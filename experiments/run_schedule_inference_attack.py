from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from random import Random
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.experiment_core import add_common_args, figures_dir, parse_seeds, resize_tuple, results_dir
from common.image_io import load_rgb_image, prepare_images
from common.plotting import write_csv
from common.security_features import (
    LEVELS,
    TRANSITIONS,
    as_matrix,
    make_payload,
    payload_features,
    trailing_deterministic_pad_len,
    transition_label,
)
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks

METHODS = ["static_uep", "random_budget_neutral_uep", "keyed_uep"]
DESIGNS = ["variable_length_payload", "fixed_deterministic_padding", "current_encrypted_dummy_tail", "full_container_masked_fixed_payload"]
ATTACKERS = ["rule_based_attacker", "logistic_regression_attacker", "random_forest_attacker"]
FEATURE_SET = "container_qc_kmer2_position"


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def rule_predict_final(payload: str, design: str, rng: Random) -> str:
    if design == "variable_length_payload":
        if len(payload) >= 160:
            return "H"
        if len(payload) >= 144:
            return "M"
        return "L"
    tail = trailing_deterministic_pad_len(payload)
    if tail >= 24:
        return "L"
    if tail >= 16:
        return "M"
    if design == "fixed_deterministic_padding":
        return "H"
    return rng.choice(LEVELS)


def rule_predict_transition(payload: str, design: str, rng: Random) -> str:
    # Observable payload features do not identify which medium blocks were
    # promoted/demoted once fixed encrypted containers are used. Variable-length
    # or deterministic padding may leak final level, which only partially maps
    # onto transition labels.
    final = rule_predict_final(payload, design, rng)
    if final == "H":
        return "promoted"
    if final == "L":
        return "demoted"
    if design in {"fixed_encrypted_dummy_padding", "current_encrypted_dummy_tail", "full_container_masked_fixed_payload"}:
        return rng.choice(TRANSITIONS)
    return "unchanged"


def collect_samples(args, resize, methods, designs, seeds):
    samples = []
    for image_path in prepare_images(args.image_folder, args.output_dir, args.smoke, resize):
        image = load_rgb_image(image_path, resize)
        blocks, _, _ = decompose_bitplane_blocks(image, args.block_size)
        for method in methods:
            for seed in seeds:
                scheduled = apply_schedule(blocks, method, seed=seed)
                for block in scheduled:
                    for design in designs:
                        payload = make_payload(block, design, seed)
                        samples.append(
                            {
                                "dataset": "smoke" if args.smoke else args.image_folder.name,
                                "image": image_path.name,
                                "method": method,
                                "container_design": design,
                                "seed": seed,
                                "group": f"{image_path.name}|{seed}",
                                "final_level": block.level,
                                "transition": transition_label(block),
                                "base_level": block.base_level,
                                "payload": payload,
                                "features": payload_features(payload),
                            }
                        )
    return samples


def evaluate_subset(samples, task, method, design, split_seed):
    subset = [s for s in samples if s["method"] == method and s["container_design"] == design]
    if task == "medium_transition":
        subset = [s for s in subset if s["base_level"] == "M"]
        classes = TRANSITIONS
        label_key = "transition"
    else:
        classes = LEVELS
        label_key = "final_level"
    if not subset:
        return [], []
    y = np.asarray([s[label_key] for s in subset])
    groups = np.asarray([s["group"] for s in subset])
    x = as_matrix(s["features"] for s in subset)
    if len(set(groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=split_seed)
        train_idx, test_idx = next(splitter.split(x, y, groups))
    else:
        stratify = y if len(set(y)) > 1 and min(Counter(y).values()) >= 2 else None
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.30,
            random_state=split_seed,
            stratify=stratify,
        )
    y_test = y[test_idx]
    test_meta = [subset[i] for i in test_idx]
    rows, cm_rows = [], []

    rng = Random(f"schedule-rule:{task}:{method}:{design}:{split_seed}")
    if task == "medium_transition":
        pred_rule = np.asarray([rule_predict_transition(s["payload"], design, rng) for s in test_meta])
    else:
        pred_rule = np.asarray([rule_predict_final(s["payload"], design, rng) for s in test_meta])
    rows.append(make_row(task, method, design, "rule_based_attacker", split_seed, test_meta, y_test, pred_rule, classes, 0.0))
    cm_rows.extend(make_cm_rows(task, method, design, "rule_based_attacker", split_seed, y_test, pred_rule, classes))

    if len(set(y[train_idx])) < 2:
        return rows, cm_rows

    models = {
        "logistic_regression_attacker": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=800, class_weight="balanced", random_state=split_seed),
        ),
        "random_forest_attacker": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=split_seed,
            n_jobs=-1,
        ),
    }
    for attacker, model in models.items():
        start = time.perf_counter()
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        runtime = time.perf_counter() - start
        rows.append(make_row(task, method, design, attacker, split_seed, test_meta, y_test, pred, classes, runtime))
        cm_rows.extend(make_cm_rows(task, method, design, attacker, split_seed, y_test, pred, classes))
    return rows, cm_rows


def make_row(task, method, design, attacker, split_seed, meta, y_true, y_pred, classes, runtime):
    return {
        "dataset": meta[0]["dataset"] if meta else "NA",
        "image": "|".join(sorted({m["image"] for m in meta})),
        "task": task,
        "method": method,
        "container_design": design,
        "attacker_type": attacker,
        "feature_set": FEATURE_SET,
        "split_seed": split_seed,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)), 6),
        "confusion_matrix_path": "results/full_simulation/schedule_inference_attack_confusion_matrix.csv",
        "runtime_seconds": round(runtime, 6),
    }


def make_cm_rows(task, method, design, attacker, split_seed, y_true, y_pred, classes):
    cm = Counter(zip(y_true, y_pred))
    return [
        {
            "task": task,
            "method": method,
            "container_design": design,
            "attacker_type": attacker,
            "split_seed": split_seed,
            "true_label": truth,
            "predicted_label": pred,
            "count": cm.get((truth, pred), 0),
        }
        for truth in classes
        for pred in classes
    ]


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["task"], r["method"], r["container_design"], r["attacker_type"], r["feature_set"])].append(r)
    out = []
    for key, items in sorted(grouped.items()):
        task, method, design, attacker, feature_set = key
        for metric in ["accuracy", "macro_f1"]:
            vals = [float(r[metric]) for r in items]
            out.append(
                {
                    "task": task,
                    "method": method,
                    "container_design": design,
                    "attacker_type": attacker,
                    "feature_set": feature_set,
                    "metric": metric,
                    "mean": round(mean(vals), 6),
                    "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
                    "num_images": len({img for r in items for img in str(r["image"]).split("|") if img}),
                    "num_splits": len(set(r["split_seed"] for r in items)),
                    "num_runs": len(items),
                }
            )
    return out


def plot_task_metric(summary, task, metric, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    designs = parse_list(",".join(DESIGNS))
    attackers = ATTACKERS
    lookup = {(r["container_design"], r["attacker_type"], r["metric"]): r for r in summary if r["task"] == task and r["method"] == "keyed_uep"}
    labels = {
        "variable_length_payload": "Variable\nlength",
        "fixed_deterministic_padding": "Deterministic\npadding",
        "current_encrypted_dummy_tail": "Tail-only\ndummy",
        "full_container_masked_fixed_payload": "Full-container\nmasked",
    }
    colors = {"rule_based_attacker": "#9AA6C8", "logistic_regression_attacker": "#56A3A6", "random_forest_attacker": "#1F5C99"}
    x = np.arange(len(designs))
    width = 0.23
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, attacker in enumerate(attackers):
        vals = [lookup[(d, attacker, metric)]["mean"] for d in designs if (d, attacker, metric) in lookup]
        errs = [lookup[(d, attacker, metric)]["std"] for d in designs if (d, attacker, metric) in lookup]
        xs = x[: len(vals)] + (i - 1) * width
        ax.bar(xs, vals, width, yerr=errs, color=colors[attacker], edgecolor="#272727", linewidth=0.5, capsize=2, label=attacker.replace("_attacker", "").replace("_", " "))
    if task == "final_level":
        ax.axhline(1 / 3, color="#767676", linestyle="--", linewidth=0.9, label="random guess")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy" if metric == "accuracy" else "Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[d] for d in designs])
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_table(summary, path):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Keyed schedule inference attack under simulated container observations. Values are mean $\pm$ standard deviation over repeated group splits. These auxiliary diagnostics are limited to the evaluated simulated attackers.}",
        r"\label{tab:schedule_inference_attack}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Task & Method & Container & Attacker & Accuracy & Macro-F1 \\",
        r"\midrule",
    ]
    lookup = {(r["task"], r["method"], r["container_design"], r["attacker_type"], r["metric"]): r for r in summary}
    keys = sorted({(r["task"], r["method"], r["container_design"], r["attacker_type"]) for r in summary})
    for task, method, design, attacker in keys:
        acc = lookup[(task, method, design, attacker, "accuracy")]
        f1 = lookup[(task, method, design, attacker, "macro_f1")]
        lines.append(f"{task.replace('_', ' ')} & {method.replace('_', ' ')} & {design.replace('_', ' ')} & {attacker.replace('_', ' ')} & {acc['mean']:.4f} $\\pm$ {acc['std']:.4f} & {f1['mean']:.4f} $\\pm$ {f1['std']:.4f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Schedule inference attack under simulated payload/container observations.")
    add_common_args(parser)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--container-designs", default=",".join(DESIGNS))
    parser.add_argument("--split-seeds", default="0,1,2,3,4")
    parser.add_argument("--tasks", default="final_level,medium_transition")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--result-prefix", default="schedule_inference_attack")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = "1"
        args.split_seeds = "0"
        resize = (64, 64)
    else:
        resize = resize_tuple(args.resize)
    if args.pilot:
        args.seeds = "0,1,2,3,4"
        args.split_seeds = "0,1,2"
    samples = collect_samples(args, resize, parse_list(args.methods), parse_list(args.container_designs), parse_seeds(args.seeds))
    rows, cm_rows = [], []
    for task in parse_list(args.tasks):
        for method in parse_list(args.methods):
            for design in parse_list(args.container_designs):
                # Medium-transition inference is meaningful for perturbed schedules;
                # static UEP has only unchanged medium blocks.
                if task == "medium_transition" and method == "static_uep":
                    continue
                for split_seed in parse_seeds(args.split_seeds):
                    r, c = evaluate_subset(samples, task, method, design, split_seed)
                    rows.extend(r)
                    cm_rows.extend(c)
    summary = summarize(rows)
    res = results_dir(args)
    figs = figures_dir(args)
    write_csv(res / f"{args.result_prefix}_raw.csv", rows)
    write_csv(res / f"{args.result_prefix}_confusion_matrix.csv", cm_rows)
    write_csv(res / f"{args.result_prefix}_summary.csv", summary)
    plot_task_metric(summary, "final_level", "accuracy", figs / f"{args.result_prefix}_final")
    plot_task_metric(summary, "medium_transition", "accuracy", figs / f"{args.result_prefix}_transition")
    write_table(summary, args.output_dir / "tables" / "generated" / f"table_{args.result_prefix}.tex")
    print(f"Wrote {len(rows)} rows to {res / (args.result_prefix + '_raw.csv')}")


if __name__ == "__main__":
    main()
