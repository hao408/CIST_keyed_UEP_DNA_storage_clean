from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
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
    as_matrix,
    make_payload,
    payload_features,
    trailing_deterministic_pad_len,
)
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks

DESIGNS = ["variable_length_payload", "fixed_deterministic_padding", "fixed_encrypted_dummy_padding"]
ATTACKERS = ["rule_based_attacker", "logistic_regression_attacker", "random_forest_attacker"]
FEATURE_SET = "length_composition_tail_kmer2_position"


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def rule_based_predict(payload: str, design: str, rng: Random) -> str:
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


def build_dataset(args, resize, designs: list[str], methods: list[str], seeds: list[int]) -> dict[str, list[dict]]:
    by_design: dict[str, list[dict]] = {d: [] for d in designs}
    images = prepare_images(args.image_folder, args.output_dir, args.smoke, resize)
    for image_path in images:
        image = load_rgb_image(image_path, resize)
        blocks, _, _ = decompose_bitplane_blocks(image, args.block_size)
        for method in methods:
            for seed in seeds:
                scheduled = apply_schedule(blocks, method, seed=seed)
                for block in scheduled:
                    for design in designs:
                        payload = make_payload(block, design, seed)
                        by_design[design].append(
                            {
                                "dataset": "smoke" if args.smoke else args.image_folder.name,
                                "image": image_path.name,
                                "method": method,
                                "seed": seed,
                                "group": f"{image_path.name}|{seed}",
                                "label": block.level,
                                "payload": payload,
                                "features": payload_features(payload),
                            }
                        )
    return by_design


def evaluate_design(design: str, samples: list[dict], split_seed: int) -> tuple[list[dict], list[dict]]:
    labels = np.asarray([s["label"] for s in samples])
    groups = np.asarray([s["group"] for s in samples])
    x = as_matrix(s["features"] for s in samples)
    if len(set(groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=split_seed)
        train_idx, test_idx = next(splitter.split(x, labels, groups))
    else:
        stratify = labels if len(set(labels)) > 1 and min(Counter(labels).values()) >= 2 else None
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=0.30,
            random_state=split_seed,
            stratify=stratify,
        )
    rows = []
    cm_rows = []
    test_meta = [samples[i] for i in test_idx]
    y_test = labels[test_idx]

    rng = Random(f"rule:{design}:{split_seed}")
    pred_rule = np.asarray([rule_based_predict(s["payload"], design, rng) for s in test_meta])
    rows.append(make_row(design, "rule_based_attacker", split_seed, test_meta, y_test, pred_rule, 0.0))
    cm_rows.extend(make_cm_rows(design, "rule_based_attacker", split_seed, y_test, pred_rule))

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
        model.fit(x[train_idx], labels[train_idx])
        pred = model.predict(x[test_idx])
        runtime = time.perf_counter() - start
        rows.append(make_row(design, attacker, split_seed, test_meta, y_test, pred, runtime))
        cm_rows.extend(make_cm_rows(design, attacker, split_seed, y_test, pred))
    return rows, cm_rows


def make_row(design, attacker, split_seed, meta, y_true, y_pred, runtime):
    images = sorted({m["image"] for m in meta})
    return {
        "dataset": meta[0]["dataset"] if meta else "NA",
        "image": "|".join(images),
        "container_design": design,
        "attacker_type": attacker,
        "feature_set": FEATURE_SET,
        "split_seed": split_seed,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LEVELS, average="macro", zero_division=0)), 6),
        "confusion_matrix_path": "results/full_simulation/leakage_ml_attack_confusion_matrix.csv",
        "runtime_seconds": round(runtime, 6),
    }


def make_cm_rows(design, attacker, split_seed, y_true, y_pred):
    cm = Counter(zip(y_true, y_pred))
    out = []
    for truth in LEVELS:
        for pred in LEVELS:
            out.append(
                {
                    "container_design": design,
                    "attacker_type": attacker,
                    "split_seed": split_seed,
                    "true_level": truth,
                    "predicted_level": pred,
                    "count": cm.get((truth, pred), 0),
                }
            )
    return out


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["container_design"], r["attacker_type"], r["feature_set"])].append(r)
    out = []
    for (design, attacker, feature_set), items in sorted(grouped.items()):
        for metric in ["accuracy", "macro_f1"]:
            vals = [float(r[metric]) for r in items]
            out.append(
                {
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


def plot_metric(summary: list[dict], metric: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attackers = ATTACKERS
    designs = DESIGNS
    colors = {"rule_based_attacker": "#7884B4", "logistic_regression_attacker": "#42949E", "random_forest_attacker": "#0F4D92"}
    lookup = {(r["container_design"], r["attacker_type"], r["metric"]): r for r in summary}
    x = np.arange(len(designs))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, attacker in enumerate(attackers):
        vals = [lookup[(d, attacker, metric)]["mean"] for d in designs]
        errs = [lookup[(d, attacker, metric)]["std"] for d in designs]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2, label=attacker.replace("_attacker", "").replace("_", " "), color=colors[attacker], edgecolor="#272727", linewidth=0.5)
    ax.axhline(1 / 3, color="#767676", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Variable\nlength", "Fixed\ndeterministic", "Fixed encrypted\ndummy"])
    ax.set_ylabel(metric.replace("_", "-"))
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_table(summary: list[dict], path: Path) -> None:
    lookup = {(r["container_design"], r["attacker_type"], r["metric"]): r for r in summary}
    labels = {"variable_length_payload": "Variable length", "fixed_deterministic_padding": "Fixed deterministic padding", "fixed_encrypted_dummy_padding": "Fixed encrypted dummy padding"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{ML-based protection-level leakage attack under simulated payload/container observations. Values are mean $\pm$ standard deviation over repeated group splits. These diagnostics are limited to the evaluated simulated attackers and do not constitute cryptographic proof.}",
        r"\label{tab:leakage_ml_attack}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Container design & Attacker & Accuracy & Macro-F1 \\",
        r"\midrule",
    ]
    for design in DESIGNS:
        for attacker in ATTACKERS:
            acc = lookup[(design, attacker, "accuracy")]
            f1 = lookup[(design, attacker, "macro_f1")]
            lines.append(f"{labels[design]} & {attacker.replace('_', ' ')} & {acc['mean']:.4f} $\\pm$ {acc['std']:.4f} & {f1['mean']:.4f} $\\pm$ {f1['std']:.4f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ML-based protection-level leakage attack.")
    add_common_args(parser)
    parser.add_argument("--container-designs", default=",".join(DESIGNS))
    parser.add_argument("--methods", default="static_uep,keyed_uep")
    parser.add_argument("--split-seeds", default="0,1,2,3,4")
    parser.add_argument("--pilot", action="store_true")
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

    designs = parse_list(args.container_designs)
    methods = parse_list(args.methods)
    seed_list = parse_seeds(args.seeds)
    split_seeds = parse_seeds(args.split_seeds)
    by_design = build_dataset(args, resize, designs, methods, seed_list)
    rows, cm_rows = [], []
    for design, samples in by_design.items():
        for split_seed in split_seeds:
            r, c = evaluate_design(design, samples, split_seed)
            rows.extend(r)
            cm_rows.extend(c)
    summary = summarize(rows)
    res = results_dir(args)
    figs = figures_dir(args)
    write_csv(res / "leakage_ml_attack_raw.csv", rows)
    write_csv(res / "leakage_ml_attack_confusion_matrix.csv", cm_rows)
    write_csv(res / "leakage_ml_attack_summary.csv", summary)
    plot_metric(summary, "accuracy", figs / "leakage_ml_attack_accuracy")
    plot_metric(summary, "macro_f1", figs / "leakage_ml_attack_f1")
    write_table(summary, args.output_dir / "tables" / "generated" / "table_leakage_ml_attack.tex")
    print(f"Wrote {len(rows)} rows to {res / 'leakage_ml_attack_raw.csv'}")


if __name__ == "__main__":
    main()
