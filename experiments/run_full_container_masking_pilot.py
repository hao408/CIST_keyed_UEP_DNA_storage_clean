from __future__ import annotations

import argparse
import csv
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.dna_channel import bytes_to_dna
from common.experiment_core import add_common_args, parse_seeds, resize_tuple
from common.image_io import load_rgb_image, prepare_images
from common.plotting import write_csv
from common.security_features import LEVELS, as_matrix, make_payload, payload_features, trailing_deterministic_pad_len
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks


DESIGNS = [
    "variable_length_payload",
    "fixed_deterministic_padding",
    "current_encrypted_dummy_tail",
    "full_container_masked_fixed_payload",
]
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


def build_design_dataset(args, design: str, methods: list[str], seeds: list[int]) -> list[dict]:
    resize = resize_tuple(args.resize)
    images = prepare_images(args.image_folder, args.output_dir, args.smoke, resize)
    samples: list[dict] = []
    for image_path in images:
        image = load_rgb_image(image_path, resize)
        blocks, _, _ = decompose_bitplane_blocks(image, args.block_size)
        for method in methods:
            for seed in seeds:
                scheduled = apply_schedule(blocks, method, seed=seed)
                for block in scheduled:
                    payload = make_payload(block, design, seed)
                    samples.append(
                        {
                            "dataset": args.image_folder.name if args.image_folder else "custom",
                            "image": image_path.name,
                            "method": method,
                            "seed": seed,
                            "group": f"{image_path.name}|{seed}",
                            "label": block.level,
                            "payload": payload,
                            "features": payload_features(payload),
                        }
                    )
    return samples


def make_row(design: str, attacker: str, split_seed: int, meta: list[dict], y_true, y_pred, runtime: float) -> dict:
    return {
        "dataset": meta[0]["dataset"] if meta else "NA",
        "image": "|".join(sorted({m["image"] for m in meta})),
        "container_design": design,
        "attacker_type": attacker,
        "feature_set": FEATURE_SET,
        "split_seed": split_seed,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LEVELS, average="macro", zero_division=0)), 6),
        "runtime_seconds": round(runtime, 6),
    }


def evaluate_design(design: str, samples: list[dict], split_seed: int) -> list[dict]:
    labels = np.asarray([s["label"] for s in samples])
    groups = np.asarray([s["group"] for s in samples])
    x = as_matrix(s["features"] for s in samples)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=split_seed)
    train_idx, test_idx = next(splitter.split(x, labels, groups))
    meta = [samples[i] for i in test_idx]
    y_test = labels[test_idx]

    rows = []
    rng = Random(f"rule:{design}:{split_seed}")
    pred_rule = np.asarray([rule_based_predict(s["payload"], design, rng) for s in meta])
    rows.append(make_row(design, "rule_based_attacker", split_seed, meta, y_test, pred_rule, 0.0))

    models = {
        "logistic_regression_attacker": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=800, class_weight="balanced", random_state=split_seed),
        ),
        "random_forest_attacker": RandomForestClassifier(
            n_estimators=48,
            max_depth=10,
            min_samples_leaf=5,
            max_samples=0.45,
            class_weight="balanced_subsample",
            random_state=split_seed,
            n_jobs=-1,
        ),
    }
    for attacker, model in models.items():
        start = time.perf_counter()
        model.fit(x[train_idx], labels[train_idx])
        pred = model.predict(x[test_idx])
        rows.append(make_row(design, attacker, split_seed, meta, y_test, pred, time.perf_counter() - start))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["container_design"], row["attacker_type"], row["feature_set"])].append(row)
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
                    "num_splits": len({r["split_seed"] for r in items}),
                    "num_runs": len(items),
                }
            )
    return out


def plot_accuracy(summary: list[dict], path_base: Path) -> None:
    lookup = {(r["container_design"], r["attacker_type"], r["metric"]): r for r in summary}
    labels = {
        "variable_length_payload": "Variable\nlength",
        "fixed_deterministic_padding": "Deterministic\npadding",
        "current_encrypted_dummy_tail": "Current encrypted\ndummy tail",
        "full_container_masked_fixed_payload": "Full-container\nmasking",
    }
    colors = {
        "rule_based_attacker": "#9AA6C8",
        "logistic_regression_attacker": "#56A3A6",
        "random_forest_attacker": "#1F5C99",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(DESIGNS))
    width = 0.23
    for i, attacker in enumerate(ATTACKERS):
        vals = [lookup[(d, attacker, "accuracy")]["mean"] for d in DESIGNS]
        errs = [lookup[(d, attacker, "accuracy")]["std"] for d in DESIGNS]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2, label=attacker.replace("_attacker", "").replace("_", " "), color=colors[attacker], edgecolor="#333333", linewidth=0.4)
    ax.axhline(1 / 3, color="#6F6F6F", linestyle="--", linewidth=1.0, label="random guess")
    ax.set_ylabel("Attacker accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[d] for d in DESIGNS])
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_metric(summary: list[dict], metric: str, path_base: Path) -> None:
    lookup = {(r["container_design"], r["attacker_type"], r["metric"]): r for r in summary}
    designs = [d for d in DESIGNS if any(r["container_design"] == d for r in summary)]
    labels = {
        "variable_length_payload": "Variable\nlength",
        "fixed_deterministic_padding": "Deterministic\npadding",
        "current_encrypted_dummy_tail": "Tail-only\ndummy",
        "full_container_masked_fixed_payload": "Full-container\nmasked",
    }
    colors = {
        "rule_based_attacker": "#9AA6C8",
        "logistic_regression_attacker": "#56A3A6",
        "random_forest_attacker": "#1F5C99",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(designs))
    width = 0.23
    for i, attacker in enumerate(ATTACKERS):
        vals = [lookup[(d, attacker, metric)]["mean"] for d in designs]
        errs = [lookup[(d, attacker, metric)]["std"] for d in designs]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2, label=attacker.replace("_attacker", "").replace("_", " "), color=colors[attacker], edgecolor="#333333", linewidth=0.4)
    ax.axhline(1 / 3, color="#6F6F6F", linestyle="--", linewidth=1.0, label="random guess")
    ax.set_ylabel("Attacker accuracy" if metric == "accuracy" else "Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[d] for d in designs])
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(summary: list[dict], path: Path) -> None:
    lookup = {(r["container_design"], r["attacker_type"], r["metric"]): r for r in summary}
    labels = {
        "variable_length_payload": "Variable length",
        "fixed_deterministic_padding": "Deterministic padding",
        "current_encrypted_dummy_tail": "Tail-only encrypted dummy",
        "full_container_masked_fixed_payload": "Full-container masked fixed payload",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Protection-level leakage attack for fixed-container designs. Values are mean $\pm$ standard deviation over grouped splits on the five representative RGB images. The results are empirical diagnostics under simulated attackers, not cryptographic proof.}",
        r"\label{tab:full_container_leakage}",
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
            lines.append(f"{labels[design]} & {attacker.replace('_', ' ')} & {float(acc['mean']):.4f} $\\pm$ {float(acc['std']):.4f} & {float(f1['mean']):.4f} $\\pm$ {float(f1['std']):.4f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot: full-container masking leakage diagnostic.")
    add_common_args(parser)
    parser.add_argument("--methods", default="static_uep,keyed_uep")
    parser.add_argument("--container-designs", default=",".join(DESIGNS))
    parser.add_argument("--split-seeds", default="0,1,2,3,4")
    parser.add_argument("--result-scope", choices=["pilot", "full_simulation"], default="pilot")
    parser.add_argument("--prefix", default="full_container_masking_leakage")
    args = parser.parse_args()
    start = time.perf_counter()
    designs = parse_list(args.container_designs)
    methods = parse_list(args.methods)
    seeds = parse_seeds(args.seeds)
    split_seeds = parse_seeds(args.split_seeds)
    rows = []
    for design in designs:
        design_start = time.perf_counter()
        samples = build_design_dataset(args, design, methods, seeds)
        for split_seed in split_seeds:
            rows.extend(evaluate_design(design, samples, split_seed))
        print(f"Finished {design} with {len(samples)} samples in {time.perf_counter() - design_start:.3f} s")
    summary = summarize(rows)
    result_dir = args.output_dir / "results" / args.result_scope
    figure_dir = args.output_dir / "figures" / args.result_scope
    write_csv(result_dir / f"{args.prefix}_raw.csv", rows)
    write_csv(result_dir / f"{args.prefix}_summary.csv", summary)
    plot_metric(summary, "accuracy", figure_dir / f"{args.prefix}_accuracy")
    plot_metric(summary, "macro_f1", figure_dir / f"{args.prefix}_macro_f1")
    if args.result_scope == "full_simulation" and args.prefix == "full_container_leakage":
        write_latex_table(summary, args.output_dir / "tables" / "generated" / "table_full_container_leakage.tex")
    print(f"Wrote {len(rows)} raw rows and {len(summary)} summary rows in {time.perf_counter() - start:.3f} s")


if __name__ == "__main__":
    main()
