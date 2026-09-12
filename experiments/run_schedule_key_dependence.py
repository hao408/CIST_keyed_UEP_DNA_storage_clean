from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import time
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

from common.image_io import iter_image_paths, load_rgb_image
from common.uep_scheduler import BitPlaneBlock, apply_schedule, decompose_bitplane_blocks


METHOD_LABELS = {
    "static_uep": "Static UEP",
    "keyed_uep": "Keyed UEP",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def schedule_signature(blocks: list[BitPlaneBlock]) -> tuple[str, ...]:
    return tuple(b.level for b in sorted(blocks, key=lambda x: x.index))


def promoted_set(blocks: list[BitPlaneBlock]) -> set[int]:
    return {b.index for b in blocks if b.bitplane == 5 and b.base_level == "M" and b.level == "H"}


def demoted_set(blocks: list[BitPlaneBlock]) -> set[int]:
    return {b.index for b in blocks if b.bitplane == 3 and b.base_level == "M" and b.level == "L"}


def mean_parity(blocks: list[BitPlaneBlock]) -> float:
    return sum(b.parity for b in blocks) / max(1, len(blocks))


def fraction_selected(blocks: list[BitPlaneBlock], bitplane: int, selected: set[int]) -> float:
    denom = sum(1 for b in blocks if b.bitplane == bitplane and b.base_level == "M")
    return len(selected) / max(1, denom)


def hamming_distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("Schedule signatures have different lengths")
    return sum(x != y for x, y in zip(a, b)) / max(1, len(a))


def summarize(raw_rows: list[dict]) -> list[dict]:
    metric_filters = {
        "schedule_match_rate": "same_key",
        "promoted_jaccard": "inter_key",
        "demoted_jaccard": "inter_key",
        "schedule_hamming_distance": "inter_key",
        "mean_parity": "all",
        "promoted_fraction": "all",
        "demoted_fraction": "all",
    }
    grouped: dict[tuple[str, str], list[float]] = {}
    images_by_method: dict[str, set[str]] = {}
    pairs_by_method_metric: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for row in raw_rows:
        method = row["method"]
        images_by_method.setdefault(method, set()).add(row["image"])
        for metric, mode in metric_filters.items():
            if mode != "all" and row["comparison_type"] != mode:
                continue
            grouped.setdefault((method, metric), []).append(float(row[metric]))
            pairs_by_method_metric.setdefault((method, metric), set()).add((row["image"], row["key_seed_a"], row["key_seed_b"]))

    summary = []
    for (method, metric), values in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "metric": metric,
                "mean": round(statistics.fmean(values), 6),
                "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
                "num_images": len(images_by_method[method]),
                "num_key_pairs": len(pairs_by_method_metric[(method, metric)]),
            }
        )
    return summary


def summary_lookup(summary: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["method"], r["metric"]): r for r in summary}


def draw_overlap(summary: list[dict], path_base: Path) -> None:
    lookup = summary_lookup(summary)
    metrics = [
        ("promoted_jaccard", "Promoted overlap"),
        ("demoted_jaccard", "Demoted overlap"),
        ("schedule_hamming_distance", "Schedule Hamming"),
        ("mean_parity", "Mean parity"),
    ]
    methods = ["static_uep", "keyed_uep"]
    colors = {"static_uep": "#D9A441", "keyed_uep": "#4C78A8"}

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.8))
    for ax, (metric, title) in zip(axes, metrics):
        vals = [float(lookup[(m, metric)]["mean"]) for m in methods]
        errs = [float(lookup[(m, metric)]["std"]) for m in methods]
        xs = range(len(methods))
        ax.bar(xs, vals, yerr=errs, color=[colors[m] for m in methods], width=0.62, capsize=2.5)
        ax.set_xticks(list(xs), ["Static\nUEP", "Keyed\nUEP"])
        ax.set_title(title, fontsize=7.2)
        if metric != "mean_parity":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(4.20, 4.28)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        ax.set_axisbelow(True)
        if metric == "promoted_jaccard":
            ax.text(0, 0.90, "fixed\nacross keys", ha="center", va="top", fontsize=5.7, color="#6B4A10")
            ax.text(1, 0.20, "changes\nacross keys", ha="center", va="bottom", fontsize=5.7, color="#244E7A")
        elif metric == "demoted_jaccard":
            ax.text(0, 0.90, "fixed", ha="center", va="top", fontsize=5.7, color="#6B4A10")
            ax.text(1, 0.24, "low overlap", ha="center", va="bottom", fontsize=5.7, color="#244E7A")
        elif metric == "schedule_hamming_distance":
            ax.text(1, 0.11, "non-zero\nschedule change", ha="center", va="bottom", fontsize=5.5, color="#244E7A")
        elif metric == "mean_parity":
            ax.text(0.5, 4.272, "unchanged parity budget", ha="center", va="top", fontsize=5.7, color="#444444")
    axes[0].set_ylabel("Value")
    fig.tight_layout(w_pad=0.6)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def fmt(summary: list[dict], method: str, metric: str) -> str:
    row = summary_lookup(summary)[(method, metric)]
    return f"{float(row['mean']):.4f} $\\pm$ {float(row['std']):.4f}"


def write_schedule_table(summary: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Schedule key-dependence diagnostic on the five representative RGB images using block size 16 and the default 10/20 keyed perturbation. Values are computed from schedule generation only; no DNA-channel simulation or image reconstruction is performed.}",
        r"\label{tab:schedule_key_dependence}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Same-key match & Promoted Jaccard & Demoted Jaccard & Schedule Hamming & Mean parity \\",
        r"\midrule",
    ]
    for method in ["static_uep", "keyed_uep"]:
        lines.append(
            f"{METHOD_LABELS[method]} & {fmt(summary, method, 'schedule_match_rate')} & "
            f"{fmt(summary, method, 'promoted_jaccard')} & {fmt(summary, method, 'demoted_jaccard')} & "
            f"{fmt(summary, method, 'schedule_hamming_distance')} & {fmt(summary, method, 'mean_parity')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tradeoff_table(path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Qualitative role of representative coding and container designs. The table summarizes the evidence chain without introducing new reconstruction metrics.}",
        r"\label{tab:method_tradeoff_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Method & Importance-aware recovery & Average parity budget & Key-synchronized schedule & Direct length/padding leakage reduction & Main role \\",
        r"\midrule",
        r"Uniform RS-4 & No & Uniform 4 bytes & No & No & Equal-protection baseline \\",
        r"Static UEP & Yes & 4.25 bytes & No & No & Deterministic importance-aware recovery \\",
        r"Random UEP & Yes & 4.25 bytes & Shared seed/metadata required & No & Budget-neutral randomized control \\",
        r"Keyed UEP + fixed encrypted container & Yes & $\approx$4.25 bytes & Yes & Yes & Static-UEP-like recovery with key-dependent schedule and reduced direct cues \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    image_paths = iter_image_paths(args.image_folder)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_folder}")

    raw_rows = []
    key_seeds = [int(x) for x in args.key_seeds.split(",") if x.strip()]
    inter_pairs = list(itertools.combinations(key_seeds, 2))
    same_pairs = [(s, s) for s in key_seeds]
    all_pairs = same_pairs + inter_pairs

    for image_path in image_paths:
        image = load_rgb_image(image_path, args.resize)
        blocks, _, _ = decompose_bitplane_blocks(image, args.block_size)
        schedules: dict[tuple[str, int], dict] = {}
        start_image = time.perf_counter()
        for method in ["static_uep", "keyed_uep"]:
            for seed in key_seeds:
                scheduled = apply_schedule(blocks, method, seed=seed)
                promote = promoted_set(scheduled)
                demote = demoted_set(scheduled)
                schedules[(method, seed)] = {
                    "signature": schedule_signature(scheduled),
                    "promote": promote,
                    "demote": demote,
                    "mean_parity": mean_parity(scheduled),
                    "promoted_fraction": fraction_selected(scheduled, 5, promote),
                    "demoted_fraction": fraction_selected(scheduled, 3, demote),
                }
        elapsed = time.perf_counter() - start_image
        runtime_per_row = elapsed / max(1, 2 * len(all_pairs))
        for method in ["static_uep", "keyed_uep"]:
            for a, b in all_pairs:
                sa = schedules[(method, a)]
                sb = schedules[(method, b)]
                is_same = a == b
                raw_rows.append(
                    {
                        "dataset": args.dataset_name,
                        "image": image_path.name,
                        "key_seed_a": a,
                        "key_seed_b": b,
                        "method": method,
                        "comparison_type": "same_key" if is_same else "inter_key",
                        "schedule_match_rate": round(1.0 if sa["signature"] == sb["signature"] else 0.0, 6),
                        "promoted_jaccard": round(jaccard(sa["promote"], sb["promote"]), 6),
                        "demoted_jaccard": round(jaccard(sa["demote"], sb["demote"]), 6),
                        "schedule_hamming_distance": round(hamming_distance(sa["signature"], sb["signature"]), 6),
                        "mean_parity": round((sa["mean_parity"] + sb["mean_parity"]) / 2, 6),
                        "promoted_fraction": round((sa["promoted_fraction"] + sb["promoted_fraction"]) / 2, 6),
                        "demoted_fraction": round((sa["demoted_fraction"] + sb["demoted_fraction"]) / 2, 6),
                        "runtime_seconds": round(runtime_per_row, 6),
                    }
                )

    result_dir = args.output_dir / "results" / "full_simulation"
    figure_dir = args.output_dir / "figures" / "full_simulation"
    table_dir = args.output_dir / "tables" / "generated"
    raw_path = result_dir / "schedule_key_dependence_raw.csv"
    summary_path = result_dir / "schedule_key_dependence_summary.csv"
    summary = summarize(raw_rows)
    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary)
    draw_overlap(summary, figure_dir / "schedule_key_dependence_overlap")
    write_schedule_table(summary, table_dir / "table_schedule_key_dependence.tex")
    write_tradeoff_table(table_dir / "table_method_tradeoff_summary.tex")
    print(f"Wrote {len(raw_rows)} schedule-only rows to {raw_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule-only key-dependence diagnostic.")
    parser.add_argument("--image-folder", type=Path, default=Path("datasets/custom_equal_budget"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--dataset-name", default="five_representative_rgb")
    parser.add_argument("--resize", type=lambda s: tuple(map(int, s.split(","))), default=(256, 256))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--key-seeds", default="0,1,2,3,4,5,6,7,8,9")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
