from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict
from dataclasses import replace
from math import floor
from random import Random
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common.experiment_core import add_common_args, parse_rates, parse_seeds, resize_tuple
from common.image_io import load_rgb_image, prepare_images
from common.metrics import full_image_success, global_ssim, important_block_recovery, mse_psnr
from common.plotting import write_csv
from common.rs_codec import simulate_rs_capacity
from common.uep_scheduler import BitPlaneBlock, apply_schedule, decompose_bitplane_blocks, reconstruct_image


METHODS = [
    "uniform_rs4",
    "static_uep",
    "random_budget_neutral_uep",
    "keyed_uep",
    "saliency_only_uep",
    "saliency_aware_keyed_uep",
]
METHOD_LABELS = {
    "uniform_rs4": "Uniform RS-4",
    "static_uep": "Static UEP",
    "random_budget_neutral_uep": "Random UEP",
    "keyed_uep": "Current Keyed UEP",
    "saliency_only_uep": "Saliency-only UEP",
    "saliency_aware_keyed_uep": "Saliency-aware Keyed UEP",
}
LEVEL_PARITY = {"H": 8, "M": 4, "L": 2}


def parse_profile(text: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in text.replace("/", ",").split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("profile must have three integers, e.g. 8/4/2")
    return vals[0], vals[1], vals[2]


def block_saliency(image: Image.Image, block_size: int) -> dict[tuple[int, int], float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    h, w = gray.shape
    pad_h = ((h + block_size - 1) // block_size) * block_size
    pad_w = ((w + block_size - 1) // block_size) * block_size
    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:h, :w] = gray
    scores = {}
    raw = []
    for by in range(0, pad_h, block_size):
        for bx in range(0, pad_w, block_size):
            patch = padded[by : by + block_size, bx : bx + block_size]
            gy, gx = np.gradient(patch)
            grad_energy = float(np.mean(gx * gx + gy * gy))
            variance = float(np.var(patch))
            edge_density = float(np.mean(np.sqrt(gx * gx + gy * gy) > 10.0))
            score = 0.5 * grad_energy + 0.4 * variance + 0.1 * edge_density
            key = (by // block_size, bx // block_size)
            scores[key] = score
            raw.append(score)
    lo, hi = min(raw), max(raw)
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def apply_saliency_schedule(
    blocks: list[BitPlaneBlock],
    saliency: dict[tuple[int, int], float],
    method: str,
    seed: int,
    alpha: float,
    profile: tuple[int, int, int],
    promote_ratio: float = 0.10,
    demote_ratio: float = 0.20,
) -> list[BitPlaneBlock]:
    if method in {"uniform_rs4", "static_uep", "random_budget_neutral_uep", "keyed_uep"}:
        return apply_schedule(blocks, method, seed=seed, profile=profile, promote_ratio=promote_ratio, demote_ratio=demote_ratio)

    scheduled = [replace(b) for b in blocks]
    parity = {"H": profile[0], "M": profile[1], "L": profile[2]}
    rng = Random(f"saliency-keyed:{seed}") if method == "saliency_aware_keyed_uep" else Random("saliency-only-fixed")
    jitter = {b.index: rng.random() for b in scheduled}
    if method == "saliency_only_uep":
        alpha_eff = 1.0
    elif method == "saliency_aware_keyed_uep":
        alpha_eff = alpha
    else:
        raise ValueError(f"Unknown method: {method}")

    def combined(b: BitPlaneBlock) -> float:
        s = saliency.get((b.block_row, b.block_col), 0.0)
        return alpha_eff * s + (1.0 - alpha_eff) * jitter[b.index]

    bit5 = [b for b in scheduled if b.bitplane == 5 and b.base_level == "M"]
    bit3 = [b for b in scheduled if b.bitplane == 3 and b.base_level == "M"]
    promote_n = floor(promote_ratio * len(bit5))
    demote_n = floor(demote_ratio * len(bit3))
    promote = {b.index for b in sorted(bit5, key=lambda x: (-combined(x), x.index))[:promote_n]}
    demote = {b.index for b in sorted(bit3, key=lambda x: (combined(x), x.index))[:demote_n]}
    for b in scheduled:
        b.level = "H" if b.index in promote else "L" if b.index in demote else b.base_level
        b.parity = parity[b.level]
    return scheduled


def weighted_visual_recovery(original: list[BitPlaneBlock], recovered: list[BitPlaneBlock]) -> float:
    total = 0.0
    got = 0.0
    for o, r in zip(original, recovered):
        weight = float(2 ** o.bitplane)
        total += weight
        if o.data == r.data:
            got += weight
    return got / max(1.0, total)


def salient_medium_block_recovery(original: list[BitPlaneBlock], recovered: list[BitPlaneBlock], saliency: dict[tuple[int, int], float]) -> float:
    candidates = [o for o in original if o.base_level == "M"]
    vals = [saliency.get((o.block_row, o.block_col), 0.0) for o in candidates]
    threshold = float(np.quantile(vals, 0.75)) if vals else 1.0
    selected = [(o, r) for o, r in zip(original, recovered) if o.base_level == "M" and saliency.get((o.block_row, o.block_col), 0.0) >= threshold]
    return sum(1 for o, r in selected if o.data == r.data) / max(1, len(selected))


def promoted_set(blocks: list[BitPlaneBlock]) -> set[int]:
    return {b.index for b in blocks if b.bitplane == 5 and b.base_level == "M" and b.level == "H"}


def demoted_set(blocks: list[BitPlaneBlock]) -> set[int]:
    return {b.index for b in blocks if b.bitplane == 3 and b.base_level == "M" and b.level == "L"}


def schedule_signature(blocks: list[BitPlaneBlock]) -> tuple[str, ...]:
    return tuple(b.level for b in sorted(blocks, key=lambda x: x.index))


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def hamming(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    return sum(x != y for x, y in zip(a, b)) / max(1, len(a))


def summarize(rows: list[dict], group_keys: list[str], metrics: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        base = {k: v for k, v in zip(group_keys, key)}
        for metric in metrics:
            vals = [float(r[metric]) for r in items]
            rec = dict(base)
            rec.update(
                {
                    "metric": metric,
                    "mean": round(mean(vals), 6),
                    "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
                    "num_images": len({r["image"] for r in items}),
                    "num_seeds": len({r["seed"] for r in items if "seed" in r}),
                    "num_runs": len(items),
                }
            )
            out.append(rec)
    return out


def plot_reliability(summary: list[dict], path_base, metric: str, ylabel: str) -> None:
    methods = METHODS
    rates = sorted({float(r["error_rate"]) for r in summary})
    lookup = {(r["method"], float(r["error_rate"]), r["metric"]): r for r in summary}
    colors = {
        "uniform_rs4": "#9BA3AF",
        "static_uep": "#D49344",
        "random_budget_neutral_uep": "#8A8A8A",
        "keyed_uep": "#4C78A8",
        "saliency_only_uep": "#7AA974",
        "saliency_aware_keyed_uep": "#1B7F79",
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for method in methods:
        vals = [lookup[(method, r, metric)]["mean"] for r in rates]
        ax.plot([100 * r for r in rates], vals, marker="o", linewidth=1.4, markersize=3.5, label=METHOD_LABELS[method], color=colors[method])
    ax.set_xlabel("Base substitution rate (%)")
    ax.set_ylabel(ylabel)
    ax.grid(color="#E7E7E7", linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_schedule(summary: list[dict], path_base) -> None:
    metrics = ["promoted_jaccard", "demoted_jaccard", "schedule_hamming_distance", "mean_parity"]
    labels = ["Promoted\nJaccard", "Demoted\nJaccard", "Schedule\nHamming", "Mean\nparity"]
    methods = ["keyed_uep", "saliency_aware_keyed_uep"]
    lookup = {(r["method"], r["metric"]): r for r in summary}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, metric, label in zip(axes, metrics, labels):
        vals = [lookup[(m, metric)]["mean"] for m in methods]
        errs = [lookup[(m, metric)]["std"] for m in methods]
        ax.bar([0, 1], vals, yerr=errs, color=["#4C78A8", "#1B7F79"], capsize=2.0, edgecolor="#333333", linewidth=0.4)
        ax.set_xticks([0, 1], ["Current\nKeyed", "Saliency-aware\nKeyed"])
        ax.set_title(label, fontsize=7)
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.7)
        ax.set_axisbelow(True)
        if metric != "mean_parity":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(4.20, 4.28)
    axes[0].set_ylabel("Value")
    fig.tight_layout(w_pad=0.5)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot: saliency-aware keyed UEP.")
    add_common_args(parser)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--profile", default="8/4/2")
    args = parser.parse_args()
    start_all = time.perf_counter()
    resize = resize_tuple(args.resize)
    profile = parse_profile(args.profile)
    images = prepare_images(args.image_folder, args.output_dir, args.smoke, resize)
    rates = parse_rates(args.error_rates)
    seeds = parse_seeds(args.seeds)

    rows = []
    sched_records = []
    schedule_cache = {}
    for image_path in images:
        image = load_rgb_image(image_path, resize)
        sal = block_saliency(image, args.block_size)
        blocks, original_size, padded_size = decompose_bitplane_blocks(image, args.block_size)
        for method in METHODS:
            for seed in seeds:
                scheduled = apply_saliency_schedule(blocks, sal, method, seed, args.alpha, profile)
                schedule_cache[(image_path.name, method, seed)] = scheduled
                promote = promoted_set(scheduled)
                demote = demoted_set(scheduled)
                bit5 = [b for b in scheduled if b.bitplane == 5 and b.base_level == "M"]
                bit3 = [b for b in scheduled if b.bitplane == 3 and b.base_level == "M"]
                for rate in rates:
                    start = time.perf_counter()
                    recovered, rs_fail = simulate_rs_capacity(scheduled, rate, seed)
                    recon = reconstruct_image(recovered, original_size, padded_size, args.block_size)
                    mse, psnr = mse_psnr(image, recon)
                    rows.append(
                        {
                            "dataset": args.image_folder.name if args.image_folder else "custom",
                            "image": image_path.name,
                            "method": method,
                            "channel_type": "simulated_dna_base_substitution",
                            "error_rate": rate,
                            "seed": seed,
                            "block_size": args.block_size,
                            "parity_profile": "/".join(str(x) for x in profile),
                            "alpha": args.alpha,
                            "psnr": round(psnr, 6),
                            "ssim": round(global_ssim(image, recon), 6),
                            "important_block_recovery": round(important_block_recovery(scheduled, recovered), 6),
                            "block_failure": round(rs_fail, 6),
                            "weighted_visual_recovery": round(weighted_visual_recovery(scheduled, recovered), 6),
                            "salient_medium_block_recovery": round(salient_medium_block_recovery(scheduled, recovered, sal), 6),
                            "mean_parity": round(mean(b.parity for b in scheduled), 6),
                            "promoted_fraction": round(len(promote) / max(1, len(bit5)), 6),
                            "demoted_fraction": round(len(demote) / max(1, len(bit3)), 6),
                            "runtime_seconds": round(time.perf_counter() - start, 6),
                        }
                    )

    for method in ["keyed_uep", "saliency_aware_keyed_uep"]:
        for image_path in images:
            same_pairs = [(s, s) for s in seeds]
            inter_pairs = list(itertools.combinations(seeds, 2))
            for a, b in same_pairs + inter_pairs:
                sa = schedule_cache[(image_path.name, method, a)]
                sb = schedule_cache[(image_path.name, method, b)]
                sched_records.append(
                    {
                        "dataset": args.image_folder.name if args.image_folder else "custom",
                        "image": image_path.name,
                        "method": method,
                        "key_seed_a": a,
                        "key_seed_b": b,
                        "comparison_type": "same_key" if a == b else "inter_key",
                        "schedule_match_rate": 1.0 if schedule_signature(sa) == schedule_signature(sb) else 0.0,
                        "promoted_jaccard": round(jaccard(promoted_set(sa), promoted_set(sb)), 6),
                        "demoted_jaccard": round(jaccard(demoted_set(sa), demoted_set(sb)), 6),
                        "schedule_hamming_distance": round(hamming(schedule_signature(sa), schedule_signature(sb)), 6),
                        "mean_parity": round(mean(x.parity for x in sa), 6),
                    }
                )

    result_dir = args.output_dir / "results" / "pilot"
    figure_dir = args.output_dir / "figures" / "pilot"
    write_csv(result_dir / "saliency_keyed_uep_raw.csv", rows)
    summary = summarize(
        rows,
        ["method", "channel_type", "error_rate"],
        ["psnr", "ssim", "important_block_recovery", "block_failure", "weighted_visual_recovery", "salient_medium_block_recovery", "mean_parity", "promoted_fraction", "demoted_fraction"],
    )
    write_csv(result_dir / "saliency_keyed_uep_summary.csv", summary)
    schedule_summary = summarize(
        [r for r in sched_records if r["comparison_type"] == "inter_key"],
        ["method"],
        ["promoted_jaccard", "demoted_jaccard", "schedule_hamming_distance", "mean_parity"],
    )
    same_summary = summarize(
        [r for r in sched_records if r["comparison_type"] == "same_key"],
        ["method"],
        ["schedule_match_rate"],
    )
    schedule_summary = same_summary + schedule_summary
    write_csv(result_dir / "saliency_keyed_uep_schedule_summary.csv", schedule_summary)
    plot_reliability(summary, figure_dir / "saliency_keyed_uep_reliability", "psnr", "PSNR (dB)")
    plot_reliability(summary, figure_dir / "saliency_keyed_uep_weighted_recovery", "weighted_visual_recovery", "Weighted visual recovery")
    plot_schedule(schedule_summary, figure_dir / "saliency_keyed_uep_schedule_diversity")
    print(f"Wrote {len(rows)} reconstruction rows and {len(schedule_summary)} schedule summary rows in {time.perf_counter() - start_all:.3f} s")


if __name__ == "__main__":
    main()
