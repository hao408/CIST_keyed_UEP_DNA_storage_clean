from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import mean, pstdev

import numpy as np

from common.experiment_core import add_common_args, figures_dir, parse_rates, parse_seeds, resize_tuple, results_dir
from common.image_io import load_rgb_image, prepare_images
from common.plotting import simple_bar_plot, write_csv
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks

METHODS = ["uniform_rs4", "static_uep", "random_budget_neutral_uep", "keyed_uep"]
CHANNELS = [
    "base_substitution",
    "base_insertion",
    "base_deletion",
    "mixed_indel_substitution",
    "oligo_dropout",
    "coverage_consensus",
    "index_corruption",
]
METRICS = ["psnr", "ssim", "important_block_recovery", "rs_failure_rate", "full_image_success"]


def consensus_error_prob(error_rate: float, coverage: int) -> float:
    if coverage <= 1:
        return error_rate
    threshold = coverage // 2 + 1
    prob = 0.0
    for k in range(threshold, coverage + 1):
        prob += math.comb(coverage, k) * (error_rate ** k) * ((1.0 - error_rate) ** (coverage - k))
    return min(max(prob, 0.0), 1.0)


def draw_binomial(rng: Random, n: int, p: float) -> int:
    if p <= 0:
        return 0
    if p >= 1:
        return n
    return sum(1 for _ in range(n) if rng.random() < p)


def simulate_channel_flags(blocks, channel_type: str, error_rate: float, coverage: int, seed: int):
    rng = Random(f"{channel_type}:{error_rate}:{coverage}:{seed}")
    success = []
    failures = 0
    effective_sub = consensus_error_prob(error_rate, coverage)
    # Indel alignment and index assignment are harder than independent
    # substitution consensus. We use a conservative residual-error model in
    # which coverage reduces but does not eliminate these events.
    residual_indel = min(1.0, error_rate / math.sqrt(max(1, coverage)))
    residual_index = min(1.0, error_rate / math.sqrt(max(1, coverage)))
    for block in blocks:
        n_bytes = len(block.data) + block.parity
        n_bases = 4 * n_bytes
        hard_fail = False
        observed_symbol_errors = 0

        if channel_type == "base_substitution":
            byte_error_prob = 1.0 - (1.0 - error_rate) ** 4
            observed_symbol_errors = draw_binomial(rng, n_bytes, byte_error_prob)
        elif channel_type == "coverage_consensus":
            byte_error_prob = 1.0 - (1.0 - effective_sub) ** 4
            observed_symbol_errors = draw_binomial(rng, n_bytes, byte_error_prob)
        elif channel_type == "base_insertion":
            hard_fail = draw_binomial(rng, n_bases, residual_indel) > 0
        elif channel_type == "base_deletion":
            hard_fail = draw_binomial(rng, n_bases, residual_indel) > 0
        elif channel_type == "mixed_indel_substitution":
            byte_error_prob = 1.0 - (1.0 - residual_indel) ** 4
            observed_symbol_errors = draw_binomial(rng, n_bytes, byte_error_prob)
            hard_fail = draw_binomial(rng, n_bases, residual_indel) > 0
        elif channel_type == "oligo_dropout":
            # Oligo dropout models a strand/template missing from the pool.
            # Extra sequencing coverage cannot recover an absent oligo without
            # cross-oligo redundancy.
            hard_fail = rng.random() < error_rate
        elif channel_type == "index_corruption":
            # The 8-nt index is consensus-decoded separately. A corrupted index
            # prevents reliable block placement.
            index_fail_prob = 1.0 - (1.0 - residual_index) ** 8
            hard_fail = rng.random() < index_fail_prob
        else:
            raise ValueError(f"Unsupported channel_type: {channel_type}")

        capacity = block.parity // 2
        ok = (not hard_fail) and observed_symbol_errors <= capacity
        success.append(ok)
        if not ok:
            failures += 1
    return success, failures / max(1, len(blocks))


def reconstruct_array_from_flags(original: np.ndarray, blocks, success_flags, block_size: int) -> np.ndarray:
    rec = original.copy()
    for block, ok in zip(blocks, success_flags):
        if ok:
            continue
        y0 = block.block_row * block_size
        x0 = block.block_col * block_size
        mask = np.uint8(255 ^ (1 << block.bitplane))
        rec[y0 : y0 + block_size, x0 : x0 + block_size, block.channel] &= mask
    return rec


def image_metrics(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float, float]:
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    mse = float(np.mean(diff * diff))
    psnr = 99.0 if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    x = original.astype(np.float32).mean(axis=2).ravel()
    y = reconstructed.astype(np.float32).mean(axis=2).ravel()
    mux = float(x.mean())
    muy = float(y.mean())
    vx = float(((x - mux) ** 2).mean())
    vy = float(((y - muy) ** 2).mean())
    cov = float(((x - mux) * (y - muy)).mean())
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))
    return psnr, float(ssim), mse


def important_recovery_from_flags(blocks, success_flags) -> float:
    selected = [ok for block, ok in zip(blocks, success_flags) if block.base_level == "H"]
    return sum(1 for ok in selected if ok) / max(1, len(selected))


def write_summary(rows: list[dict], output_path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["channel_type"], row["error_rate"], row["coverage"])].append(row)
    out = []
    for (method, channel_type, error_rate, coverage), items in sorted(grouped.items()):
        images = {r["image"] for r in items}
        seeds = {r["seed"] for r in items}
        for metric in METRICS:
            vals = [float(r[metric]) for r in items]
            out.append(
                {
                    "method": method,
                    "channel_type": channel_type,
                    "error_rate": error_rate,
                    "coverage": coverage,
                    "metric": metric,
                    "mean": round(mean(vals), 6),
                    "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
                    "num_images": len(images),
                    "num_seeds": len(seeds),
                    "num_runs": len(items),
                }
            )
    write_csv(output_path, out)


def aggregate_metric(rows: list[dict], metric: str, error_rate: str = "0.02", coverage: str = "30"):
    grouped = defaultdict(list)
    for row in rows:
        if str(row["error_rate"]) == error_rate and str(row["coverage"]) == coverage:
            grouped[row["channel_type"]].append(float(row[metric]))
    labels = sorted(grouped)
    values = [mean(grouped[label]) for label in labels]
    return labels, values


def write_latex_table(summary_path: Path, table_path: Path, error_rate: str = "0.02", coverage: str = "30") -> None:
    import csv

    keyed = {}
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == "keyed_uep" and row["error_rate"] == error_rate and row["coverage"] == coverage:
                keyed.setdefault(row["channel_type"], {})[row["metric"]] = f'{float(row["mean"]):.4f} $\\pm$ {float(row["std"]):.4f}'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Keyed UEP under simulated DNA storage channels at 2\\% error rate and 30$\\times$ coverage. Results are obtained from in-silico simulations, not wet-lab sequencing.}",
        "\\label{tab:dna_channel_sweep}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Channel & PSNR (dB) & SSIM & Imp. rec. & Block fail. \\\\",
        "\\midrule",
    ]
    for channel in sorted(keyed):
        m = keyed[channel]
        display = channel.replace("_", "\\_")
        lines.append(f"{display} & {m.get('psnr','NA')} & {m.get('ssim','NA')} & {m.get('important_block_recovery','NA')} & {m.get('rs_failure_rate','NA')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""]
    table_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full in-silico DNA storage channel sweep.")
    add_common_args(parser)
    parser.add_argument("--coverages", default="5,10,20,50,100")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--channels", default=",".join(CHANNELS))
    args = parser.parse_args()
    if args.smoke:
        args.error_rates = "0.01"
        args.seeds = "1"
        args.coverages = "5"
        args.methods = "keyed_uep"
        args.channels = "base_substitution,base_deletion,oligo_dropout,index_corruption"

    resize = (64, 64) if args.smoke else resize_tuple(args.resize)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    channels = [x.strip() for x in args.channels.split(",") if x.strip()]
    coverages = [int(x.strip()) for x in args.coverages.split(",") if x.strip()]
    rows = []
    for image_path in prepare_images(args.image_folder, args.output_dir, args.smoke, resize):
        img = load_rgb_image(image_path, resize)
        original_np = np.asarray(img, dtype=np.uint8)
        blocks, original_size, padded_size = decompose_bitplane_blocks(img, args.block_size)
        for method in methods:
            for channel_type in channels:
                for rate in parse_rates(args.error_rates):
                    for coverage in coverages:
                        for seed in parse_seeds(args.seeds):
                            start = time.perf_counter()
                            scheduled = apply_schedule(blocks, method, seed=seed)
                            success_flags, fail_rate = simulate_channel_flags(scheduled, channel_type, rate, coverage, seed)
                            recon_np = reconstruct_array_from_flags(original_np, scheduled, success_flags, args.block_size)
                            psnr, ssim, _ = image_metrics(original_np, recon_np)
                            rows.append(
                                {
                                    "dataset": "smoke" if args.smoke else (args.image_folder.name if args.image_folder else "custom"),
                                    "image": image_path.name,
                                    "method": method,
                                    "channel_type": channel_type,
                                    "error_rate": rate,
                                    "coverage": coverage,
                                    "seed": seed,
                                    "psnr": round(psnr, 6),
                                    "ssim": round(ssim, 6),
                                    "important_block_recovery": round(important_recovery_from_flags(scheduled, success_flags), 6),
                                    "rs_failure_rate": round(fail_rate, 6),
                                    "full_image_success": 1.0 if all(success_flags) else 0.0,
                                    "runtime_seconds": round(time.perf_counter() - start, 6),
                                }
                            )

    raw_name = "dna_channel_sweep.csv" if args.smoke else "dna_channel_sweep_raw.csv"
    write_csv(results_dir(args) / raw_name, rows)
    write_summary(rows, results_dir(args) / "dna_channel_sweep_summary.csv")
    for metric, suffix, ylabel in [
        ("psnr", "psnr", "mean PSNR"),
        ("ssim", "ssim", "mean SSIM"),
        ("important_block_recovery", "recovery", "important recovery"),
    ]:
        labels, values = aggregate_metric(rows, metric)
        simple_bar_plot(figures_dir(args) / f"dna_channel_sweep_{suffix}.png", f"dna_channel_sweep_{suffix}", labels, values, ylabel)
    if not args.smoke:
        write_latex_table(results_dir(args) / "dna_channel_sweep_summary.csv", args.output_dir / "tables" / "generated" / "table_dna_channel_sweep.tex")
    print(f"Wrote {len(rows)} rows to {results_dir(args) / raw_name}")


if __name__ == "__main__":
    main()
