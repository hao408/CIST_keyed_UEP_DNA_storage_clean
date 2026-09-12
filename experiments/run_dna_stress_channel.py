from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from random import Random
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.experiment_core import add_common_args, figures_dir, parse_rates, parse_seeds, resize_tuple, results_dir
from common.image_io import load_rgb_image, prepare_images
from common.plotting import write_csv
from common.security_features import make_payload, max_homopolymer
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks

METHODS = ["uniform_rs4", "static_uep", "random_budget_neutral_uep", "keyed_uep"]
METRICS = ["psnr", "ssim", "important_block_recovery", "rs_failure_rate", "full_image_success"]


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def consensus_error_prob(p: float, coverage: int) -> float:
    if coverage <= 0:
        return 1.0
    if coverage == 1:
        return p
    threshold = coverage // 2 + 1
    total = 0.0
    for k in range(threshold, coverage + 1):
        total += math.comb(coverage, k) * (p**k) * ((1 - p) ** (coverage - k))
    return min(1.0, max(0.0, total))


def sequence_factors(payload: str) -> tuple[float, float, float]:
    gc = (payload.count("G") + payload.count("C")) / max(1, len(payload))
    hp = max_homopolymer(payload)
    gc_factor = 1.0 + 1.6 * abs(gc - 0.50) / 0.50
    hp_factor = 1.0 + 0.35 * max(0, hp - 3)
    dropout_factor = 0.75 * gc_factor + 0.25 * hp_factor
    return gc_factor, hp_factor, dropout_factor


def draw_binomial(rng: Random, n: int, p: float) -> int:
    p = min(1.0, max(0.0, p))
    return sum(1 for _ in range(n) if rng.random() < p)


def simulate_stress_flags(blocks, error_rate: float, coverage: int, seed: int):
    rng = Random(f"stress:{error_rate}:{coverage}:{seed}")
    np_rng = np.random.default_rng(abs(hash(("stress", error_rate, coverage, seed))) % (2**32))
    flags = []
    failures = 0
    coverage_values = []
    for block in blocks:
        payload = make_payload(block, "fixed_encrypted_dummy_padding", seed)
        gc_factor, hp_factor, dropout_factor = sequence_factors(payload)
        cov_i = int(np_rng.poisson(max(0.1, coverage)))
        coverage_values.append(cov_i)
        if cov_i <= 0:
            flags.append(False)
            failures += 1
            continue
        dropout_p = min(0.95, error_rate * 0.35 * dropout_factor)
        if rng.random() < dropout_p:
            flags.append(False)
            failures += 1
            continue
        base_p = min(0.45, error_rate * gc_factor)
        indel_p = min(0.20, error_rate * 0.18 * hp_factor)
        residual_sub = consensus_error_prob(base_p, cov_i)
        # Homopolymer-associated residual indels are modeled as a hard residual
        # alignment failure for the affected block. This is a stress test, not a
        # calibrated sequencing simulator.
        hard_indel = rng.random() < (1.0 - (1.0 - indel_p) ** max(1, len(payload) // 12))
        n_bytes = len(block.data) + block.parity
        byte_error_prob = 1.0 - (1.0 - residual_sub) ** 4
        observed_symbol_errors = draw_binomial(rng, n_bytes, byte_error_prob)
        capacity = block.parity // 2
        ok = (not hard_indel) and observed_symbol_errors <= capacity
        flags.append(ok)
        failures += 0 if ok else 1
    return flags, failures / max(1, len(blocks)), mean(coverage_values) if coverage_values else 0.0


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


def image_metrics(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    mse = float(np.mean(diff * diff))
    psnr = 99.0 if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    x = original.astype(np.float32).mean(axis=2).ravel()
    y = reconstructed.astype(np.float32).mean(axis=2).ravel()
    mux, muy = float(x.mean()), float(y.mean())
    vx, vy = float(((x - mux) ** 2).mean()), float(((y - muy) ** 2).mean())
    cov = float(((x - mux) * (y - muy)).mean())
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))
    return psnr, min(1.0, max(0.0, float(ssim)))


def important_recovery(blocks, flags) -> float:
    selected = [ok for b, ok in zip(blocks, flags) if b.base_level == "H"]
    return sum(1 for ok in selected if ok) / max(1, len(selected))


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["method"], r["channel_type"], r["error_rate"], r["coverage"])].append(r)
    out = []
    for (method, channel_type, error_rate, coverage), items in sorted(grouped.items()):
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
                    "num_images": len(set(r["image"] for r in items)),
                    "num_seeds": len(set(r["seed"] for r in items)),
                    "num_runs": len(items),
                }
            )
    return out


def plot_metric(summary, metric, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = METHODS
    colors = {"uniform_rs4": "#B4C0E4", "static_uep": "#42949E", "random_budget_neutral_uep": "#9A4D8E", "keyed_uep": "#0F4D92"}
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for method in methods:
        sub = [r for r in summary if r["method"] == method and r["metric"] == metric and int(r["coverage"]) == 30]
        sub = sorted(sub, key=lambda r: float(r["error_rate"]))
        if not sub:
            continue
        x = [float(r["error_rate"]) * 100 for r in sub]
        y = [float(r["mean"]) for r in sub]
        e = [float(r["std"]) for r in sub]
        ax.plot(x, y, marker="o", markersize=3.2, linewidth=1.4, color=colors[method], label=method.replace("_", " "))
        ax.fill_between(x, np.asarray(y) - np.asarray(e), np.asarray(y) + np.asarray(e), color=colors[method], alpha=0.12, linewidth=0)
    ax.set_xlabel("Nominal error rate (%)")
    ax.set_ylabel(metric.replace("_", "-"))
    if metric in {"important_block_recovery", "ssim"}:
        ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_table(summary, path):
    selected = [r for r in summary if r["metric"] in {"psnr", "important_block_recovery", "rs_failure_rate"} and str(r["error_rate"]) == "0.02" and str(r["coverage"]) == "30"]
    by = defaultdict(dict)
    for r in selected:
        by[r["method"]][r["metric"]] = r
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Sequence-property-dependent in-silico DNA channel stress test at 2\% nominal error rate and 30$\times$ mean coverage. The model combines Poisson coverage, GC/homopolymer-dependent residual errors, and non-uniform dropout; it is not calibrated wet-lab sequencing.}",
        r"\label{tab:dna_stress_channel}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & PSNR (dB) & Imp. rec. & Block fail. \\",
        r"\midrule",
    ]
    for method in METHODS:
        m = by.get(method, {})
        if not m:
            continue
        lines.append(
            f"{method.replace('_', ' ')} & {m['psnr']['mean']:.4f} $\\pm$ {m['psnr']['std']:.4f} & "
            f"{m['important_block_recovery']['mean']:.4f} $\\pm$ {m['important_block_recovery']['std']:.4f} & "
            f"{m['rs_failure_rate']['mean']:.4f} $\\pm$ {m['rs_failure_rate']['std']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Sequence-property-dependent in-silico DNA channel stress test.")
    add_common_args(parser)
    parser.add_argument("--coverages", default="10,30,50")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.error_rates = "0.01"
        args.seeds = "1"
        args.coverages = "10"
        args.methods = "keyed_uep"
        resize = (64, 64)
    else:
        resize = resize_tuple(args.resize)
    if args.pilot:
        args.methods = "uniform_rs4,static_uep,keyed_uep"
        args.error_rates = "0.01,0.03"
        args.coverages = "10,30"
        args.seeds = "0,1,2"
    rows = []
    for image_path in prepare_images(args.image_folder, args.output_dir, args.smoke, resize):
        image = load_rgb_image(image_path, resize)
        original = np.asarray(image, dtype=np.uint8)
        blocks, _, _ = decompose_bitplane_blocks(image, args.block_size)
        for method in parse_list(args.methods):
            for rate in parse_rates(args.error_rates):
                for coverage in [int(x) for x in parse_list(args.coverages)]:
                    for seed in parse_seeds(args.seeds):
                        start = time.perf_counter()
                        scheduled = apply_schedule(blocks, method, seed=seed)
                        flags, fail_rate, observed_cov = simulate_stress_flags(scheduled, rate, coverage, seed)
                        recon = reconstruct_array_from_flags(original, scheduled, flags, args.block_size)
                        psnr, ssim = image_metrics(original, recon)
                        rows.append(
                            {
                                "dataset": "smoke" if args.smoke else args.image_folder.name,
                                "image": image_path.name,
                                "method": method,
                                "channel_type": "sequence_property_stress",
                                "error_rate": rate,
                                "coverage": coverage,
                                "observed_mean_coverage": round(observed_cov, 4),
                                "seed": seed,
                                "psnr": round(psnr, 6),
                                "ssim": round(ssim, 6),
                                "important_block_recovery": round(important_recovery(scheduled, flags), 6),
                                "rs_failure_rate": round(fail_rate, 6),
                                "full_image_success": 1.0 if all(flags) else 0.0,
                                "runtime_seconds": round(time.perf_counter() - start, 6),
                            }
                        )
    summary = summarize(rows)
    res = results_dir(args)
    figs = figures_dir(args)
    write_csv(res / "dna_stress_channel_raw.csv", rows)
    write_csv(res / "dna_stress_channel_summary.csv", summary)
    plot_metric(summary, "psnr", figs / "dna_stress_channel_psnr")
    plot_metric(summary, "important_block_recovery", figs / "dna_stress_channel_recovery")
    write_table(summary, args.output_dir / "tables" / "generated" / "table_dna_stress_channel.tex")
    print(f"Wrote {len(rows)} rows to {res / 'dna_stress_channel_raw.csv'}")


if __name__ == "__main__":
    main()
