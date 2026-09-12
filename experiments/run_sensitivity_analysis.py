from __future__ import annotations

import argparse
import time
from collections import defaultdict
from statistics import mean, pstdev
from pathlib import Path

from common.experiment_core import add_common_args, figures_dir, parse_rates, parse_seeds, resize_tuple, results_dir
from common.image_io import load_rgb_image, prepare_images
from common.metrics import full_image_success, global_ssim, important_block_recovery, mse_psnr
from common.plotting import simple_bar_plot, write_csv
from common.rs_codec import simulate_rs_capacity
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks, reconstruct_image


METRICS = ["psnr", "ssim", "important_block_recovery", "rs_failure_rate", "full_image_success"]
DEFAULT_BLOCK = 16
DEFAULT_PROFILE = "8/4/2"
DEFAULT_PERTURB = "10/20"


def parse_profile(text: str) -> tuple[int, int, int]:
    vals = tuple(int(x.strip()) for x in text.split("/"))
    if len(vals) != 3:
        raise ValueError(f"Expected H/M/L profile, got {text}")
    return vals


def parse_perturbation(text: str) -> tuple[float, float]:
    up, down = [float(x.strip()) / 100.0 for x in text.split("/")]
    return up, down


def split_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def build_one_factor_configs(block_sizes: list[int], profiles: list[str], perturbations: list[str]) -> list[dict]:
    configs = []
    seen = set()

    def add(factor: str, block_size: int, profile: str, perturb: str) -> None:
        key = (factor, block_size, profile, perturb)
        if key in seen:
            return
        seen.add(key)
        configs.append(
            {
                "sweep_factor": factor,
                "block_size": block_size,
                "parity_profile": profile,
                "perturbation_ratio": perturb,
            }
        )

    for block_size in block_sizes:
        add("block_size", block_size, DEFAULT_PROFILE, DEFAULT_PERTURB)
    for profile in profiles:
        add("parity_profile", DEFAULT_BLOCK, profile, DEFAULT_PERTURB)
    for perturb in perturbations:
        add("perturbation_ratio", DEFAULT_BLOCK, DEFAULT_PROFILE, perturb)
    return configs


def write_summary(rows: list[dict], out_path: Path) -> list[dict]:
    grouped = defaultdict(list)
    keys = ["method", "channel_type", "error_rate", "block_size", "parity_profile", "perturbation_ratio", "sweep_factor"]
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda kv: (kv[0][-1], float(kv[0][2]), int(kv[0][3]), kv[0][4], kv[0][5])):
        base = {k: v for k, v in zip(keys, key)}
        for metric in METRICS:
            vals = [float(r[metric]) for r in items]
            record = dict(base)
            record.update(
                {
                    "metric": metric,
                    "mean": round(mean(vals), 6),
                    "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
                    "num_images": len(set(r["image"] for r in items)),
                    "num_seeds": len(set(r["seed"] for r in items)),
                    "num_runs": len(items),
                }
            )
            out.append(record)
    write_csv(out_path, out)
    return out


def label_for_factor(row: dict, factor: str) -> str:
    if factor == "block_size":
        return str(row["block_size"])
    if factor == "parity_profile":
        return row["parity_profile"]
    return row["perturbation_ratio"]


def plot_factor(summary_rows: list[dict], factor: str, out_path: Path, metric: str = "psnr", fixed_rate: str = "0.02") -> None:
    selected = [
        r
        for r in summary_rows
        if r["sweep_factor"] == factor and r["metric"] == metric and str(r["error_rate"]) == fixed_rate
    ]
    if not selected:
        selected = [r for r in summary_rows if r["sweep_factor"] == factor and r["metric"] == metric]
    labels = [label_for_factor(r, factor) for r in selected]
    values = [float(r["mean"]) for r in selected]
    simple_bar_plot(out_path, f"sensitivity_{factor}", labels, values, f"mean {metric}")


def write_latex_table(summary_rows: list[dict], out_path: Path, fixed_rate: str = "0.02") -> None:
    rows = [
        r
        for r in summary_rows
        if str(r["error_rate"]) == fixed_rate
        and r["metric"] in {"psnr", "ssim", "important_block_recovery", "rs_failure_rate"}
    ]
    grouped = defaultdict(dict)
    for row in rows:
        label = label_for_factor(row, row["sweep_factor"])
        grouped[(row["sweep_factor"], label)][row["metric"]] = row
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{One-factor-at-a-time sensitivity analysis under simulated DNA base substitution at 2\% error rate. The default configuration is block size 16, parity profile 8/4/2, and perturbation ratio 10/20.}",
        r"\label{tab:sensitivity_analysis}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Sweep & Setting & PSNR (dB) & SSIM & Imp. rec. & Block fail. \\",
        r"\midrule",
    ]
    factor_names = {"block_size": "Block size", "parity_profile": "Parity profile", "perturbation_ratio": "Perturbation"}
    order = ["block_size", "parity_profile", "perturbation_ratio"]
    for factor in order:
        for key in sorted([k for k in grouped if k[0] == factor], key=lambda x: x[1]):
            metrics = grouped[key]
            lines.append(
                f"{factor_names[factor]} & {key[1]} & "
                f"{float(metrics['psnr']['mean']):.4f} $\\pm$ {float(metrics['psnr']['std']):.4f} & "
                f"{float(metrics['ssim']['mean']):.4f} $\\pm$ {float(metrics['ssim']['std']):.4f} & "
                f"{float(metrics['important_block_recovery']['mean']):.4f} $\\pm$ {float(metrics['important_block_recovery']['std']):.4f} & "
                f"{float(metrics['rs_failure_rate']['mean']):.4f} $\\pm$ {float(metrics['rs_failure_rate']['std']):.4f} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-factor-at-a-time sensitivity analysis for Keyed UEP.")
    add_common_args(parser)
    parser.add_argument("--block-sizes", default="8,16,32")
    parser.add_argument("--profiles", default="10/6/2,8/4/2,6/4/2,8/6/2")
    parser.add_argument("--perturbations", default="0/0,5/10,10/20,15/30,20/40")
    parser.add_argument("--method", default="keyed_uep")
    args = parser.parse_args()

    if args.smoke:
        args.error_rates = "0.01"
        args.seeds = "1"
        args.block_sizes = "16"
        args.profiles = DEFAULT_PROFILE
        args.perturbations = DEFAULT_PERTURB
        resize = (64, 64)
    else:
        resize = resize_tuple(args.resize)

    block_sizes = [int(x) for x in split_csv(args.block_sizes)]
    profiles = split_csv(args.profiles)
    perturbations = split_csv(args.perturbations)
    configs = build_one_factor_configs(block_sizes, profiles, perturbations)
    rates = parse_rates(args.error_rates)
    seeds = parse_seeds(args.seeds)

    images = prepare_images(args.image_folder, args.output_dir, args.smoke, resize)
    rows = []
    block_cache = {}
    for image_path in images:
        img = load_rgb_image(image_path, resize)
        for config in configs:
            block_size = int(config["block_size"])
            cache_key = (image_path.name, block_size)
            if cache_key not in block_cache:
                block_cache[cache_key] = decompose_bitplane_blocks(img, block_size)
            blocks, original_size, padded_size = block_cache[cache_key]
            profile = parse_profile(config["parity_profile"])
            promote, demote = parse_perturbation(config["perturbation_ratio"])
            for rate in rates:
                for seed in seeds:
                    start = time.perf_counter()
                    scheduled = apply_schedule(
                        blocks,
                        args.method,
                        seed=seed,
                        profile=profile,
                        promote_ratio=promote,
                        demote_ratio=demote,
                    )
                    recovered, fail = simulate_rs_capacity(scheduled, rate, seed)
                    recon = reconstruct_image(recovered, original_size, padded_size, block_size)
                    mse, psnr = mse_psnr(img, recon)
                    rows.append(
                        {
                            "dataset": "smoke" if args.smoke else (args.image_folder.name if args.image_folder else "custom"),
                            "image": image_path.name,
                            "method": args.method,
                            "channel_type": "simulated_dna_base_substitution",
                            "error_rate": rate,
                            "seed": seed,
                            "block_size": block_size,
                            "parity_profile": config["parity_profile"],
                            "perturbation_ratio": config["perturbation_ratio"],
                            "sweep_factor": config["sweep_factor"],
                            "psnr": round(psnr, 6),
                            "ssim": round(global_ssim(img, recon), 6),
                            "important_block_recovery": round(important_block_recovery(scheduled, recovered), 6),
                            "rs_failure_rate": round(fail, 6),
                            "full_image_success": full_image_success(scheduled, recovered),
                            "runtime_seconds": round(time.perf_counter() - start, 6),
                        }
                    )

    res = results_dir(args)
    figs = figures_dir(args)
    raw_path = res / "sensitivity_analysis_raw.csv"
    summary_path = res / "sensitivity_analysis_summary.csv"
    write_csv(raw_path, rows)
    summary = write_summary(rows, summary_path)
    plot_factor(summary, "block_size", figs / "sensitivity_block_size.png")
    plot_factor(summary, "parity_profile", figs / "sensitivity_parity_profile.png")
    plot_factor(summary, "perturbation_ratio", figs / "sensitivity_perturbation_ratio.png")
    write_latex_table(summary, args.output_dir / "tables" / "generated" / "table_sensitivity_analysis.tex")
    print(f"Wrote {len(rows)} rows to {raw_path}")


if __name__ == "__main__":
    main()
