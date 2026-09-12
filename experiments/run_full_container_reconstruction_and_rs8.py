from __future__ import annotations

import argparse
import time
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.experiment_core import add_common_args, parse_rates, parse_seeds, resize_tuple
from common.image_io import load_rgb_image, prepare_images
from common.metrics import full_image_success, global_ssim, important_block_recovery, mse_psnr
from common.plotting import write_csv
from common.rs_codec import simulate_rs_capacity
from common.security_features import full_container_round_trip, make_payload
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks, reconstruct_image


METHODS = ["uniform_rs4", "uniform_mixed_rs4_5", "static_uep", "random_budget_neutral_uep", "keyed_uep", "uniform_rs8"]
LABELS = {
    "uniform_rs4": "Uniform RS-4",
    "uniform_mixed_rs4_5": "Uniform RS-4/5",
    "static_uep": "Static UEP",
    "random_budget_neutral_uep": "Random UEP",
    "keyed_uep": "Keyed UEP",
    "uniform_rs8": "Uniform RS-8",
}


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
                    "num_seeds": len({r["seed"] for r in items}),
                    "num_runs": len(items),
                }
            )
            out.append(rec)
    return out


def plot_upper(summary: list[dict], path_base) -> None:
    colors = {
        "uniform_rs4": "#9BA3AF",
        "uniform_mixed_rs4_5": "#B9B0A4",
        "static_uep": "#D49344",
        "random_budget_neutral_uep": "#8A8A8A",
        "keyed_uep": "#4C78A8",
        "uniform_rs8": "#8E3B46",
    }
    metrics = [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("important_block_recovery", "Important-block recovery")]
    rates = sorted({float(r["error_rate"]) for r in summary})
    lookup = {(r["method"], float(r["error_rate"]), r["metric"]): r for r in summary}
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in METHODS:
            vals = [lookup[(method, r, metric)]["mean"] for r in rates]
            ax.plot([100 * r for r in rates], vals, marker="o", linewidth=1.3, markersize=3.0, color=colors[method], label=LABELS[method])
        ax.set_xlabel("Base substitution (%)")
        ax.set_ylabel(ylabel)
        ax.grid(color="#E7E7E7", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=5.8, ncol=1)
    fig.tight_layout(w_pad=0.7)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_upper_table(summary: list[dict], path) -> None:
    lookup = {(r["method"], round(float(r["error_rate"]), 6), r["metric"]): r for r in summary}
    er = 0.02
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full-parity upper-bound comparison under simulated DNA base substitution at 2\% error rate. Uniform RS-8 uses 8 effective parity bytes per block and is therefore not an equal-budget baseline for the UEP methods.}",
        r"\label{tab:upper_bound_rs8}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Mean parity & PSNR & SSIM & Important-block recovery \\",
        r"\midrule",
    ]
    for method in METHODS:
        mp = lookup[(method, er, "mean_parity")]
        ps = lookup[(method, er, "psnr")]
        ss = lookup[(method, er, "ssim")]
        ir = lookup[(method, er, "important_block_recovery")]
        lines.append(f"{LABELS[method]} & {float(mp['mean']):.4f} & {float(ps['mean']):.4f} $\\pm$ {float(ps['std']):.4f} & {float(ss['mean']):.4f} $\\pm$ {float(ss['std']):.4f} & {float(ir['mean']):.4f} $\\pm$ {float(ir['std']):.4f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_round_trip(args, images, resize) -> list[dict]:
    rows = []
    for image_path in images:
        image = load_rgb_image(image_path, resize)
        blocks, _, _ = decompose_bitplane_blocks(image, args.block_size)
        scheduled = apply_schedule(blocks, "keyed_uep", seed=0)
        for level in ["H", "M", "L"]:
            subset = [b for b in scheduled if b.level == level]
            checked = subset[: min(200, len(subset))]
            correct = 0
            wrong = 0
            lengths = set()
            base_lengths = set()
            for block in checked:
                payload = make_payload(block, "full_container_masked_fixed_payload", 0)
                base_lengths.add(len(payload))
                lengths.add(len(payload) // 4)
                original, recovered = full_container_round_trip(block, 0, wrong_key=False)
                _, wrong_recovered = full_container_round_trip(block, 0, wrong_key=True)
                correct += int(original == recovered)
                wrong += int(original == wrong_recovered)
            rows.append(
                {
                    "dataset": args.image_folder.name if args.image_folder else "custom",
                    "image": image_path.name,
                    "level": level,
                    "checked_blocks": len(checked),
                    "container_byte_lengths": "|".join(str(x) for x in sorted(lengths)),
                    "payload_base_lengths": "|".join(str(x) for x in sorted(base_lengths)),
                    "correct_key_match_rate": round(correct / max(1, len(checked)), 6),
                    "wrong_key_match_rate": round(wrong / max(1, len(checked)), 6),
                }
            )
    return rows


def run_reconstruction(args, images, resize, methods, rates, seeds) -> list[dict]:
    rows = []
    for image_path in images:
        image = load_rgb_image(image_path, resize)
        blocks, original_size, padded_size = decompose_bitplane_blocks(image, args.block_size)
        for method in methods:
            for rate in rates:
                for seed in seeds:
                    start = time.perf_counter()
                    scheduled = apply_schedule(blocks, method, seed=seed)
                    recovered, fail = simulate_rs_capacity(scheduled, rate, seed)
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
                            "psnr": round(psnr, 6),
                            "ssim": round(global_ssim(image, recon), 6),
                            "important_block_recovery": round(important_block_recovery(scheduled, recovered), 6),
                            "block_failure": round(fail, 6),
                            "mean_parity": round(mean(b.parity for b in scheduled), 6),
                            "full_image_success": full_image_success(scheduled, recovered),
                            "runtime_seconds": round(time.perf_counter() - start, 6),
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-container reconstruction check and RS-8 upper bound.")
    add_common_args(parser)
    args = parser.parse_args()
    resize = resize_tuple(args.resize)
    images = prepare_images(args.image_folder, args.output_dir, args.smoke, resize)
    seeds = parse_seeds(args.seeds)
    rates = parse_rates(args.error_rates)
    result_dir = args.output_dir / "results" / "full_simulation"
    figure_dir = args.output_dir / "figures" / "full_simulation"

    rt_rows = run_round_trip(args, images, resize)
    write_csv(result_dir / "full_container_noiseless_roundtrip.csv", rt_rows)

    keyed_rows = run_reconstruction(args, images, resize, ["keyed_uep"], [0.01, 0.02, 0.03], seeds)
    keyed_summary = summarize(keyed_rows, ["method", "channel_type", "error_rate"], ["psnr", "ssim", "important_block_recovery", "block_failure", "mean_parity", "full_image_success"])
    write_csv(result_dir / "full_container_reconstruction_check_raw.csv", keyed_rows)
    write_csv(result_dir / "full_container_reconstruction_check_summary.csv", keyed_summary)

    upper_rows = run_reconstruction(args, images, resize, METHODS, rates, seeds)
    upper_summary = summarize(upper_rows, ["method", "channel_type", "error_rate"], ["psnr", "ssim", "important_block_recovery", "block_failure", "mean_parity", "full_image_success"])
    write_csv(result_dir / "upper_bound_rs8_raw.csv", upper_rows)
    write_csv(result_dir / "upper_bound_rs8_summary.csv", upper_summary)
    plot_upper(upper_summary, figure_dir / "upper_bound_rs8_reliability")
    write_upper_table(upper_summary, args.output_dir / "tables" / "generated" / "table_upper_bound_rs8.tex")
    print(f"Wrote {len(rt_rows)} round-trip rows, {len(keyed_rows)} reconstruction rows, {len(upper_rows)} upper-bound rows")


if __name__ == "__main__":
    main()
