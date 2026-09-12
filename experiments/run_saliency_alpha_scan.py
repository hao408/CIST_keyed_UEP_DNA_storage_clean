from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.experiment_core import add_common_args, parse_rates, parse_seeds, resize_tuple
from common.image_io import load_rgb_image, prepare_images
from common.metrics import global_ssim, important_block_recovery, mse_psnr
from common.plotting import write_csv
from common.rs_codec import simulate_rs_capacity
from common.uep_scheduler import decompose_bitplane_blocks, reconstruct_image
from run_saliency_keyed_uep_pilot import (
    apply_saliency_schedule,
    block_saliency,
    demoted_set,
    hamming,
    jaccard,
    promoted_set,
    salient_medium_block_recovery,
    schedule_signature,
    weighted_visual_recovery,
)


def parse_alphas(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


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
            rec.update({"metric": metric, "mean": round(mean(vals), 6), "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0, "num_images": len({r["image"] for r in items}), "num_runs": len(items)})
            out.append(rec)
    return out


def plot_tradeoff(summary: list[dict], path_base) -> None:
    alphas = sorted({float(r["alpha"]) for r in summary})
    lookup = {(float(r["alpha"]), float(r.get("error_rate", -1)), r["metric"]): r for r in summary}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    for ax, metric, ylabel in [
        (axes[0], "psnr", "PSNR at 2% (dB)"),
        (axes[1], "salient_medium_block_recovery", "Salient M-block recovery at 2%"),
        (axes[2], "promoted_jaccard", "Promoted Jaccard across keys"),
    ]:
        vals = []
        errs = []
        for a in alphas:
            key = (a, 0.02, metric) if metric != "promoted_jaccard" else (a, -1.0, metric)
            vals.append(lookup[key]["mean"])
            errs.append(lookup[key]["std"])
        axes[0 if metric == "psnr" else 1 if metric == "salient_medium_block_recovery" else 2].errorbar(alphas, vals, yerr=errs, marker="o", color="#1B7F79", linewidth=1.4, capsize=2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylabel)
        ax.grid(color="#E7E7E7", linewidth=0.7)
    fig.tight_layout(w_pad=0.8)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot alpha scan for saliency-aware keyed UEP.")
    add_common_args(parser)
    parser.add_argument("--alphas", default="0.0,0.3,0.5,0.7,0.9,1.0")
    args = parser.parse_args()
    start_all = time.perf_counter()
    resize = resize_tuple(args.resize)
    images = prepare_images(args.image_folder, args.output_dir, args.smoke, resize)
    rates = parse_rates(args.error_rates)
    seeds = parse_seeds(args.seeds)
    rows = []
    schedule_rows = []
    for image_path in images:
        image = load_rgb_image(image_path, resize)
        sal = block_saliency(image, args.block_size)
        blocks, original_size, padded_size = decompose_bitplane_blocks(image, args.block_size)
        for alpha in parse_alphas(args.alphas):
            schedules = {}
            for seed in seeds:
                scheduled = apply_saliency_schedule(blocks, sal, "saliency_aware_keyed_uep", seed, alpha, (8, 4, 2))
                schedules[seed] = scheduled
                for rate in rates:
                    recovered, fail = simulate_rs_capacity(scheduled, rate, seed)
                    recon = reconstruct_image(recovered, original_size, padded_size, args.block_size)
                    mse, psnr = mse_psnr(image, recon)
                    rows.append(
                        {
                            "dataset": args.image_folder.name if args.image_folder else "custom",
                            "image": image_path.name,
                            "alpha": alpha,
                            "error_rate": rate,
                            "seed": seed,
                            "psnr": round(psnr, 6),
                            "ssim": round(global_ssim(image, recon), 6),
                            "important_block_recovery": round(important_block_recovery(scheduled, recovered), 6),
                            "weighted_visual_recovery": round(weighted_visual_recovery(scheduled, recovered), 6),
                            "salient_medium_block_recovery": round(salient_medium_block_recovery(scheduled, recovered, sal), 6),
                            "mean_parity": round(mean(b.parity for b in scheduled), 6),
                            "promoted_fraction": round(len(promoted_set(scheduled)) / max(1, sum(1 for b in scheduled if b.bitplane == 5 and b.base_level == "M")), 6),
                            "demoted_fraction": round(len(demoted_set(scheduled)) / max(1, sum(1 for b in scheduled if b.bitplane == 3 and b.base_level == "M")), 6),
                        }
                    )
            for a, b in [(s, s) for s in seeds] + list(itertools.combinations(seeds, 2)):
                sa, sb = schedules[a], schedules[b]
                schedule_rows.append(
                    {
                        "dataset": args.image_folder.name if args.image_folder else "custom",
                        "image": image_path.name,
                        "alpha": alpha,
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
    summary = summarize(rows, ["alpha", "error_rate"], ["psnr", "ssim", "important_block_recovery", "weighted_visual_recovery", "salient_medium_block_recovery", "mean_parity", "promoted_fraction", "demoted_fraction"])
    sched_summary = summarize([r for r in schedule_rows if r["comparison_type"] == "inter_key"], ["alpha"], ["promoted_jaccard", "demoted_jaccard", "schedule_hamming_distance", "mean_parity"])
    same_summary = summarize([r for r in schedule_rows if r["comparison_type"] == "same_key"], ["alpha"], ["schedule_match_rate"])
    combined_summary = summary + sched_summary + same_summary
    write_csv(result_dir / "saliency_alpha_scan_raw.csv", rows)
    write_csv(result_dir / "saliency_alpha_scan_schedule_raw.csv", schedule_rows)
    write_csv(result_dir / "saliency_alpha_scan_summary.csv", combined_summary)
    plot_tradeoff(combined_summary, figure_dir / "saliency_alpha_scan_tradeoff")
    print(f"Wrote alpha scan rows in {time.perf_counter() - start_all:.3f} s")


if __name__ == "__main__":
    main()
