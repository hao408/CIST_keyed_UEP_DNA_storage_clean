from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from common.experiment_core import add_common_args, parse_rates, parse_seeds, resize_tuple, results_dir
from common.image_io import load_rgb_image, prepare_images
from common.metrics import full_image_success, global_ssim, important_block_recovery, mse_psnr
from common.plotting import write_csv
from common.rs_codec import simulate_rs_capacity
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks, reconstruct_image


METRICS = ["psnr", "ssim", "important_block_recovery", "rs_failure_rate", "full_image_success"]


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def output_results_dir(args: argparse.Namespace) -> Path:
    if args.profile:
        return args.output_dir / "results" / "profiling"
    return results_dir(args)


def write_summary(rows: list[dict], out_path: Path) -> list[dict]:
    grouped = defaultdict(list)
    keys = ["dataset", "method", "channel_type", "error_rate"]
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda kv: (kv[0][1], float(kv[0][3]))):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Folder-based larger dataset evaluation.")
    add_common_args(parser)
    parser.add_argument("--methods", default="uniform_rs4,static_uep,random_budget_neutral_uep,keyed_uep")
    parser.add_argument("--max-images", type=int, default=0, help="Use only the first N images after sorting; 0 means all.")
    parser.add_argument("--profile", action="store_true", help="Write to results/profiling rather than full_simulation.")
    parser.add_argument("--raw-name", default="larger_dataset_eval_raw.csv")
    parser.add_argument("--summary-name", default="larger_dataset_eval_summary.csv")
    args = parser.parse_args()

    if args.smoke:
        args.error_rates = "0.01"
        args.seeds = "1"
        resize = (64, 64)
    else:
        resize = resize_tuple(args.resize)

    methods = parse_list(args.methods)
    rates = parse_rates(args.error_rates)
    seeds = parse_seeds(args.seeds)
    images = prepare_images(args.image_folder, args.output_dir, args.smoke, resize)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    rows = []
    for image_path in images:
        img = load_rgb_image(image_path, resize)
        blocks, original_size, padded_size = decompose_bitplane_blocks(img, args.block_size)
        for method in methods:
            for rate in rates:
                for seed in seeds:
                    start = time.perf_counter()
                    scheduled = apply_schedule(blocks, method, seed=seed)
                    recovered, fail = simulate_rs_capacity(scheduled, rate, seed)
                    recon = reconstruct_image(recovered, original_size, padded_size, args.block_size)
                    mse, psnr = mse_psnr(img, recon)
                    rows.append(
                        {
                            "dataset": "smoke" if args.smoke else (args.image_folder.name if args.image_folder else "custom"),
                            "image": image_path.name,
                            "method": method,
                            "channel_type": "simulated_dna_base_substitution",
                            "error_rate": rate,
                            "seed": seed,
                            "block_size": args.block_size,
                            "parity_profile": "8/4/2",
                            "perturbation_ratio": "10/20" if method in {"keyed_uep", "random_budget_neutral_uep"} else "NA",
                            "psnr": round(psnr, 6),
                            "ssim": round(global_ssim(img, recon), 6),
                            "important_block_recovery": round(important_block_recovery(scheduled, recovered), 6),
                            "rs_failure_rate": round(fail, 6),
                            "full_image_success": full_image_success(scheduled, recovered),
                            "runtime_seconds": round(time.perf_counter() - start, 6),
                        }
                    )

    out_dir = output_results_dir(args)
    raw_path = out_dir / args.raw_name
    summary_path = out_dir / args.summary_name
    write_csv(raw_path, rows)
    write_summary(rows, summary_path)
    print(f"Wrote {len(rows)} rows to {raw_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
