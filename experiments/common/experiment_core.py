from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, List, Sequence, Tuple

from .image_io import load_rgb_image, prepare_images
from .metrics import full_image_success, global_ssim, important_block_recovery, mse_psnr
from .plotting import simple_bar_plot, write_csv
from .rs_codec import simulate_rs_capacity
from .uep_scheduler import apply_schedule, decompose_bitplane_blocks, reconstruct_image


def parse_rates(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-folder", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resize", default="64,64")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--error-rates", default="0.001,0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--seeds", default="1,2,3")


def resize_tuple(text: str) -> Tuple[int, int]:
    w, h = [int(x.strip()) for x in text.split(",")]
    return (w, h)


def results_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / "results" / ("smoke" if args.smoke else "full_simulation")


def figures_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / "figures" / ("smoke" if args.smoke else "full_simulation")


def run_capacity_methods(args: argparse.Namespace, methods: Sequence[str], csv_name: str, fig_name: str) -> List[dict]:
    output_dir = args.output_dir
    resize = resize_tuple(args.resize)
    if args.smoke:
        args.error_rates = "0.01"
        args.seeds = "1"
        resize = (64, 64)
    images = prepare_images(args.image_folder, output_dir, args.smoke, resize)
    rows: List[dict] = []
    for image_path in images:
        image = load_rgb_image(image_path, resize)
        blocks, original_size, padded_size = decompose_bitplane_blocks(image, args.block_size)
        for method in methods:
            for rate in parse_rates(args.error_rates):
                for seed in parse_seeds(args.seeds):
                    start = time.perf_counter()
                    scheduled = apply_schedule(blocks, method, seed=seed)
                    recovered_blocks, rs_failure_rate = simulate_rs_capacity(scheduled, rate, seed)
                    recon = reconstruct_image(recovered_blocks, original_size, padded_size, args.block_size)
                    mse, psnr = mse_psnr(image, recon)
                    runtime = time.perf_counter() - start
                    rows.append(
                        {
                            "dataset": "smoke" if args.smoke else (args.image_folder.name if args.image_folder else "custom"),
                            "image": image_path.name,
                            "method": method,
                            "channel_type": "simulated_dna_base_substitution",
                            "error_rate": rate,
                            "coverage": "NA",
                            "seed": seed,
                            "block_size": args.block_size,
                            "codec_model": "rs_capacity_proxy",
                            "mean_parity": round(mean(b.parity for b in scheduled), 6),
                            "mse": round(mse, 6),
                            "psnr": round(psnr, 6),
                            "ssim": round(global_ssim(image, recon), 6),
                            "important_block_recovery": round(important_block_recovery(scheduled, recovered_blocks), 6),
                            "rs_failure_rate": round(rs_failure_rate, 6),
                            "full_image_success": full_image_success(scheduled, recovered_blocks),
                            "runtime_seconds": round(runtime, 6),
                        }
                    )
    results_path = results_dir(args) / csv_name
    write_csv(results_path, rows)
    labels, values = aggregate_for_plot(rows, "method", "psnr")
    simple_bar_plot(figures_dir(args) / fig_name, fig_name, labels, values, "mean PSNR")
    return rows


def aggregate_for_plot(rows: Sequence[dict], group_key: str, value_key: str) -> Tuple[List[str], List[float]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(float(row[value_key]))
    labels = sorted(groups)
    values = [mean(groups[label]) for label in labels]
    return labels, values


def write_aggregate(rows: Sequence[dict], output_path: Path, group_keys: Sequence[str], metrics: Sequence[str]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        record = {k: v for k, v in zip(group_keys, key)}
        for metric in metrics:
            vals = [float(r[metric]) for r in items]
            record[f"{metric}_mean"] = round(mean(vals), 6)
            record[f"{metric}_std"] = round(pstdev(vals), 6) if len(vals) > 1 else 0.0
        out.append(record)
    write_csv(output_path, out)
