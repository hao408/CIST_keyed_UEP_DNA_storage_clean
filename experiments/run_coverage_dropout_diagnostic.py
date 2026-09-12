from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from common.image_io import load_rgb_image, prepare_images
from common.nature_style import METHOD_COLORS, apply_nature_style, clean_axis, save_figure
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks


METHODS = ["uniform_rs4", "static_uep", "keyed_uep", "uniform_rs8"]
METHOD_LABELS = {
    "uniform_rs4": "Uniform RS-4",
    "static_uep": "Static UEP",
    "keyed_uep": "Keyed UEP",
    "uniform_rs8": "Uniform RS-8",
}
MAIN_METHODS = ["uniform_rs4", "static_uep", "keyed_uep"]
TABLE_METHODS = ["uniform_rs4", "static_uep", "keyed_uep", "uniform_rs8"]
METRICS = [
    "important_block_recovery",
    "block_failure_rate",
    "full_image_success",
    "missing_oligo_fraction",
    "recovered_oligo_fraction",
    "mean_coverage_after_dropout",
    "psnr",
    "ssim",
]


def parse_list(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def consensus_error_prob(error_rate: float, coverage: int) -> float:
    if error_rate <= 0.0:
        return 0.0
    if coverage <= 1:
        return error_rate
    threshold = coverage // 2 + 1
    prob = 0.0
    for k in range(threshold, coverage + 1):
        prob += math.comb(coverage, k) * (error_rate**k) * ((1.0 - error_rate) ** (coverage - k))
    return min(max(prob, 0.0), 1.0)


def draw_binomial(rng: Random, n: int, p: float) -> int:
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    return sum(1 for _ in range(n) if rng.random() < p)


def poisson_knuth(rng: Random, lam: float) -> int:
    if lam <= 0.0:
        return 0
    if lam > 80.0:
        # Normal approximation is not used in the requested coverage range, but
        # keeps the helper stable for future larger coverage values.
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    limit = math.exp(-lam)
    k = 0
    product = 1.0
    while product > limit:
        k += 1
        product *= rng.random()
    return k - 1


def simulate_coverage_dropout(blocks, method: str, coverage: int, dropout: float, seed: int, base_error_rate: float):
    rng = Random(f"coverage-dropout:{method}:{coverage}:{dropout}:{seed}:{base_error_rate}")
    success_flags = []
    read_counts = []
    failures = 0
    for block in blocks:
        dropped = rng.random() < dropout
        read_count = 0 if dropped else poisson_knuth(rng, coverage)
        read_counts.append(read_count)
        if read_count <= 0:
            success_flags.append(False)
            failures += 1
            continue

        consensus_p = consensus_error_prob(base_error_rate, read_count)
        byte_error_prob = 1.0 - (1.0 - consensus_p) ** 4
        n_bytes = len(block.data) + block.parity
        observed_symbol_errors = draw_binomial(rng, n_bytes, byte_error_prob)
        ok = observed_symbol_errors <= (block.parity // 2)
        success_flags.append(ok)
        failures += 0 if ok else 1

    read_arr = np.asarray(read_counts, dtype=float)
    missing = float(np.mean(read_arr <= 0))
    return {
        "success_flags": success_flags,
        "missing_oligo_fraction": missing,
        "recovered_oligo_fraction": 1.0 - missing,
        "mean_coverage_after_dropout": float(np.mean(read_arr)),
        "block_failure_rate": failures / max(1, len(blocks)),
    }


def reconstruct_from_flags(original: np.ndarray, blocks, success_flags, block_size: int) -> np.ndarray:
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
    mux = float(x.mean())
    muy = float(y.mean())
    vx = float(((x - mux) ** 2).mean())
    vy = float(((y - muy) ** 2).mean())
    cov = float(((x - mux) * (y - muy)).mean())
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))
    return psnr, float(ssim)


def important_recovery(blocks, success_flags) -> float:
    selected = [ok for block, ok in zip(blocks, success_flags) if block.base_level == "H"]
    return sum(1 for ok in selected if ok) / max(1, len(selected))


def summarize(raw_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (row["method"], row["channel_type"], row["coverage"], row["dropout_rate"], row["base_error_rate"])
        grouped[key].append(row)
    out = []
    for (method, channel_type, coverage, dropout, base_error), items in sorted(grouped.items()):
        images = {r["image"] for r in items}
        seeds = {r["seed"] for r in items}
        for metric in METRICS:
            vals = [float(r[metric]) for r in items]
            out.append(
                {
                    "method": method,
                    "channel_type": channel_type,
                    "coverage": coverage,
                    "dropout_rate": dropout,
                    "base_error_rate": base_error,
                    "metric": metric,
                    "mean": round(mean(vals), 6),
                    "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
                    "num_images": len(images),
                    "num_seeds": len(seeds),
                    "num_runs": len(items),
                }
            )
    return out


def heatmap_values(summary_rows: list[dict], method: str, metric: str, base_error_rate: float, coverages: list[int], dropouts: list[float]):
    lookup = {}
    for row in summary_rows:
        if row["method"] == method and row["metric"] == metric and float(row["base_error_rate"]) == base_error_rate:
            lookup[(int(row["coverage"]), float(row["dropout_rate"]))] = float(row["mean"])
    return np.asarray([[lookup[(c, d)] for c in coverages] for d in dropouts], dtype=float)


def plot_heatmaps(summary_rows: list[dict], out_stem: Path, metric: str, title: str, coverages: list[int], dropouts: list[float], base_error_rate: float, cmap: str, vmin: float, vmax: float):
    apply_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), constrained_layout=True)
    ims = []
    for ax, method, panel in zip(axes, MAIN_METHODS, ["a", "b", "c"]):
        values = heatmap_values(summary_rows, method, metric, base_error_rate, coverages, dropouts)
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ims.append(im)
        ax.set_title(METHOD_LABELS[method])
        ax.set_xticks(range(len(coverages)), [f"{c}x" for c in coverages])
        ax.set_yticks(range(len(dropouts)), [f"{int(d * 100)}%" for d in dropouts])
        ax.set_xlabel("Coverage")
        if ax is axes[0]:
            ax.set_ylabel("Dropout")
        else:
            ax.set_ylabel("")
        ax.text(-0.18, 1.08, panel, transform=ax.transAxes, fontweight="bold", fontsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.88, pad=0.018)
    cbar.set_label(title)
    save_figure(fig, out_stem)
    plt.close(fig)


def write_latex_table(summary_rows: list[dict], out_path: Path) -> None:
    selected = {}
    for row in summary_rows:
        if (
            float(row["base_error_rate"]) == 0.01
            and row["metric"] in {"important_block_recovery", "block_failure_rate", "full_image_success"}
            and int(row["coverage"]) in {5, 10, 20, 30}
            and float(row["dropout_rate"]) in {0.0, 0.1, 0.2}
            and row["method"] in TABLE_METHODS
        ):
            key = (row["method"], int(row["coverage"]), float(row["dropout_rate"]))
            selected.setdefault(key, {})[row["metric"]] = (float(row["mean"]), float(row["std"]))
    lines = [
        "\\begin{center}",
        "\\captionof{table}{Coverage/dropout robustness diagnostic under oligo-level sampling with 1\\% simulated base substitution. Values are mean important-block recovery; parentheses give block-failure rate. The experiment uses five representative RGB images, seeds 0--9, fixed 208-nt indexed oligos, 160-base payload containers, and simulated coverage/dropout sampling plus the stated base-substitution consensus approximation. Uniform RS-8 is included only as a full-parity upper bound, not as an equal-budget baseline. This is an in-silico storage-availability diagnostic, not wet-lab validation.}",
        "\\label{tab:coverage_dropout}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Method & Dropout & 5$\\times$ & 10$\\times$ & 20$\\times$ & 30$\\times$ \\\\",
        "\\midrule",
    ]
    for method in TABLE_METHODS:
        for dropout in [0.0, 0.1, 0.2]:
            cells = []
            for coverage in [5, 10, 20, 30]:
                m = selected[(method, coverage, dropout)]
                rec = m["important_block_recovery"][0]
                fail = m["block_failure_rate"][0]
                cells.append(f"{rec:.4f} ({fail:.4f})")
            lines.append(f"{METHOD_LABELS[method]} & {int(dropout * 100)}\\% & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{center}", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Oligo-level coverage/dropout robustness diagnostic.")
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--resize", default="256,256")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--coverages", default="3,5,10,20,30")
    parser.add_argument("--dropout-rates", default="0,0.01,0.05,0.10,0.20")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--base-error-rates", default="0,0.01")
    args = parser.parse_args()

    width, height = [int(x.strip()) for x in args.resize.split(",")]
    coverages = [int(x) for x in parse_list(args.coverages, int)]
    dropouts = parse_list(args.dropout_rates, float)
    seeds = parse_list(args.seeds, int)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    base_error_rates = parse_list(args.base_error_rates, float)

    results_dir = args.paper_dir / "results" / "full_simulation"
    figures_dir = args.paper_dir / "figures" / "full_simulation"
    tables_dir = args.paper_dir / "tables" / "generated"
    rows = []
    start_all = time.perf_counter()
    for image_path in prepare_images(args.image_folder, args.paper_dir, False, (width, height)):
        img = load_rgb_image(image_path, (width, height))
        original = np.asarray(img, dtype=np.uint8)
        blocks, _, _ = decompose_bitplane_blocks(img, args.block_size)
        for method in methods:
            for seed in seeds:
                scheduled = apply_schedule(blocks, method, seed=seed)
                mean_parity = mean(b.parity for b in scheduled)
                for coverage in coverages:
                    for dropout in dropouts:
                        for base_error_rate in base_error_rates:
                            tic = time.perf_counter()
                            sim = simulate_coverage_dropout(scheduled, method, coverage, dropout, seed, base_error_rate)
                            rec = reconstruct_from_flags(original, scheduled, sim["success_flags"], args.block_size)
                            psnr, ssim = image_metrics(original, rec)
                            channel_type = "coverage_dropout_only" if base_error_rate == 0.0 else "coverage_dropout_plus_1pct_substitution"
                            rows.append(
                                {
                                    "dataset": args.image_folder.name,
                                    "image": image_path.name,
                                    "method": method,
                                    "method_label": METHOD_LABELS.get(method, method),
                                    "channel_type": channel_type,
                                    "coverage": coverage,
                                    "dropout_rate": dropout,
                                    "base_error_rate": base_error_rate,
                                    "seed": seed,
                                    "image_size": f"{width}x{height}",
                                    "block_size": args.block_size,
                                    "parity_profile": "8/4/2",
                                    "container_design": "full_container_masked_fixed_payload",
                                    "oligo_length_nt": 208,
                                    "payload_bases": 160,
                                    "index_bases": 8,
                                    "primer_bases": 40,
                                    "mean_parity": round(mean_parity, 6),
                                    "important_block_recovery": round(important_recovery(scheduled, sim["success_flags"]), 6),
                                    "block_failure_rate": round(sim["block_failure_rate"], 6),
                                    "full_image_success": int(all(sim["success_flags"])),
                                    "missing_oligo_fraction": round(sim["missing_oligo_fraction"], 6),
                                    "recovered_oligo_fraction": round(sim["recovered_oligo_fraction"], 6),
                                    "mean_coverage_after_dropout": round(sim["mean_coverage_after_dropout"], 6),
                                    "psnr": round(psnr, 6),
                                    "ssim": round(ssim, 6),
                                    "runtime_seconds": round(time.perf_counter() - tic, 6),
                                }
                            )

    raw_path = results_dir / "coverage_dropout_raw.csv"
    summary_path = results_dir / "coverage_dropout_summary.csv"
    write_csv(raw_path, rows)
    summary_rows = summarize(rows)
    write_csv(summary_path, summary_rows)

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(
        summary_rows,
        figures_dir / "coverage_dropout_important_recovery",
        "important_block_recovery",
        "Important-block recovery",
        coverages,
        dropouts,
        0.01,
        "viridis",
        0.0,
        1.0,
    )
    plot_heatmaps(
        summary_rows,
        figures_dir / "coverage_dropout_full_image_success",
        "full_image_success",
        "Full-image success",
        coverages,
        dropouts,
        0.01,
        "mako" if False else "YlGnBu",
        0.0,
        1.0,
    )
    plot_heatmaps(
        summary_rows,
        figures_dir / "coverage_dropout_block_failure",
        "block_failure_rate",
        "Block failure rate",
        coverages,
        dropouts,
        0.01,
        "magma_r",
        0.0,
        1.0,
    )
    write_latex_table(summary_rows, tables_dir / "table_coverage_dropout.tex")
    print(f"Wrote {len(rows)} raw rows to {raw_path}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")
    print(f"Runtime seconds: {time.perf_counter() - start_all:.3f}")


if __name__ == "__main__":
    main()
