from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from common.experiment_core import add_common_args, figures_dir, results_dir, run_capacity_methods
from common.image_io import load_rgb_image, prepare_images
from common.plotting import write_csv
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks


METRICS = ["psnr", "ssim", "important_block_recovery", "rs_failure_rate", "full_image_success"]


def has_cryptography() -> bool:
    try:
        import cryptography  # noqa: F401
        return True
    except Exception:
        return False


def write_summary(rows: list[dict], output_path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["channel_type"], row["error_rate"])].append(row)
    out = []
    for (method, channel_type, error_rate), items in sorted(grouped.items()):
        images = {r["image"] for r in items}
        seeds = {r["seed"] for r in items}
        for metric in METRICS:
            vals = [float(r[metric]) for r in items]
            out.append(
                {
                    "method": method,
                    "channel_type": channel_type,
                    "error_rate": error_rate,
                    "metric": metric,
                    "mean": round(mean(vals), 6),
                    "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
                    "num_images": len(images),
                    "num_seeds": len(seeds),
                    "num_runs": len(items),
                }
            )
    write_csv(output_path, out)


def write_budget_check(args: argparse.Namespace, methods: list[str], output_path: Path) -> None:
    images = prepare_images(args.image_folder, args.output_dir, args.smoke, tuple(int(x) for x in args.resize.split(",")))
    rows = []
    for method in methods:
        total_blocks = 0
        total_parity = 0
        total_payload_length = 0
        total_container_length = 0
        total_data_length = 0
        for image_path in images:
            image = load_rgb_image(image_path, tuple(int(x) for x in args.resize.split(",")))
            blocks, _, _ = decompose_bitplane_blocks(image, args.block_size)
            scheduled = apply_schedule(blocks, method, seed=0)
            total_blocks += len(scheduled)
            total_parity += sum(b.parity for b in scheduled)
            total_payload_length += sum((len(b.data) + b.parity) * 4 for b in scheduled)
            total_container_length += len(scheduled) * 160
            total_data_length += sum(len(b.data) * 4 for b in scheduled)
        rows.append(
            {
                "method": method,
                "average_parity": round(total_parity / max(1, total_blocks), 6),
                "total_payload_length": total_payload_length,
                "total_container_length": total_container_length,
                "relative_overhead": round((total_container_length / max(1, total_data_length)) - 1.0, 6),
            }
        )
    write_csv(output_path, rows)


def write_latex_table(summary_path: Path, table_path: Path, error_rate: str = "0.02") -> None:
    import csv

    by_method: dict[str, dict[str, str]] = defaultdict(dict)
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row["error_rate"]) == error_rate:
                by_method[row["method"]][row["metric"]] = f'{float(row["mean"]):.4f} $\\pm$ {float(row["std"]):.4f}'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Equal-budget baseline comparison under simulated DNA base substitution at 2\\% error rate. Results are obtained from in-silico simulations.}",
        "\\label{tab:equal_budget_baselines}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & PSNR & SSIM & Imp. rec. & RS fail. & Full success \\\\",
        "\\midrule",
    ]
    for method in sorted(by_method):
        m = by_method[method]
        display_method = method.replace("_", "\\_")
        lines.append(
            f"{display_method} & {m.get('psnr','NA')} & {m.get('ssim','NA')} & {m.get('important_block_recovery','NA')} & {m.get('rs_failure_rate','NA')} & {m.get('full_image_success','NA')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""]
    table_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Equal-budget UEP baseline comparison.")
    add_common_args(parser)
    args = parser.parse_args()
    aes_available = has_cryptography()
    methods = [
        "no_rs",
        "uniform_rs4",
        "uniform_mixed_rs4_5",
        "static_uep",
        "random_budget_neutral_uep",
        "keyed_uep",
    ]
    if aes_available:
        methods += ["aes_static_uep", "aes_keyed_uep"]
    else:
        methods += ["cist_stream_static_uep", "cist_stream_keyed_uep"]
        print("cryptography is not installed; AES-CTR/ChaCha20 baselines were not run. Using CIS-T stream-based substitute labels.")
    raw_name = "equal_budget_baselines.csv" if args.smoke else "equal_budget_baselines_raw.csv"
    rows = run_capacity_methods(args, methods, raw_name, "equal_budget_baselines.png")
    write_summary(rows, results_dir(args) / "equal_budget_baselines_summary.csv")
    write_budget_check(args, methods, results_dir(args) / "equal_budget_budget_check.csv")
    if not args.smoke:
        write_latex_table(
            results_dir(args) / "equal_budget_baselines_summary.csv",
            args.output_dir / "tables" / "generated" / "table_equal_budget_baselines.tex",
        )
    print(f"Wrote {len(rows)} rows to {results_dir(args) / raw_name}")


if __name__ == "__main__":
    main()
