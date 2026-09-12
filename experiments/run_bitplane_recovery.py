from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from statistics import mean, pstdev
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from common.experiment_core import add_common_args, parse_rates, parse_seeds, resize_tuple, results_dir, figures_dir
from common.image_io import load_rgb_image, prepare_images
from common.metrics import global_ssim, important_block_recovery, mse_psnr
from common.plotting import write_csv
from common.rs_codec import simulate_rs_capacity
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks, reconstruct_image


METHODS = ["uniform_rs4", "static_uep", "random_budget_neutral_uep", "keyed_uep"]
METRICS = ["recovery_rate", "important_block_recovery", "psnr", "ssim"]
BITPLANES = list(range(7, -1, -1))


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def write_summary(rows: list[dict], out_path: Path) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["channel_type"], row["error_rate"], row["bitplane"])].append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda kv: (kv[0][0], float(kv[0][2]), -int(kv[0][3]))):
        method, channel_type, error_rate, bitplane = key
        for metric in METRICS:
            vals = [float(r[metric]) for r in items]
            out.append(
                {
                    "method": method,
                    "channel_type": channel_type,
                    "error_rate": error_rate,
                    "bitplane": bitplane,
                    "metric": metric,
                    "mean": round(mean(vals), 6),
                    "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
                    "num_images": len(set(r["image"] for r in items)),
                    "num_seeds": len(set(r["seed"] for r in items)),
                    "num_runs": len(items),
                }
            )
    write_csv(out_path, out)
    return out


def recovery_lookup(summary_rows: list[dict]) -> dict[tuple[str, str, int], float]:
    out = {}
    for row in summary_rows:
        if row["metric"] == "recovery_rate":
            out[(row["method"], str(row["error_rate"]), int(row["bitplane"]))] = float(row["mean"])
    return out


def save_pdf_from_image(img: Image.Image, pdf_path: Path) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=img.size)
    c.drawImage(ImageReader(img), 0, 0, width=img.size[0], height=img.size[1])
    c.showPage()
    c.save()


def draw_heatmap(path_png: Path, summary_rows: list[dict], methods: list[str], error_rates: list[float]) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    lookup = recovery_lookup(summary_rows)
    cell_w, cell_h = 70, 32
    panel_w = 110 + len(error_rates) * cell_w
    panel_h = 70 + len(BITPLANES) * cell_h
    w = panel_w * 2
    h = panel_h * math.ceil(len(methods) / 2)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for idx, method in enumerate(methods):
        ox = (idx % 2) * panel_w
        oy = (idx // 2) * panel_h
        d.text((ox + 12, oy + 12), method, fill="black")
        for j, rate in enumerate(error_rates):
            d.text((ox + 105 + j * cell_w, oy + 42), f"{float(rate):g}", fill="black")
        for i, bitplane in enumerate(BITPLANES):
            d.text((ox + 20, oy + 70 + i * cell_h + 8), f"bit {bitplane}", fill="black")
            for j, rate in enumerate(error_rates):
                val = lookup.get((method, str(rate), bitplane), lookup.get((method, f"{float(rate):.3f}", bitplane), 0.0))
                shade = 255 - int(200 * max(0.0, min(1.0, val)))
                x0 = ox + 100 + j * cell_w
                y0 = oy + 68 + i * cell_h
                d.rectangle([x0, y0, x0 + cell_w - 4, y0 + cell_h - 4], fill=(shade, 255, shade), outline=(180, 180, 180))
                d.text((x0 + 8, y0 + 8), f"{val:.2f}", fill="black")
    img.save(path_png)
    save_pdf_from_image(img, path_png.with_suffix(".pdf"))


def draw_line_plot(path_png: Path, summary_rows: list[dict], methods: list[str], fixed_rate: float) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    lookup = recovery_lookup(summary_rows)
    w, h = 900, 520
    left, right, top, bottom = 80, 30, 60, 430
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 20), f"Bit-plane recovery at {float(fixed_rate):g} base substitution", fill="black")
    d.line([left, bottom, w - right, bottom], fill="black")
    d.line([left, top, left, bottom], fill="black")
    for k in range(6):
        y = bottom - k * (bottom - top) / 5
        d.text((30, int(y) - 6), f"{k/5:.1f}", fill="black")
        d.line([left - 4, int(y), left, int(y)], fill="black")
    x_positions = {}
    for idx, bitplane in enumerate(BITPLANES):
        x = left + idx * (w - left - right) / (len(BITPLANES) - 1)
        x_positions[bitplane] = x
        d.text((int(x) - 10, bottom + 12), str(bitplane), fill="black")
    colors = {
        "uniform_rs4": (70, 130, 180),
        "static_uep": (210, 120, 40),
        "random_budget_neutral_uep": (90, 160, 90),
        "keyed_uep": (130, 80, 170),
    }
    for m_idx, method in enumerate(methods):
        pts = []
        for bitplane in BITPLANES:
            val = lookup.get((method, str(fixed_rate), bitplane), lookup.get((method, f"{float(fixed_rate):.3f}", bitplane), 0.0))
            x = x_positions[bitplane]
            y = bottom - val * (bottom - top)
            pts.append((x, y))
        color = colors.get(method, (0, 0, 0))
        for a, b in zip(pts, pts[1:]):
            d.line([a[0], a[1], b[0], b[1]], fill=color, width=3)
        for x, y in pts:
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)
        d.text((610, 75 + 24 * m_idx), method, fill=color)
    d.text((390, 470), "bit-plane (7 is MSB)", fill="black")
    img.save(path_png)
    save_pdf_from_image(img, path_png.with_suffix(".pdf"))


def write_latex_table(summary_rows: list[dict], out_path: Path, fixed_rate: float = 0.02) -> None:
    lookup = recovery_lookup(summary_rows)
    labels = {
        "uniform_rs4": "Uniform RS-4",
        "static_uep": "Static UEP",
        "random_budget_neutral_uep": "Random UEP",
        "keyed_uep": "Keyed UEP",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Bit-plane recovery rates under simulated DNA base substitution at 2\% error rate. Bit-plane 7 is the most significant bit-plane.}",
        r"\label{tab:bitplane_recovery}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Method & bit 7 & bit 6 & bit 5 & bit 4 & bit 3 & bit 2 & bit 1 & bit 0 \\",
        r"\midrule",
    ]
    for method in METHODS:
        vals = []
        for bitplane in BITPLANES:
            val = lookup.get((method, str(fixed_rate), bitplane), lookup.get((method, f"{float(fixed_rate):.3f}", bitplane), 0.0))
            vals.append(f"{val:.4f}")
        lines.append(f"{labels[method]} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bit-plane recovery analysis under simulated DNA base substitution.")
    add_common_args(parser)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()

    if args.smoke:
        args.error_rates = "0.01"
        args.seeds = "1"
        resize = (64, 64)
    else:
        resize = resize_tuple(args.resize)

    methods = parse_list(args.methods)
    error_rates = parse_rates(args.error_rates)
    seeds = parse_seeds(args.seeds)
    rows = []
    for image_path in prepare_images(args.image_folder, args.output_dir, args.smoke, resize):
        img = load_rgb_image(image_path, resize)
        blocks, original_size, padded_size = decompose_bitplane_blocks(img, args.block_size)
        for method in methods:
            for rate in error_rates:
                for seed in seeds:
                    start = time.perf_counter()
                    scheduled = apply_schedule(blocks, method, seed=seed)
                    recovered, _ = simulate_rs_capacity(scheduled, rate, seed)
                    recon = reconstruct_image(recovered, original_size, padded_size, args.block_size)
                    mse, psnr = mse_psnr(img, recon)
                    ssim = global_ssim(img, recon)
                    imp_rec = important_block_recovery(scheduled, recovered)
                    runtime = time.perf_counter() - start
                    for bitplane in BITPLANES:
                        pairs = [(o, r) for o, r in zip(scheduled, recovered) if o.bitplane == bitplane]
                        rec = sum(1 for o, r in pairs if o.data == r.data) / max(1, len(pairs))
                        rows.append(
                            {
                                "dataset": "smoke" if args.smoke else (args.image_folder.name if args.image_folder else "custom"),
                                "image": image_path.name,
                                "method": method,
                                "channel_type": "simulated_dna_base_substitution",
                                "error_rate": rate,
                                "seed": seed,
                                "bitplane": bitplane,
                                "recovery_rate": round(rec, 6),
                                "important_block_recovery": round(imp_rec, 6),
                                "psnr": round(psnr, 6),
                                "ssim": round(ssim, 6),
                                "runtime_seconds": round(runtime, 6),
                            }
                        )

    res = results_dir(args)
    figs = figures_dir(args)
    raw_path = res / "bitplane_recovery_raw.csv"
    summary_path = res / "bitplane_recovery_summary.csv"
    write_csv(raw_path, rows)
    summary = write_summary(rows, summary_path)
    draw_heatmap(figs / "bitplane_recovery_heatmap.png", summary, methods, error_rates)
    fixed_rate = 0.02 if 0.02 in error_rates else error_rates[len(error_rates) // 2]
    draw_line_plot(figs / "bitplane_recovery_by_method.png", summary, methods, fixed_rate)
    write_latex_table(summary, args.output_dir / "tables" / "generated" / "table_bitplane_recovery.tex", fixed_rate)
    print(f"Wrote {len(rows)} rows to {raw_path}")


if __name__ == "__main__":
    main()
