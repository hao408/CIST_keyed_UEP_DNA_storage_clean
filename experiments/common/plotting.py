from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def simple_bar_plot(path_png: Path, title: str, labels: Sequence[str], values: Sequence[float], ylabel: str = "value") -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    w, h = 900, 480
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 20), title, fill="black")
    d.text((30, 45), ylabel, fill="black")
    if not values:
        d.text((30, 100), "No rows generated.", fill="black")
    else:
        max_v = max(max(values), 1e-9)
        left, bottom, top = 60, 410, 90
        bar_w = max(12, (w - 120) // max(1, len(values)))
        for i, (label, value) in enumerate(zip(labels, values)):
            x0 = left + i * bar_w
            x1 = x0 + max(8, bar_w - 8)
            y0 = bottom - int((bottom - top) * (value / max_v))
            d.rectangle([x0, y0, x1, bottom], fill=(64, 120, 180))
            d.text((x0, bottom + 8), label[:12], fill="black")
            d.text((x0, max(top, y0 - 18)), f"{value:.3g}", fill="black")
    img.save(path_png)
    pdf_path = path_png.with_suffix(".pdf")
    c = canvas.Canvas(str(pdf_path), pagesize=(w, h))
    c.drawImage(ImageReader(img), 0, 0, width=w, height=h)
    c.showPage()
    c.save()
