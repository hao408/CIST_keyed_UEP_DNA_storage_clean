from __future__ import annotations

from math import log10
from typing import Sequence

from PIL import Image

from .uep_scheduler import BitPlaneBlock


def mse_psnr(original: Image.Image, reconstructed: Image.Image) -> tuple[float, float]:
    a = list(original.convert("RGB").getdata())
    b = list(reconstructed.convert("RGB").getdata())
    err = 0.0
    count = 0
    for p, q in zip(a, b):
        for x, y in zip(p, q):
            err += (x - y) ** 2
            count += 1
    mse = err / max(1, count)
    psnr = 99.0 if mse == 0 else 10.0 * log10((255.0 * 255.0) / mse)
    return mse, psnr


def global_ssim(original: Image.Image, reconstructed: Image.Image) -> float:
    x = [sum(p) / 3.0 for p in original.convert("RGB").getdata()]
    y = [sum(p) / 3.0 for p in reconstructed.convert("RGB").getdata()]
    n = max(1, len(x))
    mux = sum(x) / n
    muy = sum(y) / n
    vx = sum((v - mux) ** 2 for v in x) / n
    vy = sum((v - muy) ** 2 for v in y) / n
    cov = sum((a - mux) * (b - muy) for a, b in zip(x, y)) / n
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))


def important_block_recovery(original: Sequence[BitPlaneBlock], recovered: Sequence[BitPlaneBlock]) -> float:
    pairs = [(o, r) for o, r in zip(original, recovered) if o.base_level == "H"]
    return sum(1 for o, r in pairs if o.data == r.data) / max(1, len(pairs))


def full_image_success(original: Sequence[BitPlaneBlock], recovered: Sequence[BitPlaneBlock]) -> float:
    return 1.0 if all(o.data == r.data for o, r in zip(original, recovered)) else 0.0
