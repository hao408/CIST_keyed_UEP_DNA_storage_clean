from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_image_paths(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_rgb_image(path: Path, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.Resampling.BICUBIC)
    return img


def make_synthetic_image(path: Path, size: Tuple[int, int] = (32, 32)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 255) // max(1, w - 1), (y * 255) // max(1, h - 1), ((x + y) * 255) // max(1, w + h - 2))
    img.save(path)
    return path


def prepare_images(image_folder: Optional[Path], output_dir: Path, smoke: bool, resize: Tuple[int, int]) -> List[Path]:
    if smoke or image_folder is None:
        return [make_synthetic_image(output_dir / "datasets" / "smoke" / "synthetic_rgb.png", resize)]
    paths = iter_image_paths(image_folder)
    if not paths:
        raise FileNotFoundError(f"No image files found in {image_folder}")
    return paths
