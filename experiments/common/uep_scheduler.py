from __future__ import annotations

from dataclasses import dataclass
from math import floor
from random import Random
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image


@dataclass
class BitPlaneBlock:
    index: int
    channel: int
    bitplane: int
    block_row: int
    block_col: int
    data: bytes
    base_level: str
    level: str = ""
    parity: int = 0


def level_for_bitplane(bitplane: int) -> str:
    if bitplane >= 6:
        return "H"
    if bitplane >= 3:
        return "M"
    return "L"


def profile_to_dict(profile: Tuple[int, int, int]) -> Dict[str, int]:
    return {"H": profile[0], "M": profile[1], "L": profile[2]}


def decompose_bitplane_blocks(img: Image.Image, block_size: int = 16) -> Tuple[List[BitPlaneBlock], Tuple[int, int], Tuple[int, int]]:
    w, h = img.size
    pad_w = ((w + block_size - 1) // block_size) * block_size
    pad_h = ((h + block_size - 1) // block_size) * block_size
    padded = Image.new("RGB", (pad_w, pad_h))
    padded.paste(img, (0, 0))
    px = padded.load()
    blocks: List[BitPlaneBlock] = []
    idx = 0
    for channel in range(3):
        for bitplane in range(7, -1, -1):
            for by in range(0, pad_h, block_size):
                for bx in range(0, pad_w, block_size):
                    bits = []
                    for yy in range(by, by + block_size):
                        for xx in range(bx, bx + block_size):
                            bits.append((px[xx, yy][channel] >> bitplane) & 1)
                    packed = bytearray()
                    for k in range(0, len(bits), 8):
                        value = 0
                        for bit in bits[k : k + 8]:
                            value = (value << 1) | bit
                        packed.append(value)
                    base = level_for_bitplane(bitplane)
                    blocks.append(BitPlaneBlock(idx, channel, bitplane, by // block_size, bx // block_size, bytes(packed), base, base))
                    idx += 1
    return blocks, (w, h), (pad_w, pad_h)


def apply_schedule(
    blocks: Sequence[BitPlaneBlock],
    method: str,
    seed: int = 1,
    profile: Tuple[int, int, int] = (8, 4, 2),
    promote_ratio: float = 0.10,
    demote_ratio: float = 0.20,
) -> List[BitPlaneBlock]:
    scheduled = [BitPlaneBlock(**b.__dict__) for b in blocks]
    parity = profile_to_dict(profile)
    method = method.lower()

    if method == "no_rs":
        for b in scheduled:
            b.level = "N"
            b.parity = 0
        return scheduled
    if method == "uniform_rs4":
        for b in scheduled:
            b.level = "U4"
            b.parity = 4
        return scheduled
    if method in {"uniform_rs8", "all_high_rs40_32"}:
        for b in scheduled:
            b.level = "U8"
            b.parity = 8
        return scheduled
    if method == "uniform_mixed_rs4_5":
        rng = Random(seed)
        order = list(range(len(scheduled)))
        rng.shuffle(order)
        n5 = round(0.25 * len(order))
        use5 = set(order[:n5])
        for i, b in enumerate(scheduled):
            b.level = "U5" if i in use5 else "U4"
            b.parity = 5 if i in use5 else 4
        return scheduled

    if method in {"static_uep", "aes_static_uep", "chacha_static_uep", "cist_stream_static_uep"}:
        for b in scheduled:
            b.level = b.base_level
            b.parity = parity[b.level]
        return scheduled

    if method in {"keyed_uep", "aes_keyed_uep", "chacha_keyed_uep", "cist_stream_keyed_uep", "random_budget_neutral_uep"}:
        rng = Random(seed if method == "random_budget_neutral_uep" else f"keyed:{seed}")
        scores = {b.index: rng.random() for b in scheduled}
        bit5 = [b for b in scheduled if b.bitplane == 5 and b.base_level == "M"]
        bit3 = [b for b in scheduled if b.bitplane == 3 and b.base_level == "M"]
        promote_n = floor(promote_ratio * len(bit5))
        demote_n = floor(demote_ratio * len(bit3))
        promote = {b.index for b in sorted(bit5, key=lambda x: (-scores[x.index], x.index))[:promote_n]}
        demote = {b.index for b in sorted(bit3, key=lambda x: (-scores[x.index], x.index))[:demote_n]}
        for b in scheduled:
            b.level = "H" if b.index in promote else "L" if b.index in demote else b.base_level
            b.parity = parity[b.level]
        return scheduled

    raise ValueError(f"Unknown method: {method}")


def reconstruct_image(blocks: Sequence[BitPlaneBlock], original_size: Tuple[int, int], padded_size: Tuple[int, int], block_size: int = 16) -> Image.Image:
    w, h = padded_size
    planes = [[[0 for _ in range(w * h)] for _ in range(8)] for _ in range(3)]
    for b in blocks:
        bits = []
        for byte in b.data:
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
        k = 0
        for yy in range(b.block_row * block_size, (b.block_row + 1) * block_size):
            for xx in range(b.block_col * block_size, (b.block_col + 1) * block_size):
                if k < len(bits):
                    planes[b.channel][b.bitplane][yy * w + xx] = bits[k]
                k += 1
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            values = []
            for c in range(3):
                v = 0
                for bitplane in range(8):
                    v |= planes[c][bitplane][y * w + x] << bitplane
                values.append(v)
            px[x, y] = tuple(values)
    return img.crop((0, 0, original_size[0], original_size[1]))
