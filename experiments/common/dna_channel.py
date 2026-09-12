from __future__ import annotations

from collections import Counter
from random import Random
from typing import Iterable, List

BASES = "ACGT"


def bytes_to_dna(data: bytes) -> str:
    lut = "ACGT"
    out = []
    for b in data:
        out.extend(lut[(b >> shift) & 3] for shift in (6, 4, 2, 0))
    return "".join(out)


def dna_to_bytes(seq: str) -> bytes:
    inv = {"A": 0, "C": 1, "G": 2, "T": 3}
    out = bytearray()
    usable = len(seq) - (len(seq) % 4)
    for i in range(0, usable, 4):
        value = 0
        for ch in seq[i : i + 4]:
            value = (value << 2) | inv.get(ch, 0)
        out.append(value)
    return bytes(out)


def corrupt_dna(seq: str, error_rate: float, seed: int, mode: str = "substitution") -> str:
    rng = Random(seed)
    out = []
    for ch in seq:
        if mode in {"deletion", "mixed"} and rng.random() < error_rate:
            continue
        if mode in {"substitution", "mixed"} and rng.random() < error_rate:
            choices = [b for b in BASES if b != ch]
            ch = rng.choice(choices)
        out.append(ch)
        if mode in {"insertion", "mixed"} and rng.random() < error_rate:
            out.append(rng.choice(BASES))
    return "".join(out)


def consensus(reads: Iterable[str]) -> str:
    reads = list(reads)
    if not reads:
        return ""
    max_len = max(len(r) for r in reads)
    out = []
    for i in range(max_len):
        c = Counter(r[i] for r in reads if i < len(r))
        out.append(c.most_common(1)[0][0] if c else "A")
    return "".join(out)
