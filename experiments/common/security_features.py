from __future__ import annotations

import math
from collections import Counter
from random import Random
from typing import Iterable, Sequence

import numpy as np

from .dna_channel import bytes_to_dna

LEVELS = ["H", "M", "L"]
TRANSITIONS = ["promoted", "demoted", "unchanged"]
CONTAINER_BASES = 160
CONTAINER_BYTES = 40
DETERMINISTIC_PAD = "ACGT"


def pseudo_parity_bytes(block_index: int, parity: int, seed: int) -> bytes:
    rng = Random(f"parity:{seed}:{block_index}:{parity}")
    return bytes(rng.randrange(256) for _ in range(parity))


def keyed_dummy_dna(length: int, block_index: int, seed: int) -> str:
    rng = Random(f"encrypted-dummy:{seed}:{block_index}:{length}")
    return "".join(rng.choice("ACGT") for _ in range(length))


def keyed_bytes(length: int, namespace: str, block_index: int, seed: int) -> bytes:
    rng = Random(f"{namespace}:{seed}:{block_index}:{length}")
    return bytes(rng.randrange(256) for _ in range(length))


def xor_bytes(data: bytes, mask: bytes) -> bytes:
    if len(data) != len(mask):
        raise ValueError("data and mask must have identical lengths")
    return bytes(a ^ b for a, b in zip(data, mask))


def repeat_pad(length: int) -> str:
    return (DETERMINISTIC_PAD * ((length + len(DETERMINISTIC_PAD) - 1) // len(DETERMINISTIC_PAD)))[:length]


def make_payload(block, design: str, seed: int) -> str:
    design = {
        "current_encrypted_dummy_tail": "fixed_encrypted_dummy_padding",
        "current_fixed_encrypted_dummy_tail": "fixed_encrypted_dummy_padding",
        "full_container_masked_fixed_payload": "full_container_masking",
        "new_full_container_masking": "full_container_masking",
    }.get(design, design)
    codeword = block.data + pseudo_parity_bytes(block.index, block.parity, seed)
    effective_payload = bytes_to_dna(codeword)
    if design == "variable_length_payload":
        return effective_payload
    pad_len = max(0, CONTAINER_BASES - len(effective_payload))
    if design == "fixed_deterministic_padding":
        return effective_payload + repeat_pad(pad_len)
    if design == "fixed_encrypted_dummy_padding":
        return effective_payload + keyed_dummy_dna(pad_len, block.index, seed)
    if design == "full_container_masking":
        if len(codeword) > CONTAINER_BYTES:
            raise ValueError(f"effective codeword exceeds {CONTAINER_BYTES} bytes")
        dummy = keyed_bytes(CONTAINER_BYTES - len(codeword), "full-container-dummy", block.index, seed)
        container = codeword + dummy
        mask = keyed_bytes(CONTAINER_BYTES, "full-container-mask", block.index, seed)
        return bytes_to_dna(xor_bytes(container, mask))
    raise ValueError(f"Unknown container design: {design}")


def full_container_round_trip(block, seed: int, wrong_key: bool = False) -> tuple[bytes, bytes]:
    """Return original effective codeword and recovered codeword after full-container unmasking."""
    codeword = block.data + pseudo_parity_bytes(block.index, block.parity, seed)
    dummy = keyed_bytes(CONTAINER_BYTES - len(codeword), "full-container-dummy", block.index, seed)
    container = codeword + dummy
    mask_seed = seed + 1000003 if wrong_key else seed
    masked = xor_bytes(container, keyed_bytes(CONTAINER_BYTES, "full-container-mask", block.index, seed))
    unmasked = xor_bytes(masked, keyed_bytes(CONTAINER_BYTES, "full-container-mask", block.index, mask_seed))
    return codeword, unmasked[: len(codeword)]


def trailing_deterministic_pad_len(payload: str) -> int:
    count = 0
    for i in range(len(payload) - 1, -1, -1):
        expected = DETERMINISTIC_PAD[i % len(DETERMINISTIC_PAD)]
        if payload[i] != expected:
            break
        count += 1
    return count


def max_homopolymer(seq: str) -> int:
    if not seq:
        return 0
    best = cur = 1
    for a, b in zip(seq, seq[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


def shannon_entropy(seq: str) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    n = len(seq)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def kmer_freq(seq: str, k: int) -> list[float]:
    alphabet = "ACGT"
    kmers = [""]
    for _ in range(k):
        kmers = [prefix + base for prefix in kmers for base in alphabet]
    total = max(1, len(seq) - k + 1)
    counts = Counter(seq[i : i + k] for i in range(max(0, len(seq) - k + 1)))
    return [counts[kmer] / total for kmer in kmers]


def payload_features(payload: str, include_fixed_positions: bool = True) -> list[float]:
    n = len(payload)
    counts = Counter(payload)
    freqs = [counts[b] / max(1, n) for b in "ACGT"]
    gc = (counts["G"] + counts["C"]) / max(1, n)
    at = (counts["A"] + counts["T"]) / max(1, n)
    tail = payload[-32:] if payload else ""
    tail_counts = Counter(tail)
    tail_freqs = [tail_counts[b] / max(1, len(tail)) for b in "ACGT"]
    features = [
        float(n),
        float(CONTAINER_BASES - n),
        *freqs,
        gc,
        at,
        float(max_homopolymer(payload)),
        shannon_entropy(payload),
        trailing_deterministic_pad_len(payload) / max(1, n),
        *tail_freqs,
        *kmer_freq(payload, 2),
    ]
    if include_fixed_positions:
        # Sparse one-hot probes capture fixed-position padding/container cues
        # without giving access to the true schedule or secret key.
        probes = [0, 7, 15, 31, 63, 95, 127, 143, 159]
        for pos in probes:
            ch = payload[pos] if pos < n else "N"
            features.extend([1.0 if ch == b else 0.0 for b in "ACGT"])
    return features


def macro_f1(labels: Sequence[str], preds: Sequence[str], classes: Sequence[str]) -> float:
    scores = []
    for cls in classes:
        tp = sum(1 for y, p in zip(labels, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(labels, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(labels, preds) if y == cls and p != cls)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores))


def transition_label(block) -> str:
    if block.base_level == "M" and block.level == "H":
        return "promoted"
    if block.base_level == "M" and block.level == "L":
        return "demoted"
    return "unchanged"


def as_matrix(rows: Iterable[list[float]]) -> np.ndarray:
    return np.asarray(list(rows), dtype=np.float32)
