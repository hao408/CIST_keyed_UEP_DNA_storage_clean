from __future__ import annotations

from dataclasses import replace
from random import Random
from typing import List, Sequence, Tuple

from .uep_scheduler import BitPlaneBlock


def simulate_rs_capacity(
    blocks: Sequence[BitPlaneBlock],
    error_rate: float,
    seed: int,
    error_mode: str = "substitution",
) -> Tuple[List[BitPlaneBlock], float]:
    """Capacity-model RS simulation used when a real RS package is unavailable.

    Substitution correction capacity is floor(r/2); erasure capacity is r.
    Failed blocks are zero-filled so image metrics remain computable.
    """
    rng = Random(seed)
    recovered: List[BitPlaneBlock] = []
    failures = 0
    if error_mode == "erasure":
        byte_error_prob = error_rate
    else:
        # Four DNA bases represent one byte under the fixed 2-bit/base mapping.
        # A byte is corrupted if at least one of its four bases is substituted.
        byte_error_prob = 1.0 - (1.0 - error_rate) ** 4
    for b in blocks:
        n = len(b.data) + b.parity
        observed = sum(1 for _ in range(n) if rng.random() < byte_error_prob)
        capacity = b.parity if error_mode == "erasure" else b.parity // 2
        ok = observed <= capacity
        if ok:
            recovered.append(replace(b))
        else:
            failures += 1
            recovered.append(replace(b, data=bytes(len(b.data))))
    return recovered, failures / max(1, len(blocks))
