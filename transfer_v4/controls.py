"""Matched scrambling controls for the actual synthetic training objective.

Each trajectory keeps its initial conditioning frame, marker values/positions,
and supervised-prefix patch histogram. Only subsequent patch order is permuted.
The unused suffix is shuffled separately so no token enters/leaves the prefix.
This preserves per-rule frequency information; it is not an IID-noise control.
"""
from __future__ import annotations

import numpy as np

from .common import frame_length


def prefix_shuffled(tokens, c, seed):
    result = tokens.copy()
    boundary = c["model"]["context"]+1
    first = frame_length(c)
    if tokens.ndim != 2 or tokens.shape[1] < boundary:
        raise ValueError("Trajectories must contain the complete trained prefix")
    rng = np.random.default_rng(seed)
    cutoff = c["nca"]["states"]**4
    for row in result:
        for start, end in ((first, boundary), (boundary, len(row))):
            positions = np.arange(start, end)
            positions = positions[row[positions] < cutoff]
            row[positions] = rng.permutation(row[positions])
    return result


def validate_prefix_control(original, control, c):
    if original.shape != control.shape or original.dtype != control.dtype:
        raise ValueError("Prefix control shape/dtype differs")
    boundary, first = c["model"]["context"]+1, frame_length(c)
    if not np.array_equal(original[:, :first], control[:, :first]):
        raise ValueError("Initial conditioning frame changed")
    marker = original >= c["nca"]["states"]**4
    if not np.array_equal(marker, control >= c["nca"]["states"]**4) or not np.array_equal(original[marker], control[marker]):
        raise ValueError("Prefix control markers changed")
    for a, b in ((first, boundary), (boundary, original.shape[1])):
        if not np.array_equal(np.sort(original[:, a:b], axis=1), np.sort(control[:, a:b], axis=1)):
            raise ValueError("Supervised-prefix or unused-suffix histogram changed")
