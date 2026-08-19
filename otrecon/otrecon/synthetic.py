"""Synthetic known-answer case for the smoke gate: piecewise-constant o_t
with planted jumps, exact Multinomial(S, o_t) counts."""
from __future__ import annotations

import numpy as np


def make_synthetic(T: int = 90, step: int = 4, K: int = 3,
                   jump_at: tuple[int, int] = (30, 60), seed: int = 7
                   ) -> dict:
    """Grid of T positions spaced ``step`` tokens; piecewise-constant o with
    jumps before grid rows jump_at (token positions jump_at*step)."""
    idxs = np.arange(T) * step
    levels = [np.array([0.7, 0.2, 0.1]),
              np.array([0.15, 0.75, 0.1]),
              np.array([0.4, 0.1, 0.5])]
    o = np.zeros((T, K))
    seg = 0
    for i in range(T):
        if seg + 1 < len(jump_at) + 1 and i in jump_at:
            seg += 1
        o[i] = levels[seg]
    return {"idxs": idxs.tolist(), "o": o,
            "jump_tokens": [int(j * step) for j in jump_at]}


def synthetic_draws(o: np.ndarray, n_total: int, seed: int = 7) -> np.ndarray:
    """(T, n_total) iid category draws from each row of o."""
    rng = np.random.default_rng(seed)
    T, K = o.shape
    out = np.zeros((T, n_total), dtype=np.int8)
    for t in range(T):
        out[t] = rng.choice(K, size=n_total, p=o[t])
    return out


def make_ramp_jump(T: int = 90, step: int = 4, jump_row: int = 60) -> dict:
    """Drift + jump truth (#44 gate ii): rows [0, jump_row) drift linearly
    (A 0.7->0.3, B 0.2->0.6, C 0.1 const), then one jump to a constant
    level (0.15, 0.10, 0.75). True segmentation: 2 segments, 1 changepoint."""
    idxs = np.arange(T) * step
    o = np.zeros((T, 3))
    for i in range(T):
        if i < jump_row:
            f = i / max(jump_row - 1, 1)
            o[i] = [0.7 - 0.4 * f, 0.2 + 0.4 * f, 0.1]
        else:
            o[i] = [0.15, 0.10, 0.75]
    o = o / o.sum(axis=1, keepdims=True)
    return {"idxs": idxs.tolist(), "o": o,
            "jump_tokens": [int(jump_row * step)],
            "ramp_rows": (0, jump_row), "true_n_segments": 2}
