"""Metrics: TV to ground truth, held-out multinomial log-likelihood,
credible-band coverage, and the jump-proximal / flat region split."""
from __future__ import annotations

import numpy as np


def tv_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-position total variation distance, (T,)."""
    return 0.5 * np.abs(np.asarray(pred) - np.asarray(gt)).sum(axis=1)


def gt_jumps(idxs: list[int], gt: np.ndarray, thresh: float) -> list[int]:
    """Token positions of ground-truth jumps: TV between successive grid
    curve values > thresh; the jump is located at the later grid point."""
    inc = 0.5 * np.abs(np.diff(gt, axis=0)).sum(axis=1)
    return [int(idxs[i + 1]) for i in np.flatnonzero(inc > thresh)]


def region_masks(idxs: list[int], jump_tokens: list[int], radius: int = 10
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(jump_proximal, flat) boolean masks over grid positions: proximal =
    within +/- radius tokens of any jump location."""
    pos = np.asarray(idxs, float)
    if not jump_tokens:
        prox = np.zeros(len(pos), dtype=bool)
    else:
        d = np.abs(pos[:, None] - np.asarray(jump_tokens, float)[None, :])
        prox = (d <= radius).any(axis=1)
    return prox, ~prox


def heldout_loglik(pred: np.ndarray, draws: np.ndarray, lo: int,
                   eps: float = 1e-9) -> float:
    """Mean per-draw log-likelihood of held-out draws[:, lo:] under pred.

    pred: (T, K) probabilities; draws: (T, n_total) category ids.
    Averaged over draws and positions (i.e., per single held-out draw).
    """
    p = np.maximum(np.asarray(pred, float), eps)
    p = p / p.sum(axis=1, keepdims=True)
    held = draws[:, lo:]
    ll = np.log(np.take_along_axis(p, held.astype(int), axis=1))
    return float(ll.mean())


def heldout_loglik_per_position(pred: np.ndarray, draws: np.ndarray, lo: int,
                                eps: float = 1e-9) -> np.ndarray:
    p = np.maximum(np.asarray(pred, float), eps)
    p = p / p.sum(axis=1, keepdims=True)
    held = draws[:, lo:]
    ll = np.log(np.take_along_axis(p, held.astype(int), axis=1))
    return ll.mean(axis=1)


def band_coverage(lo_band: np.ndarray, hi_band: np.ndarray, gt: np.ndarray
                  ) -> np.ndarray:
    """Per-position fraction of categories whose GT value lies in the band."""
    inside = (gt >= lo_band - 1e-12) & (gt <= hi_band + 1e-12)
    return inside.mean(axis=1)


def summarize(values: np.ndarray, prox: np.ndarray) -> dict:
    """Overall / jump-proximal / flat means of a per-position metric."""
    v = np.asarray(values, float)
    out = {"overall": float(v.mean())}
    out["jump"] = float(v[prox].mean()) if prox.any() else None
    out["flat"] = float(v[~prox].mean()) if (~prox).any() else None
    return out
