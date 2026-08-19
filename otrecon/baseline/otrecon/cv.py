"""Hyperparameter tuning by k-fold CV on the cheap input draws only.

The S input draws per position are split (in draw order) into ``n_folds``
contiguous folds. For each candidate parameter set: fit the model on the
counts of the other folds' draws (at the observed positions), predict at the
observed positions, and score the held-out fold's counts by multinomial
log-likelihood; sum over folds and positions. The S=1000 reference never
enters tuning.
"""
from __future__ import annotations

import numpy as np

from .data import counts_from_draws
from .models import MODEL_REGISTRY


def fold_slices(S: int, n_folds: int = 5) -> list[tuple[int, int]]:
    edges = np.linspace(0, S, n_folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_folds)]


def _loglik_counts(pred: np.ndarray, counts: np.ndarray, eps: float = 1e-9) -> float:
    p = np.maximum(pred, eps)
    p = p / p.sum(axis=1, keepdims=True)
    return float((counts * np.log(p)).sum())


def cv_select(model_name: str, draws_obs: np.ndarray, obs_tok: np.ndarray,
              S: int, K: int, n_folds: int = 5) -> tuple[dict, dict]:
    """Pick params for ``model_name`` from its grid by CV.

    draws_obs: (n_obs, n_total) draw ids restricted to observed positions.
    Returns (best_params, {param_repr: cv_score}).
    """
    cls = MODEL_REGISTRY[model_name]
    grid = cls.grid()
    if len(grid) == 1:
        return grid[0], {repr(grid[0]): None}
    folds = fold_slices(S, n_folds)
    scores = {}
    for params in grid:
        tot = 0.0
        for (lo, hi) in folds:
            c_test = counts_from_draws(draws_obs, K, lo, hi)
            c_train = counts_from_draws(draws_obs, K, 0, lo) + \
                counts_from_draws(draws_obs, K, hi, S)
            m = cls()
            m.fit(obs_tok, c_train, S - (hi - lo), params)
            pred = m.predict(obs_tok)
            tot += _loglik_counts(pred, c_test)
        scores[repr(params)] = tot
    best = max(grid, key=lambda p: scores[repr(p)])
    return best, scores
