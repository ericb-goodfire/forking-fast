"""Hyperparameter tuning by k-fold CV on the cheap input draws only.

The S input draws per position are split (in draw order) into ``n_folds``
contiguous folds. For each candidate parameter set: fit the model on the
counts of the other folds' draws (at the observed positions), predict at the
observed positions, and score the held-out fold's counts by multinomial
log-likelihood; sum over folds and positions. The S=1000 reference never
enters tuning.

Speedups (output-identical to the original loop):

* fold count matrices are built once per call instead of once per candidate
  (they never depended on the candidate);
* per fold, candidates that provably produce the identical fitted model
  share one fit/predict/score evaluation. For the segmentation models the
  fit is a function of (breakpoints, non-penalty params) only, and PELT
  breakpoints are piecewise-constant in the penalty, so the penalty grid
  collapses to a handful of distinct fits. The reused score is the very
  float the original loop computed, and per-candidate fold scores are still
  accumulated in fold order, so ``scores`` and the (first-max) ``best``
  selection are bit-identical.
"""
from __future__ import annotations

import numpy as np

from .data import counts_from_draws
from .models import (MODEL_REGISTRY, SegmentKernelPool, SegmentPolyLogit,
                     SegmentPool, segment_positions)


def fold_slices(S: int, n_folds: int = 5) -> list[tuple[int, int]]:
    edges = np.linspace(0, S, n_folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_folds)]


def _loglik_counts(pred: np.ndarray, counts: np.ndarray, eps: float = 1e-9) -> float:
    p = np.maximum(pred, eps)
    p = p / p.sum(axis=1, keepdims=True)
    return float((counts * np.log(p)).sum())


def _fit_signature(cls, params: dict, obs_tok: np.ndarray,
                   c_train: np.ndarray):
    """Hashable key equal iff ``cls().fit(obs_tok, c_train, ., params)``
    yields an identical fitted model (same predictions everywhere).

    Mirrors each model's fit() branch exactly:
    * SegmentKernelPool (incl. the grid-restricted l2/linear subclasses):
      fit = (breakpoints, h); breakpoints from segment_positions(variant, pen).
    * SegmentPolyLogit: fit = (breakpoints, deg).
    * SegmentPool (incl. M4l2): fit = breakpoints, with fit()'s own
      ``n < 4 -> [n]`` short-circuit replicated before segment_positions.
    * everything else: params identity (no collapse).
    """
    if issubclass(cls, SegmentKernelPool):
        bk = segment_positions(obs_tok, c_train, params["variant"],
                               params["pen"])
        return ("segkernel", tuple(bk), float(params["h"]))
    if issubclass(cls, SegmentPolyLogit):
        bk = segment_positions(obs_tok, c_train, params["variant"],
                               params["pen"])
        return ("segpoly", tuple(bk), int(params["deg"]))
    if issubclass(cls, SegmentPool):
        if len(obs_tok) < 4:
            bk = [len(obs_tok)]
        else:
            bk = segment_positions(obs_tok, c_train,
                                   params.get("cost", "mult"),
                                   float(params["pen"]))
        return ("segpool", tuple(bk))
    return ("params", repr(params))


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
    # hoisted: fold count matrices are candidate-independent
    fold_data = []
    for (lo, hi) in folds:
        c_test = counts_from_draws(draws_obs, K, lo, hi)
        c_train = counts_from_draws(draws_obs, K, 0, lo) + \
            counts_from_draws(draws_obs, K, hi, S)
        fold_data.append((lo, hi, c_test, c_train))
    score_cache: dict = {}
    scores = {}
    for params in grid:
        tot = 0.0
        for fi, (lo, hi, c_test, c_train) in enumerate(fold_data):
            key = (fi, _fit_signature(cls, params, obs_tok, c_train))
            s = score_cache.get(key)
            if s is None:
                m = cls()
                m.fit(obs_tok, c_train, S - (hi - lo), params)
                pred = m.predict(obs_tok)
                s = _loglik_counts(pred, c_test)
                score_cache[key] = s
            tot += s
        scores[repr(params)] = tot
    best = max(grid, key=lambda p: scores[repr(p)])
    return best, scores
