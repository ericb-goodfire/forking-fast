"""fastseg must reproduce the ruptures-1.1.10 path bit-for-bit."""
import numpy as np
import pytest

from otrecon import fastseg
from otrecon.data import counts_from_draws
from otrecon.models import (LogitLinearCost, MultinomialCost,
                            _empirical_logits, _segment_positions_fast,
                            _segment_positions_ruptures, segment_positions)
from otrecon.synthetic import make_synthetic, synthetic_draws

PENS = (1.0, 8.0, 64.0, 512.0, 2048.0)
VARIANTS = ("mult", "linear", "l2")


def _random_counts(rng, n, K, S):
    """Piecewise-constant truth with occasional jumps -> realistic counts."""
    p = rng.dirichlet(np.ones(K))
    rows = []
    for _ in range(n):
        if rng.random() < 0.08:
            p = rng.dirichlet(np.ones(K))
        rows.append(rng.multinomial(S, p))
    return np.array(rows, float)


def _cases():
    rng = np.random.default_rng(630)
    cases = []
    for _ in range(10):
        n = int(rng.integers(6, 90))
        K = int(rng.integers(2, 6))
        S = int(rng.choice([5, 15, 30, 100]))
        step = float(rng.choice([1.0, 2.0, 4.0]))
        counts = _random_counts(rng, n, K, S)
        obs_tok = np.arange(n) * step
        cases.append((obs_tok, counts))
    # degenerate/tie-prone: deterministic single-category counts
    n = 40
    det = np.zeros((n, 3))
    det[:, 0] = 30.0
    cases.append((np.arange(n, dtype=float), det))
    # perfectly flat multi-category counts (exact ties between partitions)
    flat = np.tile([10.0, 10.0, 10.0], (30, 1))
    cases.append((np.arange(30, dtype=float), flat))
    return cases


@pytest.mark.parametrize("variant", VARIANTS)
def test_fastseg_matches_ruptures(variant):
    for obs_tok, counts in _cases():
        n = len(obs_tok)
        min_size = 2 if variant == "mult" else 3
        if n < 2 * min_size:
            continue
        for pen in PENS:
            fast = _segment_positions_fast(obs_tok, counts, variant, pen)
            legacy = _segment_positions_ruptures(obs_tok, counts, variant,
                                                 pen)
            assert fast == legacy, (variant, pen, n, fast, legacy)


def test_mult_cost_matrix_bit_exact():
    rng = np.random.default_rng(1)
    counts = _random_counts(rng, 25, 4, 20)
    C = fastseg.mult_cost_matrix(counts)
    cost = MultinomialCost().fit(counts)
    for s in range(26):
        for e in range(s + 1, 26):
            assert C[s, e] == cost.error(s, e), (s, e)


def test_linear_cost_matrix_bit_exact():
    rng = np.random.default_rng(2)
    counts = _random_counts(rng, 25, 3, 30)
    x = np.arange(25) * 4.0
    y, w = _empirical_logits(counts)
    sig = np.hstack([x[:, None], y, w])
    cost = LogitLinearCost().fit(sig)
    C = fastseg.linear_cost_matrix(x, y, w)
    for s in range(26):
        for e in range(s + 3, 26):
            assert C[s, e] == cost.error(s, e), (s, e)


def test_l2_cost_matrix_bit_exact():
    rng = np.random.default_rng(3)
    counts = _random_counts(rng, 25, 4, 20)
    C = fastseg.l2_cost_matrix(counts)
    for s in range(24):
        for e in range(s + 2, 26):
            ref = counts[s:e].var(axis=0).sum() * (e - s)
            assert C[s, e] == ref, (s, e)


def test_numba_and_numpy_dp_agree():
    rng = np.random.default_rng(4)
    counts = _random_counts(rng, 60, 3, 30)
    C = fastseg.mult_cost_matrix(counts)
    impl = fastseg._get_pelt_impl()
    for pen in PENS:
        a = list(fastseg._pelt_core_py(C, 60, pen, 2))
        b = list(impl(C, 60, pen, 2))
        assert a == b


def test_vectorized_log_selfcheck_runs():
    # On runtimes where this is False, mult falls back to the per-pair
    # reference path; either way the equivalence tests above must pass.
    assert fastseg.vectorized_log_ok() in (True, False)


def test_seg_cache_returns_same_result():
    syn = make_synthetic()
    draws = synthetic_draws(syn["o"], 30)
    counts = counts_from_draws(draws, 3, 0, 30)
    idxs = np.asarray(syn["idxs"], float)
    a = segment_positions(idxs, counts, "mult", 8.0)
    b = segment_positions(idxs, counts, "mult", 8.0)
    assert a is b  # memoized
    c = segment_positions(idxs, counts, "mult", 2048.0)
    assert isinstance(c, list) and c[-1] == len(idxs)
