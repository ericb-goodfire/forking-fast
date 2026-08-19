"""Unit tests for otrecon (toy fixtures only)."""
import numpy as np
import pytest

from otrecon.data import (collapse_curve, collapse_map, counts_from_draws,
                          mixture_draws, weighted_o_t)
from otrecon.metrics import (band_coverage, gt_jumps, heldout_loglik,
                             region_masks, tv_to_gt)
from otrecon.models import (KalmanLogit, KernelDirichletGaussian,
                            MultinomialCost, RawInterp, SegmentPool,
                            _smooth_local_level)
from otrecon.cv import cv_select, fold_slices
from otrecon.synthetic import make_synthetic, synthetic_draws


def toy_rec():
    """Two positions; t=0 has two branches (0.75 / 0.25), t=4 one branch."""
    return {
        "meta": {"row_id": 1},
        "s": 8,
        "idxs": [0, 4],
        "categories": ["A", "B", "C", "D", "Other"],
        "branches": [
            {"t": 0, "tok_id": 1, "tok_p": 0.6, "is_base": True,
             "answers": ["A"] * 8},
            {"t": 0, "tok_id": 2, "tok_p": 0.2, "is_base": False,
             "answers": ["B"] * 8},
            {"t": 4, "tok_id": 3, "tok_p": 1.0, "is_base": True,
             "answers": ["A"] * 4 + ["C"] * 4},
        ],
    }


def test_weighted_o_t():
    idxs, o = weighted_o_t(toy_rec())
    assert idxs == [0, 4]
    np.testing.assert_allclose(o[0], [0.75, 0.25, 0, 0, 0])
    np.testing.assert_allclose(o[1], [0.5, 0, 0.5, 0, 0])


def test_collapse():
    rec = toy_rec()
    _, o = weighted_o_t(rec)
    kept, idx_map = collapse_map(o, rec["categories"], min_peak=0.01)
    assert kept == ["A", "B", "C", "Other"]  # D never appears
    g = collapse_curve(o, idx_map, len(kept))
    np.testing.assert_allclose(g.sum(axis=1), 1.0)
    np.testing.assert_allclose(g[0], [0.75, 0.25, 0, 0])


def test_mixture_draws_mean_and_seeding():
    rec = toy_rec()
    # enlarge: 4000 recorded draws per branch so sampling noise is small
    for b in rec["branches"]:
        b["answers"] = b["answers"] * 500
    rec["s"] = 4000
    _, o = weighted_o_t(rec)
    kept, idx_map = collapse_map(o, rec["categories"])
    K = len(kept)
    idxs, draws, diag = mixture_draws(rec, idx_map, K, n_total=3000)
    emp = counts_from_draws(draws, K, 0, 3000)
    emp /= emp.sum(axis=1, keepdims=True)
    gt = collapse_curve(o, idx_map, K)
    assert tv_to_gt(emp, gt).max() < 0.03
    # deterministic under the same seed
    _, draws2, _ = mixture_draws(rec, idx_map, K, n_total=3000)
    assert (draws == draws2).all()


def test_mixture_exhaustion_fallback():
    rec = toy_rec()
    rec["branches"][2]["answers"] = ["A"] * 6 + ["C"] * 6  # t=4: 12 avail
    rec["branches"][0]["answers"] = ["A"] * 2  # t=0 branch 0: only 2 avail
    rec["branches"][1]["answers"] = ["B"] * 12
    # weight 0.75 on branch 0 with only 2 recorded draws guarantees the
    # exhaustion-fallback path fires within 12 mixture draws.
    _, o = weighted_o_t(rec)
    kept, idx_map = collapse_map(o, rec["categories"])
    idxs, draws, diag = mixture_draws(rec, idx_map, len(kept), n_total=12)
    assert draws.shape == (2, 12)
    assert diag["exhausted_fallbacks"] >= 1  # branch 0 at t=0 runs out


def test_counts_from_draws():
    draws = np.array([[0, 1, 1, 2], [2, 2, 2, 2]])
    c = counts_from_draws(draws, 3, 0, 4)
    np.testing.assert_allclose(c, [[1, 2, 1], [0, 0, 4]])
    c2 = counts_from_draws(draws, 3, 1, 3)
    np.testing.assert_allclose(c2, [[0, 2, 0], [0, 0, 2]])


def test_metrics_basics():
    gt = np.array([[1, 0], [0.5, 0.5], [0, 1]])
    pred = np.array([[0.8, 0.2], [0.5, 0.5], [0, 1]])
    tv = tv_to_gt(pred, gt)
    np.testing.assert_allclose(tv, [0.2, 0, 0])
    jumps = gt_jumps([0, 4, 8], gt, 0.15)
    assert jumps == [4, 8]
    prox, flat = region_masks([0, 4, 8, 40], [8], radius=10)
    assert prox.tolist() == [True, True, True, False]
    cov = band_coverage(np.array([[0.7, 0.0]]), np.array([[0.9, 0.3]]),
                        np.array([[0.8, 0.5]]))
    np.testing.assert_allclose(cov, [0.5])
    draws = np.array([[0, 0, 1, 0]])
    ll = heldout_loglik(np.array([[0.75, 0.25]]), draws, 0)
    np.testing.assert_allclose(ll, (3 * np.log(0.75) + np.log(0.25)) / 4)


def test_raw_interp():
    m = RawInterp()
    m.fit(np.array([0.0, 8.0]), np.array([[3.0, 1.0], [0.0, 4.0]]), 4,
          {"kind": "linear"})
    p = m.predict(np.array([0.0, 4.0, 8.0]))
    np.testing.assert_allclose(p[0], [0.75, 0.25])
    np.testing.assert_allclose(p[1], [0.375, 0.625])
    lo, hi = m.credible_band(np.array([0.0, 8.0]), 0.9)
    assert (lo < hi).all()
    # band brackets the Dirichlet posterior mean (not the raw MLE at 0 counts)
    post_mean = m.alpha / m.alpha.sum(axis=1, keepdims=True)
    assert (lo <= post_mean).all() and (hi >= post_mean).all()


def test_kernel_limits():
    obs = np.array([0.0, 4.0, 8.0])
    counts = np.array([[10.0, 0], [0, 10.0], [10.0, 0]])
    m = KernelDirichletGaussian()
    m.fit(obs, counts, 10, {"kernel": "gaussian", "h": 1000.0})
    p = m.predict(obs)  # near-global pooling: ~2/3 vs 1/3
    np.testing.assert_allclose(p[:, 0], 2 / 3, atol=0.02)
    m.fit(obs, counts, 10, {"kernel": "gaussian", "h": 0.5})
    p = m.predict(obs)  # local: follows raw counts
    assert p[0, 0] > 0.9 and p[1, 1] > 0.9


def test_kalman_constant_series():
    obs = np.arange(10, dtype=float) * 4
    rng = np.random.default_rng(0)
    counts = rng.multinomial(50, [0.6, 0.4], size=10).astype(float)
    m = KalmanLogit()
    m.fit(obs, counts, 50, {})
    p = m.predict(obs)
    assert np.abs(p[:, 0] - 0.6).max() < 0.12
    # interpolation at unobserved points stays within the level range
    p_mid = m.predict(np.array([2.0, 18.0]))
    assert np.abs(p_mid[:, 0] - 0.6).max() < 0.12
    lo, hi = m.credible_band(obs, 0.9)
    assert (lo < p).all() and (hi > p).all()


def test_local_level_smoother_interpolates():
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 1.0])
    R = np.array([1e-6, 1e-6])
    m, v = _smooth_local_level(x, y, R, q=0.01, x_all=np.array([5.0]))
    assert 0.3 < m[0] < 0.7
    assert v[0] > 0


def test_multinomial_cost_and_segmentation():
    c = MultinomialCost().fit(np.array([[5.0, 0]] * 4 + [[0, 5.0]] * 4))
    assert c.error(0, 4) == pytest.approx(0.0)
    assert c.error(0, 8) > 1.0
    syn = make_synthetic()
    draws = synthetic_draws(syn["o"], 30, seed=1)
    counts = counts_from_draws(draws, 3, 0, 30)
    m = SegmentPool()
    m.fit(np.asarray(syn["idxs"], float), counts, 30, {"pen": 8.0})
    bkp_toks = [0.5 * (syn["idxs"][e - 1] + syn["idxs"][e])
                for e in m.bkps[:-1]]
    for jt in syn["jump_tokens"]:
        assert min(abs(b - jt) for b in bkp_toks) <= 4
    p = m.predict(np.asarray(syn["idxs"], float))
    assert tv_to_gt(p, syn["o"]).mean() < 0.08


def test_cv_select_prefers_smooth_on_constant_truth():
    o = np.tile([0.5, 0.5], (40, 1))
    draws = synthetic_draws(o, 30, seed=3)
    obs = np.arange(40, dtype=float) * 4
    params, scores = cv_select("M1_kernel", draws, obs, 30, 2)
    assert params["h"] >= 16.0  # pooling wins when truth is constant


def test_fold_slices():
    fs = fold_slices(30, 5)
    assert fs[0] == (0, 6) and fs[-1] == (24, 30)
    assert sum(hi - lo for lo, hi in fs) == 30
    fs10 = fold_slices(10, 5)
    assert all(hi - lo == 2 for lo, hi in fs10)
