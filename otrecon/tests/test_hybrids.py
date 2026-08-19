"""Unit tests for the #44 drift-aware hybrids (toy fixtures only)."""
import numpy as np
import pytest
from ruptures.base import BaseCost

from otrecon.data import counts_from_draws
from otrecon.metrics import tv_to_gt
from otrecon.models import (LogitLinearCost, MultinomialCost,
                            MODEL_REGISTRY, SIMPLICITY_ORDER,
                            _empirical_logits, _wls_polyfit,
                            segment_positions)
from otrecon.synthetic import make_ramp_jump, make_synthetic, synthetic_draws


def test_costs_are_base_cost():
    # ruptures SILENTLY falls back to CostL2 otherwise (the #43 M4 bug).
    assert isinstance(MultinomialCost(), BaseCost)
    assert isinstance(LogitLinearCost(), BaseCost)


def test_pelt_actually_uses_custom_costs():
    import ruptures as rpt
    assert type(rpt.Pelt(custom_cost=MultinomialCost()).cost) is MultinomialCost
    assert type(rpt.Pelt(custom_cost=LogitLinearCost()).cost) is LogitLinearCost


def test_logit_linear_cost_matches_direct_wls():
    rng = np.random.default_rng(0)
    counts = rng.multinomial(30, [0.5, 0.3, 0.2], size=40).astype(float)
    x = np.arange(40) * 4.0
    y, w = _empirical_logits(counts)
    c = LogitLinearCost().fit(np.hstack([x[:, None], y, w]))
    for (s, e) in [(0, 40), (3, 17), (20, 40)]:
        direct = 0.0
        for k in range(3):
            xc = x[s:e] - x[s:e].mean()
            beta, _ = _wls_polyfit(xc, y[s:e, k], w[s:e, k], 1)
            r = y[s:e, k] - (beta[0] + beta[1] * xc)
            direct += float((w[s:e, k] * r * r).sum())
        assert c.error(s, e) == pytest.approx(direct, rel=1e-9)


def test_wls_recovers_known_line():
    x = np.linspace(-10, 10, 50)
    y = 0.3 + 0.05 * x
    w = np.full(50, 4.0)
    beta, cov = _wls_polyfit(x, y, w, 1)
    assert beta == pytest.approx([0.3, 0.05], abs=1e-10)
    # known-variance covariance: Var(b1) = 1 / sum(w * xc^2)
    assert cov[1, 1] == pytest.approx(1.0 / (w * x ** 2).sum(), rel=1e-9)


def test_linear_cost_detects_jump_not_ramp():
    """Trend-aware detection: 1 breakpoint on ramp+jump truth at moderate
    penalty; the mult cost staircases the same signal."""
    rj = make_ramp_jump()
    o, idxs = rj["o"], np.asarray(rj["idxs"], float)
    draws = synthetic_draws(o, 100, seed=3)
    counts = counts_from_draws(draws, 3, 0, 100)
    bk_lin = segment_positions(idxs, counts, "linear", 64.0)
    assert len(bk_lin) == 2
    # breakpoint within 3 grid steps of the true jump row (60)
    assert abs(bk_lin[0] - 60) <= 3
    bk_mult = segment_positions(idxs, counts, "mult", 64.0)
    assert len(bk_mult) > 2  # piecewise-constant cost staircases the ramp


@pytest.mark.parametrize("name,params", [
    ("M5a_segkernel", {"variant": "linear", "pen": 256.0, "h": 16.0}),
    ("M5b_segpoly", {"variant": "linear", "pen": 256.0, "deg": 1}),
    ("M5b_segpoly", {"variant": "mult", "pen": 64.0, "deg": 2}),
])
def test_hybrid_interface(name, params):
    syn = make_synthetic()
    o, idxs = syn["o"], np.asarray(syn["idxs"], float)
    draws = synthetic_draws(o, 30)
    counts = counts_from_draws(draws, 3, 0, 30)
    m = MODEL_REGISTRY[name]()
    m.fit(idxs, counts, 30, params)
    p = m.predict(idxs)
    assert p.shape == o.shape
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()
    lo, hi = m.credible_band(idxs, 0.9)
    assert (lo <= hi + 1e-9).all()
    assert (lo >= -1e-9).all() and (hi <= 1 + 1e-9).all()
    # point estimate is a decent reconstruction on easy truth
    assert float(tv_to_gt(p, o).mean()) < 0.15


def test_m5b_tracks_ramp_better_than_m4():
    rj = make_ramp_jump()
    o, idxs = rj["o"], np.asarray(rj["idxs"], float)
    draws = synthetic_draws(o, 100, seed=5)
    counts = counts_from_draws(draws, 3, 0, 100)
    ramp = np.zeros(len(idxs), bool)
    ramp[:60] = True
    m4 = MODEL_REGISTRY["M4_segment"]()
    m4.fit(idxs, counts, 100, {"pen": 64.0})
    m5 = MODEL_REGISTRY["M5b_segpoly"]()
    m5.fit(idxs, counts, 100, {"variant": "linear", "pen": 64.0, "deg": 1})
    tv4 = tv_to_gt(m4.predict(idxs), o)
    tv5 = tv_to_gt(m5.predict(idxs), o)
    assert tv5[ramp].mean() < tv4[ramp].mean()


def test_m5b_slope_report_shape():
    rj = make_ramp_jump()
    o, idxs = rj["o"], np.asarray(rj["idxs"], float)
    draws = synthetic_draws(o, 100, seed=5)
    counts = counts_from_draws(draws, 3, 0, 100)
    m = MODEL_REGISTRY["M5b_segpoly"]()
    m.fit(idxs, counts, 100, {"variant": "linear", "pen": 64.0, "deg": 1})
    rep = m.slope_report()
    assert len(rep) == m.n_segments
    # ramp segment: category A slope significantly negative
    seg0 = rep[0]
    assert seg0["slope"][0] < 0 and seg0["slope_z"][0] < -2


def test_m4l2_matches_legacy_fallback_behavior():
    """The 'l2' variant reproduces what #43's M4 actually did (Pelt silently
    using CostL2 when handed a non-BaseCost custom cost)."""
    import ruptures as rpt

    class _LegacyMultinomialCost:  # NOT a BaseCost: triggers the fallback
        model = "multinomial"
        min_size = 2

        def fit(self, signal):
            self.signal = signal
            return self

        def error(self, start, end):  # pragma: no cover
            return 0.0

    syn = make_synthetic()
    o, idxs = syn["o"], np.asarray(syn["idxs"], float)
    draws = synthetic_draws(o, 30)
    counts = counts_from_draws(draws, 3, 0, 30)
    legacy = rpt.Pelt(custom_cost=_LegacyMultinomialCost(), min_size=2,
                      jump=1).fit(counts).predict(pen=16.0)
    ours = segment_positions(idxs, counts, "l2", 16.0)
    assert legacy == ours


def test_default_model_is_m5a():
    from otrecon.models import DEFAULT_MODEL
    assert DEFAULT_MODEL == "M5a_segkernel"
    assert DEFAULT_MODEL in MODEL_REGISTRY


def test_registry_and_simplicity_order():
    for name in ("M4l2_segment", "M5a_segkernel", "M5b_segpoly"):
        assert name in MODEL_REGISTRY
    assert SIMPLICITY_ORDER[-2:] == ["M5a_segkernel", "M5b_segpoly"]
    assert "M4l2_segment" not in SIMPLICITY_ORDER
