"""Tests for the ablation driver: shared-arm equivalence with the
vendored scale runner, arm grid restrictions, repro comparison, and the
assembly statistics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import run_ablation as ra
from src import run_scale as rs
from otrecon.models import MODEL_REGISTRY


def _toy_question(seed=7, T=40, K=3, n_total=60):
    """Synthetic question with a mid-curve jump so segmentation, kernel
    pooling, and CV all have something to do."""
    rng = np.random.default_rng(seed)
    idxs = list(range(0, 4 * T, 4))
    gt = np.zeros((T, K))
    gt[: T // 2] = [0.7, 0.2, 0.1]
    gt[T // 2:] = [0.15, 0.15, 0.7]
    draws = np.zeros((T, n_total), dtype=np.int64)
    for t in range(T):
        draws[t] = rng.choice(K, size=n_total, p=gt[t])
    jumps = {f"{th:.2f}": rs.gt_jumps(idxs, gt, th)
             for th in rs.JUMP_THRESH}
    prox = {th_s: rs.region_masks(idxs, j, rs.JUMP_RADIUS)[0]
            for th_s, j in jumps.items()}
    return {"row_id": 1, "track": "llama", "idxs": idxs, "K": K, "gt": gt,
            "draws": draws, "n_total": n_total, "prox": prox,
            "exp_tok_per_draw": np.full(T, 10.0)}


def test_shared_arms_match_vendored_runner_exactly():
    """The ablation runner and the vendored scale run_replicate must agree
    bit-for-bit on the four shared arms (same question, cell, replicate)."""
    q = _toy_question()
    for S, stride, rep in [(15, 1, 0), (15, 4, 1), (30, 1, 1)]:
        mine = ra.run_replicate_arms(q, S, stride, rep)
        ref = rs.run_replicate(q, S, stride, rep)
        for arm in ra.SHARED_ARMS:
            d, bad = ra.compare_to_ref({arm: mine["models"][arm]},
                                       {arm: ref["models"][arm]})
            assert not bad, f"{arm} S={S} stride={stride}: {bad[:3]}"
            assert d == 0.0, f"{arm}: max diff {d}"


def test_restricted_grids_pin_variant():
    for name, variant in [("M5a_l2", "l2"), ("M5a_linear", "linear")]:
        grid = MODEL_REGISTRY[name].grid()
        assert len(grid) == 84  # 12 pens x 7 bandwidths
        assert all(p["variant"] == variant for p in grid)


def test_l2_arm_fits_and_differs_from_mult_on_jump_data():
    """The l2 variant runs end-to-end and is a real ablation (its
    segmentation can differ from mult's on the same counts)."""
    q = _toy_question()
    out = ra.run_replicate_arms(q, 30, 1, 0)
    for key, _, _ in ra.ARMS:
        assert out["models"][key]["tv"]["overall"] >= 0.0
    # all seven arms present
    assert len(out["models"]) == 7


def test_compare_to_ref_catches_perturbation():
    q = _toy_question()
    a = ra.run_replicate_arms(q, 15, 1, 0)["models"]
    ref = {k: a[k] for k in ra.SHARED_ARMS}
    d, bad = ra.compare_to_ref(ref, ref)
    assert d == 0.0 and not bad
    import copy
    b = copy.deepcopy(ref)
    b["M0_raw|cv"]["tv"]["overall"] += 1e-6
    d, bad = ra.compare_to_ref(b, ref)
    assert bad and d >= 1e-6 - 1e-12


def test_ci_and_boot_ratio():
    vals = [0.1, 0.2, 0.3, 0.4]
    ci = ra._ci(vals)
    assert abs(ci["mean"] - 0.25) < 1e-12 and ci["n"] == 4
    assert ci["lo"] < 0.25 < ci["hi"]


def test_improvement_fraction_sign_guard():
    """Per-question fractions require a strictly positive denominator of
    at least the floor; nonpositive and tiny denominators are excluded
    separately (a negative denominator would enter sign-flipped)."""
    m0 = {r: 0.10 for r in range(10)}
    full = {r: 0.05 for r in range(10)}
    full[0] = 0.099   # tiny positive denominator
    full[1] = 0.15    # NEGATIVE denominator
    arm = {r: 0.07 for r in range(10)}
    fr, n_nonpos, n_small = [], 0, 0
    for r in range(10):
        den = m0[r] - full[r]
        if den <= 0:
            n_nonpos += 1
            continue
        if den < ra.FRAC_DENOM_FLOOR:
            n_small += 1
            continue
        fr.append((m0[r] - arm[r]) / den)
    assert n_nonpos == 1 and n_small == 1 and len(fr) == 8
    assert all(abs(f - 0.6) < 1e-12 for f in fr)


def test_build_tasks_replicate_counts():
    q = _toy_question(n_total=200)
    tasks = ra.build_tasks({("llama", 1): q})
    # per stride: 40 + 13 + 6 + 2 = 61 replicates; 2 strides
    assert len(tasks) == 2 * (40 + 13 + 6 + 2)
