"""The optimized tree must reproduce the pristine baseline tree exactly:
cv_select scores/params and full run_replicate / run_replicate_arms
records on synthetic questions (the local mini reproduction gate; the
full reproduction gate replays the stored fan records)."""
import numpy as np

from otrecon import cv as fast_cv
from otrecon.metrics import gt_jumps, region_masks
from otrecon.synthetic import make_synthetic, synthetic_draws
from src import run_ablation as fast_ra
from src import run_scale as fast_rs
from tests._baseline_loader import load_baseline

N_TOTAL = 60


def _synth_question(track="llama", n_total=N_TOTAL):
    syn = make_synthetic(T=90, step=1 if track == "llama" else 4)
    o = syn["o"]
    idxs = syn["idxs"]
    draws = synthetic_draws(o, n_total, seed=11)
    jumps = {f"{th:.2f}": gt_jumps(idxs, o, th) for th in (0.10, 0.15, 0.20)}
    prox = {th_s: region_masks(idxs, j, 10)[0] for th_s, j in jumps.items()}
    return {"row_id": 0, "track": track, "idxs": idxs, "K": o.shape[1],
            "categories": ["A", "B", "Other"], "gt": o, "draws": draws,
            "diag": {}, "n_total": n_total, "jumps": jumps, "prox": prox,
            "exp_tok_per_draw": np.ones(len(idxs)), "rec_path": "",
            "question_text": "", "answer_letter": "", "answered_rate": 1.0}


def test_cv_select_matches_baseline():
    base = load_baseline()
    q = _synth_question()
    S, stride = 15, 2
    obs_idx = np.arange(0, len(q["idxs"]), stride)
    obs_tok = np.asarray(q["idxs"], float)[obs_idx]
    block = q["draws"][obs_idx][:, :S]
    for name in ("M0_raw", "M1_kernel", "M4_segment", "M5a_segkernel"):
        b_best, b_scores = base.cv.cv_select(name, block, obs_tok, S,
                                             q["K"], n_folds=5)
        f_best, f_scores = fast_cv.cv_select(name, block, obs_tok, S,
                                             q["K"], n_folds=5)
        assert f_best == b_best, name
        assert set(f_scores) == set(b_scores)
        for k in b_scores:
            assert f_scores[k] == b_scores[k], (name, k)


def test_run_replicate_matches_baseline():
    base = load_baseline()
    q = _synth_question()
    for (S, stride, rep) in [(15, 2, 1), (30, 4, 0), (5, 8, 3)]:
        b = base.run_scale.run_replicate(q, S, stride, rep)
        f = fast_rs.run_replicate(q, S, stride, rep)
        d, bad = fast_ra.compare_to_ref(f["models"], b["models"])
        assert not bad, bad[:5]
        assert d == 0.0, d


def test_run_replicate_arms_matches_baseline():
    base = load_baseline()
    q = _synth_question()
    S, stride, rep = 15, 2, 0
    b = base.run_ablation.run_replicate_arms(q, S, stride, rep)
    f = fast_ra.run_replicate_arms(q, S, stride, rep)
    assert set(f["models"]) == set(b["models"])  # all 7 arms
    d, bad = fast_ra.compare_to_ref(f["models"], b["models"])
    assert not bad, bad[:5]
    assert d == 0.0, d
