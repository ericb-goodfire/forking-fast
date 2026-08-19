"""Tests for the scale driver: LRO reference at every S, matched-budget
pairing, noise floors, spot-row reproduction, gz round trip."""
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.run_scale import (S_GRID, STRIDES, _ci, _region_summary,
                           build_tasks, load_record_gz, noise_floors,
                           prepare_question, run_replicate,
                           spot_rows_by_track)


def _toy_question(n_total=200, T=12, K=3):
    idxs = list(range(T))
    draws = np.ones((T, n_total), dtype=np.int8)
    gt = np.tile(np.array([0.0, 1.0, 0.0]), (T, 1))
    prox = {t: np.zeros(T, dtype=bool) for t in ("0.10", "0.15", "0.20")}
    return {"row_id": 1, "track": "llama", "idxs": idxs, "K": K, "gt": gt,
            "draws": draws, "n_total": n_total, "prox": prox,
            "exp_tok_per_draw": np.full(T, 10.0)}


def test_lro_reference_excludes_block_at_every_s():
    """Block draws category 0, held-out draws category 1: raw fit scores
    TV ~1 against the LRO reference at EVERY S in the grid."""
    for S in (5, 30, 100):
        q = _toy_question()
        q["draws"][:, :S] = 0     # replicate 0 pure cat 0; rest cat 1
        # full-pool curve consistent with the draws: S/200 cat0, rest cat1
        frac = S / 200.0
        q["gt"] = np.tile(np.array([frac, 1.0 - frac, 0.0]), (12, 1))
        out = run_replicate(q, S=S, stride=1, rep=0, models=["M0_raw"],
                            arms=("fixed",))
        ent = out["models"]["M0_raw|fixed"]
        assert abs(ent["tv"]["overall"] - 1.0) < 1e-6, (S, ent["tv"])
        # full-pool secondary metric mixes the block back in
        assert abs(ent["tv_full"]["overall"] - (1.0 - frac)) < 1e-6
        # tv_full now carries the same region split as tv
        assert "flat@0.15" in ent["tv_full"]


def test_lro_reference_is_block_specific():
    """Replicate 1's reference must exclude replicate 1, not replicate 0."""
    S = 30
    q = _toy_question()
    q["draws"][:, S:2 * S] = 0
    out = run_replicate(q, S=S, stride=1, rep=1, models=["M0_raw"],
                        arms=("fixed",))
    assert abs(out["models"]["M0_raw|fixed"]["tv"]["overall"] - 1.0) < 1e-6


def test_perfect_fit_scores_reference_noise_only():
    """A deterministic question: every model scores ~0 vs the LRO ref."""
    q = _toy_question()
    out = run_replicate(q, S=30, stride=1, rep=0)
    for ma, ent in out["models"].items():
        # Dirichlet pseudo-counts keep smoothed point masses at ~2e-3;
        # this is the same analytic near-zero limit gate B enforces (<0.01)
        assert ent["tv"]["overall"] < 0.01, (ma, ent["tv"])


def test_stride_subsamples_grid():
    q = _toy_question(T=12)
    out = run_replicate(q, S=10, stride=4, rep=0, models=["M0_raw"],
                        arms=("fixed",))
    assert out["n_obs"] == 3


def test_build_tasks_counts():
    q = _toy_question()
    tasks = build_tasks({("llama", 1): q})
    # per S: floor(200/S) reps x 7 strides
    exp = sum((200 // S) * len(STRIDES["llama"]) for S in S_GRID)
    assert len(tasks) == exp
    qd = _toy_question()
    qd["track"] = "deepseek"
    tasks_d = build_tasks({("deepseek", 1): qd})
    exp_d = sum((200 // S) * len(STRIDES["deepseek"]) for S in S_GRID)
    assert len(tasks_d) == exp_d


def test_region_summary_thresholds():
    vals = np.array([1.0, 0.0, 0.0, 0.0])
    prox = {"0.10": np.array([True, False, False, False]),
            "0.15": np.array([False, False, False, False]),
            "0.20": np.array([True, True, False, False])}
    out = _region_summary(vals, prox)
    assert out["jump@0.10"] == 1.0
    assert out["jump@0.15"] is None
    assert out["jump@0.20"] == 0.5
    assert abs(out["flat@0.10"] - 0.0) < 1e-12
    assert abs(out["overall"] - 0.25) < 1e-12


def test_ci_paired_math():
    ci = _ci([1.0, 2.0, 3.0])
    assert abs(ci["mean"] - 2.0) < 1e-12
    assert ci["n"] == 3 and ci["lo"] < 2.0 < ci["hi"]
    assert _ci([None, 5.0])["n"] == 1


def test_noise_floors_shrink_with_more_reference_draws():
    gt = np.tile(np.array([0.5, 0.5, 0.0]), (6, 1))
    fl = noise_floors(gt, s_grid=[5, 100], n_sim=300)
    # S=5 leaves 195 reference draws -> smaller ref floor than S=100 (100)
    assert fl["5"]["ref_floor"] < fl["100"]["ref_floor"]
    # raw floor dominated by the S-draw estimate -> S=5 much noisier
    assert fl["5"]["raw_floor"] > fl["100"]["raw_floor"]


def test_spot_rows_sequential_rng():
    """Spot rows: llama drawn first, deepseek second, from ONE rng(42)."""
    spots = spot_rows_by_track()
    rng = np.random.default_rng(42)
    a = sorted(rng.choice(50, size=10, replace=False).tolist())
    b = sorted(rng.choice(50, size=10, replace=False).tolist())
    assert spots["llama"] == a and spots["deepseek"] == b
    assert spots["llama"] != spots["deepseek"]


def test_prepare_question_gz_roundtrip(tmp_path):
    """Minimal store-format record loads through prepare_question."""
    T, S = 4, 200
    branches = []
    for t in range(T):
        branches.append({"t": t, "tok_id": 1, "tok_p": 0.7, "is_base": True,
                         "answers": ["A"] * S, "cont_lens": [3] * S})
        branches.append({"t": t, "tok_id": 2, "tok_p": 0.3, "is_base": False,
                         "answers": ["B"] * S, "cont_lens": [4] * S})
    rec = {"s": S, "categories": ["A", "B", "C", "D", "Other"],
           "idxs": list(range(T)), "branches": branches,
           "meta": {"row_id": 7, "question": "toy?", "answer_letter": "A"},
           "diagnostics": {"answered_rate": 0.9}}
    p = tmp_path / "row007.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump(rec, f)
    q = prepare_question(str(p), "llama")
    assert q["row_id"] == 7 and q["n_total"] == 200
    assert q["draws"].shape == (T, 200)
    # gt: 0.7 A + 0.3 B everywhere; K collapses to {A, B, Other}
    assert q["K"] == 3
    assert abs(q["gt"][0][q["categories"].index("A")] - 0.7) < 1e-12
    got = load_record_gz(str(p))
    assert got["s"] == 200


def test_deepseek_g1_masks(tmp_path):
    """Sentence-grid records get +-1-grid-position jump masks (critic f2)."""
    T, S = 4, 200
    idxs = [0, 100, 300, 600]   # big token gaps, as real sentence grids
    branches = []
    for i, t in enumerate(idxs):
        cat = "A" if i < 2 else "B"    # jump between grid points 1 and 2
        branches.append({"t": t, "tok_id": 1, "tok_p": 1.0, "is_base": True,
                         "answers": [cat] * S, "cont_lens": [3] * S})
    rec = {"s": S, "categories": ["A", "B", "C", "D", "Other"],
           "idxs": idxs, "branches": branches,
           "meta": {"row_id": 3, "question": "toy?", "answer_letter": "A"},
           "diagnostics": {"answered_rate": 0.5}}
    p = tmp_path / "row003.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump(rec, f)
    q = prepare_question(str(p), "deepseek")
    # token-radius mask (+-10 tok) covers only the jump grid point (t=300)
    assert q["prox"]["0.15"].tolist() == [False, False, True, False]
    # grid-radius mask covers the jump point +- 1 grid position
    assert q["prox"]["g1@0.15"].tolist() == [False, True, True, True]
    out = run_replicate(q, S=30, stride=1, rep=0, models=["M0_raw"],
                        arms=("fixed",))
    ent = out["models"]["M0_raw|fixed"]["tv"]
    assert "jumpg1@0.15" in ent and "flatg1@0.15" not in ent


def test_llama_has_no_g1_masks():
    q = _toy_question()
    assert not any(k.startswith("g1@") for k in q["prox"])


def test_load_fan_dedupes_latest(tmp_path):
    import time as _t
    from src.run_scale import load_fan
    rec_a = {"track": "deepseek", "row": 1, "S": 5, "stride": 1, "rep": 0,
             "models": {"x": 1}}
    rec_b = dict(rec_a, models={"x": 2})
    with open(tmp_path / "fan.jsonl", "w") as f:
        f.write(json.dumps(rec_a) + "\n")
    _t.sleep(0.05)
    with open(tmp_path / "fan.ds2.jsonl", "w") as f:
        f.write(json.dumps(rec_b) + "\n")
    _, results = load_fan(str(tmp_path))
    assert len(results) == 1 and results[0]["models"]["x"] == 2
