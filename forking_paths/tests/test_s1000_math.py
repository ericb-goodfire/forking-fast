"""CPU unit tests for the S=1000 merge + analysis math (synthetic draws, no GPU)."""
import os
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analyze_s1000 import (analyze_question, o_t_from_slice, tv, _B,
                               iid_null_tv, pooled_verdict)  # noqa: E402
from src.merge_s1000 import merge_row  # noqa: E402

CATS = ["A", "B", "C", "D", "Other"]


def synth_rec(seed=0, T=6, S=1000):
    """Multinomial draws from fixed per-branch distributions: pure sampling
    noise by construction, so replicate TV must fall ~1/sqrt(S)."""
    rng = np.random.default_rng(seed)
    idxs = [0, 4, 8, 12, 16, 20][:T]
    branches = []
    for t in idxs:
        for k in range(2):  # two branches per position
            p = rng.dirichlet(np.ones(5) * 2.0)
            draws = rng.choice(CATS, size=S, p=p).tolist()
            branches.append({"t": t, "tok_id": 100 + k, "tok_p": 0.5 + 0.3 * k,
                             "is_base": k == 0, "answers": draws,
                             "cont_lens": [10] * S})
    return {"meta": {"row_id": 99}, "s": S, "chunk_size": 200,
            "chunk_seeds": [100000, 200000, 300000, 400000, 500000],
            "idxs": idxs, "categories": CATS, "branches": branches,
            "gates": {}, "diagnostics": {}, "n_generated_tokens": 60000}


def synth_ref(rec, seed=1000, S=200):
    """An INDEPENDENT S=200 run of the same question (same branch dists)."""
    rng = np.random.default_rng(seed)
    ref = {"meta": dict(rec["meta"]), "s": S, "idxs": list(rec["idxs"]),
           "categories": CATS, "branches": []}
    for b in rec["branches"]:
        hist = np.zeros(5)
        for a in b["answers"]:
            hist[CATS.index(a)] += 1
        p = hist / hist.sum()
        ref["branches"].append({**{k: b[k] for k in ("t", "tok_id", "tok_p", "is_base")},
                                "answers": rng.choice(CATS, size=S, p=p).tolist(),
                                "cont_lens": [10] * S})
    bo = [_B(b) for b in ref["branches"]]
    ref["o_t_full"] = o_t_from_slice(bo, ref, 0, S)[0].tolist()
    return ref


def test_block_mean_equals_full():
    rec = synth_rec()
    branches = [_B(b) for b in rec["branches"]]
    full, _ = o_t_from_slice(branches, rec, 0, 1000)
    acc = np.zeros_like(full)
    for k in range(50):
        acc += o_t_from_slice(branches, rec, k * 20, (k + 1) * 20)[0]
    assert np.allclose(acc / 50, full, atol=1e-12)


def test_nested_rows_sum_to_one():
    q = analyze_question(synth_rec())
    for S in ("20", "50", "100", "200", "1000"):
        o = np.asarray(q["nested_o_t"][S])
        assert np.allclose(o.sum(axis=1), 1.0, atol=1e-5)


def test_interleaved_s200_close_to_contiguous_on_synthetic():
    """On iid synthetic draws, interleaved-block S=200 TV must sit near the
    contiguous-block S=200 TV (block definition doesn't matter under iid)."""
    cont, inter = [], []
    for seed in range(8):
        q = analyze_question(synth_rec(seed=seed, T=6))
        cont.append(q["replicate_tv"][200]["mean"])
        inter.append(q["replicate_tv"][200]["mean_interleaved"])
    r = np.mean(inter) / np.mean(cont)
    assert 0.9 < r < 1.1, r


def test_anchor_rejects_branch_mismatch():
    rec = synth_rec(seed=2, T=6)
    ref = synth_ref(rec, seed=7000)
    ref["branches"][0]["tok_p"] = 0.999  # corrupt one branch weight
    with pytest.raises(AssertionError):
        analyze_question(rec, ref)


def test_replicate_block_counts():
    q = analyze_question(synth_rec())
    assert q["replicate_tv"][20]["n_blocks"] == 50
    assert q["replicate_tv"][50]["n_blocks"] == 20
    assert q["replicate_tv"][100]["n_blocks"] == 10
    assert q["replicate_tv"][200]["n_blocks"] == 5
    assert q["replicate_tv"][20]["n_pairs"] == 50 * 49 // 2
    assert q["replicate_tv"][200]["n_pairs"] == 10


def test_tv_range_and_selfzero():
    a = np.array([[0.5, 0.5, 0, 0, 0]])
    b = np.array([[0, 0, 0.5, 0.5, 0]])
    assert np.allclose(tv(a, a), 0.0)
    assert np.allclose(tv(a, b), 1.0)


def test_iid_null_matches_measured_on_synthetic():
    # synthetic data IS iid multinomial, so measured TV must sit on the null
    ratios = []
    for seed in range(6):
        rec = synth_rec(seed=seed, T=6)
        q = analyze_question(rec)
        for S in (20, 100, 200):
            null_m, _ = iid_null_tv(rec, S, n_reps=100, seed=seed)
            ratios.append(q["replicate_tv"][S]["mean"] / null_m)
    m = np.mean(ratios)
    assert 0.9 < m < 1.1, m


def test_replicate_tv_falls_like_sqrt_through_200():
    tvs = {S: [] for S in (20, 50, 100, 200)}
    for seed in range(12):
        q = analyze_question(synth_rec(seed=seed, T=6))
        for S in tvs:
            tvs[S].append(q["replicate_tv"][S]["mean"])
    means = [np.mean(tvs[S]) for S in (20, 50, 100, 200)]
    assert means[0] > means[1] > means[2] > means[3]
    slope = np.polyfit(np.log([20, 50, 100, 200]), np.log(means), 1)[0]
    assert -0.65 < slope < -0.35, slope


def test_verdict_supported_on_synthetic():
    per_q = [analyze_question(synth_rec(seed=s, T=6)) for s in (0, 1)]
    _, rule = pooled_verdict(per_q)
    assert rule["verdict"] == "supported", rule


def test_verdict_refuted_on_floor():
    """Inject a common per-block systematic component: TV flattens -> refuted."""
    per_q = []
    for s in (0, 1):
        q = analyze_question(synth_rec(seed=s, T=6))
        # overwrite replicate means with a floor-dominated curve
        for S, val in zip((20, 50, 100, 200), (0.13, 0.10, 0.09, 0.089)):
            q["replicate_tv"][S]["mean"] = val
            q["replicate_tv"][S]["iid_null_mean"] = 0.13 * np.sqrt(20 / S)
        per_q.append(q)
    _, rule = pooled_verdict(per_q)
    assert rule["verdict"] == "refuted", rule


def test_cross_run_anchor_matches_within_run_on_synthetic():
    """Independent same-distribution runs: cross-run TV at S=200 should sit
    near the within-run S=200 replicate TV."""
    cross, within = [], []
    for seed in range(8):
        rec = synth_rec(seed=seed, T=6)
        ref = synth_ref(rec, seed=5000 + seed)
        q = analyze_question(rec, ref)
        cross.append(q["anchor"]["cross_run_tv_s200"])
        within.append(q["anchor"]["within_run_tv_s200"])
    r = np.mean(cross) / np.mean(within)
    assert 0.85 < r < 1.15, r


def _shardify(rec, num_shards=3):
    """Split a synthetic merged rec into shard records (inverse of merge)."""
    idxs = rec["idxs"]
    shards = [list(map(int, s)) for s in np.array_split(idxs, num_shards)]
    out = []
    for si, pos in enumerate(shards):
        bs = [b for b in rec["branches"] if b["t"] in set(pos)]
        bo = [_B(b) for b in bs]
        sub = {"branches": bs, "categories": CATS, "s": rec["s"]}
        o_t = o_t_from_slice(bo, sub, 0, rec["s"])[0]
        out.append({
            "meta": rec["meta"], "s": rec["s"], "chunk_size": rec["chunk_size"],
            "chunk_seeds": [si * 10000000 + (c + 1) * 100000 for c in range(5)],
            "shard": {"index": si, "num_shards": num_shards, "positions": pos},
            "full_idxs": list(idxs), "idxs": pos, "categories": CATS,
            "base": {"gen_ids": [1, 2, 3], "n_response_tokens": 3,
                     "finish_reason": "stop", "base_text": "x",
                     "token_texts": ["a", "b", "c"]},
            "branches": bs,
            "o_t_full": np.round(o_t, 6).tolist(),
            "gates": {"ref_match": True, "block_mean_ok": True,
                      "block_mean_max_dev": 0.0},
            "diagnostics": {"n_continuations": 1},
            "timing": {"gen_secs": 1.0, "total_secs": 2.0},
            "n_generated_tokens": 100,
            "config": {"seed": 0},
        })
    return out


def test_merge_tiles_grid_and_matches_unsharded(tmp_path):
    rec = synth_rec(seed=3, T=6)
    shard_files = []
    for sr in _shardify(rec, num_shards=3):
        fp = tmp_path / f"s1000_llama_row099_shard{sr['shard']['index']}of3.json"
        fp.write_text(json.dumps(sr))
        shard_files.append(str(fp))
    merged = merge_row(shard_files)
    assert merged["idxs"] == rec["idxs"]
    assert len(merged["branches"]) == len(rec["branches"])
    # o_t of merged == o_t computed on the unsharded record
    bo = [_B(b) for b in rec["branches"]]
    full, _ = o_t_from_slice(bo, rec, 0, 1000)
    assert np.allclose(np.asarray(merged["o_t_full"]), np.round(full, 6), atol=1e-6)
    assert merged["gates"]["merge_grid_ok"]


def test_merge_rejects_missing_shard(tmp_path):
    rec = synth_rec(seed=4, T=6)
    srs = _shardify(rec, num_shards=3)[:2]  # drop one shard
    files = []
    for sr in srs:
        fp = tmp_path / f"s1000_llama_row099_shard{sr['shard']['index']}of3.json"
        fp.write_text(json.dumps(sr))
        files.append(str(fp))
    with pytest.raises(AssertionError):
        merge_row(files)
