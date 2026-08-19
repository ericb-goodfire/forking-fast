"""CPU analysis for S=1000 stores: nested-prefix o_t at S=20/50/100/200/1000,
disjoint-replicate TV noise curve extended to S=200, exact iid multinomial
null per S, and the S=200 cross-run anchor vs an earlier independent S=200
run — from the MERGED per-question raw files written by merge_s1000.py.

Definitions (all per question, Llama track; the original S=200 metric code
semantics kept verbatim, block sizes generalized from 200 to 1000 total draws):
  * nested o_t at S: per branch, histogram over the FIRST S draws (chunk-major
    draw order as generated); combined with the standard normalized
    token-probability weights (outcomes.build_outcome_vectors, verbatim).
  * replicate TV at S: carve the 1000 draws into 1000//S disjoint consecutive
    blocks; o_t per block; TV between two blocks at one position =
    0.5 * sum_c |o_A - o_B|; average over all unordered block pairs and
    positions. 50 blocks at S=20, 20 at S=50, 10 at S=100, 5 at S=200.
  * exact iid null per S: resample every branch's draws i.i.d. from its POOLED
    1000-draw histogram, run through the identical block/TV statistic
    (the reference run's calibration, S_full=1000).
  * cross-run anchor at S=200: TV between each of our 5 disjoint S=200 blocks
    and the reference run's independent full S=200 o_t on the same rows/grids,
    averaged;
    compared with the within-run S=200 replicate TV. Both runs have uniform S
    at every position (incl. t=0), so all positions enter; the t>0 restriction
    is also reported for symmetry with the reference run. Supplementary: the
    same cross-run TV at S=20/50/100 using the reference run's nested
    prefixes.

Decision rule (pre-registered, plan Q1): supported = replicate TV monotone
decreasing over S in {20,50,100,200} with log-log slope near -1/2 AND null
ratio near 1 at every S; refuted = the curve flattens above S=100.

Writes analysis out-dir: s1000_analysis.json (full) + figdata34.json (small,
plot-ready).
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import itertools

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from forking_paths.outcomes import build_outcome_vectors

S_NESTED = [20, 50, 100, 200, 1000]
S_REP = [20, 50, 100, 200]


class _B:
    __slots__ = ("idx", "tok_id", "tok_p", "is_base")

    def __init__(self, d):
        self.idx = d["t"]; self.tok_id = d["tok_id"]
        self.tok_p = d["tok_p"]; self.is_base = d["is_base"]


def o_t_from_slice(branches, rec, lo, hi):
    """o_t using draws [lo:hi) of every branch (standard weighting)."""
    answers = [b["answers"][lo:hi] for b in rec["branches"]]
    out = build_outcome_vectors(branches, answers, rec["categories"])
    return np.asarray(out["o_t"], dtype=float), out["idxs"]


def o_t_from_take(branches, rec, take_idx):
    """o_t using an arbitrary draw-index subset (interleaved-block variant)."""
    answers = [[b["answers"][i] for i in take_idx] for b in rec["branches"]]
    out = build_outcome_vectors(branches, answers, rec["categories"])
    return np.asarray(out["o_t"], dtype=float), out["idxs"]


def tv(a, b):
    """Per-position TV between two (T, C) o_t arrays -> (T,) vector."""
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum(axis=1)


def iid_null_tv(rec, S, n_reps=200, seed=0):
    """Exact iid-null replicate TV at block size S (see analyze_s200.iid_null_tv;
    identical statistic, S_full generalized). Returns (null_mean, null_sd)."""
    rng = np.random.default_rng(seed)
    cats = rec["categories"]
    cat_idx = {c: i for i, c in enumerate(cats)}
    C = len(cats)
    S_full = rec["s"]
    nb = S_full // S
    idxs = list(rec["idxs"])
    T = len(idxs)
    pos_of = {t: i for i, t in enumerate(idxs)}

    by_pos = {}
    for b in rec["branches"]:
        by_pos.setdefault(b["t"], []).append(b)
    o = np.zeros((n_reps, nb, T, C), dtype=float)
    for t, bs in by_pos.items():
        w = np.array([b["tok_p"] for b in bs], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        for b, wb in zip(bs, w):
            hist = np.zeros(C)
            for a in b["answers"]:
                hist[cat_idx.get(a, cat_idx["Other"])] += 1
            p = hist / hist.sum()
            counts = rng.multinomial(S, p, size=(n_reps, nb))  # (R, nb, C)
            o[:, :, pos_of[t], :] += wb * counts / S
    pair_acc = np.zeros(n_reps)
    n_pairs = 0
    for i, j in itertools.combinations(range(nb), 2):
        pair_acc += 0.5 * np.abs(o[:, i] - o[:, j]).sum(axis=2).mean(axis=1)
        n_pairs += 1
    per_rep = pair_acc / n_pairs
    return float(per_rep.mean()), float(per_rep.std(ddof=1))


def analyze_question(rec, ref_rec=None):
    """rec: merged S=1000 record. ref_rec: the reference run's s200 record
    for the same row."""
    branches = [_B(b) for b in rec["branches"]]
    S_full = rec["s"]
    idxs = list(rec["idxs"])
    tpos_mask = np.array([t > 0 for t in idxs])

    # nested o_t (prefix draws) at each S
    nested = {}
    for S in S_NESTED:
        o, ii = o_t_from_slice(branches, rec, 0, S)
        assert list(ii) == idxs
        nested[S] = o

    # disjoint replicate TV per S
    rep = {}
    blocks_cache = {}
    for S in S_REP:
        nb = S_full // S
        blocks = [o_t_from_slice(branches, rec, k * S, (k + 1) * S)[0]
                  for k in range(nb)]
        blocks_cache[S] = blocks
        pair_tv = [tv(blocks[i], blocks[j])
                   for i, j in itertools.combinations(range(nb), 2)]
        pair_tv = np.asarray(pair_tv)          # (n_pairs, T)
        null_mean, null_sd = iid_null_tv(rec, S)
        rep[S] = {
            "n_blocks": nb, "n_pairs": len(pair_tv),
            "mean": float(pair_tv.mean()),
            "mean_tpos": float(pair_tv[:, tpos_mask].mean()),
            "iid_null_mean": null_mean, "iid_null_sd": null_sd,
            "ratio_to_null": float(pair_tv.mean() / null_mean) if null_mean else None,
            "per_position_mean": np.round(pair_tv.mean(axis=0), 6).tolist(),
        }

    # S=200 robustness: interleaved (mod-5) blocks, which cut ACROSS the
    # generation-chunk boundaries the contiguous S=200 blocks coincide with
    nb200 = S_full // 200
    inter_blocks = [o_t_from_take(branches, rec,
                                  list(range(k, S_full, nb200)))[0]
                    for k in range(nb200)]
    inter_tv = np.asarray([tv(inter_blocks[i], inter_blocks[j])
                           for i, j in itertools.combinations(range(nb200), 2)])
    rep[200]["mean_interleaved"] = float(inter_tv.mean())

    # cross-run anchor vs the reference run (independent, same base path & grid)
    anchor = None
    if ref_rec is not None:
        ref_idxs = list(ref_rec["idxs"])
        assert ref_idxs == idxs, \
            f"row {rec['meta']['row_id']}: reference grid differs from ours"
        # branch sets must be identical (same (t, tok_id) with equal tok_p and
        # is_base flags), otherwise the anchor compares different estimands
        ours = {(b["t"], b["tok_id"]): (round(b["tok_p"], 8), b["is_base"])
                for b in rec["branches"]}
        refs = {(b["t"], b["tok_id"]): (round(b["tok_p"], 8), b["is_base"])
                for b in ref_rec["branches"]}
        assert ours == refs, \
            f"row {rec['meta']['row_id']}: branch set / tok_p differs from the reference run"
        ref_branches = [_B(b) for b in ref_rec["branches"]]
        # headline: S=200 — our 5 disjoint blocks vs the reference full S=200 o_t
        ref200 = np.asarray(ref_rec["o_t_full"], dtype=float)
        cross200 = np.asarray([tv(blk, ref200) for blk in blocks_cache[200]])
        # supplementary: cross-run TV at every S (our block k vs reference block k,
        # min(#blocks) pairs of independent same-S estimates)
        supp = {}
        for S in S_REP:
            nb_ref = ref_rec["s"] // S
            nb = min(len(blocks_cache[S]), nb_ref)
            ref_blocks = [o_t_from_slice(ref_branches, ref_rec, k * S, (k + 1) * S)[0]
                          for k in range(nb)]
            ctv = np.asarray([tv(blocks_cache[S][k], ref_blocks[k]) for k in range(nb)])
            supp[S] = {"n_pairs": nb, "mean": float(ctv.mean()),
                       "mean_tpos": float(ctv[:, tpos_mask].mean())}
        anchor = {
            "cross_run_tv_s200": float(cross200.mean()),
            "cross_run_tv_s200_tpos": float(cross200[:, tpos_mask].mean()),
            "within_run_tv_s200": rep[200]["mean"],
            "within_run_tv_s200_tpos": rep[200]["mean_tpos"],
            "supplementary_blockwise": {str(S): v for S, v in supp.items()},
        }

    return {
        "row_id": rec["meta"]["row_id"],
        "n_positions": len(idxs),
        "idxs": idxs,
        "nested_o_t": {str(S): np.round(nested[S], 6).tolist() for S in S_NESTED},
        "replicate_tv": rep,
        "anchor": anchor,
        "n_generated_tokens": rec.get("n_generated_tokens"),
        "gates": rec.get("gates"),
        "diagnostics": {k: v for k, v in (rec.get("diagnostics") or {}).items()},
    }


def pooled_verdict(per_q):
    """Pre-registered rule over the pooled curve (mean over questions)."""
    pooled = {}
    for S in S_REP:
        vals = [q["replicate_tv"][S] for q in per_q]
        pooled[S] = {
            "mean": float(np.mean([v["mean"] for v in vals])),
            "iid_null_mean": float(np.mean([v["iid_null_mean"] for v in vals])),
            "iid_null_sd": float(np.mean([v["iid_null_sd"] for v in vals])),
            "per_question": {str(q["row_id"]): v["mean"]
                             for q, v in zip(per_q, vals)},
        }
    s_arr = np.array(S_REP, dtype=float)
    tv_arr = np.array([pooled[S]["mean"] for S in S_REP])
    null_arr = np.array([pooled[S]["iid_null_mean"] for S in S_REP])
    slope = float(np.polyfit(np.log(s_arr), np.log(tv_arr), 1)[0])
    null_slope = float(np.polyfit(np.log(s_arr), np.log(null_arr), 1)[0])
    monotone = bool(np.all(np.diff(tv_arr) < 0))
    ref_curve = tv_arr[0] * np.sqrt(s_arr[0] / s_arr)
    null_ratio = (tv_arr / null_arr).tolist()
    # tail segment S=100 -> 200 (the octave the reference run could not see)
    tail_slope = float((np.log(tv_arr[3]) - np.log(tv_arr[2]))
                       / (np.log(s_arr[3]) - np.log(s_arr[2])))
    consistent_with_null = bool(np.all(np.abs(np.log(tv_arr / null_arr)) < np.log(1.15)))

    if monotone and -0.65 <= slope <= -0.35 and consistent_with_null:
        verdict = "supported"
    elif tail_slope > -0.15:
        verdict = "refuted"   # flattens above S=100: a noise floor
    else:
        verdict = "partial"
    return pooled, {
        "loglog_slope": slope, "iid_null_slope": null_slope,
        "tail_slope_100_200": tail_slope,
        "ratio_to_null": null_ratio,
        "consistent_with_iid_null": consistent_with_null,
        "monotone_decreasing": monotone,
        "sqrt_reference": {str(S): float(v) for S, v in zip(S_REP, ref_curve)},
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-dir", required=True,
                    help="dir of merged s1000_llama_rowNNN.json files")
    ap.add_argument("--ref-s200-dir", required=True,
                    help="dir of the reference run s200_llama_rowNNN.json files")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.merged_dir, "s1000_llama_row*.json")))
    files = [f for f in files if "_shard" not in os.path.basename(f)]
    assert files, f"no merged files in {args.merged_dir}"
    per_q = []
    for fp in files:
        with open(fp) as f:
            rec = json.load(f)
        rid = rec["meta"]["row_id"]
        ref_fp = os.path.join(args.ref_s200_dir, f"s200_llama_row{rid:03d}.json")
        with open(ref_fp) as f:
            ref_rec = json.load(f)
        per_q.append(analyze_question(rec, ref_rec))
        print(f"analyzed row {rid} ({per_q[-1]['n_positions']} positions)", flush=True)

    pooled, rule = pooled_verdict(per_q)

    anchors = {str(q["row_id"]): q["anchor"] for q in per_q if q["anchor"]}
    anchor_summary = None
    if anchors:
        anchor_summary = {
            "n_questions": len(anchors),
            "cross_run_tv_s200_mean": float(np.mean(
                [a["cross_run_tv_s200"] for a in anchors.values()])),
            "within_run_tv_s200_mean": float(np.mean(
                [a["within_run_tv_s200"] for a in anchors.values()])),
            "per_question": anchors,
        }

    analysis = {
        "s_nested": S_NESTED,
        "s_replicate": S_REP,
        "pooled_replicate_tv": {str(S): pooled[S] for S in S_REP},
        **rule,
        "anchor": anchor_summary,
        "per_question": per_q,
        "total_generated_tokens": int(sum(q["n_generated_tokens"] or 0 for q in per_q)),
    }
    with open(os.path.join(args.out_dir, "s1000_analysis.json"), "w") as f:
        json.dump(analysis, f)

    figdata = {
        "noise_vs_s": {
            "s": S_REP,
            "pooled_tv": [pooled[S]["mean"] for S in S_REP],
            "sqrt_ref": [rule["sqrt_reference"][str(S)] for S in S_REP],
            "iid_null": [pooled[S]["iid_null_mean"] for S in S_REP],
            "per_question": {str(q["row_id"]):
                             [q["replicate_tv"][S]["mean"] for S in S_REP]
                             for q in per_q},
            "ratio_to_null": rule["ratio_to_null"],
        },
        "loglog_slope": rule["loglog_slope"],
        "iid_null_slope": rule["iid_null_slope"],
        "tail_slope_100_200": rule["tail_slope_100_200"],
        "consistent_with_iid_null": rule["consistent_with_iid_null"],
        "monotone_decreasing": rule["monotone_decreasing"],
        "verdict": rule["verdict"],
        "anchor": anchor_summary if anchor_summary is None else {
            "cross_run": anchor_summary["cross_run_tv_s200_mean"],
            "within_run": anchor_summary["within_run_tv_s200_mean"],
            "per_question": {
                str(q["row_id"]): {
                    "cross_run_tv_s200": q["anchor"]["cross_run_tv_s200"],
                    "within_run_tv_s200": q["anchor"]["within_run_tv_s200"],
                    "interleaved_s200": q["replicate_tv"][200]["mean_interleaved"],
                } for q in per_q if q["anchor"]},
        },
        # expected S=1000 per-position noise, extrapolated 1/sqrt(S) from the
        # measured pooled S=20 replicate TV
        "expected_s1000_tv_extrapolated": float(
            pooled[20]["mean"] * np.sqrt(20 / 1000)),
        "per_question_meta": {str(q["row_id"]):
                              {"n_positions": q["n_positions"],
                               "n_generated_tokens": q["n_generated_tokens"]}
                              for q in per_q},
        "total_generated_tokens": analysis["total_generated_tokens"],
    }
    with open(os.path.join(args.out_dir, "figdata34.json"), "w") as f:
        json.dump(figdata, f)

    # slim per-question payload for pod-side figure 2 (exemplar o_t stacks)
    slim = {"per_question": [
        {"row_id": q["row_id"], "idxs": q["idxs"],
         "nested_o_t": {S: q["nested_o_t"][S] for S in ("20", "200", "1000")}}
        for q in per_q]}
    with open(os.path.join(args.out_dir, "s1000_analysis_perq_slim.json"), "w") as f:
        json.dump(slim, f)
    print(json.dumps({k: figdata[k] for k in
                      ("noise_vs_s", "loglog_slope", "iid_null_slope",
                       "tail_slope_100_200", "consistent_with_iid_null",
                       "monotone_decreasing", "verdict", "anchor",
                       "total_generated_tokens")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
