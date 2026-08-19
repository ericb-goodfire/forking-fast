"""The cheap-sampling power analysis at scale (replicate fan + assembly).

Runs the replicate power analysis against the S=200 dataset: tinyMMLU
questions, Llama-3-8B-Instruct on an every-token grid (spacing 1-64 tokens)
and DeepSeek-R1-Distill-Llama-8B on sentence grids (spacing 1-8 sentences).
All reconstruction machinery lives in the otrecon package.

Design notes:
  * Reference is ALWAYS leave-replicate-out (LRO): each replicate block of
    S draws is scored against the empirical histogram of the 200-S draws
    outside the block. n_total=200, so a full-pool reference would share up
    to 50% of its draws with the replicate. TV vs the full-pool curve is
    recorded as a secondary diagnostic (tv_full).
  * Jump/flat masks are defined once per question on the full-pool S=200
    curve at thresholds 0.10/0.15/0.20 (radius 10 tokens).
  * CIs are across questions: the question is the split unit; every
    per-question value is first averaged over that question's replicates.
  * Per-question multinomial noise floors are simulated from the full-pool
    curve so the S-dependent reference noise is quantified, not hidden.

Modes:
  gate      -- (a) exact reproduction of the source dataset's recorded
               validation numbers (spot-row replicate TV at S=20/50/100 +
               log-log slope, both tracks);
               (b) end-to-end LRO path on a deterministic-outcome question:
               analytic near-zero TV limit for all three models.
  fan       -- replicate fan; shard JSONL output per (track, row).
  assemble  -- cells + CIs, Q1 paired fork verdict, Q2 matched-budget
               spacing comparison, Q3 transfer read, floors, frontier,
               operating points.
  dashboard -- per-combo nested-prefix reconstructions for the explorer.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from otrecon.cv import cv_select
from otrecon.data import (branch_groups, collapse_curve, collapse_map,
                          counts_from_draws, mixture_draws,
                          normalized_weights, weighted_o_t)
from otrecon.metrics import gt_jumps, region_masks, tv_to_gt
from otrecon.models import MODEL_REGISTRY

TRACKS = ("llama", "deepseek")
ROWS = list(range(50))
S_GRID = [5, 10, 15, 20, 30, 50, 75, 100]
STRIDES = {"llama": [1, 2, 4, 8, 16, 32, 64],   # grid unit = 1 token
           "deepseek": [1, 2, 4, 8]}            # grid unit = 1 sentence
N_TOTAL = 200
MODELS = ["M0_raw", "M4_segment", "M5a_segkernel"]
ARMS = ("cv", "fixed")
JUMP_THRESH = [0.10, 0.15, 0.20]
JUMP_RADIUS = 10          # tokens, both tracks (fixed convention)
BAND_LEVEL = 0.9
# Mixture-draw seed bases. 43_000_000 is the historical convention; the
# deepseek track gets its own base so the two tracks never share
# (seed_base, row_id, t) triples (row ids 0-49 collide across tracks).
SEED_BASE = {"llama": 43_000_000, "deepseek": 43_500_000}

# Fixed-default hyperparameters (picked on synthetic gates; never tuned on
# real data).
FIXED_PARAMS = {
    "M0_raw": {"kind": "linear"},
    "M4_segment": {"pen": 8.0},
    "M5a_segkernel": {"variant": "mult", "pen": 64.0, "h": 32.0},
}

# Known-answer gate targets: the source S=200 dataset's recorded validation
# summary, reproduced exactly.
GATE_TARGETS = {
    "llama": {"tv": {"20": 0.09278673114484523, "50": 0.060271479316052964,
                     "100": 0.04306018455586748},
              "slope": -0.4766807062437436},
    "deepseek": {"tv": {"20": 0.0683485215087332, "50": 0.045067751149052676,
                        "100": 0.032712149829444134},
                 "slope": -0.4576701880124801},
}

# Status-quo cell per track (the baseline FPA operating point):
# llama = S=30 at 4-token spacing (stride 4 on the every-token grid);
# deepseek = S=30 at the sentence grid (stride 1).
STATUS_QUO_STRIDE = {"llama": 4, "deepseek": 1}
SPACING_UNIT = {"llama": "tokens", "deepseek": "sentences"}


def load_record_gz(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def rec_path(data_dir: str, track: str, row: int) -> str:
    return os.path.join(data_dir, track, f"row{row:03d}.json.gz")


# ---------------------------------------------------------------------------
# Question preparation (gz + per-track seed).
# ---------------------------------------------------------------------------

def prepare_question(path: str, track: str) -> dict:
    rec = load_record_gz(path)
    cats = rec["categories"]
    idxs_re, o_recomputed = weighted_o_t(rec)
    assert list(rec["idxs"]) == idxs_re, "o_t_full grid mismatch"
    kept, idx_map = collapse_map(o_recomputed, cats)
    K = len(kept)
    gt = collapse_curve(o_recomputed, idx_map, K)
    n_total = rec["s"]
    assert n_total == N_TOTAL, f"unexpected n_total {n_total}"
    idxs, draws, diag = mixture_draws(rec, idx_map, K, n_total=n_total,
                                      seed_base=SEED_BASE[track])
    assert idxs == idxs_re
    jumps = {f"{th:.2f}": gt_jumps(idxs, gt, th) for th in JUMP_THRESH}
    prox = {th_s: region_masks(idxs, j, JUMP_RADIUS)[0]
            for th_s, j in jumps.items()}
    # Grid-position-radius variant (critic f2): on the DeepSeek sentence
    # grid the ±10-TOKEN radius collapses to the jump sentence alone
    # (inter-sentence gaps are 16-507 tokens), so also record a mask of
    # ±1 GRID position around each jump (the adjacent sentences). On the
    # llama every-token grid this variant is redundant and skipped.
    if track == "deepseek":
        pos = np.asarray(idxs, float)
        for th_s, j in jumps.items():
            if not j:
                prox[f"g1@{th_s}"] = np.zeros(len(pos), dtype=bool)
                continue
            jidx = np.flatnonzero(np.isin(pos, np.asarray(j, float)))
            m = np.zeros(len(pos), dtype=bool)
            for ji in jidx:
                m[max(0, ji - 1):ji + 2] = True
            prox[f"g1@{th_s}"] = m
    by_t = branch_groups(rec)
    exp_tok = np.zeros(len(idxs))
    for row_i, t in enumerate(idxs):
        bs = by_t[t]
        w = normalized_weights(bs)
        exp_tok[row_i] = float(sum(
            wb * np.mean(b["cont_lens"]) for b, wb in zip(bs, w)))
    return {"row_id": rec["meta"]["row_id"], "track": track,
            "idxs": idxs, "K": K, "categories": kept, "gt": gt,
            "draws": draws, "diag": diag, "n_total": n_total,
            "jumps": jumps, "prox": prox, "exp_tok_per_draw": exp_tok,
            "rec_path": path,
            "question_text": rec["meta"].get("question", ""),
            "answer_letter": rec["meta"].get("answer_letter", ""),
            "answered_rate": rec.get("diagnostics", {}).get("answered_rate")}


# ---------------------------------------------------------------------------
# Replicate metrics with the leave-replicate-out reference.
# ---------------------------------------------------------------------------

def _heldout_ll_per_position(pred, draws, lo, hi, eps=1e-9):
    p = np.maximum(np.asarray(pred, float), eps)
    p = p / p.sum(axis=1, keepdims=True)
    held = np.concatenate([draws[:, :lo], draws[:, hi:]], axis=1)
    ll = np.log(np.take_along_axis(p, held.astype(int), axis=1))
    return ll.mean(axis=1)


def _region_summary(vals: np.ndarray, prox_by_th: dict) -> dict:
    out = {"overall": float(vals.mean())}
    for th_s, prox in prox_by_th.items():
        if th_s.startswith("g1@"):   # grid-radius variant: jump side only
            out[f"jump{th_s}"] = (float(vals[prox].mean()) if prox.any()
                                  else None)
            continue
        out[f"jump@{th_s}"] = float(vals[prox].mean()) if prox.any() else None
        out[f"flat@{th_s}"] = float(vals[~prox].mean())
    return out


def run_replicate(q: dict, S: int, stride: int, rep: int,
                  models=MODELS, arms=ARMS) -> dict:
    idxs = np.asarray(q["idxs"], float)
    K = q["K"]
    obs_idx = np.arange(0, len(idxs), stride)
    obs_tok = idxs[obs_idx]
    lo, hi = rep * S, (rep + 1) * S
    block_obs = q["draws"][obs_idx][:, lo:hi]
    counts = counts_from_draws(block_obs, K, 0, S)
    # LRO reference at EVERY S: histogram of the 200-S draws outside the
    # block, at every grid position.
    held = np.concatenate([q["draws"][:, :lo], q["draws"][:, hi:]], axis=1)
    cnt = counts_from_draws(held, K, 0, held.shape[1])
    gt_lro = cnt / np.maximum(cnt.sum(axis=1, keepdims=True), 1e-12)
    prox = q["prox"]   # masks defined on the full-pool curve
    out = {"row": q["row_id"], "track": q["track"], "S": S,
           "stride": stride, "rep": rep, "n_obs": len(obs_idx),
           "models": {}}
    for name in models:
        for arm in arms:
            if arm == "cv":
                params, _ = cv_select(name, block_obs, obs_tok, S, K,
                                      n_folds=min(5, S))
            else:
                params = dict(FIXED_PARAMS[name])
            m = MODEL_REGISTRY[name]()
            m.fit(obs_tok, counts, S, params)
            pred = m.predict(idxs)
            tv = tv_to_gt(pred, gt_lro)
            tv_full = tv_to_gt(pred, q["gt"])
            ll = _heldout_ll_per_position(pred, q["draws"], lo, hi)
            lo_b, hi_b = m.credible_band(idxs, BAND_LEVEL)
            covv = ((gt_lro >= lo_b - 1e-12)
                    & (gt_lro <= hi_b + 1e-12)).mean(axis=1)
            rec = {"params": params,
                   "tv": _region_summary(tv, prox),
                   "tv_full": _region_summary(tv_full, prox),
                   "ll": _region_summary(ll, prox),
                   "cov": {"overall": float(covv.mean())}}
            if hasattr(m, "bkps"):
                rec["n_segments"] = int(len(m.bkps))
            out["models"][f"{name}|{arm}"] = rec
    return out


# ---------------------------------------------------------------------------
# Noise floors: multinomial simulation from the full-pool curve.
# ---------------------------------------------------------------------------

def noise_floors(gt: np.ndarray, s_grid=S_GRID, n_total=N_TOTAL,
                 n_sim=400, seed=58) -> dict:
    """Per-S expected pooled TV floors: ref-noise floor = E TV(hist_{n-S},
    gt) (what a perfect prediction of the true curve scores against the LRO
    reference); raw floor = E TV(hist_S, hist_{n-S}) (what the raw counts
    estimate scores)."""
    rng = np.random.default_rng(seed)
    T = gt.shape[0]
    # numpy multinomial rejects pvals whose float sum strays past 1
    gt = np.clip(np.asarray(gt, float), 0.0, None)
    gt = gt / gt.sum(axis=1, keepdims=True)
    out = {}
    for S in s_grid:
        m = n_total - S
        ref_tvs = np.zeros(T)
        raw_tvs = np.zeros(T)
        for t in range(T):
            a = rng.multinomial(m, gt[t], size=n_sim) / m
            b = rng.multinomial(S, gt[t], size=n_sim) / S
            ref_tvs[t] = 0.5 * np.abs(a - gt[t]).sum(axis=1).mean()
            raw_tvs[t] = 0.5 * np.abs(b - a).sum(axis=1).mean()
        out[str(S)] = {"ref_floor": float(ref_tvs.mean()),
                       "raw_floor": float(raw_tvs.mean())}
    return out


# ---------------------------------------------------------------------------
# Gates.
# ---------------------------------------------------------------------------

def spot_rows_by_track(n_rows=50, spot_n=10) -> dict:
    """Spot-row selection: ONE rng seeded 42, llama drawn first."""
    rng = np.random.default_rng(42)
    out = {}
    for track in TRACKS:
        out[track] = sorted(rng.choice(n_rows, size=spot_n,
                                       replace=False).tolist())
    return out


def replicate_tvs_weighted(rec: dict, s_blocks=(20, 50, 100)) -> dict:
    """Replicate TVs: mean pairwise pooled TV between disjoint-block
    weighted o_t estimates (5 base categories, all grid positions)."""
    s_full = rec["s"]
    out = {}
    for s_block in s_blocks:
        n_blk = s_full // s_block
        if n_blk < 2:
            continue
        curves = [weighted_o_t(rec, k * s_block, (k + 1) * s_block)[1]
                  for k in range(n_blk)]
        tvs = [0.5 * np.abs(curves[i] - curves[j]).sum(axis=1).mean()
               for i in range(n_blk) for j in range(i + 1, n_blk)]
        out[str(s_block)] = float(np.mean(tvs))
    return out


def gate_a_repro(data_dir: str) -> dict:
    """Reproduce the source dataset's per-track replicate TV means + slopes
    exactly."""
    spots = spot_rows_by_track()
    res = {}
    ok_all = True
    for track in TRACKS:
        per_s = {s: [] for s in ("20", "50", "100")}
        for row in spots[track]:
            rec = load_record_gz(rec_path(data_dir, track, row))
            tvs = replicate_tvs_weighted(rec)
            for s, v in tvs.items():
                per_s[s].append(v)
        pooled = {s: float(np.mean(v)) for s, v in per_s.items()}
        xs = np.log([20.0, 50.0, 100.0])
        ys = np.log([pooled[s] for s in ("20", "50", "100")])
        slope = float(np.polyfit(xs, ys, 1)[0])
        tgt = GATE_TARGETS[track]
        diffs = {s: abs(pooled[s] - tgt["tv"][s]) for s in pooled}
        ok = (max(diffs.values()) <= 1e-9
              and abs(slope - tgt["slope"]) <= 1e-6)
        ok_all = ok_all and ok
        res[track] = {"spot_rows": spots[track], "pooled_tv": pooled,
                      "slope": slope, "target": tgt, "abs_diff": diffs,
                      "slope_diff": abs(slope - tgt["slope"]),
                      "pass": bool(ok)}
    res["pass"] = bool(ok_all)
    return res


def gate_b_e2e(data_dir: str, track="llama", row=1) -> dict:
    """End-to-end LRO path on a deterministic-outcome question: near-zero
    TV analytic limit. Row 1 (llama) has nested20_tv_vs_e7 == 0.0 in the
    source dataset's manifest. Verify determinism first, then require pooled LRO TV < 0.01
    for every model/arm at S=30, stride 1, replicate 0."""
    q = prepare_question(rec_path(data_dir, track, row), track)
    frac_det = float((q["gt"].max(axis=1) >= 0.99).mean())
    out = run_replicate(q, S=30, stride=1, rep=0)
    tvs = {ma: out["models"][ma]["tv"]["overall"] for ma in out["models"]}
    ok = frac_det >= 0.99 and all(v < 0.01 for v in tvs.values())
    return {"track": track, "row": row, "frac_deterministic": frac_det,
            "tv_overall": tvs, "n_jumps@0.15": len(q["jumps"]["0.15"]),
            "pass": bool(ok)}


def run_gate(args):
    t0 = time.time()
    ga = gate_a_repro(args.data_dir)
    for track in TRACKS:
        g = ga[track]
        print(f"[{time.time()-t0:6.1f}s] gate A {track}: S=20 TV "
              f"{g['pooled_tv']['20']:.10f} (target "
              f"{g['target']['tv']['20']:.10f}), slope {g['slope']:.10f} "
              f"-> {'PASS' if g['pass'] else 'FAIL'}")
    gb = gate_b_e2e(args.data_dir)
    print(f"[{time.time()-t0:6.1f}s] gate B (llama row 1, deterministic "
          f"{gb['frac_deterministic']:.3f}): " +
          ", ".join(f"{k}={v:.5f}" for k, v in gb["tv_overall"].items()) +
          f" -> {'PASS' if gb['pass'] else 'FAIL'}")
    out = {"gate_a_repro_56": ga, "gate_b_deterministic_e2e": gb,
           "all_pass": bool(ga["pass"] and gb["pass"]),
           "runtime_s": round(time.time() - t0, 1)}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scale_gates.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"GATES {'PASS' if out['all_pass'] else 'FAIL'} -> "
          f"{args.out}/scale_gates.json")
    return 0 if out["all_pass"] else 1


# ---------------------------------------------------------------------------
# Fan.
# ---------------------------------------------------------------------------

_QUESTIONS: dict = {}


def _fan_task(task):
    track, row, S, stride, rep = task
    q = _QUESTIONS[(track, row)]
    try:
        return run_replicate(q, S, stride, rep)
    except Exception as e:
        return {"row": row, "track": track, "S": S, "stride": stride,
                "rep": rep, "error": f"{type(e).__name__}: {e}"}


def build_tasks(questions: dict) -> list:
    tasks = []
    for (track, row), q in questions.items():
        for S in S_GRID:
            n_rep = q["n_total"] // S
            for stride in STRIDES[track]:
                for rep in range(n_rep):
                    tasks.append((track, row, S, stride, rep))
    return tasks


def question_meta(q: dict, floors: dict) -> dict:
    return {"K": q["K"], "categories": q["categories"],
            "n_total": q["n_total"], "idxs": q["idxs"],
            "gt": np.round(q["gt"], 6).tolist(), "jumps": q["jumps"],
            "prox": {k: v.tolist() for k, v in q["prox"].items()},
            "exp_tok_per_draw": q["exp_tok_per_draw"].tolist(),
            "draw_diag": q["diag"], "question_text": q["question_text"],
            "answer_letter": q["answer_letter"],
            "answered_rate": q["answered_rate"], "noise_floors": floors}


def run_fan(args):
    t0 = time.time()
    global _QUESTIONS
    tracks = args.tracks.split(",")
    rows = ([int(r) for r in args.rows_list.split(",")] if args.rows_list
            else ROWS)
    for track in tracks:
        for row in rows:
            _QUESTIONS[(track, row)] = prepare_question(
                rec_path(args.data_dir, track, row), track)
        print(f"[{time.time()-t0:6.1f}s] prepared {len(rows)} {track} "
              f"questions", flush=True)
    tasks = build_tasks(_QUESTIONS)
    if args.limit:
        rng = np.random.default_rng(58)
        sel = rng.choice(len(tasks), size=min(args.limit, len(tasks)),
                        replace=False)
        tasks = [tasks[i] for i in sel]
    print(f"[{time.time()-t0:6.1f}s] {len(tasks)} replicate tasks, "
          f"{args.workers} workers", flush=True)
    def report_progress(**kw):
        pass
    os.makedirs(args.out, exist_ok=True)
    # questions meta (with noise floors) per track
    for track in tracks:
        qmeta = {str(row): question_meta(q, noise_floors(q["gt"]))
                 for (tr, row), q in _QUESTIONS.items() if tr == track}
        with open(os.path.join(args.out, f"questions_{track}.json"),
                  "w") as f:
            json.dump(qmeta, f)
    print(f"[{time.time()-t0:6.1f}s] wrote questions meta (incl. noise "
          f"floors)", flush=True)
    shard_tag = f".{args.shard_tag}" if args.shard_tag else ""
    out_path = os.path.join(args.out, f"fan{shard_tag}.jsonl")
    n_err = 0
    with Pool(args.workers) as pool, open(out_path, "w") as f:
        for i, res in enumerate(pool.imap_unordered(_fan_task, tasks,
                                                    chunksize=4)):
            f.write(json.dumps(res) + "\n")
            if "error" in res:
                n_err += 1
                print(f"ERROR {res}", flush=True)
            if (i + 1) % 500 == 0 or i + 1 == len(tasks):
                f.flush()
                report_progress(step=i + 1, total_steps=len(tasks),
                                phase="fan")
                print(f"[{time.time()-t0:8.1f}s] {i+1}/{len(tasks)} done "
                      f"({n_err} errors)", flush=True)
    meta = {"s_grid": S_GRID, "strides": STRIDES, "models": MODELS,
            "arms": list(ARMS), "fixed_params": FIXED_PARAMS,
            "jump_radius": JUMP_RADIUS, "jump_thresh": JUMP_THRESH,
            "seed_base": SEED_BASE, "tracks": tracks, "rows": rows,
            "n_tasks": len(tasks), "n_errors": n_err,
            "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, f"fan_meta{shard_tag}.json"),
              "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{time.time()-t0:8.1f}s] wrote {out_path} ({len(tasks)} "
          f"records, {n_err} errors)", flush=True)
    return 0 if n_err == 0 else 1


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------

def _ci(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None], float)
    n = len(v)
    if n == 0:
        return {"mean": None, "sd": None, "n": 0, "lo": None, "hi": None}
    mean = float(v.mean())
    if n < 2:
        return {"mean": mean, "sd": None, "n": n, "lo": None, "hi": None}
    from scipy import stats as st
    se = v.std(ddof=1) / np.sqrt(n)
    tcrit = st.t.ppf(0.975, n - 1)
    return {"mean": mean, "sd": float(v.std(ddof=1)), "n": n,
            "lo": float(mean - tcrit * se), "hi": float(mean + tcrit * se)}


THS = ["0.10", "0.15", "0.20"]
REGIONS = (["overall"] + [f"jump@{t}" for t in THS]
           + [f"flat@{t}" for t in THS]
           + [f"jumpg1@{t}" for t in THS])   # deepseek grid-radius masks


def load_fan(fan_dir: str):
    questions = {}
    for track in TRACKS:
        p = os.path.join(fan_dir, f"questions_{track}.json")
        if os.path.exists(p):
            with open(p) as f:
                questions[track] = json.load(f)
    # dedupe by task key; later-modified shard files override earlier ones
    # (a track rerun with extra mask fields replaces the original records)
    by_key = {}
    paths = sorted(glob.glob(os.path.join(fan_dir, "fan*.jsonl")),
                   key=os.path.getmtime)
    for p in paths:
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec.get("track"), rec.get("row"), rec.get("S"),
                       rec.get("stride"), rec.get("rep"))
                by_key[key] = rec
    return questions, list(by_key.values())


def assemble(args):
    t0 = time.time()
    questions, results = load_fan(args.fan_dir)
    print(f"[{time.time()-t0:.1f}s] {len(results)} fan records, "
          f"{sum(len(v) for v in questions.values())} questions")
    n_err = sum(1 for r in results if "error" in r)
    if n_err:
        print(f"WARNING: {n_err} errored records excluded")

    # per-question replicate means:
    # agg[(track,S,stride,ma,metric,region)][row] = mean over reps
    from collections import defaultdict
    vals = defaultdict(lambda: defaultdict(list))
    for rec in results:
        if "error" in rec:
            continue
        key0 = (rec["track"], rec["S"], rec["stride"])
        for ma, ent in rec["models"].items():
            for metric in ("tv", "ll", "tv_full"):
                for region in REGIONS:
                    v = ent[metric].get(region)
                    if v is not None:
                        vals[key0 + (ma, metric, region)][
                            rec["row"]].append(v)
            vals[key0 + (ma, "cov", "overall")][rec["row"]].append(
                ent["cov"]["overall"])

    def perq(track, S, stride, ma, metric, region):
        d = vals.get((track, S, stride, ma, metric, region), {})
        return {row: float(np.mean(v)) for row, v in d.items()}

    model_arms = [f"{m}|{a}" for m in MODELS for a in ARMS]

    # --- cells: CI across questions of per-question replicate means.
    cells = {}
    for track in questions:
        for S in S_GRID:
            for stride in STRIDES[track]:
                cell = {}
                for ma in model_arms:
                    ent = {}
                    for metric, region in (
                            [("tv", r) for r in REGIONS]
                            + [("tv_full", r) for r in REGIONS]
                            + [("ll", "overall"), ("cov", "overall")]):
                        pq = perq(track, S, stride, ma, metric, region)
                        ent[f"{metric}_{region}"] = _ci(pq.values())
                    cell[ma] = ent
                cells[f"{track}|{S}|{stride}"] = cell

    # --- costs per cell (expected generated tokens per question).
    costs = {}
    for track, qs in questions.items():
        sq_stride = STATUS_QUO_STRIDE[track]
        for S in S_GRID:
            for stride in STRIDES[track]:
                toks, drws = [], []
                for row, qm in qs.items():
                    et = np.asarray(qm["exp_tok_per_draw"])
                    obs = np.arange(0, len(et), stride)
                    drws.append(S * len(obs))
                    toks.append(S * float(et[obs].sum()))
                costs[f"{track}|{S}|{stride}"] = {
                    "draws_per_question": float(np.mean(drws)),
                    "tokens_per_question": float(np.mean(toks))}
        sq_tok = costs[f"{track}|30|{sq_stride}"]["tokens_per_question"]
        for S in S_GRID:
            for stride in STRIDES[track]:
                c = costs[f"{track}|{S}|{stride}"]
                c["cost_ratio_vs_sq"] = c["tokens_per_question"] / sq_tok

    # --- pooled noise floors per track/S.
    floors = {}
    for track, qs in questions.items():
        floors[track] = {str(S): {
            "ref_floor": float(np.mean(
                [qm["noise_floors"][str(S)]["ref_floor"]
                 for qm in qs.values()])),
            "raw_floor": float(np.mean(
                [qm["noise_floors"][str(S)]["raw_floor"]
                 for qm in qs.values()]))} for S in S_GRID}

    # --- Q1: paired M5a-M0 (cv) jump-region TV, per question.
    # primary flag (plan): the answered-rate ~0.007 question only (<0.01);
    # the broader <0.1 set is recorded as an annotation.
    flagged = {track: [row for row, qm in qs.items()
                       if (qm.get("answered_rate") or 1.0) < 0.01]
               for track, qs in questions.items()}
    low_answered = {track: {row: qm["answered_rate"]
                            for row, qm in qs.items()
                            if (qm.get("answered_rate") or 1.0) < 0.1}
                    for track, qs in questions.items()}
    q1 = {}
    for track in questions:
        stride = STATUS_QUO_STRIDE[track]
        ent = {}
        regions_q1 = [(th, f"jump@{th}") for th in THS]
        if track == "deepseek":
            regions_q1 += [(f"g1_{th}", f"jumpg1@{th}") for th in THS]
        for S in (15, 30):
            for th, region in regions_q1:
                a = perq(track, S, stride, "M5a_segkernel|cv", "tv",
                         region)
                b = perq(track, S, stride, "M0_raw|cv", "tv", region)
                rows_common = sorted(set(a) & set(b))
                diffs = {r: a[r] - b[r] for r in rows_common}
                ent[f"S{S}|th{th}"] = {
                    "n_questions_with_jumps": len(rows_common),
                    "paired_diff": _ci(diffs.values()),
                    "m5a": _ci([a[r] for r in rows_common]),
                    "m0": _ci([b[r] for r in rows_common]),
                    "paired_diff_excl_flagged": _ci(
                        [d for r, d in diffs.items()
                         if str(r) not in flagged[track]]),
                    "per_question_diff": {str(r): diffs[r]
                                          for r in rows_common}}
        q1[track] = {"stride": stride, "cells": ent}

    # --- Q2 (llama): matched draw-budget spacing pairs.
    # Cells (S1,st1) vs (S2,st2) with S1/st1 == S2/st2 have equal expected
    # draw counts; compare per-question paired (mean over reps).
    q2 = {"pairs": [], "note": "pairs share S/stride (equal draw budget); "
          "positive diff means the FINER spacing is worse"}
    track = "llama"
    if track in questions:
        cells_ll = [(S, st) for S in S_GRID for st in STRIDES[track]]
        from fractions import Fraction
        by_ratio = defaultdict(list)
        for S, st in cells_ll:
            by_ratio[Fraction(S, st)].append((S, st))
        for ratio, group in sorted(by_ratio.items()):
            group = sorted(group, key=lambda c: c[1])
            for i in range(len(group) - 1):
                (S1, st1), (S2, st2) = group[i], group[i + 1]
                pair = {"ratio": float(ratio), "fine": [S1, st1],
                        "coarse": [S2, st2],
                        "tokens_fine": costs[f"{track}|{S1}|{st1}"][
                            "tokens_per_question"],
                        "tokens_coarse": costs[f"{track}|{S2}|{st2}"][
                            "tokens_per_question"], "diff": {}}
                for metric, region, lbl in (
                        ("tv", "overall", "pooled"),
                        ("tv", "jump@0.10", "jump@0.10"),
                        ("tv", "jump@0.15", "jump@0.15"),
                        ("tv", "jump@0.20", "jump@0.20")):
                    a = perq(track, S1, st1, "M5a_segkernel|cv", metric,
                             region)
                    b = perq(track, S2, st2, "M5a_segkernel|cv", metric,
                             region)
                    rows_common = sorted(set(a) & set(b))
                    pair["diff"][lbl] = _ci([a[r] - b[r]
                                             for r in rows_common])
                    pair["diff"][lbl]["n_questions"] = len(rows_common)
                q2["pairs"].append(pair)

    # --- Q3: multiplier per track/stride (M5a|cv vs M0|cv, log-log).
    def pooled_mean(track, S, stride, ma, exclude=()):
        pq = perq(track, S, stride, ma, "tv", "overall")
        vs = [v for r, v in pq.items() if str(r) not in exclude]
        return float(np.mean(vs)) if vs else None

    def multiplier_for(track, exclude=()):
        out = {}
        for stride in STRIDES[track]:
            pairs = []
            for S in S_GRID:
                v0 = pooled_mean(track, S, stride, "M0_raw|cv", exclude)
                v5 = pooled_mean(track, S, stride, "M5a_segkernel|cv",
                                 exclude)
                if v0 is not None and v5 is not None and v0 > 0 and v5 > 0:
                    pairs.append((float(S), v0, v5))
            if len(pairs) < 2:
                continue
            s_arr = np.array([p[0] for p in pairs])
            m0 = np.array([p[1] for p in pairs])
            coef = np.polyfit(np.log(s_arr), np.log(m0), 1)
            per_s = {}
            for i, S in enumerate(s_arr):
                tv5 = pairs[i][2]
                if tv5 < m0.min() or tv5 > m0.max():
                    s_eq = float(np.exp((np.log(tv5) - coef[1]) / coef[0]))
                    extrapolated = ("below_m0_range" if tv5 < m0.min()
                                    else "above_m0_range")
                else:
                    order = np.argsort(m0)
                    s_eq = float(np.exp(np.interp(
                        np.log(tv5), np.log(m0[order]),
                        np.log(s_arr[order]))))
                    extrapolated = None
                per_s[str(int(S))] = {"tv_m5a": tv5, "tv_m0": float(m0[i]),
                                      "s_equiv_raw": s_eq,
                                      "multiplier": s_eq / S,
                                      "extrapolated": extrapolated}
            out[str(stride)] = {"m0_loglog_coef": coef.tolist(),
                                "per_s": per_s}
        return out

    multiplier = {track: multiplier_for(track) for track in questions}
    multiplier_excl = {
        track: multiplier_for(track, exclude=set(flagged[track]))
        for track in questions if flagged[track]}

    # --- power-law fit of TV vs S per track at status-quo stride.
    def power_law_for(track, exclude=()):
        stride = STATUS_QUO_STRIDE[track]
        ent = {"stride": stride}
        for ma in ("M0_raw|cv", "M5a_segkernel|cv"):
            pts = [(S, pooled_mean(track, S, stride, ma, exclude))
                   for S in S_GRID]
            pts = [(S, v) for S, v in pts if v is not None and v > 0]
            if len(pts) < 2:
                continue
            coef = np.polyfit(np.log([p[0] for p in pts]),
                              np.log([p[1] for p in pts]), 1)
            ent[ma] = {"slope": float(coef[0]),
                       "intercept": float(coef[1]), "n_points": len(pts)}
        return ent

    power_law = {track: power_law_for(track) for track in questions}
    power_law_excl = {
        track: power_law_for(track, exclude=set(flagged[track]))
        for track in questions if flagged[track]}

    # --- frontier per track (M5a|cv).
    frontier = {}
    for track in questions:
        pts = []
        for S in S_GRID:
            for stride in STRIDES[track]:
                key = f"{track}|{S}|{stride}"
                ent = cells[key]["M5a_segkernel|cv"]
                pts.append({"S": S, "stride": stride,
                            "tokens": costs[key]["tokens_per_question"],
                            "cost_ratio": costs[key]["cost_ratio_vs_sq"],
                            "tv": ent["tv_overall"]["mean"],
                            "tv_lo": ent["tv_overall"]["lo"],
                            "tv_hi": ent["tv_overall"]["hi"],
                            "tv_jump": ent["tv_jump@0.15"]["mean"],
                            "tv_flat": ent["tv_flat@0.15"]["mean"]})
        pts = [p for p in pts if p["tv"] is not None]
        pts.sort(key=lambda p: p["tokens"])
        best = np.inf
        for p in pts:
            p["pareto"] = p["tv"] < best
            if p["tv"] < best:
                best = p["tv"]
        frontier[track] = pts

    # --- cheapest cell within eps of each track's status quo.
    cheapest = {}
    for track in questions:
        sq_stride = STATUS_QUO_STRIDE[track]
        sq = cells[f"{track}|30|{sq_stride}"]["M5a_segkernel|cv"]
        sq_tv = sq["tv_overall"]["mean"]
        sq_tok = costs[f"{track}|30|{sq_stride}"]["tokens_per_question"]
        if sq_tv is None:
            cheapest[track] = {"status_quo": None, "table": []}
            continue
        rowsE = []
        for eps in [round(e, 3) for e in np.arange(0.0, 0.081, 0.005)]:
            ok = [p for p in frontier[track] if p["tv"] <= sq_tv + eps]
            okb = min(ok, key=lambda p: p["tokens"]) if ok else None
            rowsE.append({"eps": eps, "threshold_tv": sq_tv + eps,
                          "cell": ({"S": okb["S"], "stride": okb["stride"],
                                    "tokens": okb["tokens"], "tv": okb["tv"],
                                    "cost_ratio_vs_sq":
                                        okb["tokens"] / sq_tok}
                                   if okb else None)})
        cheapest[track] = {"status_quo": {"S": 30, "stride": sq_stride,
                                          "tv": sq_tv, "tokens": sq_tok},
                           "table": rowsE}

    out = {"meta": {"s_grid": S_GRID, "strides": STRIDES,
                    "models": MODELS, "arms": list(ARMS),
                    "jump_thresh": JUMP_THRESH,
                    "jump_radius": JUMP_RADIUS,
                    "status_quo_stride": STATUS_QUO_STRIDE,
                    "flagged_low_answered": flagged,
                    "low_answered_lt_0.1": low_answered,
                    "n_fan_records": len(results), "n_errors": n_err},
           "cells": cells, "costs": costs, "noise_floors": floors,
           "q1_fork_verdict": q1, "q2_matched_budget": q2,
           "q3_multiplier": multiplier,
           "q3_multiplier_excl_flagged": multiplier_excl,
           "power_law": power_law,
           "power_law_excl_flagged": power_law_excl,
           "frontier": frontier, "cheapest_within_eps": cheapest}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scale_analysis.json"), "w") as f:
        json.dump(out, f)
    print(f"[{time.time()-t0:.1f}s] wrote scale_analysis.json "
          f"({len(cells)} cells) to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Dashboard combos.
# ---------------------------------------------------------------------------

def _dash_task(task):
    track, row, S, stride = task
    q = _QUESTIONS[(track, row)]
    idxs = np.asarray(q["idxs"], float)
    obs_idx = np.arange(0, len(idxs), stride)
    obs_tok = idxs[obs_idx]
    block = q["draws"][obs_idx][:, :S]
    counts = counts_from_draws(block, q["K"], 0, S)
    params, _ = cv_select("M5a_segkernel", block, obs_tok, S, q["K"],
                          n_folds=min(5, S))
    m = MODEL_REGISTRY["M5a_segkernel"]()
    m.fit(obs_tok, counts, S, params)
    pred = m.predict(idxs)
    raw = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
    et = q["exp_tok_per_draw"]
    sq_stride = STATUS_QUO_STRIDE[track]
    sq_obs = np.arange(0, len(et), sq_stride)
    cost_ratio = ((S * float(et[obs_idx].sum()))
                  / (30.0 * float(et[sq_obs].sum())))
    return (track, f"{row}|{S}|{stride}", {
        "obs_tok": obs_tok.astype(int).tolist(),
        "raw": np.round(raw, 3).tolist(),
        "pred": np.round(pred, 3).tolist(),
        "params": params, "cost_ratio": round(cost_ratio, 4)})


def run_dashboard(args):
    t0 = time.time()
    global _QUESTIONS
    tracks = args.tracks.split(",")
    rows = ([int(r) for r in args.rows_list.split(",")] if args.rows_list
            else ROWS)
    if args.sq_stride:
        for track in tracks:
            STATUS_QUO_STRIDE[track] = args.sq_stride
    for track in tracks:
        for row in rows:
            _QUESTIONS[(track, row)] = prepare_question(
                rec_path(args.data_dir, track, row), track)
        print(f"[{time.time()-t0:6.1f}s] prepared {track}", flush=True)
    tasks = [(track, row, S, stride)
             for (track, row) in _QUESTIONS
             for S in S_GRID for stride in STRIDES[track]]
    print(f"[{time.time()-t0:6.1f}s] {len(tasks)} dashboard combos",
          flush=True)
    def report_progress(**kw):
        pass
    combos = {t: {} for t in tracks}
    with Pool(args.workers) as pool:
        for i, (track, key, val) in enumerate(
                pool.imap_unordered(_dash_task, tasks, chunksize=4)):
            combos[track][key] = val
            if (i + 1) % 200 == 0 or i + 1 == len(tasks):
                report_progress(step=i + 1, total_steps=len(tasks),
                                phase="dashboard")
                print(f"[{time.time()-t0:7.1f}s] {i+1}/{len(tasks)}",
                      flush=True)
    os.makedirs(args.out, exist_ok=True)
    for track in tracks:
        payload = {
            "s_grid": S_GRID, "stride_grid": STRIDES[track],
            "spacing_unit": SPACING_UNIT[track],
            "grid_tokens_per_stride": 1 if track == "llama" else None,
            "status_quo_stride": STATUS_QUO_STRIDE[track],
            "questions": {str(row): {
                "idxs": q["idxs"], "categories": q["categories"],
                "ref": np.round(q["gt"], 3).tolist(),
                "ref_label": f"S={q['n_total']} reference",
                "n_total": q["n_total"], "jumps": q["jumps"]["0.15"],
                "question_text": q["question_text"],
                "answered_rate": q["answered_rate"]}
                for (tr, row), q in _QUESTIONS.items() if tr == track},
            "combos": combos[track]}
        p = os.path.join(args.out, f"dashboard_{track}.json")
        with open(p, "w") as f:
            json.dump(payload, f)
        print(f"[{time.time()-t0:7.1f}s] wrote {p} "
              f"({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "fan", "assemble",
                                     "dashboard"])
    ap.add_argument("--data-dir", help=".../s200 (has llama/, deepseek/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fan-dir")
    ap.add_argument("--tracks", default="llama,deepseek")
    ap.add_argument("--rows-list", default="")
    ap.add_argument("--shard-tag", default="")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sq-stride", type=int, default=0,
                    help="dashboard mode only: override the cost-chip "
                         "baseline stride (e.g. 1 = every grid position)")
    args = ap.parse_args()
    if args.mode == "gate":
        sys.exit(run_gate(args))
    if args.mode == "fan":
        sys.exit(run_fan(args))
    if args.mode == "dashboard":
        sys.exit(run_dashboard(args))
    sys.exit(assemble(args))


if __name__ == "__main__":
    main()
