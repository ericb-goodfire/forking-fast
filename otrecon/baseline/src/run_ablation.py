"""M5a component ablation at scale.

M5a (SegmentKernelPool) stacks four design choices: PELT segmentation,
the fixed multinomial detection cost, within-segment kernel pooling
truncated at changepoints, and CV-tuned hyperparameters. This driver
decomposes the stack on the 50-question S=200 dataset (both tracks) by
ablating one component at a time and measuring how much of the raw-counts
-> full-M5a improvement each configuration retains, with paired CIs over
the 50 questions (the split unit; references never enter tuning).

All reconstruction machinery is vendored unchanged from the scale driver (otrecon +
src/run_scale.py). Every arm is an existing otrecon configuration; the
only new classes are grid-restricted subclasses of SegmentKernelPool that
pin the detection variant (the 'l2' and 'linear' variants already exist
in ``segment_positions``).

Ablation arms (key -> what it removes from full M5a):
  M5a_segkernel|cv     full M5a (anchor; CV over variant {mult,linear} x
                       pen x h — exactly the scale driver's M5a|cv arm)
  M1_kernel|cv         - segmentation (Gaussian kernel pooling only)
  M4_segment|cv        - kernel pooling (segment + flat Dirichlet pool)
  M5a_l2|cv            - multinomial/trend detection cost (PELT CostL2 on
                       raw counts, the legacy M4 behavior; CV pen x h)
  M5a_linear|cv        detection cost = logit-linear trend only (CV pen x h)
  M5a_segkernel|fixed  - CV tuning ({mult, pen 64, h 32}, the synthetic-gate
                       picks — exactly the scale driver's fixed arm)
  M0_raw|cv            raw floor (counts + interpolation)

Cells: S in {5, 15, 30, 100} x stride in {1, 4} grid units, both tracks
(llama grid unit = token, deepseek = sentence); floor(200/S) disjoint
replicates. Reference is always leave-replicate-out (LRO), matching the
scale driver.

Modes:
  gate      -- (a) the scale driver's gate A (exact known-answer
               reproduction) and
               gate B (deterministic-question analytic limit), vendored;
               (b) reproduction gate: this driver's replicate runner on
               the 10 spot rows per track, 8 cells, must reproduce the scale driver's
               fan_v3 per-replicate values for the four shared arms to
               <= 1e-9 (same seeds, same replicates). A mismatch stops
               the new arms.
  fan       -- 7-arm replicate fan (one managed CPU job).
  assemble  -- full-strength repro check vs fan_v3 (all 50 rows) + paired
               deltas vs full M5a + improvement-retained fractions +
               simplest-equivalent-configuration table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from otrecon.cv import cv_select
from otrecon.data import counts_from_draws
from otrecon.metrics import tv_to_gt
from otrecon.models import PEN_GRID, MODEL_REGISTRY, SegmentKernelPool
from src.run_scale import (BAND_LEVEL, FIXED_PARAMS, N_TOTAL, ROWS,
                           STATUS_QUO_STRIDE, TRACKS,
                           _heldout_ll_per_position, _region_summary,
                           gate_a_repro, gate_b_e2e, noise_floors,
                           prepare_question, question_meta, rec_path,
                           spot_rows_by_track)

# ---------------------------------------------------------------------------
# Grid-restricted M5a variants (configuration wrappers, no new model code).
# ---------------------------------------------------------------------------

_H_GRID = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


class SegmentKernelPoolL2(SegmentKernelPool):
    """M5a with the detection cost pinned to 'l2' (PELT CostL2 on raw
    counts — the legacy silent-fallback behavior). CV tunes pen x h."""
    name = "M5a_l2"

    @staticmethod
    def grid():
        return [{"variant": "l2", "pen": p, "h": h}
                for p in PEN_GRID for h in _H_GRID]


class SegmentKernelPoolLinear(SegmentKernelPool):
    """M5a with the detection cost pinned to 'linear' (logit-linear trend
    cost). CV tunes pen x h."""
    name = "M5a_linear"

    @staticmethod
    def grid():
        return [{"variant": "linear", "pen": p, "h": h}
                for p in PEN_GRID for h in _H_GRID]


MODEL_REGISTRY["M5a_l2"] = SegmentKernelPoolL2
MODEL_REGISTRY["M5a_linear"] = SegmentKernelPoolLinear

# ---------------------------------------------------------------------------
# Experiment constants.
# ---------------------------------------------------------------------------

ABL_S_GRID = [5, 15, 30, 100]
ABL_STRIDES = {"llama": [1, 4], "deepseek": [1, 4]}

# (record key, registry name, tuning arm). Keys for the four shared arms
# match the reference fan_v3 keys exactly, which makes the reproduction gate a
# straight dict comparison.
ARMS = [
    ("M0_raw|cv", "M0_raw", "cv"),
    ("M1_kernel|cv", "M1_kernel", "cv"),
    ("M4_segment|cv", "M4_segment", "cv"),
    ("M5a_segkernel|cv", "M5a_segkernel", "cv"),
    ("M5a_segkernel|fixed", "M5a_segkernel", "fixed"),
    ("M5a_l2|cv", "M5a_l2", "cv"),
    ("M5a_linear|cv", "M5a_linear", "cv"),
]
SHARED_ARMS = ["M0_raw|cv", "M4_segment|cv", "M5a_segkernel|cv",
               "M5a_segkernel|fixed"]
FULL_ARM = "M5a_segkernel|cv"
FLOOR_ARM = "M0_raw|cv"

# Pre-registered simplicity order (fewest components first) for the
# "simplest configuration indistinguishable from full M5a" table.
SIMPLICITY_ORDER = ["M0_raw|cv", "M1_kernel|cv", "M4_segment|cv",
                    "M5a_segkernel|fixed", "M5a_l2|cv", "M5a_linear|cv"]

# Human-readable component labels for the report. Note (critic f2): full
# M5a's own CV grid spans variant {mult, linear}, so the linear arm is a
# SUB-GRID of the anchor, not an independent component removal, and the l2
# arm replaces whichever detection cost CV would have picked with the
# legacy L2 cost. Labels say "detection cost ->" rather than "- cost".
ARM_LABELS = {
    "M5a_segkernel|cv": "full M5a",
    "M1_kernel|cv": "- segmentation (kernel only, M1)",
    "M4_segment|cv": "- kernel pooling (segment pool, M4)",
    "M5a_l2|cv": "detection cost -> L2 (legacy)",
    "M5a_linear|cv": "detection cost -> linear only (anchor sub-grid)",
    "M5a_segkernel|fixed": "- CV tuning (fixed mult/64/32)",
    "M0_raw|cv": "raw counts (M0)",
}

THS = ["0.10", "0.15", "0.20"]
REGIONS = (["overall"] + [f"jump@{t}" for t in THS]
           + [f"flat@{t}" for t in THS]
           + [f"jumpg1@{t}" for t in THS])
# Per-question improvement-retained fractions require a strictly positive
# per-question M0-M5a improvement of at least this floor: with a negative
# denominator the ratio is sign-flipped, and near zero it is unstable
# (critic f1: negative denominators are common on the deepseek track).
# In cells where the mean M0-M5a improvement itself is not positive the
# retained-fraction metric is undefined and only paired deltas apply.
# Verdicts never ride on fractions; they ride on paired deltas.
FRAC_DENOM_FLOOR = 0.005
REPRO_TOL = 1e-9


def run_replicate_arms(q: dict, S: int, stride: int, rep: int,
                       arms=ARMS) -> dict:
    """Mirror of run_scale.run_replicate over an explicit arm list.

    Identical observation/counts/LRO-reference construction; for the four
    shared arms the output must be float-identical to the reference fan_v3.
    """
    idxs = np.asarray(q["idxs"], float)
    K = q["K"]
    obs_idx = np.arange(0, len(idxs), stride)
    obs_tok = idxs[obs_idx]
    lo, hi = rep * S, (rep + 1) * S
    block_obs = q["draws"][obs_idx][:, lo:hi]
    counts = counts_from_draws(block_obs, K, 0, S)
    held = np.concatenate([q["draws"][:, :lo], q["draws"][:, hi:]], axis=1)
    cnt = counts_from_draws(held, K, 0, held.shape[1])
    gt_lro = cnt / np.maximum(cnt.sum(axis=1, keepdims=True), 1e-12)
    prox = q["prox"]
    out = {"row": q["row_id"], "track": q["track"], "S": S,
           "stride": stride, "rep": rep, "n_obs": len(obs_idx),
           "models": {}}
    for key, name, arm in arms:
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
        out["models"][key] = rec
    return out


# ---------------------------------------------------------------------------
# Reproduction gate vs the reference fan_v3.
# ---------------------------------------------------------------------------

def load_ref_records(ref_fan: str, tracks, rows, s_grid, strides) -> dict:
    """Stream the reference fan_v3 jsonl, keep records in our cell/row subset."""
    want_rows = set(rows)
    want_s = set(s_grid)
    refs = {}
    with open(ref_fan) as f:
        for line in f:
            rec = json.loads(line)
            if (rec["track"] in tracks and rec["row"] in want_rows
                    and rec["S"] in want_s
                    and rec["stride"] in strides[rec["track"]]):
                key = (rec["track"], rec["row"], rec["S"], rec["stride"],
                       rec["rep"])
                refs[key] = {k: rec["models"][k] for k in SHARED_ARMS}
    return refs


def _flatten_numeric(d, prefix=""):
    out = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_numeric(v, p))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[p] = float(v)
    return out


def compare_to_ref(mine: dict, ref: dict) -> tuple[float, list]:
    """Max abs diff over all numeric leaves of the arms in ``mine``;
    param dicts must match exactly. Returns (max_diff, mismatches)."""
    max_diff = 0.0
    bad = []
    for arm in mine:
        a, b = mine[arm], ref[arm]
        if a["params"] != b["params"]:
            bad.append((arm, "params", a["params"], b["params"]))
        fa, fb = _flatten_numeric(a), _flatten_numeric(b)
        if set(fa) != set(fb):
            bad.append((arm, "keys", sorted(set(fa) ^ set(fb))[:5], None))
            continue
        for k in fa:
            d = abs(fa[k] - fb[k])
            max_diff = max(max_diff, d)
            if d > REPRO_TOL:
                bad.append((arm, k, fa[k], fb[k]))
    return max_diff, bad


def run_gate(args):
    t0 = time.time()
    # (a) the known-answer gates, vendored unchanged.
    ga = gate_a_repro(args.data_dir)
    print(f"[{time.time()-t0:6.1f}s] gate A (exact repro): "
          f"{'PASS' if ga['pass'] else 'FAIL'}", flush=True)
    gb = gate_b_e2e(args.data_dir)
    print(f"[{time.time()-t0:6.1f}s] gate B (deterministic e2e): "
          f"{'PASS' if gb['pass'] else 'FAIL'}", flush=True)

    # (b) reproduction gate: spot rows, 8 cells, this driver's runner.
    spots = spot_rows_by_track()
    refs = load_ref_records(args.ref_fan, TRACKS,
                            set(spots["llama"]) | set(spots["deepseek"]),
                            ABL_S_GRID, ABL_STRIDES)
    print(f"[{time.time()-t0:6.1f}s] loaded {len(refs)} reference records "
          f"from fan_v3", flush=True)
    global _QUESTIONS
    tasks = []
    for track in TRACKS:
        for row in spots[track]:
            _QUESTIONS[(track, row)] = prepare_question(
                rec_path(args.data_dir, track, row), track)
            for S in ABL_S_GRID:
                for stride in ABL_STRIDES[track]:
                    for rep in range(N_TOTAL // S):
                        tasks.append((track, row, S, stride, rep))
    print(f"[{time.time()-t0:6.1f}s] {len(tasks)} spot replicate tasks, "
          f"{args.workers} workers", flush=True)
    max_diff = 0.0
    mism = []
    n_missing = 0
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                _gate_task, tasks, chunksize=4)):
            key = (res["track"], res["row"], res["S"], res["stride"],
                   res["rep"])
            ref = refs.get(key)
            if ref is None:
                n_missing += 1
                continue
            d, bad = compare_to_ref(
                {k: res["models"][k] for k in SHARED_ARMS}, ref)
            max_diff = max(max_diff, d)
            mism.extend([(key,) + tuple(map(str, b)) for b in bad[:3]])
            if (i + 1) % 500 == 0 or i + 1 == len(tasks):
                print(f"[{time.time()-t0:7.1f}s] {i+1}/{len(tasks)} "
                      f"compared, max_diff {max_diff:.3e}, "
                      f"{len(mism)} mismatches", flush=True)
    ok = (not mism) and n_missing == 0 and max_diff <= REPRO_TOL
    out = {"gate_a_repro_56": ga, "gate_b_deterministic_e2e": gb,
           "repro_gate": {"n_tasks": len(tasks), "n_missing_ref": n_missing,
                          "max_abs_diff": max_diff,
                          "n_mismatches": len(mism),
                          "mismatch_examples": mism[:20],
                          "shared_arms": SHARED_ARMS,
                          "spot_rows": spots, "tol": REPRO_TOL,
                          "pass": bool(ok)},
           "all_pass": bool(ga["pass"] and gb["pass"] and ok),
           "runtime_s": round(time.time() - t0, 1)}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation_gates.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"GATES {'PASS' if out['all_pass'] else 'FAIL'} "
          f"(repro max_diff {max_diff:.3e}) -> "
          f"{args.out}/ablation_gates.json", flush=True)
    return 0 if out["all_pass"] else 1


# ---------------------------------------------------------------------------
# Fan.
# ---------------------------------------------------------------------------

_QUESTIONS: dict = {}


def _gate_task(task):
    track, row, S, stride, rep = task
    return run_replicate_arms(_QUESTIONS[(track, row)], S, stride, rep)


def _fan_task(task):
    track, row, S, stride, rep = task
    q = _QUESTIONS[(track, row)]
    try:
        return run_replicate_arms(q, S, stride, rep)
    except Exception as e:
        return {"row": row, "track": track, "S": S, "stride": stride,
                "rep": rep, "error": f"{type(e).__name__}: {e}"}


def build_tasks(questions: dict) -> list:
    tasks = []
    for (track, row), q in questions.items():
        for S in ABL_S_GRID:
            for stride in ABL_STRIDES[track]:
                for rep in range(q["n_total"] // S):
                    tasks.append((track, row, S, stride, rep))
    return tasks


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
        rng = np.random.default_rng(59)
        sel = rng.choice(len(tasks), size=min(args.limit, len(tasks)),
                         replace=False)
        tasks = [tasks[i] for i in sel]
    print(f"[{time.time()-t0:6.1f}s] {len(tasks)} replicate tasks x "
          f"{len(ARMS)} arms, {args.workers} workers", flush=True)
    def report_progress(**kw):
        pass
    os.makedirs(args.out, exist_ok=True)
    for track in tracks:
        qmeta = {str(row): question_meta(q, noise_floors(q["gt"],
                                                         s_grid=ABL_S_GRID))
                 for (tr, row), q in _QUESTIONS.items() if tr == track}
        with open(os.path.join(args.out, f"questions_{track}.json"),
                  "w") as f:
            json.dump(qmeta, f)
    print(f"[{time.time()-t0:6.1f}s] wrote questions meta", flush=True)
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
    meta = {"s_grid": ABL_S_GRID, "strides": ABL_STRIDES,
            "arms": [a[0] for a in ARMS], "fixed_params": FIXED_PARAMS,
            "tracks": tracks, "rows": rows, "n_tasks": len(tasks),
            "n_errors": n_err, "runtime_s": round(time.time() - t0, 1)}
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


def load_fan(fan_dir: str):
    questions = {}
    for track in TRACKS:
        p = os.path.join(fan_dir, f"questions_{track}.json")
        if os.path.exists(p):
            with open(p) as f:
                questions[track] = json.load(f)
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
          f"{sum(len(v) for v in questions.values())} questions",
          flush=True)
    n_err = sum(1 for r in results if "error" in r)
    if n_err:
        print(f"WARNING: {n_err} errored records excluded", flush=True)

    # --- full-strength repro check vs fan_v3 (all rows, 8 cells).
    repro = None
    if args.ref_fan:
        refs = load_ref_records(args.ref_fan, TRACKS, set(ROWS),
                                ABL_S_GRID, ABL_STRIDES)
        max_diff, mism, n_cmp, n_missing = 0.0, [], 0, 0
        for rec in results:
            if "error" in rec:
                continue
            key = (rec["track"], rec["row"], rec["S"], rec["stride"],
                   rec["rep"])
            ref = refs.get(key)
            if ref is None:
                n_missing += 1
                continue
            d, bad = compare_to_ref(
                {k: rec["models"][k] for k in SHARED_ARMS}, ref)
            max_diff = max(max_diff, d)
            mism.extend(bad[:2])
            n_cmp += 1
        repro = {"n_compared": n_cmp, "n_missing_ref": n_missing,
                 "max_abs_diff": max_diff, "n_mismatches": len(mism),
                 "tol": REPRO_TOL,
                 "pass": bool(not mism and max_diff <= REPRO_TOL)}
        print(f"[{time.time()-t0:.1f}s] full repro check: {n_cmp} records, "
              f"max_diff {max_diff:.3e}, "
              f"{'PASS' if repro['pass'] else 'FAIL'}", flush=True)

    arms = [a[0] for a in ARMS]

    # --- per-question replicate means + identifiability diagnostics
    # (critic f2): per-arm segment-count distributions and the rate at
    # which each segmented arm's fit is bit-identical to the kernel-only
    # M1 fit (single-segment collapse) or to full M5a, per cell.
    vals = defaultdict(lambda: defaultdict(list))
    params_seen = defaultdict(lambda: defaultdict(int))
    seg_hist = defaultdict(lambda: defaultdict(int))
    ident = defaultdict(lambda: {"n": 0, "eq_m1": 0, "eq_full": 0})
    for rec in results:
        if "error" in rec:
            continue
        key0 = (rec["track"], rec["S"], rec["stride"])
        m1_tv = rec["models"]["M1_kernel|cv"]["tv"]["overall"]
        full_tv = rec["models"][FULL_ARM]["tv"]["overall"]
        for ma, ent in rec["models"].items():
            for metric in ("tv", "ll"):
                for region in REGIONS:
                    v = ent[metric].get(region)
                    if v is not None:
                        vals[key0 + (ma, metric, region)][
                            rec["row"]].append(v)
            if ma in ("M5a_segkernel|cv", "M5a_l2|cv", "M5a_linear|cv"):
                params_seen[key0 + (ma,)][
                    ent["params"].get("variant", "?")] += 1
            if "n_segments" in ent:
                seg_hist[key0 + (ma,)][str(ent["n_segments"])] += 1
            d = ident[key0 + (ma,)]
            d["n"] += 1
            d["eq_m1"] += int(ent["tv"]["overall"] == m1_tv)
            d["eq_full"] += int(ent["tv"]["overall"] == full_tv)

    def perq(track, S, stride, ma, metric, region):
        d = vals.get((track, S, stride, ma, metric, region), {})
        return {row: float(np.mean(v)) for row, v in d.items()}

    rng = np.random.default_rng(59)

    def boot_ratio(num_pq: dict, den_pq: dict, n_boot=2000) -> dict:
        """Ratio of question-means: mean_q(num) / mean_q(den), bootstrap
        CI over questions. Sign-guarded (critic f1): undefined unless the
        cell's mean denominator (the M0-full improvement) clears the
        positive floor; bootstrap resamples whose denominator does not
        stay positive are dropped and counted."""
        rows = sorted(set(num_pq) & set(den_pq))
        if not rows:
            return {"mean": None, "lo": None, "hi": None, "n": 0,
                    "defined": False, "reason": "no_questions"}
        num = np.array([num_pq[r] for r in rows])
        den = np.array([den_pq[r] for r in rows])
        if den.mean() < FRAC_DENOM_FLOOR:
            return {"mean": None, "lo": None, "hi": None, "n": len(rows),
                    "defined": False,
                    "reason": "mean_denominator_not_positive",
                    "den_mean": float(den.mean())}
        point = float(num.mean() / den.mean())
        idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
        bs_den = den[idx].mean(axis=1)
        okm = bs_den >= FRAC_DENOM_FLOOR
        bs = num[idx].mean(axis=1)[okm] / bs_den[okm]
        return {"mean": point, "lo": float(np.percentile(bs, 2.5)),
                "hi": float(np.percentile(bs, 97.5)), "n": len(rows),
                "defined": True, "den_mean": float(den.mean()),
                "n_boot_valid": int(okm.sum())}

    # --- the ablation table: per (track, cell, region, arm).
    cells = {}
    for track in questions:
        for S in ABL_S_GRID:
            for stride in ABL_STRIDES[track]:
                cell = {}
                for region in REGIONS:
                    if region.startswith("jumpg1") and track != "deepseek":
                        continue
                    full = perq(track, S, stride, FULL_ARM, "tv", region)
                    m0 = perq(track, S, stride, FLOOR_ARM, "tv", region)
                    reg = {}
                    for ma in arms:
                        a = perq(track, S, stride, ma, "tv", region)
                        rows_c = sorted(set(a) & set(full) & set(m0))
                        delta = {r: a[r] - full[r] for r in rows_c}
                        # improvement-retained fraction per question:
                        # strictly positive denominator required; flags
                        # broken out by sign (critic f1).
                        fr, n_nonpos, n_small = [], 0, 0
                        for r in rows_c:
                            den = m0[r] - full[r]
                            if den <= 0:
                                n_nonpos += 1
                                continue
                            if den < FRAC_DENOM_FLOOR:
                                n_small += 1
                                continue
                            fr.append((m0[r] - a[r]) / den)
                        reg[ma] = {
                            "tv": _ci([a[r] for r in rows_c]),
                            "delta_vs_full": _ci(delta.values()),
                            "frac_perq": _ci(fr),
                            "frac_perq_n_denom_nonpositive": n_nonpos,
                            "frac_perq_n_denom_small": n_small,
                            "frac_of_means": boot_ratio(
                                {r: m0[r] - a[r] for r in rows_c},
                                {r: m0[r] - full[r] for r in rows_c}),
                        }
                        # held-out log-lik companion (overall region only
                        # has full coverage; keep per-region too)
                        la = perq(track, S, stride, ma, "ll", region)
                        lf = perq(track, S, stride, FULL_ARM, "ll", region)
                        rows_l = sorted(set(la) & set(lf))
                        reg[ma]["ll_delta_vs_full"] = _ci(
                            [la[r] - lf[r] for r in rows_l])
                    cell[region] = reg
                cells[f"{track}|{S}|{stride}"] = cell

    # --- simplest configuration indistinguishable from full M5a.
    # Rule A (primary): first arm in SIMPLICITY_ORDER whose paired
    # delta-vs-full 95% CI includes 0. Rule B (thread convention): first
    # arm with mean paired delta <= 0.02 (the project's TV floor).
    simplest = {}
    for ck, cell in cells.items():
        ent = {}
        for region in ("overall", "jump@0.10", "jump@0.15", "flat@0.10",
                       "flat@0.15"):
            if region not in cell:
                continue
            rule_a = rule_b = None
            for ma in SIMPLICITY_ORDER:
                d = cell[region][ma]["delta_vs_full"]
                if d["mean"] is None or d["lo"] is None:
                    continue
                if rule_a is None and d["lo"] <= 0.0 <= d["hi"]:
                    rule_a = ma
                if rule_b is None and d["mean"] <= 0.02:
                    rule_b = ma
            ent[region] = {"ci_rule": rule_a or FULL_ARM,
                           "floor_rule_0.02": rule_b or FULL_ARM}
        simplest[ck] = ent

    # --- CV variant picks for the variant-selectable arms (diagnostic).
    variant_picks = {}
    for (track, S, stride, ma), cnt in params_seen.items():
        variant_picks.setdefault(f"{track}|{S}|{stride}", {})[ma] = dict(cnt)

    # --- identifiability diagnostics per cell/arm (critic f2).
    identifiability = {}
    for (track, S, stride, ma), d in ident.items():
        ck = f"{track}|{S}|{stride}"
        ent = identifiability.setdefault(ck, {}).setdefault(ma, {})
        ent["n_replicates"] = d["n"]
        ent["identical_tv_to_M1"] = d["eq_m1"] / max(d["n"], 1)
        ent["identical_tv_to_full"] = d["eq_full"] / max(d["n"], 1)
        sh = seg_hist.get((track, S, stride, ma))
        if sh:
            ent["n_segments_hist"] = dict(sh)
            n1 = sh.get("1", 0)
            ent["single_segment_rate"] = n1 / max(d["n"], 1)

    # --- headline cells (status-quo spacing per track, S=30).
    headline = {track: f"{track}|30|{STATUS_QUO_STRIDE[track]}"
                for track in questions
                if STATUS_QUO_STRIDE[track] in ABL_STRIDES[track]}

    out = {"meta": {"s_grid": ABL_S_GRID, "strides": ABL_STRIDES,
                    "arms": arms, "arm_labels": ARM_LABELS,
                    "shared_arms": SHARED_ARMS,
                    "simplicity_order": SIMPLICITY_ORDER,
                    "frac_denom_floor": FRAC_DENOM_FLOOR,
                    "headline_cells": headline,
                    "n_fan_records": len(results), "n_errors": n_err},
           "repro_check": repro, "cells": cells, "simplest": simplest,
           "variant_picks": variant_picks,
           "identifiability": identifiability}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation_analysis.json"), "w") as f:
        json.dump(out, f)
    print(f"[{time.time()-t0:.1f}s] wrote ablation_analysis.json "
          f"({len(cells)} cells) to {args.out}", flush=True)
    return 0 if (repro is None or repro["pass"]) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "fan", "assemble"])
    ap.add_argument("--data-dir", help=".../s200 (has llama/, deepseek/)")
    ap.add_argument("--ref-fan", help="reference fan_v3/fan.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fan-dir")
    ap.add_argument("--tracks", default="llama,deepseek")
    ap.add_argument("--rows-list", default="")
    ap.add_argument("--shard-tag", default="")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "gate":
        sys.exit(run_gate(args))
    if args.mode == "fan":
        sys.exit(run_fan(args))
    sys.exit(assemble(args))


if __name__ == "__main__":
    main()
