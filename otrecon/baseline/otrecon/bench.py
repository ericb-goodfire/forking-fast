"""Benchmark grid: (question, S, stride, model) -> metrics + predictions."""
from __future__ import annotations

import numpy as np

from .data import (collapse_curve, collapse_map, counts_from_draws,
                   load_record, mixture_draws, weighted_o_t)
from .cv import cv_select
from .metrics import (band_coverage, gt_jumps, heldout_loglik,
                      heldout_loglik_per_position, region_masks, summarize,
                      tv_to_gt)
from .models import MODEL_REGISTRY

S_GRID = [10, 30, 100]
STRIDE_GRID = [1, 2, 4, 8, 16]
MODELS = ["M0_raw", "M1_kernel", "M2_discount", "M3_kalman", "M4_segment"]
# 44-model set: fixed-cost M4 + legacy-L2 M4 (as originally run) + hybrids.
MODELS_44 = ["M0_raw", "M1_kernel", "M2_discount", "M3_kalman",
             "M4_segment", "M4l2_segment", "M5a_segkernel", "M5b_segpoly"]
HYBRIDS = ["M5a_segkernel", "M5b_segpoly"]
LL_TOL = 0.02  # nats/draw, pre-specified log-lik tolerance (Q1)
JUMP_THRESH = 0.15
JUMP_THRESH_SENS = [0.10, 0.15, 0.20]
JUMP_RADIUS = 10
BAND_LEVEL = 0.9
NOISE_FLOOR_TV = 0.02


def prepare_question(path: str, n_total: int | None = None,
                     seed_base: int = 43_000_000) -> dict:
    rec = load_record(path)
    cats = rec["categories"]
    o_stored = np.asarray(rec["o_t_full"], float)
    idxs_re, o_recomputed = weighted_o_t(rec)
    assert list(rec["idxs"]) == idxs_re, "o_t_full grid mismatch"
    kept, idx_map = collapse_map(o_recomputed, cats)
    K = len(kept)
    gt = collapse_curve(o_recomputed, idx_map, K)
    gt_stored = collapse_curve(o_stored, idx_map, K)
    n_total = n_total or rec["s"]
    idxs, draws, diag = mixture_draws(rec, idx_map, K, n_total=n_total,
                                      seed_base=seed_base)
    assert idxs == idxs_re
    jumps = {str(th): gt_jumps(idxs, gt, th) for th in JUMP_THRESH_SENS}
    prox, _ = region_masks(idxs, jumps[str(JUMP_THRESH)], JUMP_RADIUS)
    return {
        "row_id": rec["meta"]["row_id"], "rec": rec, "idxs": idxs,
        "categories": kept, "idx_map": idx_map.tolist(), "K": K,
        "gt": gt, "gt_stored": gt_stored, "draws": draws, "diag": diag,
        "jumps": jumps, "prox": prox,
        "gt_tv_stored_vs_recomputed": float(tv_to_gt(gt, gt_stored).mean()),
    }


def run_condition(q: dict, S: int, stride: int, models: list[str] = MODELS
                  ) -> dict:
    idxs = np.asarray(q["idxs"], float)
    K = q["K"]
    draws = q["draws"]
    obs_idx = np.arange(0, len(idxs), stride)
    obs_tok = idxs[obs_idx]
    draws_obs = draws[obs_idx]
    counts = counts_from_draws(draws_obs, K, 0, S)
    prox = q["prox"]
    gt = q["gt"]
    out = {"S": S, "stride": stride, "n_obs": len(obs_idx), "models": {}}
    for name in models:
        params, cv_scores = cv_select(name, draws_obs, obs_tok, S, K)
        m = MODEL_REGISTRY[name]()
        m.fit(obs_tok, counts, S, params)
        pred = m.predict(idxs)
        lo, hi = m.credible_band(idxs, BAND_LEVEL)
        tv = tv_to_gt(pred, gt)
        ll_pos = heldout_loglik_per_position(pred, draws, S)
        cov = band_coverage(lo, hi, gt)
        out["models"][name] = {
            "params": params,
            "tv": summarize(tv, prox),
            "loglik": summarize(ll_pos, prox),
            "coverage": summarize(cov, prox),
            "tv_per_position": tv.tolist(),
            "ll_per_position": ll_pos.tolist(),
            "loglik_mean": heldout_loglik(pred, draws, S),
            "pred": pred.tolist(), "band_lo": lo.tolist(),
            "band_hi": hi.tolist(),
        }
    return out


def paired_loglik_tolerance(ll_a: np.ndarray, ll_b: np.ndarray) -> float:
    """2 * SE of the mean per-position paired log-lik difference."""
    d = np.asarray(ll_a) - np.asarray(ll_b)
    return float(2.0 * d.std(ddof=1) / np.sqrt(len(d)))


def select_default(conds_with_prox: list[tuple[dict, np.ndarray]],
                   simplicity: list[str]) -> dict:
    """Pre-registered Q1 rule at the headline condition, pooled over
    questions: simplest model within NOISE_FLOOR_TV of the best mean TV in
    BOTH regions, and within 2*SE (paired, per position) of the best
    held-out log-likelihood in both regions.

    conds_with_prox: [(run_condition output, prox mask), ...] per question.
    """
    pooled = {}
    for name in simplicity:
        tv = np.concatenate([np.asarray(c["models"][name]["tv_per_position"])
                             for c, _ in conds_with_prox])
        ll = np.concatenate([np.asarray(c["models"][name]["ll_per_position"])
                             for c, _ in conds_with_prox])
        pooled[name] = {"tv": tv, "ll": ll}
    prox = np.concatenate([np.asarray(p, bool) for _, p in conds_with_prox])

    def region_mean(v, mask):
        return float(v[mask].mean()) if mask.any() else -np.inf

    best_tv = {r: min(pooled[n]["tv"][mask].mean() for n in simplicity)
               for r, mask in (("jump", prox), ("flat", ~prox))}
    best_ll_name = {r: max(simplicity,
                           key=lambda n: region_mean(pooled[n]["ll"], mask))
                    for r, mask in (("jump", prox), ("flat", ~prox))}
    report = {}
    winner_strict = None
    winner_floor = None
    for name in simplicity:
        ok_strict = True
        ok_floor = True
        detail = {}
        for r, mask in (("jump", prox), ("flat", ~prox)):
            if not mask.any():
                continue
            tv_gap = float(pooled[name]["tv"][mask].mean() - best_tv[r])
            bn = best_ll_name[r]
            ll_gap = float(pooled[bn]["ll"][mask].mean()
                           - pooled[name]["ll"][mask].mean())
            tol = paired_loglik_tolerance(pooled[bn]["ll"][mask],
                                          pooled[name]["ll"][mask])
            detail[r] = {"tv_gap_to_best": tv_gap, "ll_gap_to_best": ll_gap,
                         "ll_tolerance_2se": tol}
            if tv_gap > NOISE_FLOOR_TV:
                ok_strict = ok_floor = False
            if ll_gap > max(tol, 0.0):
                ok_strict = False
            if ll_gap > NOISE_FLOOR_TV:
                ok_floor = False
        report[name] = {"qualifies_strict": ok_strict,
                        "qualifies_floor": ok_floor, **detail}
        if ok_strict and winner_strict is None:
            winner_strict = name
        if ok_floor and winner_floor is None:
            winner_floor = name
    return {
        "winner": winner_floor, "winner_strict": winner_strict,
        "per_model": report,
        "rule": (
            "simplest model with pooled mean TV within "
            f"{NOISE_FLOOR_TV} of best in both regions; log-lik tolerance "
            "reported under two readings — 'floor': within "
            f"{NOISE_FLOOR_TV} nats/draw of best (the plan's noise-floor "
            "applied to both metrics; primary), 'strict': within 2*paired-SE "
            "of best (statistical tie)"),
    }


def _pooled_region_means(conds_with_prox: list[tuple[dict, np.ndarray]],
                         models: list[str]) -> dict:
    """Pooled (over questions) per-region mean TV / held-out log-lik."""
    prox = np.concatenate([np.asarray(p, bool) for _, p in conds_with_prox])
    out = {}
    for name in models:
        tv = np.concatenate([np.asarray(c["models"][name]["tv_per_position"])
                             for c, _ in conds_with_prox])
        ll = np.concatenate([np.asarray(c["models"][name]["ll_per_position"])
                             for c, _ in conds_with_prox])
        out[name] = {
            "tv_flat": float(tv[~prox].mean()),
            "tv_jump": float(tv[prox].mean()) if prox.any() else None,
            "ll_flat": float(ll[~prox].mean()),
            "ll_jump": float(ll[prox].mean()) if prox.any() else None,
        }
    return out


def hybrid_verdict(conds_with_prox: list[tuple[dict, np.ndarray]],
                   models: list[str], hybrids: list[str] = None) -> dict:
    """The Q1 rule at the headline condition, pooled over questions.

    A hybrid is SUPPORTED if, pooled over questions:
      flat:  TV <= best model's flat TV + NOISE_FLOOR_TV
             and log-lik >= best flat log-lik - LL_TOL
      jump:  TV <= M4's jump TV (at or below, strict)
             and log-lik >= M4's jump log-lik - LL_TOL
    """
    hybrids = hybrids or HYBRIDS
    means = _pooled_region_means(conds_with_prox, models)
    non_hybrid = [m for m in models if m not in hybrids]
    best_flat_tv = min(means[m]["tv_flat"] for m in non_hybrid)
    best_flat_ll = max(means[m]["ll_flat"] for m in non_hybrid)
    m4 = means["M4_segment"]
    per = {}
    for h in hybrids:
        mh = means[h]
        checks = {
            "flat_tv_ok": mh["tv_flat"] <= best_flat_tv + NOISE_FLOOR_TV,
            "flat_ll_ok": mh["ll_flat"] >= best_flat_ll - LL_TOL,
            "jump_tv_ok": mh["tv_jump"] <= m4["tv_jump"],
            "jump_ll_ok": mh["ll_jump"] >= m4["ll_jump"] - LL_TOL,
        }
        per[h] = {
            **checks,
            "supported": bool(all(checks.values())),
            "tv_flat": mh["tv_flat"], "tv_jump": mh["tv_jump"],
            "ll_flat": mh["ll_flat"], "ll_jump": mh["ll_jump"],
            "gap_flat_tv_to_best": mh["tv_flat"] - best_flat_tv,
            "gap_jump_tv_to_m4": mh["tv_jump"] - m4["tv_jump"],
            "gap_flat_ll_to_best": best_flat_ll - mh["ll_flat"],
            "gap_jump_ll_to_m4": m4["ll_jump"] - mh["ll_jump"],
        }
    return {
        "per_hybrid": per,
        "supported": bool(any(per[h]["supported"] for h in hybrids)),
        "references": {"best_flat_tv": best_flat_tv,
                       "best_flat_ll": best_flat_ll,
                       "best_flat_tv_model": min(non_hybrid, key=lambda m: means[m]["tv_flat"]),
                       "best_flat_ll_model": max(non_hybrid, key=lambda m: means[m]["ll_flat"]),
                       "m4_tv_jump": m4["tv_jump"], "m4_ll_jump": m4["ll_jump"]},
        "pooled_region_means": means,
        "rule": ("flat: TV within NOISE_FLOOR_TV of best non-hybrid AND "
                 "log-lik within LL_TOL of best; jump: TV <= M4 (strict) "
                 f"AND log-lik within LL_TOL of M4. LL_TOL={LL_TOL}"),
    }


def seg_count_curves(q: dict, S: int, stride: int) -> dict:
    """Q2(i): segment count vs penalty per detection variant, on the full
    S-draw counts at the given stride."""
    from .models import PEN_GRID, segment_positions
    idxs = np.asarray(q["idxs"], float)
    obs_idx = np.arange(0, len(idxs), stride)
    counts = counts_from_draws(q["draws"][obs_idx], q["K"], 0, S)
    obs_tok = idxs[obs_idx]
    out = {}
    for variant in ("mult", "linear", "l2"):
        out[variant] = {str(p): len(segment_positions(obs_tok, counts,
                                                      variant, p))
                        for p in PEN_GRID}
    return out


def epsilon_frontier(conds: dict[tuple[int, int], list[dict]], model: str
                     ) -> list[dict]:
    """For one model: pooled mean TV per (S, stride) with relative sampling
    cost S / stride (draws per base-grid position).

    conds: {(S, stride): [run_condition output per question, ...]}.
    """
    rows = []
    for (S, stride), cond_list in conds.items():
        tv = np.concatenate([
            np.asarray(c["models"][model]["tv_per_position"])
            for c in cond_list])
        rows.append({"S": S, "stride": stride, "cost": S / stride,
                     "mean_tv": float(tv.mean())})
    return sorted(rows, key=lambda r: r["cost"])
