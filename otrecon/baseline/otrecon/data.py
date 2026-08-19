"""Loading FPA raw records and building position-level mixture draws.

A record (the released S=1000 merged / S=200 store format) stores, per grid
position t, a set of branches (kept next-tokens) each with a token
probability ``tok_p`` and a recorded sequence of S_full answer categories
(one per stored rollout, in draw order). The high-fidelity estimator is

    o_t = sum_w p_norm(w) * hist(answers_w) ,

with tok_p normalized over kept branches at t (forking_paths convention,
reimplemented here).

Cheap inputs are *mixture draws*: at each position, sample a branch from its
normalized weight and consume that branch's next unconsumed recorded draw.
The resulting per-position sequence is (conditionally) an iid sample from
o_t, so counts over the first S draws are Multinomial(S, o_t) and the
remaining draws are held-out data.
"""
from __future__ import annotations

import json

import numpy as np

BASE_CATEGORIES = ["A", "B", "C", "D", "Other"]


def load_record(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def branch_groups(rec: dict) -> dict[int, list[dict]]:
    """Branches grouped by position t, in stored (stable) order."""
    by_t: dict[int, list[dict]] = {}
    for b in rec["branches"]:
        by_t.setdefault(b["t"], []).append(b)
    return by_t


def normalized_weights(branches: list[dict]) -> np.ndarray:
    w = np.array([b["tok_p"] for b in branches], dtype=float)
    z = w.sum()
    return w / z if z > 0 else np.ones_like(w) / len(w)


def weighted_o_t(rec: dict, lo: int = 0, hi: int | None = None) -> tuple[list[int], np.ndarray]:
    """Recomputed weighted o_t over draws [lo:hi) in the 5 base categories."""
    cats = rec["categories"]
    cat_idx = {c: i for i, c in enumerate(cats)}
    by_t = branch_groups(rec)
    idxs = sorted(by_t.keys())
    o = np.zeros((len(idxs), len(cats)))
    for row, t in enumerate(idxs):
        bs = by_t[t]
        w = normalized_weights(bs)
        for b, wb in zip(bs, w):
            ans = b["answers"][lo:hi]
            hist = np.zeros(len(cats))
            for a in ans:
                hist[cat_idx.get(a, cat_idx["Other"])] += 1.0
            if hist.sum() > 0:
                hist /= hist.sum()
            o[row] += wb * hist
    return idxs, o


def collapse_map(gt: np.ndarray, categories: list[str], min_peak: float = 0.01
                 ) -> tuple[list[str], np.ndarray]:
    """Collapse categories whose ground-truth curve never exceeds ``min_peak``
    into 'Other'. Returns (kept category names, index map base->collapsed)."""
    other = categories.index("Other")
    keep = [k for k in range(len(categories))
            if k == other or gt[:, k].max() >= min_peak]
    kept_names = [categories[k] for k in keep]
    idx_map = np.zeros(len(categories), dtype=int)
    out_other = kept_names.index("Other")
    for k in range(len(categories)):
        idx_map[k] = keep.index(k) if k in keep else out_other
    return kept_names, idx_map


def collapse_curve(o: np.ndarray, idx_map: np.ndarray, K: int) -> np.ndarray:
    out = np.zeros((o.shape[0], K))
    for k_base, k_new in enumerate(idx_map):
        out[:, k_new] += o[:, k_base]
    return out


def mixture_draws(rec: dict, idx_map: np.ndarray, K: int, n_total: int = 1000,
                  seed_base: int = 43_000_000) -> tuple[list[int], np.ndarray, dict]:
    """Seeded position-level mixture draw sequences.

    Returns (idxs, draws (T, n_total) int array of collapsed category ids,
    diagnostics). Per position the RNG is seeded independently; branch draws
    are consumed in recorded order (nested prefixes are therefore consistent
    across S). If a branch's recorded draws are exhausted, the branch is
    resampled among non-exhausted branches (renormalized); occurrences are
    counted in diagnostics (expected ~0).
    """
    cats = rec["categories"]
    cat_idx = {c: i for i, c in enumerate(cats)}
    by_t = branch_groups(rec)
    idxs = sorted(by_t.keys())
    row_id = rec["meta"]["row_id"]
    draws = np.zeros((len(idxs), n_total), dtype=np.int8)
    exhausted_fallbacks = 0
    for row, t in enumerate(idxs):
        bs = by_t[t]
        w = normalized_weights(bs)
        n_branch = len(bs)
        avail = np.array([len(b["answers"]) for b in bs])
        ptr = np.zeros(n_branch, dtype=int)
        rng = np.random.default_rng([seed_base, row_id, t])
        picks = rng.choice(n_branch, size=n_total, p=w)
        for j, b_i in enumerate(picks):
            if ptr[b_i] >= avail[b_i]:
                ok = np.flatnonzero(ptr < avail)
                if len(ok) == 0:
                    raise RuntimeError(f"all branches exhausted at t={t}")
                w_ok = w[ok] / w[ok].sum()
                b_i = int(rng.choice(ok, p=w_ok))
                exhausted_fallbacks += 1
            a = bs[b_i]["answers"][ptr[b_i]]
            ptr[b_i] += 1
            draws[row, j] = idx_map[cat_idx.get(a, cat_idx["Other"])]
    diag = {"exhausted_fallbacks": int(exhausted_fallbacks),
            "n_positions": len(idxs), "n_total": n_total}
    return idxs, draws, diag


def counts_from_draws(draws: np.ndarray, K: int, lo: int, hi: int) -> np.ndarray:
    """(T, K) counts over draw slice [lo:hi)."""
    T = draws.shape[0]
    out = np.zeros((T, K), dtype=float)
    for k in range(K):
        out[:, k] = (draws[:, lo:hi] == k).sum(axis=1)
    return out
