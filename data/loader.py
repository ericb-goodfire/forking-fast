"""Loader for the released outcome stores (``data/s200`` and ``data/s1000``).

Each store is one gzipped JSON record per (model track, tinyMMLU row):

  meta        question text, choices, answer_letter, row_id, track, model id
  s           number of recorded draws per branch (200 or 1000)
  idxs        observed grid positions t (response-token indices)
  categories  outcome categories, ["A", "B", "C", "D", "Other"]
  base        the greedy base path (gen_ids, token texts, finish reason)
  branches    list of {t, tok_id, tok_p, is_base, answers, cont_lens}:
              one entry per kept next-token branch at position t, with the
              recorded outcome category of each of the s resampled rollouts
              (in draw order; chunk-major, so the first S entries of every
              branch form a valid nested S-draw prefix run)
  o_t_full    the recorded outcome curve (len(idxs) x len(categories)),
              the tok_p-weighted per-branch histogram over all s draws,
              rounded to 6 decimals
  gates, diagnostics, config, ...   collection-time validation and settings

Only numpy is required. Typical use:

    import loader
    rec = loader.load_store("s200/llama/row039.json.gz")
    idxs, curve = loader.o_t(rec)          # recompute the full-pool curve
    assert loader.matches_recorded(rec)    # == rec["o_t_full"] exactly

The heavier analysis stack (mixture draws, smoothing, CV) lives in the
``otrecon`` package, which loads these records directly.
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np

BASE_CATEGORIES = ["A", "B", "C", "D", "Other"]


def load_store(path: str) -> dict:
    """Load one store record (gzipped or plain JSON)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def branch_groups(rec: dict) -> dict[int, list[dict]]:
    """Branches grouped by grid position t, in stored (stable) order."""
    by_t: dict[int, list[dict]] = {}
    for b in rec["branches"]:
        by_t.setdefault(b["t"], []).append(b)
    return by_t


def normalized_weights(branches: list[dict]) -> np.ndarray:
    """Token probabilities of the kept branches at one position, normalized."""
    w = np.array([b["tok_p"] for b in branches], dtype=float)
    z = w.sum()
    return w / z if z > 0 else np.ones_like(w) / len(w)


def o_t(rec: dict, lo: int = 0, hi: int | None = None
        ) -> tuple[list[int], np.ndarray]:
    """Recompute the weighted outcome curve over draws [lo:hi).

    With the default full range this reproduces ``rec["o_t_full"]`` exactly
    after rounding to 6 decimals (the precision the sampler recorded).
    Returns (grid positions, (T, K) curve).
    """
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


def recorded_o_t(rec: dict) -> np.ndarray:
    """The store's recorded full-pool curve, shape (T, K)."""
    return np.asarray(rec["o_t_full"], dtype=float)


def matches_recorded(rec: dict) -> bool:
    """True when the recomputed full-pool curve equals the recorded one
    exactly (at the recorded 6-decimal precision)."""
    _, o = o_t(rec)
    return bool(np.array_equal(np.round(o, 6), recorded_o_t(rec)))


def iter_stores(data_root: str | None = None):
    """Yield (relpath, path) for every ``*.json.gz`` store under data_root
    (default: the directory containing this file), sorted by relpath."""
    root = data_root or os.path.dirname(os.path.abspath(__file__))
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".json.gz"):
                p = os.path.join(dirpath, name)
                out.append((os.path.relpath(p, root), p))
    return sorted(out)
