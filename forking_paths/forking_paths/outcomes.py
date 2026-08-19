"""Probability-weighted outcome distributions o_t and o_{t,w} (paper Eq. 1-2).

o_{t,w} is the Monte-Carlo estimate E_s[R] = the normalized histogram over the
fixed outcome categories (A,B,C,D,Other) across the S continuations for that
(t, w). Because continuations are drawn from p(x_{>t} | x*_<t, x_t=w), the
uniform average over samples is the unbiased estimate of the sample-probability-
weighted expectation in Eq. 1 (this matches the reference code, whose final
`weighted` column reduces to the answer histogram). o_t = sum_w p_norm(w) o_{t,w}
with token probabilities normalized to sum to 1 over the kept tokens at index t.
"""
from __future__ import annotations

import numpy as np


def build_outcome_vectors(branches, answers_by_branch, categories):
    """
    branches: list[Branch] (aligned with answers_by_branch)
    answers_by_branch: list (per branch) of list of category strings (len S)
    Returns dict:
      idxs: sorted unique indices t
      o_t:   (T, C) array, outcome distribution per index
      o_tw:  dict[t] -> list of (tok_id, tok_p_norm, is_base, o_vec (C,))
      base_tokens: dict[t] -> greedy tok_id
    """
    C = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # group branches by idx
    by_idx = {}
    for b, ans in zip(branches, answers_by_branch):
        by_idx.setdefault(b.idx, []).append((b, ans))

    idxs = sorted(by_idx.keys())
    o_t = np.zeros((len(idxs), C), dtype=float)
    o_tw = {}
    base_tokens = {}

    for row, t in enumerate(idxs):
        items = by_idx[t]
        # token prob normalization over kept tokens at this index
        tok_ps = np.array([b.tok_p for (b, _) in items], dtype=float)
        Z = tok_ps.sum()
        tok_ps_norm = tok_ps / Z if Z > 0 else np.ones_like(tok_ps) / len(tok_ps)

        vecs = []
        o_t_row = np.zeros(C, dtype=float)
        for (b, ans), pnorm in zip(items, tok_ps_norm):
            hist = np.zeros(C, dtype=float)
            for a in ans:
                hist[cat_idx.get(a, cat_idx["Other"])] += 1.0
            if hist.sum() > 0:
                hist /= hist.sum()
            vecs.append((b.tok_id, float(pnorm), b.is_base, hist))
            o_t_row += pnorm * hist
            if b.is_base:
                base_tokens[t] = b.tok_id
        o_t[row] = o_t_row
        o_tw[t] = vecs
        if t not in base_tokens:
            # fall back to the highest-prob token as base
            base_tokens[t] = max(items, key=lambda it: it[0].tok_p)[0].tok_id

    return {"idxs": idxs, "o_t": o_t, "o_tw": o_tw, "base_tokens": base_tokens,
            "categories": list(categories)}
