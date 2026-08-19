"""Enumerate (t, w) resample branches off a base path, in token-ID space.

For each position t in [0, tok_depth), keep the alternate tokens w whose
next-token probability p(x_t=w | x*_<t) >= p_thresh, always including the greedy
token. Each branch's prefix is prompt + gen[:t] + [w]; these prefixes are nested
across t so the prefix cache computes each shared trunk once.
"""
from __future__ import annotations

import numpy as np


class Branch:
    __slots__ = ("idx", "tok_id", "tok_p", "is_base", "prefix_ids")

    def __init__(self, idx, tok_id, tok_p, is_base, prefix_ids):
        self.idx = idx
        self.tok_id = tok_id
        self.tok_p = tok_p
        self.is_base = is_base
        self.prefix_ids = prefix_ids


def enumerate_branches(base, cfg):
    """Return list[Branch] for every position (the paper default)."""
    depth = min(cfg.tok_depth, len(base.gen_ids))
    return enumerate_branches_at(base, cfg, list(range(depth)))


def enumerate_branches_at(base, cfg, positions):
    """Return list[Branch] for the given resample `positions` (subset of t).

    Identical enumeration logic to the every-token path, restricted to the
    selected positions: at each t keep alternate tokens w with p(w) >= p_thresh
    plus the greedy token. Positions where only the greedy token clears the
    threshold contribute just the greedy branch (no forking alternate to sample)
    -- this is the "skip non-fork alternates" optimization, while every selected
    position still gets its greedy continuation for the o_t series.
    """
    branches = []
    depth = min(cfg.tok_depth, len(base.gen_ids))
    for t in positions:
        if t < 0 or t >= depth:
            continue
        greedy_id = base.gen_ids[t]
        ids = base.topk_ids[t]
        lps = base.topk_logprobs[t]
        probs = {tid: float(np.exp(lp)) for tid, lp in zip(ids, lps)}
        # candidate set: top-K already (the model wrapper returned top_k logprobs); keep p>=thresh
        kept = {}
        for tid, p in probs.items():
            if p >= cfg.p_thresh or tid == greedy_id:
                kept[tid] = p
        # ensure greedy present even if it wasn't in the returned logprob dict
        if greedy_id not in kept:
            kept[greedy_id] = probs.get(greedy_id, 1.0)
        prefix_base = base.prompt_ids + base.gen_ids[:t]
        for tid, p in kept.items():
            branches.append(
                Branch(idx=t, tok_id=tid, tok_p=p, is_base=(tid == greedy_id),
                       prefix_ids=prefix_base + [tid])
            )
    return branches
