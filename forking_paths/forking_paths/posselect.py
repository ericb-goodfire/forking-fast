"""Position selectors: choose which base-path token positions t to resample.

Per question the generated-token cost is roughly
    (#positions) x (#alternates per position) x S x (continuation length),
and the position selector shrinks the first (largest) factor. Each selector is a
pure function of the base path; new selectors (e.g. entropy-gated) drop in here
without touching the sampler.

Every selector includes position 0 (the drift anchor o_0) so the reference
outcome distribution is identical across modes.
"""
from __future__ import annotations

from .segment import sentence_positions


def select_positions(base, cfg, selector="every_token", stride=4, tokenizer=None):
    """Return the sorted list of token positions to resample.

    base:     BasePath (has .gen_ids)
    selector: "every_token" | "stride" | "sentence"
    stride:   N for the stride selector (resample at t = 0, N, 2N, ...)
    tokenizer: required for the sentence selector (token<->char mapping)
    """
    depth = min(cfg.tok_depth, len(base.gen_ids))
    if depth <= 0:
        return []
    if selector == "every_token":
        pos = list(range(depth))
    elif selector == "stride":
        if stride < 1:
            raise ValueError("stride must be >= 1")
        pos = list(range(0, depth, stride))
    elif selector == "sentence":
        if tokenizer is None:
            raise ValueError("sentence selector requires a tokenizer")
        pos = sentence_positions(tokenizer, base.gen_ids, depth)
    else:
        raise ValueError(f"unknown selector: {selector}")
    if 0 not in pos:
        pos = [0] + pos
    return sorted(set(p for p in pos if 0 <= p < depth))
