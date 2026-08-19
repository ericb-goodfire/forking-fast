"""Sentence segmentation of a base-path response, mapped back to token indices.

The pipeline works in token-ID space, so a sentence-level position selector must
map each sentence-END character offset in the decoded response text back to the
token index whose decoding covers that offset. We segment with pysbd (rule-based,
no model download; keeps decimals like "9.8" and abbreviations like "e.g."/"Dr."
intact) and fall back to a decimal-guarded regex when pysbd is unavailable or
yields a single span.

Segmentation is "good enough", not perfect, by design: a mis-split changes only
*which* token positions get resampled, never the correctness of the outcome
estimated at each position.
"""
from __future__ import annotations

import re

# sentence-ending punctuation followed by whitespace, guarding a decimal point
# between two digits (so "9.8" / "3.14" never split).
_REGEX_FALLBACK = re.compile(r"(?<!\d)([.!?])(?:[\"')\]]*)(\s+)")


def _char_end_offsets(tokenizer, gen_ids):
    """char_end[t] = length (in chars) of decode(gen_ids[:t+1]).

    Monotonic non-decreasing; used to map a char offset in the full decoded text
    to the token index that first covers it.
    """
    ends = []
    for t in range(len(gen_ids)):
        ends.append(len(tokenizer.decode(gen_ids[: t + 1], skip_special_tokens=True)))
    return ends


def _char_to_token(char_end, offset):
    """First token index t whose decoded prefix reaches `offset` chars."""
    for t, e in enumerate(char_end):
        if e >= offset:
            return t
    return len(char_end) - 1


def _pysbd_sentence_ends(text):
    """Return list of sentence-end char offsets via pysbd, or None on failure."""
    try:
        import pysbd
    except Exception:
        return None
    try:
        seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
        spans = seg.segment(text)
    except Exception:
        return None
    ends = []
    for sp in spans:
        end = getattr(sp, "end", None)
        if end is None:
            return None
        # trim trailing whitespace that pysbd includes in the span
        s = getattr(sp, "sent", "")
        end = end - (len(s) - len(s.rstrip()))
        ends.append(int(end))
    return ends if len(ends) >= 2 else None


def _regex_sentence_ends(text):
    """Decimal-guarded regex fallback: char offset just after each terminator."""
    ends = []
    for m in _REGEX_FALLBACK.finditer(text):
        ends.append(m.end(1))  # offset just after the . ! or ?
    return ends


def sentence_end_char_offsets(text):
    """Sentence-end char offsets in `text` (pysbd, regex fallback). Ordered."""
    ends = _pysbd_sentence_ends(text)
    if ends is None:
        ends = _regex_sentence_ends(text)
    return sorted(set(e for e in ends if 0 < e <= len(text)))


def sentence_positions(tokenizer, gen_ids, depth):
    """Token indices (< depth) at sentence boundaries of decode(gen_ids[:depth]).

    Returns a sorted list of token positions. Always includes 0 (the drift
    anchor o_0) so the reference distribution matches the every-token run.
    """
    ids = list(gen_ids[:depth])
    if not ids:
        return [0] if depth > 0 else []
    text = tokenizer.decode(ids, skip_special_tokens=True)
    char_end = _char_end_offsets(tokenizer, ids)
    ends = sentence_end_char_offsets(text)
    pos = {0}
    for c in ends:
        t = _char_to_token(char_end, c)
        if 0 <= t < depth:
            pos.add(t)
    return sorted(pos)
