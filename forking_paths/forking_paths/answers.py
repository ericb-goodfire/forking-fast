"""Local outcome extraction R for MMLU (A/B/C/D), no external judge.

Two stages, both fully local (plan §"Outcome extraction"):
  1. regex answer-cleansing on the continuation text (Kojima-style);
  2. for continuations where regex fails, a logit-read fallback appends the
     answer-elicitation suffix and reads argmax over the {A,B,C,D} token logits
     on the *same* model (one batched forward pass).

`parse_mmlu_answer` returns "A"/"B"/"C"/"D" or None (None -> needs fallback).
"""
from __future__ import annotations

import re

VALID = ("A", "B", "C", "D")

# Ordered patterns; we take the LAST match in the text (final answer wins).
_PATTERNS = [
    r"answer\s+is\s*:?\s*\(?\s*([ABCD])\b",
    r"answer\s*:\s*\(?\s*([ABCD])\b",
    r"answer\s+is\s+\(?\s*([ABCD])\b",
    r"\bthe\s+correct\s+answer\s+is\s+\(?\s*([ABCD])\b",
    r"\boption\s+\(?\s*([ABCD])\b",
    r"\(([ABCD])\)\s*[.)]?\s*$",
    r"\b([ABCD])\)\s*$",
    r"^\s*\(?([ABCD])\)?\s*$",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _PATTERNS]


def parse_mmlu_answer(text: str):
    """Regex answer-cleansing. Returns 'A'..'D' or None if no clean match."""
    if not text:
        return None
    last = None
    last_pos = -1
    for cre in _COMPILED:
        for m in cre.finditer(text):
            if m.start() >= last_pos:
                last_pos = m.start()
                last = m.group(1).upper()
    # Also handle a bare trailing "The answer is A." with punctuation stripped.
    if last in VALID:
        return last
    return None


def letter_from_token_text(tok_text: str):
    """Map a decoded token string to a letter A-D, else None."""
    s = tok_text.strip().strip("()[].:*").upper()
    if s in VALID:
        return s
    return None
