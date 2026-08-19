"""Extract per-token base-path display strings + ground-truth answers for
all 100 tinyMMLU rows x both tracks, from the stored S=200 records.

Token derivation (needs HF tokenizers; downloads both models' tokenizers):
  gen_ids -> tokenizer.convert_ids_to_tokens (ByteLevel space)
          -> GPT-2 byte-inverse per token -> utf-8 decode(errors='replace')
This is the byte-level inverse-decoder path: decode([id]) on the R1 slow
tokenizer leaks raw BPE markers (Ġ/Ċ), a bug fixed in an earlier dashboard
revision.
One display string per token id, so plot-position alignment holds.

Assertions per row:
  - len(tokens) == len(gen_ids) == base.n_response_tokens
  - byte-exactness: b"".join(per-token bytes) == tokenizer.decode(gen_ids)
    bytes (skip_special_tokens=False, clean_up_tokenization_spaces=False)
  - cross-check vs stored record: llama token_texts match ours exactly;
    deepseek stored token_texts equal the ByteLevel strings (documents the
    stored contamination) while ours are clean
  - no U+0120/U+010A/U+010B in any output string
  - llama: idxs == range(n_tokens); deepseek: idxs strictly increasing,
    max(idxs) < n_tokens
  - answer_letter in {A,B,C,D}

Output: tokens_meta.json  {track: {row: {"answer": .., "tokens": [..]}}}
"""
import argparse
import gzip
import json
import os

DEFAULT_DATA = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "s200"))
MODELS = {"llama": "meta-llama/Meta-Llama-3-8B-Instruct",
          "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
BAD = ("\u0120", "\u010a", "\u010b")   # Ġ Ċ ċ


def bytes_to_unicode():
    """Canonical GPT-2 ByteLevel byte->unicode table (inlined; newer
    transformers no longer exports it)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def rec_path(data_dir, track, row):
    return os.path.join(data_dir, track, f"row{row:03d}.json.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="s200 store root (has llama/, deepseek/)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    u2b = {u: b for b, u in bytes_to_unicode().items()}
    out = {}
    stats = {}
    for track, model in MODELS.items():
        tok = AutoTokenizer.from_pretrained(model)
        out[track] = {}
        n_ds_contam = 0
        for row in range(100):
            r = json.load(gzip.open(rec_path(args.data, track, row), "rt"))
            ids = r["base"]["gen_ids"]
            n = r["base"]["n_response_tokens"]
            assert len(ids) == n, (track, row, len(ids), n)
            bl = tok.convert_ids_to_tokens(ids)
            assert len(bl) == n
            tok_bytes = []
            for t in bl:
                try:
                    tok_bytes.append(bytes(u2b[c] for c in t))
                except KeyError:
                    raise AssertionError(
                        f"{track} row{row}: non-ByteLevel token {t!r}")
            texts = [b.decode("utf-8", errors="replace") for b in tok_bytes]
            # vocab round-trip: token strings identify exactly these ids
            assert tok.convert_tokens_to_ids(bl) == ids, (track, row)
            # byte-exactness against the full-sequence decode
            full = tok.decode(ids, skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
            if track == "llama":
                assert b"".join(tok_bytes) == full.encode("utf-8"), \
                    (track, row)
            else:
                # the R1 tokenizer's decode itself leaks ByteLevel markers
                # (the researcher-flagged bug), so compare in byte space:
                # inverse(full decode) must equal the per-token join
                inv_full = bytes(u2b[c] for c in full)
                assert b"".join(tok_bytes) == inv_full, (track, row)
            # cross-check vs what the record stored
            stored = r["base"]["token_texts"]
            if track == "llama":
                assert stored == texts, (track, row, "stored texts differ")
            else:
                assert stored == bl, (track, row,
                                      "stored texts not ByteLevel strings")
                n_ds_contam += sum(any(m in s for m in BAD) for s in stored)
            assert not any(m in s for s in texts for m in BAD), (track, row)
            # grid alignment
            idxs = r["idxs"]
            if track == "llama":
                assert idxs == list(range(n)), (track, row, "idxs != range")
            else:
                assert all(a < b for a, b in zip(idxs, idxs[1:])), (track, row)
                assert 0 <= idxs[0] and idxs[-1] < n, (track, row, idxs[-1], n)
            ans = r["meta"]["answer_letter"]
            assert ans in "ABCD", (track, row, ans)
            out[track][str(row)] = {"answer": ans, "tokens": texts}
        ntok = sum(len(v["tokens"]) for v in out[track].values())
        stats[track] = {"rows": len(out[track]), "total_tokens": ntok}
        if track == "deepseek":
            stats[track]["stored_texts_bpe_marker_tokens"] = n_ds_contam
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(json.dumps(stats, indent=1))
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
