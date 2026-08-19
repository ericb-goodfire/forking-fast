"""Regenerate data/MANIFEST.json from the store files on disk.

Loads every store, checks that the loader's recomputed full-pool curve
reproduces the recorded o_t_full exactly (6-decimal precision), and records
per-file SHA-256, size, and identity fields plus the coverage summary that
scripts/verify_release.py asserts against.

Usage (from the repository root):  python scripts/make_data_manifest.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

_spec = importlib.util.spec_from_file_location(
    "release_loader", os.path.join(DATA, "loader.py"))
loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loader)


def main():
    import numpy as np

    files = {}
    for rel, path in loader.iter_stores(DATA):
        raw = open(path, "rb").read()
        rec = loader.load_store(path)
        assert loader.matches_recorded(rec), (rel, "o_t_full mismatch")
        files[rel] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "row_id": rec["meta"]["row_id"],
            "track": rec["meta"]["track"],
            "model": rec["meta"]["model"],
            "s": rec["s"],
            "n_positions": len(rec["idxs"]),
            "n_branches": len(rec["branches"]),
        }
        print(f"ok {rel}", file=sys.stderr)

    ref_rel = "s200/llama/row039.json.gz"
    ref = loader.load_store(os.path.join(DATA, ref_rel))
    _, o = loader.o_t(ref)
    curve = np.round(o, 6).tolist()
    assert curve == ref["o_t_full"]
    ref_sha = hashlib.sha256(
        json.dumps(curve, separators=(",", ":")).encode()).hexdigest()

    manifest = {
        "dataset": "Forking Paths outcome stores, tinyMMLU rows 0-99",
        "schema": (
            "One gzipped JSON record per (track, row). Fields: meta "
            "(question/choices/answer_letter/row_id/track/model), s (draws "
            "per branch), idxs (observed response-token grid positions), "
            "categories (A/B/C/D/Other), base (greedy base path), branches "
            "(per kept next-token branch at each position: t, tok_id, tok_p, "
            "is_base, answers[s] in chunk-major draw order, cont_lens[s]), "
            "o_t_full (recorded tok_p-weighted outcome curve, 6-decimal), "
            "plus collection-time gates/diagnostics/config. See "
            "data/loader.py for the reference loading code."
        ),
        "coverage": {
            "s200": {
                "rows": "000-099 (tinyMMLU rows 0-99, complete)",
                "tracks": {
                    "llama": "meta-llama/Meta-Llama-3-8B-Instruct, "
                             "every-token grid",
                    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B, "
                                "sentence grid",
                },
                "s": 200,
                "n_files": sum(1 for k in files if k.startswith("s200/")),
            },
            "s1000": {
                "files": {
                    "s1000/llama/row039_stride4.json.gz":
                        "row 39, Llama track, stride-4 grid, S=1000 "
                        "(independent development run)",
                    "s1000/llama/row039_everytoken.json.gz":
                        "row 39, Llama track, every-token grid, S=1000 "
                        "(extension of the S=200 store: draws 0-199 are "
                        "identical to s200/llama/row039.json.gz)",
                },
                "note": (
                    "The original development runs also produced a row-12 "
                    "stride-4 S=1000 store; its only copies were lost before "
                    "release assembly, so it is not included."
                ),
            },
        },
        "excluded": (
            "Raw-completion text sidecars (per-draw continuation text) are "
            "not part of this release; only outcome categories are shipped."
        ),
        "reference_o_t": {
            "file": ref_rel,
            "check": (
                "round(loader.o_t(rec)[1], 6) must equal rec['o_t_full'] "
                "exactly; sha256 below is over the canonical JSON of that "
                "rounded curve"
            ),
            "sha256": ref_sha,
        },
        "files": files,
    }
    out = os.path.join(DATA, "MANIFEST.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"wrote {out}: {len(files)} files")


if __name__ == "__main__":
    main()
