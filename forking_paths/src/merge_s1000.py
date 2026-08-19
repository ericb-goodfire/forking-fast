"""Merge per-shard S=1000 raw files into one per-question record (CPU, no GPU).

Asserts: all shards present, identical base path / grid / config / chunk seeds,
every grid position present exactly once across shards. Re-derives o_t_full
from the merged branches and re-runs the 50-block mean identity as a
post-merge check. Output schema matches the S=200 store files (plus
chunk/shard provenance), so downstream analysis code can treat both uniformly.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from forking_paths.outcomes import build_outcome_vectors


class _B:
    __slots__ = ("idx", "tok_id", "tok_p", "is_base")

    def __init__(self, d):
        self.idx = d["t"]; self.tok_id = d["tok_id"]
        self.tok_p = d["tok_p"]; self.is_base = d["is_base"]


def merge_row(shard_files):
    recs = []
    for fp in sorted(shard_files):
        with open(fp) as f:
            recs.append(json.load(f))
    assert recs, "no shard files"
    r0 = recs[0]
    n_shards = r0["shard"]["num_shards"]
    assert len(recs) == n_shards, \
        f"expected {n_shards} shards, found {len(recs)}"
    assert sorted(r["shard"]["index"] for r in recs) == list(range(n_shards))

    for r in recs[1:]:
        assert r["base"]["gen_ids"] == r0["base"]["gen_ids"], "base path differs across shards"
        assert r["full_idxs"] == r0["full_idxs"], "grid differs across shards"
        assert r["s"] == r0["s"] and r["chunk_size"] == r0["chunk_size"]
        assert r["config"] == r0["config"], "config differs across shards"
        assert r["gates"]["ref_match"] is not False, "shard failed gate (a)"
        assert r["gates"]["block_mean_ok"], "shard failed gate (b)"

    # chunk seeds are shard-salted by design: all (shard, chunk) seeds must be
    # distinct and none may be 0 (seed 0 replays the prior runs' draws)
    all_seeds = [s for r in recs for s in r["chunk_seeds"]]
    assert len(set(all_seeds)) == len(all_seeds), "duplicate chunk seeds across shards"
    assert 0 not in all_seeds, "seed 0 is reserved by the prior runs' draws"

    # every grid position exactly once
    covered = []
    for r in sorted(recs, key=lambda r: r["shard"]["index"]):
        covered.extend(r["shard"]["positions"])
    assert covered == r0["full_idxs"], \
        "shard positions do not tile the grid exactly once"

    branches = []
    for r in sorted(recs, key=lambda r: r["shard"]["index"]):
        branches.extend(r["branches"])
    # shards are contiguous position ranges in index order
    assert [b["t"] for b in branches] == sorted(b["t"] for b in branches), \
        "merged branches not in position order"

    cats = r0["categories"]
    bobjs = [_B(b) for b in branches]
    answers = [b["answers"] for b in branches]
    out = build_outcome_vectors(bobjs, answers, cats)
    assert list(out["idxs"]) == list(r0["full_idxs"])
    o_t_full = np.asarray(out["o_t"], dtype=float)

    # post-merge 50-block mean identity
    s, sb = r0["s"], 20
    acc = None
    for k in range(s // sb):
        sl = [a[k * sb:(k + 1) * sb] for a in answers]
        ob = np.asarray(build_outcome_vectors(bobjs, sl, cats)["o_t"])
        acc = ob if acc is None else acc + ob
    assert np.allclose(acc / (s // sb), o_t_full, atol=1e-9), "post-merge block-mean check failed"

    merged = {
        "meta": r0["meta"],
        "s": r0["s"],
        "chunk_size": r0["chunk_size"],
        "chunk_seeds": {str(r["shard"]["index"]): r["chunk_seeds"]
                        for r in sorted(recs, key=lambda r: r["shard"]["index"])},
        "num_shards": n_shards,
        "idxs": list(map(int, r0["full_idxs"])),
        "categories": cats,
        "base": r0["base"],
        "branches": branches,
        "o_t_full": np.round(o_t_full, 6).tolist(),
        "gates": {
            "ref_match": all(r["gates"]["ref_match"] for r in recs),
            "block_mean_ok": True,
            "merge_grid_ok": True,
        },
        "diagnostics": {
            "per_shard": [
                {"shard": r["shard"]["index"], **r["diagnostics"],
                 "gen_secs": r["timing"]["gen_secs"],
                 "n_generated_tokens": r["n_generated_tokens"]}
                for r in sorted(recs, key=lambda r: r["shard"]["index"])],
        },
        "n_generated_tokens": int(sum(r["n_generated_tokens"] for r in recs)),
        "config": r0["config"],
    }
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir of shard files")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = {}
    for fp in glob.glob(os.path.join(args.raw_dir, "s1000_llama_row*_shard*.json")):
        rid = int(os.path.basename(fp).split("row")[1][:3])
        rows.setdefault(rid, []).append(fp)
    assert rows, f"no shard files in {args.raw_dir}"

    manifest = {}
    for rid, files in sorted(rows.items()):
        merged = merge_row(files)
        out = os.path.join(args.out_dir, f"s1000_llama_row{rid:03d}.json")
        with open(out, "w") as f:
            json.dump(merged, f)
        manifest[rid] = {
            "n_shards": merged["num_shards"],
            "n_positions": len(merged["idxs"]),
            "n_branches": len(merged["branches"]),
            "n_generated_tokens": merged["n_generated_tokens"],
            "gates": merged["gates"],
        }
        print(f"[merge] row {rid}: {len(files)} shards -> {out} "
              f"({merged['n_generated_tokens']} tokens)", flush=True)
    with open(os.path.join(args.out_dir, "merge_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
