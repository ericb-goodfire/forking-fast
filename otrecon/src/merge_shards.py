"""Merge sharded fan JSONL outputs into one canonically-ordered file.

Raw line bytes are preserved (no JSON re-serialization); records are ordered
by the canonical single-process task order (tracks in meta order, rows in
meta order, S in s_grid order, stride in the per-track stride grid order,
rep ascending). A merged N-shard run is therefore byte-identical to a merged
1-process run of the same task set.

Usage:
    python src/merge_shards.py --fan-dir OUT [--pattern 'fan.shard*.jsonl']
                               [--out fan.jsonl]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def merge(paths: list[str], meta: dict, out_path: str,
          allow_duplicates: bool = False) -> int:
    tracks = list(meta["tracks"])
    rows = [int(r) for r in meta["rows"]]
    s_grid = [int(s) for s in meta["s_grid"]]
    strides = {t: [int(x) for x in meta["strides"][t]] for t in
               meta["strides"]}
    t_rank = {t: i for i, t in enumerate(tracks)}
    r_rank = {r: i for i, r in enumerate(rows)}
    s_rank = {s: i for i, s in enumerate(s_grid)}
    st_rank = {t: {s: i for i, s in enumerate(strides[t])} for t in strides}

    keyed: dict[tuple, bytes] = {}
    for p in paths:
        with open(p, "rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                rec = json.loads(raw)
                key = (t_rank[rec["track"]], r_rank[int(rec["row"])],
                       s_rank[int(rec["S"])],
                       st_rank[rec["track"]][int(rec["stride"])],
                       int(rec["rep"]))
                if key in keyed and not allow_duplicates:
                    raise SystemExit(f"duplicate task key {key} in {p}")
                keyed[key] = raw
    with open(out_path, "wb") as f:
        for key in sorted(keyed):
            f.write(keyed[key])
    return len(keyed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-dir", required=True)
    ap.add_argument("--pattern", default="fan.shard*.jsonl")
    ap.add_argument("--out", default="fan.jsonl")
    ap.add_argument("--meta", default="",
                    help="fan_meta json (default: first shard's)")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="keep the last record per task key instead of "
                         "erroring (mirrors load_fan dedupe)")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.fan_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"no shard files match {args.pattern} in "
                         f"{args.fan_dir}")
    meta_path = args.meta
    if not meta_path:
        metas = sorted(glob.glob(os.path.join(args.fan_dir,
                                              "fan_meta.shard*.json")))
        if not metas:
            metas = sorted(glob.glob(os.path.join(args.fan_dir,
                                                  "fan_meta*.json")))
        if not metas:
            raise SystemExit("no fan_meta*.json found; pass --meta")
        meta_path = metas[0]
    with open(meta_path) as f:
        meta = json.load(f)
    out_path = os.path.join(args.fan_dir, args.out)
    n = merge(paths, meta, out_path, args.allow_duplicates)
    print(f"merged {len(paths)} shard files -> {out_path} ({n} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
