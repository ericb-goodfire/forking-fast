"""Shard split determinism and canonical byte-identical merge."""
import json
import os

import numpy as np

from src.merge_shards import merge
from src.run_scale import (N_TOTAL, S_GRID, STRIDES, build_tasks,
                           canonical_tasks, parse_shard)


def test_canonical_tasks_matches_build_tasks():
    tracks, rows = ["llama", "deepseek"], [3, 0, 7]
    questions = {(t, r): {"n_total": N_TOTAL} for t in tracks for r in rows}
    assert canonical_tasks(tracks, rows) == build_tasks(questions)


def test_parse_shard():
    assert parse_shard("0/4") == (0, 4)
    assert parse_shard("3/4") == (3, 4)
    for bad in ("4/4", "-1/4", "1/0"):
        try:
            parse_shard(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_shards_partition_tasks():
    tasks = canonical_tasks(["llama"], list(range(5)))
    shards = [tasks[i::4] for i in range(4)]
    flat = [t for sh in shards for t in sh]
    assert sorted(flat) == sorted(tasks)
    assert len(flat) == len(tasks)
    assert len(set(map(tuple, flat))) == len(tasks)


def test_merge_byte_identical(tmp_path):
    tracks, rows = ["llama", "deepseek"], [0, 1]
    meta = {"tracks": tracks, "rows": rows, "s_grid": S_GRID,
            "strides": STRIDES}
    tasks = canonical_tasks(tracks, rows)
    rng = np.random.default_rng(0)

    def line(t):
        track, row, S, stride, rep = t
        return (json.dumps({"track": track, "row": row, "S": S,
                            "stride": stride, "rep": rep,
                            "models": {"x": float(rng.random())}})
                + "\n").encode()

    lines = {tuple(t): line(t) for t in tasks}
    # single-process run: written in arbitrary completion order
    single = tmp_path / "single.jsonl"
    order = rng.permutation(len(tasks))
    with open(single, "wb") as f:
        for i in order:
            f.write(lines[tuple(tasks[i])])
    # 4-shard run: each shard also in arbitrary order
    shard_paths = []
    for i in range(4):
        p = tmp_path / f"fan.shard{i}of4.jsonl"
        sh = tasks[i::4]
        idx = rng.permutation(len(sh))
        with open(p, "wb") as f:
            for j in idx:
                f.write(lines[tuple(sh[j])])
        shard_paths.append(str(p))

    out_single = str(tmp_path / "merged_single.jsonl")
    out_shards = str(tmp_path / "merged_shards.jsonl")
    n1 = merge([str(single)], meta, out_single)
    n2 = merge(shard_paths, meta, out_shards)
    assert n1 == n2 == len(tasks)
    assert open(out_single, "rb").read() == open(out_shards, "rb").read()


def test_merge_rejects_duplicates(tmp_path):
    meta = {"tracks": ["llama"], "rows": [0], "s_grid": [5],
            "strides": {"llama": [1]}}
    p = tmp_path / "fan.shard0of1.jsonl"
    rec = json.dumps({"track": "llama", "row": 0, "S": 5, "stride": 1,
                      "rep": 0}) + "\n"
    with open(p, "wb") as f:
        f.write(rec.encode())
        f.write(rec.encode())
    try:
        merge([str(p)], meta, str(tmp_path / "out.jsonl"))
        raise AssertionError("expected duplicate error")
    except SystemExit:
        pass
    # allowed when explicitly requested
    n = merge([str(p)], meta, str(tmp_path / "out.jsonl"),
              allow_duplicates=True)
    assert n == 1
