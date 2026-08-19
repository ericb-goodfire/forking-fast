"""Re-derive per-combination PELT changepoints for the standalone dashboard.

For every (track, row, S, stride) combo in the dashboard payloads, re-fit
M5a_segkernel with the combo's STORED CV-selected params on the same
mixture draws (same per-track seed bases via run_scale.prepare_question),
assert the re-fit reproduces the stored smoothed curve exactly (equality of
the round-3 values embedded in the payload), then export the fitted
segmentation's interior boundaries on the token axis (the midpoint between
the observed positions flanking each PELT breakpoint — exactly where the
fitted model switches segments).

Also exports a sha1 of each combo's stored pred so the builder can assert
the served payload it embeds is the same one validated here.

Output: cps_meta.json {track: {"row|S|stride": {"cps": [...], "sha": h}}}
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "otrecon"))
from otrecon.data import counts_from_draws                    # noqa: E402
from otrecon.models import MODEL_REGISTRY                     # noqa: E402
from src.run_scale import prepare_question, rec_path          # noqa: E402

DEFAULT_DATA = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "s200"))


def sha(pred_list):
    return hashlib.sha1(json.dumps(
        pred_list, separators=(",", ":")).encode()).hexdigest()


def do_row(task):
    track, row, combos, data_dir = task
    q = prepare_question(rec_path(data_dir, track, row), track)
    idxs = np.asarray(q["idxs"], float)
    out = {}
    for key, combo in combos.items():
        S, stride = int(key.split("|")[1]), int(key.split("|")[2])
        obs_idx = np.arange(0, len(idxs), stride)
        obs_tok = idxs[obs_idx]
        block = q["draws"][obs_idx][:, :S]
        counts = counts_from_draws(block, q["K"], 0, S)
        m = MODEL_REGISTRY["M5a_segkernel"]()
        m.fit(obs_tok, counts, S, combo["params"])
        pred = np.round(m.predict(idxs), 3).tolist()
        assert pred == combo["pred"], (track, key, "re-fit != stored curve")
        n = len(obs_tok)
        cps = [round(0.5 * (obs_tok[e - 1] + obs_tok[e]), 2)
               for e in m.bkps[:-1] if 0 < e < n]
        out[key] = {"cps": cps, "sha": sha(combo["pred"])}
    return track, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama", required=True,
                    help="dashboard_llama.json payload file(s), comma-separated")
    ap.add_argument("--deepseek", required=True,
                    help="dashboard_deepseek.json payload file(s), comma-separated")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="s200 store root (has llama/, deepseek/)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    t0 = time.time()
    tasks = []
    for track, paths in (("llama", args.llama.split(",")),
                         ("deepseek", args.deepseek.split(","))):
        combos = {}
        for p in paths:
            with open(p) as f:
                combos.update(json.load(f)["combos"])
        by_row = {}
        for key, c in combos.items():
            by_row.setdefault(int(key.split("|")[0]), {})[key] = c
        assert sorted(by_row) == list(range(100)), track
        tasks += [(track, row, rc, args.data)
                  for row, rc in sorted(by_row.items())]
    print(f"[{time.time()-t0:.1f}s] {len(tasks)} row-tasks", flush=True)
    res = {"llama": {}, "deepseek": {}}
    with Pool(args.workers) as pool:
        for i, (track, out) in enumerate(pool.imap_unordered(do_row, tasks)):
            res[track].update(out)
            if (i + 1) % 40 == 0 or i + 1 == len(tasks):
                print(f"[{time.time()-t0:6.1f}s] {i+1}/{len(tasks)}",
                      flush=True)
    expected = {"llama": 5600, "deepseek": 3200}  # 100 rows x combos/row
    for track in res:
        assert len(res[track]) == expected[track], (track, len(res[track]))
    n_cp = {t: sum(len(v["cps"]) for v in res[t].values()) for t in res}
    print("changepoints total:", n_cp)
    with open(args.out, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print(f"[{time.time()-t0:.1f}s] wrote {args.out} "
          f"({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
