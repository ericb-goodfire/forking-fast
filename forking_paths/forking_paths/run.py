"""Stage A (GPU): the Forking Paths sampling pipeline.

Runs base-path decode -> branch enumeration -> prefix-cached resampling ->
local outcome extraction (R), and writes a compact per-question outcome JSON
(o_t, o_{t,w}, base-path tokens, diagnostics). The CPU-cheap statistical
analysis (BEAST change-point detection + survival) runs afterward in analyze.py.

cwd-robust: resolves the package import path from __file__, so it works whether
launched from the repository root or elsewhere.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

import numpy as np

# make `import forking_paths` work regardless of cwd
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from forking_paths.config import ForkingConfig
from forking_paths.model import ForkingModel
from forking_paths.prompts import build_prompt_ids, THEREFORE_ABCD
from forking_paths.resample import enumerate_branches
from forking_paths.outcomes import build_outcome_vectors
from forking_paths.answers import parse_mmlu_answer


def _round(a, nd=6):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def extract_answers_for_branches(model, base, branches, cont_by_branch, cfg,
                                 therefore_ids, diag_sample=200, rng=None):
    """Return (answers_by_branch, diagnostics).

    answers_by_branch[i] = list of category strings (len = #continuations).
    Stage 1: regex on the full response text.
    Stage 2: logit-read fallback on the same model for regex misses.
    Also runs a regex-vs-logit agreement check on a random subsample of
    regex-resolved continuations.
    """
    rng = rng or np.random.default_rng(cfg.seed)
    tok = model.tokenizer

    # First pass: regex. Track misses (need fallback) and a diag subsample.
    answers = [[None] * len(conts) for conts in cont_by_branch]
    miss_locs = []          # (bi, ci, fallback_prefix_ids)
    regex_hits = []         # (bi, ci, letter, full_prefix_ids) for agreement diag
    n_total = 0
    n_regex = 0
    for bi, (b, conts) in enumerate(zip(branches, cont_by_branch)):
        resp_prefix = base.gen_ids[:b.idx] + [b.tok_id]
        for ci, cont in enumerate(conts):
            n_total += 1
            full_ids = resp_prefix + cont
            text = tok.decode(full_ids, skip_special_tokens=True)
            letter = parse_mmlu_answer(text)
            if letter is not None:
                answers[bi][ci] = letter
                n_regex += 1
                regex_hits.append((bi, ci, letter,
                                   base.prompt_ids + full_ids + therefore_ids))
            else:
                answers[bi][ci] = None
                miss_locs.append((bi, ci,
                                  base.prompt_ids + full_ids + therefore_ids))

    # Second pass: logit-read fallback for regex misses.
    if miss_locs:
        prefixes = [m[2] for m in miss_locs]
        letters = model.logit_read_letters(prefixes, seed=cfg.seed)
        for (bi, ci, _), L in zip(miss_locs, letters):
            answers[bi][ci] = L if L in ("A", "B", "C", "D") else "Other"

    # Agreement diagnostic on a random subsample of regex hits.
    agreement = None
    if regex_hits:
        k = min(diag_sample, len(regex_hits))
        idx = rng.choice(len(regex_hits), size=k, replace=False)
        sub = [regex_hits[i] for i in idx]
        sub_letters = model.logit_read_letters([s[3] for s in sub], seed=cfg.seed)
        agree = sum(1 for s, L in zip(sub, sub_letters) if s[2] == L)
        agreement = {"n": int(k), "agree": int(agree),
                     "rate": float(agree / k) if k else None}

    diag = {
        "n_continuations": int(n_total),
        "n_regex_resolved": int(n_regex),
        "regex_coverage": float(n_regex / n_total) if n_total else None,
        "n_logit_fallback": int(len(miss_locs)),
        "regex_vs_logit_agreement": agreement,
    }
    return answers, diag


def run_question(model, q, cfg, is_instruct, therefore_ids, diag_sample=200):
    t_start = time.time()
    prompt_ids = build_prompt_ids(model.tokenizer, q["question"], q["choices"], is_instruct)

    base = model.base_path(prompt_ids, max_tokens=cfg.base_max_tokens,
                           top_k=cfg.top_k, seed=cfg.seed)
    branches = enumerate_branches(base, cfg)

    # Split idx==0 branches (need N0 samples) from the rest (S samples).
    idx0 = [b for b in branches if b.idx == 0]
    rest = [b for b in branches if b.idx != 0]

    t_gen = time.time()
    cont_by_branch_map = {}
    if idx0:
        outs0, _ = model.resample([b.prefix_ids for b in idx0], n=cfg.n0_samples,
                                  max_tokens=cfg.cont_max_tokens,
                                  temperature=cfg.cont_temperature, seed=cfg.seed)
        for b, o in zip(idx0, outs0):
            cont_by_branch_map[id(b)] = o
    if rest:
        outs1, _ = model.resample([b.prefix_ids for b in rest], n=cfg.n_samples,
                                  max_tokens=cfg.cont_max_tokens,
                                  temperature=cfg.cont_temperature, seed=cfg.seed)
        for b, o in zip(rest, outs1):
            cont_by_branch_map[id(b)] = o
    gen_secs = time.time() - t_gen

    cont_by_branch = [cont_by_branch_map[id(b)] for b in branches]

    answers, diag = extract_answers_for_branches(
        model, base, branches, cont_by_branch, cfg, therefore_ids,
        diag_sample=diag_sample)

    outcomes = build_outcome_vectors(branches, answers, cfg.outcome_categories)

    # compact o_{t,w}
    o_tw_compact = {}
    for t, vecs in outcomes["o_tw"].items():
        o_tw_compact[str(t)] = [
            [int(tok_id), float(round(pnorm, 6)), bool(is_base), _round(vec)]
            for (tok_id, pnorm, is_base, vec) in vecs
        ]

    # per-position base token text (for highlighting change points)
    base_tok_texts = [model.tokenizer.decode([tid]) for tid in base.gen_ids]

    record = {
        "meta": {
            "row_id": q["row_id"], "question": q["question"],
            "choices": q["choices"], "answer_letter": q["answer_letter"],
            "is_instruct": bool(is_instruct),
        },
        "categories": outcomes["categories"],
        "base": {
            "gen_ids": [int(x) for x in base.gen_ids],
            "n_response_tokens": len(base.gen_ids),
            "finish_reason": base.finish_reason,
            "base_text": model.decode(base.gen_ids),
            "token_texts": base_tok_texts,
        },
        "idxs": outcomes["idxs"],
        "o_t": _round(outcomes["o_t"]),
        "o_tw": o_tw_compact,
        "base_tokens": {str(k): int(v) for k, v in outcomes["base_tokens"].items()},
        "n_branches": len(branches),
        "diagnostics": diag,
        "timing": {"gen_secs": gen_secs, "total_secs": time.time() - t_start},
        "config": cfg.to_dict(),
    }
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["base", "instruct"], required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--question-index", type=int, default=None,
                    help="run a single question by position (for smoke / efficiency)")
    ap.add_argument("--indices", default=None,
                    help="comma-separated question positions to run (e.g. '8,9')")
    ap.add_argument("--prefix-caching", choices=["on", "off"], default="on")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--n0-samples", type=int, default=None)
    ap.add_argument("--base-max-tokens", type=int, default=None)
    ap.add_argument("--cont-max-tokens", type=int, default=None)
    ap.add_argument("--diag-sample", type=int, default=200)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--tok-depth", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = ForkingConfig()
    if args.smoke:
        cfg.n_samples = 5
        cfg.n0_samples = 10
        cfg.base_max_tokens = 200
        cfg.cont_max_tokens = 200
        cfg.tok_depth = 200
    if args.n_samples is not None:
        cfg.n_samples = args.n_samples
    if args.n0_samples is not None:
        cfg.n0_samples = args.n0_samples
    if args.base_max_tokens is not None:
        cfg.base_max_tokens = args.base_max_tokens
    if args.cont_max_tokens is not None:
        cfg.cont_max_tokens = args.cont_max_tokens
    if args.tok_depth is not None:
        cfg.tok_depth = args.tok_depth

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.questions) as f:
        questions = json.load(f)
    if args.indices is not None:
        idxs = [int(x) for x in args.indices.split(",") if x.strip() != ""]
        questions = [questions[i] for i in idxs]
    elif args.question_index is not None:
        questions = [questions[args.question_index]]
    elif args.max_questions is not None:
        questions = questions[: args.max_questions]

    is_instruct = args.mode == "instruct"
    model = ForkingModel(
        args.model, enable_prefix_caching=(args.prefix_caching == "on"),
        seed=cfg.seed, gen_batch=args.gen_batch,
    )
    therefore_ids = model.tokenizer(THEREFORE_ABCD, add_special_tokens=False)["input_ids"]

    print(f"[run] model={args.model} mode={args.mode} "
          f"prefix_caching={args.prefix_caching} nq={len(questions)} "
          f"S={cfg.n_samples} N0={cfg.n0_samples}", flush=True)

    manifest = []
    t0 = time.time()
    for i, q in enumerate(questions):
        rec = run_question(model, q, cfg, is_instruct, therefore_ids,
                           diag_sample=args.diag_sample)
        suffix = f"_{args.tag}" if args.tag else ""
        fname = f"outcomes_row{q['row_id']:03d}{suffix}.json"
        fpath = os.path.join(args.out_dir, fname)
        with open(fpath, "w") as f:
            json.dump(rec, f)
        manifest.append({
            "row_id": q["row_id"], "file": fname,
            "n_response_tokens": rec["base"]["n_response_tokens"],
            "n_branches": rec["n_branches"],
            "regex_coverage": rec["diagnostics"]["regex_coverage"],
            "gen_secs": rec["timing"]["gen_secs"],
        })
        report_progress(step=i + 1, total_steps=len(questions), phase=f"forking-{args.mode}")
        print(f"[run] q{i} row={q['row_id']} L={rec['base']['n_response_tokens']} "
              f"branches={rec['n_branches']} gen={rec['timing']['gen_secs']:.1f}s "
              f"regex_cov={rec['diagnostics']['regex_coverage']:.2f}", flush=True)

    mpath = os.path.join(args.out_dir, f"manifest_{args.mode}{('_'+args.tag) if args.tag else ''}.json")
    with open(mpath, "w") as f:
        json.dump({"mode": args.mode, "model": args.model,
                   "prefix_caching": args.prefix_caching,
                   "total_secs": time.time() - t0, "questions": manifest}, f, indent=2)
    print(f"[run] DONE total={time.time()-t0:.1f}s manifest={mpath}", flush=True)


if __name__ == "__main__":
    main()
