"""HuggingFace transformers wrapper operating in token-ID space.

Token-ID space (not strings) is used end-to-end so forcing an exact alternate
token never suffers tokenization drift. The prefix-cache optimization (paper App.
"Improving Computational Efficiency") is made explicit here: `base_path` returns
the full base-path KV cache, and `resample_kv_reuse` crops that cache to position
t and reuses it for every (t, w) branch — each nested prefix is computed once.
The main run uses `resample` (plain batched generation, decode-bound at this
scale); `resample_kv_reuse` powers the ON/OFF efficiency validation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BasePath:
    prompt_ids: list
    gen_ids: list
    topk_ids: list            # per-position: list[int] candidate token ids
    topk_logprobs: list       # per-position: list[float] logprobs
    finish_reason: str


class ForkingModel:
    def __init__(self, model_path, enable_prefix_caching=True, dtype="bfloat16",
                 max_model_len=4096, gpu_memory_utilization=0.90, seed=0,
                 gen_batch=96):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda"
        self.seed = seed
        self.gen_batch = gen_batch
        self.enable_prefix_caching = enable_prefix_caching
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=td, attn_implementation="sdpa",
        ).to(self.device).eval()
        self.eos_ids = self._eos_ids()

    def _eos_ids(self):
        eos = self.tokenizer.eos_token_id
        ids = set()
        if isinstance(eos, list):
            ids.update(eos)
        elif eos is not None:
            ids.add(eos)
        # Llama-3 also uses <|eot_id|> as a stop in chat
        for t in ("<|eot_id|>", "<|end_of_text|>"):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(t)
                if tid is not None and tid >= 0:
                    ids.add(tid)
            except Exception:
                pass
        return sorted(ids)

    # ---------------------------------------------------------------- base path
    @torch.no_grad()
    def base_path(self, prompt_ids, max_tokens, top_k, seed=0, temperature=0.0) -> BasePath:
        """Decode a base path. temperature==0 -> greedy (paper default). A positive
        temperature switches to seeded sampling (used only as the R1 degeneration
        fallback, per the plan's recommended temperature 0.6 with a fixed seed);
        top-k logprobs are still recorded per step for branch enumeration."""
        input_ids = torch.tensor([prompt_ids], device=self.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens, output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.eos_ids,
        )
        if temperature and temperature > 0.0:
            torch.manual_seed(seed)
            gen_kwargs.update(do_sample=True, temperature=float(temperature),
                              top_p=1.0, top_k=0)
        else:
            gen_kwargs.update(do_sample=False, num_beams=1)
        out = self.model.generate(input_ids, **gen_kwargs)
        seq = out.sequences[0].tolist()
        gen_ids = seq[len(prompt_ids):]
        topk_ids, topk_lps = [], []
        for step_scores in out.scores:  # each (1, vocab) pre-softmax logits
            lp = torch.log_softmax(step_scores[0].float(), dim=-1)
            vals, idx = lp.topk(top_k)
            topk_ids.append(idx.tolist())
            topk_lps.append(vals.tolist())
        # trim trailing eos from gen_ids but keep alignment with scores
        finish = "stop" if (gen_ids and gen_ids[-1] in self.eos_ids) else "length"
        # drop a single trailing eos token (and its score row) for clean analysis
        if gen_ids and gen_ids[-1] in self.eos_ids:
            gen_ids = gen_ids[:-1]
            topk_ids = topk_ids[:len(gen_ids)]
            topk_lps = topk_lps[:len(gen_ids)]
        return BasePath(prompt_ids=list(prompt_ids), gen_ids=gen_ids,
                        topk_ids=topk_ids, topk_logprobs=topk_lps, finish_reason=finish)

    # ------------------------------------------------------- plain resampling
    @torch.no_grad()
    def resample(self, prefixes, n, max_tokens, temperature, seed=0):
        """prefixes: list[list[int]]. Returns (list-per-prefix of n continuation
        token-id lists, wall_seconds). Batched, left-padded generation."""
        t0 = time.time()
        results = [None] * len(prefixes)
        # chunk so chunk_size * n <= gen_batch
        per = max(1, self.gen_batch // n)
        order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
        for s in range(0, len(order), per):
            batch_idx = order[s:s + per]
            batch = [prefixes[i] for i in batch_idx]
            maxlen = max(len(p) for p in batch)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.full((len(batch), maxlen), pad, device=self.device, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), device=self.device, dtype=torch.long)
            for r, p in enumerate(batch):
                input_ids[r, maxlen - len(p):] = torch.tensor(p, device=self.device)
                attn[r, maxlen - len(p):] = 1
            torch.manual_seed(seed + s)
            out = self.model.generate(
                input_ids, attention_mask=attn, do_sample=True,
                temperature=temperature, top_p=1.0, top_k=0,
                num_return_sequences=n, max_new_tokens=max_tokens,
                pad_token_id=pad, eos_token_id=self.eos_ids,
            )
            gen = out[:, maxlen:]  # (len(batch)*n, gen_len)
            gen = gen.view(len(batch), n, -1)
            for r, i in enumerate(batch_idx):
                conts = []
                for k in range(n):
                    row = gen[r, k].tolist()
                    conts.append(self._strip(row))
                results[i] = conts
        return results, time.time() - t0

    def _strip(self, ids):
        out = []
        for t in ids:
            if t in self.eos_ids:
                break
            out.append(t)
        return out

    # ------------------------------------------------- batched raw generation
    @torch.no_grad()
    def _generate_batch(self, prefixes, n, max_tokens, temperature,
                        stop_strings=None, seed=0):
        """Draw n continuations for each prefix. Prefix-length-ordered batching so
        left-padding waste is minimized. Returns list-per-prefix of n stripped
        continuation token-id lists (EOS/stop-string trimmed)."""
        results = [None] * len(prefixes)
        per = max(1, self.gen_batch // max(1, n))
        order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
        pad = self.tokenizer.pad_token_id
        gen_kwargs = {}
        if stop_strings:
            gen_kwargs["stop_strings"] = list(stop_strings)
            gen_kwargs["tokenizer"] = self.tokenizer
        for s in range(0, len(order), per):
            batch_idx = order[s:s + per]
            batch = [prefixes[i] for i in batch_idx]
            maxlen = max(len(p) for p in batch)
            input_ids = torch.full((len(batch), maxlen), pad, device=self.device, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), device=self.device, dtype=torch.long)
            for r, p in enumerate(batch):
                input_ids[r, maxlen - len(p):] = torch.tensor(p, device=self.device)
                attn[r, maxlen - len(p):] = 1
            torch.manual_seed(seed + s)
            out = self.model.generate(
                input_ids, attention_mask=attn, do_sample=True,
                temperature=temperature, top_p=1.0, top_k=0,
                num_return_sequences=n, max_new_tokens=max_tokens,
                pad_token_id=pad, eos_token_id=self.eos_ids, **gen_kwargs,
            )
            gen = out[:, maxlen:].view(len(batch), n, -1)
            for r, i in enumerate(batch_idx):
                results[i] = [self._strip(gen[r, k].tolist()) for k in range(n)]
        return results

    # ------------------------------------------------- adaptive-S resampling
    @torch.no_grad()
    def resample_adaptive(self, branches, k_min, k_step, s_max, wilson_tol,
                          max_tokens, temperature, stop_strings=None, seed=0):
        """Round-based adaptive resampling with continuation early-stop.

        Round 1 draws k_min continuations for every branch in prefix-ordered
        batches; later rounds draw k_step more only for branches whose outcome is
        not yet decided (see forking_paths.adaptive.outcome_decided), capped at
        s_max. The stop decision uses a fast regex-only category proxy.

        Returns (conts_by_branch, stats) where conts_by_branch[i] is the list of
        drawn continuation token-id lists for branches[i], and stats has
        n_drawn (per branch), total_samples, total_gen_tokens.
        """
        from .adaptive import outcome_decided
        from .answers import parse_mmlu_answer

        conts = [[] for _ in branches]
        active = list(range(len(branches)))
        target = k_min
        rnd = 0
        while active:
            need = target - min(len(conts[i]) for i in active) if active else 0
            need = max(1, need)
            prefixes = [branches[i].prefix_ids for i in active]
            drawn = self._generate_batch(prefixes, n=need, max_tokens=max_tokens,
                                         temperature=temperature,
                                         stop_strings=stop_strings,
                                         seed=seed + 1000 * rnd)
            still = []
            for j, i in enumerate(active):
                conts[i].extend(drawn[j][: s_max - len(conts[i])])
                cats = [parse_mmlu_answer(self.tokenizer.decode(c, skip_special_tokens=True))
                        for c in conts[i]]
                if not outcome_decided(cats, k_min=k_min, s_max=s_max,
                                       wilson_tol=wilson_tol):
                    still.append(i)
            active = still
            target = min(s_max, target + k_step)
            rnd += 1
        n_drawn = [len(c) for c in conts]
        total_gen_tokens = int(sum(len(c) for cc in conts for c in cc))
        stats = {
            "n_drawn": n_drawn,
            "total_samples": int(sum(n_drawn)),
            "total_gen_tokens": total_gen_tokens,
            "rounds": rnd,
        }
        return conts, stats

    # ------------------------------------------- KV-cache prefix-reuse resampling
    @torch.no_grad()
    def resample_kv_reuse(self, base, branches_by_pos, n, max_tokens, temperature, seed=0):
        """Efficiency path: compute the base-path KV cache once, then for each
        position t crop it to (len(prompt)+t) and reuse it for every branch at t.

        base: BasePath. branches_by_pos: dict[t] -> list[Branch].
        Returns (dict[id(branch)] -> list of n continuations, wall_seconds).
        """
        from transformers import DynamicCache

        t0 = time.time()
        full_ids = torch.tensor([base.prompt_ids + base.gen_ids], device=self.device)
        base_out = self.model(full_ids, use_cache=True)
        base_cache = base_out.past_key_values  # cache for prompt+gen
        Lp = len(base.prompt_ids)

        results = {}
        for t, branches in branches_by_pos.items():
            crop_len = Lp + t  # reuse prompt + gen[:t]
            for b in branches:
                cache = _crop_cache(base_cache, crop_len, n)
                w = torch.full((n, 1), b.tok_id, device=self.device, dtype=torch.long)
                attn = torch.ones((n, crop_len + 1), device=self.device, dtype=torch.long)
                torch.manual_seed(seed + t)
                out = self.model.generate(
                    input_ids=w, attention_mask=attn, past_key_values=cache,
                    do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                    num_return_sequences=1, max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.eos_ids,
                )
                gen = out[:, 1:]  # drop the forced token position
                results[id(b)] = [self._strip(gen[k].tolist()) for k in range(n)]
        return results, time.time() - t0

    # ------------------------------------------------- logit-read answer fallback
    @torch.no_grad()
    def logit_read_letters(self, prefixes, seed=0):
        from .answers import VALID
        # candidate token ids for A/B/C/D (the letter immediately after "(")
        cand = {}
        for L in VALID:
            for tid in self.tokenizer(L, add_special_tokens=False)["input_ids"][:1]:
                cand.setdefault(L, tid)
        letters = []
        pad = self.tokenizer.pad_token_id
        cand_ids = torch.tensor([cand[L] for L in VALID], device=self.device)
        # Cap the read batch so the lm_head over long-context prefixes can't OOM
        # (row-94-style long passages produce ~1300-token prefixes).
        read_batch = min(self.gen_batch, 96)
        for s in range(0, len(prefixes), read_batch):
            batch = prefixes[s:s + read_batch]
            maxlen = max(len(p) for p in batch)
            input_ids = torch.full((len(batch), maxlen), pad, device=self.device, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), device=self.device, dtype=torch.long)
            for r, p in enumerate(batch):
                input_ids[r, maxlen - len(p):] = torch.tensor(p, device=self.device)
                attn[r, maxlen - len(p):] = 1
            # logits_to_keep=1: only compute the last-position logits (the whole
            # lm_head over every position is a 56GB tensor at batch 256 / len 1300).
            logits = self.model(input_ids, attention_mask=attn,
                                logits_to_keep=1).logits[:, -1, :]
            sub = logits[:, cand_ids]  # (B, 4)
            best = sub.argmax(dim=-1).tolist()
            letters.extend(VALID[b] for b in best)
        return letters

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def _crop_cache(cache, length, batch):
    """Crop a DynamicCache to `length` positions and expand to `batch` rows."""
    from transformers import DynamicCache

    new = DynamicCache()
    for layer_idx in range(len(cache)):
        k = cache.layers[layer_idx].keys if hasattr(cache, "layers") else cache.key_cache[layer_idx]
        v = cache.layers[layer_idx].values if hasattr(cache, "layers") else cache.value_cache[layer_idx]
        k = k[:, :, :length, :].expand(batch, -1, -1, -1).contiguous()
        v = v[:, :, :length, :].expand(batch, -1, -1, -1).contiguous()
        new.update(k, v, layer_idx)
    return new
