# Forking Fast

Efficient estimation of uncertainty dynamics in language and reasoning
models. During a single greedy response, a model's answer distribution
`o_t` — "if generation restarted from token position t, how would the final
answer be distributed?" — can drift smoothly or jump abruptly at *forking*
tokens. The original Forking Paths Analysis (Bigelow et al.,
[arXiv:2412.07961](https://arxiv.org/abs/2412.07961)) measures this by brute
force. This repository contains everything behind the paper *Forking Fast:
Efficiently Estimating Uncertainty Dynamics in Language and Reasoning
Models*: a prefix-cached sampler that makes collection cheap
(`forking_paths/`), reconstruction models that recover the full curve from a
reduced run at a fraction of the token budget (`otrecon/`), the released
outcome dataset (100 tinyMMLU questions × 2 models, `data/`), and an
interactive explorer (`dashboard/`).

## Open the explorer first

The fastest way to see what this project does:

1. Download [`dashboard/dashboard_standalone.html.zip`](dashboard/dashboard_standalone.html.zip) (6.4 MB).
2. Unzip it.
3. Double-click `dashboard_standalone.html`.

It opens directly in your browser — **fully offline, no server, no network
requests** (the 52 MB HTML file embeds plotly.js and all data). You can
browse all 100 tinyMMLU questions for both models: the reference outcome
curve from all 200 recorded samples per position, the smoothed
reconstruction from a reduced run (any S and observation spacing on the
grid), the same reduced run before smoothing, per-token hover over the
model's actual response, correct-answer chips, a generated-token cost chip
for the selected budget, and reference forks vs. PELT change points as
toggleable markers.

## Repository layout

| Path | What it is |
| --- | --- |
| `forking_paths/` | GPU sampling package: prefix-cached base-path decode, branch enumeration, batched resampling, outcome extraction, store schema. Plus the S=1000 merge/analysis utilities (`src/`) and CPU unit tests. |
| `otrecon/` | CPU reconstruction package: smoothing models (raw counts, segment pooling, the segment-kernel Full Model and hybrids), cross-validated tuning, TV/log-likelihood metrics, the replicate fan, and the power-analysis driver (`src/run_scale.py`). 63 unit tests. |
| `dashboard/` | The standalone explorer (zip of the exact released revision) and the three builder scripts that regenerate it from `data/`. |
| `data/` | The released outcome stores + `loader.py` + `MANIFEST.json` (per-file SHA-256, coverage, schema). |
| `scripts/verify_release.py` | End-to-end release verification (see below). |

## Install

Python 3.10+. The analysis stack is pure CPU (numpy / scipy / ruptures, all
prebuilt wheels):

```bash
pip install ./otrecon ./forking_paths
```

Collecting *new* samples needs a GPU stack (the released dataset was
collected with HuggingFace transformers on A100-class GPUs; analysis of the
released stores never needs it):

```bash
pip install './forking_paths[gpu]'
```

## Quickstart: sampling (GPU)

The sampler takes a JSON list of MMLU-style question dicts and writes one
compact outcome store per question (the format under `data/`):

```bash
python -m forking_paths.run \
    --model meta-llama/Meta-Llama-3-8B-Instruct --mode instruct \
    --questions questions.json --out-dir stores/ \
    --smoke            # tiny settings for a wiring check; drop for real runs
```

Defaults (S=200 draws per branch, top-k branch enumeration, temperature,
seeds) live in `forking_paths/config.py` (`ForkingConfig`); every recorded
store also embeds its exact collection config.

## Quickstart: smooth one question (CPU)

Reconstruct a question's outcome curve from a reduced run — here S=30 draws
at every 4th token instead of the full 200 draws at every token:

```python
import sys
import numpy as np
sys.path.insert(0, "data")
import loader
from otrecon import data as od
from otrecon.models import MODEL_REGISTRY

rec = loader.load_store("data/s200/llama/row039.json.gz")
idxs, gt = od.weighted_o_t(rec)                    # full-pool reference curve
cats, idx_map = od.collapse_map(gt, rec["categories"])
pos, draws, _ = od.mixture_draws(rec, idx_map, len(cats),
                                 n_total=200, seed_base=43_000_000)

S, stride = 30, 4                                  # the reduced-run budget
obs = np.arange(0, len(pos), stride)
counts = od.counts_from_draws(draws[obs], len(cats), 0, S)

model = MODEL_REGISTRY["M5a_segkernel"]()          # the Full Model
model.fit(np.asarray(pos, float)[obs], counts, S,
          {"variant": "mult", "pen": 64.0, "h": 32.0})
recon = model.predict(np.asarray(pos, float))      # smoothed o_t, full grid
```

Cross-validated per-question tuning instead of fixed hyperparameters:
`otrecon.cv.cv_select`. The full replicate-fan power analysis (gate → fan →
assemble) is driven by `otrecon/src/run_scale.py` (see its docstring).

## Dataset

`data/` ships **outcome stores only** — the per-draw answer categories and
everything needed to reconstruct outcome curves. Raw completion text is not
part of the release.

- `data/s200/{llama,deepseek}/rowNNN.json.gz` — tinyMMLU rows 0–99, S=200
  draws per branch. `llama` = Meta-Llama-3-8B-Instruct on an every-token
  grid; `deepseek` = DeepSeek-R1-Distill-Llama-8B on a sentence grid.
- `data/s1000/llama/row039_stride4.json.gz` — row 39, S=1000, stride-4 grid
  (an independent development run).
- `data/s1000/llama/row039_everytoken.json.gz` — row 39, S=1000, every-token
  grid; its first 200 draws are exactly the S=200 store (nested prefixes).
  The development runs also produced a row-12 stride-4 S=1000 store, but its
  only copies were lost before release assembly, so it is not included.

Each record stores, per observed position `t`, the kept next-token branches
(token id, probability, and the recorded outcome category of each resampled
rollout in draw order) plus the greedy base path and the recorded full-pool
curve `o_t_full`. `data/loader.py` is the reference loading code (numpy
only):

```python
import sys; sys.path.insert(0, "data")
import loader
rec = loader.load_store("data/s200/deepseek/row007.json.gz")
idxs, curve = loader.o_t(rec)          # recompute the weighted o_t curve
assert loader.matches_recorded(rec)    # reproduces rec["o_t_full"] exactly
```

`data/MANIFEST.json` records per-file SHA-256 and identity fields, the
coverage summary, and a reference curve hash;
`scripts/make_data_manifest.py` regenerates it.

## Rebuilding the dashboard

The shipped zip is the exact released revision. To rebuild an equivalent
file from the released data (CPU; `pip install plotly transformers` on top
of the packages):

```bash
# 1. per-combination reconstructions for every question / S / spacing
python otrecon/src/run_scale.py dashboard --data-dir data/s200 \
    --rows-list $(seq -s, 0 99) --out payloads/

# 2. per-token display strings for the hover panel (downloads tokenizers)
python dashboard/extract_tokens.py --out payloads/tokens_meta.json

# 3. PELT change points per combination (re-fits and cross-checks step 1)
python dashboard/extract_changepoints.py \
    --llama payloads/dashboard_llama.json \
    --deepseek payloads/dashboard_deepseek.json \
    --out payloads/cps_meta.json

# 4. assemble the single self-contained HTML file
python dashboard/build_dashboard_standalone.py \
    --llama payloads/dashboard_llama.json \
    --deepseek payloads/dashboard_deepseek.json \
    --tokens payloads/tokens_meta.json \
    --changepoints payloads/cps_meta.json \
    --out dashboard_standalone.html
```

Step 1 is the expensive one (a few CPU-hours across all 100 questions ×
both models; it parallelizes over `--workers`).

## Verifying the release

From the repository root, in a fresh virtual environment:

```bash
python scripts/verify_release.py
```

This installs both packages, runs the full `otrecon` test suite and the
CPU `forking_paths` tests, verifies every data file against
`MANIFEST.json` (including an exact loader round-trip of a recorded
curve), checks the dashboard zip is byte-identical to the released
revision with no external network references, and checks tree hygiene.

## Citation

If you use this code or dataset, please cite *Forking Fast: Efficiently
Estimating Uncertainty Dynamics in Language and Reasoning Models*, and the
original Forking Paths Analysis: Bigelow et al., *Forking Paths in Neural
Text Generation*, [arXiv:2412.07961](https://arxiv.org/abs/2412.07961).
Questions are from tinyMMLU (tinyBenchmarks); models are
Meta-Llama-3-8B-Instruct and DeepSeek-R1-Distill-Llama-8B.
