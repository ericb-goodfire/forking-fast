"""Configuration for Forking Paths Analysis.

All parameters here mirror Bigelow et al. (arXiv 2412.07961) and the authors'
reference implementation (github.com/ebigelow/forking-paths). Where a value
differs from the paper it is marked in the comment with its provenance:
  [paper]  = stated in the paper
  [repo]   = taken from the authors' reference code / analysis.ipynb
  [default]= our documented choice for the local reimplementation
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ForkingConfig:
    # ---- sampling pipeline ----
    top_k: int = 10               # [paper] k<=10 alternate tokens per position
    p_thresh: float = 0.05        # [paper] keep alternates with prob >= 5% (+ greedy always)
    n_samples: int = 20           # [paper] S; paper used 30, notes 10-20 ~ 30 (App). plan: 20
    n0_samples: int = 100         # [repo/plan] extra resamples at t=0 for a stable o_0 (paper N=300)
    base_max_tokens: int = 400    # [default] greedy base-path length cap (paper: 1200 tok_depth)
    cont_max_tokens: int = 400    # [default] continuation length cap (paper: 512)
    tok_depth: int = 400          # [paper/default] max response positions t analyzed
    base_temperature: float = 0.0 # [paper] greedy base path
    cont_temperature: float = 1.0 # [paper] temperature 1.0 for resampled continuations
    seed: int = 0                 # [repo] mcmc_seed=0; also used for sampling reproducibility

    # ---- outcome / answer ----
    # Fixed categorical outcome vector for MMLU: A,B,C,D,Other (one-hot -> histogram)
    outcome_categories: tuple = ("A", "B", "C", "D", "Other")

    # ---- semantic drift ----
    drift_metric: str = "l2"      # [paper] L2 distance d(o_0, o_t)

    # ---- BEAST change-point detection ----
    # Values below match the authors' authoritative analysis cell (analysis.ipynb
    # cell 321), which produced the paper's Table 1 numbers.
    beast_noise: float = 0.03     # [repo] jitter = np.random.normal(0, .03), clip [0,1]
    beast_prec_value: float = 20.0  # [repo] precValue=20 (constant precision prior)
    beast_alpha1: float = 0.01    # [repo] alpha1=.01
    beast_alpha2_base: float = 2.0  # [repo] rescale_alpha2 base for d_l2
    beast_alpha2_exp: float = 1000.0  # [repo] rescale_alpha2 exp base
    beast_tcp_max: int = 6        # [repo] tcp_minmax=[0,6]
    beast_torder_min: int = 0     # [repo] torder_minmax=[0,1]
    beast_torder_max: int = 1     # [repo] torder_minmax=[0,1]
    beast_mcmc_chains: int = 10   # [repo] mcmc_chains=10
    cp_bayes_factor: float = 9.0  # [paper] Bayes factor p(m>=1)/p(m=0) > 9
    cp_ncp_quantile: float = 0.1  # [paper] .1 quantile of p(m|y) point estimate

    # ---- survival analysis ----
    survival_epsilon: float = 0.6           # [paper] headline eps for S(T), L2
    survival_eps_sweep: tuple = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # [plan] sweep

    def to_dict(self):
        return asdict(self)
