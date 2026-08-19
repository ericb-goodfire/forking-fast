"""Exact fast reimplementation of the ruptures-1.1.10 PELT path behind
``segment_positions``: identical breakpoints, a fraction of the time.

Bit-exactness contract
----------------------
The reproduction gates (scale fan_v3, ablation fan) require identical
stored metrics, which requires identical breakpoint choices and therefore a
replication of ruptures' arithmetic rather than an "equivalent" algorithm.

DP recursion (ruptures ``Pelt._seg`` with ``jump=1``):

* ruptures compares candidate partitions by ``sum(partition.values())``
  where each partition dict holds one ``error(s, e) + pen`` float per
  segment (plus an initial exact 0). Dicts are extended left-to-right and
  ``sum`` adds values in insertion order, so the scalar recursion
  ``F[bkp] = F[t*] + (error(t*, bkp) + pen)`` reproduces every compared sum
  operation-for-operation (one rounding for ``error + pen``, one add).
* Tie-break: ``min(subproblems, ...)`` returns the FIRST minimum in
  admissible order; we scan with strict ``<``.
* Pruning keeps ``total_t <= best + pen`` (same ``<=``, same scalar).
* ruptures appends one admissible point per step (``bkp - min_size``).
  Points with 0 < t < min_size have no partition (KeyError) and are dropped
  the same iteration by the ``zip`` truncation in the pruning step; we never
  add them. The admissible sequences are otherwise identical.

Segment costs are computed with the same numpy expressions on the same
operands as the vendored cost classes:

* mult   -- all-pairs vectorization of ``MultinomialCost.error`` from the
  identical prefix-sum array. Per-pair sums run over the K <= 5 categories
  (numpy sums sequentially below length 8 on both sides); zero-count
  categories contribute an exact ``0.0`` term (x + 0.0 == x for every
  partial sum arising here, all-zero-term sums are +0.0 on both sides).
  The only transcendental is ``np.log`` evaluated on large arrays here vs
  length-<=K arrays in ruptures; elementwise shape-independence is verified
  at runtime by ``vectorized_log_ok()`` and the per-pair reference path is
  used if that check ever fails.
* linear -- all-pairs vectorization of ``LogitLinearCost.error`` from the
  identical cumsum arrays; only elementwise IEEE ops (+,-,*,/, where,
  maximum), so vectorization is exact by construction.
* l2     -- per-pair ``signal[s:e].var(axis=0).sum() * (e - s)`` exactly as
  ruptures ``CostL2`` (np.var's pairwise summation is segment-length
  dependent and not safely vectorizable); all pairs are computed once and
  shared across penalties.

The numba-JIT DP (numba is baked into the job-core image) performs only
float64 add/compare (no transcendentals, fastmath off), so it is bit-exact
with the pure-numpy fallback; set ``OTRECON_NO_NUMBA=1`` to force numpy.
"""
from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Runtime self-check: is np.log elementwise-deterministic across shapes?
# ---------------------------------------------------------------------------

_VECLOG_OK: bool | None = None


def vectorized_log_ok() -> bool:
    """True iff np.log on large arrays is bit-identical to np.log on the
    tiny (length <= 5) arrays ruptures' MultinomialCost.error produces,
    over a sample of the integer-count-ratio domain used by the costs."""
    global _VECLOG_OK
    if _VECLOG_OK is None:
        rng = np.random.default_rng(20260803)
        num = rng.integers(1, 70_001, size=200_000).astype(float)
        den = rng.integers(1, 70_001, size=200_000).astype(float)
        r = num / np.maximum(num, den)          # ratios in (0, 1]
        big = np.log(r)
        ok = True
        for ln in (1, 2, 3, 4, 5):              # reference nz lengths
            for off in range(0, 50_000, 617):
                sub = r[off:off + ln].copy()
                if not np.array_equal(np.log(sub), big[off:off + ln]):
                    ok = False
                    break
            if not ok:
                break
        _VECLOG_OK = bool(ok)
    return _VECLOG_OK


# ---------------------------------------------------------------------------
# Cost matrices. C[s, e] = segment cost error(s, e) for e - s >= min_size.
# Entries outside that band are never read by the DP.
# ---------------------------------------------------------------------------

_CHUNK = 64  # end-column block size to bound broadcast temporaries


def mult_cost_matrix(counts: np.ndarray) -> np.ndarray:
    """All-pairs MultinomialCost.error, bit-identical per entry."""
    counts = np.asarray(counts, float)
    n, K = counts.shape
    # identical prefix-sum construction to MultinomialCost.fit
    cum = np.vstack([np.zeros(K), np.cumsum(counts, axis=0)])
    if not vectorized_log_ok():                  # pragma: no cover
        return _mult_cost_matrix_exact(cum, n)
    C = np.zeros((n + 1, n + 1))
    for e0 in range(0, n + 1, _CHUNK):
        e1 = min(e0 + _CHUNK, n + 1)
        D = cum[None, e0:e1, :] - cum[:, None, :]        # (n+1, B, K)
        ntot = D.sum(axis=2)
        pos = D > 0
        safe_n = np.where(ntot > 0, ntot, 1.0)
        ratio = np.where(pos, D, 1.0) / safe_n[:, :, None]
        terms = np.where(pos, D * np.log(ratio), 0.0)
        block = -terms.sum(axis=2)
        block[ntot <= 0] = 0.0
        C[:, e0:e1] = block
    return C


def _mult_cost_matrix_exact(cum: np.ndarray, n: int) -> np.ndarray:
    """Per-pair fallback with reference-shaped operands (used only if
    vectorized_log_ok() fails on this runtime)."""
    C = np.zeros((n + 1, n + 1))
    for s in range(n + 1):
        for e in range(s + 1, n + 1):
            n_k = cum[e] - cum[s]
            tot = n_k.sum()
            if tot <= 0:
                continue
            nz = n_k[n_k > 0]
            C[s, e] = float(-(nz * np.log(nz / tot)).sum())
    return C


def linear_cost_matrix(x: np.ndarray, y: np.ndarray, w: np.ndarray
                       ) -> np.ndarray:
    """All-pairs LogitLinearCost.error, bit-identical per entry.

    x: (n,) token positions; y, w: (n, K) empirical logits and weights --
    the same arrays segment_positions feeds LogitLinearCost via hstack.
    Only elementwise IEEE ops, so vectorization is exact by construction.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    n, K = y.shape
    z = np.zeros((1, K))

    def cum(a):
        return np.vstack([z, np.cumsum(a, axis=0)])

    c_w = cum(w)
    c_wx = cum(w * x[:, None])
    c_wxx = cum(w * (x * x)[:, None])
    c_wy = cum(w * y)
    c_wxy = cum(w * x[:, None] * y)
    c_wyy = cum(w * y * y)

    C = np.zeros((n + 1, n + 1))
    for e0 in range(0, n + 1, _CHUNK):
        e1 = min(e0 + _CHUNK, n + 1)

        def seg(c):
            return c[None, e0:e1, :] - c[:, None, :]

        Sw, Sx, Sxx = seg(c_w), seg(c_wx), seg(c_wxx)
        Sy, Sxy, Syy = seg(c_wy), seg(c_wxy), seg(c_wyy)
        den = Sw * Sxx - Sx * Sx
        big = np.abs(den) > 1e-12
        safe = np.where(big, den, 1.0)
        b1 = np.where(big, (Sw * Sxy - Sx * Sy) / safe, 0.0)
        b0 = (Sy - b1 * Sx) / np.maximum(Sw, 1e-12)
        rss = Syy - b0 * Sy - b1 * Sxy
        C[:, e0:e1] = np.maximum(rss, 0.0).sum(axis=2)
    return C


def l2_cost_matrix(counts: np.ndarray, min_size: int = 2) -> np.ndarray:
    """All-pairs CostL2.error via the reference per-pair expression
    (np.var call per segment), computed once and shared across penalties."""
    sig = np.asarray(counts, float)
    if sig.ndim == 1:                            # CostL2.fit reshape
        sig = sig.reshape(-1, 1)
    n = len(sig)
    C = np.zeros((n + 1, n + 1))
    for s in range(n):
        for e in range(s + min_size, n + 1):
            C[s, e] = sig[s:e].var(axis=0).sum() * (e - s)
    return C


# ---------------------------------------------------------------------------
# The PELT DP, exactly as ruptures 1.1.10 with jump=1.
# ---------------------------------------------------------------------------

def _pelt_core_py(C, n, pen, min_size):
    F = np.zeros(n + 1)
    prev = np.zeros(n + 1, dtype=np.int64)
    adm = np.zeros(n + 2, dtype=np.int64)
    vals = np.zeros(n + 2)
    n_adm = 0
    for bkp in range(min_size, n + 1):
        t_new = bkp - min_size
        if t_new == 0 or t_new >= min_size:
            adm[n_adm] = t_new
            n_adm += 1
        best = np.inf
        best_i = 0
        for i in range(n_adm):
            t = adm[i]
            v = F[t] + (C[t, bkp] + pen)
            vals[i] = v
            if v < best:
                best = v
                best_i = i
        F[bkp] = best
        prev[bkp] = adm[best_i]
        thresh = best + pen
        m = 0
        for i in range(n_adm):
            if vals[i] <= thresh:
                adm[m] = adm[i]
                m += 1
        n_adm = m
    ends = np.zeros(n + 1, dtype=np.int64)
    k = 0
    t = n
    while t > 0:
        ends[k] = t
        k += 1
        t = prev[t]
    out = np.zeros(k, dtype=np.int64)
    for i in range(k):
        out[i] = ends[k - 1 - i]
    return out


_pelt_jit = None
_jit_tried = False


def _get_pelt_impl():
    global _pelt_jit, _jit_tried
    if os.environ.get("OTRECON_NO_NUMBA"):
        return _pelt_core_py
    if not _jit_tried:
        _jit_tried = True
        try:
            from numba import njit
            _pelt_jit = njit(cache=True)(_pelt_core_py)
            # trigger + verify compilation on a trivial input
            _pelt_jit(np.zeros((3, 3)), 2, 1.0, 1)
        except Exception:                        # pragma: no cover
            _pelt_jit = None
    return _pelt_jit if _pelt_jit is not None else _pelt_core_py


def pelt_breakpoints(C: np.ndarray, n: int, pen: float, min_size: int
                     ) -> list[int]:
    """ruptures-identical breakpoints (sorted end indices, last == n)."""
    impl = _get_pelt_impl()
    ends = impl(np.ascontiguousarray(C, dtype=np.float64), int(n),
                float(pen), int(min_size))
    return [int(e) for e in ends]
