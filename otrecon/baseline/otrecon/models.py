"""Reconstruction models M0-M4 behind one fit/predict interface.

All models consume observed counts c (n_obs, K) at token positions
``obs_tok`` (subset of the full grid ``all_tok``), with per-position sample
size S, and produce a probability curve at every grid position plus a
central credible band.

Interface:
    m = MODEL_REGISTRY[name]()
    m.fit(obs_tok, counts, S, params)      # params from the model's grid()
    p = m.predict(all_tok)                 # (T, K) rows sum to 1
    lo, hi = m.credible_band(all_tok, 0.9) # (T, K) each

M1/M2 are one family (KernelDirichlet): the discounted forward-backward
recursion with decay gamma per token equals a two-sided exponential kernel
with h = -1/ln(gamma), so the CV picks kernel in {gaussian, exponential} and
bandwidth h. M3 is a hand-rolled per-category local-level Kalman smoother
(exact heteroskedastic known observation variance; MLE process variance).
M4 segments with ruptures (PELT, custom multinomial cost) and pools counts
per segment into a Dirichlet posterior.
"""
from __future__ import annotations

import numpy as np
from ruptures.base import BaseCost
from scipy import optimize, stats
from scipy.special import expit, logit


def _dirichlet_band(alpha: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Marginal Beta central band for each category of Dirichlet rows."""
    a0 = alpha.sum(axis=1, keepdims=True)
    a = alpha
    b = a0 - a
    q = (1.0 - level) / 2.0
    lo = stats.beta.ppf(q, a, np.maximum(b, 1e-12))
    hi = stats.beta.ppf(1.0 - q, a, np.maximum(b, 1e-12))
    return lo, hi


def _interp_rows(x_new: np.ndarray, x_obs: np.ndarray, y: np.ndarray,
                 kind: str) -> np.ndarray:
    """Row-wise 1d interpolation of (n_obs, K) onto x_new."""
    out = np.zeros((len(x_new), y.shape[1]))
    if kind == "nearest":
        j = np.abs(x_new[:, None] - x_obs[None, :]).argmin(axis=1)
        return y[j]
    for k in range(y.shape[1]):
        out[:, k] = np.interp(x_new, x_obs, y[:, k])
    return out


class BaseModel:
    name = "base"

    @staticmethod
    def grid() -> list[dict]:
        return [{}]

    def fit(self, obs_tok, counts, S, params):  # pragma: no cover
        raise NotImplementedError


class RawInterp(BaseModel):
    """M0: c/S at observed positions, nearest/linear interpolation between.
    Band: Dirichlet(alpha0 + c) marginals, interpolated."""
    name = "M0_raw"

    @staticmethod
    def grid():
        return [{"kind": "linear"}, {"kind": "nearest"}]

    def fit(self, obs_tok, counts, S, params):
        self.obs_tok = np.asarray(obs_tok, float)
        K = counts.shape[1]
        self.p_obs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
        self.alpha = counts + 1.0 / K
        self.kind = params.get("kind", "linear")

    def predict(self, all_tok):
        return _interp_rows(np.asarray(all_tok, float), self.obs_tok,
                            self.p_obs, self.kind)

    def credible_band(self, all_tok, level=0.9):
        lo, hi = _dirichlet_band(self.alpha, level)
        x = np.asarray(all_tok, float)
        return (_interp_rows(x, self.obs_tok, lo, self.kind),
                _interp_rows(x, self.obs_tok, hi, self.kind))


class KernelDirichlet(BaseModel):
    """M1/M2 family: alpha(t) = alpha0 + sum_s K_h(|t - s|) c_s."""
    name = "M1_kernel"

    @staticmethod
    def grid():
        hs = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
        return [{"kernel": k, "h": h} for k in ("gaussian", "exponential")
                for h in hs]

    def fit(self, obs_tok, counts, S, params):
        self.obs_tok = np.asarray(obs_tok, float)
        self.counts = np.asarray(counts, float)
        self.K = counts.shape[1]
        self.kernel = params["kernel"]
        self.h = float(params["h"])

    def _alpha(self, all_tok):
        d = np.abs(np.asarray(all_tok, float)[:, None] - self.obs_tok[None, :])
        if self.kernel == "gaussian":
            w = np.exp(-0.5 * (d / self.h) ** 2)
        else:
            w = np.exp(-d / self.h)
        return 1.0 / self.K + w @ self.counts

    def predict(self, all_tok):
        a = self._alpha(all_tok)
        return a / a.sum(axis=1, keepdims=True)

    def credible_band(self, all_tok, level=0.9):
        return _dirichlet_band(self._alpha(all_tok), level)


class KernelDirichletGaussian(KernelDirichlet):
    """M1: Gaussian kernel (bandwidth tuned by CV)."""
    name = "M1_kernel"

    @staticmethod
    def grid():
        return [{"kernel": "gaussian", "h": h}
                for h in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)]


class KernelDirichletExp(KernelDirichlet):
    """M2: two-sided exponential kernel == discounted-counts fwd/bwd with
    gamma = exp(-1/h) per token."""
    name = "M2_discount"

    @staticmethod
    def grid():
        return [{"kernel": "exponential", "h": h}
                for h in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)]


class KalmanLogit(BaseModel):
    """M3: per-category local-level Kalman smoother on empirical logits.

    y_tk = logit((c_tk + 1/2) / (S + 1)); R_tk = 1 / (S * ph * (1 - ph)),
    ph = (c_tk + 1/2) / (S + 1). State evolves with variance q per token.
    q is fit by marginal MLE (shared across categories). Point estimate:
    expit(smoothed mean), renormalized. Band: expit(m +/- z*sd) per
    category (not renormalized; coverage table judges the approximation).
    """
    name = "M3_kalman"

    @staticmethod
    def grid():
        return [{}]

    def fit(self, obs_tok, counts, S, params):
        obs_tok = np.asarray(obs_tok, float)
        counts = np.asarray(counts, float)
        self.K = counts.shape[1]
        Svec = counts.sum(axis=1)
        ph = (counts + 0.5) / (Svec[:, None] + 1.0)
        self.y = logit(ph)
        self.R = 1.0 / np.maximum(Svec[:, None] * ph * (1 - ph), 1e-9)
        self.obs_tok = obs_tok

        def nll(log_q):
            q = float(np.exp(log_q))
            tot = 0.0
            for k in range(self.K):
                tot -= _ll_local_level(obs_tok, self.y[:, k], self.R[:, k], q)
            return tot

        res = optimize.minimize_scalar(nll, bounds=(np.log(1e-8), np.log(10.0)),
                                       method="bounded")
        self.q = float(np.exp(res.x))

    def _smooth(self, all_tok):
        all_tok = np.asarray(all_tok, float)
        m = np.zeros((len(all_tok), self.K))
        v = np.zeros((len(all_tok), self.K))
        for k in range(self.K):
            m[:, k], v[:, k] = _smooth_local_level(
                self.obs_tok, self.y[:, k], self.R[:, k], self.q, all_tok)
        return m, v

    def predict(self, all_tok):
        m, _ = self._smooth(all_tok)
        p = expit(m)
        return p / p.sum(axis=1, keepdims=True)

    def credible_band(self, all_tok, level=0.9):
        m, v = self._smooth(all_tok)
        z = stats.norm.ppf(0.5 + level / 2.0)
        sd = np.sqrt(np.maximum(v, 0.0))
        return expit(m - z * sd), expit(m + z * sd)


def _ll_local_level(x, y, R, q):
    """Exact log marginal likelihood of a local-level model observed at
    irregular positions x (process variance q per unit distance), with a
    diffuse initial state (first observation conditioned on)."""
    ll = 0.0
    m, P = y[0], R[0]  # diffuse init: absorb first obs
    for i in range(1, len(x)):
        P = P + q * (x[i] - x[i - 1])
        F = P + R[i]
        e = y[i] - m
        ll += -0.5 * (np.log(2 * np.pi * F) + e * e / F)
        Kg = P / F
        m = m + Kg * e
        P = P * (1 - Kg)
    return ll


def _smooth_local_level(x_obs, y, R, q, x_all):
    """RTS smoother on the union grid of x_obs and x_all; observations only
    at x_obs. Returns smoothed mean/var at x_all."""
    x_union = np.unique(np.concatenate([np.asarray(x_obs, float),
                                        np.asarray(x_all, float)]))
    obs_at = {float(t): i for i, t in enumerate(x_obs)}
    n = len(x_union)
    m_f = np.zeros(n)
    P_f = np.zeros(n)  # filtered
    m_p = np.zeros(n)
    P_p = np.zeros(n)  # predicted
    big = 1e6
    m, P = 0.0, big
    for i, t in enumerate(x_union):
        if i > 0:
            P = P + q * (x_union[i] - x_union[i - 1])
        m_p[i], P_p[i] = m, P
        j = obs_at.get(float(t))
        if j is not None:
            F = P + R[j]
            Kg = P / F
            m = m + Kg * (y[j] - m)
            P = P * (1 - Kg)
        m_f[i], P_f[i] = m, P
    # RTS backward pass
    m_s = m_f.copy()
    P_s = P_f.copy()
    for i in range(n - 2, -1, -1):
        if P_p[i + 1] <= 0:
            continue
        C = P_f[i] / P_p[i + 1]
        m_s[i] = m_f[i] + C * (m_s[i + 1] - m_p[i + 1])
        P_s[i] = P_f[i] + C * C * (P_s[i + 1] - P_p[i + 1])
    pos = {float(t): i for i, t in enumerate(x_union)}
    sel = np.array([pos[float(t)] for t in np.asarray(x_all, float)])
    return m_s[sel], P_s[sel]


class MultinomialCost(BaseCost):
    """ruptures custom cost: segment cost = -sum_k n_k log(n_k / n).

    NOTE: must subclass ``BaseCost`` — ``rpt.Pelt(custom_cost=...)``
    SILENTLY falls back to CostL2 for non-BaseCost objects, so the
    original M4 actually segmented with an L2 cost on raw counts. The 'l2'
    detection variant in ``segment_positions`` preserves that exact legacy
    behavior for reproduction/side-by-side.
    """
    model = "multinomial"
    min_size = 2

    def fit(self, signal):
        self.signal = np.asarray(signal, float)
        self.cum = np.vstack([np.zeros(signal.shape[1]),
                              np.cumsum(self.signal, axis=0)])
        return self

    def error(self, start, end):
        n_k = self.cum[end] - self.cum[start]
        n = n_k.sum()
        if n <= 0:
            return 0.0
        nz = n_k[n_k > 0]
        return float(-(nz * np.log(nz / n)).sum())

    def sum_of_costs(self, bkps):
        s = 0
        tot = 0.0
        for e in bkps:
            tot += self.error(s, e)
            s = e
        return tot


class SegmentPool(BaseModel):
    """M4: PELT segmentation with multinomial cost, per-segment Dirichlet."""
    name = "M4_segment"

    @staticmethod
    def grid():
        return [{"pen": p} for p in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
                                      128.0, 256.0, 512.0)]

    def fit(self, obs_tok, counts, S, params):
        counts = np.asarray(counts, float)
        self.obs_tok = np.asarray(obs_tok, float)
        self.K = counts.shape[1]
        n = len(obs_tok)
        if n < 4:
            bkps = [n]
        else:
            bkps = segment_positions(self.obs_tok, counts,
                                     params.get("cost", "mult"),
                                     float(params["pen"]))
        self.bkps = bkps
        self.seg_alpha = []
        self.seg_bounds_tok = []  # (lo_tok, hi_tok] boundaries at midpoints
        s = 0
        for e in bkps:
            pooled = counts[s:e].sum(axis=0)
            self.seg_alpha.append(1.0 / self.K + pooled)
            lo = -np.inf if s == 0 else 0.5 * (self.obs_tok[s - 1] + self.obs_tok[s])
            hi = np.inf if e == n else 0.5 * (self.obs_tok[e - 1] + self.obs_tok[e])
            self.seg_bounds_tok.append((lo, hi))
            s = e
        self.seg_alpha = np.array(self.seg_alpha)

    def _seg_of(self, all_tok):
        all_tok = np.asarray(all_tok, float)
        seg = np.zeros(len(all_tok), dtype=int)
        for i, (lo, hi) in enumerate(self.seg_bounds_tok):
            seg[(all_tok > lo) & (all_tok <= hi)] = i
        return seg

    def predict(self, all_tok):
        seg = self._seg_of(all_tok)
        a = self.seg_alpha[seg]
        return a / a.sum(axis=1, keepdims=True)

    def credible_band(self, all_tok, level=0.9):
        seg = self._seg_of(all_tok)
        lo, hi = _dirichlet_band(self.seg_alpha, level)
        return lo[seg], hi[seg]


MODEL_REGISTRY = {
    "M0_raw": RawInterp,
    "M1_kernel": KernelDirichletGaussian,
    "M2_discount": KernelDirichletExp,
    "M3_kalman": KalmanLogit,
    "M4_segment": SegmentPool,
}

# Simplicity order for the pre-registered selection rule (Q1).
SIMPLICITY_ORDER = ["M0_raw", "M1_kernel", "M2_discount", "M3_kalman", "M4_segment"]


# ---------------------------------------------------------------------------
# Drift-aware segment smoothing hybrids.
# ---------------------------------------------------------------------------

def _wls_polyfit(x: np.ndarray, y: np.ndarray, w: np.ndarray, deg: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares polyfit with KNOWN observation variance 1/w.

    Returns (beta ascending powers, cov) where cov = (X' W X)^{-1} is the
    exact coefficient covariance under known heteroskedastic variance.
    """
    X = np.vander(np.asarray(x, float), deg + 1, increasing=True)
    XtW = X.T * np.asarray(w, float)
    A = XtW @ X
    A_inv = np.linalg.pinv(A)
    beta = A_inv @ (XtW @ np.asarray(y, float))
    return beta, A_inv


def _empirical_logits(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(y, w): per-category empirical logits and known-variance weights.

    y_tk = logit((c_tk + 1/2) / (S_t + 1)); w_tk = 1 / Var(y_tk)
    = S_t * ph * (1 - ph) by the delta method.
    """
    counts = np.asarray(counts, float)
    Svec = counts.sum(axis=1)
    ph = (counts + 0.5) / (Svec[:, None] + 1.0)
    y = logit(ph)
    w = np.maximum(Svec[:, None] * ph * (1.0 - ph), 1e-9)
    return y, w


class LogitLinearCost(BaseCost):
    """ruptures custom cost: sum over categories of the weighted RSS of a
    degree-1 fit of empirical logits on token position.

    Signal columns: [x, y_1..y_K, w_1..w_K]. All segment statistics come
    from cumulative sums, so ``error`` is O(K).
    """
    model = "logit_linear"
    min_size = 3

    def fit(self, signal):
        sig = np.asarray(signal, float)
        self.K = (sig.shape[1] - 1) // 2
        x = sig[:, 0]
        y = sig[:, 1:1 + self.K]
        w = sig[:, 1 + self.K:1 + 2 * self.K]
        z = np.zeros((1, self.K))

        def cum(a):
            return np.vstack([z, np.cumsum(a, axis=0)])

        self.c_w = cum(w)
        self.c_wx = cum(w * x[:, None])
        self.c_wxx = cum(w * (x * x)[:, None])
        self.c_wy = cum(w * y)
        self.c_wxy = cum(w * x[:, None] * y)
        self.c_wyy = cum(w * y * y)
        self.signal = sig
        return self

    def error(self, start, end):
        Sw = self.c_w[end] - self.c_w[start]
        Sx = self.c_wx[end] - self.c_wx[start]
        Sxx = self.c_wxx[end] - self.c_wxx[start]
        Sy = self.c_wy[end] - self.c_wy[start]
        Sxy = self.c_wxy[end] - self.c_wxy[start]
        Syy = self.c_wyy[end] - self.c_wyy[start]
        den = Sw * Sxx - Sx * Sx
        safe = np.where(np.abs(den) > 1e-12, den, 1.0)
        b1 = np.where(np.abs(den) > 1e-12, (Sw * Sxy - Sx * Sy) / safe, 0.0)
        b0 = (Sy - b1 * Sx) / np.maximum(Sw, 1e-12)
        rss = Syy - b0 * Sy - b1 * Sxy
        return float(np.maximum(rss, 0.0).sum())

    def sum_of_costs(self, bkps):
        s = 0
        tot = 0.0
        for e in bkps:
            tot += self.error(s, e)
            s = e
        return tot


_SEG_CACHE: dict = {}


def segment_positions(obs_tok: np.ndarray, counts: np.ndarray,
                      variant: str, pen: float) -> list[int]:
    """PELT breakpoints (ruptures convention: end indices, last == n) under
    detection ``variant``: 'mult' (multinomial cost on counts) or 'linear'
    (LogitLinearCost on the empirical logit series), or 'l2' (CostL2 on raw
    counts — the original M4 behavior via the silent fallback).
    Memoized: the CV grid reuses segmentations across smoothing
    hyperparameters."""
    import ruptures as rpt
    obs_tok = np.asarray(obs_tok, float)
    counts = np.asarray(counts, float)
    key = (obs_tok.tobytes(), counts.tobytes(), variant, float(pen))
    hit = _SEG_CACHE.get(key)
    if hit is not None:
        return hit
    n = len(obs_tok)
    min_size = 2 if variant == "mult" else 3
    if n < 2 * min_size:
        bkps = [n]
    elif variant == "mult":
        algo = rpt.Pelt(custom_cost=MultinomialCost(), min_size=2,
                        jump=1).fit(counts)
        bkps = algo.predict(pen=float(pen))
    elif variant == "linear":
        y, w = _empirical_logits(counts)
        sig = np.hstack([obs_tok[:, None], y, w])
        algo = rpt.Pelt(custom_cost=LogitLinearCost(), min_size=3,
                        jump=1).fit(sig)
        bkps = algo.predict(pen=float(pen))
    elif variant == "l2":
        algo = rpt.Pelt(model="l2", min_size=2, jump=1).fit(counts)
        bkps = algo.predict(pen=float(pen))
    else:
        raise ValueError(f"unknown detection variant {variant!r}")
    if len(_SEG_CACHE) > 20000:
        _SEG_CACHE.clear()
    _SEG_CACHE[key] = bkps
    return bkps


class _SegmentedModel(BaseModel):
    """Shared segmentation plumbing (midpoint token boundaries, as M4)."""

    def _set_segments(self, obs_tok, counts, variant, pen):
        self.obs_tok = np.asarray(obs_tok, float)
        n = len(self.obs_tok)
        self.bkps = segment_positions(self.obs_tok, counts, variant, pen)
        self.seg_slices = []
        self.seg_bounds_tok = []
        s = 0
        for e in self.bkps:
            lo = -np.inf if s == 0 else 0.5 * (self.obs_tok[s - 1] + self.obs_tok[s])
            hi = np.inf if e == n else 0.5 * (self.obs_tok[e - 1] + self.obs_tok[e])
            self.seg_slices.append((s, e))
            self.seg_bounds_tok.append((lo, hi))
            s = e

    def _seg_of(self, all_tok):
        all_tok = np.asarray(all_tok, float)
        seg = np.zeros(len(all_tok), dtype=int)
        for i, (lo, hi) in enumerate(self.seg_bounds_tok):
            seg[(all_tok > lo) & (all_tok <= hi)] = i
        return seg

    @property
    def n_segments(self):
        return len(self.seg_slices)


PEN_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
            1024.0, 2048.0)


class SegmentKernelPool(_SegmentedModel):
    """M5a: PELT segmentation + kernel-weighted Dirichlet pooling truncated
    at the detected changepoints (SegmentPool x KernelDirichlet hybrid)."""
    name = "M5a_segkernel"

    @staticmethod
    def grid():
        hs = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
        return [{"variant": v, "pen": p, "h": h}
                for v in ("mult", "linear") for p in PEN_GRID for h in hs]

    def fit(self, obs_tok, counts, S, params):
        self.counts = np.asarray(counts, float)
        self.K = self.counts.shape[1]
        self.h = float(params["h"])
        self._set_segments(obs_tok, self.counts, params["variant"],
                           params["pen"])

    def _alpha(self, all_tok):
        all_tok = np.asarray(all_tok, float)
        seg = self._seg_of(all_tok)
        alpha = np.full((len(all_tok), self.K), 1.0 / self.K)
        for i, (s, e) in enumerate(self.seg_slices):
            mask = seg == i
            if not mask.any():
                continue
            d = np.abs(all_tok[mask][:, None] - self.obs_tok[s:e][None, :])
            w = np.exp(-0.5 * (d / self.h) ** 2)
            alpha[mask] += w @ self.counts[s:e]
        return alpha

    def predict(self, all_tok):
        a = self._alpha(all_tok)
        return a / a.sum(axis=1, keepdims=True)

    def credible_band(self, all_tok, level=0.9):
        return _dirichlet_band(self._alpha(all_tok), level)


def _binom_glm_polyfit(xc, c, s_tot, deg, n_iter=50, tol=1e-10
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Binomial GLM (logit link) polynomial fit by IRLS.

    c: category counts (n,), s_tot: per-position totals (n,). Returns
    (beta ascending powers, Fisher covariance (X' W X)^{-1}).

    Unlike WLS on empirical logits, the likelihood fit has no
    attenuation bias from noise-correlated weights, and its degree-0 case
    is exactly the pooled proportion (matching count pooling).
    """
    xc = np.asarray(xc, float)
    c = np.asarray(c, float)
    s_tot = np.asarray(s_tot, float)
    X = np.vander(xc, deg + 1, increasing=True)
    ph = (c + 0.5) / (s_tot + 1.0)
    beta, _ = _wls_polyfit(xc, logit(ph), s_tot * ph * (1 - ph), deg)
    for _ in range(n_iter):
        eta = np.clip(X @ beta, -9.0, 9.0)
        mu = expit(eta)
        w = np.maximum(s_tot * mu * (1.0 - mu), 1e-9)
        z = eta + (c - s_tot * mu) / w
        new, cov = _wls_polyfit(xc, z, w, deg)
        step = float(np.max(np.abs(new - beta)))
        beta = new
        if step < tol:
            break
    eta = np.clip(X @ beta, -9.0, 9.0)
    mu = expit(eta)
    w = np.maximum(s_tot * mu * (1.0 - mu), 1e-9)
    XtW = X.T * w
    cov = np.linalg.pinv(XtW @ X)
    return beta, cov


def _fit_seg_category(xc, c, s_tot, max_deg, z_thresh=3.0):
    """Binomial-GLM polyfit with backward degree selection: starting at
    max_deg (capped by the segment's point count), drop the top coefficient
    while it is not significant at |z| >= z_thresh (Wald, Fisher SE). On
    flat segments this shrinks to the pooled constant — the guard against
    trend fits chasing multinomial noise. Returns (beta, cov, deg)."""
    deg = int(min(max_deg, max(len(xc) - 1, 0)))
    while deg > 0:
        beta, cov = _binom_glm_polyfit(xc, c, s_tot, deg)
        se = float(np.sqrt(max(cov[deg, deg], 0.0)))
        if se > 0 and abs(beta[deg] / se) >= z_thresh:
            return beta, cov, deg
        deg -= 1
    beta, cov = _binom_glm_polyfit(xc, c, s_tot, 0)
    return beta, cov, 0


class SegmentPolyLogit(_SegmentedModel):
    """M5b: PELT segmentation + per-segment weighted polynomial regression
    on empirical logits, with backward degree selection per segment and
    category (top coefficient kept only when |z| >= 2, so flat segments
    shrink to constants); expit + renormalize across categories (M3's
    inverse-transform convention). Bands by sampling coefficient draws from
    the exact WLS covariance of the selected fit through the same
    transform. Reports per-segment slopes +/- SE."""
    name = "M5b_segpoly"
    N_BAND_DRAWS = 500
    Z_THRESH = 3.0  # per-test Wald threshold for keeping a trend coefficient

    @staticmethod
    def grid():
        return [{"variant": v, "pen": p, "deg": d}
                for v in ("mult", "linear") for p in PEN_GRID for d in (1, 2)]

    def fit(self, obs_tok, counts, S, params):
        counts = np.asarray(counts, float)
        self.K = counts.shape[1]
        self.deg = int(params["deg"])
        self._set_segments(obs_tok, counts, params["variant"], params["pen"])
        s_tot = counts.sum(axis=1)
        self.seg_fits = []
        for (s, e) in self.seg_slices:
            x = self.obs_tok[s:e]
            center = float(x.mean())
            xc = x - center
            fits = []
            for k in range(self.K):
                beta, cov, deg_k = _fit_seg_category(
                    xc, counts[s:e, k], s_tot[s:e], self.deg, self.Z_THRESH)
                fits.append({"beta": beta, "cov": cov, "deg": deg_k})
            self.seg_fits.append({"center": center, "fits": fits})

    def _logit_mean(self, all_tok):
        all_tok = np.asarray(all_tok, float)
        seg = self._seg_of(all_tok)
        m = np.zeros((len(all_tok), self.K))
        for i, sf in enumerate(self.seg_fits):
            mask = seg == i
            if not mask.any():
                continue
            xc = all_tok[mask] - sf["center"]
            for k, fk in enumerate(sf["fits"]):
                X = np.vander(xc, fk["deg"] + 1, increasing=True)
                m[mask, k] = X @ fk["beta"]
        return m

    def predict(self, all_tok):
        p = expit(self._logit_mean(all_tok))
        return p / p.sum(axis=1, keepdims=True)

    def credible_band(self, all_tok, level=0.9):
        all_tok = np.asarray(all_tok, float)
        seg = self._seg_of(all_tok)
        q = (1.0 - level) / 2.0
        lo = np.zeros((len(all_tok), self.K))
        hi = np.zeros((len(all_tok), self.K))
        rng = np.random.default_rng(44_000_001)
        for i, sf in enumerate(self.seg_fits):
            mask = seg == i
            if not mask.any():
                continue
            xc = all_tok[mask] - sf["center"]
            draws = np.zeros((self.N_BAND_DRAWS, int(mask.sum()), self.K))
            for k, fk in enumerate(sf["fits"]):
                X = np.vander(xc, fk["deg"] + 1, increasing=True)
                b = rng.multivariate_normal(fk["beta"], fk["cov"],
                                            size=self.N_BAND_DRAWS,
                                            method="svd")
                draws[:, :, k] = b @ X.T
            p = expit(draws)
            p = p / p.sum(axis=2, keepdims=True)
            lo[mask] = np.quantile(p, q, axis=0)
            hi[mask] = np.quantile(p, 1.0 - q, axis=0)
        return lo, hi

    def slope_report(self):
        """Per-segment, per-category slope of the SELECTED fit +/- SE (zero
        slope when the segment-category shrank to a constant), plus the
        selected degree."""
        out = []
        for (s, e), sf in zip(self.seg_slices, self.seg_fits):
            slopes, ses, zs, degs = [], [], [], []
            for fk in sf["fits"]:
                degs.append(fk["deg"])
                if fk["deg"] >= 1:
                    sl = float(fk["beta"][1])
                    se = float(np.sqrt(max(fk["cov"][1, 1], 0.0)))
                else:
                    sl, se = 0.0, float("nan")
                slopes.append(sl)
                ses.append(se)
                zs.append(sl / se if se and np.isfinite(se) and se > 0
                          else 0.0)
            out.append({
                "tok_lo": float(self.obs_tok[s]),
                "tok_hi": float(self.obs_tok[e - 1]),
                "n_obs": int(e - s), "deg_selected": degs,
                "slope": slopes, "slope_se": ses, "slope_z": zs,
            })
        return out


class SegmentPoolL2(SegmentPool):
    """M4 exactly as originally run: PELT with the (fallback) L2 cost
    on raw counts. Kept for reproduction and cost-fix side-by-side."""
    name = "M4l2_segment"

    @staticmethod
    def grid():
        return [{"pen": p, "cost": "l2"}
                for p in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
                          256.0, 512.0)]


# Recommended default for dense sampling regimes
# (S >= 30, observation stride <= 2). At S=10 or coarse strides the
# hybrid is not uniformly better than M1/M4 (see grid_region_summary).
DEFAULT_MODEL = "M5a_segkernel"

MODEL_REGISTRY["M4l2_segment"] = SegmentPoolL2
MODEL_REGISTRY["M5a_segkernel"] = SegmentKernelPool
MODEL_REGISTRY["M5b_segpoly"] = SegmentPolyLogit
SIMPLICITY_ORDER.extend(["M5a_segkernel", "M5b_segpoly"])
