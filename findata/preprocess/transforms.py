"""Composable processing transforms for (ticker, date) feature panels.

Each transform is sklearn-flavored (fit on train rows, transform anything) but
panel-aware: per-ticker operations never bleed across ticker boundaries and
per-date (cross-sectional) operations only use same-date information.

Chains are described by :class:`ProcessingConfig` (an ordered list of step
names + params, compiled through ``STEP_REGISTRY``) and executed by
:class:`ProcessingPipeline`, which slices the training rows once and
fit/transforms each stage in sequence.

Causality: every transform here is causal except :class:`SSADenoiser`
(batch SVD over the full series — diagnostic only). A pipeline's ``causal``
flag is the AND of its stages and is enforced at ``PipelineState.apply`` time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from findata.preprocess import rmt
from findata.preprocess.base import ScaleRule, date_values, get_scaler


# ---------------------------------------------------------------------------
# Base + shared helpers
# ---------------------------------------------------------------------------

class PanelTransform:
    causal: bool = True

    def fit(self, train_X: pd.DataFrame) -> "PanelTransform":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, train_X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_X).transform(train_X)

    def describe(self) -> dict:
        return {"class": type(self).__name__, "causal": self.causal}


def _panel_q(X: pd.DataFrame) -> float:
    """q = T/N with T = unique dates (pooled rows overstate the sample —
    a day's cross-section of correlated names is far from independent)."""
    return X.index.get_level_values("date").nunique() / X.shape[1]


def _standardize_fit(values: np.ndarray):
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    return mean, np.where(std > 1e-12, std, 1.0)


def _apply_complete_rows(values: np.ndarray, n_out: int, fn) -> np.ndarray:
    """Apply a matrix op to rows without NaNs; NaN rows stay NaN (they appear
    mid-chain, e.g. rolling-vol heads, and are dropped later by the analysis)."""
    out = np.full((values.shape[0], n_out), np.nan)
    ok = ~np.isnan(values).any(axis=1)
    if ok.any():
        out[ok] = fn(values[ok])
    return out


# ---------------------------------------------------------------------------
# Stage 0: scaling
# ---------------------------------------------------------------------------

class GroupScaler(PanelTransform):
    """Applies each group's ScaleRules (fit on train rows only).

    Declarative replacement for v1's per-class ``_scale`` overrides; columns not
    covered by any rule pass through unchanged.
    """

    def __init__(self, rules: list[ScaleRule]):
        self.rules = [r for r in rules if r.columns]

    def fit(self, train_X: pd.DataFrame) -> "GroupScaler":
        self.fitted_ = []
        for rule in self.rules:
            cols = [c for c in rule.columns if c in train_X.columns]
            if not cols:
                continue
            scaler = None
            if rule.kind == "scaler":
                scaler = get_scaler(rule.scaler).fit(train_X[cols].to_numpy())
            self.fitted_.append((rule, cols, scaler))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for rule, cols, scaler in self.fitted_:
            if rule.kind == "scaler":
                vals = scaler.transform(X[cols].to_numpy())
                if rule.arcsinh:
                    vals = np.clip(np.arcsinh(vals), -3.5, 3.5)
                out[cols] = vals
            elif rule.kind == "passthrough":
                out[cols] = X[cols]
            else:  # affine
                out[cols] = X[cols] * rule.mul + rule.add
        uncovered = [c for c in X.columns if c not in out.columns]
        if uncovered:
            print("Uncovered columns passing through:", uncovered)
            out[uncovered] = X[uncovered]
        return out[list(X.columns)]

    def describe(self) -> dict:
        return {"class": "GroupScaler",
                "rules": [(r.kind, r.scaler if r.kind == "scaler" else None, len(cols))
                          for r, cols, _ in getattr(self, "fitted_", [])]}


class Standardizer(PanelTransform):
    """Per-column (x - mean_train) / std_train — used mid-chain to re-standardize
    (e.g. after market-mode removal) so the next MP fit sees unit-variance inputs."""

    def fit(self, train_X: pd.DataFrame) -> "Standardizer":
        self.mean_ = train_X.mean()
        std = train_X.std(ddof=1)
        self.std_ = std.where(std > 1e-12, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.mean_) / self.std_


# ---------------------------------------------------------------------------
# Temporal denoising
# ---------------------------------------------------------------------------

class KalmanDenoiser(PanelTransform):
    """Causal steady-state local-level Kalman filter.

    The steady-state filter for the local-level model is an EWMA with gain K
    solving K^2 + lam*K - lam = 0, lam = sigma2_eta / sigma2_eps (signal-to-noise).
    Gains are fit PER FEATURE from pooled per-ticker first differences on the
    train rows via method-of-moments on the MA(1) structure of the differences:

        var(d)   = sigma2_eta + 2*sigma2_eps
        cov1(d)  = -sigma2_eps

    then applied as a vectorized per-ticker EWMA — no per-series MLE needed.
    ``estimator="mle"`` cross-checks lam with statsmodels' UnobservedComponents
    on a few series per feature (slow; for validation, not production).
    """
    causal = True

    def __init__(self, estimator: str = "moments", gain_floor: float = 0.02,
                 gain_cap: float = 1.0, mle_max_series: int = 5):
        if estimator not in ("moments", "mle"):
            raise ValueError("estimator must be 'moments' or 'mle'")
        self.estimator = estimator
        self.gain_floor = gain_floor
        self.gain_cap = gain_cap
        self.mle_max_series = mle_max_series

    @staticmethod
    def _gain_from_lambda(lam):
        return (-lam + np.sqrt(lam ** 2 + 4 * lam)) / 2.0

    def _moments_lambda(self, train_X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        d = train_X.groupby(level="ticker").diff()
        d1 = d.groupby(level="ticker").shift(1)
        cov1 = ((d - d.mean()) * (d1 - d1.mean())).mean()
        var_d = d.var(ddof=1)
        sigma_eps = (-cov1).clip(lower=0.0)
        sigma_eta = (var_d + 2 * cov1).clip(lower=0.0)
        lam = sigma_eta / sigma_eps.where(sigma_eps > 1e-15)  # NaN -> no obs noise
        return lam, sigma_eps

    def fit(self, train_X: pd.DataFrame) -> "KalmanDenoiser":
        lam, _ = self._moments_lambda(train_X)
        if self.estimator == "mle":
            lam = self.mle_lambdas(train_X).combine_first(lam)
        gains = self._gain_from_lambda(lam).clip(self.gain_floor, self.gain_cap)
        # sigma_eps ~ 0 (pure random walk, no measurement noise) -> identity
        self.gains_ = gains.fillna(1.0)
        return self

    def mle_lambdas(self, train_X: pd.DataFrame, columns=None) -> pd.Series:
        """Median MLE signal-to-noise per feature over the longest per-ticker
        series (statsmodels local-level). For cross-checking the moments fit."""
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        columns = list(columns) if columns is not None else list(train_X.columns)
        sizes = train_X.groupby(level="ticker").size().sort_values(ascending=False)
        tickers = sizes.index[:self.mle_max_series]
        out = {}
        for col in columns:
            lams = []
            for tkr in tickers:
                y = train_X.loc[tkr, col].dropna().to_numpy()
                if len(y) < 100:
                    continue
                try:
                    fit = UnobservedComponents(y, level="local level").fit(disp=False)
                    sigma_eps, sigma_eta = fit.params[0], fit.params[1]
                    if sigma_eps > 1e-15:
                        lams.append(sigma_eta / sigma_eps)
                except Exception:
                    continue
            out[col] = float(np.median(lams)) if lams else np.nan
        return pd.Series(out)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            a = float(self.gains_[col])
            if a >= 0.999:
                out[col] = X[col]
            else:
                out[col] = (X[col].groupby(level="ticker", group_keys=False)
                            .apply(lambda s, a=a: s.ewm(alpha=a, adjust=False).mean()))
        return pd.DataFrame(out, index=X.index)[list(X.columns)]

    def describe(self) -> dict:
        d = super().describe()
        if hasattr(self, "gains_"):
            d["gain_median"] = float(self.gains_.median())
            d["gain_range"] = [float(self.gains_.min()), float(self.gains_.max())]
        return d


class SSADenoiser(PanelTransform):
    """Singular Spectrum Analysis reconstruction per (ticker, column) series.

    NON-CAUSAL: the trajectory-matrix SVD uses the full series, so every point
    sees the future. Keep for diagnostics/upper-bound comparisons only.
    Efficient form: rank-r via eigh of the L x L Gram matrix + bincount-based
    diagonal averaging (no per-element Python loops).
    """
    causal = False

    def __init__(self, window_length: int = 40, n_components: int = 3):
        self.window_length = window_length
        self.n_components = n_components

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        L, r = self.window_length, self.n_components

        def _ssa_series(y: np.ndarray) -> np.ndarray:
            n = len(y)
            if n < 2 * L:
                return y
            K = n - L + 1
            traj = np.lib.stride_tricks.sliding_window_view(y, L).T  # (L, K)
            gram = traj @ traj.T
            _, U = np.linalg.eigh(gram)
            Ur = U[:, -r:]                       # top-r left singular vectors
            M = Ur @ (Ur.T @ traj)               # rank-r reconstruction
            idx = np.add.outer(np.arange(L), np.arange(K)).ravel()
            sums = np.bincount(idx, weights=M.ravel(), minlength=n)
            counts = np.bincount(idx, minlength=n)
            return sums / counts

        def _ssa_frame(df: pd.DataFrame) -> pd.DataFrame:
            vals = df.to_numpy(dtype=float, copy=True)
            for j in range(vals.shape[1]):
                y = vals[:, j]
                nan = np.isnan(y)
                if nan.all():
                    continue
                start = int(np.argmax(~nan))     # first valid
                seg = y[start:]
                if np.isnan(seg).any():          # interior NaNs: leave column as-is
                    continue
                vals[start:, j] = _ssa_series(seg)
            return pd.DataFrame(vals, index=df.index, columns=df.columns)

        return X.groupby(level="ticker", group_keys=False).apply(_ssa_frame)

    def describe(self) -> dict:
        return {**super().describe(), "window_length": self.window_length,
                "n_components": self.n_components}


# ---------------------------------------------------------------------------
# Cross-sectional (market-mode) normalization
# ---------------------------------------------------------------------------

class CrossSectionalNormalizer(PanelTransform):
    """Per-date market-mode removal / dispersion normalization.

    Motivation: the pooled panel concentrates into very few signal eigenvalues —
    whole-market flow. Removing the day-by-day cross-sectional mean strips that
    market component so finer (e.g. reversal) structure can clear a fresh MP
    threshold after re-standardizing.

    methods (all causal — only same-date or trailing information):
      demean : x - cross-sectional mean(date)               (market level removed)
      zscore : (x - cs_mean(date)) / cs_std(date)           (level + dispersion regime)
      vol_cs : x / cs_std(date)                             (vol-based alternative to demeaning)
      vol_ts : x / rolling std per ticker (vol_window)      (per-name vol normalization)

    NOTE on rank IC: demean/zscore/vol_cs shift/scale every ticker identically
    within a date, so within-date RANKS — and therefore per-feature rank IC —
    are provably unchanged (a standalone 'demean' variant matching 'raw+none'
    exactly is a correctness check, not a null result). Their value shows up in
    the correlation SPECTRUM and hence in whatever decorrelation step (zca /
    hpca / signal projection) runs after them. vol_ts divides each ticker by
    its own trailing vol and DOES reorder the cross-section.
    """
    causal = True
    METHODS = ("demean", "zscore", "vol_cs", "vol_ts")

    def __init__(self, method: str = "demean", vol_window: int = 20,
                 min_names: int = 5, eps: float = 1e-8):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        self.method = method
        self.vol_window = vol_window
        self.min_names = min_names
        self.eps = eps

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "vol_ts":
            vol = (X.groupby(level="ticker", group_keys=False)
                   .apply(lambda df: df.rolling(self.vol_window,
                                                min_periods=max(2, self.vol_window // 2)).std()))
            return X / (vol + self.eps)
        g = X.groupby(level="date")
        counts = g.transform("count")
        if self.method == "demean":
            out = X - g.transform("mean")
        elif self.method == "zscore":
            out = (X - g.transform("mean")) / (g.transform("std") + self.eps)
        else:  # vol_cs
            out = X / (g.transform("std") + self.eps)
        return out.where(counts >= self.min_names)

    def describe(self) -> dict:
        d = {**super().describe(), "method": self.method, "min_names": self.min_names}
        if self.method == "vol_ts":
            d["vol_window"] = self.vol_window
        return d


class SignalProjector(PanelTransform):
    """Project onto (or strip) the MP-significant signal subspace, in the
    original feature basis (name-preserving).

    mode="keep"  : X_std @ E_s E_s'   — keep only signal components
    mode="strip" : X_std (I - E_s E_s') — remove them (e.g. drop market mode)
    n_components="mp" uses the fitted MP edge; an int overrides.
    """
    causal = True

    def __init__(self, mode: str = "keep", n_components: int | str = "mp",
                 q: float | None = None, bw: float = 0.1):
        if mode not in ("keep", "strip"):
            raise ValueError("mode must be 'keep' or 'strip'")
        self.mode = mode
        self.n_components = n_components
        self.q = q
        self.bw = bw

    def fit(self, train_X: pd.DataFrame) -> "SignalProjector":
        train_X = train_X.dropna()
        self.input_cols_ = list(train_X.columns)
        vals = train_X.to_numpy(dtype=float)
        self.mean_, self.std_ = _standardize_fit(vals)
        Xs = (vals - self.mean_) / self.std_
        corr = np.corrcoef(Xs, rowvar=False)
        w, E = np.linalg.eigh(corr)
        order = w.argsort()[::-1]
        w, E = w[order], E[:, order]
        q = self.q if self.q is not None else _panel_q(train_X)
        lam_plus, n_signal, noise_var = rmt.mp_edge(w, q, self.bw)
        n = max(n_signal, 1) if self.n_components == "mp" else int(self.n_components)
        Es = E[:, :n]
        proj = Es @ Es.T
        self.proj_ = proj if self.mode == "keep" else np.eye(len(w)) - proj
        self.n_components_ = n
        self.lam_plus_ = float(lam_plus)
        self.noise_var_ = float(noise_var)
        self.top_eig_share_ = float(w[0] / w.sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        vals = X[self.input_cols_].to_numpy(dtype=float)
        out = _apply_complete_rows(vals, len(self.input_cols_),
                                   lambda v: ((v - self.mean_) / self.std_) @ self.proj_)
        return pd.DataFrame(out, index=X.index, columns=self.input_cols_)

    def describe(self) -> dict:
        d = {**super().describe(), "mode": self.mode}
        if hasattr(self, "n_components_"):
            d.update(n_components=self.n_components_, lam_plus=self.lam_plus_,
                     top_eig_share=self.top_eig_share_)
        return d


# ---------------------------------------------------------------------------
# Decorrelation / whitening
# ---------------------------------------------------------------------------

class PCADecorrelator(PanelTransform):
    """Plain PCA on the pooled train rows; emits pc1..pck."""
    causal = True

    def __init__(self, n_components=0.95, whiten: bool = True):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, train_X: pd.DataFrame) -> "PCADecorrelator":
        from sklearn.decomposition import PCA
        train_X = train_X.dropna()
        self.input_cols_ = list(train_X.columns)
        self.pca_ = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca_.fit(train_X.to_numpy(dtype=float))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        vals = X[self.input_cols_].to_numpy(dtype=float)
        k = self.pca_.n_components_
        out = _apply_complete_rows(vals, k, self.pca_.transform)
        return pd.DataFrame(out, index=X.index, columns=[f"pc{i + 1}" for i in range(k)])

    def describe(self) -> dict:
        d = {**super().describe(), "whiten": self.whiten}
        if hasattr(self, "pca_"):
            d["n_components"] = int(self.pca_.n_components_)
        return d


class ZCAWhitener(PanelTransform):
    """ZCA (correlation) whitening: W = E diag(lam^-1/2) E'.

    Output columns KEEP their original names — the whitened basis stays
    maximally close to the input basis, which is what makes per-feature rank IC
    after whitening interpretable (unlike PCA components). Optionally denoise
    the correlation matrix (RMT) before inverting.

    Rank deficiency: engineered panels contain EXACT linear combinations (e.g.
    momentum short-minus-long contrasts), so the correlation matrix is singular.
    Whitening uses a pseudo-inverse — eigendirections below ``rcond * lam_max``
    are zeroed, not inflated (a naive eps-clip amplifies null-space noise by
    ~1/sqrt(eps) and wrecks the output). Those directions carry no independent
    information, so columns involved in exact combos stay partially correlated
    after whitening — that is the honest best a linear map can do.
    """
    causal = True

    def __init__(self, eps: float = 1e-6, rcond: float = 1e-8,
                 rmt_denoise: bool = False, q: float | None = None, bw: float = 0.1):
        self.eps = eps
        self.rcond = rcond
        self.rmt_denoise = rmt_denoise
        self.q = q
        self.bw = bw

    def fit(self, train_X: pd.DataFrame) -> "ZCAWhitener":
        train_X = train_X.dropna()
        self.input_cols_ = list(train_X.columns)
        vals = train_X.to_numpy(dtype=float)
        self.mean_, self.std_ = _standardize_fit(vals)
        Xs = (vals - self.mean_) / self.std_
        corr = np.corrcoef(Xs, rowvar=False)
        if self.rmt_denoise:
            q = self.q if self.q is not None else _panel_q(train_X)
            corr = rmt.denoise_correlation(corr, q=q, bw=self.bw)
        w, E = np.linalg.eigh(corr)
        cutoff = max(self.eps, self.rcond * float(w.max()))
        w_isqrt = np.where(w > cutoff, np.clip(w, cutoff, None) ** -0.5, 0.0)
        self.rank_ = int((w > cutoff).sum())
        self.W_ = E @ np.diag(w_isqrt) @ E.T
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        vals = X[self.input_cols_].to_numpy(dtype=float)
        out = _apply_complete_rows(vals, len(self.input_cols_),
                                   lambda v: ((v - self.mean_) / self.std_) @ self.W_)
        return pd.DataFrame(out, index=X.index, columns=self.input_cols_)

    def describe(self) -> dict:
        d = {**super().describe(), "rmt_denoise": self.rmt_denoise, "eps": self.eps}
        if hasattr(self, "rank_"):
            d["rank"] = self.rank_
        return d


class HierarchicalPCA(PanelTransform):
    """Grouped PCA where the groups come from the data, not a hand taxonomy.

    Fit: standardize train rows -> correlation -> RMT denoise (default on) ->
    hierarchical clustering (rmt.cluster_corr, silhouette auto-k) -> one
    whitened PCA per cluster (singletons pass through) -> optional final PCA.
    Replaces v1's manual ``pca_groups`` substring dicts.
    """
    causal = True

    def __init__(self, k: int | str = "auto", rmt_denoise: bool = True,
                 n_components=0.95, final_pca: bool = False, final_whiten: bool = True,
                 final_n_components=0.95, k_range=(2, 12), q: float | None = None,
                 bw: float = 0.1):
        self.k = k
        self.rmt_denoise = rmt_denoise
        self.n_components = n_components
        self.final_pca = final_pca
        self.final_whiten = final_whiten
        self.final_n_components = final_n_components
        self.k_range = k_range
        self.q = q
        self.bw = bw

    def fit(self, train_X: pd.DataFrame) -> "HierarchicalPCA":
        from sklearn.decomposition import PCA

        train_X = train_X.dropna()
        self.input_cols_ = list(train_X.columns)
        vals = train_X.to_numpy(dtype=float)
        self.mean_, self.std_ = _standardize_fit(vals)
        Xs = (vals - self.mean_) / self.std_
        corr = np.corrcoef(Xs, rowvar=False)
        q = self.q if self.q is not None else _panel_q(train_X)
        corr_used = rmt.denoise_correlation(corr, q=q, bw=self.bw) if self.rmt_denoise else corr
        labels, Z, k_used = rmt.cluster_corr(corr_used, k=self.k, k_range=self.k_range)
        self.k_ = k_used
        self.linkage_ = Z

        self.clusters_ = {}
        self.pcas_ = {}
        self._members_ = {}
        for cid in np.unique(labels):
            idxs = np.where(labels == cid)[0]
            cols = [self.input_cols_[i] for i in idxs]
            self.clusters_[int(cid)] = cols
            self._members_[int(cid)] = idxs
            if len(idxs) > 1:
                pca = PCA(n_components=self.n_components, whiten=True)
                pca.fit(Xs[:, idxs])
                self.pcas_[int(cid)] = pca

        F = self._cluster_transform(Xs)
        if self.final_pca:
            self.final_pca_ = PCA(n_components=self.final_n_components,
                                  whiten=self.final_whiten).fit(F.to_numpy())
            self.final_input_cols_ = list(F.columns)
            self.output_cols_ = [f"final_pc{i + 1}" for i in range(self.final_pca_.n_components_)]
        else:
            self.output_cols_ = list(F.columns)
        return self

    def _cluster_transform(self, Xs: np.ndarray) -> pd.DataFrame:
        parts = {}
        for cid in sorted(self.clusters_):
            idxs = self._members_[cid]
            if cid in self.pcas_:
                t = self.pcas_[cid].transform(Xs[:, idxs])
                for j in range(t.shape[1]):
                    parts[f"hpc{cid}_pc{j + 1}"] = t[:, j]
            else:
                parts[self.clusters_[cid][0]] = Xs[:, idxs[0]]
        return pd.DataFrame(parts)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        vals = X[self.input_cols_].to_numpy(dtype=float)

        def _fn(v):
            Xs = (v - self.mean_) / self.std_
            F = self._cluster_transform(Xs)
            if self.final_pca:
                out = self.final_pca_.transform(F[self.final_input_cols_].to_numpy())
                return out
            return F.to_numpy()

        out = _apply_complete_rows(vals, len(self.output_cols_), _fn)
        return pd.DataFrame(out, index=X.index, columns=self.output_cols_)

    def describe(self) -> dict:
        d = {**super().describe(), "rmt_denoise": self.rmt_denoise}
        if hasattr(self, "k_"):
            d["k"] = self.k_
            d["cluster_sizes"] = {cid: len(cols) for cid, cols in self.clusters_.items()}
        return d


# ---------------------------------------------------------------------------
# Config + pipeline
# ---------------------------------------------------------------------------

STEP_REGISTRY = {
    "kalman": KalmanDenoiser,
    "ssa": SSADenoiser,
    "pca": PCADecorrelator,
    "zca": ZCAWhitener,
    "hpca": HierarchicalPCA,
    "demean": lambda **kw: CrossSectionalNormalizer(method="demean", **kw),
    "cs_zscore": lambda **kw: CrossSectionalNormalizer(method="zscore", **kw),
    "vol_cs": lambda **kw: CrossSectionalNormalizer(method="vol_cs", **kw),
    "vol_ts": lambda **kw: CrossSectionalNormalizer(method="vol_ts", **kw),
    "signal_keep": lambda **kw: SignalProjector(mode="keep", **kw),
    "signal_strip": lambda **kw: SignalProjector(mode="strip", **kw),
    "standardize": Standardizer,
}


@dataclass
class ProcessingConfig:
    """A named processing recipe.

    Either give ``steps`` explicitly (ordered (step_name, params) pairs from
    STEP_REGISTRY) or use the temporal/decorrelation convenience slots.
    ``params`` maps step name -> kwargs for the slot form.
    """
    name: str
    steps: tuple | list | None = None
    temporal: str | None = None          # None | "kalman" | "ssa"
    decorrelation: str | None = None     # None | "pca" | "zca" | "hpca"
    params: dict = field(default_factory=dict)

    def resolved_steps(self) -> tuple:
        if self.steps is not None:
            return tuple((name, dict(p) if p else {}) for name, p in self.steps)
        steps = []
        for slot in (self.temporal, self.decorrelation):
            if slot:
                steps.append((slot, dict(self.params.get(slot, {}))))
        return tuple(steps)

    def build(self, scale_rules: list[ScaleRule]) -> "ProcessingPipeline":
        transforms = [GroupScaler(scale_rules)]
        for step_name, params in self.resolved_steps():
            if step_name not in STEP_REGISTRY:
                raise KeyError(f"Unknown processing step {step_name!r}; "
                               f"available: {sorted(STEP_REGISTRY)}")
            transforms.append(STEP_REGISTRY[step_name](**params))
        return ProcessingPipeline(transforms, config=self)


def train_row_mask(index: pd.MultiIndex, splits) -> np.ndarray:
    """Boolean mask of training rows: train date window minus val-holdout tickers
    (port of v1 PCAProcessor._get_training_data)."""
    d = date_values(index)
    mask = (d >= pd.Timestamp(splits.train_start)) & (d <= pd.Timestamp(splits.train_end))
    if splits.val_holdout_tickers:
        mask &= ~index.get_level_values("ticker").isin(splits.val_holdout_tickers)
    return np.asarray(mask)


class ProcessingPipeline:
    """Ordered PanelTransforms; every stage fits on the train slice of its
    (already transformed) input, then transforms all rows for the next stage."""

    def __init__(self, transforms: list[PanelTransform], config: ProcessingConfig | None = None):
        self.transforms = list(transforms)
        self.config = config
        self.fitted_ = False

    @property
    def causal(self) -> bool:
        return all(t.causal for t in self.transforms)

    @property
    def name(self) -> str:
        return self.config.name if self.config else "pipeline"

    def fit_transform(self, X: pd.DataFrame, splits) -> pd.DataFrame:
        mask = train_row_mask(X.index, splits)
        cur = X
        for t in self.transforms:
            t.fit(cur[mask])
            cur = t.transform(cur)
        self.fitted_ = True
        return cur

    def fit(self, X: pd.DataFrame, splits) -> "ProcessingPipeline":
        self.fit_transform(X, splits)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("ProcessingPipeline.transform called before fit")
        cur = X
        for t in self.transforms:
            cur = t.transform(cur)
        return cur

    def describe(self) -> dict:
        return {"name": self.name, "causal": self.causal,
                "steps": [t.describe() for t in self.transforms]}


if __name__ == "__main__":
    from findata.preprocess import Momentum
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers

    n_tickers = 5

    tickers = get_all_tickers()[:n_tickers]
    dates = DataSplits()
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    mom = Momentum(oscillators=('rsi', 'cci'))
    X = mom.engineer(data)
    n_dates, n_feat =len(X), len(X.columns)


    mask = train_row_mask(X.index, dates)

    zca = ZCAWhitener().fit(X[mask])
    Z = zca.transform(X[mask])
    C = np.corrcoef(Z.to_numpy(), rowvar=False)
    off = np.abs(C - np.eye(n_feat)).max()
    print(f"ZCA: max |off-diagonal train corr| = {off:.2e} (want < 1e-6)")
    print(f"ZCA: train variances ~1: {np.allclose(Z.var(ddof=1), 1.0, atol=1e-6)}")

    kal = KalmanDenoiser().fit(X[mask])
    print(f"Kalman gains in (0, 1]: {bool(((kal.gains_ > 0) & (kal.gains_ <= 1)).all())}, "
          f"median={kal.gains_.median():.3f}")

    # SSA on a noisy sine: reconstruction should track the clean signal
    t = np.arange(n_dates)
    clean = np.sin(2 * np.pi * t / 60)
    rng = np.random.default_rng(7)
    noisy = clean + rng.normal(0, 0.5, n_dates)
    ssa = SSADenoiser(window_length=40, n_components=2)
    one = pd.DataFrame({"g__sine": np.tile(noisy, 1)},
                       index=pd.MultiIndex.from_product([["T0"], dates], names=["ticker", "date"]))
    rec = ssa.transform(one)["g__sine"].to_numpy()
    corr_clean = np.corrcoef(rec, clean)[0, 1]
    print(f"SSA: corr(reconstruction, clean sine) = {corr_clean:.3f} (noisy corr = "
          f"{np.corrcoef(noisy, clean)[0, 1]:.3f})")

    hp = HierarchicalPCA(rmt_denoise=False).fit(X[mask])
    print(f"HPCA: auto k = {hp.k_}, cluster sizes = "
          f"{ {cid: len(c) for cid, c in hp.clusters_.items()} }")

    csn = CrossSectionalNormalizer(method="demean")
    D = csn.transform(X)
    print(f"CS demean: max |daily mean| after = {D.groupby(level='date').mean().abs().max().max():.2e}")

    sp = SignalProjector(mode="keep").fit(X[mask])
    print(f"SignalProjector: n_signal={sp.n_components_}, top_eig_share={sp.top_eig_share_:.2f}")

    cfg = ProcessingConfig("signal+demean+std",
                           steps=[("signal_keep", {}), ("demean", {}), ("standardize", {})])
    pipe = ProcessingPipeline([Standardizer()] + [STEP_REGISTRY[n](**p) for n, p in cfg.resolved_steps()])
    chained = pipe.fit_transform(X, dates)
    print(f"Chained pipeline: causal={pipe.causal}, out shape={chained.shape}, "
          f"max |daily mean| = {chained.groupby(level='date').mean().abs().max().max():.2e}")
    print("OK")
