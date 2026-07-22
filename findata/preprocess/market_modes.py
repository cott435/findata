"""Cross-sectional (ticker-space) mode decomposition + whitening transforms.

Productionizes the market-structure research (scripts/market_structure q4-q7,
``findata.analysis.market_structure.decompose_modes``) as leakage-safe pipeline
steps: everything is FIT on train rows only, and TRANSFORM uses only same-date
information through train-fixed loadings — causal by construction.

Math per feature column (``axis="ticker"``), mirroring ``decompose_modes``:

    X            standardized wide date x ticker panel (train mean/std per
                 ticker; missing cells zero-filled = neutral after standardize)
    global stage top n_global PCs of the ticker correlation C = X'X/T:
                 unit-variance scores G_j(t) = X w_j / s_j, betas
                 beta_j = X' G_j / T, component = sum_j outer(G_j, beta_j)
    group stage  the same extraction per sector block of the global residual
                 (hierarchically orthogonalized, like decompose_modes)
    residual     X - global_comp - group_comp, renormalized to unit train
                 variance per ticker

At observed cells the identity  standardized = global + group + resid * resid_std
holds exactly. ``axis="feature"`` runs the same two stages across feature
columns on pooled train rows (groups = the ``{group}__`` taxonomy prefix), so
grouped feature-space decomposition is available without a separate transform.

Caveats:
  - tickers absent from the train window (e.g. ``val_holdout_tickers``) have no
    fitted stats and come back NaN at transform time (with a warning) — don't
    combine ticker-axis modes with held-out tickers;
  - dates with fewer than ``min_names`` present tickers pass through
    standardized-only (components 0): a thin cross-section has no usable mode;
  - partial cross-sections (some tickers present, some missing, on an
    otherwise-usable date) get a missing-mass correction: the same-date
    projection is rescaled by sqrt(available / full) weight-squared mass so a
    thin day isn't systematically shrunk — applied identically in
    ModeDecomposer and CrossSectionWhitener via the shared
    ``_mass_corrected_projection`` helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from findata.preprocess import rmt
from findata.preprocess.transforms import (STEP_REGISTRY, PanelTransform,
                                           _apply_complete_rows, _standardize_fit)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _column_panel(series: pd.Series, tickers) -> pd.DataFrame:
    """One feature column as a wide date x ticker frame (fitted ticker order)."""
    return series.unstack("ticker").reindex(columns=tickers)


def _guarded_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0, ddof=0)
    return np.where(std > 1e-12, std, 1.0)


def _mass_corrected_projection(X0: np.ndarray, M: np.ndarray, W: np.ndarray) -> np.ndarray:
    """X0 @ W with a missing-mass correction.

    Each output column j is a linear combination of all input tickers with
    weights W[:, j]; on a date where some tickers are missing (zero-filled in
    X0), the raw dot product only sums over the PRESENT ones, so it is
    systematically too small. Rescale by sqrt(available weight-squared mass /
    full weight-squared mass) to approximate what the projection would be with
    the full cross-section present. Columns whose available mass is
    negligible (``usable`` False) are zeroed rather than blown up.

    Shared by ModeDecomposer (W is N x k, k global/sector loading vectors) and
    CrossSectionWhitener (W is the N x N whitening matrix) — the math is
    identical regardless of how many output columns W has.
    """
    raw = X0 @ W
    full = (W ** 2).sum(axis=0)
    avail = M.astype(float) @ (W ** 2)
    mass = np.sqrt(np.clip(avail / np.where(full > 0, full, 1.0), 0.0, 1.0))
    usable = mass > 1e-6
    return np.where(usable, raw / np.where(usable, mass, 1.0), 0.0)


def _stack_wides(frames: dict[str, pd.DataFrame], like_index: pd.MultiIndex) -> pd.DataFrame:
    """Stack {name: date x ticker} frames back onto the input (ticker, date) index."""
    out = {}
    for name, wide in frames.items():
        s = wide.stack()
        s.index = s.index.set_names(["date", "ticker"]).reorder_levels(["ticker", "date"])
        out[name] = s
    return pd.DataFrame(out).reindex(like_index)


class _TickerAxisMixin:
    """Fit/transform plumbing shared by the two ticker-axis transforms."""

    def _fit_ticker_frame(self, train_X: pd.DataFrame):
        self.input_cols_ = list(train_X.columns)
        self.tickers_ = list(train_X.index.get_level_values("ticker").unique())
        self._n_dates_ = train_X.index.get_level_values("date").nunique()

    def _standardized_train(self, train_X: pd.DataFrame, col: str):
        wide = _column_panel(train_X[col], self.tickers_)
        mu = wide.mean()
        sd = wide.std(ddof=0)
        sd = sd.where(sd > 1e-12, 1.0)
        X0 = ((wide - mu) / sd).fillna(0.0).to_numpy(dtype=float)
        return X0, mu, sd

    def _standardized_any(self, X: pd.DataFrame, col: str, mu, sd):
        wide_all = X[col].unstack("ticker")
        unseen = [t for t in wide_all.columns if t not in set(self.tickers_)]
        wide = wide_all.reindex(columns=self.tickers_)
        M = wide.notna().to_numpy()
        Xs = ((wide - mu) / sd).to_numpy(dtype=float)   # NaN at missing cells
        X0 = np.where(M, Xs, 0.0)
        return wide.index, M, Xs, X0, unseen

    def _warn_unseen(self, unseen: list):
        if unseen and not getattr(self, "_warned_unseen_", False):
            print(f"{type(self).__name__}: {len(unseen)} tickers were not in the "
                  f"train window and come back NaN (e.g. {sorted(unseen)[:5]})")
            self._warned_unseen_ = True


# ---------------------------------------------------------------------------
# ModeDecomposer
# ---------------------------------------------------------------------------

class ModeDecomposer(PanelTransform, _TickerAxisMixin):
    """Global + per-group PCA mode removal with optional component outputs.

    axis        "ticker": modes across the ticker cross-section, one fit per
                feature column; groups come from bound ticker metadata (sector).
                "feature": modes across feature columns on pooled rows; groups
                come from the '{group}__' column prefix. No meta needed.
    group_stage fit per-group modes on the global residual (skips groups with
                fewer than min_group_size members — their group component is 0)
    n_global /  components per stage: an int, or "mp" to take the MP signal
    n_group     count from rmt.iterative_bulk_variance (capped at max_modes)
    output      "residual" (name-preserving) or "residual+components", which
                appends '{col}_gmode' / '{col}_smode' columns — the suffix sits
                after the '{group}__' prefix so provenance parsing still works
    min_names   dates with fewer present tickers pass through standardized
    renormalize divide the residual by its train std (per ticker / per column)
    """
    causal = True

    def __init__(self, axis: str = "ticker", group_stage: bool = True,
                 n_global: int | str = 1, n_group: int | str = 1,
                 output: str = "residual", min_group_size: int = 3,
                 min_names: int = 5, renormalize: bool = True, max_modes: int = 3):
        if axis not in ("ticker", "feature"):
            raise ValueError(f"axis must be 'ticker' or 'feature', got {axis!r}")
        if output not in ("residual", "residual+components"):
            raise ValueError("output must be 'residual' or 'residual+components'")
        self.axis = axis
        self.group_stage = group_stage
        self.n_global = n_global
        self.n_group = n_group
        self.output = output
        self.min_group_size = max(2, int(min_group_size))
        self.min_names = min_names
        self.renormalize = renormalize
        self.max_modes = max_modes
        self._groups_map: pd.Series | None = None

    # -- meta -----------------------------------------------------------------
    @property
    def needs_meta(self) -> bool:
        return self.axis == "ticker" and self.group_stage

    def bind_meta(self, meta) -> None:
        """Accept ticker metadata: a DataFrame with a 'sector' column (e.g. the
        ticker_info frame from get_all_data) or a ticker -> group Series."""
        if meta is None:
            return
        if isinstance(meta, pd.DataFrame):
            if "sector" not in meta.columns:
                raise ValueError("ticker meta needs a 'sector' column")
            meta = meta["sector"]
        self._groups_map = meta.fillna("ETF").astype(str)

    def _group_of(self, name: str) -> str:
        if self.axis == "feature":
            return name.split("__", 1)[0] if "__" in name else "other"
        if self._groups_map is None:
            raise RuntimeError("ModeDecomposer(axis='ticker', group_stage=True) "
                               "needs bind_meta(ticker_info) before fit — pass "
                               "ticker_meta through build_features / meta through "
                               "FeaturePipeline.fit")
        return str(self._groups_map.get(name, "ETF"))

    def _group_members(self, names: list[str]) -> dict[str, np.ndarray]:
        labels = pd.Series([self._group_of(n) for n in names])
        members = {}
        for label in labels.unique():
            idx = np.where(labels.to_numpy() == label)[0]
            if len(idx) >= self.min_group_size:
                members[label] = idx
        return members

    # -- one PCA stage ----------------------------------------------------------
    def _fit_stage(self, X: np.ndarray, denom: int, n_spec, q: float):
        """Top-k loadings/scales/betas of C = X'X/denom. Returns (W, s, B)."""
        C = X.T @ X / denom
        w, v = np.linalg.eigh(C)
        w, v = w[::-1], v[:, ::-1]
        if n_spec == "mp":
            _, _, n_sig = rmt.iterative_bulk_variance(w, q)
            k = int(np.clip(n_sig, 1, self.max_modes))
        else:
            k = max(1, int(n_spec))
        k = max(1, min(k, X.shape[1] - 1))
        W = v[:, :k].copy()
        for j in range(k):                       # sign: mean loading > 0
            if W[:, j].mean() < 0:
                W[:, j] *= -1
        scores = X @ W
        s = _guarded_std(scores)
        G = scores / s
        B = X.T @ G / denom
        return W, s, B

    @staticmethod
    def _stage_component(X0: np.ndarray, M: np.ndarray, W, s, B) -> np.ndarray:
        """Component matrix from same-date scores with a missing-mass correction
        (see ``_mass_corrected_projection``), un-scaled to unit-variance scores
        and re-expanded into ticker space via the fitted betas."""
        scores = _mass_corrected_projection(X0, M, W) / s
        return scores @ B.T

    # -- fit -------------------------------------------------------------------
    def fit(self, train_X: pd.DataFrame) -> "ModeDecomposer":
        return self._fit_feature(train_X) if self.axis == "feature" \
            else self._fit_ticker(train_X)

    def _fit_ticker(self, train_X: pd.DataFrame) -> "ModeDecomposer":
        self._fit_ticker_frame(train_X)
        n = len(self.tickers_)
        self.group_members_ = (self._group_members(self.tickers_)
                               if self.group_stage else {})
        self.state_ = {}
        for col in self.input_cols_:
            X0, mu, sd = self._standardized_train(train_X, col)
            T = X0.shape[0]
            Wg, sg, Bg = self._fit_stage(X0, T, self.n_global, q=T / n)
            R = X0 - ((X0 @ Wg) / sg) @ Bg.T
            groups = {}
            for label, idx in self.group_members_.items():
                Rg = R[:, idx]
                Ws, ss, Bs = self._fit_stage(Rg, T, self.n_group, q=T / len(idx))
                R[:, idx] = Rg - ((Rg @ Ws) / ss) @ Bs.T
                groups[label] = (Ws, ss, Bs)
            self.state_[col] = {
                "mu": mu, "sd": sd, "Wg": Wg, "sg": sg, "Bg": Bg,
                "groups": groups, "resid_sd": _guarded_std(R),
                "global_share": float((Bg ** 2).sum() / n),
            }
        return self

    def _fit_feature(self, train_X: pd.DataFrame) -> "ModeDecomposer":
        train_X = train_X.dropna()
        self.input_cols_ = list(train_X.columns)
        n = len(self.input_cols_)
        vals = train_X.to_numpy(dtype=float)
        self.mean_, self.std_ = _standardize_fit(vals)
        Xs = (vals - self.mean_) / self.std_
        rows = Xs.shape[0]
        q = train_X.index.get_level_values("date").nunique() / n
        Wg, sg, Bg = self._fit_stage(Xs, rows, self.n_global, q=q)
        R = Xs - ((Xs @ Wg) / sg) @ Bg.T
        self.group_members_ = (self._group_members(self.input_cols_)
                               if self.group_stage else {})
        groups = {}
        for label, idx in self.group_members_.items():
            Rg = R[:, idx]
            Ws, ss, Bs = self._fit_stage(Rg, rows, self.n_group, q=q)
            R[:, idx] = Rg - ((Rg @ Ws) / ss) @ Bs.T
            groups[label] = (Ws, ss, Bs)
        self.fstate_ = {"Wg": Wg, "sg": sg, "Bg": Bg, "groups": groups,
                        "resid_sd": _guarded_std(R),
                        "global_share": float((Bg ** 2).sum() / n)}
        return self

    # -- transform ---------------------------------------------------------------
    def _output_names(self) -> list[str]:
        names = list(self.input_cols_)
        if self.output == "residual+components":
            names += [f"{c}_gmode" for c in self.input_cols_]
            if self.group_stage:
                names += [f"{c}_smode" for c in self.input_cols_]
        return names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform_feature(X) if self.axis == "feature" \
            else self._transform_ticker(X)

    def _transform_ticker(self, X: pd.DataFrame) -> pd.DataFrame:
        resid_frames, gmode_frames, smode_frames = {}, {}, {}
        for col in self.input_cols_:
            st = self.state_[col]
            dates, M, Xs, X0, unseen = self._standardized_any(X, col, st["mu"], st["sd"])
            self._warn_unseen(unseen)
            comp_g = self._stage_component(X0, M, st["Wg"], st["sg"], st["Bg"])
            R = X0 - comp_g
            comp_s = np.zeros_like(X0)
            for label, (Ws, ss, Bs) in st["groups"].items():
                idx = self.group_members_[label]
                comp_s[:, idx] = self._stage_component(R[:, idx], M[:, idx], Ws, ss, Bs)
            resid = R - comp_s
            if self.renormalize:
                resid = resid / st["resid_sd"]
            thin = M.sum(axis=1) < self.min_names       # too few names for a mode
            resid[thin] = Xs[thin]
            comp_g[thin] = 0.0
            comp_s[thin] = 0.0
            resid = np.where(M, resid, np.nan)
            as_wide = lambda a: pd.DataFrame(a, index=dates, columns=self.tickers_)
            resid_frames[col] = as_wide(resid)
            if self.output == "residual+components":
                gmode_frames[f"{col}_gmode"] = as_wide(np.where(M, comp_g, np.nan))
                if self.group_stage:
                    smode_frames[f"{col}_smode"] = as_wide(np.where(M, comp_s, np.nan))
        return _stack_wides({**resid_frames, **gmode_frames, **smode_frames}, X.index)

    def _transform_feature(self, X: pd.DataFrame) -> pd.DataFrame:
        st = self.fstate_
        with_components = self.output == "residual+components"

        def _fn(v: np.ndarray) -> np.ndarray:
            Xs = (v - self.mean_) / self.std_
            comp_g = ((Xs @ st["Wg"]) / st["sg"]) @ st["Bg"].T
            R = Xs - comp_g
            comp_s = np.zeros_like(Xs)
            for label, (Ws, ss, Bs) in st["groups"].items():
                idx = self.group_members_[label]
                comp_s[:, idx] = ((R[:, idx] @ Ws) / ss) @ Bs.T
            resid = R - comp_s
            if self.renormalize:
                resid = resid / st["resid_sd"]
            parts = [resid]
            if with_components:
                parts.append(comp_g)
                if self.group_stage:
                    parts.append(comp_s)
            return np.hstack(parts)

        names = self._output_names()
        vals = X[self.input_cols_].to_numpy(dtype=float)
        out = _apply_complete_rows(vals, len(names), _fn)
        return pd.DataFrame(out, index=X.index, columns=names)

    def describe(self) -> dict:
        d = {**super().describe(), "axis": self.axis, "output": self.output,
             "group_stage": self.group_stage, "n_global": self.n_global,
             "n_group": self.n_group}
        state = getattr(self, "state_", None)
        if state:
            shares = [s["global_share"] for s in state.values()]
            d["n_features"] = len(state)
            d["global_share_median"] = float(np.median(shares))
        elif hasattr(self, "fstate_"):
            d["global_share"] = self.fstate_["global_share"]
        if hasattr(self, "group_members_"):
            d["groups"] = {g: int(len(i)) for g, i in self.group_members_.items()}
        return d


# ---------------------------------------------------------------------------
# CrossSectionWhitener
# ---------------------------------------------------------------------------

class CrossSectionWhitener(PanelTransform, _TickerAxisMixin):
    """Ticker-space ZCA per feature column: whiten the cross-section with the
    (optionally MP-denoised) train-window ticker correlation.

    The natural companion to ModeDecomposer("residual"): after the shared modes
    are stripped, whitening equalizes what correlation structure remains among
    the residuals ("whitening after denoising"). Same pseudo-inverse policy as
    the feature-space ZCAWhitener: eigendirections below rcond * lam_max are
    zeroed, never inflated. Transform applies the same missing-mass correction
    as ModeDecomposer (``_mass_corrected_projection``): W here is the full N x N
    whitening matrix rather than a k-dimensional loading matrix, but the
    per-output-column rescale is identical — a missing neighbor ticker on a
    date no longer biases every other ticker's whitened value that day.
    """
    causal = True

    def __init__(self, rmt_denoise: bool = True, method: str = "residual",
                 alpha: float = 0.0, eps: float = 1e-6, rcond: float = 1e-8,
                 min_names: int = 5, renormalize: bool = True):
        self.rmt_denoise = rmt_denoise
        self.method = method
        self.alpha = alpha
        self.eps = eps
        self.rcond = rcond
        self.min_names = min_names
        self.renormalize = renormalize

    def fit(self, train_X: pd.DataFrame) -> "CrossSectionWhitener":
        self._fit_ticker_frame(train_X)
        n = len(self.tickers_)
        self.state_ = {}
        for col in self.input_cols_:
            X0, mu, sd = self._standardized_train(train_X, col)
            T = X0.shape[0]
            C = X0.T @ X0 / T
            if self.rmt_denoise:
                C = rmt.denoise_correlation(C, q=T / n, method=self.method,
                                            alpha=self.alpha)
            w, E = np.linalg.eigh(C)
            cutoff = max(self.eps, self.rcond * float(w.max()))
            w_isqrt = np.where(w > cutoff, np.clip(w, cutoff, None) ** -0.5, 0.0)
            W = E @ np.diag(w_isqrt) @ E.T
            out_sd = _guarded_std(X0 @ W)
            self.state_[col] = {"mu": mu, "sd": sd, "W": W, "out_sd": out_sd,
                                "rank": int((w > cutoff).sum())}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = {}
        for col in self.input_cols_:
            st = self.state_[col]
            dates, M, Xs, X0, unseen = self._standardized_any(X, col, st["mu"], st["sd"])
            self._warn_unseen(unseen)
            out = _mass_corrected_projection(X0, M, st["W"])
            if self.renormalize:
                out = out / st["out_sd"]
            thin = M.sum(axis=1) < self.min_names
            out[thin] = Xs[thin]
            out = np.where(M, out, np.nan)
            frames[col] = pd.DataFrame(out, index=dates, columns=self.tickers_)
        return _stack_wides(frames, X.index)

    def describe(self) -> dict:
        d = {**super().describe(), "rmt_denoise": self.rmt_denoise}
        if hasattr(self, "state_"):
            d["rank_median"] = int(np.median([s["rank"] for s in self.state_.values()]))
        return d


# Registered here (not in transforms.py) so the import stays one-directional;
# findata.preprocess.__init__ imports this module, which populates the registry
# before any ProcessingConfig can be built.
STEP_REGISTRY["modes"] = ModeDecomposer
STEP_REGISTRY["cs_whiten"] = CrossSectionWhitener
