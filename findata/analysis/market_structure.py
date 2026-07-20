"""Market-structure RMT toolkit: cross-ticker correlation, temporal shifts,
and market-mode removal on panels of returns (or any per-ticker indicator).

Expands scripts/testing/market_signal_vs_sampling_noise.py and Nobi et al.
(arXiv:1302.6305): standardized series -> correlation matrix -> Marchenko-
Pastur bulk vs signal eigenvalues, with explicit null hypotheses.

Conventions
-----------
- "wide" panels are date x ticker DataFrames (one column per ticker).
- q = T / N as in findata.preprocess.rmt (the paper uses Q = N/T; convert).
- Noise variance uses rmt.iterative_bulk_variance (recursive bulk-mean).

Null models
-----------
- shuffle_null      : independently permute each column in time. Preserves each
                      ticker's marginal distribution (vol, fat tails); destroys
                      BOTH autocorrelation and all cross-correlation.
- circular_shift_null: independently rotate each column by a random offset.
                      Preserves each ticker's autocorrelation exactly; destroys
                      cross-sectional alignment. The right null for lead-lag
                      questions and for smoothed (autocorrelated) panels.
- random ticker groups (label exchangeability): the null for "is this SECTOR
  special" — compare a sector's statistic against same-size random groups.

Every statistic that gets a p-value states its H0 in the docstring.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from findata.configs import DATA_DIR
from findata.preprocess import rmt

# indicators (besides log returns) that take (ohlcv_df, period=...) and return a
# Series — same whitelist as preprocess.oscillators
SERIES_INDICATORS = ("rsi", "cci", "willr", "mfi", "cmf")


# ===========================================================================
# Data loading
# ===========================================================================

def parse_market_cap(value) -> float:
    """'4.66T' -> 4.66e12; handles K/M/B/T suffixes, plain numbers, NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().upper().replace(",", "").replace("$", "")
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(s[-1], None)
    return float(s[:-1]) * mult if mult else float(s)


def load_market_caps(path: str | Path | None = None) -> pd.Series:
    """Market caps from data/us_tickers.xlsx (computed at its 'Price' snapshot;
    assumes shares outstanding roughly constant — favor recent-year analyses)."""
    path = Path(path) if path else DATA_DIR / "us_tickers.xlsx"
    df = pd.read_excel(path).set_index("Ticker")
    return df["Market Cap"].map(parse_market_cap).rename("market_cap")


def cap_buckets(caps: pd.Series, n_buckets: int = 3,
                labels: Sequence[str] = ("small", "mid", "large")) -> pd.Series:
    """Equal-count market-cap buckets (by rank, so mega caps don't compress the rest)."""
    return pd.qcut(caps.rank(method="first"), n_buckets,
                   labels=list(labels)[:n_buckets]).rename("cap_bucket")


def indicator_panel(tickers: Iterable[str] | None = None, start: str = "2021-01-01",
                    end: str | None = None, indicator: str | Callable = "log_return",
                    min_coverage: float = 0.9, db_path=None,
                    indicator_kwargs: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a wide date x ticker panel of the chosen indicator + ticker meta.

    indicator:
      "log_return" (default)      log(close_t / close_{t-1})
      "rsi" / "cci" / ...         any Series-returning INDICATOR_FUNCS entry,
                                  computed per ticker (pass e.g.
                                  indicator_kwargs={"period": 14})
      callable(ohlcv_df)->Series  anything custom

    Tickers with < min_coverage non-null history are dropped, then rows with
    any gap are dropped (equal-time matrix, as in the reference script).
    Returns (wide, meta) where meta has sector/industry/market_cap/cap_bucket
    for exactly the kept tickers.
    """
    from findata import get_price_data, get_ticker_meta

    prices = get_price_data(tickers, start_date=start, end_date=end, db_path=db_path)
    if indicator == "log_return":
        wide = np.log(prices["close"].unstack("ticker").sort_index()).diff()
    else:
        if callable(indicator):
            func = indicator
        else:
            from findata.database.technical_calculators import INDICATOR_FUNCS
            if indicator not in SERIES_INDICATORS:
                raise ValueError(f"indicator must be 'log_return', a callable, or one of "
                                 f"{SERIES_INDICATORS}; got {indicator!r}")
            func = partial(INDICATOR_FUNCS[indicator], **(indicator_kwargs or {}))
        series = prices.groupby(level="ticker", group_keys=False).apply(func)
        wide = series.unstack("ticker").sort_index()

    wide = wide.dropna(axis=1, thresh=int(min_coverage * len(wide)))
    wide = wide.dropna(axis=0, how="any")
    wide = wide.loc[:, wide.std(ddof=0) > 1e-12]

    meta = get_ticker_meta(list(wide.columns)).reindex(wide.columns)
    caps = load_market_caps().reindex(wide.columns)
    meta["market_cap"] = caps
    meta["cap_bucket"] = cap_buckets(caps.fillna(caps.median()))
    return wide, meta


# ===========================================================================
# Core RMT on wide panels
# ===========================================================================

def standardize(wide: pd.DataFrame) -> np.ndarray:
    """x_it = (r_it - mean_i) / sigma_i (Nobi Eq. 2); returns a (T, N) array."""
    X = wide.to_numpy(dtype=float)
    X = X - X.mean(0, keepdims=True)
    return X / (X.std(0, ddof=0, keepdims=True) + 1e-12)


def corr_matrix(wide: pd.DataFrame) -> np.ndarray:
    """C = (1/T) X'X of the standardized panel; unit diagonal, trace = N."""
    X = standardize(wide)
    return (X.T @ X) / X.shape[0]


def offdiag(C: np.ndarray) -> np.ndarray:
    return C[np.triu_indices_from(C, k=1)]


def spectrum(wide: pd.DataFrame, q: float | None = None) -> dict:
    """Eigen-spectrum + iterative-MP diagnostics of a wide panel.

    Returns n, t, q (=T/N), mean/std of off-diagonal correlations, sigma2,
    lam_plus, n_signal, top_eig_share and the sorted eigenvalues.
    """
    C = corr_matrix(wide)
    T, N = wide.shape
    q = q if q is not None else T / N
    w = np.linalg.eigvalsh(C)[::-1]
    sigma2, lam_plus, n_signal = rmt.iterative_bulk_variance(w, q)
    off = offdiag(C)
    return {
        "n": N, "t": T, "q": float(q),
        "mean_corr": float(off.mean()), "std_corr": float(off.std()),
        "sigma2": float(sigma2), "lam_plus": float(lam_plus),
        "n_signal": int(n_signal), "top_eig": float(w[0]),
        "top_eig_share": float(w[0] / w.sum()),
        "eigvals": w, "corr": C,
    }


# ===========================================================================
# Null models
# ===========================================================================

def shuffle_null(wide: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Independent time-permutation per column. H0 material: kills auto- and
    cross-correlation, keeps marginals."""
    rng = np.random.default_rng(seed)
    R = wide.to_numpy(copy=True)
    for j in range(R.shape[1]):
        R[:, j] = R[rng.permutation(R.shape[0]), j]
    return pd.DataFrame(R, index=wide.index, columns=wide.columns)


def circular_shift_null(wide: pd.DataFrame, seed: int = 0,
                        min_shift: int = 20) -> pd.DataFrame:
    """Independent random circular rotation per column. Preserves each column's
    autocorrelation exactly, destroys cross-sectional alignment — the null for
    lead-lag questions and for smoothed panels."""
    rng = np.random.default_rng(seed)
    T = len(wide)
    R = wide.to_numpy(copy=True)
    for j in range(R.shape[1]):
        R[:, j] = np.roll(R[:, j], int(rng.integers(min_shift, T - min_shift)))
    return pd.DataFrame(R, index=wide.index, columns=wide.columns)


def null_distribution(wide: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], dict],
                      null: str = "shuffle", n_draws: int = 50,
                      transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                      seed: int = 0) -> pd.DataFrame:
    """Distribution of ``stat_fn`` under a null. ``transform`` (e.g. Kalman
    smoothing) is applied to each null draw so the null goes through the SAME
    processing as the observed data — mandatory when the processing induces
    autocorrelation that breaks the analytic MP assumptions."""
    null_fn = {"shuffle": shuffle_null, "circular": circular_shift_null}[null]
    rows = []
    for b in range(n_draws):
        draw = null_fn(wide, seed=seed + b)
        if transform is not None:
            draw = transform(draw)
        rows.append(stat_fn(draw))
    return pd.DataFrame(rows)


def permutation_pvalue(null_values: np.ndarray, observed: float, tail: str = "greater") -> float:
    """Add-one permutation p-value: P(null >= observed) (or <=, or two-sided)."""
    null_values = np.asarray(null_values, dtype=float)
    B = len(null_values)
    if tail == "greater":
        k = int((null_values >= observed).sum())
    elif tail == "less":
        k = int((null_values <= observed).sum())
    else:
        center = null_values.mean()
        k = int((np.abs(null_values - center) >= abs(observed - center)).sum())
    return (1 + k) / (1 + B)


# ===========================================================================
# Temporal smoothing (steady-state local-level Kalman, per column)
# ===========================================================================

def kalman_smooth(wide: pd.DataFrame, gain_floor: float = 0.02,
                  gain_cap: float = 1.0) -> tuple[pd.DataFrame, pd.Series]:
    """Causal steady-state local-level Kalman filter per column — the same
    method-of-moments + EWMA construction as preprocess.transforms.KalmanDenoiser,
    for plain date x ticker frames.

    On near-white series (daily returns) the model finds lam ~ 0 and the gain
    hits ``gain_floor`` — i.e. a long EWMA. Correlations of smoothed panels are
    mechanically inflated (fewer effective observations); test against a null
    that is smoothed the same way (see null_distribution(transform=...)).

    Returns (smoothed, gains).
    """
    d = wide.diff()
    d1 = d.shift(1)
    cov1 = ((d - d.mean()) * (d1 - d1.mean())).mean()
    var_d = d.var(ddof=1)
    sigma_eps = (-cov1).clip(lower=0.0)
    sigma_eta = (var_d + 2 * cov1).clip(lower=0.0)
    lam = sigma_eta / sigma_eps.where(sigma_eps > 1e-15)
    gains = ((-lam + np.sqrt(lam ** 2 + 4 * lam)) / 2.0).clip(gain_floor, gain_cap).fillna(1.0)
    out = {col: (wide[col] if gains[col] >= 0.999
                 else wide[col].ewm(alpha=float(gains[col]), adjust=False).mean())
           for col in wide.columns}
    return pd.DataFrame(out, index=wide.index)[wide.columns], gains


# ===========================================================================
# Group statistics (Q1)
# ===========================================================================

def group_stats(wide: pd.DataFrame, members: Sequence[str]) -> dict:
    """Correlation/spectrum statistics of a ticker subset (its own q = T/n).
    top_eig_share = lambda_1 / n is the size-comparable 'how collective is this
    group' number (lambda_1 itself scales with group size)."""
    members = [m for m in members if m in wide.columns]
    spec = spectrum(wide[members])
    return {"size": len(members), "mean_corr": spec["mean_corr"],
            "top_eig": spec["top_eig"], "top_eig_share": spec["top_eig_share"],
            "n_signal": spec["n_signal"], "sigma2": spec["sigma2"],
            "lam_plus": spec["lam_plus"]}


def random_group_null(wide: pd.DataFrame, size: int, n_draws: int = 500,
                      seed: int = 0) -> pd.DataFrame:
    """Statistics of uniformly random same-size ticker groups.

    H0 (label exchangeability): group membership carries no information — a
    labeled group (sector, cap bucket) behaves like a random group of its size.
    """
    rng = np.random.default_rng(seed)
    cols = np.asarray(wide.columns)
    rows = [group_stats(wide, rng.choice(cols, size=size, replace=False))
            for _ in range(n_draws)]
    return pd.DataFrame(rows)


# ===========================================================================
# Lead-lag / temporal alignment (Q2)
# ===========================================================================

def lag_correlation_cube(wide: pd.DataFrame, max_lag: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """cube[l, i, j] = corr(x_i(t), x_j(t + lag_l)) for lag_l in [-L, ..., +L].

    Positive lag at (i, j): i today correlates with j lag days LATER (i leads j).
    Standardization is global (T >> L, per-overlap re-standardization is
    negligible). Returns (lags, cube).
    """
    X = standardize(wide)
    T, N = X.shape
    lags = np.arange(-max_lag, max_lag + 1)
    cube = np.empty((len(lags), N, N))
    for ell in range(0, max_lag + 1):
        c = (X[:T - ell].T @ X[ell:]) / (T - ell)   # corr(x_i(t), x_j(t+ell))
        cube[max_lag + ell] = c
        cube[max_lag - ell] = c.T
    return lags, cube


def best_lag_stats(lags: np.ndarray, cube: np.ndarray) -> pd.DataFrame:
    """Per unordered pair: correlation at lag 0, best correlation over all lags,
    the argmax lag, and the gain (best - lag0). One row per pair (i < j)."""
    L = (len(lags) - 1) // 2
    iu = np.triu_indices(cube.shape[1], k=1)
    stack = cube[:, iu[0], iu[1]]                 # (n_lags, n_pairs)
    best_idx = stack.argmax(axis=0)
    return pd.DataFrame({
        "i": iu[0], "j": iu[1],
        "c0": stack[L], "c_best": stack[best_idx, np.arange(stack.shape[1])],
        "best_lag": lags[best_idx],
    }).assign(gain=lambda d: d.c_best - d.c0)


def market_mode_series(wide: pd.DataFrame) -> pd.Series:
    """PC1 score series (the 'global market signal'), unit variance, signed so
    that the average loading is positive."""
    X = standardize(wide)
    C = (X.T @ X) / X.shape[0]
    w, v = np.linalg.eigh(C)
    v1 = v[:, -1]
    if v1.mean() < 0:
        v1 = -v1
    score = X @ v1
    return pd.Series(score / score.std(ddof=0), index=wide.index, name="market_mode")


def align_to_reference(wide: pd.DataFrame, max_lag: int = 10,
                       reference: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Shift every ticker to maximize its correlation with a reference series
    (default: the panel's own market mode / PC1 scores).

    shift s_j = argmax_s corr(x_j(t), ref(t + s)); s_j > 0 means ticker j LEADS
    the reference by s_j days. The aligned panel delays each ticker by its own
    s_j so everything lines up, then trims the max_lag edges.

    Returns (aligned wide, shifts, lag-correlation curves per ticker).
    """
    X = standardize(wide)
    T, N = X.shape
    ref = (market_mode_series(wide) if reference is None else reference).to_numpy(dtype=float)
    ref = (ref - ref.mean()) / (ref.std(ddof=0) + 1e-12)

    lags = np.arange(-max_lag, max_lag + 1)
    curves = np.empty((len(lags), N))
    for k, s in enumerate(lags):
        if s >= 0:
            curves[k] = X[:T - s].T @ ref[s:] / (T - s)
        else:
            curves[k] = X[-s:].T @ ref[:T + s] / (T + s)
    shifts = pd.Series(lags[curves.argmax(axis=0)], index=wide.columns, name="shift")

    aligned = pd.DataFrame({col: wide[col].shift(int(shifts[col])) for col in wide.columns},
                           index=wide.index)
    aligned = aligned.iloc[max_lag: len(aligned) - max_lag]
    return aligned, shifts, pd.DataFrame(curves, index=lags, columns=wide.columns)


# ===========================================================================
# Mode removal (Q4 / Q5)
# ===========================================================================

def remove_top_pcs(wide: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """Project the top-k principal components out of the standardized panel and
    renormalize every column back to unit variance ('transform back and
    renormalize'). The residual panel is what's left after the global mode(s)."""
    X = standardize(wide)
    C = (X.T @ X) / X.shape[0]
    _, v = np.linalg.eigh(C)
    Vk = v[:, -k:]
    R = X - X @ Vk @ Vk.T
    R = R / (R.std(0, ddof=0, keepdims=True) + 1e-12)
    return pd.DataFrame(R, index=wide.index, columns=wide.columns)


def remove_group_modes(wide: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Remove each GROUP's own PC1 (its 'sector mode', which contains that
    group's share of the global mode too) from that group's tickers, then
    renormalize. Groups with < 2 members pass through standardized."""
    X = standardize(wide)
    out = X.copy()
    for g in labels.dropna().unique():
        cols = [c for c in wide.columns if labels.get(c) == g]
        idx = [wide.columns.get_loc(c) for c in cols]
        if len(idx) < 2:
            continue
        Xg = X[:, idx]
        Cg = (Xg.T @ Xg) / Xg.shape[0]
        _, v = np.linalg.eigh(Cg)
        v1 = v[:, -1:]
        out[:, idx] = Xg - Xg @ v1 @ v1.T
    out = out / (out.std(0, ddof=0, keepdims=True) + 1e-12)
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


def block_mean_corr(corr: np.ndarray, columns: Sequence[str], labels: pd.Series) -> pd.DataFrame:
    """Group x group mean correlation: diagonal = mean within-group off-diagonal
    correlation, off-diagonal = mean cross-group correlation."""
    labs = np.asarray([labels.get(c) for c in columns])
    groups = sorted(pd.unique(labs[pd.notna(labs)]))
    out = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for a in groups:
        ia = np.where(labs == a)[0]
        for b in groups:
            ib = np.where(labs == b)[0]
            block = corr[np.ix_(ia, ib)]
            if a == b:
                vals = block[np.triu_indices_from(block, k=1)]
                out.loc[a, b] = float(vals.mean()) if vals.size else np.nan
            else:
                out.loc[a, b] = float(block.mean())
    return out


# ===========================================================================
# Mode decomposition & transform layers (reward-structure visualization)
# ===========================================================================

def sample_tickers(meta: pd.DataFrame, n: int = 8, sector: str | None = None,
                   seed: int = 0) -> list[str]:
    """Seeded random ticker sample, optionally restricted to one sector."""
    pool = meta.index if sector is None else meta.index[meta["sector"] == sector]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(pool), size=min(n, len(pool)), replace=False))


def decompose_modes(wide: pd.DataFrame, labels: pd.Series) -> dict:
    """Hierarchical mode decomposition of a standardized panel:

        x_i(t) = beta_i^G * G(t)  +  beta_i^S * S_{s(i)}(t)  +  eps_i(t)

    G is the global market mode (PC1 of the full panel). Each sector mode S_s
    is PC1 of that sector's GLOBAL-RESIDUAL block, so the components are
    hierarchically orthogonalized and the identity raw = global_component +
    sector_component + residual holds exactly.

    Returns dict:
      panels    {'raw', 'global_component', 'sector_component',
                 'combined_modes', 'residual'} -> wide date x ticker frames
      modes     date x mode frame ('global', 'sector_<name>', unit variance,
                signed so mean loading > 0)
      loadings  ticker-indexed frame: sector, beta_global, beta_sector
    """
    X = standardize(wide)
    T = X.shape[0]
    g = market_mode_series(wide).to_numpy()

    beta_g = X.T @ g / T
    global_comp = np.outer(g, beta_g)
    R1 = X - global_comp

    sector_comp = np.zeros_like(X)
    modes = {"global": g}
    beta_s = np.zeros(X.shape[1])
    for s in labels.dropna().unique():
        idx = [wide.columns.get_loc(c) for c in wide.columns if labels.get(c) == s]
        if len(idx) < 2:
            continue
        Xg = R1[:, idx]
        Cg = (Xg.T @ Xg) / T
        _, v = np.linalg.eigh(Cg)
        v1 = v[:, -1]
        if v1.mean() < 0:
            v1 = -v1
        gs = Xg @ v1
        gs = gs / (gs.std(ddof=0) + 1e-12)
        b = Xg.T @ gs / T
        sector_comp[:, idx] = np.outer(gs, b)
        beta_s[idx] = b
        modes[f"sector_{s}"] = gs

    residual = R1 - sector_comp
    as_frame = lambda a: pd.DataFrame(a, index=wide.index, columns=wide.columns)
    return {
        "panels": {
            "raw": as_frame(X),
            "global_component": as_frame(global_comp),
            "sector_component": as_frame(sector_comp),
            "combined_modes": as_frame(global_comp + sector_comp),
            "residual": as_frame(residual),
        },
        "modes": pd.DataFrame(modes, index=wide.index),
        "loadings": pd.DataFrame({"sector": [labels.get(c) for c in wide.columns],
                                  "beta_global": beta_g, "beta_sector": beta_s},
                                 index=wide.columns),
    }


def _obv_velocity(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    from findata.database.technical_calculators import ema, obv
    o = obv(df)
    vel = ema(o, fast) - ema(o, slow)
    z = (vel - vel.rolling(slow * 2).mean()) / vel.rolling(slow * 2).std()
    return z.rename("obv_vel")


def _price_velocity(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    from findata.database.technical_calculators import ema
    vel = ema(df["close"], fast) - ema(df["close"], slow)
    z = (vel - vel.rolling(slow * 2).mean()) / vel.rolling(slow * 2).std()
    return z.rename("price_vel")


def _feature_func(name: str):
    from findata.database.technical_calculators import INDICATOR_FUNCS
    if name == "log_return":
        return lambda df: np.log(df["close"]).diff()
    if name == "obv_vel":
        return _obv_velocity
    if name == "price_vel":
        return _price_velocity
    if name in SERIES_INDICATORS:
        return lambda df: INDICATOR_FUNCS[name](df)
    raise ValueError(f"unknown feature {name!r}")


DEFAULT_FEATURES = ("log_return", "price_vel", "obv_vel", "rsi", "cci", "willr", "mfi", "cmf")


def feature_panels(tickers: Iterable[str] | None = None,
                   features: Sequence[str] = DEFAULT_FEATURES,
                   start: str = "2021-01-01", end: str | None = None,
                   min_coverage: float = 0.9, db_path=None) -> tuple[dict, pd.DataFrame]:
    """Build one aligned wide (date x ticker) panel per feature, all on a shared
    dense (equal-time, common-ticker) grid, plus ticker meta.

    Every feature is computed per ticker from OHLCV, then the panels are
    intersected so the FxF comparisons downstream are on identical rows.
    """
    from findata import get_price_data, get_ticker_meta

    prices = get_price_data(tickers, start_date=start, end_date=end, db_path=db_path)
    series = {name: prices.groupby(level="ticker", group_keys=False).apply(_feature_func(name))
              for name in features}
    panels = {name: s.unstack("ticker").sort_index() for name, s in series.items()}

    keep_t = None
    for p in panels.values():
        good = set(p.columns[p.notna().mean() >= min_coverage])
        keep_t = good if keep_t is None else (keep_t & good)
    keep_t = sorted(keep_t)
    panels = {f: p[keep_t] for f, p in panels.items()}

    mask = None
    for p in panels.values():
        m = p.notna()
        mask = m if mask is None else (mask & m)
    good_dates = mask.index[mask.all(axis=1)]
    panels = {f: p.loc[good_dates] for f, p in panels.items()}

    meta = get_ticker_meta(keep_t).reindex(keep_t)
    caps = load_market_caps().reindex(keep_t)
    meta["market_cap"] = caps
    meta["cap_bucket"] = cap_buckets(caps.fillna(caps.median()))
    return panels, meta


def _pooled_feature_corr(panel_dict: dict) -> np.ndarray:
    """FxF correlation over all (ticker, date) observations. Each feature's
    per-ticker series is standardized first so every ticker weighs equally."""
    cols = []
    for arr in panel_dict.values():
        a = np.asarray(arr, dtype=float)
        a = (a - a.mean(0, keepdims=True)) / (a.std(0, ddof=0, keepdims=True) + 1e-12)
        cols.append(a.reshape(-1))
    return np.corrcoef(np.column_stack(cols), rowvar=False)


def mode_feature_correlation(panels: dict, labels: pd.Series) -> dict:
    """Compare the FxF correlation of features across four representations:

      pool          pooled over T x N (the standard feature correlation)
      global        correlation of each feature's GLOBAL market mode (T-length)
      sector        mean over sectors of each feature's sector-mode correlation
      resid_global  pooled FxF after removing each feature's global mode
      resid_sector  pooled FxF after removing global + sector modes

    'pool' minus 'resid' is the feature correlation that lived in the shared
    modes; 'global' says whether those modes point the same way. Returns the
    matrices (feature-ordered), the feature list, and per-feature decompositions.
    """
    feats = list(panels)
    decomps = {f: decompose_modes(panels[f], labels) for f in feats}

    pool = _pooled_feature_corr({f: panels[f].to_numpy() for f in feats})
    resid_global = _pooled_feature_corr(
        {f: (decomps[f]["panels"]["raw"] - decomps[f]["panels"]["global_component"]).to_numpy()
         for f in feats})
    resid_sector = _pooled_feature_corr(
        {f: decomps[f]["panels"]["residual"].to_numpy() for f in feats})

    G = np.column_stack([decomps[f]["modes"]["global"].to_numpy() for f in feats])
    global_corr = np.corrcoef(G, rowvar=False)

    shared_sectors = set.intersection(*[
        {c for c in decomps[f]["modes"].columns if c.startswith("sector_")} for f in feats])
    if shared_sectors:
        mats = []
        for s in sorted(shared_sectors):
            S = np.column_stack([decomps[f]["modes"][s].to_numpy() for f in feats])
            mats.append(np.corrcoef(S, rowvar=False))
        sector_corr = np.mean(mats, axis=0)
    else:
        sector_corr = np.full_like(global_corr, np.nan)

    return {"features": feats, "pool": pool, "global": global_corr,
            "sector": sector_corr, "resid_global": resid_global,
            "resid_sector": resid_sector, "decomps": decomps,
            "n_sectors": len(shared_sectors)}


def transform_layers(wide: pd.DataFrame, labels: pd.Series | None = None,
                     which=("raw", "kalman", "demean", "minus_global_pc1",
                            "minus_sector_pc1")) -> dict[str, pd.DataFrame]:
    """The transform-comparison layers, each a wide date x ticker frame.

    Conventions: 'raw' is the standardized panel; 'kalman' filters that panel
    WITHOUT re-standardizing (the amplitude shrink IS the alteration); 'demean'
    subtracts each date's cross-sectional mean; the mode-removal layers come
    back renormalized to unit variance (their functions' contract).
    """
    X = pd.DataFrame(standardize(wide), index=wide.index, columns=wide.columns)
    out: dict[str, pd.DataFrame] = {}
    for name in which:
        if name == "raw":
            out[name] = X
        elif name == "kalman":
            out[name] = kalman_smooth(X)[0]
        elif name == "demean":
            out[name] = X - X.mean(axis=1).to_numpy()[:, None]
        elif name == "minus_global_pc1":
            out[name] = remove_top_pcs(wide, 1)
        elif name == "minus_sector_pc1":
            if labels is None:
                raise ValueError("minus_sector_pc1 needs sector labels")
            out[name] = remove_group_modes(wide, labels)
        else:
            raise ValueError(f"unknown transform layer {name!r}")
    return out


# ===========================================================================
# Greedy group selection (Q3: correlated / uncorrelated ticker sets)
# ===========================================================================

def select_correlated_group(corr: np.ndarray, names: Sequence[str], size: int = 30) -> list[str]:
    """Greedy max-mean-correlation group: seed with the highest-correlation pair,
    then repeatedly add the ticker with the highest mean correlation to the
    current members."""
    C = np.asarray(corr, dtype=float)
    iu = np.triu_indices_from(C, k=1)
    seed_flat = np.argmax(C[iu])
    members = [iu[0][seed_flat], iu[1][seed_flat]]
    while len(members) < min(size, len(names)):
        mean_to_group = C[:, members].mean(axis=1)
        mean_to_group[members] = -np.inf
        members.append(int(mean_to_group.argmax()))
    return [names[i] for i in members]


def select_uncorrelated_group(corr: np.ndarray, names: Sequence[str], size: int = 30) -> list[str]:
    """Greedy min-mean-|correlation| group (seeded with the lowest-|corr| pair)."""
    A = np.abs(np.asarray(corr, dtype=float))
    iu = np.triu_indices_from(A, k=1)
    seed_flat = np.argmin(A[iu])
    members = [iu[0][seed_flat], iu[1][seed_flat]]
    while len(members) < min(size, len(names)):
        mean_to_group = A[:, members].mean(axis=1)
        mean_to_group[members] = np.inf
        members.append(int(mean_to_group.argmin()))
    return [names[i] for i in members]
