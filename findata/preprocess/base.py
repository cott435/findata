"""Feature-group base for the v2 preprocessing pipeline.

v2 separates the concerns v1's ``PCAProcessor`` entangled:

- :class:`FeatureGroup` subclasses (momentum, trend, ...) do **feature
  engineering only** — pure functions of the raw wide frame.
- Scaling/denoising/decorrelation live in :mod:`findata.preprocess.transforms`
  as composable ``PanelTransform`` steps; each group declares how its columns
  should be scaled through :class:`ScaleRule`.
- Orchestration (concatenation, train-split fitting, state save/apply) lives in
  :mod:`findata.preprocess.pipeline`.

This module is on the production featurization path and must stay import-light
(no matplotlib/seaborn at module level).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)

from findata.configs import DATA_MAP, TECHNICAL_WINDOWS
from findata.database.technical_calculators import INDICATOR_FUNCS

# bare indicator name -> default-period column name, e.g. 'rsi' -> 'rsi_14'
PERIOD_MAP = {name: TECHNICAL_WINDOWS[base] for base, names in DATA_MAP.items() for name in names}


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

class LogStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_log = np.log1p(np.asarray(X) + self.epsilon)
        self.scaler.fit(X_log)
        return self

    def transform(self, X):
        X_log = np.log1p(np.asarray(X) + self.epsilon)
        return self.scaler.transform(X_log)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def get_scaler(scaler: str):
    if scaler == 'minmax':
        return MinMaxScaler()
    elif scaler == 'robust':
        return RobustScaler()
    elif scaler == 'standard':
        return StandardScaler()
    elif scaler == 'log_standard':
        return LogStandardScaler()
    elif scaler == 'quantile':
        return QuantileTransformer(output_distribution='normal')
    elif scaler == 'power':
        return PowerTransformer(method="yeo-johnson", standardize=True)
    else:
        raise ValueError(f'Unknown scaler {scaler!r}')


@dataclass
class ScaleRule:
    """How one set of engineered columns gets scaled.

    Declarative replacement for v1's per-class ``_scale`` overrides (e.g. Volume
    passing ``zvel`` columns through unscaled, Candle mapping ``body_size`` from
    [0, 1] to [-1, 1]).
    """
    columns: list[str]
    kind: str = "scaler"          # "scaler" | "passthrough" | "affine"
    scaler: str = "standard"      # get_scaler key, when kind == "scaler"
    arcsinh: bool = False         # arcsinh(scaled).clip(-3.5, 3.5) after scaling
    mul: float = 1.0              # when kind == "affine": x * mul + add
    add: float = 0.0

    def __post_init__(self):
        if self.kind not in ("scaler", "passthrough", "affine"):
            raise ValueError(f"Unknown ScaleRule kind {self.kind!r}")

    def with_prefix(self, prefix: str) -> "ScaleRule":
        return replace(self, columns=[f"{prefix}{c}" for c in self.columns])


# ---------------------------------------------------------------------------
# FeatureGroup
# ---------------------------------------------------------------------------

class FeatureGroup:
    """One feature family (momentum, trend, ...): feature engineering only.

    Contract for :meth:`engineer`:
      - input: the wide ``get_all_data`` frame, MultiIndex ``(ticker, date)``
      - output: engineered columns (unprefixed), same index; leading rolling
        burn-in NaNs are fine (the pipeline drops them)
      - pure: never mutate ``data`` (pandas-3 copy-on-write is on)
      - anything that shifts/rolls must run per ticker (use :func:`per_ticker`)
    """

    name: str = None                 # taxonomy prefix, e.g. "momentum"
    default_scaler: str = "standard"
    default_arcsinh: bool = False

    def __init__(self, feature_set: str = "med", verbose: bool = False):
        if feature_set not in ("low", "med", "high"):
            raise ValueError(f"feature_set must be low/med/high, got {feature_set!r}")
        self.feature_set = feature_set
        self.verbose = verbose

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement engineer()")

    def scale_rules(self, columns: Iterable[str]) -> list[ScaleRule]:
        """Default: one rule covering every engineered column."""
        return [ScaleRule(columns=list(columns), scaler=self.default_scaler,
                          arcsinh=self.default_arcsinh)]

    def _resolve_feature(self, data: pd.DataFrame, name: str, period: int) -> pd.Series:
        target = f"{name}_{period}"
        if target in data.columns:
            return data[target]
        return INDICATOR_FUNCS[name](data, period=period)

    def plot_fe(self, engineered: pd.DataFrame, raw: pd.DataFrame,
                ticker: str | None = None, tail: int = 500):
        """Optional visual check of the construction; subclasses may override."""
        raise NotImplementedError(f"{type(self).__name__} does not implement plot_fe")

    def __repr__(self):
        return f"{type(self).__name__}(feature_set={self.feature_set!r})"

    def _check_rank_deficiency(self, corr):
        rank = np.linalg.matrix_rank(corr, tol=1e-8)
        if rank < len(corr):
            print(f"Rank deficiency detected in {type(self).__name__}: {rank} < {len(corr)}")


def per_ticker(data: pd.DataFrame, func: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Apply ``func`` per ticker; result keeps the (ticker, date) MultiIndex.

    Use for anything involving shift/rolling/ewm so windows never bleed across
    ticker boundaries.
    """
    return data.groupby(level="ticker", group_keys=False).apply(func)


def date_values(index: pd.MultiIndex) -> pd.DatetimeIndex:
    """The 'date' index level as datetime64.

    The DB stores python ``datetime.date`` objects (object dtype), and pandas 3
    raises on ``Timestamp`` vs ``date`` comparisons — normalize before comparing
    against DataSplits boundaries.
    """
    return pd.DatetimeIndex(pd.to_datetime(index.get_level_values("date")))


# ---------------------------------------------------------------------------
# Column utilities (ported from v1)
# ---------------------------------------------------------------------------

def get_columns(df, keys, ignore=None):
    ignore = ignore if isinstance(ignore, list) else [ignore] if isinstance(ignore, str) else []
    if isinstance(keys, dict):
        cols = []
        for base, k in keys.items():
            k = [k] if isinstance(k, str) else k
            cols.extend([c for c in df.columns if base in c and any(cc in c for cc in k)
                         and not any(i in c for i in ignore)])
    else:
        keys = [keys] if isinstance(keys, str) else keys
        cols = [c for c in df.columns if any(cc in c for cc in keys)
                and not any(i in c for i in ignore)]
    return cols


def key_search(search_items: Iterable[str], keys: List[str] | str,
               ignore: List[str] | str = None) -> List[str]:
    ignore = ignore if isinstance(ignore, list) else [ignore] if isinstance(ignore, str) else []
    keys = [keys] if isinstance(keys, str) else keys
    return [c for c in search_items if any(cc in c for cc in keys)
            and not any(i in c for i in ignore)]


def sort_columns(df, keys):
    def col_rank(col):
        for i, kw in enumerate(keys):
            if kw in col:
                return i
        return len(keys)
    return df[sorted(df.columns, key=col_rank)]


def get_column_names(names: str | Iterable[str]):
    if isinstance(names, str):
        return f'{names}_{PERIOD_MAP[names]}' if names in PERIOD_MAP else names
    return [f'{name}_{PERIOD_MAP[name]}' if name in PERIOD_MAP else name for name in names]


def get_column(df, name) -> pd.Series:
    return df[get_column_names(name)]


# ---------------------------------------------------------------------------
# Shared feature-engineering helpers (ported from v1)
# ---------------------------------------------------------------------------

def sig_span(window):
    return max(2, round(0.35 * window))


def fe_velocity(source, ema_windows, log=False, acceleration=False, z_norm=False, z_window=None):
    """EMA-based velocity (and optionally acceleration) for consecutive window pairs.

    source : pd.Series or dict[int, pd.Series]
        Series  -> EMAs are computed internally via ewm(span=w).
        dict    -> pre-computed EMA series keyed by window size (e.g. from DB).
    log     : False -> velocity = ema_fast - ema_slow (oscillators, OBV, AD)
              True  -> velocity = log(ema_fast / ema_slow) (price EMAs; values > 0)
    """
    if isinstance(source, dict):
        ema_cache = source
        index = next(iter(source.values())).index
    else:
        ema_cache = {w: source.ewm(span=w, adjust=False).mean() for w in ema_windows}
        index = source.index

    out = pd.DataFrame(index=index)
    for i in range(len(ema_windows) - 1):
        f, s = ema_windows[i], ema_windows[i + 1]
        fast, slow = ema_cache[f], ema_cache[s]
        vel = np.log(fast / slow) if log else (fast - slow)
        if z_norm:
            win = z_window if z_window is not None else s
            vel = (vel - vel.rolling(win).mean()) / vel.rolling(win).std()
        out[f'vel{f}_{s}'] = vel
        if acceleration:
            out[f'acc{f}_{s}'] = vel - vel.ewm(span=sig_span(s), adjust=False).mean()
    return out


def fe_oscillator_momentum(df, cols, ema_windows=None, feature_set='med'):
    """Velocity uses subtraction (ema_fast - ema_slow); for bounded/symmetric indicators."""
    assert feature_set in ['low', 'med', 'high']
    if ema_windows is None:
        ema_windows = [6, 16] if feature_set == 'low' else [4, 8, 16]
    out = pd.DataFrame(index=df.index)
    for ind in cols:
        out[f'{ind}_raw'] = df[ind]
        for i in range(len(ema_windows) - 1):
            f, s = ema_windows[i], ema_windows[i + 1]
            out[f'{ind}_ema{f}'] = e1 = df[ind].ewm(span=f, adjust=False).mean()
            out[f'{ind}_ema{s}'] = e2 = df[ind].ewm(span=s, adjust=False).mean()
            out[f'{ind}_vel{f}_{s}'] = md = e1 - e2
            if feature_set == 'high':
                out[f'{ind}_acc{f}_{s}'] = md - md.ewm(span=sig_span(s), adjust=False).mean()
    return sort_columns(out, ['raw', 'ema', 'vel', 'acc'])
