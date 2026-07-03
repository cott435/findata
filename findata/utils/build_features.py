"""Combine pre-scaled processor outputs into a single feature table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from findata.configs import DataSplits
from findata.preprocess import *


DEFAULT_PROCESSORS: tuple[type[PCAProcessor], ...] = (
    Volatility, Trend, Candle, Momentum, Volume,
)


@dataclass
class FeatureBundle:
    features: pd.DataFrame          # (ticker, date) indexed, all columns pre-scaled
    provenance: dict[str, str]      # column -> processor name


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}__{c}" for c in df.columns]
    return df


def _eligible_tickers(data: pd.DataFrame, data_splits: DataSplits) -> set[str]:
    """Tickers whose raw history covers at least half the training window.

    Decided once on raw start dates (not post-burn-in), since the ~rolling-window
    burn-in is negligible against half a multi-year training window. Held-out
    validation tickers are always retained. Returns the set to keep."""
    train_days = (data_splits.train_end - data_splits.train_start).days
    half_train_cutoff = data_splits.train_start + pd.Timedelta(days=train_days // 2)
    min_dates = data.groupby(level='ticker').apply(
        lambda x: x.index.get_level_values('date').min()
    )
    insufficient = set(min_dates[min_dates > half_train_cutoff].index)
    insufficient -= set(data_splits.val_holdout_tickers)
    if insufficient:
        print(f"Dropping {len(insufficient)} tickers (cover < half training window): {sorted(insufficient)}")
    all_tickers = set(min_dates.index)
    return all_tickers - insufficient


def build_features(
    data: pd.DataFrame,
    data_splits: DataSplits,
    processors: Sequence[type[PCAProcessor]] = DEFAULT_PROCESSORS,
    feature_set: str = "med",
    verbose: bool = False,
) -> FeatureBundle:
    """Run each processor with its main-block defaults, then horizontally concatenate
    their `.dataset` outputs. Column names are prefixed with the processor name so the
    origin of each feature is recoverable. All features are already scaled; downstream
    code should pass them through TimeSeriesDataSet as-is.

    Ticker eligibility is resolved globally here (once, on raw data) so every
    processor sees the same ticker set; the inner join then only reconciles
    per-processor leading rows from differing rolling-window burn-in."""
    keep = _eligible_tickers(data, data_splits)
    data = data[data.index.get_level_values('ticker').isin(keep)]

    frames: list[pd.DataFrame] = []
    provenance: dict[str, str] = {}
    for cls in processors:
        proc = cls(data, data_splits, feature_set=feature_set, verbose=verbose)
        ds = proc.dataset
        prefixed = _prefix_columns(ds, cls.__name__.lower())
        for col in prefixed.columns:
            provenance[col] = cls.__name__
        frames.append(prefixed)
    features = pd.concat(frames, axis=1, join="inner")
    features = features.sort_index()
    return FeatureBundle(features=features, provenance=provenance)
