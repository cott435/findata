"""Combine pre-scaled processor outputs into a single feature table."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

from findata.configs import DataSplits
from findata.preprocess import *


DEFAULT_PROCESSORS: tuple[type[PCAProcessor], ...] = (
    Volatility, Trend, Candle, Momentum, Volume,
)

# Class-name -> class lookup used by FeatureState.apply. Custom processors passed to
# build_features must be registered here to be re-instantiable at apply time.
PROCESSOR_REGISTRY: dict[str, type[PCAProcessor]] = {
    cls.__name__: cls for cls in DEFAULT_PROCESSORS
}


@dataclass
class FeatureState:
    """Fitted preprocessing state from one `build_features` call — enough to project
    new raw data into the exact feature space a model was trained on (scalers and
    PCAs are reused, never refit)."""
    processor_states: dict[str, dict]   # class name -> PCAProcessor.get_state()
    feature_cols: list[str]             # final prefixed column order
    feature_set: str
    version: int = 1

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureState":
        state = joblib.load(Path(path))
        if not isinstance(state, cls):
            raise TypeError(f"{path} does not contain a FeatureState (got {type(state).__name__})")
        return state

    def apply(self, data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Transform new raw data with the fitted transformers.

        `data` is the same wide (ticker, date)-indexed frame `get_all_data` returns and
        must include enough history before the first date you need features for: rows
        inside each processor's rolling burn-in window (~210 trading days for the 'med'
        feature set, ~300 for 'high') come out NaN and are dropped.
        """
        frames: list[pd.DataFrame] = []
        for name, proc_state in self.processor_states.items():
            cls_ = PROCESSOR_REGISTRY.get(name)
            if cls_ is None:
                raise KeyError(
                    f"Processor {name!r} is not in PROCESSOR_REGISTRY; register the class "
                    "in findata.utils.build_features before applying this state."
                )
            proc = cls_(data, state=proc_state, verbose=verbose)
            frames.append(_prefix_columns(proc.dataset, name.lower()))
        features = pd.concat(frames, axis=1, join="inner").sort_index()
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            raise ValueError(f"Applied features are missing fitted columns: {missing}")
        return features[self.feature_cols]


def apply_features(state_path: str | Path, data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Convenience: load a saved FeatureState and apply it to new raw data."""
    return FeatureState.load(state_path).apply(data, verbose=verbose)


@dataclass
class FeatureBundle:
    features: pd.DataFrame          # (ticker, date) indexed, all columns pre-scaled
    provenance: dict[str, str]      # column -> processor name
    state: FeatureState | None = None  # fitted transformers, for production re-apply


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
    feat_eng=False
) -> FeatureBundle:
    """Run each processor with its main-block defaults, then horizontally concatenate
    their `.dataset` outputs. Column names are prefixed with the processor name so the
    origin of each feature is recoverable. All features are already scaled; downstream
    code should pass them through TimeSeriesDataSet as-is.

    Ticker eligibility is resolved globally here (once, on raw data) so every
    processor sees the same ticker set; the inner join then only reconciles
    per-processor leading rows from differing rolling-window burn-in.

    The returned bundle's `state` holds every fitted scaler/PCA — save it with the
    trained model and use `FeatureState.apply` to featurize new data at inference."""
    keep = _eligible_tickers(data, data_splits)
    data = data[data.index.get_level_values('ticker').isin(keep)]

    frames: list[pd.DataFrame] = []
    provenance: dict[str, str] = {}
    processor_states: dict[str, dict] = {}
    for cls in processors:
        proc = cls(data, data_splits, feature_set=feature_set, verbose=verbose)
        ds = proc.feat_eng_data if feat_eng else proc.dataset
        prefixed = _prefix_columns(ds, cls.__name__.lower())
        for col in prefixed.columns:
            provenance[col] = cls.__name__
        frames.append(prefixed)
        processor_states[cls.__name__] = proc.get_state()
    features = pd.concat(frames, axis=1, join="inner")
    features = features.sort_index()
    state = FeatureState(
        processor_states=processor_states,
        feature_cols=list(features.columns),
        feature_set=feature_set,
    )
    return FeatureBundle(features=features, provenance=provenance, state=state)
