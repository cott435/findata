"""FeaturePipeline — the main class: runs every feature group, concatenates
their engineered columns (``{group}__{col}`` prefix), and applies one
ProcessingPipeline across the combined panel.

The combined-panel design is deliberate: cross-feature transforms (ZCA,
hierarchical PCA, market-mode removal) only make sense on ALL features at
once, which v1's per-group PCA could never express.

State: :class:`PipelineState` captures the (param-only) group instances plus
the fitted ProcessingPipeline — enough to featurize new raw data exactly as at
fit time, with a causality gate for non-causal (SSA) configs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

from findata.preprocess.base import FeatureGroup, ScaleRule, date_values
from findata.preprocess.candles import Candle
from findata.preprocess.oscillators import Oscillators
from findata.preprocess.transforms import ProcessingConfig, ProcessingPipeline
from findata.preprocess.trend import Trend
from findata.preprocess.volatility import Volatility
from findata.preprocess.volume import Volume

DEFAULT_GROUPS = (Oscillators, Trend, Volatility, Volume, Candle)


def eligible_tickers(data: pd.DataFrame, data_splits) -> set[str]:
    """Tickers whose raw history covers at least half the training window.

    Decided once on raw start dates (the rolling burn-in is negligible against
    half a multi-year training window). Held-out validation tickers are always
    retained.
    """
    train_days = (data_splits.train_end - data_splits.train_start).days
    half_train_cutoff = pd.Timestamp(data_splits.train_start) + pd.Timedelta(days=train_days // 2)
    dates = pd.Series(date_values(data.index),
                      index=data.index.get_level_values('ticker'))
    min_dates = dates.groupby(level=0).min()
    insufficient = set(min_dates[min_dates > half_train_cutoff].index)
    insufficient -= set(data_splits.val_holdout_tickers)
    if insufficient:
        print(f"Dropping {len(insufficient)} tickers (cover < half training window): "
              f"{sorted(insufficient)}")
    return set(min_dates.index) - insufficient


def _as_group_instances(groups, feature_set: str, verbose: bool) -> list[FeatureGroup]:
    groups = DEFAULT_GROUPS if groups is None else groups
    out = []
    for g in groups:
        out.append(g if isinstance(g, FeatureGroup)
                   else g(feature_set=feature_set, verbose=verbose))
    names = [g.name for g in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate group names: {names}")
    return out


def _engineer(groups: Sequence[FeatureGroup], data: pd.DataFrame) -> pd.DataFrame:
    """Run every group and concatenate with the ``{group}__`` prefix.

    The double-underscore prefix is the taxonomy the structure analysis parses
    (split on first '__'), so keep it stable.
    """
    frames = []
    for group in groups:
        eng = group.engineer(data)
        eng.columns = [f"{group.name}__{c}" for c in eng.columns]
        frames.append(eng)
    return pd.concat(frames, axis=1, join="inner").sort_index()


def _align_columns(eng: pd.DataFrame, expected: list[str], context: str) -> pd.DataFrame:
    missing = [c for c in expected if c not in eng.columns]
    if missing:
        raise ValueError(
            f"{context}: engineered columns don't match the fitted state "
            f"(missing={missing}). Same group params / raw columns as at fit time?")
    return eng[expected]


class FeaturePipeline:
    """engineer -> (train-fitted) processing across all groups at once.

    processing:
      None                      -> scale-only (each group's ScaleRules)
      ProcessingConfig          -> compiled against the groups' ScaleRules
      ProcessingPipeline        -> used as given (advanced)
    """

    def __init__(self, groups=None, processing: ProcessingConfig | ProcessingPipeline | None = None,
                 feature_set: str = "med", verbose: bool = False):
        self.feature_set = feature_set
        self.verbose = verbose
        self.groups = _as_group_instances(groups, feature_set, verbose)
        self.processing = processing

    # -- feature engineering ------------------------------------------------

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        return _engineer(self.groups, data)

    def _resolve_burn_in(self, eng: pd.DataFrame, data: pd.DataFrame, splits) -> pd.DataFrame:
        """Drop rolling burn-in NaN rows; warn if NaNs come from tickers whose
        raw history should have covered the pre-train window (v1 logic)."""
        if splits is not None:
            raw_dates = pd.Series(date_values(data.index),
                                  index=data.index.get_level_values('ticker'))
            raw_min_dates = raw_dates.groupby(level=0).min()
            suspect = set(raw_min_dates[raw_min_dates > pd.Timestamp(splits.data_start)].index)
            nan_mask = eng.isna().any(axis=1)
            nan_tickers = set(eng[nan_mask].index.get_level_values('ticker').unique())
            unexpected = nan_tickers - suspect
            if unexpected and self.verbose:
                print(f"  Dropping NaN rows from unexpected tickers: {sorted(unexpected)}")
        return eng.dropna()

    def scale_rules(self) -> list[ScaleRule]:
        """Every group's rules, with columns prefixed to match the combined panel."""
        rules = []
        for group in self.groups:
            cols = [c.split('__', 1)[1] for c in self.engineered_cols_
                    if c.startswith(f"{group.name}__")]
            rules.extend(r.with_prefix(f"{group.name}__") for r in group.scale_rules(cols))
        return rules

    # -- fit / transform -----------------------------------------------------

    def fit(self, data: pd.DataFrame, splits) -> "FeaturePipeline":
        keep = eligible_tickers(data, splits)
        data = data[data.index.get_level_values('ticker').isin(keep)]

        eng = self.engineer(data)
        eng = eng[date_values(eng.index) >= pd.Timestamp(splits.train_start)]
        eng = self._resolve_burn_in(eng, data, splits)
        self.engineered_ = eng
        self.engineered_cols_ = list(eng.columns)
        self.splits = splits

        if isinstance(self.processing, ProcessingPipeline):
            self.processing_ = self.processing
        else:
            config = self.processing if self.processing is not None else ProcessingConfig("scaled")
            self.processing_ = config.build(self.scale_rules())
        self.features_ = self.processing_.fit_transform(eng, splits)

        self.provenance = {
            col: (col.split('__', 1)[0] if '__' in col else 'component')
            for col in self.features_.columns
        }
        if self.verbose:
            print(f"FeaturePipeline: {len(self.engineered_cols_)} engineered -> "
                  f"{self.features_.shape[1]} features ({self.processing_.name}), "
                  f"{len(keep)} tickers, causal={self.processing_.causal}")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Featurize new raw data with the fitted transforms (nothing refits)."""
        eng = _engineer(self.groups, data).dropna()
        eng = _align_columns(eng, self.engineered_cols_, type(self).__name__)
        return self.processing_.transform(eng)

    def fit_transform(self, data: pd.DataFrame, splits) -> pd.DataFrame:
        return self.fit(data, splits).features_

    def get_state(self) -> "PipelineState":
        return PipelineState(
            groups=self.groups,
            processing=self.processing_,
            engineered_cols=self.engineered_cols_,
            feature_cols=list(self.features_.columns),
            feature_set=self.feature_set,
            causal=self.processing_.causal,
        )


@dataclass
class PipelineState:
    """Fitted preprocessing state — enough to project new raw data into the
    exact feature space a model was trained on (nothing refits at apply time).

    Groups are param-only instances and the ProcessingPipeline holds the fitted
    scalers/filters/projections, so the state is self-describing (no class
    registry needed, unlike v1).
    """
    groups: list
    processing: ProcessingPipeline
    engineered_cols: list
    feature_cols: list
    feature_set: str
    causal: bool
    version: int = 2

    def save(self, path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path) -> "PipelineState":
        state = joblib.load(Path(path))
        if not isinstance(state, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__} "
                            f"(got {type(state).__name__})")
        return state

    def apply(self, data: pd.DataFrame, allow_non_causal: bool = False) -> pd.DataFrame:
        """Transform new raw data. ``data`` must include enough history before
        the first date you need features for (rolling burn-in rows drop as NaN).

        Raises for non-causal states (e.g. SSA in the chain) unless explicitly
        overridden — batch-SVD output at time t uses the future.
        """
        if not self.causal and not allow_non_causal:
            raise ValueError(
                "This pipeline state contains non-causal transforms (e.g. SSA): "
                "its features at time t use future data, so applying it for "
                "production/backtesting leaks. Pass allow_non_causal=True to "
                "override for diagnostics.")
        eng = _engineer(self.groups, data).dropna()
        eng = _align_columns(eng, self.engineered_cols, "PipelineState.apply")
        features = self.processing.transform(eng)
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            raise ValueError(f"Applied features are missing fitted columns: {missing}")
        return features[self.feature_cols]
