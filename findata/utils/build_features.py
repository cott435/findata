"""Back-compat wrapper over findata.preprocess.FeaturePipeline (v2).

Public surface kept from v1: ``build_features``, ``apply_features``,
``FeatureBundle``, ``FeatureState``. Differences from v1:

- Default output is SCALE-ONLY features (no baked-in grouped/final PCA — pick
  transforms explicitly via ``processing=ProcessingConfig(...)``; the
  processing search in ``findata.analysis.search`` is how you choose them).
- ``FeatureState`` is :class:`findata.preprocess.PipelineState`: self-describing
  (fitted transforms are stored directly), so the v1 ``PROCESSOR_REGISTRY`` is
  gone and custom groups need no registration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from findata.preprocess import FeaturePipeline, PipelineState, ProcessingConfig

# v1 name for the saved, fitted featurization state
FeatureState = PipelineState


@dataclass
class FeatureBundle:
    features: pd.DataFrame          # (ticker, date) indexed feature panel
    provenance: dict[str, str]      # column -> group name ('component' for pc outputs)
    state: PipelineState | None = None  # fitted transforms, for production re-apply


def build_features(
    data: pd.DataFrame,
    data_splits,
    processors=None,
    feature_set: str = "med",
    verbose: bool = False,
    feat_eng: bool = False,
    processing: ProcessingConfig | None = None,
) -> FeatureBundle:
    """Fit a FeaturePipeline and return the combined feature table.

    ``processors`` accepts FeatureGroup classes or instances (default: the five
    v2 groups). ``feat_eng=True`` returns the engineered (unscaled) panel, as in
    v1. ``processing`` optionally applies a transform chain after scaling.
    """
    pipe = FeaturePipeline(groups=processors, processing=processing,
                           feature_set=feature_set, verbose=verbose)
    pipe.fit(data, data_splits)
    features = pipe.engineered_ if feat_eng else pipe.features_
    return FeatureBundle(features=features, provenance=pipe.provenance,
                         state=pipe.get_state())


def apply_features(state_path: str | Path, data: pd.DataFrame,
                   verbose: bool = False, allow_non_causal: bool = False) -> pd.DataFrame:
    """Convenience: load a saved FeatureState/PipelineState and apply it to new raw data."""
    return PipelineState.load(state_path).apply(data, allow_non_causal=allow_non_causal)
