"""findata.preprocess v2 — feature engineering + composable panel processing.

Layout:
  calculators/   Calculator registry: technical + fundamental calculators
                 (import-light; the database layer depends on it)
  feature_building/
    base.py      FeatureGroup / ScaleRule + shared feature-engineering helpers
    oscillators.py  native-window oscillator momentum (v2 design)
    trend.py, volatility.py, volume.py, candles.py  v1 feature engineering, ported
  rmt.py         Marchenko-Pastur denoising + correlation clustering
  transforms.py  PanelTransforms (scaling, Kalman/SSA, ZCA/PCA/HPCA,
                 cross-sectional market-mode removal) + ProcessingConfig/Pipeline
  market_modes.py ModeDecomposer (global + sector PCA modes over the ticker
                 cross-section) + CrossSectionWhitener (ticker-space ZCA)
  variants.py    forecast_variants(): the named datasets the forecasting
                 search selects between
  pipeline.py    FeaturePipeline / PipelineState (the main entry points)
  v1/            frozen reference implementation

Public names resolve lazily (PEP 562) so import-light consumers — most
importantly the database layer importing ``calculators`` — don't pay the
sklearn import cost of the processing stack. Resolving any name from the
transform/pipeline stack also imports market_modes, whose import side-effect
registers the 'modes'/'cs_whiten' STEP_REGISTRY entries.
"""

import importlib

_BASE = 'findata.preprocess.feature_building.base'
_GROUPS = 'findata.preprocess.feature_building'
_PIPELINE = 'findata.preprocess.pipeline'
_TRANSFORMS = 'findata.preprocess.transforms'
_MODES = 'findata.preprocess.market_modes'
_VARIANTS = 'findata.preprocess.variants'

_LAZY = {
    # feature groups
    'FeatureGroup': _GROUPS, 'Oscillators': _GROUPS, 'Trend': _GROUPS,
    'Volatility': _GROUPS, 'Volume': _GROUPS, 'Candle': _GROUPS,
    'Fundamentals': _GROUPS,
    # pipeline
    'FeaturePipeline': _PIPELINE, 'PipelineState': _PIPELINE,
    'eligible_tickers': _PIPELINE, 'DEFAULT_GROUPS': _PIPELINE,
    # transforms
    'PanelTransform': _TRANSFORMS, 'GroupScaler': _TRANSFORMS,
    'Standardizer': _TRANSFORMS, 'KalmanDenoiser': _TRANSFORMS,
    'SSADenoiser': _TRANSFORMS, 'CrossSectionalNormalizer': _TRANSFORMS,
    'SignalProjector': _TRANSFORMS, 'PCADecorrelator': _TRANSFORMS,
    'ZCAWhitener': _TRANSFORMS, 'HierarchicalPCA': _TRANSFORMS,
    'ProcessingConfig': _TRANSFORMS, 'ProcessingPipeline': _TRANSFORMS,
    'STEP_REGISTRY': _TRANSFORMS, 'train_row_mask': _TRANSFORMS,
    # market modes
    'ModeDecomposer': _MODES, 'CrossSectionWhitener': _MODES,
    'forecast_variants': _VARIANTS,
    # scaling / helpers
    'ScaleRule': _BASE, 'get_scaler': _BASE, 'LogStandardScaler': _BASE,
    'per_ticker': _BASE, 'date_values': _BASE,
    'get_columns': _BASE, 'key_search': _BASE, 'sort_columns': _BASE,
    'get_column': _BASE, 'get_column_names': _BASE,
    'fe_velocity': _BASE, 'fe_oscillator_momentum': _BASE, 'sig_span': _BASE,
    # submodules exposed as attributes
    'rmt': 'findata.preprocess.rmt',
    'calculators': 'findata.preprocess.calculators',
}

# modules whose consumers rely on the full STEP_REGISTRY (market_modes adds
# 'modes'/'cs_whiten' to it as an import side-effect)
_REGISTRY_COUPLED = {_GROUPS, _BASE, _PIPELINE, _TRANSFORMS, _VARIANTS}

_SUBMODULE_NAMES = {'rmt', 'calculators'}

__all__ = list(_LAZY)


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'findata.preprocess' has no attribute {name!r}")
    module = importlib.import_module(target)
    if target in _REGISTRY_COUPLED:
        importlib.import_module(_MODES)
    value = module if name in _SUBMODULE_NAMES else getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
