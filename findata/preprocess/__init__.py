"""findata.preprocess v2 — feature engineering + composable panel processing.

Layout:
  base.py        FeatureGroup / ScaleRule + shared feature-engineering helpers
  oscillators.py    native-window oscillator momentum (v2 design)
  trend.py, volatility.py, volume.py, candles.py   v1 feature engineering, ported
  rmt.py         Marchenko-Pastur denoising + correlation clustering
  transforms.py  PanelTransforms (scaling, Kalman/SSA, ZCA/PCA/HPCA,
                 cross-sectional market-mode removal) + ProcessingConfig/Pipeline
  market_modes.py ModeDecomposer (global + sector PCA modes over the ticker
                 cross-section) + CrossSectionWhitener (ticker-space ZCA)
  variants.py    forecast_variants(): the named datasets the forecasting
                 search selects between
  pipeline.py    FeaturePipeline / PipelineState (the main entry points)
  v1/            frozen reference implementation
"""
from findata.preprocess import rmt
from findata.preprocess.feature_building.base import (FeatureGroup, LogStandardScaler, ScaleRule,
                                                      fe_oscillator_momentum, fe_velocity, get_column,
                                                      get_column_names, get_columns, get_scaler,
                                                      key_search, per_ticker, sig_span, sort_columns)
from .feature_building import *
from findata.preprocess.transforms import (STEP_REGISTRY, CrossSectionalNormalizer,
                                           GroupScaler, HierarchicalPCA, KalmanDenoiser,
                                           PCADecorrelator, PanelTransform,
                                           ProcessingConfig, ProcessingPipeline,
                                           SSADenoiser, SignalProjector, Standardizer,
                                           ZCAWhitener, train_row_mask)
from findata.preprocess.market_modes import CrossSectionWhitener, ModeDecomposer
from findata.preprocess.variants import forecast_variants
from findata.preprocess.pipeline import (DEFAULT_GROUPS, FeaturePipeline, PipelineState,
                                         eligible_tickers)

__all__ = [
    # groups
    "FeatureGroup", "Oscillators", "Trend", "Volatility", "Volume", "Candle",
    "DEFAULT_GROUPS",
    # pipeline
    "FeaturePipeline", "PipelineState", "eligible_tickers",
    # transforms
    "PanelTransform", "GroupScaler", "Standardizer", "KalmanDenoiser", "SSADenoiser",
    "CrossSectionalNormalizer", "SignalProjector", "PCADecorrelator", "ZCAWhitener",
    "HierarchicalPCA", "ModeDecomposer", "CrossSectionWhitener", "forecast_variants",
    "ProcessingConfig", "ProcessingPipeline", "STEP_REGISTRY",
    "train_row_mask",
    # scaling / helpers
    "ScaleRule", "get_scaler", "LogStandardScaler", "per_ticker",
    "get_columns", "key_search", "sort_columns", "get_column", "get_column_names",
    "fe_velocity", "fe_oscillator_momentum", "sig_span",
    # rmt module
    "rmt",
]
