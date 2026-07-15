"""findata.preprocess v2 — feature engineering + composable panel processing.

Layout:
  base.py        FeatureGroup / ScaleRule + shared feature-engineering helpers
  oscillators.py    native-window oscillator momentum (v2 design)
  trend.py, volatility.py, volume.py, candles.py   v1 feature engineering, ported
  rmt.py         Marchenko-Pastur denoising + correlation clustering
  transforms.py  PanelTransforms (scaling, Kalman/SSA, ZCA/PCA/HPCA,
                 cross-sectional market-mode removal) + ProcessingConfig/Pipeline
  pipeline.py    FeaturePipeline / PipelineState (the main entry points)
  v1/            frozen reference implementation
"""
from findata.preprocess import rmt
from findata.preprocess.base import (FeatureGroup, LogStandardScaler, ScaleRule,
                                     fe_oscillator_momentum, fe_velocity, get_column,
                                     get_column_names, get_columns, get_scaler,
                                     key_search, per_ticker, sig_span, sort_columns)
from findata.preprocess.candles import Candle
from findata.preprocess.oscillators import Oscillators
from findata.preprocess.trend import Trend
from findata.preprocess.volatility import Volatility
from findata.preprocess.volume import Volume
from findata.preprocess.transforms import (STEP_REGISTRY, CrossSectionalNormalizer,
                                           GroupScaler, HierarchicalPCA, KalmanDenoiser,
                                           PCADecorrelator, PanelTransform,
                                           ProcessingConfig, ProcessingPipeline,
                                           SSADenoiser, SignalProjector, Standardizer,
                                           ZCAWhitener, train_row_mask)
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
    "HierarchicalPCA", "ProcessingConfig", "ProcessingPipeline", "STEP_REGISTRY",
    "train_row_mask",
    # scaling / helpers
    "ScaleRule", "get_scaler", "LogStandardScaler", "per_ticker",
    "get_columns", "key_search", "sort_columns", "get_column", "get_column_names",
    "fe_velocity", "fe_oscillator_momentum", "sig_span",
    # rmt module
    "rmt",
]
