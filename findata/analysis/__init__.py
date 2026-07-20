"""findata.analysis — feature evaluation tooling.

  rank_ic.py    RankICAnalysis: daily cross-sectional Spearman IC vs forward
                log returns, ICIR summaries, plots (fully vectorized)
  structure.py  CorrelationStructureAnalysis: MP spectrum, within/across-group
                correlation, data-driven clustering vs taxonomy
  search.py     ProcessingSearch: evaluate processing variants x horizons
  plotting/     interactive explorers (FeatureExplorer, TransformExplorer,
                PredictionExplorer) + static savers (HoloViews/Bokeh via Panel)

Depends on findata.preprocess (never the reverse).
"""
from findata.analysis.rank_ic import RankICAnalysis
from findata.analysis.correlation_structure import CorrelationStructureAnalysis
from findata.analysis.search import ProcessingSearch, DEFAULT_VARIANTS
from findata.analysis.plotting import (FeatureExplorer, PredictionExplorer,
                                       TransformExplorer, save_calibration,
                                       save_fan_chart, save_layer_plot,
                                       save_modes_plot, save_pred_scatter,
                                       save_prediction_bands)
from findata.analysis import market_structure

__all__ = ["RankICAnalysis", "CorrelationStructureAnalysis", "ProcessingSearch",
           "DEFAULT_VARIANTS", "FeatureExplorer", "TransformExplorer",
           "PredictionExplorer", "save_layer_plot", "save_modes_plot",
           "save_fan_chart", "save_prediction_bands", "save_calibration",
           "save_pred_scatter", "market_structure"]
