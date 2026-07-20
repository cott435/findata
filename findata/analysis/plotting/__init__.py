"""findata.analysis.plotting — interactive explorers + static savers.

  base.py                 session machinery shared by every explorer
                          (SessionSync, minimap_opts, ExplorerBase)
  style.py                the layer/ticker visual encoding (palette, dashes)
  feature_explorer.py     FeatureExplorer — linked per-group feature browser
  transform_explorer.py   TransformExplorer — transform/mode reward paths
  prediction_explorer.py  PredictionExplorer — forecast vs realized quantiles
  static.py               matplotlib savers (layers, modes, fan charts,
                          prediction bands, calibration, scatter)
"""
from findata.analysis.plotting.base import (NORMALIZATIONS, ExplorerBase,
                                            SessionSync, minimap_opts)
from findata.analysis.plotting.style import LAYER_PALETTE, series_style
from findata.analysis.plotting.feature_explorer import FeatureExplorer
from findata.analysis.plotting.transform_explorer import TransformExplorer
from findata.analysis.plotting.prediction_explorer import (PredictionExplorer,
                                                           apply_scale, band_pairs,
                                                           pred_quantiles,
                                                           quantile_col, to_per_step)
from findata.analysis.plotting.static import (save_calibration, save_fan_chart,
                                              save_layer_plot, save_modes_plot,
                                              save_pred_scatter,
                                              save_prediction_bands)

__all__ = [
    "ExplorerBase", "SessionSync", "minimap_opts", "NORMALIZATIONS",
    "LAYER_PALETTE", "series_style",
    "FeatureExplorer", "TransformExplorer", "PredictionExplorer",
    "pred_quantiles", "quantile_col", "band_pairs", "apply_scale", "to_per_step",
    "save_layer_plot", "save_modes_plot", "save_fan_chart",
    "save_prediction_bands", "save_calibration", "save_pred_scatter",
]
