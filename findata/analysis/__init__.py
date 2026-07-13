"""findata.analysis — feature evaluation tooling.

  rank_ic.py    RankICAnalysis: daily cross-sectional Spearman IC vs forward
                log returns, ICIR summaries, plots (fully vectorized)
  structure.py  CorrelationStructureAnalysis: MP spectrum, within/across-group
                correlation, data-driven clustering vs taxonomy
  search.py     ProcessingSearch: evaluate processing variants x horizons
  plotting.py   FeatureExplorer: interactive linked multi-panel time-series
                browser (HoloViews/Bokeh via Panel)

Depends on findata.preprocess (never the reverse).
"""
from findata.analysis.rank_ic import RankICAnalysis
from findata.analysis.structure import CorrelationStructureAnalysis
from findata.analysis.search import ProcessingSearch, DEFAULT_VARIANTS
from findata.analysis.plotting import FeatureExplorer

__all__ = ["RankICAnalysis", "CorrelationStructureAnalysis", "ProcessingSearch",
           "DEFAULT_VARIANTS", "FeatureExplorer"]
