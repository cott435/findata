"""Multi-tab analysis workbench (Panel + HoloViews).

Launch via ``scripts/workbench.py``. Layout:
  data.py        WorkbenchData -- process-wide read-only data service + caches
  selectors.py   hierarchical indicator picker driven by the calculator registry
  presets.py     named default views rendered as buttons
  panels.py      element builders (candlestick, overlays, heatmaps, IC plots)
  stock_tab.py   single-stock explorer (price + indicator panels + transforms)
  compare_tab.py cross-ticker comparison + rank-IC analytics
  pca_tab.py     PCA / RMT structure view
  market_tab.py  production market overview (sectors, breadth, residual movers)
  app.py         WorkbenchApp gluing the tabs together

Session rules follow findata/analysis/plotting/base.py: every widget and
DynamicMap is built per session inside ``_make_app``/tab ``build`` calls;
WorkbenchData is the only shared (read-only) state.
"""

from findata.analysis.plotting.workbench.app import WorkbenchApp
from findata.analysis.plotting.workbench.data import WorkbenchData

__all__ = ['WorkbenchApp', 'WorkbenchData']
