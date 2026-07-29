"""WorkbenchApp: the four tabs glued into one served application."""

from __future__ import annotations

import logging

import panel as pn

from findata.analysis.plotting.base import ExplorerBase
from findata.analysis.plotting.workbench.compare_tab import CompareTab
from findata.analysis.plotting.workbench.data import WorkbenchData
from findata.analysis.plotting.workbench.market_tab import MarketTab
from findata.analysis.plotting.workbench.pca_tab import PcaTab
from findata.analysis.plotting.workbench.stock_tab import StockTab

logger = logging.getLogger(__name__)


class WorkbenchApp(ExplorerBase):
    """Full analysis workbench.

    ``data`` (a WorkbenchData) is shared read-only across sessions; every
    widget and DynamicMap is constructed per session inside ``_make_app``
    (the contract in findata/analysis/plotting/base.py). Tabs render
    dynamically, so a tab's first data pulls happen on first visit.
    """

    def __init__(self, data: WorkbenchData):
        self.data = data

    def _make_app(self) -> pn.Tabs:
        logger.info('New workbench session (%d tickers)', len(self.data.universe))
        return pn.Tabs(
            ('Stock', StockTab(self.data).build()),
            ('Compare', CompareTab(self.data).build()),
            ('PCA/RMT', PcaTab(self.data).build()),
            ('Market', MarketTab(self.data).build()),
            dynamic=True)
