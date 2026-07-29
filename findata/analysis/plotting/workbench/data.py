"""WorkbenchData: the workbench's shared, read-only data service.

One instance per process, shared across browser sessions (pn.serve runs a
single loop, so plain dict caches are safe). Selection keys are strings:
``'rsi_14'`` / ``'obv'`` for technical storage columns, ``'fund:<item>'``
for derived fundamentals -- panels come back as DatetimeIndex x ticker
frames ready for plotting.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from findata.analysis.rank_ic import RankICAnalysis
from findata.database.db_manager import DBManager
from findata.preprocess.calculators.base import column_name, parse_column
from findata.utils.timing import log_timing, timed

logger = logging.getLogger(__name__)

FUND_PREFIX = 'fund:'


def key_label(key: str) -> str:
    return key[len(FUND_PREFIX):] + ' (fund)' if key.startswith(FUND_PREFIX) else key


class WorkbenchData:
    def __init__(self, db: DBManager, universe: list, start: str = '2013-01-01'):
        self.db = db
        self.universe = list(universe)
        self.start = start
        self.meta = db.get_ticker_meta(self.universe)
        self.sectors = self.meta['sector'].fillna('ETF')
        self._panels: dict = {}       # key -> date x ticker frame
        self._frames: dict = {}       # ticker -> wide per-ticker frame
        self._decomps: dict = {}      # key -> decompose_modes dict
        self._ic: dict = {}           # (key, horizon) -> (daily_ic, summary)
        self._close = None
        self._returns = None
        self._breadth = None

    # ------------------------------------------------------------------ #
    # panels
    # ------------------------------------------------------------------ #

    @log_timing(logger)
    def close_panel(self) -> pd.DataFrame:
        if self._close is None:
            wide = self.db.get_items(self.universe, {'close': None}, start=self.start)
            panel = wide['close'].unstack('ticker')
            panel.index = pd.to_datetime(panel.index)
            self._close = panel.sort_index()
        return self._close

    def returns_panel(self) -> pd.DataFrame:
        if self._returns is None:
            self._returns = np.log(self.close_panel()).diff()
        return self._returns

    @log_timing(logger)
    def item_panel(self, key: str) -> pd.DataFrame:
        """Universe panel (date x ticker) for one selection key."""
        if key in self._panels:
            return self._panels[key]
        if key == 'log_ret':  # pseudo-key for the daily log-return panel
            self._panels[key] = self.returns_panel()
            return self._panels[key]
        if key.startswith(FUND_PREFIX):
            item = key[len(FUND_PREFIX):]
            wide = self.db.get_fundamentals(self.universe, items=[item],
                                            start=self.start, daily=True)
            if wide.empty or item not in wide.columns:
                panel = pd.DataFrame(index=self.close_panel().index)
            else:
                panel = wide[item].unstack('ticker')
        else:
            item, period = parse_column(key)
            wide = self.db.get_items(self.universe, {item: period or None},
                                     start=self.start)
            col = column_name(item, period)
            panel = wide[col].unstack('ticker')
        panel.index = pd.to_datetime(panel.index)
        self._panels[key] = panel.sort_index()
        return self._panels[key]

    @log_timing(logger)
    def ticker_frame(self, ticker: str) -> pd.DataFrame:
        """Everything stored for one ticker (prices + technicals + fund_*)."""
        if ticker not in self._frames:
            frame = self.db.get_all_data(ticker, start=self.start,
                                         include_fundamentals=True)
            frame.index = pd.to_datetime(frame.index)
            self._frames[ticker] = frame
        return self._frames[ticker]

    def ticker_series(self, ticker: str, key: str, persist: bool = None) -> pd.Series:
        """One selection key for one ticker; computes on the fly when the
        column isn't stored (persistence per the checkbox / save policy)."""
        frame = self.ticker_frame(ticker)
        if key.startswith(FUND_PREFIX):
            col = 'fund_' + key[len(FUND_PREFIX):]
            return frame[col] if col in frame.columns else pd.Series(dtype=float)
        if key in frame.columns:
            return frame[key]
        item, period = parse_column(key)
        got = self.db.get_items(ticker, {item: period or None}, start=self.start,
                                persist=persist)
        if key not in got.columns:
            return pd.Series(dtype=float)
        series = got[key]
        series.index = pd.to_datetime(series.index)
        if persist is not False:  # cached frame can now serve it
            self._frames.pop(ticker, None)
        return series

    # ------------------------------------------------------------------ #
    # decompositions + IC
    # ------------------------------------------------------------------ #

    @log_timing(logger)
    def decompose(self, key: str) -> dict:
        """Global + sector mode decomposition of the key's universe panel.

        The decomposition needs a complete block: tickers under 80% coverage
        are dropped, then remaining dates with any gap.
        """
        from findata.analysis.market_structure import decompose_modes
        if key not in self._decomps:
            panel = self.item_panel(key)
            panel = panel.dropna(axis=1, thresh=int(len(panel) * 0.8))
            panel = panel.dropna(axis=0, how='any')
            dropped = len(self.universe) - panel.shape[1]
            if dropped:
                logger.info('decompose(%s): %d low-coverage ticker(s) excluded',
                            key_label(key), dropped)
            labels = self.sectors.reindex(panel.columns).fillna('ETF')
            self._decomps[key] = decompose_modes(panel, labels)
        return self._decomps[key]

    @log_timing(logger)
    def fwd_ic(self, key: str, horizon: int) -> tuple:
        """(daily rank-IC Series, summary row) of the key vs forward returns."""
        cache_key = (key, horizon)
        if cache_key not in self._ic:
            analysis = RankICAnalysis()
            panel = self.item_panel(key)
            features = panel.stack().rename(key_label(key)).to_frame()
            features.index = features.index.set_names(['date', 'ticker']).reorder_levels(
                ['ticker', 'date'])
            closes = self.close_panel().stack().rename('close')
            closes.index = closes.index.set_names(['date', 'ticker']).reorder_levels(
                ['ticker', 'date'])
            fwd = analysis.forward_log_returns(closes, horizon)
            daily = analysis.daily_ic(features, analysis.prepare_target(fwd, features.index))
            summary = analysis.summarize(daily, horizon)
            self._ic[cache_key] = (daily.iloc[:, 0], summary.iloc[0])
        return self._ic[cache_key]

    def ic_decay(self, key: str, horizons: tuple = (1, 5, 10, 21, 42, 63)) -> pd.DataFrame:
        rows = {}
        for h in horizons:
            _, summary = self.fwd_ic(key, h)
            rows[h] = {'ic_mean': summary['ic_mean'], 'icir': summary['icir']}
        return pd.DataFrame(rows).T.rename_axis('horizon')

    # ------------------------------------------------------------------ #
    # market overview inputs
    # ------------------------------------------------------------------ #

    @log_timing(logger)
    def breadth(self) -> pd.DataFrame:
        """Share of the universe above its ema_close_26 / _52."""
        if self._breadth is None:
            wide = self.db.get_items(self.universe,
                                     {'close': None, 'ema_close': [26, 52]},
                                     start=self.start)
            out = {}
            for period in (26, 52):
                above = wide['close'] > wide[f'ema_close_{period}']
                share = above.groupby(level='date').mean()
                out[f'share_above_ema{period}'] = share
            breadth = pd.DataFrame(out)
            breadth.index = pd.to_datetime(breadth.index)
            self._breadth = breadth.sort_index()
        return self._breadth

    def sector_returns(self, asof: pd.Timestamp, horizons: tuple) -> pd.DataFrame:
        """Mean log return per sector over each trailing horizon ending asof."""
        returns = self.returns_panel().loc[:pd.Timestamp(asof)]
        rows = {}
        for h in horizons:
            window = returns.iloc[-h:].sum()
            rows[f'{h}d'] = window.groupby(self.sectors.reindex(window.index)).mean()
        return pd.DataFrame(rows)

    @log_timing(logger)
    def residual_movers(self, asof: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """Per-ticker horizon return split into market+sector vs residual.

        The decomposition runs on standardized returns, so components are
        rescaled by each ticker's daily return vol to land back in log-return
        units ('ret' is the actual summed log return). Flags names whose
        residual runs against (or without) the market move -- 'moving on
        its own'.
        """
        decomp = self.decompose('log_ret')
        panels = decomp['panels']
        sigma = self.returns_panel().std()
        window = slice(None, pd.Timestamp(asof))
        market_std = (panels['combined_modes'].loc[window].iloc[-horizon:].sum())
        resid_std = panels['residual'].loc[window].iloc[-horizon:].sum()
        market = market_std * sigma.reindex(market_std.index)
        resid = resid_std * sigma.reindex(resid_std.index)
        total = self.returns_panel().loc[window].iloc[-horizon:].sum()
        out = pd.DataFrame({'ret': total, 'market_sector': market, 'residual': resid})
        out = out.dropna(subset=['residual'])
        out['sector'] = self.sectors.reindex(out.index)
        threshold = out['residual'].abs().quantile(0.9)
        out['own_move'] = (np.sign(out['residual']) != np.sign(out['market_sector'])) \
            & (out['residual'].abs() >= threshold)
        return out.sort_values('residual', key=abs, ascending=False)

    def spectrum(self, key: str, tickers: list = None, date_range=None) -> dict:
        from findata.analysis import market_structure
        panel = self.item_panel(key)
        if tickers is not None:
            panel = panel[[t for t in tickers if t in panel.columns]]
        if date_range is not None:
            panel = panel.loc[date_range[0]:date_range[1]]
        panel = panel.dropna(axis=1, thresh=max(10, int(len(panel) * 0.8)))
        panel = panel.dropna(axis=0, how='any')
        with timed(logger, f'spectrum({key_label(key)}, {panel.shape})'):
            return market_structure.spectrum(panel)
