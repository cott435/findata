"""Fundamental calculators: derived financial-statement items.

``build_quarter_panel`` merges one ticker's stored statements into a
quarterly frame with TTM/market-cap helpers; seven registered
:class:`FundamentalCalculator` groups derive the modeling catalog from it;
``compute_fundamentals`` runs everything and emits the long point-in-time
frame stored in ``fundamental_data`` (observation date = filed_date of the
newest input a value uses, so daily forward-fills never look ahead).

Flow items are trailing-4-quarter (TTM) sums requiring all four quarters;
stocks are point-in-time; YoY growth compares the same fiscal quarter.
Division blowups become NaN; the FE bounds/scaling on each OutputSpec say
how the feature layer should tame the remaining tails.

Like the rest of this package: stdlib + numpy/pandas only, no DB imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from findata.preprocess.calculators.base import (Calculator, OutputSpec,
                                                 register)

# income/cashflow items summed into ttm_<item>; balance items stay point-in-time
FLOW_ITEMS = ('revenue', 'cost_of_revenue', 'gross_profit', 'operating_expenses',
              'rnd_expense', 'operating_income', 'pretax_income', 'income_tax',
              'net_income', 'interest_expense', 'operating_cash_flow',
              'investing_cash_flow', 'financing_cash_flow', 'capex',
              'depreciation_amortization', 'dividends_paid', 'stock_buybacks')

QUARTER_DURATION_MAX = 110   # days; longer cashflow windows are cumulative
MKTCAP_JUMP_LIMIT = 1.5      # |dlog mktcap| across a quarter beyond this -> NaN
CLOSE_ASOF_TOLERANCE = 10    # days a filed_date may trail the last close


@dataclass
class QuarterPanel:
    """One ticker's merged quarterly statement view.

    ``frame`` is indexed by (fiscal_year, fiscal_quarter 1-4) and holds the
    statement items, a per-row ``filed_date`` (max across contributing
    statements -- the point-in-time observation date), ``ttm_<item>`` sums,
    and the helpers mktcap / total_debt / ttm_ebitda / ev.
    """
    ticker: str
    frame: pd.DataFrame


def _pivot_statement(df: pd.DataFrame, index_cols: list) -> pd.DataFrame:
    """Long statement rows -> wide (fy, fq) frame + per-row max filed_date.

    filed_date is normalized to datetime64 first: the DB hands back python
    date objects, and pandas 3 raises comparing those against the NaNs that
    outer-joins introduce.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.assign(filed_date=pd.to_datetime(df['filed_date']))
    wide = df.pivot_table(index=index_cols, columns='item', values='value',
                          aggfunc='last')
    wide['filed_date'] = df.groupby(index_cols)['filed_date'].max()
    return wide


def _quarterly_cashflow(cashflow: pd.DataFrame) -> pd.DataFrame:
    """Single-quarter cashflow rows: as-reported quarters and derived rows,
    plus in-memory decumulation of cumulative windows where neither exists."""
    if cashflow is None or cashflow.empty:
        return pd.DataFrame()
    df = cashflow[cashflow['fiscal_quarter'].notna() & (cashflow['fiscal_quarter'] > 0)]
    single = df[df['duration_days'] <= QUARTER_DURATION_MAX]

    cumulative = df[(df['duration_days'] > QUARTER_DURATION_MAX) & (~df['derived'].astype(bool))]
    q1_lookup = (single[single['fiscal_quarter'] == 1]
                 .drop_duplicates(subset=['item', 'fiscal_year'])
                 .set_index(['item', 'fiscal_year'])['value'])
    decum_rows = []
    for (item, fy), grp in cumulative.groupby(['item', 'fiscal_year']):
        grp = grp.sort_values('duration_days')
        # Q1 arrives as a single-quarter window, so it seeds the running total
        q1 = q1_lookup.get((item, fy))
        prev_value, prev_q = (float(q1), 1) if q1 is not None and pd.notna(q1) else (0.0, 0)
        for row in grp.itertuples():
            if row.fiscal_quarter == prev_q + 1:
                decum_rows.append({'fiscal_year': fy, 'fiscal_quarter': row.fiscal_quarter,
                                   'item': item, 'value': row.value - prev_value,
                                   'filed_date': row.filed_date})
            prev_value, prev_q = row.value, row.fiscal_quarter
    parts = [single[['fiscal_year', 'fiscal_quarter', 'item', 'value', 'filed_date']]]
    if decum_rows:
        parts.append(pd.DataFrame(decum_rows))
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=['fiscal_year', 'fiscal_quarter', 'item'], keep='first')


def _close_asof(closes: pd.Series, when, tolerance_days: int = CLOSE_ASOF_TOLERANCE):
    """Last close at or before ``when``, NaN when staler than the tolerance."""
    if closes is None or closes.empty or pd.isna(when):
        return np.nan
    when = pd.Timestamp(when).date()  # index holds python dates
    dates = closes.index
    pos = dates.searchsorted(when, side='right') - 1
    if pos < 0:
        return np.nan
    if (pd.Timestamp(when) - pd.Timestamp(dates[pos])).days > tolerance_days:
        return np.nan
    return closes.iloc[pos]


def build_quarter_panel(income: pd.DataFrame, balance: pd.DataFrame,
                        cashflow: pd.DataFrame, closes: pd.Series,
                        ticker: str = '') -> QuarterPanel:
    """Merge one ticker's stored statements into the quarterly panel.

    Inputs are the long frames from DBManager.get_income / get_balance /
    get_cashflow (single ticker) and a date-indexed close Series. Quarterly
    income rows include derived Q4s (which carry the annual filing's
    filed_date); balance fiscal_quarter 0 maps to Q4.
    """
    inc = income[income['fiscal_quarter'] > 0] if income is not None and not income.empty \
        else pd.DataFrame()
    inc_w = _pivot_statement(inc, ['fiscal_year', 'fiscal_quarter'])

    bal = pd.DataFrame()
    if balance is not None and not balance.empty:
        bal = balance[balance['fiscal_quarter'].notna()].copy()
        bal['fiscal_quarter'] = bal['fiscal_quarter'].replace(0, 4).astype(int)
        bal = bal[bal['fiscal_quarter'].isin([1, 2, 3, 4])]
    bal_w = _pivot_statement(bal, ['fiscal_year', 'fiscal_quarter'])

    cf_w = _pivot_statement(_quarterly_cashflow(cashflow), ['fiscal_year', 'fiscal_quarter'])

    parts = [w for w in (inc_w, bal_w, cf_w) if not w.empty]
    if not parts:
        return QuarterPanel(ticker, pd.DataFrame())
    filed = pd.concat([w['filed_date'] for w in parts], axis=1).max(axis=1)
    frame = pd.concat([w.drop(columns='filed_date') for w in parts], axis=1).sort_index()
    frame['filed_date'] = filed.dt.date  # back to python dates (repo convention)

    if {'revenue', 'cost_of_revenue'} <= set(frame.columns):
        gp = frame['revenue'] - frame['cost_of_revenue']
        frame['gross_profit'] = frame.get('gross_profit', gp).fillna(gp)

    for item in FLOW_ITEMS:
        if item in frame.columns:
            frame[f'ttm_{item}'] = frame[item].rolling(4, min_periods=4).sum()

    shares = pd.Series(np.nan, index=frame.index)
    for col in ('shares_outstanding', 'shares_diluted', 'shares_basic'):
        if col in frame.columns:
            shares = shares.fillna(frame[col])
    frame['shares'] = shares.where(shares > 0)

    frame['mktcap'] = [
        _close_asof(closes, filed) * s if pd.notna(s) else np.nan
        for filed, s in zip(frame['filed_date'], frame['shares'])]
    jump = np.log(frame['mktcap']).diff().abs()
    bad = jump > MKTCAP_JUMP_LIMIT
    if bad.any():
        frame.loc[bad, 'mktcap'] = np.nan

    ltd = frame.get('long_term_debt')
    std = frame.get('short_term_debt')
    if ltd is not None:
        frame['total_debt'] = ltd + (std.fillna(0.0) if std is not None else 0.0)
    if {'ttm_operating_income', 'ttm_depreciation_amortization'} <= set(frame.columns):
        frame['ttm_ebitda'] = frame['ttm_operating_income'] + frame['ttm_depreciation_amortization']
    cash = frame.get('cash_and_equivalents')
    sti = frame.get('short_term_investments')
    if cash is not None and 'total_debt' in frame.columns and 'mktcap' in frame.columns:
        frame['ev'] = (frame['mktcap'] + frame['total_debt']
                       - (cash + (sti.fillna(0.0) if sti is not None else 0.0)))
    return QuarterPanel(ticker, frame)


def _avg(series: pd.Series, lag: int = 4) -> pd.Series:
    """Two-point average of now vs one year ago (falls back to now alone)."""
    prev = series.shift(lag)
    return series.where(prev.isna(), (series + prev) / 2)


def _symmetric_growth(series: pd.Series, lag: int = 4) -> pd.Series:
    prev = series.shift(lag)
    return (series - prev) / ((series.abs() + prev.abs()) / 2)


class FundamentalCalculator(Calculator):
    """One catalog group of derived fundamental items.

    ``calculate`` takes a :class:`QuarterPanel` (or its frame) and returns a
    frame of this group's items on the same (fiscal_year, fiscal_quarter)
    index; division blowups come back NaN. Subclasses implement
    :meth:`derive` over the panel frame.
    """

    family = 'fundamental'

    def calculate(self, data, **params) -> pd.DataFrame:
        frame = data.frame if isinstance(data, QuarterPanel) else data
        wanted = [spec.item for spec in self.outputs]
        if frame.empty:
            return pd.DataFrame(columns=wanted)
        out = self.derive(frame)
        for item in wanted:
            if item not in out.columns:
                out[item] = np.nan
        return out[wanted].replace([np.inf, -np.inf], np.nan)

    def derive(self, f: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _col(f: pd.DataFrame, name: str) -> pd.Series:
        return f[name] if name in f.columns else pd.Series(np.nan, index=f.index)


class Valuation(FundamentalCalculator):
    name = 'valuation'
    group = 'fundamental.valuation'
    description = 'Cheapness of the equity vs its earnings, cash flow, book value, sales, and EBITDA.'
    outputs = (
        OutputSpec('earnings_yield', 'TTM net income / market cap. Inverse P/E: stays finite as '
                   'earnings shrink and is linear in the priced quantity. Core value factor.',
                   (-0.2, 0.2), 'robust', 'winsorize 1/99% then cross-sectional z-score'),
        OutputSpec('fcf_yield', 'TTM (operating cash flow - capex) / market cap. Cash-based '
                   'cheapness, harder to manage than earnings; strong historical factor.',
                   (-0.2, 0.2), 'robust', 'winsorize 1/99% then cs z-score'),
        OutputSpec('book_to_market', 'Common equity / market cap. Fama-French value; '
                   'intangibles-heavy sectors run structurally low.',
                   (0, 5), 'log_standard', 'right-skewed: log1p positives, clip negatives; sector-neutral cs z'),
        OutputSpec('sales_yield', 'TTM revenue / market cap. Value proxy robust for unprofitable '
                   'firms since revenue is rarely negative.',
                   (0, 5), 'log_standard', 'log1p then cs z-score'),
        OutputSpec('ebitda_ev', 'TTM EBITDA / enterprise value. Capital-structure-neutral '
                   'cheapness; standard in value composites.',
                   (-0.5, 0.5), 'robust', 'winsorize 1/99% then cs z-score'),
    )

    def derive(self, f):
        mktcap = self._col(f, 'mktcap')
        return pd.DataFrame({
            'earnings_yield': self._col(f, 'ttm_net_income') / mktcap,
            'fcf_yield': (self._col(f, 'ttm_operating_cash_flow')
                          - self._col(f, 'ttm_capex')) / mktcap,
            'book_to_market': self._col(f, 'stockholders_equity') / mktcap,
            'sales_yield': self._col(f, 'ttm_revenue') / mktcap,
            'ebitda_ev': self._col(f, 'ttm_ebitda') / self._col(f, 'ev'),
        })


class Profitability(FundamentalCalculator):
    name = 'profitability'
    group = 'fundamental.profitability'
    description = 'Margins and returns on capital: how much the business earns per unit of sales/assets/equity.'
    outputs = (
        OutputSpec('gross_margin', 'TTM gross profit / revenue. Pricing power and moat; '
                   'stable, slow-moving quality signal.', (-1, 1.5), 'standard',
                   'clip [-1, 1.5]; blows up only for near-zero revenue'),
        OutputSpec('operating_margin', 'TTM operating income / revenue. Core efficiency after opex.',
                   (-1, 1), 'standard', 'clip [-1, 1] then cs z'),
        OutputSpec('net_margin', 'TTM net income / revenue. Bottom-line profitability.',
                   (-1, 1), 'standard', 'clip [-1, 1] then cs z'),
        OutputSpec('roe', 'TTM net income / average equity. Return on shareholder capital; '
                   'distorted by leverage and near-zero equity.', (-1, 1), 'robust',
                   'cap +/-1 then cs z'),
        OutputSpec('roa', 'TTM net income / average assets. Leverage-free profitability, '
                   'stabler than ROE.', (-0.5, 0.5), 'standard', 'clip [-0.5, 0.5] then cs z'),
        OutputSpec('gross_profitability', 'TTM gross profit / assets. Novy-Marx: the strongest '
                   'simple profitability predictor of returns.', (0, 2), 'standard',
                   'clip [0, 2] then cs z'),
    )

    def derive(self, f):
        rev = self._col(f, 'ttm_revenue')
        ni = self._col(f, 'ttm_net_income')
        return pd.DataFrame({
            'gross_margin': self._col(f, 'ttm_gross_profit') / rev,
            'operating_margin': self._col(f, 'ttm_operating_income') / rev,
            'net_margin': ni / rev,
            'roe': ni / _avg(self._col(f, 'stockholders_equity')),
            'roa': ni / _avg(self._col(f, 'assets')),
            'gross_profitability': self._col(f, 'ttm_gross_profit') / self._col(f, 'assets'),
        })


class Growth(FundamentalCalculator):
    name = 'growth'
    group = 'fundamental.growth'
    description = 'Year-over-year expansion, same fiscal quarter against a year earlier (kills seasonality).'
    outputs = (
        OutputSpec('revenue_growth', 'YoY same-quarter revenue growth. Top-line momentum.',
                   (-2, 2), 'robust', 'clip +/-2 then cs z or rank'),
        OutputSpec('earnings_growth', 'Symmetric YoY growth of TTM net income: '
                   '(x-y)/((|x|+|y|)/2) keeps sign flips bounded in [-4, 4].',
                   (-4, 4), 'robust', 'bounded by construction; cs rank preferred'),
        OutputSpec('asset_growth', 'YoY total-asset growth. The asset-growth anomaly: fast '
                   'balance-sheet expanders underperform (negative expected sign).',
                   (-1, 1), 'robust', 'clip +/-1 then cs z'),
    )

    def derive(self, f):
        rev = self._col(f, 'revenue')
        assets = self._col(f, 'assets')
        return pd.DataFrame({
            'revenue_growth': (rev - rev.shift(4)) / rev.shift(4).abs(),
            'earnings_growth': _symmetric_growth(self._col(f, 'ttm_net_income')),
            'asset_growth': (assets - assets.shift(4)) / assets.shift(4).abs(),
        })


class Quality(FundamentalCalculator):
    name = 'quality'
    group = 'fundamental.quality'
    description = 'Earnings quality: cash backing of profits and share-count discipline.'
    outputs = (
        OutputSpec('accruals', '(TTM net income - TTM operating cash flow) / average assets. '
                   'Sloan accruals: high accruals = low earnings quality, negatively priced.',
                   (-0.5, 0.5), 'standard', 'winsorize then cs z (flip sign in composites)'),
        OutputSpec('cash_conversion', 'TTM operating cash flow / TTM net income. Earnings '
                   'backed by actual cash.', (-5, 5), 'robust',
                   'blows up near zero income: clip [-5, 5], cs rank preferred'),
        OutputSpec('net_share_issuance', 'Negative 4-quarter log change in shares outstanding: '
                   'buybacks positive, dilution negative. Issuance is among the most robust anomalies.',
                   (-0.5, 0.5), 'standard', 'clip +/-0.5 then cs z'),
    )

    def derive(self, f):
        ni = self._col(f, 'ttm_net_income')
        cfo = self._col(f, 'ttm_operating_cash_flow')
        shares = self._col(f, 'shares')
        return pd.DataFrame({
            'accruals': (ni - cfo) / _avg(self._col(f, 'assets')),
            'cash_conversion': cfo / ni,
            'net_share_issuance': -(np.log(shares) - np.log(shares.shift(4))),
        })


class LeverageLiquidity(FundamentalCalculator):
    name = 'leverage_liquidity'
    group = 'fundamental.leverage_liquidity'
    description = 'Balance-sheet risk and distance from distress.'
    outputs = (
        OutputSpec('debt_to_equity', 'Total debt / common equity. Balance-sheet risk; conditions '
                   'value signals (cheap + levered = value trap risk).', (0, 10), 'log_standard',
                   'clip at 10, log1p, cs z; equity near zero blows up'),
        OutputSpec('net_debt_ebitda', '(Total debt - cash) / TTM EBITDA. Credit-style leverage; '
                   'distress proxy.', (-5, 10), 'robust', 'clip [-5, 10] then cs z'),
        OutputSpec('current_ratio', 'Current assets / current liabilities. Short-term liquidity '
                   'buffer.', (0, 10), 'log_standard', 'log then cs z'),
        OutputSpec('interest_coverage', 'TTM operating income / TTM interest expense. Distance '
                   'from distress; NaN when the filer reports no interest expense.',
                   (-10, 50), 'robust', 'clip [-10, 50], cs rank preferred'),
    )

    def derive(self, f):
        debt = self._col(f, 'total_debt')
        return pd.DataFrame({
            'debt_to_equity': debt / self._col(f, 'stockholders_equity'),
            'net_debt_ebitda': (debt - self._col(f, 'cash_and_equivalents'))
                               / self._col(f, 'ttm_ebitda'),
            'current_ratio': self._col(f, 'assets_current') / self._col(f, 'liabilities_current'),
            'interest_coverage': self._col(f, 'ttm_operating_income')
                                 / self._col(f, 'ttm_interest_expense'),
        })


class Efficiency(FundamentalCalculator):
    name = 'efficiency'
    group = 'fundamental.efficiency'
    description = 'Capital efficiency: revenue per unit of assets and investment burden.'
    outputs = (
        OutputSpec('asset_turnover', 'TTM revenue / average assets. DuPont complement to margin.',
                   (0, 4), 'log_standard', 'log1p then cs z, sector-neutral recommended'),
        OutputSpec('capex_intensity', 'TTM capex / revenue. Investment burden vs asset-light '
                   'models; strongly sector-flavored.', (0, 1), 'robust',
                   'winsorize then cs z, sector-neutral recommended'),
    )

    def derive(self, f):
        rev = self._col(f, 'ttm_revenue')
        return pd.DataFrame({
            'asset_turnover': rev / _avg(self._col(f, 'assets')),
            'capex_intensity': self._col(f, 'ttm_capex') / rev,
        })


class Surprise(FundamentalCalculator):
    name = 'surprise'
    group = 'fundamental.surprise'
    description = 'Standardized surprises: this quarter vs the same quarter last year, scaled by recent volatility.'
    outputs = (
        OutputSpec('sue', 'Standardized unexpected earnings: YoY EPS change / std of the last 8 '
                   'changes (min 4). Post-earnings-announcement drift is a robust anomaly.',
                   (-6, 6), 'standard', 'already standardized; clip +/-6, no cs re-scale needed'),
        OutputSpec('revenue_surprise', 'Same construction on quarterly revenue -- the less-managed '
                   'top-line analogue of SUE.', (-6, 6), 'standard', 'clip +/-6'),
    )

    @staticmethod
    def _standardized(delta: pd.Series) -> pd.Series:
        # a (near-)constant change series has no surprise scale: NaN rather
        # than dividing by float noise
        sd = delta.rolling(8, min_periods=4).std()
        floor = delta.abs().rolling(8, min_periods=4).mean() * 1e-9
        return delta / sd.where(sd > floor)

    def derive(self, f):
        eps = self._col(f, 'eps_diluted').fillna(self._col(f, 'eps_basic'))
        rev = self._col(f, 'revenue')
        return pd.DataFrame({
            'sue': self._standardized(eps.diff(4)),
            'revenue_surprise': self._standardized(rev.diff(4)),
        })


FUNDAMENTAL_CALCULATORS = tuple(register(cls()) for cls in (
    Valuation, Profitability, Growth, Quality, LeverageLiquidity,
    Efficiency, Surprise))


def compute_fundamentals(statements_by_ticker: dict, closes_by_ticker: dict) -> pd.DataFrame:
    """Run every fundamental calculator for many tickers.

    ``statements_by_ticker`` maps ticker -> {'income': df, 'balance': df,
    'cashflow': df} (the long single-ticker frames from DBManager);
    ``closes_by_ticker`` maps ticker -> date-indexed close Series. Returns
    the long frame [ticker, date, item, value, fiscal_year, fiscal_quarter]
    with date = the row's filed_date (point-in-time observation date).
    """
    out = []
    for ticker, statements in statements_by_ticker.items():
        panel = build_quarter_panel(statements.get('income'), statements.get('balance'),
                                    statements.get('cashflow'),
                                    closes_by_ticker.get(ticker), ticker=ticker)
        if panel.frame.empty or 'filed_date' not in panel.frame.columns:
            continue
        pieces = [calculator.calculate(panel) for calculator in FUNDAMENTAL_CALCULATORS]
        derived = pd.concat(pieces, axis=1)
        derived['date'] = panel.frame['filed_date']
        long = (derived.reset_index()
                .melt(id_vars=['fiscal_year', 'fiscal_quarter', 'date'],
                      var_name='item', value_name='value')
                .dropna(subset=['value', 'date']))
        long.insert(0, 'ticker', ticker)
        out.append(long[['ticker', 'date', 'item', 'value', 'fiscal_year', 'fiscal_quarter']])
    if not out:
        return pd.DataFrame(columns=['ticker', 'date', 'item', 'value',
                                     'fiscal_year', 'fiscal_quarter'])
    result = pd.concat(out, ignore_index=True)
    # a late statement can revise a derived value for the same (ticker, date,
    # item); keep the newest fiscal period's number
    result = (result.sort_values(['ticker', 'item', 'fiscal_year', 'fiscal_quarter'])
              .drop_duplicates(subset=['ticker', 'date', 'item'], keep='last'))
    return result.reset_index(drop=True)
