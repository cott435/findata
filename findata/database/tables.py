"""ORM table definitions for the stock_data SQLite database.

YF side: ticker_meta, price_data, technical_data (wide: one column per
item_period; ema_data/indicator_data are the legacy long tables the
migration script folds into it). EDGAR side: filings_data (text-section
index), form4_data (insider trades), and the three financial-statement
tables -- separate because their natural keys differ: income is keyed by
fiscal period, balance by point-in-time date, cash flow by cumulative
(start, end) window. fundamental_data holds derived point-in-time ratios.
No logic lives here.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, Date, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TickerMeta(Base):
    """One row per ticker; upserted whenever metadata refreshes."""
    __tablename__ = 'ticker_meta'

    ticker = Column(String, primary_key=True)
    cik = Column(Integer)              # SEC entity id; the join key for EDGAR data
    name = Column(String)
    sector = Column(String)
    industry = Column(String)
    first_price_date = Column(Date)
    last_price_date = Column(Date)
    last_filings_date = Column(Date)   # newest 10-K/10-Q/8-K filing processed
    last_form4_date = Column(Date)     # newest Form 4 filing processed
    last_facts_date = Column(Date)     # when XBRL facts were last fetched/parsed
    last_fundamentals_date = Column(Date)  # when derived fundamentals were last built
    inactive_since = Column(Date)      # stopped returning bars (delisted/renamed)
    yf_seeded = Column(Boolean, default=False)
    technicals_seeded = Column(Boolean, default=False)  # full indicator history stored
    edgar_seeded = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)


class PriceData(Base):
    """Split-adjusted OHLCV bars plus corporate actions. Wide format."""
    __tablename__ = 'price_data'

    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    interval = Column(String, primary_key=True)  # 'daily' | 'weekly'
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    dividend = Column(Float, default=0.0)
    split_ratio = Column(Float, default=1.0)


def _default_technical_columns() -> dict:
    """One REAL column per registry-default (item, period), so the ORM's view
    of technical_data never drifts from the calculator registry. Runtime-added
    windows live as extra columns discovered via PRAGMA table_info."""
    from findata.preprocess.calculators.technical import (EMA_WINDOWS,
                                                          INDICATOR_OUTPUTS,
                                                          INDICATOR_WINDOWS,
                                                          _as_list)
    cols = {'obv': Column(Float), 'ad': Column(Float)}
    for name, periods in INDICATOR_WINDOWS.items():
        for period in _as_list(periods):
            for item in INDICATOR_OUTPUTS[name]:
                cols[f'{item}_{period}'] = Column(Float)
    for name, periods in EMA_WINDOWS.items():
        for period in periods:
            cols[f'{name}_{period}'] = Column(Float)
    return cols


class TechnicalData(Base):
    """Derived technical items, wide: one row per bar, one column per
    item_period (rsi_14, ema_close_26, obv, ...). Upserted."""
    __tablename__ = 'technical_data'

    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    interval = Column(String, primary_key=True)  # 'daily' | 'weekly'


for _name, _column in _default_technical_columns().items():
    setattr(TechnicalData, _name, _column)


class FundamentalData(Base):
    """Derived fundamental items (ratios, growth, ...), long format.

    ``date`` is the point-in-time observation date -- the filed_date of the
    newest statement each value uses -- so forward-filling to trading dates
    never looks ahead. Upserted (late filings can revise derived values).
    """
    __tablename__ = 'fundamental_data'

    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    item = Column(String, primary_key=True)
    value = Column(Float)
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)


class FilingsData(Base):
    """Index of text filings parsed to disk. One row per 10-K/10-Q/8-K."""
    __tablename__ = 'filings_data'

    ticker = Column(String, primary_key=True)
    accession_number = Column(String, primary_key=True)
    form_type = Column(String)                   # '10-K' | '10-Q' | '8-K'
    fiscal_year = Column(Integer)                # null for 8-K
    fiscal_quarter = Column(Integer)             # 0 = annual; null for 8-K
    filing_date = Column(Date)
    period_of_report = Column(Date)
    disk_path = Column(String)                   # sec dir for the saved .txt sections
    sections_parsed = Column(String)             # JSON list of section keys saved
    parse_extra_data = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)


class Form4Data(Base):
    """Insider transactions from Form 4 filings. One row per transaction."""
    __tablename__ = 'form4_data'

    ticker = Column(String, primary_key=True)
    accession_number = Column(String, primary_key=True)
    seq = Column(Integer, primary_key=True)      # row order within the filing
    filing_date = Column(Date)
    transaction_date = Column(Date)
    insider = Column(String)
    position = Column(String)
    security = Column(String)
    transaction_code = Column(String)            # P, S, A, M, G, F, ...
    transaction_type = Column(String)
    acquired_disposed = Column(String)           # 'A' | 'D'
    direct_indirect = Column(String)             # 'D' | 'I'
    shares = Column(Float)
    price = Column(Float)
    value = Column(Float)                        # shares * price when both present
    shares_remaining = Column(Float)             # holdings after the transaction
    is_derivative = Column(Boolean, default=False)


class IncomeData(Base):
    """Income-statement items, keyed by fiscal period.

    Quarterly rows are the as-reported 3-month values; Q4 is derived as
    annual minus Q1-Q3 when not reported (derived=True). fiscal_quarter 0
    is the annual row.
    """
    __tablename__ = 'income_data'

    ticker = Column(String, primary_key=True)
    fiscal_year = Column(Integer, primary_key=True)
    fiscal_quarter = Column(Integer, primary_key=True)  # 1-4; 0 = annual
    item = Column(String, primary_key=True)
    value = Column(Float)
    start_date = Column(Date)
    end_date = Column(Date)
    filed_date = Column(Date)
    derived = Column(Boolean, default=False)


class BalanceData(Base):
    """Balance-sheet items: point-in-time values, no duration/interval."""
    __tablename__ = 'balance_data'

    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)        # balance date (period end)
    item = Column(String, primary_key=True)
    value = Column(Float)
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(Integer)             # 0 = annual report
    filed_date = Column(Date)


class CashflowData(Base):
    """Cash-flow items: issuers report cumulatively from the fiscal-year
    start, so the period window is the key and duration is stored explicitly.

    Every cumulative window is also stored de-cumulated into a standalone
    quarter (``derived=True``); ``derived`` is part of the primary key so the
    as-reported and single-quarter views of the same period coexist. Only the
    single-quarter rows are summable into trailing-twelve-month figures.
    """
    __tablename__ = 'cashflow_data'

    ticker = Column(String, primary_key=True)
    start_date = Column(Date, primary_key=True)
    end_date = Column(Date, primary_key=True)
    item = Column(String, primary_key=True)
    derived = Column(Boolean, primary_key=True, default=False)
    value = Column(Float)
    duration_days = Column(Integer)
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(Integer)             # quarter the window ends in
    filed_date = Column(Date)
