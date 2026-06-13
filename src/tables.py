"""ORM table definitions for the stock_data SQLite database.

YF side: ticker_meta, price_data, ema_data, indicator_data.
EDGAR side: filings_data (text-section index), form4_data (insider trades),
and the three financial-statement tables -- separate because their natural
keys differ: income is keyed by fiscal period, balance by point-in-time
date, cash flow by cumulative (start, end) window. No logic lives here.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, Date, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TickerMeta(Base):
    """One row per ticker; upserted whenever metadata refreshes."""
    __tablename__ = 'ticker_meta'

    ticker = Column(String, primary_key=True)
    name = Column(String)
    sector = Column(String)
    industry = Column(String)
    last_price_date = Column(Date)
    last_filings_date = Column(Date)   # newest 10-K/10-Q/8-K filing processed
    last_form4_date = Column(Date)     # newest Form 4 filing processed
    yf_seeded = Column(Boolean, default=False)
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


class EmaData(Base):
    """EMAs in long format so new (period, base) combos need no migration."""
    __tablename__ = 'ema_data'

    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    period = Column(Integer, primary_key=True)
    base = Column(String, primary_key=True)      # 'close' | 'obv' | 'ad'
    interval = Column(String, primary_key=True)  # 'daily' | 'weekly'
    ema_value = Column(Float)


class IndicatorData(Base):
    """Technical indicators in long format."""
    __tablename__ = 'indicator_data'

    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    freq = Column(String, primary_key=True)      # 'daily' | 'weekly'
    indicator = Column(String, primary_key=True)
    period = Column(Integer, primary_key=True)   # 0 = no fixed window (obv, ad)
    value = Column(Float)


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
    """Cash-flow items: cumulative from fiscal-year start, so the period
    window is the key and duration is stored explicitly. derived=True rows
    are de-cumulated single quarters (optional pipeline flag)."""
    __tablename__ = 'cashflow_data'

    ticker = Column(String, primary_key=True)
    start_date = Column(Date, primary_key=True)
    end_date = Column(Date, primary_key=True)
    item = Column(String, primary_key=True)
    value = Column(Float)
    duration_days = Column(Integer)
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(Integer)             # quarter the window ends in; 0 = full year
    filed_date = Column(Date)
    derived = Column(Boolean, default=False)
