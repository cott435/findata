"""ORM table definitions for the stock_data SQLite database.

YF-side tables only for now -- the EDGAR tables (financials_data,
filings_data, fundamentals_data) are added in part 2. No logic lives here.
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
    last_filings_date = Column(Date)
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
