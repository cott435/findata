from findata.db_manager import DBManager
from pandas import DataFrame
from typing import Iterable

def get_ticker_meta(tickers: str | Iterable[str]):
    db = DBManager(if_missing='raise')
    tickers = [tickers] if isinstance(tickers, str) else tickers
    return db.get_ticker_meta(tickers)

def get_ticker_data_df(tickers: str | Iterable[str], start_date: str = '2012-01-01', end_date: str = None) -> DataFrame:
    db = DBManager(if_missing='raise')
    tickers = [tickers] if isinstance(tickers, str) else tickers
    return db.get_all_data(tickers, 'daily', wide=True, start=start_date, end=end_date)

def get_all_data(tickers: str | Iterable[str], start_date: str = '2012-01-01', end_date: str = None) -> tuple:
    meta = get_ticker_meta(tickers)
    skipped = [t for t in tickers if t not in meta.index]
    if skipped:
        print(f"Skipping tickers not in database: {skipped}")
    return get_ticker_data_df(tickers, start_date, end_date), meta

if __name__ == '__main__':
    df = get_all_data(['AAPL', 'SOFI'])
