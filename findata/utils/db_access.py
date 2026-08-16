from findata.database.db_manager import DBManager
from pandas import DataFrame
from typing import Iterable

def _to_list(tickers):
    return [tickers] if isinstance(tickers, str) else tickers

def get_all_tickers(db_path=None):
    db = DBManager(db_path=db_path)
    return db.get_all_tickers()

def get_ticker_meta(tickers: str | Iterable[str]=None, db_path=None, manager: DBManager = None):
    db = DBManager(db_path=db_path) if manager is None else manager
    tickers = _to_list(tickers)
    return db.get_ticker_meta(tickers)

def get_ticker_data_df(tickers: str | Iterable[str]=None, start_date: str = '2012-01-01', end_date: str = None,
                       db_path=None, manager: DBManager = None, include_fundamentals=False) -> DataFrame:
    manager = DBManager(db_path=db_path) if manager is None else manager
    tickers = _to_list(tickers)
    tickers = tickers if tickers else manager.get_all_tickers()
    return manager.get_all_data(tickers, 'daily', start=start_date, end=end_date, wide=True, include_fundamentals=include_fundamentals)

def get_all_data(tickers: str | Iterable[str]=None, start_date: str = '2012-01-01', end_date: str = None,
                 db_path=None, manager: DBManager = None, include_fundamentals=False) -> tuple:
    manager = DBManager(db_path=db_path) if manager is None else manager
    tickers = _to_list(tickers)
    meta = get_ticker_meta(tickers, db_path=db_path, manager=manager)
    if not tickers:
        tickers = list(meta.index)
        skipped=[]
    else:
        skipped = [t for t in tickers if t not in meta.index]
    if skipped:
        print(f"Skipping tickers not in database: {skipped}")
    return get_ticker_data_df(tickers, start_date, end_date, db_path=db_path, manager=manager, include_fundamentals=include_fundamentals), meta

def get_price_data(tickers: str | Iterable[str]=None, start_date: str = '2012-01-01', end_date: str = None,
                   db_path=None, manager: DBManager = None) -> DataFrame:
    manager = DBManager(db_path=db_path) if manager is None else manager
    tickers = _to_list(tickers)
    tickers = tickers if tickers else manager.get_all_tickers()
    return manager.get_price_data(tickers, 'daily', start=start_date, end=end_date)

if __name__ == '__main__':
    df = get_all_data()
