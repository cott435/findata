from findata.db_manager import DBManager
from pandas import DataFrame

def get_ticker_meta(tickers: str | list):
    db = DBManager(if_missing='raise')
    tickers = [tickers] if isinstance(tickers, str) else tickers
    return db.get_ticker_meta(tickers)

def get_ticker_data_df(tickers: str | list, start_date: str = '2012-01-01', end_date: str = None) -> DataFrame:
    db = DBManager(if_missing='raise')
    tickers = [tickers] if isinstance(tickers, str) else tickers
    return db.get_all_data(tickers, 'daily', wide=True, start=start_date, end=end_date)

def get_all_data(tickers: str | list, start_date: str = '2012-01-01', end_date: str = None) -> tuple:
    return get_ticker_data_df(tickers, start_date, end_date), get_ticker_meta(tickers)

if __name__ == '__main__':
    df = get_ticker_data_df(['AAPL', 'SOFI'])
