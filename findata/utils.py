from findata.db_manager import DBManager

def get_ticker_meta(tickers: str | list):
    db = DBManager(if_missing='raise')
    tickers = [tickers] if isinstance(tickers, str) else tickers
    return db.get_ticker_meta(tickers)

def get_ticker_data_df(tickers: str | list):
    db = DBManager(if_missing='raise')
    tickers = [tickers] if isinstance(tickers, str) else tickers
    return db.get_all_data(tickers, 'daily', wide=True)


if __name__ == '__main__':
    df = get_ticker_data_df(['AAPL', 'SOFI'])
