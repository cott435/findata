from pathlib import Path

DB_NAME = "stock.db"
EDGAR_IDENTITY = "Connor ctt7729@gmail.com"
DATA_DIR = Path(__file__).parents[1] / "data"
SEC_DIR = DATA_DIR / "sec_filings"

TICKERS = ['JNJ', 'JPM', 'MRK', 'HD', 'CVX', 'C', 'PNC', 'BAC', 'BK', 'WM', 'NFLX','AAPL', 'MSFT', 'AMZN', 'LLY',
           'GOOG','META','SPG','AMT', 'F','XOM','COST','VZ','GE','NEM','CCI','NRG','MCD','KO','PG', 'TEM',' MRNA']

