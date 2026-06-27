import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DB_NAME = "stock.db"
EDGAR_IDENTITY = "Connor ctt7729@gmail.com"
DATA_DIR = Path(__file__).parents[1] / "data"
SEC_DIR = DATA_DIR / "sec_filings"
LOG_DIR = Path(__file__).parents[1] / "logs"

TICKERS = ['JNJ', 'JPM', 'MRK', 'HD', 'CVX', 'C', 'PNC', 'BAC', 'BK', 'WM', 'NFLX','AAPL', 'MSFT', 'AMZN', 'LLY',
           'GOOG','META','SPG','AMT', 'F','XOM','COST','VZ','GE','NEM','CCI','NRG','MCD','KO','PG', 'TEM',' MRNA']

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
# third-party loggers that flood INFO; kept at WARNING so our logs stay readable
_NOISY_LOGGERS = ("edgar", "urllib3", "asyncio", "httpx", "httpcore")


def setup_logging(log_file: str = "findata.log", level=logging.INFO,
                  console: bool = True) -> Path:
    """Configure root logging to a rotating file (and optionally the console).

    Replaces any existing handlers so it is safe to call more than once and
    is not silently no-op'd the way logging.basicConfig is once another
    library has already touched logging. Returns the log file path.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / log_file
    formatter = logging.Formatter(LOG_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5,
                                       encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging configured -> %s", path)
    return path

