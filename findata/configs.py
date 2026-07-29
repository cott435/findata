import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DB_NAME = "stock.db"
EDGAR_IDENTITY = "Connor ctt7729@gmail.com"
DATA_DIR = Path(__file__).parents[1] / "data"
EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments"
SEC_DIR = DATA_DIR / "sec_filings"
LOG_DIR = Path(__file__).parents[1] / "logs"

TICKERS = ['JNJ', 'JPM', 'MRK', 'HD', 'CVX', 'C', 'PNC', 'BAC', 'BK', 'WM', 'NFLX','AAPL', 'MSFT', 'AMZN', 'LLY',
           'GOOG','META','SPG','AMT', 'F','XOM','COST','VZ','GE','NEM','CCI','NRG','MCD','KO','PG', 'TEM',' MRNA',
           'AA', 'ADT', 'WFC', 'WING', 'FLY', 'SDRL']

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
# third-party loggers that flood INFO; kept at WARNING so our logs stay readable
_NOISY_LOGGERS = ("edgar", "urllib3", "asyncio", "httpx", "httpcore")

def __getattr__(name):
    # DATA_MAP / TECHNICAL_WINDOWS moved to the calculator registry; resolved
    # lazily here so configs stays stdlib-only at import time (importing the
    # calculators from the top of this module would recreate the
    # configs -> calculators -> preprocess -> configs cycle).
    if name in ('DATA_MAP', 'TECHNICAL_WINDOWS'):
        from findata.preprocess.calculators import technical
        return getattr(technical, name)
    raise AttributeError(f"module 'findata.configs' has no attribute {name!r}")


from dataclasses import dataclass as _dataclass
from dataclasses import field as _field


@_dataclass(frozen=True)
class SavePolicy:
    """Which calculator groups auto-persist their on-the-fly results.

    Rules map save-policy group keys (a Calculator's ``group``, e.g.
    'technical.core', 'fundamental.valuation') to booleans; lookup walks the
    dotted key up ('fundamental.valuation' falls back to 'fundamental'), and
    ``default`` applies when no rule prefix matches.
    """
    rules: dict = _field(default_factory=dict)
    default: bool = True

    def allows(self, group: str) -> bool:
        key = group or ''
        while key:
            if key in self.rules:
                return self.rules[key]
            key = key.rpartition('.')[0]
        return self.default


SAVE_POLICY = SavePolicy(rules={
    'technical.core': True,     # registry-default windows
    'technical.custom': True,   # ad-hoc windows requested on the fly
    'fundamental': True,        # all fundamental.<group> keys inherit this
    'feature': False,           # engineered FeatureGroup outputs never persist
})


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

from dataclasses import dataclass, fields, field
from datetime import date, timedelta
from typing import Set

@dataclass
class DataSplits:

    train_start: date = date(year=2015, month=1, day=1)
    train_end: date | None = None
    validation_start: date = date(year=2024, month=1, day=1)
    validation_end: date | None = None
    test_start: date = date(year=2025, month=1, day=1)
    test_end: date = date(year=2025, month=8, day=31)
    all_tickers: Set[str] = field(default_factory=set)
    val_holdout_tickers: Set[str] = field(default_factory=set)

    def __post_init__(self):
        self.data_start = self.train_start - timedelta(days=300)
        self.train_end = self.validation_start - timedelta(days=1) if self.train_end is None else self.train_end
        self.validation_end = self.test_start - timedelta(days=1) if self.validation_end is None else self.validation_end
        self.data_end = max(self.train_end, self.train_start, self.test_end, self.test_start)

