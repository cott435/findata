import random
from findata.configs import DATA_DIR, TICKERS
import pandas as pd
import re

def parse_market_cap(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    match = re.match(r"^([\d.]+)([MBT]?)$", val)
    if not match:
        return None
    num, suffix = match.groups()
    multiplier = {"": 1, "M": 1e6, "B": 1e9, "T": 1e12}[suffix]
    return float(num) * multiplier

class TickerSampler:
    def __init__(
        self,
        use_all=False,
        date_limit: str | None = '1-1-2018',
        market_cap_floor=300_000_000
    ):
        """
        use_sp500: include tickers from S&P 500 historical data
        use_russell3000: include current Russell 3000 tickers
        sp500_date_limit: only include S&P 500 tickers from rows on/after this date (e.g. '2010-01-01')
        """
        self._tickers: set[str] = set()
        self.market_cap_floor = market_cap_floor
        self.date_limit = pd.to_datetime(date_limit)

        if use_all:
            self._tickers.update(self._load_all())
        else:
            self._tickers.update(self._load_subset())

    def _load_subset(self):
        us_path = DATA_DIR / "all_info.xlsx"
        try:
            df = pd.read_excel(us_path)
        except FileNotFoundError as e:
            print('No ticker data file found, returning default tickers')
            return set(TICKERS)
        df = df[df['first_price_date'] < self.date_limit]
        return set(df["ticker"].dropna().astype(str))

    def _load_all(self) -> set[str]:
        us_path = DATA_DIR / "us_tickers.xlsx"
        try:
            df = pd.read_excel(us_path)
        except FileNotFoundError as e:
            print('No ticker data file found, returning default tickers')
            return set(TICKERS)
        df["market_cap_numeric"] = df["Market Cap"].apply(parse_market_cap)
        df_filtered = df[df["market_cap_numeric"] >= self.market_cap_floor].copy()
        return set(df_filtered["Ticker"].dropna().astype(str))

    def _load_sp500(self, date_limit: str | None) -> set[str]:
        sp500_path = DATA_DIR / "S&P500_Historical.csv"
        df = pd.read_csv(sp500_path)
        df["date"] = pd.to_datetime(df["date"])
        if date_limit is not None:
            df = df[df["date"] >= pd.to_datetime(date_limit)]
        tickers: set[str] = set()
        for row in df["tickers"]:
            tickers.update(row.split(","))
        return tickers

    def _load_russell3000(self) -> set[str]:
        russell_path = DATA_DIR / "Russell-3000.xlsx"
        df = pd.read_excel(russell_path)
        df = df[df["Exchange"] != "NO MARKET (E.G. UNLISTED)"]
        df = df[df['Ticker'] != "--"]
        return set(df["Ticker"].dropna().astype(str))

    @property
    def tickers(self) -> list[str]:
        return sorted(self._tickers)

    def __len__(self) -> int:
        return len(self._tickers)

    def sample(self, n: int, seed: int | None = None) -> list[str]:
        """Return n tickers sampled without replacement."""
        if n > len(self._tickers):
            raise ValueError(f"n={n} exceeds universe size {len(self._tickers)}")
        rng = random.Random(seed)
        return rng.sample(sorted(self._tickers), n)

    def sample_fraction(self, frac: float, seed: int | None = None) -> list[str]:
        """Return a fraction of the universe sampled without replacement."""
        if not 0 < frac <= 1:
            raise ValueError("frac must be in (0, 1]")
        n = max(1, round(len(self._tickers) * frac))
        return self.sample(n, seed=seed)

if __name__ == "__main__":
    t = TickerSampler()
    t.sample(30, 123)
