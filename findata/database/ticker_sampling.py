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
        market_cap_floor=300_000_000,
        domestic_only: bool = True,
    ):
        """
        use_all: draw from us_tickers.xlsx (market-cap filtered) instead of all_info.xlsx
        date_limit: only tickers with price history starting on/before this date
        domestic_only: drop foreign private issuers -- see _drop_foreign
        """
        self._tickers: set[str] = set()
        self.market_cap_floor = market_cap_floor
        self.date_limit = pd.to_datetime(date_limit)

        if use_all:
            self._tickers.update(self._load_all())
        else:
            self._tickers.update(self._load_subset())
        if domestic_only:
            self._tickers = self._drop_foreign(self._tickers)

    @staticmethod
    def _drop_foreign(tickers: set[str]) -> set[str]:
        """Remove companies that file 20-F/40-F rather than a 10-K.

        Foreign private issuers tag their XBRL in the ifrs-full taxonomy and
        report semi-annually, so the us-gaap concept lists and quarter-length
        windows in edgar_.py resolve almost nothing for them -- their stored
        statements come out empty rather than wrong. Excluding them keeps the
        universe to names the fundamentals pipeline can actually populate.

        Classification comes from which annual form each company files (cached
        under the data dir). Tickers EDGAR has no annual filing for are kept:
        absence of evidence is not evidence of a foreign filer.
        """
        try:
            from findata.database.edgar_ import filer_types
            kinds = filer_types(sorted(tickers))
        except Exception as e:
            print(f'Could not classify filers ({e}); keeping all tickers')
            return tickers
        foreign = {t for t in tickers if kinds.get(t) == 'foreign'}
        if foreign:
            print(f'Excluding {len(foreign)} foreign filer(s) from the universe')
        return tickers - foreign

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
