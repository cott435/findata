import random
from .configs import DATA_DIR
import pandas as pd


class TickerSampler:
    def __init__(
        self,
        use_sp500: bool = True,
        use_russell3000: bool = False,
        sp500_date_limit: str | None = '1-1-2010',
    ):
        """
        use_sp500: include tickers from S&P 500 historical data
        use_russell3000: include current Russell 3000 tickers
        sp500_date_limit: only include S&P 500 tickers from rows on/after this date (e.g. '2010-01-01')
        """
        if not use_sp500 and not use_russell3000:
            raise ValueError("At least one of use_sp500 or use_russell3000 must be True")

        self._tickers: set[str] = set()

        if use_sp500:
            self._tickers.update(self._load_sp500(sp500_date_limit))

        if use_russell3000:
            self._tickers.update(self._load_russell3000())

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
