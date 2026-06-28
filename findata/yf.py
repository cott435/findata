import logging
import pandas as pd
from yahooquery import Ticker
from typing import Union, List, Tuple, Dict, Any
import asyncio
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo
import numpy as np

logger = logging.getLogger(__name__)

def get_monday(date_obj, prior_week=False):
    if date_obj:
        d = date_obj - timedelta(days=date_obj.weekday())
        return d - timedelta(weeks=1) if prior_week else d
    return None

class YahooFinance:

    def __init__(self, tickers: Union[str, List[str]]):
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = [t.replace(".", "-") for t in tickers]
        self.t = Ticker(self.tickers, asynchronous=True)

    @staticmethod
    def find_liquidity_start(df, window=20, min_volume=1000, max_zero_frac=0.05):
        """Returns the first date where the ticker sustains liquid trading."""
        is_liquid = (df['volume'] >= min_volume).astype(int)
        rolling_frac = is_liquid.rolling(window, min_periods=1).mean()
        stable = rolling_frac >= (1 - max_zero_frac)
        if not stable.any():
            return None
        false_positions = np.where(~stable.fillna(False))[0]
        if len(false_positions) == 0:
            start_idx = stable.first_valid_index()
        else:
            last_false_pos = false_positions[-1]
            if last_false_pos == len(stable) - 1:
                return None
            start_idx = df.index[last_false_pos + 1]
        return start_idx[-1] if isinstance(start_idx, tuple) else start_idx

    def _filter_illiquid(self, day_hist: pd.DataFrame, week_hist: pd.DataFrame):
        """Drop pre-liquidity rows from daily data and sync weekly to match."""
        liquidity_starts = {}
        total_dropped = total_zero_vol = 0

        for ticker, grp in day_hist.groupby(level='symbol'):
            start = self.find_liquidity_start(grp)
            liquidity_starts[ticker] = start
            pre = grp if start is None else grp[grp.index.get_level_values('date') < start]
            if len(pre):
                zero_vol = int((pre['volume'] == 0).sum())
                total_dropped += len(pre)
                total_zero_vol += zero_vol
                logger.info('Liquidity filter %s: dropping %d rows (before %s), %d zero-volume',
                            ticker, len(pre), start, zero_vol)

        if total_dropped:
            logger.info('Liquidity filter total: %d daily rows dropped (%d zero-volume)',
                        total_dropped, total_zero_vol)

        def keep_daily(grp):
            ticker = grp.index.get_level_values('symbol')[0]
            start = liquidity_starts.get(ticker)
            if start is None:
                return grp.iloc[0:0]
            new = grp[grp.index.get_level_values('date') >= start]
            mask = new['volume'].eq(0)
            total = mask.sum()
            if total > 0:
                logger.info(f"Dropping {total} rows for 0 volume day: {new[mask].index}")
            return new[~mask]

        def keep_weekly(grp):
            ticker = grp.index.get_level_values('symbol')[0]
            start = liquidity_starts.get(ticker)
            if start is None:
                return grp.iloc[0:0]
            week_start = start - timedelta(days=start.weekday())
            return grp[grp.index.get_level_values('date') >= week_start]

        day_hist = day_hist.groupby(level='symbol', group_keys=False).apply(keep_daily)
        week_hist = week_hist.groupby(level='symbol', group_keys=False).apply(keep_weekly)
        return day_hist, week_hist

    @staticmethod
    def find_ohlc_violations(df):
        violations = df[
            (df['open'] > df['high']) | (df['open'] < df['low']) |
            (df["close"] > df['high']) | (df["close"] < df['low']) |
            (df['high'] < df['low'])
            ].copy()
        if len(violations) > 0:
            logger.warning(f'Found {len(violations)} violations: {violations.index}')
            df["open"] = df["open"].clip(lower=df["low"], upper=df["high"])
            df["close"] = df["close"].clip(lower=df["low"], upper=df["high"])
        return df

    @staticmethod
    def _adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """Adjust OHLC columns using adjclose (to match yfinance behavior)."""
        if "adjclose" in df and "close" in df:
            ratio = df["adjclose"] / df["close"]
            for col in ["open", "high", "low", "close"]:
                if col in df:
                    df[col] = df[col] * ratio
        return df.drop(columns=["adjclose"])

    def _fix_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._adjust_ohlc(self.find_ohlc_violations(df))

    @staticmethod
    def _unify_price_info_index(infos, candles):
        unique_prices = candles.index.get_level_values("ticker").unique()
        infos = {k: v for k, v in infos.items() if k in unique_prices}
        candles = candles[candles.index.get_level_values("ticker").isin(infos.keys())]
        return infos, candles

    def request_ticker_financials(self, start=None) -> Dict[str, Any]:
        logger.info('Requesting Yahoo Finance data for %d ticker(s)%s',
                    len(self.tickers), f' from {start}' if start else '')
        return asyncio.run(self._request_ticker_financials(start=start))

    async def _request_ticker_financials(self, start=None) -> Dict[str, Any]:

        async def get_data_p(freq):
            func = self.t.history
            return func(interval=freq, period='max', start=start)

        async def get_data_ap():
            return self.t.asset_profile

        try:
            info_task = asyncio.create_task(get_data_ap())
            daily_task = asyncio.create_task(get_data_p('1d'))
            weekly_task = asyncio.create_task(get_data_p('1wk'))

            infos, day_hist, week_hist = await asyncio.gather(info_task, daily_task, weekly_task)

            def fix_dates(df):
                dates = pd.to_datetime(df.index.get_level_values(1), utc=True).tz_convert(None).date
                df.index = pd.MultiIndex.from_arrays([df.index.get_level_values(0), dates], names=df.index.names)
            fix_dates(day_hist)
            fix_dates(week_hist)

            # ensure week date is only the last full week
            final_week = get_monday(datetime.today(), prior_week=True)
            week_hist = week_hist[week_hist.index.get_level_values('date') <= final_week.date()]
            now = datetime.now(ZoneInfo("America/New_York"))

            market_open = (
                    now.weekday() < 5 and  # Mon-Fri
                    (now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and
                    now.hour < 16
            )

            if market_open:
                today = now.date()
                day_hist = day_hist[day_hist.index.get_level_values('date') != today]

            day_hist, week_hist = self._filter_illiquid(day_hist, week_hist)

            candles = pd.concat([day_hist, week_hist], keys=['daily','weekly'], names=['interval', 'ticker', 'date'])
            candles = self._fix_ohlc(candles)
            last_price_dates = (
                candles.loc['daily']
                .groupby(level='ticker')
                .apply(lambda x: x.index.get_level_values('date').max())
            )
            first_price_dates = (
                candles.loc['daily']
                .groupby(level='ticker')
                .apply(lambda x: x.index.get_level_values('date').min())
            )

            skipped = [t for t, info in infos.items() if isinstance(info, str)]
            if skipped:
                logger.info(f"Skipping {len(skipped)} tickers due to invalid yahoo response: {skipped}")
            infos, candles = self._unify_price_info_index(infos, candles)
            t_info = [
                {
                    "ticker": ticker,
                    'sector': info.get('sector', 'ETF'),
                    'industry': info.get('industry', 'ETF'),
                    'last_price_date': last_price_dates.loc[ticker],
                    'first_price_date': first_price_dates.loc[ticker]
                } for ticker, info in infos.items() if isinstance(info, dict)
            ]
            t_info = pd.DataFrame(t_info).set_index('ticker')
            return {
                    "info": t_info,
                    "prices": candles,
                }

        except Exception as e:
            logger.exception('YahooFinance: error fetching bulk data: %s', e)
            return {ticker: {"ticker": ticker, "info": {"error": str(e)}} for ticker in self.tickers}

if __name__ == '__main__':
    yf = YahooFinance(['AAPL', 'GOOGL', 'AMZN'])
    data = yf.request_ticker_financials()
    print(data)

