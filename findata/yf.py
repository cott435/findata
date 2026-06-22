import logging
import pandas as pd
from yahooquery import Ticker
from typing import Union, List, Tuple, Dict, Any
import asyncio
from datetime import timedelta, datetime

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
        self.tickers = tickers
        self.t = Ticker(tickers, asynchronous=True)

    @staticmethod
    def _adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """Adjust OHLC columns using adjclose (to match yfinance behavior)."""
        if "adjclose" in df and "close" in df:
            ratio = df["adjclose"] / df["close"]
            for col in ["open", "high", "low", "close"]:
                if col in df:
                    df[col] = df[col] * ratio
        return df.drop(columns=["adjclose"])

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

            info, day_hist, week_hist = await asyncio.gather(info_task, daily_task, weekly_task)

            def fix_dates(df):
                dates = pd.to_datetime(df.index.get_level_values(1), utc=True).tz_convert(None).date
                df.index = pd.MultiIndex.from_arrays([df.index.get_level_values(0), dates], names=df.index.names)
            fix_dates(day_hist)
            fix_dates(week_hist)

            # ensure week date is only the last full week
            final_week = get_monday(datetime.today(), prior_week=True)
            week_hist = week_hist[week_hist.index.get_level_values('date') <= final_week.date()]

            candles = pd.concat([day_hist, week_hist], keys=['daily','weekly'], names=['interval', 'ticker', 'date'])
            candles = self._adjust_ohlc(candles)

            last_price_dates = (
                candles.loc['daily']
                .groupby(level='ticker')
                .apply(lambda x: x.index.get_level_values('date').max())
            )
            t_info = [
                {
                    "ticker": ticker,
                    'sector': info.get('sector', 'etf'),
                    'industry': info.get('industry', 'etf'),
                    'last_price_date': last_price_dates.loc[ticker],
                } for ticker, info in info.items() if isinstance(info, dict)
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

