import numpy as np
import pandas as pd

from findata.yf import YahooFinance
from findata.ticker_sampling import TickerSampler

def find_liquidity_start(df,
                         window=20,  # rolling window size (trading days)
                         min_volume=1000,  # what counts as "real" volume
                         max_zero_frac=0.05,  # max fraction of zero/low-vol days allowed in window
                         ):
    """
    df must have a 'volume' column, sorted by date ascending.
    Returns the index (date) where sustained liquidity begins.
    """
    is_liquid_day = (df['volume'] >= min_volume).astype(int)

    # rolling fraction of liquid days in the window
    rolling_liquid_frac = is_liquid_day.rolling(window, min_periods=1).mean()

    # a window "passes" if zero/low-vol days are rare enough
    stable = rolling_liquid_frac >= (1 - max_zero_frac)

    if not stable.any():
        return None  # never reaches a stable liquid regime

    start_idx = stable.idxmax()
    return start_idx


# Usage per ticker:
def filter_ticker(df):
    start = find_liquidity_start(df)
    if start is None:
        return df.iloc[0:0]  # drop entirely — never became liquid
    return df.loc[start:]


tickers = ['ACAD', 'ACHC', 'AX', 'BFC', 'CEPV', 'CURR', 'EDN', 'EH', 'ESEA',
       'IEAG', 'KRP', 'LILAK', 'MBVI', 'OPFI', 'PRTA', 'SIM', 'WGS', 'ZVRA']

data = YahooFinance(tickers).request_ticker_financials(start='1/1/2005')
info, prices = data['info'], data['prices'].loc['daily']

p = prices.groupby('ticker').apply(filter_ticker)

vol0 = prices['volume'].eq(0).groupby(['ticker']).sum()
vol5 = vol0[vol0 > 5]










