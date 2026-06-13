"""Walkthrough of every DBManager query option.

Shows, against one seeded ticker:
    1. raw price data
    2. get_all_data wide and long
    3. get_items with a mixed request dict (price cols, EMAs, indicator
       groups/columns, periodless items), wide and long
    4. if_missing='raise' vs if_missing='add' for an unstored period

Run:
    python scripts/query_db_yf.py [--ticker AAPL] [--db-path data/stock.db]

If the ticker isn't seeded yet it is pulled from Yahoo and seeded first.
"""

import argparse
import sys
from pathlib import Path


import pandas as pd

from findata.db_manager import DBManager, MissingItemsError
from findata.yf import YahooFinance


def banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description='Demo the DBManager query API.')
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--db-path', default=None)
    args = parser.parse_args()
    ticker = args.ticker.upper()

    db = DBManager(args.db_path, if_missing='raise')

    meta = db.get_ticker_meta([ticker])
    if meta.empty or not bool(meta.loc[ticker, 'yf_seeded']):
        print(f'{ticker} not seeded yet -- pulling from Yahoo Finance first.')
        data = YahooFinance(ticker).request_ticker_financials()
        db.add_ticker_data(data['info'], data['prices'])

    pd.set_option('display.width', 200)

    banner(f'1. get_price_data({ticker!r}, interval=daily, start=2024-01-01)')
    print(db.get_price_data(ticker, 'daily', start='2024-01-01').tail())

    banner(f'2a. get_all_data({ticker!r}, wide=True) -- one column per (item, period)')
    wide = db.get_all_data(ticker, 'daily', wide=True)
    print(f'shape: {wide.shape}')
    print(f'columns: {list(wide.columns)}')
    print(wide.tail(3))

    banner(f'2b. get_all_data({ticker!r}, wide=False) -- long rows [date, item, period, value]')
    long = db.get_all_data(ticker, 'daily', wide=False)
    print(f'shape: {long.shape}')
    print(long.groupby(['item', 'period']).size().rename('rows').reset_index().to_string(index=False))

    items = {
        'close': None,            # price column, period not applicable
        'volume': None,
        'rsi': 14,                # single period
        'ema_close': [12, 26],    # list of periods
        'bollinger': None,        # group name, default period -> bb_middle/upper/lower
        'stoch_k': None,          # single column from a group, default period
        'obv': None,              # cumulative, periodless (stored as period 0)
    }
    banner(f'3a. get_items({items}, wide=True)')
    print(db.get_items(ticker, items, 'daily', start='2024-01-01', wide=True).tail())

    banner('3b. same request, wide=False')
    items_long = db.get_items(ticker, items, 'daily', start='2024-01-01', wide=False)
    print(items_long.groupby(['item', 'period']).size().rename('rows').reset_index().to_string(index=False))
    print(items_long.tail())

    banner("4a. if_missing='raise' -- request rsi period 25 (not stored)")
    db.if_missing = 'raise'
    try:
        db.get_items(ticker, {'rsi': 26})
    except MissingItemsError as e:
        print(f'raised as expected -> {e}')

    banner("4b. if_missing='add' -- same request computes, stores, and returns it")
    db.if_missing = 'add'
    added = db.get_items(ticker, {'rsi': 28}, wide=True)
    print(added.tail())

    banner("4c. rsi 25 is now stored -- 'raise' mode succeeds on a second request")
    db.if_missing = 'raise'
    print(db.get_items(ticker, {'rsi': 25}, wide=True).tail(3))


if __name__ == '__main__':
    main()
