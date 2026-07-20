"""YF cold-start: bulk-download price history for a ticker list, compute the
full EMA/indicator history, and seed the database.

Tickers already seeded are skipped unless --force is given; for those the
last bar stored in the DB is printed next to the latest bar Yahoo has, so
staleness is visible at a glance (run update_daily to refresh them).

Run from the project root:
    python -m scripts.init_yf --tickers AAPL MSFT GOOGL [--force] [--db-path data/stock.db]
"""

import argparse
import logging

import pandas as pd
from yahooquery import Ticker

from findata.configs import setup_logging
from findata.database.db_manager import DBManager
from findata import YahooFinance
from findata.database.ticker_sampling import TickerSampler

logger = logging.getLogger('scripts.init_yf')

DEFAULT_TICKERS = TickerSampler().sample(800, 123)


def skip_report(db: DBManager, tickers: list) -> pd.DataFrame:
    """Last stored bar vs latest bar Yahoo has, for skipped tickers."""
    report = (db.get_ticker_meta(tickers)[['last_price_date']]
              .rename(columns={'last_price_date': 'db_last_date'}))
    try:
        recent = Ticker(tickers).history(interval='1d', period='5d').reset_index()
        recent['date'] = pd.to_datetime(recent['date'], utc=True).dt.tz_convert(None).dt.date
        latest = (recent.sort_values('date').groupby('symbol')[['date', 'close']].last()
                  .rename(columns={'date': 'yahoo_latest_date', 'close': 'yahoo_latest_close'}))
        report = report.join(latest)
    except Exception as e:
        logger.warning('Could not fetch latest bars for skip report: %s', e)
    return report


def main(use_all=False):
    parser = argparse.ArgumentParser(description='Cold-start YF price data.')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS,
                        help='Ticker symbols to seed.')
    parser.add_argument('--db-path', default=None,
                        help='SQLite file (default: data dir from configs).')
    parser.add_argument('--force', action='store_true',
                        help='Seed tickers even if already in the database.')
    args = parser.parse_args()
    setup_logging('init_yf.log')

    db = DBManager(args.db_path)
    tickers = TickerSampler().tickers if use_all else [t.upper() for t in args.tickers]

    meta = db.get_ticker_meta(tickers)
    seeded = set(meta.index[meta['yf_seeded'].fillna(False).astype(bool)])
    skipped = [] if args.force else [t for t in tickers if t in seeded]
    todo = [t for t in tickers if t not in skipped]

    if skipped:
        print('\nAlready seeded -- skipped (use --force to re-seed, update_daily to refresh):')

    if not todo:
        print('\nNothing to seed.')
        return

    max_tickers = 30

    for i in range(0, len(todo), max_tickers):
        subset = todo[i:i + max_tickers]
        data = YahooFinance(subset).request_ticker_financials(start='1/1/2005')
        if 'prices' not in data:
            logger.error('Download failed, , nothing inserted.')
            return
        counts = db.add_ticker_data(data['info'], data['prices'])

        print(f'\nRows inserted: {counts}')
    print(db.ticker_summary().loc[[t for t in todo if t in db.get_ticker_meta().index]])


if __name__ == '__main__':
    main(use_all=False)
