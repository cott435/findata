"""EDGAR cold-start: per-ticker download of 10-K/10-Q/8-K text sections,
Form 4 insider transactions, and XBRL financial facts. Slow by design (SEC
rate limits) -- intended for overnight or batched runs.

Tickers already edgar-seeded are skipped (their last stored dates are
printed) unless --force is given. Interrupted runs are resumable: filings
already on disk/DB are not refetched.

Run from the project root:
    python -m scripts.init_edgar --tickers AAPL MSFT [--extra] [--force]
        [--min-date 2015-01-01] [--db-path data/stock.db]
"""

import argparse
import logging

from findata.db_manager import DBManager
from findata.edgar_ import EdgarPipeline
from findata.ticker_sampling import TickerSampler
from scripts.init_yf import DEFAULT_TICKERS

logger = logging.getLogger('scripts.init_edgar')


def main(use_all=False):
    parser = argparse.ArgumentParser(description='Cold-start EDGAR filing + financials data.')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS, help='Ticker symbols to seed.')
    parser.add_argument('--db-path', default=None, help='SQLite file (default: data dir from configs).')
    parser.add_argument('--extra', action='store_true', help='Also parse the non-essential sections.')
    parser.add_argument('--force', action='store_true', help='Refetch even if already seeded.')
    parser.add_argument('--min-date', default=None, help='Oldest filing date to pull (YYYY-MM-DD).')
    parser.add_argument('--rps', type=float, default=8.0, help='Max SEC requests per second.')
    parser.add_argument('--concurrency', type=int, default=4, help='Max in-flight requests.')
    parser.add_argument('--decumulate-cashflow', action='store_true',
                        help='Also store de-cumulated single-quarter cash flow rows.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s | %(message)s')
    args.min_date = '2013-01-01'
    db = DBManager(args.db_path)
    tickers = TickerSampler(use_russell3000=False).tickers if use_all else [t.upper() for t in args.tickers]

    meta = db.get_ticker_meta(tickers)
    seeded = set(meta.index[meta['edgar_seeded'].fillna(False).astype(bool)])
    skipped = [] if args.force else [t for t in tickers if t in seeded]
    todo = [t for t in tickers if t not in skipped]

    if skipped:
        print('\nAlready edgar-seeded -- skipped (use --force to re-seed, update_daily to refresh):')
        print(meta.loc[skipped, ['last_filings_date', 'last_form4_date']])

    if not todo:
        print('\nNothing to seed.')
        return

    pipeline = EdgarPipeline(db=db, parse_extra_data=args.extra,
                             max_concurrency=args.concurrency, requests_per_second=args.rps,
                             decumulate_cashflow=args.decumulate_cashflow)
    counts = pipeline.run(todo, force=args.force, min_date=args.min_date)

    print(f'\nInserted: {counts}')
    print(db.get_ticker_meta(todo)[['name', 'last_filings_date', 'last_form4_date', 'edgar_seeded']])


if __name__ == '__main__':
    main(use_all=True)
