"""EDGAR cold-start: per-ticker download of 10-K/10-Q/8-K text sections,
Form 4 insider transactions, and XBRL financial facts. Slow by design (SEC
rate limits) -- intended for overnight or batched runs.

Tickers already edgar-seeded are skipped (their last stored dates are
printed) unless --force is given. Interrupted runs are resumable: filings
already on disk/DB are not refetched.

Run from the project root:
    python -m scripts.db_handling.init_edgar --tickers AAPL MSFT [--extra] [--force]
        [--min-date 2015-01-01] [--db-path data/stock.db]

Also runnable directly (e.g. `python init_edgar.py` from this directory, or
an IDE debug config that executes the file rather than the module) -- the
sys.path bootstrap below makes the ``scripts.db_handling.init_yf`` absolute
import resolve either way, so this file doesn't need relative imports.
"""

import argparse
import logging
import sys
from pathlib import Path

if __name__ == '__main__' and __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from findata.configs import setup_logging
from findata.database.db_manager import DBManager
from findata.database.edgar_ import EdgarPipeline
from findata.database.ticker_sampling import TickerSampler
from scripts.db_handling.init_yf import DEFAULT_TICKERS

logger = logging.getLogger('scripts.init_edgar')


def main(use_all=False):
    parser = argparse.ArgumentParser(description='Cold-start EDGAR filing + financials data.')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS, help='Ticker symbols to seed.')
    parser.add_argument('--db-path', default=None, help='SQLite file (default: data dir from configs).')
    parser.add_argument('--data', choices=('full', 'numeric'), default='full',
                        help="'full' = text sections + facts + Form 4; "
                             "'numeric' = XBRL facts + Form 4 only (no text downloads).")
    parser.add_argument('--extra', action='store_true', help='Also parse the non-essential sections.')
    parser.add_argument('--force', action='store_true', help='Refetch even if already seeded.')
    parser.add_argument('--min-date', default=None, help='Oldest filing date to pull (YYYY-MM-DD).')
    parser.add_argument('--rps', type=float, default=8.0, help='Max SEC requests per second.')
    parser.add_argument('--concurrency', type=int, default=4, help='Max in-flight requests.')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip the quarterly-coverage check on parsed statements.')
    parser.add_argument('--grace-years', type=int, default=2,
                        help="Leading fiscal years per item exempt from the coverage "
                             "check (early filings are routinely incomplete).")
    args = parser.parse_args()
    setup_logging('init_edgar.log')
    db = DBManager(args.db_path)
    tickers = TickerSampler(use_all=True).tickers if use_all else [t.upper() for t in args.tickers]

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
    #args.data = 'numeric'
    pipeline = EdgarPipeline(db=db, parse_extra_data=args.extra, mode=args.data,
                             max_concurrency=args.concurrency, requests_per_second=args.rps,
                             validate=not args.no_validate, grace_years=args.grace_years)

    counts = pipeline.run(todo, force=args.force, min_date=args.min_date)

    print(f'\nInserted: {counts}')
    print(db.get_ticker_meta(todo)[['name', 'last_filings_date', 'last_form4_date', 'edgar_seeded']])


if __name__ == '__main__':
    main()
