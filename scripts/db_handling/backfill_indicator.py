"""Universe-wide backfill of one technical item/period.

Computes and persists the requested (item, period) for every ticker that
doesn't have it stored yet (the on-the-fly path with persist forced on) --
use before wanting a custom window in universe-wide workbench views instead
of paying the compute on first view.

Run from the project root:
    python -m scripts.db_handling.backfill_indicator --item rsi --period 25
        [--tickers AAPL MSFT | --all-seeded] [--db-path data/stock.db]
"""

import argparse
import logging

from findata.configs import setup_logging
from findata.database.db_manager import DBManager

logger = logging.getLogger('scripts.backfill_indicator')


def main():
    parser = argparse.ArgumentParser(description='Backfill one technical item/period.')
    parser.add_argument('--item', required=True,
                        help="Output item or calculator name ('rsi', 'bollinger', ...).")
    parser.add_argument('--period', type=int, required=True)
    parser.add_argument('--tickers', nargs='+', default=None)
    parser.add_argument('--all-seeded', action='store_true',
                        help='All technicals-seeded tickers (default when no --tickers).')
    parser.add_argument('--db-path', default=None)
    args = parser.parse_args()
    setup_logging('backfill_indicator.log')

    db = DBManager(args.db_path)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        meta = db.get_ticker_meta()
        tickers = sorted(meta.index[meta['technicals_seeded'].fillna(False).astype(bool)])
    logger.info('Backfilling %s_%d for %d ticker(s)', args.item, args.period, len(tickers))

    result = db.get_items(tickers, {args.item: args.period}, persist=True)
    filled = result.notna().sum()
    print(f'\nStored values per column across {len(tickers)} tickers:')
    print(filled.to_string())


if __name__ == '__main__':
    main()
