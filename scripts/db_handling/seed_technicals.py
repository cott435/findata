"""Promote prices-only tickers to full technical coverage.

Computes the complete default technical history from the stored prices and
upserts it (the explicit replacement for the old silent cold-compute
fallback inside incremental updates). After promotion, update_daily
maintains the ticker's technicals like any fully-seeded one.

Run from the project root:
    python -m scripts.db_handling.seed_technicals --tickers AAPL MSFT
    python -m scripts.db_handling.seed_technicals --all-prices-only
"""

import argparse
import logging

from tqdm.auto import tqdm

import findata.preprocess.calculators.technical as calc
from findata.configs import setup_logging
from findata.database.db_manager import DBManager

logger = logging.getLogger('scripts.seed_technicals')


def seed(db: DBManager, tickers: list, interval: str = 'daily') -> dict:
    counts = {}
    for ticker in tqdm(tickers, desc='Seeding technicals', unit='ticker'):
        prices = db.get_price_data(ticker, interval)
        if prices.empty:
            logger.warning('%s: no stored prices, skipped', ticker)
            continue
        calc_df = calc.calculate_all(prices[['open', 'high', 'low', 'close', 'volume']])
        wide = db._calc_to_wide(calc_df, ticker, interval)
        with db._raw_txn() as conn:
            counts[ticker] = db._upsert_technicals(wide, conn=conn)
        db.upsert_ticker_meta(db.get_ticker_meta([ticker])[[]], technicals_seeded=True)
        logger.debug('%s: %d technical rows', ticker, counts[ticker])
    return counts


def main():
    parser = argparse.ArgumentParser(description='Promote prices-only tickers to full technicals.')
    parser.add_argument('--tickers', nargs='+', default=None)
    parser.add_argument('--all-prices-only', action='store_true',
                        help='Every yf-seeded ticker without technicals_seeded.')
    parser.add_argument('--db-path', default=None)
    args = parser.parse_args()
    setup_logging('seed_technicals.log')

    db = DBManager(args.db_path)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.all_prices_only:
        meta = db.get_ticker_meta()
        prices_only = (meta['yf_seeded'].fillna(False).astype(bool)
                       & ~meta['technicals_seeded'].fillna(False).astype(bool))
        tickers = sorted(meta.index[prices_only])
    else:
        parser.error('Pass --tickers or --all-prices-only.')

    if not tickers:
        print('Nothing to promote.')
        return
    counts = seed(db, tickers)
    print(f'\nPromoted {len(counts)} ticker(s); technical rows written: {sum(counts.values()):,}')


if __name__ == '__main__':
    main()
