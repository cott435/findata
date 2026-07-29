"""Compute and store derived fundamentals for edgar-seeded tickers.

Bulk-reads the stored statements + closes, runs the fundamental calculators
(findata.preprocess.calculators.fundamental), filters by SAVE_POLICY groups,
and upserts point-in-time rows into fundamental_data. Idempotent -- late
statements revise derived values in place.

Run from the project root:
    python -m scripts.db_handling.build_fundamentals --all-edgar-seeded
    python -m scripts.db_handling.build_fundamentals --tickers AAPL MSFT
"""

import argparse
import logging

from findata.configs import SAVE_POLICY, setup_logging
from findata.database.db_manager import DBManager
from findata.preprocess.calculators.fundamental import (FUNDAMENTAL_CALCULATORS,
                                                        compute_fundamentals)
from findata.utils.timing import timed

logger = logging.getLogger('scripts.build_fundamentals')

ITEM_GROUP = {spec.item: c.group for c in FUNDAMENTAL_CALCULATORS for spec in c.outputs}


def build_fundamentals(db: DBManager, tickers: list, save_policy=None) -> int:
    """Compute + upsert derived fundamentals for the given tickers."""
    save_policy = save_policy or SAVE_POLICY
    with timed(logger, f'statement + close reads ({len(tickers)} tickers)'):
        income = db.get_income(tickers)
        balance = db.get_balance(tickers)
        cashflow = db.get_cashflow(tickers)
        closes = db.get_price_data(tickers)['close']

    def per_ticker(df):
        return {t: g.drop(columns='ticker') for t, g in df.groupby('ticker')} \
            if not df.empty else {}

    inc_by, bal_by, cf_by = per_ticker(income), per_ticker(balance), per_ticker(cashflow)
    statements = {t: {'income': inc_by.get(t), 'balance': bal_by.get(t),
                      'cashflow': cf_by.get(t)}
                  for t in tickers}
    closes_by = {t: closes.xs(t, level='ticker')
                 for t in tickers if t in closes.index.get_level_values('ticker')}

    with timed(logger, f'fundamental calculators ({len(tickers)} tickers)'):
        long = compute_fundamentals(statements, closes_by)
    if long.empty:
        logger.info('No fundamentals derived (no stored statements?).')
        return 0

    allowed = {item for item, group in ITEM_GROUP.items() if save_policy.allows(group)}
    skipped = sorted(set(long['item']) - allowed)
    if skipped:
        logger.info('SAVE_POLICY skips %d item(s): %s', len(skipped), skipped)
    long = long[long['item'].isin(allowed)]
    return db.add_fundamentals(long)


def main():
    parser = argparse.ArgumentParser(description='Build derived fundamentals.')
    parser.add_argument('--tickers', nargs='+', default=None)
    parser.add_argument('--all-edgar-seeded', action='store_true')
    parser.add_argument('--db-path', default=None)
    args = parser.parse_args()
    setup_logging('build_fundamentals.log')

    db = DBManager(args.db_path)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        meta = db.get_ticker_meta()
        tickers = sorted(meta.index[meta['edgar_seeded'].fillna(False).astype(bool)])
    if not tickers:
        print('No edgar-seeded tickers, run init_edgar.py.')
        return

    n = build_fundamentals(db, tickers)
    print(f'\nUpserted {n:,} fundamental rows for {len(tickers)} ticker(s).')
    sample = db.get_fundamentals(tickers[0], daily=False, wide=True)
    if not sample.empty:
        print(f'\n{tickers[0]} latest observations:')
        print(sample.tail(3).T.to_string())


if __name__ == '__main__':
    main()
