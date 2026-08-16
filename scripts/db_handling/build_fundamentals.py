"""Compute and store derived fundamentals for edgar-seeded tickers.

Runs in batches: each batch reads its own statements + closes, computes, and
upserts before the next one starts. That bounds memory (the whole universe's
statements are ~1.5M rows) and makes an interrupted run keep everything it
had already written, rather than losing the lot at the final insert.

Only tickers whose statements are newer than their last build are processed;
``--force`` rebuilds regardless. Idempotent either way -- rows are upserted,
so late statements revise derived values in place.

Run from the project root:
    python -m scripts.db_handling.build_fundamentals --all-edgar-seeded
    python -m scripts.db_handling.build_fundamentals --tickers AAPL MSFT [--force]
"""

import argparse
import logging
from datetime import date

import pandas as pd
from tqdm.auto import tqdm

from findata.configs import SAVE_POLICY, setup_logging
from findata.database.db_manager import DBManager
from findata.preprocess.calculators.fundamental import (FUNDAMENTAL_CALCULATORS,
                                                        PRICE_EDGE_STALE_DAYS,
                                                        compute_fundamentals)

logger = logging.getLogger('scripts.build_fundamentals')

ITEM_GROUP = {spec.item: c.group for c in FUNDAMENTAL_CALCULATORS for spec in c.outputs}

# tickers per read/compute/write cycle -- small enough to keep the working set
# modest and to commit progress often, large enough that the per-batch query
# overhead stays negligible
BATCH_SIZE = 50


def tickers_needing_build(db: DBManager, tickers: list) -> list:
    """Tickers whose statements changed since their fundamentals were built.

    Two signals, because they run on different clocks. ``last_facts_date`` is
    a work timestamp like the build stamp, so "facts fetched after the last
    build" catches a newly discovered filing no matter how old the filing
    itself is -- a 10-Q filed three weeks ago and only fetched today would
    never beat today's build stamp on filing date alone. ``filed_date`` is
    the fallback for statements seeded before either stamp existed.

    The stamp is the build date rather than the newest derived observation:
    a filing that yields no derived values (missing inputs) would otherwise
    leave the observation date behind forever and re-trigger every run.
    """
    meta = db.get_ticker_meta(tickers)
    if meta.empty:
        return []
    built = pd.to_datetime(meta.get('last_fundamentals_date'), errors='coerce')
    fetched = pd.to_datetime(meta.get('last_facts_date'), errors='coerce')
    filed = pd.to_datetime(db.last_statement_dates(tickers), errors='coerce') \
              .reindex(meta.index)
    changed = pd.concat([fetched, filed], axis=1).max(axis=1)
    stale = built.isna() | (changed.notna() & (changed > built))
    return [t for t in tickers if t in meta.index and bool(stale.get(t, True))]


def _warn_if_prices_lag(db: DBManager, tickers: list):
    """Valuation items need a close near each filing date; without one the
    market cap and every ratio built on it drop out."""
    meta = db.get_ticker_meta(tickers)
    last_close = pd.to_datetime(meta['last_price_date'], errors='coerce').max()
    newest_filing = pd.to_datetime(db.last_statement_dates(tickers), errors='coerce').max()
    if pd.isna(last_close) or pd.isna(newest_filing):
        return
    lag = (newest_filing - last_close).days
    if lag > PRICE_EDGE_STALE_DAYS:
        logger.warning('Prices end %s but filings run to %s (%d days): market cap and '
                       'all valuation ratios will be NaN for the newest observations '
                       '-- run update_daily first.',
                       last_close.date(), newest_filing.date(), lag)
    elif lag > 0:
        logger.info('Prices lag the newest filing by %d day(s); market cap uses the '
                    '%s close until prices refresh.', lag, last_close.date())


def _build_batch(db: DBManager, batch: list, allowed: set) -> int:
    """Read, compute and upsert one batch; returns rows written."""
    income = db.get_income(batch)
    balance = db.get_balance(batch)
    cashflow = db.get_cashflow(batch)
    prices = db.get_price_data(batch)
    closes = prices['close'] if not prices.empty else pd.Series(dtype=float)

    def per_ticker(df):
        return {t: g.drop(columns='ticker') for t, g in df.groupby('ticker')} \
            if not df.empty else {}

    inc_by, bal_by, cf_by = per_ticker(income), per_ticker(balance), per_ticker(cashflow)
    statements = {t: {'income': inc_by.get(t), 'balance': bal_by.get(t),
                      'cashflow': cf_by.get(t)} for t in batch}
    have_closes = (set(closes.index.get_level_values('ticker'))
                   if not closes.empty else set())
    closes_by = {t: closes.xs(t, level='ticker') for t in batch if t in have_closes}

    long = compute_fundamentals(statements, closes_by)
    written = 0
    if not long.empty:
        written = db.add_fundamentals(long[long['item'].isin(allowed)])
    # stamp the whole batch, including tickers that derived nothing -- they
    # were processed, and re-running them would find the same empty result
    db.upsert_ticker_meta(pd.DataFrame(index=pd.Index(batch, name='ticker')),
                          last_fundamentals_date=date.today())
    return written


def build_fundamentals(db: DBManager, tickers: list, save_policy=None,
                       batch_size: int = BATCH_SIZE, force: bool = False,
                       progress: bool = True) -> int:
    """Compute + upsert derived fundamentals, batch by batch."""
    save_policy = save_policy or SAVE_POLICY
    tickers = list(tickers)
    targets = tickers if force else tickers_needing_build(db, tickers)
    if len(targets) < len(tickers):
        logger.info('%d of %d ticker(s) already current, skipping (--force to rebuild)',
                    len(tickers) - len(targets), len(tickers))
    if not targets:
        return 0

    allowed = {item for item, group in ITEM_GROUP.items() if save_policy.allows(group)}
    skipped = sorted(set(ITEM_GROUP) - allowed)
    if skipped:
        logger.info('SAVE_POLICY skips %d item(s): %s', len(skipped), skipped)
    _warn_if_prices_lag(db, targets)

    total = 0
    batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
    bar = tqdm(batches, desc='Fundamentals', unit='batch', disable=not progress)
    for batch in bar:
        try:
            total += _build_batch(db, batch, allowed)
        except Exception as e:
            # one bad batch should not discard the batches already committed
            logger.error('Batch %s..%s failed: %s', batch[0], batch[-1], e)
            continue
        bar.set_postfix(tickers=f'{bar.n * batch_size + len(batch)}/{len(targets)}',
                        rows=f'{total:,}')
    logger.info('Fundamentals built for %d ticker(s): %s row(s) upserted',
                len(targets), f'{total:,}')
    return total


def main():
    parser = argparse.ArgumentParser(description='Build derived fundamentals.')
    parser.add_argument('--tickers', nargs='+', default=None)
    parser.add_argument('--all-edgar-seeded', action='store_true')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild even where fundamentals are already current.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
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

    n = build_fundamentals(db, tickers, batch_size=args.batch_size, force=args.force)
    print(f'\nUpserted {n:,} fundamental rows across {len(tickers)} requested ticker(s).')
    sample = db.get_fundamentals(tickers[0], daily=False, wide=True)
    if not sample.empty:
        print(f'\n{tickers[0]} latest observations:')
        print(sample.tail(3).T.to_string())


if __name__ == '__main__':
    main()
