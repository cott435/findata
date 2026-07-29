"""Daily incremental runner: pull new bars for every yf_seeded ticker and
extend the stored EMAs/indicators from their seeds. The EDGAR refresh slots
into update_edgar() once part 2 is implemented.

Safe for cron -- inserts are INSERT OR IGNORE, so overlapping or repeated
runs are harmless.

Run from the project root:
    python -m scripts.update_daily [--tickers AAPL MSFT] [--dry-run] [--db-path data/stock.db]
"""

import argparse
import logging
from datetime import date, timedelta

from findata.configs import setup_logging
from findata.database.db_manager import DBManager
from findata import YahooFinance

logger = logging.getLogger('scripts.update_daily')

# fetch this many days behind the stalest ticker so the week that was in
# progress at the last update is re-aggregated into a complete weekly bar
WEEKLY_OVERLAP_DAYS = 7


def update_yf(db: DBManager, tickers: list = None, dry_run: bool = False) -> dict:
    """Fetch bars since each ticker's last_price_date and update the DB."""
    meta = db.get_ticker_meta()
    seeded = meta[meta['yf_seeded'].fillna(False).astype(bool)]
    tickers = list(meta.index) if tickers is None else [t.upper() for t in tickers]
    unknown = sorted(set(tickers) - set(seeded.index))
    if unknown:
        logger.warning('Not yf-seeded, skipping: %s (run init_yf first)', unknown)
    seeded = seeded[seeded.index.isin(tickers)]
    if seeded.empty:
        logger.info('No seeded tickers to update.')
        return {}

    last_dates = seeded['last_price_date'].fillna(date(1900, 1, 1))
    start = last_dates.min() - timedelta(days=WEEKLY_OVERLAP_DAYS)
    logger.info('Updating %d ticker(s); stalest last_price_date=%s, fetching from %s',
                len(seeded), last_dates.min(), start)

    data = YahooFinance(list(seeded.index)).request_ticker_financials(start=str(start))
    if 'prices' not in data:
        logger.error('Price download failed, aborting YF update.')
        return {}
    prices = data['prices']

    # report bars strictly newer than each ticker's stored history (the
    # insert itself is conflict-safe either way)
    idx = prices.index.to_frame(index=False)
    new_daily = idx[(idx['interval'] == 'daily')
                    & (idx['date'] > idx['ticker'].map(last_dates))]
    per_ticker = new_daily.groupby('ticker').size()
    for ticker in seeded.index:
        logger.info('%s: %d new daily bar(s) after %s',
                    ticker, per_ticker.get(ticker, 0), last_dates[ticker])

    if dry_run:
        logger.info('Dry run -- nothing written.')
        return {}
    return db.update_ticker_data(prices, info=data['info'])


def update_edgar(db: DBManager, tickers: list = None, dry_run: bool = False) -> dict:
    """Incremental EDGAR update: filings after last_filings_date / Form 4s
    after last_form4_date, plus a facts refresh, for edgar-seeded tickers.

    Tickers seeded numeric-only (no last_filings_date) refresh in numeric
    mode; the rest run full.
    """
    from findata.database.edgar_ import EdgarPipeline

    meta = db.get_ticker_meta()
    seeded = meta[meta['edgar_seeded'].fillna(False).astype(bool)]
    if tickers is not None:
        seeded = seeded[seeded.index.isin([t.upper() for t in tickers])]
    if seeded.empty:
        logger.info('No edgar-seeded tickers to update.')
        return {}
    if dry_run:
        logger.info('Dry run -- would check %d ticker(s) for filings after %s.',
                    len(seeded), dict(seeded['last_filings_date']))
        return {}

    full = seeded[seeded['last_filings_date'].notna()]
    numeric = seeded[seeded['last_filings_date'].isna()]
    counts = {}
    if not full.empty:
        counts = EdgarPipeline(db=db).run(list(full.index))
    if not numeric.empty:
        numeric_counts = EdgarPipeline(db=db, mode='numeric').run(list(numeric.index))
        for key, n in numeric_counts.items():
            counts[key] = counts.get(key, 0) + n
    return counts


def main():
    parser = argparse.ArgumentParser(description='Incremental daily update for seeded tickers.')
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Subset to update (default: every yf_seeded ticker).')
    parser.add_argument('--db-path', default=None,
                        help='SQLite file (default: data dir from configs).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report new bars without writing anything.')
    args = parser.parse_args()
    setup_logging('update_daily.log')

    db = DBManager(args.db_path)
    counts = update_yf(db, tickers=args.tickers, dry_run=args.dry_run)
    edgar_counts = update_edgar(db, tickers=args.tickers, dry_run=args.dry_run)

    statement_rows = sum(edgar_counts.get(k, 0) for k in ('income', 'balance', 'cashflow'))
    if statement_rows:
        from scripts.db_handling.build_fundamentals import build_fundamentals
        meta = db.get_ticker_meta()
        seeded = sorted(meta.index[meta['edgar_seeded'].fillna(False).astype(bool)])
        if args.tickers is not None:
            seeded = [t for t in seeded if t in {s.upper() for s in args.tickers}]
        n_fund = build_fundamentals(db, seeded)
        logger.info('Fundamentals refreshed: %d row(s) upserted', n_fund)

    if counts or edgar_counts:
        print(f'\nYF rows inserted: {counts}')
        print(f'EDGAR rows inserted: {edgar_counts}')
        print(db.ticker_summary())


if __name__ == '__main__':
    main()
