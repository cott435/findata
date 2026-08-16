"""Daily incremental runner, in three stages:

    1. prices     new bars for every yf_seeded ticker + their technicals
    2. edgar      new filings/Form 4s, and an XBRL facts refresh for tickers
                  that have actually reported since we last parsed them
    3. fundamentals  recomputed for whichever tickers stage 2 actually touched

Stage 2 is the expensive one: XBRL facts come as one whole-history frame per
ticker, so there is no such thing as fetching "just the new ones" -- every
refreshed ticker costs two SEC requests and a full re-parse. It is therefore
driven by an event, not a timer: SEC publishes a market-wide filing index, so
one request tells the runner which companies have filed a 10-K/10-Q newer
than the newest one already parsed, and only those get refetched. Tickers
missing from the index fall back to the --facts-max-age window, so a lookup
failure slows the cadence instead of freezing the financials.

Run the stages independently with --stages:
    python -m scripts.db_handling.update_daily --stages prices
    python -m scripts.db_handling.update_daily              # all three
    python -m scripts.db_handling.update_daily --stages edgar fundamentals
        [--facts-max-age 7] [--tickers AAPL MSFT] [--dry-run]

Safe for cron -- price inserts are conflict-safe and statement rows are
replaced per ticker, so overlapping or repeated runs are harmless.
"""

import argparse
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from findata.configs import setup_logging
from findata.database.db_manager import DBManager
from findata import YahooFinance

logger = logging.getLogger('scripts.update_daily')

# fetch this many days behind the stalest ticker so the week that was in
# progress at the last update is re-aggregated into a complete weekly bar
WEEKLY_OVERLAP_DAYS = 7

# fallback cadence for tickers the filing index has no entry for
DEFAULT_FACTS_MAX_AGE = 7

STAGES = ('prices', 'edgar', 'fundamentals')

# a session's daily bar is only final after the close; before then the running
# bar would overwrite good history with a partial one
MARKET_TZ = ZoneInfo('America/New_York')
MARKET_CLOSE_HOUR = 16

# a ticker this far behind that still returns nothing has stopped trading
# (delisted, renamed, acquired) rather than merely lagging
INACTIVE_AFTER_DAYS = 10
# ... and is left alone for this long before being tried again, so a symbol
# that resumes is picked back up without polling dead ones every day
INACTIVE_RECHECK_DAYS = 30


def latest_expected_session(now: datetime = None) -> date:
    """The most recent trading session whose daily bar should be complete.

    Weekday heuristic rather than a full exchange calendar: intraday, the
    answer is yesterday (today's bar is still moving), and weekends fall back
    to Friday. Holidays are not modelled -- the cost of thinking a holiday was
    a session is one wasted request round, and it self-corrects on the next
    real session, which is cheaper than carrying a calendar dependency.
    """
    now = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    session = now.date()
    if now.hour < MARKET_CLOSE_HOUR:
        session -= timedelta(days=1)
    while session.weekday() >= 5:            # Sat/Sun -> back to Friday
        session -= timedelta(days=1)
    return session


def update_yf(db: DBManager, tickers: list = None, dry_run: bool = False,
              include_inactive: bool = False) -> dict:
    """Bring every seeded ticker up to the latest completed session.

    Only tickers actually behind that session are fetched, and only back to
    the stalest of those: previously one long-neglected ticker set the start
    date for the whole universe, turning a daily top-up into a months-long
    re-download of 800+ symbols.
    """
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

    if not include_inactive and 'inactive_since' in seeded.columns:
        recheck = date.today() - timedelta(days=INACTIVE_RECHECK_DAYS)
        dormant = seeded['inactive_since'].notna() & (seeded['inactive_since'] >= recheck)
        if dormant.any():
            logger.info('Skipping %d inactive ticker(s) (rechecked every %d days; '
                        '--include-inactive to force): %s', int(dormant.sum()),
                        INACTIVE_RECHECK_DAYS, sorted(seeded.index[dormant])[:10])
            seeded = seeded[~dormant]
    if seeded.empty:
        logger.info('Every requested ticker is inactive.')
        return {}

    last_dates = seeded['last_price_date'].fillna(date(1900, 1, 1))
    # Target the newest session anyone could have: usually the last completed
    # one, but never behind what is already stored (so a run made before the
    # close, or on a holiday, does not declare the whole universe stale).
    stored_max = last_dates.max()
    target = max(latest_expected_session(), stored_max)
    # When the target is a date somebody already has a bar for, it is a
    # confirmed session -- an empty result then says something about the
    # ticker. When it comes from the calendar guess alone it may be a
    # holiday, and an empty result says nothing.
    session_confirmed = stored_max >= target
    behind = last_dates[last_dates < target]
    if behind.empty:
        logger.info('All %d ticker(s) already current through %s.', len(seeded), target)
        return {}

    # only the stale tickers are fetched, and only back to the stalest of
    # THOSE -- one long-neglected ticker used to drag the whole universe into
    # a full-history re-download
    stale = sorted(behind.index)
    start = behind.min() - timedelta(days=WEEKLY_OVERLAP_DAYS)
    logger.info('Target session %s: %d of %d ticker(s) behind, fetching from %s',
                target, len(stale), len(seeded), start)

    yf = YahooFinance(stale)
    data = yf.request_ticker_financials(start=str(start))
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
    if per_ticker.empty and not session_confirmed:
        logger.info('No new bars returned for %s -- likely not a trading session.', target)
    for ticker in stale:
        logger.info('%s: %d new daily bar(s) after %s',
                    ticker, per_ticker.get(ticker, 0), last_dates[ticker])

    # A ticker well past a confirmed session that still returns nothing has
    # stopped trading; one that produces bars again is revived.
    newly_inactive, revived = [], []
    if session_confirmed:
        newly_inactive = [t for t in stale if per_ticker.get(t, 0) == 0
                          and (target - last_dates[t]).days > INACTIVE_AFTER_DAYS]
        revived = [t for t in stale if per_ticker.get(t, 0) > 0
                   and pd.notna(seeded.loc[t].get('inactive_since'))]

    if dry_run:
        if newly_inactive:
            logger.info('Dry run -- would mark inactive: %s', newly_inactive)
        logger.info('Dry run -- nothing written.')
        return {}

    if newly_inactive:
        logger.info('No bars since %s for %d ticker(s); marking inactive: %s',
                    target, len(newly_inactive), newly_inactive)
        db.set_inactive(newly_inactive)
    if revived:
        logger.info('Bars returned again for %s; clearing inactive flag', revived)
        db.set_inactive(revived, active=True)
    return db.update_ticker_data(prices, info=data['info'])


def facts_are_stale(db: DBManager, candidates,
                    facts_max_age: int = DEFAULT_FACTS_MAX_AGE) -> pd.Series:
    """Boolean mask over ``candidates`` (a ticker-indexed meta frame): does
    this ticker's XBRL facts need refetching?

    The signal is SEC's own market-wide filing index: one request lists every
    filing by every company, so "who has reported since we last looked?" is
    answered for the whole universe at once and exactly the tickers that
    actually reported get refetched. Tickers absent from the index fall back
    to the age window, so a lookup failure degrades to a slower cadence
    rather than silently freezing the financials.

    The floor is the later of what we parsed and when we last fetched -- a
    filing that carries no XBRL would otherwise never advance the parsed date
    and would re-trigger forever.
    """
    from findata.database.edgar_ import latest_statement_filings

    cutoff = date.today() - timedelta(days=facts_max_age)
    by_age = ~(candidates['last_facts_date'].notna()
               & (candidates['last_facts_date'] >= cutoff)) if facts_max_age > 0 \
        else pd.Series(True, index=candidates.index)

    # everything compared here is normalized to datetime64 first: these
    # columns hold python dates, and pandas 3 refuses to reduce or compare
    # them once a missing value turns the column to object dtype
    stored = pd.to_datetime(db.last_statement_dates(list(candidates.index)),
                            errors='coerce').reindex(candidates.index)
    fetched = pd.to_datetime(candidates['last_facts_date'], errors='coerce')
    floor = pd.concat([stored, fetched], axis=1).max(axis=1)
    # look back to the oldest thing we would need to beat, so nothing is missed
    since = floor.min()
    since = None if pd.isna(since) else since.date()

    # prefer the CIKs already stored for these tickers over a fresh lookup
    stored_ciks = (candidates['cik'].dropna().astype(int)
                   if 'cik' in candidates.columns else None)
    try:
        latest = latest_statement_filings(list(candidates.index), since=since,
                                          ticker_cik=stored_ciks)
    except Exception as e:
        logger.warning('EDGAR filing index lookup failed (%s); falling back to the '
                       '%d-day age window', e, facts_max_age)
        return by_age
    if latest.empty:
        logger.info('EDGAR index returned no filings; falling back to the %d-day window',
                    facts_max_age)
        return by_age

    known = pd.to_datetime(latest, errors='coerce').reindex(candidates.index)
    covered = known.notna()
    stale = pd.Series(False, index=candidates.index)
    # a filing newer than both what we parsed and when we last fetched
    stale[covered] = (floor[covered].isna() | (known[covered] > floor[covered])).to_numpy()
    stale[~covered] = by_age[~covered]
    logger.info('Facts refresh: %d of %d ticker(s) have a newer filing '
                '(%d resolved from the EDGAR index, %d via the %d-day window)',
                int(stale.sum()), len(stale), int(covered.sum()),
                int((~covered).sum()), facts_max_age)
    return stale


def update_edgar(db: DBManager, tickers: list = None, dry_run: bool = False,
                 facts_max_age: int = DEFAULT_FACTS_MAX_AGE) -> tuple:
    """Incremental EDGAR update for edgar-seeded tickers.

    Filings and Form 4s are already incremental (fetched after the stored
    last_filings_date / last_form4_date). The XBRL facts are not -- they only
    come as one full-history frame per ticker -- so a ticker is refetched only
    when the EDGAR index shows a statement filing newer than the one parsed
    (see :func:`facts_are_stale`). Tickers seeded numeric-only (no
    last_filings_date) refresh in numeric mode; the rest run full.

    Returns ``(counts, refreshed_tickers)``.
    """
    from findata.database.edgar_ import EdgarPipeline

    meta = db.get_ticker_meta()
    seeded = meta[meta['edgar_seeded'].fillna(False).astype(bool)]
    if tickers is not None:
        seeded = seeded[seeded.index.isin([t.upper() for t in tickers])]
    if seeded.empty:
        logger.info('No edgar-seeded tickers to update.')
        return {}, []

    seeded = seeded[facts_are_stale(db, seeded, facts_max_age)]
    if seeded.empty:
        logger.info('All EDGAR data is current, nothing to refresh.')
        return {}, []

    if dry_run:
        logger.info('Dry run -- would refresh EDGAR for %d ticker(s).', len(seeded))
        return {}, []

    full = seeded[seeded['last_filings_date'].notna()]
    numeric = seeded[seeded['last_filings_date'].isna()]
    counts = {}
    if not full.empty:
        counts = EdgarPipeline(db=db).run(list(full.index))
    if not numeric.empty:
        numeric_counts = EdgarPipeline(db=db, mode='numeric').run(list(numeric.index))
        for key, n in numeric_counts.items():
            counts[key] = counts.get(key, 0) + n
    return counts, list(seeded.index)


def main():
    parser = argparse.ArgumentParser(
        description='Incremental daily update for seeded tickers.')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES),
                        metavar='STAGE',
                        help=f"Stages to run, any of {', '.join(STAGES)} (default: all).")
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Subset to update (default: every seeded ticker).')
    parser.add_argument('--facts-max-age', type=int, default=DEFAULT_FACTS_MAX_AGE,
                        help='Skip EDGAR tickers whose facts were fetched within this '
                             'many days (0 refreshes every ticker).')
    parser.add_argument('--db-path', default=None,
                        help='SQLite file (default: data dir from configs).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without writing anything.')
    parser.add_argument('--include-inactive', action='store_true',
                        help='Also poll tickers flagged as no longer trading.')
    args = parser.parse_args()
    setup_logging('update_daily.log')

    db = DBManager(args.db_path)
    logger.info('Stages: %s', ', '.join(args.stages))
    counts, edgar_counts, refreshed = {}, {}, []

    if 'prices' in args.stages:
        counts = update_yf(db, tickers=args.tickers, dry_run=args.dry_run,
                           include_inactive=args.include_inactive)
    if 'edgar' in args.stages:
        edgar_counts, refreshed = update_edgar(db, tickers=args.tickers,
                                               dry_run=args.dry_run,
                                               facts_max_age=args.facts_max_age)

    if 'fundamentals' in args.stages and not args.dry_run:
        # only the tickers EDGAR actually refreshed need recomputing; running
        # the stage on its own rebuilds the whole seeded universe
        if 'edgar' in args.stages:
            targets = refreshed
        else:
            meta = db.get_ticker_meta()
            targets = sorted(meta.index[meta['edgar_seeded'].fillna(False).astype(bool)])
            if args.tickers is not None:
                targets = [t for t in targets if t in {s.upper() for s in args.tickers}]
        if targets:
            from scripts.db_handling.build_fundamentals import build_fundamentals
            n_fund = build_fundamentals(db, targets)
            logger.info('Fundamentals refreshed for %d ticker(s): %d row(s) upserted',
                        len(targets), n_fund)
        else:
            logger.info('No tickers needed a fundamentals rebuild.')

    if counts or edgar_counts:
        print(f'\nYF rows inserted: {counts}')
        print(f'EDGAR rows inserted: {edgar_counts}')
        print(db.ticker_summary())


if __name__ == '__main__':
    main()
