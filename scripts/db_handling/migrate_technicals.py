"""One-off migration: fold the long ema_data + indicator_data tables into the
wide technical_data table (one row per bar, one column per item_period).

Additive by default and resumable: fills technical_data in ticker-batch
transactions (a killed run leaves whole tickers present-or-absent and the
next run continues), then verifies value parity against the long tables --
sampled per-ticker bitwise comparison plus global per-combo counts. Nothing
is destroyed until ``--finalize``, which renames the database to
``<name>.pre_migration`` and rebuilds a compact ``<name>`` without the two
long tables. Keep the .pre_migration file until you're satisfied.

Run from the project root (no other writers while it runs):
    python -m scripts.db_handling.migrate_technicals [--db-path data/stock.db]
        [--batch-size 40] [--verify-sample 25] [--finalize]
"""

import argparse
import logging
import random
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from findata.configs import setup_logging
from findata.database.db_manager import DBManager
from findata.preprocess.calculators.base import column_name

logger = logging.getLogger('scripts.migrate_technicals')

LONG_TABLES = ('ema_data', 'indicator_data')


def stored_combos(conn) -> list[tuple[str, int]]:
    """Every (item, period) present in the long tables (registry or not)."""
    combos = {(item, period) for item, period in
              conn.execute('SELECT DISTINCT indicator, period FROM indicator_data')}
    combos |= {(f'ema_{base}', period) for base, period in
               conn.execute('SELECT DISTINCT base, period FROM ema_data')}
    dropped = {(i, p) for i, p in combos if i is None}
    if dropped:
        logger.warning('Ignoring %d NULL-item combo(s) in long tables', len(dropped))
    return sorted(combos - dropped)


def migrate_batches(db: DBManager, conn, combos, batch_size: int) -> int:
    ind_combos = [(i, p) for i, p in combos if not i.startswith('ema_')]
    ema_combos = [(i[4:], p, column_name(i, p)) for i, p in combos if i.startswith('ema_')]

    universe = sorted({t for (t,) in conn.execute(
        'SELECT DISTINCT ticker FROM indicator_data '
        'UNION SELECT DISTINCT ticker FROM ema_data')})
    done = {t for (t,) in conn.execute('SELECT DISTINCT ticker FROM technical_data')}
    todo = [t for t in universe if t not in done]
    logger.info('Migrating %d ticker(s) (%d already done) into %d columns',
                len(todo), len(done), len(combos))
    if not todo:
        return 0

    ind_cols = [column_name(i, p) for i, p in ind_combos]
    ind_select = ', '.join('MAX(CASE WHEN indicator=? AND period=? THEN value END)'
                           for _ in ind_combos)
    ema_select = ', '.join('MAX(CASE WHEN base=? AND period=? THEN ema_value END)'
                           for _ in ema_combos)
    ema_cols = [col for _, _, col in ema_combos]

    written = 0
    for i in tqdm(range(0, len(todo), batch_size), desc='Migrating', unit='batch'):
        batch = todo[i:i + batch_size]
        marks = ','.join('?' * len(batch))
        conn.execute('BEGIN')
        try:
            quoted = ', '.join(f'"{c}"' for c in ind_cols)
            cur = conn.execute(
                f'INSERT INTO technical_data (ticker, date, interval, {quoted}) '
                f'SELECT ticker, date, freq, {ind_select} FROM indicator_data '
                f'WHERE ticker IN ({marks}) GROUP BY ticker, date, freq '
                f'ON CONFLICT(ticker, date, interval) DO UPDATE SET '
                + ', '.join(f'"{c}"=COALESCE(excluded."{c}", "{c}")' for c in ind_cols),
                [x for combo in ind_combos for x in combo] + batch)
            written += cur.rowcount
            quoted = ', '.join(f'"{c}"' for c in ema_cols)
            cur = conn.execute(
                f'INSERT INTO technical_data (ticker, date, interval, {quoted}) '
                f'SELECT ticker, date, interval, {ema_select} FROM ema_data '
                f'WHERE ticker IN ({marks}) GROUP BY ticker, date, interval '
                f'ON CONFLICT(ticker, date, interval) DO UPDATE SET '
                + ', '.join(f'"{c}"=COALESCE(excluded."{c}", "{c}")' for c in ema_cols),
                [x for base, period, _ in ema_combos for x in (base, period)] + batch)
            written += cur.rowcount
            conn.execute('COMMIT')
        except BaseException:
            conn.execute('ROLLBACK')
            raise
    return written


def verify_sampled(conn, combos, sample_n: int) -> bool:
    tickers = [t for (t,) in conn.execute('SELECT DISTINCT ticker FROM technical_data')]
    sample = sorted(random.Random(11).sample(tickers, min(sample_n, len(tickers))))
    cols = [column_name(i, p) for i, p in combos]
    ok = True
    for ticker in tqdm(sample, desc='Verifying', unit='ticker'):
        ind = pd.read_sql_query(
            'SELECT date, freq AS interval, indicator AS item, period, value '
            'FROM indicator_data WHERE ticker=?', conn, params=(ticker,))
        ema = pd.read_sql_query(
            'SELECT date, interval, base, period, ema_value AS value '
            'FROM ema_data WHERE ticker=?', conn, params=(ticker,))
        ema['item'] = 'ema_' + ema['base']
        long = pd.concat([ind, ema.drop(columns='base')], ignore_index=True).dropna(subset=['item'])
        long['column'] = [column_name(i, p) for i, p in zip(long['item'], long['period'])]
        expected = long.pivot(index=['interval', 'date'], columns='column', values='value')

        quoted = ', '.join(f'"{c}"' for c in cols)
        wide = pd.read_sql_query(
            f'SELECT interval, date, {quoted} FROM technical_data WHERE ticker=?',
            conn, params=(ticker,)).set_index(['interval', 'date'])

        for col in expected.columns:
            exp = expected[col].dropna()
            got = wide[col].dropna() if col in wide.columns else pd.Series(dtype=float)
            if len(exp) != len(got) or not (exp.sort_index() == got.sort_index().reindex(exp.sort_index().index)).all():
                logger.error('%s %s: parity FAILED (%d long vs %d wide values)',
                             ticker, col, len(exp), len(got))
                ok = False
    return ok


def verify_counts(conn, combos) -> bool:
    logger.info('Global per-combo count check (full scans)...')
    long_counts = {}
    for item, period, n in conn.execute(
            'SELECT indicator, period, COUNT(value) FROM indicator_data GROUP BY 1, 2'):
        if item is not None:
            long_counts[column_name(item, period)] = n
    for base, period, n in conn.execute(
            'SELECT base, period, COUNT(ema_value) FROM ema_data GROUP BY 1, 2'):
        long_counts[column_name(f'ema_{base}', period)] = n

    cols = [column_name(i, p) for i, p in combos]
    counts_sql = ', '.join(f'COUNT("{c}")' for c in cols)
    wide_counts = dict(zip(cols, conn.execute(
        f'SELECT {counts_sql} FROM technical_data').fetchone()))

    ok = True
    for col, n_long in sorted(long_counts.items()):
        n_wide = wide_counts.get(col, 0)
        if n_long != n_wide:
            logger.error('%s: count mismatch long=%d wide=%d', col, n_long, n_wide)
            ok = False
    logger.info('Count check %s (%d columns, %s stored values)',
                'PASSED' if ok else 'FAILED', len(long_counts),
                f'{sum(long_counts.values()):,}')
    return ok


def finalize(db_path: Path):
    """Rename the DB aside and rebuild it compact, without the long tables."""
    backup = db_path.with_name(db_path.name + '.pre_migration')
    if backup.exists():
        raise SystemExit(f'{backup} already exists -- refusing to overwrite. '
                         f'Resolve the previous finalize first.')
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    conn.close()
    db_path.rename(backup)
    logger.info('Renamed %s -> %s', db_path.name, backup.name)

    new = sqlite3.connect(db_path)
    try:
        new.execute('PRAGMA journal_mode=OFF')
        new.execute('PRAGMA synchronous=OFF')
        new.execute('ATTACH DATABASE ? AS src', (str(backup),))
        tables = new.execute(
            "SELECT name, sql FROM src.sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'").fetchall()
        kept = [(n, s) for n, s in tables if n not in LONG_TABLES]
        for name, create_sql in kept:
            new.execute(create_sql)
            new.execute(f'INSERT INTO main."{name}" SELECT * FROM src."{name}"')
        new.commit()
        for name, _ in kept:
            (n_old,) = new.execute(f'SELECT COUNT(*) FROM src."{name}"').fetchone()
            (n_new,) = new.execute(f'SELECT COUNT(*) FROM main."{name}"').fetchone()
            if n_old != n_new:
                raise SystemExit(f'{name}: rebuilt row count {n_new} != source {n_old}; '
                                 f'restore by deleting {db_path.name} and renaming '
                                 f'{backup.name} back.')
        new.execute('DETACH DATABASE src')
        new.execute('PRAGMA journal_mode=WAL')
    finally:
        new.close()
    old_gib = backup.stat().st_size / 2**30
    new_gib = db_path.stat().st_size / 2**30
    logger.info('Finalized: %.2f GiB -> %.2f GiB (backup kept at %s)',
                old_gib, new_gib, backup.name)


def main():
    parser = argparse.ArgumentParser(description='Migrate technicals to the wide table.')
    parser.add_argument('--db-path', default=None)
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--verify-sample', type=int, default=25)
    parser.add_argument('--finalize', action='store_true',
                        help='After verification passes: rename the DB aside and '
                             'rebuild it without the long tables.')
    args = parser.parse_args()
    setup_logging('migrate_technicals.log')

    db = DBManager(args.db_path)  # creates technical_data + meta columns
    db_path = db.db_path
    free_gib = shutil.disk_usage(db_path.parent).free / 2**30
    size_gib = db_path.stat().st_size / 2**30
    logger.info('DB %s: %.2f GiB, %.1f GiB free', db_path.name, size_gib, free_gib)
    if free_gib < 2 * size_gib:
        raise SystemExit('Need free disk >= 2x the database size to proceed safely.')

    conn = sqlite3.connect(db_path)
    conn.isolation_level = None  # manual BEGIN/COMMIT
    try:
        existing = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not all(t in existing for t in LONG_TABLES):
            raise SystemExit('Long tables not found -- already finalized?')

        combos = stored_combos(conn)
        db._ensure_technical_columns([column_name(i, p) for i, p in combos])
        written = migrate_batches(db, conn, combos, args.batch_size)
        logger.info('Migration pass wrote/updated %s row(s)', f'{written:,}')

        ok = verify_sampled(conn, combos, args.verify_sample) and verify_counts(conn, combos)
        if not ok:
            raise SystemExit('Parity verification FAILED -- nothing finalized.')
        logger.info('Parity verification passed.')
    finally:
        conn.close()

    if args.finalize:
        db.engine.dispose()
        finalize(db_path)
        post = DBManager(args.db_path)
        meta = post.get_ticker_meta()
        seeded = meta['yf_seeded'].fillna(False).astype(bool)
        post.upsert_ticker_meta(meta.loc[seeded, []], technicals_seeded=True)
        logger.info('technicals_seeded backfilled for %d ticker(s)', int(seeded.sum()))
    else:
        logger.info('Additive pass complete. Rerun with --finalize to swap.')


if __name__ == '__main__':
    main()
