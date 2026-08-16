"""Persistence and query layer for the stock_data SQLite database.

All DB reads/writes go through DBManager. Acquisition (yf.py) and math
(findata.preprocess.calculators) never touch the database; this module wires
them together for ingestion and exposes the query API used by downstream
projects.

Conflict policy: price/ema/indicator rows are INSERT OR IGNORE (append-only);
ticker_meta is upserted.
"""

import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm.auto import tqdm


from findata.configs import DATA_DIR, DB_NAME

import findata.preprocess.calculators.technical as calc
from findata.preprocess.calculators.base import (CALCULATOR_REGISTRY, column_name,
                                                 known_items, output_owner,
                                                 parse_column)
from findata.utils.timing import timed
from .tables import (Base, BalanceData, CashflowData, FilingsData, Form4Data,
                     FundamentalData, IncomeData, PriceData, TechnicalData,
                     TickerMeta)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(DATA_DIR) / DB_NAME

PRICE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio']

# trailing bars fetched ahead of new dates so rolling windows are exact on update
WARMUP_BARS = 150

# wide technical_data key columns (everything else is an item_period value)
TECHNICAL_KEY = ('ticker', 'date', 'interval')

# runtime-added columns must be plain lowercase identifiers
_COLUMN_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')


class MissingItemsError(LookupError):
    def __init__(self, items):
        self.items = list(items)
        super().__init__(
            f"Items not in database: {self.items}. "
            f"Set if_missing='add' to compute and store them.")


class DBManager:
    """Single entry point for writing to and querying the stock database.

    ``if_missing`` controls what get_items does when a requested
    (item, period) has no stored values: 'raise' raises MissingItemsError,
    'add' (the default) computes it on the fly from stored price data,
    returns it, and persists it when ``save_policy`` allows the calculator's
    group (``findata.configs.SAVE_POLICY`` unless one is passed).
    """

    def __init__(self, db_path=None, db_name=None, if_missing: str = 'add',
                 save_policy=None):
        if if_missing not in ('raise', 'add'):
            raise ValueError("if_missing must be 'raise' or 'add'")
        self.if_missing = if_missing
        if save_policy is None:
            from findata.configs import SAVE_POLICY
            save_policy = SAVE_POLICY
        self.save_policy = save_policy
        db_path = Path(db_path) if db_path is not None else Path(DATA_DIR) / db_name if db_name else DEFAULT_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        event.listen(self.engine, 'connect', self._set_sqlite_pragmas)
        Base.metadata.create_all(self.engine)
        self._ensure_schema()
        self._tech_cols = None  # cached technical_data value columns

    @staticmethod
    def _set_sqlite_pragmas(dbapi_conn, _record):
        """Per-connection PRAGMAs applied on every new pooled connection.

        WAL + synchronous=NORMAL is the big bulk-write win: commits no longer
        fsync the database file on every transaction (the WAL is synced at
        checkpoints instead), so the many small INSERT OR IGNORE transactions
        during ingestion stop paying a disk sync each.
        """
        cur = dbapi_conn.cursor()
        cur.execute('PRAGMA journal_mode=WAL')
        cur.execute('PRAGMA synchronous=NORMAL')
        cur.execute('PRAGMA cache_size=-262144')    # 256 MiB page cache
        cur.execute('PRAGMA mmap_size=4294967296')  # read pages via mmap, 4 GiB cap
        cur.execute('PRAGMA temp_store=MEMORY')
        cur.close()

    def _ensure_schema(self):
        """Add columns introduced after a table already exists on disk
        (create_all only creates missing tables, it never alters them)."""
        added = {'ticker_meta': {'last_form4_date': 'DATE',
                                 'technicals_seeded': 'BOOLEAN',
                                 'last_facts_date': 'DATE',
                                 'last_fundamentals_date': 'DATE',
                                 'inactive_since': 'DATE',
                                 'cik': 'INTEGER'}}
        new_columns = set()
        with self.engine.begin() as conn:
            for table, columns in added.items():
                existing = {row[1] for row in
                            conn.exec_driver_sql(f'PRAGMA table_info({table})')}
                for column, sql_type in columns.items():
                    if column not in existing:
                        logger.info('Migrating %s: adding column %s', table, column)
                        conn.exec_driver_sql(
                            f'ALTER TABLE {table} ADD COLUMN {column} {sql_type}')
                        new_columns.add(column)
        self._backfill_last_facts_date()
        self._backfill_ciks()
        self._ensure_cashflow_key()

    def _backfill_ciks(self):
        """Fill in CIKs for tickers stored before the column existed.

        Uses edgartools' bundled ticker->CIK mapping (offline, no request).
        EDGAR runs set it directly from the fetched company, so this only has
        to cover history.
        """
        with self.engine.begin() as conn:
            pending = conn.exec_driver_sql(
                'SELECT ticker FROM ticker_meta WHERE cik IS NULL').fetchall()
        names = [row[0] for row in pending]
        if not names:
            return
        try:
            from findata.database.edgar_ import ticker_ciks
            lookup = ticker_ciks(names)
        except Exception as e:
            logger.warning('Could not backfill CIKs (%s)', e)
            return
        rows = [(int(cik), ticker) for ticker, cik in lookup.items()]
        if not rows:
            return
        with self.engine.begin() as conn:
            conn.exec_driver_sql('UPDATE ticker_meta SET cik = ? WHERE ticker = ?', rows)
        logger.info('Backfilled CIK for %d of %d ticker(s)', len(rows), len(names))

    def _backfill_last_facts_date(self):
        """Seed last_facts_date for tickers whose statements predate the column.

        The newest stored filing date is a guaranteed lower bound on when the
        facts were pulled -- a filing cannot have been stored before it was
        published -- so this never claims data is fresher than it is. Tickers
        whose newest filing is genuinely old simply refresh once and then
        carry an accurate stamp. Correlated subqueries are kept one level
        deep: SQLite cannot resolve the outer row through a nested FROM.
        """
        with self.engine.begin() as conn:
            pending = conn.exec_driver_sql(
                'SELECT COUNT(*) FROM ticker_meta '
                'WHERE last_facts_date IS NULL AND edgar_seeded = 1').scalar()
            if not pending:
                return
            n = conn.exec_driver_sql("""
                UPDATE ticker_meta SET last_facts_date = COALESCE(
                    (SELECT MAX(filed_date) FROM income_data
                        WHERE ticker = ticker_meta.ticker),
                    (SELECT MAX(filed_date) FROM balance_data
                        WHERE ticker = ticker_meta.ticker),
                    (SELECT MAX(filed_date) FROM cashflow_data
                        WHERE ticker = ticker_meta.ticker))
                WHERE last_facts_date IS NULL AND edgar_seeded = 1
            """).rowcount
        logger.info('Backfilled last_facts_date for %s ticker(s) from stored filings', n)

    def _ensure_cashflow_key(self):
        """Rebuild cashflow_data when it predates 'derived' joining the PK.

        Without derived in the key, a de-cumulated quarter and the cumulative
        window it came from collide. Stored rows are dropped rather than
        migrated: they were written by the old parser, and re-running the
        EDGAR pipeline regenerates them correctly.
        """
        with self.engine.begin() as conn:
            info = list(conn.exec_driver_sql('PRAGMA table_info(cashflow_data)'))
            if not info or any(row[1] == 'derived' and row[5] for row in info):
                return  # missing table (create_all handles it) or already keyed
            n = conn.exec_driver_sql('SELECT COUNT(*) FROM cashflow_data').scalar()
            logger.warning('cashflow_data: rebuilding with derived in the primary key '
                           '(%s stale row(s) dropped -- re-run init_edgar to repopulate)', f'{n:,}')
            conn.exec_driver_sql('DROP TABLE cashflow_data')
        CashflowData.__table__.create(self.engine)

    def _technical_columns(self) -> list:
        """Value columns of technical_data on disk (PRAGMA-discovered so
        runtime-added windows are included), cached until an ALTER."""
        if self._tech_cols is None:
            with self.engine.connect() as conn:
                info = conn.exec_driver_sql('PRAGMA table_info(technical_data)')
                self._tech_cols = [row[1] for row in info if row[1] not in TECHNICAL_KEY]
        return self._tech_cols

    def _ensure_technical_columns(self, columns) -> list:
        """ALTER technical_data to add any missing value columns."""
        new = [c for c in columns if c not in self._technical_columns()]
        for col in new:
            if not _COLUMN_NAME_RE.match(col):
                raise ValueError(f'Invalid technical column name {col!r}')
            logger.info('technical_data: adding column %s', col)
            with self.engine.begin() as conn:
                conn.exec_driver_sql(f'ALTER TABLE technical_data ADD COLUMN "{col}" REAL')
        if new:
            self._tech_cols = None
        return new

    @staticmethod
    def _resolve_db_path(name) -> Path:
        """A bare name lands in the default data dir; a path is used as given.
        A missing .db suffix is appended."""
        path = Path(name)
        if path.suffix != '.db':
            path = path.with_name(path.name + '.db')
        if path.parent == Path('..'):
            path = DEFAULT_DB_PATH.parent / path.name
        return path

    def copy_subset(self, tickers, new_name, source_name=None) -> Path:
        """Write a new SQLite db holding only the given tickers' rows.

        Every table is keyed by ticker, so each is filtered with the same
        ticker list -- the result is a small, self-contained extract that is
        cheap to transfer. ``source_name`` defaults to this manager's db; a
        bare name resolves under the default data dir. Returns the new path.
        Rows stream entirely inside sqlite (ATTACH + INSERT...SELECT), so
        nothing is loaded into Python.
        """
        tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        if not tickers:
            raise ValueError('tickers must be a non-empty list')
        source = self._resolve_db_path(source_name) if source_name else self.db_path
        dest = self._resolve_db_path(new_name)
        if dest.resolve() == Path(source).resolve():
            raise ValueError('destination db must differ from source')
        if not Path(source).exists():
            raise FileNotFoundError(f'source db not found: {source}')

        # destination schema is copied verbatim from the source sqlite_master
        # (not the ORM) so runtime-added technical columns survive extraction
        dest.parent.mkdir(parents=True, exist_ok=True)
        placeholders = ','.join('?' * len(tickers))
        src = sqlite3.connect(str(source))
        try:
            schemas = src.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'").fetchall()
            dst = sqlite3.connect(str(dest))
            try:
                existing = {row[0] for row in dst.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                for name, create_sql in schemas:
                    if name not in existing:
                        dst.execute(create_sql)
                dst.commit()
            finally:
                dst.close()

            src.execute('ATTACH DATABASE ? AS dest', (str(dest),))
            counts = {}
            for name, _ in schemas:
                cur = src.execute(
                    f'INSERT INTO dest."{name}" SELECT * FROM main."{name}" '
                    f'WHERE ticker IN ({placeholders})', tickers)
                counts[name] = cur.rowcount
            src.commit()
        finally:
            src.close()

        logger.info('Copied subset %s -> %s for %d ticker(s): %s rows',
                    Path(source).name, dest.name, len(tickers), sum(counts.values()))
        logger.debug('Per-table rows copied: %s', counts)
        return dest

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #

    def add_ticker_data(self, info: pd.DataFrame, prices: pd.DataFrame,
                        progress: bool = True, daily_only=True,
                        compute_technicals: bool = True) -> dict:
        """Cold-start ingestion from the yf.py output.

        ``info`` is indexed by ticker; ``prices`` by (interval, ticker, date).
        Processes one ticker at a time -- insert bars, compute the full
        technical history (skipped for a prices-only seed with
        ``compute_technicals=False``), write, mark meta -- so the progress
        bar steps per ticker and an interrupted run resumes where it stopped.
        Idempotent; counts are rows actually written.
        """
        prices = self._standardize_prices(prices)
        groups = prices.groupby(level='ticker', sort=False)
        logger.info('Cold-start: seeding %d ticker(s), compute_technicals=%s',
                    len(groups), compute_technicals)
        counts = {'price': 0, 'technical': 0}

        for ticker, sub in tqdm(groups, desc='Seeding', unit='ticker', disable=not progress):
            sub = sub.loc[['daily']] if daily_only else sub
            with self._raw_txn() as conn:  # price + technicals in one fsync
                inserted = {'price': self._insert_prices(sub, conn=conn), 'technical': 0}
                if compute_technicals:
                    for (interval, tkr), bars in sub.groupby(level=['interval', 'ticker'],
                                                             sort=False):
                        calc_df = calc.calculate_all(bars.droplevel(['interval', 'ticker']))
                        wide = self._calc_to_wide(calc_df, tkr, interval)
                        inserted['technical'] += self._upsert_technicals(wide, conn=conn)
            if ticker in info.index:
                self._upsert_meta(info.loc[[ticker]], yf_seeded=True,
                                  technicals_seeded=compute_technicals)
            else:
                logger.warning('%s: no metadata returned, not marked yf_seeded', ticker)
            for key in counts:
                counts[key] += inserted[key]
            logger.debug('%s seeded: %s', ticker, inserted)

        logger.info('Cold-start done, rows written: %s', counts)
        return counts

    def update_ticker_data(self, prices: pd.DataFrame, info: pd.DataFrame = None,
                           progress: bool = True, daily_only=True) -> dict:
        """Incremental ingestion of new bars for already-seeded tickers.

        Inserts the new price rows, then continues each ticker's stored
        technical columns from their last values (seeded EWMs/cumsums;
        rolling windows recompute over WARMUP_BARS trailing bars). The
        columns to maintain are whatever holds values in the ticker's recent
        technical rows, so custom windows added on the fly keep updating.
        A ticker with no technical rows is prices-only and stays that way --
        promote it with scripts/db_handling/seed_technicals.py.
        """
        prices = self._standardize_prices(prices)
        names = prices.index.get_level_values('ticker').unique().tolist()
        # A re-request usually returns bars we already hold, which INSERT OR
        # IGNORE drops. Comparing stored row counts across the insert says
        # exactly which tickers gained anything (including a backfilled gap,
        # which a max-date check would miss), so the rest can skip the
        # recompute entirely instead of rewriting identical values.
        before = self._price_row_counts(names)
        counts = {'price': self._insert_prices(prices), 'technical': 0}
        after = self._price_row_counts(names) if counts['price'] else before
        changed = {t for t in names if after.get(t, 0) > before.get(t, 0)}

        if daily_only:
            prices = prices.loc[['daily']]
        groups = prices.groupby(level=['interval', 'ticker'], sort=False)
        logger.info('Incremental update: %d new price rows across %d ticker/interval '
                    'group(s); %d ticker(s) gained bars', counts['price'], len(groups),
                    len(changed))

        n_full = n_prices_only = n_unchanged = 0
        for (interval, ticker), new_bars in tqdm(groups, desc='Updating', unit='group',
                                                 disable=not progress):
            if ticker not in changed:
                n_unchanged += 1
                continue
            seeds, windows = self._technical_state(ticker, interval)
            if not windows:
                logger.debug('%s/%s has no stored technicals, prices only', ticker, interval)
                n_prices_only += 1
                continue
            window = self._price_window(ticker, interval, len(new_bars) + WARMUP_BARS)
            calc_df = calc.calculate_all(window, seeds=seeds, **windows)
            wide = self._calc_to_wide(calc_df, ticker, interval)
            with self._raw_txn() as conn:
                written = self._upsert_technicals(wide, conn=conn)
            counts['technical'] += written
            n_full += 1
            logger.debug('%s/%s updated: %d technical rows', ticker, interval, written)

        if n_unchanged:
            logger.info('%d ticker/interval group(s) had no new bars, technicals untouched',
                        n_unchanged)
        if info is None:
            info = self._last_price_dates(prices)
        if not info.empty:
            self._upsert_meta(info)
        logger.info('Incremental update done: %s (%d full, %d prices-only)',
                    counts, n_full, n_prices_only)
        return counts

    @staticmethod
    def _standardize_prices(prices: pd.DataFrame) -> pd.DataFrame:
        """Map raw yahooquery columns onto the price_data schema."""
        df = prices.rename(columns={'dividends': 'dividend', 'splits': 'split_ratio'}).copy()
        if 'dividend' not in df:
            df['dividend'] = 0.0
        if 'split_ratio' not in df:
            df['split_ratio'] = 1.0
        df['dividend'] = df['dividend'].fillna(0.0)
        df['split_ratio'] = df['split_ratio'].replace(0.0, np.nan).fillna(1.0)
        if any(df['close'].isna()):
            print(f'NA found in close, total: {sum(df['close'].isna())}')
        return df[df['close'].notna()][PRICE_COLUMNS]

    @staticmethod
    def _last_price_dates(prices: pd.DataFrame) -> pd.DataFrame:
        idx = prices.index.to_frame(index=False)
        daily = idx[idx['interval'] == 'daily']
        return daily.groupby('ticker')['date'].max().rename('last_price_date').to_frame()

    def _insert_prices(self, prices: pd.DataFrame, conn=None) -> int:
        return self._insert_ignore(PriceData, prices.reset_index(), conn=conn)

    def _price_row_counts(self, tickers) -> dict:
        """Stored price rows per ticker, for detecting what an insert added."""
        if not tickers:
            return {}
        stmt = (select(PriceData.ticker, func.count().label('n'))
                .where(PriceData.ticker.in_(list(tickers)))
                .group_by(PriceData.ticker))
        with self.engine.connect() as conn:
            return {ticker: n for ticker, n in conn.execute(stmt)}

    @staticmethod
    def _calc_to_wide(calc_df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        """calculate_all output -> technical_data row frame (key + value cols)."""
        wide = calc_df.copy()
        wide.columns = [column_name(item, period) for item, period in calc_df.columns]
        wide = wide.reset_index(names='date')
        wide.insert(0, 'ticker', ticker)
        wide.insert(2, 'interval', interval)
        return wide

    def _upsert_technicals(self, wide: pd.DataFrame, conn=None) -> int:
        """Wide upsert into technical_data.

        Only the frame's columns are touched; stored values are kept where
        the incoming value is NULL (a recompute window's leading warmup rows
        must not blank out good history).
        """
        if wide.empty:
            return 0
        value_cols = [c for c in wide.columns if c not in TECHNICAL_KEY]
        self._ensure_technical_columns(value_cols)
        cols = list(TECHNICAL_KEY) + value_cols
        quoted = ', '.join(f'"{c}"' for c in cols)
        assignments = ', '.join(f'"{c}"=COALESCE(excluded."{c}", "{c}")' for c in value_cols)
        sql = (f'INSERT INTO technical_data ({quoted}) '
               f'VALUES ({", ".join("?" * len(cols))}) '
               f'ON CONFLICT(ticker, date, interval) DO UPDATE SET {assignments}')
        own = conn is None
        connection = self.engine.raw_connection() if own else conn
        try:
            cursor = connection.cursor()
            cursor.executemany(sql, self._tuple_rows(wide[cols]))
            if own:
                connection.commit()
            return cursor.rowcount
        finally:
            if own:
                connection.close()

    def upsert_ticker_meta(self, info: pd.DataFrame, **flags):
        """Public upsert into ticker_meta (info indexed by ticker; flags are
        fixed column values applied to every row, e.g. edgar_seeded=True)."""
        self._upsert_meta(info, **flags)

    def _upsert_meta(self, info: pd.DataFrame, **flags):
        allowed = ['ticker', 'cik', 'name', 'sector', 'industry', 'last_price_date',
                   'first_price_date', 'last_filings_date', 'last_form4_date',
                   'last_facts_date', 'last_fundamentals_date', 'yf_seeded',
                   'edgar_seeded']
        df = info.reset_index()
        records = self._records(df[[c for c in df.columns if c in allowed]])
        if not records:
            return
        now = datetime.now()
        for record in records:
            record.update(flags, updated_at=now)
        stmt = sqlite_insert(TickerMeta.__table__)
        update_cols = {k: stmt.excluded[k] for k in records[0] if k != 'ticker'}
        stmt = stmt.on_conflict_do_update(index_elements=['ticker'], set_=update_cols)
        with self.engine.begin() as conn:
            conn.execute(stmt, records)

    @contextmanager
    def _raw_txn(self):
        """A raw sqlite connection wrapping one transaction (one commit/fsync).

        Pass the yielded connection to several ``_insert_ignore`` calls so
        they share a single transaction instead of committing one-by-one --
        e.g. a ticker's price + ema + indicator rows land in one fsync.
        """
        conn = self.engine.raw_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _insert_ignore(self, model, df: pd.DataFrame, conn=None) -> int:
        """Bulk INSERT OR IGNORE through the raw sqlite3 driver.

        The data is trusted (already validated upstream), so rows go in as
        plain tuples via executemany -- far faster than per-row dicts through
        the ORM layer. Returns the number of rows actually inserted
        (conflicting rows are excluded by sqlite's change counter).

        With ``conn`` the caller owns the transaction (no commit/close here),
        letting several inserts share one ``_raw_txn``; without it this opens
        its own connection and commits the batch.
        """
        if df.empty:
            return 0
        sql = (f"INSERT OR IGNORE INTO {model.__tablename__} "
               f"({', '.join(df.columns)}) VALUES ({', '.join('?' * len(df.columns))})")
        own = conn is None
        connection = self.engine.raw_connection() if own else conn
        try:
            cursor = connection.cursor()
            cursor.executemany(sql, self._tuple_rows(df))
            if own:
                connection.commit()
            return cursor.rowcount
        finally:
            if own:
                connection.close()

    @staticmethod
    def _tuple_rows(df: pd.DataFrame) -> list[tuple]:
        """Column-wise conversion to sqlite-bindable tuples.

        Dates become ISO strings (matching SQLAlchemy's Date storage format);
        numeric columns pass through -- sqlite stores NaN as NULL natively.
        """
        columns = []
        for name in df.columns:
            series = df[name]
            if series.dtype.kind in 'iufb':
                columns.append(series.to_numpy().tolist())
                continue
            first = next((v for v in series if not pd.isna(v)), None)
            if isinstance(first, (date, datetime)):
                columns.append([None if pd.isna(v) else v.isoformat() for v in series])
            else:
                columns.append([None if pd.isna(v) else v for v in series])
        return list(zip(*columns))

    @staticmethod
    def _records(df: pd.DataFrame) -> list[dict]:
        """to_dict('records') with numpy scalars and NaN made sqlite-safe."""
        def pyval(v):
            if pd.isna(v):
                return None
            return v.item() if isinstance(v, np.generic) else v

        cols = df.columns
        return [{c: pyval(v) for c, v in zip(cols, row)} for row in df.itertuples(index=False)]

    # ------------------------------------------------------------------ #
    # incremental-update helpers
    # ------------------------------------------------------------------ #

    def _price_window(self, ticker: str, interval: str, n_bars: int) -> pd.DataFrame:
        """Last n_bars price rows, oldest first, indexed by date."""
        stmt = (select(PriceData.date, *[getattr(PriceData, c) for c in PRICE_COLUMNS])
                .where(PriceData.ticker == ticker, PriceData.interval == interval)
                .order_by(PriceData.date.desc()).limit(n_bars))
        df = pd.read_sql(stmt, self.engine)
        return df.iloc[::-1].set_index('date')

    def _technical_state(self, ticker: str, interval: str) -> tuple:
        """Seeds + maintained windows from the trailing technical rows.

        Returns ``(seeds, windows)``: seeds maps seedable (item, period) keys
        to one-row Series for calculate_all; windows maps calculator names to
        sorted period lists (calculate_all overrides). Maintained columns are
        those holding any value in the trailing WARMUP_BARS rows. One indexed
        range scan replaces the old per-table GROUP-BY probes. A ticker with
        no technical rows returns ({}, {}).
        """
        cols = self._technical_columns()
        if not cols:
            return {}, {}
        col_list = ', '.join(f'"{c}"' for c in cols)
        sql = (f'SELECT date, {col_list} FROM technical_data '
               f'WHERE ticker = ? AND interval = ? ORDER BY date DESC LIMIT {WARMUP_BARS}')
        connection = self.engine.raw_connection()
        try:
            rows = connection.cursor().execute(sql, (ticker, interval)).fetchall()
        finally:
            connection.close()
        if not rows:
            return {}, {}
        df = pd.DataFrame(rows[::-1], columns=['date'] + cols)
        df['date'] = [date.fromisoformat(d) for d in df['date']]
        df = df.set_index('date')

        seeds, windows = {}, {}
        for col in cols:
            series = df[col].dropna()
            if series.empty:
                continue
            try:
                item, period = parse_column(col)
            except KeyError:  # legacy column no calculator owns -- not maintained
                logger.debug('technical_data column %r has no calculator, skipped', col)
                continue
            owner = output_owner(item)
            if period:  # period-0 items (obv, ad) are always computed
                windows.setdefault(owner, set()).add(period)
            if item in CALCULATOR_REGISTRY[owner].seeded:
                seeds[(item, period)] = series.iloc[[-1]]
        return seeds, {name: sorted(periods) for name, periods in windows.items()}

    # ------------------------------------------------------------------ #
    # query API
    # ------------------------------------------------------------------ #

    def get_all_tickers(self):
        stmt = select(TickerMeta.ticker)
        return self.engine.connect().execute(stmt).scalars().all()

    def get_ticker_meta(self, tickers: list = None) -> pd.DataFrame:
        stmt = select(TickerMeta)
        if tickers is not None:
            stmt = stmt.where(TickerMeta.ticker.in_(tickers))
        return pd.read_sql(stmt, self.engine).set_index('ticker')

    def ticker_summary(self) -> pd.DataFrame:
        """Quick look at the ticker info table plus stored price coverage.

        One row per ticker: metadata, seeded flags, bar counts per interval,
        and the stored daily date range.
        """
        meta = self.get_ticker_meta()
        stmt = (select(PriceData.ticker, PriceData.interval, func.count().label('bars'),
                       func.min(PriceData.date).label('first_date'),
                       func.max(PriceData.date).label('last_date'))
                .group_by(PriceData.ticker, PriceData.interval))
        coverage = pd.read_sql(stmt, self.engine)
        if coverage.empty:
            return meta

        bars = (coverage.pivot(index='ticker', columns='interval', values='bars')
                .rename(columns=lambda interval: f'{interval}_bars'))
        daily_range = (coverage[coverage['interval'] == 'daily']
                       .set_index('ticker')[['first_date', 'last_date']])
        cols = ['name', 'sector', 'industry', 'last_price_date', 'yf_seeded',
                'technicals_seeded', 'edgar_seeded']
        return meta[cols].join(bars).join(daily_range)

    def _stack_tickers(self, method, tickers, **kwargs) -> pd.DataFrame:
        """Run a single-ticker query for each ticker and stack the results.

        Indexed frames (date / fiscal period) gain a leading 'ticker' index
        level -- e.g. (ticker, date); long-format frames (default RangeIndex)
        gain a leading 'ticker' column instead. Tickers with no rows are
        dropped; an all-empty request returns the single-ticker empty frame.
        """
        frames = [(t, method(t, **kwargs)) for t in tickers]
        frames = [(t, df) for t, df in frames if not df.empty]
        if not frames:
            return method(tickers[0], **kwargs) if tickers else pd.DataFrame()
        if isinstance(frames[0][1].index, pd.RangeIndex):
            out = pd.concat([df.assign(ticker=t) for t, df in frames], ignore_index=True)
            return out[['ticker'] + [c for c in out.columns if c != 'ticker']]
        return pd.concat({t: df for t, df in frames}, names=['ticker'])

    def get_price_data(self, ticker, interval: str = 'daily',
                       start=None, end=None) -> pd.DataFrame:
        if not isinstance(ticker, str):
            tickers = list(ticker)
            stmt = (select(PriceData.ticker, PriceData.date,
                           *[getattr(PriceData, c) for c in PRICE_COLUMNS])
                    .where(PriceData.ticker.in_(tickers), PriceData.interval == interval)
                    .order_by(PriceData.ticker, PriceData.date))
            stmt = self._date_filter(stmt, PriceData.date, start, end)
            df = pd.read_sql(stmt, self.engine)
            if df.empty:
                return self.get_price_data(tickers[0], interval, start, end) \
                    if tickers else pd.DataFrame()
            return df.set_index(['ticker', 'date'])
        stmt = (select(PriceData.date, *[getattr(PriceData, c) for c in PRICE_COLUMNS])
                .where(PriceData.ticker == ticker, PriceData.interval == interval)
                .order_by(PriceData.date))
        stmt = self._date_filter(stmt, PriceData.date, start, end)
        return pd.read_sql(stmt, self.engine).set_index('date')

    def get_all_data(self, ticker, interval: str = 'daily',
                     start=None, end=None, wide: bool = True,
                     include_fundamentals: bool = False) -> pd.DataFrame:
        """Everything stored for ticker(s)/interval: prices + technicals.

        One price_data LEFT JOIN technical_data scan -- no pivots. wide=True
        returns a date-indexed frame (calculated columns named
        '<item>_<period>', period-0 items keep their bare name). wide=False
        returns a long frame [date, item, period, value] with the price
        columns included as period-0 items. ``include_fundamentals`` merges
        the daily-aligned derived fundamentals as ``fund_<item>`` columns.

        ``ticker`` may be a list; results gain a leading 'ticker' index
        level (wide -> (ticker, date), sorted) or 'ticker' column (long).
        """
        single = isinstance(ticker, str)
        tickers = [ticker] if single else list(ticker)
        if not tickers:
            return pd.DataFrame()
        tech_cols = sorted(self._technical_columns())
        with timed(logger, f'get_all_data ({len(tickers)} tickers)'):
            df = self._read_joined(tickers, interval, PRICE_COLUMNS, tech_cols, start, end)
            if include_fundamentals:
                fund = self.get_fundamentals(tickers, start=start, end=end, daily=True)
                if not fund.empty:
                    fund.columns = [f'fund_{c}' for c in fund.columns]
                    df = df.join(fund)
        if not wide:
            df = self._wide_to_long(df)
            return df.drop(columns='ticker') if single else df
        return df.droplevel('ticker') if single else df

    def get_items(self, ticker, items: dict, interval: str = 'daily',
                  start=None, end=None, wide: bool = True,
                  persist: bool = None) -> pd.DataFrame:
        """Fetch specific items: {name: period(s) or None}.

        Names can be price columns ('close'), output items ('rsi',
        'bb_upper', 'atr', 'obv', 'ema_close', ...) or calculator names
        ('bollinger', 'stochastic', 'adx', 'aroon' -- expanded to all their
        output columns). Periods may be an int, a list of ints, or None for
        the calculator's defaults (ignored for price columns and obv/ad).

        Missing (item, period) values follow self.if_missing: 'raise' raises
        MissingItemsError; 'add' computes them on the fly over the full
        stored price history and merges them into the result. Computed
        values persist when ``persist`` is True, or when it is None and the
        save policy allows the calculator's group ('technical.core' for
        default windows, 'technical.custom' otherwise). Tickers with no
        stored prices are skipped with a warning.

        ``ticker`` may be a list; results gain a leading 'ticker' index
        level (wide -> (ticker, date), sorted) or 'ticker' column (long).
        """
        single = isinstance(ticker, str)
        tickers = [ticker] if single else list(ticker)
        if not tickers:
            return pd.DataFrame()
        price_cols, pairs = self._resolve_items(items)
        tech_cols = [column_name(item, period) for item, period in pairs]

        stored = self._technical_columns()
        present = sorted(c for c in tech_cols if c in stored)
        with timed(logger, f'get_items ({len(tickers)} tickers, {len(tech_cols)} cols)'):
            df = self._read_joined(tickers, interval, price_cols, present, start, end)
            missing = self._missing_technicals(df, tickers, tech_cols)
            if missing:
                if self.if_missing == 'raise':
                    raise MissingItemsError(sorted({parse_column(col) for _, col in missing}))
                df = self._compute_missing(df, missing, interval, persist)

        ordered = ([c for c in price_cols if c in df.columns]
                   + sorted(c for c in df.columns if c not in price_cols))
        df = df[ordered]
        if not wide:
            df = self._wide_to_long(df)
            return df.drop(columns='ticker') if single else df
        return df.droplevel('ticker') if single else df

    def _missing_technicals(self, df: pd.DataFrame, tickers, tech_cols) -> list:
        """(ticker, column) pairs with no stored values in the read result.

        Presence keys off the index, not DataFrame.empty -- a read with rows
        but zero requested value columns is still a populated read.
        """
        present_tickers = (set(df.index.get_level_values('ticker'))
                           if len(df.index) else set())
        absent = [t for t in tickers if t not in present_tickers]
        if absent:
            logger.warning('get_items: no stored prices for %s, skipped', absent)
        counts = (df.groupby(level='ticker').count()
                  if len(df.index) and len(df.columns) else None)
        missing = []
        for col in tech_cols:
            if col not in df.columns:
                missing += [(t, col) for t in tickers if t in present_tickers]
            elif counts is not None:
                missing += [(t, col) for t in counts.index[counts[col] == 0]]
        return missing

    def _compute_missing(self, df: pd.DataFrame, missing: list, interval,
                         persist) -> pd.DataFrame:
        """On-the-fly computation of absent (ticker, column) values.

        Computes over the full stored price history (exact cold start),
        persists per calculator group policy (all sibling outputs of a
        calculator are stored together), and merges the requested columns
        into the read result.
        """
        by_ticker = {}
        for t, col in missing:
            by_ticker.setdefault(t, []).append(col)
        logger.info('Computing %d missing technical column(s) for %d ticker(s)',
                    len({col for _, col in missing}), len(by_ticker))
        prices = self.get_price_data(list(by_ticker), interval)
        if prices.empty:
            return df

        computed_frames = []
        for t, cols in by_ticker.items():
            try:
                ticker_prices = prices.xs(t, level='ticker')
            except KeyError:
                continue
            plans = {}
            for col in cols:
                item, period = parse_column(col)
                plans.setdefault((output_owner(item), period), None)
            out, to_persist = {}, {}
            for owner, period in plans:
                calculator = CALCULATOR_REGISTRY[owner]
                res = (calculator.calculate(ticker_prices, period=period)
                       if period else calculator.calculate(ticker_prices))
                res = res.to_frame() if isinstance(res, pd.Series) else res
                res.columns = [column_name(item, period) for item in res.columns]
                out.update({c: res[c] for c in res.columns})
                save = (persist if persist is not None
                        else self.save_policy.allows(self._policy_group(calculator, period)))
                if save:
                    to_persist.update({c: res[c] for c in res.columns})
            if to_persist:
                rows = pd.DataFrame(to_persist)
                rows.index.name = 'date'
                rows = rows.reset_index()
                rows.insert(0, 'ticker', t)
                rows.insert(2, 'interval', interval)
                self._upsert_technicals(rows)
            frame = pd.DataFrame(out)
            frame.index.name = 'date'
            frame['ticker'] = t
            computed_frames.append(frame.reset_index().set_index(['ticker', 'date']))

        if not computed_frames:
            return df
        computed = pd.concat(computed_frames)
        requested = {col for _, col in missing}
        computed = computed[[c for c in computed.columns if c in requested]]
        for col in computed.columns:
            if col not in df.columns:
                df[col] = np.nan
        df.update(computed)
        return df

    @staticmethod
    def _policy_group(calculator, period: int) -> str:
        defaults = calculator.default_params.get('periods', [])
        return 'technical.core' if period in defaults else 'technical.custom'

    # ------------------------------------------------------------------ #
    # get_items internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_items(items: dict):
        """Normalize the request dict into price columns and (item, period) pairs.

        Names may be price columns, registered output items, or calculator
        names (expanded to every output item). None takes the owning
        calculator's default periods; period-0 calculators (obv, ad) always
        resolve to period 0.
        """
        price_cols, pairs = [], []
        registered_items = set(known_items())
        for name, periods in items.items():
            if name in PRICE_COLUMNS:
                price_cols.append(name)
                continue
            # calculator name wins over output item where both exist ('adx'):
            # requesting the calculator expands to its whole output family
            if name in CALCULATOR_REGISTRY:
                owner = CALCULATOR_REGISTRY[name]
                out_items = [spec.item for spec in owner.outputs]
            elif name in registered_items:
                owner = CALCULATOR_REGISTRY[output_owner(name)]
                out_items = [name]
            else:
                raise KeyError(f"Unknown item '{name}'")
            default_periods = owner.default_params.get('periods', [0])
            if periods is None or default_periods == [0]:
                periods = default_periods
            for p in calc._as_list(periods):
                pairs += [(item, int(p)) for item in out_items]
        return price_cols, sorted(set(pairs))

    # ------------------------------------------------------------------ #
    # reads + shaping
    # ------------------------------------------------------------------ #

    def _read_joined(self, tickers, interval, price_cols, tech_cols,
                     start=None, end=None) -> pd.DataFrame:
        """One price LEFT JOIN technicals scan -> (ticker, date)-indexed wide
        frame. Raw sqlite cursor (dynamic column list) with dates parsed once
        per row, not once per stored item."""
        select_cols = (['p.ticker', 'p.date']
                       + [f'p.{c}' for c in price_cols]
                       + [f't."{c}"' for c in tech_cols])
        sql = (f'SELECT {", ".join(select_cols)} FROM price_data p '
               f'LEFT JOIN technical_data t ON t.ticker = p.ticker '
               f'AND t.date = p.date AND t.interval = p.interval '
               f'WHERE p.ticker IN ({",".join("?" * len(tickers))}) AND p.interval = ?')
        params = [*tickers, interval]
        if start is not None:
            sql += ' AND p.date >= ?'
            params.append(pd.to_datetime(start).date().isoformat())
        if end is not None:
            sql += ' AND p.date <= ?'
            params.append(pd.to_datetime(end).date().isoformat())
        sql += ' ORDER BY p.ticker, p.date'

        connection = self.engine.raw_connection()
        try:
            rows = connection.cursor().execute(sql, params).fetchall()
        finally:
            connection.close()
        columns = ['ticker', 'date'] + list(price_cols) + list(tech_cols)
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            return df.set_index(['ticker', 'date'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.set_index(['ticker', 'date'])
        value_cols = list(price_cols) + list(tech_cols)
        return df.astype({c: 'float64' for c in value_cols if df[c].dtype == object})

    @staticmethod
    def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
        """(ticker, date) wide frame -> long [ticker, date, item, period, value].

        Price columns become period-0 items and keep their NaNs; calculated
        columns drop NaNs (matching the stored-rows-only long contract)."""
        if df.empty:
            return pd.DataFrame(columns=['ticker', 'date', 'item', 'period', 'value'])
        parsed = {c: (c, 0) if c in PRICE_COLUMNS else parse_column(c, strict=False)
                  for c in df.columns}
        long = (df.reset_index()
                .melt(id_vars=['ticker', 'date'], var_name='column', value_name='value'))
        keep_na = long['column'].isin(PRICE_COLUMNS)
        long = long[keep_na | long['value'].notna()]
        long['item'] = long['column'].map(lambda c: parsed[c][0])
        long['period'] = long['column'].map(lambda c: parsed[c][1])
        return (long[['ticker', 'date', 'item', 'period', 'value']]
                .sort_values(['ticker', 'item', 'period', 'date'], ignore_index=True))

    @staticmethod
    def _date_filter(stmt, column, start, end):
        if start is not None:
            stmt = stmt.where(column >= pd.to_datetime(start).date())
        if end is not None:
            stmt = stmt.where(column <= pd.to_datetime(end).date())
        return stmt


    # ------------------------------------------------------------------ #
    # EDGAR ingestion
    # ------------------------------------------------------------------ #

    def add_filings_meta(self, df: pd.DataFrame) -> int:
        """Insert filings_data rows (one per parsed 10-K/10-Q/8-K)."""
        n = self._insert_ignore(FilingsData, df)
        logger.debug('filings_data: %d row(s) inserted', n)
        return n

    def add_form4(self, df: pd.DataFrame) -> int:
        """Insert Form 4 transaction rows."""
        n = self._insert_ignore(Form4Data, df)
        logger.debug('form4_data: %d row(s) inserted', n)
        return n

    def add_financials(self, statements: dict) -> dict:
        """Insert parsed financial statements.

        ``statements`` maps 'income'/'balance'/'cashflow' to DataFrames with
        columns matching the respective table.
        """
        models = {'income': IncomeData, 'balance': BalanceData, 'cashflow': CashflowData}
        counts = {}
        for name, frame in statements.items():
            if frame is None or frame.empty:
                counts[name] = 0
                continue
            counts[name] = self._insert_ignore(models[name], frame)
        logger.debug('financials inserted: %s', counts)
        return counts

    def last_statement_dates(self, tickers=None) -> pd.Series:
        """Newest filed_date per ticker across the three statement tables.

        The high-water mark of what has actually been parsed, so callers can
        compare it against a filing list to see whether new financials exist.
        """
        single = isinstance(tickers, str)
        parts = []
        for model in (IncomeData, BalanceData, CashflowData):
            stmt = select(model.ticker, func.max(model.filed_date).label('filed'))
            if tickers is not None:
                names = [tickers] if single else list(tickers)
                stmt = stmt.where(model.ticker.in_(names))
            parts.append(pd.read_sql(stmt.group_by(model.ticker), self.engine))
        combined = pd.concat(parts, ignore_index=True).dropna(subset=['filed'])
        if combined.empty:
            return pd.Series(dtype='object', name='filed')
        combined['filed'] = pd.to_datetime(combined['filed']).dt.date
        return combined.groupby('ticker')['filed'].max()

    def set_inactive(self, tickers, active: bool = False) -> int:
        """Flag tickers as no longer trading (or revive them).

        A delisted or renamed symbol keeps its stored history but stops
        returning bars, so it would otherwise be re-requested on every run
        forever. ``active=True`` clears the flag, which is what happens
        automatically the moment a symbol produces bars again.
        """
        tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        if not tickers:
            return 0
        stamp = None if active else date.today().isoformat()
        with self.engine.begin() as conn:
            n = conn.exec_driver_sql(
                f'UPDATE ticker_meta SET inactive_since = ? '
                f'WHERE ticker IN ({",".join("?" * len(tickers))})',
                (stamp, *tickers)).rowcount
        logger.info('Marked %s ticker(s) %s', n, 'active' if active else 'inactive')
        return n

    def replace_financials(self, ticker: str, statements: dict) -> dict:
        """Delete a ticker's stored statements, then insert the parsed set.

        The facts request always returns the issuer's full history, so the
        parse is authoritative and a plain INSERT OR IGNORE would preserve
        rows that a parser fix has since corrected (or re-keyed onto a
        different fiscal year). Replacing makes re-runs self-healing.
        """
        models = {'income': IncomeData, 'balance': BalanceData, 'cashflow': CashflowData}
        with self.engine.begin() as conn:
            for model in models.values():
                conn.execute(model.__table__.delete().where(model.ticker == ticker))
        return self.add_financials(statements)

    # ------------------------------------------------------------------ #
    # derived fundamentals
    # ------------------------------------------------------------------ #

    def add_fundamentals(self, df: pd.DataFrame) -> int:
        """Upsert derived fundamental rows [ticker, date, item, value,
        fiscal_year, fiscal_quarter] (late statements revise in place)."""
        if df is None or df.empty:
            return 0
        cols = ['ticker', 'date', 'item', 'value', 'fiscal_year', 'fiscal_quarter']
        sql = (f'INSERT INTO fundamental_data ({", ".join(cols)}) '
               f'VALUES ({", ".join("?" * len(cols))}) '
               f'ON CONFLICT(ticker, date, item) DO UPDATE SET value=excluded.value, '
               f'fiscal_year=excluded.fiscal_year, fiscal_quarter=excluded.fiscal_quarter')
        connection = self.engine.raw_connection()
        try:
            cursor = connection.cursor()
            cursor.executemany(sql, self._tuple_rows(df[cols]))
            connection.commit()
            n = cursor.rowcount
        finally:
            connection.close()
        logger.info('fundamental_data: %d row(s) upserted', n)
        return n

    def get_fundamentals(self, ticker, items: list = None, start=None, end=None,
                         wide: bool = True, daily: bool = False,
                         stale_limit_days: int = 120) -> pd.DataFrame:
        """Derived fundamental items, point-in-time by filed date.

        Default: observation rows (one per filed date), wide pivots items to
        columns. ``daily=True`` (always wide) aligns each item onto the
        stored trading dates with a grouped backward as-of join -- values go
        NaN once older than ``stale_limit_days``; observations before
        ``start`` still fill forward into the window. Single-ticker input
        drops the ticker level/column.
        """
        single = isinstance(ticker, str)
        tickers = [ticker] if single else list(ticker)
        if not tickers:
            return pd.DataFrame()
        stmt = (select(FundamentalData).where(FundamentalData.ticker.in_(tickers))
                .order_by(FundamentalData.ticker, FundamentalData.date))
        if items is not None:
            stmt = stmt.where(FundamentalData.item.in_(items))
        if not daily:  # daily needs pre-start observations to fill forward
            stmt = self._date_filter(stmt, FundamentalData.date, start, end)
        else:
            stmt = self._date_filter(stmt, FundamentalData.date, None, end)
        df = pd.read_sql(stmt, self.engine)

        if not daily:
            if wide and not df.empty:
                out = df.pivot(index=['ticker', 'date'], columns='item', values='value')
                return out.droplevel('ticker') if single else out
            if single and not df.empty:
                df = df.drop(columns='ticker')
            return df

        if df.empty:
            return pd.DataFrame()
        obs = df.pivot(index=['ticker', 'date'], columns='item', values='value').reset_index()
        obs['ts'] = pd.to_datetime(obs['date'])
        trading = self.get_price_data(tickers, 'daily', start, end)
        if trading.empty:
            return pd.DataFrame()
        left = trading.reset_index()[['ticker', 'date']]
        left['ts'] = pd.to_datetime(left['date'])
        merged = pd.merge_asof(left.sort_values('ts', kind='stable'),
                               obs.drop(columns='date').sort_values('ts', kind='stable'),
                               on='ts', by='ticker', direction='backward',
                               tolerance=pd.Timedelta(days=stale_limit_days))
        out = merged.drop(columns='ts').set_index(['ticker', 'date']).sort_index()
        return out.droplevel('ticker') if single else out

    # ------------------------------------------------------------------ #
    # EDGAR queries
    # ------------------------------------------------------------------ #

    def last_edgar_dates(self, ticker: str) -> dict:
        """Newest stored filing_date per EDGAR stream, for ticker_meta upkeep."""
        with self.engine.connect() as conn:
            filings = conn.execute(select(func.max(FilingsData.filing_date))
                                   .where(FilingsData.ticker == ticker)).scalar()
            form4 = conn.execute(select(func.max(Form4Data.filing_date))
                                 .where(Form4Data.ticker == ticker)).scalar()
        return {'last_filings_date': filings, 'last_form4_date': form4}

    def get_filings_meta(self, ticker, form: str = None) -> pd.DataFrame:
        """filings_data rows for a ticker, newest first."""
        if not isinstance(ticker, str):
            stmt = (select(FilingsData).where(FilingsData.ticker.in_(list(ticker)))
                    .order_by(FilingsData.ticker, FilingsData.filing_date.desc()))
            if form is not None:
                stmt = stmt.where(FilingsData.form_type == form)
            return pd.read_sql(stmt, self.engine)
        stmt = (select(FilingsData).where(FilingsData.ticker == ticker)
                .order_by(FilingsData.filing_date.desc()))
        if form is not None:
            stmt = stmt.where(FilingsData.form_type == form)
        return pd.read_sql(stmt, self.engine)

    def get_form4(self, ticker, start=None, end=None) -> pd.DataFrame:
        """Insider transactions, oldest first."""
        if not isinstance(ticker, str):
            stmt = (select(Form4Data).where(Form4Data.ticker.in_(list(ticker)))
                    .order_by(Form4Data.ticker, Form4Data.transaction_date,
                              Form4Data.accession_number, Form4Data.seq))
            stmt = self._date_filter(stmt, Form4Data.transaction_date, start, end)
            return pd.read_sql(stmt, self.engine)
        stmt = (select(Form4Data).where(Form4Data.ticker == ticker)
                .order_by(Form4Data.transaction_date, Form4Data.accession_number,
                          Form4Data.seq))
        stmt = self._date_filter(stmt, Form4Data.transaction_date, start, end)
        return pd.read_sql(stmt, self.engine)

    def get_income(self, ticker, items: list = None, interval: str = None,
                   wide: bool = False) -> pd.DataFrame:
        """Income-statement rows.

        ``interval``: 'quarterly' (fiscal_quarter 1-4), 'annual'
        (fiscal_quarter 0), or None for both. wide pivots items to columns
        indexed by (fiscal_year, fiscal_quarter).

        ``ticker`` may be a list; results gain a leading 'ticker' index
        level (wide) or 'ticker' column (long).
        """
        if not isinstance(ticker, str):
            stmt = (select(IncomeData).where(IncomeData.ticker.in_(list(ticker)))
                    .order_by(IncomeData.ticker, IncomeData.fiscal_year,
                              IncomeData.fiscal_quarter))
            stmt = self._statement_filters(stmt, IncomeData, items, interval)
            df = pd.read_sql(stmt, self.engine)
            if wide and not df.empty:
                pivot = df.pivot(index=['ticker', 'fiscal_year', 'fiscal_quarter'],
                                 columns='item', values='value')
                if interval == 'annual':
                    pivot.index = pivot.index.droplevel('fiscal_quarter')
                return pivot
            return df
        stmt = (select(IncomeData).where(IncomeData.ticker == ticker)
                .order_by(IncomeData.fiscal_year, IncomeData.fiscal_quarter))
        stmt = self._statement_filters(stmt, IncomeData, items, interval)
        df = pd.read_sql(stmt, self.engine)
        if wide and not df.empty:
            pivot = df.pivot(index=['fiscal_year', 'fiscal_quarter'],
                             columns='item', values='value')
            if interval == 'annual':
                pivot.index = pivot.index.get_level_values('fiscal_year')
            return pivot
        return df

    @staticmethod
    def _statement_filters(stmt, model, items, interval):
        if items is not None:
            stmt = stmt.where(model.item.in_(items))
        if interval == 'quarterly':
            stmt = stmt.where(model.fiscal_quarter > 0)
        elif interval == 'annual':
            stmt = stmt.where(model.fiscal_quarter == 0)
        return stmt

    def get_balance(self, ticker, items: list = None, start=None, end=None,
                    wide: bool = False) -> pd.DataFrame:
        """Balance-sheet rows (point-in-time). wide pivots items to columns
        indexed by balance date.

        ``ticker`` may be a list; results gain a leading 'ticker' index
        level (wide -> (ticker, date)) or 'ticker' column (long).
        """
        if not isinstance(ticker, str):
            stmt = (select(BalanceData).where(BalanceData.ticker.in_(list(ticker)))
                    .order_by(BalanceData.ticker, BalanceData.date))
            if items is not None:
                stmt = stmt.where(BalanceData.item.in_(items))
            stmt = self._date_filter(stmt, BalanceData.date, start, end)
            df = pd.read_sql(stmt, self.engine)
            if wide and not df.empty:
                return df.pivot(index=['ticker', 'date'], columns='item', values='value')
            return df
        stmt = (select(BalanceData).where(BalanceData.ticker == ticker)
                .order_by(BalanceData.date))
        if items is not None:
            stmt = stmt.where(BalanceData.item.in_(items))
        stmt = self._date_filter(stmt, BalanceData.date, start, end)
        df = pd.read_sql(stmt, self.engine)
        if wide and not df.empty:
            return df.pivot(index='date', columns='item', values='value')
        return df

    def get_cashflow(self, ticker, items: list = None, derived: bool = None,
                     wide: bool = False) -> pd.DataFrame:
        """Cash-flow rows (cumulative windows; duration_days tells how far
        into the fiscal year each window reaches).

        ``derived``: True -> only de-cumulated single quarters, False -> only
        as-reported cumulative rows, None -> both. wide pivots items to
        columns indexed by (start_date, end_date).

        ``ticker`` may be a list; results gain a leading 'ticker' index
        level (wide) or 'ticker' column (long).
        """
        if not isinstance(ticker, str):
            stmt = (select(CashflowData).where(CashflowData.ticker.in_(list(ticker)))
                    .order_by(CashflowData.ticker, CashflowData.end_date,
                              CashflowData.duration_days))
            if items is not None:
                stmt = stmt.where(CashflowData.item.in_(items))
            if derived is not None:
                stmt = stmt.where(CashflowData.derived == derived)
            df = pd.read_sql(stmt, self.engine)
            if wide and not df.empty:
                return df.pivot(index=['ticker', 'start_date', 'end_date'],
                                columns='item', values='value')
            return df
        stmt = (select(CashflowData).where(CashflowData.ticker == ticker)
                .order_by(CashflowData.end_date, CashflowData.duration_days))
        if items is not None:
            stmt = stmt.where(CashflowData.item.in_(items))
        if derived is not None:
            stmt = stmt.where(CashflowData.derived == derived)
        df = pd.read_sql(stmt, self.engine)
        if wide and not df.empty:
            return df.pivot(index=['start_date', 'end_date'], columns='item',
                            values='value')
        return df

    def get_financials(self, ticker, interval: str = 'quarterly',
                       items: list = None) -> pd.DataFrame:
        """One wide frame of all financial items across the three statements.

        interval='annual' is indexed by fiscal_year; 'quarterly' by the
        MultiIndex (fiscal_year, fiscal_quarter). Per statement:
        income -- annual rows or as-reported quarters (incl. derived Q4);
        balance -- fiscal-year-end values, shown as quarter 4 in the
        quarterly view; cash flow -- full-year windows for annual,
        single-quarter windows for quarterly (the as-reported Q1 plus the
        derived rows the parser de-cumulates for Q2-Q4).

        ``ticker`` may be a list; results gain a leading 'ticker' index level
        -> (ticker, fiscal_year[, fiscal_quarter]).
        """
        if interval not in ('annual', 'quarterly'):
            raise ValueError("interval must be 'annual' or 'quarterly'")
        if not isinstance(ticker, str):
            return self._stack_tickers(self.get_financials, ticker,
                                       interval=interval, items=items)
        annual = interval == 'annual'

        income = self.get_income(ticker, items=items)
        balance = self.get_balance(ticker, items=items)
        cashflow = self.get_cashflow(ticker, items=items)

        parts = []
        if not income.empty:
            parts.append(income[income['fiscal_quarter'] == 0] if annual
                         else income[income['fiscal_quarter'] > 0])
        if not balance.empty:
            balance = balance[balance['fiscal_quarter'].notna()].copy()
            if annual:
                parts.append(balance[balance['fiscal_quarter'] == 0])
            else:
                balance['fiscal_quarter'] = balance['fiscal_quarter'].replace(0, 4)
                parts.append(balance[balance['fiscal_quarter'].isin([1, 2, 3, 4])])
        if not cashflow.empty:
            if annual:
                parts.append(cashflow[cashflow['fiscal_quarter'] == 0])
            else:
                quarters = cashflow[cashflow['duration_days'] <= 110].copy()
                quarters['fiscal_quarter'] = quarters['fiscal_quarter'].replace(0, 4)
                parts.append(quarters)

        parts = [p[['fiscal_year', 'fiscal_quarter', 'item', 'value']] for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame()
        long = (pd.concat(parts, ignore_index=True)
                .drop_duplicates(subset=['fiscal_year', 'fiscal_quarter', 'item'], keep='last'))
        if annual:
            return (long.pivot(index='fiscal_year', columns='item', values='value')
                    .sort_index())
        long['fiscal_quarter'] = long['fiscal_quarter'].astype(int)
        return (long.pivot(index=['fiscal_year', 'fiscal_quarter'], columns='item',
                           values='value').sort_index())
