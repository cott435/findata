"""Persistence and query layer for the stock_data SQLite database.

All DB reads/writes go through DBManager. Acquisition (yf.py) and math
(technical_calculators.py) never touch the database; this module wires them together
for ingestion and exposes the query API used by downstream projects.

Conflict policy: price/ema/indicator rows are INSERT OR IGNORE (append-only);
ticker_meta is upserted.
"""

import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import and_, create_engine, func, select, tuple_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm.auto import tqdm


from configs import DATA_DIR, DB_NAME

import src.technical_calculators as calc
from .tables import Base, EmaData, IndicatorData, PriceData, TickerMeta

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(DATA_DIR) / DB_NAME

PRICE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio']
EMA_ITEMS = ('ema_close', 'ema_obv', 'ema_ad')
CUMULATIVE_ITEMS = ('obv', 'ad')

# trailing bars fetched ahead of new dates so rolling windows are exact on update
WARMUP_BARS = 150

# reverse map: indicator column -> calculator group that produces it
_COLUMN_TO_GROUP = {col: name for name, cols in calc.INDICATOR_OUTPUTS.items() for col in cols}


class MissingItemsError(LookupError):
    def __init__(self, items):
        self.items = list(items)
        super().__init__(
            f"Items not in database: {self.items}. "
            f"Set if_missing='add' to compute and store them.")


class DBManager:
    """Single entry point for writing to and querying the stock database.

    ``if_missing`` controls what get_items does when a requested
    (item, period) has no stored rows: 'raise' raises MissingItemsError,
    'add' computes it from stored price data, inserts it, and returns it.
    """

    def __init__(self, db_path=None, if_missing: str = 'raise'):
        if if_missing not in ('raise', 'add'):
            raise ValueError("if_missing must be 'raise' or 'add'")
        self.if_missing = if_missing
        db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #

    def add_ticker_data(self, info: pd.DataFrame, prices: pd.DataFrame,
                        progress: bool = True) -> dict:
        """Cold-start ingestion from the yf.py output.

        ``info`` is indexed by ticker; ``prices`` by (interval, ticker, date).
        Processes one ticker at a time -- insert bars, compute the full
        EMA/indicator history, insert, mark yf_seeded -- so the progress bar
        steps per ticker and an interrupted run resumes where it stopped.
        Idempotent (INSERT OR IGNORE); counts are rows actually inserted.
        """
        prices = self._standardize_prices(prices)
        tickers = prices.index.get_level_values('ticker').unique()
        logger.info('Cold-start: seeding %d ticker(s)', len(tickers))
        counts = {'price': 0, 'ema': 0, 'indicator': 0}

        for ticker in tqdm(tickers, desc='Seeding', unit='ticker', disable=not progress):
            sub = prices.xs(ticker, level='ticker', drop_level=False)
            inserted = {'price': self._insert_prices(sub)}
            calc_df = calc.calculate_all(sub)
            inserted |= self._insert_calc_long(calc_df.melt(ignore_index=False).reset_index())
            if ticker in info.index:
                self._upsert_meta(info.loc[[ticker]], yf_seeded=True)
            else:
                logger.warning('%s: no metadata returned, not marked yf_seeded', ticker)
            for key in counts:
                counts[key] += inserted[key]
            logger.debug('%s seeded: %s', ticker, inserted)

        logger.info('Cold-start done, rows inserted: %s', counts)
        return counts

    def update_ticker_data(self, prices: pd.DataFrame, info: pd.DataFrame = None,
                           progress: bool = True) -> dict:
        """Incremental ingestion of new bars for already-seeded tickers.

        Inserts the new price rows, then continues each EMA/indicator from
        its last stored value (seeded EWMs/cumsums; rolling windows recompute
        over WARMUP_BARS trailing bars). The periods to maintain are taken
        from what is already stored for the ticker, so custom periods added
        via if_missing='add' keep updating too. A ticker/interval with no
        stored calculations falls back to a full cold-start computation.
        Counts are rows actually inserted (overlap is ignored on conflict).
        """
        prices = self._standardize_prices(prices)
        counts = {'price': self._insert_prices(prices), 'ema': 0, 'indicator': 0}
        groups = list(prices.groupby(level=['interval', 'ticker']))
        logger.info('Incremental update: %d new price rows across %d ticker/interval group(s)',
                    counts['price'], len(groups))

        for (interval, ticker), new_bars in tqdm(groups, desc='Updating', unit='group',
                                                 disable=not progress):
            seeds = self._load_seeds(ticker, interval)
            if seeds:
                window = self._price_window(ticker, interval, len(new_bars) + WARMUP_BARS)
                stored_windows = self._stored_windows(ticker, interval)
                calc_df = calc.calculate_all(window, seeds=seeds, **stored_windows)
            else:
                logger.debug('%s/%s has no stored calculations, computing cold', ticker, interval)
                calc_df = calc.calculate_all(self.get_price_data(ticker, interval))
            long = calc_df.melt(ignore_index=False).reset_index(names='date')
            long['ticker'], long['interval'] = ticker, interval
            inserted = self._insert_calc_long(long)
            counts['ema'] += inserted['ema']
            counts['indicator'] += inserted['indicator']
            logger.debug('%s/%s updated: %s', ticker, interval, inserted)

        if info is None:
            info = self._last_price_dates(prices)
        if not info.empty:
            self._upsert_meta(info)
        logger.info('Incremental update done, rows inserted: %s', counts)
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
        return df[df['close'].notna()][PRICE_COLUMNS]

    @staticmethod
    def _last_price_dates(prices: pd.DataFrame) -> pd.DataFrame:
        idx = prices.index.to_frame(index=False)
        daily = idx[idx['interval'] == 'daily']
        return daily.groupby('ticker')['date'].max().rename('last_price_date').to_frame()

    def _insert_prices(self, prices: pd.DataFrame) -> int:
        return self._insert_ignore(PriceData, prices.reset_index())

    def _insert_calc_long(self, long: pd.DataFrame) -> dict:
        """Split a melted calculate_all frame into ema_data / indicator_data rows.

        Expects columns [interval, ticker, date, item, period, value].
        """
        long = long.dropna(subset=['value'])
        long['period'] = long['period'].astype(int)
        is_ema = long['item'].str.startswith('ema_')

        emas = long[is_ema].copy()
        emas['base'] = emas['item'].str[4:]
        emas = emas.rename(columns={'value': 'ema_value'})
        n_ema = self._insert_ignore(
            EmaData, emas[['ticker', 'date', 'period', 'base', 'interval', 'ema_value']])

        inds = long[~is_ema].rename(columns={'item': 'indicator', 'interval': 'freq'})
        n_ind = self._insert_ignore(
            IndicatorData, inds[['ticker', 'date', 'freq', 'indicator', 'period', 'value']])
        return {'ema': n_ema, 'indicator': n_ind}

    def _upsert_meta(self, info: pd.DataFrame, **flags):
        allowed = ['ticker', 'name', 'sector', 'industry', 'last_price_date',
                   'last_filings_date', 'yf_seeded', 'edgar_seeded']
        df = info.reset_index()
        records = self._records(df[[c for c in df.columns if c in allowed]])
        now = datetime.now()
        with self.engine.begin() as conn:
            for record in records:
                record.update(flags, updated_at=now)
                stmt = sqlite_insert(TickerMeta.__table__).values(**record)
                update_cols = {k: stmt.excluded[k] for k in record if k != 'ticker'}
                conn.execute(stmt.on_conflict_do_update(index_elements=['ticker'], set_=update_cols))

    def _insert_ignore(self, model, df: pd.DataFrame) -> int:
        """Bulk INSERT OR IGNORE through the raw sqlite3 driver.

        The data is trusted (already validated upstream), so rows go in as
        plain tuples via executemany -- far faster than per-row dicts through
        the ORM layer. Returns the number of rows actually inserted
        (conflicting rows are excluded by sqlite's change counter).
        """
        if df.empty:
            return 0
        sql = (f"INSERT OR IGNORE INTO {model.__tablename__} "
               f"({', '.join(df.columns)}) VALUES ({', '.join('?' * len(df.columns))})")
        connection = self.engine.raw_connection()
        try:
            cursor = connection.cursor()
            cursor.executemany(sql, self._tuple_rows(df))
            connection.commit()
            return cursor.rowcount
        finally:
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
            if series.dtype.kind in 'iuf':
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

    def _load_seeds(self, ticker: str, interval: str) -> dict:
        """Last stored value per seedable (item, period) -> calculate_all seeds."""
        seeds = {}
        last_ema = (select(EmaData.base, EmaData.period, func.max(EmaData.date).label('date'))
                    .where(EmaData.ticker == ticker, EmaData.interval == interval)
                    .group_by(EmaData.base, EmaData.period).subquery())
        ema_stmt = (select(EmaData.base, EmaData.period, EmaData.date, EmaData.ema_value)
                    .join(last_ema, and_(EmaData.base == last_ema.c.base,
                                         EmaData.period == last_ema.c.period,
                                         EmaData.date == last_ema.c.date))
                    .where(EmaData.ticker == ticker, EmaData.interval == interval))

        last_ind = (select(IndicatorData.indicator, IndicatorData.period,
                           func.max(IndicatorData.date).label('date'))
                    .where(IndicatorData.ticker == ticker, IndicatorData.freq == interval,
                           IndicatorData.indicator.in_(calc.SEEDED_ITEMS))
                    .group_by(IndicatorData.indicator, IndicatorData.period).subquery())
        ind_stmt = (select(IndicatorData.indicator, IndicatorData.period,
                           IndicatorData.date, IndicatorData.value)
                    .join(last_ind, and_(IndicatorData.indicator == last_ind.c.indicator,
                                         IndicatorData.period == last_ind.c.period,
                                         IndicatorData.date == last_ind.c.date))
                    .where(IndicatorData.ticker == ticker, IndicatorData.freq == interval))

        with self.engine.connect() as conn:
            for base, period, date, value in conn.execute(ema_stmt):
                seeds[(f'ema_{base}', period)] = pd.Series([value], index=[date])
            for indicator, period, date, value in conn.execute(ind_stmt):
                seeds[(indicator, period)] = pd.Series([value], index=[date])
        return seeds

    def _stored_windows(self, ticker: str, interval: str) -> dict:
        """Periods already stored for this ticker -> calculate_all overrides.

        Ensures incremental updates maintain every (item, period) combo in
        the DB, not just the configured defaults.
        """
        ema_stmt = (select(EmaData.base, EmaData.period).distinct()
                    .where(EmaData.ticker == ticker, EmaData.interval == interval))
        ind_stmt = (select(IndicatorData.indicator, IndicatorData.period).distinct()
                    .where(IndicatorData.ticker == ticker, IndicatorData.freq == interval,
                           IndicatorData.period > 0))
        windows = {}
        with self.engine.connect() as conn:
            for base, period in conn.execute(ema_stmt):
                windows.setdefault(f'ema_{base}', set()).add(period)
            for indicator, period in conn.execute(ind_stmt):
                windows.setdefault(_COLUMN_TO_GROUP[indicator], set()).add(period)
        return {name: sorted(periods) for name, periods in windows.items()}

    # ------------------------------------------------------------------ #
    # query API
    # ------------------------------------------------------------------ #

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
        cols = ['name', 'sector', 'industry', 'last_price_date', 'yf_seeded', 'edgar_seeded']
        return meta[cols].join(bars).join(daily_range)

    def get_price_data(self, ticker: str, interval: str = 'daily',
                       start=None, end=None) -> pd.DataFrame:
        stmt = (select(PriceData.date, *[getattr(PriceData, c) for c in PRICE_COLUMNS])
                .where(PriceData.ticker == ticker, PriceData.interval == interval)
                .order_by(PriceData.date))
        stmt = self._date_filter(stmt, PriceData.date, start, end)
        return pd.read_sql(stmt, self.engine).set_index('date')

    def get_all_data(self, ticker: str, interval: str = 'daily',
                     start=None, end=None, wide: bool = True) -> pd.DataFrame:
        """Everything stored for a ticker/interval: prices + EMAs + indicators.

        wide=True returns one date-indexed frame (calculated columns named
        '<item>_<period>', period-0 items keep their bare name). wide=False
        returns a long frame [date, item, period, value] with the price
        columns included as period-0 items.
        """
        price = self.get_price_data(ticker, interval, start, end)
        calc_long = pd.concat([
            self._read_ema_long(ticker, interval, None, start, end),
            self._read_ind_long(ticker, interval, None, start, end),
        ], ignore_index=True)
        return self._shape(price, calc_long, wide)

    def get_items(self, ticker: str, items: dict, interval: str = 'daily',
                  start=None, end=None, wide: bool = True) -> pd.DataFrame:
        """Fetch specific items: {name: period(s) or None}.

        Names can be price columns ('close'), EMA items ('ema_close',
        'ema_obv', 'ema_ad'), indicator columns ('rsi', 'bb_upper', 'atr',
        'obv', ...) or calculator group names ('bollinger', 'stochastic',
        'adx', 'aroon' -- expanded to all their output columns). Periods may
        be an int, a list of ints, or None for the configured default
        (ignored for price columns and obv/ad).

        Missing (item, period) combos follow self.if_missing: 'raise' raises
        MissingItemsError; 'add' computes them from stored price data,
        inserts them, and returns them.
        """
        price_cols, ema_pairs, ind_pairs = self._resolve_items(items)

        missing_emas = self._missing_pairs(ticker, interval, EmaData, ema_pairs)
        missing_inds = self._missing_pairs(ticker, interval, IndicatorData, ind_pairs)
        if missing_emas or missing_inds:
            if self.if_missing == 'raise':
                raise MissingItemsError(
                    [(f'ema_{base}', p) for base, p in missing_emas] + missing_inds)
            self._add_missing(ticker, interval, missing_emas, missing_inds)

        parts = []
        if ema_pairs:
            parts.append(self._read_ema_long(ticker, interval, ema_pairs, start, end))
        if ind_pairs:
            parts.append(self._read_ind_long(ticker, interval, ind_pairs, start, end))
        calc_long = pd.concat(parts, ignore_index=True) if parts else None
        price = (self.get_price_data(ticker, interval, start, end)[price_cols]
                 if price_cols else None)
        return self._shape(price, calc_long, wide)

    # ------------------------------------------------------------------ #
    # get_items internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_items(items: dict):
        """Normalize the request dict into price columns and (item, period) pairs."""
        price_cols, ema_pairs, ind_pairs = [], [], []
        for name, periods in items.items():
            if name in PRICE_COLUMNS:
                price_cols.append(name)
            elif name in EMA_ITEMS:
                periods = calc.EMA_WINDOWS[name] if periods is None else periods
                ema_pairs += [(name[4:], int(p)) for p in calc._as_list(periods)]
            elif name in CUMULATIVE_ITEMS:
                ind_pairs.append((name, 0))
            elif name in calc.INDICATOR_FUNCS:  # group name -> all output columns
                periods = calc.INDICATOR_WINDOWS[name] if periods is None else periods
                for p in calc._as_list(periods):
                    ind_pairs += [(col, int(p)) for col in calc.INDICATOR_OUTPUTS[name]]
            elif name in _COLUMN_TO_GROUP:  # single output column
                group = _COLUMN_TO_GROUP[name]
                periods = calc.INDICATOR_WINDOWS[group] if periods is None else periods
                ind_pairs += [(name, int(p)) for p in calc._as_list(periods)]
            else:
                raise KeyError(f"Unknown item '{name}'")
        return price_cols, sorted(set(ema_pairs)), sorted(set(ind_pairs))

    def _missing_pairs(self, ticker, interval, model, pairs) -> list:
        if not pairs:
            return []
        if model is EmaData:
            stmt = (select(EmaData.base, EmaData.period).distinct()
                    .where(EmaData.ticker == ticker, EmaData.interval == interval))
        else:
            stmt = (select(IndicatorData.indicator, IndicatorData.period).distinct()
                    .where(IndicatorData.ticker == ticker, IndicatorData.freq == interval))
        with self.engine.connect() as conn:
            existing = set(map(tuple, conn.execute(stmt)))
        return [pair for pair in pairs if pair not in existing]

    def _add_missing(self, ticker, interval, ema_pairs, ind_pairs):
        """if_missing='add': compute missing items over full stored history."""
        logger.info('Computing missing items for %s/%s: emas=%s indicators=%s',
                    ticker, interval, ema_pairs, ind_pairs)
        prices = self.get_price_data(ticker, interval)
        if prices.empty:
            raise MissingItemsError(
                [('price_data', interval)])  # nothing to compute from

        # indicators first -- ema_obv/ema_ad may need a freshly added base
        groups = {(_COLUMN_TO_GROUP[col], p) if col not in CUMULATIVE_ITEMS else (col, 0)
                  for col, p in ind_pairs}
        for group, period in sorted(groups):
            if group in CUMULATIVE_ITEMS:
                result = getattr(calc, group)(prices)
            else:
                result = calc.INDICATOR_FUNCS[group](prices, period=period)
            if isinstance(result, pd.Series):
                result = result.to_frame()
            self._insert_indicator_frame(result, ticker, interval, period)

        for base, period in ema_pairs:
            series = prices['close'] if base == 'close' else \
                self._base_series(ticker, interval, prices, base)
            values = calc.ema(series, period).dropna()
            rows = pd.DataFrame({
                'ticker': ticker, 'date': values.index, 'period': period,
                'base': base, 'interval': interval, 'ema_value': values.values})
            self._insert_ignore(EmaData, rows)

    def _insert_indicator_frame(self, frame, ticker, interval, period):
        long = (frame.melt(ignore_index=False, var_name='indicator', value_name='value')
                .reset_index(names='date').dropna(subset=['value']))
        long['ticker'], long['freq'], long['period'] = ticker, interval, period
        self._insert_ignore(
            IndicatorData, long[['ticker', 'date', 'freq', 'indicator', 'period', 'value']])

    def _base_series(self, ticker, interval, prices, base) -> pd.Series:
        """obv/ad series for EMA bases, computing and storing it if absent."""
        stored = self._read_ind_long(ticker, interval, [(base, 0)])
        if not stored.empty:
            return stored.set_index('date')['value']
        series = getattr(calc, base)(prices)
        self._insert_indicator_frame(series.to_frame(), ticker, interval, 0)
        return series

    # ------------------------------------------------------------------ #
    # reads + shaping
    # ------------------------------------------------------------------ #

    def _read_ema_long(self, ticker, interval, pairs=None, start=None, end=None) -> pd.DataFrame:
        stmt = (select(EmaData.date, EmaData.base, EmaData.period,
                       EmaData.ema_value.label('value'))
                .where(EmaData.ticker == ticker, EmaData.interval == interval)
                .order_by(EmaData.date))
        if pairs is not None:
            stmt = stmt.where(tuple_(EmaData.base, EmaData.period).in_(pairs))
        stmt = self._date_filter(stmt, EmaData.date, start, end)
        df = pd.read_sql(stmt, self.engine)
        df['item'] = 'ema_' + df['base']
        return df[['date', 'item', 'period', 'value']]

    def _read_ind_long(self, ticker, interval, pairs=None, start=None, end=None) -> pd.DataFrame:
        stmt = (select(IndicatorData.date, IndicatorData.indicator.label('item'),
                       IndicatorData.period, IndicatorData.value)
                .where(IndicatorData.ticker == ticker, IndicatorData.freq == interval)
                .order_by(IndicatorData.date))
        if pairs is not None:
            stmt = stmt.where(tuple_(IndicatorData.indicator, IndicatorData.period).in_(pairs))
        stmt = self._date_filter(stmt, IndicatorData.date, start, end)
        return pd.read_sql(stmt, self.engine)[['date', 'item', 'period', 'value']]

    @staticmethod
    def _date_filter(stmt, column, start, end):
        if start is not None:
            stmt = stmt.where(column >= pd.to_datetime(start).date())
        if end is not None:
            stmt = stmt.where(column <= pd.to_datetime(end).date())
        return stmt

    @staticmethod
    def _shape(price: pd.DataFrame, calc_long: pd.DataFrame, wide: bool) -> pd.DataFrame:
        """Assemble price (wide, date-indexed) + calc rows into the output format."""
        has_price = price is not None and not price.empty
        has_calc = calc_long is not None and not calc_long.empty

        if wide:
            parts = []
            if has_price:
                parts.append(price)
            if has_calc:
                pivot = calc_long.pivot(index='date', columns=['item', 'period'],
                                        values='value').sort_index(axis=1)
                pivot.columns = [item if not period else f'{item}_{period}'
                                 for item, period in pivot.columns]
                parts.append(pivot)
            if not parts:
                return pd.DataFrame()
            return pd.concat(parts, axis=1).sort_index()

        parts = []
        if has_price:
            parts.append(price.reset_index()
                         .melt(id_vars='date', var_name='item', value_name='value')
                         .assign(period=0))
        if has_calc:
            parts.append(calc_long)
        if not parts:
            return pd.DataFrame(columns=['date', 'item', 'period', 'value'])
        long = pd.concat(parts, ignore_index=True)[['date', 'item', 'period', 'value']]
        return long.sort_values(['item', 'period', 'date'], ignore_index=True)
