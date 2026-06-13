"""EDGAR acquisition pipeline (edgartools).

One synchronous entry point -- ``EdgarPipeline.run(tickers)`` -- covering both
cold-start and incremental update (decided per ticker from ticker_meta).
Only HTTP runs async; orchestration, parsing and DB writes stay synchronous.

The fetch side is four queue-fed stages, mirroring the edgartools call chain:

    1. company queue:  ticker            -> Company()        (submissions request)
    2. filings queue:  company           -> get_filings()    (filing list, paged)
    3. obj queue:      filing            -> filing.obj()     (download + parse)
    4. facts queue:    company           -> get_facts()      (all XBRL facts, one request)

Stage workers share one concurrency semaphore and one rate limiter so total
request pressure stays under the SEC ceiling regardless of stage. Workers
reduce each filing object to a small payload (section texts / trade frames)
immediately, so full documents never accumulate in memory. Tickers are
fetched in small batches; each batch is then processed synchronously:

    10-K / 10-Q -> section .txt files on disk + filings_data row
    8-K         -> per-item .txt files (date + item number prepended) + filings_data row
    Form 4      -> one standardized DataFrame -> form4_data table
    facts       -> income_data / balance_data / cashflow_data tables

Financial facts are parsed from a single ``facts.query().to_dataframe()``
frame per ticker -- concepts are resolved locally with priority-ordered tag
lists (companies report the "same" item under different us-gaap tags), so no
per-concept queries are made.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from edgar import Company, find, set_identity
from tqdm.auto import tqdm

from findata.configs import EDGAR_IDENTITY, SEC_DIR

from .db_manager import DBManager

logging.getLogger("edgar").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# section inventories (spec 4.1 / 4.2): key -> (edgartools item, essential)
# ---------------------------------------------------------------------- #

TENK_SECTIONS = {
    'business': ('Item 1', True),
    'risk_factors': ('Item 1A', True),
    'legal_proceedings': ('Item 3', True),
    'mda': ('Item 7', True),
    'market_risk': ('Item 7A', True),
    'financial_statements': ('Item 8', True),
    'cybersecurity': ('Item 1C', False),
    'properties': ('Item 2', False),
    'controls_and_procedures': ('Item 9A', False),
    'directors_and_governance': ('Item 10', False),
    'executive_compensation': ('Item 11', False),
    'security_ownership': ('Item 12', False),
    'related_party_transactions': ('Item 13', False),
    'accountant_fees': ('Item 14', False),
    'exhibits_index': ('Item 15', False),
}

TENQ_SECTIONS = {
    'financial_statements': ('Part I, Item 1', True),
    'mda': ('Part I, Item 2', True),
    'legal_proceedings': ('Part II, Item 1', True),
    'market_risk_changes': ('Part I, Item 3', False),
    'controls_changes': ('Part I, Item 4', False),
    'risk_factor_changes': ('Part II, Item 1A', False),
    'share_repurchases': ('Part II, Item 2', False),
    'other_information': ('Part II, Item 5', False),
}

# ---------------------------------------------------------------------- #
# concept priorities: item -> ordered us-gaap/dei tag candidates. The first
# tag that reports a given period wins, so tag migrations (e.g. AAPL moving
# to RevenueFromContractWithCustomer...) and issuer quirks (SOFI's
# RevenuesNetOfInterestExpense, VZ's plain Revenues) resolve cleanly.
# Extend by appending tags / items here -- nothing else needs to change.
# ---------------------------------------------------------------------- #

INCOME_ITEMS = {
    'revenue': ['RevenueFromContractWithCustomerExcludingAssessedTax',
                'RevenuesNetOfInterestExpense', 'Revenues', 'SalesRevenueNet',
                'SalesRevenueGoodsNet', 'SalesRevenueServicesNet'],
    'cost_of_revenue': ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfGoodsSold'],
    'gross_profit': ['GrossProfit'],
    'operating_expenses': ['OperatingExpenses', 'CostsAndExpenses'],
    'rnd_expense': ['ResearchAndDevelopmentExpense'],
    'operating_income': ['OperatingIncomeLoss'],
    'pretax_income': ['IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                      'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'],
    'income_tax': ['IncomeTaxExpenseBenefit'],
    'net_income': ['NetIncomeLoss', 'ProfitLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic'],
    'eps_basic': ['EarningsPerShareBasic'],
    'eps_diluted': ['EarningsPerShareDiluted'],
    'shares_basic': ['WeightedAverageNumberOfSharesOutstandingBasic'],
    'shares_diluted': ['WeightedAverageNumberOfDilutedSharesOutstanding'],
}

# averages are not additive across quarters -- never derive a Q4 for these
NON_ADDITIVE_ITEMS = ('shares_basic', 'shares_diluted')

BALANCE_ITEMS = {
    'assets': ['Assets'],
    'assets_current': ['AssetsCurrent'],
    'assets_noncurrent': ['AssetsNoncurrent'],
    'liabilities': ['Liabilities'],
    'liabilities_current': ['LiabilitiesCurrent'],
    'liabilities_noncurrent': ['LiabilitiesNoncurrent'],
    'stockholders_equity': ['StockholdersEquity',
                            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'cash_and_equivalents': ['CashAndCashEquivalentsAtCarryingValue',
                             'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
    'short_term_investments': ['ShortTermInvestments', 'MarketableSecuritiesCurrent'],
    'inventory': ['InventoryNet'],
    'goodwill': ['Goodwill'],
    'long_term_debt': ['LongTermDebtNoncurrent', 'LongTermDebt',
                       'LongTermDebtAndCapitalLeaseObligations'],
    'shares_outstanding': ['CommonStockSharesOutstanding', 'EntityCommonStockSharesOutstanding'],
}

CASHFLOW_ITEMS = {
    'operating_cash_flow': ['NetCashProvidedByUsedInOperatingActivities',
                            'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'],
    'investing_cash_flow': ['NetCashProvidedByUsedInInvestingActivities',
                            'NetCashProvidedByUsedInInvestingActivitiesContinuingOperations'],
    'financing_cash_flow': ['NetCashProvidedByUsedInFinancingActivities',
                            'NetCashProvidedByUsedInFinancingActivitiesContinuingOperations'],
    'capex': ['PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets'],
    'depreciation_amortization': ['DepreciationDepletionAndAmortization',
                                  'DepreciationAmortizationAndAccretionNet', 'Depreciation'],
    'dividends_paid': ['PaymentsOfDividendsCommonStock', 'PaymentsOfDividends'],
    'stock_buybacks': ['PaymentsForRepurchaseOfCommonStock'],
}

QUARTER_DAYS = (70, 110)    # one fiscal quarter (13/14-week)
ANNUAL_DAYS = (330, 400)    # one fiscal year (52/53-week)

TEXT_FORMS = ('10-K', '10-Q', '8-K')


def _to_date(value):
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == '':
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value).date()
    except (ValueError, TypeError):
        return None


def _slug(name: str) -> str:
    return name.lower().replace('.', '_').replace(',', '').replace(' ', '_')


# ---------------------------------------------------------------------- #
# financial facts parsing (pure functions -- no I/O)
# ---------------------------------------------------------------------- #

def parse_financials(facts: pd.DataFrame, ticker: str,
                     decumulate_cashflow: bool = False) -> dict:
    """Parse one ticker's full facts frame into the three statement tables.

    Returns {'income': df, 'balance': df, 'cashflow': df} with columns
    matching income_data / balance_data / cashflow_data. Values are
    as-originally-reported: when a period shows up again in later filings
    (comparatives, restatements) the earliest filing wins.
    """
    if facts is None or facts.empty:
        return {'income': pd.DataFrame(), 'balance': pd.DataFrame(), 'cashflow': pd.DataFrame()}

    needed = ['concept', 'numeric_value', 'period_start', 'period_end', 'period_type',
              'fiscal_year', 'fiscal_period', 'filing_date']
    df = facts.loc[facts['numeric_value'].notna(), needed].copy()
    df['tag'] = df['concept'].str.split(':').str[-1]
    df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
    df['period_start'] = pd.to_datetime(df['period_start'], errors='coerce')
    df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    df['fiscal_quarter'] = df['fiscal_period'].map({'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4, 'FY': 0})
    df = df[df['period_end'].notna()]

    instant = df[df['period_type'] == 'instant']
    duration = df[df['period_type'] == 'duration'].copy()
    duration = duration[duration['period_start'].notna()]
    duration['duration_days'] = (duration['period_end'] - duration['period_start']).dt.days

    income = _parse_income(duration)
    balance = _parse_balance(instant)
    cashflow = _parse_cashflow(duration, decumulate=decumulate_cashflow)
    for frame in (income, balance, cashflow):
        if not frame.empty:
            frame.insert(0, 'ticker', ticker)
    return {'income': income, 'balance': balance, 'cashflow': cashflow}


def _resolve(pool: pd.DataFrame, items: dict, key_cols: list) -> pd.DataFrame:
    """Priority-ordered concept resolution.

    For every item, gather rows of all candidate tags; per reporting period
    (key_cols) keep the highest-priority tag, then the earliest filing.
    """
    frames = []
    for item, tags in items.items():
        sub = pool[pool['tag'].isin(tags)]
        if sub.empty:
            continue
        rank = {tag: i for i, tag in enumerate(tags)}
        frames.append(sub.assign(item=item, rank=sub['tag'].map(rank)))
    if not frames:
        return pd.DataFrame(columns=list(pool.columns) + ['item', 'rank'])
    resolved = pd.concat(frames, ignore_index=True)
    return (resolved.sort_values(['item'] + key_cols + ['rank', 'filing_date'])
            .drop_duplicates(subset=['item'] + key_cols, keep='first'))


def _finalize(frame: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Rename to table columns, convert timestamps to dates, order columns."""
    out = frame.rename(columns={'numeric_value': 'value', 'period_start': 'start_date',
                                'period_end': 'end_date', 'filing_date': 'filed_date'})
    for col in ('start_date', 'end_date', 'filed_date', 'date'):
        if col in out.columns and col in columns:
            out[col] = pd.to_datetime(out[col]).dt.date
    return out[columns].reset_index(drop=True)


def _parse_income(duration: pd.DataFrame) -> pd.DataFrame:
    """Quarterly (as-reported 3-month) + annual income rows, deriving Q4
    as annual minus Q1-Q3 when the company never reports it standalone."""
    rows = _resolve(duration, INCOME_ITEMS, ['period_start', 'period_end'])
    if rows.empty:
        return pd.DataFrame()

    annual = rows[rows['duration_days'].between(*ANNUAL_DAYS)].copy()
    annual['fiscal_quarter'] = 0
    quarterly = rows[rows['duration_days'].between(*QUARTER_DAYS)].copy()

    # a 3-month window ending on a fiscal-year end is Q4 even when the fact
    # carries the 10-K's 'FY' label (some issuers report Q4 in the 10-K)
    annual_ends = set(zip(annual['item'], annual['period_end']))
    ends_on_fy = [(item, end) in annual_ends
                  for item, end in zip(quarterly['item'], quarterly['period_end'])]
    quarterly.loc[~quarterly['fiscal_quarter'].isin([1, 2, 3, 4]) & pd.Series(ends_on_fy, index=quarterly.index),
                  'fiscal_quarter'] = 4
    quarterly = quarterly[quarterly['fiscal_quarter'].isin([1, 2, 3, 4])]

    derived = []
    for (item, fy), ann in annual.groupby(['item', 'fiscal_year']):
        if item in NON_ADDITIVE_ITEMS:
            continue
        ann = ann.iloc[0]
        qs = quarterly[(quarterly['item'] == item) & (quarterly['fiscal_year'] == fy)]
        present = set(qs['fiscal_quarter'])
        if {1, 2, 3} <= present and 4 not in present:
            q123 = qs[qs['fiscal_quarter'].isin([1, 2, 3])]
            derived.append({
                'item': item, 'fiscal_year': fy, 'fiscal_quarter': 4,
                'numeric_value': ann['numeric_value'] - q123['numeric_value'].sum(),
                'period_start': q123['period_end'].max() + pd.Timedelta(days=1),
                'period_end': ann['period_end'],
                'filing_date': ann['filing_date'], 'derived': True,
            })

    out = pd.concat([quarterly.assign(derived=False), annual.assign(derived=False),
                     pd.DataFrame(derived)], ignore_index=True)
    out = out[out['fiscal_year'].notna()]
    out['fiscal_year'] = out['fiscal_year'].astype(int)
    out['fiscal_quarter'] = out['fiscal_quarter'].astype(int)
    out = (out.sort_values(['item', 'fiscal_year', 'fiscal_quarter', 'filing_date'])
           .drop_duplicates(subset=['item', 'fiscal_year', 'fiscal_quarter'], keep='first'))
    return _finalize(out, ['fiscal_year', 'fiscal_quarter', 'item', 'value',
                           'start_date', 'end_date', 'filed_date', 'derived'])


def _parse_balance(instant: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time balance rows keyed by balance date -- no interval."""
    rows = _resolve(instant, BALANCE_ITEMS, ['period_end'])
    if rows.empty:
        return pd.DataFrame()
    rows = rows[rows['fiscal_year'].notna()].copy()
    rows['fiscal_year'] = rows['fiscal_year'].astype(int)
    rows['date'] = rows['period_end']
    rows = rows.drop_duplicates(subset=['item', 'date'], keep='first')
    return _finalize(rows, ['date', 'item', 'value', 'fiscal_year',
                            'fiscal_quarter', 'filed_date'])


def _parse_cashflow(duration: pd.DataFrame, decumulate: bool = False) -> pd.DataFrame:
    """Cumulative-from-fiscal-year-start cash-flow windows, duration kept.

    With ``decumulate`` True, consecutive windows sharing a fiscal-year start
    are differenced into standalone quarters and appended as derived rows.
    """
    rows = _resolve(duration, CASHFLOW_ITEMS, ['period_start', 'period_end'])
    if rows.empty:
        return pd.DataFrame()
    rows = rows[rows['duration_days'].between(40, 400) & rows['fiscal_year'].notna()].copy()
    rows['derived'] = False

    derived = []
    if decumulate:
        for (item, start), group in rows.groupby(['item', 'period_start']):
            group = group.sort_values('period_end')
            if len(group) < 2:
                continue
            previous = None
            for _, row in group.iterrows():
                if previous is not None:
                    quarter = row['fiscal_quarter'] if row['fiscal_quarter'] in (1, 2, 3) else 4
                    derived.append({
                        'item': item, 'numeric_value': row['numeric_value'] - previous['numeric_value'],
                        'period_start': previous['period_end'] + pd.Timedelta(days=1),
                        'period_end': row['period_end'],
                        'fiscal_year': row['fiscal_year'], 'fiscal_quarter': quarter,
                        'filing_date': row['filing_date'], 'derived': True,
                    })
                previous = row

    out = pd.concat([rows, pd.DataFrame(derived)], ignore_index=True)
    out['fiscal_year'] = out['fiscal_year'].astype(int)
    out['duration_days'] = (out['period_end'] - out['period_start']).dt.days
    out = out.drop_duplicates(subset=['item', 'period_start', 'period_end'], keep='first')
    return _finalize(out, ['start_date', 'end_date', 'item', 'value', 'duration_days',
                           'fiscal_year', 'fiscal_quarter', 'filed_date', 'derived'])


# ---------------------------------------------------------------------- #
# async fetch machinery
# ---------------------------------------------------------------------- #

class _RateLimiter:
    """Evenly spaces request starts: at most ``per_second`` per second."""

    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self):
        async with self._lock:
            now = asyncio.get_running_loop().time()
            delay = self._next - now
            self._next = max(self._next, now) + self._interval
        if delay > 0:
            await asyncio.sleep(delay)


@dataclass
class _TickerJob:
    ticker: str
    text_since: date = None      # fetch 10-K/10-Q/8-K filed strictly after this
    form4_since: date = None     # fetch Form 4 filed strictly after this
    known_accessions: set = field(default_factory=set)


@dataclass
class _TickerResult:
    ticker: str
    company_name: str = None
    cik: int = None
    filings: list = field(default_factory=list)   # reduced payload dicts
    facts: pd.DataFrame = None
    errors: list = field(default_factory=list)


class EdgarPipeline:
    """Fetches, parses and stores EDGAR filings + financial facts.

    ``run(tickers)`` is the single entry point: per ticker it decides between
    cold-start (full history) and incremental update (filings after the
    last_filings_date / last_form4_date stored in ticker_meta), fetches via
    the staged async queues, then writes text sections to ``sec_dir`` and
    rows to the database. Already-stored accession numbers are skipped, so
    an interrupted init resumes where it stopped.
    """

    def __init__(self, db: DBManager = None, sec_dir=None, identity: str = EDGAR_IDENTITY,
                 parse_extra_data: bool = False, forms=('10-K', '10-Q', '8-K', '4'),
                 max_concurrency: int = 4, requests_per_second: float = 8.0,
                 batch_size: int = 2, decumulate_cashflow: bool = False):
        set_identity(identity)
        self.db = db or DBManager()
        self.sec_dir = Path(sec_dir) if sec_dir is not None else Path(SEC_DIR)
        self.parse_extra_data = parse_extra_data
        self.forms = tuple(forms)
        self.max_concurrency = max_concurrency
        self.requests_per_second = requests_per_second
        self.batch_size = batch_size
        self.decumulate_cashflow = decumulate_cashflow

    # ------------------------------ entry ------------------------------ #

    def run(self, tickers, force: bool = False, min_date=None, progress: bool = True) -> dict:
        """Init or update EDGAR data for the given ticker(s).

        ``force`` refetches everything regardless of stored state;
        ``min_date`` caps how far back filings are pulled (init only).
        """
        tickers = [tickers.upper()] if isinstance(tickers, str) else [t.upper() for t in tickers]
        min_date = _to_date(min_date)
        jobs = [self._build_job(t, force, min_date) for t in tickers]
        logger.info('EDGAR run: %d ticker(s), forms=%s, extra_sections=%s',
                    len(jobs), self.forms, self.parse_extra_data)

        totals = {}
        for i in range(0, len(jobs), self.batch_size):
            batch = jobs[i:i + self.batch_size]
            results = asyncio.run(self._fetch_batch(batch, progress))
            for job, result in zip(batch, results):
                counts = self._process_result(job, result)
                for key, n in counts.items():
                    totals[key] = totals.get(key, 0) + n
        logger.info('EDGAR run done: %s', totals)
        return totals

    def _build_job(self, ticker: str, force: bool, min_date) -> _TickerJob:
        text_since = form4_since = min_date
        known = set()
        if not force:
            meta = self.db.get_ticker_meta([ticker])
            if not meta.empty and bool(meta['edgar_seeded'].fillna(False).iloc[0]):
                text_since = meta['last_filings_date'].iloc[0] or min_date
                form4_since = meta['last_form4_date'].iloc[0] or min_date
            known = set(self.db.get_filings_meta(ticker)['accession_number'])
            form4 = self.db.get_form4(ticker)
            if not form4.empty:
                known |= set(form4['accession_number'])
        return _TickerJob(ticker, text_since, form4_since, known)

    # --------------------------- fetch stages -------------------------- #

    async def _request(self, fn, *args):
        """Run one edgartools call in a worker thread, bounded by the shared
        concurrency semaphore and rate limiter."""
        async with self._sem:
            await self._limiter.wait()
            return await asyncio.to_thread(fn, *args)

    async def _fetch_batch(self, jobs: list, progress: bool) -> list:
        # loop-bound primitives must be created inside the running loop
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self._limiter = _RateLimiter(self.requests_per_second)

        results = {job.ticker: _TickerResult(job.ticker) for job in jobs}
        company_q, filings_q, obj_q, facts_q = (asyncio.Queue() for _ in range(4))
        for job in jobs:
            company_q.put_nowait(job)
        bar = tqdm(total=0, unit='filing', disable=not progress,
                   desc=f"EDGAR {'/'.join(job.ticker for job in jobs)}",
                   dynamic_ncols=True, leave=True)

        async def company_worker():
            while True:
                job = await company_q.get()
                try:
                    company = await self._request(Company, job.ticker)
                    results[job.ticker].company_name = company.name
                    results[job.ticker].cik = company.cik
                    filings_q.put_nowait((job, company))
                    facts_q.put_nowait((job, company))
                except Exception as e:
                    results[job.ticker].errors.append(f'company: {e}')
                    logger.error('%s: Company() failed: %s', job.ticker, e)
                finally:
                    company_q.task_done()

        async def filings_worker():
            while True:
                job, company = await filings_q.get()
                try:
                    filings = await self._request(self._list_filings, company, job)
                    bar.total = (bar.total or 0) + len(filings)

                    for filing in filings:
                        obj_q.put_nowait((job, filing))
                except Exception as e:
                    results[job.ticker].errors.append(f'get_filings: {e}')
                    logger.error('%s: get_filings failed: %s', job.ticker, e)
                finally:
                    filings_q.task_done()

        async def obj_worker():
            while True:
                job, filing = await obj_q.get()
                try:
                    payload = await self._request(self._fetch_and_reduce, filing)
                    results[job.ticker].filings.append(payload)
                except Exception as e:
                    results[job.ticker].errors.append(
                        f'{filing.form} {filing.accession_number}: {e}')
                    logger.warning('%s: %s %s failed: %s', job.ticker, filing.form,
                                   filing.accession_number, e)
                finally:
                    bar.update(1)
                    obj_q.task_done()

        async def facts_worker():
            while True:
                job, company = await facts_q.get()
                try:
                    results[job.ticker].facts = await self._request(self._fetch_facts_frame, company)
                except Exception as e:
                    results[job.ticker].errors.append(f'get_facts: {e}')
                    logger.error('%s: get_facts failed: %s', job.ticker, e)
                finally:
                    facts_q.task_done()

        workers = ([asyncio.create_task(company_worker()) for _ in range(2)]
                   + [asyncio.create_task(filings_worker()) for _ in range(2)]
                   + [asyncio.create_task(obj_worker()) for _ in range(self.max_concurrency)]
                   + [asyncio.create_task(facts_worker()) for _ in range(2)])
        await company_q.join()
        await filings_q.join()
        bar.refresh()
        await obj_q.join()
        await facts_q.join()
        bar.close()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        return [results[job.ticker] for job in jobs]

    def _list_filings(self, company, job: _TickerJob) -> list:
        """Stage 2: enumerate the filings that still need fetching."""
        kwargs = {'form': list(self.forms), 'amendments': False}
        cutoffs = [d for d in (job.text_since, job.form4_since) if d is not None]
        if len(cutoffs) == 2 or (cutoffs and job.text_since == job.form4_since):
            kwargs['filing_date'] = f'{min(cutoffs)}:'
        filings = company.get_filings(**kwargs)

        keep = []
        for filing in filings:
            if filing.accession_number in job.known_accessions:
                continue
            since = job.form4_since if filing.form == '4' else job.text_since
            if since is not None and filing.filing_date <= since:
                continue
            keep.append(filing)
        logger.info('%s: %d filing(s) to fetch (%d already stored)',
                    job.ticker, len(keep), len(job.known_accessions))
        return keep

    def _fetch_and_reduce(self, filing) -> dict:
        """Stage 3: download/parse one filing and reduce it to a small payload."""
        payload = {
            'form': filing.form,
            'accession': filing.accession_number,
            'filing_date': filing.filing_date,
            'period_of_report': _to_date(filing.period_of_report),
        }
        obj = filing.obj()
        if obj is None:
            raise ValueError('no data object for filing')

        if filing.form == '10-K':
            payload['sections'] = self._extract_sections(obj, TENK_SECTIONS)
        elif filing.form == '10-Q':
            payload['sections'] = self._extract_sections(obj, TENQ_SECTIONS)
        elif filing.form == '8-K':
            items = {}
            for item in (obj.items or []):
                try:
                    text = obj[item]
                except Exception:
                    text = None
                if text and text.strip():
                    items[item] = text
            payload['items'] = items
            payload['report_date'] = _to_date(obj.date_of_report) or filing.filing_date
        elif filing.form == '4':
            frames = []
            for attr, is_derivative in (('market_trades', False), ('derivative_trades', True)):
                try:
                    trades = getattr(obj, attr)
                except Exception:
                    trades = None
                if trades is not None and not isinstance(trades, pd.DataFrame):
                    trades = getattr(trades, 'data', None)  # unwrap edgartools DataHolder
                if trades is not None and len(trades):
                    frames.append((trades, is_derivative))
            payload['frames'] = frames
            payload['insider'] = obj.insider_name
            payload['position'] = obj.position
        return payload

    def _extract_sections(self, obj, inventory: dict) -> dict:
        sections = {}
        for key, (item, essential) in inventory.items():
            if not essential and not self.parse_extra_data:
                continue
            try:
                text = obj[item]
            except Exception:
                text = None
            if text and text.strip():
                sections[key] = text
        return sections

    @staticmethod
    def _fetch_facts_frame(company) -> pd.DataFrame:
        """Stage 4: all XBRL facts for the company in one frame."""
        facts = company.get_facts()
        if facts is None:
            return pd.DataFrame()
        return facts.query().to_dataframe()

    # --------------------------- processing ---------------------------- #

    def _process_result(self, job: _TickerJob, result: _TickerResult) -> dict:
        """Synchronous side: save section files, insert DB rows, update meta."""
        ticker = result.ticker
        counts = {'text_filings': 0, 'sections': 0, 'form4_rows': 0,
                  'income': 0, 'balance': 0, 'cashflow': 0, 'errors': len(result.errors)}
        if result.company_name is None and not result.filings:
            logger.error('%s: nothing fetched, skipping processing', ticker)
            return counts

        # fiscal-year anchors (10-K period ends) from this batch + the DB
        stored_tenk = self.db.get_filings_meta(ticker, '10-K')
        anchors = {_to_date(d) for d in stored_tenk['period_of_report']} if not stored_tenk.empty else set()
        anchors |= {p['period_of_report'] for p in result.filings
                    if p['form'] == '10-K' and p['period_of_report']}
        anchors = sorted(a for a in anchors if a)

        meta_rows, form4_payloads = [], []
        for payload in sorted(result.filings, key=lambda p: p['filing_date']):
            if payload['form'] in ('10-K', '10-Q'):
                meta_rows.append(self._process_text_filing(ticker, payload, anchors))
                counts['text_filings'] += 1
                counts['sections'] += len(payload['sections'])
            elif payload['form'] == '8-K':
                meta_rows.append(self._process_eightk(ticker, payload))
                counts['text_filings'] += 1
                counts['sections'] += len(payload['items'])
            elif payload['form'] == '4':
                form4_payloads.append(payload)

        if meta_rows:
            self.db.add_filings_meta(pd.DataFrame(meta_rows))
        if form4_payloads:
            form4_df = self._standardize_form4(form4_payloads, ticker)
            counts['form4_rows'] = self.db.add_form4(form4_df) if not form4_df.empty else 0

        if result.facts is not None and not result.facts.empty:
            statements = parse_financials(result.facts, ticker, self.decumulate_cashflow)
            inserted = self.db.add_financials(statements)
            counts.update(inserted)

        self._update_meta(result)
        logger.info('%s processed: %s', ticker, counts)
        return counts

    def _process_text_filing(self, ticker: str, payload: dict, anchors: list) -> dict:
        period = payload['period_of_report']
        if payload['form'] == '10-K':
            fiscal_year, fiscal_quarter = (period.year if period else None), 0
        else:
            fiscal_year, fiscal_quarter = self._fiscal_label(period, anchors)
        directory = self._save_filing(ticker, payload, payload['sections'])
        return self._meta_row(ticker, payload, fiscal_year, fiscal_quarter,
                              directory, list(payload['sections']))

    def _process_eightk(self, ticker: str, payload: dict) -> dict:
        files = {_slug(item): f"{payload['report_date']} | {item}\n\n{text}"
                 for item, text in payload['items'].items()}
        directory = self._save_filing(ticker, payload, files)
        return self._meta_row(ticker, payload, None, None, directory, list(payload['items']))

    def _save_filing(self, ticker: str, payload: dict, files: dict) -> Path:
        directory = self.sec_dir / ticker / payload['form'] / payload['accession']
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (directory / f'{name}.txt').write_text(text, encoding='utf-8')
        metadata = {key: payload[key] for key in
                    ('form', 'accession', 'filing_date', 'period_of_report')}
        metadata['sections'] = list(files)
        metadata['parse_extra_data'] = self.parse_extra_data
        (directory / 'metadata.json').write_text(json.dumps(metadata, default=str, indent=2))
        return directory

    def _meta_row(self, ticker, payload, fiscal_year, fiscal_quarter, directory, sections):
        return {
            'ticker': ticker, 'accession_number': payload['accession'],
            'form_type': payload['form'], 'fiscal_year': fiscal_year,
            'fiscal_quarter': fiscal_quarter, 'filing_date': payload['filing_date'],
            'period_of_report': payload['period_of_report'], 'disk_path': str(directory),
            'sections_parsed': json.dumps(sections), 'parse_extra_data': self.parse_extra_data,
        }

    @staticmethod
    def _fiscal_label(period, anchors: list):
        """Fiscal (year, quarter) of a 10-Q period using 10-K period ends as
        fiscal-year anchors; predicts the next anchor when the year is open."""
        if period is None:
            return None, None
        following = next((a for a in anchors if a >= period), None)
        if following is None and anchors:
            following = anchors[-1] + timedelta(days=365)
        if following is None:
            return period.year, (period.month - 1) // 3 + 1
        quarter = 4 - round((following - period).days / 91)
        return following.year, min(max(quarter, 1), 4)

    @staticmethod
    def _standardize_form4(payloads: list, ticker: str) -> pd.DataFrame:
        """Concat per-filing trade frames into one standardized DataFrame."""
        def column(frame, name):
            return frame[name] if name in frame.columns else pd.Series([None] * len(frame))

        rows = []
        for payload in payloads:
            for trades, is_derivative in payload['frames']:
                trades = trades.reset_index(drop=True)
                rows.append(pd.DataFrame({
                    'transaction_date': pd.to_datetime(column(trades, 'Date'), errors='coerce').dt.date,
                    'security': column(trades, 'Security'),
                    'transaction_code': column(trades, 'Code'),
                    'transaction_type': column(trades, 'TransactionType'),
                    'acquired_disposed': column(trades, 'AcquiredDisposed'),
                    'direct_indirect': column(trades, 'DirectIndirect'),
                    'shares': pd.to_numeric(column(trades, 'Shares'), errors='coerce'),
                    'price': pd.to_numeric(column(trades, 'Price'), errors='coerce'),
                    'shares_remaining': pd.to_numeric(column(trades, 'Remaining'), errors='coerce'),
                }).assign(is_derivative=is_derivative, accession_number=payload['accession'],
                          filing_date=payload['filing_date'], insider=payload['insider'],
                          position=payload['position']))
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, ignore_index=True)
        out['value'] = out['shares'] * out['price']
        out['seq'] = out.groupby('accession_number').cumcount()
        out['ticker'] = ticker
        return out[['ticker', 'accession_number', 'seq', 'filing_date', 'transaction_date',
                    'insider', 'position', 'security', 'transaction_code', 'transaction_type',
                    'acquired_disposed', 'direct_indirect', 'shares', 'price', 'value',
                    'shares_remaining', 'is_derivative']]

    def _update_meta(self, result: _TickerResult):
        # derive the last dates from what is actually stored, so resumed or
        # partially-skipped runs still leave ticker_meta accurate
        row = {key: value for key, value in self.db.last_edgar_dates(result.ticker).items()
               if value is not None}
        if result.company_name:
            row['name'] = result.company_name
        info = pd.DataFrame(row or {'name': None}, index=pd.Index([result.ticker], name='ticker'))
        self.db.upsert_ticker_meta(info, edgar_seeded=True)

    # ----------------------------- access ------------------------------ #

    def get_10k(self, ticker: str, year: int, obj: bool = False) -> dict:
        """Saved 10-K sections for a fiscal year -> {section_key: text}.
        With obj=True the live edgartools TenK is fetched into data['obj']."""
        return self._load_filing_dict(ticker, '10-K', year, None, obj)

    def get_10q(self, ticker: str, year: int, quarter: int, obj: bool = False) -> dict:
        """Saved 10-Q sections for a fiscal (year, quarter)."""
        return self._load_filing_dict(ticker, '10-Q', year, quarter, obj)

    def _load_filing_dict(self, ticker, form, year, quarter, with_obj) -> dict:
        meta = self.db.get_filings_meta(ticker.upper(), form)
        meta = meta[meta['fiscal_year'] == year]
        if quarter is not None:
            meta = meta[meta['fiscal_quarter'] == quarter]
        if meta.empty:
            raise FileNotFoundError(
                f'No {form} stored for {ticker} fiscal year {year}'
                + (f' Q{quarter}' if quarter is not None else ''))
        row = meta.iloc[0]  # newest filing first
        data = {path.stem: path.read_text(encoding='utf-8')
                for path in sorted(Path(row['disk_path']).glob('*.txt'))}
        if with_obj:
            data['obj'] = find(row['accession_number']).obj()
        return data
