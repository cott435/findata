"""EDGAR acquisition pipeline (edgartools).

One synchronous entry point -- ``EdgarPipeline.run(tickers)`` -- covering both
cold-start and incremental update (decided per ticker from ticker_meta).
Only HTTP runs async; orchestration, parsing and DB writes stay synchronous.

The fetch side is four queue-fed stages, mirroring the edgartools call chain:

    1. company queue:  ticker            -> Company()        (submissions request)
    2. filings queue:  company           -> get_filings()    (filing list, paged)
    3. obj queue:      filing            -> filing.obj()     (download + parse)
    4. facts queue:    company           -> get_facts()      (all XBRL facts, one request)

``mode='numeric'`` starts ONLY stages 1 and 4: no filing list is requested
and no filing is ever downloaded or parsed, so a ticker costs exactly two
requests and yields just the income/balance/cashflow tables. ``mode='full'``
(the default) starts all four. For Form 4 transactions without any
10-K/10-Q/8-K text, use ``mode='full', forms=('4',)`` -- that runs the
filings stages against Form 4 alone.

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
                'RevenueFromContractWithCustomerIncludingAssessedTax',
                'RevenuesNetOfInterestExpense', 'Revenues', 'SalesRevenueNet',
                'SalesRevenueGoodsNet', 'SalesRevenueServicesNet',
                # sector top lines: banks report interest + noninterest income
                # rather than "revenue"; REITs report rental revenue
                'InterestAndDividendIncomeOperating', 'RealEstateRevenueNet',
                'HealthCareOrganizationRevenue', 'RegulatedAndUnregulatedOperatingRevenue'],
    'cost_of_revenue': ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfGoodsSold',
                        'CostOfServices', 'CostOfSales',
                        # ex-D&A variants (AA and other capital-intensive issuers
                        # report only these -- D&A is a separate line for them)
                        'CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization',
                        'CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization',
                        'CostOfRevenueExcludingDepreciationDepletionAndAmortization'],
    'gross_profit': ['GrossProfit'],
    'operating_expenses': ['OperatingExpenses', 'CostsAndExpenses'],
    # total operating costs -- kept as its own item so operating_income can be
    # reconstructed (revenue - costs) for issuers that never tag it directly
    'costs_and_expenses': ['CostsAndExpenses', 'OperatingCostsAndExpenses'],
    'rnd_expense': ['ResearchAndDevelopmentExpense'],
    'operating_income': ['OperatingIncomeLoss'],
    'pretax_income': ['IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                      'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'],
    'income_tax': ['IncomeTaxExpenseBenefit'],
    'net_income': ['NetIncomeLoss', 'ProfitLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic'],
    # 'BasicAndDiluted' is one combined tag used by issuers whose two figures
    # are identical (loss-makers with no dilution, e.g. ABEO 2015-2021)
    'eps_basic': ['EarningsPerShareBasic', 'EarningsPerShareBasicAndDiluted'],
    'eps_diluted': ['EarningsPerShareDiluted', 'EarningsPerShareBasicAndDiluted'],
    'shares_basic': ['WeightedAverageNumberOfSharesOutstandingBasic',
                     'WeightedAverageNumberOfShareOutstandingBasicAndDiluted'],
    'shares_diluted': ['WeightedAverageNumberOfDilutedSharesOutstanding',
                       'WeightedAverageNumberOfShareOutstandingBasicAndDiluted'],
    'interest_expense': ['InterestExpense', 'InterestExpenseDebt',
                         'InterestAndDebtExpense', 'InterestExpenseNonoperating',
                         'InterestExpenseOther'],
}

# Weighted-average share counts are averages, not sums: the fiscal-year figure
# is the mean of the quarters, so a missing Q4 is 4*FY - (Q1+Q2+Q3) rather than
# FY - (Q1+Q2+Q3). EPS is left in the additive set -- annual EPS is the sum of
# the quarters up to share-count drift, which is the standard convention.
AVERAGE_ITEMS = ('shares_basic', 'shares_diluted')

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
    'short_term_debt': ['DebtCurrent', 'LongTermDebtCurrent', 'ShortTermBorrowings',
                        'LongTermDebtAndCapitalLeaseObligationsCurrent'],
    'shares_outstanding': ['CommonStockSharesOutstanding', 'EntityCommonStockSharesOutstanding'],
}

CASHFLOW_ITEMS = {
    'operating_cash_flow': ['NetCashProvidedByUsedInOperatingActivities',
                            'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'],
    'investing_cash_flow': ['NetCashProvidedByUsedInInvestingActivities',
                            'NetCashProvidedByUsedInInvestingActivitiesContinuingOperations'],
    'financing_cash_flow': ['NetCashProvidedByUsedInFinancingActivities',
                            'NetCashProvidedByUsedInFinancingActivitiesContinuingOperations'],
    'capex': ['PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets',
              'PaymentsForCapitalImprovements', 'PaymentsToAcquireMachineryAndEquipment',
              'PaymentsToAcquireRealEstate', 'PaymentsToAcquireOtherPropertyPlantAndEquipment'],
    'depreciation_amortization': ['DepreciationDepletionAndAmortization',
                                  'DepreciationAndAmortization',
                                  'DepreciationAmortizationAndAccretionNet', 'Depreciation'],
    'dividends_paid': ['PaymentsOfDividendsCommonStock', 'PaymentsOfDividends'],
    'stock_buybacks': ['PaymentsForRepurchaseOfCommonStock'],
}

QUARTER_DAYS = (70, 115)    # one fiscal quarter (13/14-week)
ANNUAL_DAYS = (330, 400)    # one fiscal year (52/53-week)

# 52/53-week fiscal years can end a few days either side of the month boundary
# (e.g. a "2022" year ending 2023-01-01); shifting back before reading the
# month keeps those on the right fiscal year
FY_EDGE_DAYS = 10

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


def _fiscal_year_end_month(duration: pd.DataFrame) -> int:
    """The issuer's fiscal-year-end month, from the mode of its annual periods."""
    annual = duration[duration['duration_days'].between(*ANNUAL_DAYS)]
    if annual.empty:
        return 12
    month = (annual['period_end'] - pd.Timedelta(days=FY_EDGE_DAYS)).dt.month.mode()
    return int(month.iloc[0]) if not month.empty else 12


def _fiscal_labels(period_end: pd.Series, fye_month: int):
    """(fiscal_year, fiscal_quarter) derived from the period's own end date.

    XBRL's ``fiscal_year``/``fiscal_period`` describe the FILING a fact was
    taken from, not the period it covers, so comparatives restated in a later
    filing arrive mislabeled -- AA's FY2015 revenue appears as fiscal_year
    2017 because it is a comparative column in the FY2017 10-K. Deriving the
    labels from the period itself is filing-independent, so the same period
    lands on the same fiscal year no matter which filing reported it.
    """
    shifted = period_end - pd.Timedelta(days=FY_EDGE_DAYS)
    year, month = shifted.dt.year, shifted.dt.month
    fiscal_year = year.where(month <= fye_month, year + 1)
    months_to_year_end = (fye_month - month) % 12
    quarter = (4 - (months_to_year_end / 3).round()).clip(1, 4)
    return fiscal_year.astype('Int64'), quarter.astype('Int64')


# ---------------------------------------------------------------------- #
# financial facts parsing (pure functions -- no I/O)
# ---------------------------------------------------------------------- #

def parse_financials(facts: pd.DataFrame, ticker: str) -> dict:
    """Parse one ticker's full facts frame into the three statement tables.

    Returns {'income': df, 'balance': df, 'cashflow': df} with columns
    matching income_data / balance_data / cashflow_data. Values are
    as-originally-reported: when a period shows up again in later filings
    (comparatives, restatements) the earliest filing wins.

    Fiscal year/quarter are derived from each period's own dates rather than
    XBRL's filing-scoped labels (see :func:`_fiscal_labels`), so a period
    reported only as a comparative in a later filing still lands on its true
    fiscal year. Cash flow is always de-cumulated into single quarters
    alongside the as-reported cumulative rows.
    """
    if facts is None or facts.empty:
        return {'income': pd.DataFrame(), 'balance': pd.DataFrame(), 'cashflow': pd.DataFrame()}

    needed = ['concept', 'numeric_value', 'period_start', 'period_end', 'period_type',
              'filing_date']
    df = facts.loc[facts['numeric_value'].notna(), needed].copy()
    df['tag'] = df['concept'].str.split(':').str[-1]
    df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
    df['period_start'] = pd.to_datetime(df['period_start'], errors='coerce')
    df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    df = df[df['period_end'].notna()]

    duration = df[df['period_type'] == 'duration'].copy()
    duration = duration[duration['period_start'].notna()]
    duration['duration_days'] = (duration['period_end'] - duration['period_start']).dt.days

    fye_month = _fiscal_year_end_month(duration)
    duration['fiscal_year'], duration['fiscal_quarter'] = _fiscal_labels(
        duration['period_end'], fye_month)
    instant = df[df['period_type'] == 'instant'].copy()
    instant['fiscal_year'], instant['fiscal_quarter'] = _fiscal_labels(
        instant['period_end'], fye_month)

    income = _parse_income(duration)
    balance = _parse_balance(instant)
    cashflow = _parse_cashflow(duration)
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
    as annual minus Q1-Q3 when the company never reports it standalone.

    Weighted-average share counts use the average relationship instead
    (4*FY - Q1-Q3); a non-positive result means the inputs disagree, so the
    quarter is left missing rather than stored as nonsense.
    """
    rows = _resolve(duration, INCOME_ITEMS, ['period_start', 'period_end'])
    if rows.empty:
        return pd.DataFrame()

    annual = rows[rows['duration_days'].between(*ANNUAL_DAYS)].copy()
    annual['fiscal_quarter'] = 0
    quarterly = rows[rows['duration_days'].between(*QUARTER_DAYS)].copy()

    derived = []
    for (item, fy), ann in annual.groupby(['item', 'fiscal_year']):
        ann = ann.iloc[0]
        qs = quarterly[(quarterly['item'] == item) & (quarterly['fiscal_year'] == fy)]
        present = set(qs['fiscal_quarter'])
        if not ({1, 2, 3} <= present) or 4 in present:
            continue
        q123 = qs[qs['fiscal_quarter'].isin([1, 2, 3])]['numeric_value'].sum()
        value = (4 * ann['numeric_value'] - q123 if item in AVERAGE_ITEMS
                 else ann['numeric_value'] - q123)
        if item in AVERAGE_ITEMS and not value > 0:
            continue
        derived.append({
            'item': item, 'fiscal_year': fy, 'fiscal_quarter': 4,
            'numeric_value': value,
            'period_start': qs['period_end'].max() + pd.Timedelta(days=1),
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
    rows['fiscal_quarter'] = rows['fiscal_quarter'].astype(int)
    rows['date'] = rows['period_end']
    rows = rows.drop_duplicates(subset=['item', 'date'], keep='first')
    return _finalize(rows, ['date', 'item', 'value', 'fiscal_year',
                            'fiscal_quarter', 'filed_date'])


def _parse_cashflow(duration: pd.DataFrame) -> pd.DataFrame:
    """Cash-flow windows: the as-reported cumulative rows PLUS the
    single-quarter rows differenced out of them.

    Issuers report cash flow cumulatively from the fiscal-year start (Q1,
    H1, 9M, FY), which cannot be summed into a trailing-twelve-month figure.
    Consecutive windows sharing a start are therefore always differenced into
    standalone quarters and stored as ``derived=True`` rows next to the
    originals -- ``derived`` is part of the table's primary key, so both
    coexist. The Q1 window is already a single quarter and needs no
    derived twin.
    """
    rows = _resolve(duration, CASHFLOW_ITEMS, ['period_start', 'period_end'])
    if rows.empty:
        return pd.DataFrame()
    rows = rows[rows['duration_days'].between(40, 400) & rows['fiscal_year'].notna()].copy()
    rows['derived'] = False

    derived = []
    for _, group in rows.groupby(['item', 'period_start']):
        group = group.sort_values('period_end')
        previous = None
        for _, row in group.iterrows():
            if previous is not None:
                derived.append({
                    'item': row['item'],
                    'numeric_value': row['numeric_value'] - previous['numeric_value'],
                    'period_start': previous['period_end'] + pd.Timedelta(days=1),
                    'period_end': row['period_end'],
                    'fiscal_year': row['fiscal_year'],
                    'fiscal_quarter': row['fiscal_quarter'],
                    'filing_date': row['filing_date'], 'derived': True,
                })
            previous = row

    out = pd.concat([rows, pd.DataFrame(derived)], ignore_index=True)
    out['fiscal_year'] = out['fiscal_year'].astype(int)
    out['fiscal_quarter'] = out['fiscal_quarter'].astype(int)
    out['duration_days'] = (out['period_end'] - out['period_start']).dt.days
    out = out.drop_duplicates(subset=['item', 'period_start', 'period_end', 'derived'],
                              keep='first')
    return _finalize(out, ['start_date', 'end_date', 'item', 'value', 'duration_days',
                           'fiscal_year', 'fiscal_quarter', 'filed_date', 'derived'])


# ---------------------------------------------------------------------- #
# market-wide filing index
# ---------------------------------------------------------------------- #

# forms that carry XBRL financial statements for a domestic filer. Foreign
# private issuers report on 20-F/40-F (annual) and 6-K (interim) instead --
# see filer_types(); their facts arrive in the ifrs-full taxonomy, which the
# us-gaap concept lists above do not resolve.
STATEMENT_FORMS = ('10-K', '10-Q', '10-K/A', '10-Q/A')
FOREIGN_ANNUAL_FORMS = ('20-F', '40-F')
FILER_TYPE_CACHE = 'filer_types.csv'


def _ensure_identity():
    """SEC requires a User-Agent; the pipeline sets it in __init__, but the
    index helpers below are usable without ever constructing one."""
    set_identity(EDGAR_IDENTITY)


def ticker_ciks(tickers=None) -> pd.Series:
    """ticker -> CIK, from the bundled offline mapping.

    Mapped this direction on purpose: a CIK can list several tickers (share
    classes, an ADR alongside its OTC line -- 1,449 of them do), so a
    CIK->ticker dict silently keeps one and drops the rest. BTI, for
    instance, shares CIK 1303523 with BTAFF.
    """
    from edgar import get_company_tickers
    _ensure_identity()
    frame = get_company_tickers().dropna(subset=['ticker', 'cik'])
    lookup = frame.drop_duplicates('ticker').set_index('ticker')['cik']
    if tickers is None:
        return lookup
    return lookup.reindex(sorted({t.upper() for t in tickers})).dropna().astype(int)


def latest_statement_filings(tickers=None, since=None,
                             forms: tuple = STATEMENT_FORMS,
                             ticker_cik: pd.Series = None) -> pd.Series:
    """Newest statement filing date per ticker, from the market-wide index.

    SEC publishes one index covering every filing by every company, so
    answering "who has reported since X?" for a whole universe is a single
    request -- as opposed to per-company endpoints, which need one request
    each and get rate-limited long before 800 of them finish.

    Returns a ticker-indexed Series of dates; tickers with no such filing in
    the window are absent. ``since`` defaults to 30 days back. Pass
    ``ticker_cik`` to join on CIKs already stored in ticker_meta -- those are
    the entities the facts were actually fetched under, so they stay correct
    through a ticker rename.
    """
    from edgar import get_filings
    _ensure_identity()
    since = _to_date(since) or (date.today() - timedelta(days=30))
    if ticker_cik is None or ticker_cik.empty:
        ticker_cik = ticker_ciks(tickers)
    if ticker_cik.empty:
        return pd.Series(dtype='object')

    filings = get_filings(filing_date=f'{since}:', form=list(forms))
    if filings is None:
        logger.warning('EDGAR filing index returned nothing for %s onward', since)
        return pd.Series(dtype='object')
    frame = filings.to_pandas()
    if frame.empty:
        return pd.Series(dtype='object')

    newest = frame.groupby('cik')['filing_date'].max()
    latest = ticker_cik.map(newest).dropna()
    if latest.empty:
        return pd.Series(dtype='object')
    latest = pd.to_datetime(latest).dt.date
    logger.info('EDGAR index since %s: %d filing(s), resolving %d of %d requested ticker(s)',
                since, len(frame), len(latest), len(ticker_cik))
    return latest


def filer_types(tickers=None, since=None, refresh: bool = False,
                max_age_days: int = 90) -> pd.Series:
    """Ticker -> 'domestic' (files 10-K) or 'foreign' (files 20-F/40-F).

    The distinction matters because foreign private issuers tag their facts
    in the ifrs-full taxonomy and report semi-annually, neither of which the
    us-gaap concept lists and quarter-length filters here handle -- so their
    statements come out largely empty. Classification is derived from which
    annual form each company actually files, cached under the data dir, and
    refreshed when older than ``max_age_days``.

    Tickers that filed no annual report in the window are simply absent,
    rather than guessed at.
    """
    from findata.configs import DATA_DIR
    from edgar import get_filings

    cache = Path(DATA_DIR) / FILER_TYPE_CACHE
    if cache.exists() and not refresh:
        stored = pd.read_csv(cache)
        checked = _to_date(stored['checked'].iloc[0]) if 'checked' in stored else None
        if checked and (date.today() - checked).days <= max_age_days:
            series = stored.set_index('ticker')['filer_type']
            return series if tickers is None else series.reindex(
                [t.upper() for t in tickers]).dropna()

    _ensure_identity()
    since = _to_date(since) or (date.today() - timedelta(days=730))
    ticker_cik = ticker_ciks(None)   # classify every known ticker, then cache
    frame = get_filings(filing_date=f'{since}:',
                        form=['10-K', *FOREIGN_ANNUAL_FORMS]).to_pandas()
    # a company filing both (a transition year) is treated as domestic: its
    # newer statements arrive in us-gaap, which is what the parser reads
    by_cik = frame.groupby('cik')['form'].apply(
        lambda forms: 'domestic' if '10-K' in set(forms) else 'foreign')
    kinds = ticker_cik.map(by_cik).dropna()
    kinds.index.name = 'ticker'
    kinds.name = 'filer_type'

    out = kinds.reset_index()
    out['checked'] = date.today().isoformat()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    logger.info('Classified %d ticker(s) by annual form -> %s (cached to %s)',
                len(kinds), kinds.value_counts().to_dict(), cache.name)
    return kinds if tickers is None else kinds.reindex(
        [t.upper() for t in tickers]).dropna()


# ---------------------------------------------------------------------- #
# completeness validation
# ---------------------------------------------------------------------- #

def validate_statements(statements: dict, ticker: str = '', grace_years: int = 2,
                        items: tuple = None) -> pd.DataFrame:
    """Fiscal years whose quarterly coverage is incomplete.

    One row per (statement, item, fiscal_year) holding fewer than four
    quarters. The first ``grace_years`` fiscal years of a ticker's history
    are skipped: early filings routinely lack a quarter or two before the
    issuer's XBRL tagging settles, which is expected rather than a defect.
    The newest fiscal year is skipped too -- it is simply still in progress.

    ``items`` restricts the check to specific item names (default: every
    item present). Returns an empty frame when everything checks out.
    """
    frames = []
    for name, frame in statements.items():
        if frame is None or frame.empty:
            continue
        counted = _quarter_counts(name, frame)
        if counted.empty:
            continue
        counted.insert(0, 'statement', name)
        frames.append(counted)
    if not frames:
        return pd.DataFrame(columns=['ticker', 'statement', 'item', 'fiscal_year',
                                     'n_quarters', 'missing_quarters'])

    out = pd.concat(frames, ignore_index=True)
    if items is not None:
        out = out[out['item'].isin(items)]
    if out.empty:
        return out.assign(ticker=ticker)

    out = out[out['fiscal_year'] < out['fiscal_year'].max()]  # year still in progress

    kept = []
    for _, group in out.groupby(['statement', 'item'], sort=False):
        # an item the issuer only ever tags annually is a reporting choice,
        # not a gap -- nothing to flag unless it reaches 4 quarters somewhere
        if group['n_quarters'].max() < 4:
            continue
        # grace runs from each item's OWN first year, so an item that starts
        # being reported late isn't flagged for its ramp-up
        kept.append(group[group['fiscal_year'] >= group['fiscal_year'].min() + grace_years])
    if not kept:
        return out.iloc[0:0].assign(ticker=ticker)

    out = pd.concat(kept)
    out = out[out['n_quarters'] < 4]
    out.insert(0, 'ticker', ticker)
    return out.sort_values(['statement', 'item', 'fiscal_year'], ignore_index=True)


def _quarter_counts(statement: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Quarters present per (item, fiscal_year) for one statement."""
    df = frame
    if statement == 'income':
        df = df[df['fiscal_quarter'].between(1, 4)]
    elif statement == 'cashflow':
        # only the single-quarter rows count: the cumulative originals would
        # double-count, and derived rows cover Q2-Q4 with Q1 already single
        df = df[df['duration_days'] <= QUARTER_DAYS[1]]
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(['item', 'fiscal_year'])['fiscal_quarter']
    counted = grouped.nunique().rename('n_quarters').reset_index()
    present = grouped.agg(lambda s: sorted(set(s)))
    counted['missing_quarters'] = [
        [q for q in (1, 2, 3, 4) if q not in got] for got in present.values]
    return counted


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
                 batch_size: int = 5, mode: str = 'full',
                 validate: bool = True, grace_years: int = 2):
        """``mode='numeric'`` fetches ONLY the XBRL company facts: one
        submissions request plus one facts request per ticker. The filings
        stages never start, so no filing list is enumerated, nothing is
        downloaded or parsed, and neither filing text nor Form 4 rows are
        written. 'full' (default) runs both streams; for Form 4 without any
        text filings use mode='full' with forms=('4',)."""
        if mode not in ('full', 'numeric'):
            raise ValueError("mode must be 'full' or 'numeric'")
        set_identity(identity)
        self.db = db or DBManager()
        self.sec_dir = Path(sec_dir) if sec_dir is not None else Path(SEC_DIR)
        self.parse_extra_data = parse_extra_data
        self.mode = mode
        self.fetch_filings = mode == 'full'
        self.forms = tuple(forms) if self.fetch_filings else ()
        self.max_concurrency = max_concurrency
        self.requests_per_second = requests_per_second
        self.batch_size = batch_size
        self.validate = validate
        self.grace_years = grace_years

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
        # every field below only feeds _list_filings, so numeric mode skips
        # the three DB lookups entirely
        if not force and self.fetch_filings:
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
        bar = tqdm(total=0 if self.fetch_filings else len(jobs),
                   unit='filing' if self.fetch_filings else 'ticker',
                   disable=not progress,
                   desc=f"EDGAR {'/'.join(job.ticker for job in jobs)}",
                   dynamic_ncols=True, leave=True)

        async def company_worker():
            while True:
                job = await company_q.get()
                try:
                    company = await self._request(Company, job.ticker)
                    results[job.ticker].company_name = company.name
                    results[job.ticker].cik = company.cik
                    if self.fetch_filings:
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
                    if not self.fetch_filings:
                        bar.update(1)
                    facts_q.task_done()

        # numeric mode never starts the filings/obj workers, so nothing is
        # listed, downloaded or parsed -- the facts request is the only extra
        workers = ([asyncio.create_task(company_worker()) for _ in range(2)]
                   + [asyncio.create_task(facts_worker()) for _ in range(2)])
        if self.fetch_filings:
            workers += ([asyncio.create_task(filings_worker()) for _ in range(2)]
                        + [asyncio.create_task(obj_worker()) for _ in range(self.max_concurrency)])
        await company_q.join()
        if self.fetch_filings:
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

        if result.filings:  # numeric mode fetches none, so this whole side is skipped
            # fiscal-year anchors (10-K period ends) from this batch + the DB
            stored_tenk = self.db.get_filings_meta(ticker, '10-K')
            anchors = ({_to_date(d) for d in stored_tenk['period_of_report']}
                       if not stored_tenk.empty else set())
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
            statements = parse_financials(result.facts, ticker)
            # replace, not append: the facts request returns the full history,
            # so the parse is authoritative and supersedes anything stored
            inserted = self.db.replace_financials(ticker, statements)
            counts.update(inserted)
            if self.validate:
                gaps = validate_statements(statements, ticker, self.grace_years)
                counts['incomplete_years'] = len(gaps)
                if not gaps.empty:
                    summary = (gaps.groupby(['statement', 'item'])['fiscal_year']
                               .agg(lambda s: sorted(s)).to_dict())
                    logger.warning('%s: incomplete quarterly coverage for %d (item, year) '
                                   'pair(s): %s', ticker, len(gaps), summary)

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
        if result.cik is not None:
            row['cik'] = int(result.cik)
        if result.facts is not None and not result.facts.empty:
            # when the facts were last pulled, so daily runs can skip tickers
            # whose financials cannot have changed (issuers file quarterly)
            row['last_facts_date'] = date.today()
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
