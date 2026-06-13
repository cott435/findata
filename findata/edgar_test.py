"""
edgar_financials.py
-------------------
Fetches company financials from SEC EDGAR via edgartools and returns two DataFrames:

  quarterly_df  — MultiIndex (year, quarter), columns = financial metrics
  annual_df     — Index = year,               columns = financial metrics

Async design:
  Stage 1: Company() calls run concurrently across tickers
  Stage 2: get_facts() calls run concurrently (one per company, one HTTP request each)
  Stage 3: All parsing / DataFrame assembly is CPU-only — no more network calls

Usage:
  quarterly, annual = asyncio.run(build_financials(["AAPL", "MSFT"]))
"""

import asyncio
import logging
from typing import Optional
import numpy as np
import pandas as pd
from edgar import Company, set_identity
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


log = logging.getLogger(__name__)

TENK_SECTIONS: List[Tuple[str, str, List[str], int, bool, Tuple[str, ...]]] = [
    # --- essential (parse_extra_data=False still saves these) ---
    ("Item 1", "business", ["Item 1"], 9, True, ("business",)),
    ("Item 1A", "risk_factors", ["Item 1A"], 8, True, ("risk_factors",)),
    ("Item 3", "legal_proceedings", ["Item 3"], 7, True, ()),
    ("Item 7", "mda", ["Item 7"], 10, True, ("management_discussion",)),
    ("Item 7A", "market_risk", ["Item 7A"], 7, True, ()),
    ("Item 8", "financial_statements", ["Item 8"], 10, True, ()),
    # --- extra (only when parse_extra_data=True) ---
    ("Item 1C", "cybersecurity", ["Item 1C"], 6, False, ()),
    ("Item 2", "properties", ["Item 2"], 4, False, ()),
    ("Item 9A", "controls_and_procedures", ["Item 9A"], 6, False, ()),
    ("Item 10", "directors_and_governance", ["Item 10"], 5, False, ("directors_officers_and_governance",)),
    ("Item 11", "executive_compensation", ["Item 11"], 5, False, ()),
    ("Item 12", "security_ownership", ["Item 12"], 4, False, ()),
    ("Item 13", "related_party_transactions", ["Item 13"], 5, False, ()),
    ("Item 14", "accountant_fees", ["Item 14"], 3, False, ()),
    ("Item 15", "exhibits_index", ["Item 15"], 4, False, ()),
]

TENQ_SECTIONS: List[Tuple[str, str, List[str], int, bool, Tuple[str, ...]]] = [
    # --- essential ---
    ("Part I, Item 1", "financial_statements", ["Part I, Item 1", "Part I Item 1"], 10, True, ()),
    ("Part I, Item 2", "mda", ["Part I, Item 2", "Part I Item 2"], 10, True, ("management_discussion",)),
    ("Part II, Item 1", "legal_proceedings", ["Part II, Item 1", "Part II Item 1"], 7, True, ()),
    # --- extra ---
    ("Part I, Item 3", "market_risk_changes", ["Part I, Item 3", "Part I Item 3"], 5, False, ()),
    ("Part I, Item 4", "controls_changes", ["Part I, Item 4", "Part I Item 4"], 5, False, ()),
    ("Part II, Item 1A", "risk_factor_changes", ["Part II, Item 1A", "Part II Item 1A"], 6, False, ()),
    ("Part II, Item 2", "share_repurchases", ["Part II, Item 2", "Part II Item 2"], 5, False, ()),
    ("Part II, Item 5", "other_information", ["Part II, Item 5", "Part II Item 5"], 4, False, ()),
]

# Map a filing form to its on-disk subfolder.
FORM_FOLDER = {"10-K": "10k", "10-Q": "10q", "8-K": "8k", '4': 'form4'}

# SEC requires an identity header on every request. Set your own name/email.
IDENTITY = "Connor ctt7729@gmail.com"



def _extract_concept(facts, concept: str, concept_keys=None) -> pd.DataFrame:
    """
    Query a single concept from an EntityFacts object and return a deduplicated
    Series indexed by (fiscal_year, fiscal_period).

    Deduplication strategy: when the same (fiscal_year, fiscal_period) appears
    more than once (e.g., quarterly data re-stated in a 10-K), prefer the row
    sourced from the target form_type, then take the latest filing_date.
    """
    df = (
        facts.query()
        .by_concept(concept)                    # standardized fuzzy match
        .to_dataframe()
    )

    df['period_start'] = pd.to_datetime(df['period_start'])
    df['period_end'] = pd.to_datetime(df['period_end'])
    df['duration'] = (df['period_end'] - df['period_start']).dt.days
    df["fiscal_year"] = (df["period_end"]- pd.Timedelta(days=10)).dt.year
    df["fiscal_period"] = (df["period_end"]- pd.Timedelta(days=10)).dt.quarter  # use 10 day buffer

    # Target categories
    targets = np.array([90, 180, 270, 360])

    # Find nearest target
    nearest = targets[np.abs(df['duration'].values[:, None] - targets).argmin(axis=1)]

    # Assign category only if within 10 days
    df['duration_category'] = np.where(
        np.abs(df['duration'] - nearest) <= 14,
        nearest,
        np.nan
    )

    df['filing_date'] = pd.to_datetime(df['filing_date'])
    #TODO determine if ascending is true or false. If true, accession number is correct, if false, most recent value pulled (best if value corrected in future filing)
    df = (
        df.sort_values("filing_date", ascending=True)
        .drop_duplicates(subset=["fiscal_year", "fiscal_period", "duration_category", "concept"], keep="first")
    )

    df = df.sort_values(by=["fiscal_year", "fiscal_period"], ascending=True)
    df["concept_key"] = df["concept"].str.lower().str.split(":").str[-1]

    if concept_keys:
        concept_keys = [k.lower() for k in concept_keys]
        df = df[df["concept_key"].isin(concept_keys)]

    return df

def _dedup(df):
    return (
        df.sort_values("numeric_value", ascending=False)
        .drop_duplicates(subset=["fiscal_year", "fiscal_period"], keep="first")
        .sort_values(by=["fiscal_year", "fiscal_period"], ascending=True)
    )

def extract_quarterly_and_annual(df: pd.DataFrame) -> tuple:
    q = df[df["duration_category"] == 90]
    a = df[df["duration_category"] == 360]
    return q, a

def base_get(facts, concept, keys, dedup=False, split=True):
    keys = [k.lower() for k in keys]
    raw_data = _extract_concept(facts, concept)

    # Filter to only matching keys
    data = raw_data[raw_data["concept_key"].isin(keys)] if keys else raw_data
    q = data[data["duration_category"] == 90]
    a = data[data["duration_category"] == 360]
    if not dedup:
        return q, a

    def dedup(df):
        return (
            df.sort_values("numeric_value", ascending=False)
              .drop_duplicates(subset=["fiscal_year", "fiscal_period"], keep="first")
              .sort_values(by=["fiscal_year", "fiscal_period"], ascending=True)
        )
    q = dedup(q)
    a = dedup(a)
    # TODO: make quarterly rectifier to calculate missing quarters with annual sum
    return q, a
"Balance Sheet: revenue (priority search), income (net and comprehensive), eps, operating expense (appl splits by r&d and good&services)"
""
def get_revenue(facts) -> tuple:
    keys = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "RevenuesNetOfInterestExpense", "SalesRevenueNet", "TotalRevenues", "NetSales"]
    raw_data = _extract_concept(facts, 'revenue', concept_keys=keys)
    q, a = extract_quarterly_and_annual(raw_data)
    return _dedup(q), _dedup(a)

def get_net_income(facts) -> tuple:
    keys = ['NetIncomeLoss', 'ProfitLoss', 'NetIncome', 'NetEarnings', 'NetIncomeLossAttributableToParent']
    raw_data = _extract_concept(facts, 'income', concept_keys=keys)
    q, a = extract_quarterly_and_annual(raw_data)
    return _dedup(q), _dedup(a)

def get_eps(facts) -> tuple:
    keys = ['EarningsPerShareBasic', 'EarningsPerShareDiluted']
    raw_data = _extract_concept(facts, 'earnings', concept_keys=keys)
    q, a = extract_quarterly_and_annual(raw_data)
    return q, a

def get_total_liabilities(facts) -> pd.DataFrame:
    keys = ['Liabilities', 'LiabilitiesCurrent', 'TotalLiabilities', 'LiabilitiesAndStockholdersEquity']
    data = _extract_concept(facts, 'liabilities', concept_keys=keys)
    return data

def get_total_assets(facts) -> pd.DataFrame:
    keys=['Assets', 'TotalAssets', 'AssetsCurrent']
    data = _extract_concept(facts, 'assets', concept_keys=keys)
    return data

def get_financials(company):
    facts = company.get_facts()
    data = [fn(facts) for fn in [get_eps, get_revenue, get_net_income, get_total_assets, get_total_liabilities]]
    return data

def get_10k(filings):
    data = filings.filter('10-K')
    obj = data[0].obj()
    data = {item[1]: obj[item[0]] for item in TENK_SECTIONS if item[4]}
    return data

def get_10q(filings):
    data = filings.filter('10-Q')
    obj = data[0].obj()
    data = {item[1]: obj[item[0]] for item in TENQ_SECTIONS if item[4]}

    return data

def get_8k(filings):
    data = filings.filter('8-K')
    obj = data[10].obj()
    return data

def get_form4(filings):
    all_dfs = []
    all_objs = []
    all_none=[]
    data = filings.filter('4')
    for d in data[:100]:
        obj = d.obj()
        df = obj.market_trades
        if df is not None:
            df['Name'] = obj.reporting_owners[0].name
            df['Position'] = obj.reporting_owners[0].position
            all_dfs.append(df)
            all_objs.append(obj)
        else:
            all_none.append(obj)
    return pd.concat(all_dfs).reset_index(drop=True)

def get_filings(company):
    pass



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_identity(IDENTITY)
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    ticker = 'AAPL'
    company = Company(ticker)
    filings = company.get_filings()
    tenk = get_10k(filings)


    TICKERS = ["AAPL", "PNC", "SOFI", "MRNA", "VZ"]
    companies = [Company(ticker) for ticker in TICKERS]
    facts = [company.get_facts() for company in companies]

    d = [fn(facts[0]) for fn in [get_eps, get_revenue, get_net_income, get_total_assets, get_total_liabilities]]


    datas = [get_revenue(fact) for fact in facts]
    wb = Workbook()
    for ticker, data in zip(TICKERS, datas):
        data = data[0] if isinstance(data, tuple) else data
        wb.create_sheet(ticker)
        sheet = wb[ticker]
        data = data[data['fiscal_year']==2024]

        rows = dataframe_to_rows(data)

        for r_idx, row in enumerate(rows, 1):
            for c_idx, value in enumerate(row, 1):
                sheet.cell(row=r_idx, column=c_idx, value=value)

    wb.save("financials.xlsx")
