"""
sec_pipeline.py
===============

An async, rate-limited ingestion pipeline for SEC filings built on `edgartools`.

It does two jobs:

1. **Numerical database** -- builds quarterly balance-sheet / income / cash-flow
   DataFrames (one row per concept per fiscal quarter) that concat across years
   so you can watch any line item move over time. Quarterly figures for flow
   statements (income, cash flow) are derived from the cumulative XBRL facts via
   year-to-date differencing, which is also how the "missing" Q4 (only reported
   annually in the 10-K) is recovered.

2. **RAG corpus** -- saves the high-signal text sections of every 10-K / 10-Q
   (per the ingestion scoring sheet) as .txt files, plus each filing's financial
   statements as .csv, so a downstream retriever can index them.

Layout on disk (rooted at `data_dir`)::

    <data_dir>/<TICKER>/
        10k/<filing_date>_<accession>/
            meta.json
            item_07_mda.txt
            ...
            income_statement.csv
            balance_sheet.csv
            cash_flow.csv
        10q/<filing_date>_<accession>/...
        8k/<filing_date>_<accession>/...
        financials/
            income_statement_quarterly.csv     <- concatenated numeric DB
            balance_sheet_quarterly.csv
            cash_flow_quarterly.csv
            income_statement_wide.csv           <- concept x period pivot
            ...
            facts_raw.csv                        <- full fact dump (audit trail)

Quick start::

    import asyncio
    from sec_pipeline import SECPipeline

    pipe = SECPipeline("./data", identity="Jane Doe jane@example.com")
    asyncio.run(pipe.init_ticker("AAPL", min_date="2010-01-01"))

    tenk   = pipe.get_latest_10k("AAPL")
    tenq   = pipe.get_latest_10q("AAPL")
    k2018  = pipe.get_10k_for_year("AAPL", 2018)
    fins   = pipe.get_financials("AAPL", years=5)   # {"balance_sheet": df, ...}

Note: hitting SEC EDGAR requires network access and a valid identity string
(SEC mandates a descriptive User-Agent: "Name email@domain"). Set it in the
constructor or via the EDGAR_IDENTITY environment variable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger("sec_pipeline")


# --------------------------------------------------------------------------- #
# Section scoring sheet (from the ingestion reference doc)
# --------------------------------------------------------------------------- #
# Each entry: (item_id, human_name, [candidate lookup keys], score, essential,
#              (named-attribute fallbacks on the report object))
# `essential` items are always saved; the rest are saved only when
# parse_extra_data=True. Items rated 1-2 (boilerplate) are simply omitted.

TENK_SECTIONS: List[Tuple[str, str, List[str], int, bool, Tuple[str, ...]]] = [
    # --- essential (parse_extra_data=False still saves these) ---
    ("Item 1",  "business",                  ["Item 1"],  9, True,  ("business",)),
    ("Item 1A", "risk_factors",              ["Item 1A"], 8, True,  ("risk_factors",)),
    ("Item 3",  "legal_proceedings",         ["Item 3"],  7, True,  ()),
    ("Item 7",  "mda",                       ["Item 7"],  10, True, ("management_discussion",)),
    ("Item 7A", "market_risk",               ["Item 7A"], 7, True,  ()),
    ("Item 8",  "financial_statements",      ["Item 8"],  10, True, ()),
    # --- extra (only when parse_extra_data=True) ---
    ("Item 1C", "cybersecurity",             ["Item 1C"], 6, False, ()),
    ("Item 2",  "properties",                ["Item 2"],  4, False, ()),
    ("Item 9A", "controls_and_procedures",   ["Item 9A"], 6, False, ()),
    ("Item 10", "directors_and_governance",  ["Item 10"], 5, False, ("directors_officers_and_governance",)),
    ("Item 11", "executive_compensation",    ["Item 11"], 5, False, ()),
    ("Item 12", "security_ownership",         ["Item 12"], 4, False, ()),
    ("Item 13", "related_party_transactions",["Item 13"], 5, False, ()),
    ("Item 14", "accountant_fees",           ["Item 14"], 3, False, ()),
    ("Item 15", "exhibits_index",            ["Item 15"], 4, False, ()),
]

TENQ_SECTIONS: List[Tuple[str, str, List[str], int, bool, Tuple[str, ...]]] = [
    # --- essential ---
    ("Part I, Item 1",  "financial_statements", ["Part I, Item 1", "Part I Item 1"],  10, True, ()),
    ("Part I, Item 2",  "mda",                  ["Part I, Item 2", "Part I Item 2"],  10, True, ("management_discussion",)),
    ("Part II, Item 1", "legal_proceedings",    ["Part II, Item 1", "Part II Item 1"], 7, True, ()),
    # --- extra ---
    ("Part I, Item 3",  "market_risk_changes",  ["Part I, Item 3", "Part I Item 3"],   5, False, ()),
    ("Part I, Item 4",  "controls_changes",     ["Part I, Item 4", "Part I Item 4"],   5, False, ()),
    ("Part II, Item 1A","risk_factor_changes",  ["Part II, Item 1A", "Part II Item 1A"],6, False, ()),
    ("Part II, Item 2", "share_repurchases",    ["Part II, Item 2", "Part II Item 2"],  5, False, ()),
    ("Part II, Item 5", "other_information",     ["Part II, Item 5", "Part II Item 5"],  4, False, ()),
]

# Map a filing form to its on-disk subfolder.
FORM_FOLDER = {"10-K": "10k", "10-Q": "10q", "8-K": "8k"}

# Concept-name heuristics used only as a fallback when edgartools' own
# statement_type classification is missing (it is admittedly incomplete).
_CF_PATTERNS = (
    "cashprovidedbyusedin", "netcashprovided", "netcashused", "paymentsto",
    "paymentsfor", "paymentsofdividends", "proceedsfrom", "repaymentsof",
    "cashandcashequivalentsperiodincreasedecrease", "cashcashequivalents",
    "depreciationdepletionandamortization", "sharebasedcompensation",
)
_IS_PATTERNS = (
    "revenue", "costof", "grossprofit", "operatingexpenses", "operatingincome",
    "researchanddevelopment", "sellinggeneralandadministrative", "interestexpense",
    "incometaxexpense", "netincome", "earningspershare", "comprehensiveincome",
    "costsandexpenses", "nonoperating",
)
_BS_PATTERNS = (
    "assets", "liabilities", "stockholdersequity", "cashandcashequivalentsat",
    "accountsreceivable", "inventory", "propertyplantandequipment", "goodwill",
    "retainedearnings", "longtermdebt", "accountspayable", "commonstock",
    "intangibleassets", "deferred", "marketablesecurities",
)


@dataclass
class LoadedFiling:
    """A filing reconstructed from disk (or freshly downloaded then persisted)."""
    ticker: str
    form: str
    filing_date: str
    accession: str
    period_of_report: Optional[str]
    fiscal_year: Optional[int]
    path: Path
    sections: Dict[str, str] = field(default_factory=dict)          # human_name -> text
    statements: Dict[str, pd.DataFrame] = field(default_factory=dict)  # name -> df

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        secs = ", ".join(self.sections) or "none"
        stmts = ", ".join(self.statements) or "none"
        return (f"<LoadedFiling {self.ticker} {self.form} filed={self.filing_date} "
                f"acc={self.accession} sections=[{secs}] statements=[{stmts}]>")


# --------------------------------------------------------------------------- #
# Async rate limiter (token-bucket style start-time spacing)
# --------------------------------------------------------------------------- #
class AsyncRateLimiter:
    """Ensures scheduled operations start no faster than `rate_per_sec`.

    SEC asks clients to stay at/under 10 requests/second. Because a single
    high-level edgartools call can fan out into several HTTP requests, keep the
    configured rate comfortably below 10.
    """

    def __init__(self, rate_per_sec: float):
        self._min_interval = (1.0 / rate_per_sec) if rate_per_sec and rate_per_sec > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            wait = self._next_time - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_time = max(now, self._next_time) + self._min_interval


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
class SECPipeline:
    def __init__(
        self,
        data_dir: str | os.PathLike,
        identity: Optional[str] = None,
        *,
        parse_extra_data: bool = False,
        requests_per_second: float = 6.0,
        max_concurrency: int = 5,
        save_full_text: bool = False,
        use_local_storage: bool = False,
        log_level: int = logging.INFO,
    ):
        """
        Parameters
        ----------
        data_dir : base directory; per-ticker folders are created beneath it.
        identity : SEC-required identity string "Name email@domain". Falls back
                   to the EDGAR_IDENTITY env var.
        parse_extra_data : when False only the "essential" sections are saved;
                   when True the lower-value extra sections are saved too.
        requests_per_second : ceiling on how fast operations are scheduled.
        max_concurrency : max filings processed in parallel.
        save_full_text : also dump the full filing text (filing.text()) per
                   filing. Off by default to honour selective ingestion.
        use_local_storage : enable edgartools' bulk local cache.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parse_extra_data = parse_extra_data
        self.save_full_text = save_full_text

        logging.basicConfig(level=log_level,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logger.setLevel(log_level)

        # Import edgar lazily so the module imports even without it installed.
        try:
            import edgar  # noqa: F401
            from edgar import Company, set_identity
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "edgartools is required: pip install edgartools"
            ) from exc
        self._edgar = edgar
        self._Company = Company

        identity = identity or os.environ.get("EDGAR_IDENTITY")
        if not identity:
            raise ValueError(
                "SEC requires an identity string 'Name email@domain'. "
                "Pass identity=... or set EDGAR_IDENTITY."
            )
        set_identity(identity)
        if use_local_storage:
            self._edgar.use_local_storage()

        self._limiter = AsyncRateLimiter(requests_per_second)
        self._sem = asyncio.Semaphore(max_concurrency)

    # ------------------------------------------------------------------ #
    # async plumbing
    # ------------------------------------------------------------------ #
    async def _run(self, fn, *args, **kwargs):
        """Run a blocking edgar call off-thread under the rate limiter + semaphore."""
        async with self._sem:
            await self._limiter.acquire()
            return await asyncio.to_thread(fn, *args, **kwargs)

    @staticmethod
    def _maybe_run(coro):
        """Run a coroutine to completion from sync code (raises if a loop is live)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "Call the async method directly; a sync wrapper was invoked inside a "
            "running event loop."
        )

    # ------------------------------------------------------------------ #
    # path helpers
    # ------------------------------------------------------------------ #
    def _ticker_dir(self, ticker: str) -> Path:
        return self.data_dir / ticker.upper()

    def _form_dir(self, ticker: str, form: str) -> Path:
        return self._ticker_dir(ticker) / FORM_FOLDER[self._base_form(form)]

    @staticmethod
    def _base_form(form: str) -> str:
        return form.split("/")[0].upper()  # "10-K/A" -> "10-K"

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    @staticmethod
    def _fmt_date(value) -> str:
        if value is None:
            return ""
        return str(value)[:10]

    def _filing_dir(self, ticker: str, form: str, filing_date, accession: str) -> Path:
        acc = accession.replace(":", "-")
        return self._form_dir(ticker, form) / f"{self._fmt_date(filing_date)}_{acc}"

    # ================================================================== #
    # 1. INGEST
    # ================================================================== #
    async def init_ticker(
        self,
        ticker: str,
        min_date: str = "2010-01-01",
        force: bool = False,
        forms: Sequence[str] = ("10-K", "10-Q", "8-K"),
        build_numeric: bool = True,
    ) -> Dict[str, int]:
        """Download every `forms` filing for `ticker` back to `min_date`.

        Existing filing folders are skipped unless `force=True`. Returns a count
        of saved/skipped/failed per form.
        """
        ticker = ticker.upper()
        self._ticker_dir(ticker).mkdir(parents=True, exist_ok=True)
        logger.info("init_ticker(%s) since %s (force=%s)", ticker, min_date, force)

        company = await self._run(self._Company, ticker)
        if getattr(company, "not_found", False):
            raise ValueError(f"Company not found for ticker {ticker!r}")

        results: Dict[str, int] = {}
        for form in forms:
            filings = await self._run(
                company.get_filings, form=form, filing_date=f"{min_date}:"
            )
            filing_list = list(filings) if filings is not None else []
            logger.info("  %s: %d filings", form, len(filing_list))

            tasks = [
                self._process_filing(ticker, form, f, force)
                for f in filing_list
            ]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            saved = sum(1 for o in outcomes if o == "saved")
            skipped = sum(1 for o in outcomes if o == "skipped")
            failed = sum(1 for o in outcomes if isinstance(o, Exception) or o == "failed")
            for o in outcomes:
                if isinstance(o, Exception):
                    logger.warning("  filing error: %s", o)
            results[form] = saved
            logger.info("  %s done: %d saved, %d skipped, %d failed",
                        form, saved, skipped, failed)

        if build_numeric:
            try:
                await self.build_financials(ticker, min_date=min_date,
                                            company=company, force=force)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("  numeric DB build failed: %s", exc)

        return results

    async def _process_filing(self, ticker: str, form: str, filing, force: bool) -> str:
        """Parse one filing and persist its sections + statement CSVs."""
        filing_date = getattr(filing, "filing_date", None)
        accession = getattr(filing, "accession_number", "") or getattr(filing, "accession", "")
        out_dir = self._filing_dir(ticker, form, filing_date, accession)

        if out_dir.exists() and not force:
            return "skipped"

        try:
            obj = await self._run(filing.obj)
        except Exception as exc:  # unsupported / malformed filing
            logger.warning("    obj() failed for %s %s: %s", form, accession, exc)
            return "failed"

        out_dir.mkdir(parents=True, exist_ok=True)
        base = self._base_form(form)

        period = getattr(obj, "period_of_report", None) or getattr(filing, "period_of_report", None)
        fiscal_year = self._fiscal_year_of(obj, period)

        meta = {
            "ticker": ticker,
            "form": form,
            "filing_date": self._fmt_date(filing_date),
            "accession": accession,
            "period_of_report": self._fmt_date(period) if period else None,
            "fiscal_year": fiscal_year,
            "sections_saved": [],
        }

        # --- text sections (10-K / 10-Q) ---
        if base in ("10-K", "10-Q"):
            spec = TENK_SECTIONS if base == "10-K" else TENQ_SECTIONS
            for item_id, name, keys, score, essential, named in spec:
                if not essential and not self.parse_extra_data:
                    continue
                text = self._get_section_text(obj, keys, named)
                if not text:
                    continue
                fname = f"{self._slug(item_id)}_{name}.txt"
                header = f"# {item_id} - {name} (retrieval score {score}/10)\n# {ticker} {form} filed {meta['filing_date']}\n\n"
                (out_dir / fname).write_text(header + text, encoding="utf-8")
                meta["sections_saved"].append({"item": item_id, "name": name,
                                                "score": score, "file": fname})
            # statement CSVs
            self._save_statements(obj, out_dir)

        # --- 8-K: event-driven, short; save items + full text + press releases ---
        elif base == "8-K":
            items = getattr(obj, "items", None)
            if items is not None:
                (out_dir / "items.txt").write_text(str(items), encoding="utf-8")
                meta["items"] = self._jsonable(items)
            full = self._safe_text(obj)
            if full:
                (out_dir / "full.txt").write_text(full, encoding="utf-8")
            for i, pr in enumerate(getattr(obj, "press_releases", []) or [], start=1):
                pr_text = self._safe_text(pr)
                if pr_text:
                    (out_dir / f"press_release_{i}.txt").write_text(pr_text, encoding="utf-8")

        if self.save_full_text and base in ("10-K", "10-Q"):
            full = self._safe_text(obj)
            if full:
                (out_dir / "full.txt").write_text(full, encoding="utf-8")

        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return "saved"

    # ------------------------------------------------------------------ #
    # extraction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_section_text(report, keys: Iterable[str], named: Iterable[str]) -> Optional[str]:
        """Robustly pull a section's text, tolerating None / KeyError / combined headings."""
        for key in keys:
            try:
                val = report[key]
            except Exception:
                val = None
            if val:
                s = str(val).strip()
                if s and s.lower() != "none":
                    return s
        for attr in named:
            try:
                val = getattr(report, attr, None)
            except Exception:
                val = None
            if val:
                s = str(val).strip()
                if s and s.lower() != "none":
                    return s
        return None

    @staticmethod
    def _safe_text(obj) -> Optional[str]:
        for attr in ("text", "markdown"):
            fn = getattr(obj, attr, None)
            try:
                val = fn() if callable(fn) else fn
            except Exception:
                val = None
            if val:
                s = str(val).strip()
                if s:
                    return s
        return None

    def _save_statements(self, report, out_dir: Path) -> None:
        """Persist the filing's three financial statements as CSV."""
        fin = getattr(report, "financials", None)
        if fin is None:
            return
        getters = {
            "income_statement": ("income_statement",),
            "balance_sheet": ("balance_sheet",),
            "cash_flow": ("cashflow_statement", "cash_flow_statement"),
        }
        for out_name, attrs in getters.items():
            stmt = None
            for attr in attrs:
                f = getattr(fin, attr, None)
                if f is None:
                    continue
                try:
                    stmt = f() if callable(f) else f
                except Exception:
                    stmt = None
                if stmt is not None:
                    break
            if stmt is None:
                continue
            df = self._statement_to_df(stmt)
            if df is not None and not df.empty:
                df.to_csv(out_dir / f"{out_name}.csv", index=True)

    @staticmethod
    def _statement_to_df(stmt) -> Optional[pd.DataFrame]:
        if isinstance(stmt, pd.DataFrame):
            return stmt
        for attr in ("to_dataframe", "to_pandas"):
            fn = getattr(stmt, attr, None)
            if callable(fn):
                try:
                    df = fn()
                    if isinstance(df, pd.DataFrame):
                        return df
                except Exception:
                    pass
        return None

    @staticmethod
    def _fiscal_year_of(obj, period) -> Optional[int]:
        fy = getattr(obj, "fiscal_year", None)
        if isinstance(fy, int):
            return fy
        if period:
            try:
                return int(str(period)[:4])
            except ValueError:
                return None
        return None

    @staticmethod
    def _jsonable(value):
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    # ================================================================== #
    # 2. NUMERIC DATABASE (quarterly BS / IS / CF across years)
    # ================================================================== #
    async def build_financials(
        self,
        ticker: str,
        min_date: str = "2010-01-01",
        *,
        company=None,
        force: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Build & persist the quarterly numeric DB from XBRL company facts."""
        ticker = ticker.upper()
        fin_dir = self._ticker_dir(ticker) / "financials"
        fin_dir.mkdir(parents=True, exist_ok=True)

        if company is None:
            company = await self._run(self._Company, ticker)
        facts = await self._run(company.get_facts)
        if facts is None:
            logger.warning("  no facts available for %s", ticker)
            return {}

        # One bulk pull (local CPU after facts are fetched).
        raw = facts.query().date_range(start=min_date).to_dataframe()
        if raw is None or raw.empty:
            logger.warning("  empty facts for %s", ticker)
            return {}

        raw = raw.copy()
        raw["ticker"] = ticker
        raw.to_csv(fin_dir / "facts_raw.csv", index=False)

        statements = {
            "balance_sheet": self._balance_sheet_quarters(raw, ticker),
            "income_statement": self._quarterize_flows(raw, "IncomeStatement", ticker),
            "cash_flow": self._quarterize_flows(raw, "CashFlow", ticker),
        }
        for name, df in statements.items():
            if df is None or df.empty:
                continue
            df.to_csv(fin_dir / f"{name}_quarterly.csv", index=False)
            wide = self._pivot_wide(df)
            if wide is not None and not wide.empty:
                wide.to_csv(fin_dir / f"{name}_wide.csv")
        logger.info("  numeric DB written to %s", fin_dir)
        return statements

    # ---- pure (offline-testable) transforms ---------------------------- #
    @classmethod
    def _classify_concept(cls, concept: str) -> Optional[str]:
        c = (concept or "").lower()
        # dei / cover-page facts (shares outstanding, public float, etc.) are not
        # financial-statement line items -- never classify them into a statement.
        if c.startswith("dei:") or c.startswith("entity"):
            return None
        c = c.replace("us-gaap:", "")
        if any(p in c for p in _CF_PATTERNS):
            return "CashFlow"
        if any(p in c for p in _IS_PATTERNS):
            return "IncomeStatement"
        if any(p in c for p in _BS_PATTERNS):
            return "BalanceSheet"
        return None

    @staticmethod
    def _coerce(raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["numeric_value"] = pd.to_numeric(df.get("numeric_value"), errors="coerce")
        for col in ("period_start", "period_end", "filing_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    @staticmethod
    def _snap_months(days: float) -> Optional[int]:
        if pd.isna(days):
            return None
        m = days / 30.44
        for target in (3, 6, 9, 12):
            if abs(m - target) <= 1.2:
                return target
        return None

    @classmethod
    def _balance_sheet_quarters(cls, raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = cls._coerce(raw)
        df = df[df["period_type"] == "instant"].copy()
        if df.empty:
            return pd.DataFrame()
        st = df.get("statement_type")
        keep = (st == "BalanceSheet") | df["concept"].map(
            lambda c: cls._classify_concept(c) == "BalanceSheet")
        df = df[keep & df["numeric_value"].notna() & df["period_end"].notna()].copy()
        if df.empty:
            return pd.DataFrame()
        # newest restatement wins for a given (concept, as-of date)
        df = df.sort_values("filing_date").drop_duplicates(
            ["concept", "period_end"], keep="last")
        df["fiscal_quarter"] = df["fiscal_period"].replace({"FY": "Q4"})
        out = pd.DataFrame({
            "ticker": ticker,
            "statement": "balance_sheet",
            "concept": df["concept"],
            "label": df["label"],
            "fiscal_year": df["fiscal_year"],
            "fiscal_quarter": df["fiscal_quarter"],
            "period_start": df["period_end"],   # snapshot -> start == end (as-of)
            "period_end": df["period_end"],
            "filing_date": df["filing_date"],
            "form_type": df["form_type"],
            "accession": df["accession"],
            "value": df["numeric_value"],
            "unit": df.get("unit"),
            "derived": False,
        })
        return cls._sort_statement(out)

    @classmethod
    def _quarterize_flows(cls, raw: pd.DataFrame, target: str, ticker: str) -> pd.DataFrame:
        """Recover discrete 3-month quarters from cumulative YTD facts.

        Flow statements report cumulative figures (Q1=3mo, Q2=6mo YTD, Q3=9mo YTD,
        FY=12mo from the 10-K). We difference them:
            Q1 = c3 ; Q2 = c6-c3 ; Q3 = c9-c6 ; Q4 = c12-c9
        which simultaneously recovers Q4 (only filed annually). If the YTD chain
        is broken we fall back to any directly-reported discrete 3-month fact and
        otherwise leave the quarter empty rather than emit a wrong number.
        """
        df = cls._coerce(raw)
        df = df[df["period_type"] == "duration"].copy()
        if df.empty:
            return pd.DataFrame()
        st = df.get("statement_type")
        keep = (st == target) | df["concept"].map(
            lambda c: cls._classify_concept(c) == target)
        df = df[keep].copy()
        df = df.dropna(subset=["numeric_value", "period_start", "period_end"])
        if df.empty:
            return pd.DataFrame()

        df["months"] = ((df["period_end"] - df["period_start"]).dt.days).map(cls._snap_months)
        df = df.dropna(subset=["months"])
        df["months"] = df["months"].astype(int)
        # newest restatement wins for a given (concept, exact period)
        df = df.sort_values("filing_date").drop_duplicates(
            ["concept", "period_start", "period_end", "months"], keep="last")

        statement_name = {"IncomeStatement": "income_statement",
                          "CashFlow": "cash_flow"}.get(target, target.lower())
        rows: List[dict] = []

        for (concept, fy), g in df.groupby(["concept", "fiscal_year"], sort=False):
            label = g["label"].iloc[0]
            unit = g["unit"].iloc[0] if "unit" in g else None
            twelve = g[g["months"] == 12]
            fy_start = (twelve["period_start"].min() if not twelve.empty
                        else g["period_start"].min())
            ytd = g[g["period_start"] == fy_start]
            cum = {int(m): r for m, r in
                   ytd.set_index("months").iterrows()} if not ytd.empty else {}
            discrete3 = g[(g["months"] == 3)]

            prev_val = 0.0
            prev_end = None
            for q, m in (("Q1", 3), ("Q2", 6), ("Q3", 9), ("Q4", 12)):
                cur_row = None
                value = None
                derived = False
                if m in cum:
                    r = cum[m]
                    cur = float(r["numeric_value"])
                    value = cur - prev_val
                    derived = (m != 3)          # Q1 is reported directly
                    cur_row = r
                    q_start = fy_start if prev_end is None else (prev_end + pd.Timedelta(days=1))
                    q_end = r["period_end"]
                    prev_val, prev_end = cur, q_end
                else:
                    # YTD chain broke: try a directly-reported discrete 3mo fact
                    cand = discrete3[discrete3["fiscal_period"] == q]
                    if cand.empty:
                        break  # don't fabricate downstream quarters
                    r = cand.sort_values("filing_date").iloc[-1]
                    value = float(r["numeric_value"])
                    cur_row = r
                    q_start = r["period_start"]
                    q_end = r["period_end"]
                    prev_end = q_end
                    prev_val = prev_val  # cumulative unknown; stop chaining after
                if cur_row is None:
                    continue
                rows.append({
                    "ticker": ticker,
                    "statement": statement_name,
                    "concept": concept,
                    "label": label,
                    "fiscal_year": int(fy) if pd.notna(fy) else None,
                    "fiscal_quarter": q,
                    "period_start": q_start,
                    "period_end": q_end,
                    "filing_date": cur_row["filing_date"],
                    "form_type": cur_row["form_type"],
                    "accession": cur_row["accession"],
                    "value": value,
                    "unit": unit,
                    "derived": bool(derived),
                })

        out = pd.DataFrame(rows)
        return cls._sort_statement(out)

    @staticmethod
    def _sort_statement(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        qorder = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        df = df.copy()
        df["_q"] = df["fiscal_quarter"].map(qorder).fillna(0)
        df = df.sort_values(["concept", "fiscal_year", "_q"]).drop(columns="_q")
        return df.reset_index(drop=True)

    @staticmethod
    def _pivot_wide(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df.empty:
            return None
        d = df.copy()
        d["period"] = (d["fiscal_year"].astype("Int64").astype(str)
                       + "-" + d["fiscal_quarter"].astype(str))
        wide = d.pivot_table(index="label", columns="period",
                            values="value", aggfunc="last")
        # chronological column order
        def _key(p):
            try:
                y, q = p.split("-Q")
                return (int(y), int(q))
            except Exception:
                return (0, 0)
        wide = wide[sorted(wide.columns, key=_key)]
        return wide

    # ================================================================== #
    # 3. RETRIEVAL
    # ================================================================== #
    def get_financials(self, ticker: str, years: int = 5) -> Dict[str, pd.DataFrame]:
        """Return {balance_sheet, income_statement, cash_flow} quarterly DataFrames
        for the most recent `years`, building the DB first if it is missing."""
        ticker = ticker.upper()
        fin_dir = self._ticker_dir(ticker) / "financials"
        needed = ["balance_sheet", "income_statement", "cash_flow"]
        if not all((fin_dir / f"{n}_quarterly.csv").exists() for n in needed):
            self._maybe_run(self.build_financials(ticker))

        out: Dict[str, pd.DataFrame] = {}
        for name in needed:
            path = fin_dir / f"{name}_quarterly.csv"
            if not path.exists():
                out[name] = pd.DataFrame()
                continue
            df = pd.read_csv(path, parse_dates=["period_start", "period_end", "filing_date"])
            if years and not df.empty:
                cutoff = df["period_end"].max() - pd.DateOffset(years=years)
                df = df[df["period_end"] >= cutoff].reset_index(drop=True)
            out[name] = df
        return out

    def get_latest_10k(self, ticker: str) -> LoadedFiling:
        return self._maybe_run(self._aget_filing(ticker, "10-K", which="latest"))

    async def aget_latest_10k(self, ticker: str) -> LoadedFiling:
        return await self._aget_filing(ticker, "10-K", which="latest")

    def get_latest_10q(self, ticker: str) -> LoadedFiling:
        return self._maybe_run(self._aget_filing(ticker, "10-Q", which="latest"))

    async def aget_latest_10q(self, ticker: str) -> LoadedFiling:
        return await self._aget_filing(ticker, "10-Q", which="latest")

    def get_10k_for_year(self, ticker: str, year: int) -> LoadedFiling:
        return self._maybe_run(self._aget_filing(ticker, "10-K", year=year))

    async def aget_10k_for_year(self, ticker: str, year: int) -> LoadedFiling:
        return await self._aget_filing(ticker, "10-K", year=year)

    def get_filing(self, ticker: str, form: str, *, which: str = "latest",
                year: Optional[int] = None) -> LoadedFiling:
        return self._maybe_run(self._aget_filing(ticker, form, which=which, year=year))

    async def aget_filing(self, ticker: str, form: str, *, which: str = "latest",
                year: Optional[int] = None) -> LoadedFiling:
        return await self._aget_filing(ticker, form, which=which, year=year)

    async def aget_financials(self, ticker: str, years: int = 5) -> Dict[str, pd.DataFrame]:
        """Async version of get_financials; safe to call inside a running event loop."""
        ticker = ticker.upper()
        fin_dir = self._ticker_dir(ticker) / "financials"
        needed = ["balance_sheet", "income_statement", "cash_flow"]
        if not all((fin_dir / f"{n}_quarterly.csv").exists() for n in needed):
            await self.build_financials(ticker)

        out: Dict[str, pd.DataFrame] = {}
        for name in needed:
            path = fin_dir / f"{name}_quarterly.csv"
            if not path.exists():
                out[name] = pd.DataFrame()
                continue
            df = pd.read_csv(path, parse_dates=["period_start", "period_end", "filing_date"])
            if years and not df.empty:
                cutoff = df["period_end"].max() - pd.DateOffset(years=years)
                df = df[df["period_end"] >= cutoff].reset_index(drop=True)
            out[name] = df
        return out

    async def _aget_filing(self, ticker: str, form: str, *, which: str = "latest",
                        year: Optional[int] = None) -> LoadedFiling:
        """Load a filing from disk if present, else download + persist + load."""
        ticker = ticker.upper()
        existing = self._find_on_disk(ticker, form, which=which, year=year)
        if existing is not None:
            return self._load_from_disk(ticker, form, existing)

        company = await self._run(self._Company, ticker)
        filings = await self._run(company.get_filings, form=form)
        filing = self._select_filing(filings, which=which, year=year)
        if filing is None:
            raise LookupError(f"No {form} found for {ticker} ({which=}, {year=})")

        await self._process_filing(ticker, form, filing, force=False)
        filing_date = getattr(filing, "filing_date", None)
        accession = getattr(filing, "accession_number", "") or getattr(filing, "accession", "")
        out_dir = self._filing_dir(ticker, form, filing_date, accession)
        return self._load_from_disk(ticker, form, out_dir)

    def _select_filing(self, filings, *, which: str, year: Optional[int]):
        if filings is None:
            return None
        if year is not None:
            best = None
            for f in filings:
                fd = self._fmt_date(getattr(f, "filing_date", None))
                pr = self._fmt_date(getattr(f, "period_of_report", None))
                fy = int(fd[:4]) if fd else None
                py = int(pr[:4]) if pr else None
                if year in (fy, py):
                    if best is None or fd > self._fmt_date(getattr(best, "filing_date", None)):
                        best = f
            return best
        latest = getattr(filings, "latest", None)
        if callable(latest):
            return latest()
        return filings[0] if len(filings) else None

    def _find_on_disk(self, ticker: str, form: str, *, which: str,
                    year: Optional[int]) -> Optional[Path]:
        form_dir = self._form_dir(ticker, form)
        if not form_dir.exists():
            return None
        dirs = sorted([d for d in form_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.name)  # name starts with filing_date
        if not dirs:
            return None
        if year is not None:
            matches = []
            for d in dirs:
                meta = self._read_meta(d)
                fd = (meta.get("filing_date") or d.name[:10])
                fy = meta.get("fiscal_year")
                if (fd[:4] == str(year)) or (fy == year):
                    matches.append(d)
            return matches[-1] if matches else None
        return dirs[-1]  # latest by filing date

    @staticmethod
    def _read_meta(d: Path) -> dict:
        p = d / "meta.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _load_from_disk(self, ticker: str, form: str, d: Path) -> LoadedFiling:
        meta = self._read_meta(d)
        sections: Dict[str, str] = {}
        for txt in sorted(d.glob("*.txt")):
            sections[txt.stem] = txt.read_text(encoding="utf-8")
        statements: Dict[str, pd.DataFrame] = {}
        for csv in sorted(d.glob("*.csv")):
            try:
                statements[csv.stem] = pd.read_csv(csv, index_col=0)
            except Exception:
                pass
        return LoadedFiling(
            ticker=ticker,
            form=meta.get("form", form),
            filing_date=meta.get("filing_date", d.name[:10]),
            accession=meta.get("accession", d.name[11:]),
            period_of_report=meta.get("period_of_report"),
            fiscal_year=meta.get("fiscal_year"),
            path=d,
            sections=sections,
            statements=statements,
        )