**Database Specification**

Stock Data Library --- Standalone Package

**Version:** 0.2 (Draft) **Date:** 2026-06-07

**Engine:** SQLite via SQLAlchemy (Python)

**Packaging:** Standalone Python package --- imported locally by the
forecaster and any future projects

**Scope:** Data layer only --- acquisition, storage, standardization,
and derived calculations. Model training and inference are out of scope.

# 1. Purpose & Scope

This document specifies the **stock_data** library --- a self-contained
SQLite database package that any downstream project can import to access
historical price data, SEC filings, and pre-computed technical and
fundamental metrics. The spec supersedes v0.1 and reflects the following
structural changes:

- **Standalone package:** No longer a subfolder of the forecaster. Lives
  in its own repo and is installed via pip install -e ../stock_data.

- **Async-first I/O:** Both yf.py and edgar.py expose async download
  methods.

- **edgar.py split:** The old download.py is now yf.py (Yahoo Finance /
  yahooquery) and edgar.py (edgartools). Both can get long.

- **EDGAR filing text storage:** edgartools parses and stores structured
  text sections from 10-K and 10-Q filings in addition to numerical
  financials.

- **Separated init pipelines:** YF and EDGAR inits are independent
  methods because their performance characteristics differ
  fundamentally.

- **indicator_data schema:** interval renamed to period so the
  calculation window can be stored directly.

## 1.1 Goals

- Authoritative, append-only local store for price, filing, and
  calculated data.

- Support cold-start and incremental daily updates per ticker without
  full recomputation.

- Expose a clean query API so any importing project gets ready-to-use
  DataFrames.

- Raw source data and derived calculations are in separate tables and
  independently recomputable.

## 1.2 Non-Goals

- Real-time or tick-level data.

- Multi-user concurrency (single writer assumed).

- Cloud sync or replication.

- Model weights, predictions, or any ML artifacts.

# 2. Data Sources

  -----------------------------------------------------------------------
  **Source**            Description
  --------------------- -------------------------------------------------
  **yahooquery          OHLCV price history, dividends, splits, ticker
  (yf.py)**             metadata (sector, industry, exchange). Supports
                        bulk multi-ticker queries.

  **edgartools          SEC EDGAR filings. Provides structured numerical
  (edgar.py)**          data (income statement, balance sheet, cash flow)
                        and raw text sections from 10-K and 10-Q forms.
  -----------------------------------------------------------------------

**Intervals for price/EMA/indicator tables:** \'daily\' and \'weekly\'.

**Intervals for financials/fundamentals tables:** \'quarterly\' and
\'annual\'.

# 3. Package File Structure

The package root is stock_data/. Nothing outside this package writes
directly to the database.

  -----------------------------------------------------------------------
  **File**                            **Responsibility**
  ----------------------------------- -----------------------------------
  stock_data/tables.py                **ORM definitions.** All SQLAlchemy
                                      model classes, primary keys,
                                      indexes, and column types. No
                                      logic.

  stock_data/yf.py                    **Yahoo Finance acquisition.**
                                      Async wrapper around yahooquery.
                                      Multi-ticker bulk queries for OHLCV
                                      and metadata. Returns DataFrames;
                                      does not touch the DB.

  stock_data/edgar.py                 **EDGAR acquisition.** Async
                                      wrapper around edgartools.
                                      Per-ticker sequential filing
                                      downloads. Parses both numerical
                                      financials and text sections. Does
                                      not touch the DB.

  stock_data/calculators.py           **Derived metrics.** EMAs,
                                      technical indicators, fundamental
                                      ratios. Stateless functions with
                                      optional seed param for incremental
                                      mode.

  stock_data/db_manager.py            **Persistence & query.** All DB
                                      reads/writes. Implements per-ticker
                                      init/update logic and the public
                                      query API.

  scripts/init_yf.py                  **YF init script.** Bulk-seeds
                                      price data for a ticker list. Fast
                                      --- yahooquery handles many tickers
                                      in parallel.

  scripts/init_edgar.py               **EDGAR init script.** Seeds
                                      financial and filing text data
                                      ticker-by-ticker. Slow --- intended
                                      to be run overnight or in batches.

  scripts/update_daily.py             **Daily runner.** Incremental
                                      update for all seeded tickers. Safe
                                      to run via cron.

  scripts/query_db_yf.py              **Ad-hoc CLI.** Interactive query
                                      interface for debugging and data
                                      inspection for yahooquery.

  scripts/query_db_edger.py           **Ad-hoc CLI.** Interactive query
                                      interface for debugging and data
                                      inspection for edgartools.

                                      
  -----------------------------------------------------------------------

# 4. EDGAR File Storage (Disk)

SEC filing text sections are stored on disk alongside the SQLite
database (filing metadata and numerical data live in the DB). The
directory layout mirrors the filing hierarchy:

sec_data/\<ticker\>/\<form\>/\<accession_number\>/ --- e.g.
sec_data/AAPL/10-K/0000320193-23-000106/

  -----------------------------------------------------------------------
  **File pattern**        **Content**             **Format**
  ----------------------- ----------------------- -----------------------
  \<section_key\>.txt     Raw extracted text for  Plain text
                          one filing section      
                          (e.g. mda.txt,          
                          risk_factors.txt).      

  financials.csv          Numerical line items    CSV
                          from the filing ---     
                          income statement,       
                          balance sheet, cash     
                          flow rows.              

  metadata.json           Filing metadata:        JSON
                          accession number, form  
                          type, filing_date,      
                          fiscal_year,            
                          fiscal_quarter,         
                          parse_extra_data flag.  
  -----------------------------------------------------------------------

## 4.1 10-K Sections

Sections are defined by TENK_SECTIONS. Each entry carries a
parse_extra_data flag --- the six essential sections are always
extracted; the remaining nine are only extracted when the flag is
enabled.

  --------------------------------------------------------------------------
  **Section key**              **Item**               **Essential?**
  ---------------------------- ---------------------- ----------------------
  business                     Item 1                 **Yes**

  risk_factors                 Item 1A                **Yes**

  legal_proceedings            Item 3                 **Yes**

  mda                          Item 7                 **Yes**

  market_risk                  Item 7A                **Yes**

  financial_statements         Item 8                 **Yes**

  cybersecurity                Item 1C                *Extra only*

  properties                   Item 2                 *Extra only*

  controls_and_procedures      Item 9A                *Extra only*

  directors_and_governance     Item 10                *Extra only*

  executive_compensation       Item 11                *Extra only*

  security_ownership           Item 12                *Extra only*

  related_party_transactions   Item 13                *Extra only*

  accountant_fees              Item 14                *Extra only*

  exhibits_index               Item 15                *Extra only*
  --------------------------------------------------------------------------

## 4.2 10-Q Sections

  -----------------------------------------------------------------------
  **Section key**         **Item**                **Essential?**
  ----------------------- ----------------------- -----------------------
  financial_statements    Part I, Item 1          **Yes**

  mda                     Part I, Item 2          **Yes**

  legal_proceedings       Part II, Item 1         **Yes**

  market_risk_changes     Part I, Item 3          *Extra only*

  controls_changes        Part I, Item 4          *Extra only*

  risk_factor_changes     Part II, Item 1A        *Extra only*

  share_repurchases       Part II, Item 2         *Extra only*

  other_information       Part II, Item 5         *Extra only*
  -----------------------------------------------------------------------

# 5. Database Table Specifications

**PK convention for time-series tables:** (ticker, date, interval)
unless noted. interval is \'daily\'/\'weekly\' for market-frequency
tables and \'quarterly\'/\'annual\' for filing-frequency tables.

**indicator_data exception:** Uses period (integer window size) instead
of interval --- see §5.5.

## 5.1 ticker_meta

**Not a time series.** One row per ticker. Updated on each daily run
when metadata is stale.

  ------------------------------------------------------------------------
  **Column**            **Type**         **PK?**         **Notes**
  --------------------- ---------------- --------------- -----------------
  ticker                TEXT             ✓               Yahoo Finance
                                                         ticker symbol.

  name                  TEXT                             Full company
                                                         name.

  sector                TEXT                             GICS sector.

  industry              TEXT                             GICS industry.

  last_price_date       DATE                             Most recent date
                                                         in price_data for
                                                         this ticker.

  last_filings_date     DATE                             Most recent
                                                         filing_date in
                                                         financials_data
                                                         for this ticker.

  created_at            DATETIME                         Row insertion
                                                         timestamp.

  updated_at            DATETIME                         Timestamp of last
                                                         metadata refresh.
  ------------------------------------------------------------------------

## 5.2 price_data

Raw OHLCV bars and corporate actions from yahooquery. Wide format. All
prices are split-adjusted at download time.

  -----------------------------------------------------------------------
  **Column**        **Type**          **PK?**           **Notes**
  ----------------- ----------------- ----------------- -----------------
  ticker            TEXT              ✓                 

  date              DATE              ✓                 Bar date (market
                                                        close date, ET).

  interval          TEXT              ✓                 \'daily\' or
                                                        \'weekly\'.

  open              REAL                                Split-adjusted
                                                        open.

  high              REAL                                Split-adjusted
                                                        high.

  low               REAL                                Split-adjusted
                                                        low.

  close             REAL                                Split-adjusted
                                                        close.

  volume            INTEGER                             Trade volume.

  dividend          REAL                                Dividend amount
                                                        on this date (0
                                                        if none).

  split_ratio       REAL                                Split multiplier
                                                        on this date (1.0
                                                        if none).
  -----------------------------------------------------------------------

## 5.3 ema_data

Exponential moving averages in long format so new (period, base)
combinations can be added without schema migration.

  -----------------------------------------------------------------------
  **Column**        **Type**          **PK?**           **Notes**
  ----------------- ----------------- ----------------- -----------------
  ticker            TEXT              ✓                 

  date              DATE              ✓                 Bar date.

  period            INTEGER           ✓                 EMA window in
                                                        bars (e.g. 9, 21,
                                                        50, 200).

  base              TEXT              ✓                 \'close\',
                                                        \'obv\', \'ad\',
                                                        etc.

  interval          TEXT              ✓                 \'daily\' or
                                                        \'weekly\'.

  ema_value         REAL                                Computed EMA
                                                        value.
  -----------------------------------------------------------------------

**Seeding:** Cold-start uses ewm(span=period, adjust=True) over full
history. Incremental update prepends the last stored ema_value as seed,
runs ewm(span=period, adjust=False) over new bars only, then drops the
seed row before insertion.

## 5.4 indicator_data

Technical indicators in long format. **Schema change from v0.1:**
interval is renamed to period (INTEGER) so the calculation window is
stored explicitly alongside the indicator name. The bar frequency is
captured by the separate freq column.

  -----------------------------------------------------------------------
  **Column**        **Type**          **PK?**           **Notes**
  ----------------- ----------------- ----------------- -----------------
  ticker            TEXT              ✓                 

  date              DATE              ✓                 Bar date.

  freq              TEXT              ✓                 \'daily\' or
                                                        \'weekly\' ---
                                                        bar frequency.

  indicator         TEXT              ✓                 Snake-case
                                                        indicator name,
                                                        e.g. \'rsi\',
                                                        \'adx\',
                                                        \'bb_upper\',
                                                        \'obv\'.

  period            INTEGER           ✓                 Calculation
                                                        window in bars. 0
                                                        for indicators
                                                        with no fixed
                                                        window (e.g.
                                                        \'obv\', \'ad\').

  value             REAL                                Computed
                                                        indicator value.
  -----------------------------------------------------------------------

### Indicator inventory (from calculators.py)

  --------------------------------------------------------------------------------
  **Indicator(s)**   **Period(s)**    **Seeded EWM?**  **Notes**
  ------------------ ---------------- ---------------- ---------------------------
  rsi                14               No               Standard Wilder RSI ---
                                                       rolling gain/loss means.

  bb_middle,         20               No               Bollinger Bands (2σ).
  bb_upper, bb_lower                                   

  stoch_k, stoch_d   14, 3            No               Stochastic Oscillator.

  plus_dm, minus_dm  14               Yes              Directional movement
                                                       components --- seeded EWM.

  atr                14               Yes              Average True Range ---
                                                       seeded EWM.

  plus_di, minus_di, 14               Yes              Directional Index and ADX
  adx                                                  --- ADX is a seeded EWM of
                                                       DX.

  cci                20               No               Commodity Channel Index ---
                                                       SMA + mean absolute
                                                       deviation.

  obv                0                No               On-Balance Volume ---
                                                       cumulative; seeded by last
                                                       stored value on update
                                                       (combine_first).

  ad                 0                No               Accumulation/Distribution
                                                       --- cumulative; same
                                                       seeding strategy as OBV.

  willr              14               No               Williams %R.

  aroon_up,          14               No               Aroon indicator.
  aroon_down                                           

  mfi                20               No               Money Flow Index.

  cmf                20               No               Chaikin Money Flow.
  --------------------------------------------------------------------------------

**Cumulative indicator seeding (OBV, AD):** On incremental update, the
last stored value is fetched and combine_first() is used to prepend it
before the new cumsum() runs --- matching the existing implementation.

## 5.5 financials_data

Raw numerical line items from EDGAR filings. Wide format (finite, known
column set). One row per ticker per fiscal period.

  ---------------------------------------------------------------------------
  **Column**             **Type**          **PK?**          **Notes**
  ---------------------- ----------------- ---------------- -----------------
  ticker                 TEXT              ✓                

  fiscal_year            INTEGER           ✓                Four-digit fiscal
                                                            year.

  fiscal_quarter         INTEGER           ✓                1--4 for
                                                            quarterly; 0 for
                                                            annual.

  interval               TEXT              ✓                \'quarterly\' or
                                                            \'annual\'.

  filing_date            DATE                               **Not a PK.** SEC
                                                            acceptance date
                                                            --- used for
                                                            point-in-time
                                                            alignment.

  accession_number       TEXT                               SEC accession
                                                            number --- links
                                                            to disk filing in
                                                            sec_data/.

  revenue                REAL                               Total revenues.

  net_income             REAL                               Net income to
                                                            common
                                                            shareholders.

  ebitda                 REAL                               EBITDA.

  total_assets           REAL                               Total assets.

  total_liabilities      REAL                               Total
                                                            liabilities.

  shareholders_equity    REAL                               Total
                                                            stockholders\'
                                                            equity.

  operating_cash_flow    REAL                               Net cash from
                                                            operations.

  capex                  REAL                               Capital
                                                            expenditures
                                                            (absolute value).

  free_cash_flow         REAL                               Operating cash
                                                            flow minus capex.

  shares_outstanding     REAL                               Diluted weighted
                                                            average shares.

  long_term_debt         REAL                               Long-term debt +
                                                            finance lease
                                                            obligations.

  cash_and_equivalents   REAL                               Cash + short-term
                                                            investments.

  ...                    REAL                               Additional EDGAR
                                                            tags appended as
                                                            columns. Prefer
                                                            extending this
                                                            wide table for
                                                            raw filing items.
  ---------------------------------------------------------------------------

## 5.6 filings_data (new in v0.2)

Filing metadata index. One row per filing. Points to the disk directory
where extracted text sections live. Does not store the text itself (kept
on disk due to size).

  ---------------------------------------------------------------------------------------
  **Column**         **Type**    **PK?**   **Notes**
  ------------------ ----------- --------- ----------------------------------------------
  ticker             TEXT        ✓         

  accession_number   TEXT        ✓         SEC accession number --- uniquely identifies
                                           the filing.

  form_type          TEXT                  \'10-K\' or \'10-Q\'.

  fiscal_year        INTEGER               Four-digit fiscal year.

  fiscal_quarter     INTEGER               0 for annual.

  filing_date        DATE                  SEC acceptance date.

  disk_path          TEXT                  Absolute or package-relative path to
                                           sec_data/\<ticker\>/\<form\>/\<accession\>/.

  sections_parsed    TEXT                  JSON list of section keys successfully
                                           extracted, e.g. \[\"mda\",\"risk_factors\"\].

  parse_extra_data   BOOLEAN               Whether extra sections were parsed for this
                                           filing.

  created_at         DATETIME              Row insertion timestamp.
  ---------------------------------------------------------------------------------------

## 5.7 fundamentals_data

**Long format.** Every row is a computed ratio or metric derived from
financials_data and/or price_data. Not a raw filing value.

  -----------------------------------------------------------------------
  **Column**        **Type**          **PK?**           **Notes**
  ----------------- ----------------- ----------------- -----------------
  ticker            TEXT              ✓                 

  fiscal_year       INTEGER           ✓                 

  fiscal_quarter    INTEGER           ✓                 0 for annual.

  interval          TEXT              ✓                 \'quarterly\' or
                                                        \'annual\'.

  metric            TEXT              ✓                 Snake-case metric
                                                        name.

  value             REAL                                Computed value.

  as_of_date        DATE                                **Not a PK.**
                                                        Closing price
                                                        date used when a
                                                        market-price
                                                        input was
                                                        required (P/E,
                                                        P/B, etc.).
  -----------------------------------------------------------------------

**Maintained metric set:**

- price_to_earnings --- trailing twelve months

- price_to_book, price_to_sales

- ev_to_ebitda

- debt_to_equity, current_ratio

- return_on_equity, return_on_assets

- gross_margin, operating_margin, net_margin

- fcf_yield --- FCF / market cap

# 6. Init Pipelines --- YF vs. EDGAR

The YF and EDGAR cold-start pipelines are **independent** because their
performance characteristics are fundamentally different. yahooquery
handles many tickers efficiently in bulk; edgartools is strictly
sequential and much slower due to SEC rate limits and per-filing parsing
overhead.

  -----------------------------------------------------------------------
  **Characteristic**      **YF pipeline**         **EDGAR pipeline**
  ----------------------- ----------------------- -----------------------
  Parallelism             Multi-ticker bulk query One ticker at a time.
                          via yahooquery.         SEC rate limits prevent
                                                  parallelism.

  Speed                   Fast --- minutes for a  Slow --- hours for a
                          full universe.          large universe.
                                                  Intended for overnight
                                                  batch.

  Seeded flag             yf_seeded = True        edgar_seeded = True

  Script                  scripts/init_yf.py      scripts/init_edgar.py

  Can run independently?  **Yes**                 **Yes --- forecaster
                                                  can use price data
                                                  before EDGAR data is
                                                  ready.**
  -----------------------------------------------------------------------

## 6.1 YF Cold-Start (yf_seeded = False)

  -----------------------------------------------------------------------
  **Step**                            **Action**
  ----------------------------------- -----------------------------------
  **1**                               **Bulk download.** yahooquery
                                      called for all unseeded tickers in
                                      one query. Full history, no start
                                      date cap.

  **2**                               **Standardize & insert
                                      price_data.** INSERT OR IGNORE on
                                      PK conflict.

  **3**                               **Insert ticker_meta.** Upsert
                                      sector, industry, exchange, etc.

  **4**                               **Seed EMAs.** For each (period,
                                      base, interval): run
                                      ewm(span=period, adjust=True) over
                                      full history. Write to ema_data.

  **5**                               **Compute indicators.** Run full
                                      indicator suite over full history.
                                      Seeded EWM indicators (ADX
                                      components, ATR) use adjust=True on
                                      cold-start. Write to
                                      indicator_data.

  **6**                               **Set yf_seeded = True**, update
                                      last_price_date, updated_at.
  -----------------------------------------------------------------------

## 6.2 EDGAR Cold-Start (edgar_seeded = False)

  -----------------------------------------------------------------------
  **Step**                            **Action**
  ----------------------------------- -----------------------------------
  **1**                               **Fetch all filings.** edgartools
                                      called for this ticker. All 10-K
                                      and 10-Q filings retrieved.

  **2**                               **Parse essential sections.** For
                                      each filing, extract the **6
                                      essential 10-K** / **3 essential
                                      10-Q** sections. Save as
                                      \<section_key\>.txt on disk.

  **3**                               **Parse extra sections (if
                                      enabled).** Extra sections saved on
                                      disk; parse_extra_data flag
                                      recorded in filings_data.

  **4**                               **Parse numerical financials.**
                                      Save as financials.csv on disk and
                                      insert rows into financials_data.

  **5**                               **Write filing metadata.** Insert
                                      row into filings_data (accession
                                      number, disk_path, sections_parsed,
                                      filing_date, etc.).

  **6**                               **Compute fundamentals.** For each
                                      fiscal period, compute all metrics
                                      using financials_data + closest
                                      prior price_data close. Insert into
                                      fundamentals_data.

  **7**                               **Set edgar_seeded = True**, update
                                      last_fundamentals_date, updated_at.
  -----------------------------------------------------------------------

## 6.3 YF Incremental Update

Run daily for each yf_seeded = True ticker.

  -----------------------------------------------------------------------
  **Step**                            **Action**
  ----------------------------------- -----------------------------------
  **1**                               **Fetch new bars.** yahooquery
                                      called with start =
                                      last_price_date + 1 day. Skip if
                                      empty.

  **2**                               **Insert price rows.** INSERT OR
                                      IGNORE.

  **3**                               **EMA incremental update.** Fetch
                                      last ema_value per (period, base,
                                      interval). Prepend as seed →
                                      ewm(adjust=False) over new bars →
                                      drop seed row → insert.

  **4**                               **Indicator incremental update.**
                                      Fetch trailing window_size rows →
                                      append new bars → compute tail →
                                      INSERT OR IGNORE new date rows.
                                      Seeded EWM indicators (plus_dm,
                                      minus_dm, atr, adx) fetch last
                                      value and pass as seed.

  **5**                               **Update ticker_meta.**
                                      last_price_date, updated_at.
  -----------------------------------------------------------------------

## 6.4 EDGAR Incremental Update

Run daily for each edgar_seeded = True ticker. Much lighter than
cold-start --- only new filings since last_fundamentals_date are
processed.

  -----------------------------------------------------------------------
  **Step**                            **Action**
  ----------------------------------- -----------------------------------
  **1**                               **Poll for new filings.**
                                      edgartools checks for filings with
                                      acceptance_date \>
                                      last_fundamentals_date. Skip ticker
                                      if none.

  **2**                               **Parse and store new filing.**
                                      Same disk + DB process as
                                      cold-start §6.2 steps 2--5.

  **3**                               **Compute fundamentals** for new
                                      fiscal periods.

  **4**                               **Update ticker_meta.**
                                      last_fundamentals_date, updated_at.
  -----------------------------------------------------------------------

## 6.5 Conflict & Staleness Policy

  -----------------------------------------------------------------------
  **Table**                           **Policy**
  ----------------------------------- -----------------------------------
  price_data, ema_data,               **INSERT OR IGNORE** on PK
  indicator_data, financials_data,    conflict. Existing rows never
  fundamentals_data, filings_data     overwritten on routine update.
                                      Force refresh: delete affected rows
                                      and re-run cold-start for that
                                      ticker.

  ticker_meta                         **INSERT OR REPLACE** (upsert) ---
                                      metadata can change over time.
  -----------------------------------------------------------------------

# 7. Async Architecture

Both acquisition modules expose **async methods** to allow concurrent
I/O without blocking. The DB write layer remains synchronous (SQLite is
not async-safe without a dedicated async driver).

  -----------------------------------------------------------------------
  **Module**              **Async pattern**       **Notes**
  ----------------------- ----------------------- -----------------------
  yf.py                   asyncio with            yahooquery handles bulk
                          asyncio.gather() for    internally; async
                          multi-ticker batches.   wrapping allows the
                                                  caller to await
                                                  alongside other tasks.

  edgar.py                asyncio with sequential One ticker at a time by
                          per-ticker await.       design (SEC rate
                                                  limits). Async allows
                                                  cancellation and
                                                  timeout handling.

  db_manager.py           Synchronous.            All DB writes use
                                                  standard SQLAlchemy
                                                  sessions. Caller awaits
                                                  I/O, then calls sync DB
                                                  methods.
  -----------------------------------------------------------------------

# 8. Query API (db_manager.py)

Importing projects use only these functions. All return pandas
DataFrames. All SQL is parameterized.

  ---------------------------------------------------------------------------
  **Function**                 **Returns**            **Notes**
  ---------------------------- ---------------------- -----------------------
  get_price_data(ticker,       DataFrame              OHLCV + actions for one
  interval, start, end)                               ticker over a date
                                                      range.

  get_ema(ticker, interval,    DataFrame              EMA series for a
  period, base, start, end)                           specific (period, base)
                                                      combination.

  get_indicators(ticker, freq, DataFrame (wide)       Long → wide pivot on
  indicators, start, end)                             indicator. indicators
                                                      is a list of (name,
                                                      period) tuples.

  get_financials(ticker,       DataFrame              All filing rows; caller
  interval)                                           filters by
                                                      fiscal_year/quarter.

  get_fundamentals(ticker,     DataFrame (wide)       Long → wide pivot on
  interval, metrics)                                  metric.

  get_ticker_meta(tickers)     DataFrame              Metadata rows for a
                                                      list of tickers.

  get_filing_path(ticker,      Path                   Resolves the disk path
  accession_number)                                   for a specific
                                                      filing\'s text
                                                      sections.

  get_feature_bundle(ticker,   DataFrame              Convenience join:
  freq, start, end)                                   price + EMAs +
                                                      indicators aligned on
                                                      date. Primary ingestion
                                                      path for the
                                                      forecaster.
  ---------------------------------------------------------------------------

# 9. Operational Scripts

  --------------------------------------------------------------------------
  **Script**                 **Purpose**             **Key flags**
  -------------------------- ----------------------- -----------------------
  scripts/init_yf.py         Cold-start for YF data. \--tickers, \--db-path
                             Bulk downloads price +  
                             computes EMAs and       
                             indicators.             

  scripts/init_edgar.py      Cold-start for EDGAR    \--tickers, \--extra
                             data. Per-ticker, slow  (parse extra sections),
                             --- run as overnight    \--db-path
                             batch.                  

  scripts/init_combined.py   Runs YF then EDGAR      \--tickers, \--extra,
                             sequentially.           \--db-path
                             Convenience wrapper.    

  scripts/update_daily.py    Incremental update for  \--db-path, \--dry-run,
                             all seeded tickers.     \--tickers
                             Runs YF update (fast)   
                             then EDGAR update       
                             (checks for new         
                             filings). Safe for      
                             cron.                   

  scripts/query_db.py        Ad-hoc CLI query        Subcommands mirror
                             interface.              query API; \--csv flag
                                                     for pipe output.
  --------------------------------------------------------------------------

# 10. Open Questions

  -----------------------------------------------------------------------
  **ID**                  **Question**            **Impact**
  ----------------------- ----------------------- -----------------------
  **OQ-1**                Should weekly bars be   Affects whether yf.py
                          derived by resampling   makes one or two
                          daily bars in-DB, or    queries per ticker.
                          fetched directly from   
                          yahooquery?             

  **OQ-2**                What is the canonical   Needed to estimate
                          list of (period, base,  ema_data volume and
                          interval) EMA           design the index.
                          combinations the        
                          forecaster will use?    

  **OQ-3**                Should restated EDGAR   Affects point-in-time
                          filings (amended        correctness.
                          10-K/Q) overwrite the   
                          existing row, or insert 
                          a new row with an       
                          amendment flag?         

  **OQ-4**                Is SQLite WAL mode      Minor --- one-line
                          needed? (Relevant if a  pragma, but concurrency
                          forecaster process      model should be
                          reads while             confirmed.
                          update_daily.py         
                          writes.)                

  **OQ-5**                What is the right batch Determines runtime of
                          size and delay between  init_edgar.py for a
                          EDGAR ticker requests   large universe.
                          to respect SEC rate     
                          limits?                 

  **OQ-6**                Should parse_extra_data If True, each new
                          default to True or      quarterly filing incurs
                          False for the daily     extra parsing time.
                          update runner?          
  -----------------------------------------------------------------------

# 11. Revision History

  -----------------------------------------------------------------------
  **Version**       **Date**          **Author**        **Summary**
  ----------------- ----------------- ----------------- -----------------
  0.1               2026-06-07        Connor            Initial draft.

  0.2               2026-06-07        Connor            Restructured as
                                                        standalone
                                                        package.
                                                        download.py split
                                                        into yf.py +
                                                        edgar.py. Async
                                                        I/O added to both
                                                        acquisition
                                                        modules. EDGAR
                                                        text section
                                                        storage and
                                                        filings_data
                                                        table added.
                                                        10-K/10-Q section
                                                        inventory
                                                        documented.
                                                        is_seeded split
                                                        into yf_seeded +
                                                        edgar_seeded. YF
                                                        and EDGAR init
                                                        pipelines
                                                        separated.
                                                        indicator_data:
                                                        interval → period
                                                        (INTEGER) + freq
                                                        column added.
                                                        Indicator
                                                        inventory
                                                        documented.
                                                        fundamentals
                                                        metric set
                                                        maintained from
                                                        v0.1.
  -----------------------------------------------------------------------
