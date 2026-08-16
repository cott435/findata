# scripts/db_handling — database lifecycle CLIs

All operate on the SQLite DB at `data/stock.db` (override with `--db-path`).
Schema lives in `findata/database/tables.py`; ingestion in `yf.py` /
`edgar_.py`; indicator/fundamental math in `findata/preprocess/calculators/`.

| Script | Purpose                                                                                                                                                                                                                                                                           |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `init_yf.py` | Yahoo cold-start. `--level full` (default) stores prices + the full default technical history; `--level prices` stores bars only (indicators compute on the fly on first request). Already-seeded tickers skipped unless `--force`; `--use-all` seeds the whole sampler universe. |
| `seed_technicals.py` | Promote prices-only tickers to full technical coverage (`--tickers ...` or `--all-prices-only`). The explicit replacement for the old silent cold-compute fallback.                                                                                                               |
| `backfill_indicator.py` | Universe-wide compute + persist of one item/period (`--item rsi --period 25 [--all-seeded]`) — use before wanting a custom window in universe workbench views.                                                                                                                    |
| `init_edgar.py` | EDGAR cold-start. `--data full` (default) = text sections + XBRL facts + Form 4; `--data numeric` = XBRL facts (no text downloads). Resumable; `--force` to reseed.                                                                                                               |
| `build_fundamentals.py` | Compute + upsert the derived fundamentals catalog (`--all-edgar-seeded` or `--tickers ...`), gated by `SAVE_POLICY` groups.                                                                                                                                                       |
| `migrate_technicals.py` | One-off: fold the legacy long `ema_data`/`indicator_data` into the wide `technical_data` table. Additive + resumable; `--finalize` renames the DB to `<name>.pre_migration` and rebuilds it compact.                                                                              |
| `update_daily.py` | Daily incremental refresh: new bars for every yf-seeded ticker + technical continuation from stored state; EDGAR refresh (full/numeric per ticker); fundamentals rebuild for tickers with new statements. Cron-safe.                                                              |
| `bench_reads.py` | Times `get_price_data` / `get_all_data` / `get_items` at N tickers (before/after read-path changes).                                                                                                                                                                              |
| `check_seeding.py` | Parity check: cold-start vs seeded-incremental technical recomputation for one ticker.                                                                                                                                                                                            |
| `query_db_yf.py` / `query_db_edgar.py` | Walkthroughs of the YF-side / EDGAR-side query APIs.                                                                                                                                                                                                                              |

```bash
python -m scripts.db_handling.init_yf --tickers AAPL MSFT GOOGL --level full
python -m scripts.db_handling.init_edgar --tickers AAPL MSFT --data numeric
python -m scripts.db_handling.build_fundamentals --tickers AAPL MSFT
python -m scripts.db_handling.update_daily --dry-run
```

## Workflow

Four data layers, each depending on the one above it: **prices → technicals →
financials → fundamentals**. Seed them in that order; afterwards a single
`update_daily` run maintains all four.

### 1. Prices

```bash
python -m scripts.db_handling.init_yf --use-all --level full
```

`--level full` also computes the whole default technical history in the same
pass. `--level prices` stores bars only, leaving indicators to compute on
first request — pick it when you want the universe on disk quickly and don't
yet know which windows you need. Re-runs skip seeded tickers unless `--force`,
and print each skipped ticker's stored last bar next to Yahoo's, so staleness
is visible at a glance.

### 2. Technicals

Nothing to run if you seeded `--level full`. Otherwise:

```bash
python -m scripts.db_handling.seed_technicals --all-prices-only   # promote prices-only tickers
python -m scripts.db_handling.backfill_indicator --item rsi --period 25 --all-seeded
```

Any indicator/window is also computed **on demand** by `get_items` and
persisted per `SAVE_POLICY`, so `backfill_indicator` is only needed to
pre-warm a custom window before a universe-wide view pays for it lazily.

### 3. Financials (EDGAR)

```bash
python -m scripts.db_handling.init_edgar --use-all --data numeric
```

`--data numeric` fetches only the XBRL facts — two requests per ticker, no
filing text. `--data full` additionally downloads and stores 10-K/10-Q/8-K
sections to `data/sec_filings/` plus Form 4 rows; it costs one request per
filing, so it is an overnight job. Interrupted runs resume. Facts are always
fetched as one whole-history frame, so a re-run is authoritative: statements
are **replaced** per ticker rather than appended, which makes parser fixes
self-healing.

### 4. Fundamentals

```bash
python -m scripts.db_handling.build_fundamentals --all-edgar-seeded
```

Derives the 24-item catalog (valuation, profitability, growth, quality,
leverage, efficiency, surprise) from the stored statements plus prices.
Valuation items need a close near each filing date, so **run the price update
first** — the script warns when prices lag the newest filing, since market cap
and every ratio built on it drop out otherwise.

Runs in batches of 50 tickers (`--batch-size`), each reading, computing and
upserting before the next begins, with a progress bar. That bounds memory and
means an interrupted run keeps everything already written — a resume picks up
only the tickers left. Only tickers whose statements are newer than their last
build are processed; `--force` rebuilds regardless.

### Updating

```bash
python -m scripts.db_handling.update_daily                  # all three stages
python -m scripts.db_handling.update_daily --stages prices  # just bars + technicals
```

Three stages, selectable with `--stages`:

- **prices** — targets the last completed session (weekday heuristic in ET;
  before the close the target is yesterday, since today's bar is still
  moving) and fetches **only the tickers behind it**, back only as far as the
  stalest of those. Technicals then continue from stored state for the
  tickers that actually gained bars — EWM/cumulative items resume from seeds,
  rolling windows recompute over a warmup. Whatever columns a ticker has are
  what get maintained, so custom windows ride along automatically. Tickers
  with no technicals stay prices-only.

  Symbols that fall well behind a **confirmed** session and still return
  nothing are flagged `inactive_since` and dropped from the polling set —
  delisted and renamed tickers otherwise get re-requested forever. They are
  retried every 30 days, and the flag clears itself the moment a symbol
  produces bars again. `--include-inactive` forces them back in. A session
  only counts as confirmed when some ticker already holds a bar for it, so a
  holiday can never retire the universe.
- **edgar** — filings and Form 4s are already incremental. The XBRL facts are
  not (they only come as a full-history frame), so the stage is **event
  driven**: SEC publishes a market-wide filing index, so a single request
  lists every filing by every company and only tickers with a 10-K/10-Q newer
  than the newest one already parsed get refetched — ~4s for an 800-ticker
  universe. Tickers absent from the index fall back to the `--facts-max-age`
  window (default 7 days), so a lookup failure slows the cadence instead of
  freezing the data. The comparison floors on both the newest statement
  parsed and `last_facts_date`, so a filing carrying no XBRL cannot re-trigger
  forever.
- **fundamentals** — recomputed only for the tickers stage 2 actually
  refreshed. Run the stage alone to rebuild the whole seeded universe.

All stages are cron-safe and idempotent. `--dry-run` reports what would change.

### Checks

```bash
python -m scripts.db_handling.check_seeding --ticker A      # seeded vs cold-start parity
python -m scripts.db_handling.bench_reads --tickers-n 800   # read-path timing
python -m scripts.db_handling.query_db_yf                   # query API walkthroughs
python -m scripts.db_handling.query_db_edgar
```

Statement **completeness** is checked automatically during every EDGAR run:
each ticker is reported for fiscal years holding fewer than four quarters.
The check is deliberately quiet where silence is correct — an item's first
couple of years are exempt (early filings are routinely partial), the
in-progress year is skipped, and items an issuer only ever tags annually are
ignored. Disable with `--no-validate`, widen with `--grace-years`.

## Data cleaning

Most of the work sits in the EDGAR parse, where the raw XBRL is messier than
it looks:

- **Fiscal periods are re-derived, not trusted.** XBRL's `fiscal_year` field
  describes the *filing* a fact came from, so a period reported only as a
  comparative in a later filing arrives mislabeled (AA's FY2015 revenue shows
  up as 2017). Year and quarter are instead derived from each period's own end
  date against the issuer's fiscal-year-end month, which is filing-independent
  and survives restatements. A 10-day shift keeps 52/53-week years that end in
  early January on the right year.
- **Concepts resolve through priority lists.** Issuers tag the "same" item
  differently (and migrate tags over time), so each item holds an ordered list
  of candidates — first tag reporting a period wins, earliest filing breaks
  ties, giving as-originally-reported values.
- **Missing quarters are derived where the arithmetic is sound.** A Q4 that is
  never reported standalone is annual minus Q1–Q3. Weighted-average share
  counts are averages, not sums, so they use `4*FY − Q1..Q3` instead, and a
  non-positive result is discarded rather than stored.
- **Cash flow is always de-cumulated.** Issuers report cumulatively from the
  fiscal-year start (Q1, H1, 9M, FY), which cannot be summed into a
  trailing-twelve-month figure. Consecutive windows are differenced into
  standalone quarters and stored alongside the originals — `derived` is part
  of the primary key, so both views coexist.
- **Balance levels carry forward, flows never do.** Balance items are stocks:
  the last reported figure stands until the next report (bounded to four
  quarters), which matters because many issuers break out debt or investments
  only in the 10-K. Flows belong to their own period and are never carried.
- **Market cap degrades honestly.** The close used is the last one at or
  before the filing date — never after, so there is no lookahead. A gap inside
  the price history means the name wasn't trading and the value is NaN; past
  the end of the series it just means prices haven't refreshed, so the newest
  close stands for a bounded window and corrects itself on the next build.

**Universe scope.** `TickerSampler(domestic_only=True)` (the default) drops
foreign private issuers, classified by which annual form each company
actually files — 10-K means domestic, 20-F/40-F means foreign. They are
excluded rather than fixed because they tag XBRL in the `ifrs-full` taxonomy
and report semi-annually, so the us-gaap concept lists and quarter-length
windows above resolve almost nothing for them (BTI: 9,061 of 9,073 facts are
`ifrs-full`, and cash flow matched 0 of 7 items). Their statements came out
empty rather than wrong. Classification is cached to `data/filer_types.csv`
and refreshed every 90 days; tickers EDGAR has no annual filing for are kept,
since absence of evidence is not evidence of a foreign filer.

Ratios are left raw here — bounds, winsorizing and cross-sectional scaling are
feature-engineering concerns and live with the calculators (see
`findata/preprocess/calculators/README.md`). Some emptiness is real rather
than a defect: banks and REITs have no cost of revenue or gross profit, so
those items are structurally absent for them.

## Storage model

`technical_data` is **wide**: one row per (ticker, date, interval), one REAL
column per `item_period` (`rsi_14`, `ema_close_26`, `obv`, …). New windows
requested on the fly are added as columns (`ALTER TABLE ADD COLUMN`) and
persisted per `SAVE_POLICY`. `fundamental_data` is **long** and
point-in-time: keyed by (ticker, filed_date, item), forward-filled onto
trading dates at read time via `get_fundamentals(daily=True)`. See
`findata/preprocess/calculators/README.md` for the calculator/save-policy
model.
