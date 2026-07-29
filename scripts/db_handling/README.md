# scripts/db_handling — database lifecycle CLIs

All operate on the SQLite DB at `data/stock.db` (override with `--db-path`).
Schema lives in `findata/database/tables.py`; ingestion in `yf.py` /
`edgar_.py`; indicator/fundamental math in `findata/preprocess/calculators/`.

| Script | Purpose |
|---|---|
| `init_yf.py` | Yahoo cold-start. `--level full` (default) stores prices + the full default technical history; `--level prices` stores bars only (indicators compute on the fly on first request). Already-seeded tickers skipped unless `--force`; `--use-all` seeds the whole sampler universe. |
| `seed_technicals.py` | Promote prices-only tickers to full technical coverage (`--tickers ...` or `--all-prices-only`). The explicit replacement for the old silent cold-compute fallback. |
| `backfill_indicator.py` | Universe-wide compute + persist of one item/period (`--item rsi --period 25 [--all-seeded]`) — use before wanting a custom window in universe workbench views. |
| `init_edgar.py` | EDGAR cold-start. `--data full` (default) = text sections + XBRL facts + Form 4; `--data numeric` = XBRL facts + Form 4 only (no text downloads). Resumable; `--force` to reseed. |
| `build_fundamentals.py` | Compute + upsert the derived fundamentals catalog (`--all-edgar-seeded` or `--tickers ...`), gated by `SAVE_POLICY` groups. |
| `migrate_technicals.py` | One-off: fold the legacy long `ema_data`/`indicator_data` into the wide `technical_data` table. Additive + resumable; `--finalize` renames the DB to `<name>.pre_migration` and rebuilds it compact. |
| `update_daily.py` | Daily incremental refresh: new bars for every yf-seeded ticker + technical continuation from stored state; EDGAR refresh (full/numeric per ticker); fundamentals rebuild for tickers with new statements. Cron-safe. |
| `bench_reads.py` | Times `get_price_data` / `get_all_data` / `get_items` at N tickers (before/after read-path changes). |
| `check_seeding.py` | Parity check: cold-start vs seeded-incremental technical recomputation for one ticker. |
| `query_db_yf.py` / `query_db_edgar.py` | Walkthroughs of the YF-side / EDGAR-side query APIs. |

```bash
python -m scripts.db_handling.init_yf --tickers AAPL MSFT GOOGL --level full
python -m scripts.db_handling.init_edgar --tickers AAPL MSFT --data numeric
python -m scripts.db_handling.build_fundamentals --tickers AAPL MSFT
python -m scripts.db_handling.update_daily --dry-run
```

## Storage model

`technical_data` is **wide**: one row per (ticker, date, interval), one REAL
column per `item_period` (`rsi_14`, `ema_close_26`, `obv`, …). New windows
requested on the fly are added as columns (`ALTER TABLE ADD COLUMN`) and
persisted per `SAVE_POLICY`. `fundamental_data` is **long** and
point-in-time: keyed by (ticker, filed_date, item), forward-filled onto
trading dates at read time via `get_fundamentals(daily=True)`. See
`findata/preprocess/calculators/README.md` for the calculator/save-policy
model.
