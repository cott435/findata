# scripts/db_handling — database lifecycle CLIs

All operate on the SQLite DB at `data/stock.db` (override with `--db-path`).
Schema lives in `findata/database/tables.py`; ingestion in `yf.py` / `edgar_.py`.

| Script | Purpose |
|---|---|
| `init_yf.py` | Yahoo cold-start: bulk price history + full EMA/indicator history for a ticker list. Already-seeded tickers are skipped (stored vs latest bar printed) unless `--force`. |
| `init_edgar.py` | EDGAR cold-start: 10-K/10-Q/8-K text sections, Form 4 insider transactions, XBRL financial facts. Slow by design (SEC rate limits); resumable; `--force` to reseed. |
| `update_daily.py` | Daily incremental refresh: new bars for every yf-seeded ticker + EMA/indicator extension from stored seeds. INSERT OR IGNORE — cron-safe. The trading repo's production loop shells out to this. |
| `query_db_yf.py` | Walkthrough of every DBManager query option against one ticker (prices, wide/long `get_all_data`, mixed `get_items`, `if_missing` behaviors). |
| `query_db_edgar.py` | Walkthrough of the EDGAR-side query API (filings index, 10-K/10-Q sections, 8-K items, Form 4, income/balance/cashflow). Local reads only. |

```bash
python -m scripts.db_handling.init_yf --tickers AAPL MSFT GOOGL
python -m scripts.db_handling.update_daily --dry-run
```
