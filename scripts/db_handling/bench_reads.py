"""Read-path benchmark: times the multi-ticker read API and prints rows/s.

Run before/after read-path changes for evidence:
    python -m scripts.db_handling.bench_reads --tickers-n 20 [--start 2015-01-01]
"""

import argparse
import time

from findata.configs import setup_logging
from findata.database.db_manager import DBManager


def _time(label, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    n = len(result)
    print(f'{label:<45} {dt:8.2f}s  {n:>12,} rows  {n / max(dt, 1e-9):>14,.0f} rows/s')
    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark multi-ticker reads.')
    parser.add_argument('--tickers-n', type=int, default=20)
    parser.add_argument('--start', default=None, help='Optional start date filter.')
    parser.add_argument('--db-path', default=None)
    args = parser.parse_args()
    setup_logging('bench_reads.log', console=False)

    db = DBManager(args.db_path)
    meta = db.get_ticker_meta()
    seeded = sorted(meta.index[meta['yf_seeded'].fillna(False).astype(bool)])
    tickers = seeded[:args.tickers_n]
    print(f'{len(tickers)} tickers (of {len(seeded)} seeded), start={args.start}')

    _time(f'get_price_data({len(tickers)})',
          lambda: db.get_price_data(tickers, start=args.start))
    _time(f'get_all_data({len(tickers)}, wide=True)',
          lambda: db.get_all_data(tickers, start=args.start))
    _time(f'get_items({len(tickers)}, rsi_14 + close)',
          lambda: db.get_items(tickers, {'rsi': 14, 'close': None}, start=args.start))


if __name__ == '__main__':
    main()
