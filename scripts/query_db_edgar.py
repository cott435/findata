"""Walkthrough of the EDGAR-side query API against a seeded ticker.

Shows:
    1. filings_data index (what text filings are stored where)
    2. get_10k / get_10q section dicts read back from disk
    3. a saved 8-K item (date + item number prepended)
    4. Form 4 insider transactions
    5. income (quarterly with derived Q4 + annual), balance, cash flow

Everything here is read from the local DB/disk -- no SEC requests (the
``obj=True`` option on get_10k/get_10q would fetch live; not used here).

Run from the project root after init_edgar:
    python -m scripts.query_db_edgar [--ticker TEM] [--db-path data/stock.db]
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.db_manager import DBManager
from src.edgar_ import EdgarPipeline


def banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description='Demo the EDGAR query API.')
    parser.add_argument('--ticker', default='TEM')
    parser.add_argument('--db-path', default=None)
    args = parser.parse_args()
    ticker = args.ticker.upper()
    logging.basicConfig(level=logging.WARNING)

    db = DBManager(args.db_path)
    pipe = EdgarPipeline(db=db)
    pd.set_option('display.width', 220)

    meta = db.get_filings_meta(ticker)
    if meta.empty:
        print(f'{ticker} has no EDGAR data -- run: python -m scripts.init_edgar --tickers {ticker}')
        return

    banner(f'1. filings_data for {ticker}')
    print(meta.groupby('form_type').agg(filings=('accession_number', 'count'),
                                        first=('filing_date', 'min'),
                                        last=('filing_date', 'max')))
    print(meta[['form_type', 'fiscal_year', 'fiscal_quarter', 'filing_date',
                'period_of_report', 'sections_parsed']].head(8).to_string(index=False))

    tenk_meta = meta[meta['form_type'] == '10-K']
    if not tenk_meta.empty:
        year = int(tenk_meta['fiscal_year'].max())
        banner(f'2a. get_10k({ticker!r}, {year}) -- dict of saved sections')
        tenk = pipe.get_10k(ticker, year)
        print('keys:', list(tenk))
        print('\nmda preview:', ' '.join(tenk.get('mda', '')[:400].split()))

    tenq_meta = meta[meta['form_type'] == '10-Q']
    if not tenq_meta.empty:
        year, quarter = (int(tenq_meta.iloc[0]['fiscal_year']),
                         int(tenq_meta.iloc[0]['fiscal_quarter']))
        banner(f'2b. get_10q({ticker!r}, {year}, {quarter})')
        tenq = pipe.get_10q(ticker, year, quarter)
        print('keys:', list(tenq))

    eightk_meta = meta[meta['form_type'] == '8-K']
    if not eightk_meta.empty:
        banner('3. one saved 8-K item (date | item number prepended)')
        directory = Path(eightk_meta.iloc[0]['disk_path'])
        files = sorted(directory.glob('*.txt'))
        if files:
            print(f'{directory.name}/{files[0].name}:')
            print(files[0].read_text()[:400])

    banner(f'4. get_form4({ticker!r}) -- insider transactions')
    form4 = db.get_form4(ticker)
    print(f'{len(form4)} transaction rows')
    if not form4.empty:
        cols = ['transaction_date', 'insider', 'position', 'transaction_code',
                'acquired_disposed', 'shares', 'price', 'value', 'is_derivative']
        print(form4[cols].tail(8).to_string(index=False))

    banner(f"5a. get_income({ticker!r}, interval='quarterly', wide=True) -- Q4 derived")
    print(db.get_income(ticker, interval='quarterly', wide=True).tail(8))
    print('\nderived rows (computed Q4 = annual - Q1..Q3):')
    income = db.get_income(ticker)
    print(income[income['derived'].astype(bool)].tail(4).to_string(index=False))

    banner(f"5b. get_income({ticker!r}, interval='annual', wide=True)")
    print(db.get_income(ticker, interval='annual', wide=True).tail(4))

    banner(f'5c. get_balance({ticker!r}, wide=True) -- point-in-time, no interval')
    print(db.get_balance(ticker, wide=True).tail(4))

    banner(f'5d. get_cashflow({ticker!r}) -- cumulative windows with duration')
    cashflow = db.get_cashflow(ticker, items=['operating_cash_flow'])
    print(cashflow.tail(6).to_string(index=False))
    print("\n(derived=True rows are de-cumulated quarters; populated when the "
          "pipeline runs with decumulate_cashflow=True)")

    banner(f"6a. get_financials({ticker!r}, interval='annual') -- indexed by fiscal_year")
    print(db.get_financials(ticker, interval='annual'))

    banner(f"6b. get_financials({ticker!r}, interval='quarterly') -- (fiscal_year, fiscal_quarter)")
    quarterly = db.get_financials(ticker, interval='quarterly')
    print(f'shape: {quarterly.shape} | columns: {list(quarterly.columns)}')
    print(quarterly[['revenue', 'net_income', 'eps_diluted', 'assets', 'operating_cash_flow']].tail(8))


if __name__ == '__main__':
    main()
