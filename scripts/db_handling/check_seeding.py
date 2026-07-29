"""Seeding efficacy check for the technical calculators.

Cold-starts the full daily history for one ticker, then seeds from the last
bar before --cutoff and recomputes onward only. The two results should agree
to numerical noise. Lives here (not in the calculator module's __main__)
because ``python -m`` would execute a registering module twice.

Run from the project root:
    python -m scripts.db_handling.check_seeding [--ticker A] [--cutoff 2020-01-01]
"""

import argparse

import pandas as pd

from findata.database.yf import YahooFinance
from findata.preprocess.calculators.technical import build_seeds, calculate_all

WARMUP_BARS = 150  # trailing bars so rolling-window indicators are exact


def main():
    parser = argparse.ArgumentParser(description='Check seeded incremental parity.')
    parser.add_argument('--ticker', default='A')
    parser.add_argument('--cutoff', default='2020-01-01')
    args = parser.parse_args()
    ticker = args.ticker.upper()
    cutoff = pd.Timestamp(args.cutoff).date()

    print(f'Pulling full daily history for {ticker}...')
    data = YahooFinance(ticker).request_ticker_financials()
    prices = data['prices'].xs(('daily', ticker))

    full = calculate_all(prices)
    print(f'Cold-start: {full.shape[0]} rows x {full.shape[1]} columns '
          f'({prices.index[0]} -> {prices.index[-1]})')

    history = full[full.index < cutoff]
    seeds = build_seeds(history)
    seed_date = history.index[-1]
    seed_pos = prices.index.get_loc(seed_date)
    window = prices.iloc[max(seed_pos - WARMUP_BARS, 0):]

    seeded = calculate_all(window, seeds=seeds)
    seeded = seeded[seeded.index >= cutoff]
    print(f'Seeded from {seed_date}: recomputed {seeded.shape[0]} rows '
          f'({len(seeds)} seed values, {WARMUP_BARS} warmup bars)')

    expected = full[full.index >= cutoff]
    diff = (expected - seeded).abs()
    summary = pd.DataFrame({
        'max_abs_diff': diff.max(),
        'mean_abs_diff': diff.mean(),
        'last_full': expected.iloc[-1],
        'last_seeded': seeded.iloc[-1],
    }).sort_values('max_abs_diff', ascending=False)

    with pd.option_context('display.max_rows', None, 'display.width', 200,
                           'display.float_format', '{:,.6g}'.format):
        print(f'\nFull-history vs seeded recomputation ({cutoff} onward):')
        print(summary)


if __name__ == '__main__':
    main()
