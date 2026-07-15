#!/usr/bin/env python
"""Run the full feature-processing search + correlation-structure analysis.

Examples:
    .venv/bin/python scripts/run_feature_search.py --tickers 15 --quick
    .venv/bin/python scripts/run_feature_search.py                       # full grid
    .venv/bin/python scripts/run_feature_search.py --oscillators rsi cci --horizons 5 21 63
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from findata import get_all_data, get_all_tickers
from findata.analysis import CorrelationStructureAnalysis, ProcessingSearch
from findata.analysis.search import DEFAULT_VARIANTS
from findata.configs import EXPERIMENT_DIR, DataSplits
from findata.preprocess import Candle, Oscillators, Trend, Volatility, Volume

QUICK_VARIANTS = {"raw+none", "kalman+none", "raw+zca", "raw+hpca", "demean", "signal+demean"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", type=int, default=None,
                   help="limit to the first N tickers (default: all)")
    p.add_argument("--quick", action="store_true",
                   help="small variant subset + horizons [5, 21]")
    p.add_argument("--horizons", type=int, nargs="*", default=None,
                   help="forward horizons in trading days (default: 1 5 10 21 63)")
    p.add_argument("--feature-set", choices=("low", "med", "high"), default="med")
    p.add_argument("--oscillators", nargs="*", default=None,
                   help="momentum oscillators, e.g. --oscillators rsi cci")
    p.add_argument("--min-assets", type=int, default=10)
    p.add_argument("--save-states", action="store_true",
                   help="save a fitted PipelineState per variant")
    return p.parse_args()


def main():
    args = parse_args()
    splits = DataSplits()
    args.tickers = 70
    tickers = get_all_tickers()
    if args.tickers:
        tickers = tickers[:args.tickers]
    print(f"Loading {len(tickers)} tickers: {splits.data_start} -> {splits.data_end}")
    data, _ = get_all_data(tickers, splits.data_start, splits.data_end)

    groups = None
    if args.oscillators:
        groups = [Oscillators(oscillators=args.oscillators, feature_set=args.feature_set),
                  Trend, Volatility, Volume, Candle]

    horizons = args.horizons or ([5, 21] if args.quick else [1, 5, 10, 21, 63])
    variants = ([v for v in DEFAULT_VARIANTS if v.name in QUICK_VARIANTS]
                if args.quick else None)

    search = ProcessingSearch(data, splits, variants=variants, horizons=horizons,
                              groups=groups, feature_set=args.feature_set,
                              min_assets=args.min_assets)
    search.run(save_states=args.save_states)

    # headline: validation top-5 mean |ICIR| per variant x horizon
    val = search.summary_[search.summary_["split"] == "val"]
    pivot = val.pivot(index="variant", columns="horizon", values="top5_mean_abs_icir")
    pivot = pivot.reindex([v.name for v in search.variants])
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print("\n=== Validation top-5 mean |ICIR| (variant x horizon) ===")
        print(pivot)
        print("\n=== Variant spectrum (train panel) ===")
        meta = pd.DataFrame(search.variant_meta_).T
        print(meta[["n_features", "causal", "n_signal", "lam_plus", "top_eig_share"]])

    # correlation-structure analysis on the kept panels (baseline + market-removed)
    for name, panel in search.panels_.items():
        out_dir = EXPERIMENT_DIR / "feature_structure" / name.replace("+", "_")
        print(f"\n--- Structure analysis: {name} -> {out_dir} ---")
        CorrelationStructureAnalysis(output_dir=out_dir).run(panel)

    print(f"\nAll search artifacts: {search.output_dir.resolve()}")


if __name__ == "__main__":
    main()
