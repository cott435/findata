"""Shared loading/plumbing for the feature-structure analysis scripts.

These are DESCRIPTIVE structure studies (the feature-centered sibling of
scripts/market_structure): everything is fit over the FULL sample — all dates
T, all tickers N — not a train window. The leakage-safe production version of
the mode decomposition is ``findata.preprocess.market_modes.ModeDecomposer``
used inside the pipeline; here it is fit on the whole panel purely to
characterize structure.

Two studies sit on two axes of the (time x ticker x feature) cube:

  feature axis (F x F)   feature_correlation_structure.py — how the engineered
                         features correlate, via CorrelationStructureAnalysis,
                         and whether removing the cross-sectional global/sector
                         market modes changes it (ModeDecomposer axis="ticker").
  ticker axis  (N x N)   mp_information_search.py — the Marchenko-Pastur signal
                         content of the N-stock cross-section, measured on
                         returns AND on each feature (market_structure).

Both expose base / global_removed / sector_removed representations.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from findata import build_features, get_all_data, get_all_tickers  # noqa: E402
from findata.configs import EXPERIMENT_DIR, DataSplits              # noqa: E402
from findata.preprocess import ModeDecomposer                       # noqa: E402

OUT_ROOT = EXPERIMENT_DIR / "feature_structure"
REPRESENTATIONS = ("base", "global_removed", "sector_removed")


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", type=int, default=None,
                   help="first N tickers (default: all in DB)")
    p.add_argument("--feature-set", default="med", choices=["low", "med", "high"])
    p.add_argument("--start", default="2021-01-01", help="ticker-axis panel start")
    p.add_argument("--end", default=None, help="panel end date (default: all)")
    p.add_argument("--min-sector-size", type=int, default=3,
                   help="skip the sector stage for sectors smaller than this")
    p.add_argument("--bw", type=float, default=0.1, help="MP KDE bandwidth (legacy fits)")
    p.add_argument("--db-path", default=None)
    return p


def resolve_tickers(args) -> list[str]:
    tickers = get_all_tickers(db_path=args.db_path)
    return tickers[:args.tickers] if args.tickers else tickers


def load_raw(args):
    """Raw OHLCV+indicator panel + ticker_info + DataSplits (full range)."""
    splits = DataSplits()
    tickers = resolve_tickers(args)
    print(f"Loading {len(tickers)} tickers: {splits.data_start} -> {splits.data_end}")
    raw, info = get_all_data(tickers, splits.data_start, splits.data_end, db_path=args.db_path)
    return raw, info, splits


def feature_reps(raw, info, splits, args) -> tuple[dict, dict]:
    """base / global_removed / sector_removed FEATURE panels (fit over all T, N).

    Returns ({representation: (ticker,date) panel}, provenance). The global and
    sector panels are the ModeDecomposer residuals (name-preserving), so all
    three share the same feature columns for a like-for-like F x F comparison.
    """
    fb = build_features(raw, splits, feature_set=args.feature_set, ticker_meta=info)
    base = fb.features
    print(f"feature panel: {base.shape[0]:,} (ticker,date) rows x {base.shape[1]} features")

    g = ModeDecomposer(group_stage=False, output="residual",
                       min_group_size=args.min_sector_size)
    g.bind_meta(info)
    global_removed = g.fit_transform(base)

    s = ModeDecomposer(group_stage=True, output="residual",
                       min_group_size=args.min_sector_size)
    s.bind_meta(info)
    sector_removed = s.fit_transform(base)
    print(f"sector stage groups: {s.describe().get('groups')}")

    return ({"base": base, "global_removed": global_removed,
             "sector_removed": sector_removed}, fb.provenance)
