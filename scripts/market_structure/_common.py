"""Shared loading/argument plumbing for the market-structure question scripts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from findata.analysis import market_structure as ms   # noqa: E402
from findata.configs import EXPERIMENT_DIR             # noqa: E402

OUT_ROOT = EXPERIMENT_DIR / "market_structure"


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--start", default="2021-01-01", help="panel start date")
    p.add_argument("--end", default=None, help="panel end date (default: all)")
    p.add_argument("--tickers", type=int, default=None, help="first N tickers (default: all)")
    p.add_argument("--indicator", default="log_return",
                   help="log_return | rsi | cci | willr | mfi | cmf")
    p.add_argument("--period", type=int, default=None,
                   help="indicator period for non-return indicators (e.g. 14 for rsi)")
    p.add_argument("--n-draws", type=int, default=300, help="permutation/null draws")
    p.add_argument("--max-lag", type=int, default=10, help="lead-lag search window (days)")
    p.add_argument("--seed", type=int, default=0)
    return p


def load_panel(args):
    from findata import get_all_tickers

    tickers = get_all_tickers()
    if args.tickers:
        tickers = tickers[:args.tickers]
    kw = {"period": args.period} if args.period else None
    wide, meta = ms.indicator_panel(tickers, start=args.start, end=args.end,
                                    indicator=args.indicator, indicator_kwargs=kw)
    print(f"panel: T={len(wide)} dates x N={wide.shape[1]} tickers, "
          f"indicator={args.indicator}, {wide.index[0]} -> {wide.index[-1]}")
    return wide, meta


def sector_groups(meta, min_size: int = 6) -> dict:
    groups = {s: list(idx) for s, idx in meta.groupby("sector").groups.items()}
    return {s: m for s, m in groups.items() if len(m) >= min_size}


def cap_groups(meta) -> dict:
    return {f"cap_{b}": list(idx) for b, idx in meta.groupby("cap_bucket", observed=True).groups.items()}
