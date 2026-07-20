"""Reproducible ticker-universe sampling for training/search runs.

The forecasting default of a handful of tickers is degenerate for
cross-sectional (PCA mode) work — sector blocks need members. This sampler
draws a liquid, sector-stratified universe straight from the price DB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import func, select


def sample_universe(n: int = 150, start=None, end=None, min_coverage: float = 0.9,
                    seed: int = 42, exclude_sectors=("ETF",),
                    db_path=None) -> list[str]:
    """Deterministic sample of ``n`` tickers with dense price coverage.

    Coverage = stored daily bars in [start, end] relative to the best-covered
    ticker (defaults: the DataSplits train window through validation end, so
    every pick has history for fitting AND validation). Sampling is
    proportional by sector (largest-remainder allocation) with a fixed seed,
    so the same arguments always return the same universe.
    """
    from findata.configs import DataSplits
    from findata.database.db_manager import DBManager
    from findata.database.tables import PriceData

    splits = DataSplits()
    start = splits.train_start if start is None else start
    end = splits.validation_end if end is None else end

    db = DBManager(if_missing="raise", db_path=db_path)
    stmt = (select(PriceData.ticker, func.count().label("bars"))
            .where(PriceData.interval == "daily",
                   PriceData.date >= start, PriceData.date <= end)
            .group_by(PriceData.ticker))
    bars = pd.read_sql(stmt, db.engine).set_index("ticker")["bars"]
    if bars.empty:
        raise ValueError(f"No daily price rows between {start} and {end}")
    covered = bars[bars >= min_coverage * bars.max()]

    sectors = db.get_ticker_meta(list(covered.index))["sector"].fillna("ETF")
    keep = sectors[~sectors.isin(set(exclude_sectors or ()))]
    if len(keep) <= n:
        return sorted(keep.index)

    rng = np.random.default_rng(seed)
    sizes = keep.value_counts()
    exact = sizes / sizes.sum() * n
    alloc = exact.astype(int)
    for sector in exact.sub(alloc).sort_values(ascending=False).index:
        if alloc.sum() >= n:
            break
        if alloc[sector] < sizes[sector]:
            alloc[sector] += 1

    picked: list[str] = []
    for sector, k in alloc.items():
        pool = sorted(keep.index[keep == sector])
        if k > 0:
            picked += [str(t) for t in
                       rng.choice(pool, size=min(int(k), len(pool)), replace=False)]
    return sorted(picked)
