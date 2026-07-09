"""
rank_ic_analysis.py

Computes cross-sectional Rank IC (Spearman) for a bank of features against
forward log returns, and produces summary deliverables (table + plots).

Expected inputs
----------------
features_df : pd.DataFrame
    MultiIndex (ticker, date), columns = feature names (numeric).
closes : pd.Series or pd.DataFrame with a single 'close' column
    MultiIndex (ticker, date), values = close prices.
H : int
    Forward horizon in periods (e.g. 10 trading days).

Outputs
-------
- fwd_log_returns.csv        forward H-period log returns per (ticker, date)
- daily_ic.csv                date x feature matrix of daily cross-sectional Rank IC
- ic_summary.csv              per-feature IC mean, std, ICIR, t-stat, p-value, n_days
- cumulative_ic.png           cumulative sum of daily IC per feature (top N by |ICIR|)
- icir_bar.png                bar chart of ICIR ranked

Usage
-----
    from rank_ic_analysis import run_rank_ic_pipeline

    results = run_rank_ic_pipeline(
        features_df=features_df,
        closes=closes,
        H=10,
        min_assets=20,
        output_dir="rank_ic_outputs",
    )
    results["ic_summary"]   # ranked summary table
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# 1. Forward log returns
# --------------------------------------------------------------------------

def compute_forward_log_returns(
    closes: pd.Series | pd.DataFrame,
    H: int,
    ticker_level: str | int = 0,
    date_level: str | int = 1,
) -> pd.Series:
    """
    Transform close prices into forward H-period log returns, computed
    independently per ticker (so returns never bleed across ticker boundaries).

    fwd_log_ret_t = log(price_{t+H} / price_t)

    Parameters
    ----------
    closes : Series or single-column DataFrame, MultiIndex (ticker, date)
    H : forward horizon in periods
    ticker_level, date_level : index level names/positions for ticker & date

    Returns
    -------
    pd.Series named 'fwd_log_ret', same MultiIndex as input, with the last
    H observations per ticker set to NaN (no forward data available).
    """
    if isinstance(closes, pd.DataFrame):
        if closes.shape[1] != 1:
            raise ValueError("closes DataFrame must have exactly one column of prices")
        closes = closes.iloc[:, 0]
    closes = closes.copy()
    closes.name = "close"

    # Ensure sorted by (ticker, date) so shifting works correctly per group
    closes = closes.sort_index(level=[ticker_level, date_level])

    log_price = np.log(closes)

    def _fwd_ret(group: pd.Series) -> pd.Series:
        return group.shift(-H) - group

    fwd_log_ret = log_price.groupby(level=ticker_level, group_keys=False).apply(_fwd_ret)
    fwd_log_ret.name = "fwd_log_ret"
    return fwd_log_ret


# --------------------------------------------------------------------------
# 2. Daily cross-sectional Rank IC
# --------------------------------------------------------------------------

def compute_daily_rank_ic(
    features_df: pd.DataFrame,
    fwd_ret: pd.Series,
    min_assets: int = 20,
    ticker_level: str | int = 0,
    date_level: str | int = 1,
) -> pd.DataFrame:
    """
    For each date, compute the Spearman rank correlation between each
    feature and the forward return, across all tickers available that date.

    Parameters
    ----------
    features_df : DataFrame, MultiIndex (ticker, date), numeric feature columns
    fwd_ret : Series, same MultiIndex, forward log returns
    min_assets : minimum number of non-NaN cross-sectional observations
                 required on a date to compute IC that day (else NaN)

    Returns
    -------
    pd.DataFrame indexed by date, columns = feature names, values = daily Rank IC
    """
    panel = features_df.copy()
    panel["__fwd_ret__"] = fwd_ret.reindex(panel.index)

    feature_cols = [c for c in panel.columns if c != "__fwd_ret__"]

    records = {}
    for date, group in panel.groupby(level=date_level):
        y = group["__fwd_ret__"]
        valid_y = y.notna()
        row = {}
        for col in feature_cols:
            x = group[col]
            mask = valid_y & x.notna()
            n = mask.sum()
            if n < min_assets:
                row[col] = np.nan
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic, _ = spearmanr(x[mask], y[mask])
            row[col] = ic
        records[date] = row

    daily_ic = pd.DataFrame.from_dict(records, orient="index")
    daily_ic.index.name = "date"
    daily_ic = daily_ic.sort_index()
    return daily_ic


# --------------------------------------------------------------------------
# 3. Summarize IC (mean, ICIR, autocorrelation-adjusted t-stat)
# --------------------------------------------------------------------------

def summarize_ic(daily_ic: pd.DataFrame, H: int) -> pd.DataFrame:
    """
    Aggregate daily IC into per-feature summary stats.

    Because forward returns overlap across H periods, consecutive daily IC
    values are autocorrelated. We approximate the effective sample size as
    n_eff = n_obs / H (a standard, conservative overlap correction) when
    computing the t-stat, so significance isn't overstated.

    Returns
    -------
    pd.DataFrame indexed by feature, sorted by |ICIR| descending, columns:
        ic_mean, ic_std, icir, n_days, n_eff, t_stat, p_value
    """
    rows = []
    for col in daily_ic.columns:
        series = daily_ic[col].dropna()
        n_days = len(series)
        if n_days < 2:
            rows.append({
                "feature": col, "ic_mean": np.nan, "ic_std": np.nan,
                "icir": np.nan, "n_days": n_days, "n_eff": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
            })
            continue

        ic_mean = series.mean()
        ic_std = series.std(ddof=1)
        icir = ic_mean / ic_std if ic_std > 0 else np.nan

        n_eff = max(n_days / H, 2.0)
        se = ic_std / np.sqrt(n_eff) if ic_std > 0 else np.nan
        t_stat = ic_mean / se if se and se > 0 else np.nan
        p_value = (
            2 * (1 - t_dist.cdf(abs(t_stat), df=max(n_eff - 1, 1)))
            if t_stat is not None and not np.isnan(t_stat)
            else np.nan
        )

        rows.append({
            "feature": col,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "n_days": n_days,
            "n_eff": n_eff,
            "t_stat": t_stat,
            "p_value": p_value,
        })

    summary = pd.DataFrame(rows).set_index("feature")
    summary = summary.reindex(summary["icir"].abs().sort_values(ascending=False).index)
    return summary


# --------------------------------------------------------------------------
# 4. Plots
# --------------------------------------------------------------------------

def plot_cumulative_ic(daily_ic: pd.DataFrame, summary: pd.DataFrame, top_n: int, path: str):
    top_features = summary.head(top_n).index.tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    for feat in top_features:
        cum = daily_ic[feat].fillna(0).cumsum()
        ax.plot(cum.index, cum.values, label=feat, linewidth=1.4)
    ax.set_title(f"Cumulative Daily Rank IC — Top {top_n} Features by |ICIR|")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative IC")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_icir_bar(summary: pd.DataFrame, top_n: int, path: str):
    top = summary.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * top_n)))
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in top["icir"]]
    ax.barh(top.index[::-1], top["icir"][::-1], color=colors[::-1])
    ax.set_title(f"ICIR by Feature — Top {top_n}")
    ax.set_xlabel("Information Coefficient IR (mean IC / std IC)")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Full pipeline
# --------------------------------------------------------------------------

def run_rank_ic_pipeline(
    features_df: pd.DataFrame,
    closes: pd.Series | pd.DataFrame,
    H: int,
    min_assets: int = 20,
    top_n_plot: int = 15,
    output_dir: str = "rank_ic_outputs",
    ticker_level: str | int = 0,
    date_level: str | int = 1,
) -> dict:
    """
    Run the full Rank IC pipeline end to end and write deliverables to disk.

    Returns a dict with keys: 'fwd_log_ret', 'daily_ic', 'ic_summary'
    (all also written to output_dir as CSV/PNG).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Computing forward log returns (H={H})...")
    fwd_ret = compute_forward_log_returns(closes, H, ticker_level, date_level)
    fwd_ret.to_frame().to_csv(os.path.join(output_dir, "fwd_log_returns.csv"))

    print(f"[2/4] Computing daily cross-sectional Rank IC (min_assets={min_assets})...")
    daily_ic = compute_daily_rank_ic(features_df, fwd_ret, min_assets, ticker_level, date_level)
    daily_ic.to_csv(os.path.join(output_dir, "daily_ic.csv"))

    print("[3/4] Summarizing IC per feature...")
    summary = summarize_ic(daily_ic, H)
    summary.to_csv(os.path.join(output_dir, "ic_summary.csv"))

    print("[4/4] Generating plots...")
    n_plot = min(top_n_plot, summary.shape[0])
    plot_cumulative_ic(daily_ic, summary, n_plot, os.path.join(output_dir, "cumulative_ic.png"))
    plot_icir_bar(summary, n_plot, os.path.join(output_dir, "icir_bar.png"))

    print(f"\nDone. Deliverables written to: {os.path.abspath(output_dir)}")
    print("\n=== IC Summary (ranked by |ICIR|) ===")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.round(4))

    return {"fwd_log_ret": fwd_ret, "daily_ic": daily_ic, "ic_summary": summary}


# --------------------------------------------------------------------------
# Example usage / smoke test with synthetic data
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from findata.configs import EXPERIMENT_DIR
    from scripts.testing.momentum_analysis import build_momentum_features

    from findata import get_all_tickers, get_price_data

    tickers = get_all_tickers()[:30]
    data = get_price_data(tickers)
    closes = data["close"]
    INDICATOR = 'rsi'

    def calc(dat: pd.Series) -> pd.DataFrame:
        momentum_features, momentum_intermediates = build_momentum_features(dat, oscillator_name=INDICATOR)
        return momentum_features

    features_df = closes.groupby("ticker", group_keys=False).apply(calc).dropna()



    results = run_rank_ic_pipeline(
        features_df=features_df,
        closes=closes,
        H=10,
        min_assets=20,
        output_dir=EXPERIMENT_DIR / 'rank_ic_analysis',
    )