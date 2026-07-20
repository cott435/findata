"""Cross-sectional Rank IC analysis (port of scripts/testing/rank_ic_analysis.py,
fully vectorized — no per-date Python loops).

Spearman rank IC per (date, feature) is Pearson on within-date ranks, so the
whole date x feature IC matrix reduces to groupby rank / transform / sum ops:

    rx  = rank of each feature within date       (average ties == spearmanr)
    ry  = rank of forward returns within date
    ic(date, f) = sum(rxc * ryc) / sqrt(sum(rxc^2) * sum(ryc^2))   per date

The overlap-corrected significance follows the original script: consecutive
daily ICs of an H-day horizon overlap, so n_eff = n_days / H (conservative).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


@dataclass
class TargetRanks:
    """Precomputed y-side of the IC computation for one horizon (reusable across
    feature panels sharing the same index — the search exploits this)."""
    index: pd.MultiIndex        # rows where fwd_ret is valid
    ryc: pd.Series              # centered within-date ranks of fwd returns
    deny: pd.Series             # per-date sum(ryc^2)
    counts: pd.Series           # per-date cross-section size
    fwd_ret: pd.Series          # valid forward returns (aligned to index)


class RankICAnalysis:

    def __init__(self, min_assets: int = 10, top_n_plot: int = 15):
        self.min_assets = min_assets
        self.top_n_plot = top_n_plot

    # ------------------------------------------------------------------ #
    # forward returns
    # ------------------------------------------------------------------ #

    @staticmethod
    def forward_log_returns(closes: pd.Series | pd.DataFrame, H: int) -> pd.Series:
        """log(price_{t+H} / price_t) per ticker (no bleed across boundaries);
        the last H rows per ticker are NaN."""
        if isinstance(closes, pd.DataFrame):
            if closes.shape[1] != 1:
                raise ValueError("closes DataFrame must have exactly one price column")
            closes = closes.iloc[:, 0]
        closes = closes.sort_index()
        log_price = np.log(closes)
        fwd = log_price.groupby(level="ticker").shift(-H) - log_price
        fwd.name = "fwd_log_ret"
        return fwd

    # ------------------------------------------------------------------ #
    # daily rank IC
    # ------------------------------------------------------------------ #

    def prepare_target(self, fwd_ret: pd.Series, index: pd.MultiIndex) -> TargetRanks:
        """Rank/center the forward returns once for a given feature-panel index."""
        y = fwd_ret.reindex(index)
        y = y[y.notna()]
        ry = y.groupby(level="date").rank()
        ryc = ry - ry.groupby(level="date").transform("mean")
        deny = (ryc ** 2).groupby(level="date").sum()
        counts = ry.groupby(level="date").size()
        return TargetRanks(index=y.index, ryc=ryc, deny=deny, counts=counts, fwd_ret=y)

    def daily_ic(self, features: pd.DataFrame, target: pd.Series | TargetRanks,
                 method: str = "vectorized") -> pd.DataFrame:
        """date x feature matrix of daily cross-sectional Spearman rank IC.

        ``target`` is the forward-return Series or a prepared TargetRanks.
        Dates with fewer than ``min_assets`` valid pairs are NaN.
        """
        if not isinstance(target, TargetRanks):
            target = self.prepare_target(target, features.index)
        if method == "spearmanr":
            return self._daily_ic_spearman_loop(features, target)

        X = features.reindex(target.index)
        if not X.isna().any().any():
            ic = self._ic_all_columns(X, target)
        else:
            # per-column masks (mid-chain NaNs, e.g. rolling-vol heads):
            # same vectorized math per column, never per date
            cols = {}
            for col in X.columns:
                cols[col] = self._ic_one_column(X[col], target.fwd_ret)
            ic = pd.DataFrame(cols)
        ic.index = pd.to_datetime(ic.index)
        ic.index.name = "date"
        return ic.sort_index()

    def _ic_all_columns(self, X: pd.DataFrame, target: TargetRanks) -> pd.DataFrame:
        """Fast joint path: every column shares the target's valid-row mask."""
        rx = X.groupby(level="date").rank()
        rxc = rx - rx.groupby(level="date").transform("mean")
        num = rxc.mul(target.ryc, axis=0).groupby(level="date").sum()
        denx = (rxc ** 2).groupby(level="date").sum()
        den = np.sqrt(denx.mul(target.deny, axis=0))
        ic = num / den.where(den > 0)
        return ic.where(target.counts >= self.min_assets, axis=0)

    def _ic_one_column(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """Vectorized single-column path honoring this column's own NaN mask
        (Spearman requires ranking within rows where BOTH x and y are valid)."""
        m = x.notna()
        xs, ys = x[m], y[m]
        rx = xs.groupby(level="date").rank()
        ry = ys.groupby(level="date").rank()
        rxc = rx - rx.groupby(level="date").transform("mean")
        ryc = ry - ry.groupby(level="date").transform("mean")
        num = (rxc * ryc).groupby(level="date").sum()
        den = np.sqrt((rxc ** 2).groupby(level="date").sum()
                      * (ryc ** 2).groupby(level="date").sum())
        counts = rx.groupby(level="date").size()
        ic = num / den.where(den > 0)
        return ic.where(counts >= self.min_assets)

    def _daily_ic_spearman_loop(self, features: pd.DataFrame, target: TargetRanks) -> pd.DataFrame:
        """Original per-date scipy loop — kept ONLY as a parity check for the
        vectorized path (slow; do not use in the search)."""
        from scipy.stats import spearmanr

        panel = features.reindex(target.index).copy()
        panel["__fwd_ret__"] = target.fwd_ret
        feature_cols = [c for c in panel.columns if c != "__fwd_ret__"]
        records = {}
        for date, group in panel.groupby(level="date"):
            y = group["__fwd_ret__"]
            row = {}
            for col in feature_cols:
                x = group[col]
                mask = x.notna() & y.notna()
                if mask.sum() < self.min_assets:
                    row[col] = np.nan
                    continue
                row[col] = spearmanr(x[mask], y[mask])[0]
            records[date] = row
        out = pd.DataFrame.from_dict(records, orient="index")
        out.index = pd.to_datetime(out.index)
        out.index.name = "date"
        return out.sort_index()

    # ------------------------------------------------------------------ #
    # summary (vectorized — no per-feature loop)
    # ------------------------------------------------------------------ #

    def summarize(self, daily_ic: pd.DataFrame, H: int) -> pd.DataFrame:
        """Per-feature ic_mean, ic_std, icir, n_days, n_eff, t_stat, p_value,
        sorted by |ICIR| descending. n_eff = max(n_days/H, 2) corrects for the
        H-day overlap of consecutive forward returns."""
        n_days = daily_ic.notna().sum()
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std(ddof=1)
        icir = ic_mean / ic_std.where(ic_std > 0)
        n_eff = (n_days / H).clip(lower=2.0)
        se = ic_std.where(ic_std > 0) / np.sqrt(n_eff)
        t_stat = ic_mean / se
        p_value = pd.Series(
            2 * t_dist.sf(np.abs(t_stat.to_numpy(dtype=float)),
                          df=np.clip(n_eff.to_numpy(dtype=float) - 1, 1, None)),
            index=t_stat.index)
        summary = pd.DataFrame({
            "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
            "n_days": n_days, "n_eff": n_eff, "t_stat": t_stat, "p_value": p_value,
        })
        summary.loc[n_days < 2, ["ic_mean", "ic_std", "icir", "n_eff", "t_stat", "p_value"]] = np.nan
        summary.index.name = "feature"
        return summary.reindex(summary["icir"].abs().sort_values(ascending=False).index)

    # ------------------------------------------------------------------ #
    # plots + pipeline (same artifacts as the original script)
    # ------------------------------------------------------------------ #

    def plot_cumulative_ic(self, daily_ic: pd.DataFrame, summary: pd.DataFrame,
                           top_n: int, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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

    def plot_icir_bar(self, summary: pd.DataFrame, top_n: int, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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

    def run(self, features: pd.DataFrame, closes: pd.Series | pd.DataFrame, H: int,
            output_dir: str | Path) -> dict:
        """Full pipeline: forward returns -> daily IC -> summary -> plots, with
        the same five artifacts as the original script."""
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[1/4] Forward log returns (H={H})...")
        fwd_ret = self.forward_log_returns(closes, H)
        fwd_ret.to_frame().to_csv(output_dir / "fwd_log_returns.csv")

        print(f"[2/4] Daily cross-sectional rank IC (min_assets={self.min_assets})...")
        daily_ic = self.daily_ic(features, fwd_ret)
        daily_ic.to_csv(output_dir / "daily_ic.csv")

        print("[3/4] Per-feature summary...")
        summary = self.summarize(daily_ic, H)
        summary.to_csv(output_dir / "ic_summary.csv")

        print("[4/4] Plots...")
        n_plot = min(self.top_n_plot, summary.shape[0])
        self.plot_cumulative_ic(daily_ic, summary, n_plot, output_dir / "cumulative_ic.png")
        self.plot_icir_bar(summary, n_plot, output_dir / "icir_bar.png")

        print(f"Done. Deliverables written to: {output_dir.resolve()}")
        return {"fwd_log_ret": fwd_ret, "daily_ic": daily_ic, "ic_summary": summary}


if __name__ == "__main__":
    # Parity check: vectorized vs scipy spearmanr loop on real data
    from findata import get_all_tickers, get_price_data

    tickers = get_all_tickers()[:5]
    closes = get_price_data(tickers)["close"]
    rng = np.random.default_rng(3)
    feats = pd.DataFrame({
        "f_mom": closes.groupby(level="ticker").pct_change(10),
        "f_noise": rng.normal(size=len(closes)),
        "f_level": np.log(closes),
    }, index=closes.index).dropna()

    ana = RankICAnalysis(min_assets=4)
    fwd = ana.forward_log_returns(closes, H=10)
    fast = ana.daily_ic(feats, fwd)
    slow = ana.daily_ic(feats, fwd, method="spearmanr")
    aligned = fast.align(slow)
    diff = (aligned[0] - aligned[1]).abs().max().max()
    print(f"vectorized vs spearmanr: max |diff| = {diff:.2e} (want < 1e-12)")
    print(ana.summarize(fast, H=10).round(4))
