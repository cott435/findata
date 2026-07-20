#!/usr/bin/env python
"""Rank IC decomposition: where does a feature's predictive power live — in
predicting the market mode, the sector mode, or idiosyncratic residual return?

Cross-sectional Spearman rank IC (feature vs forward return) is computed for
three representations, neutralizing BOTH sides consistently:

  base            all features            -> all forward returns (N stocks)
  global_removed  market-mode-removed     -> market-neutral forward returns
                  features                   (residual after PC1)
  sector_removed  global+sector-removed   -> sector-neutral forward returns
                  features                   (residual after each sector's PC1)

Feature neutralization is ModeDecomposer(axis="ticker") (per-feature global +
sector modes removed across the ticker cross-section); return neutralization is
the market_structure residual return series, forward-summed over the horizon.

Reading it: a feature whose |ICIR| is high on `base` but collapses on
`global_removed` was largely predicting the market mode (systematic timing) —
once the mode is stripped from both the feature and the target there is nothing
left to predict. A feature whose |ICIR| survives (or strengthens, as market
noise is removed from the target) on the residual returns is genuine
cross-sectional stock selection. Everything is fit over the full sample.

    .venv/bin/python scripts/feature_structure/rank_ic_decomposition.py --tickers 250
    .venv/bin/python scripts/feature_structure/rank_ic_decomposition.py --horizons 5 21 --plot-horizon 21
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUT_ROOT, REPRESENTATIONS, build_parser, feature_reps, load_raw

from findata.analysis import market_structure as ms
from findata.analysis.rank_ic import RankICAnalysis

OUT = OUT_ROOT / "rank_ic_decomposition"


def forward_target(wide: pd.DataFrame, H: int) -> pd.Series:
    """Forward sum of a wide date x ticker (residual) return panel over the next
    H days -> (ticker, date) Series (last H rows per ticker NaN, no bleed)."""
    fwd = wide.rolling(H).sum().shift(-H)
    s = fwd.stack()
    s.index = s.index.set_names(["date", "ticker"]).reorder_levels(["ticker", "date"])
    return s.sort_index().dropna()


def return_targets(info, splits, args) -> dict:
    """base / global_removed / sector_removed wide daily-return panels, aligned to
    the feature date span (from train_start)."""
    wide, meta = ms.indicator_panel(list(info.index), start=str(splits.train_start),
                                    end=args.end, indicator="log_return", db_path=args.db_path)
    print(f"return panel: T={len(wide)} dates x N={wide.shape[1]} tickers, "
          f"{wide.index[0]} -> {wide.index[-1]}")
    return {"base": wide,
            "global_removed": ms.remove_top_pcs(wide, 1),
            "sector_removed": ms.remove_group_modes(wide, meta["sector"])}


def plot_icir_comparison(pivot: pd.DataFrame, H: int, path):
    """Grouped bars: top features by base |ICIR|, ICIR under each representation."""
    top = pivot.reindex(pivot["base"].abs().sort_values(ascending=False).index).head(18)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.5 * len(top))))
    y = np.arange(len(top))
    h = 0.26
    for i, rep in enumerate(REPRESENTATIONS):
        ax.barh(y + (1 - i) * h, top[rep].to_numpy(), height=h, label=rep)
    ax.set_yticks(y); ax.set_yticklabels(top.index, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("ICIR (mean IC / std IC)")
    ax.set_title(f"Rank ICIR by representation (H={H}) — top features by base |ICIR|\n"
                 "shrink from base = predictive power that lived in the market/sector mode")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_attribution(pivot: pd.DataFrame, H: int, path):
    """Scatter base |ICIR| vs sector-neutral |ICIR|: on/above the diagonal =
    IC survives neutralization (real selection); far below = market/sector-driven."""
    b = pivot["base"].abs()
    s = pivot["sector_removed"].abs()
    lim = float(max(b.max(), s.max())) * 1.05
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(b, s, s=28, alpha=0.6, color="#4c72b0", edgecolor="k", linewidth=0.4)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="IC unchanged by neutralization")
    for f in b.sort_values(ascending=False).head(12).index:
        ax.annotate(f, (b[f], s[f]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("|ICIR| — base (raw features vs raw returns)")
    ax.set_ylabel("|ICIR| — sector-neutral (both sides)")
    ax.set_title(f"Rank IC attribution (H={H})\n"
                 "below diagonal = predictive power was market/sector co-movement")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = build_parser(__doc__)
    p.add_argument("--horizons", type=int, nargs="*", default=[1, 5, 10, 21],
                   help="forward horizons in trading days")
    p.add_argument("--plot-horizon", type=int, default=10, help="horizon for the comparison plots")
    p.add_argument("--min-assets", type=int, default=10)
    args = p.parse_args()

    raw, info, splits = load_raw(args)
    fpanels, _ = feature_reps(raw, info, splits, args)
    rpanels = return_targets(info, splits, args)
    OUT.mkdir(parents=True, exist_ok=True)

    ana = RankICAnalysis(min_assets=args.min_assets)
    plot_h = args.plot_horizon if args.plot_horizon in args.horizons else max(args.horizons)

    rows = []                       # long: representation x horizon x feature
    icir_at_plot = {}               # representation -> per-feature ICIR at plot_h
    for rep in REPRESENTATIONS:
        feats = fpanels[rep]
        rep_dir = OUT / rep
        rep_dir.mkdir(exist_ok=True)
        for H in args.horizons:
            target = forward_target(rpanels[rep], H)
            daily = ana.daily_ic(feats, target)
            summ = ana.summarize(daily, H)
            out = summ.reset_index()
            out.insert(0, "representation", rep)
            out.insert(1, "horizon", H)
            rows.append(out)
            if H == plot_h:
                icir_at_plot[rep] = summ["icir"]
                summ.to_csv(rep_dir / f"ic_summary_H{H}.csv")
                ana.plot_cumulative_ic(daily, summ, min(ana.top_n_plot, len(summ)),
                                       rep_dir / f"cumulative_ic_H{H}.png")

    results = pd.concat(rows, ignore_index=True)
    results.to_csv(OUT / "rank_ic_summary.csv", index=False)

    # ---- comparison at the plot horizon ---- #
    pivot = pd.DataFrame(icir_at_plot).dropna(how="all")
    pivot.to_csv(OUT / f"icir_pivot_H{plot_h}.csv")
    plot_icir_comparison(pivot, plot_h, OUT / f"icir_comparison_H{plot_h}.png")
    plot_attribution(pivot, plot_h, OUT / f"ic_attribution_H{plot_h}.png")

    # ---- headline: top-5 mean |ICIR| per representation x horizon ---- #
    def top5(g):
        a = g["icir"].abs()
        return pd.Series({"top5_mean_abs_icir": a.nlargest(5).mean(),
                          "mean_abs_icir": a.mean(),
                          "n_sig_t2": int((g["t_stat"].abs() >= 2).sum())})
    headline = (results.groupby(["representation", "horizon"]).apply(top5, include_groups=False)
                .reset_index())
    headline.to_csv(OUT / "headline_top5_icir.csv", index=False)

    print("\n===== rank IC decomposition — top-5 mean |ICIR| =====")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(headline.pivot(index="horizon", columns="representation",
                             values="top5_mean_abs_icir")[list(REPRESENTATIONS)])
    print(f"\nper-feature ICIR at H={plot_h} (base -> global -> sector):")
    for f in pivot["base"].abs().sort_values(ascending=False).head(12).index:
        b, g, s = pivot.loc[f, "base"], pivot.loc[f, "global_removed"], pivot.loc[f, "sector_removed"]
        if abs(s) >= abs(b):
            keep = "STRENGTHENS"          # neutral target reveals selection signal
        elif abs(s) >= 0.5 * abs(b):
            keep = "survives"
        else:
            keep = "market/sector-driven"
        print(f"  {f:34s} {b:+.3f} -> {g:+.3f} -> {s:+.3f}   {keep}")
    base5 = headline[headline.horizon == plot_h].set_index("representation")["top5_mean_abs_icir"]
    if "base" in base5 and base5["base"] > 0:
        ratio = base5.get("sector_removed", np.nan) / base5["base"]
        if ratio >= 1:
            print(f"\ntop-5 |ICIR| STRENGTHENS to {ratio:.1f}x after full (global+sector) "
                  f"neutralization at H={plot_h} — the market mode was drowning the")
            print("  cross-sectional signal; the features predict residual returns, not the index.")
        else:
            print(f"\ntop-5 |ICIR| retains {ratio:.0%} of its strength after full "
                  f"(global+sector) neutralization at H={plot_h}")
            print("  high retention -> genuine cross-sectional selection signal;")
            print("  low retention  -> the features were mostly predicting the market/sector mode.")
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
