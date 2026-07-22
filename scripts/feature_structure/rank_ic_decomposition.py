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

A separate, exploratory SECTOR-LEVEL rank IC treats the ~11 sectors as the
cross-section (features and returns equal-weight-aggregated per sector): does a
feature predict sector ROTATION? With only ~11 sectors the daily IC is noisy —
a rough look, not a verdict.

Artifacts: an Excel workbook split by sheet (per-representation feature x
horizon, ICIR pivots, headline, sector) plus per-horizon comparison / attribution
figures and a horizon-curve.

    .venv/bin/python scripts/feature_structure/rank_ic_decomposition.py --tickers 250
    .venv/bin/python scripts/feature_structure/rank_ic_decomposition.py \
        --horizons 1 5 10 21 63 --plot-horizons 1 10 63
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


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #

def forward_target(wide: pd.DataFrame, H: int, entity: str = "ticker") -> pd.Series:
    """Forward sum of a wide date x <entity> return panel over the next H days
    -> (entity, date) Series (last H rows per entity NaN, no bleed)."""
    fwd = wide.rolling(H).sum().shift(-H)
    s = fwd.stack()
    s.index = s.index.set_names(["date", entity]).reorder_levels([entity, "date"])
    return s.sort_index().dropna()


def return_targets(info, splits, args):
    """base / global_removed / sector_removed wide daily-return panels + meta,
    aligned to the feature date span (from train_start)."""
    wide, meta = ms.indicator_panel(list(info.index), start=str(splits.train_start),
                                    end=args.end, indicator="log_return", db_path=args.db_path)
    print(f"return panel: T={len(wide)} dates x N={wide.shape[1]} tickers, "
          f"{wide.index[0]} -> {wide.index[-1]}")
    return ({"base": wide,
             "global_removed": ms.remove_top_pcs(wide, 1),
             "sector_removed": ms.remove_group_modes(wide, meta["sector"])}, meta)


def sector_aggregate(feats: pd.DataFrame, ret_wide: pd.DataFrame, sector_of: pd.Series):
    """Equal-weight aggregate features and returns to the sector cross-section.

    Returns ((sector, date) x feature frame, date x sector daily-return frame).
    """
    tick = feats.index.get_level_values("ticker")
    date = feats.index.get_level_values("date")
    sec = pd.Index(tick.map(sector_of), name="sector")
    feats_sec = feats.groupby([sec, pd.Index(date, name="date")]).mean()
    feats_sec.index = feats_sec.index.set_names(["sector", "date"])

    col_sec = pd.Index(ret_wide.columns.map(sector_of), name="sector")
    ret_sec = ret_wide.T.groupby(col_sec).mean().T          # date x sector
    return feats_sec, ret_sec


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #

def plot_grouped_barh(df: pd.DataFrame, order_col, title: str, path, *,
                      xlabel: str = "ICIR", top: int = 18):
    """Grouped horizontal bars: top rows by |order_col|, one bar group per column."""
    order = df[order_col].abs().sort_values(ascending=False).index
    top_df = df.reindex(order).head(top)
    cols = list(df.columns)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.6 * len(top_df))))
    y = np.arange(len(top_df))
    hgt = 0.8 / len(cols)
    for i, c in enumerate(cols):
        off = (len(cols) - 1) / 2 - i
        ax.barh(y + off * hgt, top_df[c].to_numpy(dtype=float), height=hgt, label=str(c))
    ax.set_yticks(y); ax.set_yticklabels(top_df.index, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
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


def plot_horizon_curves(headline: pd.DataFrame, path):
    """top-5 mean |ICIR| vs horizon, one line per representation."""
    piv = headline.pivot(index="horizon", columns="representation",
                         values="top5_mean_abs_icir")[list(REPRESENTATIONS)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for rep in REPRESENTATIONS:
        ax.plot(piv.index, piv[rep], "o-", lw=1.8, label=rep)
    ax.set_xlabel("forward horizon (days)")
    ax.set_ylabel("top-5 mean |ICIR|")
    ax.set_title("Predictive strength vs horizon by representation\n"
                 "rising with neutralization = market mode was drowning the cross-sectional signal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# workbook
# --------------------------------------------------------------------------- #

def _feature_horizon_sheet(sub: pd.DataFrame) -> pd.DataFrame:
    """feature x (horizon, {icir, t_stat}) wide frame, ordered by mean |ICIR|."""
    wide = sub.pivot(index="feature", columns="horizon", values=["icir", "t_stat"])
    wide = wide.swaplevel(axis=1).sort_index(axis=1, level=0)
    order = (wide.xs("icir", axis=1, level=1).abs().mean(axis=1)
             .sort_values(ascending=False).index)
    return wide.loc[order].round(4)


def write_workbook(results: pd.DataFrame, headline: pd.DataFrame,
                   pivots: dict, sector_results: pd.DataFrame,
                   sector_headline: pd.DataFrame, path):
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        headline.pivot(index="horizon", columns="representation",
                       values="top5_mean_abs_icir")[list(REPRESENTATIONS)] \
            .round(4).to_excel(xl, sheet_name="headline_top5")
        headline.round(4).to_excel(xl, sheet_name="headline_full", index=False)
        for rep in REPRESENTATIONS:
            _feature_horizon_sheet(results[results.representation == rep]) \
                .to_excel(xl, sheet_name=f"ticker_{rep}"[:31])
        for H, piv in pivots.items():
            piv.round(4).to_excel(xl, sheet_name=f"pivot_icir_H{H}"[:31])
        if sector_results is not None:
            _feature_horizon_sheet(sector_results).to_excel(xl, sheet_name="sector_rank_ic")
            sector_headline.round(4).to_excel(xl, sheet_name="sector_headline", index=False)


# --------------------------------------------------------------------------- #

def _top5(g):
    a = g["icir"].abs()
    return pd.Series({"top5_mean_abs_icir": a.nlargest(5).mean(),
                      "mean_abs_icir": a.mean(),
                      "n_sig_t2": int((g["t_stat"].abs() >= 2).sum())})


def main():
    p = build_parser(__doc__)
    p.add_argument("--horizons", type=int, nargs="*", default=[1, 5, 10, 21],
                   help="forward horizons in trading days")
    p.add_argument("--plot-horizons", type=int, nargs="*", default=[1, 10, 21],
                   help="up to 3 horizons to render comparison/attribution plots for")
    p.add_argument("--min-assets", type=int, default=10)
    p.add_argument("--sector-min-assets", type=int, default=8,
                   help="min sectors present for a valid sector-level IC day")
    p.add_argument("--no-sector", action="store_true", help="skip the sector-level rank IC")
    args = p.parse_args()

    raw, info, splits = load_raw(args)
    fpanels, _ = feature_reps(raw, info, splits, args)
    rpanels, meta = return_targets(info, splits, args)
    OUT.mkdir(parents=True, exist_ok=True)

    ana = RankICAnalysis(min_assets=args.min_assets)
    horizons = list(dict.fromkeys(args.horizons))
    plot_horizons = [h for h in dict.fromkeys(args.plot_horizons) if h in horizons][:3] \
        or horizons[:3]

    # ---- ticker cross-section: representation x horizon ---- #
    rows = []
    icir_by_h = {rep: {} for rep in REPRESENTATIONS}   # rep -> {H: icir Series}
    for rep in REPRESENTATIONS:
        feats = fpanels[rep]
        rep_dir = OUT / rep
        rep_dir.mkdir(exist_ok=True)
        for H in horizons:
            target = forward_target(rpanels[rep], H)
            daily = ana.daily_ic(feats, target)
            summ = ana.summarize(daily, H)
            out = summ.reset_index()
            out.insert(0, "representation", rep)
            out.insert(1, "horizon", H)
            rows.append(out)
            icir_by_h[rep][H] = summ["icir"]
            if H in plot_horizons:
                ana.plot_cumulative_ic(daily, summ, min(ana.top_n_plot, len(summ)),
                                       rep_dir / f"cumulative_ic_H{H}.png")
    results = pd.concat(rows, ignore_index=True)
    results.to_csv(OUT / "rank_ic_summary.csv", index=False)
    headline = (results.groupby(["representation", "horizon"]).apply(_top5, include_groups=False)
                .reset_index())

    # ---- per-plot-horizon comparison + attribution ---- #
    pivots = {}
    for H in plot_horizons:
        pivot = pd.DataFrame({rep: icir_by_h[rep][H] for rep in REPRESENTATIONS}).dropna(how="all")
        pivots[H] = pivot
        plot_grouped_barh(pivot, "base",
                          f"Rank ICIR by representation (H={H}) — top features by base |ICIR|\n"
                          "shrink from base = predictive power that lived in the market/sector mode",
                          OUT / f"icir_comparison_H{H}.png")
        plot_attribution(pivot, H, OUT / f"ic_attribution_H{H}.png")
    plot_horizon_curves(headline, OUT / "top5_icir_vs_horizon.png")

    # ---- exploratory sector-level rank IC (rotation) ---- #
    sector_results = sector_headline = None
    if not args.no_sector:
        sec_dir = OUT / "sector"
        sec_dir.mkdir(exist_ok=True)
        sector_of = info["sector"].fillna("ETF").astype(str)
        feats_sec, ret_sec = sector_aggregate(fpanels["base"], rpanels["base"], sector_of)
        n_sec = ret_sec.shape[1]
        sec_ana = RankICAnalysis(min_assets=min(args.sector_min_assets, n_sec))
        print(f"\nsector cross-section: {n_sec} sectors (rough — daily IC over ~{n_sec} names)")
        s_rows, s_icir = [], {}
        for H in horizons:
            target = forward_target(ret_sec, H, entity="sector")
            daily = sec_ana.daily_ic(feats_sec, target)
            summ = sec_ana.summarize(daily, H)
            out = summ.reset_index(); out.insert(0, "horizon", H)
            s_rows.append(out); s_icir[H] = summ["icir"]
            if H in plot_horizons:
                sec_ana.plot_cumulative_ic(daily, summ, min(sec_ana.top_n_plot, len(summ)),
                                           sec_dir / f"cumulative_ic_H{H}.png")
        sector_results = pd.concat(s_rows, ignore_index=True)
        sector_results.to_csv(sec_dir / "sector_rank_ic_summary.csv", index=False)
        sector_headline = (sector_results.groupby("horizon").apply(_top5, include_groups=False)
                           .reset_index())
        sec_pivot = pd.DataFrame({f"H={H}": s_icir[H] for H in plot_horizons}).dropna(how="all")
        plot_grouped_barh(sec_pivot, sec_pivot.columns[0],
                          f"SECTOR-level rank ICIR ({n_sec} sectors) — top features across horizons\n"
                          "does the feature predict sector rotation? (rough — few names)",
                          sec_dir / "sector_icir_by_horizon.png")

    # ---- workbook (split by sheet) ---- #
    write_workbook(results, headline, pivots, sector_results, sector_headline,
                   OUT / "rank_ic_summary.xlsx")

    # ---- console ---- #
    print("\n===== rank IC decomposition — top-5 mean |ICIR| (ticker cross-section) =====")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(headline.pivot(index="horizon", columns="representation",
                             values="top5_mean_abs_icir")[list(REPRESENTATIONS)])
    for H in plot_horizons:
        piv = pivots[H]
        print(f"\nper-feature ICIR at H={H} (base -> global -> sector), top by base |ICIR|:")
        for f in piv["base"].abs().sort_values(ascending=False).head(8).index:
            b, g, s = piv.loc[f, "base"], piv.loc[f, "global_removed"], piv.loc[f, "sector_removed"]
            tag = ("STRENGTHENS" if abs(s) >= abs(b)
                   else "survives" if abs(s) >= 0.5 * abs(b) else "market/sector-driven")
            print(f"  {f:34s} {b:+.3f} -> {g:+.3f} -> {s:+.3f}   {tag}")
    if sector_headline is not None:
        print("\n===== SECTOR-level top-5 mean |ICIR| (rough, ~11 names) =====")
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(sector_headline.set_index("horizon")[["top5_mean_abs_icir", "n_sig_t2"]])
    print(f"\nworkbook -> {OUT / 'rank_ic_summary.xlsx'}")
    print(f"artifacts -> {OUT}")


if __name__ == "__main__":
    main()
