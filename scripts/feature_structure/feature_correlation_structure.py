#!/usr/bin/env python
"""Feature correlation structure vs the market: does removing the cross-sectional
GLOBAL and SECTOR modes change the FEATURE x FEATURE structure?

Runs CorrelationStructureAnalysis (MP spectrum, denoised within/across-taxonomy
correlation, data-driven clustering vs the trend/vol/momentum/volume taxonomy,
ARI) on three versions of the FULL engineered feature panel — fit over all T
and all N — then compares them:

  base            scaled features (all features by default)
  global_removed  each feature's global market mode (PC1 across tickers) removed
  sector_removed  global + per-sector modes removed

Then two closer looks at the residual:

  * residual vs components — how much each RESIDUAL feature still correlates with
    the global and sector COMPONENTS that were stripped out (via ModeDecomposer
    output="residual+components": {feat}, {feat}_gmode, {feat}_smode). A residual
    is orthogonal to its OWN component by construction, so any surviving
    correlation is with a DIFFERENT feature's mode channel — leftover shared
    market/sector structure.
  * surviving redundancy — the top-N feature pairs that are STILL highly
    correlated after global+sector removal (intrinsic within-name indicator
    redundancy, e.g. RSI~CCI), vs how many strong pairs the neutralization killed.

    .venv/bin/python scripts/feature_structure/feature_correlation_structure.py --tickers 200
    .venv/bin/python scripts/feature_structure/feature_correlation_structure.py --top-pairs 15
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUT_ROOT, REPRESENTATIONS, build_parser, feature_reps, load_raw

from findata.analysis import CorrelationStructureAnalysis
from findata.analysis.correlation_structure import parse_group
from findata.preprocess import ModeDecomposer

OUT = OUT_ROOT / "feature_vs_market"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _taxonomy_ticks(ax, groups):
    """Draw block separators + one centered tick label per taxonomy group."""
    boundaries = [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", lw=0.8)
        ax.axvline(b - 0.5, color="black", lw=0.8)
    edges = [0] + boundaries + [len(groups)]
    centers = [(edges[i] + edges[i + 1]) / 2 - 0.5 for i in range(len(edges) - 1)]
    names = [groups[e] for e in edges[:-1]]
    ax.set_xticks(centers); ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(centers); ax.set_yticklabels(names, fontsize=8)


def aligned_corrs(panels: dict, provenance: dict):
    """Pooled F x F correlation for each representation on a COMMON feature set
    (columns present and non-degenerate in every panel), taxonomy-ordered so the
    delta heatmaps line up block-for-block."""
    cols = None
    dense = {}
    for name, p in panels.items():
        X = p.dropna()
        good = set(X.columns[X.std(ddof=0) > 1e-12])
        cols = good if cols is None else (cols & good)
        dense[name] = X
    cols = sorted(cols, key=lambda c: (provenance.get(c, parse_group(c)), c))
    groups = [provenance.get(c, parse_group(c)) for c in cols]
    corrs = {name: np.corrcoef(dense[name][cols].to_numpy(dtype=float), rowvar=False)
             for name in panels}
    return cols, groups, corrs


def plot_delta(delta, cols, groups, title, path):
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-1, vmax=1)
    _taxonomy_ticks(ax, groups)
    ax.set_title(title)
    fig.colorbar(im, label="Δ correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def top_changed_pairs(delta, cols, groups, n=20):
    iu = np.triu_indices(len(cols), k=1)
    vals = delta[iu]
    order = np.argsort(np.abs(vals))[::-1][:n]
    return pd.DataFrame({
        "feature_i": [cols[iu[0][k]] for k in order],
        "feature_j": [cols[iu[1][k]] for k in order],
        "group_i": [groups[iu[0][k]] for k in order],
        "group_j": [groups[iu[1][k]] for k in order],
        "delta_corr": vals[order],
    })


# --------------------------------------------------------------------------- #
# residual vs global/sector components
# --------------------------------------------------------------------------- #

def component_correlations(base_panel: pd.DataFrame, info, provenance: dict, args):
    """Correlate each RESIDUAL feature with the global- and sector-component
    columns from a residual+components decomposition (fit over all T, N).

    Returns (RG, RS, feats, groups, leakage_df):
      RG / RS  F x F pooled correlation of residual (rows) vs global / sector
               components (cols), taxonomy-ordered; the diagonal is ~0 by
               construction (a residual is orthogonal to its own component).
      leakage_df  per residual feature: its strongest |corr| with any global /
                  sector component (and which feature's channel) — a residual
                  still riding a shared market/sector component shows up here.
    """
    md = ModeDecomposer(group_stage=True, output="residual+components",
                        min_group_size=args.min_sector_size)
    md.bind_meta(info)
    comp = md.fit_transform(base_panel).dropna()

    feats = sorted((c for c in base_panel.columns
                    if comp[c].std(ddof=0) > 1e-12),
                   key=lambda c: (provenance.get(c, parse_group(c)), c))
    groups = [provenance.get(c, parse_group(c)) for c in feats]
    gcols = [f"{c}_gmode" for c in feats]
    scols = [f"{c}_smode" for c in feats]

    block = comp[feats + gcols + scols].to_numpy(dtype=float)
    std = block.std(axis=0, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(block, rowvar=False)
    C[std <= 1e-12, :] = np.nan
    C[:, std <= 1e-12] = np.nan
    nf = len(feats)
    RG = C[:nf, nf:2 * nf]
    RS = C[:nf, 2 * nf:3 * nf]

    def _leak(M):
        A = np.abs(M).copy()
        np.fill_diagonal(A, np.nan)                       # own-component corr is ~0
        best = np.nanargmax(np.where(np.isnan(A), -1, A), axis=1)
        return (np.nanmax(A, axis=1), [feats[b] for b in best], np.nanmean(A, axis=1))

    g_max, g_partner, g_mean = _leak(RG)
    s_max, s_partner, s_mean = _leak(RS)
    leakage = pd.DataFrame({
        "feature": feats, "group": groups,
        "max_abs_corr_global": g_max, "global_partner": g_partner, "mean_abs_corr_global": g_mean,
        "max_abs_corr_sector": s_max, "sector_partner": s_partner, "mean_abs_corr_sector": s_mean,
    }).sort_values("max_abs_corr_global", ascending=False).reset_index(drop=True)
    return RG, RS, feats, groups, leakage


def plot_component_matrix(M, groups, title, path):
    off = M[~np.eye(len(M), dtype=bool)]
    vmax = max(0.2, float(np.nanpercentile(np.abs(off), 99)))
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _taxonomy_ticks(ax, groups)
    ax.set_xlabel("component channel (by feature)")
    ax.set_ylabel("residual feature")
    ax.set_title(title)
    fig.colorbar(im, label="correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# surviving redundancy (top-N still-correlated pairs)
# --------------------------------------------------------------------------- #

def surviving_pairs(corr_base, corr_resid, cols, groups, n=10):
    """Top-n feature pairs by |corr| in the sector-removed residual, alongside
    their base |corr| — the redundancy that survives global+sector removal."""
    iu = np.triu_indices(len(cols), k=1)
    order = np.argsort(np.abs(corr_resid[iu]))[::-1][:n]
    rows = []
    for k in order:
        i, j = iu[0][k], iu[1][k]
        cb, cr = corr_base[i, j], corr_resid[i, j]
        rows.append({"feature_i": cols[i], "feature_j": cols[j],
                     "group_i": groups[i], "group_j": groups[j],
                     "corr_base": cb, "corr_residual": cr,
                     "retained_frac": abs(cr) / max(abs(cb), 1e-9)})
    return pd.DataFrame(rows)


def plot_surviving_pairs(pairs: pd.DataFrame, path):
    labels = [f"{r.feature_i}·{r.feature_j}" for r in pairs.itertuples()]
    y = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(pairs))))
    ax.barh(y + 0.2, pairs["corr_base"].abs(), height=0.4, label="|corr| base", color="#b0b0b0")
    ax.barh(y - 0.2, pairs["corr_residual"].abs(), height=0.4,
            label="|corr| residual (global+sector removed)", color="#4c72b0")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("|correlation|")
    ax.set_title("Feature pairs still correlated after global+sector removal\n"
                 "(bars nearly equal = intrinsic redundancy the market/sector modes never explained)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _count_above(corr, cols, thr):
    iu = np.triu_indices(len(cols), k=1)
    return int((np.abs(corr[iu]) >= thr).sum())


# --------------------------------------------------------------------------- #

def main():
    p = build_parser(__doc__)
    p.add_argument("--top-pairs", type=int, default=10,
                   help="how many still-correlated feature pairs to surface")
    p.add_argument("--corr-threshold", type=float, default=0.5,
                   help="|corr| threshold for the 'strong pair' survival count")
    args = p.parse_args()

    raw, info, splits = load_raw(args)
    panels, provenance = feature_reps(raw, info, splits, args)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- full CorrelationStructureAnalysis per representation ---- #
    rows = []
    for rep in REPRESENTATIONS:
        print(f"\n===== CorrelationStructureAnalysis: {rep} =====")
        res = CorrelationStructureAnalysis(bw=args.bw, output_dir=OUT / rep) \
            .run(panels[rep], provenance=provenance)
        rows.append({
            "representation": rep,
            "n_features": len(res["clusters"]),
            "n_signal": res["n_signal"],
            "top_eig_share": res["top_eig_share"],
            "lambda_plus": res["lambda_plus"],
            "within_raw": res["raw_stats"]["within_mean"],
            "across_raw": res["raw_stats"]["across_mean"],
            "within_denoised": res["denoised_stats"]["within_mean"],
            "across_denoised": res["denoised_stats"]["across_mean"],
            "k_auto": res["k_auto"],
            "ari_vs_taxonomy": res["adjusted_rand_index"],
        })
    summary = pd.DataFrame(rows).set_index("representation")
    summary.to_csv(OUT / "representation_summary.csv")

    # ---- where do the correlations differ? ---- #
    cols, groups, corrs = aligned_corrs(panels, provenance)
    deltas = {
        "base_minus_global": corrs["base"] - corrs["global_removed"],
        "global_minus_sector": corrs["global_removed"] - corrs["sector_removed"],
        "base_minus_sector": corrs["base"] - corrs["sector_removed"],
    }
    for name, d in deltas.items():
        plot_delta(d, cols, groups,
                   f"Δ feature correlation: {name.replace('_', ' ')}\n"
                   f"(red = correlation removed with the mode; mean|Δ|="
                   f"{np.abs(d[np.triu_indices(len(cols), 1)]).mean():.3f})",
                   OUT / f"delta_{name}.png")
        top_changed_pairs(d, cols, groups).to_csv(OUT / f"top_changed_{name}.csv", index=False)

    def mean_abs_off(C):
        return float(np.abs(C[np.triu_indices(len(cols), 1)]).mean())

    # ---- residual vs global/sector components ---- #
    print("\n===== residual vs global / sector components =====")
    RG, RS, feats, cgroups, leakage = component_correlations(panels["base"], info, provenance, args)
    leakage.to_csv(OUT / "residual_component_leakage.csv", index=False)
    plot_component_matrix(RG, cgroups, "Residual feature vs GLOBAL component (pooled corr)\n"
                          "off-diagonal = residual still shares another feature's global mode",
                          OUT / "residual_vs_global_components.png")
    plot_component_matrix(RS, cgroups, "Residual feature vs SECTOR component (pooled corr)\n"
                          "off-diagonal = residual still shares another feature's sector mode",
                          OUT / "residual_vs_sector_components.png")
    print(f"mean |corr| residual↔global components = {np.nanmean(np.abs(RG)):.3f}, "
          f"residual↔sector components = {np.nanmean(np.abs(RS)):.3f}")
    print("residuals still most aligned with a global component (leftover shared market structure):")
    for r in leakage.head(6).itertuples():
        print(f"  {r.feature:34s} |corr|≤{r.max_abs_corr_global:.2f} with {r.global_partner}_gmode")

    # ---- surviving redundancy: top-N still-correlated pairs ---- #
    print("\n===== surviving redundancy (top pairs still correlated) =====")
    surv = surviving_pairs(corrs["base"], corrs["sector_removed"], cols, groups, n=args.top_pairs)
    surv.to_csv(OUT / "surviving_pairs.csv", index=False)
    plot_surviving_pairs(surv, OUT / "surviving_pairs.png")
    thr = args.corr_threshold
    n_base = _count_above(corrs["base"], cols, thr)
    n_resid = _count_above(corrs["sector_removed"], cols, thr)
    print(f"pairs with |corr| ≥ {thr:.2f}:  base={n_base}  sector_removed={n_resid}  "
          f"({n_resid / max(n_base, 1):.0%} of strong redundancy is intrinsic, survives neutralization)")
    print(f"top {len(surv)} still-correlated pairs (residual |corr|):")
    for r in surv.itertuples():
        print(f"  {r.feature_i:26s} · {r.feature_j:26s}  base {r.corr_base:+.2f} -> "
              f"resid {r.corr_residual:+.2f}  ({r.retained_frac:.0%} retained)")

    # ---- headline ---- #
    print("\n===== summary =====")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(summary)
    base_off, glob_off, sect_off = (mean_abs_off(corrs[r]) for r in REPRESENTATIONS)
    print(f"\nmean|feature corr|:  base={base_off:.3f}  "
          f"global_removed={glob_off:.3f}  sector_removed={sect_off:.3f}")
    print(f"  global mode carries {(base_off - glob_off) / base_off:+.1%} of feature correlation")
    print(f"  global+sector carry {(base_off - sect_off) / base_off:+.1%} of feature correlation")
    if (base_off - sect_off) / base_off < 0.15:
        print("  -> feature correlation is mostly INTRINSIC (within-name indicator redundancy);")
        print("     market-mode removal barely decorrelates features (feature-axis PCA/ZCA still needed).")
    else:
        print("  -> a meaningful share of feature correlation is market-driven.")
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
