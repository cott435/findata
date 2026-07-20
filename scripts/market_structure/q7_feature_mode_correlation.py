#!/usr/bin/env python
"""Q7 — Two PCAs on two axes of the data cube (time x ticker x feature): how do
the per-feature GLOBAL and SECTOR market modes (PCA over tickers, N) relate to
the pooled feature-feature correlation (PCA over features, F), and where do the
two decompositions agree vs diverge?

For every feature (rsi, obv_vel, ...) we run the cross-sectional (N) mode
decomposition, then build the FxF correlation four ways:

  pool          feature corr pooled over T x N  (what the F-PCA / HierarchicalPCA sees)
  global        corr of the features' global market modes
  sector        corr of the features' sector modes (mean over sectors)
  resid_global  pooled FxF after removing each feature's global mode

Null hypothesis (per feature pair): the pooled feature correlation is entirely
carried by the shared global market mode, i.e. resid_global[f,g] = 0. Rejection
(residual survives) = feature co-movement the market-mode PCA cannot explain —
the axis where the two PCAs do NOT reduce to one another.

Also repeats the analysis on a highly-correlated feature subset (auto-detected
from the pooled matrix, or --features).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.metrics import adjusted_rand_score

from _common import OUT_ROOT, build_parser, ms

from findata.preprocess import rmt

OUT = OUT_ROOT / "q7_feature_modes"


def leaf_order(corr: np.ndarray) -> np.ndarray:
    if corr.shape[0] < 3:
        return np.arange(corr.shape[0])
    Z = linkage(rmt.corr_distance(corr)[np.triu_indices(len(corr), 1)], method="average")
    return leaves_list(Z)


def heat(ax, M, names, title):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(title, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(M[i, j]) > 0.55 else "black")
    return im


def mean_abs_offdiag(M):
    iu = np.triu_indices(len(M), 1)
    v = M[iu]
    return float(np.nanmean(np.abs(v)))


def analyze(panels, labels, out, tag):
    out.mkdir(parents=True, exist_ok=True)
    res = ms.mode_feature_correlation(panels, labels)
    feats = np.array(res["features"])
    order = leaf_order(res["pool"])
    names = feats[order].tolist()

    def reorder(M):
        return M[np.ix_(order, order)]

    mats = {k: reorder(res[k]) for k in ("pool", "global", "sector", "resid_global", "resid_sector")}
    for k, M in mats.items():
        pd.DataFrame(M, index=names, columns=names).to_csv(out / f"corr_{k}.csv")

    # ---- 4-panel: pooled vs modes vs residual ---- #
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    im = None
    for ax, key, title in zip(
            axes, ("pool", "global", "resid_global", "resid_sector"),
            (f"POOLED FxF (over T x N)\nmean|off|={mean_abs_offdiag(res['pool']):.2f}",
             f"GLOBAL-mode FxF\nmean|off|={mean_abs_offdiag(res['global']):.2f}",
             f"RESIDUAL FxF (global removed)\nmean|off|={mean_abs_offdiag(res['resid_global']):.2f}",
             f"RESIDUAL FxF (global+sector)\nmean|off|={mean_abs_offdiag(res['resid_sector']):.2f}")):
        im = heat(ax, mats[key], names, title)
    fig.colorbar(im, ax=axes, shrink=0.7, label="correlation")
    fig.suptitle(f"[{tag}] Feature correlation: pooled = market-mode-driven + residual "
                 f"({len(feats)} features, {res['n_sectors']} shared sectors)")
    fig.savefig(out / "1_fourway_heatmaps.png", dpi=140)
    plt.close(fig)

    # ---- global vs sector mode correlation ---- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    heat(axes[0], mats["global"], names, "GLOBAL-mode feature correlation")
    im = heat(axes[1], mats["sector"], names, "SECTOR-mode feature correlation (avg)")
    fig.colorbar(im, ax=axes, shrink=0.7, label="correlation")
    fig.suptitle(f"[{tag}] Do features' market aggregates co-move the same way "
                 f"globally vs within sectors?")
    fig.savefig(out / "2_global_vs_sector_modes.png", dpi=140)
    plt.close(fig)

    # ---- agreement scatter: where do the two PCAs reduce to one another? ---- #
    iu = np.triu_indices(len(feats), 1)
    pool_v, resid_v, glob_v = res["pool"][iu], res["resid_global"][iu], res["global"][iu]
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(pool_v, resid_v, c=glob_v, cmap="coolwarm", vmin=-1, vmax=1,
                    s=90, edgecolor="k", linewidth=0.5)
    lim = [min(pool_v.min(), resid_v.min()) - 0.05, max(pool_v.max(), resid_v.max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=1, label="resid = pool (market mode irrelevant)")
    ax.axhline(0, color="gray", lw=0.7)
    for a, b, fi, fj in zip(pool_v, resid_v, feats[iu[0]], feats[iu[1]]):
        ax.annotate(f"{fi}·{fj}", (a, b), fontsize=6, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("pooled feature correlation (F-PCA sees this)")
    ax.set_ylabel("residual correlation after global-mode removal")
    ax.set_title(f"[{tag}] Where the two PCAs interact\n"
                 "near diagonal = market mode irrelevant to this pair (idiosyncratic co-move);\n"
                 "far below diagonal = correlation was market-driven (the two PCAs are one structure)")
    fig.colorbar(sc, label="global-mode corr(f,g)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "3_agreement_scatter.png", dpi=140)
    plt.close(fig)

    # ---- does the FEATURE clustering reorganize once the market mode is gone? ---- #
    lab_pool, _, k1 = rmt.cluster_corr(res["pool"], k="auto", k_range=(2, min(6, len(feats) - 1)))
    lab_resid, _, k2 = rmt.cluster_corr(res["resid_global"], k="auto",
                                        k_range=(2, min(6, len(feats) - 1)))
    ari = adjusted_rand_score(lab_pool, lab_resid)

    reduction = 1 - mean_abs_offdiag(res["resid_global"]) / max(mean_abs_offdiag(res["pool"]), 1e-9)
    summary = {
        "tag": tag, "n_features": len(feats),
        "mean_abs_pool": mean_abs_offdiag(res["pool"]),
        "mean_abs_global": mean_abs_offdiag(res["global"]),
        "mean_abs_sector": mean_abs_offdiag(res["sector"]),
        "mean_abs_resid_global": mean_abs_offdiag(res["resid_global"]),
        "mean_abs_resid_sector": mean_abs_offdiag(res["resid_sector"]),
        "frac_feature_corr_from_market": reduction,
        "k_pool": k1, "k_resid": k2, "cluster_ARI_pool_vs_resid": ari,
    }
    pd.Series(summary).to_csv(out / "summary.csv")

    print(f"\n===== [{tag}] {len(feats)} features =====")
    print(f"  mean|corr|  pooled={summary['mean_abs_pool']:.3f}  "
          f"global-mode={summary['mean_abs_global']:.3f}  "
          f"sector-mode={summary['mean_abs_sector']:.3f}")
    print(f"  mean|corr|  residual(global)={summary['mean_abs_resid_global']:.3f}  "
          f"residual(global+sector)={summary['mean_abs_resid_sector']:.3f}")
    print(f"  -> {reduction:.0%} of feature correlation was carried by the global market mode")
    print(f"  feature clusters: pool k={k1} vs residual k={k2}, ARI={ari:.3f} "
          f"({'stable' if ari > 0.6 else 'REORGANIZES'} once the market mode is removed)")
    return res, summary


def pick_correlated_subset(res, min_size=3):
    """Largest tight feature cluster from the pooled matrix (highest within-mean|corr|)."""
    feats = np.array(res["features"])
    if len(feats) < min_size + 1:
        return None
    labels, _, _ = rmt.cluster_corr(res["pool"], k="auto", k_range=(2, len(feats) - 1))
    best, best_score = None, -np.inf
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < min_size:
            continue
        block = np.abs(res["pool"][np.ix_(idx, idx)])
        score = block[np.triu_indices(len(idx), 1)].mean()
        if score > best_score:
            best, best_score = idx, score
    return feats[best].tolist() if best is not None else None


def main():
    p = build_parser(__doc__)
    p.add_argument("--features", nargs="*", default=None,
                   help=f"features to use (default: {list(ms.DEFAULT_FEATURES)})")
    p.add_argument("--subset", nargs="*", default=None,
                   help="explicit highly-correlated subset (default: auto-detect)")
    args = p.parse_args()

    from findata import get_all_tickers
    tickers = get_all_tickers()
    if args.tickers:
        tickers = tickers[:args.tickers]
    features = args.features or list(ms.DEFAULT_FEATURES)

    print(f"building {len(features)} feature panels over up to {len(tickers)} tickers...")
    panels, meta = ms.feature_panels(tickers, features=features, start=args.start, end=args.end)
    any_panel = next(iter(panels.values()))
    print(f"aligned grid: T={any_panel.shape[0]} dates x N={any_panel.shape[1]} tickers")
    labels = meta["sector"]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, panel in panels.items():
        panel.to_parquet(OUT / f"panel_{name}.parquet")
    meta.to_csv(OUT / "meta.csv")

    res, _ = analyze(panels, labels, OUT / "all_features", "all features")

    subset = args.subset or pick_correlated_subset(res)
    if subset and len(subset) >= 3:
        print(f"\nhighly-correlated subset: {subset}")
        analyze({f: panels[f] for f in subset}, labels, OUT / "correlated_subset",
                "correlated subset")
    else:
        print("\nno correlated subset of >=3 features found; skipping subset analysis")

    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
