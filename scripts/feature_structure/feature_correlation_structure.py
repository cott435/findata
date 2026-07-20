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

The comparison shows WHERE the feature correlations differ: the delta matrices
(base - global, global - sector), the feature pairs that lose the most
correlation, and whether the MP signal count / taxonomy agreement move once the
market structure is stripped. If feature correlation is intrinsic (RSI ~ CCI on
the same name), the three panels look nearly identical; if it is market-driven,
removing the modes collapses it.

    .venv/bin/python scripts/feature_structure/feature_correlation_structure.py
    .venv/bin/python scripts/feature_structure/feature_correlation_structure.py --tickers 200
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUT_ROOT, REPRESENTATIONS, build_parser, feature_reps, load_raw

from findata.analysis import CorrelationStructureAnalysis
from findata.analysis.correlation_structure import parse_group

OUT = OUT_ROOT / "feature_vs_market"


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
    boundaries = [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", lw=0.8)
        ax.axvline(b - 0.5, color="black", lw=0.8)
    # one centered tick label per taxonomy block
    edges = [0] + boundaries + [len(groups)]
    centers = [(edges[i] + edges[i + 1]) / 2 - 0.5 for i in range(len(edges) - 1)]
    block_names = [groups[e] for e in edges[:-1]]
    ax.set_xticks(centers); ax.set_xticklabels(block_names, rotation=90, fontsize=8)
    ax.set_yticks(centers); ax.set_yticklabels(block_names, fontsize=8)
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


def main():
    args = build_parser(__doc__).parse_args()
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

    print("\n===== summary =====")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(summary)
    base_off = mean_abs_off(corrs["base"])
    glob_off = mean_abs_off(corrs["global_removed"])
    sect_off = mean_abs_off(corrs["sector_removed"])
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
