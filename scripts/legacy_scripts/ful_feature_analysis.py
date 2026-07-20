"""
full_feature_taxonomy_analysis.py

Full-pipeline deliverable: takes your ENTIRE feature-engineered set, concatenated
across trend / volatility / momentum / volume groups (group name prepended to each
feature name, e.g. "trend_sma20", "volatility_realized20", "momentum_rsi14",
"volume_obv"), and answers the question this analysis is actually FOR:

    Does your assumed group taxonomy (trend/vol/momentum/volume) match the REAL
    statistical structure in the data, or does real cross-group correlation mean
    the taxonomy is a convenient fiction?

It does this by running the whole feature set through ONE pipeline (no upfront
block assumption) and comparing against the taxonomy at each stage:

  1. Raw correlation, ordered by your assumed taxonomy -> what you'd present naively
  2. Within-group vs across-group average |correlation|, RAW
  3. MP/RMT denoising of the FULL matrix (no per-block split)
  4. Within-group vs across-group average |correlation|, DENOISED
     -> if across-group correlation shrinks much more than within-group correlation
        under denoising, that's evidence cross-group correlation was largely noise.
        If it survives denoising, that's evidence it's REAL structure your taxonomy
        is ignoring.
  5. Hierarchical clustering on the DENOISED distance matrix (data-driven groups,
     no taxonomy assumption)
  6. Dendrogram with leaves colored by your ASSUMED taxonomy labels -> visual check
     of whether the data agrees with your naming
  7. Adjusted Rand Index: quantifies taxonomy-vs-cluster agreement in one number
  8. Correlation heatmap reordered by DATA-DRIVEN cluster order vs TAXONOMY order,
     side by side -> the single most direct visual for "does the data agree with
     my labels"
  9. Overall (whole-system, not per-block) signal eigenvalue count via MP

HOW TO USE WITH YOUR REAL DATA:
  Replace `build_synthetic_grouped_panel()` in `main()` with your real concatenated
  feature df. Required: feature column names prefixed with their group, using
  an underscore separator, e.g. "trend_sma_20", "volatility_atr_14". The group
  parser below splits on the first underscore -- adjust `parse_group()` if your
  naming convention differs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

OUTPUT_DIR = Path("../taxonomy_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# GROUP PARSING
# =============================================================================

def parse_group(feature_name):
    """'trend_sma_20' -> 'trend'. Adjust this if your naming convention differs."""
    return feature_name.split("_")[0]


# =============================================================================
# MP / RMT CORE (same math as the prior script, reused here on the full matrix)
# =============================================================================

def mp_lambda_plus(q, sigma2=1.0):
    return sigma2 * (1 + np.sqrt(1.0 / q)) ** 2


def mp_denoise_correlation(corr, q, sigma2=1.0):
    eigvals, eigvecs = np.linalg.eigh(corr)
    lam_plus = mp_lambda_plus(q, sigma2)
    signal_mask = eigvals > lam_plus
    n_signal = signal_mask.sum()
    denoised_eigvals = eigvals.copy()
    if (~signal_mask).sum() > 0:
        denoised_eigvals[~signal_mask] = eigvals[~signal_mask].mean()
    denoised_corr = eigvecs @ np.diag(denoised_eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(denoised_corr))
    denoised_corr = denoised_corr / np.outer(d, d)
    np.fill_diagonal(denoised_corr, 1.0)
    return denoised_corr, eigvals, n_signal, lam_plus


# =============================================================================
# WITHIN- VS ACROSS-GROUP CORRELATION STATS
# =============================================================================

def within_across_group_stats(corr, feature_names, groups):
    """
    Mean |correlation| for feature pairs within the same taxonomy group vs.
    pairs across different groups. This is the number that tells you whether
    your naming taxonomy corresponds to a real statistical boundary.
    """
    n = len(feature_names)
    within_vals, across_vals = [], []
    within_pairs_by_group_combo = {}

    for i, j in combinations(range(n), 2):
        val = abs(corr[i, j])
        gi, gj = groups[i], groups[j]
        if gi == gj:
            within_vals.append(val)
        else:
            across_vals.append(val)
            key = tuple(sorted([gi, gj]))
            within_pairs_by_group_combo.setdefault(key, []).append(val)

    combo_means = {k: np.mean(v) for k, v in within_pairs_by_group_combo.items()}
    return {
        "within_mean": np.mean(within_vals),
        "across_mean": np.mean(across_vals),
        "across_by_group_pair": combo_means,
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_mp_spectrum(eigvals, q, lam_plus, n_signal):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eigvals, bins=40, density=True, alpha=0.6, color="steelblue", label="empirical eigenvalues")
    x_grid = np.linspace(0.001, max(eigvals.max(), lam_plus) * 1.1, 500)
    lam_minus = max(1 - np.sqrt(1 / q), 0) ** 2
    density = np.zeros_like(x_grid)
    mask = (x_grid >= lam_minus) & (x_grid <= lam_plus)
    density[mask] = np.sqrt(np.maximum((lam_plus - x_grid[mask]) * (x_grid[mask] - lam_minus), 0)) / (
        2 * np.pi * q * x_grid[mask]
    )
    ax.plot(x_grid, density, color="black", lw=2, label="MP theoretical density (pure noise)")
    ax.axvline(lam_plus, color="crimson", ls="--", lw=2, label=f"lambda+ = {lam_plus:.2f}")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_title(f"Whole-system MP spectrum (all groups pooled, no block assumption)\n"
                 f"{n_signal} signal eigenvalue(s) above the noise ceiling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_wholesystem_mp_spectrum.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '1_wholesystem_mp_spectrum.png'}")


def plot_taxonomy_ordered_heatmap(corr, feature_names, groups, title, fname):
    order = np.argsort(groups)  # groups sorted alphabetically -> taxonomy blocks
    ordered_corr = corr[np.ix_(order, order)]
    ordered_names = [feature_names[i] for i in order]
    ordered_groups = [groups[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(ordered_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title)
    # draw block boundary lines between taxonomy groups
    boundaries = [i for i in range(1, len(ordered_groups)) if ordered_groups[i] != ordered_groups[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", lw=1)
        ax.axvline(b - 0.5, color="black", lw=1)
    if len(feature_names) <= 40:
        ax.set_xticks(range(len(ordered_names)))
        ax.set_yticks(range(len(ordered_names)))
        ax.set_xticklabels(ordered_names, rotation=90, fontsize=6)
        ax.set_yticklabels(ordered_names, fontsize=6)
    fig.colorbar(im, label="correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / fname}")


def plot_within_across_bars(raw_stats, denoised_stats):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["within-group\n(raw)", "across-group\n(raw)", "within-group\n(denoised)", "across-group\n(denoised)"]
    values = [raw_stats["within_mean"], raw_stats["across_mean"],
              denoised_stats["within_mean"], denoised_stats["across_mean"]]
    colors = ["steelblue", "lightcoral", "steelblue", "lightcoral"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean |correlation|")
    ax.set_title("Within- vs. across-group correlation, raw vs. denoised\n"
                  "If across-group correlation survives denoising more than it shrinks,\n"
                  "that's real cross-group structure -- not noise")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_within_across_group_comparison.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '2_within_across_group_comparison.png'}")


def plot_dendrogram_with_taxonomy(denoised_corr, feature_names, groups, n_taxonomy_groups):
    dist = np.sqrt(np.maximum(2 * (1 - denoised_corr), 0))
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    unique_groups = sorted(set(groups))
    group_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
    color_map = dict(zip(unique_groups, group_colors))

    fig, ax = plt.subplots(figsize=(12, 7))
    dendro = dendrogram(Z, labels=feature_names, ax=ax, leaf_rotation=90, leaf_font_size=7)
    # color the x-tick labels by taxonomy group (not by cluster -- this is the point:
    # do the data-driven merges group same-colored leaves together or not?)
    ordered_names = dendro["ivl"]
    for tick_label in ax.get_xticklabels():
        name = tick_label.get_text()
        grp = parse_group(name)
        tick_label.set_color(color_map.get(grp, "black"))

    # legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[g], markersize=8, label=g)
               for g in unique_groups]
    ax.legend(handles=handles, title="assumed taxonomy group", loc="upper right")
    ax.set_title("Hierarchical clustering on DENOISED distance (data-driven)\n"
                 "Leaf label colors = your assumed taxonomy. Mixed colors merging early\n"
                 "= data disagrees with the taxonomy at that point.")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_dendrogram_taxonomy_colored.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '3_dendrogram_taxonomy_colored.png'}")

    cluster_labels = fcluster(Z, t=n_taxonomy_groups, criterion="maxclust")
    return cluster_labels, ordered_names


def plot_cluster_vs_taxonomy_heatmap(denoised_corr, feature_names, groups, cluster_labels):
    taxonomy_order = np.argsort(groups)
    cluster_order = np.argsort(cluster_labels)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, order, title in zip(
        axes, [taxonomy_order, cluster_order],
        ["Ordered by ASSUMED taxonomy", "Ordered by DATA-DRIVEN clusters"]
    ):
        mat = denoised_corr[np.ix_(order, order)]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title)
        if len(feature_names) <= 40:
            names = [feature_names[i] for i in order]
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_yticklabels(names, fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")
    fig.suptitle("If these two orderings produce very different block structure,\n"
                 "the data-driven grouping is telling you something your taxonomy misses")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "4_taxonomy_vs_cluster_heatmaps.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '4_taxonomy_vs_cluster_heatmaps.png'}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_full_taxonomy_analysis(df, feature_cols, date_col, horizon_h):
    print("=" * 70)
    print("FULL FEATURE TAXONOMY ANALYSIS (trend / volatility / momentum / volume)")
    print("=" * 70)

    groups = [parse_group(f) for f in feature_cols]
    unique_groups = sorted(set(groups))
    print(f"Features: {len(feature_cols)}, taxonomy groups found: {unique_groups}")

    X = df[feature_cols].dropna()
    X_z = (X - X.mean()) / X.std()
    corr = X_z.corr().values

    N = len(feature_cols)
    T_raw = df[date_col].nunique()
    T_eff = T_raw / horizon_h
    q = T_eff / N
    print(f"N = {N}, T_raw = {T_raw}, H = {horizon_h}, T_eff = {T_eff:.1f}, q = {q:.3f}")

    # --- whole-system MP denoising, no block split ---
    denoised_corr, eigvals, n_signal, lam_plus = mp_denoise_correlation(corr, q)
    print(f"Whole-system signal eigenvalues (above lambda+ = {lam_plus:.3f}): {n_signal} / {N}")
    plot_mp_spectrum(eigvals, q, lam_plus, n_signal)

    # --- taxonomy-ordered raw heatmap (what you'd present naively) ---
    plot_taxonomy_ordered_heatmap(corr, feature_cols, groups,
                                   "RAW correlation, ordered by assumed taxonomy",
                                   "0a_raw_taxonomy_ordered.png")
    plot_taxonomy_ordered_heatmap(denoised_corr, feature_cols, groups,
                                   "DENOISED correlation, ordered by assumed taxonomy",
                                   "0b_denoised_taxonomy_ordered.png")

    # --- within vs across group stats, raw and denoised ---
    raw_stats = within_across_group_stats(corr, feature_cols, groups)
    denoised_stats = within_across_group_stats(denoised_corr, feature_cols, groups)

    print(f"\nRAW:      within-group mean|corr| = {raw_stats['within_mean']:.3f}, "
          f"across-group mean|corr| = {raw_stats['across_mean']:.3f}")
    print(f"DENOISED: within-group mean|corr| = {denoised_stats['within_mean']:.3f}, "
          f"across-group mean|corr| = {denoised_stats['across_mean']:.3f}")
    print("\nAcross-group correlation by group pair (denoised):")
    for pair, val in sorted(denoised_stats["across_by_group_pair"].items(), key=lambda x: -x[1]):
        print(f"  {pair[0]:12s} <-> {pair[1]:12s}: {val:.3f}")

    within_shrink_pct = 100 * (1 - denoised_stats["within_mean"] / max(raw_stats["within_mean"], 1e-9))
    across_shrink_pct = 100 * (1 - denoised_stats["across_mean"] / max(raw_stats["across_mean"], 1e-9))
    print(f"\nWithin-group correlation shrank {within_shrink_pct:.1f}% under denoising")
    print(f"Across-group correlation shrank {across_shrink_pct:.1f}% under denoising")
    if across_shrink_pct > within_shrink_pct + 10:
        print("--> Across-group correlation shrank MORE than within-group: consistent with "
              "cross-group correlation being largely noise, taxonomy holds up reasonably well.")
    elif across_shrink_pct < within_shrink_pct - 10:
        print("--> Across-group correlation survived denoising MORE than within-group: this is "
              "REAL cross-group structure. Your taxonomy is missing something the data sees.")
    else:
        print("--> Within- and across-group correlation shrank similarly: no strong evidence "
              "either way from this comparison alone -- lean on the clustering/ARI result below.")

    plot_within_across_bars(raw_stats, denoised_stats)

    # --- data-driven clustering vs taxonomy ---
    cluster_labels, ordered_names = plot_dendrogram_with_taxonomy(
        denoised_corr, feature_cols, groups, n_taxonomy_groups=len(unique_groups)
    )

    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    taxonomy_int_labels = [group_to_int[g] for g in groups]
    ari = adjusted_rand_score(taxonomy_int_labels, cluster_labels)
    print(f"\nAdjusted Rand Index (taxonomy vs. data-driven clusters, k={len(unique_groups)}): {ari:.3f}")
    print("  ARI = 1.0  -> clusters perfectly match your taxonomy")
    print("  ARI ~ 0.0  -> clusters look like a random relabeling of your taxonomy (no agreement)")
    print("  ARI < 0.0  -> clusters actively disagree with your taxonomy structure")

    plot_cluster_vs_taxonomy_heatmap(denoised_corr, feature_cols, groups, cluster_labels)

    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")

    return {
        "n_signal_components": n_signal,
        "lambda_plus": lam_plus,
        "raw_stats": raw_stats,
        "denoised_stats": denoised_stats,
        "adjusted_rand_index": ari,
        "cluster_labels": cluster_labels,
    }


# =============================================================================
# SYNTHETIC DATA (demo -- replace with your real concatenated feature df)
# =============================================================================

def build_synthetic_grouped_panel(n_dates=750, n_tickers=200, seed=11):
    """
    Builds a 4-group synthetic feature set with REALISTIC cross-group leakage,
    to demonstrate what "taxonomy vs. real structure" disagreement looks like:
      - trend and momentum share a common latent driver (momentum is trend-derived
        in real feature engineering -- e.g. rate-of-change of a trend signal)
      - volatility and volume share a common latent driver (volume spikes with
        volatility -- also realistic)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    tickers = [f"T{i:04d}" for i in range(n_tickers)]

    n_per_group = 6
    trend_names = [f"trend_feat{i}" for i in range(n_per_group)]
    momentum_names = [f"momentum_feat{i}" for i in range(n_per_group)]
    volatility_names = [f"volatility_feat{i}" for i in range(n_per_group)]
    volume_names = [f"volume_feat{i}" for i in range(n_per_group)]
    all_names = trend_names + momentum_names + volatility_names + volume_names

    rows = []
    for date in dates:
        f_trend = rng.normal(0, 1)
        f_momentum_idio = rng.normal(0, 1)
        f_vol = rng.normal(0, 1)
        f_volume_idio = rng.normal(0, 1)

        for ticker in tickers:
            trend_vals = f_trend + rng.normal(0, 1.3, n_per_group)
            # momentum = 55% shared with trend, 45% its own idiosyncratic factor -- REAL cross-group leakage
            momentum_vals = 0.55 * f_trend + 0.45 * f_momentum_idio + rng.normal(0, 1.3, n_per_group)
            vol_vals = f_vol + rng.normal(0, 1.3, n_per_group)
            # volume = 50% shared with volatility, 50% its own -- REAL cross-group leakage
            volume_vals = 0.50 * f_vol + 0.50 * f_volume_idio + rng.normal(0, 1.3, n_per_group)

            feats = np.concatenate([trend_vals, momentum_vals, vol_vals, volume_vals])
            rows.append([date, ticker] + list(feats))

    cols = ["date", "ticker"] + all_names
    df = pd.DataFrame(rows, columns=cols)
    return df, all_names


def main():
    from findata import build_features
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:10]
    dates = DataSplits()
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    features = build_features(data, dates, feature_set='low', feat_eng=True)
    fe = features.features
    FEATURE_COLS = fe.columns
    df = fe.reset_index(drop=False)
    H = 10

    run_full_taxonomy_analysis(df, FEATURE_COLS, date_col="date", horizon_h=H)


if __name__ == "__main__":
    main()