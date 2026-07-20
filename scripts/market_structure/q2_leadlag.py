#!/usr/bin/env python
"""Q2 — Temporal dynamics: does shifting tickers in time raise correlations,
and when every ticker is shifted to maximize correlation, do groups emerge by
sector or by market cap?

H0-A (no lead-lag): the max-over-lags correlation of a pair exceeds its lag-0
correlation only by selection bias (max of 2L+1 noisy estimates). Quantified by
the SAME statistic on a circular-shift null — each ticker's autocorrelation is
preserved exactly, cross-sectional alignment is destroyed.

H0-B (shifts carry no group information): (i) per-sector / per-cap shift
distributions match the pooled distribution (Kruskal-Wallis); (ii) clustering
the ALIGNED correlation matrix agrees with sector/cap labels no better than
with permuted labels (ARI permutation test).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import adjusted_rand_score

from _common import OUT_ROOT, build_parser, load_panel, ms

from findata.preprocess import rmt

OUT = OUT_ROOT / "q2_leadlag"


def pair_stats(wide, max_lag):
    lags, cube = ms.lag_correlation_cube(wide, max_lag)
    return ms.best_lag_stats(lags, cube)


def ari_permutation_p(cluster_labels, group_labels, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = adjusted_rand_score(group_labels, cluster_labels)
    perm = np.array([adjusted_rand_score(rng.permutation(group_labels), cluster_labels)
                     for _ in range(n_perm)])
    return obs, ms.permutation_pvalue(perm, obs, "greater")


def main():
    args = build_parser(__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wide, meta = load_panel(args)
    L = args.max_lag
    n_null = min(20, args.n_draws)

    # ---------------- pairwise lead-lag vs circular null ---------------- #
    real = pair_stats(wide, L)
    null_frames = [pair_stats(ms.circular_shift_null(wide, seed=args.seed + b), L)
                   for b in range(n_null)]
    null = pd.concat(null_frames, ignore_index=True)

    frac_nonzero = (real["best_lag"] != 0).mean()
    frac_nonzero_null = (null["best_lag"] != 0).mean()
    p_gain = ms.permutation_pvalue(
        [f["gain"].mean() for f in null_frames], real["gain"].mean(), "greater")

    print("\n===== pairwise lead-lag =====")
    print(f"  mean corr at lag 0        : {real['c0'].mean():.4f}")
    print(f"  mean corr at best lag     : {real['c_best'].mean():.4f}  "
          f"(mean gain {real['gain'].mean():.4f})")
    print(f"  null mean gain (circular) : {null['gain'].mean():.4f}   p(gain) = {p_gain:.3f}")
    print(f"  pairs with best lag != 0  : {frac_nonzero:.1%}  (null: {frac_nonzero_null:.1%})")
    print("  -> the gain from shifting is real only if it clears the selection-bias null")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.arange(-L - 0.5, L + 1.5)
    axes[0].hist(real["best_lag"], bins=bins, density=True, alpha=0.6,
                 color="#d64545", label="real")
    axes[0].hist(null["best_lag"], bins=bins, density=True, alpha=0.5,
                 color="#4c72b0", label="circular null")
    axes[0].set_xlabel("correlation-maximizing lag (days)")
    axes[0].set_title("Best lag per pair — spike at 0 = contemporaneous market")
    axes[0].legend()
    gbins = np.linspace(0, max(real["gain"].max(), null["gain"].max()), 60)
    axes[1].hist(real["gain"], bins=gbins, density=True, alpha=0.6,
                 color="#d64545", label="real")
    axes[1].hist(null["gain"], bins=gbins, density=True, alpha=0.5,
                 color="#4c72b0", label="circular null")
    axes[1].set_xlabel("gain = corr(best lag) - corr(lag 0)")
    axes[1].set_title(f"Shift gain vs selection-bias null (p={p_gain:.3f})")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "q2_pairwise_leadlag.png", dpi=140)
    plt.close(fig)

    # ---------------- align all tickers to the market mode ---------------- #
    aligned, shifts, curves = ms.align_to_reference(wide, L)
    trimmed = wide.iloc[L: len(wide) - L]              # same window as aligned
    mc_before = ms.spectrum(trimmed)["mean_corr"]
    mc_after = ms.spectrum(aligned)["mean_corr"]

    null_gains = []
    for b in range(n_null):
        nw = ms.circular_shift_null(wide, seed=args.seed + 100 + b)
        na, _, _ = ms.align_to_reference(nw, L)
        null_gains.append(ms.spectrum(na)["mean_corr"]
                          - ms.spectrum(nw.iloc[L: len(nw) - L])["mean_corr"])
    p_align = ms.permutation_pvalue(null_gains, mc_after - mc_before, "greater")

    print("\n===== global alignment to the market mode (PC1) =====")
    print(f"  mean corr before alignment: {mc_before:.4f}")
    print(f"  mean corr after alignment : {mc_after:.4f}  (gain {mc_after - mc_before:+.4f}, "
          f"null gain {np.mean(null_gains):+.4f}, p = {p_align:.3f})")
    print(f"  shifts: {(shifts == 0).mean():.1%} at 0, "
          f"range [{shifts.min()}, {shifts.max()}]")

    shift_df = pd.DataFrame({"shift": shifts, "sector": meta["sector"],
                             "cap_bucket": meta["cap_bucket"],
                             "market_cap": meta["market_cap"]})
    shift_df.to_csv(OUT / "q2_shifts.csv")

    print("\n  shift distribution by group (H0: same distribution across groups):")
    for col in ("sector", "cap_bucket"):
        groups = [g["shift"].to_numpy() for _, g in shift_df.groupby(col, observed=True)
                  if len(g) >= 5]
        if all((g == groups[0][0]).all() for g in groups):
            print(f"    {col}: all shifts identical — Kruskal-Wallis undefined")
            continue
        stat, p = kruskal(*groups)
        print(f"    {col}: Kruskal-Wallis H={stat:.2f}, p={p:.4f}")
        summary = shift_df.groupby(col, observed=True)["shift"].agg(["mean", "std", "count"])
        print(summary.round(3).to_string())

    # ---------------- do clusters of the corr matrix match labels? --------- #
    # Raw-return clusters are dominated by the market mode, so also cluster the
    # market-mode-removed panel; auto-k tends to pick 2, so also cut at the
    # label count (k = n sectors / n cap buckets) for a like-for-like ARI.
    print("\n===== clustering (H0: cluster/label agreement no better than chance) =====")
    rows = []
    panels_to_cluster = (("unaligned", trimmed), ("aligned", aligned),
                         ("minus_global_pc1", ms.remove_top_pcs(trimmed, 1)))
    label_sets = (("sector", meta["sector"]), ("cap", meta["cap_bucket"]))
    for name, panel in panels_to_cluster:
        C = ms.corr_matrix(panel)
        for lab_name, lab in label_sets:
            gl = pd.factorize(lab.reindex(panel.columns).astype(str))[0]
            for k_req in ("auto", len(np.unique(gl))):
                labels, _, k = rmt.cluster_corr(C, k=k_req, k_range=(2, 15))
                ari, p = ari_permutation_p(labels, gl, seed=args.seed)
                rows.append({"panel": name, "labels": lab_name,
                             "k_requested": str(k_req), "k": k, "ari": ari, "p": p})
                print(f"  {name:16s} clusters (k={k:2d}) vs {lab_name:6s}: "
                      f"ARI={ari:+.3f}, p={p:.4f}")

    pd.DataFrame(rows).to_csv(OUT / "q2_cluster_ari.csv", index=False)
    real.to_csv(OUT / "q2_pair_stats.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    order = meta["sector"].reindex(shift_df.index).sort_values().index
    sectors_sorted = shift_df.loc[order]
    positions, labels_x = [], []
    for i, (sec, g) in enumerate(sectors_sorted.groupby("sector")):
        ax.scatter(np.full(len(g), i) + np.random.default_rng(1).uniform(-0.18, 0.18, len(g)),
                   g["shift"], s=14, alpha=0.7)
        positions.append(i)
        labels_x.append(f"{sec}\n(n={len(g)})")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_x, fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("correlation-maximizing shift vs market mode (days)")
    ax.set_title("Q2 — optimal temporal shift per ticker, by sector (positive = leads the market)")
    fig.tight_layout()
    fig.savefig(OUT / "q2_shifts_by_sector.png", dpi=140)
    plt.close(fig)
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
