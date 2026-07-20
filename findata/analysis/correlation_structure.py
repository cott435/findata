"""Correlation-structure analysis: does the group taxonomy match the data?

Port of scripts/testing/ful_feature_analysis.py with two substitutions:
  - MP denoising uses the FITTED noise variance (findata.preprocess.rmt), not
    the sigma^2=1 hard threshold of the original script;
  - clustering goes through rmt.cluster_corr — the SAME clustering
    HierarchicalPCA uses, so the clusters you see here are the clusters the
    hpca transform groups by.

Within/across-group statistics are computed with boolean masks on the
correlation matrix (no per-pair Python loops).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from findata.configs import EXPERIMENT_DIR
from findata.preprocess import rmt


def parse_group(feature_name: str) -> str:
    """'momentum__rsi_short_raw' -> 'momentum'; no '__' -> 'component'."""
    return feature_name.split("__", 1)[0] if "__" in feature_name else feature_name.split("_")[0]


class CorrelationStructureAnalysis:

    def __init__(self, q: float | None = None, horizon_h: int | None = None,
                 bw: float = 0.1, cluster_k: int | str = "auto", k_range=(2, 12),
                 output_dir: str | Path | None = None):
        """``q`` overrides the MP ratio; default q = T/N with T = unique dates
        (set ``horizon_h`` to use T/h instead — the original script's convention
        for H-overlapping labels)."""
        self.q = q
        self.horizon_h = horizon_h
        self.bw = bw
        self.cluster_k = cluster_k
        self.k_range = k_range
        self.output_dir = Path(output_dir) if output_dir else EXPERIMENT_DIR / "Correlation Structure"

    # ------------------------------------------------------------------ #
    # stats (vectorized)
    # ------------------------------------------------------------------ #

    @staticmethod
    def within_across_stats(corr: np.ndarray, groups: list[str]) -> dict:
        """Mean |corr| within vs across taxonomy groups + per group-pair means."""
        labels = np.asarray(groups)
        A = np.abs(np.asarray(corr))
        same = labels[:, None] == labels[None, :]
        iu = np.triu_indices(len(labels), k=1)
        vals, same_u = A[iu], same[iu]
        by_pair = {}
        for gi, gj in combinations(sorted(set(groups)), 2):
            mask = ((labels[iu[0]] == gi) & (labels[iu[1]] == gj)) | \
                   ((labels[iu[0]] == gj) & (labels[iu[1]] == gi))
            if mask.any():
                by_pair[(gi, gj)] = float(vals[mask].mean())
        return {
            "within_mean": float(vals[same_u].mean()) if same_u.any() else np.nan,
            "across_mean": float(vals[~same_u].mean()) if (~same_u).any() else np.nan,
            "across_by_group_pair": by_pair,
        }

    # ------------------------------------------------------------------ #
    # plots
    # ------------------------------------------------------------------ #

    def _plot_mp_spectrum(self, eigvals, q, lam_plus, noise_var, n_signal, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(eigvals, bins=40, density=True, alpha=0.6, color="steelblue",
                label="empirical eigenvalues")
        grid, pdf, _, _ = rmt.mp_pdf(noise_var, q)
        ax.plot(grid, pdf, color="black", lw=2,
                label=f"MP density (fitted noise var={noise_var:.2f})")
        ax.axvline(lam_plus, color="crimson", ls="--", lw=2, label=f"lambda+ = {lam_plus:.2f}")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_title(f"Whole-panel MP spectrum: {n_signal} signal eigenvalue(s) above the bulk\n"
                     f"(top eigenvalue share = {eigvals.max() / eigvals.sum():.1%})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def _plot_ordered_heatmap(self, corr, names, groups, title, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(groups, kind="stable")
        mat = corr[np.ix_(order, order)]
        ordered_groups = [groups[i] for i in order]
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title)
        boundaries = [i for i in range(1, len(ordered_groups))
                      if ordered_groups[i] != ordered_groups[i - 1]]
        for b in boundaries:
            ax.axhline(b - 0.5, color="black", lw=1)
            ax.axvline(b - 0.5, color="black", lw=1)
        if len(names) <= 80:
            ordered_names = [names[i] for i in order]
            ax.set_xticks(range(len(ordered_names)))
            ax.set_yticks(range(len(ordered_names)))
            ax.set_xticklabels(ordered_names, rotation=90, fontsize=5)
            ax.set_yticklabels(ordered_names, fontsize=5)
        fig.colorbar(im, label="correlation")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def _plot_within_across_bars(self, raw_stats, den_stats, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["within\n(raw)", "across\n(raw)", "within\n(denoised)", "across\n(denoised)"]
        values = [raw_stats["within_mean"], raw_stats["across_mean"],
                  den_stats["within_mean"], den_stats["across_mean"]]
        colors = ["steelblue", "lightcoral", "steelblue", "lightcoral"]
        bars = ax.bar(labels, values, color=colors, alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", fontsize=9)
        ax.set_ylabel("Mean |correlation|")
        ax.set_title("Within- vs across-group correlation, raw vs denoised\n"
                     "Across-group correlation surviving denoising = real structure\n"
                     "your taxonomy is missing")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def _plot_dendrogram(self, Z, names, groups, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        unique_groups = sorted(set(groups))
        palette = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        color_map = dict(zip(unique_groups, palette))

        fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.22), 7))
        dendrogram(Z, labels=names, ax=ax, leaf_rotation=90, leaf_font_size=6)
        for tick_label in ax.get_xticklabels():
            tick_label.set_color(color_map.get(parse_group(tick_label.get_text()), "black"))
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=color_map[g], markersize=8, label=g)
                   for g in unique_groups]
        ax.legend(handles=handles, title="assumed taxonomy", loc="upper right")
        ax.set_title("Hierarchical clustering on DENOISED correlation distance\n"
                     "(leaf colors = your taxonomy; early mixed-color merges = disagreement)")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def _plot_taxonomy_vs_clusters(self, corr, names, groups, cluster_labels, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        taxonomy_order = np.argsort(groups, kind="stable")
        cluster_order = np.argsort(cluster_labels, kind="stable")
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        im = None
        for ax, order, title in zip(
                axes, [taxonomy_order, cluster_order],
                ["Ordered by ASSUMED taxonomy", "Ordered by DATA-DRIVEN clusters"]):
            mat = corr[np.ix_(order, order)]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(title)
            labels_ordered = (np.asarray(groups)[order] if title.startswith("Ordered by ASSUMED")
                              else np.asarray(cluster_labels)[order])
            boundaries = [i for i in range(1, len(labels_ordered))
                          if labels_ordered[i] != labels_ordered[i - 1]]
            for b in boundaries:
                ax.axhline(b - 0.5, color="black", lw=0.8)
                ax.axvline(b - 0.5, color="black", lw=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")
        fig.suptitle("Different block structure between the two orderings = the data-driven\n"
                     "grouping sees something your taxonomy misses")
        fig.savefig(path, dpi=130)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #

    def run(self, features: pd.DataFrame, provenance: dict[str, str] | None = None,
            verbose: bool = True) -> dict:
        """Full structure analysis of one feature panel.

        ``provenance``: column -> group name (FeaturePipeline.provenance);
        defaults to parsing the '{group}__' prefix.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        X = features.dropna()
        names = list(X.columns)
        groups = [(provenance or {}).get(c, parse_group(c)) for c in names]
        n_groups = len(set(groups))

        vals = X.to_numpy(dtype=float)
        std = vals.std(axis=0, ddof=1)
        keep = std > 1e-12
        if not keep.all():
            dropped = [n for n, k in zip(names, keep) if not k]
            print(f"  Dropping zero-variance columns: {dropped}")
            vals = vals[:, keep]
            names = [n for n, k in zip(names, keep) if k]
            groups = [g for g, k in zip(groups, keep) if k]
        corr = np.corrcoef(vals, rowvar=False)

        T = X.index.get_level_values("date").nunique()
        N = len(names)
        q = self.q if self.q is not None else (T / self.horizon_h if self.horizon_h else T) / N

        eigvals = np.linalg.eigvalsh(corr)[::-1]
        lam_plus, n_signal, noise_var = rmt.mp_edge(eigvals, q, self.bw)
        denoised = rmt.denoise_correlation(corr, q, self.bw)

        raw_stats = self.within_across_stats(corr, groups)
        den_stats = self.within_across_stats(denoised, groups)

        # data-driven clusters: auto-k (what HierarchicalPCA uses) + a cut at the
        # taxonomy's k for a like-for-like ARI
        labels_auto, Z, k_auto = rmt.cluster_corr(denoised, k=self.cluster_k,
                                                  k_range=self.k_range)
        from scipy.cluster.hierarchy import fcluster
        labels_tax_k = fcluster(Z, t=max(n_groups, 2), criterion="maxclust")

        from sklearn.metrics import adjusted_rand_score
        group_ints = pd.factorize(np.asarray(groups))[0]
        ari_auto = adjusted_rand_score(group_ints, labels_auto)
        ari_tax_k = adjusted_rand_score(group_ints, labels_tax_k)

        # ---- artifacts -------------------------------------------------- #
        self._plot_mp_spectrum(eigvals, q, lam_plus, noise_var, n_signal,
                               self.output_dir / "1_mp_spectrum.png")
        self._plot_ordered_heatmap(corr, names, groups,
                                   "RAW correlation (taxonomy order)",
                                   self.output_dir / "0a_raw_taxonomy_ordered.png")
        self._plot_ordered_heatmap(denoised, names, groups,
                                   "DENOISED correlation (taxonomy order)",
                                   self.output_dir / "0b_denoised_taxonomy_ordered.png")
        self._plot_within_across_bars(raw_stats, den_stats,
                                      self.output_dir / "2_within_across_bars.png")
        self._plot_dendrogram(Z, names, groups, self.output_dir / "3_dendrogram.png")
        self._plot_taxonomy_vs_clusters(denoised, names, groups, labels_auto,
                                        self.output_dir / "4_taxonomy_vs_clusters.png")
        clusters = pd.DataFrame({"feature": names, "group": groups,
                                 "cluster_auto": labels_auto, "cluster_tax_k": labels_tax_k})
        clusters.to_csv(self.output_dir / "clusters.csv", index=False)

        if verbose:
            within_shrink = 100 * (1 - den_stats["within_mean"] / max(raw_stats["within_mean"], 1e-9))
            across_shrink = 100 * (1 - den_stats["across_mean"] / max(raw_stats["across_mean"], 1e-9))
            print(f"N={N} features, T={T} dates, q={q:.2f} -> lambda+={lam_plus:.3f}, "
                  f"n_signal={n_signal}, top eig share={eigvals[0] / eigvals.sum():.1%}")
            print(f"RAW      within={raw_stats['within_mean']:.3f} across={raw_stats['across_mean']:.3f}")
            print(f"DENOISED within={den_stats['within_mean']:.3f} across={den_stats['across_mean']:.3f}")
            print(f"Shrink under denoising: within {within_shrink:.1f}%, across {across_shrink:.1f}%")
            if across_shrink > within_shrink + 10:
                print("--> cross-group correlation was largely noise; taxonomy holds up")
            elif across_shrink < within_shrink - 10:
                print("--> REAL cross-group structure survives denoising; taxonomy is missing it")
            else:
                print("--> inconclusive shrink comparison; lean on clustering/ARI")
            print(f"Auto clusters: k={k_auto}, ARI vs taxonomy = {ari_auto:.3f} "
                  f"(at taxonomy k={n_groups}: {ari_tax_k:.3f})")
            print(f"Artifacts -> {self.output_dir.resolve()}")

        return {
            "q": q, "lambda_plus": float(lam_plus), "noise_var": float(noise_var),
            "n_signal": int(n_signal), "top_eig_share": float(eigvals[0] / eigvals.sum()),
            "raw_stats": raw_stats, "denoised_stats": den_stats,
            "cluster_labels": labels_auto, "k_auto": int(k_auto),
            "adjusted_rand_index": float(ari_auto),
            "adjusted_rand_index_tax_k": float(ari_tax_k),
            "clusters": clusters,
        }
