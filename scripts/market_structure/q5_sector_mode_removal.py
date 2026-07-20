#!/usr/bin/env python
"""Q5 — Once each sector's own dominant mode is removed, is there MORE cross-
correlation across sectors (i.e., shared structure beyond the sector modes)?

Three panels compared:
  raw               standardized returns
  minus_global_pc1  global market mode projected out, renormalized
  minus_sector_pc1  EACH SECTOR's own PC1 projected out of its members
                    (a sector's PC1 contains its share of the global mode too)

For each panel: the sector-block mean-correlation matrix (diagonal = mean
within-sector pair corr, off-diagonal = mean cross-sector corr) and the pooled
cross-sector mean correlation.

H0: after sector-mode removal, the pooled cross-sector correlation is
consistent with independent columns — permutation p from a shuffle null of the
residual panel. Rejection means sectors still share common structure (macro /
style factors) beyond their own modes.

Interpretation caveat: projecting a block's PC1 out forces that block's mean
residual within-correlation slightly NEGATIVE by construction — read the
within diagonal accordingly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUT_ROOT, build_parser, load_panel, ms

OUT = OUT_ROOT / "q5_sector_modes"


def pooled_cross_sector_mean(corr: np.ndarray, columns, labels: pd.Series) -> float:
    labs = np.asarray([labels.get(c) for c in columns])
    same = labs[:, None] == labs[None, :]
    iu = np.triu_indices(len(labs), k=1)
    cross = corr[iu][~same[iu]]
    return float(cross.mean())


def main():
    args = build_parser(__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wide, meta = load_panel(args)
    sectors = meta["sector"]

    panels = {
        "raw": wide,
        "minus_global_pc1": ms.remove_top_pcs(wide, 1),
        "minus_sector_pc1": ms.remove_group_modes(wide, sectors),
    }

    blocks, rows = {}, []
    for name, panel in panels.items():
        spec = ms.spectrum(panel)
        C = spec["corr"]
        blocks[name] = ms.block_mean_corr(C, panel.columns, sectors)
        cross = pooled_cross_sector_mean(C, panel.columns, sectors)

        null = ms.null_distribution(
            panel, lambda w: {"cross": pooled_cross_sector_mean(
                ms.corr_matrix(w), w.columns, sectors)},
            null="shuffle", n_draws=min(100, args.n_draws), seed=args.seed)
        p = ms.permutation_pvalue(np.abs(null["cross"] - null["cross"].mean()),
                                  abs(cross - null["cross"].mean()), "greater")

        rows.append({"panel": name, "cross_sector_mean_corr": cross,
                     "null_mean": float(null["cross"].mean()),
                     "null_std": float(null["cross"].std()),
                     "p_two_sided": p,
                     "n_signal": spec["n_signal"], "top_eig": spec["top_eig"],
                     "mean_corr_all": spec["mean_corr"]})
        print(f"[{name:18s}] cross-sector mean corr = {cross:+.4f} "
              f"(null {null['cross'].mean():+.4f} ± {null['cross'].std():.4f}, "
              f"p = {p:.3f}); n_signal = {spec['n_signal']}, "
              f"top eig = {spec['top_eig']:.1f}")

    summary = pd.DataFrame(rows).set_index("panel")
    summary.to_csv(OUT / "q5_cross_sector_summary.csv")
    for name, block in blocks.items():
        block.to_csv(OUT / f"q5_block_corr_{name}.csv")

    # side-by-side sector-block heatmaps, shared color scale
    vmax = max(np.nanmax(np.abs(b.to_numpy(dtype=float))) for b in blocks.values())
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    im = None
    for ax, (name, block) in zip(axes, blocks.items()):
        im = ax.imshow(block.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(block)))
        ax.set_yticks(range(len(block)))
        ax.set_xticklabels(block.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(block.index, fontsize=7)
        for i in range(len(block)):
            for j in range(len(block)):
                v = block.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > 0.5 * vmax else "black")
        ax.set_title(f"{name}\npooled cross-sector mean = "
                     f"{summary.loc[name, 'cross_sector_mean_corr']:+.4f} "
                     f"(p={summary.loc[name, 'p_two_sided']:.3f})")
    fig.colorbar(im, ax=axes, shrink=0.75, label="mean correlation")
    fig.suptitle("Q5 — sector-block mean correlation: raw / global mode removed / sector modes removed\n"
                 "(diagonal = within-sector; off-diagonal = cross-sector; "
                 "within turns slightly negative by construction after own-mode removal)")
    fig.savefig(OUT / "q5_block_heatmaps.png", dpi=140)
    plt.close(fig)

    print(f"\nsaved: {OUT}")
    print("\n=== Q5 verdict ===")
    raw_c = summary.loc["raw", "cross_sector_mean_corr"]
    res_c = summary.loc["minus_sector_pc1", "cross_sector_mean_corr"]
    print(f"Cross-sector mean corr: raw {raw_c:+.4f} -> after sector-mode removal {res_c:+.4f}")
    if summary.loc["minus_sector_pc1", "p_two_sided"] < 0.05:
        print("Residual cross-sector correlation is NOT consistent with independence "
              "(p<0.05): sectors share structure beyond their own modes.")
    else:
        print("Residual cross-sector correlation is consistent with the independence "
              "null: sector modes account for the shared structure.")


if __name__ == "__main__":
    main()
