#!/usr/bin/env python
"""MP information search on the N-stock cross-section: where does the information
live — in the global market mode, in sector modes, or idiosyncratic — and does
that split look the same for returns as for each feature?

For the N-stock cross-section (N x N correlation, the object Marchenko-Pastur is
built for) we sweep every panel — the return panel AND each feature panel — and
three representations of each:

  base            -> total signal (eigenvalues above the MP bulk edge λ₊)
  global_removed  -> market mode (PC1) projected out, columns renormalized
  sector_removed  -> each sector's own PC1 projected out

The change in n_signal / top-eigenvalue share / mean correlation across
base → global_removed → sector_removed says how much information each mode layer
carries, and comparing returns against features (rsi, obv_vel, …) shows whether
the market/sector structure is a returns-only phenomenon or shared across the
whole feature cube.

MP needs a real cross-section: run with a few hundred tickers (the mode-removed
spectra are rank-reduced, so on a tiny N the noise fit degenerates).

    .venv/bin/python scripts/feature_structure/mp_information_search.py
    .venv/bin/python scripts/feature_structure/mp_information_search.py --tickers 300 \
        --features log_return rsi obv_vel cci
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUT_ROOT, REPRESENTATIONS, build_parser, resolve_tickers

from findata.analysis import market_structure as ms
from findata.preprocess import rmt

OUT = OUT_ROOT / "mp_information_search"
_ROW_KEYS = ("n", "t", "q", "mean_corr", "n_signal", "lam_plus", "sigma2", "top_eig_share")


def spectra(panel, labels) -> dict:
    """base / global_removed / sector_removed N x N spectra of one wide panel."""
    return {
        "base": ms.spectrum(panel),
        "global_removed": ms.spectrum(ms.remove_top_pcs(panel, 1)),
        "sector_removed": ms.spectrum(ms.remove_group_modes(panel, labels)),
    }


def plot_density_panel(specs: dict, feature: str, path):
    """Empirical eigenvalue density vs the fitted MP bulk, one panel per rep."""
    fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 5), squeeze=False)
    for ax, rep in zip(axes[0], REPRESENTATIONS):
        s = specs[rep]
        w = s["eigvals"]
        ax.hist(w, bins=60, density=True, alpha=0.6, color="steelblue",
                label="empirical eigenvalues")
        grid, pdf, _, _ = rmt.mp_pdf(max(s["sigma2"], 1e-6), s["q"])
        ax.plot(grid, pdf, color="black", lw=2, label=f"MP bulk (σ²={s['sigma2']:.2f})")
        ax.axvline(s["lam_plus"], color="crimson", ls="--", lw=1.5,
                   label=f"λ₊={s['lam_plus']:.2f}")
        second = np.sort(w)[-2] if len(w) > 1 else w[0]
        ax.set_xlim(0, max(s["lam_plus"] * 2.2, second * 1.1))  # zoom past the market mode
        ax.set_title(f"{rep}\nn_signal={s['n_signal']}, top share={s['top_eig_share']:.1%}")
        ax.set_xlabel("eigenvalue"); ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(f"{feature} — N×N eigenvalue density vs Marchenko-Pastur "
                 f"(N={specs['base']['n']}, T={specs['base']['t']}, q={specs['base']['q']:.1f})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_feature_bars(summary: pd.DataFrame, metric: str, label: str, path):
    pivot = summary.pivot(index="feature", columns="representation", values=metric)[list(REPRESENTATIONS)]
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(pivot)), 5))
    pivot.plot.bar(ax=ax, rot=30)
    ax.set_title(label)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = build_parser(__doc__)
    p.add_argument("--features", nargs="*", default=None,
                   help=f"panels to analyze (default: {list(ms.DEFAULT_FEATURES)}); "
                        "log_return is the return panel")
    args = p.parse_args()

    tickers = resolve_tickers(args)
    features = args.features or list(ms.DEFAULT_FEATURES)
    print(f"building {len(features)} feature panels over up to {len(tickers)} tickers "
          f"({args.start} -> {args.end or 'latest'})...")
    panels, meta = ms.feature_panels(tickers, features=features, start=args.start, end=args.end)
    labels = meta["sector"]
    any_panel = next(iter(panels.values()))
    print(f"aligned grid: T={any_panel.shape[0]} dates x N={any_panel.shape[1]} tickers, "
          f"{len(set(labels.dropna()))} sectors")
    OUT.mkdir(parents=True, exist_ok=True)

    rows, all_specs = [], {}
    for feat, panel in panels.items():
        specs = spectra(panel, labels)
        all_specs[feat] = specs
        for rep in REPRESENTATIONS:
            row = {"feature": feat, "representation": rep}
            row.update({k: specs[rep][k] for k in _ROW_KEYS})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.rename(columns={"sigma2": "noise_var"}).to_csv(OUT / "mp_summary.csv", index=False)

    # returns is the headline; density panel for it, plus per-feature comparison bars
    ret_key = "log_return" if "log_return" in all_specs else features[0]
    plot_density_panel(all_specs[ret_key], ret_key, OUT / "mp_returns_density.png")
    plot_feature_bars(summary, "n_signal", "N×N signal eigenvalues (above λ₊) by feature",
                      OUT / "mp_signal_by_feature.png")
    plot_feature_bars(summary, "top_eig_share", "Top-eigenvalue share (market dominance) by feature",
                      OUT / "mp_top_share_by_feature.png")

    print("\n===== MP information search (N×N cross-section) =====")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 170):
        print(summary.set_index(["feature", "representation"])
              [["n", "t", "mean_corr", "n_signal", "lam_plus", "top_eig_share"]])
    print("\ninformation carried by each mode layer (base → global → sector):")
    for feat in panels:
        b, g, s = (all_specs[feat][r]["n_signal"] for r in REPRESENTATIONS)
        tb = all_specs[feat]["base"]["top_eig_share"]
        tag = "  <- returns" if feat == ret_key else ""
        print(f"  {feat:11s}  n_signal {b:>3d} → {g:>3d} → {s:>3d}   "
              f"top-eig share(base)={tb:.1%}   global mode = {b - g:+d} signal dir(s){tag}")
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
