#!/usr/bin/env python
"""Q4 — How do noise estimates change when PC1 is removed from the data, the
data transformed back and renormalized? Do NEW signal eigenvalues emerge?

Procedure: standardize -> project out the top-k principal components ->
renormalize every column to unit variance -> re-estimate the noise variance and
MP edge with the ITERATIVE bulk-mean method -> count signal eigenvalues again.

H0: after removing the k dominant modes, the residual panel is pure sampling
noise — zero eigenvalues above the refitted MP edge. "New signal emerges" means
n_signal(after) > n_signal(before) - k: eigenvalues that were previously hidden
under an edge inflated by the market mode now clear the recalibrated threshold.

Control: the same PC1-removal applied to a shuffle null (independent columns)
must NOT create signal — verifying the operation itself doesn't manufacture
structure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUT_ROOT, build_parser, load_panel, ms

from findata.preprocess import rmt

OUT = OUT_ROOT / "q4_pc1_removal"
MAX_K = 3


def spectrum_row(name, spec):
    return {"panel": name, "sigma2": spec["sigma2"], "lam_plus": spec["lam_plus"],
            "n_signal": spec["n_signal"], "top_eig": spec["top_eig"],
            "top_eig_share": spec["top_eig_share"], "mean_corr": spec["mean_corr"]}


def plot_spectra(specs: dict, path):
    fig, axes = plt.subplots(1, len(specs), figsize=(5.5 * len(specs), 4.6), sharey=False)
    for ax, (name, spec) in zip(np.atleast_1d(axes), specs.items()):
        w = spec["eigvals"]
        q = spec["q"]
        bins = np.linspace(0, max(spec["lam_plus"] * 1.6, w[min(1, len(w) - 1)] * 1.1), 60)
        ax.hist(w, bins=bins, density=True, alpha=0.6, color="#d64545",
                label="eigenvalues")
        grid, pdf, _, _ = rmt.mp_pdf(spec["sigma2"], q)
        ax.plot(grid, pdf, "k-", lw=2, label=f"MP (sigma2={spec['sigma2']:.2f})")
        ax.axvline(spec["lam_plus"], color="black", ls="--", lw=1,
                   label=f"lam+={spec['lam_plus']:.2f}")
        n_show = spec["n_signal"]
        ax.set_title(f"{name}\nn_signal={n_show}, top eig={spec['top_eig']:.1f} "
                     f"({spec['top_eig_share']:.1%})")
        ax.set_xlabel("eigenvalue")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved: {path}")


def main():
    args = build_parser(__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wide, _ = load_panel(args)

    specs = {"raw": ms.spectrum(wide)}
    for k in range(1, MAX_K + 1):
        specs[f"minus_pc{k}"] = ms.spectrum(ms.remove_top_pcs(wide, k))

    shuffled = ms.shuffle_null(wide, seed=args.seed)
    specs_null = {"null_raw": ms.spectrum(shuffled),
                  "null_minus_pc1": ms.spectrum(ms.remove_top_pcs(shuffled, 1))}

    rows = [spectrum_row(n, s) for n, s in {**specs, **specs_null}.items()]
    table = pd.DataFrame(rows).set_index("panel")
    table.to_csv(OUT / "q4_spectrum_table.csv")

    print("\n===== spectra before/after mode removal (iterative MP) =====")
    print(table.round(4).to_string())

    raw_n = specs["raw"]["n_signal"]
    for k in range(1, MAX_K + 1):
        after = specs[f"minus_pc{k}"]["n_signal"]
        newly = after - (raw_n - k)
        print(f"\nremoved top {k} PC(s): n_signal {raw_n} -> {after} "
              f"(edge {specs['raw']['lam_plus']:.3f} -> {specs[f'minus_pc{k}']['lam_plus']:.3f})")
        print(f"  newly emerged vs naive expectation (n_signal - {k}): {newly:+d}")
    print(f"\ncontrol (shuffle null): n_signal raw={specs_null['null_raw']['n_signal']}, "
          f"after PC1 removal={specs_null['null_minus_pc1']['n_signal']} "
          f"(must both be ~0 — the operation itself doesn't create signal)")

    plot_spectra({k: specs[k] for k in ("raw", "minus_pc1", "minus_pc2")},
                 OUT / "q4_spectra.png")
    plot_spectra(specs_null, OUT / "q4_spectra_null_control.png")

    # rank plot: top 15 eigenvalues at each removal stage
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, spec in specs.items():
        ax.plot(range(1, 16), spec["eigvals"][:15], "o-", ms=4, label=f"{name} "
                f"(lam+={spec['lam_plus']:.2f}, n_sig={spec['n_signal']})")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue (log)")
    ax.set_title("Q4 — top eigenvalues as global modes are peeled off\n"
                 "(renormalization redistributes the removed variance; the refit edge is the new threshold)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "q4_top_eigenvalues.png", dpi=140)
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
