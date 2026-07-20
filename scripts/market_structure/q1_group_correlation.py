#!/usr/bin/env python
"""Q1 — Do sector (or market-cap) groups correlate more than the whole market,
is their largest eigenvalue (share) much higher, and does Kalman smoothing
change the picture?

Null hypothesis (label exchangeability): a named group of size n behaves like a
uniformly random group of n tickers drawn from the same panel — same mean
pairwise correlation, same top-eigenvalue share (lambda_1 / n), same MP signal
count. p = (1 + #{random-group stat >= observed}) / (1 + B), one-sided.

Kalman branch: the identical test on the causally smoothed panel. Smoothing
mechanically inflates ALL correlations (fewer effective observations), so the
random-group null is drawn from the SAME smoothed panel — only the labels are
random, which keeps the test valid under autocorrelation. The whole-panel MP
n_signal on smoothed data is additionally checked against a shuffle-then-smooth
empirical null (the analytic MP edge assumes iid rows, which smoothing breaks).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUT_ROOT, build_parser, cap_groups, load_panel, ms, sector_groups

OUT = OUT_ROOT / "q1_groups"


def group_table(wide, groups: dict, n_draws: int, seed: int) -> pd.DataFrame:
    rows = []
    for name, members in sorted(groups.items()):
        obs = ms.group_stats(wide, members)
        null = ms.random_group_null(wide, obs["size"], n_draws=n_draws, seed=seed)
        row = {"group": name, **obs}
        for stat in ("mean_corr", "top_eig_share", "n_signal"):
            row[f"p_{stat}"] = ms.permutation_pvalue(null[stat], obs[stat], "greater")
            row[f"null_mean_{stat}"] = float(null[stat].mean())
            row[f"null_std_{stat}"] = float(null[stat].std())
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def plot_group_bars(tab: pd.DataFrame, market: dict, title: str, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(tab) + 3))
    for ax, stat, mkt_val in zip(
            axes, ("mean_corr", "top_eig_share"),
            (market["mean_corr"], market["top_eig_share"])):
        y = np.arange(len(tab))
        ax.barh(y, tab[stat], color="#4c72b0", alpha=0.85, label="observed")
        ax.errorbar(tab[f"null_mean_{stat}"], y, xerr=2 * tab[f"null_std_{stat}"],
                    fmt="o", color="#d64545", ms=4, capsize=3,
                    label="random-group null (mean ± 2σ)")
        ax.axvline(mkt_val, color="black", ls="--", lw=1.2,
                   label=f"whole market ({mkt_val:.3f})")
        ax.set_yticks(y)
        labels = [f"{g} (n={int(tab.loc[g, 'size'])}, p={tab.loc[g, f'p_{stat}']:.3f})"
                  for g in tab.index]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(stat)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved: {path}")


def main():
    args = build_parser(__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wide, meta = load_panel(args)

    sectors = sector_groups(meta)
    caps = cap_groups(meta)

    tables = []
    for panel_name, panel in (("raw", wide), ("kalman", None)):
        if panel_name == "kalman":
            panel, gains = ms.kalman_smooth(wide)
            print(f"\nKalman gains: min={gains.min():.3f} median={gains.median():.3f} "
                  f"max={gains.max():.3f} (floor-bound = heavy smoothing on near-white returns)")

        market = ms.spectrum(panel)
        print(f"\n===== {panel_name.upper()} panel: whole market =====")
        print(f"  mean corr = {market['mean_corr']:.4f}, top eig = {market['top_eig']:.2f} "
              f"({market['top_eig_share']:.1%} of trace), n_signal = {market['n_signal']}, "
              f"lam+ = {market['lam_plus']:.3f}")
        if panel_name == "kalman":
            # analytic MP assumes iid rows; check the whole-panel top eigenvalue
            # against a shuffle-then-smooth empirical null
            null = ms.null_distribution(
                wide, lambda w: {"top_eig": ms.spectrum(w)["top_eig"]},
                null="shuffle", n_draws=min(30, args.n_draws),
                transform=lambda w: ms.kalman_smooth(w)[0], seed=args.seed)
            p = ms.permutation_pvalue(null["top_eig"], market["top_eig"], "greater")
            print(f"  smoothed-panel top eig vs shuffle-then-smooth null: "
                  f"null mean={null['top_eig'].mean():.2f}, observed={market['top_eig']:.2f}, p={p:.3f}")

        for grouping_name, groups in (("sector", sectors), ("cap", caps)):
            tab = group_table(panel, groups, n_draws=args.n_draws, seed=args.seed)
            tab.insert(0, "panel", panel_name)
            tab.insert(1, "grouping", grouping_name)
            tab.insert(2, "market_mean_corr", market["mean_corr"])
            tables.append(tab)
            plot_group_bars(
                tab, market,
                f"Q1 [{panel_name} / {grouping_name}] — group corr vs random-group null "
                f"(H0: labels exchangeable)",
                OUT / f"q1_{panel_name}_{grouping_name}.png")
            sig = tab[tab["p_mean_corr"] < 0.05]
            print(f"  [{grouping_name}] groups beating random at p<0.05 (mean corr): "
                  f"{list(sig.index) or 'NONE'}")

    result = pd.concat(tables)
    result.to_csv(OUT / "q1_group_stats.csv")
    print(f"\nsaved: {OUT / 'q1_group_stats.csv'}")

    print("\n=== Q1 verdict ===")
    raw_sec = result[(result["panel"] == "raw") & (result["grouping"] == "sector")]
    print(f"Sectors with mean corr above their random-group null (p<0.05): "
          f"{(raw_sec['p_mean_corr'] < 0.05).sum()} / {len(raw_sec)}")
    print(f"Sectors with top-eig SHARE above null (p<0.05): "
          f"{(raw_sec['p_top_eig_share'] < 0.05).sum()} / {len(raw_sec)} "
          f"— note lambda_1 itself scales with n; the share is the comparable number.")
    raw_cap = result[(result["panel"] == "raw") & (result["grouping"] == "cap")]
    print(f"Cap buckets beating null (mean corr, p<0.05): "
          f"{(raw_cap['p_mean_corr'] < 0.05).sum()} / {len(raw_cap)}")


if __name__ == "__main__":
    main()
