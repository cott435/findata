#!/usr/bin/env python
"""Q6 — How does the cumulative reward (log-return) signal split under
different PC decompositions, and does the ORDER of removal matter?

Two exact, in-sample-orthogonal decompositions of each stock's standardized
return x_i (so variance shares sum to 1 and cumulative components stack
exactly to the stock's demeaned cumulative return):

  Pipeline A (global-first):
      x_i = beta_i * g            global mode = PC1 of the WHOLE panel
          + gamma_i * s_k         sector mode = PC1 of sector k's block of the
                                  global-removed residual (g ⊥ s_k exactly)
          + e_i                   idiosyncratic residual

  Pipeline B (sector-first):
      m_k   = PC1 of sector k's RAW block          (contains global + sector)
      g_B   = PC1 across the K sector-mode series  (the shared/global part)
      u_k   = m_k - delta_k * g_B                  (sector-SPECIFIC mode)
      x_i = gammaB_i*delta_k * g_B + gammaB_i * u_k + e_i

Deliverables:
  - step-wise correlation heatmaps (raw -> global removed -> sector removed)
    for A; raw -> sector-mode removed for B, plus the K x K sector-mode
    correlation before/after removing ITS PC1
  - variance-share split (global / sector / idio) per sector, A vs B
  - agreement between the two orderings: corr(g_A, g_B), corr(s_k^A, u_k^B)
  - per-ticker cumulative log-return decomposition (few tickers per plot),
    for a highly correlated sector vs the weakly correlated ones
    (Healthcare, Consumer Defensive, Communication Services)
  - cumulative plots of the global and sector-specific signals themselves

All attribution is IN-SAMPLE and descriptive (full-period PCs) — this is a
structure-understanding exercise, not a tradeable signal.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUT_ROOT, build_parser, load_panel, ms

OUT = OUT_ROOT / "q6_reward_split"
LOW_CORR_SECTORS = ["Healthcare", "Consumer Defensive", "Communication Services"]
TICKERS_PER_PLOT = 3


# --------------------------------------------------------------------------
# PC helpers (T x m standardized blocks)
# --------------------------------------------------------------------------

def pc1(Xb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-variance PC1 score series + per-column loadings (score covariance).

    Sign-fixed so the mean loading is positive. Because columns are unit
    variance, loading_i = corr(x_i, score) and loading_i^2 = variance share.
    """
    T = Xb.shape[0]
    C = (Xb.T @ Xb) / T
    _, v = np.linalg.eigh(C)
    v1 = v[:, -1]
    if v1.mean() < 0:
        v1 = -v1
    s = Xb @ v1
    s = s / (s.std(ddof=0) + 1e-12)
    beta = (Xb.T @ s) / T
    return s, beta


def sector_blocks(columns, sectors: pd.Series) -> dict[str, np.ndarray]:
    """sector -> integer column indices (sectors with >= 2 members)."""
    out = {}
    labs = np.asarray([sectors.get(c) for c in columns])
    for k in pd.unique(labs[pd.notna(labs)]):
        idx = np.where(labs == k)[0]
        if len(idx) >= 2:
            out[k] = idx
    return out


# --------------------------------------------------------------------------
# The two decompositions. X is the standardized (T, N) panel.
# --------------------------------------------------------------------------

def decompose_global_first(X: np.ndarray, blocks: dict) -> dict:
    g, beta = pc1(X)
    G = np.outer(g, beta)
    R1 = X - G
    S = np.zeros_like(X)
    sector_scores, sector_load = {}, np.zeros(X.shape[1])
    for k, idx in blocks.items():
        s_k, gamma = pc1(R1[:, idx])
        sector_scores[k] = s_k
        sector_load[idx] = gamma
        S[:, idx] = np.outer(s_k, gamma)
    E = R1 - S
    return {"global_score": g, "global_load": beta, "global_comp": G,
            "sector_scores": sector_scores, "sector_load": sector_load,
            "sector_comp": S, "resid": E,
            "share_global": beta ** 2, "share_sector": sector_load ** 2}


def decompose_sector_first(X: np.ndarray, blocks: dict) -> dict:
    M = np.zeros_like(X)
    mode_series, mode_load = {}, np.zeros(X.shape[1])
    for k, idx in blocks.items():
        m_k, gamma = pc1(X[:, idx])
        mode_series[k] = m_k
        mode_load[idx] = gamma
        M[:, idx] = np.outer(m_k, gamma)
    E = X - M

    keys = list(mode_series)
    M_panel = np.column_stack([mode_series[k] for k in keys])      # (T, K), unit var
    g_B, delta = pc1(M_panel)
    U = M_panel - np.outer(g_B, delta)                             # sector-specific modes
    delta_map = dict(zip(keys, delta))
    u_map = {k: U[:, i] for i, k in enumerate(keys)}

    N = X.shape[1]
    share_global = np.zeros(N)
    share_sector = np.zeros(N)
    G = np.zeros_like(X)
    S = np.zeros_like(X)
    for k, idx in blocks.items():
        d = delta_map[k]
        share_global[idx] = (mode_load[idx] * d) ** 2
        share_sector[idx] = mode_load[idx] ** 2 * (1 - d ** 2)
        G[:, idx] = np.outer(g_B * d, mode_load[idx])
        S[:, idx] = np.outer(u_map[k], mode_load[idx])
    return {"global_score": g_B, "global_comp": G,
            "mode_series": mode_series, "mode_delta": delta_map,
            "sector_scores": u_map, "sector_comp": S, "resid": E,
            "share_global": share_global, "share_sector": share_sector,
            "mode_panel": pd.DataFrame(M_panel, columns=keys)}


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------

def renorm(X: np.ndarray) -> np.ndarray:
    return X / (X.std(0, ddof=0, keepdims=True) + 1e-12)


def corr_of(X: np.ndarray) -> np.ndarray:
    return (X.T @ X) / X.shape[0]


def plot_corr_steps(mats: dict, columns, sectors: pd.Series, title, path):
    order = np.argsort([str(sectors.get(c)) for c in columns], kind="stable")
    labs_ordered = [str(sectors.get(columns[i])) for i in order]
    boundaries = [i for i in range(1, len(labs_ordered))
                  if labs_ordered[i] != labs_ordered[i - 1]]
    fig, axes = plt.subplots(1, len(mats), figsize=(5.6 * len(mats), 5.4))
    im = None
    for ax, (name, C) in zip(np.atleast_1d(axes), mats.items()):
        Cn = renorm_corr = C  # already correlation matrices
        im = ax.imshow(Cn[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
        off = ms.offdiag(Cn)
        ax.set_title(f"{name}\nmean corr {off.mean():+.3f}", fontsize=10)
        for b in boundaries:
            ax.axhline(b - 0.5, color="black", lw=0.5)
            ax.axvline(b - 0.5, color="black", lw=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")
    fig.suptitle(title + "  (sector-ordered; lines = sector boundaries)")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved: {path}")


def plot_mode_corr(mode_panel: pd.DataFrame, g_B, path):
    Cm = corr_of(renorm(mode_panel.to_numpy()))
    U = mode_panel.to_numpy() - np.outer(g_B, (mode_panel.to_numpy().T @ g_B) / len(g_B))
    Cu = corr_of(renorm(U))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (name, C) in zip(axes, (("sector modes (raw)", Cm),
                                    ("sector modes minus their PC1", Cu))):
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mode_panel.columns)))
        ax.set_yticks(range(len(mode_panel.columns)))
        ax.set_xticklabels(mode_panel.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(mode_panel.columns, fontsize=7)
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(C[i, j]) > 0.5 else "black")
        ax.set_title(f"{name} (mean offdiag {ms.offdiag(C).mean():+.3f})", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle("Pipeline B: how much the sector modes share (their PC1 = the global signal)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved: {path}")


def plot_variance_split(shares: pd.DataFrame, path):
    """Stacked mean variance shares per sector, pipeline A vs B side by side."""
    agg = shares.groupby(["sector", "pipeline"])[["global", "sector_specific", "idio"]].mean()
    sectors = sorted(shares["sector"].unique())
    x = np.arange(len(sectors))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = {"global": "#4c72b0", "sector_specific": "#dd8452", "idio": "#55a868"}
    for off, pipe in ((-width / 2, "A"), (width / 2, "B")):
        bottoms = np.zeros(len(sectors))
        for comp in ("global", "sector_specific", "idio"):
            vals = np.array([agg.loc[(s, pipe), comp] if (s, pipe) in agg.index else 0
                             for s in sectors])
            ax.bar(x + off, vals, width, bottom=bottoms, color=colors[comp],
                   edgecolor="white", lw=0.4,
                   label=comp if pipe == "A" else None,
                   alpha=1.0 if pipe == "A" else 0.65)
            bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean variance share")
    ax.set_title("Q6 — where each sector's return variance lives\n"
                 "left bar = pipeline A (global-first), right = pipeline B (sector-first, faded)")
    ax.legend(title="component")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved: {path}")


def plot_ticker_decomposition(wide, mu, sigma, A, blocks, columns, sectors, caps,
                              sector_name, path):
    """Few tickers of one sector: cumulative log return split into
    global + sector + idio (pipeline A), stacking exactly to the total."""
    idx = blocks[sector_name]
    members = [columns[i] for i in idx]
    top = caps.reindex(members).sort_values(ascending=False).index[:TICKERS_PER_PLOT]

    fig, axes = plt.subplots(len(top), 1, figsize=(12, 3.2 * len(top)), sharex=True)
    dates = pd.to_datetime(wide.index)
    for ax, tkr in zip(np.atleast_1d(axes), top):
        i = columns.index(tkr)
        total = wide[tkr].cumsum()
        glob = pd.Series(sigma[i] * A["global_comp"][:, i], index=wide.index).cumsum()
        sect = pd.Series(sigma[i] * A["sector_comp"][:, i], index=wide.index).cumsum()
        idio = pd.Series(sigma[i] * A["resid"][:, i] + mu[i], index=wide.index).cumsum()
        ax.plot(dates, total, color="black", lw=1.6, label="total cum log-return")
        ax.plot(dates, glob, color="#4c72b0", lw=1.2, label="global component")
        ax.plot(dates, sect, color="#dd8452", lw=1.2, label="sector component")
        ax.plot(dates, idio, color="#55a868", lw=1.2, label="idiosyncratic (incl. drift)")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        sh_g, sh_s = A["share_global"][i], A["share_sector"][i]
        ax.set_title(f"{tkr} — variance split: global {sh_g:.0%} / sector {sh_s:.0%} / "
                     f"idio {1 - sh_g - sh_s:.0%}", fontsize=10)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    fig.suptitle(f"Q6 — {sector_name}: cumulative reward decomposition "
                 f"(components sum exactly to the total)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved: {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = build_parser(__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wide, meta = load_panel(args)
    sectors = meta["sector"]
    columns = list(wide.columns)

    V = wide.to_numpy(dtype=float)
    mu = V.mean(0)
    sigma = V.std(0, ddof=0)
    X = (V - mu) / (sigma + 1e-12)
    blocks = sector_blocks(columns, sectors)

    A = decompose_global_first(X, blocks)
    B = decompose_sector_first(X, blocks)

    # ---- agreement between the two orderings ---------------------------- #
    g_corr = float(np.corrcoef(A["global_score"], B["global_score"])[0, 1])
    print(f"\n===== ordering comparison =====")
    print(f"corr(global A [panel PC1], global B [PC1 of sector modes]) = {g_corr:.4f}")
    rows = []
    for k in blocks:
        c = float(np.corrcoef(A["sector_scores"][k], B["sector_scores"][k])[0, 1])
        rows.append({"sector": k, "corr_sectorA_sectorB": c,
                     "delta_k (sector-mode loading on global B)": B["mode_delta"][k]})
        print(f"  {k:24s} corr(s_A, u_B) = {c:+.3f}   delta_k = {B['mode_delta'][k]:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "q6_ordering_agreement.csv", index=False)

    # ---- correlation matrix at each step -------------------------------- #
    plot_corr_steps(
        {"raw": corr_of(X),
         "A: minus global PC1": corr_of(renorm(X - A["global_comp"])),
         "A: minus global & sector": corr_of(renorm(A["resid"]))},
        columns, sectors, "Pipeline A (global-first)", OUT / "q6_corr_steps_A.png")
    plot_corr_steps(
        {"raw": corr_of(X),
         "B: minus sector modes": corr_of(renorm(B["resid"]))},
        columns, sectors, "Pipeline B (sector-first)", OUT / "q6_corr_steps_B.png")
    plot_mode_corr(B["mode_panel"], B["global_score"], OUT / "q6_sector_mode_corr.png")

    for name, resid in (("A", A["resid"]), ("B", B["resid"])):
        spec = ms.spectrum(pd.DataFrame(renorm(resid), index=wide.index, columns=columns))
        print(f"residual panel {name}: mean corr {spec['mean_corr']:+.4f}, "
              f"n_signal {spec['n_signal']}, top eig {spec['top_eig']:.1f}")

    # ---- variance shares ------------------------------------------------- #
    share_rows = []
    for pipe, D in (("A", A), ("B", B)):
        for i, c in enumerate(columns):
            share_rows.append({
                "ticker": c, "sector": sectors.get(c), "pipeline": pipe,
                "global": D["share_global"][i], "sector_specific": D["share_sector"][i],
                "idio": 1 - D["share_global"][i] - D["share_sector"][i]})
    shares = pd.DataFrame(share_rows)
    shares.to_csv(OUT / "q6_variance_shares.csv", index=False)
    plot_variance_split(shares, OUT / "q6_variance_split.png")

    # ---- the signals themselves (global + sector-specific), cumulative --- #
    dates = pd.to_datetime(wide.index)
    within_corr = {k: float(ms.offdiag(corr_of(X[:, idx])).mean())
                   for k, idx in blocks.items()}
    high_sectors = sorted(within_corr, key=within_corr.get, reverse=True)[:2]
    chosen = high_sectors + [s for s in LOW_CORR_SECTORS if s in blocks]
    print(f"\nwithin-sector mean corr: "
          f"{ {k: round(v, 3) for k, v in sorted(within_corr.items(), key=lambda x: -x[1])} }")
    print(f"chosen sectors (high-corr {high_sectors} + low-corr {LOW_CORR_SECTORS})")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(dates, np.cumsum(A["global_score"]), color="#4c72b0", lw=1.5,
                 label="global A (panel PC1)")
    axes[0].plot(dates, np.cumsum(B["global_score"]), color="#d64545", lw=1.2, ls="--",
                 label=f"global B (PC1 of sector modes), corr={g_corr:.3f}")
    axes[0].set_title("Cumulative GLOBAL signals (standardized units)")
    axes[0].legend(fontsize=8)
    for k in chosen:
        axes[1].plot(dates, np.cumsum(A["sector_scores"][k]), lw=1.2,
                     label=f"{k} (within-corr {within_corr[k]:.2f})")
    axes[1].axhline(0, color="gray", lw=0.6, ls=":")
    axes[1].set_title("Cumulative SECTOR-SPECIFIC signals, pipeline A (global already removed)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "q6_signals_cumulative.png", dpi=140)
    plt.close(fig)
    print(f"saved: {OUT / 'q6_signals_cumulative.png'}")

    # ---- per-ticker decomposition plots ---------------------------------- #
    for k in chosen:
        safe = k.replace(" ", "_").lower()
        plot_ticker_decomposition(wide, mu, sigma, A, blocks, columns, sectors,
                                  meta["market_cap"], k,
                                  OUT / f"q6_cumret_{safe}.png")

    print(f"\n=== Q6 verdict ===")
    agg = shares.groupby(["pipeline"])[["global", "sector_specific", "idio"]].mean()
    print("mean variance shares across all tickers:")
    print(agg.round(3).to_string())
    hi, lo = high_sectors[0], [s for s in LOW_CORR_SECTORS if s in blocks][0]
    a = shares[(shares.pipeline == "A")]
    print(f"\n[{hi}] global {a[a.sector == hi]['global'].mean():.0%} + sector "
          f"{a[a.sector == hi]['sector_specific'].mean():.0%} vs "
          f"[{lo}] global {a[a.sector == lo]['global'].mean():.0%} + sector "
          f"{a[a.sector == lo]['sector_specific'].mean():.0%}")
    print(f"artifacts -> {OUT}")


if __name__ == "__main__":
    main()
