"""
===========
Reproduce the core Random Matrix Theory analysis from
  - Potters, Bouchaud & Laloux, "Financial Applications of RMT" (2005)
  - Nobi et al., "RMT and Cross-correlations in ... Stock Indices"

on a panel of stock prices.

WHAT IT SHOWS
-------------
1. P(C_ij)         -- the distribution of pairwise correlation coefficients
                      (Nobi Fig. 2). Real market: shifted positive. Random: ~0.
2. Eigenvalue density vs the Marchenko-Pastur (MP) law (Potters Fig. 2,
   Nobi Fig. 3). The bulk sits inside [lambda_-, lambda_+]; the "market mode"
   leaks out far to the right.
3. THE INDEPENDENCE DEMONSTRATION. We independently shuffle each ticker's
   return series in time. That preserves every ticker's own volatility/fat
   tails but destroys all *cross*-correlation -> genuinely independent columns.
   Its spectrum collapses onto the MP law. That is what "the data is
   independent / the bulk is noise" means, made visible.

DROP-IN USAGE
-------------
    import pandas as pd
    from rmt_demo import run_analysis
    # price_df: MultiIndex (ticker, date), one price column (e.g. 'close')
    run_analysis(price_df, price_col="close", out_prefix="myrun")

Run standalone (no data needed) to see it on synthetic one-factor data:
    python rmt_demo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from findata.preprocess.rmt import iterative_bulk_variance
from findata.analysis.market_structure import (select_correlated_group,
                                               select_uncorrelated_group)
from _common import OUT_ROOT, load_panel, build_parser

OUT = OUT_ROOT/'rsi' / "q3_signal_vs_sample_noise"

# ---------------------------------------------------------------------------
# 1. Synthetic panel -- a ONE-FACTOR model (Potters Eq. 33-34).
#    Ground truth = 1 real cross-sectional factor ("market") + pure noise.
#    Everything below should therefore recover exactly ONE eigenvalue outside
#    the MP bulk, and a bulk that matches MP. Delete this when using real data.
# ---------------------------------------------------------------------------
def make_synthetic_prices(n_tickers=200, n_days=1300,
                          market_strength=0.4, seed=0) -> pd.DataFrame:
    """r_it = beta_i * phi_t + eps_it.  phi_t = common market factor (shared),
    eps_it = idiosyncratic noise (independent across tickers). Returns a price
    DataFrame with a (ticker, date) MultiIndex and a 'close' column."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"STK{i:03d}" for i in range(n_tickers)]

    phi = rng.standard_normal(n_days)                       # market factor (T,)
    beta = rng.uniform(0.5, 1.5, size=n_tickers)            # exposures    (N,)
    eps = rng.standard_normal((n_days, n_tickers))          # noise    (T, N)

    rets = (market_strength * np.outer(phi, beta)
            + np.sqrt(1 - market_strength**2) * eps) * 0.01
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))

    wide = pd.DataFrame(prices, index=dates, columns=tickers)
    long = wide.stack().rename("close")
    long.index = long.index.set_names(["date", "ticker"])
    return long.reorder_levels(["ticker", "date"]).sort_index().to_frame()


# ---------------------------------------------------------------------------
# 2. Panel -> standardized returns -> correlation matrix
# ---------------------------------------------------------------------------
def panel_to_returns(price_df: pd.DataFrame, price_col: str | None = None) -> pd.DataFrame:
    """(ticker, date) price panel -> (T x N) log-return frame (rows=dates)."""
    if price_col is None:
        price_col = price_df.columns[0]
    wide = price_df[price_col].unstack("ticker").sort_index()   # date x ticker
    logret = np.log(wide).diff()
    # keep tickers with full history, then rows with no gaps (equal-time matrix)
    logret = logret.dropna(axis=1, thresh=int(0.9 * len(logret)))
    logret = logret.dropna(axis=0, how="any")
    return logret


def standardize(returns: pd.DataFrame) -> np.ndarray:
    """x_it = (r_it - mean_i) / sigma_i  (Nobi Eq. 2). Correlation of these == C."""
    X = returns.values.astype(float)
    X = X - X.mean(0, keepdims=True)
    X = X / (X.std(0, ddof=0, keepdims=True) + 1e-12)
    return X                                                    # (T, N)


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """E_ij = (1/T) sum_t x_it x_jt.  Unit diagonal; trace = N; sum(eig) = N."""
    T = X.shape[0]
    return (X.T @ X) / T


# ---------------------------------------------------------------------------
# 3. Marchenko-Pastur law (Potters Eq. 23; equivalently Nobi Eq. 4-5)
# ---------------------------------------------------------------------------
def mp_bounds(q: float, sigma2: float = 1.0):
    """q = N/T (<=1). Bulk edges lambda_+/- = sigma2 * (1 +/- sqrt(q))^2."""
    s = np.sqrt(q)
    return sigma2 * (1 - s) ** 2, sigma2 * (1 + s) ** 2


def mp_density(lam, q: float, sigma2: float = 1.0):
    """Theoretical MP density; zero outside [lambda_-, lambda_+]."""
    lo, hi = mp_bounds(q, sigma2)
    lam = np.asarray(lam, float)
    out = np.zeros_like(lam)
    m = (lam > lo) & (lam < hi)
    out[m] = np.sqrt((hi - lam[m]) * (lam[m] - lo)) / (2 * np.pi * q * sigma2 * lam[m])
    return out


# ---------------------------------------------------------------------------
# 4. THE INDEPENDENCE NULL: shuffle each ticker in time, independently.
#    Preserves marginals (vol, tails), destroys all cross-correlation.
# ---------------------------------------------------------------------------
def shuffle_null(returns: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    R = returns.values.copy()
    for j in range(R.shape[1]):
        R[:, j] = R[rng.permutation(R.shape[0]), j]
    return pd.DataFrame(R, index=returns.index, columns=returns.columns)


# ---------------------------------------------------------------------------
# 5. Helpers: off-diagonal coefficients, eigenvalues, bulk noise variance
# ---------------------------------------------------------------------------
def offdiag(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(C, k=1)
    return C[iu]


def eigvals_desc(C: np.ndarray) -> np.ndarray:
    w = np.linalg.eigvalsh(C)
    return w[::-1]


def fit_bulk_sigma2(eigs: np.ndarray, q: float) -> tuple[float, float, int]:
    """Iterative bulk-mean noise variance: sigma2 = mean(bulk), recompute the
    MP edge from it, reclassify, and RECURSE until no more eigenvalues shift to
    signal (the single-pass version stops after one reclassification and
    overestimates sigma2 whenever the shrunken edge exposes more signal).

    Delegates to findata.preprocess.rmt.iterative_bulk_variance — note the
    convention flip: this script uses q = N/T, rmt uses q = T/N.

    Returns (sigma2, lambda_plus, n_signal).
    """
    return iterative_bulk_variance(eigs, 1.0 / q)


# ---------------------------------------------------------------------------
# 6. Deliverables
# ---------------------------------------------------------------------------
def run_analysis(wide, out_prefix='rmt'):
    OUT.mkdir(parents=True, exist_ok=True)
    T, N = wide.shape
    q = N / T

    X = standardize(wide)
    C = correlation_matrix(X)
    eigs = eigvals_desc(C)

    null_ret = shuffle_null(wide, seed=args.seed)
    Cn = correlation_matrix(standardize(null_ret))
    eigs_n = eigvals_desc(Cn)

    lo, hi = mp_bounds(q, 1.0)
    sigma2, hi_s, n_signal = fit_bulk_sigma2(eigs, q)
    lo_s, _ = mp_bounds(q, sigma2)

    off, off_n = offdiag(C), offdiag(Cn)

    # ---- diagnostics table ----
    print("=" * 66)
    print(f"  N tickers        : {N}")
    print(f"  T return obs      : {T}")
    print(f"  q = N/T           : {q:.4f}      (RMT is trustworthy when q<1 and N,T large)")
    print(f"  MP bulk (sig2=1)  : [{lo:.3f}, {hi:.3f}]")
    print(f"  fitted bulk sig2  : {sigma2:.3f}   (mean of bulk eigenvalues)")
    print(f"  MP bulk (fitted)  : [{lo_s:.3f}, {hi_s:.3f}]")
    print(f"  # eigs > edge     : {n_signal}   <- real 'signal' modes (market + sectors)")
    print(f"  largest eigenvalue: {eigs[0]:.3f}   <- the 'market mode'")
    print("-" * 66)
    print(f"  REAL   <C_ij> = {off.mean():+.4f}   std = {off.std():.4f}")
    print(f"  SHUFFLED <C_ij>= {off_n.mean():+.4f}   std = {off_n.std():.4f}  (~1/sqrt(T)={1/np.sqrt(T):.4f})")
    print("=" * 66)

    # ---- Figure 1: P(C_ij) ----
    fig1, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.4, 1.0, 60)
    ax.hist(off, bins=bins, density=True, alpha=0.6, color="#d64545",
            label=f"REAL market  (mean {off.mean():+.2f})")
    ax.hist(off_n, bins=bins, density=True, alpha=0.6, color="#4c72b0",
            label=f"SHUFFLED null (mean {off_n.mean():+.2f})")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$C_{ij}$  (pairwise correlation)")
    ax.set_ylabel(r"$P(C_{ij})$")
    ax.set_title("Distribution of pairwise correlations (Nobi Fig. 2)\n"
                 "real market is shifted positive; independent null centers on 0")
    ax.legend()
    fig1.tight_layout()
    fig1.savefig(OUT / f"{out_prefix}_1_Cij_distribution.png", dpi=130)

    # ---- Figure 2: eigenvalue density vs MP ----
    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [3, 1]})
    grid = np.linspace(max(1e-3, lo_s * 0.5), hi_s * 1.3, 400)
    axL.hist(eigs, bins=np.linspace(0, hi_s * 1.4, 60), density=True,
             alpha=0.55, color="#d64545", label="REAL eigenvalues")
    axL.hist(eigs_n, bins=np.linspace(0, hi_s * 1.4, 60), density=True,
             alpha=0.45, color="#4c72b0", label="SHUFFLED eigenvalues")
    axL.plot(grid, mp_density(grid, q, sigma2), "k-", lw=2,
             label=f"MP law (q={q:.3f}, $\\sigma^2$={sigma2:.2f})")
    axL.axvline(hi_s, color="k", ls="--", lw=1, label=r"MP edge $\lambda_+$")
    axL.set_xlabel(r"$\lambda$"); axL.set_ylabel(r"$\rho(\lambda)$")
    axL.set_title("Bulk matches MP; shuffled matches even better\n"
                  "(bulk = noise; deviations = real structure)")
    axL.legend(fontsize=8)

    # zoom on the leaked signal eigenvalues
    ntop = min(11, N)
    axR.plot(range(1, ntop + 1), eigs[:ntop], "o-", color="#d64545",
             label="REAL")
    axR.plot(range(1, ntop + 1), eigs_n[:ntop], "s-", color="#4c72b0",
             label="SHUFFLED")
    axR.axhline(hi_s, color="k", ls="--", lw=1)
    axR.annotate("market\nmode", xy=(1, eigs[0]),
                 xytext=(3, eigs[0] * 0.8), fontsize=9,
                 arrowprops=dict(arrowstyle="->"))
    axR.set_xlabel("rank"); axR.set_ylabel(r"$\lambda$")
    axR.set_title("Top eigenvalues")
    axR.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / f"{out_prefix}_2_eigenvalue_density.png", dpi=130)

    return dict(q=q, sigma2=sigma2, n_signal=n_signal,
                lambda_plus=hi_s, largest=eigs[0], eigs=eigs, C=C)


GROUP_SIZE = 30



if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    wide, meta = load_panel(args)

    print("\n############ FULL PANEL ############")
    full = run_analysis(wide, out_prefix=args.indicator)

    names = list(wide.columns)
    corr_group = select_correlated_group(full["C"], names, size=GROUP_SIZE)
    uncorr_group = select_uncorrelated_group(full["C"], names, size=GROUP_SIZE)
    print(f"\ncorrelated group   ({len(corr_group)}): {corr_group}")
    print(f"uncorrelated group ({len(uncorr_group)}): {uncorr_group}")

    print("\n############ CORRELATED GROUP ############")
    run_analysis(wide[corr_group], out_prefix=f"{args.indicator}_correlated")

    print("\n############ UNCORRELATED GROUP ############")
    run_analysis(wide[uncorr_group], out_prefix=f"{args.indicator}_uncorrelated")

    print("\nsaved: rmt{,_correlated,_uncorrelated}_1_Cij_distribution.png / _2_eigenvalue_density.png")
