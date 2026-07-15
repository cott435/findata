#!/usr/bin/env python3
"""
RMT feature-validation pipeline for multi-feature, multi-ticker panels.

Stages (each produces a deliverable plot demonstrating WHY the choice matters):

  1. Null construction ........ circular-shift null vs full permutation vs analytic
                                Marchenko-Pastur. Shows that autocorrelated features
                                need an autocorrelation-preserving null (T_eff, not T).
  2. Per-feature N x N ........ spectrum per feature vs its null band; lambda_1/N as a
                                "market dominance" score per feature.
  3. Market-mode removal ...... regression residuals (Plerou Eq. 19), re-standardize,
                                re-diagonalize. Sector structure emerges from the bulk.
  4. F x F redundancy map ..... pooled cross-feature correlation; data-driven groups
                                vs semantic groups; effective rank of the feature set.
  5. Lower-edge structure ..... IPR localization at BOTH spectrum edges; smallest
                                eigenvectors = pairs / reversion book. Why naive
                                eigenvalue clipping of the bottom edge is destructive.
  6. ZCA stress test .......... out-of-sample variance of whitened components: raw
                                ZCA amplifies estimation noise in small-eigenvalue
                                directions; clip-cleaned ZCA does not.
  7. Predictive gate .......... rectangular SVD of (features -> forward returns)
                                cross-correlation vs circular-shift null
                                (Bouchaud-Potters random SVD / CCA). Counts how many
                                independent predictive directions exist at all.

The synthetic market has KNOWN ground truth (1 market mode with vol clustering,
6 sectors, 4 strongly-correlated pairs, and a weak OU mean-reversion component in
log-prices), so every plot can be checked against what was planted.

To run on real data: replace `simulate_market()` with a loader that returns a
(T x N) log-price DataFrame, and keep everything downstream unchanged.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import os, json, time
from findata.configs import EXPERIMENT_DIR

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED       = 7
T_RAW      = 2700              # simulated days before warmup trim
N          = 120               # tickers
N_SECTORS  = 6
PAIRS      = [(0, 1), (25, 26), (50, 51), (90, 91)]   # planted "TXN/MU"-style pairs
WARMUP     = 200               # rows dropped so all features are well-defined
NULL_REPS  = 80                # circular-shift replicates for N x N nulls
SVD_REPS   = 250               # circular-shift replicates for predictive SVD null
H_LIST     = [1, 5, 10, 20]    # forward-return horizons for the predictive gate
OU_THETA   = 0.05              # planted mean reversion: ~20-day half-life scale
OUT        = EXPERIMENT_DIR / "rmt_pipeline"

rng = np.random.default_rng(SEED)
OUT.mkdir(parents=True, exist_ok=True)
summary = {}

# ----------------------------------------------------------------------------
# 0. Synthetic market with known ground truth
# ----------------------------------------------------------------------------
def simulate_market():
    """Returns log-price DataFrame (T_RAW x N) and sector labels.

    r_it = beta_i * m_t  +  s_{sector(i),t}  +  pair_{k(i),t}  +  eps_it  +  du_it
      m_t   : market factor with AR(1) log-volatility (vol clustering)
      s     : 6 sector factors
      pair  : extra shared factor for 4 planted pairs (strong co-movement)
      u     : OU process in log-price -> genuine, weak mean-reversion signal
    """
    # market factor with volatility clustering
    logv = np.zeros(T_RAW)
    for t in range(1, T_RAW):
        logv[t] = 0.97 * logv[t - 1] + 0.10 * rng.normal()
    m = np.exp(logv) * rng.normal(size=T_RAW)

    beta    = rng.normal(1.0, 0.2, N)
    sectors = np.repeat(np.arange(N_SECTORS), N // N_SECTORS)
    s       = rng.normal(0.0, 0.6, (T_RAW, N_SECTORS))
    idio    = rng.normal(0.0, 1.0, (T_RAW, N))

    r = beta[None, :] * m[:, None] + s[:, sectors] + idio
    pf = rng.normal(0.0, 2.0, (T_RAW, len(PAIRS)))   # strong pairs, rho ~ 0.8
    for k, (i, j) in enumerate(PAIRS):
        r[:, i] += pf[:, k]
        r[:, j] += pf[:, k]

    # OU mispricing in log-price: adds Delta-u to returns; gap features can see u
    u = np.zeros((T_RAW, N))
    eta = np.sqrt(2 * OU_THETA) * 1.0
    for t in range(1, T_RAW):
        u[t] = (1 - OU_THETA) * u[t - 1] + eta * rng.normal(size=N)
    du = np.vstack([u[:1] * 0, np.diff(u, axis=0)])

    r_total = 0.01 * (r + du)                       # ~1-2% daily vol scale
    lp = np.log(100.0) + np.cumsum(r_total, axis=0)
    return pd.DataFrame(lp), sectors


# ----------------------------------------------------------------------------
# 0b. Feature engineering (RSI at native windows, EMA velocity / acceleration,
#     log-ratio price velocity, gap-to-EMA, realized vol)
# ----------------------------------------------------------------------------
def ema(df, span):
    return df.ewm(span=span, adjust=False).mean()

def compute_features(lp):
    p = np.exp(lp)
    r = lp.diff()
    f = {}
    for w in (14, 28):                              # native windows, NOT double-smoothed
        d  = p.diff()
        ru = d.clip(lower=0).ewm(alpha=1 / w, adjust=False).mean()
        rd = (-d).clip(lower=0).ewm(alpha=1 / w, adjust=False).mean()
        f[f"rsi{w}"] = 100 * ru / (ru + rd + 1e-12)
    sma, sd = p.rolling(20).mean(), p.rolling(20).std()
    f["cci20"]     = (p - sma) / (0.015 * sd * np.sqrt(np.pi / 2) + 1e-12)
    f["mom20"]     = lp - lp.shift(20)
    vel            = np.log(ema(p, 12)) - np.log(ema(p, 26))   # log-ratio velocity
    f["ema_vel"]   = vel
    f["ema_acc"]   = vel - vel.ewm(span=9, adjust=False).mean()
    f["gap50"]     = lp - np.log(ema(p, 50))                   # carries the OU signal
    f["vol20"]     = r.rolling(20).std()
    f["rsi14_vel"] = f["rsi14"] - f["rsi14"].ewm(span=9, adjust=False).mean()
    f = {k: v.iloc[WARMUP:].reset_index(drop=True) for k, v in f.items()}
    return f


# ----------------------------------------------------------------------------
# Core RMT machinery
# ----------------------------------------------------------------------------
def standardize(A):
    A = np.asarray(A, float)
    return (A - A.mean(0)) / (A.std(0) + 1e-12)

def corr_eig(Z, vectors=False):
    C = Z.T @ Z / len(Z)
    return np.linalg.eigh(C) if vectors else np.linalg.eigvalsh(C)

def circular_shift_null(Z, reps=NULL_REPS, rng=rng):
    """Roll each column by an independent random offset.
    Preserves every series' autocorrelation exactly; destroys cross-sectional
    alignment. This is the correct empirical null for autocorrelated features."""
    Tn, Nn = Z.shape
    ev = np.empty((reps, Nn))
    for k in range(reps):
        off = rng.integers(1, Tn - 1, Nn)
        Zs = np.stack([np.roll(Z[:, j], off[j]) for j in range(Nn)], axis=1)
        ev[k] = corr_eig(Zs)
    return ev

def permutation_null(Z, reps=NULL_REPS, rng=rng):
    """Independent full time-permutation per column: destroys cross-sectional
    alignment AND autocorrelation -> null band is too narrow for windowed features."""
    Tn, Nn = Z.shape
    ev = np.empty((reps, Nn))
    for k in range(reps):
        Zs = np.stack([rng.permutation(Z[:, j]) for j in range(Nn)], axis=1)
        ev[k] = corr_eig(Zs)
    return ev

def mp_edges(q, s2=1.0):
    return s2 * (1 - np.sqrt(q)) ** 2, s2 * (1 + np.sqrt(q)) ** 2

def tau_int(x, max_lag=250):
    """Integrated autocorrelation time  tau = 1 + 2 * sum rho_k  (Sokal-style cutoff)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    tau = 1.0
    for k in range(1, min(max_lag, len(ac) - 1)):
        if ac[k] < 0.05:
            break
        tau += 2 * ac[k]
    return tau, ac[:max_lag]

def ipr(U):
    return (U ** 4).sum(0)


# ----------------------------------------------------------------------------
# Plot 1  --  Why the null must preserve autocorrelation
# ----------------------------------------------------------------------------
def plot_null_choice(lp_indep):
    """Feature panel from INDEPENDENT tickers (zero true cross-correlation).
    Any eigenvalue flagged above a null band is, by construction, a false positive."""
    feats = compute_features(lp_indep)
    Z = standardize(feats["rsi14"].values)
    Tn, Nn = Z.shape
    ev_real = corr_eig(Z)

    ev_circ = circular_shift_null(Z, reps=NULL_REPS)
    ev_perm = permutation_null(Z, reps=NULL_REPS)
    circ_hi = np.quantile(ev_circ.max(1), 0.99)
    perm_hi = np.quantile(ev_perm.max(1), 0.99)

    taus = [tau_int(Z[:, j])[0] for j in range(0, Nn, 4)]
    tau = float(np.median(taus))
    q, q_eff = Nn / Tn, Nn * tau / Tn
    lam_mp_hi = mp_edges(q)[1]
    lam_eff_hi = mp_edges(q_eff)[1]

    fp_mp   = int((ev_real > lam_mp_hi).sum())
    fp_perm = int((ev_real > perm_hi).sum())
    fp_circ = int((ev_real > circ_hi).sum())
    summary["null_choice"] = dict(tau=tau, q=q, q_eff=q_eff,
                                  false_pos_MP=fp_mp, false_pos_perm=fp_perm,
                                  false_pos_circ=fp_circ)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    _, acf_r = tau_int(np.diff(lp_indep.values[:, 0]))
    _, acf_f = tau_int(Z[:, 0])
    ax[0].plot(acf_r[:80], lw=1.5, label="daily log returns")
    ax[0].plot(acf_f[:80], lw=1.5, label="RSI-14 feature")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(title=f"Windowed features are heavily autocorrelated\n"
                    f"(integrated autocorr. time \u03c4 \u2248 {tau:.0f} days "
                    f"\u2192 T_eff \u2248 T/{tau:.0f})",
              xlabel="lag (days)", ylabel="autocorrelation")
    ax[0].legend()

    ax[1].hist(ev_real, bins=60, density=True, color="steelblue", alpha=0.75,
               label="RSI-14 spectrum, ZERO true cross-corr")
    ax[1].axvline(lam_mp_hi, color="crimson", ls="--", lw=2,
                  label=f"MP edge, q=N/T \u2192 {fp_mp} false positives")
    ax[1].axvline(perm_hi, color="darkorange", ls="--", lw=2,
                  label=f"permutation null \u2192 {fp_perm} false positives")
    ax[1].axvline(circ_hi, color="seagreen", ls="-", lw=2.5,
                  label=f"circular-shift null \u2192 {fp_circ} false positives")
    ax[1].axvline(lam_eff_hi, color="gray", ls=":", lw=2,
                  label=f"MP with q_eff=N\u03c4/T (\u2248 shift null)")
    ax[1].set(title="Null bands on a panel with NO real correlation:\n"
                    "only the autocorrelation-preserving null is calibrated",
              xlabel="eigenvalue \u03bb", ylabel="density")
    ax[1].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_null_choice.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 2  --  Per-feature N x N spectra and market dominance
# ----------------------------------------------------------------------------
def plot_per_feature_spectra(feats, ret):
    show = ["rsi14", "ema_vel", "vol20"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [("returns", standardize(ret))] + [(k, standardize(feats[k].values)) for k in show]
    lam1_over_N, dev_counts = {}, {}

    for ax, (name, Z) in zip(axes, panels):
        ev = corr_eig(Z)
        null = circular_shift_null(Z, reps=NULL_REPS)
        hi = np.quantile(null.max(1), 0.99)
        lo = np.quantile(null.min(1), 0.01)
        n_dev = int((ev > hi).sum())
        lam1_over_N[name] = ev[-1] / len(ev)
        dev_counts[name] = n_dev
        ax.hist(ev[:-1], bins=50, density=True, color="steelblue", alpha=0.8)
        ax.axvspan(lo, hi, color="gray", alpha=0.25, label="null band (99%)")
        ax.plot(ev[ev > hi], np.zeros((ev > hi).sum()), "v", color="crimson", ms=8,
                label=f"{n_dev} deviating \u03bb")
        ax.axvline(ev[-1], color="crimson", lw=1)
        ax.set(title=f"{name}\n\u03bb\u2081={ev[-1]:.1f}  (\u03bb\u2081/N={ev[-1]/len(ev):.2f})",
               xlabel="\u03bb", yscale="log")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density (log)")
    fig.suptitle("Per-feature N\u00d7N spectra: each feature has its own market mode, "
                 "sector tier, and noise bulk", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_per_feature_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # market-dominance bar for ALL features
    fig, ax = plt.subplots(figsize=(9, 4))
    names, vals = [], []
    for k, v in [("returns", ret)] + [(k, f.values) for k, f in feats.items()]:
        ev = corr_eig(standardize(v))
        names.append(k); vals.append(ev[-1] / len(ev))
    order = np.argsort(vals)[::-1]
    ax.bar(np.array(names)[order], np.array(vals)[order], color="steelblue")
    ax.set(ylabel="\u03bb\u2081 / N   (share of variance in market mode)",
           title="Market dominance per feature: high \u03bb\u2081/N = mostly redundant beta;\n"
                 "low \u03bb\u2081/N = cross-sectional differentiation lives here")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(f"{OUT}/02b_market_dominance.png", dpi=150)
    plt.close(fig)
    summary["market_dominance"] = dict(zip(names, map(float, vals)))


# ----------------------------------------------------------------------------
# Plot 3  --  Market-mode removal by regression (Plerou Eq. 19)
# ----------------------------------------------------------------------------
def remove_mode(Z, u1):
    g = Z @ u1                                   # market-mode time series
    beta = (Z.T @ g) / (g @ g)
    resid = Z - np.outer(g, beta)
    return standardize(resid)

def plot_mode_removal(ret, sectors):
    Z = standardize(ret)
    ev, U = corr_eig(Z, vectors=True)
    hi0 = np.quantile(circular_shift_null(Z).max(1), 0.99)

    Zr = remove_mode(Z, U[:, -1])
    ev_r, U_r = corr_eig(Zr, vectors=True)
    hi1 = np.quantile(circular_shift_null(Zr).max(1), 0.99)
    n_dev0 = int((ev > hi0).sum())
    n_dev1 = int((ev_r > hi1).sum())
    summary["mode_removal"] = dict(dev_before=n_dev0, dev_after=n_dev1,
                                   planted_sectors=N_SECTORS)

    fig = plt.figure(figsize=(14, 4.6))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.hist(ev, bins=60, density=True, color="steelblue", alpha=0.8)
    ax0.axvline(hi0, color="seagreen", lw=2, label="null edge")
    ax0.axvline(ev[-1], color="crimson", lw=1.5, label=f"\u03bb\u2081={ev[-1]:.0f}")
    ax0.set(title=f"Before removal: \u03bb\u2081 dwarfs everything\n({n_dev0} deviating incl. market)",
            xlabel="\u03bb", yscale="log"); ax0.legend(fontsize=8)

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.hist(ev_r, bins=60, density=True, color="steelblue", alpha=0.8)
    ax1.axvline(hi1, color="seagreen", lw=2, label="recomputed null edge")
    for lam in ev_r[ev_r > hi1]:
        ax1.axvline(lam, color="crimson", lw=1, alpha=0.7)
    ax1.set(title=f"After regression on market mode:\n{n_dev1} deviating \u03bb "
                  f"(planted: {N_SECTORS} sectors + {len(PAIRS)} pair modes)", xlabel="\u03bb", yscale="log")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3)
    k = max(n_dev1, 1)
    L = U_r[:, -k:][:, ::-1]                      # top residual eigenvectors
    order = np.argsort(sectors)
    im = ax2.imshow(L[order], aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    for b in np.where(np.diff(np.sort(sectors)))[0]:
        ax2.axhline(b + 0.5, color="k", lw=0.6)
    ax2.set(title="Residual eigenvector loadings\n(rows sorted by true sector \u2192 blocks)",
            xlabel="residual eigenvector rank", ylabel="ticker (sector-sorted)")
    fig.colorbar(im, ax=ax2, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_mode_removal.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 4  --  F x F redundancy map (pooled across tickers and time)
# ----------------------------------------------------------------------------
def pool_features(feats, tmax=None):
    names = list(feats)
    cols = []
    for k in names:
        A = feats[k].values if tmax is None else feats[k].values[:tmax]
        cols.append(standardize(A).ravel(order="F"))    # per-ticker z-score, ticker-major
    return np.column_stack(cols), names

def plot_redundancy(feats):
    X, names = pool_features(feats)
    C = np.corrcoef(X.T)
    d = squareform(1 - np.abs(C), checks=False)
    order = hierarchy.leaves_list(hierarchy.linkage(d, "average"))
    ev = np.linalg.eigvalsh(C)
    eff_rank = ev.sum() ** 2 / (ev ** 2).sum()
    summary["redundancy"] = dict(F=len(names), effective_rank=float(eff_rank))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    im = ax[0].imshow(C[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(names))); ax[0].set_yticks(range(len(names)))
    ax[0].set_xticklabels(np.array(names)[order], rotation=45, ha="right", fontsize=8)
    ax[0].set_yticklabels(np.array(names)[order], fontsize=8)
    ax[0].set_title("F\u00d7F pooled correlation, hierarchically ordered:\n"
                    "data-driven blocks \u2260 semantic groups")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    ax[1].bar(range(1, len(ev) + 1), ev[::-1], color="steelblue")
    ax[1].axhline(1, color="gray", ls=":")
    ax[1].set(title=f"Feature-space scree: effective rank \u2248 {eff_rank:.1f} of F={len(names)}\n"
                    "(overlapping windows collapse; contrast features add rank)",
              xlabel="component", ylabel="eigenvalue of F\u00d7F corr")
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_feature_redundancy.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 5  --  Lower-edge localization: the reversion book lives there
# ----------------------------------------------------------------------------
def plot_lower_edge(ret):
    Z = standardize(ret)
    ev, U = corr_eig(Z, vectors=True)
    null = circular_shift_null(Z)
    hi = np.quantile(null.max(1), 0.99)
    lo = np.quantile(null.min(1), 0.01)
    I = ipr(U)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].semilogy(ev, I, "o", ms=4, color="steelblue", alpha=0.8)
    ax[0].axhline(3 / N, color="gray", ls=":", label="GOE expectation 3/N")
    ax[0].axvspan(lo, hi, color="gray", alpha=0.2, label="null eigenvalue band")
    ax[0].set(title="IPR vs eigenvalue: localization at BOTH edges\n"
                    "top edge = market/sectors, bottom edge = pairs (reversion)",
              xlabel="\u03bb", ylabel="IPR (log)")
    ax[0].legend(fontsize=8)

    k_star = 0                                      # smallest eigenvector
    u_min = U[:, k_star]
    small4 = [sorted(int(t) for t in np.argsort(np.abs(U[:, k]))[-2:]) for k in range(4)]
    n_pairs_found = sum(any(set(t2) == set(p) for p in PAIRS) for t2 in small4)
    ax[1].stem(np.arange(N), u_min)
    top2 = np.argsort(np.abs(u_min))[-2:]
    is_pair = any(set(top2) == set(p) for p in PAIRS)
    ax[1].set(title=f"Smallest eigenvector (\u03bb={ev[k_star]:.2f} < null lower edge {lo:.2f}):\n"
                    f"localized, opposite-sign on tickers {sorted(top2)} "
                    f"({'a PLANTED pair' if is_pair else 'pair candidates'}) \u2192 long-short book.\n"
                    f"The {n_pairs_found}/4 smallest eigenvectors recover the 4 planted pairs.\n"
                    "Clipping the bottom edge would erase exactly this.",
              xlabel="ticker index", ylabel="component")
    summary["lower_edge"] = dict(lambda_min=float(ev[k_star]), null_lo=float(lo),
                                 smallest4_top2=small4, pairs_recovered=int(n_pairs_found),
                                 planted_pair=bool(is_pair),
                                 sign_opposite=bool(u_min[top2[0]] * u_min[top2[1]] < 0))
    fig.tight_layout()
    fig.savefig(f"{OUT}/05_lower_edge.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 6  --  ZCA stress test: raw vs cleaned whitening, out of sample
# ----------------------------------------------------------------------------
def plot_zca(feats, fname="ema_vel"):
    Z = standardize(feats[fname].values)            # autocorrelated panel: the case
    Tn = len(Z)                                     # that actually bites when
    Ztr, Zte = standardize(Z[: Tn // 2]), standardize(Z[Tn // 2:])   # whitening features
    ev, U = corr_eig(Ztr, vectors=True)
    ev = np.maximum(ev, 1e-10)
    null = circular_shift_null(Ztr)
    hi = np.quantile(null.max(1), 0.99)
    lo = np.quantile(null.min(1), 0.01)
    tau = float(np.median([tau_int(Ztr[:, j])[0] for j in range(0, N, 6)]))
    q, q_eff = N / len(Ztr), N * tau / len(Ztr)

    # cleaned spectrum: clip the BULK to its trace-preserving mean; keep both
    # deviating tops (market/sectors) and deviating bottoms (localized pairs)
    bulk = (ev > lo) & (ev < hi)
    ev_clean = ev.copy()
    ev_clean[bulk] = ev[bulk].mean()

    proj = Zte @ U                                  # test data in train eigenbasis
    var_te = proj.var(0)
    ratio_raw = var_te / ev                         # variance of raw-ZCA components oos
    ratio_cln = var_te / ev_clean

    summary["zca"] = dict(feature=fname, tau=tau, q=q, q_eff=q_eff,
                          worst_raw_bulk=float(ratio_raw[bulk].max()),
                          worst_clean_bulk=float(ratio_cln[bulk].max()),
                          bulk_fraction=float(bulk.mean()))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ranks = np.arange(1, N + 1)                     # ascending eigenvalue rank
    ax.semilogy(ranks, ratio_raw, "o-", ms=3, lw=0.8, color="crimson",
                label=f"raw ZCA  C^{{-1/2}}  (worst bulk blow-up {ratio_raw[bulk].max():.0f}\u00d7)")
    ax.semilogy(ranks, ratio_cln, "o-", ms=3, lw=0.8, color="seagreen",
                label=f"cleaned ZCA (bulk clipped, edges kept; worst {ratio_cln[bulk].max():.1f}\u00d7)")
    ax.axhline(1, color="k", lw=0.8)
    ax.fill_between(ranks, ratio_cln.min() / 2, ratio_raw.max() * 2, where=bulk,
                    color="gray", alpha=0.15, label="bulk (noise) directions")
    ax.set(xlabel="eigenvalue rank (small \u03bb \u2192 large \u03bb)",
           ylabel="out-of-sample variance of whitened component (log)",
           title=f"ZCA stress test on '{fname}' panel: q=N/T={q:.2f} looks safe, but "
                 f"\u03c4\u2248{tau:.0f} \u21d2 q_eff\u2248{q_eff:.1f}\n"
                 "raw 1/\u221a\u03bb weighting amplifies exactly the directions whose \u03bb "
                 "was underestimated in-sample")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/06_zca_stress.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 7  --  Predictive gate: rectangular SVD of features -> forward returns
# ----------------------------------------------------------------------------
def plot_predictive_svd(feats, lp_trim):
    names = list(feats)
    Tf = len(next(iter(feats.values())))
    tmax = Tf - max(H_LIST)

    Zf = {k: standardize(feats[k].values[:tmax]) for k in names}          # (tmax x N)
    Y = []
    for H in H_LIST:
        fwd = (lp_trim.shift(-H) - lp_trim).values[:tmax]
        Y.append(standardize(fwd).ravel(order="F"))
    Y = np.column_stack(Y)                                                # (tmax*N x M)

    def pooled_E(offsets=None):
        cols = []
        for k in names:
            A = Zf[k]
            if offsets is not None:                                       # same shift for
                A = np.stack([np.roll(A[:, j], offsets[j])                # all features of
                              for j in range(N)], axis=1)                 # a ticker
            cols.append(A.ravel(order="F"))
        X = np.column_stack(cols)
        return X.T @ Y / len(Y)

    E = pooled_E()
    Uu, s, Vt = np.linalg.svd(E, full_matrices=False)

    s_null = np.empty(SVD_REPS)
    for k in range(SVD_REPS):
        off = rng.integers(1, tmax - 1, N)
        s_null[k] = np.linalg.svd(pooled_E(off), compute_uv=False)[0]
    band = np.quantile(s_null, 0.99)
    n_sig = int((s > band).sum())
    summary["predictive_svd"] = dict(singular_values=[float(x) for x in s],
                                     null_99=float(band), n_above=n_sig)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].bar(range(1, len(s) + 1), s, color=["crimson" if x > band else "steelblue" for x in s])
    ax[0].axhline(band, color="seagreen", lw=2, label="circular-shift null (99%)\n"
                  "handles feature autocorr AND overlapping fwd returns")
    ax[0].set(title=f"Singular values of corr(features, forward returns):\n"
                    f"{n_sig} independent predictive direction(s) exist "
                    f"(1 planted: OU reversion)",
              xlabel="singular value rank", ylabel="singular value")
    ax[0].legend(fontsize=8.5)

    ax[1].bar(names, Uu[:, 0], color="steelblue")
    ax[1].tick_params(axis="x", rotation=35)
    sgn = "reversion (gap/momentum block loads against forward returns)" \
          if Uu[np.array(names) == "gap50", 0] * Vt[0].mean() < 0 else "trend-following"
    ax[1].set(title=f"Feature loadings of top predictive direction\n"
                    f"horizon loadings {np.round(Vt[0], 2)} at H={H_LIST}\n\u2192 {sgn}",
              ylabel="loading")
    fig.tight_layout()
    fig.savefig(f"{OUT}/07_predictive_gate.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    """from findata.preprocess import Volume
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:10]"""

    t0 = time.time()
    lp, sectors = simulate_market()

    # independent-ticker world for null calibration (Plot 1): idio returns only
    lp_indep = pd.DataFrame(np.log(100) + np.cumsum(0.01 * rng.normal(size=(T_RAW, N)), 0))

    lp_trim = lp.iloc[WARMUP:].reset_index(drop=True)
    ret = lp_trim.diff().dropna().values
    feats = compute_features(lp)



    print("[1/7] null construction ..."); plot_null_choice(lp_indep)
    print("[2/7] per-feature spectra ..."); plot_per_feature_spectra(feats, ret)
    print("[3/7] market-mode removal ..."); plot_mode_removal(ret, sectors)
    print("[4/7] feature redundancy ..."); plot_redundancy(feats)
    print("[5/7] lower-edge structure ..."); plot_lower_edge(ret)
    print("[6/7] ZCA stress test ..."); plot_zca(feats)
    print("[7/7] predictive gate ..."); plot_predictive_svd(feats, lp_trim)

    with open(f"{OUT}/pipeline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time()-t0:.1f}s -> {OUT}")

if __name__ == "__main__":
    main()

