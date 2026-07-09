"""
feature_denoising_diagnostics.py

Diagnostic suite for feature-processing decisions BEFORE PCA / final denoising choice.
Two independent axes are tested here, and the script keeps them separate on purpose:

  AXIS 1 - CROSS-SECTIONAL (across your ~300-500 tickers, at fixed time):
      raw correlation  ->  Marchenko-Pastur eigenvalue diagnostic
                        ->  hard-threshold MP denoising
                        ->  Ledoit-Wolf shrinkage (linear + nonlinear-ish comparison)

  AXIS 2 - TEMPORAL (within a single feature's time series):
      Kalman filter (causal, local-level model, MLE-fit via statsmodels)
      Wavelet denoising (batch/non-causal - demo only, see warning below)
      SSA / trajectory-matrix SVD (batch/non-causal - demo only, see warning below)

WARNING ON CAUSALITY:
  Wavelet and SSA denoising as implemented here use the ENTIRE series (past and future)
  to compute each point's denoised value. That is look-ahead leakage if you feed the
  output into training data. Kalman filtering here is properly causal (each point only
  uses data up to and including that point). Treat the wavelet/SSA outputs as a
  visualization/diagnostic of "how noisy is this feature, and does smoothing reveal
  structure" -- not as a drop-in causal feature transform. If SSA/wavelets show real
  promise, implement a rolling/streaming causal variant before using them for training data.

HOW TO USE WITH YOUR REAL DATA:
  Replace `build_synthetic_panel()` in `main()` with your actual feature-engineered df.
  Expected shape: long/tidy panel with columns ['date', 'ticker', feature_1, feature_2, ...]
  Set FEATURE_COLS to your actual feature column names, and H to your forecast horizon
  (used to compute T_eff = T / H, matching the RMT convention you're already using
  for label overlap).

Dependencies beyond your existing stack: pip install pywavelets statsmodels
(both optional -- if pywt or statsmodels aren't installed, those sections are skipped
with a printed note, everything else still runs).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


import pywt
from statsmodels.tsa.statespace.structural import UnobservedComponents

from sklearn.covariance import LedoitWolf

OUTPUT_DIR = Path("./denoising_diagnostics_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# AXIS 1: CROSS-SECTIONAL (feature x feature correlation structure)
# =============================================================================

def marchenko_pastur_lambda_plus(q, sigma2=1.0):
    """Upper edge of the MP distribution support. q = T_eff / N."""
    return sigma2 * (1 + np.sqrt(1.0 / q)) ** 2


def marchenko_pastur_lambda_minus(q, sigma2=1.0):
    """Lower edge of the MP distribution support."""
    return sigma2 * (1 - np.sqrt(1.0 / q)) ** 2


def marchenko_pastur_density(x, q, sigma2=1.0):
    """MP theoretical density, for overlay on the empirical eigenvalue histogram."""
    lam_plus = marchenko_pastur_lambda_plus(q, sigma2)
    lam_minus = max(marchenko_pastur_lambda_minus(q, sigma2), 1e-10)
    x = np.asarray(x, dtype=float)
    density = np.zeros_like(x)
    mask = (x >= lam_minus) & (x <= lam_plus)
    density[mask] = np.sqrt(np.maximum((lam_plus - x[mask]) * (x[mask] - lam_minus), 0)) / (
        2 * np.pi * sigma2 * q * x[mask]
    )
    return density


def mp_denoise_correlation(corr, q, sigma2=1.0):
    """
    Hard-threshold RMT denoising: eigenvalues below lambda_plus are averaged
    and replaced by that average (trace-preserving); eigenvalues above lambda_plus
    are kept as-is. Correlation matrix is reconstructed from the modified spectrum.
    """
    eigvals, eigvecs = np.linalg.eigh(corr)
    lam_plus = marchenko_pastur_lambda_plus(q, sigma2)

    signal_mask = eigvals > lam_plus
    n_signal = signal_mask.sum()

    denoised_eigvals = eigvals.copy()
    if (~signal_mask).sum() > 0:
        noise_avg = eigvals[~signal_mask].mean()
        denoised_eigvals[~signal_mask] = noise_avg

    denoised_corr = eigvecs @ np.diag(denoised_eigvals) @ eigvecs.T
    # re-normalize to a proper correlation matrix (unit diagonal)
    d = np.sqrt(np.diag(denoised_corr))
    denoised_corr = denoised_corr / np.outer(d, d)
    np.fill_diagonal(denoised_corr, 1.0)

    return denoised_corr, eigvals, denoised_eigvals, n_signal, lam_plus


def run_cross_sectional_diagnostics(df, feature_cols, date_col, horizon_h):
    """
    Pools all (date, ticker) rows, z-scores each feature, computes the pooled
    feature-feature correlation matrix, and runs the MP diagnostic against it.
    """
    print("\n" + "=" * 70)
    print("AXIS 1: CROSS-SECTIONAL FEATURE CORRELATION DIAGNOSTICS")
    print("=" * 70)

    X = df[feature_cols].dropna()
    X_z = (X - X.mean()) / X.std()
    corr = X_z.corr().values
    N = len(feature_cols)

    T_raw = df[date_col].nunique()
    T_eff = T_raw / horizon_h
    q = len(df) / N

    print(f"N features = {N}")
    print(f"T_raw (unique dates) = {T_raw}, H = {horizon_h}, T_eff = T_raw/H = {T_eff:.1f}")
    print(f"q = T_eff / N = {q:.3f}  (q < 1 means MORE features than effective independent samples --"
          f" if this is your regime, denoising is not optional)")

    denoised_corr, raw_eigvals, denoised_eigvals, n_signal, lam_plus = mp_denoise_correlation(corr, q)

    print(f"MP theoretical noise ceiling (lambda+) = {lam_plus:.3f}")
    print(f"Eigenvalues above lambda+ (statistically real signal components): {n_signal} / {N}")
    print(f"--> This is your PCA component count under the 'signal vs noise' rule,"
          f" NOT a 95%-variance cutoff.")

    # --- Ledoit-Wolf shrinkage for comparison ---
    lw = LedoitWolf().fit(X_z.values)
    lw_cov = lw.covariance_
    d = np.sqrt(np.diag(lw_cov))
    lw_corr = lw_cov / np.outer(d, d)
    lw_eigvals = np.linalg.eigvalsh(lw_corr)
    print(f"Ledoit-Wolf shrinkage intensity = {lw.shrinkage_:.3f} "
          f"(0 = no shrinkage/trust raw corr, 1 = shrink fully to target)")

    _plot_mp_spectrum(raw_eigvals, q, lam_plus, n_signal)
    _plot_eigenvalue_comparison(raw_eigvals, denoised_eigvals, lw_eigvals, lam_plus)
    _plot_corr_heatmaps(corr, denoised_corr, feature_cols)

    return {
        "raw_corr": corr,
        "denoised_corr": denoised_corr,
        "n_signal_components": n_signal,
        "lambda_plus": lam_plus,
        "q": q,
        "lw_shrinkage_intensity": lw.shrinkage_,
    }


def _plot_mp_spectrum(raw_eigvals, q, lam_plus, n_signal):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(raw_eigvals, bins=40, density=True, alpha=0.6, color="steelblue",
            label="empirical eigenvalues (your data)")
    x_grid = np.linspace(0.001, max(raw_eigvals.max(), lam_plus) * 1.1, 500)
    mp_density = marchenko_pastur_density(x_grid, q)
    ax.plot(x_grid, mp_density, color="black", lw=2, label="MP theoretical density (pure noise)")
    ax.axvline(lam_plus, color="crimson", ls="--", lw=2, label=f"lambda+ = {lam_plus:.2f} (noise ceiling)")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_title(f"Empirical spectrum vs. Marchenko-Pastur null\n"
                 f"{n_signal} eigenvalue(s) exceed the noise ceiling = real signal components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_mp_eigenvalue_spectrum.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '1_mp_eigenvalue_spectrum.png'}")


def _plot_eigenvalue_comparison(raw_eigvals, mp_denoised_eigvals, lw_eigvals, lam_plus):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(raw_eigvals)[::-1]
    idx = np.arange(1, len(raw_eigvals) + 1)
    ax.plot(idx, raw_eigvals[order], "o-", label="raw", color="gray", alpha=0.8)
    ax.plot(idx, mp_denoised_eigvals[order], "s-", label="MP hard-threshold denoised", color="crimson")
    ax.plot(idx, np.sort(lw_eigvals)[::-1], "^-", label="Ledoit-Wolf shrunk", color="seagreen")
    ax.axhline(lam_plus, color="black", ls=":", lw=1.5, label="lambda+ (noise ceiling)")
    ax.set_yscale("log")
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Scree plot: raw vs. MP-denoised vs. Ledoit-Wolf shrunk spectra")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_eigenvalue_shrinkage_comparison.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '2_eigenvalue_shrinkage_comparison.png'}")


def _plot_corr_heatmaps(raw_corr, denoised_corr, feature_cols):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, title in zip(axes, [raw_corr, denoised_corr], ["Raw correlation", "MP-denoised correlation"]):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title)
        if len(feature_cols) <= 25:
            ax.set_xticks(range(len(feature_cols)))
            ax.set_yticks(range(len(feature_cols)))
            ax.set_xticklabels(feature_cols, rotation=90, fontsize=7)
            ax.set_yticklabels(feature_cols, fontsize=7)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")
    fig.suptitle("Denoising should visibly flatten spurious off-block noise while\n"
                 "preserving genuine block structure (e.g. vol-family, trend-family clusters)")
    fig.savefig(OUTPUT_DIR / "3_correlation_heatmaps.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '3_correlation_heatmaps.png'}")


# =============================================================================
# AXIS 2: TEMPORAL (single feature series denoising)
# =============================================================================

def kalman_local_level_filter(y):
    """
    Causal local-level Kalman filter. Fits observation/state noise variances by
    MLE via statsmodels' UnobservedComponents, then returns the FILTERED
    (not smoothed) state estimate -- filtered means each point only uses data
    up to and including that point, which is what makes this safe for training data.
    """
    model = UnobservedComponents(y, level="local level")
    fit = model.fit(disp=False)
    filtered_state = fit.filtered_state[0]
    return filtered_state


def wavelet_denoise(y, wavelet="db4", level=None):
    """
    Batch wavelet denoising via soft-thresholding of detail coefficients.
    NOT causal -- see module docstring warning. Demo/diagnostic only.
    """

    if level is None:
        level = min(pywt.dwt_max_level(len(y), wavelet), 4)
    coeffs = pywt.wavedec(y, wavelet, level=level)
    # universal threshold, sigma estimated via MAD of finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(y)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, uthresh, mode="soft") for c in coeffs[1:]]
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[: len(y)]


def ssa_denoise(y, window_length, n_components):
    """
    Singular Spectrum Analysis via trajectory-matrix SVD + diagonal averaging.
    NOT causal -- see module docstring warning. Demo/diagnostic only.
    Returns (denoised_series, singular_values) so you can scree-plot the spectrum.
    """
    n = len(y)
    k = n - window_length + 1
    traj = np.column_stack([y[i:i + window_length] for i in range(k)])
    U, S, Vt = np.linalg.svd(traj, full_matrices=False)

    def diagonal_average(mat):
        L, K = mat.shape
        total_len = L + K - 1
        out = np.zeros(total_len)
        counts = np.zeros(total_len)
        for i in range(L):
            for j in range(K):
                out[i + j] += mat[i, j]
                counts[i + j] += 1
        return out / counts

    recon = np.zeros(n)
    for r in range(n_components):
        comp = S[r] * np.outer(U[:, r], Vt[r, :])
        recon += diagonal_average(comp)
    return recon, S


def run_temporal_diagnostics(series, series_name="feature", ssa_window=40, ssa_n_components=3):
    print("\n" + "=" * 70)
    print(f"AXIS 2: TEMPORAL DENOISING DIAGNOSTICS -- '{series_name}'")
    print("=" * 70)

    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]

    results = {"original": y}

    kalman_out = kalman_local_level_filter(y)
    results["kalman"] = kalman_out
    print("Kalman filter: fit via MLE (statsmodels local-level model), causal.")

    wavelet_out = wavelet_denoise(y)
    results["wavelet"] = wavelet_out
    print("Wavelet denoising: db4, soft-threshold, batch (non-causal -- diagnostic only).")

    ssa_out, ssa_singvals = ssa_denoise(y, ssa_window, ssa_n_components)
    results["ssa"] = ssa_out
    print(f"SSA: window={ssa_window}, kept top {ssa_n_components} components (batch, non-causal -- diagnostic only).")

    _plot_temporal_comparison(y, kalman_out, wavelet_out, ssa_out, series_name)
    _plot_ssa_scree(ssa_singvals, ssa_n_components, series_name)

    return results


def _plot_temporal_comparison(y, kalman_out, wavelet_out, ssa_out, series_name):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    t = np.arange(len(y))

    ax = axes[0]
    ax.plot(t, y, color="lightgray", lw=1, label="raw (noisy) series")
    if kalman_out is not None:
        ax.plot(t, kalman_out, color="crimson", lw=1.5, label="Kalman filtered (causal)")
    if wavelet_out is not None:
        ax.plot(t, wavelet_out, color="seagreen", lw=1.5, label="wavelet denoised (batch)")
    ax.plot(t, ssa_out, color="darkorange", lw=1.5, label=f"SSA denoised (batch)")
    ax.set_title(f"'{series_name}': raw vs. denoised (full series)")
    ax.legend(loc="upper right", fontsize=8)

    # zoomed-in window to actually see the smoothing effect
    zoom_start = len(y) // 3
    zoom_end = zoom_start + min(150, len(y) // 4)
    ax2 = axes[1]
    ax2.plot(t[zoom_start:zoom_end], y[zoom_start:zoom_end], color="lightgray", lw=1.2, marker=".",
             label="raw")
    if kalman_out is not None:
        ax2.plot(t[zoom_start:zoom_end], kalman_out[zoom_start:zoom_end], color="crimson", lw=1.8, label="Kalman")
    if wavelet_out is not None:
        ax2.plot(t[zoom_start:zoom_end], wavelet_out[zoom_start:zoom_end], color="seagreen", lw=1.8, label="wavelet")
    ax2.plot(t[zoom_start:zoom_end], ssa_out[zoom_start:zoom_end], color="darkorange", lw=1.8, label="SSA")
    ax2.set_title("Zoomed window -- this is where you judge if smoothing preserves real turns\n"
                  "or just lags behind them")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlabel("time index")

    fig.tight_layout()
    fname = OUTPUT_DIR / f"4_temporal_denoising_{series_name}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"Saved: {fname}")


def _plot_ssa_scree(singvals, n_components, series_name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    frac_var = (singvals ** 2) / np.sum(singvals ** 2)
    ax.bar(range(1, min(len(singvals), 20) + 1), frac_var[:20], color="steelblue")
    ax.axvline(n_components + 0.5, color="crimson", ls="--", label=f"kept top {n_components}")
    ax.set_xlabel("SSA component rank")
    ax.set_ylabel("Fraction of variance")
    ax.set_title(f"SSA singular value spectrum -- '{series_name}'\n"
                 f"Look for the elbow: components after it are typically noise")
    ax.legend()
    fig.tight_layout()
    fname = OUTPUT_DIR / f"5_ssa_scree_{series_name}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"Saved: {fname}")


# =============================================================================
# SYNTHETIC DATA (for demo -- replace with your real df in main())
# =============================================================================

def build_synthetic_panel(n_dates=750, n_tickers=200, n_true_factors=3, n_features=30, seed=7):
    """
    Builds a panel that mimics the STRUCTURE your MP diagnostic should react to:
    a handful of features driven by shared latent factors + noise, plus pure-noise
    features, plus a single noisy time series for the temporal-axis demo.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    tickers = [f"T{i:04d}" for i in range(n_tickers)]

    factor_loadings = rng.normal(0, 1, size=(n_features, n_true_factors))
    rows = []
    for date_idx, date in enumerate(dates):
        factors = rng.normal(0, 1, size=n_true_factors)
        for ticker in tickers:
            base = factor_loadings @ factors
            noise = rng.normal(0, 1.5, size=n_features)  # dominant noise, deliberately
            feats = base + noise
            rows.append([date, ticker] + list(feats))

    cols = ["date", "ticker"] + [f"feat_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(rows, columns=cols)

    # a single noisy temporal series for the Axis-2 demo: true slow random walk + noise + a few spikes
    n_t = 500
    true_state = np.cumsum(rng.normal(0, 0.05, n_t))
    obs_noise = rng.normal(0, 0.5, n_t)
    spikes = np.zeros(n_t)
    spike_idx = rng.choice(n_t, size=8, replace=False)
    spikes[spike_idx] = rng.normal(0, 3, size=8)
    noisy_series = true_state + obs_noise + spikes

    return df, cols[2:], noisy_series


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # REPLACE THIS BLOCK with your real feature-engineered df:
    #
    #   df = pd.read_parquet("your_features.parquet")
    #   FEATURE_COLS = [c for c in df.columns if c not in ("date", "ticker")]
    #   H = 10  # your forecast horizon
    #   temporal_series = df[df.ticker == "AAPL"]["some_feature"].values
    #
    from findata.preprocess import Volume
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:10]
    dates = DataSplits()
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    processor = Volume(data, DataSplits(), verbose=True, scaler='robust', arcsinh=True)
    fe = processor.feat_eng_data
    FEATURE_COLS = [f for f in fe.columns if 'zvel' in f]
    df = fe[FEATURE_COLS].reset_index(drop=False)
    temporal_series = fe.loc[tickers[0]]['obv_zvel12_26'].values
    H = 10

    cross_sectional_results = run_cross_sectional_diagnostics(
        df, FEATURE_COLS, date_col="date", horizon_h=H
    )

    temporal_results = run_temporal_diagnostics(
        temporal_series, series_name="demo_feature", ssa_window=40, ssa_n_components=3
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Significant (signal) eigenvalues: {cross_sectional_results['n_signal_components']} "
          f"out of {len(FEATURE_COLS)} features (q = {cross_sectional_results['q']:.3f})")
    print(f"Ledoit-Wolf shrinkage intensity: {cross_sectional_results['lw_shrinkage_intensity']:.3f}")
    print(f"All plots saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()