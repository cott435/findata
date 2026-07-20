"""Random-matrix-theory tools: Marchenko-Pastur denoising + correlation clustering.

The single home for MP math in the repo (consolidates the three divergent
copies that lived in scripts/testing/{denoise,denoise_test,ful_feature_analysis}.py).

Convention: q = T / N (observations per variable), q > 1 for the non-singular
case, per Lopez de Prado, "Machine Learning for Asset Managers" (2020), ch. 2.
For pooled (ticker, date) feature panels use T = number of unique dates — a
single day's cross-section of correlated names is far from independent, so
counting pooled rows wildly overstates T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Marchenko-Pastur (lifted from scripts/testing/denoise_test.py)
# ---------------------------------------------------------------------------

def mp_pdf(var: float, q: float, pts: int = 500):
    """Theoretical MP density for noise variance `var`.

    Returns (eigenvalue_grid, density, lambda_minus, lambda_plus).
    q = T / N. The bulk edges are var * (1 +/- sqrt(N/T))**2.
    """
    a = (1.0 / q) ** 0.5                       # sqrt(N/T)
    e_min = var * (1.0 - a) ** 2
    e_max = var * (1.0 + a) ** 2
    e = np.linspace(e_min, e_max, pts)
    pdf = q / (2.0 * np.pi * var * e) * np.sqrt(np.maximum((e_max - e) * (e - e_min), 0.0))
    return e, pdf, e_min, e_max


def fit_noise_variance(eigenvalues: np.ndarray, q: float, bw: float = 0.05) -> float:
    """Estimate the noise variance by matching the MP density to the empirical
    eigenvalue density (KDE) over the bulk. Signal eigenvalues sit outside the
    fitted support and so do not drag the estimate.

    `bw` is an ABSOLUTE kernel bandwidth (in eigenvalue units), not a multiple
    of the sample std — otherwise the few huge signal eigenvalues inflate the
    std, over-smooth the kernel, and the bulk fit collapses to a boundary.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    std = float(np.std(eigenvalues))
    if std < 1e-9:
        # degenerate spectrum (e.g. an already-whitened panel): no bulk to fit
        return float(np.clip(eigenvalues.mean(), 1e-4, 1.0 - 1e-4))
    kde = gaussian_kde(eigenvalues, bw_method=bw / std)

    def sse(var: float) -> float:
        e, pdf, *_ = mp_pdf(var, q)
        emp = kde(e)
        return float(np.sum((pdf - emp) ** 2))

    res = minimize_scalar(sse, bounds=(1e-4, 1.0 - 1e-4), method="bounded")
    return float(res.x)


def iterative_bulk_variance(eigenvalues: np.ndarray, q: float, max_iter: int = 100):
    """Noise variance by iterative bulk-mean recursion (the preferred method).

    Start with every eigenvalue in the bulk. sigma2 = mean(bulk) — for a
    correlation matrix this is trace-preserving: (N - sum(signal)) / (N -
    n_signal), i.e. the noise budget left after the signal eigenvalues absorb
    their share. The MP edge from that sigma2 reclassifies eigenvalues; repeat
    until no more eigenvalues shift to signal. Convergence is guaranteed:
    removing large eigenvalues only lowers the bulk mean, so the edge only
    moves down and n_signal only grows.

    Unlike the KDE density fit (``fit_noise_variance``) this has no bandwidth
    knob, no optimizer bounds to saturate, and behaves sensibly on degenerate
    spectra (already-whitened panels -> sigma2 = 1, n_signal = 0).

    Returns (sigma2, lambda_plus, n_signal).
    """
    w = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    edge_factor = (1.0 + (1.0 / q) ** 0.5) ** 2
    n_signal = 0
    sigma2 = float(w.mean())
    lam_plus = sigma2 * edge_factor
    for _ in range(max_iter):
        new_n = min(int((w > lam_plus).sum()), len(w) - 1)  # keep a nonempty bulk
        if new_n == n_signal:
            break
        n_signal = new_n
        sigma2 = float(w[n_signal:].mean())
        lam_plus = sigma2 * edge_factor
    return sigma2, lam_plus, n_signal


def mp_edge(eigenvalues: np.ndarray, q: float, bw: float = 0.1, method: str = "iterative"):
    """Return (lambda_plus, n_signal, noise_var): the upper bulk edge, the count
    of eigenvalues above it, and the fitted noise variance.

    method="iterative" (default): recursive bulk-mean, see iterative_bulk_variance.
    method="kde": legacy density-matching fit (``bw`` only applies here).
    """
    if method == "iterative":
        var, e_max, n_signal = iterative_bulk_variance(eigenvalues, q)
        return e_max, n_signal, var
    var = fit_noise_variance(eigenvalues, q, bw)
    _, _, _, e_max = mp_pdf(var, q)
    n_signal = int((np.asarray(eigenvalues) > e_max).sum())
    return e_max, n_signal, var


def denoise_correlation(corr: np.ndarray, q: float, bw: float = 0.1,
                        method: str = "residual", alpha: float = 0.0) -> np.ndarray:
    """Denoise a correlation matrix.

    method="residual": collapse all sub-edge eigenvalues to their mean
        (trace-preserving; the classic Laloux/de Prado method).
    method="shrink": shrink only the noise subspace toward its diagonal by
        (1 - alpha), leaving the signal subspace untouched. alpha in [0, 1].

    Returns the denoised correlation matrix (unit diagonal).
    """
    corr = np.asarray(corr, dtype=float)
    w, v = np.linalg.eigh(corr)
    idx = w.argsort()[::-1]
    w, v = w[idx], v[:, idx]

    e_max, n_signal, _ = mp_edge(w, q, bw)
    n_signal = max(n_signal, 1)  # always keep at least the top factor

    if method == "residual":
        w_d = w.copy()
        if n_signal < len(w):
            w_d[n_signal:] = w_d[n_signal:].mean()
        c = v @ np.diag(w_d) @ v.T
    elif method == "shrink":
        v_sig, w_sig = v[:, :n_signal], w[:n_signal]
        v_noi, w_noi = v[:, n_signal:], w[n_signal:]
        c_sig = v_sig @ np.diag(w_sig) @ v_sig.T
        c_noi = v_noi @ np.diag(w_noi) @ v_noi.T
        c = c_sig + alpha * c_noi + (1 - alpha) * np.diag(np.diag(c_noi))
    else:
        raise ValueError("method must be 'residual' or 'shrink'")

    d = np.sqrt(np.clip(np.diag(c), 1e-12, None))
    return c / np.outer(d, d)


# ---------------------------------------------------------------------------
# Correlation clustering (shared by HierarchicalPCA and the structure analysis)
# ---------------------------------------------------------------------------

def corr_distance(corr: np.ndarray) -> np.ndarray:
    """Distance d_ij = sqrt(2 (1 - corr_ij)), zero diagonal."""
    dist = np.sqrt(np.clip(2.0 * (1.0 - np.asarray(corr, dtype=float)), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return dist


def choose_k_silhouette(dist: np.ndarray, Z: np.ndarray, k_range=(2, 12)) -> int:
    """Pick the cluster count maximizing mean silhouette on the precomputed
    distance — threshold-free and cheap for N up to a few hundred features."""
    from sklearn.metrics import silhouette_score

    n = dist.shape[0]
    lo = max(2, int(k_range[0]))
    hi = min(int(k_range[1]), n - 1)
    best_k, best_score = 2, -np.inf
    for k in range(lo, hi + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(dist, labels, metric="precomputed")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_corr(corr: np.ndarray, k: int | str = "auto", method: str = "average",
                 k_range=(2, 12), threshold: float | None = None):
    """Hierarchically cluster features on correlation distance.

    Returns (labels, linkage_matrix, k_used); labels are 1-based like fcluster.
    k="auto" picks k by silhouette; an int k or a distance `threshold` override.
    """
    corr = np.asarray(corr, dtype=float)
    dist = corr_distance(corr)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    if threshold is not None:
        labels = fcluster(Z, t=threshold, criterion="distance")
        k_used = int(len(np.unique(labels)))
    else:
        k_used = choose_k_silhouette(dist, Z, k_range) if k == "auto" else int(k)
        labels = fcluster(Z, t=k_used, criterion="maxclust")
    return labels, Z, k_used


# ---------------------------------------------------------------------------
# Panel spectrum probe (used by the search to report each variant's structure)
# ---------------------------------------------------------------------------

def spectrum_probe(panel: pd.DataFrame, q: float | None = None, bw: float = 0.1) -> dict:
    """Standardize a (ticker, date) feature panel, eigendecompose its correlation,
    and report the MP diagnostics: how concentrated is the signal?

    Returns dict with n_features, q, lam_plus, noise_var, n_signal,
    top_eig_share (largest eigenvalue / trace) and the sorted eigenvalues.
    """
    X = panel.dropna()
    values = X.values.astype(float)
    std = values.std(axis=0, ddof=1)
    keep = std > 1e-12
    values = values[:, keep]
    n = values.shape[1]
    if q is None:
        q = X.index.get_level_values("date").nunique() / n
    corr = np.corrcoef(values, rowvar=False)
    w = np.linalg.eigvalsh(corr)[::-1]
    lam_plus, n_signal, noise_var = mp_edge(w, q, bw)
    return {
        "n_features": n,
        "q": float(q),
        "lam_plus": float(lam_plus),
        "noise_var": float(noise_var),
        "n_signal": int(n_signal),
        "top_eig_share": float(w[0] / w.sum()),
        "eigvals": w,
    }
