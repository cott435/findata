"""
Random-matrix-theory denoising of correlation matrices (Marchenko-Pastur).

Given N variables observed over T periods, the sample correlation matrix has
eigenvalues that, under the null of pure noise (IID columns), fall inside the
Marchenko-Pastur bulk [lambda_-, lambda_+]. Eigenvalues ABOVE lambda_+ cannot
be explained by finite-sample sampling error, so they are treated as signal.
Everything inside the bulk is collapsed to its mean (constant-residual method),
which denoises while preserving the trace (total variance).

Convention here: q = T / N  (observations per variable), q > 1 for the
non-singular case. This matches Lopez de Prado, "Machine Learning for Asset
Managers" (2020), ch. 2.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde


def mp_pdf(var: float, q: float, pts: int = 500):
    """Theoretical Marchenko-Pastur density for noise variance `var`.

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

    `bw` is an ABSOLUTE kernel bandwidth (in eigenvalue units), not a multiple of
    the sample std -- otherwise the few huge signal eigenvalues inflate the std,
    over-smooth the kernel, and the bulk fit collapses to a boundary.
    """
    std = float(np.std(eigenvalues))
    kde = gaussian_kde(eigenvalues, bw_method=(bw / std) if std > 0 else bw)

    def sse(var: float) -> float:
        e, pdf, *_ = mp_pdf(var, q)
        emp = kde(e)
        return float(np.sum((pdf - emp) ** 2))

    res = minimize_scalar(sse, bounds=(1e-4, 1.0 - 1e-4), method="bounded")
    return float(res.x)


def mp_edge(eigenvalues: np.ndarray, q: float, bw: float = 0.1):
    """Return (lambda_plus, n_signal, noise_var): the upper bulk edge, the count
    of eigenvalues above it, and the fitted noise variance."""
    var = fit_noise_variance(eigenvalues, q, bw)
    _, _, _, e_max = mp_pdf(var, q)
    n_signal = int((eigenvalues > e_max).sum())
    return e_max, n_signal, var


def denoise_correlation(corr: np.ndarray, q: float, bw: float = 0.1,
                        method: str = "residual", alpha: float = 0.0):
    """Denoise a correlation matrix.

    method="residual": collapse all sub-edge eigenvalues to their mean
        (trace-preserving; the classic Laloux/de Prado method).
    method="shrink": targeted shrinkage -- shrink only the noise subspace toward
        its diagonal by (1 - alpha), leaving the signal subspace untouched.
        alpha in [0, 1]; alpha=0 reproduces full residual denoising in spirit.

    Returns the denoised correlation matrix (unit diagonal).
    """
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


if __name__ == '__main__':
    from findata.preprocess import Volume
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:10]
    dates = DataSplits()
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    processor = Volatility(data, DataSplits(), verbose=True, scaler='robust', arcsinh=True)
    corr = processor.feat_eng_data.corr()
    q = len(processor.feat_eng_data) / len(processor.feat_eng_data.columns)

    out = denoise_correlation(corr, q, bw=0.1)
    d=1

