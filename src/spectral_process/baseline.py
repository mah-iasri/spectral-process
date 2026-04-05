from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ._validation import validate_spectra_2d


def als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10):
    """
    Apply Asymmetric Least Squares (ALS) baseline correction to one spectrum.

    Parameters
    ----------
    y : np.ndarray
        1D input spectrum.
    lam : float, default=1e5
        Smoothness parameter. Higher values produce a smoother baseline.
    p : float, default=0.01
        Asymmetry parameter. Must be between 0 and 1.
    niter : int, default=10
        Number of ALS iterations.

    Returns
    -------
    corrected : np.ndarray
        Baseline-corrected spectrum.
    baseline : np.ndarray
        Estimated baseline.
    """
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be a 1D numpy array")
    if y.size < 3:
        raise ValueError("y must contain at least 3 points")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if not (0 < p < 1):
        raise ValueError("p must be between 0 and 1")
    if niter < 1:
        raise ValueError("niter must be >= 1")

    n = y.size
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    h = lam * (d.T @ d)

    w = np.ones(n, dtype=float)
    z = np.zeros(n, dtype=float)

    for _ in range(niter):
        w_mat = sparse.diags(w, 0, shape=(n, n), format="csc")
        z = spsolve(w_mat + h, w * y)
        w = np.where(y > z, p, 1.0 - p)

    return y - z, z


def baseline_correct(
    spectra: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
    return_baselines: bool = False,
):
    """
    Apply ALS baseline correction to multiple spectra.

    Parameters
    ----------
    spectra : np.ndarray
        2D spectral matrix of shape (n_samples, n_features).
    lam : float, default=1e5
        Smoothness parameter.
    p : float, default=0.01
        Asymmetry parameter.
    niter : int, default=10
        Number of ALS iterations.
    return_baselines : bool, default=False
        If True, return both corrected spectra and estimated baselines.

    Returns
    -------
    corrected : np.ndarray
        Baseline-corrected spectra.
    baselines : np.ndarray, optional
        Estimated baselines, only if return_baselines=True.
    """
    spectra = validate_spectra_2d(spectra)

    corrected = np.empty_like(spectra, dtype=float)
    baselines = np.empty_like(spectra, dtype=float)

    for i, spectrum in enumerate(spectra):
        corrected[i], baselines[i] = als(spectrum, lam=lam, p=p, niter=niter)

    if return_baselines:
        return corrected, baselines
    return corrected