from __future__ import annotations

import warnings
import numpy as np

from ._validation import validate_spectra_2d


def snv_transformation(spectra: np.ndarray, ddof: int = 0):
    """
    Apply Standard Normal Variate (SNV) transformation to each spectrum.

    Parameters
    ----------
    spectra : np.ndarray
        2D spectral matrix of shape (n_samples, n_features).
    ddof : int, default=0
        Delta degrees of freedom for standard deviation.

    Returns
    -------
    np.ndarray
        SNV-transformed spectra.
    """
    spectra = validate_spectra_2d(spectra)

    if ddof < 0:
        raise ValueError("ddof must be >= 0")
    if spectra.shape[1] <= ddof:
        raise ValueError("Each spectrum must have more points than ddof")

    means = np.mean(spectra, axis=1, keepdims=True)
    stds = np.std(spectra, axis=1, keepdims=True, ddof=ddof)

    zero_std_mask = np.isclose(stds, 0.0)
    if np.any(zero_std_mask):
        warnings.warn(
            "Constant spectra detected during SNV; corresponding rows will become zeros.",
            RuntimeWarning,
            stacklevel=2,
        )
        stds = stds.copy()
        stds[zero_std_mask] = 1.0

    return (spectra - means) / stds


def unit_vector_normalization(spectra: np.ndarray, handle_zero: str = "warn"):
    """
    Apply row-wise unit vector (L2) normalization.

    Parameters
    ----------
    spectra : np.ndarray
        2D spectral matrix of shape (n_samples, n_features).
    handle_zero : {"warn", "ignore", "error"}, default="warn"
        How to handle zero-norm spectra.

    Returns
    -------
    np.ndarray
        L2-normalized spectra.
    """
    spectra = validate_spectra_2d(spectra)

    if handle_zero not in {"warn", "ignore", "error"}:
        raise ValueError("handle_zero must be one of: 'warn', 'ignore', 'error'")

    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    zero_norm_mask = np.isclose(norms, 0.0)

    if np.any(zero_norm_mask):
        count = int(np.sum(zero_norm_mask))
        if handle_zero == "error":
            raise ValueError(f"Found {count} zero-norm spectra")
        if handle_zero == "warn":
            warnings.warn(
                f"Found {count} zero-norm spectra; those rows will remain zeros.",
                RuntimeWarning,
                stacklevel=2,
            )

        norms = norms.copy()
        norms[zero_norm_mask] = 1.0

    return spectra / norms