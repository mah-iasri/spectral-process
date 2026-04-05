from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ._validation import validate_spectra_2d


def apply_savitzky_golay_derivative(
    spectra: np.ndarray,
    window_length: int = 21,
    polyorder: int = 2,
    deriv: int = 1,
    delta: float = 1.0,
    mode: str = "interp",
):
    """
    Apply Savitzky-Golay filtering/derivative to each spectrum.

    Parameters
    ----------
    spectra : np.ndarray
        2D spectral matrix of shape (n_samples, n_features).
    window_length : int, default=21
        Length of the filter window. Must be odd.
    polyorder : int, default=2
        Polynomial order. Must be less than window_length.
    deriv : int, default=1
        Derivative order.
    delta : float, default=1.0
        Sample spacing for derivative calculation.
    mode : str, default="interp"
        Padding mode passed to scipy.signal.savgol_filter.

    Returns
    -------
    np.ndarray
        Filtered/derivative spectra with the same shape as input.
    """
    spectra = validate_spectra_2d(spectra)

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder")
    if window_length > spectra.shape[1]:
        raise ValueError(
            f"window_length ({window_length}) must be <= spectrum length ({spectra.shape[1]})"
        )
    if deriv < 0:
        raise ValueError("deriv must be >= 0")
    if deriv > polyorder:
        raise ValueError("deriv must be <= polyorder")
    if delta <= 0:
        raise ValueError("delta must be > 0")

    return savgol_filter(
        spectra,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        axis=1,
        mode=mode,
    )