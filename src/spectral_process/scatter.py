from __future__ import annotations

import numpy as np

from ._validation import validate_reference, validate_spectra_2d


def multiplicative_scatter_correction(
    spectra: np.ndarray,
    reference: np.ndarray | None = None,
    return_reference: bool = False,
    eps: float = 1e-12,
):
    """
    Apply Multiplicative Scatter Correction (MSC) to spectral data.

    Parameters
    ----------
    spectra : np.ndarray
        2D spectral matrix of shape (n_samples, n_features).
    reference : np.ndarray, optional
        1D reference spectrum. If None, the mean spectrum is used.
    return_reference : bool, default=False
        If True, also return the reference used for MSC.
    eps : float, default=1e-12
        Small threshold to detect near-zero slopes.

    Returns
    -------
    corrected : np.ndarray
        MSC-corrected spectra.
    reference : np.ndarray, optional
        Reference spectrum used, only if return_reference=True.
    """
    spectra = validate_spectra_2d(spectra)

    if reference is None:
        reference = np.mean(spectra, axis=0)
    else:
        reference = validate_reference(reference, spectra.shape[1])

    corrected = np.empty_like(spectra, dtype=float)
    a = np.column_stack([reference, np.ones(reference.shape[0])])

    for i, spectrum in enumerate(spectra):
        slope, intercept = np.linalg.lstsq(a, spectrum, rcond=None)[0]

        if abs(slope) < eps:
            raise ValueError(
                f"MSC failed for spectrum index {i}: estimated slope is too close to zero."
            )

        corrected[i] = (spectrum - intercept) / slope

    if return_reference:
        return corrected, reference
    return corrected