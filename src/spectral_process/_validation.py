from __future__ import annotations

import numpy as np


def validate_spectra_2d(spectra: np.ndarray, name: str = "spectra") -> np.ndarray:
    spectra = np.asarray(spectra, dtype=float)

    if spectra.ndim != 2:
        raise ValueError(f"{name} must be a 2D numpy array")

    if spectra.shape[0] == 0 or spectra.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")

    return spectra


def validate_reference(reference: np.ndarray, n_features: int) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)

    if reference.ndim != 1:
        raise ValueError("reference must be a 1D numpy array")

    if reference.shape[0] != n_features:
        raise ValueError(
            f"reference length ({reference.shape[0]}) must match "
            f"number of features ({n_features})"
        )

    return reference