import numpy as np

from spectral_process import (
    als,
    baseline_correct,
    apply_savitzky_golay_derivative,
    snv_transformation,
    unit_vector_normalization,
    multiplicative_scatter_correction,
)


def test_als_shapes():
    x = np.linspace(0, 1, 200)
    y = 2.0 + 0.5 * x + np.sin(10 * x)
    corrected, baseline = als(y)

    assert corrected.shape == y.shape
    assert baseline.shape == y.shape


def test_baseline_correct_shape():
    spectra = np.random.rand(4, 200)
    corrected = baseline_correct(spectra)
    assert corrected.shape == spectra.shape


def test_savgol_derivative_shape():
    spectra = np.random.rand(3, 101)
    out = apply_savitzky_golay_derivative(
        spectra,
        window_length=11,
        polyorder=2,
        deriv=1,
    )
    assert out.shape == spectra.shape


def test_snv_row_mean_and_std():
    spectra = np.random.rand(5, 50)
    out = snv_transformation(spectra)

    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-10)
    assert np.allclose(out.std(axis=1), 1.0, atol=1e-10)


def test_unit_vector_normalization_row_norm():
    spectra = np.random.rand(5, 50)
    out = unit_vector_normalization(spectra)

    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_msc_shape():
    spectra = np.random.rand(6, 120)
    out = multiplicative_scatter_correction(spectra)
    assert out.shape == spectra.shape