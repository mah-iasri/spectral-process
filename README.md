# spectral-process

> Python tools for spectral data preprocessing — baseline correction, 
> scatter correction, normalization, and derivative filtering.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/mah-iasri/spectral-process/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/spectral-process.svg)](https://pypi.org/project/spectral-process/)

---

## Overview

`spectral-process` is a python library for applying preprocessing techniques on the NIR, MIR, hyperspectral, 
and other spectral dataset. It provides a set of well-tested implementations of the 
most widely used data preprocessing techniques, which can be applied to chemometrics and agricultural 
spectroscopic datasets.

---

## Installation

**From GitHub (recommended for now):**

```bash
git clone https://github.com/mah-iasri/spectral-process.git
cd spectral-process
pip install -e .
```

**From PyPI (coming soon):**

```bash
pip install spectral-process
```

---

## Usage Examples

### Import the package

```python
from spectral_process import (
    baseline_correct,
    apply_savitzky_golay_derivative,
    snv_transformation,
    unit_vector_normalization,
    multiplicative_scatter_correction,
)
import numpy as np
```

---

### 1. ALS-based Baseline Correction

Removes fluorescence background or sloping baselines common in the spectra.

```python
# Simulate 10 spectra with 200 wavelength points
spectra = np.random.rand(10, 200) + np.linspace(0, 1, 200)

# Apply ALS baseline correction
corrected = baseline_correct(spectra, lam=1e5, p=0.01, niter=10)

# Also retrieve the estimated baselines
corrected, baselines = baseline_correct(
    spectra, lam=1e5, p=0.01, niter=10, return_baselines=True
)
```

**Parameters to tune:**
- `lam` — smoothness of baseline (typical range: `1e4` to `1e7`)
- `p` — asymmetry (typical range: `0.001` to `0.1`)

---

### 2. Savitzky-Golay Derivative Filtering

Smooths spectra and computes derivatives to remove baseline offsets and 
enhance spectral features.

```python
# First derivative with 21-point window, 2nd order polynomial
first_derivative = apply_savitzky_golay_derivative(
    spectra,
    window_length=21,
    polyorder=2,
    deriv=1
)

# Second derivative
second_derivative = apply_savitzky_golay_derivative(
    spectra,
    window_length=21,
    polyorder=2,
    deriv=2
)
```

---

### 3. Standard Normal Variate (SNV)

Corrects for multiplicative scatter effects and baseline shifts on a 
per-sample basis. Each spectrum will have mean = 0 and std = 1.

```python
snv_spectra = snv_transformation(spectra)

# Using sample standard deviation (ddof=1)
snv_spectra = snv_transformation(spectra, ddof=1)
```
---
### 4. Unit Vector Normalization

Scales each spectrum to unit L2 norm. Useful before applying machine learning models.

```python
normalized = unit_vector_normalization(spectra)

# Raise an error if any zero-norm spectra are found
normalized = unit_vector_normalization(spectra, handle_zero='error')
```
---

### 5. Multiplicative Scatter Correction (MSC)

Corrects for light scattering differences between samples. Uses the mean 
spectrum as reference by default.

```python
msc_spectra = multiplicative_scatter_correction(spectra)

# Use a custom reference spectrum (e.g. from a calibration set)
reference = np.mean(spectra[:50], axis=0)
msc_spectra, ref_used = multiplicative_scatter_correction(
    spectra,
    reference=reference,
    return_reference=True
)
```

---

### Full Preprocessing Pipeline Example

```python
import numpy as np
from spectral_prep import (
    baseline_correct,
    apply_savitzky_golay_derivative,
    snv_transformation,
)

# Load your spectral data (n_samples × n_wavelengths)
spectra = np.load("nir_spectra.npy")   # shape: (100, 1050)

# Step 1: Baseline correction
spectra_bc = baseline_correct(spectra, lam=1e5, p=0.01, niter=10)

# Step 2: Savitzky-Golay first derivative
spectra_sg = apply_savitzky_golay_derivative(
    spectra_bc, window_length=21, polyorder=2, deriv=1
)

# Step 3: SNV normalization
spectra_ready = snv_transformation(spectra_sg)

print(spectra_ready.shape)  # (100, 1050)
```

---

## API Reference

### Baseline Correction

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `als(y, lam, p, niter)` | ALS baseline correction for a single spectrum | `lam`: smoothness; `p`: asymmetry |
| `baseline_correct(spectra, lam, p, niter, return_baselines)` | ALS correction for multiple spectra | `return_baselines=True` to get baselines |

### Derivative Filtering

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `apply_savitzky_golay_derivative(spectra, window_length, polyorder, deriv, delta, mode)` | SG filter / derivative | `deriv=0` for smoothing only; `deriv=1` or `2` for derivatives |

### Normalization

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `snv_transformation(spectra, ddof)` | Standard Normal Variate | `ddof=0` (population) or `1` (sample) |
| `unit_vector_normalization(spectra, handle_zero)` | L2 unit vector normalization | `handle_zero`: `'warn'`, `'ignore'`, `'error'` |

### Scatter Correction

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `multiplicative_scatter_correction(spectra, reference, return_reference, eps)` | MSC scatter correction | `reference`: custom 1D reference; `return_reference=True` to retrieve it |

---

## Input Requirements

All functions expect:
- **Input:** 2D `numpy.ndarray` of shape `(n_samples, n_features)`
- **Output:** 2D `numpy.ndarray` of the same shape
- Invalid inputs raise a `ValueError` with a descriptive message

---

## Dependencies

| Package | Version |
|---------|---------|
| `numpy` | ≥ 1.23 |
| `scipy` | ≥ 1.10 |

---

## Authors

| Name              | Affiliation                  | Contact                    |
|-------------------|------------------------------|----------------------------|
| Md Ashraful Haque | ICAR-IASRI, New Delhi, India | ashrafulhaque664@gmail.com |
| Avijit Ghosh      | ICAR-IGFRI, Jhansi, India    | avijitghosh19892@gmail.com |



## Citation

If you use `spectral-process` in your research, please cite:

```bibtex
@software{haque2026spectralprocess,
  author    = {Haque, Md Ashraful and 
                Ghosh, Avijit},
  title     = {spectral-process: Python tools for spectral data preprocessing},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/mah-iasri/spectral-process}
}
```

Or in plain text:

> Haque, M.A. & Ghosh, A. (2026). *spectral-process: Python tools for spectral data 
> preprocessing*. GitHub. https://github.com/mah-iasri/spectral-process

---


## License

This project is licensed under the MIT License — see the 
[LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please open an issue first to discuss any 
proposed changes. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.