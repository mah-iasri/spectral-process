# spectral-process

Python package for common spectral preprocessing operations.

## Features

- ALS baseline correction
- Savitzky-Golay derivative filtering
- Standard Normal Variate (SNV)
- Unit vector normalization
- Multiplicative Scatter Correction (MSC)

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Package import

```python
from spectral_process import (
    als,
    baseline_correct,
    apply_savitzky_golay_derivative,
    snv_transformation,
    unit_vector_normalization,
    multiplicative_scatter_correction,
)
```

## Example

```python
import numpy as np
from spectral_process import (
    baseline_correct,
    apply_savitzky_golay_derivative,
    snv_transformation,
)

spectra = np.random.rand(5, 200)

corrected = baseline_correct(spectra, lam=1e5, p=0.01, niter=10)
first_derivative = apply_savitzky_golay_derivative(
    corrected,
    window_length=21,
    polyorder=2,
    deriv=1
)
snv = snv_transformation(first_derivative)
```

## Notes

- Input spectra must be a 2D NumPy array of shape `(n_samples, n_features)`.
- Functions raise exceptions for invalid input instead of printing errors.
- Constant spectra are handled with warnings where appropriate.

## License

MIT