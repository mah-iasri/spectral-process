from .baseline import als, baseline_correct
from .derivatives import apply_savitzky_golay_derivative
from .normalization import snv_transformation, unit_vector_normalization
from .scatter import multiplicative_scatter_correction

__all__ = [
    "als",
    "baseline_correct",
    "apply_savitzky_golay_derivative",
    "snv_transformation",
    "unit_vector_normalization",
    "multiplicative_scatter_correction",
]

__version__ = "0.1.0"