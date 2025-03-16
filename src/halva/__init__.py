"""
Halva
=====

Multivariate analysis of ordinal data with missing values and latent variables.
"""
from importlib.metadata import version as _version
from halva.inference import infer_precision

__all__ = [
    "infer_precision",
]

try:
    __version__ = _version("halva")
except Exception:
    __version__ = "unknown version"
