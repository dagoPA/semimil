"""semimil - MIL & (semi/self)-supervised anomaly detection for IC microscopy images.

The package targets the MIIC dataset (Microscopic Images of Integrated Circuits,
Huang et al., ICIP 2021): a strongly imbalanced set of ~25k normal and ~116
abnormal die images, well suited to weakly-supervised Multiple-Instance Learning
where each image is a *bag* of patches.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("semimil")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0+unknown"

from semimil.config import Settings, settings

__all__ = ["Settings", "__version__", "settings"]
