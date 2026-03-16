# ssl_dataset/__init__.py
#
# Top-level package init.
# Exposes the version and all three sub-libraries from one place.

__version__ = "1.0.0"
__author__  = "Jayasha"

from .landmarks    import SSLLandmarkDataset
from .skeleton     import SSLSkeletonDataset
from .preprocessed import SSLPreprocessedDataset
from ._constants   import CLASS_LABELS, NUM_CLASSES

__all__ = [
    "SSLLandmarkDataset",
    "SSLSkeletonDataset",
    "SSLPreprocessedDataset",
    "CLASS_LABELS",
    "NUM_CLASSES",
]
