# ssl_dataset/landmarks/__init__.py
#
# This file makes `ssl_dataset/landmarks/` a Python package and controls
# what is visible when a user writes:
#   from ssl_dataset.landmarks import SSLLandmarkDataset

from .dataset import SSLLandmarkDataset

__all__ = ["SSLLandmarkDataset"]
