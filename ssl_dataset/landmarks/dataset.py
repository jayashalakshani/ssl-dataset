# ssl_dataset/landmarks/dataset.py
#
# SSLLandmarkDataset — loads raw MediaPipe hand landmark coordinates.
#
# What this class does:
#   1. Downloads landmark .json files from HuggingFace (once, then cached)
#   2. Parses the JSON into numpy arrays
#   3. Applies the same stratified train/val/test split as the original thesis
#   4. Returns data in the shape the user requests
#
# Output shapes (controlled by the `format` parameter):
#   "lstm"  → (N, 30, 63)     flattened coords per frame — direct LSTM/Transformer input
#   "raw"   → (N, 30, 21, 3)  one row per landmark, 3 columns for x/y/z — for custom models
#
# Why two formats?
#   "lstm" matches exactly what your thesis used for the LSTM/Transformer/Ensemble models.
#   "raw"  keeps the landmark structure intact for researchers who want to do their
#          own feature engineering (e.g. compute joint angles, distances, etc.)

import json
import numpy as np
from sklearn.model_selection import train_test_split

from .._constants import (
    CLASS_LABELS,
    CLASS_SLUGS,
    STATIC_CLASS_IDS,
    DYNAMIC_CLASS_IDS,
    NUM_CLASSES,
    NUM_FRAMES,
    NUM_LANDMARKS,
    NUM_COORDS,
    SAMPLES_PER_STATIC_CLASS,
    SAMPLES_PER_DYNAMIC_CLASS,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST,
    SPLIT_SEED,
)
from .._downloader import download_landmark_file

# Valid values for the `format` parameter
VALID_FORMATS = ("lstm", "raw")

# Valid values for the `split` parameter
VALID_SPLITS = ("train", "val", "test", "all")


class SSLLandmarkDataset:
    """
    Sinhala Sign Language landmark coordinate dataset.

    Downloads and loads 3D hand landmark coordinates (x, y, z) extracted
    by MediaPipe for all 55 SSL sign classes.

    Parameters
    ----------
    split : str
        Which portion of the dataset to load.
        "train" → 70% of samples  (2842 samples)
        "val"   → 15% of samples  (609 samples)
        "test"  → 15% of samples  (609 samples)
        "all"   → entire dataset  (4060 samples)

    format : str
        Output shape for landmark arrays.
        "lstm" → (N, 30, 63)    — flattened, ready for LSTM/Transformer
        "raw"  → (N, 30, 21, 3) — structured, one entry per landmark

    return_labels : bool
        If True (default), also return a label array y of shape (N,)
        containing integer class indices (0–54).

    Examples
    --------
    >>> from ssl_dataset.landmarks import SSLLandmarkDataset
    >>>
    >>> # Load training data in LSTM format
    >>> ds = SSLLandmarkDataset(split="train", format="lstm")
    >>> X, y = ds.load()
    >>> print(X.shape)   # (2842, 30, 63)
    >>> print(y.shape)   # (2842,)
    >>>
    >>> # Load all data in raw format, labels as sign names
    >>> ds = SSLLandmarkDataset(split="all", format="raw")
    >>> X, y = ds.load()
    >>> print(X.shape)   # (4060, 30, 21, 3)
    >>> label_names = [CLASS_LABELS[i] for i in y]
    """

    def __init__(
        self,
        split: str = "train",
        format: str = "lstm",
        return_labels: bool = True,
    ):
        if split not in VALID_SPLITS:
            raise ValueError(
                f"split must be one of {VALID_SPLITS}, got {split!r}"
            )
        if format not in VALID_FORMATS:
            raise ValueError(
                f"format must be one of {VALID_FORMATS}, got {format!r}"
            )

        self.split = split
        self.format = format
        self.return_labels = return_labels

        # These are populated lazily when .load() is called
        self._X = None
        self._y = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self):
        """
        Download (if needed) and load the landmark data.

        Returns
        -------
        X : np.ndarray
            Landmark array. Shape depends on `format`:
            "lstm" → (N, 30, 63)
            "raw"  → (N, 30, 21, 3)

        y : np.ndarray  (only if return_labels=True)
            Integer class labels, shape (N,), values 0–54.

        Notes
        -----
        Calling load() a second time returns the cached result immediately
        without re-reading any files.
        """
        if self._X is None:
            self._X, self._y = self._build_dataset()

        if self.return_labels:
            return self._X, self._y
        return self._X

    @property
    def class_labels(self) -> dict:
        """Return the full class index → sign label mapping."""
        return CLASS_LABELS

    @property
    def num_classes(self) -> int:
        """Total number of sign classes (55)."""
        return NUM_CLASSES

    def __repr__(self) -> str:
        return (
            f"SSLLandmarkDataset("
            f"split={self.split!r}, "
            f"format={self.format!r})"
        )

    # ------------------------------------------------------------------
    # Internal: building the full dataset
    # ------------------------------------------------------------------

    def _build_dataset(self):
        """
        Load every landmark file, stack into arrays, apply the split,
        and return (X, y).
        """
        all_X = []
        all_y = []

        # --- Static signs (classes 0–36) ---
        # Each class has 100 samples, each sample is a single-frame
        # .json file with key "hands".
        # We replicate the single frame 30 times so static signs have
        # the same temporal shape as dynamic signs — this is exactly
        # what the thesis preprocessing pipeline did.
        print("Loading static landmark files...")
        for class_id in STATIC_CLASS_IDS:
            slug = CLASS_SLUGS[class_id]
            for sample_idx in range(SAMPLES_PER_STATIC_CLASS):
                remote_path = f"static/{slug}/{sample_idx}.json"
                local_path = download_landmark_file(remote_path)
                frame = self._parse_static_json(local_path)
                # frame shape: (21, 3)
                # replicate to 30 frames → (30, 21, 3)
                sequence = np.stack([frame] * NUM_FRAMES, axis=0)
                all_X.append(sequence)
                all_y.append(class_id)

        # --- Dynamic signs (classes 37–54) ---
        # Each class has 20 samples, each sample is a 30-frame
        # .json file with key "sequence".
        print("Loading dynamic landmark files...")
        for class_id in DYNAMIC_CLASS_IDS:
            slug = CLASS_SLUGS[class_id]
            for sample_idx in range(SAMPLES_PER_DYNAMIC_CLASS):
                remote_path = f"dynamic/{slug}/{sample_idx}.json"
                local_path = download_landmark_file(remote_path)
                sequence = self._parse_dynamic_json(local_path)
                # sequence shape: (30, 21, 3)
                all_X.append(sequence)
                all_y.append(class_id)

        # Stack all samples into a single array
        # X shape at this point: (4060, 30, 21, 3)
        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)

        # Apply train/val/test split
        X, y = self._apply_split(X, y)

        # Reshape based on requested format
        X = self._apply_format(X)

        return X, y

    # ------------------------------------------------------------------
    # Internal: JSON parsers
    # ------------------------------------------------------------------

    def _parse_static_json(self, path) -> np.ndarray:
        """
        Parse a static landmark .json file.

        File structure:
            { "hands": [ [ {x,y,z}, {x,y,z}, ... 21 landmarks ] ] }

        Returns
        -------
        np.ndarray of shape (21, 3)  — one row per landmark, columns: x, y, z
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data["hands"][0] is the first (and only) detected hand
        # It is a list of 21 dicts, each with keys "x", "y", "z"
        landmarks = data["hands"][0]
        return np.array(
            [[lm["x"], lm["y"], lm["z"]] for lm in landmarks],
            dtype=np.float32,
        )  # shape: (21, 3)

    def _parse_dynamic_json(self, path) -> np.ndarray:
        """
        Parse a dynamic landmark .json file.

        File structure:
            { "sequence": [
                [ [ {x,y,z}, ... 21 landmarks ] ],   ← frame 0
                [ [ {x,y,z}, ... 21 landmarks ] ],   ← frame 1
                ...                                   (30 frames total)
            ]}

        Returns
        -------
        np.ndarray of shape (30, 21, 3)
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames = []
        for frame_data in data["sequence"]:
            # frame_data[0] is the first (dominant) hand in this frame
            landmarks = frame_data[0]
            frame = np.array(
                [[lm["x"], lm["y"], lm["z"]] for lm in landmarks],
                dtype=np.float32,
            )  # shape: (21, 3)
            frames.append(frame)

        return np.stack(frames, axis=0)  # shape: (30, 21, 3)

    # ------------------------------------------------------------------
    # Internal: split logic
    # ------------------------------------------------------------------

    def _apply_split(self, X: np.ndarray, y: np.ndarray):
        """
        Apply stratified train/val/test split matching the thesis methodology.

        Strategy (same two-stage approach as the thesis):
          Stage 1: split off 70% train from 30% temp  (stratified)
          Stage 2: split the 30% temp into 15% val and 15% test (stratified)

        `stratify=y` ensures every class is proportionally represented
        in all three splits — critical for rare classes.
        """
        if self.split == "all":
            return X, y

        # Stage 1: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(SPLIT_VAL + SPLIT_TEST),
            stratify=y,
            random_state=SPLIT_SEED,
        )

        # Stage 2: 50% of temp → val, 50% of temp → test
        # (50% of 30% = 15% each)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=SPLIT_SEED,
        )

        splits = {
            "train": (X_train, y_train),
            "val":   (X_val,   y_val),
            "test":  (X_test,  y_test),
        }
        return splits[self.split]

    # ------------------------------------------------------------------
    # Internal: format reshaping
    # ------------------------------------------------------------------

    def _apply_format(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape X from (N, 30, 21, 3) into the requested output format.

        "raw"  → (N, 30, 21, 3)  — no change, return as-is
        "lstm" → (N, 30, 63)     — flatten the last two dims (21*3=63)
                                    this is the direct LSTM/Transformer input
        """
        if self.format == "raw":
            return X  # (N, 30, 21, 3) — no change

        if self.format == "lstm":
            N = X.shape[0]
            return X.reshape(N, NUM_FRAMES, NUM_LANDMARKS * NUM_COORDS)
            # (N, 30, 63)

        # Should never reach here due to __init__ validation
        raise ValueError(f"Unknown format: {self.format!r}")
