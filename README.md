# ssl-dataset

Python library for the **Sinhala Sign Language (SSL) dataset** — 55 sign classes, 4,060 samples, three ready-to-use sub-libraries.

## Installation

```bash
pip install ssl-dataset
```

## Dataset overview

| Type | Classes | Samples per class | Total samples |
|---|---|---|---|
| Static signs (images) | 0 – 36 | 100 | 3,700 |
| Dynamic signs (videos) | 37 – 54 | 20 | 360 |
| **Total** | **55** | | **4,060** |

---

## Sub-libraries

### `ssl_dataset.landmarks`
Raw 3D hand landmark coordinates extracted by MediaPipe.

```python
from ssl_dataset.landmarks import SSLLandmarkDataset

# LSTM / Transformer input — shape (N, 30, 63)
ds = SSLLandmarkDataset(split="train", format="lstm")
X, y = ds.load()
print(X.shape)   # (2842, 30, 63)
print(y.shape)   # (2842,)

# Raw structured format — shape (N, 30, 21, 3)
ds = SSLLandmarkDataset(split="train", format="raw")
X, y = ds.load()
print(X.shape)   # (2842, 30, 21, 3)
```

---

### `ssl_dataset.skeleton`
28×28 RGB hand skeleton images generated from MediaPipe landmarks.

```python
from ssl_dataset.skeleton import SSLSkeletonDataset

# CNN-LSTM input — shape (N, 30, 28, 28, 3)
ds = SSLSkeletonDataset(split="train", format="cnn_lstm")
X, y = ds.load()
print(X.shape)   # (2842, 30, 28, 28, 3)

# MLP input — shape (N, 70560)
ds = SSLSkeletonDataset(split="test", format="mlp")
X, y = ds.load()
print(X.shape)   # (609, 70560)
```

---

### `ssl_dataset.preprocessed`
Pre-split, pre-labelled numpy arrays — load and train immediately.

```python
from ssl_dataset.preprocessed import SSLPreprocessedDataset

X_train, y_train = SSLPreprocessedDataset("train").load()
X_val,   y_val   = SSLPreprocessedDataset("val").load()
X_test,  y_test  = SSLPreprocessedDataset("test").load()

print(X_train.shape)   # (2842, 30, 63)
print(y_train.shape)   # (2842, 55)  — one-hot encoded

# input_shape helper for model building
ds = SSLPreprocessedDataset("train")
print(ds.input_shape)  # (30, 63)
```

---

## Split details

All sub-libraries use the same stratified split (seed = 42) matching the original thesis methodology:

| Split | Samples | Percentage |
|---|---|---|
| Train | 2,842 | 70% |
| Val | 609 | 15% |
| Test | 609 | 15% |

---

## Class labels

```python
from ssl_dataset import CLASS_LABELS

print(CLASS_LABELS[0])   # 'අ'
print(CLASS_LABELS[37])  # 'ඈ'
print(CLASS_LABELS[54])  # 'මගේ'
```

---

## Citation

If you use this dataset in your research, please cite:

```
Jayasha (2025). Sinhala Sign Language Dataset.
GitHub: https://github.com/jayashalakshani/ssl-dataset
```

## License

This dataset and library are licensed under **CC BY-NC 4.0** (Creative Commons Attribution Non-Commercial 4.0).

You are free to use, share, and adapt this dataset for **research and educational purposes**, as long as you give appropriate credit. **Commercial use is not permitted.**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
