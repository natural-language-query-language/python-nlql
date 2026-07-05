"""Vector type and helpers.

Vectors are 1-D ``float32`` numpy arrays. By convention they are **unit-normalized**
at ingestion and query time, so a dot product equals cosine similarity — this keeps
the similarity metric honest (raw cosine in ``[-1, 1]``) instead of the reference
implementation's opaque ``(cos + 1) / 2`` rescaling.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# A vector is just a 1-D float32 ndarray; the alias documents intent at call sites.
Vector = np.ndarray


def as_array(v: Sequence[float] | np.ndarray, dtype: type = np.float32) -> np.ndarray:
    """Coerce a sequence / array into a contiguous 1-D array of ``dtype``."""
    arr: np.ndarray = np.asarray(v, dtype=dtype)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.ascontiguousarray(arr)


def normalize(v: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the unit-normalized vector; a zero vector is returned unchanged."""
    arr = as_array(v)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def to_list(v: Sequence[float] | np.ndarray) -> list[float]:
    """Convert a vector to a plain ``list[float]`` (e.g. for JSON / transport)."""
    return [float(x) for x in as_array(v)]
