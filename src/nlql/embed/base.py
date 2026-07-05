"""Embedder protocol and shared base.

By convention every embedder returns **unit-normalized** ``float32`` row vectors, so a
dot product equals cosine similarity. Normalization is centralized here in
:class:`BaseEmbedder` so concrete backends only implement ``_embed_raw``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Structural type for anything that turns texts into vectors."""

    @property
    def model_id(self) -> str:
        """Stable identifier of the model/config; part of the cache key."""
        ...

    @property
    def dim(self) -> int:
        """Output vector dimensionality."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 matrix of unit-normalized row vectors."""
        ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Unit-normalize each row; zero rows are left as zeros."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(m / norms, dtype=np.float32)


class BaseEmbedder(ABC):
    """Base class handling the empty-input case and row normalization."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        """Compute raw (un-normalized) ``(n, dim)`` embeddings."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.empty((0, self.dim), dtype=np.float32)
        return normalize_rows(self._embed_raw(items))
