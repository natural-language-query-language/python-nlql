"""Embedding cache — the direct fix for the reference implementation's #1 defect
(recomputing every embedding on every query).

The cache key binds ``model_id + dim + modality + normalized_text``, so switching model
or output dimension can never return a stale / mis-sized vector. :class:`CachedEmbedder`
wraps any :class:`~nlql.embed.base.Embedder`, serving hits from the cache and computing
only the misses in a single batch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from nlql.embed.base import Embedder
from nlql.errors import NLQLEmbeddingError


def cache_key(model_id: str, dim: int, modality: str, text: str) -> str:
    """Stable content-addressed key for one embedding."""
    payload = f"{model_id}\x00{dim}\x00{modality}\x00{text}".encode()
    return hashlib.sha256(payload).hexdigest()


class EmbeddingCache:
    """An in-memory embedding cache with optional on-disk persistence."""

    def __init__(self) -> None:
        self._store: dict[str, np.ndarray] = {}

    def get(self, key: str) -> np.ndarray | None:
        return self._store.get(key)

    def set(self, key: str, vector: np.ndarray) -> None:
        self._store[key] = np.asarray(vector, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def save(self, path: str | Path) -> None:
        """Persist to a ``.npz`` file (keys + stacked vectors)."""
        keys = list(self._store)
        matrix = (
            np.stack([self._store[k] for k in keys])
            if keys
            else np.empty((0, 0), dtype=np.float32)
        )
        np.savez(path, keys=np.array(keys, dtype=object), vectors=matrix)

    def load(self, path: str | Path) -> None:
        """Load entries from a ``.npz`` file, merging into the current cache."""
        data = np.load(path, allow_pickle=True)
        keys = data["keys"]
        vectors = data["vectors"]
        for i, key in enumerate(keys):
            self._store[str(key)] = np.asarray(vectors[i], dtype=np.float32)


class CachedEmbedder:
    """Wraps an embedder so repeated texts are embedded at most once."""

    def __init__(
        self,
        inner: Embedder,
        cache: EmbeddingCache | None = None,
        *,
        modality: str = "text",
    ) -> None:
        self._inner = inner
        self._cache = cache if cache is not None else EmbeddingCache()
        self._modality = modality

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    @property
    def inner(self) -> Embedder:
        """The wrapped embedder (used to detect multimodal support)."""
        return self._inner

    def _key(self, text: str) -> str:
        return cache_key(self._inner.model_id, self._inner.dim, self._modality, text)

    def _image_key(self, image: bytes | str) -> str:
        raw = image if isinstance(image, (bytes, bytearray)) else str(image).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        return cache_key(self._inner.model_id, self._inner.dim, "image", digest)

    def _cached(
        self,
        items: list,
        key_fn: Callable[[object], str],
        compute: Callable[[list], np.ndarray],
    ) -> np.ndarray:
        if not items:
            return np.empty((0, self.dim), dtype=np.float32)
        results: list[np.ndarray | None] = [None] * len(items)
        misses: list = []
        miss_index: list[int] = []
        for i, item in enumerate(items):
            hit = self._cache.get(key_fn(item))
            if hit is None:
                misses.append(item)
                miss_index.append(i)
            else:
                results[i] = hit
        if misses:
            computed = compute(misses)
            for j, i in enumerate(miss_index):
                vec = np.asarray(computed[j], dtype=np.float32)
                results[i] = vec
                self._cache.set(key_fn(items[i]), vec)
        return np.stack([r for r in results if r is not None])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._cached(list(texts), self._key, self._inner.embed)  # type: ignore[arg-type]

    def embed_images(self, images: Sequence[bytes | str]) -> np.ndarray:
        compute = getattr(self._inner, "embed_images", None)
        if not callable(compute):
            raise NLQLEmbeddingError(f"{self._inner.model_id} cannot embed images")
        return self._cached(list(images), self._image_key, compute)  # type: ignore[arg-type]
