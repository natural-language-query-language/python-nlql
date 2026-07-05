"""Optional local embedder backed by sentence-transformers (lazy-loaded).

Never a hard dependency: the ``sentence_transformers`` import happens on first use, and a
missing install raises a clear, actionable error pointing at the ``nlql[local]`` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nlql.embed.base import BaseEmbedder
from nlql.errors import NLQLEmbeddingError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wraps a ``sentence_transformers.SentenceTransformer`` model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", *, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise NLQLEmbeddingError(
                "SentenceTransformerEmbedder requires the 'local' extra: pip install python-nlql[local]"
            ) from e
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_id(self) -> str:
        return f"st:{self._model_name}"

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        vectors = self._model.encode(texts, convert_to_numpy=True)  # type: ignore[attr-defined]
        return np.asarray(vectors, dtype=np.float32)
