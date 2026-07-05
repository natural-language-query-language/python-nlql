"""Optional CLIP embedder — real text/image shared space via sentence-transformers.

Puts text queries and images in one vector space (e.g. ``clip-ViT-B-32``), so text→image
retrieval works over the same index. Lazy-loaded and optional (``pip install python-nlql[local]``
plus ``Pillow`` for image input); not exercised in CI. The deterministic
:class:`~nlql.embed.multimodal.FakeMultimodalEmbedder` covers the pipeline in tests.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import numpy as np

from nlql.embed.base import BaseEmbedder, normalize_rows
from nlql.errors import NLQLEmbeddingError


class ClipEmbedder(BaseEmbedder):
    """A CLIP model wrapper embedding both text and images into one space."""

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise NLQLEmbeddingError(
                "ClipEmbedder needs sentence-transformers: pip install python-nlql[local]"
            ) from e
        self._model = SentenceTransformer(self._model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_id(self) -> str:
        return f"clip:{self._model_name}"

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_images(self, images: Sequence[bytes | str]) -> np.ndarray:
        """Embed images (bytes or file paths/URIs) into the shared space (normalized)."""
        self._ensure_loaded()
        assert self._model is not None
        try:
            from PIL import Image
        except ImportError as e:  # pragma: no cover
            raise NLQLEmbeddingError("ClipEmbedder image input needs Pillow") from e
        loaded = [
            Image.open(io.BytesIO(img)) if isinstance(img, (bytes, bytearray)) else Image.open(img)
            for img in images
        ]
        vectors = self._model.encode(loaded, convert_to_numpy=True)
        return normalize_rows(np.asarray(vectors, dtype=np.float32))
