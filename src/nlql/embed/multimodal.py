"""Multimodal embedding — text and images in one vector space (M5).

The whole point of the modality-agnostic model pays off here: a multimodal embedder places
text and images in the *same* space (like CLIP), so a text query retrieves image units over
the exact same IR and index path — only the vector's origin differs.

``FakeMultimodalEmbedder`` is the deterministic, offline stand-in: it decodes an image's
bytes to a caption and embeds it in the same bag-of-tokens space as text, so a text query
whose words overlap the caption scores high. No model, no network. Real CLIP lives in
``nlql.embed.clip`` (optional).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from nlql.embed.base import Embedder
from nlql.embed.fake import FakeEmbedder


@runtime_checkable
class MultimodalEmbedder(Embedder, Protocol):
    """An embedder that can also embed images into the same space as text."""

    def embed_images(self, images: Sequence[bytes | str]) -> np.ndarray:
        """Return an ``(n, dim)`` matrix of unit-normalized image vectors."""
        ...


def supports_images(embedder: Embedder) -> bool:
    """Whether an embedder (unwrapping a cache wrapper) can embed images."""
    inner = getattr(embedder, "inner", embedder)
    return callable(getattr(inner, "embed_images", None))


class FakeMultimodalEmbedder(FakeEmbedder):
    """Deterministic multimodal fake: an image's bytes are read as a caption and embedded in
    the same bag-of-tokens space as text."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim, model_id="fake-mm")

    def embed_images(self, images: Sequence[bytes | str]) -> np.ndarray:
        captions = [
            img.decode("utf-8", "ignore") if isinstance(img, (bytes, bytearray)) else str(img)
            for img in images
        ]
        return self.embed(captions)
