"""Deterministic, offline fake embedder for tests and demos.

Each token maps to a fixed pseudo-random unit vector (seeded from its hash), and a
text embeds to the normalized sum of its token vectors — a "bag of hashed tokens".
Consequence: texts that share words get higher cosine similarity, so semantic-pipeline
tests can assert ranking/threshold behaviour deterministically, with no model download
and no network. Identical texts always embed identically across processes.
"""

from __future__ import annotations

import hashlib

import numpy as np

from nlql.embed.base import BaseEmbedder


class FakeEmbedder(BaseEmbedder):
    """A hash-based embedder producing stable vectors without any model."""

    def __init__(self, dim: int = 64, model_id: str = "fake") -> None:
        self._dim = dim
        self._model_id = f"{model_id}:{dim}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _token_vector(self, token: str) -> np.ndarray:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self._dim).astype(np.float32)

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split() or [text.lower()]
            for tok in tokens:
                out[i] += self._token_vector(tok)
        return out
