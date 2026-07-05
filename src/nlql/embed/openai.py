"""OpenAI-compatible embedder.

Talks to any ``/v1/embeddings`` endpoint (OpenAI, Azure, or a compatible gateway) over
plain HTTP — no heavy SDK, no native deps. This is the "production" showcase backend and
ties NLQL directly to modern AI stacks. A custom ``httpx.Client`` can be injected, which
makes the network layer trivially mockable in tests.
"""

from __future__ import annotations

import os

import httpx
import numpy as np

from nlql.embed.base import BaseEmbedder
from nlql.errors import NLQLEmbeddingError

# Native output dimensions for common OpenAI models (used when `dimensions` is unset).
_DEFAULT_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """Embedder backed by an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._dim = dimensions or _DEFAULT_DIMS.get(model, 1536)
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("NLQL_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise NLQLEmbeddingError(
                    "OpenAIEmbedder needs an api_key (or NLQL_OPENAI_API_KEY / OPENAI_API_KEY env var)"
                )
            self._client = httpx.Client(
                base_url=base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {key}"},
                timeout=timeout,
            )

    @property
    def model_id(self) -> str:
        # Dimension is part of the id so truncated-dim vectors never collide in the cache.
        return f"openai:{self._model}:{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        payload: dict[str, object] = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        try:
            response = self._client.post("/embeddings", json=payload)
        except httpx.HTTPError as e:  # network/timeout
            raise NLQLEmbeddingError(f"embedding request failed: {e}") from e
        if response.status_code != 200:
            raise NLQLEmbeddingError(
                f"embedding endpoint returned {response.status_code}: {response.text[:200]}"
            )
        data = response.json().get("data", [])
        # Responses may arrive out of order; sort by the returned index.
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in ordered]
        if len(vectors) != len(texts):
            raise NLQLEmbeddingError(
                f"expected {len(texts)} embeddings, got {len(vectors)}"
            )
        return np.array(vectors, dtype=np.float32)

    def close(self) -> None:
        self._client.close()
