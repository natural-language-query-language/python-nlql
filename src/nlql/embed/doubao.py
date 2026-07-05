"""Doubao (Volcengine Ark) vision embedder — text and images in one space over HTTP.

Uses the Ark ``/embeddings/multimodal`` endpoint (``doubao-embedding-vision``): text and
images share one 2048-d space, so text→image retrieval works with **no local model and no
torch** — pure HTTP. It implements the :class:`~nlql.embed.multimodal.MultimodalEmbedder`
protocol, so it drops into any :class:`~nlql.sdk.engine.Engine`, proving the vision extension
point on a hosted production model. A custom ``httpx.Client`` can be injected for tests.

The multimodal endpoint embeds one input per request, so batches are sent as sequential
calls; pair with a :class:`~nlql.embed.cache.CachedEmbedder` to avoid recomputation.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Sequence

import httpx
import numpy as np

from nlql.embed.base import BaseEmbedder, normalize_rows
from nlql.errors import NLQLEmbeddingError


def _data_uri(image: bytes | str) -> str:
    """Turn image bytes into a data URI; pass http(s) / data URIs through unchanged."""
    if isinstance(image, str):
        return image  # already a URL or data URI
    raw = bytes(image)
    if raw[:8].startswith(b"\x89PNG"):
        mime = "image/png"
    elif raw[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif raw[:4] == b"RIFF":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


class DoubaoVisionEmbedder(BaseEmbedder):
    """Multimodal embedder backed by Volcengine Ark's ``doubao-embedding-vision``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        model: str = "doubao-embedding-vision",
        dim: int = 2048,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._dim = dim
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("NLQL_ARK_API_KEY") or os.environ.get("ARK_API_KEY")
            if not key:
                raise NLQLEmbeddingError(
                    "DoubaoVisionEmbedder needs an api_key (or NLQL_ARK_API_KEY / ARK_API_KEY env var)"
                )
            self._client = httpx.Client(
                base_url=base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {key}"},
                timeout=timeout,
            )

    @property
    def model_id(self) -> str:
        return f"doubao-vision:{self._model}:{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, parts: list[dict]) -> np.ndarray:
        try:
            response = self._client.post(
                "/embeddings/multimodal", json={"model": self._model, "input": parts}
            )
        except httpx.HTTPError as e:
            raise NLQLEmbeddingError(f"ark embedding request failed: {e}") from e
        if response.status_code != 200:
            raise NLQLEmbeddingError(
                f"ark embedding endpoint returned {response.status_code}: {response.text[:200]}"
            )
        return np.asarray(response.json()["data"]["embedding"], dtype=np.float32)

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_one([{"type": "text", "text": t}]) for t in texts])

    def embed_images(self, images: Sequence[bytes | str]) -> np.ndarray:
        """Embed images (bytes, file/URL string, or data URI) into the shared text space."""
        items = list(images)
        if not items:
            return np.empty((0, self._dim), dtype=np.float32)
        vectors = np.stack(
            [
                self._embed_one([{"type": "image_url", "image_url": {"url": _data_uri(img)}}])
                for img in items
            ]
        )
        return normalize_rows(vectors)

    def close(self) -> None:
        self._client.close()
