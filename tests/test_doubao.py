"""Tests for DoubaoVisionEmbedder (mocked HTTP; no real API calls)."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from nlql.embed import DoubaoVisionEmbedder, supports_images
from nlql.embed.doubao import _data_uri
from nlql.errors import NLQLEmbeddingError


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test/api/v3")


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    part = body["input"][0]
    # Distinct vectors for text vs image so tests can tell them apart.
    vec = [float(len(part.get("text", ""))), 1.0, 2.0, 3.0] if part["type"] == "text" else [9.0, 1.0, 2.0, 3.0]
    return httpx.Response(200, json={"data": {"embedding": vec}, "model": "doubao-embedding-vision"})


class TestDoubaoVisionEmbedder:
    def test_embed_text_normalized(self) -> None:
        emb = DoubaoVisionEmbedder(dim=4, client=_mock_client(_handler))
        out = emb.embed(["aa", "bbbb"])
        assert out.shape == (2, 4)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_embed_images_normalized(self) -> None:
        emb = DoubaoVisionEmbedder(dim=4, client=_mock_client(_handler))
        out = emb.embed_images([b"\x89PNG\r\n", b"\xff\xd8\xff"])
        assert out.shape == (2, 4)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_is_multimodal(self) -> None:
        assert supports_images(DoubaoVisionEmbedder(dim=4, client=_mock_client(_handler)))

    def test_model_id(self) -> None:
        emb = DoubaoVisionEmbedder(dim=2048, client=_mock_client(_handler))
        assert emb.model_id == "doubao-vision:doubao-embedding-vision:2048"

    def test_http_error_raises(self) -> None:
        def bad(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="Image dimensions are too small")

        emb = DoubaoVisionEmbedder(dim=4, client=_mock_client(bad))
        with pytest.raises(NLQLEmbeddingError):
            emb.embed(["x"])

    def test_missing_api_key_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("NLQL_ARK_API_KEY", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        with pytest.raises(NLQLEmbeddingError):
            DoubaoVisionEmbedder()


class TestDataUri:
    def test_bytes_png(self) -> None:
        assert _data_uri(b"\x89PNG\r\n\x1a\n").startswith("data:image/png;base64,")

    def test_bytes_jpeg(self) -> None:
        assert _data_uri(b"\xff\xd8\xff\xe0").startswith("data:image/jpeg;base64,")

    def test_url_passthrough(self) -> None:
        assert _data_uri("https://example.com/x.jpg") == "https://example.com/x.jpg"
