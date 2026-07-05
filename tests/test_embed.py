"""Tests for embedders and the embedding cache."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from nlql.embed import (
    CachedEmbedder,
    EmbeddingCache,
    FakeEmbedder,
    OpenAIEmbedder,
    cache_key,
    normalize_rows,
)
from nlql.errors import NLQLEmbeddingError


class TestFakeEmbedder:
    def test_shape_and_normalization(self, fake_embedder: FakeEmbedder) -> None:
        vecs = fake_embedder.embed(["hello world", "another text"])
        assert vecs.shape == (2, 64)
        assert vecs.dtype == np.float32
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    def test_deterministic_across_instances(self) -> None:
        a = FakeEmbedder(dim=32).embed(["stable text"])
        b = FakeEmbedder(dim=32).embed(["stable text"])
        assert np.array_equal(a, b)

    def test_shared_tokens_more_similar(self, fake_embedder: FakeEmbedder) -> None:
        v = fake_embedder.embed(
            [
                "machine learning models",
                "machine learning systems",  # shares 2/3 tokens with the first
                "banana bread recipe",  # shares nothing
            ]
        )
        sim_related = float(np.dot(v[0], v[1]))
        sim_unrelated = float(np.dot(v[0], v[2]))
        assert sim_related > sim_unrelated

    def test_empty_input(self, fake_embedder: FakeEmbedder) -> None:
        assert fake_embedder.embed([]).shape == (0, 64)

    def test_model_id_includes_dim(self) -> None:
        assert FakeEmbedder(dim=16).model_id == "fake:16"


class TestNormalizeRows:
    def test_zero_row_preserved(self) -> None:
        out = normalize_rows(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
        assert np.allclose(out[0], [0.0, 0.0])
        assert np.allclose(out[1], [0.6, 0.8])


class TestEmbeddingCache:
    def test_set_get_contains(self) -> None:
        cache = EmbeddingCache()
        key = cache_key("m", 3, "text", "hello")
        assert key not in cache
        cache.set(key, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert key in cache
        assert len(cache) == 1
        assert np.array_equal(cache.get(key), np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_key_depends_on_model_and_dim(self) -> None:
        assert cache_key("m1", 3, "text", "x") != cache_key("m2", 3, "text", "x")
        assert cache_key("m1", 3, "text", "x") != cache_key("m1", 4, "text", "x")

    def test_save_load_round_trip(self, tmp_path) -> None:
        cache = EmbeddingCache()
        cache.set("k1", np.array([1.0, 0.0], dtype=np.float32))
        cache.set("k2", np.array([0.0, 1.0], dtype=np.float32))
        path = tmp_path / "cache.npz"
        cache.save(path)

        restored = EmbeddingCache()
        restored.load(path)
        assert len(restored) == 2
        assert np.array_equal(restored.get("k1"), np.array([1.0, 0.0], dtype=np.float32))


class _CountingEmbedder(FakeEmbedder):
    """Fake embedder that records how many texts it actually embedded."""

    def __init__(self) -> None:
        super().__init__(dim=16)
        self.embedded: list[str] = []

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        self.embedded.extend(texts)
        return super()._embed_raw(texts)


class TestCachedEmbedder:
    def test_second_call_is_all_hits(self) -> None:
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner)
        cached.embed(["a", "b"])
        assert inner.embedded == ["a", "b"]
        cached.embed(["a", "b"])  # fully cached — inner must not run again
        assert inner.embedded == ["a", "b"]
        assert len(cached.cache) == 2

    def test_mixed_hits_and_misses_preserve_order(self) -> None:
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner)
        first = cached.embed(["a", "b"])
        inner.embedded.clear()
        second = cached.embed(["b", "a", "c"])  # b,a cached; only c is new
        assert inner.embedded == ["c"]
        assert np.array_equal(second[0], first[1])  # "b"
        assert np.array_equal(second[1], first[0])  # "a"

    def test_empty(self) -> None:
        assert CachedEmbedder(FakeEmbedder()).embed([]).shape[0] == 0


def _openai_mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test/v1")


class TestOpenAIEmbedder:
    def test_returns_normalized_vectors(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            # Return one 4-dim vector per input, deliberately out of order by index.
            data = [
                {"index": i, "embedding": [float(len(t)), 1.0, 2.0, 3.0]}
                for i, t in enumerate(body["input"])
            ]
            return httpx.Response(200, json={"data": list(reversed(data))})

        emb = OpenAIEmbedder(model="text-embedding-3-small", dimensions=4, client=_openai_mock_client(handler))
        out = emb.embed(["aa", "bbbb"])
        assert out.shape == (2, 4)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        assert emb.dim == 4
        assert emb.model_id == "openai:text-embedding-3-small:4"

    def test_http_error_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="rate limited")

        emb = OpenAIEmbedder(dimensions=4, client=_openai_mock_client(handler))
        with pytest.raises(NLQLEmbeddingError):
            emb.embed(["x"])

    def test_missing_api_key_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("NLQL_OPENAI_API_KEY", raising=False)
        with pytest.raises(NLQLEmbeddingError):
            OpenAIEmbedder()

    def test_default_dim_for_known_model(self) -> None:
        emb = OpenAIEmbedder(client=_openai_mock_client(lambda r: httpx.Response(200, json={"data": []})))
        assert emb.dim == 1536
