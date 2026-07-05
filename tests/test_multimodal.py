"""Tests for multimodal embedding and image ingestion (M5)."""

from __future__ import annotations

import numpy as np
import pytest

from nlql import Document, Engine
from nlql.embed import (
    CachedEmbedder,
    FakeEmbedder,
    FakeMultimodalEmbedder,
    supports_images,
)
from nlql.errors import NLQLEmbeddingError
from nlql.ingest import IngestionPipeline
from nlql.model import Modality, Payload
from nlql.store import LocalStore


class TestFakeMultimodal:
    def test_image_and_text_share_space(self) -> None:
        emb = FakeMultimodalEmbedder(dim=64)
        # An image whose bytes decode to a caption lands near a matching text query.
        img = emb.embed_images([b"a photo of a fluffy cat"])
        related = emb.embed(["fluffy cat photo"])
        unrelated = emb.embed(["quarterly financial report"])
        assert float(img[0] @ related[0]) > float(img[0] @ unrelated[0])

    def test_deterministic(self) -> None:
        a = FakeMultimodalEmbedder(dim=32).embed_images([b"same bytes"])
        b = FakeMultimodalEmbedder(dim=32).embed_images([b"same bytes"])
        assert np.array_equal(a, b)


class TestSupportsImages:
    def test_detection(self) -> None:
        assert supports_images(FakeMultimodalEmbedder()) is True
        assert supports_images(FakeEmbedder()) is False
        assert supports_images(CachedEmbedder(FakeMultimodalEmbedder())) is True
        assert supports_images(CachedEmbedder(FakeEmbedder())) is False


class TestCachedEmbedderImages:
    def test_caches_and_delegates(self) -> None:
        cached = CachedEmbedder(FakeMultimodalEmbedder())
        v1 = cached.embed_images([b"cat"])
        v2 = cached.embed_images([b"cat"])
        assert np.array_equal(v1, v2)
        assert len(cached.cache) == 1

    def test_raises_for_non_multimodal(self) -> None:
        cached = CachedEmbedder(FakeEmbedder())
        with pytest.raises(NLQLEmbeddingError):
            cached.embed_images([b"cat"])


class TestImageIngestion:
    def test_image_payload_becomes_unit(self) -> None:
        pipe = IngestionPipeline(CachedEmbedder(FakeMultimodalEmbedder()), granularity="chunk")
        doc = Document(
            id="img1",
            payloads=[Payload(modality=Modality.IMAGE, data=b"a red bicycle", mime="image/png")],
            metadata={"kind": "image"},
        )
        units = pipe.process([doc])
        assert len(units) == 1
        assert units[0].payload.modality is Modality.IMAGE
        assert units[0].vector is not None

    def test_images_skipped_without_multimodal_embedder(self) -> None:
        pipe = IngestionPipeline(CachedEmbedder(FakeEmbedder()), granularity="chunk")
        doc = Document(id="img1", payloads=[Payload(modality=Modality.IMAGE, data=b"x")])
        assert pipe.process([doc]) == []


class TestCrossModalRetrieval:
    def test_text_query_retrieves_image(self) -> None:
        engine = Engine(FakeMultimodalEmbedder(), store=LocalStore(), granularity="chunk")
        engine.add_documents(
            [
                Document(id="cat", payloads=[Payload(Modality.IMAGE, b"a photo of a fluffy cat")], metadata={"t": "img"}),
                Document(id="car", payloads=[Payload(Modality.IMAGE, b"a red sports car on a road")], metadata={"t": "img"}),
                Document(id="dog", payloads=[Payload(Modality.IMAGE, b"a happy dog in a park")], metadata={"t": "img"}),
            ]
        )
        results = engine.search(
            'SELECT CHUNK LET rel = SIMILARITY(content, "fluffy cat animal") ORDER BY rel DESC LIMIT 1'
        )
        assert results[0].doc_id == "cat"  # text query → image result, same IR/index path

    def test_mixed_text_and_image_corpus(self) -> None:
        engine = Engine(FakeMultimodalEmbedder(), granularity="chunk")
        engine.add_documents(
            [
                Document.from_text("A written article about mountain hiking trails.", id="text1"),
                Document(id="img1", payloads=[Payload(Modality.IMAGE, b"a scenic mountain hiking trail")]),
            ]
        )
        results = engine.search(
            'SELECT CHUNK LET rel = SIMILARITY(content, "mountain hiking") ORDER BY rel DESC'
        )
        assert {r.doc_id for r in results} == {"text1", "img1"}  # both modalities retrievable


class TestAddImage:
    def _engine(self) -> Engine:
        return Engine(FakeMultimodalEmbedder(), granularity="chunk")

    def test_add_image_bytes(self) -> None:
        engine = self._engine()
        doc_id = engine.add_image(b"a photo of a fluffy cat", metadata={"kind": "photo"})
        assert isinstance(doc_id, str) and len(engine) == 1
        results = engine.search('SELECT CHUNK LET rel = SIMILARITY(content, "cat") ORDER BY rel DESC LIMIT 1')
        assert results[0].doc_id == doc_id
        assert results[0].payload.modality is Modality.IMAGE

    def test_add_image_from_path(self, tmp_path) -> None:
        path = tmp_path / "img.bin"
        path.write_bytes(b"a red bicycle on a street")
        engine = self._engine()
        assert engine.add_image(str(path), id="bike") == "bike"
        assert len(engine) == 1

    def test_add_image_url_passthrough(self) -> None:
        engine = self._engine()
        engine.add_image("https://example.com/x.jpg", id="u")
        assert len(engine) == 1


class TestMultimodalExternalStore:
    """Image-derived vectors go into any store; text queries retrieve them cross-modally."""

    def test_image_vectors_in_chroma(self) -> None:
        pytest.importorskip("chromadb")
        from nlql.store.chroma_store import ChromaStore

        engine = Engine(FakeMultimodalEmbedder(), store=ChromaStore(), granularity="chunk")
        engine.add_image(b"a photo of a fluffy cat", id="cat", metadata={"kind": "animal"})
        engine.add_image(b"a red sports car on a road", id="car", metadata={"kind": "vehicle"})

        # Text query retrieves the image whose vector Chroma holds (shared space).
        results = engine.search(
            'SELECT CHUNK LET rel = SIMILARITY(content, "fluffy cat animal") ORDER BY rel DESC LIMIT 1'
        )
        assert results[0].doc_id == "cat"

        # Native metadata filter (pushed to Chroma) works over image records too.
        filtered = engine.search(
            'SELECT CHUNK LET rel = SIMILARITY(content, "vehicle") '
            'WHERE meta.kind == "vehicle" ORDER BY rel DESC LIMIT 1'
        )
        assert filtered[0].doc_id == "car"
