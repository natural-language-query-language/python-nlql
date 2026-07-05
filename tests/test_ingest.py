"""Tests for normalization, splitting, and the ingestion pipeline."""

from __future__ import annotations

import numpy as np

from nlql.embed import FakeEmbedder
from nlql.ingest import DefaultNormalizer, IngestionPipeline, split_chunks, split_sentences
from nlql.model import Document, Modality, Payload
from nlql.registry import GLOBAL_REGISTRY


class TestNormalizer:
    def test_collapses_inline_whitespace(self) -> None:
        assert DefaultNormalizer().normalize("a   b\t c") == "a b c"

    def test_preserves_single_newline(self) -> None:
        assert DefaultNormalizer().normalize("line1\nline2") == "line1\nline2"

    def test_collapses_blank_lines_and_trims(self) -> None:
        assert DefaultNormalizer().normalize("  a \n\n\n b  ") == "a\nb"


class TestSentenceSplitter:
    def test_western(self) -> None:
        assert split_sentences("Hello world. How are you? Fine!") == [
            "Hello world.",
            "How are you?",
            "Fine!",
        ]

    def test_cjk(self) -> None:
        assert split_sentences("你好。今天天气不错！真的吗？") == [
            "你好。",
            "今天天气不错！",
            "真的吗？",
        ]

    def test_newline_boundary(self) -> None:
        assert split_sentences("first line\nsecond line") == ["first line", "second line"]

    def test_empty(self) -> None:
        assert split_sentences("   ") == []

    def test_single_unpunctuated(self) -> None:
        assert split_sentences("just a fragment") == ["just a fragment"]


class TestChunkSplitter:
    def test_groups_by_max_chars(self) -> None:
        text = "Sentence one. Sentence two. Sentence three."
        chunks = split_chunks(text, max_chars=25)
        assert len(chunks) >= 2
        assert all(len(c) <= 40 for c in chunks)


class TestIngestionPipeline:
    def test_produces_units_with_vectors(self, fake_embedder: FakeEmbedder) -> None:
        pipe = IngestionPipeline(fake_embedder)
        doc = Document.from_text(
            "AI agents plan. They use memory. They call tools.",
            id="d1",
            metadata={"status": "published"},
        )
        units = pipe.process([doc])
        assert len(units) == 3
        assert [u.id for u in units] == ["d1#sentence:0", "d1#sentence:1", "d1#sentence:2"]
        assert [u.ordinal for u in units] == [0, 1, 2]
        assert all(u.kind == "sentence" and u.doc_id == "d1" for u in units)
        assert all(u.vector is not None for u in units)
        assert np.allclose(np.linalg.norm(units[0].vector), 1.0, atol=1e-5)

    def test_metadata_is_copied_not_shared(self, fake_embedder: FakeEmbedder) -> None:
        doc = Document.from_text("One. Two.", id="d1", metadata={"k": "v"})
        units = IngestionPipeline(fake_embedder).process([doc])
        units[0].metadata["k"] = "mutated"
        assert units[1].metadata["k"] == "v"  # siblings unaffected
        assert doc.metadata["k"] == "v"  # source doc unaffected

    def test_non_text_payload_skipped(self, fake_embedder: FakeEmbedder) -> None:
        doc = Document(
            id="d1",
            payloads=[Payload(modality=Modality.IMAGE, data=b"\x89PNG"), Payload.text("Hello.")],
        )
        units = IngestionPipeline(fake_embedder).process([doc])
        assert len(units) == 1
        assert units[0].content == "Hello."

    def test_custom_splitter_via_child_registry(self, fake_embedder: FakeEmbedder) -> None:
        reg = GLOBAL_REGISTRY.child()
        reg.register("splitter", "SENTENCE", lambda t: t.split("|"), overwrite=False)
        pipe = IngestionPipeline(fake_embedder, registry=reg)
        units = pipe.process([Document.from_text("a|b|c", id="d1")])
        assert [u.content for u in units] == ["a", "b", "c"]

    def test_empty_document_yields_no_units(self, fake_embedder: FakeEmbedder) -> None:
        units = IngestionPipeline(fake_embedder).process([Document.from_text("   ", id="d1")])
        assert units == []
