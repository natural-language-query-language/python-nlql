"""Tests for the modality-agnostic data model."""

import numpy as np
import pytest

from nlql.model import (
    Document,
    Modality,
    Payload,
    Span,
    Unit,
    as_array,
    normalize,
    to_list,
)


class TestPayload:
    def test_text_constructor(self) -> None:
        p = Payload.text("hello")
        assert p.modality is Modality.TEXT
        assert p.is_text
        assert p.as_text == "hello"

    def test_non_text_as_text_is_empty(self) -> None:
        p = Payload(modality=Modality.IMAGE, data=b"\x89PNG", mime="image/png")
        assert not p.is_text
        assert p.as_text == ""

    def test_text_payload_rejects_bytes(self) -> None:
        with pytest.raises(TypeError):
            Payload(modality=Modality.TEXT, data=b"nope")

    def test_modality_serializes_as_string(self) -> None:
        assert Modality.TEXT == "text"
        assert str(Modality.IMAGE) == "image"


class TestDocument:
    def test_from_text(self) -> None:
        d = Document.from_text("body", id="d1", metadata={"status": "published"})
        assert d.id == "d1"
        assert len(d.payloads) == 1
        assert d.payloads[0].as_text == "body"
        assert d.metadata["status"] == "published"

    def test_empty_payloads_rejected(self) -> None:
        with pytest.raises(ValueError):
            Document(id="d1", payloads=[])

    def test_metadata_defaults_to_empty_dict(self) -> None:
        d = Document(id="d1", payloads=[Payload.text("x")])
        assert d.metadata == {}
        assert d.source is None


class TestUnit:
    def test_content_from_text_payload(self) -> None:
        u = Unit(id="u1", doc_id="d1", kind="sentence", payload=Payload.text("a sentence"))
        assert u.content == "a sentence"
        assert u.scores == {}
        assert u.vector is None

    def test_content_empty_for_non_text(self) -> None:
        u = Unit(
            id="u1",
            doc_id="d1",
            kind="chunk",
            payload=Payload(modality=Modality.BLOB, data=b"\x00"),
        )
        assert u.content == ""

    def test_named_scores(self) -> None:
        u = Unit(id="u1", doc_id="d1", kind="sentence", payload=Payload.text("x"))
        u.scores["relevance"] = 0.87
        u.scores["novelty"] = 0.42
        assert u.scores["relevance"] == 0.87
        assert u.scores["novelty"] == 0.42

    def test_span_info(self) -> None:
        span = Span(center=3, start=2, end=4)
        u = Unit(
            id="u1", doc_id="d1", kind="span", payload=Payload.text("ctx"), span=span, ordinal=3
        )
        assert u.span is not None
        assert (u.span.start, u.span.center, u.span.end) == (2, 3, 4)
        assert u.ordinal == 3


class TestVector:
    def test_as_array_flattens_and_casts(self) -> None:
        arr = as_array([[1.0, 2.0]])
        assert arr.ndim == 1
        assert arr.dtype == np.float32
        assert arr.tolist() == [1.0, 2.0]

    def test_normalize_unit_length(self) -> None:
        v = normalize([3.0, 4.0])
        assert np.isclose(np.linalg.norm(v), 1.0)
        assert np.allclose(v, [0.6, 0.8])

    def test_normalize_zero_vector_unchanged(self) -> None:
        v = normalize([0.0, 0.0, 0.0])
        assert np.allclose(v, [0.0, 0.0, 0.0])

    def test_to_list(self) -> None:
        assert to_list(np.array([1.0, 2.0], dtype=np.float32)) == [1.0, 2.0]

    def test_normalized_dot_is_cosine(self) -> None:
        # The core invariant the store relies on: dot of unit vectors == cosine.
        a = normalize([1.0, 2.0, 3.0])
        b = normalize([2.0, 1.0, 0.5])
        cosine = float(np.dot(a, b))
        assert -1.0 <= cosine <= 1.0
