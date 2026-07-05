"""Unit and Span — the retrieval / return granularity.

A :class:`Unit` is the unified body for document / chunk / sentence / span results.
Named ``scores`` replace the reference implementation's magic ``metadata["similarity"]``,
so a single query can carry several semantic scores (``relevance``, ``novelty``, …).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from nlql.model.payload import Payload
from nlql.model.vector import Vector

UnitKind = Literal["document", "chunk", "sentence", "span"]


@dataclass(slots=True)
class Span:
    """Context-window info for ``kind="span"`` units.

    Indices are positions within the parent document's ordered unit sequence
    (of the base granularity that was expanded), inclusive on both ends.
    """

    center: int
    start: int
    end: int
    char_start: int | None = None
    char_end: int | None = None


@dataclass(slots=True)
class Unit:
    """The atomic unit of retrieval and of results.

    Args:
        id: Stable unit id (unique within a store).
        doc_id: Owning document id.
        kind: Granularity of this unit.
        payload: The unit's content.
        metadata: Inherited business metadata from the document (read-only view).
        vector: Embedding, computed at ingestion; ``None`` before embedding.
        span: Present when ``kind == "span"``.
        scores: Named scalar scores attached during query (e.g. ``{"relevance": .87}``).
        ordinal: Position of this unit within its document's sequence, used for
            stable ordering and SPAN window expansion.
    """

    id: str
    doc_id: str
    kind: UnitKind
    payload: Payload
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: Vector | None = None
    span: Span | None = None
    scores: dict[str, float] = field(default_factory=dict)
    ordinal: int = 0
    vectors: dict[str, Vector] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Text content when the payload is textual; ``""`` otherwise."""
        return self.payload.as_text

    def get_vector(self, name: str = "default") -> Vector | None:
        """Return a named vector; ``"default"`` / ``"content"`` map to the primary ``vector``.

        A unit may carry several modality vectors (e.g. a ``text`` and an ``image`` vector),
        queried separately via ``SIMILARITY(vec.<name>, "…")``.
        """
        if name in ("default", "content"):
            return self.vector
        return self.vectors.get(name)
