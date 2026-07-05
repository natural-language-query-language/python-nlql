"""Reranker protocol and a deterministic fake.

A reranker is the **second stage** of retrieval: after the (coarse) vector recall over-fetches
candidates, a reranker re-scores each ``(query, passage)`` pair to reorder them precisely. This
fixes the bi-encoder weakness the vector stage has — a short chunk's embedding vs a long query's
embedding is only a rough match — because a reranker (e.g. a cross-encoder) scores the pair
*jointly* rather than comparing two independent vectors.

Plug your own by implementing :class:`Reranker` and passing it to ``Engine(reranker=...)`` or
``engine.search(..., reranker=...)``. Built-ins: :class:`FakeReranker` (offline, deterministic)
and :class:`~nlql.rerank.cross_encoder.CrossEncoderReranker` (real, optional).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from nlql.model import Unit


@runtime_checkable
class Reranker(Protocol):
    """Re-scores and reorders candidate units against a query."""

    def rerank(self, query: str, units: Sequence[Unit]) -> list[Unit]:
        """Return ``units`` reordered best-first; should set ``unit.scores['rerank']``."""
        ...


class FakeReranker:
    """Deterministic offline reranker: scores by query/passage token overlap.

    A stand-in for a cross-encoder — it re-scores each candidate by how many query tokens the
    passage contains (independent of the embedding), so tests can assert that reranking reorders
    results. No model, no network.
    """

    def rerank(self, query: str, units: Sequence[Unit]) -> list[Unit]:
        query_tokens = set(query.lower().split())
        denom = len(query_tokens) or 1
        scored = list(units)
        for unit in scored:
            passage_tokens = set(unit.content.lower().split())
            unit.scores["rerank"] = len(query_tokens & passage_tokens) / denom
        scored.sort(key=lambda u: u.scores.get("rerank", 0.0), reverse=True)
        return scored
