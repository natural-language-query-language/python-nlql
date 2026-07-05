"""Optional cross-encoder reranker (sentence-transformers).

A cross-encoder scores a ``(query, passage)`` pair *jointly* through a transformer, so it is
far more precise than the bi-encoder vector recall — the standard fix for "a short chunk vs a
long query". Lazy-loaded and optional (``pip install python-nlql[local]``); not exercised in CI (the
deterministic :class:`~nlql.rerank.base.FakeReranker` covers the wiring).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nlql.errors import NLQLError
from nlql.model import Unit


class CrossEncoderReranker:
    """Reranks candidates with a sentence-transformers ``CrossEncoder`` model."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise NLQLError(
                "CrossEncoderReranker needs sentence-transformers: pip install python-nlql[local]"
            ) from e
        self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, units: Sequence[Unit]) -> list[Unit]:
        items = list(units)
        if not items:
            return items
        self._ensure_loaded()
        scores = self._model.predict([(query, u.content) for u in items])
        for unit, score in zip(items, scores, strict=True):
            unit.scores["rerank"] = float(score)
        items.sort(key=lambda u: u.scores.get("rerank", 0.0), reverse=True)
        return items
