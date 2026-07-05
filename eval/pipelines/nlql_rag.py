"""NLQL RAG pipeline.

Uses the hand-written NLQL per question (see datasets.QUESTIONS). The metadata
'date' field is declared as DATE so range comparisons (>=, <=) sort chronologically.
"""

from __future__ import annotations

import nlql
from nlql import TypeTag

from ..llm import make_nlql_embedder


class NlqlRag:
    name = "NLQL"

    def __init__(self) -> None:
        self.engine = nlql.Engine(
            make_nlql_embedder(),
            field_types={"date": TypeTag.DATE},
        )

    def ingest(self, docs: list[tuple[str, str, dict]]) -> None:
        for doc_id, text, meta in docs:
            self.engine.add_text(text, id=doc_id, metadata=meta)

    def retrieve(self, question: dict, k: int = 3) -> list[tuple[str, str]]:
        units = self.engine.search(question["nlql"])
        return [(u.doc_id, u.content) for u in units[:k]]
