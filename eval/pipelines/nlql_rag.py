"""NLQL RAG pipeline.

Uses the hand-written NLQL per question (see datasets.QUESTIONS). The metadata
'date' field is declared as DATE so range comparisons sort chronologically.
Embeddings persist across runs via a disk-backed EmbeddingCache (the MS MARCO
corpus is fixed — only the first run embeds it).
"""

from __future__ import annotations

import os

import nlql
from nlql import TypeTag
from nlql.embed import EmbeddingCache

from ..llm import make_nlql_embedder

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "nlql_embeddings.npz"
)


class NlqlRag:
    name = "NLQL"

    def __init__(self, granularity: str = "sentence") -> None:
        # SELECT <unit> must match the ingest granularity; store it for retrieve_semantic.
        self._unit = granularity.upper()
        self._cache = EmbeddingCache()
        if os.path.exists(_CACHE_PATH):
            self._cache.load(_CACHE_PATH)
        self.engine = nlql.Engine(
            make_nlql_embedder(),
            field_types={"date": TypeTag.DATE},
            granularity=granularity,
            cache=self._cache,
        )

    def ingest(self, docs: list[tuple[str, str, dict]]) -> None:
        from nlql.model import Document

        items = [Document.from_text(t, id=i, metadata=m) for i, t, m in docs]
        BATCH = 64  # text-embedding-3-small has 8K context; larger batch = fewer calls
        for i in range(0, len(items), BATCH):
            self.engine.add_documents(items[i : i + BATCH])
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        self._cache.save(_CACHE_PATH)

    def retrieve(self, question: dict, k: int = 3) -> list[tuple[str, str]]:
        units = self.engine.search(question["nlql"])
        return [(u.doc_id, u.content) for u in units[:k]]

    def retrieve_semantic(self, query: str, k: int = 10) -> list[str]:
        q = query.replace('"', " ").replace("\n", " ").strip()
        nlql = (
            f'SELECT {self._unit} LET rel = SIMILARITY(content, "{q}") '
            f"ORDER BY rel DESC LIMIT {k}"
        )
        units = self.engine.search(nlql)
        return [u.doc_id for u in units[:k]]
