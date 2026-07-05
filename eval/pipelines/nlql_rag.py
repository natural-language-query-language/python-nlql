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
        if granularity == "document":
            # NLQL has no built-in document splitter; register an identity one so a
            # whole document becomes a single unit (matches LangChain's passage store,
            # avoids chunk-split embedding losing context on long docs).
            from nlql.registry import GLOBAL_REGISTRY

            GLOBAL_REGISTRY.register(
                "splitter", "document",
                lambda t: [t] if t.strip() else [],
                doc="whole-document unit (identity, no split)",
                overwrite=True,
            )
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
        # Over-fetch (3x) then deduplicate by doc_id: long documents get split into
        # multiple chunks and would otherwise occupy several top-k slots with the
        # same doc_id, starving the result of unique documents.
        nlql = (
            f'SELECT {self._unit} LET rel = SIMILARITY(content, "{q}") '
            f"ORDER BY rel DESC LIMIT {k * 3}"
        )
        units = self.engine.search(nlql)
        seen: set[str] = set()
        unique: list[str] = []
        for u in units:
            if u.doc_id not in seen:
                seen.add(u.doc_id)
                unique.append(u.doc_id)
            if len(unique) >= k:
                break
        return unique

    def llm_retrieve(self, question_text: str, k: int = 3) -> list[tuple[str, str]]:
        """LLM-driven retrieval: the model writes an NLQL query string (SQL-like,
        far shorter and more controllable than nested IR JSON) and the lark parser
        executes it. Parse errors → empty result (counts as a miss).
        """
        from ..llm import chat

        prompt = (
            "Write an NLQL query to retrieve the answer. Reply with ONLY the query.\n\n"
            "NLQL syntax (SQL-like):\n"
            "  SELECT SENTENCE|CHUNK|DOCUMENT [SPAN(UNIT, window => N)]\n"
            "  LET name = SIMILARITY(content, \"semantic query\")\n"
            "  WHERE <conditions>: meta.field == \"v\" | meta.date >= \"2024-01-01\" |\n"
            "    content CONTAINS \"keyword\" | AND / OR / NOT | ==, !=, >=, <=\n"
            "  ORDER BY name DESC\n"
            "  LIMIT n\n\n"
            "Functions: SIMILARITY(content, \"...\") = semantic relevance;\n"
            "  CONTAINS = literal substring.\n"
            "Metadata: status (published/draft), date (YYYY-MM-DD), category,\n"
            "  priority (high/medium/low), done (true/false).\n\n"
            "Examples:\n"
            "  SELECT SENTENCE LET rel = SIMILARITY(content, \"AI agents\") WHERE meta.status == \"published\" ORDER BY rel DESC LIMIT 3\n"
            "  SELECT SENTENCE WHERE content CONTAINS \"transformer\" LIMIT 5\n"
            "  SELECT SENTENCE LET rel = SIMILARITY(content, \"RAG\") WHERE meta.status != \"draft\" ORDER BY rel DESC LIMIT 5\n\n"
            f"Question: {question_text}"
        )
        raw = chat(prompt).strip().strip("`")
        try:
            units = self.engine.search(raw)
        except Exception:
            return []
        return [(u.doc_id, u.content) for u in units[:k]]
