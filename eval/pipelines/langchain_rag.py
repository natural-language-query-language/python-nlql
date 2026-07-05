"""LangChain RAG pipeline (reference baseline).

Uses langchain-chroma + langchain-openai, pointed at the same embedding endpoint
as NLQL so the only difference is the retriever.

Capability note: LangChain's standard retriever exposes equality metadata filters
via `similarity_search(filter=...)` but has no native range query (date >= / <=)
and no lexical CONTAINS. For date/keyword/composite scenarios we therefore fall
back to the best the retriever can do (plain similarity search, optionally with
the equality status filter), which is exactly the capability gap this eval is
designed to surface.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ..llm import API_KEY, BASE_URL, EMBED_MODEL


class LangChainRag:
    name = "LangChain"

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL, base_url=BASE_URL, api_key=API_KEY
        )
        self.store: Chroma | None = None

    def ingest(self, docs: list[tuple[str, str, dict]]) -> None:
        texts = [t for _, t, _ in docs]
        metadatas = [{"doc_id": did, **m} for did, _, m in docs]
        self.store = Chroma.from_texts(texts, self.embeddings, metadatas=metadatas)

    def retrieve(self, question: dict, k: int = 3) -> list[tuple[str, str]]:
        assert self.store is not None, "ingest() not called"
        q = question["q"]
        if question.get("filter"):
            docs = self.store.similarity_search(q, k=k, filter=question["filter"])
        else:
            docs = self.store.similarity_search(q, k=k)
        return [(d.metadata.get("doc_id", ""), d.page_content) for d in docs]
