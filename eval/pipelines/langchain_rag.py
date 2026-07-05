"""LangChain RAG pipeline (reference baseline).

Uses langchain-chroma + langchain-openai, pointed at the same embedding endpoint
as NLQL so the only difference is the retriever. Embeddings persist across runs
via a disk cache (the MS MARCO corpus is fixed).

Capability note: LangChain's standard retriever exposes equality metadata filters
via `similarity_search(filter=...)` but has no native range query (date >= / <=)
and no lexical CONTAINS. For date/keyword/composite scenarios we therefore fall
back to the best the retriever can do, which is exactly the capability gap this
eval is designed to surface.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ..llm import API_KEY, BASE_URL, EMBED_MODEL

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "lc_embeddings"
)


class _CachedEmbeddings:
    """Disk-backed cache wrapping an LC Embeddings object.

    Avoids re-embedding the same fixed corpus on every run (LangChain has no
    built-in disk cache without the heavy `langchain` package, so we roll a
    minimal one keyed by sha1(text)).
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        os.makedirs(_CACHE_DIR, exist_ok=True)

    def _path(self, text: str) -> str:
        return os.path.join(_CACHE_DIR, hashlib.sha1(text.encode("utf-8")).hexdigest() + ".npy")

    def _embed_many(self, texts: list[str], fn) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        miss_i: list[int] = []
        miss_t: list[str] = []
        for i, t in enumerate(texts):
            p = self._path(t)
            if os.path.exists(p):
                out[i] = np.load(p).tolist()
            else:
                miss_i.append(i)
                miss_t.append(t)
        if miss_t:
            vecs = fn(miss_t)  # single batched API call for all misses
            for idx, v in zip(miss_i, vecs):
                arr = np.asarray(v, dtype=np.float32)
                np.save(self._path(miss_t[miss_i.index(idx)] if False else texts[idx]), arr)
                out[idx] = arr.tolist()
        return out  # type: ignore[return-value]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many(texts, self.inner.embed_documents)

    def embed_query(self, text: str) -> list[float]:
        # embed_query takes a single str, not a list — handle it directly.
        p = self._path(text)
        if os.path.exists(p):
            return np.load(p).tolist()
        v = self.inner.embed_query(text)
        np.save(p, np.asarray(v, dtype=np.float32))
        return v


class LangChainRag:
    name = "LangChain"

    def __init__(self) -> None:
        self.embeddings = _CachedEmbeddings(
            OpenAIEmbeddings(
                model=EMBED_MODEL, base_url=BASE_URL, api_key=API_KEY, request_timeout=120
            )
        )
        self.store: Chroma | None = None

    @staticmethod
    def _to_chroma_filter(flt: dict) -> dict:
        # Chroma 'where' allows exactly one operator; multi-key equality must be
        # wrapped in $and.
        if len(flt) <= 1:
            return flt
        return {"$and": [{k: v} for k, v in flt.items()]}

    def ingest(self, docs: list[tuple[str, str, dict]]) -> None:
        texts = [t for _, t, _ in docs]
        metadatas = [{"doc_id": did, **m} for did, _, m in docs]
        self.store = Chroma(embedding_function=self.embeddings)
        BATCH = 64
        for i in range(0, len(texts), BATCH):
            self.store.add_texts(
                texts[i : i + BATCH], metadatas=metadatas[i : i + BATCH]
            )

    def retrieve(self, question: dict, k: int = 3) -> list[tuple[str, str]]:
        assert self.store is not None, "ingest() not called"
        q = question["q"]
        flt = question.get("filter")
        if flt:
            docs = self.store.similarity_search(
                q, k=k, filter=self._to_chroma_filter(flt)
            )
        else:
            docs = self.store.similarity_search(q, k=k)
        return [(d.metadata.get("doc_id", ""), d.page_content) for d in docs]

    def retrieve_semantic(self, query: str, k: int = 10) -> list[str]:
        assert self.store is not None, "ingest() not called"
        docs = self.store.similarity_search(query, k=k)
        return [d.metadata.get("doc_id", "") for d in docs]

    def llm_retrieve(self, question_text: str, k: int = 3) -> list[tuple[str, str]]:
        """LLM-driven retrieval: the model emits {query, filter}; the LangChain
        retriever executes it. Fair counterpart to NlqlRag.llm_retrieve."""
        import json

        from ..llm import chat

        prompt = (
            "Produce a JSON object for retrieval. Reply with ONLY the JSON, no prose.\n"
            'Format: {"query": "<semantic query string>", '
            '"filter": {"<field>": "<value>"} or null}\n'
            "Metadata fields (EQUALITY ONLY — no ranges, no !=, no substring): "
            "status, date (a YYYY-MM-DD string, exact match only), category, priority, done.\n"
            "If the question implies a range, negation, or substring, leave filter null "
            "and put the best semantic query in 'query' (the retriever cannot express those).\n"
            f"Question: {question_text}"
        )
        raw = chat(prompt).strip().strip("`")
        try:
            spec = json.loads(raw)
        except Exception:
            spec = {"query": question_text}
        flt = spec.get("filter") or None
        if flt and len(flt) > 1:
            flt = {"$and": [{kk: vv} for kk, vv in flt.items()]}
        assert self.store is not None
        docs = self.store.similarity_search(spec.get("query", question_text), k=k, filter=flt)
        return [(d.metadata.get("doc_id", ""), d.page_content) for d in docs]
