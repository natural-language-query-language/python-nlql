"""Pipeline protocol: each RAG implementation only differs in retrieval.

ingest(docs) indexes the corpus; retrieve(question, k) returns the top-k chunk
texts that will be fed to the (shared) answer LLM. Keeping generation outside the
pipeline isolates the comparison to retrieval quality alone.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RagPipeline(Protocol):
    name: str

    def ingest(self, docs: list[tuple[str, str, dict]]) -> None: ...

    def retrieve(self, question: dict, k: int = 3) -> list[tuple[str, str]]: ...
