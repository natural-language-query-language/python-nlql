"""Quickstart: ingest text and run NLQL queries three equivalent ways.

Run: python examples/quickstart.py
Uses the deterministic FakeEmbedder, so it needs no network or model download.
"""

from __future__ import annotations

import nlql
from nlql.sdk.builder import F, Meta, select, similarity

DOCS = [
    ("AI agents plan tasks, keep memory, and call external tools.", {"status": "published", "topic": "agents"}),
    ("Retrieval-augmented generation grounds LLM answers in your documents.", {"status": "published", "topic": "rag"}),
    ("Banana bread needs flour, sugar, and about forty minutes to bake.", {"status": "draft", "topic": "cooking"}),
]


def main() -> None:
    engine = nlql.Engine(nlql.FakeEmbedder())
    for i, (text, meta) in enumerate(DOCS):
        engine.add_text(text, id=f"doc-{i}", metadata=meta)
    print(f"indexed {len(engine)} sentence units\n")

    query = """
        SELECT SENTENCE
        LET relevance = SIMILARITY(content, "autonomous agents and tools")
        WHERE meta.status == "published"
        ORDER BY relevance DESC
        LIMIT 3
    """

    print("== via NLQL string ==")
    for unit in engine.search(query):
        print(f"  [{unit.scores['relevance']:+.3f}] {unit.content}")

    print("\n== via Query Builder (equivalent) ==")
    built = (
        select("sentence")
        .let("relevance", similarity("content", "autonomous agents and tools"))
        .where(Meta("status") == "published")
        .order_by("relevance", desc=True)
        .limit(3)
        .build()
    )
    for unit in engine.search(built):
        print(f"  [{unit.scores['relevance']:+.3f}] {unit.content}")

    print("\n== EXPLAIN ==")
    import json

    print(json.dumps(engine.explain(query), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
