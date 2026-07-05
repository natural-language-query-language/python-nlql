"""Real semantic search over an OpenAI-compatible embeddings endpoint.

Run:
    export NLQL_OPENAI_API_KEY=sk-...
    export NLQL_OPENAI_BASE_URL=https://api.openai.com/v1   # or your gateway
    python examples/openai_semantic.py

Note the similarity scores are honest raw cosine in [-1, 1] — related text scores clearly
positive, unrelated text near zero — not the reference implementation's opaque (cos+1)/2.
"""

from __future__ import annotations

import os

import nlql


def main() -> None:
    api_key = os.environ.get("NLQL_OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("set NLQL_OPENAI_API_KEY (and optionally NLQL_OPENAI_BASE_URL) to run this")

    engine = nlql.Engine(
        nlql.OpenAIEmbedder(
            api_key=api_key,
            base_url=os.environ.get("NLQL_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model="text-embedding-3-small",
            dimensions=512,
        )
    )
    engine.add_documents(
        [
            nlql.Document.from_text(
                "Machine learning models learn patterns from large datasets.",
                id="ml",
                metadata={"topic": "ai", "status": "published"},
            ),
            nlql.Document.from_text(
                "Neural networks and deep learning power modern AI agents.",
                id="dl",
                metadata={"topic": "ai", "status": "published"},
            ),
            nlql.Document.from_text(
                "Banana bread needs flour, sugar, and forty minutes to bake.",
                id="food",
                metadata={"topic": "cooking", "status": "published"},
            ),
        ]
    )

    results = engine.search(
        """
        SELECT SENTENCE
        LET rel = SIMILARITY(content, "artificial intelligence and neural nets")
        WHERE meta.status == "published"
        ORDER BY rel DESC
        LIMIT 3
        """
    )
    for unit in results:
        print(f"  [{unit.scores['rel']:+.4f}] ({unit.doc_id}) {unit.content}")


if __name__ == "__main__":
    main()
