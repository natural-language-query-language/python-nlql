"""Named multi-vectors: one record, several vectors, queried separately.

Run: python examples/multivector.py   (offline)

A product carries a text description (the default vector), plus an ``image`` vector and a
``title`` vector. Query each independently with ``SIMILARITY(vec.<name>, "…")`` — match by
image, by title, or by description. Per-name embedders are configured on the engine (here
Fake ones stand in for CLIP / OpenAI).
"""

from __future__ import annotations

import nlql
from nlql.embed import FakeMultimodalEmbedder


def main() -> None:
    engine = nlql.Engine(
        nlql.FakeEmbedder(),
        granularity="chunk",
        named_embedders={"image": FakeMultimodalEmbedder(), "title": nlql.FakeEmbedder()},
    )
    engine.add_multivector(
        "dress",
        content="a comfortable summer outfit",
        named={"image": b"a photo of a red dress", "title": "Red Summer Dress"},
    )
    engine.add_multivector(
        "car",
        content="a fast vehicle for the open road",
        named={"image": b"a photo of a blue sports car", "title": "Blue Sports Car"},
    )

    for target, query in [
        ("vec.image", "red dress"),
        ("vec.title", "sports car"),
        ("content", "comfortable outfit"),
    ]:
        results = engine.search(
            f'SELECT CHUNK LET rel = SIMILARITY({target}, "{query}") ORDER BY rel DESC LIMIT 1'
        )
        print(f'  SIMILARITY({target}, "{query}")  ->  {results[0].id}  (rel={results[0].scores["rel"]:+.3f})')


if __name__ == "__main__":
    main()
