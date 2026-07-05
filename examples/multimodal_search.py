"""Cross-modal retrieval: a text query finds images (M5).

Run: python examples/multimodal_search.py   (offline; uses FakeMultimodalEmbedder)

Text and images share one vector space, so a text query retrieves image units over the same
IR and index path — only the vector's origin differs. Swap in ``nlql.embed.clip.ClipEmbedder``
for real CLIP embeddings over actual pixels.
"""

from __future__ import annotations

import nlql
from nlql.embed import FakeMultimodalEmbedder
from nlql.model import Modality, Payload


def main() -> None:
    engine = nlql.Engine(FakeMultimodalEmbedder(), granularity="chunk")
    engine.add_documents(
        [
            nlql.Document(id="cat", payloads=[Payload(Modality.IMAGE, b"a photo of a fluffy cat")], metadata={"kind": "image"}),
            nlql.Document(id="car", payloads=[Payload(Modality.IMAGE, b"a red sports car on a road")], metadata={"kind": "image"}),
            nlql.Document(id="dog", payloads=[Payload(Modality.IMAGE, b"a happy dog running in a park")], metadata={"kind": "image"}),
            nlql.Document.from_text("An article about adopting kittens and cats.", id="article"),
        ]
    )

    query = 'SELECT CHUNK LET rel = SIMILARITY(content, "fluffy cat kitten") ORDER BY rel DESC LIMIT 3'
    print("text query 'fluffy cat kitten' retrieves across modalities:\n")
    for unit in engine.search(query):
        modality = unit.payload.modality.value
        print(f"  [{unit.scores['rel']:+.3f}] ({unit.doc_id}, {modality})")


if __name__ == "__main__":
    main()
