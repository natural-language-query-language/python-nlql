"""Real cross-modal search over a hosted vision embedding model (Doubao / Volcengine Ark).

Run:
    export NLQL_ARK_API_KEY=ark-...
    python examples/doubao_vision_search.py

Text and images share one 2048-d space via ``doubao-embedding-vision`` — no torch, no local
model, just HTTP. A text query retrieves real images over the same IR and index path as text.
Images are downloaded here and sent as base64 (the Ark endpoint cannot reach arbitrary URLs).
"""

from __future__ import annotations

import os

import httpx

import nlql
from nlql.embed import DoubaoVisionEmbedder
from nlql.model import Modality, Payload

# Stable, labelled sample images (PyTorch / YOLO tutorials).
IMAGES = {
    "dog": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "bus": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "players": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
}
QUERIES = ["a dog", "a bus on the street", "football players on a field"]


def main() -> None:
    if not os.environ.get("NLQL_ARK_API_KEY"):
        raise SystemExit("set NLQL_ARK_API_KEY to run this example")

    downloader = httpx.Client(timeout=30, follow_redirects=True)
    engine = nlql.Engine(
        DoubaoVisionEmbedder(base_url="https://ark.cn-beijing.volces.com/api/v3"),
        granularity="chunk",
    )
    for label, url in IMAGES.items():
        data = downloader.get(url).content
        engine.add(nlql.Document(id=label, payloads=[Payload(Modality.IMAGE, data)], metadata={"kind": "image"}))
    print(f"indexed {len(engine)} real image units\n")

    for query in QUERIES:
        results = engine.search(
            f'SELECT CHUNK LET rel = SIMILARITY(content, "{query}") ORDER BY rel DESC LIMIT 3'
        )
        ranked = [(u.doc_id, round(u.scores["rel"], 3)) for u in results]
        print(f"  {query!r:34} -> {ranked}")


if __name__ == "__main__":
    main()
