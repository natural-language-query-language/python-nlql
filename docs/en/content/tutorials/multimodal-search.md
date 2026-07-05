# Multimodal Search

A multimodal embedder maps text and images into the same vector space. Retrieving images with a text query follows exactly the same query path as text retrieval; the only difference is the source of the vectors. This example uses `FakeMultimodalEmbedder` for an offline demonstration.

```python
import nlql
from nlql.embed import FakeMultimodalEmbedder
from nlql.model import Modality, Payload

engine = nlql.Engine(FakeMultimodalEmbedder(), granularity="chunk")
```

`granularity="chunk"` makes each ingested document a single retrieval unit, which suits collections organized per image.

## Ingesting images and text

`Payload(Modality.IMAGE, data)` describes an image payload. Pass image documents together with ordinary text documents to `add_documents`; the engine generates vectors according to each modality.

```python
engine.add_documents(
    [
        nlql.Document(
            id="cat",
            payloads=[Payload(Modality.IMAGE, b"a photo of a fluffy cat")],
            metadata={"kind": "image"},
        ),
        nlql.Document(
            id="car",
            payloads=[Payload(Modality.IMAGE, b"a red sports car on a road")],
            metadata={"kind": "image"},
        ),
        nlql.Document(
            id="dog",
            payloads=[Payload(Modality.IMAGE, b"a happy dog running in a park")],
            metadata={"kind": "image"},
        ),
        nlql.Document.from_text("An article about adopting kittens and cats.", id="article"),
    ]
)
```

You can also ingest a single image with `engine.add_image(bytes_or_path_or_url, id=..., metadata=...)`; the argument accepts raw bytes, a local file path, or an `http(s)` / `data:` URL.

## Retrieving images with text

The query statement is unchanged. `SIMILARITY(content, "…")` works equally on image units under a multimodal embedder.

```python
query = 'SELECT CHUNK LET rel = SIMILARITY(content, "fluffy cat kitten") ORDER BY rel DESC LIMIT 3'
print("text query 'fluffy cat kitten' retrieves across modalities:\n")

for unit in engine.search(query):
    modality = unit.payload.modality.value
    print(f"  [{unit.scores['rel']:+.3f}] ({unit.doc_id}, {modality})")
```

The results span both image and text modalities, ordered by relevance. Each `unit`'s `payload.modality` indicates whether the result came from an image or text.

## Swapping the embedder for production

`FakeMultimodalEmbedder` is for demonstration and testing only. In production, replace it with a real vision embedder:

```python
from nlql.embed import ClipEmbedder       # or DoubaoVisionEmbedder

engine = nlql.Engine(ClipEmbedder(model="openai/clip-vit-base-patch32"),
                     granularity="chunk")
```

- `ClipEmbedder` is based on OpenAI CLIP and requires `pip install "python-nlql[local]"`.
- `DoubaoVisionEmbedder` calls the Doubao vision model API, with the same calling convention as any other OpenAI-compatible embedder.

After swapping, the query and ingestion code remain completely unchanged.

!!! tip "Adding images to a text collection"
    Write images and text into the same engine, and a text query will hit both kinds of content. Use `payload.modality` to distinguish the source in the results.

## Next steps

- [Quick start](./quickstart.md)
- [Two-stage reranking](./reranking.md)
- [Payload and Unit data model](../concepts/overview.md)
- [Embedder API](../../reference/embed.md)

---

**Full source**: [`examples/multimodal_search.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/multimodal_search.py)
