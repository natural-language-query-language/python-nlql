# Multimodal

## Text and Images in the Same Vector Space

The premise of multimodal retrieval is mapping text and images into the same vector space. Once both live in that space, the vector of a text query can be compared for similarity with the vector of an image — text can retrieve images directly. NLQL's data model is modality-agnostic by design, so this path introduces no new query syntax: an image is just another `Payload`, and an image unit is just a `Unit` carrying an image vector.

```python
import nlql
from nlql.embed import FakeMultimodalEmbedder

mm = nlql.Engine(FakeMultimodalEmbedder(), granularity="chunk")
mm.add_image(b"a photo of a fluffy cat", metadata={"kind": "photo"})
mm.add_text("An article about adopting kittens.", metadata={"kind": "text"})

for unit in mm.search('SELECT CHUNK LET rel = SIMILARITY(content, "fluffy kitten") ORDER BY rel DESC LIMIT 3'):
    print(f"  [{unit.scores['rel']:+.3f}] {unit.doc_id} ({unit.payload.modality.value})")
```

The text query "fluffy kitten" hits both image and text units, ranked by relevance. The query statement is identical to plain text retrieval.

## MultimodalEmbedder

`MultimodalEmbedder` extends a regular `Embedder` with an `embed_images` method. The key point is that text and images share the same vector space — `embed` handles text, `embed_images` handles images, and the vectors both produce can be compared directly via cosine.

```python
from typing import Protocol, runtime_checkable
import numpy as np
from nlql.embed.base import Embedder

@runtime_checkable
class MultimodalEmbedder(Embedder, Protocol):
    def embed_images(self, images: list[bytes | str]) -> np.ndarray:
        """Returns an (n, dim) matrix of unit-normalized image vectors."""
        ...
```

`embed_images` takes a sequence of bytes (or image paths / URLs) and returns vectors of the same dimension and space as `embed`. The engine calls it when writing images and calls `embed` when writing text; both paths ultimately land in the same index.

## Production Multimodal Backends

`FakeMultimodalEmbedder` is an offline, deterministic test double: it embeds image bytes as if they were a text description, with no model or network. In production, swap in a real multimodal model:

- **`DoubaoVisionEmbedder`** — Volcano Ark-hosted `doubao-embedding-vision` (2048-dim), pure HTTP, no torch dependency, suited to cloud deployment.
- **`ClipEmbedder`** — locally-run CLIP (requires `nlql[local]`), text and images in the same space, suited to offline or private deployment.

Switching only requires replacing the embedder in the constructor; query code, indexing paths, and filtering behavior all stay the same.

```python
from nlql.embed import DoubaoVisionEmbedder   # or ClipEmbedder
mm = nlql.Engine(DoubaoVisionEmbedder(), granularity="chunk")
```

## Writing Images

`engine.add_image` is the convenience entry point for writing images, symmetric to `add_text`. It accepts bytes, a path, or a URL, along with metadata:

```python
mm.add_image(image_bytes, metadata={"source": "s3://bucket/cat.jpg", "kind": "photo"})
mm.add_image("/data/photos/dog.png", metadata={"kind": "photo"})
```

!!! tip "Store a reference, not the bytes"
    In production, keep the image body in object storage (S3, OSS) and store only the vector, metadata, and an image reference (URL or key) in the record. Either `Payload(IMAGE, uri_str)` or `metadata["image_url"]` works; the record stays lightweight, and the vector store's retrieval performance is unaffected.

## Queries for Cross-Modal Retrieval

The query statement is no different from text retrieval. `SIMILARITY(content, "text")` applies equally to image units — `content` is a modality-agnostic field path, and the engine uses the vector matching the unit's modality for scoring.

```sql
SELECT CHUNK
LET   rel = SIMILARITY(content, "a red sports car")
WHERE  meta.kind == "photo"
ORDER BY rel DESC
LIMIT 5
```

Metadata filtering works as usual and can be handed off to backends that support native filtering (such as Qdrant and Chroma) during recall — this property holds equally for image records and text records.

## Combining with Vector Databases

Image vectors and text vectors are essentially both vectors, so they can be stored in any backend that supports vector search. Once a multimodal embedder places both in the same space, image units can be stored in Qdrant / Chroma / PgVector and retrieved cross-modally by a text query. It has been verified that under Chroma, a text query can hit image records and that metadata filtering is performed natively by Chroma.

## Next steps

- A runnable example of multimodal writes and retrieval: see `examples/multimodal_search.py` in the repository
- Attaching multiple vectors to one record, each independently queryable: see [Multi-vector Example](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/multivector.py)
- How backends handle metadata filtering: see [Store Interface](./store-protocol.md)
- The embedder protocol and caching: see [Embedder Reference](../../reference/embed.md)
