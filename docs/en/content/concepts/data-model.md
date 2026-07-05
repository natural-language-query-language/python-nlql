# Data model

The NLQL data model consists of four kinds of objects: `Modality` identifies the content type, `Payload` carries the actual content, `Document` is the ingestion unit, and `Unit` is the basic unit of retrieval and return. The model is modality-agnostic by design — text is just one kind of `Payload`.

```python
from nlql import Document, Payload, Unit, Modality
```

## Modality and Payload

`Modality` is an enum, currently with three values: `TEXT`, `IMAGE`, and `BLOB`. `Payload` binds content together with its modality:

```python
from nlql import Payload, Modality

text_payload = Payload.text("a piece of text")    # modality=TEXT, data=str
image_payload = Payload(modality=Modality.IMAGE,
                        data=image_bytes, mime="image/png")
```

`Payload.as_text` returns the string when the content is text, otherwise an empty string. This lets text and image share the same indexing and query path — only the embedding method differs.

## Document

`Document` is the ingestion unit, holding one or more `Payload`s plus business metadata:

```python
doc = Document.from_text(
    "AI agents plan tasks and call external tools.",
    id="doc-0",
    metadata={"status": "published", "topic": "agents"},
)
```

`Document.from_text` is a convenience constructor for a single text payload; a multimodal document can pass multiple payloads. `metadata` is a dict free for the user to use, accessed at query time via `meta.<field>`.

```python
doc = Document(
    id="doc-1",
    payloads=[text_payload, image_payload],
    metadata={"author": "ada"},
)
```

## Unit

At ingestion time, a Document is split into multiple `Unit`s. A `Unit` is the atomic unit of retrieval and return, carrying content, a vector, metadata, and named scores:

| Field | Meaning |
|---|---|
| `kind` | Granularity: `document` / `chunk` / `sentence` / `span` |
| `payload` | The content of this unit |
| `vector` | The embedding computed at ingestion |
| `metadata` | Business metadata inherited from the document |
| `scores` | Named scores attached during the query, e.g. `{"relevance": 0.87}` |
| `span` | Context window information when `kind="span"` |
| `ordinal` | This unit's position index within the document |

`Unit.content` is a convenience accessor for text content; non-text payloads return an empty string. `scores` lets a single query carry multiple semantic scores (for example, computing both `relevance` and `novelty`), so that when sorting by one, the other remains readable.

## Granularity

The `SELECT` clause determines the granularity of query results:

```sql
SELECT SENTENCE       -- return by sentence
SELECT SPAN(SENTENCE, window => 1)   -- a sentence plus one neighbor on each side
SELECT CHUNK          -- return by the chunks split at ingestion
SELECT DOCUMENT       -- the whole document
```

The splitter used at ingestion also serves granularity transformation at query time, so `SPAN` can expand the context window based on stable unit ordinals instead of re-splitting text during the query.

## Ingestion entry points

The Engine provides three ingestion entry points, covering the range from convenience to structured:

```python
engine.add_text("a piece of text", id="doc-0", metadata={"status": "published"})
engine.add(Document.from_text("...", id="doc-1"))
engine.add_documents(iterable_of_documents, batch=256)
```

The ingestion pipeline computes a vector once per Unit and caches it by content hash; repeated content is not re-embedded. System fields (`doc_id`, `kind`, `span`, `scores`) are stored separately from business `metadata` and never pollute each other.

## Multimodal

`MultimodalEmbedder` embeds text and images into the same vector space, so text can retrieve images with the same query statement as text retrieval:

```python
mm = Engine(MultimodalEmbedder(), granularity="chunk")
mm.add_image(image_bytes, metadata={"kind": "photo"})
mm.search('SELECT CHUNK LET rel = SIMILARITY(content, "a cat") ORDER BY rel DESC')
```

The data model makes no distinction between the text and image retrieval paths — the only difference is how the embedder vectorizes content.

!!! note "Named vectors"
    A Unit may carry multiple sets of vectors (the `vectors` dict), for example both a text vector and an image vector. At query time, use `SIMILARITY(vec.<name>, "…")` to specify which set to use.

## Next steps

- [Query syntax](./syntax.md): how to reference `content`, `meta.*`, and granularity in queries
- [Architecture](./architecture.md): the stages of the ingestion pipeline
- [API reference · model](../../reference/model.md): fields of `Document` / `Payload` / `Unit`
- [Multimodal retrieval](../tutorials/multimodal-search.md): a complete example of retrieving images with text
