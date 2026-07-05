# Ingestion

When a document enters NLQL it passes through four stages in order: normalize, split, embed, and index. These all happen at ingestion time; the query stage reuses the results directly.

```python
import nlql

engine = nlql.Engine(nlql.embed.OpenAIEmbedder(base_url="...", api_key="..."))
doc_id = engine.add_text(
    "AI agents plan tasks. They keep memory and call external tools.",
    metadata={"status": "published", "year": 2026},
)
results = engine.search(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "how agents work") ORDER BY rel DESC LIMIT 3'
)
```

## Four Stages

### 1. Normalize

Before splitting, text is normalized: whitespace and line breaks are unified, and formatting differences that would disturb segmentation are removed. This ensures that the same text, however it is wrapped on input, is split into the same units and hits the same embedding cache.

### 2. Split

The normalized text is sliced into units by the splitter for the active granularity. The default is by sentence (`SENTENCE`), and the built-in splitter covers Chinese, English, Japanese, and CJK punctuation.

```python
engine = nlql.Engine(nlql.embed.FakeEmbedder(), granularity="sentence")  # default
engine = nlql.Engine(nlql.embed.FakeEmbedder(), granularity="chunk")     # use the chunk splitter instead
```

Splitting happens at ingest and is reused at query time — the boundaries returned by `SELECT SENTENCE` and `SELECT SPAN(SENTENCE, window => n)` both come from this stage; there is no on-the-fly re-splitting at query time.

### 3. Embed

Each unit's text is vectorized once by the embedder, then L2-normalized. By default the engine wraps the passed-in embedder with `CachedEmbedder`, so identical text is computed only once (see [Index and Cache](./index-cache.md) for caching details).

### 4. Index

Units with their vectors and metadata are written to the store. The default is the in-process `LocalStore`, but it can also be Qdrant, Chroma, Faiss, HnswLib, or pgvector — switching backends does not change ingestion code.

## Ingestion Entry Points

The engine offers ingestion APIs at four granularities, all routed through the same pipeline internally:

```python
# A single text passage
engine.add_text("A passage. Second sentence.", metadata={"status": "published"})

# A single file (dispatched by extension: .txt / .md / .docx / .pdf)
engine.add_file("notes.md")
engine.add_files(["a.txt", "b.md", "report.pdf"])

# Structured documents (multiple payloads, multimodal, custom ids)
from nlql import Document, Payload
engine.add_documents([
    Document(id="d1", payloads=[Payload.text("...")], metadata={"kind": "note"}),
])
```

`add_file` selects a loader by extension: `.txt` / `.md` use built-in loaders; `.docx` requires `python-nlql[loaders]` (python-docx); `.pdf` requires the same extras (pypdf). The loader parses the file into a `Document`, which then goes through the same pipeline.

```python
import nlql

engine = nlql.Engine(nlql.embed.FakeEmbedder())
ids = engine.add_files(["agents.txt", "rag.md"])
print(f"loaded into {len(engine)} units: {ids}")
```

## Choosing a Granularity

`granularity` determines what kind of unit documents are split into at ingest, and directly shapes retrieval behavior:

- **`sentence`** (default) — one unit per sentence; fine-grained, with precise relevance localization. Suited to Q&A and citation scenarios that need to point at a specific sentence.
- **`chunk`** — splits into larger passages; each unit carries more complete information. Suited to RAG where answers need full contextual paragraphs.
- **Custom granularity** — register your own splitter (see [Registry and Extension](./registry.md)), for example by paragraph or by chapter.

```python
engine = nlql.Engine(nlql.embed.FakeEmbedder(), granularity="chunk")
engine.add_file("long_document.md")
# each chunk is one retrieval unit
```

!!! note "Granularity is fixed at ingest"
    The engine indexes under a single `granularity`. At query time you can request larger return units (`SELECT SPAN(SENTENCE, window => 2)` joins the matching sentence with its neighbors, or `SELECT DOCUMENT` returns the whole document), but you cannot split below the ingested granularity.

## Metadata

`metadata` is a free-form field space for user data: it is attached to the document at ingest and accessed at query time via the `meta.<field>` path, on equal footing with `content`.

```python
engine.add_text(
    "Retrieval-augmented generation grounds LLM answers in your documents.",
    metadata={"status": "published", "year": 2026, "topic": "rag"},
)
# At query time: meta.status == "published" AND meta.year > 2024
```

!!! tip "Declare field types for more accurate comparisons"
    Passing `field_types={"year": TypeTag.DATE}` and similar when constructing the engine lets dates compare as dates and numbers as numbers, avoiding the ambiguity of string comparison.

## Next steps

- For how vectors are cached and reused, see [Index and Cache](./index-cache.md).
- For how to plug in custom splitters, see [Registry and Extension](./registry.md).
- For a complete ingestion and retrieval example, see [Quickstart](../tutorials/quickstart.md).
- For the full signature of the Engine ingestion API, see [SDK Reference](../../reference/sdk.md).
