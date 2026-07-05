# Document Loading

`engine.add_file` and `engine.add_files` select a loader automatically by file extension, convert the file into `Document`s, and ingest them into the index. `.txt` / `.md` / `.docx` / `.pdf` are supported out of the box, and more formats can be added via `register_loader`.

This example uses `FakeEmbedder` for an offline demonstration.

```python
import tempfile
from pathlib import Path
import nlql

tmp = Path(tempfile.mkdtemp())
(tmp / "agents.txt").write_text(
    "AI agents plan tasks. They keep memory and call external tools.", encoding="utf-8"
)
(tmp / "rag.md").write_text(
    "# RAG\n\nRetrieval-augmented generation grounds LLM answers in your documents.",
    encoding="utf-8",
)
files = [tmp / "agents.txt", tmp / "rag.md"]
```

## Loading by extension

`add_files` takes a list of paths and dispatches a loader per file based on its extension. `.txt` and `.md` go through the plain-text loader, which works out of the box.

```python
engine = nlql.Engine(nlql.FakeEmbedder())
ids = engine.add_files([str(f) for f in files])
print(f"loaded {len(ids)} files -> {len(engine)} sentence units: {ids}")
```

The return value is the list of ingested document ids. A single file may produce multiple `Document`s (for example, a PDF page by page), so a list is returned.

## Loading DOCX / PDF

DOCX and PDF depend on additional parsing libraries; install them to enable loading:

```bash
pip install "python-nlql[loaders]"
```

The following snippet adds a `.docx` file when `python-docx` is available, otherwise it skips it:

```python
try:
    import docx

    document = docx.Document()
    document.add_paragraph("Vector databases store embeddings for similarity search.")
    document.save(str(tmp / "vectors.docx"))
    files.append(tmp / "vectors.docx")
except ImportError:
    print("(python-docx not installed — skipping the .docx file)")

engine = nlql.Engine(nlql.FakeEmbedder())
ids = engine.add_files([str(f) for f in files])
```

`add_file` loads a single file and returns the list of document ids for that file:

```python
ids = engine.add_file("report.pdf", metadata={"source": "annual-report"})
```

The PDF loader supports `PdfLoader(extract_images=True)` to extract image payloads from the file.

## Specifying a loader explicitly

`add_file(path, loader=...)` bypasses automatic dispatch and uses a specific loader directly:

```python
from nlql.loaders import PdfLoader

engine.add_file("scan.pdf", loader=PdfLoader(by_page=True))
```

## Registering custom formats

`register_loader` binds a loader implementing the `Loader` protocol (`load(path) -> list[Document]`) to one or more extensions. Importing `nlql.loaders` registers the built-in formats.

```python
from nlql.loaders import register_loader, TextLoader

# Make .log also load as plain text
register_loader(TextLoader(), ".log")

# A custom loader
class EpubLoader:
    def load(self, path):
        ...
        return [nlql.Document.from_text(text, ...)]

register_loader(EpubLoader(), ".epub")
```

After registration, `engine.add_file("book.epub")` goes through your custom logic.

!!! note "DOCX / PDF require extras"
    Plain-text formats need no extra dependencies; install `python-nlql[loaders]` before loading `.docx` and `.pdf`, otherwise an `ImportError` is raised.

## Querying

Once ingestion is complete, querying is the same as the other entry points:

```python
results = engine.search(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "how do agents use tools") '
    "ORDER BY rel DESC LIMIT 3"
)
for unit in results:
    print(f"  ({unit.doc_id}) {unit.content}")
```

## Next steps

- [Quick start](./quickstart.md)
- [Multimodal search](./multimodal-search.md)
- [Document and Unit data model](../concepts/overview.md)
- [Loader API](../../reference/loaders.md)

---

**Full source**: [`examples/document_loading.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/document_loading.py)
