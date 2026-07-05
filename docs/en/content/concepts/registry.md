# Registry and Extension

NLQL's operators, functions, splitters, and embedders all plug in through a single registration protocol. Once registered, they are usable across all three entry points — NLQL statements, the Query Builder, and LLM IR — without modifying the grammar.

## Unified Registration

Built-in capabilities (`SIMILARITY`, `CONTAINS`, `LENGTH`, and so on) are registered automatically when `nlql` is imported. User extensions go through exactly the same path:

```python
import nlql
from nlql import Signature, TypeTag

@nlql.register_function(
    "WORD_COUNT",
    signature=Signature((TypeTag.TEXT,), TypeTag.NUMBER),
)
def word_count(text: str) -> float:
    return float(len(text.split()))
```

Once registered, `WORD_COUNT` is available just like a built-in:

```sql
SELECT SENTENCE
LET wc = WORD_COUNT(content)
WHERE wc > 10 AND meta.status == "published"
ORDER BY wc DESC LIMIT 5
```

```python
from nlql import QueryBuilder as Q, F
results = engine.search(Q.select("sentence").let("wc", Q.call("WORD_COUNT", Q.content())).where(F("wc") > 10))
```

The `Signature` declared at registration drives argument count and type checks. It can be omitted if you do not care about types.

## Registration Scope

Registration has two levels, corresponding to different visibility.

**Process-level registration** — shared across all engine instances, suited to utility functions and reusable extensions:

```python
import nlql

@nlql.register_function("MY_FN")
def my_fn(text: str) -> float: ...
```

**Instance-level registration** — applies only to the current engine and does not leak to other instances:

```python
engine = nlql.Engine(nlql.embed.FakeEmbedder())

@engine.register_function("TEMP_SCORE")
def temp_score(text: str) -> float: ...
```

Instance-level registration overrides a same-named process-level registration — lookup checks the instance first, then the parent. This lets multiple engines running in the same process keep their extensions isolated.

## Custom Splitters

The ingestion pipeline calls the splitter for the active `granularity` (default `SENTENCE`) to slice a document into units. The default sentence splitter covers Chinese, English, Japanese, and CJK punctuation. When you need more robust segmentation (abbreviations, specialized domains), register your own splitter:

```python
import nlql
import pysbd  # pip install python-nlql[segment]

@nlql.register_splitter("SENTENCE", overwrite=True)
def pysbd_sentences(text: str) -> list[str]:
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)

engine = nlql.Engine(nlql.embed.FakeEmbedder())  # the splitter above is used automatically at ingest
```

The same mechanism uses the splitter at both ingest and query time, so the boundaries returned by `SELECT SENTENCE` / `SELECT SPAN(SENTENCE, window => n)` match those from ingestion — there is no mismatch from re-splitting on the fly at query time.

You can also register a new granularity name (such as `"paragraph"`) and specify it when constructing the engine:

```python
@nlql.register_splitter("PARAGRAPH")
def split_paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]

engine = nlql.Engine(nlql.embed.FakeEmbedder(), granularity="paragraph")
```

## Custom Embedders

Any embedder is an object that implements the `Embedder` protocol. Pass it to `Engine` — that is the only way to construct an engine; there are no channel-specific factory methods.

```python
from nlql import Engine
from nlql.embed.base import BaseEmbedder
import numpy as np

class MyEmbedder(BaseEmbedder):
    @property
    def model_id(self) -> str:
        return "my-model"

    @property
    def dim(self) -> int:
        return 128

    def embed(self, texts):
        # returns a (len(texts), dim) float32 matrix
        return np.random.rand(len(texts), self.dim).astype(np.float32)

engine = Engine(MyEmbedder())
```

OpenAI-compatible endpoints use `OpenAIEmbedder(base_url=..., api_key=...)` directly; for other vendors, write a class that implements the `Embedder` protocol. Multimodal extension only requires additionally implementing `embed_images`, placing images in the same vector space as text.

## Capability Declarations

When registering a function you can attach capability declarations that affect query planning:

- `signature` — argument and return types, used for type checking and argument validation
- `pushdownable` — hints that the function can be translated into a backend's native query (such as a SQL function)
- `provides_score` — marks the function's value as supplied by the recall stage (`SIMILARITY` does this), so it is not invoked per row

```python
@nlql.register_function(
    "LENGTH",
    signature=Signature((TypeTag.TEXT,), TypeTag.NUMBER),
    pushdownable=True,
)
def length(text: str) -> float:
    return float(len(text))
```

These declarations are optional metadata. Without them, the function is evaluated row by row as an ordinary scalar function; with them, the planner decides which parts can be delegated to the backend and which stay in memory.

## Next steps

- For how the ingestion pipeline uses registered splitters, see [Ingestion](./ingestion.md).
- For how custom functions are evaluated within a query, see [Execution Flow](./evaluation.md).
- For the full registration API, see [Registry Reference](../../reference/registry.md).
- For the protocol definition of a custom embedder, see [Embedder Reference](../../reference/embed.md).
