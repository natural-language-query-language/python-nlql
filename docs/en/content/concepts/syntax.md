# Query syntax

NLQL is a declarative query whose structure mirrors SQL: `SELECT` specifies the return granularity, `LET` computes relevance, `WHERE` filters, and `ORDER BY` and `LIMIT` sort and cap. Below is a complete query covering the main clauses:

```sql
SELECT SPAN(SENTENCE, window => 1)
LET   relevance = SIMILARITY(content, "AI agent architecture"),
      novelty   = SIMILARITY(content, "novel unpublished ideas")
WHERE relevance >= 0.8
  AND meta.status != "draft"
  AND (content CONTAINS "planning" OR content CONTAINS "memory")
ORDER BY relevance DESC, novelty DESC
LIMIT 5
```

Line by line:

- **`SELECT SPAN(SENTENCE, window => 1)`** — use the sentence as the base granularity; each result carries one neighbor sentence on each side as a context window
- **`LET relevance = SIMILARITY(...)`** — compute the cosine similarity between the query text and the unit content, bound to the name `relevance`
- **`LET novelty = ...`** — a single query may declare multiple named scores, independent of each other
- **`WHERE relevance >= 0.8`** — filter by score, referencing the name bound above
- **`meta.status != "draft"`** — filter by business metadata
- **`content CONTAINS "planning"`** — string containment test, with an explicit subject
- **`ORDER BY relevance DESC, novelty DESC`** — sort by scores, multiple fields allowed
- **`LIMIT 5`** — return at most 5 rows

## Clause overview

| Clause | Purpose | Example |
|---|---|---|
| `SELECT` | Return granularity | `SELECT SENTENCE` / `SELECT CHUNK` / `SELECT SPAN(SENTENCE, window => 1)` |
| `LET` | Declare a named score or computed value | `LET rel = SIMILARITY(content, "x")` |
| `WHERE` | Filter condition | `WHERE rel >= 0.8 AND meta.status == "published"` |
| `ORDER BY` | Sort | `ORDER BY rel DESC` |
| `LIMIT` | Cap | `LIMIT 5` |

## Field references

Fields are referenced uniformly with dot paths in queries:

- **`content`** — the text content of the unit
- **`meta.<field>`** — the document's business metadata, e.g. `meta.status`, `meta.created_at`
- **`vec.<vector>`** — a named vector (multimodal or multi-vector scenarios), e.g. `vec.image`
- **`relevance`**, etc. — names bound by `LET`, reusable in `WHERE` and `ORDER BY`

Metadata uses dot paths; there is no longer a dedicated `META("x")` function form. `content` and `meta.status` are the same kind of path expression syntactically.

## Functions

Function calls are uniformly written as `NAME(args)` and return a scalar value. Common built-in functions:

| Function | Returns | Example |
|---|---|---|
| `SIMILARITY(content, "query")` | number | relevance score, computed once during recall |
| `LENGTH(content)` | number | `LENGTH(content)` returns the character length |
| `COUNT(content, "word")` | number | number of substring occurrences |
| `CONTAINS(content, "x")` | boolean | `content CONTAINS "x"` (infix form) |

Boolean-returning operators like `CONTAINS`, `MATCH`, and `LIKE` are written in infix form only as syntactic sugar; after parsing they are equivalent to function calls. New functions can be registered via the Registry and used in queries without changing the grammar.

!!! note "Keyword case"
    Structural keywords (`SELECT` / `WHERE` / `LET` / `ORDER BY` / `AND` / `OR` / `NOT`) must be uppercase and are case-sensitive. This distinguishes them from lowercase metadata keys (such as `meta.status`, `meta.window`) and avoids conflicts.

## Predicates and logical combinations

Predicates support common comparisons and infix forms:

```sql
WHERE relevance >= 0.8
  AND meta.year > 2023
  AND content CONTAINS "agent"
  AND (meta.topic == "rag" OR meta.topic == "agents")
```

- Comparisons: `==`, `!=`, `>`, `>=`, `<`, `<=`
- Logic: `AND`, `OR`, `NOT`, grouped with parentheses
- Numbers and dates are compared after type coercion (`meta.year > 2023` is a numeric comparison; ISO dates are compared by time)

## Named scores

`LET` lets a single query carry multiple semantic scores and reuse them when sorting:

```sql
LET   relevance = SIMILARITY(content, "AI agents"),
      novelty   = SIMILARITY(content, "novel ideas")
WHERE relevance >= 0.8
ORDER BY relevance DESC, novelty DESC
```

A `SIMILARITY` call with identical arguments is computed only once, whether bound in `LET` or inlined in `WHERE`. The result is read via `unit.scores["relevance"]`.

## Similarity value range

`SIMILARITY` returns the raw cosine value in the range `[-1, 1]`, with no `(cos+1)/2` folding. Thresholds therefore match intuition: highly relevant approaches 1, unrelated approaches 0, and semantically opposite is negative.

## Next steps

- [Three forms](./three-entries.md): rewrite the same query as a Builder or JSON IR
- [Data model](./data-model.md): the field definitions behind `content` and `meta.*`
- [Quick start](../tutorials/quickstart.md): runnable query examples
- [API reference · lang](../../reference/lang.md): parser and grammar details
