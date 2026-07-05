# Field Types

## Giving Comparisons a Type

Metadata field values are usually stored in `metadata` as strings, but when compared they are often something else: dates, numbers, booleans. If the engine is not told a field's type, it can only guess — and a wrong guess yields wrong ordering.

`field_types` declares the type of each metadata field when constructing the `Engine`:

```python
import nlql
from nlql import TypeTag

engine = nlql.Engine(
    embedder,
    field_types={
        "published": TypeTag.DATE,
        "views":     TypeTag.NUMBER,
        "code":      TypeTag.TEXT,
        "featured":  TypeTag.BOOL,
    },
)
```

Once declared, comparisons undergo type coercion before execution: dates are compared as dates, numbers as numbers, text as strings. Comparison semantics are consistent across the built-in store and all external backends.

## The TypeTag Enum

`TypeTag` is a string enum covering the value types that flow through a query:

```python
from enum import StrEnum

class TypeTag(StrEnum):
    ANY = "any"
    TEXT = "text"
    NUMBER = "number"
    BOOL = "bool"
    DATE = "date"
    VECTOR = "vector"
    NULL = "null"
```

Beyond driving metadata comparisons, `TypeTag` is also used for signature validation of registered functions (arity and argument types) and for passing correctly typed filter values to external backends. A `Signature((TypeTag.TEXT,), TypeTag.NUMBER)` describes a function that "takes one text and returns a number".

## Why Type Declarations Are Needed

Consider a `published` field storing the ISO date string `"2024-03-01"`. Without a type declaration:

```sql
WHERE meta.published > "2024-01-01"
```

String comparison is lexicographic. `"2024-03-01" > "2024-01-01"` happens to hold lexicographically, but the moment the date format becomes `"2024-3-1"` (no leading zeros), lexicographic order breaks down — `"2024-3-1"` would sort after `"2024-12-01"`.

Declare `TypeTag.DATE` and both sides are coerced to `datetime` before comparison, regardless of format:

```python
engine = nlql.Engine(embedder, field_types={"published": TypeTag.DATE})
# meta.published > "2024-01-01"  →  datetime comparison, result is correct
```

The same applies to numeric fields. For `meta.views > 1000`, if `views` is stored as the string `"999"`, string comparison would yield `"999" > "1000"` (lexicographic), but numeric comparison would not.

## TEXT Suppresses Numeric Coercion

Type declarations do not only "add" types; sometimes they "block" a wrong automatic coercion. A product code, a postal code, or a version number may look like a number but should be compared as a string.

```python
engine_a = nlql.Engine(embedder)                      # no declaration, auto-inferred
engine_b = nlql.Engine(embedder, field_types={"zip": TypeTag.TEXT})
```

| Field value | Result of `meta.zip > "02100"` | Reason |
|---|---|---|
| `"02134"` | Engine A: `False` (2134 is not greater than 2100) | auto-inferred as numeric |
| `"02134"` | Engine B: `True` (lexicographic `"02134" > "02100"`) | forced string comparison |

`TypeTag.TEXT` tells the engine explicitly: do not treat this field as a number. This matters for fields like ID card numbers, order numbers, and SKUs.

## Actual Behavior of Coercion

Before comparison, the engine coerces the operands according to the declared type; if coercion fails (for example, `DATE` declared on a non-date string), it falls back to comparing the raw values. A `null` participating in an ordered comparison causes the row to be excluded (SQL semantics); in `==` / `!=` it is judged by equality.

```python
from nlql.types.coerce import compare_values
from nlql import TypeTag

compare_values(">", "2024-03-01", "2024-01-01", hint=TypeTag.DATE)  # True
compare_values(">", "02134", "02100", hint=TypeTag.TEXT)            # True
```

This comparison logic is shared by the built-in store and external backends, so the same `WHERE` behaves consistently across backends.

!!! tip "The more you declare, the more predictable the behavior"
    Only fields that get compared need a type declared. Tagging every metadata field that appears in `WHERE` / `ORDER BY` with a type keeps query results stable across backends and data formats. Undeclared fields fall back to value inference: two values that both look numeric are compared numerically, otherwise as strings.

## Relationship with Backend Filtering

Declared-type fields still take part in metadata filtering. The engine passes the type information along to the backend, so external backends translate filter conditions using the correct type (for example, pgvector needs to know whether a column is a date or a number). Which conditions can be handed off to the backend is decided by the backend's `StoreCaps`. See [Store Interface](./store-protocol.md).

## Next steps

- How types affect where filtering happens: see [Store Interface](./store-protocol.md)
- How fields are written in queries: see [Quick Start](../tutorials/quickstart.md)
- Full signatures of `TypeTag` and `Signature`: see [Types Reference](../../reference/types.md)
