# Benchmarks & the M6 (Rust) decision

Run: `python benchmarks/bench.py [n_docs]` (uses `FakeEmbedder`, so timings measure *our*
retrieval code — splitting, the flat-index matmul, the evaluation loop — not embedding).

## Numbers (15,097 units from 5,000 docs, dim=384, Windows / Python 3.14)

| stage | result |
|---|---|
| ingest | ~2,000 units/s (embed+cache, split, index) |
| **semantic top-k** (P50) | **1.6–2 ms** (flat-index matmul + argsort) |
| semantic + metadata filter | 83 ms → 41 ms (coercion fast-reject) → **4.4 ms** (columnar) |
| metadata-only | 98 ms → 45 ms → **6.2 ms** (columnar) |

Two profile-driven optimizations landed: (1) a fast-reject in `as_number`/`as_date` (don't
parse-and-fail non-numeric/date strings) — halved filtered latency; (2) **vectorized column
filtering** — a `MetadataColumns` numpy mask replacing the per-row Python loop, coercing each
*distinct* field value once. Together: filtered queries **83 → 4.4 ms (~19×)**, and now in the
same ballpark as pure semantic top-k. Results stay identical to the per-row path (a test
asserts `column mask == matches_filter`).

## What the profiler showed

- **Semantic top-k is already fast** and dominated by the vectorized flat-index matmul +
  `argsort` (numpy). At 15k units that is ~1.6 ms; it grows O(N), so million-scale wants an
  ANN index — which is exactly what the optional `FaissStore` / a future hnswlib backend
  provide behind the same `Store` protocol. **No Rust needed.**
- **Filtered / metadata queries were 40–50× slower**, and the hotspot was *not* vector math —
  it was the **per-row Python filter loop**, dominated by value coercion (`as_number` /
  `as_date`) run on every operand of every row. `datetime.fromisoformat` alone was called
  ~3M times, mostly failing on non-date strings like `"published"`.
- **Acting on that**: a one-line fast-reject in `as_number` / `as_date` (skip the
  parse-and-fail when a string can't be a number/date) **halved** filtered-query latency
  (83→41 ms, 98→45 ms) with zero semantic change. That is the profile paying for itself.

## M6 decision: defer Rust — Python has the bigger wins first

The DESIGN gates Rust on profiling. The data says **do not write Rust yet** — the Python-level
wins came first:

1. ✅ **Columnar / vectorized metadata filtering** (`store/columns.py`): metadata as numpy
   columns, `meta.status == "published"` as a boolean mask, coercion once per distinct value.
   Delivered the ~10× above. **Done.**
2. **ANN backend (hnswlib)** behind the `Store` protocol for sub-linear semantic recall at
   million-scale (Faiss already available) — the next lever once N ≫ 100k.
3. **One-time typed coercion at ingestion** for declared `field_types` — minor, optional.

Only after those, if the expression kernel is still hot, is a `PyO3`/`maturin` hotspot worth
it — and the columnar approach may already make it unnecessary. Interfaces are unchanged (the
`Store` / evaluator boundaries), so each is a drop-in.
