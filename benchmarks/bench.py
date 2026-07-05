"""Micro-benchmark + profile for the retrieval path (M6 groundwork).

Run: python benchmarks/bench.py [n_docs]

Uses the deterministic FakeEmbedder so timings reflect *our* code (splitting, the flat-index
matmul, the Python evaluation loop) rather than embedding/network latency. Query embeddings
are cached, so per-query time is pure retrieval. The cProfile output shows where the time
goes — the input to deciding whether an M6 Rust hotspot is worth it.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import random
import sys
import time

from nlql import Document, Engine
from nlql.embed import FakeEmbedder
from nlql.lang import parse
from nlql.store import LocalStore

WORDS = (
    "agent memory tool plan vector search index query language model neural network data "
    "learning system retrieval semantic embedding chunk sentence pipeline store planner"
).split()


def make_docs(n: int, rng: random.Random) -> list[Document]:
    docs = []
    for i in range(n):
        sentences = [
            " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 12))) + "."
            for _ in range(rng.randint(2, 4))
        ]
        docs.append(
            Document.from_text(
                " ".join(sentences),
                id=f"d{i}",
                metadata={"status": rng.choice(["published", "draft"]), "year": rng.randint(2018, 2025)},
            )
        )
    return docs


def percentiles_ms(samples: list[float]) -> tuple[float, float]:
    s = sorted(samples)
    n = len(s)
    return s[int(0.50 * n)] * 1000, s[min(n - 1, int(0.95 * n))] * 1000


def bench(fn, runs: int = 200) -> tuple[float, float]:
    latencies = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        latencies.append(time.perf_counter() - t)
    return percentiles_ms(latencies)


def main() -> None:
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    rng = random.Random(42)
    docs = make_docs(n_docs, rng)

    engine = Engine(FakeEmbedder(dim=384), store=LocalStore())
    t = time.perf_counter()
    engine.add_documents(docs)
    ingest_dt = time.perf_counter() - t
    units = len(engine)
    print(f"ingest: {units} units from {n_docs} docs in {ingest_dt:.2f}s "
          f"= {units / ingest_dt:,.0f} units/s  (dim=384, FakeEmbedder+cache)\n")

    q_topk = 'SELECT SENTENCE LET rel = SIMILARITY(content, "agent memory tool") ORDER BY rel DESC LIMIT 10'
    q_filter = ('SELECT SENTENCE LET rel = SIMILARITY(content, "neural network model") '
                'WHERE meta.status == "published" ORDER BY rel DESC LIMIT 10')
    q_meta = 'SELECT SENTENCE WHERE meta.status == "published" AND meta.year >= 2022 LIMIT 10'
    ir_topk = parse(q_topk)

    print(f"query latency over {units:,} units (P50 / P95 ms):")
    for name, fn in [
        ("semantic top-k (string)", lambda: engine.search(q_topk)),
        ("semantic top-k (IR)", lambda: engine.search(ir_topk)),
        ("semantic + filter", lambda: engine.search(q_filter)),
        ("metadata only", lambda: engine.search(q_meta)),
    ]:
        p50, p95 = bench(fn)
        print(f"  {name:26} P50={p50:6.2f}  P95={p95:6.2f}")

    print("\ncProfile — 300 semantic top-k queries, top cumulative:")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(300):
        engine.search(ir_topk)
    profiler.disable()
    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(10)
    print("\n".join(buf.getvalue().splitlines()[:20]))


if __name__ == "__main__":
    main()
