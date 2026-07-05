"""eval v2 — MS MARCO semantic recall + constructed filter scenarios.

Two independent evaluations, run concurrently via ThreadPoolExecutor:

  1. MS MARCO passage retrieval — REAL Bing queries with judged relevant
     passages, no metadata. Both pipelines share the same embedding and neither
     can use filters, so this is the dimension where NLQL is NOT inherently
     favored. Metrics: recall@10, MRR.

  2. Constructed scenarios (with metadata) — end-to-end answer accuracy judged
     by a PANEL (averaged across qwen3.7-max + gpt-5.4 + minimax-m3 to cancel
     single-model noise) + recall + precision.

Run from the repo root:

    export NLQL_EVAL_API_KEY=...
    python -m eval.run
"""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.data.datasets import DOCUMENTS, QUESTIONS, SCENARIOS  # noqa: E402
from eval.data.ms_marco import load_subset  # noqa: E402
from eval.judge import judge_answer_multi  # noqa: E402
from eval.llm import (  # noqa: E402
    BASE_URL,
    CHAT_MODEL,
    EMBED_MODEL,
    JUDGE_MODELS,
    chat,
)
from eval.pipelines.langchain_rag import LangChainRag  # noqa: E402
from eval.pipelines.nlql_rag import NlqlRag  # noqa: E402

PIPELINE_NAMES = ["NLQL", "LangChain"]
WORKERS = int(os.environ.get("NLQL_EVAL_WORKERS", "8"))

ANSWER_PROMPT = """Answer the question using ONLY the context below. Be concise (one or two sentences). If the context does not support an answer, say "insufficient context".

Question: {q}

Context:
{ctx}

Answer:"""


def answer(q: str, ctxs: list[tuple[str, str]]) -> str:
    ctx = "\n".join(f"- {t}" for _, t in ctxs) or "(no context retrieved)"
    return chat(ANSWER_PROMPT.format(q=q, ctx=ctx))


# --- retrieval metrics ---------------------------------------------------------


def hit_rate(expected: list[str], retrieved_ids: list[str]) -> float:
    if not expected:
        return 0.0
    got = set(retrieved_ids)
    return sum(1 for e in expected if e in got) / len(expected)


def recall_at_k(gold: list[str], retrieved: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return sum(1 for g in gold if g in retrieved[:k]) / len(gold)


def mrr(gold: list[str], retrieved: list[str]) -> float:
    for i, rid in enumerate(retrieved, 1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def _contains_term(question: dict) -> str | None:
    m = re.search(r'CONTAINS "([^"]+)"', question.get("nlql", ""))
    return m.group(1) if m else None


def precision(retrieved_ids: list[str], question: dict, docs_by_id: dict) -> float:
    if not retrieved_ids:
        return 0.0
    flt = question.get("filter") or {}
    not_flt = question.get("filter_not") or {}
    gte = question.get("filter_date_gte")
    lte = question.get("filter_date_lte")
    contains = _contains_term(question)

    def ok(doc_id: str) -> bool:
        if doc_id not in docs_by_id:
            return False
        _, text, meta = docs_by_id[doc_id]
        for k, v in flt.items():
            if meta.get(k) != v:
                return False
        for k, v in not_flt.items():
            if meta.get(k) == v:
                return False
        if gte and meta.get("date", "") < gte:
            return False
        if lte and meta.get("date", "") > lte:
            return False
        if contains and contains.lower() not in text.lower():
            return False
        return True

    return sum(1 for rid in retrieved_ids if ok(rid)) / len(retrieved_ids)


# --- evaluation 1: MS MARCO ----------------------------------------------------


def run_msmarco() -> tuple[dict, int, int, list[dict]]:
    print("loading MS MARCO...")
    corpus, queries = load_subset()
    print(f"  {len(corpus)} passages, {len(queries)} judged queries")

    # NLQL uses chunk granularity so each MS MARCO passage is one retrieval unit
    # (CHUNK splitter merges up to 1000 chars → whole passage). This aligns with
    # LangChain, which stores whole passages — otherwise NLQL would retrieve
    # sentence fragments and lose recall to duplicate-doc_id slot occupation.
    pipelines = [NlqlRag(granularity="chunk"), LangChainRag()]
    t_ingest = time.time()
    for p in pipelines:
        p.ingest(corpus)
        print(f"  [{time.strftime('%H:%M:%S')}] {p.name}: corpus ingested ({time.time() - t_ingest:.0f}s)")

    def one(q: dict) -> dict:
        out: dict = {"q": q["q"][:60], "gold": q["gold"]}
        for p in pipelines:
            ids = p.retrieve_semantic(q["q"], k=10)
            out[p.name] = {
                "recall@10": recall_at_k(q["gold"], ids, 10),
                "mrr": mrr(q["gold"], ids),
            }
        return out

    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, q) for q in queries]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 10 == 0:
                print(f"  msmarco {i}/{len(queries)} ({time.time() - t0:.0f}s)")

    agg = {
        name: {
            "recall@10": mean(r[name]["recall@10"] for r in rows),
            "mrr": mean(r[name]["mrr"] for r in rows),
        }
        for name in PIPELINE_NAMES
    }
    return agg, len(corpus), len(queries), rows


# --- evaluation 2: constructed scenarios ---------------------------------------


def run_scenarios() -> list[dict]:
    docs_by_id = {d[0]: d for d in DOCUMENTS}
    pipelines = []
    for cls in (NlqlRag, LangChainRag):
        p = cls()
        p.ingest(DOCUMENTS)
        pipelines.append(p)
    print(f"  {len(DOCUMENTS)} docs ingested by both pipelines; judges panel = {JUDGE_MODELS}")

    def one(q: dict) -> dict:
        row = {"scenario": q["scenario"], "q": q["q"], "expected": q["expected"]}
        for p in pipelines:
            retrieved = p.retrieve(q)
            ans = answer(q["q"], retrieved)
            score, breakdown = judge_answer_multi(
                q["q"], q["points"], ans, JUDGE_MODELS
            )
            retrieved_ids = [rid for rid, _ in retrieved]
            row[p.name] = {
                "score": score,
                "hit": hit_rate(q["expected"], retrieved_ids),
                "precision": precision(retrieved_ids, q, docs_by_id),
                "judges": breakdown,
                "retrieved": retrieved_ids,
                "answer": ans.replace("\n", " ")[:140],
            }
        return row

    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, q) for q in QUESTIONS]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            print(f"  scenario {i}/{len(QUESTIONS)} ({time.time() - t0:.0f}s)")
    rows.sort(key=lambda r: (SCENARIOS.index(r["scenario"]), r["q"]))
    return rows


# --- report --------------------------------------------------------------------


def write_report(msmarco: tuple[dict, int, int, list[dict]], scen_rows: list[dict]) -> None:
    agg, n_corpus, n_q, _ = msmarco
    out: list[str] = []
    out.append("# NLQL vs LangChain — retrieval eval v2\n")
    out.append(
        f"Embedding `{EMBED_MODEL}` · answer `{CHAT_MODEL}` · "
        f"judge panel `{', '.join(JUDGE_MODELS)}` (averaged) · via `{BASE_URL}`.\n"
    )
    out.append(
        "Both pipelines share the same embedding and answer LLM — only the retriever differs. "
        "Scenario scores are the **panel average** to cancel single-judge noise.\n"
    )

    out.append("\n## 1. MS MARCO — real Bing queries, pure semantic recall\n")
    out.append(
        f"Corpus: {n_corpus} deduped passages · {n_q} queries with judged relevant passages. "
        "Passages carry no metadata, so neither pipeline can use filters — this is the dimension "
        "where **NLQL is not inherently favored** (keeps the benchmark honest).\n"
    )
    out.append("| pipeline | recall@10 | MRR |")
    out.append("|---|---|---|")
    for name in PIPELINE_NAMES:
        a = agg[name]
        out.append(f"| {name} | {a['recall@10']:.1%} | {a['mrr']:.3f} |")

    out.append("\n## 2. Constructed scenarios — filter-aware, end-to-end\n")
    out.append("| scenario | NLQL score | NLQL hit | NLQL prec | LC score | LC hit | LC prec |")
    out.append("|---|---|---|---|---|---|---|")

    def scen_avg(scen: str, name: str, key: str) -> float:
        sr = [r for r in scen_rows if r["scenario"] == scen]
        return mean(r[name][key] for r in sr) if sr else 0.0

    for s in SCENARIOS:
        if not any(r["scenario"] == s for r in scen_rows):
            continue
        out.append(
            f"| {s} | {scen_avg(s, 'NLQL', 'score'):.2f} | {scen_avg(s, 'NLQL', 'hit'):.0%} | "
            f"{scen_avg(s, 'NLQL', 'precision'):.0%} | {scen_avg(s, 'LangChain', 'score'):.2f} | "
            f"{scen_avg(s, 'LangChain', 'hit'):.0%} | {scen_avg(s, 'LangChain', 'precision'):.0%} |"
        )
    out.append(
        "| **overall** | {:.2f} | {:.0%} | {:.0%} | {:.2f} | {:.0%} | {:.0%} |".format(
            mean(r["NLQL"]["score"] for r in scen_rows),
            mean(r["NLQL"]["hit"] for r in scen_rows),
            mean(r["NLQL"]["precision"] for r in scen_rows),
            mean(r["LangChain"]["score"] for r in scen_rows),
            mean(r["LangChain"]["hit"] for r in scen_rows),
            mean(r["LangChain"]["precision"] for r in scen_rows),
        )
    )

    out.append("\n## 3. Per-question (panel-averaged score)\n")
    out.append("| # | scenario | question | NLQL (score/hit/prec) | LangChain (score/hit/prec) |")
    out.append("|---|---|---|---|---|")
    for i, r in enumerate(scen_rows, 1):
        n = r["NLQL"]
        lc = r["LangChain"]
        out.append(
            f"| {i} | {r['scenario']} | {r['q'][:46]} | "
            f"{n['score']:.1f} / {n['hit']:.0%} / {n['precision']:.0%} | "
            f"{lc['score']:.1f} / {lc['hit']:.0%} / {lc['precision']:.0%} |"
        )

    out.append("\n## 4. Per-question judge breakdown & retrieved ids\n")
    for i, r in enumerate(scen_rows, 1):
        out.append(f"### Q{i} ({r['scenario']}) — {r['q']}\n")
        out.append(f"- expected: `{r['expected']}`")
        for name in PIPELINE_NAMES:
            d = r[name]
            out.append(
                f"- **{name}** — score {d['score']:.1f} ({d['judges']}), "
                f"hit {d['hit']:.0%}, prec {d['precision']:.0%}, retrieved `{d['retrieved']}`"
            )
            out.append(f"  - answer: {d['answer']}")
        out.append("")

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main() -> None:
    print(f"embed={EMBED_MODEL} chat={CHAT_MODEL} judges={JUDGE_MODELS} workers={WORKERS}")
    print("\n=== 1) MS MARCO (real data, pure semantic recall) ===")
    msmarco = run_msmarco()
    for name in PIPELINE_NAMES:
        a = msmarco[0][name]
        print(f"  {name:10} recall@10={a['recall@10']:.1%}  MRR={a['mrr']:.3f}")

    print("\n=== 2) constructed scenarios (filter-aware, panel-judged) ===")
    scen_rows = run_scenarios()

    write_report(msmarco, scen_rows)
    print("\nreport -> eval/report.md")


if __name__ == "__main__":
    main()
