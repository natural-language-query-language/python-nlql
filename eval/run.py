"""Run the NLQL-vs-LangChain retrieval eval and write eval/report.md.

Run from the repo root:

    export NLQL_EVAL_API_KEY=sk-...
    python -m eval.run

Metrics per cell:
  - score     : LLM-judge answer accuracy (0-5)
  - hit       : recall — share of expected docs present in top-k
  - precision : share of retrieved docs that satisfy the scenario's conditions
"""

from __future__ import annotations

import os
import re
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.data.datasets import DOCUMENTS, QUESTIONS, SCENARIOS  # noqa: E402
from eval.judge import judge_answer  # noqa: E402
from eval.llm import BASE_URL, CHAT_MODEL, EMBED_MODEL, JUDGE_MODEL, chat  # noqa: E402
from eval.pipelines.langchain_rag import LangChainRag  # noqa: E402
from eval.pipelines.nlql_rag import NlqlRag  # noqa: E402

PIPELINE_NAMES = ["NLQL", "LangChain"]

ANSWER_PROMPT = """Answer the question using ONLY the context below. Be concise (one or two sentences). If the context does not support an answer, say "insufficient context".

Question: {q}

Context:
{ctx}

Answer:"""


def answer(q: str, ctxs: list[tuple[str, str]]) -> str:
    ctx = "\n".join(f"- {t}" for _, t in ctxs) or "(no context retrieved)"
    return chat(ANSWER_PROMPT.format(q=q, ctx=ctx))


def hit_rate(expected: list[str], retrieved_ids: list[str]) -> float:
    if not expected:
        return 0.0
    got = set(retrieved_ids)
    return sum(1 for e in expected if e in got) / len(expected)


def _contains_term(question: dict) -> str | None:
    m = re.search(r'CONTAINS "([^"]+)"', question.get("nlql", ""))
    return m.group(1) if m else None


def precision(retrieved_ids: list[str], question: dict, docs_by_id: dict) -> float:
    """Share of retrieved docs that actually satisfy the scenario's conditions
    (status/done equality, date range, CONTAINS). Exposes the false-positive gap
    that hit-rate misses when the expected doc happens to be present anyway."""
    if not retrieved_ids:
        return 0.0
    flt = question.get("filter") or {}
    gte = question.get("filter_date_gte")
    contains = _contains_term(question)

    def ok(doc_id: str) -> bool:
        if doc_id not in docs_by_id:
            return False
        _, text, meta = docs_by_id[doc_id]
        for k, v in flt.items():
            if meta.get(k) != v:
                return False
        if gte and meta.get("date", "") < gte:
            return False
        if contains and contains.lower() not in text.lower():
            return False
        return True

    return sum(1 for rid in retrieved_ids if ok(rid)) / len(retrieved_ids)


def main() -> None:
    print(f"embed={EMBED_MODEL} chat={CHAT_MODEL} judge={JUDGE_MODEL} via {BASE_URL}")
    print(f"corpus: {len(DOCUMENTS)} docs, {len(QUESTIONS)} questions across {len(SCENARIOS)} scenarios")
    print("ingesting corpus...")
    pipelines = [NlqlRag(), LangChainRag()]
    for p in pipelines:
        p.ingest(DOCUMENTS)
        print(f"  {p.name}: {len(DOCUMENTS)} docs indexed")

    docs_by_id = {d[0]: d for d in DOCUMENTS}
    rows: list[dict] = []
    for qi, q in enumerate(QUESTIONS, 1):
        print(f"\n[{qi}/{len(QUESTIONS)}] {q['scenario']}: {q['q']}")
        row = {"scenario": q["scenario"], "q": q["q"], "expected": q["expected"]}
        for p in pipelines:
            t0 = time.time()
            retrieved = p.retrieve(q)
            ans = answer(q["q"], retrieved)
            score, reason = judge_answer(q["q"], q["points"], ans)
            retrieved_ids = [rid for rid, _ in retrieved]
            hit = hit_rate(q["expected"], retrieved_ids)
            prec = precision(retrieved_ids, q, docs_by_id)
            row[p.name] = {
                "score": score,
                "hit": hit,
                "precision": prec,
                "reason": reason,
                "retrieved": retrieved_ids,
                "answer": ans.replace("\n", " ")[:140],
            }
            print(
                f"  {p.name:10} score={score} hit={hit:.0%} prec={prec:.0%} "
                f"retrieved={retrieved_ids} ({time.time() - t0:.1f}s)"
            )
        rows.append(row)

    write_report(rows)
    print("\nreport -> eval/report.md")


def write_report(rows: list[dict]) -> None:
    overall = {}
    by_scenario = {s: {} for s in SCENARIOS}
    for name in PIPELINE_NAMES:
        overall[name] = {
            "score": mean(r[name]["score"] for r in rows),
            "hit": mean(r[name]["hit"] for r in rows),
            "precision": mean(r[name]["precision"] for r in rows),
        }
        for s in SCENARIOS:
            sr = [r for r in rows if r["scenario"] == s]
            by_scenario[s][name] = {
                "score": mean(r[name]["score"] for r in sr),
                "hit": mean(r[name]["hit"] for r in sr),
                "precision": mean(r[name]["precision"] for r in sr),
            }

    out = []
    out.append("# NLQL vs LangChain — retrieval eval\n")
    out.append(
        f"Shared embedding `{EMBED_MODEL}` · answer LLM `{CHAT_MODEL}` · "
        f"judge `{JUDGE_MODEL}` via `{BASE_URL}`.\n"
    )
    out.append("Both pipelines share the same embedding and answer LLM — the only difference is the **retriever**.\n")
    out.append(
        "Metrics: **score** = LLM-judge accuracy (0-5) · **hit** = recall "
        "(share of expected docs found) · **precision** = share of retrieved "
        "docs satisfying the scenario's hard conditions (no false positives).\n"
    )
    out.append(
        "LangChain's standard retriever has no native range or CONTAINS query, "
        "so on `date` / `keyword` / `composite` / `vague` it cannot enforce those "
        "conditions and degrades — that gap is what this eval measures.\n"
    )

    out.append("\n## Overall\n")
    out.append("| pipeline | avg score (0-5) | recall (hit) | precision |")
    out.append("|---|---|---|---|")
    for name in PIPELINE_NAMES:
        o = overall[name]
        out.append(f"| {name} | {o['score']:.2f} | {o['hit']:.0%} | {o['precision']:.0%} |")

    out.append("\n## Per scenario\n")
    out.append("| scenario | NLQL score | NLQL hit | NLQL prec | LC score | LC hit | LC prec |")
    out.append("|---|---|---|---|---|---|---|")
    for s in SCENARIOS:
        n = by_scenario[s]["NLQL"]
        lc = by_scenario[s]["LangChain"]
        out.append(
            f"| {s} | {n['score']:.2f} | {n['hit']:.0%} | {n['precision']:.0%} | "
            f"{lc['score']:.2f} | {lc['hit']:.0%} | {lc['precision']:.0%} |"
        )

    out.append("\n## Per question\n")
    out.append("| # | scenario | question | NLQL (score/hit/prec) | LangChain (score/hit/prec) |")
    out.append("|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        n = r["NLQL"]
        lc = r["LangChain"]
        qshort = r["q"][:46]
        out.append(
            f"| {i} | {r['scenario']} | {qshort} | "
            f"{n['score']} / {n['hit']:.0%} / {n['precision']:.0%} | "
            f"{lc['score']} / {lc['hit']:.0%} / {lc['precision']:.0%} |"
        )

    out.append("\n## Detail (retrieved ids, answer, judge reason)\n")
    for i, r in enumerate(rows, 1):
        out.append(f"### Q{i} ({r['scenario']}) — {r['q']}\n")
        out.append(f"- expected: `{r['expected']}`")
        for name in PIPELINE_NAMES:
            d = r[name]
            out.append(
                f"- **{name}** — score {d['score']}, hit {d['hit']:.0%}, precision {d['precision']:.0%}, "
                f"retrieved `{d['retrieved']}`"
            )
            out.append(f"  - answer: {d['answer']}")
            out.append(f"  - judge : {d['reason']}")
        out.append("")

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
