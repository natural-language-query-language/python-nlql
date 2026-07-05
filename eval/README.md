# NLQL vs LangChain — retrieval eval

Compares NLQL retrieval against LangChain's standard RAG retriever across 5 scenarios.
The two pipelines share the same embedding model and answer LLM, so the only difference
is the retriever — the score reflects retrieval quality, not model differences.

## Setup

```bash
pip install langchain-chroma langchain-openai
export NLQL_EVAL_API_KEY=sk-...        # OpenAI-compatible key for hjmai.yby.zone
# optional overrides (defaults shown):
# export NLQL_EVAL_BASE_URL=https://ai.su.ki/v1
# export NLQL_EVAL_EMBED=text-embedding-3-small
# export NLQL_EVAL_CHAT=doubao-seed-2.0-mini
# export NLQL_EVAL_JUDGE=gpt-5.4
```

## Run

```bash
python -m eval.run
```

Writes [`report.md`](report.md) with overall + per-scenario + per-question tables, plus a short
judge rationale per cell.

## Scenarios

| scenario | what it tests | LangChain retriever capability |
|---|---|---|
| `semantic` | pure similarity | supported |
| `hybrid` | semantic + status equality filter | supported (metadata filter) |
| `date` | semantic + date range (`>=` / `between`) | **not supported** — degrades to similarity |
| `keyword` | CONTAINS (lexical substring) | **not supported** — semantic only |
| `composite` | semantic + status + date range | partial (equality filter only) |

## Fairness guarantees

- Same embedding for both pipelines (`text-embedding-3-small`).
- Same answer LLM (`doubao-seed-2.0-mini`), generation outside the pipeline.
- A different model judges (`gpt-5.4`), to avoid self-bias.
- Each question's NLQL is hand-written (no LLM→NLQL translation noise); the LangChain
  side uses the equivalent metadata filter where the retriever supports it. Range and
  CONTAINS have no LangChain-retriever equivalent — that gap is part of what this eval
  measures, not a flaw in the harness.

## Layout

```
eval/
  llm.py                     # OpenAI-compatible client + config (env-driven)
  data/datasets.py           # 24 docs + 15 questions across 5 scenarios
  pipelines/base.py          # RagPipeline protocol
  pipelines/nlql_rag.py      # NLQL retrieval
  pipelines/langchain_rag.py # LangChain + Chroma baseline
  judge.py                   # LLM-as-judge scoring (0-5)
  run.py                     # entry point → eval/report.md
```
