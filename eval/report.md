# NLQL vs LangChain — retrieval benchmark

Embedding `text-embedding-3-small` · answer `qwen3.7-plus` · judge panel `qwen3.7-max, gpt-5.5, minimax-m3` (averaged) · via `https://ai.su.ki/v1`.


## Fairness & scope

- **Same embedding** (`text-embedding-3-small`) and **same answer LLM** for both pipelines — only the retriever differs.
- Answer scores are the **panel average** across multiple judges to cancel single-model noise.
- **Section 1 — public IR benchmarks** (MS MARCO, BEIR/scifact): standard recall@10 / MRR on real queries with judged relevance. Passages carry no metadata, so neither pipeline can use filters — this is the NLQL-agnostic dimension where NLQL is *not* inherently favored.
- **Section 2 — constructed capability scenarios**: questions we authored to exercise NLQL-specific capabilities (filters, OR/AND/CONTAINS, negation, multi-condition). This is a **self-authored capability probe, not a neutral benchmark**; it shows where the two retrievers differ — and where LangChain's standard retriever degrades (no native range / CONTAINS / !=).
- **Limitations**: small per-scenario sample (3 each), one embedding model, judge panel is itself LLM-based and can be wrong per-question — see the per-judge breakdown in section 4.


## 1. Public IR benchmarks — pure semantic recall (fair, no filters)

| dataset | pipeline | recall@10 | MRR |
|---|---|---|---|
| MS MARCO · 1609 docs / 100 q | NLQL | 100.0% | 0.591 |
| MS MARCO · 1609 docs / 100 q | LangChain | 100.0% | 0.592 |
| BEIR/scifact · 1100 docs / 100 q | NLQL | 94.5% | 0.847 |
| BEIR/scifact · 1100 docs / 100 q | LangChain | 95.5% | 0.853 |

## 2. Constructed capability scenarios (self-authored probe)

> Authored by us to target NLQL's strengths. Read as 'where the retrievers differ', not 'who is objectively better at IR'.

| scenario | NLQL score | NLQL hit | NLQL prec | LC score | LC hit | LC prec |
|---|---|---|---|---|---|---|
| semantic | 5.00 | 100% | 100% | 5.00 | 100% | 89% |
| hybrid | 4.89 | 100% | 100% | 4.78 | 100% | 100% |
| date | 5.00 | 100% | 100% | 5.00 | 100% | 78% |
| keyword | 3.33 | 100% | 100% | 3.33 | 100% | 56% |
| composite | 5.00 | 100% | 100% | 5.00 | 100% | 89% |
| vague | 3.67 | 70% | 100% | 3.11 | 57% | 78% |
| negation | 3.00 | 58% | 100% | 1.33 | 25% | 89% |
| distractor | 4.78 | 100% | 100% | 2.78 | 56% | 78% |
| boolean | 4.11 | 92% | 89% | 3.33 | 69% | 67% |
| **overall** | 4.31 | 91% | 99% | 3.74 | 79% | 80% |

## 3. Per-question (panel-averaged)

| # | scenario | question | NLQL (score/hit/prec) | LangChain (score/hit/prec) |
|---|---|---|---|---|
| 1 | semantic | How do AI agents plan multi-step tasks? | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 2 | semantic | How does the attention mechanism work? | 5.0 / 100% / 100% | 5.0 / 100% / 67% |
| 3 | semantic | What does RAG retrieval do? | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 4 | hybrid | How do AI agents keep memory across turns? | 4.7 / 100% / 100% | 4.3 / 100% / 100% |
| 5 | hybrid | How does RAG retrieve relevant chunks? | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 6 | hybrid | How is the transformer architecture structured | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 7 | date | How do AI agents plan tasks? | 5.0 / 100% / 100% | 5.0 / 100% / 67% |
| 8 | date | How do GPT models predict text? | 5.0 / 100% / 100% | 5.0 / 100% / 67% |
| 9 | date | How does RAG retrieval work? | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 10 | keyword | How does mixture of experts route each token? | 5.0 / 100% / 100% | 5.0 / 100% / 33% |
| 11 | keyword | How does the attention mechanism weigh tokens? | 0.0 / 100% / 100% | 0.0 / 100% / 67% |
| 12 | keyword | How is the transformer architecture built? | 5.0 / 100% / 100% | 5.0 / 100% / 67% |
| 13 | composite | How do AI agents plan multi-step tasks? | 5.0 / 100% / 100% | 5.0 / 100% / 67% |
| 14 | composite | How does RAG retrieval select chunks? | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 15 | composite | How is RAG retrieval evaluated? | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 16 | vague | What have I shipped recently? | 4.0 / 75% / 100% | 4.0 / 75% / 100% |
| 17 | vague | What's coming up in 2025? | 3.0 / 60% / 100% | 1.7 / 20% / 33% |
| 18 | vague | What's still pending on my plate? | 4.0 / 75% / 100% | 3.7 / 75% / 100% |
| 19 | negation | List the published transformer docs (no drafts | 3.7 / 75% / 100% | 2.0 / 25% / 100% |
| 20 | negation | Show me the published agent docs (not the draf | 2.3 / 50% / 100% | 0.0 / 0% / 67% |
| 21 | negation | Which RAG docs are published (exclude drafts)? | 3.0 / 50% / 100% | 2.0 / 50% / 100% |
| 22 | distractor | Published RAG docs from 2024 (there are draft  | 5.0 / 100% / 100% | 3.0 / 67% / 100% |
| 23 | distractor | Published agent docs from 2024 only. | 4.3 / 100% / 100% | 0.3 / 0% / 33% |
| 24 | distractor | Show me the high-priority todos that are done. | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 25 | boolean | Docs that mention both 'retrieval' and 'chunks | 3.0 / 100% / 100% | 4.3 / 100% / 67% |
| 26 | boolean | Published docs mentioning 'agents' or 'tools'. | 4.7 / 100% / 100% | 1.3 / 33% / 67% |
| 27 | boolean | Published docs that mention 'transformer' or ' | 4.7 / 75% / 67% | 4.3 / 75% / 67% |

## 4. Per-question detail (retrieved ids, answer, judge breakdown)

### Q1 (semantic) — How do AI agents plan multi-step tasks?

- expected: `['agt-planning']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['agt-planning', 'agt-multi', 'llm-moe']`
  - answer: By decomposing high-level goals into ordered subtasks.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks.

### Q2 (semantic) — How does the attention mechanism work?

- expected: `['tx-attention']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['tx-attention', 'llm-moe', 'rag-chunk']`
  - answer: The attention mechanism lets a model weigh different tokens when producing each output.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 67%, retrieved `['tx-attention', 'tx-arch', 'scifact-1710116']`
  - answer: The attention mechanism lets a model weigh different tokens when producing each output.

### Q3 (semantic) — What does RAG retrieval do?

- expected: `['rag-retrieve']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'rag-chunk']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'todo-rag-doc', 'rag-eval']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.

### Q4 (hybrid) — How do AI agents keep memory across turns?

- expected: `['agt-memory']`
- **NLQL** — score 4.7 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=4), hit 100%, prec 100%, retrieved `['agt-memory', 'agt-planning', 'rag-eval']`
  - answer: By keeping a working memory of past actions.
- **LangChain** — score 4.3 (qwen3.7-max=5 gpt-5.5=4 minimax-m3=4), hit 100%, prec 100%, retrieved `['agt-memory', 'agt-planning', 'agt-multi']`
  - answer: They keep a working memory of past actions.

### Q5 (hybrid) — How does RAG retrieve relevant chunks?

- expected: `['rag-retrieve']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.

### Q6 (hybrid) — How is the transformer architecture structured?

- expected: `['tx-arch']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['tx-arch', 'tx-bert', 'llm-gpt']`
  - answer: The transformer architecture stacks self-attention and feed-forward layers, with variants structured as encoder-only (e.g., BERT) or decoder
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['tx-arch', 'tx-bert', 'llm-gpt']`
  - answer: The transformer architecture stacks self-attention and feed-forward layers, with variants structured as encoder-only (e.g., BERT) or decoder

### Q7 (date) — How do AI agents plan tasks?

- expected: `['agt-planning']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['agt-planning', 'agt-tools', 'todo-rag-doc']`
  - answer: AI agents plan tasks by decomposing high-level goals into ordered subtasks.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 67%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: AI agents plan tasks by decomposing high-level goals into ordered subtasks. In multi-agent systems, they split complex goals across speciali

### Q8 (date) — How do GPT models predict text?

- expected: `['llm-gpt']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['llm-gpt', 'rag-index', 'todo-eval-harness']`
  - answer: GPT models are decoder-only transformers trained to predict the next token.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 67%, retrieved `['llm-gpt', 'news-2024', 'tx-bert']`
  - answer: GPT models are decoder-only transformers trained to predict the next token.

### Q9 (date) — How does RAG retrieval work?

- expected: `['rag-retrieve']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'todo-rag-doc', 'rag-eval']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'todo-rag-doc', 'rag-eval']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.

### Q10 (keyword) — How does mixture of experts route each token?

- expected: `['llm-moe']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['llm-moe']`
  - answer: It routes each token to a sparse subset of parameters.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 33%, retrieved `['llm-moe', 'tx-attention', 'agt-multi']`
  - answer: To a sparse subset of parameters.

### Q11 (keyword) — How does the attention mechanism weigh tokens?

- expected: `['tx-attention']`
- **NLQL** — score 0.0 (qwen3.7-max=0 gpt-5.5=0 minimax-m3=0), hit 100%, prec 100%, retrieved `['tx-attention', 'tx-arch']`
  - answer: insufficient context
- **LangChain** — score 0.0 (qwen3.7-max=0 gpt-5.5=0 minimax-m3=0), hit 100%, prec 67%, retrieved `['tx-attention', 'llm-moe', 'tx-arch']`
  - answer: insufficient context

### Q12 (keyword) — How is the transformer architecture built?

- expected: `['tx-arch']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['tx-arch', 'tx-bert', 'llm-gpt']`
  - answer: It stacks self-attention and feed-forward layers.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 67%, retrieved `['tx-arch', 'todo-eval-harness', 'tx-bert']`
  - answer: It stacks self-attention and feed-forward layers.

### Q13 (composite) — How do AI agents plan multi-step tasks?

- expected: `['agt-planning']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['agt-planning', 'todo-rag-doc', 'todo-meeting']`
  - answer: By decomposing high-level goals into ordered subtasks.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 67%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks.

### Q14 (composite) — How does RAG retrieval select chunks?

- expected: `['rag-retrieve']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.

### Q15 (composite) — How is RAG retrieval evaluated?

- expected: `['rag-eval']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-eval', 'todo-rag-doc', 'rag-retrieve']`
  - answer: RAG evaluation measures both retrieval recall and answer faithfulness.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-eval', 'rag-retrieve', 'todo-rag-doc']`
  - answer: RAG evaluation measures both retrieval recall and answer faithfulness.

### Q16 (vague) — What have I shipped recently?

- expected: `['todo-rag-doc', 'todo-eval-harness', 'todo-i18n-docs', 'todo-meeting']`
- **NLQL** — score 4.0 (qwen3.7-max=4 gpt-5.5=4 minimax-m3=4), hit 75%, prec 100%, retrieved `['todo-meeting', 'todo-eval-harness', 'todo-rag-doc']`
  - answer: You recently shipped: - Meeting notes: roadmap sync with the platform team (2024-09-30) - The retrieval eval harness comparing frameworks (2
- **LangChain** — score 4.0 (qwen3.7-max=4 gpt-5.5=4 minimax-m3=4), hit 75%, prec 100%, retrieved `['todo-eval-harness', 'todo-meeting', 'todo-rag-doc']`
  - answer: You recently shipped: - Built the retrieval eval harness comparing frameworks (2024-12-05) - Wrote the RAG retrieval documentation draft (20

### Q17 (vague) — What's coming up in 2025?

- expected: `['todo-qdrant-migrate', 'todo-bench-rerun', 'todo-i18n-docs', 'todo-release-02', 'news-2025']`
- **NLQL** — score 3.0 (qwen3.7-max=3 gpt-5.5=3 minimax-m3=3), hit 60%, prec 100%, retrieved `['news-2025', 'todo-qdrant-migrate', 'todo-release-02']`
  - answer: The upcoming tasks in 2025 are: - Migrate the store backend to Qdrant for production scale (Jan 10) - Cut the v0.2 release and publish pytho
- **LangChain** — score 1.7 (qwen3.7-max=1 gpt-5.5=2 minimax-m3=2), hit 20%, prec 33%, retrieved `['news-2025', 'news-2023', 'news-2024']`
  - answer: AI agents that operate browsers and tools autonomously took off.

### Q18 (vague) — What's still pending on my plate?

- expected: `['todo-qdrant-migrate', 'todo-bench-rerun', 'todo-release-02', 'todo-rerank-fix']`
- **NLQL** — score 4.0 (qwen3.7-max=4 gpt-5.5=4 minimax-m3=4), hit 75%, prec 100%, retrieved `['todo-bench-rerun', 'todo-rerank-fix', 'todo-qdrant-migrate']`
  - answer: All three items are pending (`done=false`): - Re-run benchmarks after the columnar filter change - Investigate the reranker regression on sh
- **LangChain** — score 3.7 (qwen3.7-max=4 gpt-5.5=4 minimax-m3=3), hit 75%, prec 100%, retrieved `['todo-bench-rerun', 'todo-release-02', 'todo-rerank-fix']`
  - answer: All three items are pending (done=false): - Re-run benchmarks after the columnar filter change. - Cut the v0.2 release and publish python-nl

### Q19 (negation) — List the published transformer docs (no drafts).

- expected: `['tx-attention', 'tx-arch', 'tx-bert', 'llm-gpt']`
- **NLQL** — score 3.7 (qwen3.7-max=4 gpt-5.5=4 minimax-m3=3), hit 75%, prec 100%, retrieved `['tx-arch', 'tx-attention', 'tx-bert']`
  - answer: - The transformer architecture stacks self-attention and feed-forward layers. - The attention mechanism lets a model weigh different tokens 
- **LangChain** — score 2.0 (qwen3.7-max=2 gpt-5.5=2 minimax-m3=2), hit 25%, prec 100%, retrieved `['todo-i18n-docs', 'todo-rag-doc', 'tx-bert']`
  - answer: BERT is an encoder-only transformer pretrained with masked language modeling.

### Q20 (negation) — Show me the published agent docs (not the drafts).

- expected: `['agt-planning', 'agt-memory']`
- **NLQL** — score 2.3 (qwen3.7-max=3 gpt-5.5=2 minimax-m3=2), hit 50%, prec 100%, retrieved `['agt-planning', 'news-2025', 'agt-multi']`
  - answer: - AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks. - Multi-agent systems split a complex goal across s
- **LangChain** — score 0.0 (qwen3.7-max=0 gpt-5.5=0 minimax-m3=0), hit 0%, prec 67%, retrieved `['todo-i18n-docs', 'todo-rag-doc', 'agt-tools']`
  - answer: insufficient context

### Q21 (negation) — Which RAG docs are published (exclude drafts)?

- expected: `['rag-index', 'rag-retrieve', 'rag-rerank', 'rag-eval']`
- **NLQL** — score 3.0 (qwen3.7-max=3 gpt-5.5=3 minimax-m3=3), hit 50%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: - RAG retrieval selects the top-k chunks most relevant to a query before generation. - RAG indexing splits documents into chunks and embeds 
- **LangChain** — score 2.0 (qwen3.7-max=2 gpt-5.5=2 minimax-m3=2), hit 50%, prec 100%, retrieved `['todo-rag-doc', 'rag-index', 'rag-retrieve']`
  - answer: Based on the metadata (`status=published`), all three RAG documents are published:  - Wrote the RAG retrieval documentation draft. - RAG ind

### Q22 (distractor) — Published RAG docs from 2024 (there are draft and 2025 ones too).

- expected: `['rag-index', 'rag-retrieve', 'rag-eval']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'rag-eval']`
  - answer: 1. RAG retrieval selects the top-k chunks most relevant to a query before generation. 2. RAG indexing splits documents into chunks and embed
- **LangChain** — score 3.0 (qwen3.7-max=3 gpt-5.5=3 minimax-m3=3), hit 67%, prec 100%, retrieved `['todo-rag-doc', 'rag-index', 'rag-retrieve']`
  - answer: All three items in the context are published RAG docs from 2024:  1. Wrote the RAG retrieval documentation draft. (2024-11-20) 2. RAG indexi

### Q23 (distractor) — Published agent docs from 2024 only.

- expected: `['agt-planning', 'agt-memory']`
- **NLQL** — score 4.3 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=3), hit 100%, prec 100%, retrieved `['agt-planning', 'agt-memory', 'todo-rag-doc']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks. Modern AI agents keep a working memory of past action
- **LangChain** — score 0.3 (qwen3.7-max=0 gpt-5.5=1 minimax-m3=0), hit 0%, prec 33%, retrieved `['news-2025', 'todo-i18n-docs', 'todo-rag-doc']`
  - answer: Wrote the RAG retrieval documentation draft.

### Q24 (distractor) — Show me the high-priority todos that are done.

- expected: `['todo-rag-doc', 'todo-eval-harness']`
- **NLQL** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['todo-rag-doc', 'todo-eval-harness']`
  - answer: - Wrote the RAG retrieval documentation draft. - Built the retrieval eval harness comparing frameworks.
- **LangChain** — score 5.0 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['todo-eval-harness', 'todo-rag-doc']`
  - answer: - Built the retrieval eval harness comparing frameworks. - Wrote the RAG retrieval documentation draft.

### Q25 (boolean) — Docs that mention both 'retrieval' and 'chunks'.

- expected: `['rag-retrieve']`
- **NLQL** — score 3.0 (qwen3.7-max=2 gpt-5.5=5 minimax-m3=2), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-chunk']`
  - answer: Both documents.
- **LangChain** — score 4.3 (qwen3.7-max=5 gpt-5.5=4 minimax-m3=4), hit 100%, prec 67%, retrieved `['rag-chunk', 'rag-retrieve', 'rag-index']`
  - answer: - Chunk size affects retrieval quality: small chunks stay focused, large ones keep context. - RAG retrieval selects the top-k chunks most re

### Q26 (boolean) — Published docs mentioning 'agents' or 'tools'.

- expected: `['agt-planning', 'agt-memory', 'agt-multi']`
- **NLQL** — score 4.7 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=4), hit 100%, prec 100%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: All three documents are published and mention 'agents' (none mention 'tools'):  1. AI agents plan multi-step tasks by decomposing high-level
- **LangChain** — score 1.3 (qwen3.7-max=1 gpt-5.5=1 minimax-m3=2), hit 33%, prec 67%, retrieved `['news-2025', 'agt-multi', 'todo-i18n-docs']`
  - answer: - In 2025 AI agents that operate browsers and tools autonomously took off. - Multi-agent systems split a complex goal across several special

### Q27 (boolean) — Published docs that mention 'transformer' or 'attention'.

- expected: `['tx-arch', 'tx-attention', 'tx-bert', 'llm-gpt']`
- **NLQL** — score 4.7 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=4), hit 75%, prec 67%, retrieved `['tx-attention', 'tx-arch', 'tx-bert']`
  - answer: - The attention mechanism lets a model weigh different tokens when producing each output. (2023-06-11) - The transformer architecture stacks
- **LangChain** — score 4.3 (qwen3.7-max=5 gpt-5.5=5 minimax-m3=3), hit 75%, prec 67%, retrieved `['tx-arch', 'tx-attention', 'tx-bert']`
  - answer: All three documents are published and mention 'transformer' or 'attention':  - The transformer architecture stacks self-attention and feed-f

