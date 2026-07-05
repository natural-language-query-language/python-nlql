# NLQL vs LangChain — retrieval eval v2

Embedding `text-embedding-3-small` · answer `qwen3.7-plus` · judge panel `qwen3.7-max, deepseek-v4-flash, minimax-m3` (averaged) · via `https://ai.su.ki/v1`.

Both pipelines share the same embedding and answer LLM — only the retriever differs. Scenario scores are the **panel average** to cancel single-judge noise.


## 1. MS MARCO — real Bing queries, pure semantic recall

Corpus: 657 deduped passages · 40 queries with judged relevant passages. Passages carry no metadata, so neither pipeline can use filters — this is the dimension where **NLQL is not inherently favored** (keeps the benchmark honest).

| pipeline | recall@10 | MRR |
|---|---|---|
| NLQL | 100.0% | 0.632 |
| LangChain | 100.0% | 0.633 |

## 2. Constructed scenarios — filter-aware, end-to-end

| scenario | NLQL score | NLQL hit | NLQL prec | LC score | LC hit | LC prec |
|---|---|---|---|---|---|---|
| semantic | 5.00 | 100% | 100% | 5.00 | 100% | 100% |
| hybrid | 5.00 | 100% | 100% | 4.89 | 100% | 100% |
| date | 5.00 | 100% | 100% | 5.00 | 100% | 78% |
| keyword | 3.33 | 100% | 100% | 3.33 | 100% | 56% |
| composite | 5.00 | 100% | 100% | 5.00 | 100% | 89% |
| vague | 2.89 | 70% | 100% | 1.89 | 57% | 78% |
| negation | 0.00 | 58% | 100% | 0.00 | 25% | 89% |
| distractor | 0.00 | 100% | 100% | 0.00 | 56% | 78% |
| **overall** | 3.28 | 91% | 100% | 3.14 | 80% | 83% |

## 3. Per-question (panel-averaged score)

| # | scenario | question | NLQL (score/hit/prec) | LangChain (score/hit/prec) |
|---|---|---|---|---|
| 1 | semantic | How do AI agents plan multi-step tasks? | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 2 | semantic | How does the attention mechanism work? | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 3 | semantic | What does RAG retrieval do? | 5.0 / 100% / 100% | 5.0 / 100% / 100% |
| 4 | hybrid | How do AI agents keep memory across turns? | 5.0 / 100% / 100% | 4.7 / 100% / 100% |
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
| 16 | vague | What have I shipped recently? | 3.0 / 75% / 100% | 0.0 / 75% / 100% |
| 17 | vague | What's coming up in 2025? | 1.7 / 60% / 100% | 1.7 / 20% / 33% |
| 18 | vague | What's still pending on my plate? | 4.0 / 75% / 100% | 4.0 / 75% / 100% |
| 19 | negation | List the published transformer docs (no drafts | 0.0 / 75% / 100% | 0.0 / 25% / 100% |
| 20 | negation | Show me the published agent docs (not the draf | 0.0 / 50% / 100% | 0.0 / 0% / 67% |
| 21 | negation | Which RAG docs are published (exclude drafts)? | 0.0 / 50% / 100% | 0.0 / 50% / 100% |
| 22 | distractor | Published RAG docs from 2024 (there are draft  | 0.0 / 100% / 100% | 0.0 / 67% / 100% |
| 23 | distractor | Published agent docs from 2024 only. | 0.0 / 100% / 100% | 0.0 / 0% / 33% |
| 24 | distractor | Show me the high-priority todos that are done. | 0.0 / 100% / 100% | 0.0 / 100% / 100% |

## 4. Per-question judge breakdown & retrieved ids

### Q1 (semantic) — How do AI agents plan multi-step tasks?

- expected: `['agt-planning']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['agt-planning', 'agt-multi', 'llm-moe']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks, or by splitting complex goals across specialized agen

### Q2 (semantic) — How does the attention mechanism work?

- expected: `['tx-attention']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['tx-attention', 'llm-moe', 'rag-chunk']`
  - answer: The attention mechanism works by letting a model weigh different tokens when producing each output.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['tx-attention', 'tx-arch', 'agt-memory']`
  - answer: The attention mechanism works by letting a model weigh different tokens when producing each output.

### Q3 (semantic) — What does RAG retrieval do?

- expected: `['rag-retrieve']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'rag-chunk']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'todo-rag-doc', 'rag-eval']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.

### Q4 (hybrid) — How do AI agents keep memory across turns?

- expected: `['agt-memory']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['agt-memory', 'agt-planning', 'rag-eval']`
  - answer: AI agents keep memory across turns by maintaining a working memory of past actions.
- **LangChain** — score 4.7 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=4), hit 100%, prec 100%, retrieved `['agt-memory', 'agt-planning', 'agt-multi']`
  - answer: AI agents keep memory across turns by keeping a working memory of past actions.

### Q5 (hybrid) — How does RAG retrieve relevant chunks?

- expected: `['rag-retrieve']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieves relevant chunks by selecting the top-k chunks most relevant to a query.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieves relevant chunks by selecting the top-k chunks most relevant to a query.

### Q6 (hybrid) — How is the transformer architecture structured?

- expected: `['tx-arch']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['tx-arch', 'tx-bert', 'llm-gpt']`
  - answer: The transformer architecture is structured by stacking self-attention and feed-forward layers.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['tx-arch', 'tx-bert', 'llm-gpt']`
  - answer: The transformer architecture is structured by stacking self-attention and feed-forward layers.

### Q7 (date) — How do AI agents plan tasks?

- expected: `['agt-planning']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['agt-planning', 'agt-tools', 'todo-rag-doc']`
  - answer: AI agents plan tasks by decomposing high-level goals into ordered subtasks.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 67%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: AI agents plan tasks by decomposing high-level goals into ordered subtasks, and multi-agent systems split complex goals across specialized a

### Q8 (date) — How do GPT models predict text?

- expected: `['llm-gpt']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['llm-gpt', 'rag-index', 'todo-eval-harness']`
  - answer: GPT models are decoder-only transformers trained to predict the next token.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 67%, retrieved `['llm-gpt', 'news-2024', 'tx-bert']`
  - answer: GPT models are decoder-only transformers trained to predict the next token.

### Q9 (date) — How does RAG retrieval work?

- expected: `['rag-retrieve']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'todo-rag-doc', 'rag-eval']`
  - answer: RAG retrieval works by selecting the top-k chunks most relevant to a query before generation.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'todo-rag-doc', 'rag-eval']`
  - answer: RAG retrieval works by selecting the top-k chunks most relevant to a query before generation.

### Q10 (keyword) — How does mixture of experts route each token?

- expected: `['llm-moe']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['llm-moe']`
  - answer: Mixture of Experts routes each token to a sparse subset of parameters.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 33%, retrieved `['llm-moe', 'tx-attention', 'agt-multi']`
  - answer: Mixture of Experts routes each token to a sparse subset of parameters.

### Q11 (keyword) — How does the attention mechanism weigh tokens?

- expected: `['tx-attention']`
- **NLQL** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 100%, prec 100%, retrieved `['tx-attention', 'tx-arch']`
  - answer: insufficient context
- **LangChain** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 100%, prec 67%, retrieved `['tx-attention', 'llm-moe', 'tx-arch']`
  - answer: insufficient context

### Q12 (keyword) — How is the transformer architecture built?

- expected: `['tx-arch']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['tx-arch', 'tx-bert', 'llm-gpt']`
  - answer: The transformer architecture is built by stacking self-attention and feed-forward layers.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 67%, retrieved `['tx-arch', 'todo-eval-harness', 'tx-bert']`
  - answer: The transformer architecture is built by stacking self-attention and feed-forward layers.

### Q13 (composite) — How do AI agents plan multi-step tasks?

- expected: `['agt-planning']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['agt-planning', 'todo-rag-doc', 'todo-meeting']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 67%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks, or by splitting complex goals across specialized agen

### Q14 (composite) — How does RAG retrieval select chunks?

- expected: `['rag-retrieve']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.

### Q15 (composite) — How is RAG retrieval evaluated?

- expected: `['rag-eval']`
- **NLQL** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-eval', 'todo-rag-doc', 'rag-retrieve']`
  - answer: RAG evaluation measures both retrieval recall and answer faithfulness.
- **LangChain** — score 5.0 (qwen3.7-max=5 deepseek-v4-flash=5 minimax-m3=5), hit 100%, prec 100%, retrieved `['rag-eval', 'rag-retrieve', 'todo-rag-doc']`
  - answer: RAG evaluation measures both retrieval recall and answer faithfulness.

### Q16 (vague) — What have I shipped recently?

- expected: `['todo-rag-doc', 'todo-eval-harness', 'todo-i18n-docs', 'todo-meeting']`
- **NLQL** — score 3.0 (qwen3.7-max=3 deepseek-v4-flash=3 minimax-m3=3), hit 75%, prec 100%, retrieved `['todo-meeting', 'todo-eval-harness', 'todo-rag-doc']`
  - answer: You recently shipped the retrieval eval harness comparing frameworks and wrote a draft of the RAG retrieval documentation.
- **LangChain** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 75%, prec 100%, retrieved `['todo-eval-harness', 'todo-meeting', 'todo-rag-doc']`
  - answer: insufficient context

### Q17 (vague) — What's coming up in 2025?

- expected: `['todo-qdrant-migrate', 'todo-bench-rerun', 'todo-i18n-docs', 'todo-release-02', 'news-2025']`
- **NLQL** — score 1.7 (qwen3.7-max=2 deepseek-v4-flash=2 minimax-m3=1), hit 60%, prec 100%, retrieved `['news-2025', 'todo-qdrant-migrate', 'todo-release-02']`
  - answer: In 2025, AI agents that operate browsers and tools autonomously took off.
- **LangChain** — score 1.7 (qwen3.7-max=2 deepseek-v4-flash=2 minimax-m3=1), hit 20%, prec 33%, retrieved `['news-2025', 'news-2023', 'news-2024']`
  - answer: In 2025, AI agents that operate browsers and tools autonomously took off.

### Q18 (vague) — What's still pending on my plate?

- expected: `['todo-qdrant-migrate', 'todo-bench-rerun', 'todo-release-02', 'todo-rerank-fix']`
- **NLQL** — score 4.0 (qwen3.7-max=4 deepseek-v4-flash=4 minimax-m3=4), hit 75%, prec 100%, retrieved `['todo-bench-rerun', 'todo-rerank-fix', 'todo-qdrant-migrate']`
  - answer: Your pending tasks are to re-run benchmarks after the columnar filter change, investigate the reranker regression on short queries, and migr
- **LangChain** — score 4.0 (qwen3.7-max=4 deepseek-v4-flash=4 minimax-m3=4), hit 75%, prec 100%, retrieved `['todo-bench-rerun', 'todo-release-02', 'todo-rerank-fix']`
  - answer: Your pending tasks are to re-run benchmarks after the columnar filter change, cut the v0.2 release and publish python-nlql to PyPI, and inve

### Q19 (negation) — List the published transformer docs (no drafts).

- expected: `['tx-attention', 'tx-arch', 'tx-bert', 'llm-gpt']`
- **NLQL** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 75%, prec 100%, retrieved `['tx-arch', 'tx-attention', 'tx-bert']`
  - answer: insufficient context
- **LangChain** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 25%, prec 100%, retrieved `['todo-i18n-docs', 'tx-arch', 'todo-rag-doc']`
  - answer: insufficient context

### Q20 (negation) — Show me the published agent docs (not the drafts).

- expected: `['agt-planning', 'agt-memory']`
- **NLQL** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 50%, prec 100%, retrieved `['agt-planning', 'news-2025', 'agt-multi']`
  - answer: insufficient context
- **LangChain** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 0%, prec 67%, retrieved `['todo-i18n-docs', 'todo-rag-doc', 'agt-tools']`
  - answer: insufficient context

### Q21 (negation) — Which RAG docs are published (exclude drafts)?

- expected: `['rag-index', 'rag-retrieve', 'rag-rerank', 'rag-eval']`
- **NLQL** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 50%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: insufficient context
- **LangChain** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 50%, prec 100%, retrieved `['todo-rag-doc', 'rag-index', 'rag-retrieve']`
  - answer: insufficient context

### Q22 (distractor) — Published RAG docs from 2024 (there are draft and 2025 ones too).

- expected: `['rag-index', 'rag-retrieve', 'rag-eval']`
- **NLQL** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 100%, prec 100%, retrieved `['rag-retrieve', 'rag-index', 'rag-eval']`
  - answer: insufficient context
- **LangChain** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 67%, prec 100%, retrieved `['todo-rag-doc', 'rag-index', 'rag-retrieve']`
  - answer: insufficient context

### Q23 (distractor) — Published agent docs from 2024 only.

- expected: `['agt-planning', 'agt-memory']`
- **NLQL** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 100%, prec 100%, retrieved `['agt-planning', 'agt-memory', 'todo-rag-doc']`
  - answer: insufficient context
- **LangChain** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 0%, prec 33%, retrieved `['news-2025', 'todo-i18n-docs', 'todo-rag-doc']`
  - answer: insufficient context

### Q24 (distractor) — Show me the high-priority todos that are done.

- expected: `['todo-rag-doc', 'todo-eval-harness']`
- **NLQL** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 100%, prec 100%, retrieved `['todo-rag-doc', 'todo-eval-harness']`
  - answer: insufficient context
- **LangChain** — score 0.0 (qwen3.7-max=0 deepseek-v4-flash=0 minimax-m3=0), hit 100%, prec 100%, retrieved `['todo-eval-harness', 'todo-rag-doc']`
  - answer: insufficient context

