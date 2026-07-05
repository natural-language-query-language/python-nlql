# NLQL vs LangChain — retrieval eval

Shared embedding `text-embedding-3-small` · answer LLM `doubao-seed-2.0-mini` · judge `gpt-5.4` via `https://ai.su.ki/v1`.

Both pipelines share the same embedding and answer LLM — the only difference is the **retriever**.

Metrics: **score** = LLM-judge accuracy (0-5) · **hit** = recall (share of expected docs found) · **precision** = share of retrieved docs satisfying the scenario's hard conditions (no false positives).

LangChain's standard retriever has no native range or CONTAINS query, so on `date` / `keyword` / `composite` / `vague` it cannot enforce those conditions and degrades — that gap is what this eval measures.


## Overall

| pipeline | avg score (0-5) | recall (hit) | precision |
|---|---|---|---|
| NLQL | 4.44 | 95% | 100% |
| LangChain | 4.33 | 93% | 83% |

## Per scenario

| scenario | NLQL score | NLQL hit | NLQL prec | LC score | LC hit | LC prec |
|---|---|---|---|---|---|---|
| semantic | 5.00 | 100% | 100% | 5.00 | 100% | 100% |
| hybrid | 5.00 | 100% | 100% | 5.00 | 100% | 100% |
| date | 5.00 | 100% | 100% | 5.00 | 100% | 78% |
| keyword | 3.33 | 100% | 100% | 4.67 | 100% | 56% |
| composite | 5.00 | 100% | 100% | 4.33 | 100% | 89% |
| vague | 3.33 | 70% | 100% | 2.00 | 57% | 78% |

## Per question

| # | scenario | question | NLQL (score/hit/prec) | LangChain (score/hit/prec) |
|---|---|---|---|---|
| 1 | semantic | How do AI agents plan multi-step tasks? | 5 / 100% / 100% | 5 / 100% / 100% |
| 2 | semantic | What does RAG retrieval do? | 5 / 100% / 100% | 5 / 100% / 100% |
| 3 | semantic | How does the attention mechanism work? | 5 / 100% / 100% | 5 / 100% / 100% |
| 4 | hybrid | How do AI agents keep memory across turns? | 5 / 100% / 100% | 5 / 100% / 100% |
| 5 | hybrid | How does RAG retrieve relevant chunks? | 5 / 100% / 100% | 5 / 100% / 100% |
| 6 | hybrid | How is the transformer architecture structured | 5 / 100% / 100% | 5 / 100% / 100% |
| 7 | date | How do AI agents plan tasks? | 5 / 100% / 100% | 5 / 100% / 67% |
| 8 | date | How does RAG retrieval work? | 5 / 100% / 100% | 5 / 100% / 100% |
| 9 | date | How do GPT models predict text? | 5 / 100% / 100% | 5 / 100% / 67% |
| 10 | keyword | How is the transformer architecture built? | 5 / 100% / 100% | 5 / 100% / 67% |
| 11 | keyword | How does the attention mechanism weigh tokens? | 0 / 100% / 100% | 4 / 100% / 67% |
| 12 | keyword | How does mixture of experts route each token? | 5 / 100% / 100% | 5 / 100% / 33% |
| 13 | composite | How does RAG retrieval select chunks? | 5 / 100% / 100% | 5 / 100% / 100% |
| 14 | composite | How do AI agents plan multi-step tasks? | 5 / 100% / 100% | 5 / 100% / 67% |
| 15 | composite | How is RAG retrieval evaluated? | 5 / 100% / 100% | 3 / 100% / 100% |
| 16 | vague | What have I shipped recently? | 3 / 75% / 100% | 0 / 75% / 100% |
| 17 | vague | What's still pending on my plate? | 4 / 75% / 100% | 4 / 75% / 100% |
| 18 | vague | What's coming up in 2025? | 3 / 60% / 100% | 2 / 20% / 33% |

## Detail (retrieved ids, answer, judge reason)

### Q1 (semantic) — How do AI agents plan multi-step tasks?

- expected: `['agt-planning']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['agt-planning', 'agt-multi', 'llm-moe']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks.
  - judge : It exactly states that agents decompose high-level goals into ordered subtasks.
- **LangChain** — score 5, hit 100%, precision 100%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks. They may also use working memory to stay coherent acr
  - judge : It correctly states that agents decompose high-level goals into ordered subtasks, with only additional relevant details.

### Q2 (semantic) — What does RAG retrieval do?

- expected: `['rag-retrieve']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['rag-retrieve', 'rag-index', 'rag-chunk']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
  - judge : It exactly states that retrieval selects the top-k query-relevant chunks before generation.
- **LangChain** — score 5, hit 100%, precision 100%, retrieved `['rag-retrieve', 'todo-rag-doc', 'rag-eval']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
  - judge : It exactly states that retrieval selects the top-k chunks most relevant to the query before generation.

### Q3 (semantic) — How does the attention mechanism work?

- expected: `['tx-attention']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['tx-attention', 'llm-moe', 'rag-chunk']`
  - answer: The attention mechanism lets a model weigh different tokens when producing each output.
  - judge : It exactly states that attention weighs different tokens for each output, covering the key point fully.
- **LangChain** — score 5, hit 100%, precision 100%, retrieved `['tx-attention', 'tx-arch', 'agt-memory']`
  - answer: The attention mechanism lets a model weigh different tokens when producing each output.
  - judge : It exactly states that attention weighs different tokens when producing each output.

### Q4 (hybrid) — How do AI agents keep memory across turns?

- expected: `['agt-memory']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['agt-memory', 'agt-planning', 'rag-eval']`
  - answer: Modern AI agents keep a working memory of past actions to stay coherent across turns.
  - judge : It directly states that agents use a working memory of past actions to remain coherent across turns.
- **LangChain** — score 5, hit 100%, precision 100%, retrieved `['agt-memory', 'agt-planning', 'agt-multi']`
  - answer: Modern AI agents keep a working memory of past actions to stay coherent across turns.
  - judge : It directly states that agents use a working memory of past actions to remain coherent across turns.

### Q5 (hybrid) — How does RAG retrieve relevant chunks?

- expected: `['rag-retrieve']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG first splits documents into chunks and embeds them into a vector store during indexing, then selects the top-k chunks most relevant to a
  - judge : It correctly states that RAG retrieves the top-k chunks most relevant to the query before generation.
- **LangChain** — score 5, hit 100%, precision 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG first splits documents into chunks and embeds them into a vector store during indexing, then selects the top-k chunks most relevant to a
  - judge : It correctly states that RAG retrieves the top-k chunks most relevant to the query, with accurate extra context about indexing.

### Q6 (hybrid) — How is the transformer architecture structured?

- expected: `['tx-arch']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['tx-arch', 'tx-bert', 'llm-gpt']`
  - answer: The transformer architecture stacks self-attention and feed-forward layers. There are also encoder-only and decoder-only variants like BERT 
  - judge : It correctly states that transformers stack self-attention and feed-forward layers and adds accurate variant examples.
- **LangChain** — score 5, hit 100%, precision 100%, retrieved `['tx-arch', 'tx-bert', 'llm-gpt']`
  - answer: The transformer architecture stacks self-attention and feed-forward layers. There are also encoder-only and decoder-only variants like BERT 
  - judge : It correctly states that transformers stack self-attention and feed-forward layers and adds accurate variant examples.

### Q7 (date) — How do AI agents plan tasks?

- expected: `['agt-planning']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['agt-planning', 'agt-tools', 'todo-rag-doc']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks.
  - judge : It correctly states that AI agents plan by breaking high-level goals into ordered subtasks.
- **LangChain** — score 5, hit 100%, precision 67%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks. They may also use working memory of past actions to s
  - judge : It correctly states that AI agents plan by decomposing high-level goals into ordered subtasks and adds relevant details without contradicting the key point.

### Q8 (date) — How does RAG retrieval work?

- expected: `['rag-retrieve']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['rag-retrieve', 'todo-rag-doc', 'rag-eval']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
  - judge : It exactly states that retrieval picks the top-k most relevant chunks before generation.
- **LangChain** — score 5, hit 100%, precision 100%, retrieved `['rag-retrieve', 'todo-rag-doc', 'rag-eval']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
  - judge : It exactly states that retrieval picks the top-k most relevant chunks before generation.

### Q9 (date) — How do GPT models predict text?

- expected: `['llm-gpt']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['llm-gpt', 'rag-index', 'todo-eval-harness']`
  - answer: GPT models are decoder-only transformers trained to predict the next token, which is how they predict text.
  - judge : It correctly states that GPT models are decoder-only transformers trained to predict the next token.
- **LangChain** — score 5, hit 100%, precision 67%, retrieved `['llm-gpt', 'news-2024', 'tx-bert']`
  - answer: GPT models are decoder-only transformers trained to predict the next token, which is how they perform text prediction.
  - judge : It exactly states that GPT models are decoder-only transformers trained to predict the next token.

### Q10 (keyword) — How is the transformer architecture built?

- expected: `['tx-arch']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['tx-arch', 'tx-bert', 'llm-gpt']`
  - answer: The transformer architecture stacks self-attention and feed-forward layers.
  - judge : It exactly states that transformers are built by stacking self-attention and feed-forward layers.
- **LangChain** — score 5, hit 100%, precision 67%, retrieved `['tx-arch', 'todo-eval-harness', 'tx-bert']`
  - answer: The transformer architecture stacks self-attention and feed-forward layers.
  - judge : It exactly states that transformers are built by stacking self-attention and feed-forward layers.

### Q11 (keyword) — How does the attention mechanism weigh tokens?

- expected: `['tx-attention']`
- **NLQL** — score 0, hit 100%, precision 100%, retrieved `['tx-attention', 'tx-arch']`
  - answer: insufficient context
  - judge : It gives no explanation and does not address how attention weighs tokens.
- **LangChain** — score 4, hit 100%, precision 67%, retrieved `['tx-attention', 'llm-moe', 'tx-arch']`
  - answer: The context does not specify how the attention mechanism weighs tokens, only that it does weigh different tokens when producing each output.
  - judge : It captures the key point but unnecessarily claims insufficient context instead of answering directly.

### Q12 (keyword) — How does mixture of experts route each token?

- expected: `['llm-moe']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['llm-moe']`
  - answer: Mixture of Experts routes each token to a sparse subset of parameters.
  - judge : It exactly states that each token is routed to a sparse subset of parameters.
- **LangChain** — score 5, hit 100%, precision 33%, retrieved `['llm-moe', 'tx-attention', 'agt-multi']`
  - answer: Mixture of Experts routes each token to a sparse subset of parameters.
  - judge : It exactly states that each token is routed to a sparse subset of parameters.

### Q13 (composite) — How does RAG retrieval select chunks?

- expected: `['rag-retrieve']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
  - judge : It exactly states that retrieval picks the top-k chunks most relevant to the query before generation.
- **LangChain** — score 5, hit 100%, precision 100%, retrieved `['rag-retrieve', 'rag-index', 'todo-rag-doc']`
  - answer: RAG retrieval selects the top-k chunks most relevant to a query before generation.
  - judge : It exactly states that RAG retrieves the top-k chunks most relevant to the query before generation.

### Q14 (composite) — How do AI agents plan multi-step tasks?

- expected: `['agt-planning']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['agt-planning', 'todo-rag-doc', 'todo-meeting']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks.
  - judge : It exactly states that agents decompose high-level goals into ordered subtasks.
- **LangChain** — score 5, hit 100%, precision 67%, retrieved `['agt-planning', 'agt-memory', 'agt-multi']`
  - answer: AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks. They may also use working memory to stay coherent acr
  - judge : It directly states that agents decompose high-level goals into ordered subtasks, fully covering the key point.

### Q15 (composite) — How is RAG retrieval evaluated?

- expected: `['rag-eval']`
- **NLQL** — score 5, hit 100%, precision 100%, retrieved `['rag-eval', 'todo-rag-doc', 'rag-retrieve']`
  - answer: RAG retrieval is evaluated using retrieval recall, alongside answer faithfulness for overall RAG evaluation.
  - judge : It correctly includes both retrieval recall and answer faithfulness.
- **LangChain** — score 3, hit 100%, precision 100%, retrieved `['rag-eval', 'rag-retrieve', 'todo-rag-doc']`
  - answer: RAG retrieval is evaluated using retrieval recall.
  - judge : It correctly mentions retrieval recall but omits answer faithfulness.

### Q16 (vague) — What have I shipped recently?

- expected: `['todo-rag-doc', 'todo-eval-harness', 'todo-i18n-docs', 'todo-meeting']`
- **NLQL** — score 3, hit 75%, precision 100%, retrieved `['todo-meeting', 'todo-eval-harness', 'todo-rag-doc']`
  - answer: You have recently built the retrieval eval harness comparing frameworks and written the RAG retrieval documentation draft.
  - judge : It correctly mentions the eval harness and RAG docs but omits the i18n docs and meeting notes.
- **LangChain** — score 0, hit 75%, precision 100%, retrieved `['todo-eval-harness', 'todo-meeting', 'todo-rag-doc']`
  - answer: insufficient context
  - judge : It provides no useful answer and misses all recently shipped items.

### Q17 (vague) — What's still pending on my plate?

- expected: `['todo-qdrant-migrate', 'todo-bench-rerun', 'todo-release-02', 'todo-rerank-fix']`
- **NLQL** — score 4, hit 75%, precision 100%, retrieved `['todo-bench-rerun', 'todo-rerank-fix', 'todo-qdrant-migrate']`
  - answer: Your pending tasks are re-running benchmarks after the columnar filter change, investigating the reranker regression on short queries, and m
  - judge : It correctly lists the benchmark rerun, reranker fix, and Qdrant migration, but misses the v0.2 release.
- **LangChain** — score 4, hit 75%, precision 100%, retrieved `['todo-bench-rerun', 'todo-release-02', 'todo-rerank-fix']`
  - answer: Your pending tasks are re-running benchmarks after the columnar filter change, cutting the v0.2 release and publishing python-nlql to PyPI, 
  - judge : Covers bench rerun, v0.2 release, and reranker fix, but misses the qdrant migration.

### Q18 (vague) — What's coming up in 2025?

- expected: `['todo-qdrant-migrate', 'todo-bench-rerun', 'todo-i18n-docs', 'todo-release-02', 'news-2025']`
- **NLQL** — score 3, hit 60%, precision 100%, retrieved `['news-2025', 'todo-qdrant-migrate', 'todo-release-02']`
  - answer: In 2025, AI agents that operate browsers and tools autonomously took off, there will also be the migration of the store backend to Qdrant fo
  - judge : Mentions Qdrant migration, v0.2, and agent news, but misses bench rerun and i18n docs and adds an unsupported PyPI item.
- **LangChain** — score 2, hit 20%, precision 33%, retrieved `['news-2025', 'news-2023', 'news-2024']`
  - answer: AI agents that operate browsers and tools autonomously took off in 2025.
  - judge : It only mentions 2025 agent news and misses the qdrant migration, bench rerun, i18n docs, and v0.2 release.

