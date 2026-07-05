"""Eval dataset: documents with metadata + questions across 6 scenarios.

Each question has a **knowledge question** (what the answer LLM answers and the
judge scores) and an independent **retrieval spec** (the NLQL form + LangChain
filter). Retrieval conditions never leak into the knowledge question.

Scenarios:
  - semantic  : pure similarity, no filter
  - hybrid    : semantic + metadata (status) equality filter
  - date      : semantic + date range (>= / between)
  - keyword   : CONTAINS (full-text substring)
  - composite : semantic + status + date range
  - vague     : fuzzy natural-language queries ("what did I ship", "what's pending")
                with multiple expected docs — tests recall completeness AND precision
                (no false positives) under loose intent + hard filters

Metrics tracked per cell:
  - score     : LLM-judge answer accuracy (0-5)
  - hit       : recall — share of expected docs present in top-k (completeness)
  - precision : share of retrieved docs that satisfy the scenario's conditions
                (status / date range / CONTAINS) — exposes false positives
"""

from __future__ import annotations

DOCUMENTS: list[tuple[str, str, dict]] = [
    # --- AI agents ---
    ("agt-planning", "AI agents plan multi-step tasks by decomposing high-level goals into ordered subtasks.",
     {"status": "published", "date": "2024-03-10", "category": "agents"}),
    ("agt-memory", "Modern AI agents keep a working memory of past actions to stay coherent across turns.",
     {"status": "published", "date": "2024-05-22", "category": "agents"}),
    ("agt-tools", "Tool use lets an AI agent call external APIs and act on the world.",
     {"status": "draft", "date": "2024-02-14", "category": "agents"}),
    ("agt-multi", "Multi-agent systems split a complex goal across several specialized agents.",
     {"status": "published", "date": "2023-11-05", "category": "agents"}),
    ("agt-reflect", "Self-reflection lets an agent critique its own outputs and retry on failure.",
     {"status": "draft", "date": "2025-01-18", "category": "agents"}),
    # --- RAG ---
    ("rag-index", "RAG indexing splits documents into chunks and embeds them into a vector store.",
     {"status": "published", "date": "2024-01-30", "category": "rag"}),
    ("rag-retrieve", "RAG retrieval selects the top-k chunks most relevant to a query before generation.",
     {"status": "published", "date": "2024-07-12", "category": "rag"}),
    ("rag-rerank", "A cross-encoder reranker refines the top-k candidates for higher accuracy.",
     {"status": "published", "date": "2025-02-08", "category": "rag"}),
    ("rag-chunk", "Chunk size affects retrieval quality: small chunks stay focused, large ones keep context.",
     {"status": "draft", "date": "2024-04-03", "category": "rag"}),
    ("rag-eval", "RAG evaluation measures both retrieval recall and answer faithfulness.",
     {"status": "published", "date": "2024-09-19", "category": "rag"}),
    # --- transformers / LLMs ---
    ("tx-attention", "The attention mechanism lets a model weigh different tokens when producing each output.",
     {"status": "published", "date": "2023-06-11", "category": "transformers"}),
    ("tx-arch", "The transformer architecture stacks self-attention and feed-forward layers.",
     {"status": "published", "date": "2023-08-21", "category": "transformers"}),
    ("tx-bert", "BERT is an encoder-only transformer pretrained with masked language modeling.",
     {"status": "published", "date": "2023-05-09", "category": "transformers"}),
    ("llm-gpt", "GPT models are decoder-only transformers trained to predict the next token.",
     {"status": "published", "date": "2024-11-15", "category": "transformers"}),
    ("llm-moe", "Mixture of Experts (MoE) routes each token to a sparse subset of parameters.",
     {"status": "draft", "date": "2025-03-01", "category": "transformers"}),
    # --- other domains (distractors + variety) ---
    ("sci-physics", "Quantum entanglement links particles so that measuring one affects the other.",
     {"status": "published", "date": "2023-09-14", "category": "science"}),
    ("sci-bio", "CRISPR lets researchers edit genes at precise positions in the genome.",
     {"status": "published", "date": "2024-06-27", "category": "science"}),
    ("sci-chem", "Enzymes lower the activation energy of reactions inside living cells.",
     {"status": "draft", "date": "2024-10-02", "category": "science"}),
    ("news-2023", "In 2023 a record number of open-source LLMs were released to the public.",
     {"status": "published", "date": "2023-12-20", "category": "news"}),
    ("news-2024", "During 2024 multimodal models that read images and text became mainstream.",
     {"status": "published", "date": "2024-12-15", "category": "news"}),
    ("news-2025", "In 2025 AI agents that operate browsers and tools autonomously took off.",
     {"status": "published", "date": "2025-05-30", "category": "news"}),
    ("cook-bread", "Banana bread needs flour, sugar, and about forty minutes to bake.",
     {"status": "published", "date": "2024-02-01", "category": "cooking"}),
    ("cook-pasta", "Pasta water should be salted generously before adding the noodles.",
     {"status": "draft", "date": "2024-07-08", "category": "cooking"}),
    ("travel-jp", "Kyoto's temples are quietest in the early morning before tour groups arrive.",
     {"status": "published", "date": "2025-04-12", "category": "travel"}),
    # --- todos / work log (for the 'vague' scenario; multi-doc recall + precision) ---
    ("todo-rag-doc", "Wrote the RAG retrieval documentation draft.",
     {"status": "published", "date": "2024-11-20", "category": "todo", "priority": "high", "done": "true"}),
    ("todo-eval-harness", "Built the retrieval eval harness comparing frameworks.",
     {"status": "published", "date": "2024-12-05", "category": "todo", "priority": "high", "done": "true"}),
    ("todo-qdrant-migrate", "Migrate the store backend to Qdrant for production scale.",
     {"status": "published", "date": "2025-01-10", "category": "todo", "priority": "high", "done": "false"}),
    ("todo-bench-rerun", "Re-run benchmarks after the columnar filter change.",
     {"status": "draft", "date": "2025-02-15", "category": "todo", "priority": "medium", "done": "false"}),
    ("todo-i18n-docs", "Translate the docs site into English.",
     {"status": "published", "date": "2025-03-01", "category": "todo", "priority": "medium", "done": "true"}),
    ("todo-release-02", "Cut the v0.2 release and publish python-nlql to PyPI.",
     {"status": "published", "date": "2025-03-20", "category": "todo", "priority": "high", "done": "false"}),
    ("todo-rerank-fix", "Investigate the reranker regression on short queries.",
     {"status": "draft", "date": "2024-10-08", "category": "todo", "priority": "low", "done": "false"}),
    ("todo-meeting", "Meeting notes: roadmap sync with the platform team.",
     {"status": "published", "date": "2024-09-30", "category": "todo", "priority": "medium", "done": "true"}),
]


# Each question:
#   q          : a knowledge question the answer LLM answers and the judge scores
#   nlql       : the NLQL retrieval (hand-written, fair)
#   filter     : equivalent LangChain equality filter (or None)
#   expected   : ground-truth doc ids the retrieval should surface
#   points     : key factual points the judge checks the answer against
QUESTIONS: list[dict] = [
    # 1. semantic — pure similarity
    {"scenario": "semantic", "q": "How do AI agents plan multi-step tasks?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "agent planning multi-step task decomposition") ORDER BY rel DESC LIMIT 3',
     "filter": None, "expected": ["agt-planning"],
     "points": "agents decompose high-level goals into ordered subtasks"},
    {"scenario": "semantic", "q": "What does RAG retrieval do?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "RAG retrieval top-k relevant chunks") ORDER BY rel DESC LIMIT 3',
     "filter": None, "expected": ["rag-retrieve"],
     "points": "selects the top-k chunks most relevant to a query before generation"},
    {"scenario": "semantic", "q": "How does the attention mechanism work?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "attention mechanism weighing tokens") ORDER BY rel DESC LIMIT 3',
     "filter": None, "expected": ["tx-attention"],
     "points": "lets a model weigh different tokens when producing each output"},

    # 2. hybrid — semantic + status filter
    {"scenario": "hybrid", "q": "How do AI agents keep memory across turns?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "agent memory past actions") WHERE meta.status == "published" ORDER BY rel DESC LIMIT 3',
     "filter": {"status": "published"}, "expected": ["agt-memory"],
     "points": "agents keep a working memory of past actions to stay coherent"},
    {"scenario": "hybrid", "q": "How does RAG retrieve relevant chunks?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "RAG retrieval chunks") WHERE meta.status == "published" ORDER BY rel DESC LIMIT 3',
     "filter": {"status": "published"}, "expected": ["rag-retrieve"],
     "points": "selects the top-k chunks most relevant to a query before generation"},
    {"scenario": "hybrid", "q": "How is the transformer architecture structured?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "transformer architecture layers") WHERE meta.status == "published" ORDER BY rel DESC LIMIT 3',
     "filter": {"status": "published"}, "expected": ["tx-arch"],
     "points": "stacks self-attention and feed-forward layers"},

    # 3. date — semantic + date range. LangChain's retriever has no native range query.
    {"scenario": "date", "q": "How do AI agents plan tasks?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "agent planning tasks") WHERE meta.date >= "2024-01-01" ORDER BY rel DESC LIMIT 3',
     "filter": None, "filter_date_gte": "2024-01-01", "expected": ["agt-planning"],
     "points": "agents decompose high-level goals into ordered subtasks"},
    {"scenario": "date", "q": "How does RAG retrieval work?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "RAG retrieval") WHERE meta.date >= "2024-01-01" ORDER BY rel DESC LIMIT 3',
     "filter": None, "filter_date_gte": "2024-01-01", "expected": ["rag-retrieve"],
     "points": "selects the top-k chunks most relevant to a query before generation"},
    {"scenario": "date", "q": "How do GPT models predict text?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "GPT decoder next token") WHERE meta.date >= "2024-01-01" AND meta.date <= "2024-12-31" ORDER BY rel DESC LIMIT 3',
     "filter": None, "filter_date_gte": "2024-01-01", "expected": ["llm-gpt"],
     "points": "decoder-only transformers trained to predict the next token"},

    # 4. keyword — CONTAINS. LangChain retriever is semantic, not lexical.
    {"scenario": "keyword", "q": "How is the transformer architecture built?",
     "nlql": 'SELECT SENTENCE WHERE content CONTAINS "transformer" LIMIT 5',
     "filter": None, "expected": ["tx-arch"],
     "points": "stacks self-attention and feed-forward layers"},
    {"scenario": "keyword", "q": "How does the attention mechanism weigh tokens?",
     "nlql": 'SELECT SENTENCE WHERE content CONTAINS "attention" LIMIT 5',
     "filter": None, "expected": ["tx-attention"],
     "points": "lets a model weigh different tokens when producing each output"},
    {"scenario": "keyword", "q": "How does mixture of experts route each token?",
     "nlql": 'SELECT SENTENCE WHERE content CONTAINS "Experts" LIMIT 5',
     "filter": None, "expected": ["llm-moe"],
     "points": "routes each token to a sparse subset of parameters"},

    # 5. composite — semantic + status + date range
    {"scenario": "composite", "q": "How does RAG retrieval select chunks?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "RAG retrieval chunks") WHERE meta.status == "published" AND meta.date >= "2024-01-01" ORDER BY rel DESC LIMIT 3',
     "filter": {"status": "published"}, "filter_date_gte": "2024-01-01", "expected": ["rag-retrieve"],
     "points": "selects the top-k chunks most relevant to a query before generation"},
    {"scenario": "composite", "q": "How do AI agents plan multi-step tasks?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "agent planning tasks") WHERE meta.status == "published" AND meta.date >= "2024-01-01" AND meta.date <= "2024-12-31" ORDER BY rel DESC LIMIT 3',
     "filter": {"status": "published"}, "filter_date_gte": "2024-01-01", "expected": ["agt-planning"],
     "points": "agents decompose high-level goals into ordered subtasks"},
    {"scenario": "composite", "q": "How is RAG retrieval evaluated?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "RAG evaluation") WHERE meta.status == "published" AND meta.date >= "2024-01-01" ORDER BY rel DESC LIMIT 3',
     "filter": {"status": "published"}, "filter_date_gte": "2024-01-01", "expected": ["rag-eval"],
     "points": "measures both retrieval recall and answer faithfulness"},

    # 6. vague — fuzzy natural-language queries, multiple expected docs.
    #    Tests recall completeness (did it surface ALL relevant items?) and
    #    precision (did it sneak in items that violate the hard filter?).
    {"scenario": "vague", "q": "What have I shipped recently?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "shipped completed work done") WHERE meta.done == "true" ORDER BY rel DESC LIMIT 5',
     "filter": {"done": "true"}, "expected": ["todo-rag-doc", "todo-eval-harness", "todo-i18n-docs", "todo-meeting"],
     "points": "completed items: RAG docs, eval harness, i18n docs, meeting notes"},
    {"scenario": "vague", "q": "What's still pending on my plate?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "pending open todo task") WHERE meta.done == "false" ORDER BY rel DESC LIMIT 5',
     "filter": {"done": "false"}, "expected": ["todo-qdrant-migrate", "todo-bench-rerun", "todo-release-02", "todo-rerank-fix"],
     "points": "pending items: qdrant migration, bench rerun, v0.2 release, reranker fix"},
    {"scenario": "vague", "q": "What's coming up in 2025?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "2025 planned upcoming work") WHERE meta.date >= "2025-01-01" ORDER BY rel DESC LIMIT 5',
     "filter": None, "filter_date_gte": "2025-01-01",
     "expected": ["todo-qdrant-migrate", "todo-bench-rerun", "todo-i18n-docs", "todo-release-02", "news-2025"],
     "points": "2025 items: qdrant migration, bench rerun, i18n docs, v0.2 release, 2025 agent news"},

    # 7. negation — exclusion (status != "draft"). LangChain's retriever has no
    #    native != filter, so drafts leak into its results.
    {"scenario": "negation", "q": "Show me the published agent docs (not the drafts).",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "AI agent") WHERE meta.status != "draft" ORDER BY rel DESC LIMIT 3',
     "filter": None, "filter_not": {"status": "draft"}, "expected": ["agt-planning", "agt-memory"],
     "points": "agent docs that are published: planning, memory (NOT the draft ones)"},
    {"scenario": "negation", "q": "Which RAG docs are published (exclude drafts)?",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "RAG retrieval indexing") WHERE meta.status != "draft" ORDER BY rel DESC LIMIT 5',
     "filter": None, "filter_not": {"status": "draft"},
     "expected": ["rag-index", "rag-retrieve", "rag-rerank", "rag-eval"],
     "points": "published RAG docs: indexing, retrieval, reranking, evaluation (NOT the draft chunking one)"},
    {"scenario": "negation", "q": "List the published transformer docs (no drafts).",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "transformer architecture attention") WHERE meta.status != "draft" ORDER BY rel DESC LIMIT 5',
     "filter": None, "filter_not": {"status": "draft"},
     "expected": ["tx-attention", "tx-arch", "tx-bert", "llm-gpt"],
     "points": "published transformer docs: attention, architecture, BERT, GPT (NOT the draft MoE one)"},

    # 8. distractor — high-similarity noise: several docs are semantically close
    #    but only a subset satisfies the hard conditions; precision under pressure.
    {"scenario": "distractor", "q": "Published RAG docs from 2024 (there are draft and 2025 ones too).",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "RAG retrieval indexing rerank") WHERE meta.status == "published" AND meta.date >= "2024-01-01" AND meta.date <= "2024-12-31" ORDER BY rel DESC LIMIT 5',
     "filter": {"status": "published"}, "filter_date_gte": "2024-01-01", "filter_date_lte": "2024-12-31",
     "expected": ["rag-index", "rag-retrieve", "rag-eval"],
     "points": "published 2024 RAG docs: indexing, retrieval, evaluation (NOT the draft chunking or the 2025 rerank)"},
    {"scenario": "distractor", "q": "Published agent docs from 2024 only.",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "AI agent") WHERE meta.status == "published" AND meta.date >= "2024-01-01" AND meta.date <= "2024-12-31" ORDER BY rel DESC LIMIT 5',
     "filter": {"status": "published"}, "filter_date_gte": "2024-01-01", "filter_date_lte": "2024-12-31",
     "expected": ["agt-planning", "agt-memory"],
     "points": "published 2024 agent docs: planning, memory (NOT the 2024 draft tools or the 2023 multi-agent)"},
    {"scenario": "distractor", "q": "Show me the high-priority todos that are done.",
     "nlql": 'SELECT SENTENCE LET rel = SIMILARITY(content, "completed high priority todo work") WHERE meta.priority == "high" AND meta.done == "true" ORDER BY rel DESC LIMIT 5',
     "filter": {"priority": "high", "done": "true"},
     "expected": ["todo-rag-doc", "todo-eval-harness"],
     "points": "high-priority AND done todos: RAG docs, eval harness (NOT the pending high-priority migration or the done medium ones)"},
]

SCENARIOS = ["semantic", "hybrid", "date", "keyword", "composite", "vague", "negation", "distractor"]
