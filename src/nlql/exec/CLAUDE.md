[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **exec**

# exec — 执行：Evaluator + Executor

## 模块职责

把 IR 跑出 `Unit` 列表。两个角色：

- **`Evaluator`**：表达式代数求值器，**无 special-case**。
- **`Executor`**：编排"召回 → 谓词过滤 → 粒度变换 → 重排 → LIMIT"。

## 入口与对外接口

主要由 `sdk.Engine` 持有一个 `Executor` 并调用 `.execute(query, *, reranker=..., rerank_query=...)`；`.plan(query)` 产出可 `explain()` 的 `QueryPlan`。

## 关键设计点（DESIGN §6）

- **无特判**：`SIMILARITY` / `meta.*` / `CONTAINS` / 自定义函数都是普通 IR 节点；唯一分支是 `cap.provides_score`——为真读 `unit.scores[score_key(call)]`，为假 `cap.impl(*args)`。能力元数据分派，**非函数名硬编码**。
- **provider 型打分**：Planner 收集所有打分 `Call`，去重后 Executor 在召回阶段把候选向量堆成矩阵与查询向量矩阵**一次 matmul** 得 cosine，写入 `unit.scores`。`LET` 别名穿透到绑定表达式 → 内联与命名调用共享 key、只算一次。
- **归一化向量** ⇒ `matrix @ query` 即 cosine（[-1,1] 原始值，不做 (cos+1)/2 折叠）。
- **短路求值 + 类型规约**：先算便宜谓词（元数据/字符串）；null 参与有序比较则该行落选（SQL 式）。
- **粒度变换**：`SELECT SPAN(SENTENCE, window=>n)` 走 `Store.neighbors(doc_id, ordinal, window)`。

## 关键依赖与配置

- 上游：`plan` / `store` / `registry` / `embed`（查询向量）/ `rerank`（可选）/ `types`（类型规约）。
- M6 列式过滤（`store/columns.py`）让带过滤查询从 83ms 降到 4.4ms（~19×）。

## 测试与质量

- `tests/test_exec.py`（执行全链路）、`tests/test_expressions.py`（表达式求值）、`tests/test_pushdown.py`（下推拆分）、`tests/test_cross_store.py`（跨后端一致）、`tests/test_rerank.py`、`tests/test_multivector.py`（命名 scorer 走 scan + `unit.get_vector(name)`）。

## 相关文件清单

- [`evaluate.py`](evaluate.py) — `Evaluator`（表达式求值）
- [`executor.py`](executor.py) — `Executor`（召回 + 过滤 + 重排 + limit）
- [`__init__.py`](__init__.py)

## 变更记录

- 2026-07-05：首次生成。
