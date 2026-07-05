[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **plan**

# plan — 查询规划

## 模块职责

把 IR 分析成可执行的 `QueryPlan`：提取所有打分 `Call`、计算 `score_key`、决定 WHERE 子表达式哪些**能下推**到 Store、哪些**必须内存**算。

## 入口与对外接口

```python
from nlql.plan import Planner, QueryPlan, Scorer, score_key, FilterSplit, split_filter, is_pushable, metadata_field
```

- `Planner.plan(query) -> QueryPlan`，`QueryPlan.explain()` 给 EXPLAIN 输出。
- `score_key(call)` — 打分调用的稳定 key（canonical JSON），用于 `unit.scores` 去重 + LET 别名穿透。
- `split_filter(expr, caps) -> FilterSplit` — 按 `StoreCaps` 把 WHERE 拆成 `pushdown`（IR 子集，交给 Store）与 `residual`（内存后置）。

## 关键设计点（DESIGN §6 / §10）

- **provider 型函数收集**：遍历 IR，收集所有 `provides_score=True` 的 Call；LET 别名穿透到绑定表达式。
- **下推判定**：基于能力（`metadata_pushdown` / `text_pushdown`）+ 表达式可表达性；`is_pushable` / `metadata_field` 是判定原语。
- **纯委托优先**：若后端能表达整条查询 → 不拆分；只有用到后端表达不了的能力（自定义 Python 函数谓词、复杂正则、跨源逻辑）才拆分残余。

## 关键依赖与配置

- 上游：`ir` / `registry`（查 capability）；下游：`store`（拿 `StoreCaps`）/ `exec`（消费 `QueryPlan`）。

## 测试与质量

- `tests/test_pushdown.py`（拆分正确性）、`tests/test_exec.py`（plan→execute）、`tests/test_pgvector_store.py`（CONTAINS→ILIKE 下推）、`tests/test_cross_store.py`（EXPLAIN 下推决策差异）。

## 相关文件清单

- [`plan.py`](plan.py) — `QueryPlan` / `Scorer` / `score_key`
- [`planner.py`](planner.py) — `Planner`
- [`pushdown.py`](pushdown.py) — `FilterSplit` / `split_filter` / `is_pushable` / `metadata_field`

## 变更记录

- 2026-07-05：首次生成。
