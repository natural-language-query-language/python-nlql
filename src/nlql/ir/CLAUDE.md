[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **ir**

# ir — Query IR：规范中间表示

## 模块职责

Query IR 是 NLQL 查询的**规范形态**——可 JSON 序列化、可 EXPLAIN、是 LLM function-calling 的载体。三种入口（NLQL 字符串、Query Builder、LLM JSON）都编译到它。

## 入口与对外接口

```python
from nlql.ir import Query, Select, Binding, OrderKey, Expr, Literal, Path, Ref, Call, Compare, And, Or, Not, expr_from_dict, query_json_schema
```

- 节点类型（`nodes.py`，dataclass）：`Query / Select / Binding / OrderKey`（结构）+ `Expr / Literal / Path / Ref / Call / Compare / And / Or / Not`（表达式代数）。
- `Query.from_dict(d)` / `query.to_dict()` — IR ↔ dict（LLM 载体）。
- `expr_from_dict(d)` — 表达式节点反序列化。
- `query_json_schema(call_names)` — 产出 IR 的 JSON Schema，可把 `Call` 名限制为某 registry 已注册函数集（用于 `Engine.function_schema()`）。

## 关键设计点

- **正交表达式代数**：四类正交构件——`Path`（`content` / `meta.status`）/ `Call`（标量与谓词函数，统一 `NAME(args)`）/ `Compare` / `And-Or-Not`。**无 special-case**。
- **具名分数**：`Binding(name, expr)` 在 `Query.let`；别名让 `WHERE`/`ORDER BY` 复用分数；多语义查询 + 组合排序成为一等能力。
- **`Ref` vs `Path`**：解析后 LET 别名集把 `Path(alias)`→`Ref`（如 `relevance`），`content`/`meta.status` 保持 `Path`。
- **provider 型 Call**：`Call("SIMILARITY", ...)` 自身无 row-wise impl；Executor 在召回阶段用 `score_key = canonical(call)`（其 `to_dict()` 稳定 JSON）去重并填入 `unit.scores`。

## 关键依赖与配置

- 仅 Python 标准库；不依赖 parser/store。

## 数据模型

- IR 节点是 dataclass；序列化为 JSON 即 function-calling 参数。

## 测试与质量

- `tests/test_ir.py`（节点 / round-trip / JSON Schema）、`tests/test_lang.py`（字符串→IR）、`tests/test_builder.py`（Builder→IR）、`tests/test_expressions.py`（表达式代数）。

## 相关文件清单

- [`nodes.py`](nodes.py) — 全部 IR 节点 + `from_dict`/`to_dict`
- [`schema.py`](schema.py) — `query_json_schema`（JSON Schema 导出）
- [`__init__.py`](__init__.py)

## 变更记录

- 2026-07-05：首次生成。
