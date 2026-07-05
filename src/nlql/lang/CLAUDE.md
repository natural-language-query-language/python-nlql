[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **lang**

# lang — NLQL 字符串前端

## 模块职责

把 NLQL 字符串（NLQL/2 语法）解析为 `Query IR`。文法由 lark 定义，函数名**不写死进文法**——`IDENT(args)` 通用调用，合法性在语义阶段查 Registry。

## 入口与对外接口

```python
from nlql import parse
from nlql.lang import NLQLParser
q = parse('SELECT SENTENCE LET rel = SIMILARITY(content, "x") WHERE rel >= 0.8 ORDER BY rel DESC LIMIT 5')
parser = NLQLParser(); parser.parse(text)
```

## NLQL/2 语法要点

```sql
SELECT SPAN(SENTENCE, window => 1)
LET   relevance = SIMILARITY(content, "AI Agent architecture"),
      novelty   = SIMILARITY(content, "novel unpublished ideas")
WHERE relevance >= 0.8
  AND meta.status != "draft"
  AND (content CONTAINS "planning" OR content CONTAINS "memory")
ORDER BY relevance DESC, novelty DESC
LIMIT 5
```

- **关键字大写、大小写敏感**（避免与小写 metadata 键 `meta.status` 冲突）。
- **`LET` 显式声明具名分数** → 可多个、可组合排序（根治 v1 "只能一个 SIMILAR_TO"）。
- **`meta.x` 与 `content` 同为 Path**，元数据统一为点路径，无 `META("x")` 函数形态。
- **中缀谓词 `content CONTAINS "x"` 只是语法糖** → `Call("CONTAINS", [content, "x"])`；`CONTAINS/MATCH/LIKE` 是返回 BOOL 的内置**函数**。
- 关键字参数用 `=>`（如 `window => 1`）。
- `IDENT(args)` 通用调用 → 新增函数无需改文法。

## 关键依赖与配置

- `lark>=1.1.0`（核心依赖）。
- 文件：`grammar.lark`（文法）、`parser.py`（lark 加载 + 缓存）、`transformer.py`（lark Tree → IR）。

## 测试与质量

- `tests/test_lang.py`（解析正确性 + 边界 + round-trip 与 Builder/IR 一致）。

## 相关文件清单

- [`grammar.lark`](grammar.lark) — lark 文法
- [`parser.py`](parser.py) — `NLQLParser` / `parse`
- [`transformer.py`](transformer.py) — lark Tree → IR 节点

## 变更记录

- 2026-07-05：首次生成。
