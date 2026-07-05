[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **model**

# model — 模态无关数据模型

## 模块职责

定义贯穿全系统的核心数据结构。**从第一天起模态无关**——文本只是 `Payload` 的一种 modality；修复 v1 "全系统 `content: str`、无多模态入口"。

## 入口与对外接口

```python
from nlql import Document, Payload, Unit, Modality, Vector, Span, UnitKind
```

- `Modality(str, Enum)`：`TEXT / IMAGE / BLOB`。
- `Payload(modality, data: str | bytes, mime=None)`；`Payload.text(s)` 便捷构造。
- `Document(id, payloads: list[Payload], metadata: dict, source=None)`；`Document.from_text(...)`。
- `Unit(id, doc_id, kind, payload, vector=None, vectors=None, span=None, metadata, scores, ordinal)`：
  - `kind` ∈ `document / chunk / sentence / span`（检索/返回的基本单位）。
  - `vector` — 默认向量；`vectors: dict[name→Vector]` — 命名多向量（M5）。
  - `scores: dict[score_key → float]` — **具名分数**（替代 v1 "塞进 metadata 的魔法字段"），支持一查询多个语义分数。
  - `metadata: dict` — 业务元数据（用户空间）；系统字段（doc_id/kind/span）分离，杜绝污染。
  - `get_vector(name)` — 取命名向量（缺省回退默认）。
- `Vector` / `as_array` / `normalize` / `to_list`（`vector.py`）。
- `Span` / `UnitKind`（`unit.py`）。

## 关键设计点（DESIGN §3）

- **`content: str` → `Payload{modality, data}`**：一步到位多模态。
- **`scores` 字典**：让"多语义查询 + 按某分数排序"成为一等能力；根治 v1 "全局仅一个 SIMILAR_TO 分数"。
- **系统字段 vs 业务 metadata 分离**。

## 关键依赖与配置

- `numpy`（Vector）；被几乎所有其它模块依赖。

## 测试与质量

- `tests/test_model.py`（构造 / 序列化 / 命名向量 / 模态）。

## 相关文件清单

- [`document.py`](document.py) — `Document`
- [`payload.py`](payload.py) — `Payload` / `Modality`
- [`unit.py`](unit.py) — `Unit` / `UnitKind` / `Span`
- [`vector.py`](vector.py) — `Vector` / `as_array` / `normalize` / `to_list`

## 变更记录

- 2026-07-05：首次生成。
