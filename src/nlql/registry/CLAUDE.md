[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **registry**

# registry — 统一能力注册中心

## 模块职责

所有扩展点的**单一注册中心**：四类 capability 同构注册（`function` / `splitter` / `embedder` / `modality`）。`function` 已吸收 v1 的 operator——`CONTAINS/MATCH/LIKE` 都是返回 BOOL 的函数。

## 入口与对外接口

```python
from nlql import GLOBAL_REGISTRY, Registry, register_function, register_splitter

@GLOBAL_REGISTRY.function("WORD_COUNT", signature=Signature((TypeTag.TEXT,), TypeTag.NUMBER))
def word_count(s): ...

engine.register_function("MY_FN", provides_score=False, pushdownable=True)  # 实例作用域
```

- `Registry.register(kind, name, *, impl, signature, provides_score=False, pushdownable=False, overwrite=False)`。
- `Capability` dataclass：`kind / name / impl / signature / provides_score / pushdownable`。
- 装饰器糖：`registry.function(name, ...)` / `registry.splitter(name, ...)`。
- `Registry.child()` — 父子链作用域；查找 self→parent；实例注册天然 shadow 全局，互不泄漏。
- `registry.names(kind)` — 列出已注册名（供 JSON Schema 收紧 Call 名）。

## 关键设计点（DESIGN §8）

- **作用域用父子链而非 `scope` 字符串**：进程级 `GLOBAL_REGISTRY` 为根，每个 `Engine` 持一个 child；`instance > global` 自然成立。
- **内置与用户扩展走同一注册路径**——删除 v1 的 placeholder 死代码。
- **Parser/Planner 从 Registry 取**"可调用名 + 签名 + 可下推性"——实现"注册即可用 + 参与下推决策"。
- 导入 `nlql.registry` 即副作用 seed 内置函数（`builtins.py`：CONTAINS/SIMILARITY/LENGTH/...）。

## 关键依赖与配置

- 上游：`types`（Signature/TypeTag）；被 `lang` / `plan` / `exec` / `ingest` / `sdk` 依赖。

## 测试与质量

- `tests/test_registry.py`（注册 / 作用域 / 装饰器 / shadow）。

## 相关文件清单

- [`core.py`](core.py) — `Registry` / `Capability` / `GLOBAL_REGISTRY` / `register_function` / `register_splitter` / `CAPABILITY_KINDS`
- [`builtins.py`](builtins.py) — 内置函数 seed（CONTAINS / SIMILARITY / LENGTH / ...）

## 变更记录

- 2026-07-05：首次生成。
