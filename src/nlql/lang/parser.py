"""NLQL string front-end: parse an NLQL/2 query into Query IR."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path as _FsPath
from typing import Any

from lark import Lark
from lark.exceptions import LarkError, UnexpectedInput, VisitError

from nlql.errors import NLQLError, NLQLParseError
from nlql.ir.nodes import And, Call, Compare, Expr, Not, Or, Path, Query, Ref
from nlql.lang.transformer import NLQLTransformer

_GRAMMAR_FILE = _FsPath(__file__).parent / "grammar.lark"


def _resolve_expr(e: Expr, aliases: frozenset[str]) -> Expr:
    """Rewrite bare paths that name a LET alias into Ref nodes."""
    if isinstance(e, Path):
        if not e.segments and e.root in aliases:
            return Ref(e.root)
        return e
    if isinstance(e, Call):
        e.args = [_resolve_expr(a, aliases) for a in e.args]
        return e
    if isinstance(e, Compare):
        e.left = _resolve_expr(e.left, aliases)
        e.right = _resolve_expr(e.right, aliases)
        return e
    if isinstance(e, (And, Or)):
        e.operands = [_resolve_expr(o, aliases) for o in e.operands]
        return e
    if isinstance(e, Not):
        e.operand = _resolve_expr(e.operand, aliases)
        return e
    return e  # Literal, Ref — nothing to resolve


def _resolve_query(q: Query) -> Query:
    aliases = frozenset(b.name for b in q.let)
    for b in q.let:
        b.expr = _resolve_expr(b.expr, aliases)
    if q.where is not None:
        q.where = _resolve_expr(q.where, aliases)
    for key in q.order_by:
        key.expr = _resolve_expr(key.expr, aliases)
    return q


class NLQLParser:
    """Parses NLQL/2 strings into :class:`~nlql.ir.nodes.Query` IR."""

    def __init__(self) -> None:
        grammar = _GRAMMAR_FILE.read_text(encoding="utf-8")
        self._parser = Lark(grammar, parser="lalr", start="start", propagate_positions=True)
        self._transformer = NLQLTransformer()

    def parse(self, query: str) -> Query:
        """Parse a query string into resolved Query IR.

        Raises:
            NLQLParseError: on any syntax or transform error.
        """
        try:
            tree = self._parser.parse(query)
            result: Query = self._transformer.transform(tree)
        except VisitError as e:
            if isinstance(e.orig_exc, NLQLError):
                raise e.orig_exc from e
            raise NLQLParseError(f"parse error: {e}") from e
        except UnexpectedInput as e:
            self._raise_parse_error(e, query)
        except LarkError as e:
            raise NLQLParseError(f"parse error: {e}") from e
        return _resolve_query(result)

    @staticmethod
    def _raise_parse_error(error: Any, query: str) -> None:
        line = getattr(error, "line", None)
        column = getattr(error, "column", None)
        context = None
        if line is not None:
            lines = query.split("\n")
            if 0 < line <= len(lines):
                rendered = []
                for i in range(max(0, line - 2), min(len(lines), line + 1)):
                    prefix = ">>> " if i == line - 1 else "    "
                    rendered.append(f"{prefix}{lines[i]}")
                context = "\n".join(rendered)
        raise NLQLParseError(str(error).strip(), line=line, column=column, context=context) from error


@lru_cache(maxsize=1)
def _default_parser() -> NLQLParser:
    return NLQLParser()


def parse(query: str) -> Query:
    """Parse an NLQL/2 string into Query IR using a shared parser instance."""
    return _default_parser().parse(query)
