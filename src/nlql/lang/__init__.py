"""NLQL string front-end: grammar, transformer, and parser to Query IR."""

from nlql.lang.parser import NLQLParser, parse

__all__ = ["NLQLParser", "parse"]
