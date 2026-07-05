"""LLM integration: expose NLQL as a function-calling tool and run the IR an LLM returns.

Run: python examples/llm_function_calling.py

The engine hands an LLM a JSON-Schema tool definition. The model emits a *structured*
Query IR (more reliable than generating a query string), which the engine executes
directly via ``search_ir``. This is the recommended agent/RAG retrieval path.
"""

from __future__ import annotations

import json

import nlql


def main() -> None:
    engine = nlql.Engine(nlql.FakeEmbedder())
    engine.add_text("AI agents use planning, memory, and tool use.", metadata={"status": "published"})
    engine.add_text("Vector databases store embeddings for similarity search.", metadata={"status": "published"})

    # 1. What you pass to the LLM as a tool/function definition:
    tool = engine.function_tool(name="nlql_query")
    print("== tool definition handed to the LLM (truncated) ==")
    print(json.dumps({"type": tool["type"], "function": {"name": tool["function"]["name"]}}, indent=2))
    print(f"  (parameters: JSON Schema '{tool['function']['parameters']['title']}' with "
          f"{len(tool['function']['parameters']['$defs'])} node types)\n")

    # 2. What the LLM would return as tool-call arguments — a Query IR document:
    llm_output = {
        "select": {"unit": "sentence"},
        "let": [
            {
                "name": "relevance",
                "expr": {
                    "node": "call",
                    "name": "SIMILARITY",
                    "args": [
                        {"node": "path", "root": "content", "segments": []},
                        {"node": "literal", "value": "how do agents work"},
                    ],
                },
            }
        ],
        "where": {
            "node": "compare",
            "op": "==",
            "left": {"node": "path", "root": "meta", "segments": ["status"]},
            "right": {"node": "literal", "value": "published"},
        },
        "order_by": [{"expr": {"node": "ref", "name": "relevance"}, "desc": True}],
        "limit": 3,
    }

    # 3. Execute it directly — no string parsing, no ambiguity:
    print("== results from executing the LLM's Query IR ==")
    for unit in engine.search_ir(llm_output):
        print(f"  [{unit.scores['relevance']:+.3f}] {unit.content}")


if __name__ == "__main__":
    main()
