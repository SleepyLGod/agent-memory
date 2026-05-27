"""Smoke demo for the current ClaudeMemory v0.0 interface."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from sys import path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

import agent_memory as am  # noqa: E402
from agent_memory.logical import QueryExpr  # noqa: E402


def _format_value(value: Any) -> Any:
    if isinstance(value, tuple) and all(hasattr(item, "name") for item in value):
        return [item.name for item in value]
    return value


def _format_params(params: Mapping[str, Any]) -> str:
    visible = {
        key: _format_value(value)
        for key, value in params.items()
        if key in {"columns", "input_cols", "output_cols", "key", "instruction", "k"}
    }
    if not visible:
        return ""
    return " " + repr(visible)


def print_expr(expr: QueryExpr, *, indent: int = 0) -> None:
    """Print a compact query tree."""

    prefix = "  " * indent
    print(f"{prefix}- {expr.op}{_format_params(expr.params)}")
    for input_expr in expr.inputs:
        print_expr(input_expr, indent=indent + 1)


def main() -> None:
    """Inspect ClaudeMemory's logical interface without executing it."""

    spec = am.ClaudeMemory.spec()
    log_columns = [column.name for column in spec.log.expr.params["columns"]]

    print("ClaudeMemory interface smoke")
    print(f"log columns: {log_columns}")
    print(f"private relations: {sorted(spec.private_relations)}")
    print(f"views: {sorted(spec.views)}")

    for name, view in spec.views.items():
        print(f"\nview: {name}")
        print_expr(view.query)

    memory = am.ClaudeMemory()

    try:
        memory.add(am.Message(content="Please remember concise design docs."))
    except NotImplementedError as error:
        print(f"\nadd: {error}")

    try:
        memory.query("design docs")
    except NotImplementedError as error:
        print(f"query: {error}")


if __name__ == "__main__":
    main()
