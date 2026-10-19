"""How a policy's declared output columns reach DocETL.

A declared column is represented through two channels: the operation's `output.schema`, which DocETL turns
into the tool signature the model fills, and the prompt, which carries its description. Both are built here,
so a change to one is visible from the other.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_memory.policy.logical import ColumnSpec, QueryExpr

# Reason: `policy/logical.py::ColumnSpec` has no type field, so every declared column is asked for as text:
#   a policy needing a number or boolean cannot say so yet, and inferring one from the column name was the
#   alternative (a wrong guess drops rows). When `ColumnSpec` grows a type, this is the one place to change.
SCALAR_TYPE = "string"


def declared_outputs(query: QueryExpr) -> tuple[ColumnSpec, ...]:
    """Return the columns an operator adds, which are the ones the model must fill."""

    declared = query.params.get("output_cols")
    if not declared:
        raise ValueError(f"{query.op} requires output_cols")
    return tuple(declared)


def output_column_names(query: QueryExpr, source_columns: Sequence[str]) -> list[str]:
    """Return the frame's columns: the input's, then the declared ones.

    Deriving them from the frame rather than from `policy.schema.output_columns` keeps this correct when
    the input is a bare `log`, whose columns are known only at execution time.
    """

    columns = list(source_columns)
    for column in declared_outputs(query):
        if column.name not in columns:
            columns.append(column.name)
    return columns


def scalar_schema(columns: Sequence[ColumnSpec]) -> dict[str, str]:
    """Return a DocETL `output.schema` for a row-local operator, one field per declared column."""

    return {column.name: SCALAR_TYPE for column in columns}


def rows_type(columns: Sequence[ColumnSpec]) -> str:
    """Return the DocETL type of a list of emitted rows, which is how a 1:1 `map` produces 0..n rows.

    `docetl.operations.utils.validation::convert_val` parses `list[{a: string}]` into an array whose items
    are objects with those properties, and validates them recursively.
    """

    fields = ", ".join(f"{column.name}: {SCALAR_TYPE}" for column in columns)
    return f"list[{{{fields}}}]"


def describe_outputs(columns: Sequence[ColumnSpec]) -> str:
    """Return the output columns as prompt lines.

    DocETL's schema is `{name: type}` and carries no descriptions, so the prompt is the only route by which
    a policy's intended meaning reaches the model.
    """

    described = [
        f"- {column.name}: {column.description}" for column in columns if column.description
    ]
    if not described:
        return ""
    return "Fill these fields:\n" + "\n".join(described)
