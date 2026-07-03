"""Shared QueryExpr output-column inference helpers."""

from __future__ import annotations

from collections.abc import Sequence

from agent_memory.logical import ColumnSpec, QueryExpr


def output_columns(
    query: QueryExpr,
    *,
    window_source_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Infer best-known output columns for a logical query expression."""

    if query.op == "select":
        return tuple(str(column) for column in query.params["columns"])
    if query.op == "log":
        return tuple(column.name for column in query.params.get("columns", ()))
    if query.op == "window_source":
        return tuple(
            str(column)
            for column in query.params.get("columns", window_source_columns or ())
        )
    if query.op == "materialized_view":
        return tuple(str(column) for column in query.params.get("columns", ()))
    if query.op == "over":
        return output_columns(query.inputs[0], window_source_columns=window_source_columns)
    if query.op == "array_agg":
        if query.inputs and query.inputs[0].op == "over":
            return _append_unique(
                output_columns(query.inputs[0], window_source_columns=window_source_columns),
                (str(query.params["output_col"]),),
            )
        return (str(query.params["output_col"]),)
    if query.op == "array_cat":
        return (str(query.params["column"]),)
    if query.op == "sem_agg":
        output_cols = query.params.get("output_cols")
        if query.inputs and query.inputs[0].op == "over":
            if output_cols is None:
                raise NotImplementedError("over sem_agg requires explicit output_cols")
            return _append_unique(
                output_columns(query.inputs[0], window_source_columns=window_source_columns),
                _column_names(output_cols),
            )
        if output_cols is not None:
            return _column_names(output_cols)
        input_cols = query.params.get("input_cols")
        if input_cols is not None:
            return tuple(str(column) for column in input_cols)
    if query.op in {"sem_map", "sem_flat_map"}:
        return _append_unique(
            output_columns(query.inputs[0], window_source_columns=window_source_columns),
            _column_names(query.params.get("output_cols") or ()),
        )
    if query.op in {
        "count_window",
        "sem_filter",
        "sem_groupby",
        "sem_topk",
        "drop_duplicates",
        "filter",
        "assign",
    }:
        return output_columns(query.inputs[0], window_source_columns=window_source_columns)
    if query.op == "process_window":
        return output_columns(query.inputs[1], window_source_columns=window_source_columns)
    if query.op == "join":
        return join_output_columns(query, window_source_columns=window_source_columns)
    if query.op == "sem_join":
        return semantic_join_columns(
            query,
            window_source_columns=window_source_columns,
        )
    if query.op in {"union", "concat", "subtract"}:
        columns: list[str] = []
        for input_query in query.inputs:
            for column in output_columns(
                input_query,
                window_source_columns=window_source_columns,
            ):
                if column not in columns:
                    columns.append(column)
        return tuple(columns)
    raise NotImplementedError(f"Cannot infer output columns for QueryExpr op {query.op!r}.")


def join_output_columns(
    query: QueryExpr,
    *,
    window_source_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Infer pandas merge output columns for deterministic same-key joins."""

    left_columns = output_columns(
        query.inputs[0],
        window_source_columns=window_source_columns,
    )
    right_columns = output_columns(
        query.inputs[1],
        window_source_columns=window_source_columns,
    )
    keys = tuple(str(column) for column in query.params["on"])
    overlapping = set(left_columns).intersection(right_columns).difference(keys)

    columns: list[str] = []
    for column in left_columns:
        columns.append(f"{column}:left" if column in overlapping else column)
    for column in right_columns:
        if column in keys:
            continue
        columns.append(f"{column}:right" if column in overlapping else column)
    return tuple(columns)


def semantic_join_columns(
    query: QueryExpr,
    *,
    window_source_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Infer no-key semantic join output columns."""

    left_columns = output_columns(
        query.inputs[0],
        window_source_columns=window_source_columns,
    )
    right_columns = output_columns(
        query.inputs[1],
        window_source_columns=window_source_columns,
    )
    overlapping = set(left_columns).intersection(right_columns)

    columns: list[str] = []
    for column in left_columns:
        columns.append(f"{column}:left" if column in overlapping else column)
    for column in right_columns:
        columns.append(f"{column}:right" if column in overlapping else column)
    return tuple(columns)


def _column_names(columns: object) -> tuple[str, ...]:
    """Return string names from ColumnSpec-like values."""

    return tuple(
        column.name if isinstance(column, ColumnSpec) else str(column)
        for column in columns  # type: ignore[union-attr]
    )


def _append_unique(base: Sequence[str], additions: Sequence[str]) -> tuple[str, ...]:
    """Append columns while preserving first occurrence order."""

    columns = list(base)
    for column in additions:
        if column not in columns:
            columns.append(column)
    return tuple(columns)
