"""Output-column inference for logical query expressions."""

from __future__ import annotations

from collections.abc import Sequence

from .aggregates import (
    ArrayAggregateSpec,
    CollectListAggregateSpec,
    MinAggregateSpec,
    SemanticAggregateSpec,
)
from .logical import ColumnSpec, QueryExpr


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
    if query.op == "search":
        source_columns = output_columns(
            query.inputs[0],
            window_source_columns=window_source_columns,
        )
        metadata_columns = ("record_id", "rank", "score")
        conflicts = sorted(set(source_columns).intersection(metadata_columns))
        if conflicts:
            raise ValueError(
                "search metadata columns conflict with source columns: "
                f"{conflicts}"
            )
        return source_columns + metadata_columns
    if query.op in {"alias", "group_by"}:
        return output_columns(query.inputs[0], window_source_columns=window_source_columns)
    if query.op == "over":
        return output_columns(query.inputs[0], window_source_columns=window_source_columns)
    if query.op == "array_agg":
        if query.inputs and query.inputs[0].op == "over":
            return _append_unique(
                output_columns(query.inputs[0], window_source_columns=window_source_columns),
                (str(query.params["output_col"]),),
            )
        if query.inputs and query.inputs[0].op == "group_by":
            keys = tuple(str(key) for key in query.inputs[0].params["keys"])
            output_col = str(query.params["output_col"])
            if output_col in keys:
                raise ValueError(
                    f"grouped array_agg output column conflicts with group key: {output_col!r}"
                )
            return _append_unique(keys, (output_col,))
        if query.inputs and query.inputs[0].op == "sem_groupby":
            return (str(query.params["output_col"]),)
        return (str(query.params["output_col"]),)
    if query.op == "array_cat":
        return (str(query.params["column"]),)
    if query.op == "min":
        output_col = str(query.params["output_col"])
        if query.inputs and query.inputs[0].op == "group_by":
            keys = tuple(str(key) for key in query.inputs[0].params["keys"])
            if output_col in keys:
                raise ValueError(
                    f"grouped min output column conflicts with group key: {output_col!r}"
                )
            return _append_unique(keys, (output_col,))
        if query.inputs and query.inputs[0].op == "sem_groupby":
            raise NotImplementedError("sem_groupby(...).min(...) is not supported")
        return (output_col,)
    if query.op == "flatten":
        source_columns = output_columns(
            query.inputs[0],
            window_source_columns=window_source_columns,
        )
        column = str(query.params["column"])
        if column not in source_columns:
            raise ValueError(f"flatten input column not found: {column!r}")
        output_col = query.params.get("output_col")
        if output_col is None:
            return source_columns
        if str(output_col) != column and str(output_col) in source_columns:
            raise ValueError(f"flatten output column already exists: {output_col!r}")
        return _append_unique(source_columns, (str(output_col),))
    if query.op == "explode":
        source_columns = output_columns(
            query.inputs[0],
            window_source_columns=window_source_columns,
        )
        output_col = query.params.get("output_col")
        if output_col is None:
            return source_columns
        if str(output_col) in source_columns:
            raise ValueError(f"explode output column already exists: {output_col!r}")
        return _append_unique(source_columns, (str(output_col),))
    if query.op == "unnest":
        source_columns = output_columns(
            query.inputs[0],
            window_source_columns=window_source_columns,
        )
        column = str(query.params["column"])
        if column not in source_columns:
            raise ValueError(f"unnest input column not found: {column!r}")
        outputs = tuple(str(output) for _, output in query.params["fields"])
        duplicates = sorted({output for output in outputs if outputs.count(output) > 1})
        if duplicates:
            raise ValueError(f"unnest output columns must be unique: {duplicates}")
        parent_columns = tuple(name for name in source_columns if name != column)
        conflicts = sorted(set(parent_columns).intersection(outputs))
        if conflicts:
            raise ValueError(f"unnest output columns conflict with existing columns: {conflicts}")
        return _append_unique(parent_columns, outputs)
    if query.op == "sem_agg":
        output_cols = query.params.get("output_cols")
        if query.inputs and query.inputs[0].op == "group_by":
            keys = tuple(str(key) for key in query.inputs[0].params["keys"])
            aggregate_outputs = (
                _column_names(output_cols)
                if output_cols is not None
                else tuple(str(column) for column in query.params.get("input_cols") or ())
            )
            return _append_unique(keys, aggregate_outputs)
        if query.inputs and query.inputs[0].op == "sem_groupby":
            aggregate_outputs = (
                _column_names(output_cols)
                if output_cols is not None
                else tuple(str(column) for column in query.params.get("input_cols") or ())
            )
            return _append_unique(
                _sem_groupby_partition_keys(query.inputs[0]),
                aggregate_outputs,
            )
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
    if query.op == "agg":
        grouped = query.inputs[0]
        outputs = _aggregate_output_columns(query.params.get("aggregates", ()))
        if grouped.op == "group_by":
            return _append_unique(
                tuple(str(key) for key in grouped.params["keys"]),
                outputs,
            )
        if grouped.op == "sem_groupby":
            return _append_unique(_sem_groupby_partition_keys(grouped), outputs)
        raise NotImplementedError("agg output columns require group_by or sem_groupby input")
    if query.op in {"sem_map", "sem_flat_map"}:
        columns = _append_unique(
            output_columns(query.inputs[0], window_source_columns=window_source_columns),
            _column_names(query.params.get("output_cols") or ()),
        )
        ordinal_col = (
            query.params.get("ordinal_col")
            if query.op == "sem_flat_map"
            else None
        )
        if ordinal_col is None:
            return columns
        if str(ordinal_col) in columns:
            raise ValueError(
                "sem_flat_map ordinal_col conflicts with an existing or output "
                f"column: {ordinal_col!r}"
            )
        return _append_unique(columns, (str(ordinal_col),))
    if query.op in {
        "count_window",
        "sem_filter",
        "sem_groupby",
        "sem_topk",
        "drop_duplicates",
        "filter",
    }:
        return output_columns(query.inputs[0], window_source_columns=window_source_columns)
    if query.op == "assign":
        return _append_unique(
            output_columns(query.inputs[0], window_source_columns=window_source_columns),
            tuple(str(column) for column in query.params.get("assignments", {})),
        )
    if query.op == "process_window":
        return output_columns(query.inputs[1], window_source_columns=window_source_columns)
    if query.op == "join":
        return join_output_columns(query, window_source_columns=window_source_columns)
    if query.op == "sem_join":
        return semantic_join_columns(
            query,
            window_source_columns=window_source_columns,
        )
    if query.op in {"union", "union_by_name", "concat", "subtract"}:
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
    if str(query.params.get("how", "inner")) == "left_anti":
        return left_columns
    right_columns = output_columns(
        query.inputs[1],
        window_source_columns=window_source_columns,
    )
    keys = _join_key_columns(query.params["on"])
    left_alias = _relation_alias(query.inputs[0])
    right_alias = _relation_alias(query.inputs[1])
    overlapping = set(left_columns).intersection(right_columns).difference(keys)
    predicate_join = not keys

    columns: list[str] = []
    for column in left_columns:
        columns.append(
            _joined_column_name(
                column,
                side_alias=left_alias,
                default_side="left",
                should_suffix=predicate_join or column in overlapping,
            )
        )
    for column in right_columns:
        if keys and column in keys:
            continue
        columns.append(
            _joined_column_name(
                column,
                side_alias=right_alias,
                default_side="right",
                should_suffix=predicate_join or column in overlapping,
            )
        )
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


def _aggregate_output_columns(aggregates: object) -> tuple[str, ...]:
    """Return output names from aggregate specs."""

    outputs: list[str] = []
    for aggregate in aggregates:  # type: ignore[union-attr]
        if isinstance(aggregate, SemanticAggregateSpec):
            names = tuple(column.name for column in aggregate.output_cols)
        elif isinstance(
            aggregate,
            (ArrayAggregateSpec, CollectListAggregateSpec, MinAggregateSpec),
        ):
            names = (aggregate.output_col,)
        else:
            names = ()
        for name in names:
            if name not in outputs:
                outputs.append(name)
    return tuple(outputs)


def _sem_groupby_partition_keys(query: QueryExpr) -> tuple[str, ...]:
    """Return deterministic partition keys carried by sem_groupby."""

    return tuple(str(key) for key in query.params.get("partition_by", ()))


def _append_unique(base: Sequence[str], additions: Sequence[str]) -> tuple[str, ...]:
    """Append columns while preserving first occurrence order."""

    columns = list(base)
    for column in additions:
        if column not in columns:
            columns.append(column)
    return tuple(columns)


def _join_key_columns(on: object) -> tuple[str, ...]:
    """Return exact key join columns, or empty for predicate joins."""

    if isinstance(on, tuple) and all(isinstance(column, str) for column in on):
        return tuple(str(column) for column in on)
    return ()


def _relation_alias(query: QueryExpr) -> str | None:
    """Return a relation alias name if present."""

    if query.op == "alias":
        name = query.params.get("name")
        return str(name) if name is not None else None
    return None


def _joined_column_name(
    column: str,
    *,
    side_alias: str | None,
    default_side: str,
    should_suffix: bool,
) -> str:
    """Return the public column name emitted by deterministic joins."""

    if should_suffix:
        return f"{column}:{side_alias or default_side}"
    return column
