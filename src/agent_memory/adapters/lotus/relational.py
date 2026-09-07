"""Deterministic relational lowering for the LOTUS adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any

import numpy as np
import pandas as pd

from agent_memory.policy.aggregates import (
    ArrayAggregateSpec,
    CollectListAggregateSpec,
    MinAggregateSpec,
    SemanticAggregateSpec,
)
from agent_memory.adapters.lotus.sem_agg import (
    aggregate_groups_with_keys,
    aggregate_input_columns,
    execute_native_sem_agg_group,
    execute_native_sem_agg_groups,
    execute_structured_sem_agg_group,
    execute_structured_sem_agg_groups,
)
from agent_memory.policy.expressions import (
    ArithmeticExpr,
    ArrayCatExpr,
    BooleanExpr,
    CaseWhenExpr,
    ColumnExpr,
    ComparisonExpr,
    Expr,
    LeastExpr,
    LiteralExpr,
    TryCastExpr,
    expr_from_param,
    is_scalar,
)
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.schema import output_columns
from agent_memory.runtime.window import over_frames


def execute_select(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic column projection."""

    source = execute(query.inputs[0], inputs)
    return source.loc[:, list(query.params["columns"])].copy()


def execute_alias(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute a relation alias marker without changing physical rows."""

    return execute(query.inputs[0], inputs).copy()


def execute_group_by(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute a deterministic group marker without aggregating rows."""

    source = execute(query.inputs[0], inputs).copy()
    keys = tuple(str(key) for key in query.params["keys"])
    missing = [key for key in keys if key not in source.columns]
    if missing:
        raise ValueError(f"group_by key columns not found in DataFrame: {missing}")
    source.attrs["agent_memory_groupby_keys"] = keys
    return source


def execute_let(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Evaluate one relation once and bind it while executing a local query body."""

    if len(query.inputs) != 2:
        raise ValueError("let expects one bound relation and one query body")
    name = str(query.params.get("name", ""))
    if not name:
        raise ValueError("let requires a non-empty binding name")
    if name in inputs:
        raise ValueError(f"let binding conflicts with an existing input: {name!r}")
    value = execute(query.inputs[0], inputs)
    scoped_inputs = dict(inputs)
    scoped_inputs[name] = value
    return execute(query.inputs[1], scoped_inputs)


def execute_concat(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute union-all row append semantics."""

    left, right = _execute_binary_inputs(query, inputs, execute)
    _require_matching_columns(left, right, op="concat")
    return _concat_rows(left, right)


def execute_union(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute exact row-set union semantics."""

    concatenated = execute_concat(query, inputs, execute)
    return concatenated.drop_duplicates(ignore_index=True)


def execute_union_by_name(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute name-aligned row-set union semantics."""

    left, right = _execute_binary_inputs(query, inputs, execute)
    allow_missing = query.params.get("allow_missing_columns", True)
    if not isinstance(allow_missing, bool):
        raise TypeError("union_by_name allow_missing_columns must be a bool")
    columns = _union_by_name_columns(left, right, allow_missing_columns=allow_missing)
    left_aligned = _align_columns_by_name(
        left, columns, allow_missing_columns=allow_missing
    )
    right_aligned = _align_columns_by_name(
        right, columns, allow_missing_columns=allow_missing
    )
    concatenated = _concat_rows(left_aligned, right_aligned)
    return concatenated.drop_duplicates(ignore_index=True)


def execute_assign(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic scalar or column-copy assignment."""

    source = execute(query.inputs[0], inputs).copy()
    assignments = query.params.get("assignments", {})
    if not isinstance(assignments, Mapping):
        raise TypeError("assign assignments must be a mapping")
    for column, value in assignments.items():
        column_name = str(column)
        if is_scalar(value):
            source[column_name] = value
            continue
        expr = expr_from_param(value)
        if not isinstance(
            expr,
            (
                LiteralExpr,
                ColumnExpr,
                ArithmeticExpr,
                ArrayCatExpr,
                LeastExpr,
                TryCastExpr,
                CaseWhenExpr,
            ),
        ):
            raise TypeError(
                "assign values must be scalar literals or supported row expressions"
            )
        source[column_name] = evaluate_expr(expr, source)
    return source


def execute_filter(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic row-filter expressions."""

    source = execute(query.inputs[0], inputs)
    predicate = expr_from_param(query.params.get("predicate"))
    mask = evaluate_predicate(predicate, source)
    return source.loc[mask].copy().reset_index(drop=True)


def execute_subtract(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute exact row-set difference semantics."""

    left, right = _execute_binary_inputs(query, inputs, execute)
    _require_matching_columns(left, right, op="subtract")
    if right.empty:
        return left.copy().reset_index(drop=True)

    marker = "_agent_memory_subtract_marker"
    right_unique = right.drop_duplicates()
    merged = left.merge(right_unique.assign(**{marker: True}), how="left")
    result = merged[merged[marker].isna()].drop(columns=[marker])
    return result.loc[:, list(left.columns)].reset_index(drop=True)


def execute_join(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic same-key or predicate relational join semantics."""

    left, right = _execute_binary_inputs(query, inputs, execute)
    on = tuple(query.params["on"])
    how = str(query.params.get("how", "inner"))
    _require_supported_join_how(how)

    if _is_predicate_join(on):
        return _execute_predicate_join(
            query,
            left,
            right,
            predicates=tuple(expr_from_param(predicate) for predicate in on),
            how=how,
        )

    keys = tuple(str(column) for column in on)
    _require_join_keys(left, right, keys)
    _require_non_null_join_keys(left, keys, side="left")
    _require_non_null_join_keys(right, keys, side="right")

    if how == "left_anti":
        if right.empty:
            return left.copy().reset_index(drop=True)
        right_keys = right.loc[:, list(keys)].drop_duplicates()
        marker = "_agent_memory_left_anti_marker"
        merged = left.merge(
            right_keys.assign(**{marker: True}),
            how="left",
            on=list(keys),
            sort=False,
        )
        result = merged[merged[marker].isna()].drop(columns=[marker])
        return result.loc[:, list(left.columns)].reset_index(drop=True)

    return left.merge(
        right,
        how=how,
        on=list(keys),
        suffixes=_join_suffixes(query),
        sort=False,
    ).reset_index(drop=True)


def predicate_join_row_ids(
    query: QueryExpr,
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, Any], ...]:
    """Return input row-ID pairs that satisfy a deterministic join predicate."""

    predicates = tuple(on)
    if not _is_predicate_join(predicates):
        raise ValueError("predicate join row IDs require normalized predicate expressions")
    if len(query.inputs) != 2:
        raise ValueError("predicate join row IDs require exactly two query inputs")

    left_id_column = _unused_column_name(
        left,
        right,
        base="_agent_memory_join_left_row_id",
    )
    right_id_column = _unused_column_name(
        left,
        right,
        base="_agent_memory_join_right_row_id",
    )
    left_with_ids = left.copy()
    right_with_ids = right.copy()
    left_with_ids[left_id_column] = list(left.index)
    right_with_ids[right_id_column] = list(right.index)
    joined = _execute_predicate_join(
        query,
        left_with_ids,
        right_with_ids,
        predicates=tuple(expr_from_param(predicate) for predicate in predicates),
        how="inner",
    )

    left_alias = _relation_alias(query.inputs[0]) or "left"
    right_alias = _relation_alias(query.inputs[1]) or "right"
    left_output = f"{left_id_column}:{left_alias}"
    right_output = f"{right_id_column}:{right_alias}"
    return tuple(
        zip(
            joined[left_output],
            joined[right_output],
            strict=True,
        )
    )


def execute_drop_duplicates(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute exact duplicate-row removal."""

    source = execute(query.inputs[0], inputs)
    subset = query.params.get("subset")
    return source.drop_duplicates(
        subset=list(subset) if subset is not None else None,
        keep="first",
        ignore_index=True,
    )


def execute_array_agg(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic array-of-records aggregation."""

    if query.inputs[0].op == "over":
        return execute_over_array_agg(query, inputs, execute)
    if query.inputs[0].op in {"group_by", "sem_groupby"}:
        return execute_grouped_array_agg(query, inputs, execute)

    source = execute(query.inputs[0], inputs)
    columns = tuple(str(column) for column in query.params["columns"])
    output_col = str(query.params["output_col"])
    missing = [column for column in columns if column not in source.columns]
    if missing:
        raise ValueError(f"array_agg input columns not found in DataFrame: {missing}")

    records = _strict_json_records(source, columns)
    value = json.dumps(
        records, ensure_ascii=False, default=_json_default, allow_nan=False
    )
    return pd.DataFrame([{output_col: value}], columns=[output_col])


def execute_grouped_array_agg(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute grouped array-of-records aggregation."""

    group_query = query.inputs[0]
    source = execute(group_query, inputs)
    columns = tuple(str(column) for column in query.params["columns"])
    output_col = str(query.params["output_col"])
    missing = [column for column in columns if column not in source.columns]
    if missing:
        raise ValueError(
            f"grouped array_agg input columns not found in DataFrame: {missing}"
        )

    if group_query.op == "group_by":
        group_keys = tuple(str(key) for key in group_query.params["keys"])
        missing_keys = [key for key in group_keys if key not in source.columns]
        if missing_keys:
            raise ValueError(
                f"group_by key columns not found in DataFrame: {missing_keys}"
            )
        return _array_agg_by_keys(
            source,
            group_keys=group_keys,
            columns=columns,
            output_col=output_col,
            include_keys=True,
        )

    raise NotImplementedError(
        "sem_groupby(...).array_agg(...) is not supported; use "
        "sem_groupby(...).agg(sem_agg(...), array_agg(...)) so semantic keys "
        "are produced by sem_agg output columns."
    )


def execute_min(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute a global or deterministic grouped minimum aggregate."""

    source_query = query.inputs[0]
    source = execute(source_query, inputs)
    columns = tuple(str(column) for column in query.params["columns"])
    output_col = str(query.params["output_col"])
    missing = [column for column in columns if column not in source.columns]
    if missing:
        if len(missing) == 1:
            raise ValueError(f"min input column not found in DataFrame: {missing[0]!r}")
        raise ValueError(f"min input columns not found in DataFrame: {missing}")

    if source_query.op == "group_by":
        keys = tuple(str(key) for key in source_query.params["keys"])
        if output_col in keys:
            raise ValueError(
                f"grouped min output column conflicts with group key: {output_col!r}"
            )
        rows = [
            {**key_values, output_col: _minimum_value(group, columns)}
            for key_values, group in aggregate_groups_with_keys(source)
        ]
        result = pd.DataFrame(rows, columns=[*keys, output_col])
        result[output_col] = pd.Series(
            [row[output_col] for row in rows],
            dtype=object,
        )
        return result
    if source_query.op == "sem_groupby":
        raise NotImplementedError("sem_groupby(...).min(...) is not supported")
    return pd.DataFrame(
        {output_col: pd.Series([_minimum_value(source, columns)], dtype=object)}
    )


def execute_agg(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: Any,
) -> Any:
    """Execute grouped aggregate specs over one grouped input."""

    group_query = query.inputs[0]
    source = execute(group_query, inputs)
    aggregates = tuple(query.params.get("aggregates", ()))
    if group_query.op not in {"group_by", "sem_groupby"}:
        raise ValueError("agg expects group_by or sem_groupby input")
    if source.empty:
        return pd.DataFrame(columns=list(output_columns(query)))
    if (
        context.config.sem_agg_dispatch != "sequential"
        or context.config.prompt_batching is not None
    ):
        return _execute_agg_with_group_batching(
            query,
            source,
            aggregates,
            context=context,
        )

    rows: list[dict[str, Any]] = []
    for group_index, (key_values, group) in enumerate(
        aggregate_groups_with_keys(source)
    ):
        row: dict[str, Any] = dict(key_values)
        for aggregate in aggregates:
            if isinstance(aggregate, ArrayAggregateSpec):
                if aggregate.output_col in row:
                    continue
                row[aggregate.output_col] = _array_records_json(
                    group, aggregate.columns
                )
                continue
            if isinstance(aggregate, CollectListAggregateSpec):
                if aggregate.output_col in row:
                    continue
                row[aggregate.output_col] = _collect_list_json(group, aggregate.column)
                continue
            if isinstance(aggregate, MinAggregateSpec):
                if aggregate.output_col in row:
                    continue
                missing = [
                    column
                    for column in aggregate.columns
                    if column not in group.columns
                ]
                if missing:
                    raise ValueError(
                        f"min input columns not found in aggregate group: {missing}"
                    )
                row[aggregate.output_col] = _minimum_value(group, aggregate.columns)
                continue
            if isinstance(aggregate, SemanticAggregateSpec):
                semantic_values = _execute_grouped_semantic_aggregate_spec(
                    aggregate,
                    group,
                    group_index=group_index,
                    context=context,
                )
                for column, value in semantic_values.items():
                    if column not in row:
                        row[column] = value
                continue
            raise TypeError(f"Unsupported aggregate spec: {type(aggregate).__name__}")
        rows.append(row)
    result = pd.DataFrame(rows, columns=list(output_columns(query)))
    for aggregate in aggregates:
        if isinstance(aggregate, MinAggregateSpec):
            result[aggregate.output_col] = pd.Series(
                [row.get(aggregate.output_col) for row in rows],
                dtype=object,
            )
    return result


def _execute_agg_with_group_batching(
    query: QueryExpr,
    source: pd.DataFrame,
    aggregates: Sequence[object],
    *,
    context: Any,
) -> pd.DataFrame:
    """Execute semantic aggregate specs across independent groups in batches."""

    grouped = aggregate_groups_with_keys(source)
    groups = [group for _key_values, group in grouped]
    rows = [dict(key_values) for key_values, _group in grouped]
    for aggregate in aggregates:
        if isinstance(aggregate, SemanticAggregateSpec):
            semantic_values = _execute_grouped_semantic_aggregate_spec_many(
                aggregate,
                groups,
                context=context,
            )
            for row, values in zip(rows, semantic_values, strict=True):
                for column, value in values.items():
                    if column not in row:
                        row[column] = value
            continue
        for row, group in zip(rows, groups, strict=True):
            _apply_deterministic_aggregate(aggregate, group, row)

    result = pd.DataFrame(rows, columns=list(output_columns(query)))
    for aggregate in aggregates:
        if isinstance(aggregate, MinAggregateSpec):
            result[aggregate.output_col] = pd.Series(
                [row.get(aggregate.output_col) for row in rows],
                dtype=object,
            )
    return result


def _apply_deterministic_aggregate(
    aggregate: object,
    group: pd.DataFrame,
    row: dict[str, Any],
) -> None:
    """Apply one non-semantic aggregate spec to one grouped frame."""

    if isinstance(aggregate, ArrayAggregateSpec):
        if aggregate.output_col not in row:
            row[aggregate.output_col] = _array_records_json(group, aggregate.columns)
        return
    if isinstance(aggregate, CollectListAggregateSpec):
        if aggregate.output_col not in row:
            row[aggregate.output_col] = _collect_list_json(group, aggregate.column)
        return
    if isinstance(aggregate, MinAggregateSpec):
        if aggregate.output_col in row:
            return
        missing = [
            column for column in aggregate.columns if column not in group.columns
        ]
        if missing:
            raise ValueError(
                f"min input columns not found in aggregate group: {missing}"
            )
        row[aggregate.output_col] = _minimum_value(group, aggregate.columns)
        return
    raise TypeError(f"Unsupported aggregate spec: {type(aggregate).__name__}")


def _execute_grouped_semantic_aggregate_spec_many(
    aggregate: SemanticAggregateSpec,
    groups: Sequence[pd.DataFrame],
    *,
    context: Any,
) -> list[dict[str, Any]]:
    """Execute one semantic aggregate spec over independent grouped frames."""

    if not groups:
        return []
    query = QueryExpr(
        op="sem_agg",
        params={
            "input_cols": aggregate.input_cols,
            "output_cols": aggregate.output_cols,
            "instruction": aggregate.instruction,
        },
    )
    input_cols = aggregate_input_columns(groups[0], query.params.get("input_cols"))
    if len(aggregate.output_cols) == 1:
        output_col = aggregate.output_cols[0]
        values = execute_native_sem_agg_groups(
            query,
            groups,
            input_cols,
            context.config,
        )
        return [{output_col.name: value} for value in values]
    parsed = execute_structured_sem_agg_groups(
        query,
        groups,
        input_cols,
        aggregate.output_cols,
        context.config,
    )
    return [dict(values) for values in parsed]


def _execute_grouped_semantic_aggregate_spec(
    aggregate: SemanticAggregateSpec,
    group: pd.DataFrame,
    *,
    group_index: int,
    context: Any,
) -> dict[str, Any]:
    """Execute one semantic aggregate spec against one grouped frame."""

    query = QueryExpr(
        op="sem_agg",
        params={
            "input_cols": aggregate.input_cols,
            "output_cols": aggregate.output_cols,
            "instruction": aggregate.instruction,
        },
    )
    input_cols = aggregate_input_columns(group, query.params.get("input_cols"))
    if len(aggregate.output_cols) == 1:
        output_col = aggregate.output_cols[0]
        raw = execute_native_sem_agg_group(query, group, input_cols, context.config)
        return {output_col.name: raw}
    parsed = execute_structured_sem_agg_group(
        query,
        group,
        input_cols,
        aggregate.output_cols,
        context.config,
        group_index=group_index,
    )
    return dict(parsed)


def execute_over_array_agg(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute row-preserving over-window array aggregation."""

    over_query = query.inputs[0]
    emit_source = execute(over_query.inputs[0], inputs)
    frame_source_query = over_query.params.get("frame_source")
    frame_source = (
        execute(frame_source_query, inputs)
        if isinstance(frame_source_query, QueryExpr)
        else emit_source
    )
    columns = tuple(str(column) for column in query.params["columns"])
    output_col = str(query.params["output_col"])
    emit_columns = tuple(output_columns(over_query.inputs[0])) or tuple(
        emit_source.columns
    )
    missing = [column for column in columns if column not in frame_source.columns]
    if missing:
        raise ValueError(
            f"over array_agg input columns not found in DataFrame: {missing}"
        )
    missing_emit = [
        column for column in emit_columns if column not in emit_source.columns
    ]
    if missing_emit:
        raise ValueError(
            f"over array_agg emit columns not found in DataFrame: {missing_emit}"
        )

    rows: list[dict[str, Any]] = []
    for frame in over_frames(emit_source, frame_source, over_query.params):
        value = json.dumps(
            _strict_json_records(frame.frame, columns),
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        )
        row = frame.emit_row.loc[list(emit_columns)].to_dict()
        row[output_col] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=[*emit_columns, output_col])


def execute_array_cat(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Execute deterministic JSON array-state concatenation."""

    left, right = _execute_binary_inputs(query, inputs, execute)
    column = str(query.params["column"])
    _require_array_column(left, column, side="left")
    _require_array_column(right, column, side="right")
    _require_at_most_one_row(left, op="array_cat", side="left")
    _require_at_most_one_row(right, op="array_cat", side="right")

    if left.empty and right.empty:
        return pd.DataFrame(columns=[column])
    if left.empty:
        _load_array_json(right.iloc[0][column], column=column, side="right")
        return right.loc[:, [column]].copy().reset_index(drop=True)
    if right.empty:
        _load_array_json(left.iloc[0][column], column=column, side="left")
        return left.loc[:, [column]].copy().reset_index(drop=True)

    left_items = _load_array_json(left.iloc[0][column], column=column, side="left")
    right_items = _load_array_json(right.iloc[0][column], column=column, side="right")
    value = json.dumps(
        [*left_items, *right_items],
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )
    return pd.DataFrame([{column: value}], columns=[column])


def execute_flatten(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Flatten one JSON array-of-arrays column into a JSON array column."""

    source = execute(query.inputs[0], inputs).copy()
    column = str(query.params["column"])
    output_col = query.params.get("output_col")
    output_column = column if output_col is None else str(output_col)
    if column not in source.columns:
        raise ValueError(f"flatten input column not found in DataFrame: {column!r}")
    if output_column != column and output_column in source.columns:
        raise ValueError(f"flatten output column already exists: {output_column!r}")
    source[output_column] = source[column].map(
        lambda value: _flatten_array_value(value, column=column)
    )
    return source


def execute_explode(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Expand one JSON array column into one row per element."""

    source = execute(query.inputs[0], inputs).copy()
    column = str(query.params["column"])
    output_col = query.params.get("output_col")
    output_column = column if output_col is None else str(output_col)
    if column not in source.columns:
        raise ValueError(f"explode input column not found in DataFrame: {column!r}")
    if output_col is not None and output_column in source.columns:
        raise ValueError(f"explode output column already exists: {output_column!r}")

    output_rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        elements = _load_optional_json_array(row[column], op="explode", column=column)
        for element in elements:
            output_row = row.to_dict()
            output_row[output_column] = element
            output_rows.append(output_row)
    return pd.DataFrame(output_rows, columns=list(output_columns(query)))


def execute_unnest(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> Any:
    """Expand one JSON object column into ordinary columns."""

    expected_columns = list(output_columns(query))
    source = execute(query.inputs[0], inputs).copy()
    column = str(query.params["column"])
    fields = tuple(
        (str(field), str(output)) for field, output in query.params["fields"]
    )
    if column not in source.columns:
        raise ValueError(f"unnest input column not found in DataFrame: {column!r}")

    parent_columns = [name for name in source.columns if name != column]
    output_names = [output for _, output in fields]
    conflicts = sorted(set(parent_columns).intersection(output_names))
    if conflicts:
        raise ValueError(
            f"unnest output columns conflict with existing columns: {conflicts}"
        )

    output_rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        value = _load_json_object(row[column], op="unnest", column=column)
        output_row = {name: row[name] for name in parent_columns}
        for field, output in fields:
            if field not in value:
                raise ValueError(
                    f"unnest field {field!r} not found in object column {column!r}"
                )
            output_row[output] = value[field]
        output_rows.append(output_row)
    return pd.DataFrame(output_rows, columns=expected_columns)


def _execute_binary_inputs(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
) -> tuple[Any, Any]:
    """Execute and validate a binary relational expression."""

    if len(query.inputs) != 2:
        raise ValueError(f"{query.op} expects exactly two inputs")
    return execute(query.inputs[0], inputs), execute(query.inputs[1], inputs)


def _concat_rows(*frames: pd.DataFrame) -> pd.DataFrame:
    """Append aligned rows without pandas' deprecated all-null dtype inference."""

    if not frames:
        raise ValueError("row concatenation requires at least one DataFrame")
    columns = list(frames[0].columns)
    non_empty_frames = tuple(frame for frame in frames if not frame.empty)
    if not non_empty_frames:
        return frames[0].copy().reset_index(drop=True)
    if not columns:
        return pd.DataFrame(index=range(sum(len(frame) for frame in non_empty_frames)))
    return pd.DataFrame(
        {
            column: pd.concat(
                [frame[column] for frame in non_empty_frames],
                ignore_index=True,
            )
            for column in columns
        }
    )


def _require_matching_columns(left: Any, right: Any, *, op: str) -> None:
    """Require identical column order for exact row-set operators."""

    if list(left.columns) != list(right.columns):
        raise ValueError(f"{op} requires matching columns")


def _union_by_name_columns(
    left: Any,
    right: Any,
    *,
    allow_missing_columns: bool,
) -> list[str]:
    """Return stable name-aligned output columns for union_by_name."""

    left_columns = [str(column) for column in left.columns]
    right_columns = [str(column) for column in right.columns]
    if not allow_missing_columns and set(left_columns) != set(right_columns):
        raise ValueError(
            "union_by_name requires the same column names unless "
            "allow_missing_columns=True"
        )

    columns = list(left_columns)
    for column in right_columns:
        if column not in columns:
            columns.append(column)
    return columns


def _align_columns_by_name(
    frame: Any,
    columns: Sequence[str],
    *,
    allow_missing_columns: bool,
) -> Any:
    """Project a frame into name-aligned union columns."""

    missing = [column for column in columns if column not in frame.columns]
    if missing and not allow_missing_columns:
        raise ValueError(f"union_by_name input is missing columns: {missing}")

    aligned = frame.copy()
    for column in missing:
        aligned[column] = pd.NA
    return aligned.loc[:, list(columns)].copy()


def _require_supported_join_how(how: str) -> None:
    """Require a pandas relational join mode supported by the public API."""

    if how not in {"inner", "left", "right", "outer", "left_anti"}:
        raise ValueError(
            "join how must be one of: inner, left, right, outer, left_anti"
        )


def _is_predicate_join(on: tuple[Any, ...]) -> bool:
    """Return whether a join uses boolean predicate expression params."""

    return bool(on) and all(isinstance(item, Mapping) for item in on)


def _join_suffixes(query: QueryExpr) -> tuple[str, str]:
    """Return pandas merge suffixes for key joins."""

    left_alias = _relation_alias(query.inputs[0]) or "left"
    right_alias = _relation_alias(query.inputs[1]) or "right"
    return f":{left_alias}", f":{right_alias}"


def _unused_column_name(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    base: str,
) -> str:
    """Return a temporary column name absent from both join inputs."""

    candidate = base
    while candidate in left.columns or candidate in right.columns:
        candidate = f"_{candidate}"
    return candidate


def _relation_alias(query: QueryExpr) -> str | None:
    """Return relation alias name if present."""

    if query.op == "alias":
        name = query.params.get("name")
        return str(name) if name is not None else None
    return None


def _execute_predicate_join(
    query: QueryExpr,
    left: Any,
    right: Any,
    *,
    predicates: tuple[Expr, ...],
    how: str,
) -> Any:
    """Execute an inner predicate join with optional equi-join pruning."""

    if how != "inner":
        raise NotImplementedError("predicate join currently supports only how='inner'")

    left_alias = _relation_alias(query.inputs[0]) or "left"
    right_alias = _relation_alias(query.inputs[1]) or "right"
    equality_pairs = _predicate_equality_pairs(
        predicates,
        left_alias=left_alias,
        right_alias=right_alias,
        left_columns=tuple(str(column) for column in left.columns),
        right_columns=tuple(str(column) for column in right.columns),
    )

    left_prepared = _rename_join_side(left, alias=left_alias)
    right_prepared = _rename_join_side(right, alias=right_alias)
    if equality_pairs:
        left_on = [f"{left_col}:{left_alias}" for left_col, _ in equality_pairs]
        right_on = [f"{right_col}:{right_alias}" for _, right_col in equality_pairs]
        joined = left_prepared.merge(
            right_prepared,
            how="inner",
            left_on=left_on,
            right_on=right_on,
            sort=False,
        )
    else:
        joined = left_prepared.merge(right_prepared, how="cross", sort=False)

    mask = evaluate_predicate(BooleanExpr(op="and", operands=predicates), joined)
    return joined.loc[mask].reset_index(drop=True)


def _rename_join_side(frame: Any, *, alias: str) -> Any:
    """Rename every column in one predicate-join side with its alias suffix."""

    renamed = {column: f"{column}:{alias}" for column in frame.columns}
    return frame.rename(columns=renamed).copy()


def _predicate_equality_pairs(
    predicates: tuple[Expr, ...],
    *,
    left_alias: str,
    right_alias: str,
    left_columns: tuple[str, ...],
    right_columns: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Extract left/right column equality pairs usable as merge keys."""

    pairs: list[tuple[str, str]] = []
    for predicate in predicates:
        if not isinstance(predicate, ComparisonExpr) or predicate.op != "eq":
            continue
        left_side = _predicate_column_side(
            predicate.left,
            left_alias=left_alias,
            right_alias=right_alias,
            left_columns=left_columns,
            right_columns=right_columns,
        )
        right_side = _predicate_column_side(
            predicate.right,
            left_alias=left_alias,
            right_alias=right_alias,
            left_columns=left_columns,
            right_columns=right_columns,
        )
        if left_side is None or right_side is None or left_side[0] == right_side[0]:
            continue
        if left_side[0] == "left":
            pairs.append((left_side[1], right_side[1]))
        else:
            pairs.append((right_side[1], left_side[1]))
    return tuple(pairs)


def _predicate_column_side(
    expr: Expr,
    *,
    left_alias: str,
    right_alias: str,
    left_columns: tuple[str, ...],
    right_columns: tuple[str, ...],
) -> tuple[str, str] | None:
    """Return which join side a column expression references."""

    if not isinstance(expr, ColumnExpr):
        return None
    if expr.qualifier == left_alias:
        return "left", expr.name
    if expr.qualifier == right_alias:
        return "right", expr.name
    if expr.qualifier is None:
        in_left = expr.name in left_columns
        in_right = expr.name in right_columns
        if in_left and not in_right:
            return "left", expr.name
        if in_right and not in_left:
            return "right", expr.name
    return None


def evaluate_predicate(expr: Expr, frame: Any) -> Any:
    """Evaluate a boolean expression against a DataFrame."""

    values = evaluate_expr(expr, frame)
    if isinstance(values, pd.Series):
        non_null = values.dropna()
        invalid = non_null.map(lambda value: not isinstance(value, (bool, np.bool_)))
        if bool(invalid.any()):
            raise TypeError("filter predicate must evaluate to boolean values")
        return values.map(lambda value: False if pd.isna(value) else bool(value))
    if not isinstance(values, (bool, np.bool_)):
        raise TypeError("filter predicate must evaluate to boolean values")
    return pd.Series([bool(values)] * len(frame), index=frame.index)


def evaluate_expr(expr: Expr, frame: Any) -> Any:
    """Evaluate a relational expression against a DataFrame."""

    if isinstance(expr, ColumnExpr):
        return _column_values(frame, expr)
    if isinstance(expr, LiteralExpr):
        return expr.value
    if isinstance(expr, ComparisonExpr):
        return _evaluate_comparison(expr, frame)
    if isinstance(expr, ArithmeticExpr):
        return _evaluate_arithmetic(expr, frame)
    if isinstance(expr, TryCastExpr):
        return _evaluate_try_cast(expr, frame)
    if isinstance(expr, CaseWhenExpr):
        return _evaluate_case_when(expr, frame)
    if isinstance(expr, BooleanExpr):
        return _evaluate_boolean(expr, frame)
    if isinstance(expr, ArrayCatExpr):
        return _evaluate_array_cat(expr, frame)
    if isinstance(expr, LeastExpr):
        return _evaluate_least(expr, frame)
    raise TypeError(f"Unsupported expression type: {type(expr).__name__}")


def _evaluate_arithmetic(expr: ArithmeticExpr, frame: Any) -> Any:
    """Evaluate numeric arithmetic with null propagation and explicit zero checks."""

    left = evaluate_expr(expr.left, frame)
    right = evaluate_expr(expr.right, frame)
    if not isinstance(left, pd.Series) and not isinstance(right, pd.Series):
        return _evaluate_arithmetic_value(expr.op, left, right)
    left_series = (
        left
        if isinstance(left, pd.Series)
        else pd.Series([left] * len(frame), index=frame.index, dtype=object)
    )
    right_series = (
        right
        if isinstance(right, pd.Series)
        else pd.Series([right] * len(frame), index=frame.index, dtype=object)
    )
    return pd.Series(
        [
            _evaluate_arithmetic_value(expr.op, left_value, right_value)
            for left_value, right_value in zip(left_series, right_series, strict=True)
        ],
        index=frame.index,
        dtype=object,
    )


def _evaluate_try_cast(expr: TryCastExpr, frame: Any) -> Any:
    """Convert scalars to floats, returning null for failed conversions."""

    if expr.target != "float":
        raise ValueError("try_cast currently only supports the 'float' target")
    value = evaluate_expr(expr.value, frame)
    if isinstance(value, pd.Series):
        return value.map(_try_cast_float)
    return _try_cast_float(value)


def _try_cast_float(value: Any) -> float | None:
    """Convert one nullable scalar using Python float semantics."""

    if _is_arithmetic_null(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _evaluate_case_when(expr: CaseWhenExpr, frame: Any) -> pd.Series:
    """Evaluate CASE WHEN, treating a null condition as false."""

    condition = evaluate_predicate(expr.condition, frame)
    selected = condition.to_numpy(dtype=bool)
    result = np.empty(len(frame), dtype=object)
    for branch, mask in (
        (expr.then_value, selected),
        (expr.else_value, ~selected),
    ):
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            continue
        branch_frame = frame.iloc[positions]
        branch_values = _as_expression_series(
            evaluate_expr(branch, branch_frame),
            branch_frame,
        )
        result[positions] = branch_values.to_numpy(dtype=object)
    return pd.Series(result, index=frame.index, dtype=object)


def _as_expression_series(value: Any, frame: pd.DataFrame) -> pd.Series:
    """Broadcast a scalar expression result to the current frame."""

    if isinstance(value, pd.Series):
        return value
    return pd.Series([value] * len(frame), index=frame.index, dtype=object)


def _evaluate_arithmetic_value(op: str, left: Any, right: Any) -> Any:
    """Evaluate one pair of nullable numeric scalar operands."""

    if _is_arithmetic_null(left) or _is_arithmetic_null(right):
        return None
    _require_arithmetic_numeric(left)
    _require_arithmetic_numeric(right)
    if op == "add":
        return left + right
    if op == "subtract":
        return left - right
    if op == "multiply":
        return left * right
    if op == "divide":
        if right == 0:
            raise ZeroDivisionError("arithmetic division by zero")
        return left / right
    raise ValueError(f"Unsupported arithmetic expression op: {op!r}")


def _require_arithmetic_numeric(value: Any) -> None:
    """Require one non-null arithmetic operand to be numeric but not boolean."""

    from numbers import Number

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
        raise TypeError("arithmetic expressions require numeric non-null operands")


def _is_arithmetic_null(value: Any) -> bool:
    """Return whether one scalar arithmetic operand is null."""

    if value is None or value is pd.NA:
        return True
    result = pd.isna(value)
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _column_values(frame: Any, expr: ColumnExpr) -> Any:
    """Return the DataFrame column referenced by an expression."""

    candidates = []
    if expr.qualifier is not None:
        candidates.append(f"{expr.name}:{expr.qualifier}")
    candidates.append(expr.name)
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    raise ValueError(f"Column {expr.name!r} not found in DataFrame")


def _evaluate_comparison(expr: ComparisonExpr, frame: Any) -> Any:
    """Evaluate a binary comparison expression."""

    left = evaluate_expr(expr.left, frame)
    right = evaluate_expr(expr.right, frame)
    if expr.op == "eq":
        return left == right
    if expr.op == "ne":
        return left != right
    if expr.op == "lt":
        return left < right
    if expr.op == "le":
        return left <= right
    if expr.op == "gt":
        return left > right
    if expr.op == "ge":
        return left >= right
    raise ValueError(f"Unsupported comparison expression op: {expr.op!r}")


def _evaluate_boolean(expr: BooleanExpr, frame: Any) -> Any:
    """Evaluate a boolean expression."""

    if expr.op == "and":
        masks = [evaluate_predicate(operand, frame) for operand in expr.operands]
        if not masks:
            return pd.Series([True] * len(frame), index=frame.index)
        result = masks[0]
        for mask in masks[1:]:
            result = result & mask
        return result
    if expr.op == "or":
        masks = [evaluate_predicate(operand, frame) for operand in expr.operands]
        if not masks:
            return pd.Series([False] * len(frame), index=frame.index)
        result = masks[0]
        for mask in masks[1:]:
            result = result | mask
        return result
    if expr.op == "not":
        if len(expr.operands) != 1:
            raise ValueError("not expression requires exactly one operand")
        return ~evaluate_predicate(expr.operands[0], frame)
    if expr.op == "in":
        if not expr.operands:
            raise ValueError("in expression requires at least one operand")
        values = evaluate_expr(expr.operands[0], frame)
        literals = [evaluate_expr(operand, frame) for operand in expr.operands[1:]]
        if not isinstance(values, pd.Series):
            return values in literals
        return values.isin(literals)
    if expr.op == "is_null":
        if len(expr.operands) != 1:
            raise ValueError("is_null expression requires exactly one operand")
        values = evaluate_expr(expr.operands[0], frame)
        return pd.isna(values)
    if expr.op == "is_not_null":
        if len(expr.operands) != 1:
            raise ValueError("is_not_null expression requires exactly one operand")
        values = evaluate_expr(expr.operands[0], frame)
        return ~pd.isna(values)
    raise ValueError(f"Unsupported boolean expression op: {expr.op!r}")


def _evaluate_array_cat(expr: ArrayCatExpr, frame: Any) -> pd.Series:
    """Evaluate a row-wise JSON array-state concatenation expression."""

    left = _expr_as_series(evaluate_expr(expr.left, frame), frame)
    right = _expr_as_series(evaluate_expr(expr.right, frame), frame)
    values: list[str] = []
    for left_value, right_value in zip(left.array, right.array, strict=True):
        items = [
            *_load_optional_array_json(left_value, column="array_cat", side="left"),
            *_load_optional_array_json(right_value, column="array_cat", side="right"),
        ]
        values.append(
            json.dumps(
                items,
                ensure_ascii=False,
                default=_json_default,
                allow_nan=False,
            )
        )
    return pd.Series(values, index=frame.index)


def _evaluate_least(expr: LeastExpr, frame: Any) -> pd.Series:
    """Evaluate a row-wise minimum while ignoring null operands."""

    operands = [
        _expr_as_series(evaluate_expr(operand, frame), frame)
        for operand in expr.operands
    ]
    values: list[Any] = []
    for position in range(len(frame)):
        candidates = [
            operand.iloc[position]
            for operand in operands
            if not _is_null_scalar(operand.iloc[position])
        ]
        if not candidates:
            values.append(None)
            continue
        try:
            values.append(min(candidates))
        except TypeError as exc:
            raise TypeError(
                f"least operands are not mutually comparable at row position {position}"
            ) from exc
    return pd.Series(values, index=frame.index)


def _is_null_scalar(value: Any) -> bool:
    """Return whether one scalar expression result is null."""

    result = pd.isna(value)
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _minimum_value(frame: pd.DataFrame, columns: Sequence[str]) -> Any:
    """Return a scalar or lexicographic tuple minimum over complete rows."""

    complete = frame.loc[:, list(columns)].dropna(how="any")
    if complete.empty:
        return None
    if len(columns) == 1:
        return complete.iloc[:, 0].min()
    values = [tuple(row) for row in complete.itertuples(index=False, name=None)]
    try:
        return min(values)
    except TypeError as error:
        raise TypeError("min input tuples are not mutually comparable") from error


def _expr_as_series(value: Any, frame: Any) -> pd.Series:
    """Broadcast scalar expression values to a DataFrame-indexed Series."""

    if isinstance(value, pd.Series):
        return value
    return pd.Series([value] * len(frame), index=frame.index)


def _array_agg_by_keys(
    source: Any,
    *,
    group_keys: tuple[str, ...],
    columns: tuple[str, ...],
    output_col: str,
    include_keys: bool,
) -> Any:
    """Aggregate source rows into JSON arrays within each group."""

    if include_keys and output_col in group_keys:
        raise ValueError(
            f"grouped array_agg output column conflicts with group key: {output_col!r}"
        )
    output_rows: list[dict[str, Any]] = []
    if source.empty:
        return pd.DataFrame(
            columns=[*group_keys, output_col] if include_keys else [output_col]
        )

    grouped = source.groupby(list(group_keys), sort=False, dropna=False)
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = {}
        if include_keys:
            row.update(dict(zip(group_keys, key_values, strict=True)))
        row[output_col] = json.dumps(
            _strict_json_records(group, columns),
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        )
        output_rows.append(row)
    return pd.DataFrame(
        output_rows,
        columns=[*group_keys, output_col] if include_keys else [output_col],
    )


def _array_records_json(
    group: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    """Serialize selected grouped rows as one JSON array state."""

    missing = [column for column in columns if column not in group.columns]
    if missing:
        raise ValueError(f"grouped agg array_agg input columns not found: {missing}")
    return json.dumps(
        _strict_json_records(group, columns),
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )


def _collect_list_json(group: pd.DataFrame, column: str) -> str:
    """Serialize one grouped column as a JSON value list."""

    if column not in group.columns:
        raise ValueError(f"collect_list input column not found: {column!r}")
    values = [None if _is_missing_value(value) else value for value in group[column]]
    return json.dumps(
        values,
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )


def _strict_json_records(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Return JSON-safe records with every pandas missing value normalized to None."""

    projected = frame.loc[:, list(columns)].astype(object)
    clean = projected.where(pd.notna(projected), None)
    return clean.to_dict(orient="records")


def _flatten_array_value(value: Any, *, column: str) -> str:
    """Flatten one JSON array containing JSON array states."""

    arrays = _load_optional_array_json(value, column=column, side="flatten")
    items: list[Any] = []
    for nested in arrays:
        items.extend(_load_optional_array_json(nested, column=column, side="flatten"))
    return json.dumps(
        items,
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )


def _load_json_array(value: Any, *, op: str, column: str) -> list[Any]:
    """Parse one JSON array value for a relational array operator."""

    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{op} value in column {column!r} must be a JSON array"
        ) from error
    if not isinstance(parsed, list):
        raise ValueError(f"{op} value in column {column!r} must be a JSON array")
    return parsed


def _load_optional_json_array(value: Any, *, op: str, column: str) -> list[Any]:
    """Parse one optional JSON array value."""

    if _is_missing_value(value):
        return []
    return _load_json_array(value, op=op, column=column)


def _load_json_object(value: Any, *, op: str, column: str) -> dict[str, Any]:
    """Parse one JSON object value for a relational struct operator."""

    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{op} value in column {column!r} must be a JSON object"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{op} value in column {column!r} must be a JSON object")
    return parsed


def _require_join_keys(left: Any, right: Any, keys: tuple[str, ...]) -> None:
    """Require join key columns to exist on both sides."""

    missing_left = [key for key in keys if key not in left.columns]
    missing_right = [key for key in keys if key not in right.columns]
    if missing_left or missing_right:
        details = []
        if missing_left:
            details.append(f"left missing {missing_left}")
        if missing_right:
            details.append(f"right missing {missing_right}")
        raise ValueError(
            f"join key columns must exist on both sides: {', '.join(details)}"
        )


def _require_non_null_join_keys(
    frame: Any, keys: tuple[str, ...], *, side: str
) -> None:
    """Reject null join keys to avoid pandas null-null matching surprises."""

    if frame.loc[:, list(keys)].isna().any().any():
        raise ValueError(f"join key columns cannot contain null values on {side} side")


def _require_array_column(frame: Any, column: str, *, side: str) -> None:
    """Require one JSON array column for aggregate-state concatenation."""

    if column not in frame.columns:
        raise ValueError(f"array_cat {side} input is missing column {column!r}")


def _require_at_most_one_row(frame: Any, *, op: str, side: str) -> None:
    """Require a one-row aggregate-state relation, allowing empty state."""

    if len(frame) > 1:
        raise ValueError(f"{op} expects {side} input to contain at most one row")


def _load_array_json(value: Any, *, column: str, side: str) -> list[Any]:
    """Parse one JSON array aggregate-state value."""

    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"array_cat {side} value in column {column!r} must be a JSON array"
        ) from error
    if not isinstance(parsed, list):
        raise ValueError(
            f"array_cat {side} value in column {column!r} must be a JSON array"
        )
    return parsed


def _load_optional_array_json(value: Any, *, column: str, side: str) -> list[Any]:
    """Parse one optional JSON array aggregate-state value."""

    if _is_missing_value(value):
        return []
    return _load_array_json(value, column=column, side=side)


def _is_missing_value(value: Any) -> bool:
    """Return whether a scalar-ish value should be treated as missing."""

    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _json_default(value: Any) -> Any:
    """Convert pandas/numpy scalar values before string fallback."""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)
