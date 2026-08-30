"""DataFrame-style relation authoring handles for memory policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .aggregates import (
    AggregateSpec,
    SemanticAggregateSpec,
    aggregate_output_names,
    normalize_aggregate_specs,
    normalize_min_columns,
)
from .expressions import (
    ArrayCatExpr,
    ColumnExpr,
    LeastExpr,
    ensure_boolean_expr,
    ensure_expr,
    expr_to_param,
    is_scalar,
)
from .logical import ColumnSpec, QueryExpr, UserQuery
from .schema import output_columns

LOG_ROW_ID_COLUMN = "_row_id"
LOG_ADDED_AT_COLUMN = "_added_at"
LOG_ADD_SEQ_COLUMN = "_add_seq"
LOG_SYSTEM_COLUMNS = (
    LOG_ROW_ID_COLUMN,
    LOG_ADDED_AT_COLUMN,
    LOG_ADD_SEQ_COLUMN,
)
LOG_SYSTEM_COLUMN_SPECS = (
    ColumnSpec(LOG_ROW_ID_COLUMN, "Framework-generated source row UUID."),
    ColumnSpec(LOG_ADDED_AT_COLUMN, "Framework-generated UTC append time."),
    ColumnSpec(LOG_ADD_SEQ_COLUMN, "Framework-generated zero-based append position."),
)

ColumnInput = Sequence[str] | None
ColumnOutput = Sequence[str] | Mapping[str, str]
JoinOn = str | Sequence[str] | object
SEM_GROUPBY_MEMBERSHIPS = ("exclusive", "overlapping")


def _normalize_input_cols(input_cols: ColumnInput) -> tuple[str, ...] | None:
    """Normalize optional input column names into immutable expression params."""

    if input_cols is None:
        return None
    return tuple(input_cols)


def _normalize_output_cols(output_cols: ColumnOutput | None) -> tuple[ColumnSpec, ...] | None:
    """Normalize output column declarations into ColumnSpec tuples."""

    if output_cols is None:
        return None
    if isinstance(output_cols, Mapping):
        return tuple(
            ColumnSpec(name=name, description=desc) for name, desc in output_cols.items()
        )
    return tuple(ColumnSpec(name=name) for name in output_cols)


def _normalize_group_labels(
    labels: Mapping[str, str] | None,
) -> tuple[ColumnSpec, ...] | None:
    """Normalize optional closed-world group labels into ColumnSpec tuples."""

    if labels is None:
        return None
    if not labels:
        raise ValueError("sem_groupby labels cannot be empty")
    return tuple(ColumnSpec(name=name, description=desc) for name, desc in labels.items())


def _normalize_sem_groupby_membership(membership: str | None) -> str | None:
    """Normalize an optional semantic-group membership contract."""

    if membership is None:
        return None
    if membership not in SEM_GROUPBY_MEMBERSHIPS:
        raise ValueError(
            "sem_groupby membership must be one of: "
            + ", ".join(SEM_GROUPBY_MEMBERSHIPS)
        )
    return membership


def _require_relation(value: Any, *, argument: str) -> "Relation":
    """Validate inputs to binary relation operators."""

    if not isinstance(value, Relation):
        raise TypeError(f"{argument} must be a Relation")
    return value


def _normalize_join_on(on: JoinOn) -> tuple[str, ...] | tuple[Mapping[str, Any], ...]:
    """Normalize join keys or predicate expressions into QueryExpr params."""

    if isinstance(on, str):
        keys = (on,)
        if not keys[0]:
            raise ValueError("join key column cannot be empty")
        return keys
    if isinstance(on, Sequence) and not isinstance(on, (str, bytes)):
        values = tuple(on)
        if not values:
            raise ValueError("join requires at least one key column or predicate")
        if all(isinstance(value, str) for value in values):
            keys = tuple(str(value) for value in values)
            if any(not key for key in keys):
                raise ValueError("join key columns cannot be empty")
            return keys
        if any(isinstance(value, str) for value in values):
            raise ValueError("join on sequence cannot mix key column names and predicate expressions")
        return tuple(expr_to_param(ensure_boolean_expr(value)) for value in values)
    return (expr_to_param(ensure_boolean_expr(on)),)


def _normalize_group_keys(keys: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize deterministic group key columns."""

    if isinstance(keys, str):
        normalized = (keys,)
    else:
        normalized = tuple(str(key) for key in keys)
    if not normalized:
        raise ValueError("group_by requires at least one key column")
    if any(not key for key in normalized):
        raise ValueError("group_by key columns cannot be empty")
    return normalized


def _normalize_sem_join_keys(keys: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize exact semantic-join key columns."""

    if isinstance(keys, str):
        normalized = (keys,)
    else:
        normalized = tuple(str(key) for key in keys)
    if not normalized:
        raise ValueError("sem_join on requires at least one key column")
    if any(not key for key in normalized):
        raise ValueError("sem_join on key columns cannot be empty")
    return normalized


def _normalize_drop_duplicates_subset(
    subset: str | Sequence[str],
) -> tuple[str, ...]:
    """Normalize columns that define exact duplicate identity."""

    if isinstance(subset, str):
        normalized = (subset,)
    elif isinstance(subset, Sequence) and not isinstance(subset, (str, bytes)):
        normalized = tuple(str(column) for column in subset)
    else:
        raise TypeError(
            "drop_duplicates subset must be a column name or sequence of "
            "column names"
        )
    if not normalized:
        raise ValueError("drop_duplicates subset cannot be empty")
    if any(not column for column in normalized):
        raise ValueError("drop_duplicates subset column names cannot be empty")
    duplicates = sorted(
        {column for column in normalized if normalized.count(column) > 1}
    )
    if duplicates:
        raise ValueError(
            "drop_duplicates subset column names must be unique: "
            f"{duplicates}"
        )
    return normalized


def _normalize_partition_by(partition_by: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Normalize optional sem_groupby deterministic partition columns."""

    if partition_by is None:
        return None
    if isinstance(partition_by, str):
        normalized = (partition_by,)
    else:
        normalized = tuple(str(key) for key in partition_by)
    if not normalized:
        raise ValueError("sem_groupby partition_by cannot be empty")
    if any(not key for key in normalized):
        raise ValueError("sem_groupby partition_by columns cannot be empty")
    return normalized


def _alias_name(expr: QueryExpr) -> str | None:
    """Return relation alias name if the current handle wraps an alias node."""

    if expr.op == "alias":
        name = expr.params.get("name")
        return str(name) if name is not None else None
    return None


def _normalize_assignment_value(value: object) -> Mapping[str, Any]:
    """Normalize assign values to literal, column, or supported array expression params."""

    expr = ensure_expr(value)
    if isinstance(expr, (ColumnExpr, ArrayCatExpr, LeastExpr)) or is_scalar(value):
        return expr.to_param()
    raise TypeError(
        "assign values must be scalar literals, column expressions, array_cat expressions, "
        "or least expressions"
    )


def _semantic_aggregate_output_names(specs: Sequence[AggregateSpec]) -> tuple[str, ...]:
    """Return output names produced by semantic aggregate specs."""

    outputs: list[str] = []
    for spec in specs:
        if isinstance(spec, SemanticAggregateSpec):
            outputs.extend(aggregate_output_names(spec))
    return tuple(outputs)


class RelationHandle:
    """Internal base for policy authoring handles that wrap a QueryExpr."""

    def __init__(self, expr: QueryExpr) -> None:
        self.expr = expr


class Relation(RelationHandle):
    """DataFrame-like chain handle for policy authoring.

    Operator methods only build QueryExpr nodes. They do not execute queries,
    call models, or materialize memory state.
    """

    def _derive(
        self,
        op: str,
        *,
        inputs: Sequence[QueryExpr] | None = None,
        **params: Any,
    ) -> "Relation":
        """Build a new Relation by appending one logical expression node."""

        return Relation(
            QueryExpr(
                op=op,
                inputs=tuple(inputs) if inputs is not None else (self.expr,),
                params=params,
            )
        )

    def search(
        self,
        query: UserQuery,
        *,
        methods: Sequence[Any],
        reranker: Any | None,
        limit: int,
    ) -> "SearchRelation":
        """Build one storage-backed ranked search relation."""

        from .retrieval import normalize_search

        if not isinstance(query, UserQuery):
            raise TypeError("search query must be UserQuery")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("search limit must be a positive integer")
        source_columns = output_columns(self.expr)
        metadata_columns = {"record_id", "rank", "score"}
        conflicts = sorted(metadata_columns.intersection(source_columns))
        if conflicts:
            raise ValueError(
                f"search metadata columns conflict with source columns: {conflicts}"
            )
        method_specs, reranker_spec, dependencies = normalize_search(methods, reranker)
        return SearchRelation(
            QueryExpr(
                op="search",
                inputs=(self.expr, *dependencies),
                params={
                    "query": query,
                    "methods": method_specs,
                    "reranker": reranker_spec,
                    "limit": limit,
                },
            )
        )

    def select(self, columns: Sequence[str]) -> "Relation":
        """Project relation columns."""

        return self._derive("select", columns=tuple(columns))

    def alias(self, name: str) -> "Relation":
        """Assign a temporary relation qualifier for self-joins."""

        if not isinstance(name, str) or not name:
            raise ValueError("alias name must be a non-empty string")
        return self._derive("alias", name=name)

    def col(self, name: str) -> ColumnExpr:
        """Return a relation-bound column expression."""

        if not isinstance(name, str) or not name:
            raise ValueError("column name must be a non-empty string")
        return ColumnExpr(name=name, qualifier=_alias_name(self.expr))

    def __getitem__(self, columns: Sequence[str]) -> "Relation":
        """Pandas-style column projection sugar."""

        if isinstance(columns, str):
            raise TypeError("single-column selection is not supported in the v0.0 interface")
        return self.select(columns)

    def filter(self, predicate: Any) -> "Relation":
        """Add a deterministic row filter expression."""

        return self._derive("filter", predicate=expr_to_param(ensure_boolean_expr(predicate)))

    def assign(self, **assignments: Any) -> "Relation":
        """Add deterministic column assignment expressions."""

        return self._derive(
            "assign",
            assignments={
                str(column): _normalize_assignment_value(value)
                for column, value in assignments.items()
            },
        )

    def concat(self, other: "Relation") -> "Relation":
        """Append rows with union-all semantics."""

        other = _require_relation(other, argument="other")
        return self._derive("concat", inputs=(self.expr, other.expr))

    def union(self, other: "Relation") -> "Relation":
        """Append rows and remove exact duplicates."""

        other = _require_relation(other, argument="other")
        return self._derive("union", inputs=(self.expr, other.expr))

    def union_by_name(
        self,
        other: "Relation",
        *,
        allow_missing_columns: bool = True,
    ) -> "Relation":
        """Append rows by column name and remove exact duplicates."""

        other = _require_relation(other, argument="other")
        if not isinstance(allow_missing_columns, bool):
            raise TypeError("union_by_name allow_missing_columns must be a bool")
        return self._derive(
            "union_by_name",
            inputs=(self.expr, other.expr),
            allow_missing_columns=allow_missing_columns,
        )

    def subtract(self, other: "Relation") -> "Relation":
        """Remove rows using exact set-difference semantics."""

        other = _require_relation(other, argument="other")
        return self._derive("subtract", inputs=(self.expr, other.expr))

    def join(
        self,
        other: "Relation",
        *,
        on: JoinOn,
        how: str = "inner",
    ) -> "Relation":
        """Join two relations using exact keys or deterministic predicates."""

        other = _require_relation(other, argument="other")
        return self._derive(
            "join",
            inputs=(self.expr, other.expr),
            on=_normalize_join_on(on),
            how=how,
        )

    def group_by(self, keys: str | Sequence[str]) -> "GroupedRelation":
        """Create a deterministic grouped relation expression."""

        return GroupedRelation(
            QueryExpr(
                op="group_by",
                inputs=(self.expr,),
                params={"keys": _normalize_group_keys(keys)},
            )
        )

    def drop_duplicates(
        self,
        *,
        subset: str | Sequence[str] | None = None,
    ) -> "Relation":
        """Remove duplicate rows, optionally comparing only selected columns."""

        if subset is None:
            return self._derive("drop_duplicates")
        normalized = _normalize_drop_duplicates_subset(subset)
        missing = sorted(set(normalized).difference(output_columns(self.expr)))
        if missing:
            raise ValueError(
                f"drop_duplicates subset columns not found: {missing}"
            )
        return self._derive("drop_duplicates", subset=normalized)

    def array_agg(self, *, columns: Sequence[str], output_col: str) -> "Relation":
        """Aggregate relation rows into one JSON array-of-records column."""

        cols = tuple(str(column) for column in columns)
        if not cols:
            raise ValueError("array_agg columns cannot be empty")
        if not output_col:
            raise ValueError("array_agg output_col cannot be empty")
        return self._derive("array_agg", columns=cols, output_col=str(output_col))

    def min(
        self,
        *,
        column: str | None = None,
        columns: Sequence[str] | None = None,
        output_col: str,
    ) -> "Relation":
        """Aggregate this relation into one minimum value."""

        normalized_columns = normalize_min_columns(column=column, columns=columns)
        if not output_col:
            raise ValueError("min output_col cannot be empty")
        return self._derive(
            "min",
            columns=normalized_columns,
            output_col=str(output_col),
        )

    def array_cat(self, other: "Relation", *, column: str) -> "Relation":
        """Concatenate one JSON array aggregate-state column."""

        other = _require_relation(other, argument="other")
        if not column:
            raise ValueError("array_cat column cannot be empty")
        return self._derive("array_cat", inputs=(self.expr, other.expr), column=str(column))

    def flatten(self, *, column: str, output_col: str | None = None) -> "Relation":
        """Flatten one JSON array-of-arrays column into a JSON array column."""

        if not column:
            raise ValueError("flatten column cannot be empty")
        if output_col is not None and not output_col:
            raise ValueError("flatten output_col cannot be empty")
        return self._derive(
            "flatten",
            column=str(column),
            output_col=None if output_col is None else str(output_col),
        )

    def explode(self, *, column: str, output_col: str | None = None) -> "Relation":
        """Expand one JSON array column into one row per element."""

        if not column:
            raise ValueError("explode column cannot be empty")
        if output_col is not None and not output_col:
            raise ValueError("explode output_col cannot be empty")
        return self._derive(
            "explode",
            column=str(column),
            output_col=None if output_col is None else str(output_col),
        )

    def unnest(self, *, column: str, fields: Mapping[str, str]) -> "Relation":
        """Expand one JSON object column into ordinary columns."""

        if not column:
            raise ValueError("unnest column cannot be empty")
        if not fields:
            raise ValueError("unnest fields cannot be empty")
        normalized_fields = tuple((str(field), str(output)) for field, output in fields.items())
        if any(not field for field, _ in normalized_fields):
            raise ValueError("unnest field names cannot be empty")
        if any(not output for _, output in normalized_fields):
            raise ValueError("unnest output columns cannot be empty")
        outputs = [output for _, output in normalized_fields]
        duplicates = sorted({output for output in outputs if outputs.count(output) > 1})
        if duplicates:
            raise ValueError(f"unnest output columns must be unique: {duplicates}")
        return self._derive("unnest", column=str(column), fields=normalized_fields)

    def count_window(
        self,
        *,
        size: int,
        slide: int = 1,
        trigger: None = None,
    ) -> "WindowedRelation":
        """Assign relation rows to completed count windows."""

        if trigger is not None:
            raise NotImplementedError("count_window currently supports only trigger=None")
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("count_window size must be an integer")
        if isinstance(slide, bool) or not isinstance(slide, int):
            raise TypeError("count_window slide must be an integer")
        if size <= 0:
            raise ValueError("count_window size must be positive")
        if slide <= 0:
            raise ValueError("count_window slide must be positive")
        return WindowedRelation(
            QueryExpr(
                op="count_window",
                inputs=(self.expr,),
                params={
                    "size": size,
                    "slide": slide,
                    "trigger": trigger,
                },
            )
        )

    def over(self, *, rows: tuple[int, int]) -> "OverRelation":
        """Create a row-preserving over-window handle."""

        if (
            not isinstance(rows, tuple)
            or len(rows) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in rows)
        ):
            raise TypeError("over rows must be a tuple of two integers")
        start, end = rows
        if start > end:
            raise ValueError("over rows start must be <= end")
        if end > 0:
            raise NotImplementedError("over currently supports only rows with N <= 0")
        return OverRelation(
            QueryExpr(
                op="over",
                inputs=(self.expr,),
                params={"rows": rows},
            )
        )

    def sem_filter(self, *, instruction: str) -> "Relation":
        """Add a semantic row filter expression."""

        return self._derive("sem_filter", instruction=instruction)

    def sem_map(
        self,
        *,
        input_cols: ColumnInput = None,
        output_cols: ColumnOutput,
        instruction: str,
    ) -> "Relation":
        """Add or replace columns using a semantic row transform."""

        return self._derive(
            "sem_map",
            input_cols=_normalize_input_cols(input_cols),
            output_cols=_normalize_output_cols(output_cols),
            instruction=instruction,
        )

    def sem_flat_map(
        self,
        *,
        input_cols: ColumnInput = None,
        output_cols: ColumnOutput,
        instruction: str,
        ordinal_col: str | None = None,
    ) -> "Relation":
        """Add columns while allowing one input row to emit many output rows."""

        normalized_output_cols = _normalize_output_cols(output_cols)
        if ordinal_col is not None:
            if not ordinal_col:
                raise ValueError("sem_flat_map ordinal_col cannot be empty")
            existing_columns = set(output_columns(self.expr))
            generated_columns = {column.name for column in normalized_output_cols or ()}
            if ordinal_col in existing_columns or ordinal_col in generated_columns:
                raise ValueError(
                    "sem_flat_map ordinal_col conflicts with an existing or "
                    f"output column: {ordinal_col!r}"
                )
        params: dict[str, Any] = {
            "input_cols": _normalize_input_cols(input_cols),
            "output_cols": normalized_output_cols,
            "instruction": instruction,
        }
        if ordinal_col is not None:
            params["ordinal_col"] = str(ordinal_col)
        return self._derive("sem_flat_map", **params)

    def sem_groupby(
        self,
        *,
        input_cols: Sequence[str],
        instruction: str,
        partition_by: str | Sequence[str] | None = None,
        labels: Mapping[str, str] | None = None,
        label_col: str = "_label",
        membership: str | None = None,
    ) -> "GroupedRelation":
        """Create a grouped semantic relation expression."""

        normalized_partition_by = _normalize_partition_by(partition_by)
        normalized_membership = _normalize_sem_groupby_membership(membership)
        params: dict[str, Any] = {
            "input_cols": _normalize_input_cols(input_cols),
            "instruction": instruction,
            "labels": _normalize_group_labels(labels),
            "label_col": label_col,
        }
        if normalized_partition_by is not None:
            params["partition_by"] = normalized_partition_by
        if normalized_membership is not None:
            params["membership"] = normalized_membership
        expr = QueryExpr(
            op="sem_groupby",
            inputs=(self.expr,),
            params=params,
        )
        return GroupedRelation(expr)

    def sem_join(
        self,
        other: "Relation",
        *,
        instruction: str,
        how: str = "inner",
        on: str | Sequence[str] | None = None,
        k: int | None = None,
    ) -> "Relation":
        """Add a semantic join expression."""

        other = _require_relation(other, argument="other")
        params: dict[str, Any] = {
            "instruction": instruction,
            "how": how,
        }
        if on is not None:
            params["on"] = _normalize_sem_join_keys(on)
        if k is not None:
            if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                raise ValueError("sem_join k must be a positive integer")
            params["k"] = k
        return self._derive("sem_join", inputs=(self.expr, other.expr), **params)

    def sem_topk(
        self,
        instruction: str | UserQuery,
        k: int,
    ) -> "Relation":
        """Add a semantic top-k expression."""

        return self._derive(
            "sem_topk",
            instruction=instruction,
            k=k,
        )

    def sem_agg(
        self,
        *,
        input_cols: ColumnInput = None,
        output_cols: ColumnOutput | None = None,
        instruction: str,
    ) -> "Relation":
        """Aggregate this whole relation into semantic output columns."""

        return self._derive(
            "sem_agg",
            input_cols=_normalize_input_cols(input_cols),
            output_cols=_normalize_output_cols(output_cols),
            instruction=instruction,
        )


class SearchRelation(Relation):
    """Query-time relation handle that never becomes maintained memory state."""

    def _derive(
        self,
        op: str,
        *,
        inputs: Sequence[QueryExpr] | None = None,
        **params: Any,
    ) -> "SearchRelation":
        return SearchRelation(
            QueryExpr(
                op=op,
                inputs=tuple(inputs) if inputs is not None else (self.expr,),
                params=params,
            )
        )


class Log(Relation):
    """Append-only source relation for one memory policy."""

    def __init__(
        self,
        columns: Mapping[str, str] | None = None,
        *,
        system_columns: bool = False,
    ) -> None:
        if not isinstance(system_columns, bool):
            raise TypeError("Log system_columns must be a bool")
        declared_columns = columns or {"message": "Raw memory log message."}
        reserved = sorted(set(declared_columns).intersection(LOG_SYSTEM_COLUMNS))
        if reserved:
            raise ValueError(f"Log columns contain reserved system column names: {reserved}")
        column_defs = tuple(
            ColumnSpec(name=name, description=description)
            for name, description in declared_columns.items()
        )
        if system_columns:
            column_defs += LOG_SYSTEM_COLUMN_SPECS
        params: dict[str, Any] = {"columns": column_defs}
        if system_columns:
            params["system_columns"] = True
        super().__init__(QueryExpr(op="log", params=params))


class GroupedRelation(RelationHandle):
    """Intermediate grouped expression returned by sem_groupby.

    A GroupedRelation represents semantic partition state before aggregation. It
    becomes a normal Relation only after sem_agg is called.
    """

    def sem_agg(
        self,
        *,
        input_cols: ColumnInput = None,
        output_cols: ColumnOutput | None = None,
        instruction: str,
    ) -> Relation:
        """Aggregate each semantic group into output columns."""

        return Relation(
            QueryExpr(
                op="sem_agg",
                inputs=(self.expr,),
                params={
                    "input_cols": _normalize_input_cols(input_cols),
                    "output_cols": _normalize_output_cols(output_cols),
                    "instruction": instruction,
                },
            )
        )

    def array_agg(self, *, columns: Sequence[str], output_col: str) -> Relation:
        """Aggregate each group into one JSON array-of-records column."""

        if self.expr.op == "sem_groupby":
            raise NotImplementedError(
                "sem_groupby(...).array_agg(...) is not supported; use "
                "sem_groupby(...).agg(sem_agg(...), array_agg(...)) so semantic "
                "keys are produced by sem_agg output columns."
            )
        cols = tuple(str(column) for column in columns)
        if not cols:
            raise ValueError("grouped array_agg columns cannot be empty")
        if not output_col:
            raise ValueError("grouped array_agg output_col cannot be empty")
        if self.expr.op == "group_by":
            keys = tuple(str(key) for key in self.expr.params["keys"])
            if output_col in keys:
                raise ValueError(
                    f"grouped array_agg output column conflicts with group key: {output_col!r}"
                )
        return Relation(
            QueryExpr(
                op="array_agg",
                inputs=(self.expr,),
                params={"columns": cols, "output_col": str(output_col)},
            )
        )

    def min(
        self,
        *,
        column: str | None = None,
        columns: Sequence[str] | None = None,
        output_col: str,
    ) -> Relation:
        """Aggregate each deterministic group into one minimum value."""

        if self.expr.op == "sem_groupby":
            raise NotImplementedError(
                "sem_groupby(...).min(...) is not supported; use "
                "sem_groupby(...).agg(sem_agg(...), min(...))."
            )
        normalized_columns = normalize_min_columns(column=column, columns=columns)
        if not output_col:
            raise ValueError("grouped min output_col cannot be empty")
        keys = tuple(str(key) for key in self.expr.params["keys"])
        if output_col in keys:
            raise ValueError(
                f"grouped min output column conflicts with group key: {output_col!r}"
            )
        return Relation(
            QueryExpr(
                op="min",
                inputs=(self.expr,),
                params={"columns": normalized_columns, "output_col": str(output_col)},
            )
        )

    def agg(self, *aggregates: AggregateSpec) -> Relation:
        """Aggregate each group with one or more declared aggregate functions."""

        specs = normalize_aggregate_specs(aggregates)
        if self.expr.op == "sem_groupby":
            semantic_outputs = set(_semantic_aggregate_output_names(specs))
            if not semantic_outputs:
                raise NotImplementedError(
                    "sem_groupby(...).agg(...) requires at least one sem_agg(...) "
                    "spec; deterministic aggregates cannot generate semantic keys."
                )
            semantic_keys = set(str(column) for column in self.expr.params["input_cols"])
            missing_keys = sorted(semantic_keys.difference(semantic_outputs))
            if missing_keys:
                raise ValueError(
                    "sem_groupby(...).agg(...) requires semantic keys to appear "
                    f"in sem_agg output_cols; missing {missing_keys}."
                )
        return Relation(
            QueryExpr(
                op="agg",
                inputs=(self.expr,),
                params={"aggregates": specs},
            )
        )


class WindowedRelation(RelationHandle):
    """Count-window expression that must be closed by process_window."""

    def process_window(self, builder: Callable[[Relation], Relation]) -> Relation:
        """Build a window function subtree and return a normal relation."""

        if self.expr.op != "count_window" or len(self.expr.inputs) != 1:
            raise ValueError("process_window requires a count_window relation")
        window_source = Relation(
            QueryExpr(
                op="window_source",
                params={"columns": output_columns(self.expr.inputs[0])},
            )
        )
        result = builder(window_source)
        if not isinstance(result, Relation):
            raise TypeError("process_window builder must return a Relation")
        return Relation(
            QueryExpr(
                op="process_window",
                inputs=(self.expr, result.expr),
            )
        )


class OverRelation(RelationHandle):
    """Over-window expression that must be closed by one window function."""

    def array_agg(self, *, columns: Sequence[str], output_col: str) -> Relation:
        """Add one row-preserving JSON array frame aggregate column."""

        cols = tuple(str(column) for column in columns)
        if not cols:
            raise ValueError("over array_agg columns cannot be empty")
        if not output_col:
            raise ValueError("over array_agg output_col cannot be empty")
        return Relation(
            QueryExpr(
                op="array_agg",
                inputs=(self.expr,),
                params={"columns": cols, "output_col": str(output_col)},
            )
        )

    def sem_agg(
        self,
        *,
        input_cols: ColumnInput = None,
        output_cols: ColumnOutput,
        instruction: str,
    ) -> Relation:
        """Add row-preserving semantic aggregate columns over each frame."""

        if output_cols is None:
            raise ValueError("over sem_agg requires explicit output_cols")
        return Relation(
            QueryExpr(
                op="sem_agg",
                inputs=(self.expr,),
                params={
                    "input_cols": _normalize_input_cols(input_cols),
                    "output_cols": _normalize_output_cols(output_cols),
                    "instruction": instruction,
                },
            )
        )
