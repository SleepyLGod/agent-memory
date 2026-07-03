"""DataFrame-style relation authoring handles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .logical import ColumnSpec, QueryExpr, UserQuery
from .query_schema import output_columns

ColumnInput = Sequence[str] | None
ColumnOutput = Sequence[str] | Mapping[str, str]


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


def _require_relation(value: Any, *, argument: str) -> "Relation":
    """Validate inputs to binary relation operators."""

    if not isinstance(value, Relation):
        raise TypeError(f"{argument} must be a Relation")
    return value


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

    def select(self, columns: Sequence[str]) -> "Relation":
        """Project relation columns."""

        return self._derive("select", columns=tuple(columns))

    def __getitem__(self, columns: Sequence[str]) -> "Relation":
        """Pandas-style column projection sugar."""

        if isinstance(columns, str):
            raise TypeError("single-column selection is not supported in the v0.0 interface")
        return self.select(columns)

    def filter(self, predicate: Any) -> "Relation":
        """Add a deterministic row filter expression.

        The current v0.0 interface accepts Any for now. Before planner
        execution, this should be narrowed to a serializable column expression
        instead of arbitrary Python callables.
        """

        return self._derive("filter", predicate=predicate)

    def assign(self, **assignments: Any) -> "Relation":
        """Add deterministic column assignment expressions.

        The current v0.0 interface accepts Any for now. Before planner
        execution, assignments should be represented as serializable column
        expressions instead of arbitrary Python callables.
        """

        return self._derive("assign", assignments=dict(assignments))

    def concat(self, other: "Relation") -> "Relation":
        """Append rows with union-all semantics."""

        other = _require_relation(other, argument="other")
        return self._derive("concat", inputs=(self.expr, other.expr))

    def union(self, other: "Relation") -> "Relation":
        """Append rows and remove exact duplicates."""

        other = _require_relation(other, argument="other")
        return self._derive("union", inputs=(self.expr, other.expr))

    def subtract(self, other: "Relation") -> "Relation":
        """Remove rows using exact set-difference semantics."""

        other = _require_relation(other, argument="other")
        return self._derive("subtract", inputs=(self.expr, other.expr))

    def drop_duplicates(self) -> "Relation":
        """Remove exact duplicate rows."""

        return self._derive("drop_duplicates")

    def array_agg(self, *, columns: Sequence[str], output_col: str) -> "Relation":
        """Aggregate relation rows into one JSON array-of-records column."""

        cols = tuple(str(column) for column in columns)
        if not cols:
            raise ValueError("array_agg columns cannot be empty")
        if not output_col:
            raise ValueError("array_agg output_col cannot be empty")
        return self._derive("array_agg", columns=cols, output_col=str(output_col))

    def array_cat(self, other: "Relation", *, column: str) -> "Relation":
        """Concatenate one JSON array aggregate-state column."""

        other = _require_relation(other, argument="other")
        if not column:
            raise ValueError("array_cat column cannot be empty")
        return self._derive("array_cat", inputs=(self.expr, other.expr), column=str(column))

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
    ) -> "Relation":
        """Add columns while allowing one input row to emit many output rows."""

        return self._derive(
            "sem_flat_map",
            input_cols=_normalize_input_cols(input_cols),
            output_cols=_normalize_output_cols(output_cols),
            instruction=instruction,
        )

    def sem_groupby(
        self,
        *,
        input_cols: Sequence[str],
        instruction: str,
        labels: Mapping[str, str] | None = None,
        label_col: str = "_label",
    ) -> "GroupedRelation":
        """Create a grouped semantic relation expression."""

        expr = QueryExpr(
            op="sem_groupby",
            inputs=(self.expr,),
            params={
                "input_cols": _normalize_input_cols(input_cols),
                "instruction": instruction,
                "labels": _normalize_group_labels(labels),
                "label_col": label_col,
            },
        )
        return GroupedRelation(expr)

    def sem_join(
        self,
        other: "Relation",
        *,
        instruction: str,
        how: str = "inner",
    ) -> "Relation":
        """Add a semantic join expression."""

        other = _require_relation(other, argument="other")
        return self._derive(
            "sem_join",
            inputs=(self.expr, other.expr),
            instruction=instruction,
            how=how,
        )

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
