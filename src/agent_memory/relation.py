"""DataFrame-style relation authoring handles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .logical import ColumnSpec, QueryExpr, UserQuery

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


class Relation:
    """DataFrame-like chain handle for policy authoring.

    Operator methods only build QueryExpr nodes. They do not execute queries,
    call models, or materialize memory state.
    """

    def __init__(self, expr: QueryExpr) -> None:
        self.expr = expr

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

    def join(
        self,
        other: "Relation",
        *,
        on: str | Sequence[str],
        how: str = "inner",
    ) -> "Relation":
        """Join two relations on same-named deterministic key columns."""

        other = _require_relation(other, argument="other")
        if isinstance(on, str):
            keys = (on,)
        else:
            keys = tuple(on)
        if not keys:
            raise ValueError("join requires at least one key column")
        return self._derive(
            "join",
            inputs=(self.expr, other.expr),
            on=keys,
            how=how,
        )

    def drop_duplicates(self) -> "Relation":
        """Remove exact duplicate rows."""

        return self._derive("drop_duplicates")

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


class GroupedRelation:
    """Intermediate grouped expression returned by sem_groupby.

    A GroupedRelation represents semantic partition state before aggregation. It
    becomes a normal Relation only after sem_agg is called.
    """

    def __init__(self, expr: QueryExpr) -> None:
        self.expr = expr

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
