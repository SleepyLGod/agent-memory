"""Independent semantic grouping contracts and authoring handles."""

from dataclasses import replace
from collections.abc import Sequence
from typing import Any

from .aggregates import (
    AggregateSpec,
    SemanticAggregateSpec,
    aggregate_output_names,
    is_algebraic_aggregate_spec,
)
from .logical import ColumnSpec, QueryExpr
from .relation import (
    ColumnInput,
    ColumnOutput,
    GroupedRelation,
    LOG_SYSTEM_COLUMNS,
    Relation,
    _normalize_partition_by,
)
from .schema import output_columns

RECORD_ID = "_agent_memory_record_id"
GROUP_ID = "_agent_memory_group_id"


class SemanticGroupedRelation(Relation):
    """Readable memberships with convenience aggregation by group identity."""

    def _derive(
        self, op: str, *, inputs: Sequence[QueryExpr] | None = None, **params: Any
    ) -> "SemanticGroupedRelation":
        if op not in {"filter", "select", "alias"}:
            raise NotImplementedError(f"{op} does not preserve semantic grouping yet")
        scope = self.expr.params["grouping"]
        if op == "select" and not set(scope["keys"]).issubset(params["columns"]):
            raise ValueError("select must retain all semantic group keys")
        if op == "select" and not set(params["columns"]).issubset(
            output_columns(self.expr)
        ):
            raise ValueError("select references unknown columns")
        return SemanticGroupedRelation(
            QueryExpr(
                op=op,
                inputs=tuple(inputs) if inputs is not None else (self.expr,),
                params={**params, "grouping": scope},
            )
        )

    def _groups(self) -> GroupedRelation:
        scope = self.expr.params["grouping"]
        return GroupedRelation(
            QueryExpr(
                op="group_by",
                inputs=(self.expr,),
                params={
                    "keys": scope["keys"],
                    "explicit_grouping": True,
                },
            )
        )

    def _check_outputs(self, names: Sequence[str]) -> None:
        grouped = self._groups().expr
        if set(names) & set(grouped.params["keys"]):
            raise ValueError("aggregate output conflicts with group identity")

    def sem_agg(
        self,
        *,
        input_cols: ColumnInput = None,
        output_cols: ColumnOutput | None = None,
        instruction: str,
        S: object = None,
    ) -> Relation:
        """Aggregate each group without regenerating its identity."""
        selected = business_columns(self.expr, input_cols)
        grouped = self._groups().expr
        retained = set(grouped.params["keys"])
        outputs = (
            tuple(col for col in selected if col not in retained)
            if output_cols is None
            else output_cols
        )
        if not outputs:
            raise ValueError("sem_agg requires at least one non-key output column")
        self._check_outputs(tuple(outputs))
        result = self._groups().sem_agg(
            input_cols=selected,
            output_cols=outputs,
            instruction=instruction,
            S=S,
        )
        return _fallback(result)

    def agg(self, *aggregates: AggregateSpec) -> Relation:
        """Apply existing aggregate descriptors independently to every group."""
        specs = tuple(
            replace(spec, input_cols=business_columns(self.expr, spec.input_cols))
            if isinstance(spec, SemanticAggregateSpec)
            else spec
            for spec in aggregates
        )
        self._check_outputs(
            tuple(name for spec in specs for name in aggregate_output_names(spec))
        )
        result = self._groups().agg(*specs)
        if all(is_algebraic_aggregate_spec(spec) for spec in specs):
            return result
        return _fallback(result)


def business_columns(
    query: QueryExpr, explicit: Sequence[str] | None = None
) -> tuple[str, ...]:
    """Resolve model inputs from declared columns, not incidental frame contents."""
    columns = output_columns(query)
    if explicit is not None:
        selected = tuple(explicit)
    else:
        excluded = set(LOG_SYSTEM_COLUMNS)
        scope = query.params.get("grouping", {})
        excluded.update(scope.get("metadata", ()))
        selected = tuple(col for col in columns if col not in excluded)
    if not selected or not set(selected).issubset(columns):
        raise ValueError("semantic input columns must be nonempty existing columns")
    return selected


def normalize_fallback(query: QueryExpr) -> QueryExpr:
    """Bind standalone fallback inputs and inferred outputs before execution."""
    if query.op != "sem_agg":
        return query
    parent = query.inputs[0]
    source = parent.inputs[0] if parent.op == "group_by" else parent
    selected = business_columns(source, query.params.get("input_cols"))
    outputs = query.params.get("output_cols")
    if outputs is None:
        outputs = tuple(ColumnSpec(name=col) for col in selected)
    return QueryExpr(
        op=query.op,
        inputs=query.inputs,
        params={
            **query.params,
            "input_cols": selected,
            "output_cols": outputs,
            "maintenance_rule": "agg-fallback",
        },
    )


def _fallback(relation: Relation) -> Relation:
    query = relation.expr
    return Relation(
        QueryExpr(
            op=query.op,
            inputs=query.inputs,
            params={**query.params, "maintenance_rule": "agg-fallback"},
        )
    )


def make_semantic_groups(
    source: Relation,
    *,
    input_cols: Sequence[str] | None,
    instruction: str,
    partition_by: str | Sequence[str] | None,
    membership: str,
    C: int | None,
) -> SemanticGroupedRelation:
    """Validate and construct an independent membership relation."""
    columns = output_columns(source.expr)
    reserved = (RECORD_ID, GROUP_ID)
    if set(columns) & set(reserved):
        raise ValueError(
            "semantic grouping metadata columns conflict with input columns"
        )
    selected = business_columns(source.expr, input_cols)
    partitions = _normalize_partition_by(partition_by) or ()
    if not selected or not instruction:
        raise ValueError(
            "semantic grouping requires business input columns and instruction"
        )
    if not set((*selected, *partitions)).issubset(columns):
        raise ValueError("semantic grouping references unknown input columns")
    return SemanticGroupedRelation(
        QueryExpr(
            op="sem_groupby",
            inputs=(source.expr,),
            params={
                "input_cols": selected,
                "instruction": instruction,
                "membership": membership,
                "partition_by": partitions,
                "C": C,
                "independent": True,
                "maintenance_rule": "group-fallback",
                "grouping": {
                    "keys": (*partitions, GROUP_ID),
                    "metadata": reserved,
                },
                "columns": (*columns, *reserved),
            },
        )
    )
