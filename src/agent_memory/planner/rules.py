"""Differential rules for hardcoded Q -> Q' rewrites."""

from __future__ import annotations

from collections.abc import Sequence
import re

from agent_memory.policy.aggregates import (
    ArrayAggregateSpec,
    CollectListAggregateSpec,
    MinAggregateSpec,
    SemanticAggregateSpec,
    aggregate_output_names,
)
from agent_memory.policy.expressions import ColumnExpr, least
from agent_memory.policy.logical import ColumnSpec, QueryExpr
from agent_memory.policy.schema import output_columns

ROW_LOCAL_UNARY_OPS = {
    "select",
    "explode",
    "unnest",
    "sem_filter",
    "sem_map",
    "sem_flat_map",
}
CHANGED_FRAGMENT_BINARY_OPS = {"concat", "union"}
GROUP_ID_COLUMN = "_agent_memory_group_id"
CHANGED_MARKER_COLUMN = "_changed"
JOIN_MAP_LEFT_ID_COLUMN = "_agent_memory_join_map_left_id"
JOIN_MAP_RIGHT_ID_COLUMN = "_agent_memory_join_map_right_id"
JOIN_MAP_BINDING_NAME = "__agent_memory_join_map_pairs"
GROUPED_AGG_RULE_ALIASES = {
    "compressed": "rule-all-group",
    "changed-aware": "rule-all-group-optimized",
    "join-map": "rule-join-map",
}
GROUPED_AGG_RULES = (
    "compressed",
    "changed-aware",
    "join-map",
    "rule-join-map",
    "rule-re-group",
    "rule-all-group",
    "rule-all-group-optimized",
)
PLACEHOLDER_PATTERN = re.compile(
    r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)(?::(left|right))?\}(?!\})"
)


class DifferentialInstructionRewriter:
    """Instruction rewrite hook for differentiated semantic operators."""

    def groupby_to_join(
        self,
        instruction: str,
        *,
        input_cols: Sequence[str],
    ) -> str:
        """Rewrite a sem_groupby instruction for differentiated sem_join."""

        return self._rewrite_placeholders(
            instruction,
            input_cols=input_cols,
            output_cols=(),
        )

    def agg_to_map(
        self,
        instruction: str,
        *,
        input_cols: Sequence[str],
        output_cols: Sequence[ColumnSpec],
    ) -> str:
        """Rewrite a sem_agg instruction for differentiated sem_map."""

        return self._rewrite_placeholders(
            instruction,
            input_cols=input_cols,
            output_cols=tuple(column.name for column in output_cols),
        )

    def state_reaggregation(
        self,
        instruction: str,
        *,
        state_cols: Sequence[str],
        raw_input_cols: Sequence[str],
    ) -> str:
        """Adapt raw-input placeholders to an aggregate-state reaggregation."""

        state_names = {str(column) for column in state_cols}
        raw_input_names = {str(column) for column in raw_input_cols}

        def replace(match: re.Match[str]) -> str:
            column = match.group(1)
            side = match.group(2)
            state_column = column if side is None else f"{column}:{side}"
            if state_column in state_names:
                return match.group(0)
            if side is None and column in raw_input_names:
                return column
            raise ValueError(
                f"Cannot reaggregate instruction placeholder {{{column}}}: it is not "
                "present in aggregate state and is not a declared raw input column."
            )

        return PLACEHOLDER_PATTERN.sub(replace, instruction)

    def _rewrite_placeholders(
        self,
        instruction: str,
        *,
        input_cols: Sequence[str],
        output_cols: Sequence[str],
    ) -> str:
        """Rewrite simple column placeholders after a schema-changing rewrite."""

        input_names = {str(column) for column in input_cols}
        output_names = {str(column) for column in output_cols}

        def replace(match: re.Match[str]) -> str:
            column = match.group(1)
            side = match.group(2)

            if side is not None:
                return match.group(0)
            if column in input_names:
                return f"{{{column}:left}} and {{{column}:right}}"
            if column in output_names:
                return match.group(0)
            raise ValueError(
                f"Cannot rewrite instruction placeholder {{{column}}}: it is not "
                "declared as an input or output column. Use double braces like "
                f"{{{{{column}}}}} for literal text that is not a column placeholder."
            )

        return PLACEHOLDER_PATTERN.sub(replace, instruction)


class DifferentialRules:
    """Operator-level and pattern-level differential rules."""

    def __init__(self, *, grouped_agg_rule: str = "compressed") -> None:
        """Configure differential rule strategy experiments."""

        if grouped_agg_rule not in GROUPED_AGG_RULES:
            raise ValueError(
                "grouped_agg_rule must be one of: " + ", ".join(GROUPED_AGG_RULES)
            )
        self._grouped_agg_rule = GROUPED_AGG_RULE_ALIASES.get(
            grouped_agg_rule,
            grouped_agg_rule,
        )

    @property
    def grouped_agg_rule(self) -> str:
        """Return the canonical grouped aggregate rule name."""

        return self._grouped_agg_rule

    def differentiate(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None = None,
        is_view_boundary: bool = False,
        instruction_rewriter: DifferentialInstructionRewriter | None = None,
    ) -> QueryExpr:
        """Differentiate a query into a changed fragment or full Q'."""

        rewriter = instruction_rewriter or DifferentialInstructionRewriter()
        if is_view_boundary:
            return self._differentiate_to_view(
                query,
                source_input=source_input,
                current_view=current_view,
                source_query=source_query,
                instruction_rewriter=rewriter,
            )

        return self._differentiate_to_rows(
            query,
            source_input=source_input,
            current_view=current_view,
            source_query=source_query,
            instruction_rewriter=rewriter,
        )

    def _differentiate_to_view(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Differentiate a view definition query into a full next-view query."""

        for rule in (
            self._differentiate_sem_groupby_agg_view,
            self._differentiate_grouped_agg_view,
            self._differentiate_min_view,
            self._differentiate_array_agg_view,
            self._differentiate_sem_agg_view,
        ):
            differentiated = rule(
                query,
                source_input=source_input,
                current_view=current_view,
                source_query=source_query,
                instruction_rewriter=instruction_rewriter,
            )
            if differentiated is not None:
                return differentiated

        changed_rows = self._differentiate_to_rows(
            query,
            source_input=source_input,
            current_view=current_view,
            source_query=source_query,
            instruction_rewriter=instruction_rewriter,
        )
        return QueryExpr(op="union", inputs=(current_view, changed_rows))

    def _differentiate_to_rows(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Differentiate one query subtree into changed output rows."""

        changed_rows = self._differentiate_source_input(
            query,
            source_input=source_input,
            current_view=current_view,
            source_query=source_query,
            instruction_rewriter=instruction_rewriter,
        )
        if changed_rows is not None:
            return changed_rows
        self._raise_if_rows_unsupported(query)
        for rule in (
            self._differentiate_row_local_operator,
            self._differentiate_append_binary_operator,
            self._differentiate_sem_join,
        ):
            changed_rows = rule(
                query,
                source_input=source_input,
                current_view=current_view,
                source_query=source_query,
                instruction_rewriter=instruction_rewriter,
            )
            if changed_rows is not None:
                return changed_rows
        raise NotImplementedError(f"No differential rule for QueryExpr op {query.op!r}.")

    def _raise_if_rows_unsupported(self, query: QueryExpr) -> None:
        """Reject patterns that cannot produce changed rows in v0.0."""

        if query.op == "materialized_view":
            raise NotImplementedError(
                "materialized_view appeared during fragment differentiation; "
                "v0.0 supports cascaded view dependencies only when the dependent "
                "view contains no base log dependencies. Mixed log + upstream "
                "materialized view differentials require upstream changed-view "
                "state and are not implemented yet."
            )
        if query.op == "sem_groupby":
            raise NotImplementedError(
                "sem_groupby differential is only supported as part of a view-boundary sem_groupby(...).sem_agg(...) pattern."
            )
        if query.op == "sem_agg":
            raise NotImplementedError(
                "standalone sem_agg differential is only supported at a view boundary."
            )
        if query.op == "array_agg":
            raise NotImplementedError(
                "array_agg changed-row differential is not implemented; array_agg is only supported at a view boundary or inside process_window."
            )
        if query.op == "min":
            raise NotImplementedError(
                "min changed-row differential is only supported at a view boundary."
            )
        if query.op == "agg":
            raise NotImplementedError(
                "agg changed-row differential is not implemented; grouped agg is only supported at a view boundary."
            )

    def _differentiate_source_input(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Rewrite the source log leaf to the runtime-bound changed rows input."""

        if source_query is not None:
            return source_input if query == source_query else None
        if query.op != "log":
            return None
        return source_input

    def _differentiate_row_local_operator(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Differentiate row-local unary operator patterns."""

        if query.op not in ROW_LOCAL_UNARY_OPS:
            return None

        if len(query.inputs) != 1:
            raise ValueError(
                f"QueryExpr op {query.op!r} expects exactly one input; got {len(query.inputs)}."
            )

        return QueryExpr(
            op=query.op,
            inputs=(
                self._differentiate_to_rows(
                    query.inputs[0],
                    source_input=source_input,
                    current_view=current_view,
                    source_query=source_query,
                    instruction_rewriter=instruction_rewriter,
                ),
            ),
            params=query.params,
        )

    def _differentiate_append_binary_operator(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Differentiate append-preserving binary operator patterns."""

        if query.op not in CHANGED_FRAGMENT_BINARY_OPS:
            return None

        if len(query.inputs) != 2:
            raise ValueError(
                f"QueryExpr op {query.op!r} expects exactly two inputs; got {len(query.inputs)}."
            )

        return QueryExpr(
            op=query.op,
            inputs=tuple(
                self._differentiate_to_rows(
                    input_query,
                    source_input=source_input,
                    current_view=current_view,
                    source_query=source_query,
                    instruction_rewriter=instruction_rewriter,
                )
                for input_query in query.inputs
            ),
            params=query.params,
        )

    def _differentiate_sem_join(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Differentiate generic sem_join changed rows when state is available."""

        if query.op != "sem_join":
            return None
        how = str(query.params.get("how", "inner")).lower()
        if how != "inner":
            raise NotImplementedError(
                "Generic sem_join differential currently supports only how='inner'."
            )
        raise NotImplementedError(
            "Generic sem_join differential requires materialized old L/R state and is not implemented yet."
        )

    def _differentiate_min_view(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Differentiate global or deterministic grouped minimum state."""

        selected = query.op == "select" and len(query.inputs) == 1
        aggregate = query.inputs[0] if selected else query
        if aggregate.op != "min" or len(aggregate.inputs) != 1:
            return None

        source = aggregate.inputs[0]
        if source.op == "group_by":
            changed_input = self._differentiate_to_rows(
                source.inputs[0],
                source_input=source_input,
                current_view=current_view,
                source_query=source_query,
                instruction_rewriter=instruction_rewriter,
            )
            changed_grouped = QueryExpr(
                op="group_by",
                inputs=(changed_input,),
                params=source.params,
            )
            changed_aggregate = QueryExpr(
                op="min",
                inputs=(changed_grouped,),
                params=aggregate.params,
            )
            keys = tuple(str(key) for key in source.params["keys"])
            output_col = str(aggregate.params["output_col"])
            if self._grouped_agg_rule == "rule-join-map":
                joined = QueryExpr(
                    op="join",
                    inputs=(changed_aggregate, current_view),
                    params={"on": keys, "how": "outer"},
                )
                merged = QueryExpr(
                    op="assign",
                    inputs=(joined,),
                    params={
                        "assignments": {
                            output_col: least(
                                ColumnExpr(output_col, qualifier="left"),
                                ColumnExpr(output_col, qualifier="right"),
                            ).to_param()
                        }
                    },
                )
                differentiated: QueryExpr = QueryExpr(
                    op="select",
                    inputs=(merged,),
                    params={"columns": (*keys, output_col)},
                )
            else:
                combined = QueryExpr(
                    op="concat",
                    inputs=(changed_aggregate, current_view),
                )
                regrouped = QueryExpr(
                    op="group_by",
                    inputs=(combined,),
                    params=source.params,
                )
                differentiated = QueryExpr(
                    op="min",
                    inputs=(regrouped,),
                    params={"columns": (output_col,), "output_col": output_col},
                )
        else:
            changed_input = self._differentiate_to_rows(
                source,
                source_input=source_input,
                current_view=current_view,
                source_query=source_query,
                instruction_rewriter=instruction_rewriter,
            )
            changed_aggregate = QueryExpr(
                op="min",
                inputs=(changed_input,),
                params=aggregate.params,
            )
            output_col = str(aggregate.params["output_col"])
            differentiated = QueryExpr(
                op="min",
                inputs=(
                    QueryExpr(
                        op="concat",
                        inputs=(changed_aggregate, current_view),
                    ),
                ),
                params={"columns": (output_col,), "output_col": output_col},
            )

        if not selected:
            return differentiated
        return QueryExpr(
            op="select",
            inputs=(differentiated,),
            params=query.params,
        )

    def _differentiate_array_agg_view(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Differentiate view-boundary array_agg via array aggregate-state concat."""

        if query.op != "array_agg":
            return None
        if len(query.inputs) != 1:
            raise ValueError("array_agg expects exactly one input")
        if query.inputs[0].op == "group_by":
            groupby = query.inputs[0]
            changed_group_input = self._differentiate_to_rows(
                groupby.inputs[0],
                source_input=source_input,
                current_view=current_view,
                source_query=source_query,
                instruction_rewriter=instruction_rewriter,
            )
            changed_groupby = QueryExpr(
                op="group_by",
                inputs=(changed_group_input,),
                params=groupby.params,
            )
            changed_aggregate = QueryExpr(
                op="array_agg",
                inputs=(changed_groupby,),
                params=query.params,
            )
            combined = QueryExpr(
                op="concat",
                inputs=(current_view, changed_aggregate),
            )
            regrouped = QueryExpr(
                op="group_by",
                inputs=(combined,),
                params=groupby.params,
            )
            return QueryExpr(
                op="flatten",
                inputs=(
                    QueryExpr(
                        op="agg",
                        inputs=(regrouped,),
                        params={
                            "aggregates": (
                                CollectListAggregateSpec(
                                    column=str(query.params["output_col"]),
                                    output_col=str(query.params["output_col"]),
                                ),
                            ),
                        },
                    ),
                ),
                params={"column": str(query.params["output_col"]), "output_col": None},
            )
        if query.inputs[0].op == "sem_groupby":
            raise NotImplementedError("sem_groupby(...).array_agg(...) is not supported.")

        changed_input = self._differentiate_to_rows(
            query.inputs[0],
            source_input=source_input,
            current_view=current_view,
            source_query=source_query,
            instruction_rewriter=instruction_rewriter,
        )
        changed_aggregate = QueryExpr(
            op="array_agg",
            inputs=(changed_input,),
            params=query.params,
        )
        return QueryExpr(
            op="array_cat",
            inputs=(current_view, changed_aggregate),
            params={"column": str(query.params["output_col"])},
        )

    def _differentiate_grouped_agg_view(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Differentiate grouped agg(A*) view-boundary patterns."""

        final_columns, aggregate = self._select_over_grouped_agg(query)
        if aggregate is None:
            return None
        groupby = aggregate.inputs[0]
        changed_group_input = self._differentiate_to_rows(
            groupby.inputs[0],
            source_input=source_input,
            current_view=current_view,
            source_query=source_query,
            instruction_rewriter=instruction_rewriter,
        )
        if self._grouped_agg_rule == "rule-join-map":
            return self._build_grouped_agg_join_map_candidate(
                aggregate=aggregate,
                groupby=groupby,
                changed_group_input=changed_group_input,
                current_view=current_view,
                final_columns=final_columns,
                instruction_rewriter=instruction_rewriter,
            )
        return self._build_grouped_agg_re_group_candidate(
            aggregate=aggregate,
            groupby=groupby,
            changed_group_input=changed_group_input,
            current_view=current_view,
            final_columns=final_columns,
            instruction_rewriter=instruction_rewriter,
        )

    def _build_grouped_agg_re_group_candidate(
        self,
        *,
        aggregate: QueryExpr,
        groupby: QueryExpr,
        changed_group_input: QueryExpr,
        current_view: QueryExpr,
        final_columns: Sequence[str],
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Build rule-re-group for grouped agg(A*) patterns."""

        changed_groupby = QueryExpr(
            op=groupby.op,
            inputs=(changed_group_input,),
            params=groupby.params,
        )
        changed_aggregate = QueryExpr(
            op="agg",
            inputs=(changed_groupby,),
            params=aggregate.params,
        )
        combined = QueryExpr(op="concat", inputs=(current_view, changed_aggregate))
        regrouped = QueryExpr(
            op=groupby.op,
            inputs=(combined,),
            params=groupby.params,
        )
        merged = self._build_aggregate_remerge(
            grouped=regrouped,
            aggregate=aggregate,
            instruction_rewriter=instruction_rewriter,
        )
        return self._select_after_array_remerge_flatten(merged, aggregate, final_columns)

    def _build_aggregate_remerge(
        self,
        *,
        grouped: QueryExpr,
        aggregate: QueryExpr,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Build grouped aggregate remerge using ordinary aggregate specs."""

        return QueryExpr(
            op="agg",
            inputs=(grouped,),
            params={
                "aggregates": self._remerge_aggregate_specs(
                    aggregate,
                    instruction_rewriter=instruction_rewriter,
                )
            },
        )

    def _remerge_aggregate_specs(
        self,
        aggregate: QueryExpr,
        *,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> tuple[object, ...]:
        """Return aggregate specs for remerging grouped aggregate rows."""

        specs: list[object] = []
        for spec in aggregate.params.get("aggregates", ()):
            if isinstance(spec, SemanticAggregateSpec):
                state_cols = aggregate_output_names(spec)
                specs.append(
                    SemanticAggregateSpec(
                        input_cols=state_cols,
                        output_cols=spec.output_cols,
                        instruction=instruction_rewriter.state_reaggregation(
                            spec.instruction,
                            state_cols=state_cols,
                            raw_input_cols=spec.input_cols or (),
                        ),
                    )
                )
                continue
            if isinstance(spec, (ArrayAggregateSpec, CollectListAggregateSpec)):
                output_col = spec.output_col
                specs.append(CollectListAggregateSpec(column=output_col, output_col=output_col))
                continue
            if isinstance(spec, MinAggregateSpec):
                specs.append(
                    MinAggregateSpec(
                        columns=(spec.output_col,),
                        output_col=spec.output_col,
                    )
                )
                continue
            raise TypeError(f"Unsupported aggregate spec: {type(spec).__name__}")
        return tuple(specs)

    def _select_after_array_remerge_flatten(
        self,
        query: QueryExpr,
        aggregate: QueryExpr,
        final_columns: Sequence[str],
    ) -> QueryExpr:
        """Flatten array remerge columns before final projection."""

        flattened = query
        for column in self._array_remerge_output_columns(aggregate):
            flattened = QueryExpr(
                op="flatten",
                inputs=(flattened,),
                params={"column": column, "output_col": None},
            )
        return QueryExpr(
            op="select",
            inputs=(flattened,),
            params={"columns": tuple(final_columns)},
        )

    def _array_remerge_output_columns(self, aggregate: QueryExpr) -> tuple[str, ...]:
        """Return output columns that need collect_list(...).flatten() remerge."""

        outputs: list[str] = []
        for spec in aggregate.params.get("aggregates", ()):
            if isinstance(spec, (ArrayAggregateSpec, CollectListAggregateSpec)):
                outputs.append(spec.output_col)
        return tuple(outputs)

    def _build_grouped_agg_join_map_candidate(
        self,
        *,
        aggregate: QueryExpr,
        groupby: QueryExpr,
        changed_group_input: QueryExpr,
        current_view: QueryExpr,
        final_columns: Sequence[str],
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Build rule-join-map for grouped agg(A*) patterns without array specs."""

        changed_groupby = QueryExpr(
            op=groupby.op,
            inputs=(changed_group_input,),
            params=groupby.params,
        )
        changed_aggregate = QueryExpr(
            op="agg",
            inputs=(changed_groupby,),
            params=aggregate.params,
        )
        if groupby.op == "sem_groupby":
            return self._build_semantic_join_map(
                aggregate=aggregate,
                groupby=groupby,
                changed_aggregate=changed_aggregate,
                current_view=current_view,
                final_columns=final_columns,
                instruction_rewriter=instruction_rewriter,
            )
        if groupby.op == "group_by":
            joined: QueryExpr = QueryExpr(
                op="join",
                inputs=(changed_aggregate, current_view),
                params={"on": tuple(str(key) for key in groupby.params["keys"]), "how": "outer"},
            )
        else:
            raise TypeError(f"Unsupported grouped aggregate input: {groupby.op}")
        preserved_columns = self._grouped_preserved_columns(groupby)
        mapped = joined
        array_outputs: list[str] = []
        min_outputs: list[str] = []
        for spec in aggregate.params.get("aggregates", ()):
            if isinstance(spec, (ArrayAggregateSpec, CollectListAggregateSpec)):
                array_outputs.append(spec.output_col)
                continue
            if isinstance(spec, MinAggregateSpec):
                min_outputs.append(spec.output_col)
                continue
            if not isinstance(spec, SemanticAggregateSpec):
                continue
            output_cols = self._aggregate_output_cols_for_columns(
                spec.output_cols,
                tuple(column.name for column in spec.output_cols),
                preserved_columns=preserved_columns,
            )
            mapped = QueryExpr(
                op="sem_map",
                inputs=(mapped,),
                params={
                    "input_cols": None,
                    "output_cols": output_cols,
                    "instruction": instruction_rewriter.agg_to_map(
                        spec.instruction,
                        input_cols=tuple(str(column) for column in (spec.input_cols or ())),
                        output_cols=output_cols,
                    ),
                },
            )
        if array_outputs or min_outputs:
            assignments = {
                output: ColumnExpr(name=output, qualifier="right")
                .array_cat(ColumnExpr(name=output, qualifier="left"))
                .to_param()
                for output in array_outputs
            }
            assignments.update(
                {
                    output: least(
                        ColumnExpr(name=output, qualifier="left"),
                        ColumnExpr(name=output, qualifier="right"),
                    ).to_param()
                    for output in min_outputs
                }
            )
            mapped = QueryExpr(
                op="assign",
                inputs=(mapped,),
                params={"assignments": assignments},
            )
        return QueryExpr(
            op="select",
            inputs=(mapped,),
            params={"columns": tuple(final_columns)},
        )

    def _build_semantic_join_map(
        self,
        *,
        aggregate: QueryExpr,
        groupby: QueryExpr,
        changed_aggregate: QueryExpr,
        current_view: QueryExpr,
        final_columns: Sequence[str],
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Join changed groups to current targets and merge each target once."""

        reserved = {
            JOIN_MAP_LEFT_ID_COLUMN,
            JOIN_MAP_RIGHT_ID_COLUMN,
        }.intersection(final_columns)
        if reserved:
            raise ValueError(
                "join-map internal ID columns conflict with view columns: "
                f"{sorted(reserved)}"
            )
        partition_keys = tuple(
            str(key) for key in groupby.params.get("partition_by", ())
        )
        join_params: dict[str, object] = {
            "instruction": instruction_rewriter.groupby_to_join(
                str(groupby.params["instruction"]),
                input_cols=tuple(
                    str(column) for column in groupby.params["input_cols"]
                ),
            ),
            "how": "outer",
            "id_columns": (
                JOIN_MAP_LEFT_ID_COLUMN,
                JOIN_MAP_RIGHT_ID_COLUMN,
            ),
        }
        membership = groupby.params.get("membership")
        if membership == "exclusive":
            join_params["k"] = 1
        elif membership == "overlapping":
            raise NotImplementedError(
                "overlapping sem_groupby membership is not implemented"
            )
        elif membership is not None:
            raise ValueError(f"Unsupported sem_groupby membership: {membership!r}")
        if partition_keys:
            join_params["on"] = partition_keys
        joined = QueryExpr(
            op="sem_join",
            inputs=(changed_aggregate, current_view),
            params=join_params,
        )
        joined_ref = QueryExpr(
            op="materialized_view",
            params={
                "name": JOIN_MAP_BINDING_NAME,
                "columns": output_columns(joined),
            },
        )

        matched = self._filter_join_map_rows(
            joined_ref,
            left_present=True,
            right_present=True,
        )
        left_only = self._filter_join_map_rows(
            joined_ref,
            left_present=True,
            right_present=False,
        )
        right_only = self._filter_join_map_rows(
            joined_ref,
            left_present=False,
            right_present=True,
        )

        matched_changed = self._project_join_map_side(
            matched,
            side="left",
            final_columns=final_columns,
            include_target_id=True,
        )
        matched_current = QueryExpr(
            op="drop_duplicates",
            inputs=(
                self._project_join_map_side(
                    matched,
                    side="right",
                    final_columns=final_columns,
                    include_target_id=True,
                ),
            ),
            params={"subset": (JOIN_MAP_RIGHT_ID_COLUMN,)},
        )
        state_rows = QueryExpr(
            op="concat",
            inputs=(matched_changed, matched_current),
        )
        grouped_states = QueryExpr(
            op="group_by",
            inputs=(state_rows,),
            params={
                "keys": (
                    JOIN_MAP_RIGHT_ID_COLUMN,
                    *partition_keys,
                )
            },
        )
        updates = self._reaggregate_join_map_states(
            grouped_states,
            aggregate=aggregate,
            final_columns=final_columns,
            instruction_rewriter=instruction_rewriter,
        )
        new_groups = self._project_join_map_side(
            left_only,
            side="left",
            final_columns=final_columns,
        )
        untouched = self._project_join_map_side(
            right_only,
            side="right",
            final_columns=final_columns,
        )
        kept_and_updated = QueryExpr(
            op="union_by_name",
            inputs=(untouched, updates),
            params={"allow_missing_columns": False},
        )
        body = QueryExpr(
            op="union_by_name",
            inputs=(kept_and_updated, new_groups),
            params={"allow_missing_columns": False},
        )
        return QueryExpr(
            op="let",
            inputs=(joined, body),
            params={"name": JOIN_MAP_BINDING_NAME},
        )

    def _filter_join_map_rows(
        self,
        joined: QueryExpr,
        *,
        left_present: bool,
        right_present: bool,
    ) -> QueryExpr:
        """Select one matched or unmatched side of an outer semantic join."""

        left_id = ColumnExpr(JOIN_MAP_LEFT_ID_COLUMN)
        right_id = ColumnExpr(JOIN_MAP_RIGHT_ID_COLUMN)
        predicate = (
            left_id.is_not_null() if left_present else left_id.is_null()
        ) & (
            right_id.is_not_null() if right_present else right_id.is_null()
        )
        return QueryExpr(
            op="filter",
            inputs=(joined,),
            params={"predicate": predicate.to_param()},
        )

    def _project_join_map_side(
        self,
        source: QueryExpr,
        *,
        side: str,
        final_columns: Sequence[str],
        include_target_id: bool = False,
    ) -> QueryExpr:
        """Project one outer-join side back to the materialized view schema."""

        assignments = {
            str(column): ColumnExpr(str(column), qualifier=side).to_param()
            for column in final_columns
        }
        columns: tuple[str, ...] = tuple(str(column) for column in final_columns)
        if include_target_id:
            assignments[JOIN_MAP_RIGHT_ID_COLUMN] = ColumnExpr(
                JOIN_MAP_RIGHT_ID_COLUMN
            ).to_param()
            columns = (JOIN_MAP_RIGHT_ID_COLUMN, *columns)
        assigned = QueryExpr(
            op="assign",
            inputs=(source,),
            params={"assignments": assignments},
        )
        return QueryExpr(
            op="select",
            inputs=(assigned,),
            params={"columns": columns},
        )

    def _reaggregate_join_map_states(
        self,
        grouped_states: QueryExpr,
        *,
        aggregate: QueryExpr,
        final_columns: Sequence[str],
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Merge current and changed aggregate state once for each selected target."""

        if aggregate.op == "agg":
            merged = self._build_aggregate_remerge(
                grouped=grouped_states,
                aggregate=aggregate,
                instruction_rewriter=instruction_rewriter,
            )
            return self._select_after_array_remerge_flatten(
                merged,
                aggregate,
                final_columns,
            )
        if aggregate.op != "sem_agg":
            raise TypeError(f"Unsupported join-map aggregate: {aggregate.op}")

        reaggregate_params = dict(aggregate.params)
        state_cols = self._output_column_names(aggregate.params.get("output_cols"))
        reaggregate_params["input_cols"] = state_cols
        reaggregate_params["instruction"] = instruction_rewriter.state_reaggregation(
            str(aggregate.params["instruction"]),
            state_cols=state_cols,
            raw_input_cols=tuple(
                str(column) for column in (aggregate.params.get("input_cols") or ())
            ),
        )
        reaggregated = QueryExpr(
            op="sem_agg",
            inputs=(grouped_states,),
            params=reaggregate_params,
        )
        return QueryExpr(
            op="select",
            inputs=(reaggregated,),
            params={"columns": tuple(final_columns)},
        )

    def _differentiate_sem_groupby_agg_view(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Differentiate sem_groupby(...).sem_agg(...) into a full V' query."""

        final_columns, aggregate = self._select_over_grouped_aggregate(query)
        if aggregate is None:
            return None

        groupby = aggregate.inputs[0]
        changed_group_input = self._differentiate_to_rows(
            groupby.inputs[0],
            source_input=source_input,
            current_view=current_view,
            source_query=source_query,
            instruction_rewriter=instruction_rewriter,
        )

        selected_rule = self._grouped_agg_rule
        if selected_rule == "rule-re-group":
            return self._build_grouped_sem_agg_re_group_candidate(
                aggregate=aggregate,
                groupby=groupby,
                changed_group_input=changed_group_input,
                current_view=current_view,
                final_columns=final_columns,
                instruction_rewriter=instruction_rewriter,
            )
        if selected_rule == "rule-all-group-optimized":
            return self._build_sem_groupby_agg_changed_aware_candidate(
                aggregate=aggregate,
                groupby=groupby,
                changed_group_input=changed_group_input,
                current_view=current_view,
                final_columns=final_columns,
            )
        if selected_rule == "rule-join-map":
            return self._build_sem_groupby_agg_join_map_candidate(
                query,
                changed_group_input=changed_group_input,
                current_view=current_view,
                instruction_rewriter=instruction_rewriter,
            )

        return self._build_sem_groupby_agg_compressed_candidate(
            aggregate=aggregate,
            groupby=groupby,
            changed_group_input=changed_group_input,
            current_view=current_view,
            final_columns=final_columns,
        )

    def _build_sem_groupby_agg_compressed_candidate(
        self,
        *,
        aggregate: QueryExpr,
        groupby: QueryExpr,
        changed_group_input: QueryExpr,
        current_view: QueryExpr,
        final_columns: Sequence[str],
    ) -> QueryExpr:
        """Build the default compressed-state grouped aggregate candidate."""

        compressed_input = QueryExpr(
            op="union_by_name",
            inputs=(changed_group_input, current_view),
            params={"allow_missing_columns": True},
        )
        regrouped = QueryExpr(
            op="sem_groupby",
            inputs=(compressed_input,),
            params=groupby.params,
        )
        reaggregated = QueryExpr(
            op="sem_agg",
            inputs=(regrouped,),
            params=aggregate.params,
        )
        return QueryExpr(
            op="select",
            inputs=(reaggregated,),
            params={"columns": tuple(final_columns)},
        )

    def _build_grouped_sem_agg_re_group_candidate(
        self,
        *,
        aggregate: QueryExpr,
        groupby: QueryExpr,
        changed_group_input: QueryExpr,
        current_view: QueryExpr,
        final_columns: Sequence[str],
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Build rule-re-group for grouped semantic aggregates."""

        changed_groupby = QueryExpr(
            op=groupby.op,
            inputs=(changed_group_input,),
            params=groupby.params,
        )
        changed_aggregate = QueryExpr(
            op="sem_agg",
            inputs=(changed_groupby,),
            params=aggregate.params,
        )
        combined = QueryExpr(
            op="union_by_name",
            inputs=(changed_aggregate, current_view),
            params={"allow_missing_columns": True},
        )
        regrouped = QueryExpr(
            op=groupby.op,
            inputs=(combined,),
            params=groupby.params,
        )
        reaggregate_params = dict(aggregate.params)
        state_cols = self._output_column_names(aggregate.params.get("output_cols"))
        reaggregate_params["input_cols"] = state_cols
        reaggregate_params["instruction"] = instruction_rewriter.state_reaggregation(
            str(aggregate.params["instruction"]),
            state_cols=state_cols,
            raw_input_cols=tuple(
                str(column) for column in (aggregate.params.get("input_cols") or ())
            ),
        )
        reaggregated = QueryExpr(
            op="sem_agg",
            inputs=(regrouped,),
            params=reaggregate_params,
        )
        return QueryExpr(
            op="select",
            inputs=(reaggregated,),
            params={"columns": tuple(final_columns)},
        )

    def _build_sem_groupby_agg_changed_aware_candidate(
        self,
        *,
        aggregate: QueryExpr,
        groupby: QueryExpr,
        changed_group_input: QueryExpr,
        current_view: QueryExpr,
        final_columns: Sequence[str],
    ) -> QueryExpr:
        """Build the changed-aware grouped aggregate optimization candidate."""

        if aggregate.params.get("input_cols") is None:
            raise NotImplementedError(
                "changed-aware grouped aggregate rule requires explicit sem_agg "
                "input_cols so internal marker columns do not enter semantic prompts."
            )

        changed = QueryExpr(
            op="assign",
            inputs=(changed_group_input,),
            params={"assignments": {CHANGED_MARKER_COLUMN: True}},
        )
        current = QueryExpr(
            op="assign",
            inputs=(current_view,),
            params={"assignments": {CHANGED_MARKER_COLUMN: False}},
        )
        mixed = QueryExpr(
            op="union_by_name",
            inputs=(changed, current),
            params={"allow_missing_columns": True},
        )
        grouped = QueryExpr(
            op=groupby.op,
            inputs=(mixed,),
            params=groupby.params,
        )
        changed_group_rows = QueryExpr(
            op="filter",
            inputs=(grouped,),
            params={
                "predicate": ColumnExpr(CHANGED_MARKER_COLUMN).to_param(),
            },
        )
        touched_columns = (
            (
                *tuple(str(key) for key in groupby.params.get("partition_by", ())),
                GROUP_ID_COLUMN,
            )
            if groupby.op == "sem_groupby"
            else tuple(str(key) for key in groupby.params["keys"])
        )
        touched_groups = QueryExpr(
            op="drop_duplicates",
            inputs=(
                QueryExpr(
                    op="select",
                    inputs=(changed_group_rows,),
                    params={"columns": touched_columns},
                ),
            ),
        )
        join_on = touched_columns
        updated_input = QueryExpr(
            op="join",
            inputs=(grouped, touched_groups),
            params={"on": join_on, "how": "inner"},
        )
        updated = QueryExpr(
            op="sem_agg",
            inputs=(updated_input,),
            params=aggregate.params,
        )
        updates = QueryExpr(
            op="select",
            inputs=(updated,),
            params={"columns": tuple(final_columns)},
        )
        kept_input = QueryExpr(
            op="join",
            inputs=(grouped, touched_groups),
            params={"on": join_on, "how": "left_anti"},
        )
        kept = QueryExpr(
            op="select",
            inputs=(kept_input,),
            params={"columns": tuple(final_columns)},
        )
        combined = QueryExpr(
            op="union",
            inputs=(kept, updates),
            params={},
        )
        return combined

    def _build_sem_groupby_agg_join_map_candidate(
        self,
        query: QueryExpr,
        *,
        changed_group_input: QueryExpr,
        current_view: QueryExpr,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Build the sem_join -> sem_map grouped aggregate candidate."""

        final_columns, aggregate = self._select_over_grouped_aggregate(query)
        if aggregate is None:
            return None

        groupby = aggregate.inputs[0]
        changed_groupby = QueryExpr(
            op="sem_groupby",
            inputs=(changed_group_input,),
            params=groupby.params,
        )
        changed_aggregate = QueryExpr(
            op="sem_agg",
            inputs=(changed_groupby,),
            params=aggregate.params,
        )
        if groupby.op == "sem_groupby":
            return self._build_semantic_join_map(
                aggregate=aggregate,
                groupby=groupby,
                changed_aggregate=changed_aggregate,
                current_view=current_view,
                final_columns=final_columns,
                instruction_rewriter=instruction_rewriter,
            )
        preserved_columns = self._grouped_preserved_columns(groupby)
        if groupby.op == "group_by":
            joined = QueryExpr(
                op="join",
                inputs=(changed_aggregate, current_view),
                params={"on": tuple(str(key) for key in groupby.params["keys"]), "how": "outer"},
            )
        else:
            joined = QueryExpr(
                op="sem_join",
                inputs=(changed_aggregate, current_view),
                params={
                    "instruction": instruction_rewriter.groupby_to_join(
                        str(groupby.params["instruction"]),
                        input_cols=tuple(str(column) for column in groupby.params["input_cols"]),
                    ),
                    "how": "outer",
                },
            )
        output_cols = self._aggregate_output_cols_for_columns(
            aggregate.params.get("output_cols"),
            final_columns,
            preserved_columns=preserved_columns,
        )
        mapped = QueryExpr(
            op="sem_map",
            inputs=(joined,),
            params={
                "input_cols": None,
                "output_cols": output_cols,
                "instruction": instruction_rewriter.agg_to_map(
                    str(aggregate.params["instruction"]),
                    input_cols=tuple(
                        str(column)
                        for column in (aggregate.params.get("input_cols") or ())
                    ),
                    output_cols=output_cols,
                ),
            },
        )
        return QueryExpr(
            op="select",
            inputs=(mapped,),
            params={"columns": tuple(final_columns)},
        )

    def _differentiate_sem_agg_view(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        source_query: QueryExpr | None,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Differentiate standalone sem_agg using compressed aggregate state."""

        final_columns, aggregate = self._select_over_standalone_aggregate(query)
        if aggregate is None:
            return None
        if aggregate.inputs[0].op == "sem_groupby":
            return None

        input_cols = aggregate.params.get("input_cols")
        if input_cols is None:
            raise NotImplementedError(
                "standalone sem_agg compressed-state differential requires explicit input_cols."
            )
        required_cols = tuple(str(column) for column in input_cols)
        if not required_cols:
            raise NotImplementedError(
                "standalone sem_agg compressed-state differential requires non-empty input_cols."
            )

        changed_fragment = self._differentiate_to_rows(
            aggregate.inputs[0],
            source_input=source_input,
            current_view=current_view,
            source_query=source_query,
            instruction_rewriter=instruction_rewriter,
        )
        self._require_columns(
            required_cols,
            available=self._output_column_names_for_query(current_view),
            source="current view",
        )
        self._require_columns(
            required_cols,
            available=self._output_column_names_for_query(aggregate.inputs[0]),
            source="changed sem_agg input fragment",
        )

        current_state = QueryExpr(
            op="select",
            inputs=(current_view,),
            params={"columns": required_cols},
        )
        changed_state = QueryExpr(
            op="select",
            inputs=(changed_fragment,),
            params={"columns": required_cols},
        )
        compressed_input = QueryExpr(
            op="union",
            inputs=(current_state, changed_state),
        )
        differentiated = QueryExpr(
            op="sem_agg",
            inputs=(compressed_input,),
            params=aggregate.params,
        )
        if final_columns is None:
            return differentiated
        return QueryExpr(
            op="select",
            inputs=(differentiated,),
            params={"columns": final_columns},
        )

    def _select_over_standalone_aggregate(
        self,
        query: QueryExpr,
    ) -> tuple[tuple[str, ...] | None, QueryExpr | None]:
        """Return final select columns and standalone aggregate if matched."""

        if query.op == "select" and len(query.inputs) == 1:
            final_columns = tuple(str(column) for column in query.params["columns"])
            aggregate = query.inputs[0]
        else:
            final_columns = None
            aggregate = query

        if aggregate.op != "sem_agg" or len(aggregate.inputs) != 1:
            return final_columns, None
        if aggregate.inputs[0].op == "sem_groupby":
            return final_columns, None
        return final_columns, aggregate

    def _select_over_grouped_aggregate(
        self,
        query: QueryExpr,
    ) -> tuple[tuple[str, ...], QueryExpr | None]:
        """Return final columns and grouped aggregate if the pattern matches."""

        if query.op == "select" and len(query.inputs) == 1:
            final_columns = tuple(str(column) for column in query.params["columns"])
            aggregate = query.inputs[0]
        else:
            aggregate = query
            final_columns = output_columns(aggregate)

        if aggregate.op != "sem_agg" or len(aggregate.inputs) != 1:
            return final_columns, None
        groupby = aggregate.inputs[0]
        if groupby.op not in {"sem_groupby", "group_by"} or len(groupby.inputs) != 1:
            return final_columns, None
        if not final_columns:
            raise ValueError("grouped sem_agg view requires output columns")
        return final_columns, aggregate

    def _select_over_grouped_agg(
        self,
        query: QueryExpr,
    ) -> tuple[tuple[str, ...], QueryExpr | None]:
        """Return final columns and grouped agg(A*) node if the pattern matches."""

        if query.op == "select" and len(query.inputs) == 1:
            final_columns = tuple(str(column) for column in query.params["columns"])
            aggregate = query.inputs[0]
        else:
            aggregate = query
            final_columns = output_columns(aggregate)

        if aggregate.op != "agg" or len(aggregate.inputs) != 1:
            return final_columns, None
        groupby = aggregate.inputs[0]
        if groupby.op not in {"group_by", "sem_groupby"} or len(groupby.inputs) != 1:
            return final_columns, None
        if not final_columns:
            raise ValueError("grouped agg(A*) view requires output columns")
        return final_columns, aggregate

    def _output_cols_for_columns(
        self,
        output_cols: object,
        columns: Sequence[str],
    ) -> tuple[ColumnSpec, ...]:
        """Return aggregate output column specs matching final view columns."""

        by_name: dict[str, ColumnSpec] = {}
        if output_cols is not None:
            for column in output_cols:  # type: ignore[union-attr]
                if isinstance(column, ColumnSpec):
                    by_name[column.name] = column

        return tuple(by_name.get(column, ColumnSpec(name=column)) for column in columns)

    def _aggregate_output_cols_for_columns(
        self,
        output_cols: object,
        columns: Sequence[str],
        *,
        preserved_columns: Sequence[str],
    ) -> tuple[ColumnSpec, ...]:
        """Return semantic output specs excluding deterministic preserved columns."""

        preserved = set(preserved_columns)
        return self._output_cols_for_columns(
            output_cols,
            tuple(column for column in columns if column not in preserved),
        )

    def _grouped_preserved_columns(self, groupby: QueryExpr) -> tuple[str, ...]:
        """Return deterministic columns preserved by a grouped aggregate."""

        if groupby.op == "group_by":
            return tuple(str(key) for key in groupby.params["keys"])
        if groupby.op == "sem_groupby":
            return tuple(str(key) for key in groupby.params.get("partition_by", ()))
        return ()

    def _output_column_names(self, output_cols: object) -> tuple[str, ...]:
        """Return output column names from ColumnSpec tuples."""

        if output_cols is None:
            return ()
        return tuple(
            column.name if isinstance(column, ColumnSpec) else str(column)
            for column in output_cols  # type: ignore[union-attr]
        )

    def _output_column_names_for_query(self, query: QueryExpr) -> tuple[str, ...]:
        """Infer output column names for schema-only differential checks."""

        return output_columns(query)

    def _require_columns(
        self,
        required: Sequence[str],
        *,
        available: Sequence[str],
        source: str,
    ) -> None:
        """Require columns needed by compressed-state aggregate maintenance."""

        missing = [column for column in required if column not in set(available)]
        if missing:
            raise NotImplementedError(
                "standalone sem_agg compressed-state differential requires "
                f"{source} to contain input_cols {tuple(required)!r}; missing {missing!r}."
            )
