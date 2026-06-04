"""Differential rules for hardcoded Q -> Q' rewrites."""

from __future__ import annotations

from collections.abc import Sequence
import re

from agent_memory.logical import ColumnSpec, QueryExpr

ROW_LOCAL_UNARY_OPS = {
    "select",
    "sem_filter",
    "sem_map",
    "sem_flat_map",
    "drop_duplicates",
}
CHANGED_FRAGMENT_BINARY_OPS = {"concat", "union"}
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

    def differentiate(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
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
                instruction_rewriter=rewriter,
            )

        return self._differentiate_to_rows(
            query,
            source_input=source_input,
            current_view=current_view,
            instruction_rewriter=rewriter,
        )

    def _differentiate_to_view(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Differentiate a view definition query into a full next-view query."""

        for rule in (
            self._differentiate_sem_groupby_agg_view,
            self._differentiate_sem_agg_view,
        ):
            differentiated = rule(
                query,
                source_input=source_input,
                current_view=current_view,
                instruction_rewriter=instruction_rewriter,
            )
            if differentiated is not None:
                return differentiated

        changed_rows = self._differentiate_to_rows(
            query,
            source_input=source_input,
            current_view=current_view,
            instruction_rewriter=instruction_rewriter,
        )
        return QueryExpr(op="union", inputs=(current_view, changed_rows))

    def _differentiate_to_rows(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr:
        """Differentiate one query subtree into changed output rows."""

        self._raise_if_rows_unsupported(query)
        for rule in (
            self._differentiate_source_input,
            self._differentiate_row_local_operator,
            self._differentiate_append_binary_operator,
            self._differentiate_sem_join,
        ):
            changed_rows = rule(
                query,
                source_input=source_input,
                current_view=current_view,
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

    def _differentiate_source_input(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
        instruction_rewriter: DifferentialInstructionRewriter,
    ) -> QueryExpr | None:
        """Rewrite the source log leaf to the runtime-bound changed rows input."""

        if query.op != "log":
            return None
        return source_input

    def _differentiate_row_local_operator(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
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

    def _differentiate_sem_groupby_agg_view(
        self,
        query: QueryExpr,
        *,
        source_input: QueryExpr,
        current_view: QueryExpr,
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
            instruction_rewriter=instruction_rewriter,
        )
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
        output_cols = self._output_cols_for_columns(
            aggregate.params.get("output_cols"),
            final_columns,
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
            final_columns = self._output_column_names(aggregate.params.get("output_cols"))

        if aggregate.op != "sem_agg" or len(aggregate.inputs) != 1:
            return final_columns, None
        groupby = aggregate.inputs[0]
        if groupby.op != "sem_groupby" or len(groupby.inputs) != 1:
            return final_columns, None
        if not final_columns:
            raise ValueError("sem_groupby(...).sem_agg(...) view requires output columns")
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

        if query.op == "select":
            return tuple(str(column) for column in query.params["columns"])
        if query.op == "log":
            columns = query.params.get("columns", ())
            return tuple(column.name for column in columns if isinstance(column, ColumnSpec))
        if query.op == "materialized_view":
            return self._output_column_names(query.params.get("columns"))
        if query.op == "sem_agg":
            output_cols = query.params.get("output_cols")
            if output_cols is not None:
                return self._output_column_names(output_cols)
            input_cols = query.params.get("input_cols")
            if input_cols is not None:
                return tuple(str(column) for column in input_cols)
            raise NotImplementedError(
                "Cannot infer sem_agg output columns without input_cols or output_cols."
            )
        if query.op in {"sem_map", "sem_flat_map"}:
            columns = list(self._output_column_names_for_query(query.inputs[0]))
            for column in query.params.get("output_cols") or ():
                if isinstance(column, ColumnSpec) and column.name not in columns:
                    columns.append(column.name)
            return tuple(columns)
        if query.op in {"sem_filter", "sem_groupby", "sem_topk", "drop_duplicates"}:
            return self._output_column_names_for_query(query.inputs[0])
        if query.op in {"union", "concat", "subtract", "sem_join"}:
            columns: list[str] = []
            for input_query in query.inputs:
                for column in self._output_column_names_for_query(input_query):
                    if column not in columns:
                        columns.append(column)
            return tuple(columns)
        raise NotImplementedError(
            f"Cannot infer output columns for QueryExpr op {query.op!r}."
        )

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
