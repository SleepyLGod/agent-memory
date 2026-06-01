"""LOTUS-backed semantic aggregation lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import (
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.adapters.lotus.sem_groupby import GROUP_ID_COLUMN
from agent_memory.adapters.lotus.structured import StructuredLMExecutor
from agent_memory.logical import ColumnSpec, QueryExpr


def execute_sem_agg(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Execute whole-relation or grouped semantic aggregation."""

    context.configure()
    source = execute(query.inputs[0], inputs)
    input_cols = aggregate_input_columns(source, query.params.get("input_cols"))
    output_cols = aggregate_output_columns(query, input_cols)
    if len(output_cols) == 1:
        return execute_native_sem_agg(
            query,
            source,
            input_cols,
            output_cols[0],
            context.config,
        )
    return execute_structured_sem_agg(
        query,
        source,
        input_cols,
        output_cols,
        context.config,
    )


def execute_native_sem_agg(
    query: QueryExpr,
    source: pd.DataFrame,
    input_cols: Sequence[str],
    output_col: ColumnSpec,
    config: LotusExecutionConfig | None = None,
) -> pd.DataFrame:
    """Execute single-output aggregation through LOTUS sem_agg."""

    if source.empty:
        return pd.DataFrame(columns=[output_col.name])

    outputs = [
        execute_native_sem_agg_group(query, group, input_cols, config)
        for group in aggregate_groups(source)
    ]
    return pd.DataFrame({output_col.name: outputs})


def execute_native_sem_agg_group(
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    config: LotusExecutionConfig | None = None,
) -> str:
    """Aggregate one group into one native LOTUS string answer."""

    import lotus
    from lotus.sem_ops.sem_agg import sem_agg

    config = config or LotusExecutionConfig()
    docs = aggregate_group_text(group, input_cols)
    instruction = lotus.nl_expression.nle2str(
        str(query.params["instruction"]),
        list(input_cols),
    )
    output = sem_agg(
        docs,
        lotus.settings.lm,
        instruction,
        [0] * len(docs),
        safe_mode=config.sem_agg_safe_mode,
        progress_bar_desc=config.sem_agg_progress_bar_desc,
    )
    if not output.outputs:
        return ""
    return str(output.outputs[0])


def execute_structured_sem_agg(
    query: QueryExpr,
    source: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig | None = None,
) -> pd.DataFrame:
    """Execute multi-output aggregation with structured JSON objects."""

    if source.empty:
        return pd.DataFrame(columns=[column.name for column in output_cols])

    config = config or LotusExecutionConfig()
    parsed_outputs = [
        execute_structured_sem_agg_group(
            query,
            group,
            input_cols,
            output_cols,
            config,
        )
        for group in aggregate_groups(source)
    ]
    return apply_structured_aggregate_outputs(parsed_outputs, output_cols)


def execute_structured_sem_agg_group(
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
    *,
    context_kind: str = "source rows",
) -> Mapping[str, str]:
    """Aggregate one group with the configured structured strategy."""

    strategy = structured_sem_agg_strategy(config)
    if strategy == "single_batch":
        return execute_structured_sem_agg_leaf(
            query,
            group,
            input_cols,
            output_cols,
            config,
            context_kind=context_kind,
        )

    if strategy == "lotus_hierarchical":
        return execute_lotus_hierarchical_structured_sem_agg_group(
            query,
            group,
            input_cols,
            output_cols,
            config,
            context_kind=context_kind,
        )

    raise ValueError(
        "sem_agg_structured_strategy must be 'single_batch' or 'lotus_hierarchical'"
    )


def execute_lotus_hierarchical_structured_sem_agg_group(
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
    *,
    context_kind: str = "source rows",
) -> Mapping[str, str]:
    """Aggregate one group through LOTUS native hierarchy, then structure it."""

    summary_query = QueryExpr(
        op=query.op,
        inputs=query.inputs,
        params={
            **dict(query.params),
            "instruction": lotus_hierarchical_summary_instruction(
                query,
                output_cols,
            ),
        },
    )
    intermediate_summary = execute_native_sem_agg_group(
        summary_query,
        group,
        input_cols,
        config,
    )
    if not intermediate_summary.strip():
        raise ValueError("sem_agg lotus_hierarchical returned an empty intermediate summary")

    summary_frame = pd.DataFrame({"context": [intermediate_summary]})
    generation = execute_structured_sem_agg_context(
        query,
        summary_frame,
        output_cols,
        config,
        context_kind="LOTUS hierarchical aggregate",
    )
    if not generation.parsed_outputs:
        return {column.name: "" for column in output_cols}
    return generation.parsed_outputs[0]


def execute_structured_sem_agg_leaf(
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
    *,
    context_kind: str,
) -> Mapping[str, str]:
    """Aggregate one tree-fold leaf into one structured object."""

    context_frame = aggregate_context_frame(group, input_cols)
    generation = execute_structured_sem_agg_context(
        query,
        context_frame,
        output_cols,
        config,
        context_kind=context_kind,
    )
    if not generation.parsed_outputs:
        return {column.name: "" for column in output_cols}
    return generation.parsed_outputs[0]


def execute_structured_sem_agg_context(
    query: QueryExpr,
    context_frame: pd.DataFrame,
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig | None = None,
    *,
    context_kind: str = "grouped rows",
) -> Any:
    """Execute one structured aggregate context row."""

    config = config or LotusExecutionConfig()
    executor = StructuredLMExecutor(context_frame)
    return executor(
        input_cols=("context",),
        output_cols=tuple(output_cols),
        instruction=structured_sem_agg_instruction(query, context_kind=context_kind),
        shape="object",
        safe_mode=config.sem_agg_safe_mode,
        progress_bar_desc=config.sem_agg_progress_bar_desc,
        model_kwargs=structured_sem_agg_model_kwargs(config),
        operator="sem_agg",
    )


def structured_sem_agg_instruction(query: QueryExpr, *, context_kind: str) -> str:
    """Return structured aggregate instruction for raw or partial rows."""

    return (
        f"{query.params['instruction']}\n\n"
        f"Use the {context_kind} in {{context}} and produce one aggregate object."
    )


def lotus_hierarchical_summary_instruction(
    query: QueryExpr,
    output_cols: Sequence[ColumnSpec],
) -> str:
    """Return the intermediate instruction for LOTUS native hierarchy."""

    field_names = ", ".join(column.name for column in output_cols)
    return (
        f"{query.params['instruction']}\n\n"
        "Produce an intermediate aggregate that preserves all information needed "
        f"to fill these final output fields later: {field_names}."
    )


def structured_sem_agg_model_kwargs(
    config: LotusExecutionConfig | None = None,
) -> dict[str, Any]:
    """Return conservative generation kwargs for structured aggregate JSON."""

    import lotus

    config = config or LotusExecutionConfig()
    current = int(getattr(lotus.settings.lm, "max_tokens", 512))
    kwargs = {"max_tokens": max(current, 1024)}
    kwargs.update(dict(config.sem_agg_model_kwargs))
    return kwargs


def structured_sem_agg_strategy(config: LotusExecutionConfig) -> str:
    """Return validated structured aggregate strategy."""

    strategy = str(config.sem_agg_structured_strategy)
    if strategy not in {"single_batch", "lotus_hierarchical"}:
        raise ValueError(
            "sem_agg_structured_strategy must be 'single_batch', "
            "or 'lotus_hierarchical'"
        )
    return strategy


def aggregate_input_columns(
    source: pd.DataFrame,
    input_cols: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve input columns for semantic aggregation."""

    if input_cols is not None:
        columns = tuple(str(column) for column in input_cols)
    else:
        groupby_input_cols = tuple(source.attrs.get("agent_memory_groupby_input_cols", ()))
        excluded = set(groupby_input_cols).union({GROUP_ID_COLUMN})
        columns = tuple(
            str(column) for column in source.columns if column not in excluded
        )

    if not columns:
        raise ValueError("sem_agg requires at least one input column")
    missing = [column for column in columns if column not in source.columns]
    if missing:
        raise ValueError(f"sem_agg input columns not found in DataFrame: {missing}")
    return columns


def aggregate_output_columns(
    query: QueryExpr,
    input_cols: Sequence[str],
) -> tuple[ColumnSpec, ...]:
    """Resolve output columns for semantic aggregation."""

    output_cols = query.params.get("output_cols")
    if output_cols is None:
        return tuple(ColumnSpec(name=column) for column in input_cols)
    return tuple(output_cols)


def aggregate_context_frame(
    source: pd.DataFrame,
    input_cols: Sequence[str],
) -> pd.DataFrame:
    """Build one text context row per aggregate group."""

    if source.empty:
        return pd.DataFrame({"context": []})

    contexts = [
        "\n".join(aggregate_group_text(group, input_cols))
        for group in aggregate_groups(source)
    ]
    return pd.DataFrame({"context": contexts})


def aggregate_groups(source: pd.DataFrame) -> list[pd.DataFrame]:
    """Return one dataframe per aggregate group without internal group ids."""

    if GROUP_ID_COLUMN not in source.columns:
        return [source.reset_index(drop=True).copy()]
    return [
        group.drop(columns=[GROUP_ID_COLUMN]).reset_index(drop=True)
        for _group_id, group in source.groupby(GROUP_ID_COLUMN, sort=True)
    ]


def grouped_frames(source: pd.DataFrame) -> list[pd.DataFrame]:
    """Return grouped frames or one whole-relation frame."""

    return aggregate_groups(source)


def aggregate_group_text(
    group: pd.DataFrame,
    input_cols: Sequence[str],
) -> list[str]:
    """Format one aggregate group with LOTUS dataframe text formatting."""

    from lotus.templates import task_instructions

    return task_instructions.df2text(group, list(input_cols))


def apply_structured_aggregate_outputs(
    parsed_outputs: Sequence[Mapping[str, str]],
    output_cols: Sequence[ColumnSpec],
) -> pd.DataFrame:
    """Convert structured aggregate outputs into a DataFrame."""

    return pd.DataFrame(
        [
            {column.name: parsed[column.name] for column in output_cols}
            for parsed in parsed_outputs
        ],
        columns=[column.name for column in output_cols],
    )
