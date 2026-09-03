"""LOTUS-backed semantic aggregation lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import (
    LotusExecutionConfig,
    LotusExecutionContext,
)
from agent_memory.tracing.semantic import write_structured_generation_trace
from agent_memory.adapters.lotus.sem_groupby import GROUP_ID_COLUMN
from agent_memory.adapters.lotus.sem_agg_batch_prompting import (
    execute_batch_prompted_sem_agg,
)
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.structured import (
    StructuredLMRetryResult,
    escape_structured_formatter_placeholders,
    execute_structured_lm_retry_result,
    parse_structured_object_json,
    structured_parse_error,
    structured_scalar_values,
    write_structured_failure_artifacts,
)
from agent_memory.policy.logical import ColumnSpec, QueryExpr
from agent_memory.policy.schema import output_columns
from agent_memory.runtime.window import over_frames

JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}


@dataclass(frozen=True)
class _SemAggExecution:
    """Final output and structured retry metadata for one aggregate group."""

    raw_output: str
    retry_result: StructuredLMRetryResult | None = None


@dataclass
class _SemAggGroupState:
    """Mutable hierarchical state for one independent aggregate group."""

    group_index: int
    documents: list[str]
    tree_level: int = 0
    execution: _SemAggExecution | None = None


def execute_sem_agg(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> Any:
    """Execute whole-relation or grouped semantic aggregation."""

    context.configure()
    if query.inputs[0].op == "over":
        return execute_over_sem_agg(query, inputs, execute, context)

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


def execute_over_sem_agg(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: LotusExecutionContext,
) -> pd.DataFrame:
    """Execute row-preserving over-window semantic aggregation."""

    over_query = query.inputs[0]
    declared_output_cols = query.params.get("output_cols")
    if declared_output_cols is None:
        raise ValueError("over sem_agg requires explicit output_cols")
    output_cols = tuple(declared_output_cols)
    emit_source = execute(over_query.inputs[0], inputs)
    emit_columns = tuple(output_columns(over_query.inputs[0])) or tuple(
        emit_source.columns
    )
    missing_emit = [
        column for column in emit_columns if column not in emit_source.columns
    ]
    if missing_emit:
        raise ValueError(
            f"over sem_agg emit columns not found in DataFrame: {missing_emit}"
        )
    frame_source_query = over_query.params.get("frame_source")
    frame_source = (
        execute(frame_source_query, inputs)
        if isinstance(frame_source_query, QueryExpr)
        else emit_source
    )
    rows: list[dict[str, Any]] = []
    for frame in over_frames(emit_source, frame_source, over_query.params):
        row = frame.emit_row.loc[list(emit_columns)].to_dict()
        if frame.frame.empty:
            for column in output_cols:
                row[column.name] = None
            rows.append(row)
            continue

        input_cols = aggregate_input_columns(
            frame.frame, query.params.get("input_cols")
        )
        if len(output_cols) == 1:
            result = execute_native_sem_agg(
                query,
                frame.frame,
                input_cols,
                output_cols[0],
                context.config,
            )
        else:
            result = execute_structured_sem_agg(
                query,
                frame.frame,
                input_cols,
                output_cols,
                context.config,
            )
        values = result.iloc[0].to_dict() if not result.empty else {}
        for column in output_cols:
            row[column.name] = values.get(column.name)
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[*emit_columns, *(column.name for column in output_cols)],
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
        return pd.DataFrame(columns=list(output_columns(query)))

    config = config or LotusExecutionConfig()
    grouped = aggregate_groups_with_keys(source)
    raw_outputs = execute_native_sem_agg_groups(
        query,
        [group for _key_values, group in grouped],
        input_cols,
        config,
    )
    rows: list[dict[str, Any]] = []
    for group_index, ((key_values, group), raw_output) in enumerate(
        zip(grouped, raw_outputs, strict=True)
    ):
        row = dict(key_values)
        if output_col.name not in row:
            row[output_col.name] = raw_output
        rows.append(row)
        if _prompt_batching(config) is None:
            write_sem_agg_audit(
                config,
                query=query,
                group=group,
                input_cols=input_cols,
                output_cols=(output_col,),
                group_index=group_index,
                raw_output=raw_output,
                parsed_output={output_col.name: raw_output},
            )
    return pd.DataFrame(rows, columns=list(output_columns(query)))


def _prompt_batching(config: LotusExecutionConfig) -> PromptBatching | None:
    """Return the shared prompt batching contract."""

    return config.prompt_batching


def execute_native_sem_agg_groups(
    query: QueryExpr,
    groups: Sequence[pd.DataFrame],
    input_cols: Sequence[str],
    config: LotusExecutionConfig,
) -> list[str]:
    """Aggregate independent groups with the configured physical dispatch."""

    prompt_batching = _prompt_batching(config)
    if prompt_batching is not None:
        import lotus

        output_col = aggregate_output_columns(query, input_cols)[0]
        result = execute_batch_prompted_sem_agg(
            [aggregate_group_text(group, input_cols) for group in groups],
            instruction=aggregate_instruction(query, input_cols),
            output_cols=(output_col,),
            model=lotus.settings.lm,
            prompt_batching=prompt_batching,
            max_retries=config.structured_parse_retries,
            model_kwargs=structured_sem_agg_model_kwargs(config),
            progress_bar_desc=config.sem_agg_progress_bar_desc,
            trace_dir=config.trace_dir(),
        )
        outputs = [str(values[output_col.name]) for values in result.outputs]
        for group_index, (group, output, raw_output_attempts, repair_method) in enumerate(
            zip(
                groups,
                outputs,
                result.raw_output_attempts,
                result.repair_methods,
                strict=True,
            )
        ):
            write_sem_agg_audit(
                config,
                query=query,
                group=group,
                input_cols=input_cols,
                output_cols=(output_col,),
                group_index=group_index,
                raw_output=raw_output_attempts[-1],
                raw_output_attempts=raw_output_attempts,
                parsed_output={output_col.name: output},
                syntax_repair_method=repair_method,
            )
        return outputs
    if config.sem_agg_dispatch == "sequential" or len(groups) < 2:
        return [
            execute_native_sem_agg_group(query, group, input_cols, config)
            for group in groups
        ]

    import lotus

    instruction = aggregate_instruction(query, input_cols)
    executions = _execute_independent_sem_agg_groups(
        [aggregate_group_text(group, input_cols) for group in groups],
        lotus.settings.lm,
        instruction,
        config=config,
    )
    return [execution.raw_output for execution in executions]


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
    instruction = aggregate_instruction(query, input_cols)
    kwargs: dict[str, Any] = {
        "safe_mode": config.sem_agg_safe_mode,
        "progress_bar_desc": config.sem_agg_progress_bar_desc,
    }
    output = sem_agg(
        docs,
        lotus.settings.lm,
        instruction,
        [0] * len(docs),
        **kwargs,
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
    """Execute multi-output aggregation with LOTUS-style structured final output."""

    if source.empty:
        return pd.DataFrame(columns=list(output_columns(query)))

    config = config or LotusExecutionConfig()
    grouped = aggregate_groups_with_keys(source)
    parsed_outputs = execute_structured_sem_agg_groups(
        query,
        [group for _key_values, group in grouped],
        input_cols,
        output_cols,
        config,
    )
    rows: list[dict[str, Any]] = []
    for (key_values, _group), parsed in zip(
        grouped,
        parsed_outputs,
        strict=True,
    ):
        row = dict(key_values)
        for column in output_cols:
            if column.name not in row:
                row[column.name] = parsed[column.name]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(output_columns(query)))


def execute_structured_sem_agg_groups(
    query: QueryExpr,
    groups: Sequence[pd.DataFrame],
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
) -> list[Mapping[str, Any]]:
    """Aggregate independent groups into declared structured fields."""

    prompt_batching = _prompt_batching(config)
    if prompt_batching is not None:
        import lotus

        instruction = aggregate_instruction(query, input_cols)
        result = execute_batch_prompted_sem_agg(
            [aggregate_group_text(group, input_cols) for group in groups],
            instruction=instruction,
            output_cols=output_cols,
            model=lotus.settings.lm,
            prompt_batching=prompt_batching,
            max_retries=config.structured_parse_retries,
            model_kwargs=structured_sem_agg_model_kwargs(config),
            progress_bar_desc=config.sem_agg_progress_bar_desc,
            trace_dir=config.trace_dir(),
        )
        parsed_outputs: list[Mapping[str, Any]] = [
            dict(values) for values in result.outputs
        ]
        for group_index, (group, parsed, repair_method) in enumerate(
            zip(groups, parsed_outputs, result.repair_methods, strict=True)
        ):
            raw_output_attempts = result.raw_output_attempts[group_index]
            raw_output = raw_output_attempts[-1]
            write_sem_agg_audit(
                config,
                query=query,
                group=group,
                input_cols=input_cols,
                output_cols=output_cols,
                group_index=group_index,
                raw_output=raw_output,
                raw_output_attempts=raw_output_attempts,
                parsed_output=parsed,
                syntax_repair_method=repair_method,
            )
        return parsed_outputs
    if config.sem_agg_dispatch == "sequential" or len(groups) < 2:
        return [
            execute_structured_sem_agg_group(
                query,
                group,
                input_cols,
                output_cols,
                config,
                group_index=group_index,
            )
            for group_index, group in enumerate(groups)
        ]

    import lotus

    instruction = structured_aggregate_instruction(query, input_cols, output_cols)
    executions = _execute_independent_sem_agg_groups(
        [aggregate_group_text(group, input_cols) for group in groups],
        lotus.settings.lm,
        instruction,
        config=config,
        output_cols=output_cols,
        failure_extra_by_group=[
            {
                "group_index": group_index,
                "final_instruction": instruction,
                "group_row_preview": group.head(5)
                .astype(str)
                .to_dict(orient="records"),
            }
            for group_index, group in enumerate(groups)
        ],
    )
    return [
        _parse_structured_sem_agg_execution(
            execution,
            query=query,
            group=group,
            input_cols=input_cols,
            output_cols=output_cols,
            config=config,
            group_index=group_index,
            instruction=instruction,
        )
        for group_index, (group, execution) in enumerate(
            zip(groups, executions, strict=True)
        )
    ]


def execute_structured_sem_agg_group(
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
    *,
    group_index: int = 0,
) -> Mapping[str, Any]:
    """Aggregate one group into declared structured fields."""

    instruction = structured_aggregate_instruction(query, input_cols, output_cols)
    retry_result = execute_lotus_style_structured_sem_agg_group(
        query,
        group,
        input_cols,
        output_cols,
        config,
        group_index=group_index,
    )
    execution = _SemAggExecution(
        raw_output=str(retry_result.raw_outputs[0]),
        retry_result=retry_result,
    )
    return _parse_structured_sem_agg_execution(
        execution,
        query=query,
        group=group,
        input_cols=input_cols,
        output_cols=output_cols,
        config=config,
        group_index=group_index,
        instruction=instruction,
    )


def _parse_structured_sem_agg_execution(
    execution: _SemAggExecution,
    *,
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
    group_index: int,
    instruction: str,
) -> Mapping[str, Any]:
    """Parse, audit, and return one structured aggregate execution."""

    retry_result = execution.retry_result
    if retry_result is None:
        raise ValueError("structured sem_agg execution is missing retry metadata")
    raw_output = execution.raw_output
    raw_output_attempts = tuple(
        str(value) for value in retry_result.raw_output_attempts[0]
    )
    if retry_result.invalid_indices:
        parse_error = structured_parse_error(
            raw_output,
            output_cols=output_cols,
            shape="object",
            require_explanation=False,
            operator="sem_agg",
        )
        artifact_path = (
            retry_result.failure_artifact_paths[0]
            if retry_result.failure_artifact_paths
            else write_sem_agg_failure_artifact(
                raw_output_attempts,
                group,
                output_cols,
                instruction=instruction,
                group_index=group_index,
            )
        )
        write_sem_agg_audit(
            config,
            query=query,
            group=group,
            input_cols=input_cols,
            output_cols=output_cols,
            group_index=group_index,
            raw_output=raw_output,
            raw_output_attempts=raw_output_attempts,
            parsed_output=None,
            parse_error=parse_error,
            failure_artifact=artifact_path,
        )
        raise ValueError(f"{parse_error}; structured failure artifact: {artifact_path}")

    parsed = parse_structured_sem_agg_output(raw_output, output_cols)
    write_sem_agg_audit(
        config,
        query=query,
        group=group,
        input_cols=input_cols,
        output_cols=output_cols,
        group_index=group_index,
        raw_output=raw_output,
        raw_output_attempts=raw_output_attempts,
        parsed_output=parsed,
    )
    return parsed


def execute_lotus_style_structured_sem_agg_group(
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
    *,
    group_index: int = 0,
) -> StructuredLMRetryResult:
    """Run a LOTUS-main-style hierarchical aggregate with final JSON output."""

    import lotus

    docs = aggregate_group_text(group, input_cols)
    instruction = structured_aggregate_instruction(query, input_cols, output_cols)
    return lotus_style_structured_sem_agg(
        docs,
        lotus.settings.lm,
        instruction,
        [0] * len(docs),
        safe_mode=config.sem_agg_safe_mode,
        progress_bar_desc=config.sem_agg_progress_bar_desc,
        output_cols=output_cols,
        max_retries=config.structured_parse_retries,
        final_model_kwargs=structured_sem_agg_model_kwargs(config),
        failure_extra_by_index={
            0: {
                "group_index": group_index,
                "final_instruction": instruction,
                "group_row_preview": group.head(5)
                .astype(str)
                .to_dict(orient="records"),
            }
        },
    )


def structured_aggregate_instruction(
    query: QueryExpr,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
) -> str:
    """Build a structured aggregate instruction after resolving input placeholders."""

    instruction = aggregate_instruction(query, input_cols)
    field_lines = "\n".join(
        f"- {column.name}: {column.description or 'string'}" for column in output_cols
    )
    shape = json.dumps(
        {column.name: "string" for column in output_cols},
        ensure_ascii=True,
    )
    return (
        f"{instruction}\n\n"
        "Return exactly one valid JSON object for the aggregate result.\n"
        "Required output fields:\n"
        f"{field_lines}\n"
        "All values must be strings. Do not include extra keys.\n"
        f"Expected JSON shape: {shape}"
    )


def aggregate_instruction(query: QueryExpr, input_cols: Sequence[str]) -> str:
    """Resolve semantic aggregate placeholders against aggregate input columns."""

    import lotus

    formatter_instruction = escape_structured_formatter_placeholders(
        str(query.params["instruction"]),
        input_cols=input_cols,
        output_cols=aggregate_output_columns(query, input_cols),
    )
    return lotus.nl_expression.nle2str(
        formatter_instruction,
        list(input_cols),
    )


def write_sem_agg_audit(
    config: LotusExecutionConfig,
    *,
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    group_index: int,
    raw_output: Any,
    parsed_output: Mapping[str, Any] | None,
    raw_output_attempts: Sequence[str] | None = None,
    parse_error: str = "",
    failure_artifact: Path | str | None = None,
    syntax_repair_method: str | None = None,
) -> None:
    """Write a structured audit row for one semantic aggregate group."""

    instruction = str(query.params["instruction"])
    formatted_instruction = aggregate_instruction(query, input_cols)
    final_instruction = (
        structured_aggregate_instruction(query, input_cols, output_cols)
        if len(output_cols) > 1
        else formatted_instruction
    )
    input_preview = (
        group.loc[:, list(input_cols)].head(20).astype(str).to_dict(orient="records")
    )
    attempts = [str(value) for value in (raw_output_attempts or (raw_output,))]
    audit_row = {
        "operator": "sem_agg",
        "group_index": group_index,
        "shape": "object" if len(output_cols) > 1 else "string",
        "input_cols": list(input_cols),
        "input_preview": input_preview,
        "instruction": instruction,
        "formatted_instruction": formatted_instruction,
        "final_instruction": final_instruction,
        "required_output_cols": [
            {"name": column.name, "description": column.description}
            for column in output_cols
        ],
        "raw_output": str(raw_output),
        "raw_output_attempts": attempts,
        "parse_retry_attempts": max(len(attempts) - 1, 0),
        "parsed_output": parsed_output,
        "parse_error": parse_error,
        "failure_artifact": "" if failure_artifact is None else str(failure_artifact),
    }
    if (
        config.sem_agg_dispatch != "sequential"
        or config.prompt_batching is not None
    ):
        audit_row.update(
            {
                "dispatch": config.sem_agg_dispatch,
                "prompt_batching": (
                    None
                    if config.prompt_batching is None
                    else config.prompt_batching.to_dict()
                ),
                "structured_output_repaired": syntax_repair_method is not None,
                "structured_output_repair_method": syntax_repair_method or "",
            }
        )
    write_structured_generation_trace(
        config.trace_dir(),
        operator="sem_agg",
        rows=[audit_row],
        snapshots={"group": group.loc[:, list(input_cols)].copy()},
    )


def _execute_independent_sem_agg_groups(
    documents_by_group: Sequence[Sequence[str]],
    model: Any,
    instruction: str,
    *,
    config: LotusExecutionConfig,
    output_cols: Sequence[ColumnSpec] | None = None,
    failure_extra_by_group: Sequence[Mapping[str, Any]] | None = None,
) -> list[_SemAggExecution]:
    """Batch ready prompts while preserving independent aggregate groups."""

    if failure_extra_by_group is not None and len(failure_extra_by_group) != len(
        documents_by_group
    ):
        raise ValueError("sem_agg failure metadata must match aggregate groups")
    states = [
        _SemAggGroupState(
            group_index=group_index,
            documents=[str(document) for document in documents],
        )
        for group_index, documents in enumerate(documents_by_group)
    ]
    if any(not state.documents for state in states):
        raise ValueError("sem_agg aggregate groups cannot be empty")

    while any(state.execution is None for state in states):
        regular_prompts: list[list[dict[str, str]]] = []
        regular_refs: list[tuple[_SemAggGroupState, int]] = []
        regular_counts: dict[int, int] = {}
        structured_prompts: list[list[dict[str, str]]] = []
        structured_states: list[_SemAggGroupState] = []

        for state in states:
            if state.execution is not None:
                continue
            prompts = _build_independent_sem_agg_level_prompts(
                state.documents,
                model,
                instruction,
                tree_level=state.tree_level,
                lotus_native_prompt=output_cols is None,
            )
            if output_cols is not None and len(prompts) == 1:
                structured_prompts.append(prompts[0])
                structured_states.append(state)
                continue
            regular_counts[state.group_index] = len(prompts)
            for prompt_index, prompt in enumerate(prompts):
                regular_prompts.append(prompt)
                regular_refs.append((state, prompt_index))

        if regular_prompts:
            regular_output = model(
                regular_prompts,
                progress_bar_desc=config.sem_agg_progress_bar_desc,
            )
            raw_outputs = [str(value) for value in regular_output.outputs]
            if len(raw_outputs) != len(regular_prompts):
                raise ValueError(
                    "sem_agg provider batch returned an unexpected number of "
                    f"outputs: expected {len(regular_prompts)}, got {len(raw_outputs)}"
                )
            outputs_by_group: dict[int, list[str]] = {
                group_index: [""] * count
                for group_index, count in regular_counts.items()
            }
            for (state, prompt_index), raw_output in zip(
                regular_refs,
                raw_outputs,
                strict=True,
            ):
                outputs_by_group[state.group_index][prompt_index] = raw_output
            for state in states:
                outputs = outputs_by_group.get(state.group_index)
                if outputs is None:
                    continue
                state.documents = outputs
                state.tree_level += 1
                if len(outputs) == 1 and output_cols is None:
                    state.execution = _SemAggExecution(raw_output=outputs[0])

        if structured_prompts:
            failure_extra = (
                {
                    prompt_index: dict(failure_extra_by_group[state.group_index])
                    for prompt_index, state in enumerate(structured_states)
                }
                if failure_extra_by_group is not None
                else None
            )
            retry_result = execute_structured_lm_retry_result(
                model,
                structured_prompts,
                lm_kwargs={
                    "progress_bar_desc": config.sem_agg_progress_bar_desc,
                    **structured_sem_agg_model_kwargs(config),
                    "response_format": JSON_OBJECT_RESPONSE_FORMAT,
                },
                output_cols=output_cols or (),
                shape="object",
                require_explanation=False,
                operator="sem_agg",
                max_retries=config.structured_parse_retries,
                failure_extra_by_index=failure_extra,
            )
            artifact_by_index = dict(
                zip(
                    retry_result.invalid_indices,
                    retry_result.failure_artifact_paths,
                    strict=True,
                )
            )
            invalid_indices = set(retry_result.invalid_indices)
            for prompt_index, state in enumerate(structured_states):
                invalid = prompt_index in invalid_indices
                state.execution = _SemAggExecution(
                    raw_output=str(retry_result.raw_outputs[prompt_index]),
                    retry_result=StructuredLMRetryResult(
                        raw_outputs=(retry_result.raw_outputs[prompt_index],),
                        raw_output_attempts=(
                            retry_result.raw_output_attempts[prompt_index],
                        ),
                        invalid_indices=(0,) if invalid else (),
                        failure_artifact_paths=(
                            (artifact_by_index[prompt_index],) if invalid else ()
                        ),
                    ),
                )

        if config.sem_agg_safe_mode:
            model.print_total_usage()

    executions = [state.execution for state in states]
    if any(execution is None for execution in executions):
        raise RuntimeError("sem_agg provider batching did not complete every group")
    return [execution for execution in executions if execution is not None]


def _build_independent_sem_agg_level_prompts(
    documents: Sequence[str],
    model: Any,
    instruction: str,
    *,
    tree_level: int,
    lotus_native_prompt: bool,
) -> list[list[dict[str, str]]]:
    """Build one LOTUS-compatible tree level for one aggregate group."""

    template = (
        leaf_instruction_template(
            instruction,
            lotus_native_spacing=lotus_native_prompt,
        )
        if tree_level == 0
        else node_instruction_template(
            instruction,
            lotus_native_spacing=lotus_native_prompt,
        )
    )
    template_tokens = model.count_tokens(template)
    context_str = ""
    context_tokens = 0
    document_counter = 1
    prompts: list[list[dict[str, str]]] = []

    for document in documents:
        formatted = format_aggregate_doc(
            tree_level,
            str(document),
            document_counter,
        )
        new_tokens = model.count_tokens(formatted)
        if (
            new_tokens + context_tokens + template_tokens
            > model.max_ctx_len - model.max_tokens
        ):
            prompt = template.replace("{{docs_str}}", context_str)
            prompts.append([{"role": "user", "content": prompt}])
            document_counter = 1
            formatted = format_aggregate_doc(
                tree_level, str(document), document_counter
            )
            context_str = formatted
            context_tokens = new_tokens
            document_counter += 1
            continue
        context_str += formatted
        context_tokens += new_tokens
        document_counter += 1

    if document_counter > 1 or len(documents) == 1:
        prompt = template.replace("{{docs_str}}", context_str)
        prompts.append([{"role": "user", "content": prompt}])
    return prompts


def lotus_style_sem_agg(
    docs: Sequence[str],
    model: Any,
    user_instruction: str,
    partition_ids: Sequence[int],
    *,
    safe_mode: bool = False,
    progress_bar_desc: str = "Aggregating",
    response_format: Any = None,
    final_model_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Compatibility copy of LOTUS main sem_agg structured-final-pass behavior."""

    output, _retry_result = _execute_lotus_style_sem_agg(
        docs,
        model,
        user_instruction,
        partition_ids,
        safe_mode=safe_mode,
        progress_bar_desc=progress_bar_desc,
        response_format=response_format,
        final_model_kwargs=final_model_kwargs,
    )
    return output


def lotus_style_structured_sem_agg(
    docs: Sequence[str],
    model: Any,
    user_instruction: str,
    partition_ids: Sequence[int],
    *,
    output_cols: Sequence[ColumnSpec],
    max_retries: int,
    safe_mode: bool = False,
    progress_bar_desc: str = "Aggregating",
    final_model_kwargs: Mapping[str, Any] | None = None,
    failure_extra_by_index: Mapping[int, Mapping[str, Any]] | None = None,
) -> StructuredLMRetryResult:
    """Run a hierarchical aggregate and retry only its final structured call."""

    _output, retry_result = _execute_lotus_style_sem_agg(
        docs,
        model,
        user_instruction,
        partition_ids,
        safe_mode=safe_mode,
        progress_bar_desc=progress_bar_desc,
        response_format=JSON_OBJECT_RESPONSE_FORMAT,
        final_model_kwargs=final_model_kwargs,
        structured_output_cols=output_cols,
        structured_parse_retries=max_retries,
        failure_extra_by_index=failure_extra_by_index,
    )
    if retry_result is None:
        raise ValueError("structured sem_agg requires at least one input document")
    return retry_result


def _execute_lotus_style_sem_agg(
    docs: Sequence[str],
    model: Any,
    user_instruction: str,
    partition_ids: Sequence[int],
    *,
    safe_mode: bool,
    progress_bar_desc: str,
    response_format: Any,
    final_model_kwargs: Mapping[str, Any] | None,
    structured_output_cols: Sequence[ColumnSpec] | None = None,
    structured_parse_retries: int = 0,
    failure_extra_by_index: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[str, StructuredLMRetryResult | None]:
    """Execute LOTUS-style aggregation and expose final structured retry metadata."""

    import lotus

    if safe_mode:
        lotus.logger.warning("Safe mode is not implemented yet")

    doc_list = [str(doc) for doc in docs]
    current_partition_ids = list(partition_ids)
    if not doc_list:
        return "", None

    tree_level = 0
    summaries: list[str] = []
    structured_retry_result: StructuredLMRetryResult | None = None
    while len(doc_list) != 1 or summaries == []:
        current_partition_id = current_partition_ids[0]
        do_fold = len(current_partition_ids) == len(set(current_partition_ids))
        context_str = ""
        batch = []
        template = (
            leaf_instruction_template(user_instruction)
            if tree_level == 0
            else node_instruction_template(user_instruction)
        )
        template_tokens = model.count_tokens(template)
        context_tokens = 0
        doc_counter = 1
        new_partition_ids: list[int] = []

        for idx, doc in enumerate(doc_list):
            partition_id = current_partition_ids[idx]
            formatted_doc = format_aggregate_doc(tree_level, doc, doc_counter)
            new_tokens = model.count_tokens(formatted_doc)

            if (
                new_tokens + context_tokens + template_tokens
                > model.max_ctx_len - model.max_tokens
            ) or (partition_id != current_partition_id and not do_fold):
                prompt = template.replace("{{docs_str}}", context_str)
                lotus.logger.debug(f"Prompt added to batch: {prompt}")
                batch.append([{"role": "user", "content": prompt}])
                new_partition_ids.append(current_partition_id)
                current_partition_id = partition_id
                doc_counter = 1

                formatted_doc = format_aggregate_doc(tree_level, doc, doc_counter)
                context_str = formatted_doc
                context_tokens = new_tokens
                doc_counter += 1
            else:
                context_str += formatted_doc
                context_tokens += new_tokens
                doc_counter += 1

        if doc_counter > 1 or len(doc_list) == 1:
            prompt = template.replace("{{docs_str}}", context_str)
            lotus.logger.debug(f"Prompt added to batch: {prompt}")
            batch.append([{"role": "user", "content": prompt}])
            new_partition_ids.append(current_partition_id)

        model_kwargs: dict[str, Any] = {}
        is_final_pass = len(batch) == 1
        if is_final_pass:
            model_kwargs.update(final_model_kwargs or {})
        if is_final_pass and response_format is not None:
            model_kwargs["response_format"] = response_format

        if is_final_pass and structured_output_cols is not None:
            structured_retry_result = execute_structured_lm_retry_result(
                model,
                batch,
                lm_kwargs={
                    "progress_bar_desc": progress_bar_desc,
                    **model_kwargs,
                },
                output_cols=structured_output_cols,
                shape="object",
                require_explanation=False,
                operator="sem_agg",
                max_retries=structured_parse_retries,
                failure_extra_by_index=failure_extra_by_index,
            )
            summaries = [str(output) for output in structured_retry_result.raw_outputs]
        else:
            lm_output = model(
                batch,
                progress_bar_desc=progress_bar_desc,
                **model_kwargs,
            )
            summaries = [str(output) for output in lm_output.outputs]
        doc_list = summaries
        current_partition_ids = new_partition_ids
        lotus.logger.debug(f"Model outputs from tree level {tree_level}: {summaries}")
        tree_level += 1
        if safe_mode:
            model.print_total_usage()

    if not summaries:
        return "", structured_retry_result
    return summaries[0], structured_retry_result


def structured_sem_agg_model_kwargs(
    config: LotusExecutionConfig | None = None,
) -> dict[str, Any]:
    """Return final-pass model kwargs for structured aggregate JSON."""

    import lotus

    config = config or LotusExecutionConfig()
    kwargs = dict(config.sem_agg_model_kwargs)
    if "response_format" in kwargs:
        raise ValueError("sem_agg_model_kwargs cannot override response_format")
    if "progress_bar_desc" in kwargs:
        raise ValueError("sem_agg_model_kwargs cannot override progress_bar_desc")

    current = int(getattr(lotus.settings.lm, "max_tokens", 512) or 512)
    return {"max_tokens": max(current, config.structured_max_tokens), **kwargs}


def write_sem_agg_failure_artifact(
    raw_output_attempts: Sequence[str],
    group: pd.DataFrame,
    output_cols: Sequence[ColumnSpec],
    *,
    instruction: str,
    group_index: int,
) -> Path:
    """Write a structured sem_agg failure artifact and return its path."""

    paths = write_structured_failure_artifacts(
        [instruction],
        [list(raw_output_attempts)],
        [0],
        output_cols=output_cols,
        shape="object",
        require_explanation=False,
        operator="sem_agg",
        extra_by_index={
            0: {
                "group_index": group_index,
                "final_instruction": instruction,
                "group_row_preview": group.head(5)
                .astype(str)
                .to_dict(orient="records"),
            }
        },
    )
    return paths[0]


def leaf_instruction_template(
    user_instruction: str,
    *,
    lotus_native_spacing: bool = False,
) -> str:
    """Return the LOTUS leaf-level semantic aggregation prompt template."""

    instruction_prefix = "Instruction:  " if lotus_native_spacing else "Instruction: "
    return (
        "Your job is to provide an answer to the user's instruction given the context below from multiple documents.\n"
        "Remember that your job is to answer the user's instruction by combining all relevant information from all provided documents, into a single coherent answer.\n"
        "Do NOT copy the format of the sources! Instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
        "You have limited space to provide your answer, so be concise and to the point.\n\n---\n\n"
        "Follow the following format.\n\nContext: relevant facts from multiple documents\n\n"
        "Instruction: the instruction provided by the user\n\nAnswer: Write your answer\n\n---\n\n"
        "Context: {{docs_str}}\n\n"
        f"{instruction_prefix}{user_instruction}\n\nAnswer:\n"
    )


def node_instruction_template(
    user_instruction: str,
    *,
    lotus_native_spacing: bool = False,
) -> str:
    """Return the LOTUS intermediate-node semantic aggregation prompt template."""

    instruction_prefix = "Instruction:  " if lotus_native_spacing else "Instruction: "
    return (
        "Your job is to provide an answer to the user's instruction given the context below from multiple sources.\n"
        "Note that each source may be formatted differently and contain information about several different documents.\n"
        "Remember that your job is to answer the user's instruction by combining all relevant information from all provided sources, into a single coherent answer.\n"
        "The sources may provide opposing viewpoints or complementary information.\n"
        "Be sure to include information from ALL relevant sources in your answer.\n"
        "Do NOT copy the format of the sources, instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
        "You have limited space to provide your answer, so be concise and to the point.\n"
        "You may need to draw connections between sources to provide a complete answer.\n\n---\n\n"
        "Follow the following format.\n\nContext: relevant facts from multiple sources\n\n"
        "Instruction: the instruction provided by the user\n\nAnswer: Write your answer\n\n---\n\n"
        "Context: {{docs_str}}\n\n"
        f"{instruction_prefix}{user_instruction}\n\nAnswer:\n"
    )


def format_aggregate_doc(tree_level: int, doc: str, counter: int) -> str:
    """Format a leaf document or intermediate summary for aggregation."""

    label = "Document" if tree_level == 0 else "Source"
    return f"\n\t{label} {counter}: {doc}"


def parse_structured_sem_agg_output(
    raw_output: Any,
    output_cols: Sequence[ColumnSpec],
) -> dict[str, Any]:
    """Parse and validate one structured LOTUS sem_agg output."""

    if isinstance(raw_output, Mapping):
        missing = [
            column.name for column in output_cols if column.name not in raw_output
        ]
        if missing:
            raise ValueError(f"sem_agg JSON output is missing required keys: {missing}")
        return structured_scalar_values(
            raw_output,
            output_cols,
            operator="sem_agg",
        )

    parsed, _explanation = parse_structured_object_json(
        str(raw_output),
        output_cols,
        operator="sem_agg",
    )
    return parsed


def aggregate_input_columns(
    source: pd.DataFrame,
    input_cols: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve input columns for semantic aggregation."""

    if input_cols is not None:
        columns = tuple(str(column) for column in input_cols)
    else:
        columns = tuple(
            str(column) for column in source.columns if column != GROUP_ID_COLUMN
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


def aggregate_groups(source: pd.DataFrame) -> list[pd.DataFrame]:
    """Return one dataframe per aggregate group without internal group ids."""

    return [group for _key_values, group in aggregate_groups_with_keys(source)]


def aggregate_groups_with_keys(
    source: pd.DataFrame,
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """Return grouped frames plus deterministic key values to preserve."""

    deterministic_keys = tuple(
        str(key) for key in source.attrs.get("agent_memory_groupby_keys", ())
    )
    partition_keys = tuple(
        str(key)
        for key in source.attrs.get("agent_memory_sem_groupby_partition_by", ())
    )
    if deterministic_keys:
        return _aggregate_groups_by_keys(
            source,
            group_keys=deterministic_keys,
            output_keys=deterministic_keys,
            drop_columns=(),
        )
    if GROUP_ID_COLUMN not in source.columns:
        return [({}, source.reset_index(drop=True).copy())]
    return _aggregate_groups_by_keys(
        source,
        group_keys=(*partition_keys, GROUP_ID_COLUMN),
        output_keys=partition_keys,
        drop_columns=(GROUP_ID_COLUMN,),
    )


def _aggregate_groups_by_keys(
    source: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    output_keys: Sequence[str],
    drop_columns: Sequence[str],
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """Group frames and carry selected key values into aggregate output rows."""

    missing = [column for column in group_keys if column not in source.columns]
    if missing:
        raise ValueError(
            f"aggregate group key columns not found in DataFrame: {missing}"
        )
    if source.empty:
        return []
    groups: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for key, group in source.groupby(list(group_keys), sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        by_key = dict(zip(group_keys, key_values, strict=True))
        output_key_values = {key_name: by_key[key_name] for key_name in output_keys}
        groups.append(
            (
                output_key_values,
                group.drop(columns=list(drop_columns), errors="ignore").reset_index(
                    drop=True
                ),
            )
        )
    return groups


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
    parsed_outputs: Sequence[Mapping[str, Any]],
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
