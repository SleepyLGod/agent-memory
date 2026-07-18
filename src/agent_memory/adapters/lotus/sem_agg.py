"""LOTUS-backed semantic aggregation lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    emit_columns = tuple(output_columns(over_query.inputs[0])) or tuple(emit_source.columns)
    missing_emit = [column for column in emit_columns if column not in emit_source.columns]
    if missing_emit:
        raise ValueError(f"over sem_agg emit columns not found in DataFrame: {missing_emit}")
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

        input_cols = aggregate_input_columns(frame.frame, query.params.get("input_cols"))
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
    rows: list[dict[str, Any]] = []
    for group_index, (key_values, group) in enumerate(aggregate_groups_with_keys(source)):
        raw_output = execute_native_sem_agg_group(query, group, input_cols, config)
        row = dict(key_values)
        if output_col.name not in row:
            row[output_col.name] = raw_output
        rows.append(row)
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
    rows: list[dict[str, Any]] = []
    for group_index, (key_values, group) in enumerate(aggregate_groups_with_keys(source)):
        parsed = execute_structured_sem_agg_group(
            query,
            group,
            input_cols,
            output_cols,
            config,
            group_index=group_index,
        )
        row = dict(key_values)
        for column in output_cols:
            if column.name not in row:
                row[column.name] = parsed[column.name]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(output_columns(query)))


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
    raw_output = str(retry_result.raw_outputs[0])
    raw_output_attempts = tuple(str(value) for value in retry_result.raw_output_attempts[0])
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
                "group_row_preview": group.head(5).astype(str).to_dict(orient="records"),
            }
        },
    )


def structured_aggregate_instruction(
    query: QueryExpr,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
) -> str:
    """Build a structured aggregate instruction after resolving input placeholders."""

    import lotus

    instruction = aggregate_instruction(query, input_cols)
    field_lines = "\n".join(
        f"- {column.name}: {column.description or 'string'}"
        for column in output_cols
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
) -> None:
    """Write a structured audit row for one semantic aggregate group."""

    instruction = str(query.params["instruction"])
    formatted_instruction = aggregate_instruction(query, input_cols)
    final_instruction = (
        structured_aggregate_instruction(query, input_cols, output_cols)
        if len(output_cols) > 1
        else formatted_instruction
    )
    input_preview = group.loc[:, list(input_cols)].head(20).astype(str).to_dict(
        orient="records"
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
    write_structured_generation_trace(
        config.trace_dir(),
        operator="sem_agg",
        rows=[audit_row],
        snapshots={"group": group.loc[:, list(input_cols)].copy()},
    )


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
                "group_row_preview": group.head(5).astype(str).to_dict(orient="records"),
            }
        },
    )
    return paths[0]


def leaf_instruction_template(user_instruction: str) -> str:
    """Return the LOTUS leaf-level semantic aggregation prompt template."""

    return (
        "Your job is to provide an answer to the user's instruction given the context below from multiple documents.\n"
        "Remember that your job is to answer the user's instruction by combining all relevant information from all provided documents, into a single coherent answer.\n"
        "Do NOT copy the format of the sources! Instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
        "You have limited space to provide your answer, so be concise and to the point.\n\n---\n\n"
        "Follow the following format.\n\nContext: relevant facts from multiple documents\n\n"
        "Instruction: the instruction provided by the user\n\nAnswer: Write your answer\n\n---\n\n"
        "Context: {{docs_str}}\n\n"
        f"Instruction: {user_instruction}\n\nAnswer:\n"
    )


def node_instruction_template(user_instruction: str) -> str:
    """Return the LOTUS intermediate-node semantic aggregation prompt template."""

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
        f"Instruction: {user_instruction}\n\nAnswer:\n"
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
        missing = [column.name for column in output_cols if column.name not in raw_output]
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
        str(key) for key in source.attrs.get("agent_memory_sem_groupby_partition_by", ())
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
        raise ValueError(f"aggregate group key columns not found in DataFrame: {missing}")
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
                group.drop(columns=list(drop_columns), errors="ignore").reset_index(drop=True),
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
