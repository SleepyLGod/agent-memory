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
from agent_memory.adapters.lotus.sem_groupby import GROUP_ID_COLUMN
from agent_memory.adapters.lotus.structured import (
    parse_structured_object_json,
    write_structured_failure_artifacts,
)
from agent_memory.logical import ColumnSpec, QueryExpr

JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}


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
        return pd.DataFrame(columns=[column.name for column in output_cols])

    config = config or LotusExecutionConfig()
    parsed_outputs = [
        execute_structured_sem_agg_group(
            query,
            group,
            input_cols,
            output_cols,
            config,
            group_index=group_index,
        )
        for group_index, group in enumerate(aggregate_groups(source))
    ]
    return apply_structured_aggregate_outputs(parsed_outputs, output_cols)


def execute_structured_sem_agg_group(
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
    *,
    group_index: int = 0,
) -> Mapping[str, str]:
    """Aggregate one group into declared structured fields."""

    instruction = structured_aggregate_instruction(query, input_cols, output_cols)
    raw_output = execute_lotus_style_structured_sem_agg_group(
        query,
        group,
        input_cols,
        output_cols,
        config,
    )
    try:
        return parse_structured_sem_agg_output(raw_output, output_cols)
    except ValueError as error:
        artifact_path = write_sem_agg_failure_artifact(
            raw_output,
            group,
            output_cols,
            instruction=instruction,
            group_index=group_index,
        )
        raise ValueError(
            f"{error}; structured failure artifact: {artifact_path}"
        ) from error


def execute_lotus_style_structured_sem_agg_group(
    query: QueryExpr,
    group: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    config: LotusExecutionConfig,
) -> str:
    """Run a LOTUS-main-style hierarchical aggregate with final JSON output."""

    import lotus

    docs = aggregate_group_text(group, input_cols)
    instruction = structured_aggregate_instruction(query, input_cols, output_cols)
    return lotus_style_sem_agg(
        docs,
        lotus.settings.lm,
        instruction,
        [0] * len(docs),
        safe_mode=config.sem_agg_safe_mode,
        progress_bar_desc=config.sem_agg_progress_bar_desc,
        response_format=JSON_OBJECT_RESPONSE_FORMAT,
        final_model_kwargs=structured_sem_agg_model_kwargs(config),
    )


def structured_aggregate_instruction(
    query: QueryExpr,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
) -> str:
    """Build a structured aggregate instruction after resolving input placeholders."""

    import lotus

    instruction = lotus.nl_expression.nle2str(
        str(query.params["instruction"]),
        list(input_cols),
    )
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

    import lotus

    if safe_mode:
        lotus.logger.warning("Safe mode is not implemented yet")

    doc_list = [str(doc) for doc in docs]
    current_partition_ids = list(partition_ids)
    if not doc_list:
        return ""

    tree_level = 0
    summaries: list[str] = []
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
        return ""
    return summaries[0]


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
    raw_output: Any,
    group: pd.DataFrame,
    output_cols: Sequence[ColumnSpec],
    *,
    instruction: str,
    group_index: int,
) -> Path:
    """Write a structured sem_agg failure artifact and return its path."""

    paths = write_structured_failure_artifacts(
        [instruction],
        [[str(raw_output)]],
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
) -> dict[str, str]:
    """Parse and validate one structured LOTUS sem_agg output."""

    if isinstance(raw_output, Mapping):
        missing = [column.name for column in output_cols if column.name not in raw_output]
        if missing:
            raise ValueError(f"sem_agg JSON output is missing required keys: {missing}")
        return {column.name: str(raw_output[column.name]) for column in output_cols}

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
