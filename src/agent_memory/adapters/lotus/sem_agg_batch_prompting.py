"""Batch-prompting access path for independent semantic aggregate groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from agent_memory.adapters.lotus.prompt_batching import (
    ParsedPromptBatch,
    PromptBatchItem,
    PromptBatchRequest,
    PromptBatching,
    run_prompt_batches,
)
from agent_memory.adapters.lotus.structured import (
    load_structured_json_with_syntax_repair,
    structured_scalar_values,
)
from agent_memory.policy.logical import ColumnSpec


SEM_AGG_BATCH_SYSTEM_PROMPT = (
    "The user will provide one semantic aggregation instruction and several "
    "independent groups. Aggregate every group separately using all documents in "
    "that group. Never combine, compare, or copy information across group IDs. "
    "Return every supplied group_id exactly once, with exactly the requested "
    "output fields. Return only the requested JSON object."
)


@dataclass(frozen=True)
class BatchPromptedSemAggResult:
    """Per-group outputs plus prompt accounting for one aggregate call."""

    outputs: tuple[Mapping[str, Any], ...]
    raw_output_attempts: tuple[tuple[str, ...], ...]
    prompt_count: int
    retry_count: int
    repair_methods: tuple[str | None, ...]


@dataclass(frozen=True)
class _BatchPromptTask:
    group_index: int
    group_id: str
    documents: tuple[str, ...]


def execute_batch_prompted_sem_agg(
    documents_by_group: Sequence[Sequence[str]],
    *,
    instruction: str,
    output_cols: Sequence[ColumnSpec],
    model: Any,
    prompt_batching: PromptBatching,
    max_retries: int,
    model_kwargs: Mapping[str, Any],
    progress_bar_desc: str,
    trace_dir: Any = None,
) -> BatchPromptedSemAggResult:
    """Aggregate independent groups in bounded shared prompts."""

    max_tokens = int(model_kwargs.get("max_tokens", model.max_tokens))
    tasks = tuple(
        _BatchPromptTask(
            group_index=group_index,
            group_id=f"group_{group_index}",
            documents=tuple(str(document) for document in documents),
        )
        for group_index, documents in enumerate(documents_by_group)
    )
    if not tasks:
        return BatchPromptedSemAggResult((), (), 0, 0, ())

    request_model_kwargs = dict(model_kwargs)
    request_model_kwargs.pop("max_tokens", None)
    execution = run_prompt_batches(
        tasks,
        task_id=lambda task: task.group_id,
        build_request=lambda batch: _build_request(
            batch,
            instruction=instruction,
            output_cols=output_cols,
            max_tokens=max_tokens,
        ),
        parse_results=lambda raw_output: _parse_outputs(
            raw_output,
            output_cols=output_cols,
        ),
        model=model,
        config=prompt_batching,
        max_retries=max_retries,
        progress_bar_desc=progress_bar_desc,
        operator="sem_agg",
        trace_dir=trace_dir,
        model_kwargs=request_model_kwargs,
    )
    return BatchPromptedSemAggResult(
        outputs=execution.outputs,
        raw_output_attempts=execution.raw_output_attempts,
        prompt_count=execution.prompt_count,
        retry_count=execution.retry_count,
        repair_methods=execution.repair_methods,
    )


def _build_request(
    tasks: tuple[_BatchPromptTask, ...],
    *,
    instruction: str,
    output_cols: Sequence[ColumnSpec],
    max_tokens: int,
) -> PromptBatchRequest:
    payload = {
        "instruction": instruction,
        "groups": [
            {
                "group_id": task.group_id,
                "documents": list(task.documents),
            }
            for task in tasks
        ],
        "output_fields": {
            column.name: column.description or "string" for column in output_cols
        },
        "output_schema": {
            "results": [
                {
                    "group_id": "group_id",
                    "output": {column.name: "string" for column in output_cols},
                }
            ]
        },
    }
    return PromptBatchRequest(
        task_ids=tuple(task.group_id for task in tasks),
        prompt=[
            {"role": "system", "content": SEM_AGG_BATCH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ],
        max_tokens=max_tokens,
    )


def _parse_outputs(
    raw_output: str,
    *,
    output_cols: Sequence[ColumnSpec],
) -> ParsedPromptBatch[Mapping[str, Any]]:
    if not raw_output.strip():
        raise ValueError("batch-prompting sem_agg output is empty")
    decoded = load_structured_json_with_syntax_repair(
        raw_output,
        operator="batch-prompting sem_agg",
        expected_shape='JSON object with a "results" array',
    )
    payload = decoded.value
    if not isinstance(payload, dict) or set(payload) != {"results"}:
        raise ValueError("batch-prompting sem_agg output must contain only results")
    results = payload["results"]
    if not isinstance(results, list):
        raise ValueError("batch-prompting sem_agg results must be a list")

    parsed: list[PromptBatchItem[Mapping[str, Any]]] = []
    for result in results:
        if not isinstance(result, dict) or set(result) != {"group_id", "output"}:
            raise ValueError(
                "batch-prompting sem_agg results require group_id and output"
            )
        group_id = result["group_id"]
        output = result["output"]
        if not isinstance(group_id, str):
            raise ValueError("batch-prompting sem_agg group_id must be a string")
        if not isinstance(output, Mapping):
            raise ValueError("batch-prompting sem_agg output must be an object")
        expected_fields = {column.name for column in output_cols}
        if set(output) != expected_fields:
            raise ValueError(
                "batch-prompting sem_agg output fields do not match the declared "
                f"schema for {group_id!r}"
            )
        parsed.append(
            PromptBatchItem(
                group_id,
                structured_scalar_values(
                    output,
                    output_cols,
                    operator="sem_agg",
                ),
            )
        )
    return ParsedPromptBatch(
        items=tuple(parsed),
        repair_method=decoded.repair_method,
    )


__all__ = [
    "BatchPromptedSemAggResult",
    "execute_batch_prompted_sem_agg",
]
