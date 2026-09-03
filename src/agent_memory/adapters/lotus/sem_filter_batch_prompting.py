"""Batch-prompting access path for independent semantic filter tuples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

import pandas as pd
from lotus.cache import operator_cache

from agent_memory.adapters.lotus.prompt_batching import (
    ParsedPromptBatch,
    PromptBatchItem,
    PromptBatchRequest,
    PromptBatching,
    run_prompt_batches,
)

if TYPE_CHECKING:
    from agent_memory.adapters.lotus.context import LotusExecutionContext


SEM_FILTER_BATCH_SYSTEM_PROMPT = (
    "The user will provide one semantic filter claim and several independent row "
    "contexts. Evaluate the claim independently for every row. Set keep to true "
    "exactly when the claim is true for that row's context; otherwise set it to "
    "false. Copy every supplied row_id exactly once without adding or changing IDs. "
    "Do not compare rows with each other or let one row affect another row's "
    "decision. Return only the requested JSON object."
)


@dataclass(frozen=True)
class BatchPromptingResult:
    """Filtered rows and logical prompt accounting for one operator call."""

    frame: pd.DataFrame
    prompt_count: int
    retry_count: int
    tuple_count: int
    decisions: tuple[bool, ...] = ()
    raw_output_attempts: tuple[tuple[str, ...], ...] = ()
    repair_methods: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class _BatchPromptingTask:
    position: int
    row_id: str
    context: str


@dataclass
class _BatchPromptingExecutor:
    """Reuse LOTUS whole-operator caching for the opt-in access path."""

    _obj: pd.DataFrame

    @operator_cache
    def __call__(
        self,
        *,
        instruction: str,
        prompt_batching: PromptBatching,
        structured_parse_retries: int,
        structured_max_tokens: int,
        progress_bar_desc: str,
        trace_dir: Any = None,
        operator: str = "sem_filter",
    ) -> BatchPromptingResult:
        """Evaluate independent tuples in bounded structured prompts."""

        import lotus
        from lotus.templates import task_instructions

        lm = lotus.settings.lm
        if lm is None:
            raise ValueError(
                "batch-prompting sem_filter requires a configured language model"
            )
        columns = tuple(lotus.nl_expression.parse_cols(instruction))
        if not columns:
            raise ValueError(
                "batch-prompting sem_filter requires instruction column placeholders"
            )
        missing = [column for column in columns if column not in self._obj.columns]
        if missing:
            raise ValueError(
                f"batch-prompting sem_filter columns not found in DataFrame: {missing}"
            )
        docs = task_instructions.df2multimodal_info(self._obj, list(columns))
        if any(doc.get("image") for doc in docs):
            raise NotImplementedError(
                "batch-prompting sem_filter currently supports text columns only"
            )
        claim = _render_claim(instruction, columns)
        tasks = _build_tasks(docs)
        if not tasks:
            return BatchPromptingResult(
                frame=self._obj.copy(),
                prompt_count=0,
                retry_count=0,
                tuple_count=0,
            )

        current_max_tokens = int(getattr(lm, "max_tokens", 512) or 512)
        execution = run_prompt_batches(
            tasks,
            task_id=lambda task: task.row_id,
            build_request=lambda batch: _build_request(
                batch,
                claim=claim,
                max_tokens=max(current_max_tokens, structured_max_tokens),
            ),
            parse_results=_parse_decisions,
            model=lm,
            config=prompt_batching,
            max_retries=structured_parse_retries,
            progress_bar_desc=progress_bar_desc,
            operator=operator,
            trace_dir=trace_dir,
            model_kwargs={
                "show_progress_bar": True,
            },
        )
        selected_positions = [
            task.position
            for task, keep in zip(tasks, execution.outputs, strict=True)
            if keep
        ]
        return BatchPromptingResult(
            frame=self._obj.iloc[selected_positions].copy(),
            prompt_count=execution.prompt_count,
            retry_count=execution.retry_count,
            tuple_count=len(self._obj),
            decisions=execution.outputs,
            raw_output_attempts=execution.raw_output_attempts,
            repair_methods=execution.repair_methods,
        )


def execute_batch_prompted_sem_filter(
    source: pd.DataFrame,
    *,
    instruction: str,
    context: LotusExecutionContext,
    prompt_batching: PromptBatching,
) -> BatchPromptingResult:
    """Execute one explicitly configured batch-prompting semantic filter."""

    return execute_batch_prompted_predicate(
        source,
        instruction=instruction,
        prompt_batching=prompt_batching,
        structured_parse_retries=context.config.structured_parse_retries,
        structured_max_tokens=context.config.structured_max_tokens,
        progress_bar_desc=context.config.sem_filter_progress_bar_desc,
        trace_dir=context.config.trace_dir(),
        operator="sem_filter",
    )


def execute_batch_prompted_predicate(
    source: pd.DataFrame,
    *,
    instruction: str,
    prompt_batching: PromptBatching,
    structured_parse_retries: int,
    structured_max_tokens: int,
    progress_bar_desc: str,
    trace_dir: Any = None,
    operator: str = "sem_filter",
) -> BatchPromptingResult:
    """Evaluate independent boolean predicate tasks in shared prompts."""

    return _BatchPromptingExecutor(source)(
        instruction=instruction,
        prompt_batching=prompt_batching,
        structured_parse_retries=structured_parse_retries,
        structured_max_tokens=structured_max_tokens,
        progress_bar_desc=progress_bar_desc,
        trace_dir=trace_dir,
        operator=operator,
    )


def validate_batch_prompting_sem_filter_config(config: Any) -> None:
    """Reject LOTUS options the new access path cannot preserve."""

    unsupported: list[str] = []
    if config.sem_filter_examples is not None:
        unsupported.append("examples")
    if config.sem_filter_helper_examples is not None:
        unsupported.append("helper_examples")
    if config.sem_filter_strategy is not None:
        unsupported.append("strategy")
    if config.sem_filter_default is not True:
        unsupported.append("default")
    if config.sem_filter_cascade_args is not None:
        unsupported.append("cascade_args")
    if config.sem_filter_safe_mode:
        unsupported.append("safe_mode")
    if config.sem_filter_additional_cot_instructions:
        unsupported.append("additional_cot_instructions")
    if unsupported:
        raise ValueError(
            "batch-prompting sem_filter does not support LOTUS options: "
            + ", ".join(unsupported)
        )


def _build_tasks(
    docs: Sequence[Mapping[str, Any]],
) -> tuple[_BatchPromptingTask, ...]:
    return tuple(
        _BatchPromptingTask(
            position=position,
            row_id=f"row_{position}",
            context=str(doc.get("text", "")),
        )
        for position, doc in enumerate(docs)
    )


def _render_claim(instruction: str, columns: Sequence[str]) -> str:
    """Render LOTUS placeholders without treating side labels as format specs."""

    claim = instruction
    for column in columns:
        claim = claim.replace(f"{{{column}}}", column.capitalize())
    return claim


def _build_request(
    tasks: tuple[_BatchPromptingTask, ...],
    *,
    claim: str,
    max_tokens: int,
) -> PromptBatchRequest:
    payload = {
        "claim": claim,
        "rows": [
            {"row_id": task.row_id, "context": task.context} for task in tasks
        ],
        "output_schema": {"decisions": [{"row_id": "row_id", "keep": True}]},
    }
    return PromptBatchRequest(
        task_ids=tuple(task.row_id for task in tasks),
        prompt=[
            {"role": "system", "content": SEM_FILTER_BATCH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ],
        max_tokens=max_tokens,
    )


def _parse_decisions(
    raw_output: str,
) -> ParsedPromptBatch[bool]:
    from agent_memory.adapters.lotus.structured import (
        load_structured_json_with_syntax_repair,
    )

    if not raw_output.strip():
        raise ValueError("batch-prompting sem_filter output is empty")
    decoded = load_structured_json_with_syntax_repair(
        raw_output,
        operator="batch-prompting sem_filter",
        expected_shape='JSON object with a "decisions" array',
    )
    payload = decoded.value
    if not isinstance(payload, dict) or set(payload) != {"decisions"}:
        raise ValueError(
            "batch-prompting sem_filter output must contain only decisions"
        )
    raw_decisions = payload["decisions"]
    if not isinstance(raw_decisions, list):
        raise ValueError("batch-prompting sem_filter decisions must be a list")

    parsed: list[PromptBatchItem[bool]] = []
    for decision in raw_decisions:
        if not isinstance(decision, dict) or set(decision) != {"row_id", "keep"}:
            raise ValueError(
                "batch-prompting sem_filter decisions require row_id and keep"
            )
        row_id = decision["row_id"]
        keep = decision["keep"]
        if not isinstance(row_id, str):
            raise ValueError("batch-prompting sem_filter row_id must be a string")
        if type(keep) is not bool:
            raise ValueError("batch-prompting sem_filter keep must be a boolean")
        parsed.append(PromptBatchItem(row_id, keep))
    return ParsedPromptBatch(
        items=tuple(parsed),
        repair_method=decoded.repair_method,
    )


__all__ = [
    "BatchPromptingResult",
    "execute_batch_prompted_predicate",
    "execute_batch_prompted_sem_filter",
    "validate_batch_prompting_sem_filter_config",
]
