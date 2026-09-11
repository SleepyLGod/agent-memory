"""Single-context structured predicates, independent of prompt batching."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.structured import execute_structured_lm_retry_result
from agent_memory.policy.logical import ColumnSpec
from agent_memory.tracing.semantic import write_trace_event


PREDICATE_SCHEMA = {
    "type": "object",
    "properties": {"keep": {"type": "boolean"}},
    "required": ["keep"],
    "additionalProperties": False,
}
PREDICATE_SYSTEM_PROMPT = (
    "The user will provide a claim and relevant context. Determine whether the "
    "claim is true for that context. Return only a JSON object with the boolean "
    "field keep: true when the claim is true, false otherwise."
)


def single_schema_predicate(config: LotusExecutionConfig) -> bool:
    """Select only the explicitly requested non-batching schema path."""

    return (
        config.prompt_batching is None
        and config.structured_output_transport == "responses-json-schema"
    )


def validate_schema_predicate(config: LotusExecutionConfig, operator: str) -> None:
    """Reject unsupported native options rather than silently dropping them."""

    names = ["examples", "strategy", "cascade_args", "safe_mode"]
    if operator == "sem_filter":
        names += ["helper_examples", "additional_cot_instructions"]
    unsupported = []
    for name in names:
        value = getattr(config, f"{operator}_{name}")
        if (
            value is not None
            and value is not False
            and not (isinstance(value, str) and value == "")
        ):
            unsupported.append(name)
    default = getattr(config, f"{operator}_default")
    if default != (operator == "sem_filter"):
        unsupported.append("default")
    if unsupported:
        raise ValueError(
            f"single-task schema {operator} does not support: {', '.join(unsupported)}"
        )


def _validate_output(raw: str) -> None:
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or set(value) != {"keep"}
        or type(value["keep"]) is not bool
    ):
        raise ValueError(
            'predicate output must contain exactly one boolean field "keep"'
        )


@dataclass(frozen=True)
class PredicateResult:
    """Aligned decisions and raw attempts for existing operator trace assembly."""

    decisions: tuple[bool, ...]
    raw_outputs: tuple[str, ...]


def execute_schema_predicate(
    source: pd.DataFrame,
    *,
    instruction: str,
    config: LotusExecutionConfig,
    operator: str,
) -> PredicateResult:
    """Evaluate one context per prompt using the shared structured retry executor."""

    import lotus
    from lotus.nl_expression import parse_cols
    from lotus.templates import task_instructions

    validate_schema_predicate(config, operator)
    columns = list(parse_cols(instruction))
    if not columns or any(column not in source.columns for column in columns):
        raise ValueError(
            "schema predicate requires valid instruction column placeholders"
        )
    if source.empty:
        return PredicateResult((), ())
    docs = task_instructions.df2multimodal_info(source, columns)
    if any(doc.get("image") for doc in docs):
        raise NotImplementedError("single-task schema predicates support text only")
    prompts = [
        [
            {"role": "system", "content": PREDICATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{doc['text']}\n\nClaim: {instruction}",
            },
        ]
        for doc in docs
    ]
    payload: dict[str, Any] = {
        "instruction": instruction,
        "structured_output_transport": config.structured_output_transport,
        "predicate_prompt_mode": "single-task",
        "tasks_per_prompt": 1,
        "prompt_count": len(prompts),
    }
    try:
        result = execute_structured_lm_retry_result(
            lotus.settings.lm,
            prompts,
            lm_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "predicate",
                        "strict": True,
                        "schema": PREDICATE_SCHEMA,
                    },
                },
                "max_tokens": max(
                    int(getattr(lotus.settings.lm, "max_tokens", 512) or 512),
                    config.structured_max_tokens,
                ),
                "progress_bar_desc": getattr(config, f"{operator}_progress_bar_desc"),
            },
            output_cols=(ColumnSpec("keep", "Whether the claim is true."),),
            shape="object",
            require_explanation=False,
            operator=operator,
            max_retries=config.structured_parse_retries,
            output_validator=_validate_output,
        )
    except Exception as error:
        write_trace_event(
            config.trace_dir(),
            operator=operator,
            event_type="predicate_failure",
            payload={**payload, "error": str(error)},
            raw_output=str(error),
        )
        raise
    write_trace_event(
        config.trace_dir(),
        operator=operator,
        event_type="predicate_generation",
        payload={**payload, "invalid_indices": list(result.invalid_indices)},
        raw_output={"prompts": prompts, "attempts": result.raw_output_attempts},
        parsed_output=[
            None if index in result.invalid_indices else json.loads(raw)["keep"]
            for index, raw in enumerate(result.raw_outputs)
        ],
    )
    if result.invalid_indices:
        raise ValueError(
            f"{operator} schema predicate failed for rows {result.invalid_indices}; artifacts: {result.failure_artifact_paths}"
        )
    return PredicateResult(
        tuple(json.loads(raw)["keep"] for raw in result.raw_outputs),
        tuple(result.raw_outputs),
    )
