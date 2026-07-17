"""Shared structured LOTUS generation helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from lotus.cache import operator_cache

from agent_memory.adapters.lotus.context import (
    DEFAULT_STRUCTURED_MAX_TOKENS,
    DEFAULT_STRUCTURED_PARSE_RETRIES,
)
from agent_memory.tracing.semantic import write_structured_generation_trace
from agent_memory.logical import ColumnSpec, QueryExpr

EXPLANATION_FIELD = "_explanation"
FLAT_MAP_ROWS_FIELD = "rows"
STRUCTURED_RESERVED_MODEL_KWARGS = {"progress_bar_desc", "response_format"}
RAW_OUTPUT_PREVIEW_CHARS = 240
STRUCTURED_FAILURE_DIR = Path(".memory-test") / "structured-failures" / "latest"
MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
PLACEHOLDER_PATTERN = re.compile(
    r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)(?::(left|right))?\}(?!\})"
)


@dataclass
class StructuredRetryStats:
    """Process-local counters for custom structured JSON retry observability."""

    retry_batches: int = 0
    retry_rows: int = 0
    failure_artifacts: int = 0


_STRUCTURED_RETRY_STATS = StructuredRetryStats()


def structured_retry_stats() -> StructuredRetryStats:
    """Return current process-local structured retry counters."""

    return StructuredRetryStats(
        retry_batches=_STRUCTURED_RETRY_STATS.retry_batches,
        retry_rows=_STRUCTURED_RETRY_STATS.retry_rows,
        failure_artifacts=_STRUCTURED_RETRY_STATS.failure_artifacts,
    )


def reset_structured_retry_stats() -> None:
    """Reset process-local structured retry counters."""

    _STRUCTURED_RETRY_STATS.retry_batches = 0
    _STRUCTURED_RETRY_STATS.retry_rows = 0
    _STRUCTURED_RETRY_STATS.failure_artifacts = 0


@dataclass(frozen=True)
class StructuredGenerationResult:
    """Structured LM outputs before DataFrame-specific writeback."""

    parsed_outputs: Sequence[Any]
    raw_outputs: Sequence[str]
    explanations: Sequence[str | None]
    raw_output_attempts: Sequence[Sequence[str]] = ()


@dataclass(frozen=True)
class StructuredLMRetryResult:
    """Raw structured LM outputs plus retry and validation metadata."""

    raw_outputs: Sequence[str]
    raw_output_attempts: Sequence[Sequence[str]]
    invalid_indices: Sequence[int]
    failure_artifact_paths: Sequence[Path]


def normalize_strategy(strategy: Any) -> Any:
    """Normalize string reasoning strategy names to LOTUS ReasoningStrategy."""

    if strategy is None:
        return None

    from lotus.types import ReasoningStrategy

    if isinstance(strategy, ReasoningStrategy):
        return strategy
    if isinstance(strategy, str):
        key = strategy.upper().replace("-", "_")
        try:
            return ReasoningStrategy[key]
        except KeyError as error:
            raise ValueError(f"Unknown LOTUS reasoning strategy {strategy!r}") from error
    return strategy


def strategy_requests_explanation(strategy: Any) -> bool:
    """Return whether a LOTUS reasoning strategy should emit explanations."""

    normalized = normalize_strategy(strategy)
    if normalized is None:
        return False

    from lotus.types import ReasoningStrategy

    return normalized in {ReasoningStrategy.COT, ReasoningStrategy.ZS_COT}


def validate_model_kwargs(
    model_kwargs: Mapping[str, Any],
    *,
    reserved: set[str],
    operator: str,
) -> None:
    """Reject model kwargs that would override lowering-owned execution args."""

    conflicts = sorted(set(model_kwargs).intersection(reserved))
    if conflicts:
        raise ValueError(f"{operator} model_kwargs cannot override adapter kwargs: {conflicts}")


def output_columns(query: QueryExpr, *, operator: str) -> tuple[ColumnSpec, ...]:
    """Return required output columns for a structured semantic operator."""

    output_cols = query.params.get("output_cols")
    if not output_cols:
        raise ValueError(f"{operator} requires output_cols")
    return tuple(output_cols)


def resolve_input_cols(source: Any, query: QueryExpr, *, operator: str) -> tuple[str, ...]:
    """Resolve input columns for structured LOTUS lowering."""

    explicit = query.params.get("input_cols")
    if explicit is not None:
        columns = tuple(str(column) for column in explicit)
    else:
        import lotus

        try:
            parsed = tuple(
                column
                for column in lotus.nl_expression.parse_cols(
                    str(query.params["instruction"])
                )
                if column in source.columns
            )
        except ValueError:
            parsed = ()
        columns = parsed or tuple(
            str(column) for column in getattr(source, "columns", ())
        )

    if not columns:
        raise ValueError(f"{operator} requires at least one input column")

    missing = [column for column in columns if column not in source.columns]
    if missing:
        raise ValueError(f"{operator} input columns not found in DataFrame: {missing}")
    return columns


def examples_dataframe(examples: Any) -> pd.DataFrame | None:
    """Convert row-like public examples into a LOTUS examples DataFrame."""

    if examples is None:
        return None
    return pd.DataFrame([dict(row) for row in examples])


def structured_instruction(
    instruction: str,
    output_cols: Sequence[ColumnSpec],
    *,
    shape: Literal["object", "array"],
    require_explanation: bool = False,
) -> str:
    """Append a structured JSON output contract to an operator instruction."""

    schema = {
        column.name: column.description or column.name
        for column in output_cols
    }
    fields = [f'"{column.name}"' for column in output_cols]
    if require_explanation:
        fields.append(f'"{EXPLANATION_FIELD}"')
        schema[EXPLANATION_FIELD] = "Brief explanation for the generated fields."

    field_text = ", ".join(fields)
    schema_text = json.dumps(schema, ensure_ascii=False)
    object_example = json.dumps(
        {column.name: "string" for column in output_cols},
        ensure_ascii=False,
    )
    if shape == "object":
        return (
            f"{instruction}\n\n"
            "Return only a valid JSON object. Do not include markdown fences, "
            "comments, or extra prose.\n"
            f"The JSON object must include these fields: {field_text}.\n"
            f"Field descriptions: {schema_text}.\n"
            f"Output shape example: {object_example}.\n"
            "Use string values for every field."
        )

    rows_example = json.dumps(
        {FLAT_MAP_ROWS_FIELD: [{column.name: "string" for column in output_cols}]},
        ensure_ascii=False,
    )
    return (
        f"{instruction}\n\n"
        "Return only a valid JSON object. Do not include markdown fences, "
        "comments, or extra prose.\n"
        f'The JSON object must include a "{FLAT_MAP_ROWS_FIELD}" field containing '
        "an array of output row objects. Return an empty rows array when there "
        "are no output rows.\n"
        f"Every object in rows must include these fields: {field_text}.\n"
        f"Field descriptions: {schema_text}.\n"
        f"Output shape example: {rows_example}.\n"
        "Use string values for every field."
    )


def escape_structured_formatter_placeholders(
    instruction: str,
    *,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
) -> str:
    """Escape placeholders that Python str.format must not treat as input keys."""

    input_names = {str(column) for column in input_cols}
    output_names = {column.name for column in output_cols}

    def replace(match: re.Match[str]) -> str:
        column = match.group(1)
        side = match.group(2)
        if side is not None:
            return f"{{{{{column}:{side}}}}}"
        if column in output_names and column not in input_names:
            return f"{{{{{column}}}}}"
        return match.group(0)

    return PLACEHOLDER_PATTERN.sub(replace, instruction)


def parse_structured_object_json(
    raw_output: str,
    output_cols: Sequence[ColumnSpec],
    *,
    require_explanation: bool = False,
    operator: str = "sem_map",
) -> tuple[dict[str, str], str | None]:
    """Parse one JSON object output and return columns plus optional explanation."""

    parsed = _load_structured_json(
        raw_output,
        operator=operator,
        expected_shape="JSON object",
    )

    if not isinstance(parsed, Mapping):
        raise ValueError(f"{operator} returned non-object JSON: {raw_output!r}")

    required = [column.name for column in output_cols]
    if require_explanation:
        required.append(EXPLANATION_FIELD)
    missing = [name for name in required if name not in parsed]
    if missing:
        raise ValueError(f"{operator} JSON output is missing required keys: {missing}")

    output = {column.name: str(parsed[column.name]) for column in output_cols}
    explanation = str(parsed[EXPLANATION_FIELD]) if require_explanation else None
    return output, explanation


def parse_structured_array_json(
    raw_output: str,
    output_cols: Sequence[ColumnSpec],
    *,
    operator: str = "sem_flat_map",
) -> list[dict[str, str]]:
    """Parse one JSON array output for flat-map style row expansion."""

    expected_shape = f'JSON object with "{FLAT_MAP_ROWS_FIELD}" array'
    parsed = _load_structured_json(
        raw_output,
        operator=operator,
        expected_shape=expected_shape,
    )

    if not isinstance(parsed, Mapping):
        raise ValueError(
            f"{operator} returned non-object JSON wrapper; expected {expected_shape}; "
            f"raw_output={_preview_raw_output(raw_output)!r}"
        )
    if FLAT_MAP_ROWS_FIELD not in parsed:
        raise ValueError(
            f"{operator} JSON output is missing required key {FLAT_MAP_ROWS_FIELD!r}; "
            f"expected {expected_shape}; raw_output={_preview_raw_output(raw_output)!r}"
        )
    emitted_rows = parsed[FLAT_MAP_ROWS_FIELD]
    if not isinstance(emitted_rows, list):
        raise ValueError(
            f"{operator} JSON {FLAT_MAP_ROWS_FIELD!r} value is not an array; "
            f"expected {expected_shape}; raw_output={_preview_raw_output(raw_output)!r}"
        )

    rows: list[dict[str, str]] = []
    for index, item in enumerate(emitted_rows):
        if not isinstance(item, Mapping):
            raise ValueError(f"{operator} JSON item {index} is not an object: {item!r}")

        missing = [column.name for column in output_cols if column.name not in item]
        if missing:
            raise ValueError(
                f"{operator} JSON item {index} is missing required keys: {missing}"
            )
        rows.append({column.name: str(item[column.name]) for column in output_cols})
    return rows


def _strip_markdown_fence(raw_output: str) -> str:
    """Strip markdown code fences if present, returning bare JSON string."""
    m = MARKDOWN_FENCE_RE.match(raw_output.strip())
    return m.group(1) if m else raw_output


def _load_structured_json(raw_output: str, *, operator: str, expected_shape: str) -> Any:
    """Parse JSON with concise operator diagnostics."""

    try:
        return json.loads(_strip_markdown_fence(raw_output))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{operator} returned invalid JSON; expected {expected_shape}; "
            f"raw_output={_preview_raw_output(raw_output)!r}"
        ) from error


def _preview_raw_output(raw_output: str) -> str:
    """Return a bounded raw output preview for errors."""

    if len(raw_output) <= RAW_OUTPUT_PREVIEW_CHARS:
        return raw_output
    return raw_output[:RAW_OUTPUT_PREVIEW_CHARS] + "..."


def example_answers(
    examples: Any,
    output_cols: Sequence[ColumnSpec],
    *,
    require_explanation: bool,
) -> tuple[list[dict[str, Any]] | None, list[str] | None]:
    """Split row-like examples into LOTUS multimodal rows and structured answers."""

    if examples is None:
        return None, None

    rows = [dict(row) for row in examples]
    answers: list[str] = []

    for row in rows:
        reasoning = row.pop("Reasoning", None)
        if "Answer" in row:
            answer = row.pop("Answer")
        else:
            answer = {column.name: row.pop(column.name) for column in output_cols if column.name in row}
            missing = [column.name for column in output_cols if column.name not in answer]
            if missing:
                raise ValueError(
                    "structured examples must include Answer or all output columns; "
                    f"missing {missing}"
                )

        if isinstance(answer, Mapping):
            answer_dict = dict(answer)
            if require_explanation and EXPLANATION_FIELD not in answer_dict:
                answer_dict[EXPLANATION_FIELD] = "" if reasoning is None else str(reasoning)
            answers.append(json.dumps(answer_dict, ensure_ascii=False))
        else:
            answers.append(str(answer))

    return rows, answers


@dataclass
class StructuredLMExecutor:
    """Internal executor that reuses LOTUS operator caching without pandas accessors."""

    _obj: Any

    @operator_cache
    def __call__(
        self,
        *,
        input_cols: tuple[str, ...],
        output_cols: tuple[ColumnSpec, ...],
        instruction: str,
        shape: Literal["object", "array"],
        system_prompt: str | None = None,
        examples: Any = None,
        strategy: Any = None,
        safe_mode: bool = False,
        return_explanations: bool = False,
        progress_bar_desc: str,
        model_kwargs: Mapping[str, Any],
        structured_max_tokens: int = DEFAULT_STRUCTURED_MAX_TOKENS,
        structured_parse_retries: int = DEFAULT_STRUCTURED_PARSE_RETRIES,
        semantic_trace_dir: Any = None,
        operator: str = "sem_map",
    ) -> StructuredGenerationResult:
        """Run a structured LOTUS-backed LM batch and parse JSON outputs."""

        import lotus
        from lotus.templates import task_instructions
        from lotus.types import LMOutput
        from lotus.utils import show_safe_mode

        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure "
                "a valid language model using lotus.settings.configure()"
            )

        validate_model_kwargs(
            model_kwargs,
            reserved=STRUCTURED_RESERVED_MODEL_KWARGS,
            operator="structured semantic operator",
        )

        if self._obj.empty:
            return StructuredGenerationResult(
                parsed_outputs=[],
                raw_outputs=[],
                explanations=[],
            )

        require_explanation = return_explanations or strategy_requests_explanation(strategy)
        formatter_instruction = escape_structured_formatter_placeholders(
            instruction,
            input_cols=input_cols,
            output_cols=output_cols,
        )
        formatted_instruction = lotus.nl_expression.nle2str(
            formatter_instruction,
            list(input_cols),
        )
        user_instruction = structured_instruction(
            formatted_instruction,
            output_cols,
            shape=shape,
            require_explanation=require_explanation and shape == "object",
        )
        docs = task_instructions.df2multimodal_info(self._obj, list(input_cols))

        example_rows, answers = example_answers(
            examples,
            output_cols,
            require_explanation=require_explanation and shape == "object",
        )
        examples_multimodal_data = (
            task_instructions.df2multimodal_info(pd.DataFrame(example_rows), list(input_cols))
            if example_rows is not None
            else None
        )

        prompts = [
            task_instructions.map_formatter(
                lotus.settings.lm,
                doc,
                user_instruction,
                examples_multimodal_data=examples_multimodal_data,
                examples_answer=answers,
                cot_reasoning=None,
                strategy=None,
                system_prompt=system_prompt,
            )
            for doc in docs
        ]

        if safe_mode:
            estimated_cost = sum(lotus.settings.lm.count_tokens(prompt) for prompt in prompts)
            show_safe_mode(estimated_cost, len(prompts))

        current_max_tokens = int(getattr(lotus.settings.lm, "max_tokens", 512) or 512)
        effective_max_tokens = max(current_max_tokens, structured_max_tokens)
        lm_kwargs: dict[str, Any] = {
            "progress_bar_desc": progress_bar_desc,
            "max_tokens": effective_max_tokens,
            "max_completion_tokens": effective_max_tokens,
            **dict(model_kwargs),
            "response_format": {"type": "json_object"},
        }
        retry_result = execute_structured_lm_retry_result(
            lotus.settings.lm,
            prompts,
            lm_kwargs=lm_kwargs,
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation and shape == "object",
            operator=operator,
            max_retries=structured_parse_retries,
        )
        audit_rows = structured_generation_audit_rows(
            source=self._obj,
            input_cols=input_cols,
            output_cols=output_cols,
            instruction=instruction,
            formatted_instruction=formatted_instruction,
            final_instruction=user_instruction,
            raw_outputs=retry_result.raw_outputs,
            raw_output_attempts=retry_result.raw_output_attempts,
            shape=shape,
            require_explanation=require_explanation and shape == "object",
            invalid_indices=retry_result.invalid_indices,
            failure_artifact_paths=retry_result.failure_artifact_paths,
            operator=operator,
        )
        write_structured_generation_trace(
            semantic_trace_dir,
            operator=operator,
            rows=audit_rows,
            snapshots={"input": self._obj.loc[:, list(input_cols)].copy()},
        )
        if retry_result.invalid_indices:
            raise_structured_lm_retry_error(
                retry_result,
                output_cols=output_cols,
                shape=shape,
                require_explanation=require_explanation and shape == "object",
                operator=operator,
            )

        raw_outputs = list(retry_result.raw_outputs)
        parsed_outputs, explanations = parse_structured_outputs(
            raw_outputs,
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation and shape == "object",
            operator=operator,
        )

        if safe_mode:
            lotus.settings.lm.print_total_usage()

        return StructuredGenerationResult(
            parsed_outputs=parsed_outputs,
            raw_outputs=raw_outputs,
            explanations=explanations,
            raw_output_attempts=retry_result.raw_output_attempts,
        )


def execute_structured_lm_with_retries(
    model: Any,
    prompts: Sequence[Any],
    *,
    lm_kwargs: Mapping[str, Any],
    output_cols: Sequence[ColumnSpec],
    shape: Literal["object", "array"],
    require_explanation: bool,
    operator: str,
    max_retries: int,
) -> list[str]:
    """Call the LM and retry only rows with invalid structured JSON."""

    result = execute_structured_lm_retry_result(
        model,
        prompts,
        lm_kwargs=lm_kwargs,
        output_cols=output_cols,
        shape=shape,
        require_explanation=require_explanation,
        operator=operator,
        max_retries=max_retries,
    )
    if result.invalid_indices:
        raise_structured_lm_retry_error(
            result,
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation,
            operator=operator,
        )
    return list(result.raw_outputs)


def execute_structured_lm_retry_result(
    model: Any,
    prompts: Sequence[Any],
    *,
    lm_kwargs: Mapping[str, Any],
    output_cols: Sequence[ColumnSpec],
    shape: Literal["object", "array"],
    require_explanation: bool,
    operator: str,
    max_retries: int,
) -> StructuredLMRetryResult:
    """Call the LM and return retry metadata without hiding parse failures."""

    from lotus.types import LMOutput

    output: LMOutput = model(prompts, **dict(lm_kwargs))
    raw_outputs = list(output.outputs)
    raw_output_attempts = [[raw_output] for raw_output in raw_outputs]
    invalid = invalid_structured_output_indices(
        raw_outputs,
        output_cols=output_cols,
        shape=shape,
        require_explanation=require_explanation,
        operator=operator,
    )

    retries_left = max_retries
    while invalid and retries_left > 0:
        retry_prompts = [prompts[index] for index in invalid]
        _STRUCTURED_RETRY_STATS.retry_batches += 1
        _STRUCTURED_RETRY_STATS.retry_rows += len(retry_prompts)
        retry_kwargs = dict(lm_kwargs)
        retry_kwargs["progress_bar_desc"] = (
            f"{lm_kwargs.get('progress_bar_desc', 'Structured generation')} retry"
        )
        retry_output: LMOutput = model(retry_prompts, **retry_kwargs)
        for index, raw_output in zip(invalid, retry_output.outputs):
            raw_outputs[index] = raw_output
            raw_output_attempts[index].append(raw_output)
        retries_left -= 1
        invalid = invalid_structured_output_indices(
            raw_outputs,
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation,
            operator=operator,
        )

    if invalid:
        artifact_paths = write_structured_failure_artifacts(
            prompts,
            raw_output_attempts,
            invalid,
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation,
            operator=operator,
        )
        _STRUCTURED_RETRY_STATS.failure_artifacts += len(artifact_paths)
    else:
        artifact_paths = []

    return StructuredLMRetryResult(
        raw_outputs=raw_outputs,
        raw_output_attempts=tuple(tuple(attempts) for attempts in raw_output_attempts),
        invalid_indices=tuple(invalid),
        failure_artifact_paths=tuple(artifact_paths),
    )


def raise_structured_lm_retry_error(
    result: StructuredLMRetryResult,
    *,
    output_cols: Sequence[ColumnSpec],
    shape: Literal["object", "array"],
    require_explanation: bool,
    operator: str,
) -> None:
    """Raise the original structured parse failure after audit has run."""

    if not result.invalid_indices:
        return

    first = result.invalid_indices[0]
    parse_error = structured_parse_error(
        result.raw_outputs[first],
        output_cols=output_cols,
        shape=shape,
        require_explanation=require_explanation,
        operator=operator,
    )
    artifact = result.failure_artifact_paths[0]
    raise ValueError(f"{parse_error}; structured failure artifact: {artifact}")


def structured_generation_audit_rows(
    *,
    source: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[ColumnSpec],
    instruction: str,
    formatted_instruction: str,
    final_instruction: str,
    raw_outputs: Sequence[str],
    raw_output_attempts: Sequence[Sequence[str]],
    shape: Literal["object", "array"],
    require_explanation: bool,
    invalid_indices: Sequence[int],
    failure_artifact_paths: Sequence[Path],
    operator: str,
) -> list[dict[str, Any]]:
    """Build structured generation audit rows without mutating source data."""

    invalid = set(invalid_indices)
    artifact_by_index = {
        index: str(path)
        for index, path in zip(invalid_indices, failure_artifact_paths)
    }
    input_records = source.loc[:, list(input_cols)].astype(str).to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    for index, raw_output in enumerate(raw_outputs):
        parse_error = ""
        parsed_output: Any = None
        if index in invalid:
            parse_error = structured_parse_error(
                raw_output,
                output_cols=output_cols,
                shape=shape,
                require_explanation=require_explanation,
                operator=operator,
            )
        else:
            parsed, _explanations = parse_structured_outputs(
                [raw_output],
                output_cols=output_cols,
                shape=shape,
                require_explanation=require_explanation,
                operator=operator,
            )
            parsed_output = parsed[0] if parsed else None

        attempts = (
            list(raw_output_attempts[index])
            if index < len(raw_output_attempts)
            else [raw_output]
        )
        rows.append(
            {
                "operator": operator,
                "row_index": index,
                "shape": shape,
                "input_cols": list(input_cols),
                "input_preview": input_records[index] if index < len(input_records) else {},
                "instruction": instruction,
                "formatted_instruction": formatted_instruction,
                "final_instruction": final_instruction,
                "required_output_cols": [
                    {"name": column.name, "description": column.description}
                    for column in output_cols
                ],
                "raw_output": raw_output,
                "raw_output_attempts": attempts,
                "parse_retry_attempts": max(len(attempts) - 1, 0),
                "parsed_output": parsed_output,
                "parse_error": parse_error,
                "failure_artifact": artifact_by_index.get(index, ""),
            }
        )
    return rows


def write_structured_failure_artifacts(
    prompts: Sequence[Any],
    raw_output_attempts: Sequence[Sequence[str]],
    invalid_indices: Sequence[int],
    *,
    output_cols: Sequence[ColumnSpec],
    shape: Literal["object", "array"],
    require_explanation: bool,
    operator: str,
    extra_by_index: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[Path]:
    """Write structured generation failure artifacts for local inspection."""

    STRUCTURED_FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    paths: list[Path] = []
    for index in invalid_indices:
        final_raw_output = raw_output_attempts[index][-1]
        parse_error = structured_parse_error(
            final_raw_output,
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation,
            operator=operator,
        )
        artifact = {
            "operator": operator,
            "shape": shape,
            "row_index": index,
            "expected_output_columns": [
                {
                    "name": column.name,
                    "description": column.description,
                }
                for column in output_cols
            ],
            "require_explanation": require_explanation,
            "parse_error": parse_error,
            "prompt": str(prompts[index]),
            "raw_outputs": list(raw_output_attempts[index]),
        }
        if extra_by_index and index in extra_by_index:
            artifact.update(dict(extra_by_index[index]))
        path = STRUCTURED_FAILURE_DIR / (
            f"{timestamp}-{operator}-row-{index}-{uuid4().hex[:8]}.json"
        )
        path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def structured_parse_error(
    raw_output: str,
    *,
    output_cols: Sequence[ColumnSpec],
    shape: Literal["object", "array"],
    require_explanation: bool,
    operator: str,
) -> str:
    """Return the parser error message for one structured raw output."""

    try:
        parse_one_structured_output(
            raw_output,
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation,
            operator=operator,
        )
    except ValueError as error:
        return str(error)
    return "structured output unexpectedly parsed successfully"


def invalid_structured_output_indices(
    raw_outputs: Sequence[str],
    *,
    output_cols: Sequence[ColumnSpec],
    shape: Literal["object", "array"],
    require_explanation: bool,
    operator: str,
) -> list[int]:
    """Return output indices that fail the structured parser."""

    invalid: list[int] = []
    for index, raw_output in enumerate(raw_outputs):
        try:
            parse_one_structured_output(
                raw_output,
                output_cols=output_cols,
                shape=shape,
                require_explanation=require_explanation,
                operator=operator,
            )
        except ValueError:
            invalid.append(index)
    return invalid


def parse_structured_outputs(
    raw_outputs: Sequence[str],
    *,
    output_cols: Sequence[ColumnSpec],
    shape: Literal["object", "array"],
    require_explanation: bool,
    operator: str,
) -> tuple[list[Any], list[str | None]]:
    """Parse final raw outputs after all allowed retries."""

    parsed_outputs: list[Any] = []
    explanations: list[str | None] = []
    for raw_output in raw_outputs:
        parsed_output, explanation = parse_one_structured_output(
            raw_output,
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation,
            operator=operator,
        )
        parsed_outputs.append(parsed_output)
        explanations.append(explanation)
    return parsed_outputs, explanations


def parse_one_structured_output(
    raw_output: str,
    *,
    output_cols: Sequence[ColumnSpec],
    shape: Literal["object", "array"],
    require_explanation: bool,
    operator: str,
) -> tuple[Any, str | None]:
    """Parse one raw structured output for map or flat-map style shapes."""

    if shape == "object":
        return parse_structured_object_json(
            raw_output,
            output_cols,
            require_explanation=require_explanation,
            operator=operator,
        )
    return parse_structured_array_json(
        raw_output,
        output_cols,
        operator=operator,
    ), None
