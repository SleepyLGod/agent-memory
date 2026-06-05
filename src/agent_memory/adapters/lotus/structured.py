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

from agent_memory.adapters.lotus.context import DEFAULT_STRUCTURED_MAX_TOKENS
from agent_memory.logical import ColumnSpec, QueryExpr

EXPLANATION_FIELD = "_explanation"
FLAT_MAP_ROWS_FIELD = "rows"
STRUCTURED_RESERVED_MODEL_KWARGS = {"progress_bar_desc", "response_format"}
RAW_OUTPUT_PREVIEW_CHARS = 240
STRUCTURED_MAX_RETRIES = 3
STRUCTURED_FAILURE_DIR = Path(".memory-test") / "structured-failures" / "latest"
PLACEHOLDER_PATTERN = re.compile(
    r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)(?::(left|right))?\}(?!\})"
)


@dataclass(frozen=True)
class StructuredGenerationResult:
    """Structured LM outputs before DataFrame-specific writeback."""

    parsed_outputs: Sequence[Any]
    raw_outputs: Sequence[str]
    explanations: Sequence[str | None]


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


def _load_structured_json(raw_output: str, *, operator: str, expected_shape: str) -> Any:
    """Parse JSON with concise operator diagnostics."""

    try:
        return json.loads(raw_output)
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
        lm_kwargs: dict[str, Any] = {
            "progress_bar_desc": progress_bar_desc,
            "max_tokens": max(current_max_tokens, structured_max_tokens),
            **dict(model_kwargs),
            "response_format": {"type": "json_object"},
        }
        raw_outputs = execute_structured_lm_with_retries(
            lotus.settings.lm,
            prompts,
            lm_kwargs=lm_kwargs,
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation and shape == "object",
            operator=operator,
            max_retries=STRUCTURED_MAX_RETRIES,
        )
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
        first = invalid[0]
        parse_error = structured_parse_error(
            raw_outputs[first],
            output_cols=output_cols,
            shape=shape,
            require_explanation=require_explanation,
            operator=operator,
        )
        raise ValueError(
            f"{parse_error}; structured failure artifact: {artifact_paths[0]}"
        )

    return raw_outputs


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
