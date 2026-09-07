"""Shared execution mechanics for multi-task semantic prompts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from time import perf_counter
from typing import Any, Generic, TypeVar

from agent_memory.adapters.lotus.json_output import (
    JSON_REPAIR_VERSION,
    repair_json_structure,
)
from agent_memory.tracing.semantic import write_trace_event


TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")
STRUCTURED_OUTPUT_TRANSPORTS = ("chat-json-object", "responses-json-schema")


def validate_structured_output_transport(
    transport: str, *, model: str | None = None
) -> None:
    """Validate an explicitly selected physical structured-output path."""

    if transport not in STRUCTURED_OUTPUT_TRANSPORTS:
        raise ValueError(f"unsupported structured_output_transport: {transport!r}")
    if transport == "responses-json-schema" and model is not None:
        from agent_memory.adapters.lotus.deepseek_responses_lm import (
            validate_deepseek_responses_model,
        )

        validate_deepseek_responses_model(model)


@dataclass(frozen=True)
class PromptBatching:
    """Bound the ready semantic tasks placed in one prompt."""

    max_tasks: int | None = None
    repair_version: str = field(default=JSON_REPAIR_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the optional task-count limit."""

        if self.max_tasks is None:
            return
        if not isinstance(self.max_tasks, int) or isinstance(self.max_tasks, bool):
            raise TypeError("PromptBatching.max_tasks must be an integer or None")
        if self.max_tasks < 1:
            raise ValueError("PromptBatching.max_tasks must be at least 1")

    def to_dict(self) -> dict[str, int | None]:
        """Return the canonical physical execution contract."""

        return {"max_tasks": self.max_tasks}

    @property
    def fingerprint(self) -> str:
        """Return a stable identity for checkpoint isolation."""

        payload = json.dumps(
            {**self.to_dict(), "repair_version": self.repair_version},
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def parse_prompt_batch_size(value: str) -> PromptBatching:
    """Parse the benchmark CLI's ``all`` or positive-integer contract."""

    normalized = value.strip().lower()
    if normalized == "all":
        return PromptBatching()
    try:
        max_tasks = int(normalized)
    except ValueError as error:
        raise ValueError("expected 'all' or a positive integer") from error
    try:
        return PromptBatching(max_tasks=max_tasks)
    except (TypeError, ValueError) as error:
        raise ValueError("expected 'all' or a positive integer") from error


@dataclass(frozen=True)
class PromptBatchRequest:
    """One model request built from a stable sequence of semantic tasks."""

    task_ids: tuple[str, ...]
    prompt: list[dict[str, str]]
    max_tokens: int


@dataclass(frozen=True)
class PromptBatchItem(Generic[ResultT]):
    """One task result returned by an operator-specific parser."""

    task_id: str
    value: ResultT


@dataclass(frozen=True)
class ParsedPromptBatch(Generic[ResultT]):
    """Parsed task results plus optional syntax-repair provenance."""

    items: tuple[PromptBatchItem[ResultT], ...]
    repair_method: str | None = None


@dataclass(frozen=True)
class PromptBatchRunResult(Generic[ResultT]):
    """Task outputs and physical prompt execution accounting."""

    outputs: tuple[ResultT, ...]
    raw_output_attempts: tuple[tuple[str, ...], ...]
    repair_methods: tuple[str | None, ...]
    prompt_count: int
    retry_count: int
    chunk_sizes: tuple[int, ...]
    output_token_limit: int | None = None
    repairs: tuple[dict[str, Any], ...] = ()


def run_prompt_batches(
    tasks: Sequence[TaskT],
    *,
    task_id: Callable[[TaskT], str],
    build_request: Callable[[tuple[TaskT, ...]], PromptBatchRequest],
    parse_results: Callable[[str], ParsedPromptBatch[ResultT]],
    parse_single_result: Callable[[str], ResultT] | None = None,
    parse_repaired_results: Callable[[str], ParsedPromptBatch[ResultT]] | None = None,
    output_schema: Mapping[str, Any] | None = None,
    structured_output_transport: str = "chat-json-object",
    model: Any,
    config: PromptBatching,
    max_retries: int,
    progress_bar_desc: str,
    operator: str,
    trace_dir: Any = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> PromptBatchRunResult[ResultT]:
    """Run ready semantic tasks in deterministic, context-bounded prompts."""

    started = perf_counter()
    validate_structured_output_transport(structured_output_transport)
    if structured_output_transport == "responses-json-schema" and output_schema is None:
        raise ValueError("responses-json-schema requires an operator output schema")
    response_format: dict[str, Any] = {"type": "json_object"}
    if structured_output_transport == "responses-json-schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_batch",
                "schema": dict(output_schema or {}),
            },
        }
    if max_retries < 0:
        raise ValueError("prompt batch max_retries cannot be negative")

    task_sequence = tuple(tasks)
    task_ids = tuple(task_id(task) for task in task_sequence)
    _validate_unique_task_ids(task_ids)
    if not task_sequence:
        result = PromptBatchRunResult((), (), (), 0, 0, ())
        _write_prompt_batch_trace(
            trace_dir,
            operator=operator,
            config=config,
            task_count=0,
            result=result,
            latency_ms=(perf_counter() - started) * 1000,
        )
        return result

    try:
        requests = _build_requests(
            task_sequence,
            task_ids=task_ids,
            build_request=build_request,
            model=model,
            config=config,
            output_schema=(
                output_schema
                if structured_output_transport == "responses-json-schema"
                else None
            ),
        )
    except ValueError as error:
        _write_batch_failure(
            trace_dir,
            operator=operator,
            attempt=0,
            task_count=len(task_sequence),
            error=error,
            phase="preflight",
        )
        raise
    request_max_tokens = _fixed_output_token_limit(requests)
    pending = list(requests)
    outputs: dict[str, ResultT] = {}
    raw_output_attempts = {identifier: [] for identifier in task_ids}
    repair_methods: dict[str, str | None] = {}
    prompt_count = 0
    retry_count = 0
    repairs: list[dict[str, Any]] = []
    last_error: ValueError | None = None
    kwargs = dict(model_kwargs or {})
    conflicts = sorted({"max_tokens", "progress_bar_desc", "response_format"} & kwargs.keys())
    if conflicts:
        raise ValueError(
            "prompt batching model_kwargs cannot override runner options: "
            f"{conflicts}"
        )

    for _attempt in range(max_retries + 1):
        if not pending:
            break
        prompt_count += len(pending)
        try:
            model_output = model(
                [request.prompt for request in pending],
                progress_bar_desc=progress_bar_desc,
                max_tokens=request_max_tokens,
                **kwargs,
                response_format=response_format,
            )
        except Exception as error:
            _write_batch_failure(
                trace_dir,
                operator=operator,
                attempt=_attempt + 1,
                task_count=sum(len(request.task_ids) for request in pending),
                error=error,
                phase="provider",
            )
            raise
        raw_outputs = [str(value) for value in getattr(model_output, "outputs", ())]
        if len(raw_outputs) != len(pending):
            raise ValueError(
                "prompt batching returned an unexpected number of outputs: "
                f"expected {len(pending)}, got {len(raw_outputs)}"
            )

        retry: list[PromptBatchRequest] = []
        metadata = getattr(model_output, "response_metadata", None)
        for index, (request, raw_output) in enumerate(
            zip(pending, raw_outputs, strict=True)
        ):
            for identifier in request.task_ids:
                raw_output_attempts[identifier].append(raw_output)
            completion = metadata[index] if metadata is not None else {}
            try:
                if _incomplete(completion):
                    raise ValueError(f"incomplete structured response: {completion}")
                parsed = parse_results(raw_output)
                request_outputs = _validate_parsed_items(
                    parsed.items,
                    expected_ids=request.task_ids,
                )
            except ValueError as error:
                try:
                    if _incomplete(completion):
                        raise error
                    parsed, edits = _repair_batch(
                        raw_output,
                        expected_ids=request.task_ids,
                        parser=parse_repaired_results or parse_results,
                        singleton_parser=parse_single_result,
                    )
                    request_outputs = _validate_parsed_items(
                        parsed.items, expected_ids=request.task_ids
                    )
                except ValueError as repair_error:
                    last_error = error
                    retry.append(request)
                    _write_batch_failure(
                        trace_dir,
                        operator=operator,
                        attempt=_attempt + 1,
                        task_count=len(request.task_ids),
                        error=error,
                        phase="parse",
                        repair_error=repair_error,
                    )
                    continue
                repairs.append(
                    {
                        "method": parsed.repair_method,
                        "version": JSON_REPAIR_VERSION,
                        "attempt": _attempt + 1,
                        "affected_task_count": len(request.task_ids),
                        "edits": edits,
                    }
                )
            else:
                if parsed.repair_method is not None:
                    repairs.append(
                        {
                            "method": parsed.repair_method,
                            "version": JSON_REPAIR_VERSION,
                            "attempt": _attempt + 1,
                            "affected_task_count": len(request.task_ids),
                            "edits": (),
                        }
                    )
            outputs.update(request_outputs)
            repair_methods.update(
                dict.fromkeys(request.task_ids, parsed.repair_method)
            )
        retry_count += len(retry)
        pending = retry

    if pending:
        assert last_error is not None
        raise ValueError(
            "prompt batching returned invalid structured output after "
            f"{max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    result = PromptBatchRunResult(
        outputs=tuple(outputs[identifier] for identifier in task_ids),
        raw_output_attempts=tuple(
            tuple(raw_output_attempts[identifier]) for identifier in task_ids
        ),
        repair_methods=tuple(repair_methods[identifier] for identifier in task_ids),
        prompt_count=prompt_count,
        retry_count=retry_count,
        chunk_sizes=tuple(len(request.task_ids) for request in requests),
        output_token_limit=request_max_tokens,
        repairs=tuple(repairs),
    )
    _write_prompt_batch_trace(
        trace_dir,
        operator=operator,
        config=config,
        task_count=len(task_sequence),
        result=result,
        latency_ms=(perf_counter() - started) * 1000,
    )
    return result


def _repair_batch(
    raw: str,
    *,
    expected_ids: tuple[str, ...],
    parser: Callable[[str], ParsedPromptBatch[ResultT]],
    singleton_parser: Callable[[str], ResultT] | None,
) -> tuple[ParsedPromptBatch[ResultT], tuple[tuple[str, int, str], ...]]:
    if len(expected_ids) == 1 and singleton_parser is not None:
        try:
            value = singleton_parser(raw)
        except ValueError:
            pass
        else:
            return ParsedPromptBatch(
                (PromptBatchItem(expected_ids[0], value),), "singleton-envelope"
            ), ()

    def validate(candidate: str) -> ParsedPromptBatch[ResultT]:
        parsed = parser(candidate)
        _validate_parsed_items(parsed.items, expected_ids=expected_ids)
        return parsed

    repaired = repair_json_structure(raw, validator=validate)
    return replace(repaired.value, repair_method="bounded-json-syntax"), repaired.edits


def _write_batch_failure(
    trace_dir: Any,
    *,
    operator: str,
    attempt: int,
    task_count: int,
    error: Exception,
    phase: str,
    repair_error: ValueError | None = None,
) -> None:
    try:
        write_trace_event(
            trace_dir,
            operator=operator,
            event_type="prompt_batching_failure",
            payload={
                "status": "error",
                "phase": phase,
                "attempt": attempt,
                "task_count": task_count,
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "repair_result": "not-accepted",
                "repair_error": str(repair_error)[:1000] if repair_error else None,
                "repair_version": JSON_REPAIR_VERSION,
            },
        )
    except Exception as trace_error:
        error.add_note(f"Prompt batching failure trace also failed: {trace_error}")


def _incomplete(metadata: Mapping[str, Any]) -> bool:
    return metadata.get("finish_reason") in {
        "length",
        "content_filter",
    } or metadata.get("status") in {"incomplete", "failed", "cancelled"}


def _write_prompt_batch_trace(
    trace_dir: Any,
    *,
    operator: str,
    config: PromptBatching,
    task_count: int,
    result: PromptBatchRunResult[Any],
    latency_ms: float,
) -> None:
    """Record compact mechanics; provider usage remains the token source of truth."""

    repair_methods = sorted(
        {method for method in result.repair_methods if method is not None}
    )
    syntax_repair_methods = [
        method for method in repair_methods if method != "singleton-envelope"
    ]
    write_trace_event(
        trace_dir,
        operator=operator,
        event_type="prompt_batching",
        payload={
            "prompt_batching": config.to_dict(),
            "task_count": task_count,
            "prompt_count": result.prompt_count,
            "chunk_sizes": list(result.chunk_sizes),
            "structured_output_token_limit": result.output_token_limit,
            "retry_count": result.retry_count,
            "structured_output_repair_count": len(result.repairs),
            "structured_output_repaired_task_count": sum(
                method is not None for method in result.repair_methods
            ),
            "structured_output_repairs": list(result.repairs),
            "structured_output_repair_version": JSON_REPAIR_VERSION,
            "structured_output_repair_methods": repair_methods,
            "syntax_repair_count": sum(
                repair["method"] != "singleton-envelope" for repair in result.repairs
            ),
            "syntax_repair_methods": syntax_repair_methods,
            "prompt_batching_latency_ms": latency_ms,
        },
    )


def _build_requests(
    tasks: tuple[TaskT, ...],
    *,
    task_ids: tuple[str, ...],
    build_request: Callable[[tuple[TaskT, ...]], PromptBatchRequest],
    model: Any,
    config: PromptBatching,
    output_schema: Mapping[str, Any] | None = None,
) -> tuple[PromptBatchRequest, ...]:
    requests: list[PromptBatchRequest] = []
    limit = config.max_tasks or len(tasks)
    for start in range(0, len(tasks), limit):
        stop = min(start + limit, len(tasks))
        request = build_request(tasks[start:stop])
        expected_ids = task_ids[start:stop]
        if request.task_ids != expected_ids:
            raise ValueError("prompt batch request task IDs must preserve task order")
        if request.max_tokens < 1:
            raise ValueError("prompt batch request max_tokens must be positive")
        max_ctx_len = getattr(model, "max_ctx_len", None)
        if max_ctx_len is None:
            requests.append(request)
            continue
        prompt_tokens = int(model.count_tokens(request.prompt))
        schema_tokens = (
            int(
                model.count_tokens(
                    [{"role": "user", "content": json.dumps(output_schema)}]
                )
            )
            if output_schema is not None
            else 0
        )
        if prompt_tokens + schema_tokens + request.max_tokens > int(max_ctx_len):
            raise ValueError(
                "configured prompt batch does not fit the model context; "
                f"batch_size={len(expected_ids)}, "
                f"estimated_input_tokens={prompt_tokens}, estimated_schema_tokens={schema_tokens}, "
                f"reserved_output_tokens={request.max_tokens}, context_limit={max_ctx_len}, "
                f"first_task_id={expected_ids[0]!r}, "
                f"last_task_id={expected_ids[-1]!r}"
            )
        requests.append(request)
    return tuple(requests)


def _fixed_output_token_limit(requests: Sequence[PromptBatchRequest]) -> int:
    """Return the fixed per-request output ceiling for one batch run."""

    limits = {request.max_tokens for request in requests}
    if len(limits) != 1:
        raise ValueError(
            "all prompt batch requests must use the same output token limit"
        )
    return next(iter(limits))


def _validate_unique_task_ids(task_ids: Sequence[str]) -> None:
    seen: set[str] = set()
    for identifier in task_ids:
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("prompt batching task IDs must be non-empty strings")
        if identifier in seen:
            raise ValueError(f"prompt batching input contains duplicate task ID {identifier!r}")
        seen.add(identifier)


def _validate_parsed_items(
    items: Sequence[PromptBatchItem[ResultT]],
    *,
    expected_ids: Sequence[str],
) -> dict[str, ResultT]:
    parsed: dict[str, ResultT] = {}
    for item in items:
        if item.task_id in parsed:
            raise ValueError(
                f"prompt batching returned duplicate task ID {item.task_id!r}"
            )
        parsed[item.task_id] = item.value

    expected = set(expected_ids)
    unknown = sorted(set(parsed) - expected)
    missing = sorted(expected - set(parsed))
    if unknown:
        raise ValueError(f"prompt batching returned unknown task IDs: {unknown}")
    if missing:
        raise ValueError(
            f"prompt batching omitted or returned missing task IDs: {missing}"
        )
    return parsed


__all__ = [
    "ParsedPromptBatch",
    "PromptBatchItem",
    "PromptBatchRequest",
    "PromptBatchRunResult",
    "PromptBatching",
    "parse_prompt_batch_size",
    "run_prompt_batches",
]
