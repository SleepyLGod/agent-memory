"""Shared execution mechanics for multi-task semantic prompts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter
from typing import Any, Generic, TypeVar

from agent_memory.tracing.semantic import write_trace_event


TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class PromptBatching:
    """Bound the ready semantic tasks placed in one prompt."""

    max_tasks: int | None = None

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

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
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


def run_prompt_batches(
    tasks: Sequence[TaskT],
    *,
    task_id: Callable[[TaskT], str],
    build_request: Callable[[tuple[TaskT, ...]], PromptBatchRequest],
    parse_results: Callable[[str], ParsedPromptBatch[ResultT]],
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

    requests = _build_requests(
        task_sequence,
        task_ids=task_ids,
        build_request=build_request,
        model=model,
        config=config,
    )
    pending = list(requests)
    outputs: dict[str, ResultT] = {}
    raw_output_attempts = {identifier: [] for identifier in task_ids}
    repair_methods: dict[str, str | None] = {}
    prompt_count = 0
    retry_count = 0
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
        model_output = model(
            [request.prompt for request in pending],
            progress_bar_desc=progress_bar_desc,
            max_tokens=max(request.max_tokens for request in pending),
            **kwargs,
            response_format={"type": "json_object"},
        )
        raw_outputs = [str(value) for value in getattr(model_output, "outputs", ())]
        if len(raw_outputs) != len(pending):
            raise ValueError(
                "prompt batching returned an unexpected number of outputs: "
                f"expected {len(pending)}, got {len(raw_outputs)}"
            )

        retry: list[PromptBatchRequest] = []
        for request, raw_output in zip(pending, raw_outputs, strict=True):
            for identifier in request.task_ids:
                raw_output_attempts[identifier].append(raw_output)
            try:
                parsed = parse_results(raw_output)
                request_outputs = _validate_parsed_items(
                    parsed.items,
                    expected_ids=request.task_ids,
                )
            except ValueError as error:
                last_error = error
                retry.append(request)
                continue
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
    write_trace_event(
        trace_dir,
        operator=operator,
        event_type="prompt_batching",
        payload={
            "prompt_batching": config.to_dict(),
            "task_count": task_count,
            "prompt_count": result.prompt_count,
            "chunk_sizes": list(result.chunk_sizes),
            "retry_count": result.retry_count,
            "syntax_repair_count": sum(
                method is not None for method in result.repair_methods
            ),
            "syntax_repair_methods": repair_methods,
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
) -> tuple[PromptBatchRequest, ...]:
    requests: list[PromptBatchRequest] = []
    start = 0
    while start < len(tasks):
        limit = config.max_tasks or len(tasks)
        stop = min(start + limit, len(tasks))
        accepted: PromptBatchRequest | None = None
        while stop > start:
            request_tasks = tasks[start:stop]
            request = build_request(request_tasks)
            expected_ids = task_ids[start:stop]
            if request.task_ids != expected_ids:
                raise ValueError(
                    "prompt batch request task IDs must preserve task order"
                )
            if request.max_tokens < 1:
                raise ValueError("prompt batch request max_tokens must be positive")
            if _request_fits_context(request, model=model):
                accepted = request
                break
            stop -= 1
        if accepted is None:
            raise ValueError(
                "prompt batching task does not fit the model context; "
                f"task_id={task_ids[start]!r}"
            )
        requests.append(accepted)
        start = stop
    return tuple(requests)


def _request_fits_context(request: PromptBatchRequest, *, model: Any) -> bool:
    max_ctx_len = getattr(model, "max_ctx_len", None)
    if max_ctx_len is None:
        return True
    prompt_tokens = int(model.count_tokens(request.prompt))
    return prompt_tokens + request.max_tokens <= int(max_ctx_len)


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
