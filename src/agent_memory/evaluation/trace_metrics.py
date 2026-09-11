"""Normalize provider traces emitted by the four benchmark systems."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from .pricing import PricingSnapshot

_STANDARD_PHASES = ("insertion", "retrieval", "answering", "grading")
_PHASE_ALIASES = {
    "add": "insertion",
    "extract": "insertion",
    "extraction": "insertion",
    "store": "insertion",
    "answer": "answering",
    "judge": "grading",
}


def normalize_provider_calls(
    events: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    pricing: PricingSnapshot | None = None,
    include_cost: bool = True,
) -> list[dict[str, Any]]:
    """Return one normalized row per provider response.

    LOTUS writes both semantic call metadata and a provider-usage event. The
    provider event is authoritative when both exist, avoiding double counting.
    Responses retain their framework batch identity so reports can distinguish
    logical operator calls, framework batches, and provider calls.
    With include_cost=False, both cost fields remain unknown; no default price
    is applied. Existing callers retain the historical pricing behavior.
    """

    pricing = (pricing or PricingSnapshot.deepseek_2026_07_17()) if include_cost else None
    pending_semantic_calls: dict[
        tuple[str, int], deque[Mapping[str, Any]]
    ] = defaultdict(deque)
    for event in events:
        if event.get("event_type") == "llm_call" and "status" not in event:
            key = _agent_call_key(event, item_key="llm_item_index")
            if key is not None:
                pending_semantic_calls[key].append(event)
    paired_provider_calls: dict[int, Mapping[str, Any]] = {}
    paired_semantic_calls: set[int] = set()
    for event_index, event in enumerate(events):
        if event.get("event_type") != "provider_usage":
            continue
        key = _agent_call_key(event, item_key="provider_item_index")
        if key is not None and pending_semantic_calls[key]:
            semantic_event = pending_semantic_calls[key].popleft()
            paired_provider_calls[event_index] = semantic_event
            paired_semantic_calls.add(id(semantic_event))

    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        event_type = event.get("event_type")
        if event_type == "provider_usage":
            rows.append(
                _agent_provider_row(
                    event,
                    output_dir,
                    pricing,
                    semantic_event=paired_provider_calls.get(event_index),
                )
            )
        elif event_type in {"llm_call_finish", "llm_call_error"}:
            rows.append(_native_claude_row(event, pricing))
        elif event_type == "llm_call" and "status" in event:
            rows.append(_native_graphiti_row(event, pricing))
        elif event_type in {"llm_call", "llm_batch_error"}:
            if int(event.get("llm_item_index") or 0) != 0:
                continue
            if event_type == "llm_call" and id(event) in paired_semantic_calls:
                continue
            rows.append(_agent_runner_row(event, pricing))
    return rows


def summarize_provider_calls(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize normalized calls without replacing unknown usage with guesses."""

    phases: set[str] = set(_STANDARD_PHASES)
    phases.update(str(row.get("phase") or "unknown") for row in rows)
    summary = _summarize_rows(rows)
    summary["phases"] = {
        phase: _summarize_rows(
            [row for row in rows if str(row.get("phase") or "unknown") == phase]
        )
        for phase in sorted(phases, key=_phase_sort_key)
    }
    return summary


def normalize_framework_cache_usage(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one normalized row per cache-observed semantic operation."""

    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") != "framework_cache_usage":
            continue
        rows.append(
            {
                "trace_id": str(event.get("trace_id") or ""),
                "logical_call_id": str(event.get("operator_call_id") or ""),
                "case_id": str(event.get("case_id") or ""),
                "event_id": str(event.get("event_id") or ""),
                "session_id": str(event.get("session_id") or ""),
                "question_id": str(event.get("question_id") or ""),
                "phase": _canonical_phase(event.get("phase")),
                "operator": str(event.get("operator") or ""),
                "operation": str(event.get("operation") or ""),
                "attempt": _optional_int(event.get("attempt")),
                "execution_attempt": _optional_int(
                    event.get("execution_attempt")
                ),
                "unit_attempt": _optional_int(event.get("unit_attempt")),
                "query_digest": str(event.get("query_digest") or ""),
                "cache_mode": str(event.get("cache_mode") or ""),
                "status": str(event.get("status") or ""),
                "output_row_count": _optional_int(
                    event.get("output_row_count")
                ),
                "lm_cache_hits": _counter(event, "lm_cache_hits"),
                "operator_cache_hits": _counter(
                    event,
                    "operator_cache_hits",
                ),
                "physical_prompt_tokens": _counter(
                    event,
                    "physical_prompt_tokens",
                ),
                "physical_completion_tokens": _counter(
                    event,
                    "physical_completion_tokens",
                ),
                "physical_total_tokens": _counter(
                    event,
                    "physical_total_tokens",
                ),
                "virtual_prompt_tokens": _counter(
                    event,
                    "virtual_prompt_tokens",
                ),
                "virtual_completion_tokens": _counter(
                    event,
                    "virtual_completion_tokens",
                ),
                "virtual_total_tokens": _counter(
                    event,
                    "virtual_total_tokens",
                ),
                "error_type": str(event.get("error_type") or ""),
                "error_message": str(event.get("error_message") or ""),
            }
        )
    return rows


def summarize_framework_cache_usage(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize framework cache counters separately from provider usage."""

    counter_keys = (
        "lm_cache_hits",
        "operator_cache_hits",
        "physical_prompt_tokens",
        "physical_completion_tokens",
        "physical_total_tokens",
        "virtual_prompt_tokens",
        "virtual_completion_tokens",
        "virtual_total_tokens",
    )
    return {
        "observed_operation_count": len(rows),
        "error_count": sum(row.get("status") == "error" for row in rows),
        **{
            key: sum(int(row.get(key) or 0) for row in rows)
            for key in counter_keys
        },
    }


def _agent_provider_row(
    event: Mapping[str, Any],
    output_dir: Path,
    pricing: PricingSnapshot | None,
    *,
    semantic_event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prompt = _optional_int(event.get("provider_prompt_tokens"))
    hit = _optional_int(event.get("provider_prompt_cache_hit_tokens"))
    miss = _optional_int(event.get("provider_prompt_cache_miss_tokens"))
    completion = _optional_int(event.get("provider_completion_tokens"))
    raw_usage = _read_usage_artifact(output_dir, event.get("provider_raw_usage_path"))
    reasoning = _reasoning_tokens(raw_usage)
    return _row(
        event,
        source="agent-provider",
        status="success",
        latency_ms=(
            _seconds_to_milliseconds(semantic_event.get("latency_sec"))
            if semantic_event is not None
            and int(event.get("provider_item_index") or 0) == 0
            else None
        ),
        prompt_tokens=prompt,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        usage_available=bool(event.get("provider_usage_available")),
        pricing=pricing,
    )


def _agent_runner_row(
    event: Mapping[str, Any],
    pricing: PricingSnapshot | None,
) -> dict[str, Any]:
    error = event.get("event_type") == "llm_batch_error"
    return _row(
        event,
        source="agent-runner",
        status="error" if error else "success",
        latency_ms=_seconds_to_milliseconds(event.get("latency_sec")),
        prompt_tokens=_optional_int(event.get("usage_prompt_tokens")),
        cache_hit_tokens=_optional_int(
            event.get("usage_prompt_cache_hit_tokens")
        ),
        cache_miss_tokens=_optional_int(
            event.get("usage_prompt_cache_miss_tokens")
        ),
        completion_tokens=_optional_int(event.get("usage_completion_tokens")),
        reasoning_tokens=_optional_int(event.get("usage_reasoning_tokens")),
        usage_available=not error and any(
            key.startswith("usage_") for key in event
        ),
        pricing=pricing,
    )


def _native_graphiti_row(
    event: Mapping[str, Any],
    pricing: PricingSnapshot | None,
) -> dict[str, Any]:
    usage = _mapping(event.get("usage"))
    provider_cache = _mapping(event.get("provider_cache"))
    prompt = _optional_int(usage.get("prompt_tokens"))
    completion = _optional_int(usage.get("completion_tokens"))
    return _row(
        event,
        source="native-graphiti",
        status=str(event.get("status") or "success"),
        latency_ms=_optional_float(event.get("latency_ms")),
        prompt_tokens=prompt,
        cache_hit_tokens=_optional_int(provider_cache.get("hit_tokens")),
        cache_miss_tokens=_optional_int(provider_cache.get("miss_tokens")),
        completion_tokens=completion,
        reasoning_tokens=_reasoning_tokens(usage),
        usage_available=prompt is not None and completion is not None,
        pricing=pricing,
    )


def _native_claude_row(
    event: Mapping[str, Any],
    pricing: PricingSnapshot | None,
) -> dict[str, Any]:
    usage = _mapping(event.get("usage"))
    error = event.get("event_type") == "llm_call_error"
    cache_hit = _optional_int(usage.get("cache_read_input_tokens"))
    input_tokens = _optional_int(usage.get("input_tokens"))
    cache_creation = _optional_int(usage.get("cache_creation_input_tokens"))
    cache_miss = _sum_optional(input_tokens, cache_creation)
    prompt = _sum_optional(cache_hit, cache_miss)
    completion = _optional_int(usage.get("output_tokens"))
    return _row(
        event,
        source="native-claude",
        status="error" if error else "success",
        latency_ms=_optional_float(event.get("latency_ms")),
        prompt_tokens=prompt,
        cache_hit_tokens=cache_hit,
        cache_miss_tokens=cache_miss,
        completion_tokens=completion,
        reasoning_tokens=None,
        usage_available=not error and prompt is not None and completion is not None,
        pricing=pricing,
    )


def _row(
    event: Mapping[str, Any],
    *,
    source: str,
    status: str,
    latency_ms: float | None,
    prompt_tokens: int | None,
    cache_hit_tokens: int | None,
    cache_miss_tokens: int | None,
    completion_tokens: int | None,
    reasoning_tokens: int | None,
    usage_available: bool,
    pricing: PricingSnapshot | None,
) -> dict[str, Any]:
    total = _sum_optional(prompt_tokens, completion_tokens)
    cost = None if pricing is None else pricing.estimate_cost_usd(
        cache_hit_input_tokens=cache_hit_tokens,
        cache_miss_input_tokens=cache_miss_tokens,
        output_tokens=completion_tokens,
    )
    return {
        "trace_id": str(event.get("trace_id") or event.get("logical_call_id") or ""),
        "logical_call_id": str(
            event.get("operator_call_id")
            or event.get("logical_call_id")
            or event.get("llm_batch_id")
            or ""
        ),
        "framework_batch_id": str(
            event.get("provider_batch_id")
            or event.get("llm_batch_id")
            or event.get("trace_id")
            or event.get("logical_call_id")
            or ""
        ),
        "case_id": str(event.get("case_id") or ""),
        "event_id": str(event.get("event_id") or ""),
        "session_id": str(event.get("session_id") or ""),
        "question_id": str(event.get("question_id") or ""),
        "phase": _canonical_phase(event.get("phase")),
        "operator": str(
            event.get("semantic_operator") or event.get("operator") or ""
        ),
        "operation": str(event.get("operation") or ""),
        "attempt": _optional_int(event.get("attempt")),
        "execution_attempt": _optional_int(event.get("execution_attempt")),
        "unit_attempt": _optional_int(event.get("unit_attempt")),
        "batch_size": _optional_int(
            event.get("provider_batch_size") or event.get("llm_batch_size")
        ),
        "item_index": _optional_int(
            event.get("provider_item_index") or event.get("llm_item_index") or 0
        ),
        "source": source,
        "status": status,
        "model": str(event.get("model") or ""),
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total,
        "usage_available": usage_available,
        "known_cost_usd": (
            None if pricing is None else float(round(cost, 12)) if cost is not None else 0.0
        ),
        "estimated_cost_usd": (
            float(round(cost, 12)) if cost is not None else None
        ),
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    costs = [row.get("estimated_cost_usd") for row in rows]
    known_cost = sum(Decimal(str(cost)) for cost in costs if cost is not None)
    return {
        "provider_call_count": len(rows),
        "provider_error_count": sum(row.get("status") == "error" for row in rows),
        "provider_usage_available_count": sum(
            bool(row.get("usage_available")) for row in rows
        ),
        "usage_complete": all(bool(row.get("usage_available")) for row in rows),
        "known_cost_usd": float(round(known_cost, 12)),
        "latency_ms": round(
            sum(float(row.get("latency_ms") or 0) for row in rows), 3
        ),
        "prompt_tokens": _known_token_sum(rows, "prompt_tokens"),
        "cache_hit_tokens": _known_token_sum(rows, "cache_hit_tokens"),
        "cache_miss_tokens": _known_token_sum(rows, "cache_miss_tokens"),
        "completion_tokens": _known_token_sum(rows, "completion_tokens"),
        "reasoning_tokens": _complete_token_sum(rows, "reasoning_tokens"),
        "total_tokens": _known_token_sum(rows, "total_tokens"),
        "estimated_cost_usd": (
            float(round(known_cost, 12))
            if costs and all(cost is not None for cost in costs)
            else (0.0 if not costs else None)
        ),
    }


def _known_token_sum(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(int(row[key]) for row in rows if row.get(key) is not None)


def _complete_token_sum(
    rows: Sequence[Mapping[str, Any]], key: str
) -> int | None:
    if rows and any(row.get(key) is None for row in rows):
        return None
    return _known_token_sum(rows, key)


def _agent_call_key(
    event: Mapping[str, Any], *, item_key: str
) -> tuple[str, int] | None:
    call_id = event.get("operator_call_id")
    if not isinstance(call_id, str) or not call_id:
        return None
    return call_id, int(event.get(item_key) or 0)


def _read_usage_artifact(output_dir: Path, path_value: Any) -> Mapping[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        return {}
    path = Path(path_value)
    if not path.is_absolute():
        path = output_dir / path
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, Mapping) else {}


def _reasoning_tokens(usage: Mapping[str, Any]) -> int | None:
    details = _mapping(usage.get("completion_tokens_details"))
    return _optional_int(details.get("reasoning_tokens"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _canonical_phase(value: Any) -> str:
    phase = str(value or "unknown")
    return _PHASE_ALIASES.get(phase, phase)


def _phase_sort_key(phase: str) -> tuple[int, str]:
    try:
        return _STANDARD_PHASES.index(phase), phase
    except ValueError:
        return len(_STANDARD_PHASES), phase


def _sum_optional(*values: int | None) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _seconds_to_milliseconds(value: Any) -> float | None:
    seconds = _optional_float(value)
    return round(seconds * 1000, 3) if seconds is not None else None


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    number = int(value)
    if number < 0:
        raise ValueError("provider token counters must be non-negative")
    return number


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return round(float(value), 3)


def _counter(event: Mapping[str, Any], key: str) -> int:
    value = _optional_int(event.get(key))
    return value if value is not None else 0


__all__ = [
    "normalize_framework_cache_usage",
    "normalize_provider_calls",
    "summarize_framework_cache_usage",
    "summarize_provider_calls",
]
