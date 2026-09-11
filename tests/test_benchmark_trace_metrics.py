from __future__ import annotations

import json
from pathlib import Path

from agent_memory.evaluation.trace_metrics import (
    normalize_framework_cache_usage,
    normalize_provider_calls,
    summarize_framework_cache_usage,
    summarize_provider_calls,
)


def test_normalize_provider_calls_can_omit_pricing(tmp_path: Path) -> None:
    events = [{
        "event_type": "provider_usage", "provider_usage_available": True,
        "provider_prompt_tokens": 10, "provider_completion_tokens": 2,
        "provider_prompt_cache_hit_tokens": 4, "provider_prompt_cache_miss_tokens": 6,
    }]
    priced = normalize_provider_calls(events, output_dir=tmp_path)[0]
    unpriced = normalize_provider_calls(events, output_dir=tmp_path, include_cost=False)[0]
    assert priced["estimated_cost_usd"] is not None
    for key, value in priced.items():
        assert unpriced[key] == (None if key.endswith("cost_usd") else value)


def test_normalize_provider_calls_deduplicates_agent_semantic_usage(
    tmp_path: Path,
) -> None:
    usage_path = tmp_path / "trace" / "outputs" / "usage.json"
    usage_path.parent.mkdir(parents=True)
    usage_path.write_text(
        json.dumps(
            {
                "completion_tokens_details": {"reasoning_tokens": 3},
            }
        ),
        encoding="utf-8",
    )
    events = [
        {
            "trace_id": "semantic-call",
            "event_type": "llm_call",
            "operator": "sem_agg",
            "operator_call_id": "call-1",
            "llm_item_index": 0,
            "phase": "insertion",
            "model": "deepseek-v4-flash",
            "latency_sec": 0.5,
        },
        {
            "trace_id": "provider-call",
            "event_type": "provider_usage",
            "operator": "sem_agg",
            "operator_call_id": "call-1",
            "provider_item_index": 0,
            "phase": "insertion",
            "model": "deepseek-v4-flash",
            "provider_usage_available": True,
            "provider_prompt_tokens": 11,
            "provider_prompt_cache_hit_tokens": 8,
            "provider_prompt_cache_miss_tokens": 3,
            "provider_completion_tokens": 5,
            "provider_raw_usage_path": "trace/outputs/usage.json",
        },
    ]

    rows = normalize_provider_calls(events, output_dir=tmp_path)

    assert len(rows) == 1
    assert rows[0] == {
        "trace_id": "provider-call",
        "logical_call_id": "call-1",
        "framework_batch_id": "provider-call",
        "case_id": "",
        "event_id": "",
        "session_id": "",
        "question_id": "",
        "phase": "insertion",
        "operator": "sem_agg",
        "operation": "",
        "attempt": None,
        "execution_attempt": None,
        "unit_attempt": None,
        "batch_size": None,
        "item_index": 0,
        "source": "agent-provider",
        "status": "success",
        "model": "deepseek-v4-flash",
        "latency_ms": 500.0,
        "prompt_tokens": 11,
        "cache_hit_tokens": 8,
        "cache_miss_tokens": 3,
        "completion_tokens": 5,
        "reasoning_tokens": 3,
        "total_tokens": 16,
        "usage_available": True,
        "known_cost_usd": 1.8424e-06,
        "estimated_cost_usd": 1.8424e-06,
    }


def test_normalize_provider_calls_supports_runner_graphiti_and_native_claude(
    tmp_path: Path,
) -> None:
    events = [
        {
            "trace_id": "runner",
            "event_type": "llm_call",
            "operator": "llm",
            "phase": "answering",
            "model": "deepseek/deepseek-v4-flash",
            "latency_sec": 1.25,
            "usage_prompt_tokens": 7,
            "usage_prompt_cache_hit_tokens": 4,
            "usage_prompt_cache_miss_tokens": 3,
            "usage_completion_tokens": 2,
            "usage_reasoning_tokens": 1,
        },
        {
            "trace_id": "graphiti",
            "event_type": "llm_call",
            "status": "success",
            "phase": "insertion",
            "model": "deepseek-v4-flash",
            "latency_ms": 10.5,
            "provider_cache": {"hit_tokens": 6, "miss_tokens": 4},
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
        {
            "trace_id": "claude",
            "event_type": "llm_call_finish",
            "phase": "extract",
            "model": "deepseek-v4-flash",
            "latency_ms": 20,
            "usage": {
                "input_tokens": 5,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 9,
                "output_tokens": 4,
            },
        },
    ]

    rows = normalize_provider_calls(events, output_dir=tmp_path)

    assert [row["source"] for row in rows] == [
        "agent-runner",
        "native-graphiti",
        "native-claude",
    ]
    assert rows[0]["latency_ms"] == 1250.0
    assert rows[1]["reasoning_tokens"] == 2
    assert rows[2]["phase"] == "insertion"
    assert rows[2]["prompt_tokens"] == 16
    assert rows[2]["cache_hit_tokens"] == 9
    assert rows[2]["cache_miss_tokens"] == 7
    assert rows[2]["reasoning_tokens"] is None


def test_provider_summary_keeps_unknown_cost_explicit(tmp_path: Path) -> None:
    rows = normalize_provider_calls(
        [
            {
                "trace_id": "ok",
                "event_type": "llm_call",
                "operator": "llm",
                "phase": "answering",
                "usage_prompt_tokens": 2,
                "usage_prompt_cache_hit_tokens": 0,
                "usage_prompt_cache_miss_tokens": 2,
                "usage_completion_tokens": 1,
                "usage_reasoning_tokens": 0,
            },
            {
                "trace_id": "error",
                "event_type": "llm_call_error",
                "phase": "retrieval",
                "latency_ms": 8,
            },
        ],
        output_dir=tmp_path,
    )

    summary = summarize_provider_calls(rows)

    assert summary["provider_call_count"] == 2
    assert summary["provider_error_count"] == 1
    assert summary["usage_complete"] is False
    assert summary["estimated_cost_usd"] is None
    assert summary["phases"]["answering"]["estimated_cost_usd"] == 5.6e-07
    assert summary["phases"]["retrieval"]["estimated_cost_usd"] is None


def test_normalize_provider_calls_preserves_batch_identity_and_usage(
    tmp_path: Path,
) -> None:
    events = []
    for batch_id, size in (("batch-1", 3), ("batch-2", 2), ("batch-3", 1)):
        for item_index in range(size):
            events.append(
                {
                    "trace_id": f"{batch_id}-{item_index}",
                    "event_type": "provider_usage",
                    "operator": "sem_topk",
                    "operator_call_id": "logical-1",
                    "provider_batch_id": batch_id,
                    "provider_item_index": item_index,
                    "phase": "retrieval",
                    "model": "deepseek-v4-flash",
                    "provider_usage_available": True,
                    "provider_prompt_tokens": 10,
                    "provider_prompt_cache_hit_tokens": 6,
                    "provider_prompt_cache_miss_tokens": 4,
                    "provider_completion_tokens": 2,
                }
            )

    rows = normalize_provider_calls(events, output_dir=tmp_path)
    summary = summarize_provider_calls(rows)

    assert {row["logical_call_id"] for row in rows} == {"logical-1"}
    assert {row["framework_batch_id"] for row in rows} == {
        "batch-1",
        "batch-2",
        "batch-3",
    }
    assert summary["provider_call_count"] == 6
    assert summary["prompt_tokens"] == 60
    assert summary["cache_hit_tokens"] == 36
    assert summary["cache_miss_tokens"] == 24
    assert summary["completion_tokens"] == 12
    assert summary["estimated_cost_usd"] == 6.8208e-06


def test_framework_cache_usage_is_separate_from_provider_usage(
    tmp_path: Path,
) -> None:
    events = [
        {
            "trace_id": "cache-1",
            "operator_call_id": "operator-1",
            "event_type": "framework_cache_usage",
            "operator": "sem_filter",
            "phase": "insertion",
            "case_id": "case-1",
            "event_id": "event-1",
            "cache_mode": "lotus-memory:1024",
            "status": "success",
            "output_row_count": 2,
            "lm_cache_hits": 1,
            "operator_cache_hits": 2,
            "physical_prompt_tokens": 3,
            "physical_completion_tokens": 1,
            "physical_total_tokens": 4,
            "virtual_prompt_tokens": 9,
            "virtual_completion_tokens": 3,
            "virtual_total_tokens": 12,
        }
    ]

    rows = normalize_framework_cache_usage(events)
    summary = summarize_framework_cache_usage(rows)

    assert normalize_provider_calls(events, output_dir=tmp_path) == []
    assert rows[0]["cache_mode"] == "lotus-memory:1024"
    assert summary == {
        "observed_operation_count": 1,
        "error_count": 0,
        "lm_cache_hits": 1,
        "operator_cache_hits": 2,
        "physical_prompt_tokens": 3,
        "physical_completion_tokens": 1,
        "physical_total_tokens": 4,
        "virtual_prompt_tokens": 9,
        "virtual_completion_tokens": 3,
        "virtual_total_tokens": 12,
    }
