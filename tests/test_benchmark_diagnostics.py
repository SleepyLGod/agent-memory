"""Tests for benchmark trace-derived diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_memory.benchmarks.diagnostics import (
    build_cause_trace_rows,
    build_llm_anomaly_rows,
    build_provider_usage_rows,
    build_provider_usage_summary_rows,
)
from agent_memory.benchmarks.types import BenchmarkQuestion
from agent_memory.adapters.lotus import provider_usage_lm
from agent_memory.tracing.semantic import write_provider_usage_trace


def question(evidence_event_ids: tuple[str, ...]) -> BenchmarkQuestion:
    """Return one tiny benchmark question."""

    return BenchmarkQuestion(
        question_id="q1",
        sample_id="sample",
        question="What happened?",
        gold_answer="answer",
        evidence_event_ids=evidence_event_ids,
        category="2",
    )


def write_events(trace_dir: Path, events: list[dict[str, Any]]) -> None:
    """Write semantic trace events."""

    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event))
            handle.write("\n")


def write_parsed_output(trace_dir: Path, name: str, value: Any) -> str:
    """Write one parsed output artifact and return its trace-relative path."""

    output_dir = trace_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return f"trace/outputs/{name}"


def write_raw_output(trace_dir: Path, name: str, value: Any) -> str:
    """Write one raw output artifact and return its trace-relative path."""

    output_dir = trace_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return f"trace/outputs/{name}"


def test_cause_trace_marks_questions_without_evidence(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    write_events(trace_dir, [])

    rows = build_cause_trace_rows(
        questions=[question(())],
        ingested_event_ids=set(),
        trace_dir=trace_dir,
    )

    assert rows[0]["source_status"] == "no_evidence_id"
    assert rows[0]["trace_status"] == "no_evidence_id"


def test_cause_trace_marks_not_ingested_evidence(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    write_events(trace_dir, [])

    rows = build_cause_trace_rows(
        questions=[question(("D1:5",))],
        ingested_event_ids=set(),
        trace_dir=trace_dir,
    )

    assert rows[0]["source_status"] == "not_ingested"
    assert rows[0]["trace_status"] == "not_ingested"


def test_cause_trace_marks_missing_trace_for_ingested_evidence(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    write_events(trace_dir, [])

    rows = build_cause_trace_rows(
        questions=[question(("D1:5",))],
        ingested_event_ids={"D1:5"},
        trace_dir=trace_dir,
    )

    assert rows[0]["source_status"] == "ingested"
    assert rows[0]["trace_status"] == "missing_trace"


def test_cause_trace_marks_first_zero_output_operator(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    parsed_path = write_parsed_output(trace_dir, "flat-map-parsed.json", [])
    write_events(
        trace_dir,
        [
            {
                "trace_id": "t1",
                "phase": "add",
                "event_id": "D1:5",
                "operator": "sem_flat_map",
                "event_type": "structured_generation",
                "parsed_output_path": parsed_path,
            },
            {
                "trace_id": "t2",
                "phase": "add",
                "event_id": "D1:5",
                "operator": "sem_join",
                "event_type": "operator_result",
                "output_rows": 1,
            },
        ],
    )

    rows = build_cause_trace_rows(
        questions=[question(("D1:5",))],
        ingested_event_ids={"D1:5"},
        trace_dir=trace_dir,
    )

    assert rows[0]["trace_status"] == "stopped_at_zero_output"
    assert rows[0]["first_nonzero_operator"] == ""
    assert rows[0]["first_nonzero_trace_id"] == ""
    assert rows[0]["first_nonzero_output_rows"] == ""
    assert rows[0]["first_zero_output_operator"] == "sem_flat_map"
    assert rows[0]["first_zero_output_trace_id"] == "t1"
    assert rows[0]["operator_output_rows"] == 0


def test_cause_trace_marks_observed_with_output(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    parsed_path = write_parsed_output(trace_dir, "flat-map-parsed.json", [{"name": "memory"}])
    write_events(
        trace_dir,
        [
            {
                "trace_id": "t1",
                "phase": "add",
                "event_id": "D1:9",
                "operator": "sem_flat_map",
                "event_type": "structured_generation",
                "parsed_output_path": parsed_path,
            },
            {
                "trace_id": "t2",
                "phase": "add",
                "event_id": "D1:9",
                "operator": "select",
                "event_type": "operator_result",
                "output_rows": 3,
            }
        ],
    )

    rows = build_cause_trace_rows(
        questions=[question(("D1:9",))],
        ingested_event_ids={"D1:9"},
        trace_dir=trace_dir,
    )

    assert rows[0]["trace_status"] == "observed_with_output"
    assert rows[0]["first_nonzero_operator"] == "sem_flat_map"
    assert rows[0]["first_nonzero_trace_id"] == "t1"
    assert rows[0]["first_nonzero_output_rows"] == 1
    assert rows[0]["last_observed_operator"] == "select"
    assert rows[0]["last_observed_trace_id"] == "t2"
    assert rows[0]["operator_output_rows"] == 3


def test_cause_trace_records_nonzero_before_later_zero(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    parsed_path = write_parsed_output(trace_dir, "flat-map-parsed.json", [{"name": "memory"}])
    write_events(
        trace_dir,
        [
            {
                "trace_id": "t1",
                "phase": "add",
                "event_id": "D1:9",
                "operator": "sem_flat_map",
                "event_type": "structured_generation",
                "parsed_output_path": parsed_path,
            },
            {
                "trace_id": "t2",
                "phase": "add",
                "event_id": "D1:9",
                "operator": "sem_map",
                "event_type": "operator_result",
                "output_rows": 0,
            },
        ],
    )

    rows = build_cause_trace_rows(
        questions=[question(("D1:9",))],
        ingested_event_ids={"D1:9"},
        trace_dir=trace_dir,
    )

    assert rows[0]["trace_status"] == "stopped_at_zero_output"
    assert rows[0]["first_nonzero_operator"] == "sem_flat_map"
    assert rows[0]["first_nonzero_trace_id"] == "t1"
    assert rows[0]["first_nonzero_output_rows"] == 1
    assert rows[0]["first_zero_output_operator"] == "sem_map"
    assert rows[0]["first_zero_output_trace_id"] == "t2"


def test_cause_trace_prefers_event_output_rows(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    parsed_path = write_parsed_output(trace_dir, "map-parsed.json", [{"name": "memory"}])
    write_events(
        trace_dir,
        [
            {
                "trace_id": "t1",
                "phase": "add",
                "event_id": "D1:5",
                "operator": "sem_map",
                "event_type": "structured_generation",
                "output_rows": 0,
                "parsed_output_path": parsed_path,
            }
        ],
    )

    rows = build_cause_trace_rows(
        questions=[question(("D1:5",))],
        ingested_event_ids={"D1:5"},
        trace_dir=trace_dir,
    )

    assert rows[0]["trace_status"] == "stopped_at_zero_output"
    assert rows[0]["operator_output_rows"] == 0


def test_llm_anomaly_marks_empty_raw_output(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    raw_path = write_raw_output(trace_dir, "topk-raw.json", {"output": ""})
    write_events(
        trace_dir,
        [
            {
                "trace_id": "t1",
                "phase": "retrieval",
                "question_id": "q1",
                "operator": "sem_topk",
                "event_type": "llm_call",
                "raw_output_path": raw_path,
                "model": "test-model",
            }
        ],
    )

    rows = build_llm_anomaly_rows(trace_dir=trace_dir)

    assert len(rows) == 1
    assert rows[0]["phase"] == "retrieval"
    assert rows[0]["operator"] == "sem_topk"
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["issue"] == "empty_output"
    assert rows[0]["raw_output_path"] == raw_path
    assert rows[0]["model"] == "test-model"


def test_llm_anomaly_ignores_nonempty_raw_output(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    raw_path = write_raw_output(trace_dir, "topk-raw.json", {"output": "Document 1"})
    write_events(
        trace_dir,
        [
            {
                "trace_id": "t1",
                "phase": "retrieval",
                "operator": "sem_topk",
                "event_type": "llm_call",
                "raw_output_path": raw_path,
            }
        ],
    )

    assert build_llm_anomaly_rows(trace_dir=trace_dir) == []


def test_llm_anomaly_marks_missing_raw_output(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    write_events(
        trace_dir,
        [
            {
                "trace_id": "t1",
                "phase": "answer",
                "operator": "answer",
                "event_type": "llm_call",
            }
        ],
    )

    rows = build_llm_anomaly_rows(trace_dir=trace_dir)

    assert len(rows) == 1
    assert rows[0]["operator"] == "answer"
    assert rows[0]["issue"] == "missing_raw_output"


def test_llm_anomaly_preserves_answer_question_id(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    raw_path = write_raw_output(trace_dir, "answer-raw.json", {"output": ""})
    write_events(
        trace_dir,
        [
            {
                "trace_id": "t1",
                "phase": "answer",
                "question_id": "conv-26:q5",
                "operator": "answer",
                "event_type": "llm_call",
                "raw_output_path": raw_path,
            }
        ],
    )

    rows = build_llm_anomaly_rows(trace_dir=trace_dir)

    assert len(rows) == 1
    assert rows[0]["phase"] == "answer"
    assert rows[0]["operator"] == "answer"
    assert rows[0]["question_id"] == "conv-26:q5"
    assert rows[0]["issue"] == "empty_output"


def test_llm_anomaly_marks_batch_error(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    write_events(
        trace_dir,
        [
            {
                "trace_id": "t1",
                "phase": "retrieval",
                "question_id": "conv-26:q3",
                "operator": "sem_topk",
                "event_type": "llm_batch_error",
                "prompt_path": "trace/prompts/topk-prompt.json",
                "error_type": "InternalServerError",
                "error_message": "SSL EOF",
                "model": "test-model",
            }
        ],
    )

    rows = build_llm_anomaly_rows(trace_dir=trace_dir)

    assert len(rows) == 1
    assert rows[0]["phase"] == "retrieval"
    assert rows[0]["operator"] == "sem_topk"
    assert rows[0]["question_id"] == "conv-26:q3"
    assert rows[0]["issue"] == "llm_batch_error"
    assert rows[0]["prompt_path"] == "trace/prompts/topk-prompt.json"
    assert rows[0]["error_type"] == "InternalServerError"
    assert rows[0]["error_message"] == "SSL EOF"


def test_provider_usage_rows_ignore_other_events(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    write_events(
        trace_dir,
        [
            {
                "trace_id": "llm-1",
                "phase": "retrieval",
                "operator": "sem_topk",
                "event_type": "llm_call",
            },
            {
                "trace_id": "usage-1",
                "phase": "retrieval",
                "operator": "sem_topk",
                "question_id": "q1",
                "event_type": "provider_usage",
                "provider_usage_available": True,
                "provider_prompt_tokens": 11,
                "provider_completion_tokens": 3,
                "provider_total_tokens": 14,
                "provider_prompt_cache_hit_tokens": 5,
                "provider_prompt_cache_miss_tokens": 6,
                "provider_raw_usage_path": "trace/outputs/usage.json",
            },
        ],
    )

    rows = build_provider_usage_rows(trace_dir=trace_dir)

    assert len(rows) == 1
    assert rows[0]["trace_id"] == "usage-1"
    assert rows[0]["phase"] == "retrieval"
    assert rows[0]["operator"] == "sem_topk"
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["provider_usage_available"] is True
    assert rows[0]["provider_prompt_tokens"] == 11
    assert rows[0]["provider_completion_tokens"] == 3
    assert rows[0]["provider_total_tokens"] == 14
    assert rows[0]["provider_prompt_cache_hit_tokens"] == 5
    assert rows[0]["provider_prompt_cache_miss_tokens"] == 6
    assert rows[0]["provider_raw_usage_path"] == "trace/outputs/usage.json"


def test_provider_usage_rows_preserve_unavailable_usage(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    write_events(
        trace_dir,
        [
            {
                "trace_id": "usage-1",
                "phase": "answer",
                "operator": "answer",
                "event_type": "provider_usage",
                "provider_usage_available": False,
            },
        ],
    )

    rows = build_provider_usage_rows(trace_dir=trace_dir)

    assert len(rows) == 1
    assert rows[0]["provider_usage_available"] is False
    assert rows[0]["provider_total_tokens"] == 0


def test_provider_usage_rows_respect_excluded_trace_ranges(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    write_events(
        trace_dir,
        [
            {
                "trace_id": "usage-1",
                "phase": "retrieval",
                "event_type": "provider_usage",
                "provider_total_tokens": 10,
            },
            {
                "trace_id": "usage-2",
                "phase": "retrieval",
                "event_type": "provider_usage",
                "provider_total_tokens": 20,
            },
        ],
    )

    rows = build_provider_usage_rows(
        trace_dir=trace_dir,
        excluded_event_ranges=((1, 2),),
    )

    assert [row["trace_id"] for row in rows] == ["usage-1"]


def test_provider_usage_trace_compacts_request_metadata(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"

    class FakeUsage:
        def dict(self) -> dict[str, int]:
            return {"total_tokens": 3}

    class FakeResponse:
        usage = FakeUsage()

    write_provider_usage_trace(
        trace_dir,
        model="test-model",
        responses=[FakeResponse()],
        request_metadata={
            "instruction": "x" * 500,
            "provider_batch_size": 2,
        },
    )

    events = (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    event = json.loads(events[0])
    assert "instruction" not in event["provider_request"]
    assert event["provider_request"]["instruction_preview"].endswith("...")
    assert event["provider_request"]["provider_batch_size"] == 2


def test_provider_usage_tracing_mixin_is_best_effort(monkeypatch: Any) -> None:
    class BaseLM:
        model = "test-model"

        def _process_uncached_messages(self, *_args: Any, **_kwargs: Any) -> list[str]:
            return ["ok"]

    class TracedLM(provider_usage_lm.ProviderUsageTracingMixin, BaseLM):
        pass

    captured: dict[str, Any] = {}

    def fail_trace(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(provider_usage_lm, "write_provider_usage_trace", fail_trace)
    lm = TracedLM(trace_dir="/tmp/trace")

    responses = lm._process_uncached_messages(
        [([{"role": "user", "content": "hi"}], "cache-key")],
        {"max_tokens": 5, "api_key": "secret"},
        False,
        "",
    )

    assert responses == ["ok"]
    assert captured["request_metadata"]["provider_kwargs"] == {"max_tokens": 5}


def test_provider_usage_summary_groups_by_phase_and_total() -> None:
    rows = [
        {
            "phase": "retrieval",
            "provider_usage_available": True,
            "provider_prompt_tokens": 10,
            "provider_completion_tokens": 2,
            "provider_total_tokens": 12,
            "provider_prompt_cache_hit_tokens": 4,
        },
        {
            "phase": "retrieval",
            "provider_usage_available": False,
            "provider_prompt_tokens": 0,
            "provider_total_tokens": 0,
        },
        {
            "phase": "answer",
            "provider_usage_available": True,
            "provider_prompt_tokens": 7,
            "provider_completion_tokens": 1,
            "provider_total_tokens": 8,
            "provider_prompt_cache_miss_tokens": 7,
        },
    ]

    summary = build_provider_usage_summary_rows(rows)

    by_phase = {row["phase"]: row for row in summary}
    assert by_phase["answer"]["provider_usage_event_count"] == 1
    assert by_phase["answer"]["provider_usage_available_count"] == 1
    assert by_phase["answer"]["provider_total_tokens"] == 8
    assert by_phase["retrieval"]["provider_usage_event_count"] == 2
    assert by_phase["retrieval"]["provider_usage_available_count"] == 1
    assert by_phase["retrieval"]["provider_total_tokens"] == 12
    assert by_phase["_total"]["provider_usage_event_count"] == 3
    assert by_phase["_total"]["provider_usage_available_count"] == 2
    assert by_phase["_total"]["provider_total_tokens"] == 20
    assert by_phase["_total"]["provider_prompt_cache_hit_tokens"] == 4
    assert by_phase["_total"]["provider_prompt_cache_miss_tokens"] == 7
