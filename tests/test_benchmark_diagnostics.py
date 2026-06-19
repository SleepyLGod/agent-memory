"""Tests for benchmark trace-derived diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_memory.benchmarks.diagnostics import build_cause_trace_rows, build_llm_anomaly_rows
from agent_memory.benchmarks.types import BenchmarkQuestion


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
