"""Tests for compact semantic trace artifacts and I/O accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.policy.logical import QueryExpr
from agent_memory.tracing.semantic import (
    measure_semantic_trace_io,
    semantic_trace_scope,
    write_pair_trace,
    write_trace_event,
)


def _events(trace_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _select_query() -> QueryExpr:
    return QueryExpr(
        op="select",
        inputs=(QueryExpr(op="materialized_view", params={"name": "rows"}),),
        params={"columns": ("memory",)},
    )


def test_lotus_trace_defaults_to_compact_without_frame_snapshots(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    adapter = LotusAdapter(
        config=LotusExecutionConfig(semantic_trace_dir=trace_dir)
    )

    result = adapter.execute(
        _select_query(),
        {"rows": pd.DataFrame({"memory": ["likes tea"]})},
    )

    assert result.to_dict("records") == [{"memory": "likes tea"}]
    [event] = _events(trace_dir)
    assert event["output_rows"] == 1
    assert event["output_columns"] == ["memory"]
    assert event["semantic_trace_snapshot_mode"] == "compact"
    assert "output_snapshot_path" not in event
    assert not (trace_dir / "snapshots").exists()


def test_lotus_full_trace_explicitly_writes_frame_snapshots(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    adapter = LotusAdapter(
        config=LotusExecutionConfig(
            semantic_trace_dir=trace_dir,
            semantic_trace_snapshot_mode="full",
        )
    )

    adapter.execute(
        _select_query(),
        {"rows": pd.DataFrame({"memory": ["likes tea"]})},
    )

    [event] = _events(trace_dir)
    assert event["semantic_trace_snapshot_mode"] == "full"
    snapshot = trace_dir / str(event["output_snapshot_path"]).removeprefix("trace/")
    assert pd.read_csv(snapshot).to_dict("records") == [{"memory": "likes tea"}]


def test_lotus_trace_rejects_unknown_snapshot_mode() -> None:
    with pytest.raises(ValueError, match="semantic_trace_snapshot_mode"):
        LotusExecutionConfig(semantic_trace_snapshot_mode="verbose")


def test_compact_pair_trace_inlines_decision_without_duplicate_artifacts(
    tmp_path: Path,
) -> None:
    pairs = pd.DataFrame({"left": ["tea"], "right": ["tea preference"]})

    with semantic_trace_scope(semantic_trace_snapshot_mode="compact"):
        write_pair_trace(
            tmp_path,
            operator="sem_join",
            rows=(
                {
                    "left": "tea",
                    "right": "tea preference",
                    "parsed_output": True,
                    "raw_output": "True",
                },
            ),
            snapshots={"pairs": pairs},
        )

    [event] = _events(tmp_path)
    assert event["decision"] is True
    assert "parsed_output_path" not in event
    assert "raw_output_path" not in event
    assert "pairs_snapshot_path" not in event
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "snapshots").exists()


def test_trace_io_measurement_counts_written_bytes(tmp_path: Path) -> None:
    with measure_semantic_trace_io() as measurement:
        write_trace_event(
            tmp_path,
            operator="select",
            event_type="operator_result",
            payload={"output_rows": 1},
        )

    assert measurement.bytes_written == (tmp_path / "events.jsonl").stat().st_size
    assert measurement.latency_ms >= 0
