from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import csv
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from typing import Any

import pytest

from tools.analysis.semantic_pair_candidates import (
    CandidateStrategy,
    DirectorySource,
    PairGroup,
    PairRecord,
    PairTraceError,
    SentenceTransformerCosineScorer,
    SourceStats,
    TarSource,
    _write_single_report,
    analyze_sources,
    build_strategies,
)


class FakeScorer:
    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = scores

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"kind": "fake-cosine"}

    def score(self, group: PairGroup) -> Sequence[float]:
        return [self.scores[pair.pair_id] for pair in group.pairs]


class StaticSource:
    def __init__(self, groups: Sequence[PairGroup], name: str = "static") -> None:
        self.groups = tuple(groups)
        self.description = name
        self.stats = SourceStats(source=name)

    def iter_groups(self, *, phase: str) -> Iterator[PairGroup]:
        assert phase == "insertion"
        self.stats.scanned_group_count = len(self.groups)
        yield from self.groups

    def contains_output(self, output: Path) -> bool:
        return False


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, str]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _groupby_event(
    *,
    call_id: str,
    trace_id: str,
    left_id: int,
    right_id: int,
    left: str,
    right: str,
    parsed_path: str,
    timestamp: str | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "trace_id": trace_id,
        "run_kind": "zep-locomo",
        "phase": "insertion",
        "case_id": "case-1",
        "session_id": "session-1",
        "event_id": "event-1",
        "query_digest": "query-1",
        "operator": "sem_groupby",
        "operator_call_id": call_id,
        "event_type": "pair_decision",
        "source_instruction": "same entity",
        "left_unique_id": left_id,
        "right_unique_id": right_id,
        "left": left,
        "right": right,
        "parsed_output_path": parsed_path,
    }
    if timestamp is not None:
        event["timestamp"] = timestamp
    return event


def _sem_join_event(
    *,
    call_id: str,
    trace_id: str,
    left_id: int,
    right_id: int,
    left: str,
    right: str,
    parsed_path: str,
) -> dict[str, object]:
    return {
        **_groupby_event(
            call_id=call_id,
            trace_id=trace_id,
            left_id=left_id,
            right_id=right_id,
            left=left,
            right=right,
            parsed_path=parsed_path,
        ),
        "run_kind": "claude-locomo",
        "phase": "add",
        "case_id": "",
        "sample_id": "conv-26",
        "operator": "sem_join",
        "semantic_operator": "sem_join",
        "left_id": left_id,
        "right_id": right_id,
    }


def _write_attempt_ledger(
    root: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    _write_jsonl(root / "checkpoint/control-events.jsonl", rows)


def _write_legacy_recovery(
    root: Path,
    *,
    ranges: Sequence[Mapping[str, int]],
    trace_event_count: int,
) -> None:
    path = root / "diagnostics/recovery.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "excluded_trace_event_ranges": list(ranges),
                "excluded_trace_event_count": sum(row["count"] for row in ranges),
                "trace_event_count": trace_event_count,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_groupby_run(
    root: Path,
    *,
    call_id: str,
    labels: Sequence[bool],
) -> None:
    pairs = ((0, 1, "A", "B"), (0, 2, "A", "C"), (1, 2, "B", "C"))
    events: list[dict[str, object]] = []
    for index, ((left_id, right_id, left, right), label) in enumerate(
        zip(pairs, labels, strict=True)
    ):
        path = f"trace/outputs/pair-{index}.json"
        events.append(
            _groupby_event(
                call_id=call_id,
                trace_id=f"pair-{index}",
                left_id=left_id,
                right_id=right_id,
                left=left,
                right=right,
                parsed_path=path,
            )
        )
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(label) + "\n", encoding="utf-8")
    _write_jsonl(root / "trace/events.jsonl", events)


def _sem_filter_event() -> dict[str, object]:
    return {
        "trace_id": "filter",
        "run_kind": "mem0-locomo",
        "phase": "insertion",
        "case_id": "case-1",
        "session_id": "session-1",
        "event_id": "event-2",
        "query_digest": "query-2",
        "operator": "sem_filter",
        "operator_call_id": "filter-call",
        "event_type": "operator_result",
        "lowered_instruction": "Existing {memory_earlier}; New {memory_later}",
        "input_rows": 3,
        "output_rows": 1,
        "input_snapshot_path": "trace/snapshots/input.csv",
        "output_snapshot_path": "trace/snapshots/output.csv",
    }


def _write_sem_filter_run(root: Path) -> None:
    columns = (
        "_row_id:earlier",
        "_memory_ordinal:earlier",
        "memory:earlier",
        "_row_id:later",
        "_memory_ordinal:later",
        "memory:later",
    )
    duplicate = {
        "_row_id:earlier": "old-1",
        "_memory_ordinal:earlier": "0",
        "memory:earlier": "User likes Paris",
        "_row_id:later": "new-1",
        "_memory_ordinal:later": "0",
        "memory:later": "The user likes Paris",
    }
    second_duplicate = {**duplicate, "_row_id:earlier": "old-2"}
    distinct = {
        **duplicate,
        "_row_id:earlier": "old-3",
        "memory:earlier": "User has a dog",
    }
    _write_csv(
        root / "trace/snapshots/input.csv",
        [duplicate, second_duplicate, distinct],
        columns,
    )
    _write_csv(root / "trace/snapshots/output.csv", [duplicate], columns)
    _write_jsonl(root / "trace/events.jsonl", [_sem_filter_event()])


def _manual_group(direction: str) -> PairGroup:
    pairs = (
        PairRecord("ab", "A", "B", "A", "B", True),
        PairRecord("ac", "A", "C", "A", "C", False),
        PairRecord("bc", "B", "C", "B", "C", False),
    )
    if direction == "right-to-left":
        pairs = tuple(
            PairRecord(
                pair.pair_id,
                pair.left_id,
                "new",
                pair.left,
                "new memory",
                pair.baseline_match,
            )
            for pair in pairs
        )
    return PairGroup(
        group_id=f"group-{direction}",
        operator="sem_filter" if direction == "right-to-left" else "sem_groupby",
        direction=direction,
        source="static",
        case_id="case-1",
        session_id="session-1",
        event_id="event-1",
        query_digest="query-1",
        pairs=pairs,
    )


def test_newest_source_replaces_replayed_group(tmp_path: Path) -> None:
    newest = tmp_path / "newest"
    oldest = tmp_path / "oldest"
    _write_groupby_run(newest, call_id="new-call", labels=[False, False, False])
    _write_groupby_run(oldest, call_id="old-call", labels=[True, False, False])
    sources = [DirectorySource(newest), DirectorySource(oldest)]

    report = analyze_sources(sources)

    assert report["baseline"] == {
        "group_count": 1,
        "pair_count": 3,
        "positive_pair_count": 0,
    }
    assert report["sources"][0]["accepted_group_count"] == 1
    assert report["sources"][1]["accepted_group_count"] == 0
    assert all(row["events_sha256"] for row in report["sources"])


def test_sem_filter_preserves_direction_and_multiplicity(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_sem_filter_run(run)

    groups = list(DirectorySource(run).iter_groups(phase="insertion"))

    assert len(groups) == 1
    assert groups[0].direction == "right-to-left"
    assert len(groups[0].pairs) == 3
    assert sum(pair.baseline_match for pair in groups[0].pairs) == 1
    assert len({pair.pair_id for pair in groups[0].pairs}) == 3
    assert {pair.right_id for pair in groups[0].pairs} == {groups[0].pairs[0].right_id}


@pytest.mark.parametrize("read_workers", [1, 4, 8])
def test_directory_read_workers_preserve_pair_order_and_labels(
    tmp_path: Path, read_workers: int
) -> None:
    run = tmp_path / "run"
    _write_groupby_run(run, call_id="call", labels=[True, False, True])

    source = DirectorySource(run, read_workers=read_workers)
    groups = list(source.iter_groups(phase="insertion"))

    assert len(groups) == 1
    assert [pair.baseline_match for pair in groups[0].pairs] == [True, False, True]
    assert [(pair.left, pair.right) for pair in groups[0].pairs] == [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ]
    assert source.stats.requested_read_workers == read_workers
    assert source.stats.effective_read_workers == read_workers


def test_parallel_reader_reports_same_first_error_as_serial(tmp_path: Path) -> None:
    run = tmp_path / "run"
    events = [
        _groupby_event(
            call_id="call",
            trace_id="first",
            left_id=0,
            right_id=1,
            left="A",
            right="B",
            parsed_path="trace/outputs/first.json",
        ),
        _groupby_event(
            call_id="call",
            trace_id="second",
            left_id=0,
            right_id=2,
            left="A",
            right="C",
            parsed_path="trace/outputs/second.json",
        ),
    ]
    _write_jsonl(run / "trace/events.jsonl", events)

    errors: list[str] = []
    for read_workers in (1, 4, 8):
        with pytest.raises(PairTraceError) as captured:
            list(
                DirectorySource(run, read_workers=read_workers).iter_groups(
                    phase="insertion"
                )
            )
        errors.append(str(captured.value))

    assert errors[0] == errors[1] == errors[2]
    assert "first.json" in errors[0]


def test_symmetric_top_k_union_and_combined_threshold() -> None:
    group = _manual_group("symmetric")
    source = StaticSource([group])
    strategies = build_strategies(top_ks=[1], thresholds=[0.85])
    scorer = FakeScorer({"ab": 0.9, "ac": 0.8, "bc": 0.7})

    report = analyze_sources([source], strategies=strategies, scorer=scorer)

    metrics = {row["strategy"]: row for row in report["strategies"]}
    assert metrics["threshold:0.85"]["selected_pair_count"] == 1
    assert metrics["top-k:1"]["selected_pair_count"] == 2
    assert metrics["top-k:1+threshold:0.85"]["selected_pair_count"] == 1
    assert all(row["positive_pair_recall"] == 1.0 for row in metrics.values())


def test_right_to_left_top_k_is_per_later_row_and_reports_missed_positive() -> None:
    group = _manual_group("right-to-left")
    source = StaticSource([group])
    scorer = FakeScorer({"ab": 0.1, "ac": 0.9, "bc": 0.8})

    report = analyze_sources(
        [source],
        strategies=[CandidateStrategy(top_k=1)],
        scorer=scorer,
        max_examples=1,
    )

    metric = report["strategies"][0]
    assert metric["selected_pair_count"] == 1
    assert metric["positive_pair_recall"] == 0.0
    assert metric["groups_losing_positive_count"] == 1
    assert len(metric["counterexamples"]) == 1
    assert metric["counterexamples"][0]["pair_id"] == "ab"


def test_sem_join_uses_left_to_right_direction_and_explicit_ids(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    pairs = (
        (0, 10, "new A", "old X", True),
        (0, 11, "new A", "old Y", False),
        (1, 10, "new B", "old X", False),
        (1, 12, "new B", "old Z", True),
    )
    events: list[dict[str, object]] = []
    for index, (left_id, right_id, left, right, label) in enumerate(pairs):
        path = f"trace/outputs/pair-{index}.json"
        events.append(
            _sem_join_event(
                call_id="join-call",
                trace_id=f"join-{index}",
                left_id=left_id,
                right_id=right_id,
                left=left,
                right=right,
                parsed_path=path,
            )
        )
        target = run / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(label) + "\n", encoding="utf-8")
    _write_jsonl(run / "trace/events.jsonl", events)

    groups = list(DirectorySource(run).iter_groups(phase="add"))

    assert len(groups) == 1
    assert groups[0].operator == "sem_join"
    assert groups[0].direction == "left-to-right"
    assert groups[0].case_id == "conv-26"
    assert [(pair.left_id, pair.right_id) for pair in groups[0].pairs] == [
        ("0", "10"),
        ("0", "11"),
        ("1", "10"),
        ("1", "12"),
    ]


def test_left_to_right_top_k_is_per_left_query() -> None:
    group = PairGroup(
        group_id="join-group",
        operator="sem_join",
        direction="left-to-right",
        source="static",
        case_id="case-1",
        session_id="session-1",
        event_id="event-1",
        query_digest="query-1",
        pairs=(
            PairRecord("ax", "A", "X", "new A", "old X", True),
            PairRecord("ay", "A", "Y", "new A", "old Y", False),
            PairRecord("bx", "B", "X", "new B", "old X", False),
            PairRecord("bz", "B", "Z", "new B", "old Z", True),
        ),
    )
    scorer = FakeScorer({"ax": 0.1, "ay": 0.9, "bx": 0.6, "bz": 0.7})

    report = analyze_sources(
        [StaticSource([group])],
        strategies=[CandidateStrategy(top_k=1)],
        scorer=scorer,
    )

    metric = report["strategies"][0]
    assert metric["selected_pair_count"] == 2
    assert metric["positive_pair_recall"] == 0.5
    assert metric["groups_losing_positive_count"] == 1


def test_tar_and_directory_sem_filter_reports_match(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    run = source_root / "archived/run"
    _write_sem_filter_run(run)
    archive = tmp_path / "runs.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(run, arcname="archived/run")

    directory_report = analyze_sources([DirectorySource(run)])
    tar_report = analyze_sources(
        [TarSource(archive, "archived/run", read_workers=8)]
    )

    assert tar_report["baseline"] == directory_report["baseline"]
    assert tar_report["by_operator"] == directory_report["by_operator"]
    assert tar_report["sources"][0]["requested_read_workers"] == 8
    assert tar_report["sources"][0]["effective_read_workers"] == 1
    assert not list(tmp_path.rglob("*.sqlite"))
    assert not list(tmp_path.rglob("pairs.jsonl"))


def test_tar_sem_groupby_requires_directory_source(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    run = source_root / "archived/run"
    _write_groupby_run(run, call_id="call", labels=[True, False, False])
    archive = tmp_path / "runs.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(run, arcname="archived/run")

    with pytest.raises(PairTraceError, match="directory source"):
        analyze_sources([TarSource(archive, "archived/run")])


def test_compressed_tar_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    archive = tmp_path / "runs.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="run")

    with pytest.raises(PairTraceError, match="compressed tar"):
        TarSource(archive, "run")


def test_interleaved_group_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    events = [
        _groupby_event(
            call_id="call-a",
            trace_id="a1",
            left_id=0,
            right_id=1,
            left="A",
            right="B",
            parsed_path="trace/outputs/a1.json",
        ),
        _groupby_event(
            call_id="call-b",
            trace_id="b1",
            left_id=0,
            right_id=1,
            left="C",
            right="D",
            parsed_path="trace/outputs/b1.json",
        ),
        _groupby_event(
            call_id="call-a",
            trace_id="a2",
            left_id=0,
            right_id=2,
            left="A",
            right="E",
            parsed_path="trace/outputs/a2.json",
        ),
    ]
    _write_jsonl(run / "trace/events.jsonl", events)
    for name in ("a1", "a2", "b1"):
        target = run / f"trace/outputs/{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("false\n", encoding="utf-8")

    with pytest.raises(PairTraceError, match="interleaved semantic group"):
        analyze_sources([DirectorySource(run)])


def test_conflicting_groupby_label_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    first = _groupby_event(
        call_id="call",
        trace_id="first",
        left_id=0,
        right_id=1,
        left="A",
        right="B",
        parsed_path="trace/outputs/first.json",
    )
    second = {
        **first,
        "trace_id": "second",
        "parsed_output_path": "trace/outputs/second.json",
    }
    _write_jsonl(run / "trace/events.jsonl", [first, second])
    (run / "trace/outputs").mkdir(parents=True)
    (run / "trace/outputs/first.json").write_text("true\n", encoding="utf-8")
    (run / "trace/outputs/second.json").write_text("false\n", encoding="utf-8")

    with pytest.raises(PairTraceError, match="conflicting baseline labels"):
        analyze_sources([DirectorySource(run)])


def test_identical_group_replay_within_source_is_counted_once(tmp_path: Path) -> None:
    run = tmp_path / "run"
    pairs = ((0, 1, "A", "B"), (0, 2, "A", "C"), (1, 2, "B", "C"))
    events: list[dict[str, object]] = []
    for call_id in ("first-call", "replayed-call"):
        for index, (left_id, right_id, left, right) in enumerate(pairs):
            path = f"trace/outputs/{call_id}-{index}.json"
            events.append(
                _groupby_event(
                    call_id=call_id,
                    trace_id=f"{call_id}-{index}",
                    left_id=left_id,
                    right_id=right_id,
                    left=left,
                    right=right,
                    parsed_path=path,
                )
            )
            target = run / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("true\n" if index == 0 else "false\n", encoding="utf-8")
    _write_jsonl(run / "trace/events.jsonl", events)

    report = analyze_sources([DirectorySource(run, read_workers=8)])

    assert report["baseline"] == {
        "group_count": 1,
        "pair_count": 3,
        "positive_pair_count": 1,
    }
    assert report["sources"][0]["scanned_group_count"] == 2
    assert report["sources"][0]["accepted_group_count"] == 1
    assert report["sources"][0]["idempotent_replay_group_count"] == 1


def test_conflicting_group_replay_within_source_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    events: list[dict[str, object]] = []
    for call_id, label in (("first-call", True), ("replayed-call", False)):
        path = f"trace/outputs/{call_id}.json"
        events.append(
            _groupby_event(
                call_id=call_id,
                trace_id=call_id,
                left_id=0,
                right_id=1,
                left="A",
                right="B",
                parsed_path=path,
            )
        )
        target = run / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(label) + "\n", encoding="utf-8")
    _write_jsonl(run / "trace/events.jsonl", events)

    with pytest.raises(PairTraceError, match="conflicting pair order or labels"):
        analyze_sources([DirectorySource(run, read_workers=8)])


@pytest.mark.parametrize("read_workers", [1, 4])
def test_failed_attempt_replay_is_excluded_by_control_ledger(
    tmp_path: Path, read_workers: int
) -> None:
    run = tmp_path / "run"
    events: list[dict[str, object]] = []
    for call_id, label, timestamp in (
        ("failed-call", False, "20260803T143700000000Z"),
        ("completed-call", True, "20260803T150000000000Z"),
    ):
        path = f"trace/outputs/{call_id}.json"
        events.append(
            _groupby_event(
                call_id=call_id,
                trace_id=call_id,
                left_id=0,
                right_id=1,
                left="photo",
                right="posters",
                parsed_path=path,
                timestamp=timestamp,
            )
        )
        target = run / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(label) + "\n", encoding="utf-8")
    _write_jsonl(run / "trace/events.jsonl", events)
    _write_attempt_ledger(
        run,
        [
            {
                "timestamp": "2026-08-03T14:36:00+00:00",
                "event_type": "unit_started",
                "unit_id": "event:event-1",
                "attempt": 1,
            },
            {
                "timestamp": "2026-08-03T14:55:00+00:00",
                "event_type": "unit_failed",
                "unit_id": "event:event-1",
                "attempt": 1,
            },
            {
                "timestamp": "2026-08-03T14:59:00+00:00",
                "event_type": "unit_started",
                "unit_id": "event:event-1",
                "attempt": 2,
            },
            {
                "timestamp": "2026-08-03T15:18:00+00:00",
                "event_type": "unit_completed",
                "unit_id": "event:event-1",
                "attempt": 2,
            },
        ],
    )

    report = analyze_sources([DirectorySource(run, read_workers=read_workers)])

    assert report["baseline"] == {
        "group_count": 1,
        "pair_count": 1,
        "positive_pair_count": 1,
    }
    assert report["sources"][0]["scanned_group_count"] == 2
    assert report["sources"][0]["excluded_failed_attempt_group_count"] == 1
    assert report["sources"][0]["attempt_ledger_sha256"]


def test_failed_sem_join_replay_is_excluded_by_legacy_recovery(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    events: list[dict[str, object]] = []
    for call_id, labels in (
        ("failed-call", (False, False)),
        ("completed-call", (True, False)),
    ):
        for index, (right, label) in enumerate(zip(("old X", "old Y"), labels)):
            path = f"trace/outputs/{call_id}-{index}.json"
            events.append(
                _sem_join_event(
                    call_id=call_id,
                    trace_id=f"{call_id}-{index}",
                    left_id=0,
                    right_id=index,
                    left="new A",
                    right=right,
                    parsed_path=path,
                )
            )
            target = run / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(label) + "\n", encoding="utf-8")
    _write_jsonl(run / "trace/events.jsonl", events)
    _write_legacy_recovery(
        run,
        ranges=[{"start": 0, "end": 2, "count": 2}],
        trace_event_count=4,
    )

    report = analyze_sources([DirectorySource(run)], phase="add")

    assert report["baseline"] == {
        "group_count": 1,
        "pair_count": 2,
        "positive_pair_count": 1,
    }
    source = report["sources"][0]
    assert source["scanned_group_count"] == 2
    assert source["excluded_failed_attempt_group_count"] == 1
    assert source["excluded_legacy_recovery_group_count"] == 1
    assert source["legacy_recovery_sha256"]


def test_legacy_recovery_cannot_split_one_semantic_group(tmp_path: Path) -> None:
    run = tmp_path / "run"
    events: list[dict[str, object]] = []
    for index, right in enumerate(("old X", "old Y")):
        path = f"trace/outputs/pair-{index}.json"
        events.append(
            _sem_join_event(
                call_id="join-call",
                trace_id=f"join-{index}",
                left_id=0,
                right_id=index,
                left="new A",
                right=right,
                parsed_path=path,
            )
        )
        target = run / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("false\n", encoding="utf-8")
    _write_jsonl(run / "trace/events.jsonl", events)
    _write_legacy_recovery(
        run,
        ranges=[{"start": 0, "end": 1, "count": 1}],
        trace_event_count=2,
    )

    with pytest.raises(PairTraceError, match="crosses legacy recovery ranges"):
        list(DirectorySource(run).iter_groups(phase="add"))


def test_malformed_legacy_recovery_range_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_groupby_run(run, call_id="call", labels=[True, False, False])
    _write_legacy_recovery(
        run,
        ranges=[{"start": 0, "end": 2, "count": 1}],
        trace_event_count=3,
    )

    with pytest.raises(PairTraceError, match="legacy recovery range"):
        list(DirectorySource(run).iter_groups(phase="insertion"))


def test_control_ledger_takes_precedence_over_legacy_recovery(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _write_groupby_run(run, call_id="call", labels=[True, False, False])
    _write_attempt_ledger(run, [])
    _write_legacy_recovery(
        run,
        ranges=[{"start": 0, "end": 2, "count": 1}],
        trace_event_count=3,
    )

    report = analyze_sources([DirectorySource(run)])

    assert report["baseline"]["pair_count"] == 3
    assert report["sources"][0]["legacy_recovery_sha256"] == ""


def test_unclosed_attempt_ledger_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_groupby_run(run, call_id="call", labels=[True, False, False])
    _write_attempt_ledger(
        run,
        [
            {
                "timestamp": "2026-08-03T14:36:00+00:00",
                "event_type": "unit_started",
                "unit_id": "event:event-1",
                "attempt": 1,
            }
        ],
    )

    with pytest.raises(PairTraceError, match="attempt has no terminal event"):
        list(DirectorySource(run).iter_groups(phase="insertion"))


def test_overlapping_attempt_intervals_are_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_groupby_run(run, call_id="call", labels=[True, False, False])
    _write_attempt_ledger(
        run,
        [
            {
                "timestamp": "2026-08-03T14:36:00+00:00",
                "event_type": "unit_started",
                "unit_id": "event:event-1",
                "attempt": 1,
            },
            {
                "timestamp": "2026-08-03T14:40:00+00:00",
                "event_type": "unit_started",
                "unit_id": "event:event-1",
                "attempt": 2,
            },
            {
                "timestamp": "2026-08-03T14:45:00+00:00",
                "event_type": "unit_failed",
                "unit_id": "event:event-1",
                "attempt": 1,
            },
            {
                "timestamp": "2026-08-03T14:50:00+00:00",
                "event_type": "unit_completed",
                "unit_id": "event:event-1",
                "attempt": 2,
            },
        ],
    )

    with pytest.raises(PairTraceError, match="attempt intervals overlap"):
        list(DirectorySource(run).iter_groups(phase="insertion"))


def test_pair_trace_outside_attempt_intervals_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    path = "trace/outputs/call.json"
    event = _groupby_event(
        call_id="call",
        trace_id="call",
        left_id=0,
        right_id=1,
        left="A",
        right="B",
        parsed_path=path,
        timestamp="20260803T150000000000Z",
    )
    target = run / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("true\n", encoding="utf-8")
    _write_jsonl(run / "trace/events.jsonl", [event])
    _write_attempt_ledger(
        run,
        [
            {
                "timestamp": "2026-08-03T14:36:00+00:00",
                "event_type": "unit_started",
                "unit_id": "event:event-1",
                "attempt": 1,
            },
            {
                "timestamp": "2026-08-03T14:55:00+00:00",
                "event_type": "unit_completed",
                "unit_id": "event:event-1",
                "attempt": 1,
            },
        ],
    )

    with pytest.raises(PairTraceError, match="outside recorded attempt intervals"):
        list(DirectorySource(run).iter_groups(phase="insertion"))


def test_missing_artifact_is_not_guessed_as_false(tmp_path: Path) -> None:
    run = tmp_path / "run"
    event = _groupby_event(
        call_id="call",
        trace_id="missing",
        left_id=0,
        right_id=1,
        left="A",
        right="B",
        parsed_path="trace/outputs/missing.json",
    )
    _write_jsonl(run / "trace/events.jsonl", [event])

    with pytest.raises(PairTraceError, match="missing artifact"):
        analyze_sources([DirectorySource(run)])


def test_single_report_refuses_source_and_existing_paths(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    source = DirectorySource(run)

    with pytest.raises(PairTraceError, match="inside a source run"):
        _write_single_report(run / "report.json", "{}\n", [source])

    report = tmp_path / "report.json"
    _write_single_report(report, "{}\n", [source])
    assert report.read_text(encoding="utf-8") == "{}\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.json", "run"]
    with pytest.raises(FileExistsError, match="already exists"):
        _write_single_report(report, "{}\n", [source])


def test_scorer_and_strategy_must_be_configured_together() -> None:
    source = StaticSource([])

    with pytest.raises(ValueError, match="require a pair scorer"):
        analyze_sources([source], strategies=[CandidateStrategy(top_k=1)])
    with pytest.raises(ValueError, match="requires at least one"):
        analyze_sources([source], scorer=FakeScorer({}))


def test_embedding_batch_size_changes_only_physical_encode_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    encode_calls: list[dict[str, object]] = []

    class FakeSentenceTransformer:
        device = "cpu"

        def __init__(self, model: str, *, revision: str, device: str) -> None:
            assert (model, revision, device) == ("model", "revision", "cpu")

        def float(self) -> FakeSentenceTransformer:
            return self

        def encode(self, texts: Sequence[str], **kwargs: object) -> np.ndarray:
            encode_calls.append(dict(kwargs))
            vectors = {
                "A": (1.0, 0.0),
                "B": (0.9, 0.4358899),
                "C": (0.0, 1.0),
            }
            return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    strategies = build_strategies(top_ks=[1], thresholds=[0.8])
    reports = []
    for batch_size in (1, 64):
        scorer = SentenceTransformerCosineScorer(
            model="model",
            revision="revision",
            batch_size=batch_size,
        )
        reports.append(
            analyze_sources(
                [StaticSource([_manual_group("symmetric")])],
                strategies=strategies,
                scorer=scorer,
            )
        )

    assert reports[0]["baseline"] == reports[1]["baseline"]
    assert reports[0]["strategies"] == reports[1]["strategies"]
    assert [call["batch_size"] for call in encode_calls] == [1, 64]
    assert all(call["precision"] == "float32" for call in encode_calls)
    assert reports[0]["scorer"]["embedding_batch_size"] == 1
    assert reports[1]["scorer"]["embedding_batch_size"] == 64


@pytest.mark.parametrize("value", [0, -1])
def test_parallelism_arguments_must_be_positive(
    tmp_path: Path, value: int
) -> None:
    with pytest.raises(ValueError, match="read_workers"):
        DirectorySource(tmp_path, read_workers=value)
    with pytest.raises(ValueError, match="embedding batch size"):
        SentenceTransformerCosineScorer(
            model="model",
            revision="revision",
            batch_size=value,
        )
