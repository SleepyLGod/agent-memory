"""Pure attempt-lineage classification for benchmark metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DurableUnit = tuple[str, str, str, int, int]


def classify_attempt_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    durable_units: set[DurableUnit],
    authoritative_cases: set[str],
) -> list[dict[str, Any]]:
    """Mark actual calls as final-successful or recovery overhead."""

    classified: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        case_id = str(row.get("case_id") or "")
        phase = str(row.get("phase") or "")
        unit_phase = _unit_phase(phase)
        unit_id = str(
            row.get("event_id")
            if unit_phase == "insertion"
            else row.get("question_id")
            or ""
        )
        lineage = (
            case_id,
            unit_phase,
            unit_id,
            int(row.get("execution_attempt") or 0),
            int(row.get("unit_attempt") or 0),
        )
        durable = case_id not in authoritative_cases or lineage in durable_units
        row["successful_path"] = durable and row.get("status") != "error"
        classified.append(row)
    return classified


def partition_attempt_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split classified rows into final-successful and recovery attempts."""

    final: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        (final if row.get("successful_path") else recovery).append(row)
    return final, recovery


def _unit_phase(phase: str) -> str:
    if phase in {"insertion", "consolidation", "checkpoint"}:
        return "insertion"
    if phase in {"retrieval", "answering", "grading"}:
        return "question"
    return phase


__all__ = [
    "DurableUnit",
    "classify_attempt_rows",
    "partition_attempt_rows",
]
