"""Portable normalized benchmark bundles shared by all memory systems."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .types import BenchmarkCase, BenchmarkEvent, BenchmarkQuestion


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


@dataclass(frozen=True)
class BenchmarkBundle:
    """Pinned normalized cases and the evidence needed to reproduce them."""

    benchmark_id: str
    dataset_revision: str
    dataset_sha256: str
    cases: tuple[BenchmarkCase, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if not self.dataset_revision:
            raise ValueError("dataset_revision must be non-empty")
        if not self.dataset_sha256:
            raise ValueError("dataset_sha256 must be non-empty")
        if not self.cases:
            raise ValueError("a benchmark bundle must contain at least one case")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case IDs must be unique")

    @property
    def fingerprint(self) -> str:
        """Fingerprint the full normalized bundle, including evaluation labels."""

        return _fingerprint(self._payload(include_labels=True))

    @property
    def policy_input_fingerprint(self) -> str:
        """Fingerprint only the events that memory systems are allowed to ingest."""

        return _fingerprint(self._payload(include_labels=False))

    def _payload(self, *, include_labels: bool) -> dict[str, Any]:
        cases: list[dict[str, Any]] = []
        for case in self.cases:
            case_payload: dict[str, Any] = {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "events": [asdict(event) for event in case.events],
            }
            if include_labels:
                case_payload["questions"] = [
                    asdict(question) for question in case.questions
                ]
                case_payload["metadata"] = dict(case.metadata)
            cases.append(case_payload)
        return {
            "benchmark_id": self.benchmark_id,
            "dataset_revision": self.dataset_revision,
            "dataset_sha256": self.dataset_sha256,
            "cases": cases,
            "metadata": dict(self.metadata) if include_labels else {},
        }


def write_bundle(bundle: BenchmarkBundle, directory: Path) -> None:
    """Write one canonical bundle without replacing an existing run directory."""

    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"bundle directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    event_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for case in bundle.cases:
        event_ids: list[str] = []
        question_ids: list[str] = []
        for event in case.events:
            row = asdict(event)
            row["case_id"] = case.case_id
            event_rows.append(row)
            event_ids.append(event.event_id)
        for question in case.questions:
            row = asdict(question)
            row["case_id"] = case.case_id
            question_rows.append(row)
            question_ids.append(question.question_id)
        case_rows.append(
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "event_ids": event_ids,
                "question_ids": question_ids,
                "metadata": dict(case.metadata),
            }
        )

    manifest = {
        "schema_version": 1,
        "benchmark_id": bundle.benchmark_id,
        "dataset_revision": bundle.dataset_revision,
        "dataset_sha256": bundle.dataset_sha256,
        "case_count": len(bundle.cases),
        "event_count": len(event_rows),
        "question_count": len(question_rows),
        "fingerprint": bundle.fingerprint,
        "policy_input_fingerprint": bundle.policy_input_fingerprint,
        "metadata": dict(bundle.metadata),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(directory / "cases.jsonl", case_rows)
    _write_jsonl(directory / "events.jsonl", event_rows)
    _write_jsonl(directory / "questions.jsonl", question_rows)


def read_bundle(directory: Path) -> BenchmarkBundle:
    """Read and validate a canonical bundle created by :func:`write_bundle`."""

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported benchmark bundle schema version")

    event_rows = _read_jsonl(directory / "events.jsonl")
    question_rows = _read_jsonl(directory / "questions.jsonl")
    case_rows = _read_jsonl(directory / "cases.jsonl")
    events_by_id: dict[str, BenchmarkEvent] = {}
    questions_by_id: dict[str, BenchmarkQuestion] = {}

    for row in event_rows:
        case_id = row.pop("case_id", None)
        event = BenchmarkEvent(**row)
        if case_id != event.sample_id:
            raise ValueError(f"event {event.event_id!r} has inconsistent case_id")
        if event.event_id in events_by_id:
            raise ValueError(f"duplicate event ID {event.event_id!r}")
        events_by_id[event.event_id] = event

    for row in question_rows:
        case_id = row.pop("case_id", None)
        row["evidence_event_ids"] = tuple(row.get("evidence_event_ids", ()))
        question = BenchmarkQuestion(**row)
        if case_id != question.sample_id:
            raise ValueError(
                f"question {question.question_id!r} has inconsistent case_id"
            )
        if question.question_id in questions_by_id:
            raise ValueError(f"duplicate question ID {question.question_id!r}")
        questions_by_id[question.question_id] = question

    cases: list[BenchmarkCase] = []
    used_event_ids: set[str] = set()
    used_question_ids: set[str] = set()
    for row in case_rows:
        event_ids = tuple(row.get("event_ids", ()))
        question_ids = tuple(row.get("question_ids", ()))
        try:
            case_events = tuple(events_by_id[event_id] for event_id in event_ids)
            case_questions = tuple(
                questions_by_id[question_id] for question_id in question_ids
            )
        except KeyError as exc:
            raise ValueError(f"case references unknown row {exc.args[0]!r}") from exc
        if used_event_ids.intersection(event_ids):
            raise ValueError("an event cannot belong to multiple benchmark cases")
        if used_question_ids.intersection(question_ids):
            raise ValueError("a question cannot belong to multiple benchmark cases")
        used_event_ids.update(event_ids)
        used_question_ids.update(question_ids)
        cases.append(
            BenchmarkCase(
                case_id=row["case_id"],
                task_id=row["task_id"],
                events=case_events,
                questions=case_questions,
                metadata=row.get("metadata", {}),
            )
        )

    if used_event_ids != set(events_by_id):
        raise ValueError("bundle contains events not referenced by a case")
    if used_question_ids != set(questions_by_id):
        raise ValueError("bundle contains questions not referenced by a case")

    bundle = BenchmarkBundle(
        benchmark_id=manifest["benchmark_id"],
        dataset_revision=manifest["dataset_revision"],
        dataset_sha256=manifest["dataset_sha256"],
        cases=tuple(cases),
        metadata=manifest.get("metadata", {}),
    )
    if bundle.fingerprint != manifest.get("fingerprint"):
        raise ValueError("benchmark bundle fingerprint does not match manifest")
    if bundle.policy_input_fingerprint != manifest.get("policy_input_fingerprint"):
        raise ValueError("benchmark policy-input fingerprint does not match manifest")
    return bundle
