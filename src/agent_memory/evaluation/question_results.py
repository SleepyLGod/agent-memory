"""Atomic benchmark question results used as the resume authority."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_memory.evaluation.recovery import (
    ArtifactContractError,
    AttemptLineage,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(item) for item in value]
    if type(value).__module__.startswith("neo4j.time"):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _json_value(row),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    os.replace(temporary, path)


class AtomicQuestionStore:
    """Publish and rebuild one case's question artifacts atomically."""

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = case_dir
        self.root = case_dir / "question-results"

    def completed(
        self,
        *,
        question_id: str,
        contract_fingerprint: str,
        scorer_ids: Sequence[str],
    ) -> bool:
        """Return whether a complete result matches the current contract."""

        complete = self._directory(question_id) / "complete.json"
        if not complete.is_file():
            return False
        metadata = self._read_mapping(complete)
        if metadata.get("question_id") != question_id:
            raise ArtifactContractError("question result identity mismatch")
        if metadata.get("contract_fingerprint") != contract_fingerprint:
            raise ArtifactContractError("question result contract mismatch")
        if metadata.get("scorer_ids") != list(scorer_ids):
            raise ArtifactContractError(
                "question result scorer contract mismatch"
            )
        return True

    def lineage(self, question_id: str) -> AttemptLineage | None:
        """Return the durable lineage for one completed question."""

        complete = self._directory(question_id) / "complete.json"
        if not complete.is_file():
            return None
        metadata = self._read_mapping(complete)
        return (
            "question",
            question_id,
            int(metadata.get("execution_attempt") or 0),
            int(metadata.get("unit_attempt") or 0),
        )

    def publish(
        self,
        *,
        question_id: str,
        contract_fingerprint: str,
        retrieval: Mapping[str, Any],
        answer: Mapping[str, Any],
        grades: Sequence[Mapping[str, Any]],
        execution_attempt: int,
        unit_attempt: int,
    ) -> None:
        """Publish retrieval, answer, and all grades as one result."""

        final = self._directory(question_id)
        scorer_ids = [str(row["scorer_id"]) for row in grades]
        if final.exists():
            if not self.completed(
                question_id=question_id,
                contract_fingerprint=contract_fingerprint,
                scorer_ids=scorer_ids,
            ):
                raise ArtifactContractError(
                    "question result directory is incomplete"
                )
            return
        staging = final.parent / f".{final.name}.{uuid4().hex}.staging"
        _write_json(staging / "retrieval.json", dict(retrieval))
        _write_json(staging / "answer.json", dict(answer))
        _write_json(staging / "grades.json", [dict(row) for row in grades])
        _write_json(
            staging / "complete.json",
            {
                "question_id": question_id,
                "contract_fingerprint": contract_fingerprint,
                "scorer_ids": scorer_ids,
                "execution_attempt": execution_attempt,
                "unit_attempt": unit_attempt,
            },
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)

    def rebuild_jsonl(self, question_ids: Sequence[str]) -> None:
        """Rebuild compatibility JSONL files from complete results only."""

        retrievals: list[Mapping[str, Any]] = []
        answers: list[Mapping[str, Any]] = []
        grades: list[Mapping[str, Any]] = []
        for question_id in question_ids:
            directory = self._directory(question_id)
            if not (directory / "complete.json").is_file():
                continue
            retrievals.append(self._read_mapping(directory / "retrieval.json"))
            answers.append(self._read_mapping(directory / "answer.json"))
            grade_rows = json.loads(
                (directory / "grades.json").read_text(encoding="utf-8")
            )
            if not isinstance(grade_rows, list) or not all(
                isinstance(row, Mapping) for row in grade_rows
            ):
                raise ArtifactContractError(
                    "question grades must be a list of objects"
                )
            grades.extend(grade_rows)
        _write_jsonl_atomic(self.case_dir / "retrieval.jsonl", retrievals)
        _write_jsonl_atomic(self.case_dir / "answers.jsonl", answers)
        _write_jsonl_atomic(self.case_dir / "grades.jsonl", grades)

    def completion_metadata(self, question_id: str) -> Mapping[str, Any]:
        """Return immutable completion metadata for metrics."""

        complete = self._directory(question_id) / "complete.json"
        return self._read_mapping(complete) if complete.is_file() else {}

    def _directory(self, question_id: str) -> Path:
        digest = sha256(question_id.encode("utf-8")).hexdigest()[:12]
        return self.root / digest

    @staticmethod
    def _read_mapping(path: Path) -> Mapping[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ArtifactContractError(
                f"question result must be an object: {path.name}"
            )
        return value


__all__ = ["AtomicQuestionStore"]
