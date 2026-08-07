"""Benchmark unit-attempt accounting independent of memory state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

MAX_UNIT_ATTEMPTS = 20
AttemptLineage = tuple[str, str, int, int]


class ArtifactContractError(ValueError):
    """Raised when persisted benchmark evidence violates its contract."""


class UnitAttemptExhausted(RuntimeError):
    """Raised before a benchmark unit would exceed its retry budget."""


def is_retryable_unit_error(error: BaseException) -> bool:
    """Return whether a transient provider failure may succeed on retry."""

    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 429}:
        return True
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return True
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "BadGatewayError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        )


class UnitAttemptStore:
    """Persist retry budgets while leaving state authority to checkpoints."""

    def __init__(self, control_dir: Path) -> None:
        self.control_dir = control_dir
        self.state_path = control_dir / "unit-attempts.json"
        self.events_path = control_dir / "events.jsonl"

    def begin(
        self,
        *,
        phase: str,
        unit_id: str,
        execution_attempt: int,
    ) -> int:
        """Begin one retryable operation without calling external resources."""

        state = self._read_state()
        units = self._units(state)
        key = self._key(phase, unit_id)
        current = units.get(key, {})
        failures = int(current.get("retryable_failure_count", 0))
        if failures >= MAX_UNIT_ATTEMPTS:
            raise UnitAttemptExhausted(
                f"{key} exhausted {MAX_UNIT_ATTEMPTS} attempts"
            )
        unit_attempt = failures + 1
        units[key] = {
            "phase": phase,
            "unit_id": unit_id,
            "retryable_failure_count": failures,
            "unit_attempt": unit_attempt,
            "last_status": "running",
            "execution_attempt": execution_attempt,
        }
        _write_json_atomic(self.state_path, state)
        self._append_event(
            "unit_attempt_started",
            phase,
            unit_id,
            execution_attempt,
            unit_attempt,
        )
        return unit_attempt

    def finish(
        self,
        *,
        phase: str,
        unit_id: str,
        execution_attempt: int,
        unit_attempt: int,
        status: str,
        error: BaseException | None = None,
    ) -> bool:
        """Finish a unit and return whether 20 retryable failures were used."""

        if status not in {"success", "failed"}:
            raise ValueError("unit attempt status must be success or failed")
        state = self._read_state()
        current = self._current(
            state,
            phase=phase,
            unit_id=unit_id,
            execution_attempt=execution_attempt,
            unit_attempt=unit_attempt,
        )
        current["last_status"] = status
        if status == "failed":
            current["retryable_failure_count"] = unit_attempt
        if error is not None:
            current["error_type"] = type(error).__name__
            current["error"] = str(error)
        _write_json_atomic(self.state_path, state)
        self._append_event(
            f"unit_attempt_{status}",
            phase,
            unit_id,
            execution_attempt,
            unit_attempt,
            error=error,
        )
        return status == "failed" and unit_attempt >= MAX_UNIT_ATTEMPTS

    def reconcile_success(self, lineage: AttemptLineage) -> None:
        """Mark a checkpointed or atomically published unit as successful."""

        phase, unit_id, execution_attempt, unit_attempt = lineage
        state = self._read_state()
        current = self._current(
            state,
            phase=phase,
            unit_id=unit_id,
            execution_attempt=execution_attempt,
            unit_attempt=unit_attempt,
        )
        if current.get("last_status") == "success":
            return
        current["last_status"] = "success"
        _write_json_atomic(self.state_path, state)
        self._append_event(
            "unit_attempt_reconciled_success",
            phase,
            unit_id,
            execution_attempt,
            unit_attempt,
        )

    def reconcile_many(self, lineages: Iterable[AttemptLineage]) -> None:
        """Reconcile all durable event or question lineages."""

        for lineage in lineages:
            self.reconcile_success(lineage)

    def successful_lineages(self) -> set[AttemptLineage]:
        """Return ledger entries already reconciled as durable successes."""

        if not self.state_path.is_file():
            return set()
        successful: set[AttemptLineage] = set()
        for unit in self._units(self._read_state()).values():
            if not isinstance(unit, Mapping) or unit.get("last_status") != "success":
                continue
            successful.add(
                (
                    str(unit.get("phase") or ""),
                    str(unit.get("unit_id") or ""),
                    int(unit.get("execution_attempt") or 0),
                    int(unit.get("unit_attempt") or 0),
                )
            )
        return successful

    def retryable_failure_count(self) -> int:
        """Return the number of failed unit attempts recorded for reporting."""

        if not self.events_path.is_file():
            return 0
        return sum(
            json.loads(line).get("event_type") == "unit_attempt_failed"
            for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"schema_version": 1, "units": {}}
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ArtifactContractError("unit attempt state must be an object")
        return state

    @staticmethod
    def _units(state: dict[str, Any]) -> dict[str, Any]:
        units = state.get("units")
        if not isinstance(units, dict):
            raise ArtifactContractError(
                "unit attempt state must contain an object of units"
            )
        return units

    def _current(
        self,
        state: dict[str, Any],
        *,
        phase: str,
        unit_id: str,
        execution_attempt: int,
        unit_attempt: int,
    ) -> dict[str, Any]:
        current = self._units(state).get(self._key(phase, unit_id))
        if (
            not isinstance(current, dict)
            or int(current.get("execution_attempt", 0)) != execution_attempt
            or int(current.get("unit_attempt", 0)) != unit_attempt
        ):
            raise ArtifactContractError(
                "unit attempt completion does not match current state"
            )
        return current

    def _append_event(
        self,
        event_type: str,
        phase: str,
        unit_id: str,
        execution_attempt: int,
        unit_attempt: int,
        *,
        error: BaseException | None = None,
    ) -> None:
        _append_jsonl(
            self.events_path,
            {
                "event_type": event_type,
                "phase": phase,
                "unit_id": unit_id,
                "execution_attempt": execution_attempt,
                "unit_attempt": unit_attempt,
                **(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    if error is not None
                    else {}
                ),
            },
        )

    @staticmethod
    def _key(phase: str, unit_id: str) -> str:
        return f"{phase}:{unit_id}"


__all__ = [
    "ArtifactContractError",
    "AttemptLineage",
    "MAX_UNIT_ATTEMPTS",
    "UnitAttemptExhausted",
    "UnitAttemptStore",
    "is_retryable_unit_error",
]
