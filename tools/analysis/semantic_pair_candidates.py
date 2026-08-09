"""Read semantic-pair traces and evaluate in-memory candidate strategies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, Protocol

SOURCE_PREFIXES = ("dir:", "tar:")
PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)_(earlier|later)\}")
COMPRESSED_ARCHIVE_MAGIC = (
    b"\x1f\x8b",  # gzip
    b"BZh",  # bzip2
    b"\xfd7zXZ\x00",  # xz
    b"\x28\xb5\x2f\xfd",  # zstd
)


class PairTraceError(ValueError):
    """Raised when semantic-pair evidence cannot be interpreted safely."""


@dataclass(frozen=True)
class PairRecord:
    """One pair and its exact-execution baseline decision."""

    pair_id: str
    left_id: str
    right_id: str
    left: str
    right: str
    baseline_match: bool


@dataclass(frozen=True)
class PairGroup:
    """One logical semantic operator invocation."""

    group_id: str
    operator: str
    direction: str
    source: str
    case_id: str
    session_id: str
    event_id: str
    query_digest: str
    pairs: tuple[PairRecord, ...]


@dataclass(frozen=True)
class AttemptInterval:
    """One completed or failed benchmark event attempt."""

    start: datetime
    end: datetime
    attempt: int
    status: str


@dataclass(frozen=True)
class LegacyRecovery:
    """Validated zero-based half-open trace ranges from an older runner."""

    ranges: tuple[tuple[int, int], ...]
    trace_event_count: int


@dataclass(frozen=True)
class CurrentAttemptLedger:
    """Successful insertion lineages from case-local harness state."""

    case_ids: frozenset[str]
    successful_lineages: frozenset[tuple[str, str, str, int, int]]
    digest: str


@dataclass(frozen=True)
class CandidateStrategy:
    """A threshold, top-k, or combined candidate strategy."""

    top_k: int | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.top_k is None and self.threshold is None:
            raise ValueError("candidate strategy requires top_k or threshold")
        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k < 1
        ):
            raise ValueError("candidate top_k must be a positive integer")
        if self.threshold is not None and (
            not isinstance(self.threshold, (int, float))
            or isinstance(self.threshold, bool)
            or not math.isfinite(float(self.threshold))
        ):
            raise ValueError("candidate threshold must be a finite number")

    @property
    def strategy_id(self) -> str:
        """Return a stable human-readable strategy identifier."""

        parts: list[str] = []
        if self.top_k is not None:
            parts.append(f"top-k:{self.top_k}")
        if self.threshold is not None:
            parts.append(f"threshold:{self.threshold:g}")
        return "+".join(parts)


@dataclass
class SourceStats:
    """Read-only provenance collected while scanning one source."""

    source: str
    requested_read_workers: int = 1
    effective_read_workers: int = 1
    events_sha256: str = ""
    event_count: int = 0
    scanned_group_count: int = 0
    accepted_group_count: int = 0
    idempotent_replay_group_count: int = 0
    excluded_failed_attempt_group_count: int = 0
    attempt_ledger_sha256: str = ""
    excluded_legacy_recovery_group_count: int = 0
    legacy_recovery_sha256: str = ""


class PairScorer(Protocol):
    """Assign one finite similarity score to each pair in a group."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Describe the scorer used for the report."""

        ...

    def score(self, group: PairGroup) -> Sequence[float]:
        """Score pairs in input order without persisting embeddings."""

        ...


class PairSource(Protocol):
    """Stream pair groups from an immutable artifact source."""

    description: str
    stats: SourceStats

    def iter_groups(self, *, phase: str) -> Iterator[PairGroup]:
        """Yield complete groups while leaving source artifacts untouched."""

        ...

    def contains_output(self, output: Path) -> bool:
        """Return whether an output path would modify source evidence."""

        ...


class DirectorySource:
    """Stream pair traces from one run directory."""

    def __init__(self, root: Path, *, read_workers: int = 1) -> None:
        _validate_positive_integer(read_workers, name="read_workers")
        self.root = root.resolve()
        self.description = f"dir:{self.root}"
        self.read_workers = read_workers
        self.stats = SourceStats(
            source=self.description,
            requested_read_workers=read_workers,
            effective_read_workers=read_workers,
        )

    def iter_groups(self, *, phase: str) -> Iterator[PairGroup]:
        """Yield contiguous semantic groups from events.jsonl."""

        if self.read_workers == 1:
            yield from self._iter_groups(phase=phase, executor=None)
            return
        with ThreadPoolExecutor(max_workers=self.read_workers) as executor:
            yield from self._iter_groups(phase=phase, executor=executor)

    def _iter_groups(
        self,
        *,
        phase: str,
        executor: ThreadPoolExecutor | None,
    ) -> Iterator[PairGroup]:
        """Use one ordered reader pool for the complete directory source."""

        current_attempts = self._read_current_attempt_ledger()
        if current_attempts is not None:
            self.stats.attempt_ledger_sha256 = current_attempts.digest
        control_ledger_exists = (
            self.root / "checkpoint/control-events.jsonl"
        ).is_file()
        attempt_intervals = self._read_attempt_intervals()
        legacy_recovery = (
            None if control_ledger_exists else self._read_legacy_recovery()
        )
        events_path = self.root / "trace/events.jsonl"
        try:
            handle = events_path.open("rb")
        except FileNotFoundError as error:
            raise PairTraceError(f"missing artifact: {events_path}") from error

        events_digest = hashlib.sha256()
        pending_call_id: str | None = None
        pending_events: list[dict[str, Any]] = []
        pending_trace_indices: list[int] = []
        closed_call_ids: set[str] = set()
        group_evidence_digests: dict[str, str] = {}
        unproven_attempts: dict[tuple[str, str, str], tuple[int, int]] = {}

        def is_new_group(group: PairGroup) -> bool:
            evidence_digest = _group_evidence_digest(group)
            prior_digest = group_evidence_digests.get(group.group_id)
            if prior_digest is None:
                group_evidence_digests[group.group_id] = evidence_digest
                return True
            if prior_digest != evidence_digest:
                raise PairTraceError(
                    "replayed semantic group has conflicting pair order or labels: "
                    f"{group.group_id}"
                )
            self.stats.idempotent_replay_group_count += 1
            return False

        def group_status(
            events: Sequence[Mapping[str, Any]], trace_indices: Sequence[int]
        ) -> str | None:
            statuses = {
                _event_attempt_status(
                    event,
                    trace_index=trace_index,
                    current_attempts=current_attempts,
                    attempt_intervals=attempt_intervals,
                    legacy_recovery=legacy_recovery,
                    unproven_attempts=unproven_attempts,
                )
                for event, trace_index in zip(events, trace_indices, strict=True)
            }
            if len(statuses) != 1:
                if legacy_recovery is not None:
                    raise PairTraceError(
                        "semantic group crosses legacy recovery ranges"
                    )
                raise PairTraceError("semantic group crosses recorded attempt provenance")
            return statuses.pop()

        def flush_group() -> PairGroup | None:
            nonlocal pending_call_id, pending_events, pending_trace_indices
            if not pending_events:
                return None
            self.stats.scanned_group_count += 1
            status = group_status(pending_events, pending_trace_indices)
            if status in {"failed", "legacy-failed"}:
                self.stats.excluded_failed_attempt_group_count += 1
                if status == "legacy-failed":
                    self.stats.excluded_legacy_recovery_group_count += 1
                pending_call_id = None
                pending_events = []
                pending_trace_indices = []
                return None
            group = _pair_decision_group(
                pending_events,
                source=self.description,
                read_bytes=self._read_bytes,
                executor=executor,
            )
            pending_call_id = None
            pending_events = []
            pending_trace_indices = []
            return group if is_new_group(group) else None

        with handle:
            for trace_index, raw_line in enumerate(handle):
                events_digest.update(raw_line)
                self.stats.event_count += 1
                event = _parse_json_line(
                    raw_line,
                    source=self.description,
                    line_number=trace_index + 1,
                )
                kind = _pair_event_kind(event, phase=phase)
                if kind == "pair_decision":
                    call_id = _required_string(event, "operator_call_id")
                    if pending_call_id is None:
                        if call_id in closed_call_ids:
                            raise PairTraceError(
                                f"interleaved semantic group in {self.description}: {call_id}"
                            )
                        pending_call_id = call_id
                    elif call_id != pending_call_id:
                        previous_call_id = pending_call_id
                        group = flush_group()
                        closed_call_ids.add(previous_call_id)
                        if group is not None:
                            yield group
                        if call_id in closed_call_ids:
                            raise PairTraceError(
                                f"interleaved semantic group in {self.description}: {call_id}"
                            )
                        pending_call_id = call_id
                    pending_events.append(event)
                    pending_trace_indices.append(trace_index)
                    continue
                if kind == "sem_filter":
                    previous_call_id = pending_call_id
                    group = flush_group()
                    if previous_call_id is not None:
                        closed_call_ids.add(previous_call_id)
                    if group is not None:
                        yield group
                    self.stats.scanned_group_count += 1
                    status = group_status((event,), (trace_index,))
                    if status in {"failed", "legacy-failed"}:
                        self.stats.excluded_failed_attempt_group_count += 1
                        if status == "legacy-failed":
                            self.stats.excluded_legacy_recovery_group_count += 1
                        continue
                    filter_group = _sem_filter_group(
                        event,
                        source=self.description,
                        read_bytes=self._read_bytes,
                    )
                    if is_new_group(filter_group):
                        yield filter_group

        previous_call_id = pending_call_id
        group = flush_group()
        if previous_call_id is not None:
            closed_call_ids.add(previous_call_id)
        if group is not None:
            yield group
        self.stats.events_sha256 = events_digest.hexdigest()
        if (
            legacy_recovery is not None
            and legacy_recovery.trace_event_count > self.stats.event_count
        ):
            raise PairTraceError(
                "legacy recovery trace_event_count exceeds events.jsonl length"
            )

    def _read_current_attempt_ledger(self) -> CurrentAttemptLedger | None:
        cases_root = self.root / "cases"
        if not cases_root.is_dir():
            return None
        artifacts: list[tuple[str, bytes, bytes]] = []
        for state_path in sorted(cases_root.glob("*/control/unit-attempts.json")):
            case_dir = state_path.parent.parent
            relative_case_dir = case_dir.relative_to(self.root).as_posix()
            artifacts.append(
                (
                    relative_case_dir,
                    self._read_bytes(
                        (case_dir / "case.json").relative_to(self.root).as_posix()
                    ),
                    self._read_bytes(state_path.relative_to(self.root).as_posix()),
                )
            )
        if not artifacts:
            return None
        return _parse_current_attempt_ledger(artifacts, source=self.description)

    def _read_attempt_intervals(self) -> dict[str, tuple[AttemptInterval, ...]]:
        ledger_path = self.root / "checkpoint/control-events.jsonl"
        if not ledger_path.is_file():
            return {}
        digest = hashlib.sha256()
        starts: dict[tuple[str, int], datetime] = {}
        intervals: dict[str, list[AttemptInterval]] = defaultdict(list)
        with ledger_path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                digest.update(raw_line)
                event = _parse_json_line(
                    raw_line,
                    source=f"control:{ledger_path}",
                    line_number=line_number,
                )
                event_type = str(event.get("event_type") or "")
                unit_id = str(event.get("unit_id") or "")
                if not unit_id.startswith("event:") or event_type not in {
                    "unit_started",
                    "unit_completed",
                    "unit_failed",
                }:
                    continue
                event_id = unit_id.removeprefix("event:")
                attempt = event.get("attempt")
                if (
                    not event_id
                    or not isinstance(attempt, int)
                    or isinstance(attempt, bool)
                    or attempt < 1
                ):
                    raise PairTraceError(
                        f"invalid event attempt at control:{ledger_path}:{line_number}"
                    )
                timestamp = _parse_timestamp(
                    event.get("timestamp"),
                    source=f"control:{ledger_path}:{line_number}",
                )
                key = (event_id, attempt)
                if event_type == "unit_started":
                    if key in starts:
                        raise PairTraceError(
                            f"event attempt starts more than once: {event_id}:{attempt}"
                        )
                    starts[key] = timestamp
                    continue
                start = starts.pop(key, None)
                if start is None:
                    raise PairTraceError(
                        f"event attempt terminal has no start: {event_id}:{attempt}"
                    )
                if timestamp <= start:
                    raise PairTraceError(
                        f"event attempt terminal precedes its start: {event_id}:{attempt}"
                    )
                intervals[event_id].append(
                    AttemptInterval(
                        start=start,
                        end=timestamp,
                        attempt=attempt,
                        status=(
                            "completed"
                            if event_type == "unit_completed"
                            else "failed"
                        ),
                    )
                )
        if starts:
            event_id, attempt = sorted(starts)[0]
            raise PairTraceError(
                f"event attempt has no terminal event: {event_id}:{attempt}"
            )
        for event_id, values in intervals.items():
            values.sort(key=lambda value: (value.start, value.end, value.attempt))
            for previous, current in zip(values, values[1:], strict=False):
                if current.start <= previous.end:
                    raise PairTraceError(
                        f"event attempt intervals overlap: {event_id}"
                    )
        if not self.stats.attempt_ledger_sha256:
            self.stats.attempt_ledger_sha256 = digest.hexdigest()
        return {
            event_id: tuple(values) for event_id, values in intervals.items()
        }

    def _read_legacy_recovery(self) -> LegacyRecovery | None:
        recovery_path = self.root / "diagnostics/recovery.json"
        if not recovery_path.is_file():
            return None
        try:
            raw_value = recovery_path.read_bytes()
            value = json.loads(raw_value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PairTraceError(
                f"invalid legacy recovery artifact: {recovery_path}"
            ) from error
        if not isinstance(value, dict):
            raise PairTraceError("legacy recovery artifact must be an object")
        trace_event_count = _required_nonnegative_integer(
            value.get("trace_event_count"), name="legacy recovery trace_event_count"
        )
        rows = value.get("excluded_trace_event_ranges")
        if not isinstance(rows, list):
            raise PairTraceError("legacy recovery ranges must be a list")
        ranges: list[tuple[int, int]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise PairTraceError("legacy recovery range must be an object")
            start = _required_nonnegative_integer(
                row.get("start"), name="legacy recovery range start"
            )
            end = _required_nonnegative_integer(
                row.get("end"), name="legacy recovery range end"
            )
            count = _required_nonnegative_integer(
                row.get("count"), name="legacy recovery range count"
            )
            if start >= end or count != end - start or end > trace_event_count:
                raise PairTraceError("invalid legacy recovery range")
            ranges.append((start, end))
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:], strict=False):
            if current[0] < previous[1]:
                raise PairTraceError("legacy recovery ranges overlap")
        excluded_count = _required_nonnegative_integer(
            value.get("excluded_trace_event_count"),
            name="legacy recovery excluded_trace_event_count",
        )
        if excluded_count != sum(end - start for start, end in ranges):
            raise PairTraceError("legacy recovery excluded count does not match ranges")
        self.stats.legacy_recovery_sha256 = hashlib.sha256(raw_value).hexdigest()
        return LegacyRecovery(tuple(ranges), trace_event_count)

    def _read_bytes(self, relative_path: str) -> bytes:
        target = (self.root / _safe_relative_path(relative_path)).resolve()
        if not target.is_relative_to(self.root):
            raise PairTraceError(f"artifact path escapes run root: {relative_path!r}")
        try:
            return target.read_bytes()
        except FileNotFoundError as error:
            raise PairTraceError(f"missing artifact: {target}") from error
        except OSError as error:
            raise PairTraceError(f"cannot read artifact: {target}") from error

    def contains_output(self, output: Path) -> bool:
        """Protect the complete run directory from analyzer writes."""

        return output.resolve().is_relative_to(self.root)


class TarSource:
    """Read Mem0-style pairwise sem_filter evidence directly from an uncompressed tar."""

    def __init__(
        self, archive: Path, member_root: str, *, read_workers: int = 1
    ) -> None:
        _validate_positive_integer(read_workers, name="read_workers")
        self.archive = archive.resolve()
        self.member_root = _safe_member_root(member_root)
        self.description = f"tar:{self.archive}::{self.member_root}"
        self.stats = SourceStats(
            source=self.description,
            requested_read_workers=read_workers,
            effective_read_workers=1,
        )
        _validate_uncompressed_tar(self.archive)

    def iter_groups(self, *, phase: str) -> Iterator[PairGroup]:
        """Scan twice: discover snapshot references, then read only those members."""

        events_name = self._member_name("trace/events.jsonl")
        events: list[dict[str, Any]] = []
        events_digest = hashlib.sha256()
        found_events = False
        attempt_artifacts: dict[str, dict[str, bytes]] = defaultdict(dict)
        with self._open_stream() as archive:
            for member in archive:
                artifact_key = self._attempt_artifact_key(member.name)
                if member.name != events_name and artifact_key is None:
                    continue
                if not member.isfile():
                    raise PairTraceError(f"tar member is not a file: {member.name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise PairTraceError(f"cannot read tar member: {member.name}")
                if artifact_key is not None:
                    case_root, kind = artifact_key
                    if kind in attempt_artifacts[case_root]:
                        raise PairTraceError(
                            f"duplicate tar attempt artifact: {member.name}"
                        )
                    attempt_artifacts[case_root][kind] = handle.read()
                    continue
                if found_events:
                    raise PairTraceError(f"invalid tar events member: {events_name}")
                found_events = True
                for line_number, raw_line in enumerate(handle, 1):
                    events_digest.update(raw_line)
                    self.stats.event_count += 1
                    event = _parse_json_line(
                        raw_line,
                        source=self.description,
                        line_number=line_number,
                    )
                    kind = _pair_event_kind(event, phase=phase)
                    if kind == "pair_decision":
                        raise PairTraceError(
                            "tar pair-decision traces use external per-pair labels; "
                            "analyze them from a directory source instead"
                        )
                    if kind == "sem_filter":
                        events.append(event)
        if not found_events:
            raise PairTraceError(f"missing tar member: {events_name}")
        self.stats.events_sha256 = events_digest.hexdigest()

        current_attempts = self._parse_attempt_artifacts(attempt_artifacts)
        if current_attempts is not None:
            self.stats.attempt_ledger_sha256 = current_attempts.digest
        unproven_attempts: dict[tuple[str, str, str], tuple[int, int]] = {}
        selected_events: list[dict[str, Any]] = []
        for trace_index, event in enumerate(events):
            self.stats.scanned_group_count += 1
            status = _event_attempt_status(
                event,
                trace_index=trace_index,
                current_attempts=current_attempts,
                attempt_intervals={},
                legacy_recovery=None,
                unproven_attempts=unproven_attempts,
            )
            if status == "failed":
                self.stats.excluded_failed_attempt_group_count += 1
                continue
            selected_events.append(event)
        events = selected_events

        references: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for index, event in enumerate(events):
            references[
                self._member_name(_required_string(event, "input_snapshot_path"))
            ].append((index, "input"))
            references[
                self._member_name(_required_string(event, "output_snapshot_path"))
            ].append((index, "output"))

        snapshots: dict[int, dict[str, bytes]] = defaultdict(dict)
        completed: set[int] = set()
        stable_group_ids: set[str] = set()
        with self._open_stream() as archive:
            for member in archive:
                targets = references.get(member.name)
                if not targets:
                    continue
                if not member.isfile():
                    raise PairTraceError(f"tar member is not a file: {member.name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise PairTraceError(f"cannot read tar member: {member.name}")
                value = handle.read()
                for index, side in targets:
                    snapshots[index][side] = value
                    if set(snapshots[index]) != {"input", "output"}:
                        continue
                    group = _sem_filter_group_from_bytes(
                        events[index],
                        source=self.description,
                        input_value=snapshots[index]["input"],
                        output_value=snapshots[index]["output"],
                    )
                    if group.group_id in stable_group_ids:
                        raise PairTraceError(
                            f"semantic group repeats within source {self.description}: "
                            f"{group.group_id}"
                        )
                    stable_group_ids.add(group.group_id)
                    completed.add(index)
                    del snapshots[index]
                    yield group

        missing = sorted(set(range(len(events))) - completed)
        if missing:
            event = events[missing[0]]
            raise PairTraceError(
                "missing tar snapshot for semantic group: "
                f"{event.get('operator_call_id') or event.get('trace_id') or missing[0]}"
            )

    def _attempt_artifact_key(self, member_name: str) -> tuple[str, str] | None:
        prefix = f"{self.member_root}/"
        if not member_name.startswith(prefix):
            return None
        parts = PurePosixPath(member_name.removeprefix(prefix)).parts
        if len(parts) == 3 and parts[0] == "cases" and parts[2] == "case.json":
            return "/".join(parts[:2]), "case"
        if (
            len(parts) == 4
            and parts[0] == "cases"
            and parts[2:] == ("control", "unit-attempts.json")
        ):
            return "/".join(parts[:2]), "state"
        return None

    def _parse_attempt_artifacts(
        self, artifacts: Mapping[str, Mapping[str, bytes]]
    ) -> CurrentAttemptLedger | None:
        rows: list[tuple[str, bytes, bytes]] = []
        for case_root, values in sorted(artifacts.items()):
            state = values.get("state")
            if state is None:
                continue
            case = values.get("case")
            if case is None:
                raise PairTraceError(
                    f"missing tar case artifact for attempt state: {case_root}"
                )
            rows.append((case_root, case, state))
        if not rows:
            return None
        return _parse_current_attempt_ledger(rows, source=self.description)

    def _member_name(self, relative_path: str) -> str:
        return f"{self.member_root}/{_safe_relative_path(relative_path).as_posix()}"

    def _open_stream(self) -> tarfile.TarFile:
        try:
            return tarfile.open(self.archive, mode="r|")
        except (FileNotFoundError, tarfile.TarError) as error:
            raise PairTraceError(
                f"cannot stream tar archive: {self.archive}"
            ) from error

    def contains_output(self, output: Path) -> bool:
        """Protect the source archive itself from overwrite."""

        return output.resolve() == self.archive


class SentenceTransformerCosineScorer:
    """Compute cosine scores in memory for one group at a time."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        if not model or not revision or not device:
            raise ValueError("embedding model, revision, and device must be non-empty")
        _validate_positive_integer(batch_size, name="embedding batch size")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "embedding analysis requires the sentence-transformers dependency"
            ) from error
        self.model = model
        self.revision = revision
        self.requested_device = device
        self.batch_size = batch_size
        self._dimensions: int | None = None
        _configure_float32_execution(device)
        self._model = SentenceTransformer(model, revision=revision, device=device)
        self._model.float()
        self.device = str(self._model.device)

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return reproducibility fields without persisting vectors."""

        return {
            "kind": "sentence-transformers-cosine",
            "model": self.model,
            "revision": self.revision,
            "requested_device": self.requested_device,
            "effective_device": self.device,
            "device": self.device,
            "embedding_batch_size": self.batch_size,
            "precision": "float32",
            "tf32_enabled": False,
            "mixed_precision": False,
            "normalize_embeddings": True,
            "dimensions": self._dimensions,
        }

    def score(self, group: PairGroup) -> Sequence[float]:
        """Embed unique endpoint text once within the current group."""

        import numpy as np

        texts = list(
            dict.fromkeys(
                text for pair in group.pairs for text in (pair.left, pair.right)
            )
        )
        if not texts:
            return []
        values = np.asarray(
            self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                precision="float32",
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if values.ndim != 2 or values.shape[0] != len(texts):
            raise PairTraceError("embedding model returned an unexpected matrix shape")
        self._dimensions = int(values.shape[1])
        text_index = {text: index for index, text in enumerate(texts)}
        scores = [
            float(values[text_index[pair.left]] @ values[text_index[pair.right]])
            for pair in group.pairs
        ]
        if not all(math.isfinite(score) for score in scores):
            raise PairTraceError(
                "embedding model returned a non-finite similarity score"
            )
        return scores


@dataclass
class _StrategyCounts:
    selected_pair_count: int = 0
    selected_positive_pair_count: int = 0
    positive_group_count: int = 0
    groups_losing_positive_count: int = 0
    counterexamples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Bucket:
    group_count: int = 0
    pair_count: int = 0
    positive_pair_count: int = 0
    strategies: dict[str, _StrategyCounts] = field(default_factory=dict)


class AnalysisAccumulator:
    """Aggregate reports without retaining prior pair groups."""

    def __init__(
        self, strategies: Sequence[CandidateStrategy], max_examples: int
    ) -> None:
        if max_examples < 0:
            raise ValueError("max_examples cannot be negative")
        self.strategies = tuple(strategies)
        self.max_examples = max_examples
        self._buckets: dict[str, _Bucket] = {"__all__": _Bucket()}

    def add_group(self, group: PairGroup, scores: Sequence[float] | None) -> None:
        """Add one group and immediately release its pair data after aggregation."""

        if self.strategies:
            if scores is None or len(scores) != len(group.pairs):
                raise PairTraceError("candidate scoring must return one score per pair")
            if not all(math.isfinite(float(score)) for score in scores):
                raise PairTraceError("candidate scores must be finite")
        for bucket_name in ("__all__", group.operator):
            bucket = self._buckets.setdefault(bucket_name, _Bucket())
            self._add_to_bucket(bucket, group, scores)

    def _add_to_bucket(
        self,
        bucket: _Bucket,
        group: PairGroup,
        scores: Sequence[float] | None,
    ) -> None:
        bucket.group_count += 1
        bucket.pair_count += len(group.pairs)
        positive_indices = {
            index for index, pair in enumerate(group.pairs) if pair.baseline_match
        }
        bucket.positive_pair_count += len(positive_indices)
        for strategy in self.strategies:
            counts = bucket.strategies.setdefault(
                strategy.strategy_id, _StrategyCounts()
            )
            selected = _select_candidates(group, scores or (), strategy)
            selected_positive = positive_indices & selected
            counts.selected_pair_count += len(selected)
            counts.selected_positive_pair_count += len(selected_positive)
            if positive_indices:
                counts.positive_group_count += 1
            missed = positive_indices - selected
            if missed:
                counts.groups_losing_positive_count += 1
                if bucket is self._buckets["__all__"]:
                    for index in sorted(missed):
                        if len(counts.counterexamples) >= self.max_examples:
                            break
                        pair = group.pairs[index]
                        counts.counterexamples.append(
                            {
                                "strategy": strategy.strategy_id,
                                "group_id": group.group_id,
                                "operator": group.operator,
                                "direction": group.direction,
                                "case_id": group.case_id,
                                "event_id": group.event_id,
                                "pair_id": pair.pair_id,
                                "left": pair.left,
                                "right": pair.right,
                                "score": float((scores or ())[index]),
                            }
                        )

    def report(self) -> dict[str, Any]:
        """Return overall and per-operator metrics."""

        overall = self._bucket_report(self._buckets["__all__"])
        return {
            "baseline": overall["baseline"],
            "strategies": overall["strategies"],
            "by_operator": {
                name: self._bucket_report(bucket)
                for name, bucket in sorted(self._buckets.items())
                if name != "__all__"
            },
        }

    @staticmethod
    def _bucket_report(bucket: _Bucket) -> dict[str, Any]:
        strategies: list[dict[str, Any]] = []
        for strategy_id, counts in sorted(bucket.strategies.items()):
            strategies.append(
                {
                    "strategy": strategy_id,
                    "selected_pair_count": counts.selected_pair_count,
                    "pair_reduction": (
                        1.0 - _ratio(counts.selected_pair_count, bucket.pair_count)
                        if bucket.pair_count
                        else 0.0
                    ),
                    "selected_positive_pair_count": counts.selected_positive_pair_count,
                    "positive_pair_recall": (
                        _ratio(
                            counts.selected_positive_pair_count,
                            bucket.positive_pair_count,
                        )
                        if bucket.positive_pair_count
                        else None
                    ),
                    "positive_group_count": counts.positive_group_count,
                    "groups_losing_positive_count": counts.groups_losing_positive_count,
                    "counterexamples": counts.counterexamples,
                }
            )
        return {
            "baseline": {
                "group_count": bucket.group_count,
                "pair_count": bucket.pair_count,
                "positive_pair_count": bucket.positive_pair_count,
            },
            "strategies": strategies,
        }


def analyze_sources(
    sources: Sequence[PairSource],
    *,
    phase: str = "insertion",
    strategies: Sequence[CandidateStrategy] = (),
    scorer: PairScorer | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Analyze newest-to-oldest sources without modifying or copying evidence."""

    if strategies and scorer is None:
        raise ValueError("candidate strategies require a pair scorer")
    if not strategies and scorer is not None:
        raise ValueError("pair scorer requires at least one candidate strategy")
    accumulator = AnalysisAccumulator(strategies, max_examples)
    accepted_group_ids: set[str] = set()
    for source in sources:
        for group in source.iter_groups(phase=phase):
            if group.group_id in accepted_group_ids:
                continue
            accepted_group_ids.add(group.group_id)
            source.stats.accepted_group_count += 1
            scores = scorer.score(group) if scorer is not None else None
            accumulator.add_group(group, scores)
    report = accumulator.report()
    return {
        "schema_version": 3,
        "analysis": "semantic-pair-candidates",
        "read_only_inputs": True,
        "source_order": "newest-to-oldest",
        "phase": phase,
        "sources": [vars(source.stats) for source in sources],
        "scorer": dict(scorer.metadata) if scorer is not None else None,
        **report,
    }


def build_strategies(
    *,
    top_ks: Sequence[int],
    thresholds: Sequence[float],
) -> tuple[CandidateStrategy, ...]:
    """Build top-k, threshold, and combined profiles without hidden defaults."""

    unique_top_ks = sorted(set(top_ks))
    unique_thresholds = sorted(set(float(value) for value in thresholds))
    strategies = [CandidateStrategy(top_k=value) for value in unique_top_ks]
    strategies.extend(CandidateStrategy(threshold=value) for value in unique_thresholds)
    strategies.extend(
        CandidateStrategy(top_k=top_k, threshold=threshold)
        for top_k in unique_top_ks
        for threshold in unique_thresholds
    )
    return tuple(sorted(strategies, key=lambda value: value.strategy_id))


def parse_source(value: str, *, read_workers: int = 1) -> PairSource:
    """Parse one newest-to-oldest source specification."""

    if value.startswith("dir:"):
        path = value.removeprefix("dir:")
        if not path:
            raise PairTraceError("directory source path cannot be empty")
        return DirectorySource(Path(path), read_workers=read_workers)
    if value.startswith("tar:"):
        archive, separator, member_root = value.removeprefix("tar:").partition("::")
        if not separator or not archive or not member_root:
            raise PairTraceError(
                "tar source must be tar:/absolute/archive.tar::member/run/root"
            )
        return TarSource(Path(archive), member_root, read_workers=read_workers)
    raise PairTraceError(f"source must start with one of {SOURCE_PREFIXES}: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build the single-report command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Read-only source in newest-to-oldest order: dir:/run/root or "
            "tar:/archive.tar::archived/run/root"
        ),
    )
    parser.add_argument("--phase", default="insertion")
    parser.add_argument("--top-k", type=int, action="append", default=[])
    parser.add_argument("--threshold", type=float, action="append", default=[])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--read-workers", type=int, default=1)
    parser.add_argument("--max-counterexamples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print one report or write exactly one explicitly requested JSON file."""

    args = build_parser().parse_args(argv)
    try:
        _validate_positive_integer(args.read_workers, name="read_workers")
        _validate_positive_integer(
            args.embedding_batch_size, name="embedding batch size"
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    sources = [
        parse_source(value, read_workers=args.read_workers) for value in args.source
    ]
    strategies = build_strategies(top_ks=args.top_k, thresholds=args.threshold)
    scorer: PairScorer | None = None
    if strategies:
        if not args.embedding_model or not args.embedding_revision:
            raise SystemExit(
                "candidate strategies require --embedding-model and --embedding-revision"
            )
        scorer = SentenceTransformerCosineScorer(
            model=args.embedding_model,
            revision=args.embedding_revision,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
    elif args.embedding_model or args.embedding_revision:
        raise SystemExit("embedding configuration requires --top-k or --threshold")

    report = analyze_sources(
        sources,
        phase=args.phase,
        strategies=strategies,
        scorer=scorer,
        max_examples=args.max_counterexamples,
    )
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
        return 0
    _write_single_report(args.output, rendered, sources)
    return 0


def _pair_event_kind(event: Mapping[str, Any], *, phase: str) -> str | None:
    if str(event.get("phase") or "") != phase:
        return None
    if (
        event.get("operator") in {"sem_groupby", "sem_join"}
        and event.get("event_type") == "pair_decision"
    ):
        return "pair_decision"
    if (
        event.get("operator") == "sem_filter"
        and event.get("event_type") == "operator_result"
        and _is_pairwise_sem_filter_event(event)
    ):
        return "sem_filter"
    return None


def _pair_decision_group(
    events: Sequence[Mapping[str, Any]],
    *,
    source: str,
    read_bytes: Callable[[str], bytes],
    executor: ThreadPoolExecutor | None,
) -> PairGroup:
    first = events[0]
    operator = str(first.get("operator") or "")
    direction = {
        "sem_groupby": "symmetric",
        "sem_join": "left-to-right",
    }.get(operator)
    if direction is None:
        raise PairTraceError(f"unsupported pair-decision operator: {operator}")
    base = _group_base(first, direction=direction)
    drafts: list[tuple[str, str, str, str, bool]] = []
    labels_by_pair: dict[tuple[str, str, str, str], bool] = {}
    def read_output(event: Mapping[str, Any]) -> tuple[str, bytes]:
        parsed_path = _required_string(event, "parsed_output_path")
        return parsed_path, read_bytes(parsed_path)

    def build_group(outputs: Iterator[tuple[str, bytes]]) -> PairGroup:
        for event, (parsed_path, raw_output) in zip(events, outputs, strict=True):
            if _group_base(event, direction=direction) != base:
                raise PairTraceError(
                    "one physical pair-decision call contains mixed logical groups"
                )
            try:
                parsed = json.loads(raw_output)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PairTraceError(
                    f"invalid semantic pair parsed output: {parsed_path}"
                ) from error
            if not isinstance(parsed, bool):
                raise PairTraceError(
                    f"semantic pair parsed output must be boolean: {parsed_path}"
                )
            left = _required_string(event, "left")
            right = _required_string(event, "right")
            left_id = _endpoint_id(event, "left", left)
            right_id = _endpoint_id(event, "right", right)
            pair_key = (left_id, right_id, left, right)
            prior_label = labels_by_pair.get(pair_key)
            if prior_label is not None and prior_label != parsed:
                raise PairTraceError(
                    f"{operator} pair has conflicting baseline labels"
                )
            labels_by_pair[pair_key] = parsed
            drafts.append(
                (
                    left_id,
                    right_id,
                    left,
                    right,
                    parsed,
                )
            )
        return _finalize_group(
            first, source=source, direction=direction, drafts=drafts
        )

    if executor is None:
        return build_group(map(read_output, events))
    return build_group(executor.map(read_output, events))


def _sem_filter_group(
    event: Mapping[str, Any],
    *,
    source: str,
    read_bytes: Any,
) -> PairGroup:
    return _sem_filter_group_from_bytes(
        event,
        source=source,
        input_value=read_bytes(_required_string(event, "input_snapshot_path")),
        output_value=read_bytes(_required_string(event, "output_snapshot_path")),
    )


def _sem_filter_group_from_bytes(
    event: Mapping[str, Any],
    *,
    source: str,
    input_value: bytes,
    output_value: bytes,
) -> PairGroup:
    input_rows = _parse_csv(input_value, source="input_snapshot_path")
    output_rows = _parse_csv(output_value, source="output_snapshot_path")
    if _optional_nonnegative_integer(
        event, "input_rows", default=len(input_rows)
    ) != len(input_rows):
        raise PairTraceError("sem_filter input row count does not match snapshot")
    if _optional_nonnegative_integer(
        event, "output_rows", default=len(output_rows)
    ) != len(output_rows):
        raise PairTraceError("sem_filter output row count does not match snapshot")
    paired_bases = _paired_placeholder_bases(event, input_rows)
    output_counts = Counter(_row_signature(row) for row in output_rows)
    drafts: list[tuple[str, str, str, str, bool]] = []
    for row in input_rows:
        signature = _row_signature(row)
        matched = output_counts[signature] > 0
        if matched:
            output_counts[signature] -= 1
        left = _side_text(row, paired_bases, "earlier")
        right = _side_text(row, paired_bases, "later")
        drafts.append(
            (
                _side_id(row, "earlier", left),
                _side_id(row, "later", right),
                left,
                right,
                matched,
            )
        )
    if any(output_counts.values()):
        raise PairTraceError(
            "sem_filter output contains rows absent from input snapshot"
        )
    return _finalize_group(
        event, source=source, direction="right-to-left", drafts=drafts
    )


def _finalize_group(
    event: Mapping[str, Any],
    *,
    source: str,
    direction: str,
    drafts: Sequence[tuple[str, str, str, str, bool]],
) -> PairGroup:
    occurrences: Counter[str] = Counter()
    pair_tokens: list[str] = []
    normalized: list[tuple[str, str, str, str, bool, str]] = []
    for left_id, right_id, left, right, matched in drafts:
        signature = _stable_digest(
            {
                "direction": direction,
                "left_id": left_id,
                "right_id": right_id,
                "left": left,
                "right": right,
            }
        )
        occurrence = occurrences[signature]
        occurrences[signature] += 1
        token = f"{signature}:{occurrence}"
        pair_tokens.append(token)
        normalized.append((left_id, right_id, left, right, matched, token))
    group_id = _stable_digest(
        {**_group_base(event, direction=direction), "pair_tokens": sorted(pair_tokens)}
    )
    pairs = tuple(
        PairRecord(
            pair_id=_stable_digest({"group_id": group_id, "pair_token": token}),
            left_id=left_id,
            right_id=right_id,
            left=left,
            right=right,
            baseline_match=matched,
        )
        for left_id, right_id, left, right, matched, token in normalized
    )
    return PairGroup(
        group_id=group_id,
        operator=str(event.get("operator") or ""),
        direction=direction,
        source=source,
        case_id=_event_case_id(event),
        session_id=str(event.get("session_id") or ""),
        event_id=str(event.get("event_id") or ""),
        query_digest=str(event.get("query_digest") or ""),
        pairs=pairs,
    )


def _group_evidence_digest(group: PairGroup) -> str:
    return _stable_digest(
        {
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "baseline_match": pair.baseline_match,
                }
                for pair in group.pairs
            ]
        }
    )


def _group_base(event: Mapping[str, Any], *, direction: str) -> dict[str, str]:
    return {
        "run_kind": str(event.get("run_kind") or ""),
        "case_id": _event_case_id(event),
        "session_id": str(event.get("session_id") or ""),
        "event_id": str(event.get("event_id") or ""),
        "query_digest": str(event.get("query_digest") or ""),
        "operator": str(event.get("operator") or ""),
        "direction": direction,
        "instruction": str(
            event.get("source_instruction")
            or event.get("instruction")
            or event.get("lowered_instruction")
            or ""
        ),
    }


def _select_candidates(
    group: PairGroup,
    scores: Sequence[float],
    strategy: CandidateStrategy,
) -> set[int]:
    eligible = {
        index
        for index, score in enumerate(scores)
        if strategy.threshold is None or float(score) >= strategy.threshold
    }
    if strategy.top_k is None:
        return eligible
    buckets: dict[str, list[int]] = defaultdict(list)
    for index in eligible:
        pair = group.pairs[index]
        if group.direction == "left-to-right":
            buckets[f"left:{pair.left_id}"].append(index)
        elif group.direction == "right-to-left":
            buckets[f"right:{pair.right_id}"].append(index)
        elif group.direction == "symmetric":
            buckets[f"node:{pair.left_id}"].append(index)
            buckets[f"node:{pair.right_id}"].append(index)
        else:
            raise PairTraceError(f"unsupported pair direction: {group.direction}")
    selected: set[int] = set()
    for indices in buckets.values():
        indices.sort(
            key=lambda index: (-float(scores[index]), group.pairs[index].pair_id)
        )
        selected.update(indices[: strategy.top_k])
    return selected


def _is_pairwise_sem_filter_event(event: Mapping[str, Any]) -> bool:
    instruction = str(event.get("lowered_instruction") or "")
    sides: dict[str, set[str]] = defaultdict(set)
    for base, side in PLACEHOLDER_PATTERN.findall(instruction):
        sides[base].add(side)
    return any(values == {"earlier", "later"} for values in sides.values())


def _paired_placeholder_bases(
    event: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, str]],
) -> tuple[str, ...]:
    instruction = str(event.get("lowered_instruction") or "")
    sides: dict[str, set[str]] = defaultdict(set)
    for base, side in PLACEHOLDER_PATTERN.findall(instruction):
        sides[base].add(side)
    bases = tuple(
        sorted(base for base, values in sides.items() if values == {"earlier", "later"})
    )
    columns = (
        set(input_rows[0]) if input_rows else set(event.get("input_columns") or ())
    )
    valid = tuple(
        base
        for base in bases
        if f"{base}:earlier" in columns and f"{base}:later" in columns
    )
    if not valid:
        raise PairTraceError(
            "pairwise sem_filter trace must expose matching earlier/later placeholders"
        )
    return valid


def _side_text(row: Mapping[str, str], bases: Sequence[str], side: str) -> str:
    return "\n".join(f"{base}: {row[f'{base}:{side}']}" for base in bases)


def _side_id(row: Mapping[str, str], side: str, text: str) -> str:
    values = {
        key: value
        for key, value in row.items()
        if key.endswith(f":{side}")
        and (key.startswith("_row_id:") or key.startswith("_memory_ordinal:"))
    }
    return _stable_digest(values or {"text": text})


def _endpoint_id(event: Mapping[str, Any], side: str, text: str) -> str:
    for key in (f"{side}_unique_id", f"{side}_id"):
        value = event.get(key)
        if value is not None and not isinstance(value, bool):
            return str(value)
    return text


def _event_case_id(event: Mapping[str, Any]) -> str:
    return str(event.get("case_id") or event.get("sample_id") or "")


def _parse_current_attempt_ledger(
    artifacts: Sequence[tuple[str, bytes, bytes]], *, source: str
) -> CurrentAttemptLedger:
    case_ids: set[str] = set()
    successful: set[tuple[str, str, str, int, int]] = set()
    digest = hashlib.sha256()
    for case_root, case_raw, state_raw in sorted(artifacts):
        digest.update(case_root.encode("utf-8"))
        digest.update(b"\x00case\x00")
        digest.update(case_raw)
        digest.update(b"\x00state\x00")
        digest.update(state_raw)
        case_value = _parse_json_object_bytes(
            case_raw, source=f"{source}:{case_root}/case.json"
        )
        state_value = _parse_json_object_bytes(
            state_raw,
            source=f"{source}:{case_root}/control/unit-attempts.json",
        )
        case_id = _required_string(case_value, "case_id")
        if case_id in case_ids:
            raise PairTraceError(f"duplicate case attempt state: {case_id}")
        case_ids.add(case_id)
        units = state_value.get("units")
        if not isinstance(units, dict):
            raise PairTraceError(
                f"unit attempt state must contain an object of units: {case_id}"
            )
        for unit in units.values():
            if not isinstance(unit, dict):
                raise PairTraceError(f"unit attempt entry must be an object: {case_id}")
            phase = _required_string(unit, "phase")
            unit_id = _required_string(unit, "unit_id")
            execution_attempt = _required_positive_integer(
                unit.get("execution_attempt"), name="execution_attempt"
            )
            unit_attempt = _required_positive_integer(
                unit.get("unit_attempt"), name="unit_attempt"
            )
            status = unit.get("last_status")
            if status not in {"running", "failed", "success"}:
                raise PairTraceError(
                    f"invalid unit attempt status for {case_id}:{phase}:{unit_id}"
                )
            if status == "success":
                successful.add(
                    (
                        case_id,
                        phase,
                        unit_id,
                        execution_attempt,
                        unit_attempt,
                    )
                )
    return CurrentAttemptLedger(
        case_ids=frozenset(case_ids),
        successful_lineages=frozenset(successful),
        digest=digest.hexdigest(),
    )


def _event_attempt_status(
    event: Mapping[str, Any],
    *,
    trace_index: int,
    current_attempts: CurrentAttemptLedger | None,
    attempt_intervals: Mapping[str, Sequence[AttemptInterval]],
    legacy_recovery: LegacyRecovery | None,
    unproven_attempts: dict[tuple[str, str, str], tuple[int, int]],
) -> str | None:
    case_id = _event_case_id(event)
    if current_attempts is not None and case_id in current_attempts.case_ids:
        lineage = _trace_attempt_lineage(event, required=True)
        assert lineage is not None
        return (
            "completed"
            if lineage in current_attempts.successful_lineages
            else "failed"
        )

    event_id = str(event.get("event_id") or "")
    candidates = attempt_intervals.get(event_id)
    if candidates is not None:
        timestamp = _parse_timestamp(
            event.get("timestamp"),
            source=f"trace event {event.get('trace_id') or event_id}",
        )
        matches = [
            interval
            for interval in candidates
            if interval.start <= timestamp <= interval.end
        ]
        if len(matches) != 1:
            raise PairTraceError(
                "semantic pair trace is outside recorded attempt intervals: "
                f"{event_id} at {event.get('timestamp')}"
            )
        return matches[0].status

    if legacy_recovery is not None:
        return (
            "legacy-failed"
            if _is_excluded_trace_index(trace_index, legacy_recovery.ranges)
            else "legacy-completed"
        )

    lineage = _trace_attempt_lineage(event, required=False)
    if lineage is None:
        return None
    key = lineage[:3]
    attempt = lineage[3:]
    prior = unproven_attempts.setdefault(key, attempt)
    if prior != attempt:
        raise PairTraceError(
            "multiple attempt lineages require authoritative recovery metadata: "
            f"{case_id}:{event_id}"
        )
    return None


def _trace_attempt_lineage(
    event: Mapping[str, Any], *, required: bool
) -> tuple[str, str, str, int, int] | None:
    execution_attempt = event.get("execution_attempt")
    unit_attempt = event.get("unit_attempt")
    if execution_attempt is None and unit_attempt is None:
        if required:
            raise PairTraceError(
                "trace event is missing execution_attempt and unit_attempt"
            )
        return None
    case_id = _event_case_id(event)
    phase = str(event.get("phase") or "")
    event_id = str(event.get("event_id") or "")
    if not case_id or not phase or not event_id:
        raise PairTraceError("trace attempt lineage is missing case, phase, or event ID")
    return (
        case_id,
        phase,
        event_id,
        _required_positive_integer(
            execution_attempt, name="trace execution_attempt"
        ),
        _required_positive_integer(unit_attempt, name="trace unit_attempt"),
    )


def _is_excluded_trace_index(
    trace_index: int, ranges: Sequence[tuple[int, int]]
) -> bool:
    return any(start <= trace_index < end for start, end in ranges)


def _row_signature(row: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in row.items()))


def _parse_json_object_bytes(value: bytes, *, source: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairTraceError(f"invalid JSON object: {source}") from error
    if not isinstance(parsed, dict):
        raise PairTraceError(f"JSON value must be an object: {source}")
    return parsed


def _parse_json_line(
    raw_line: bytes,
    *,
    source: str,
    line_number: int,
) -> dict[str, Any]:
    if not raw_line.strip():
        return {}
    try:
        row = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairTraceError(f"invalid JSONL at {source}:{line_number}") from error
    if not isinstance(row, dict):
        raise PairTraceError(f"JSONL row must be an object at {source}:{line_number}")
    return row


def _parse_timestamp(value: object, *, source: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise PairTraceError(f"missing timestamp at {source}")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise PairTraceError(f"invalid timestamp at {source}: {value!r}") from error
    if timestamp.tzinfo is None:
        raise PairTraceError(f"timestamp must include a timezone at {source}")
    return timestamp.astimezone(timezone.utc)


def _parse_csv(value: bytes, *, source: str) -> list[dict[str, str]]:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PairTraceError(f"CSV is not UTF-8: {source}") from error
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise PairTraceError(f"trace event is missing {key}")
    return value


def _required_nonnegative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PairTraceError(f"{name} must be a non-negative integer")
    return value


def _required_positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PairTraceError(f"{name} must be a positive integer")
    return value


def _optional_nonnegative_integer(
    row: Mapping[str, Any], key: str, *, default: int
) -> int:
    if key not in row:
        return default
    return _required_nonnegative_integer(row[key], name=f"sem_filter {key}")


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PairTraceError(f"artifact path must be relative and contained: {value!r}")
    return path


def _safe_member_root(value: str) -> str:
    return _safe_relative_path(value.strip("/")).as_posix().rstrip("/")


def _validate_uncompressed_tar(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            magic = handle.read(8)
    except FileNotFoundError as error:
        raise PairTraceError(f"cannot open tar archive: {path}") from error
    if any(magic.startswith(prefix) for prefix in COMPRESSED_ARCHIVE_MAGIC):
        raise PairTraceError(
            "compressed tar archives are not supported for direct analysis"
        )


def _validate_positive_integer(value: object, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _configure_float32_execution(device: str) -> None:
    if not device.lower().startswith("cuda"):
        return
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _write_single_report(
    output: Path,
    rendered: str,
    sources: Sequence[PairSource],
) -> None:
    target = output.resolve()
    if target.exists():
        raise FileExistsError(f"analysis report already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"analysis report parent does not exist: {target.parent}"
        )
    if any(source.contains_output(target) for source in sources):
        raise PairTraceError("analysis output cannot be written inside a source run")
    target.write_text(rendered, encoding="utf-8")


def _stable_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
