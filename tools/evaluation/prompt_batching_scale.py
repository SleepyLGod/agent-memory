"""Run the controlled 128-event Claude prompt-batching scale experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_memory.evaluation.bundle import write_bundle  # noqa: E402
from agent_memory.evaluation.locomo import (  # noqa: E402
    LOCOMO_COMMIT,
    LOCOMO_SHA256,
    eligible_questions,
    load_locomo_sample,
    locomo_bundle,
)
from agent_memory.evaluation.provenance import (  # noqa: E402
    SOURCE_EVIDENCE_ENV,
    build_source_evidence,
    collect_runtime_provenance,
    validate_run_provenance,
)


HONG_KONG = ZoneInfo("Asia/Hong_Kong")
EVENT_COUNT = 128
QUESTION_COUNT = 71
JUDGED_QUESTION_COUNT = 54
FIRST_EVENT_ID = "D1:1"
LAST_EVENT_ID = "D7:20"
CLIENT_REQUEST_MAX_WORKERS = 64
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
SOFT_STOP_CNY = Decimal("48")
HARD_STOP_CNY = Decimal("50")
MIN_DISK_GIB = 100.0
MAX_CONDITION_BYTES = 2 * 1024**3
MAX_CONDITION_SECONDS = 6 * 60 * 60
POLL_SECONDS = 15
OFF_PEAK_PRICES = {
    "cache_hit": Decimal("0.05"),
    "cache_miss": Decimal("1.5"),
    "completion": Decimal("4.5"),
}
PEAK_MULTIPLIER = Decimal("2")
OFFICIAL_SCORER_ID = "locomo_official:v1"
JUDGE_SCORER_ID = "locomo_zep_judge:deepseek/deepseek-v4-flash"


@dataclass(frozen=True)
class ConditionSpec:
    """One independent end-to-end batching-scale condition."""

    pass_index: int
    label: str
    refresh_every: int
    prompt_batch: int | str | None

    @property
    def run_id(self) -> str:
        """Return the stable directory identifier for this condition."""

        return f"pass-{self.pass_index}-{self.label}"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe condition contract."""

        return {"run_id": self.run_id, **asdict(self)}


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved filesystem paths for one experiment run."""

    root: Path
    control: Path
    bundle: Path
    conditions: Path
    analysis: Path
    source: Path
    venv: Path
    deps: Path | None
    hf_home: Path

    @classmethod
    def from_contract(cls, root: Path, contract: Mapping[str, Any]) -> ExperimentPaths:
        """Build paths from one persisted experiment contract."""

        deps = contract["runtime"].get("deps_dir")
        return cls(
            root=root.resolve(),
            control=root.resolve() / "control",
            bundle=root.resolve() / "bundle",
            conditions=root.resolve() / "conditions",
            analysis=root.resolve() / "analysis",
            source=Path(str(contract["source"]["directory"])).resolve(),
            venv=Path(str(contract["runtime"]["venv"])).resolve(),
            deps=Path(str(deps)).resolve() if deps is not None else None,
            hf_home=Path(str(contract["runtime"]["hf_home"])).resolve(),
        )


def condition_matrix() -> tuple[ConditionSpec, ...]:
    """Return the two identical ordered passes of the ten-condition sweep."""

    levels: tuple[tuple[str, int, int | str | None], ...] = (
        ("original", 1, None),
        ("packed-1", 1, 1),
        ("packed-2", 2, 2),
        ("packed-4", 4, 4),
        ("packed-8", 8, 8),
        ("packed-16", 16, 16),
        ("packed-32", 32, 32),
        ("packed-64", 64, 64),
        ("packed-128", 128, 128),
        ("packed-all", 128, "all"),
    )
    return tuple(
        ConditionSpec(pass_index, label, refresh, prompt)
        for pass_index in (1, 2)
        for label, refresh, prompt in levels
    )


def build_condition_command(
    condition: ConditionSpec,
    *,
    source: Path,
    venv: Path,
    bundle: Path,
    output: Path,
) -> list[str]:
    """Build the canonical LOCOMO command for one condition."""

    command = [
        str(venv / "bin" / "python"),
        str(source / "tools" / "evaluation" / "locomo.py"),
        "run",
        "--bundle-dir",
        str(bundle),
        "--output-dir",
        str(output),
        "--system",
        "claude-memory",
        "--condition-id",
        f"Claude-BatchingScale-{condition.run_id}-128e",
        "--memory-model",
        "deepseek/deepseek-v4-flash",
        "--answer-model",
        "deepseek/deepseek-v4-flash",
        "--judge-model",
        "deepseek/deepseek-v4-flash",
        "--grouped-agg-rule",
        "rule-join-map",
        "--sem-topk-method",
        "listwise",
        "--semantic-pair-profile",
        "search-filter",
        "--semantic-pair-top-k",
        "20",
        "--semantic-pair-min-similarity",
        "0.5",
        "--embedding-device",
        "cuda",
        "--lotus-cache-mode",
        "disabled",
        "--semantic-trace-snapshot-mode",
        "compact",
        "--refresh-every",
        str(condition.refresh_every),
    ]
    if condition.prompt_batch is not None:
        command.extend(("--prompt-batch-size", str(condition.prompt_batch)))
    return command


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(repository), *arguments), text=True
    ).strip()


def _directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _directories, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                continue
    return total


def _disk_available_gib(path: Path) -> float:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize / 1024**3


def _is_peak(local: datetime) -> bool:
    if local.weekday() >= 5:
        return False
    minute = local.hour * 60 + local.minute
    return 9 * 60 <= minute < 12 * 60 or 14 * 60 <= minute < 18 * 60


def _pricing_period(timestamp: str) -> str:
    observed = datetime.strptime(timestamp, "%Y%m%dT%H%M%S%fZ").replace(
        tzinfo=timezone.utc
    )
    return "peak" if _is_peak(observed.astimezone(HONG_KONG)) else "off_peak"


def _usage_cost_cny(pricing_usage: Mapping[str, Mapping[str, int]]) -> Decimal:
    total = Decimal(0)
    million = Decimal(1_000_000)
    for period, usage in pricing_usage.items():
        multiplier = PEAK_MULTIPLIER if period == "peak" else Decimal(1)
        total += multiplier * (
            Decimal(usage.get("cache_hit", 0)) * OFF_PEAK_PRICES["cache_hit"]
            + Decimal(usage.get("cache_miss", 0))
            * OFF_PEAK_PRICES["cache_miss"]
            + Decimal(usage.get("completion", 0))
            * OFF_PEAK_PRICES["completion"]
        ) / million
    return total.quantize(Decimal("0.000001"))


def _empty_monitor() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "offset": 0,
        "provider_responses": 0,
        "reasoning_tokens": 0,
        "reasoning_usage_reports": 0,
        "provider_contract_violations": [],
        "pricing_usage": {
            "off_peak": {"cache_hit": 0, "cache_miss": 0, "completion": 0},
            "peak": {"cache_hit": 0, "cache_miss": 0, "completion": 0},
        },
        "phase_usage": {},
        "insertion_results": 0,
        "insertion_latency_ms": 0.0,
        "trace_io_latency_ms": 0.0,
        "trace_bytes_written": 0,
        "retrieval_results": 0,
        "answer_results": 0,
        "grade_results": 0,
        "candidate_pairs": {},
        "operator_usage": {},
        "prompt_batching": {},
    }


def _integer_token(event: Mapping[str, Any], key: str) -> int:
    value = event.get(key)
    if value in (None, ""):
        raise ValueError(f"provider usage is missing {key}")
    return int(value)


def _number_equals(value: Any, expected: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) == expected
    )


def _counter_mapping(value: Mapping[str, Mapping[str, Any]]) -> dict[str, Counter[str]]:
    return {key: Counter(row) for key, row in value.items()}


def _trace_site(event: Mapping[str, Any]) -> str:
    operator = str(event.get("operator") or "unknown")
    digest = str(
        event.get("semantic_pair_site_id")
        or event.get("query_digest")
        or event.get("query_fingerprint")
        or "default"
    )
    return f"{operator}:{digest}"


def _physical_usage_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize direct answer/judge usage without counting LOTUS mirrors twice."""

    if (
        event.get("event_type") != "llm_call"
        or event.get("operator") != "llm"
        or event.get("phase") not in {"answering", "grading"}
    ):
        return event
    if event.get("usage_scope") != "batch" or event.get("llm_batch_size") != 1:
        raise RuntimeError("unsupported direct answer/judge usage scope")
    details = event.get("usage_completion_tokens_details") or {}
    return {
        **event,
        "event_type": "provider_usage",
        "provider_prompt_cache_hit_tokens": event.get("usage_prompt_cache_hit_tokens"),
        "provider_prompt_cache_miss_tokens": event.get("usage_prompt_cache_miss_tokens"),
        "provider_completion_tokens": event.get("usage_completion_tokens"),
        "provider_reasoning_tokens": details.get("reasoning_tokens"),
        "provider_request": {"provider_kwargs": event.get("llm_kwargs") or {}},
    }


def _update_monitor(trace: Path, state_path: Path) -> dict[str, Any]:
    state = json.loads(state_path.read_text()) if state_path.exists() else _empty_monitor()
    if state.get("schema_version") != 1:
        raise RuntimeError(f"invalid monitor state: {state_path}")
    offset = int(state["offset"])
    if trace.stat().st_size < offset:
        raise RuntimeError(f"semantic trace shrank: {trace}")
    candidates = _counter_mapping(state["candidate_pairs"])
    operators = _counter_mapping(state["operator_usage"])
    batching = _counter_mapping(state["prompt_batching"])
    phases = _counter_mapping(state["phase_usage"])
    violations = list(state["provider_contract_violations"])
    with trace.open("rb") as handle:
        handle.seek(offset)
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.seek(line_start)
                break
            event = _physical_usage_event(json.loads(line))
            event_type = str(event.get("event_type") or "")
            site = _trace_site(event)
            if event_type == "provider_usage":
                try:
                    hit = _integer_token(event, "provider_prompt_cache_hit_tokens")
                    miss = _integer_token(event, "provider_prompt_cache_miss_tokens")
                    completion = _integer_token(event, "provider_completion_tokens")
                    period = _pricing_period(str(event["timestamp"]))
                except (KeyError, TypeError, ValueError) as error:
                    violations.append(f"incomplete_provider_usage:{error}")
                    hit = miss = completion = 0
                    period = "off_peak"
                response_count = int(event.get("response_count") or 1)
                state["provider_responses"] += response_count
                state["pricing_usage"][period]["cache_hit"] += hit
                state["pricing_usage"][period]["cache_miss"] += miss
                state["pricing_usage"][period]["completion"] += completion
                phase = str(event.get("phase") or "unknown")
                phases.setdefault(phase, Counter())
                phases[phase].update(
                    responses=response_count,
                    cache_hit=hit,
                    cache_miss=miss,
                    completion=completion,
                )
                operators.setdefault(site, Counter())
                operators[site].update(
                    responses=response_count,
                    tokens=hit + miss + completion,
                )
                reasoning = event.get("provider_reasoning_tokens")
                if reasoning not in (None, ""):
                    state["reasoning_usage_reports"] += 1
                    state["reasoning_tokens"] += int(reasoning)
                request = event.get("provider_request") or {}
                kwargs = request.get("provider_kwargs") or {}
                if float(kwargs.get("temperature", 0.0)) != 0.0:
                    violations.append(f"temperature:{kwargs.get('temperature')}")
                if kwargs.get("thinking") != {"type": "disabled"}:
                    violations.append(f"thinking:{kwargs.get('thinking')!r}")
            elif event_type == "insertion_result":
                state["insertion_results"] += 1
                state["insertion_latency_ms"] += float(event.get("latency_ms") or 0)
                state["trace_io_latency_ms"] += float(
                    event.get("semantic_trace_io_latency_ms") or 0
                )
                state["trace_bytes_written"] += int(
                    event.get("semantic_trace_bytes_written") or 0
                )
            elif event_type == "retrieval_result":
                state["retrieval_results"] += 1
            elif event_type == "answer_result":
                state["answer_results"] += 1
            elif event_type == "grade_result":
                state["grade_results"] += 1
            elif event_type == "candidate_generation":
                candidates.setdefault(site, Counter())
                candidates[site].update(
                    events=1,
                    eligible=int(event.get("total_pair_count") or 0),
                    selected=int(event.get("candidate_pair_count") or 0),
                )
            elif event_type == "prompt_batching":
                batching.setdefault(site, Counter())
                chunks = [int(item) for item in event.get("chunk_sizes") or ()]
                batching[site].update(
                    events=1,
                    tasks=int(event.get("task_count") or 0),
                    prompts=int(event.get("prompt_count") or 0),
                    retries=int(event.get("retry_count") or 0),
                    repairs=int(
                        event.get(
                            "structured_output_repair_count",
                            event.get("syntax_repair_count"),
                        )
                        or 0
                    ),
                )
                if chunks:
                    batching[site]["max_chunk"] = max(
                        batching[site]["max_chunk"], max(chunks)
                    )
            offset = handle.tell()
    state["offset"] = offset
    state["provider_contract_violations"] = violations
    state["candidate_pairs"] = {
        key: dict(value) for key, value in sorted(candidates.items())
    }
    state["operator_usage"] = {
        key: dict(value) for key, value in sorted(operators.items())
    }
    state["prompt_batching"] = {
        key: dict(value) for key, value in sorted(batching.items())
    }
    state["phase_usage"] = {
        key: dict(value) for key, value in sorted(phases.items())
    }
    state["cost_cny"] = float(_usage_cost_cny(state["pricing_usage"]))
    _atomic_json(state_path, state)
    return state


def _read_contract(root: Path) -> dict[str, Any]:
    return json.loads((root / "control" / "experiment-contract.json").read_text())


def _paths(root: Path) -> ExperimentPaths:
    contract = _read_contract(root)
    return ExperimentPaths.from_contract(root, contract)


def _log(paths: ExperimentPaths, message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with (paths.control / "controller.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()


def _state(paths: ExperimentPaths, name: str) -> str:
    path = paths.control / name
    return path.read_text().strip() if path.exists() else ""


def _set_state(paths: ExperimentPaths, name: str, value: str) -> None:
    _atomic_text(paths.control / name, f"{value}\n")


def _condition_from_dict(value: Mapping[str, Any]) -> ConditionSpec:
    return ConditionSpec(
        pass_index=int(value["pass_index"]),
        label=str(value["label"]),
        refresh_every=int(value["refresh_every"]),
        prompt_batch=value.get("prompt_batch"),
    )


def _total_cost(paths: ExperimentPaths) -> Decimal:
    continuation = _read_contract(paths.root).get("continuation") or {}
    total = Decimal(str(continuation.get("prior_cost_cny", "0")))
    for path in paths.conditions.glob("*/monitor-state.json"):
        payload = json.loads(path.read_text())
        total += Decimal(str(payload.get("cost_cny") or 0))
    return total


def _capture_continuation(
    origin: Path, run_ids: Sequence[str], contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Pin completed results and all prior physical spending without writing to origin."""

    origin = origin.resolve()
    previous = _read_contract(origin)
    if previous.get("continuation"):
        raise RuntimeError("nested continuations are not supported")
    if (origin / "control/experiment-state").read_text().strip() not in {
        "failed", "safety-stopped", "completed",
    }:
        raise RuntimeError("origin experiment must be stopped")
    for key in ("experiment", "input", "fixed_execution", "runtime", "pricing", "conditions"):
        if previous.get(key) != contract.get(key):
            raise RuntimeError(f"continuation contract mismatch: {key}")
    specs = {row["run_id"]: row for row in previous["conditions"]}
    if not run_ids or len(set(run_ids)) != len(run_ids) or not set(run_ids) <= specs.keys():
        raise ValueError("reuse IDs must be known, unique, and nonempty")
    files = {
        origin / "control/experiment-contract.json",
        origin / "control/source-evidence.json",
    }
    reused = {}
    for run_id in run_ids:
        directory = origin / "conditions" / run_id
        output = directory / "output"
        if (directory / "state").read_text().strip() != "completed":
            raise RuntimeError(f"reuse condition is not completed: {run_id}")
        result = json.loads((directory / "validation.json").read_text())
        summary = json.loads((output / "metrics/summary.json").read_text())
        monitor = json.loads((directory / "monitor-state.json").read_text())
        scores = summary.get("scores_by_scorer") or {}
        if (
            any(result.get(key) != value for key, value in specs[run_id].items())
            or monitor.get("insertion_results") != EVENT_COUNT
            or summary.get("completed_cases") != 1
            or summary.get("failed_cases") != 0
            or summary.get("question_count") != QUESTION_COUNT
            or summary.get("memory_system_error_question_count") != 0
            or summary.get("retrieval_system_error_count") != 0
            or _completed_questions(output) != QUESTION_COUNT
            or (scores.get(OFFICIAL_SCORER_ID) or {}).get("grade_count") != QUESTION_COUNT
            or (scores.get(JUDGE_SCORER_ID) or {}).get("grade_count") != JUDGED_QUESTION_COUNT
        ):
            raise RuntimeError(f"reuse result is incomplete or incompatible: {run_id}")
        # Hash the small completed artifacts, including checkpoint and question payloads.
        files.update(p for p in directory.rglob("*") if p.is_file())
        reused[run_id] = str(directory)
    condition_costs = {}
    for trace in sorted(origin.glob("conditions/*/output/trace/events.jsonl")):
        files.add(trace)
        usage = {period: Counter() for period in ("off_peak", "peak")}
        with trace.open() as handle:
            for line in handle:
                event = _physical_usage_event(json.loads(line))
                if event.get("event_type") == "provider_usage":
                    usage[_pricing_period(event["timestamp"])].update(
                        cache_hit=_integer_token(event, "provider_prompt_cache_hit_tokens"),
                        cache_miss=_integer_token(event, "provider_prompt_cache_miss_tokens"),
                        completion=_integer_token(event, "provider_completion_tokens"),
                    )
        condition_costs[trace.parents[2].name] = str(_usage_cost_cny(usage))
    return {
        "origin_root": str(origin),
        "source": previous["source"],
        "reused_conditions": reused,
        "files_sha256": {str(p.relative_to(origin)): _sha256_file(p) for p in sorted(files)},
        "prior_cost_cny": str(sum((Decimal(value) for value in condition_costs.values()), Decimal(0))),
        "prior_condition_costs_cny": condition_costs,
        "interpretation": "Exploratory mixed-repair first pass; second pass uses the new source.",
    }


def _validate_continuation(contract: Mapping[str, Any]) -> None:
    continuation = contract.get("continuation")
    if not continuation:
        return
    origin = Path(continuation["origin_root"])
    for name, digest in continuation["files_sha256"].items():
        path = origin / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise RuntimeError(f"reused evidence changed: {path}")


def _result_directory(root: Path, contract: Mapping[str, Any], run_id: str) -> Path:
    reused = (contract.get("continuation") or {}).get("reused_conditions", {})
    return Path(reused[run_id]) if run_id in reused else root / "conditions" / run_id


def _completed_questions(output: Path) -> int:
    return sum(
        1 for _path in output.glob("cases/*/question-results/*/complete.json")
    )


def _question_numbers(dataset: Path) -> tuple[int, ...]:
    sample = load_locomo_sample(dataset, sample_index=0)
    events = sample.events[:EVENT_COUNT]
    questions = eligible_questions(
        sample.questions,
        ingested_event_ids=[event.event_id for event in events],
    )
    if (
        len(events) != EVENT_COUNT
        or events[0].event_id != FIRST_EVENT_ID
        or events[-1].event_id != LAST_EVENT_ID
        or len(questions) != QUESTION_COUNT
    ):
        raise RuntimeError("the pinned LOCOMO 128-event prefix contract changed")
    categories = Counter(int(question.category) for question in questions)
    if categories != Counter({1: 7, 2: 16, 3: 6, 4: 25, 5: 17}):
        raise RuntimeError(f"unexpected question categories: {categories}")
    return tuple(int(question.metadata["question_number"]) for question in questions)


def _validate_initialization_root(root: Path) -> None:
    if not root.exists():
        return
    unexpected = sorted(
        path.name for path in root.iterdir() if path.name not in {"package", "source"}
    )
    if unexpected:
        raise FileExistsError(
            f"experiment root contains unexpected entries: {root}: {unexpected}"
        )


def initialize(
    *,
    root: Path,
    source: Path,
    venv: Path,
    deps: Path | None,
    hf_home: Path,
    dataset: Path,
    source_package_inventory: Path,
    reuse_root: Path | None = None,
    reuse_conditions: Sequence[str] = (),
) -> None:
    """Create the immutable experiment contract and canonical input bundle."""

    root = root.resolve()
    source = source.resolve()
    _validate_initialization_root(root)
    control = root / "control"
    bundle_dir = root / "bundle"
    conditions_dir = root / "conditions"
    analysis_dir = root / "analysis"
    for path in (control, bundle_dir, conditions_dir, analysis_dir):
        path.mkdir(parents=True, exist_ok=True)
    numbers = _question_numbers(dataset)
    bundle = locomo_bundle(
        dataset,
        sample_index=0,
        start_row=1,
        row_limit=EVENT_COUNT,
        question_numbers=numbers,
        include_adversarial=True,
        run_mode="prompt-batching-scale",
    )
    write_bundle(bundle, bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    evidence = build_source_evidence(source)
    _atomic_json(control / "source-evidence.json", evidence)
    inventory_copy = control / "source-package-inventory.json"
    _atomic_text(inventory_copy, source_package_inventory.read_text())
    contract = {
        "schema_version": 1,
        "experiment": "claude-prompt-batching-scale-128e",
        "source": {
            "directory": str(source),
            "base_commit": _git(source, "rev-parse", "HEAD"),
            "source_snapshot_sha256": evidence["source_snapshot_sha256"],
            "dirty_patch_sha256": evidence["dirty_patch_sha256"],
            "source_evidence_sha256": _sha256_file(control / "source-evidence.json"),
            "source_package_inventory_sha256": _sha256_file(inventory_copy),
            "lock_sha256": _sha256_file(source / "uv.lock"),
        },
        "runtime": {
            "venv": str(venv.resolve()),
            "deps_dir": str(deps.resolve()) if deps is not None else None,
            "hf_home": str(hf_home.resolve()),
            "python": "3.12",
            "lotus": "1.1.4",
            "sentence_transformers": "3.4.1",
            "torch": "2.7.0+cu118",
            "client_request_max_workers": CLIENT_REQUEST_MAX_WORKERS,
        },
        "input": {
            "dataset_commit": LOCOMO_COMMIT,
            "dataset_sha256": LOCOMO_SHA256,
            "bundle_fingerprint": manifest["fingerprint"],
            "policy_input_fingerprint": manifest["policy_input_fingerprint"],
            "case_id": "conv-26",
            "first_event_id": FIRST_EVENT_ID,
            "last_event_id": LAST_EVENT_ID,
            "event_count": EVENT_COUNT,
            "question_count": QUESTION_COUNT,
            "question_numbers": list(numbers),
            "category_counts": {"1": 7, "2": 16, "3": 6, "4": 25, "5": 17},
        },
        "fixed_execution": {
            "system": "claude-memory",
            "grouped_agg_rule": "rule-join-map",
            "semantic_pair_profile": "search-filter",
            "semantic_pair_top_k": 20,
            "semantic_pair_min_similarity": 0.5,
            "embedding_model": "BAAI/bge-m3",
            "embedding_revision": BGE_M3_REVISION,
            "embedding_device": "cuda",
            "retrieval": "listwise",
            "model": "deepseek/deepseek-v4-flash",
            "temperature": 0,
            "thinking": "disabled",
            "lotus_cache": "disabled",
            "semantic_trace": "compact",
        },
        "pricing": {
            "currency": "CNY",
            "per_million_tokens_off_peak": {
                key: str(value) for key, value in OFF_PEAK_PRICES.items()
            },
            "peak_multiplier": str(PEAK_MULTIPLIER),
            "soft_stop_cny": str(SOFT_STOP_CNY),
            "hard_stop_cny": str(HARD_STOP_CNY),
        },
        "conditions": [condition.to_dict() for condition in condition_matrix()],
    }
    from agent_memory.adapters.lotus.json_output import JSON_REPAIR_VERSION

    contract["structured_output"] = {
        "transport": "chat-json-object", "repair_version": JSON_REPAIR_VERSION,
    }
    if (reuse_root is None) != (not reuse_conditions):
        raise ValueError("reuse-root and reuse-condition must be supplied together")
    if reuse_root is not None:
        contract["continuation"] = _capture_continuation(reuse_root, reuse_conditions, contract)
    _atomic_json(control / "experiment-contract.json", contract)
    _set_state(
        ExperimentPaths.from_contract(root, contract), "experiment-state", "initialized"
    )
    _set_state(
        ExperimentPaths.from_contract(root, contract), "current-condition", "none"
    )
    status_script = f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH={shlex.quote(str(source / 'src'))}
exec {shlex.quote(str(venv / 'bin' / 'python'))} \\
  {shlex.quote(str(source / 'tools' / 'evaluation' / 'prompt_batching_scale.py'))} \\
  status --run-root {shlex.quote(str(root))}
"""
    _atomic_text(control / "status.sh", status_script)
    (control / "status.sh").chmod(0o755)


def preflight(root: Path) -> dict[str, Any]:
    """Validate source, input, runtime, model cache, credentials, and capacity."""

    contract = _read_contract(root)
    paths = ExperimentPaths.from_contract(root, contract)
    _validate_continuation(contract)
    evidence_path = paths.control / "source-evidence.json"
    os.environ[SOURCE_EVIDENCE_ENV] = str(evidence_path)
    provenance = collect_runtime_provenance(
        paths.source,
        lockfile="uv.lock",
        dependencies=("agent-memory", "lotus-ai", "sentence-transformers", "torch"),
    )
    validate_run_provenance(provenance, run_mode="prompt-batching-scale")
    source = contract["source"]
    if _git(paths.source, "rev-parse", "HEAD") != source["base_commit"]:
        raise RuntimeError("source base commit changed")
    if _sha256_file(paths.source / "uv.lock") != source["lock_sha256"]:
        raise RuntimeError("source lock digest changed")
    if provenance["source"].get("source_snapshot_sha256") != source[
        "source_snapshot_sha256"
    ]:
        raise RuntimeError("source snapshot does not match the experiment contract")
    if _sha256_file(evidence_path) != source["source_evidence_sha256"]:
        raise RuntimeError("source evidence digest changed")
    if _sha256_file(paths.control / "source-package-inventory.json") != source[
        "source_package_inventory_sha256"
    ]:
        raise RuntimeError("source package inventory digest changed")
    manifest = json.loads((paths.bundle / "manifest.json").read_text())
    expected_input = contract["input"]
    if (
        manifest.get("fingerprint") != expected_input["bundle_fingerprint"]
        or manifest.get("event_count") != EVENT_COUNT
        or manifest.get("question_count") != QUESTION_COUNT
    ):
        raise RuntimeError(f"bundle manifest mismatch: {manifest}")
    events = [json.loads(line) for line in (paths.bundle / "events.jsonl").read_text().splitlines()]
    questions = [
        json.loads(line) for line in (paths.bundle / "questions.jsonl").read_text().splitlines()
    ]
    if (
        len(events) != EVENT_COUNT
        or events[0].get("event_id") != FIRST_EVENT_ID
        or events[-1].get("event_id") != LAST_EVENT_ID
        or len(questions) != QUESTION_COUNT
    ):
        raise RuntimeError("bundle contents do not match the 128-event contract")

    from importlib.metadata import version

    import agent_memory
    import torch
    from agent_memory.adapters.lotus.context import LotusExecutionConfig
    from lotus.models import LM

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"unexpected Python runtime: {sys.version}")
    expected_versions = contract["runtime"]
    if version("lotus-ai") != expected_versions["lotus"]:
        raise RuntimeError(f"unexpected LOTUS version: {version('lotus-ai')}")
    if version("sentence-transformers") != expected_versions["sentence_transformers"]:
        raise RuntimeError(
            f"unexpected SentenceTransformers version: {version('sentence-transformers')}"
        )
    if torch.__version__ != expected_versions["torch"] or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA contract failed: torch={torch.__version__}")
    if paths.source not in Path(agent_memory.__file__).resolve().parents:
        raise RuntimeError(f"wrong agent_memory source: {agent_memory.__file__}")
    config = LotusExecutionConfig()
    from agent_memory.adapters.lotus.json_output import JSON_REPAIR_VERSION

    if contract.get("structured_output") != {
        "transport": config.structured_output_transport, "repair_version": JSON_REPAIR_VERSION,
    }:
        raise RuntimeError("structured output contract changed")
    signature = inspect.signature(LM.__init__)
    if (
        config.lm_max_batch_size != CLIENT_REQUEST_MAX_WORKERS
        or signature.parameters["max_batch_size"].default
        != CLIENT_REQUEST_MAX_WORKERS
        or signature.parameters["temperature"].default != 0.0
    ):
        raise RuntimeError("LOTUS LM defaults do not match the experiment contract")
    bge_path = (
        paths.hf_home
        / "hub"
        / "models--BAAI--bge-m3"
        / "snapshots"
        / BGE_M3_REVISION
    )
    if not bge_path.is_dir():
        raise RuntimeError(f"pinned BGE-M3 revision is not cached: {bge_path}")
    if not (paths.venv / "bin" / "python").is_file():
        raise RuntimeError(f"validated Python environment is missing: {paths.venv}")
    if paths.deps is not None and not paths.deps.is_dir():
        raise RuntimeError(f"dependency overlay is missing: {paths.deps}")
    if _disk_available_gib(Path("/mnt/data")) < MIN_DISK_GIB:
        raise RuntimeError("insufficient /mnt/data free space")
    if _is_peak(datetime.now(tz=HONG_KONG)):
        raise RuntimeError("new experiments start only in a DeepSeek off-peak window")
    active = subprocess.run(
        ("pgrep", "-af", "tools/evaluation/locomo.py run"),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if active:
        raise RuntimeError(f"another LOCOMO worker is active:\n{active}")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is unavailable")
    request = Request(
        "https://api.deepseek.com/user/balance",
        headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urlopen(request, timeout=30) as response:
        balance_payload = json.loads(response.read())
    balances = [
        Decimal(str(item["total_balance"]))
        for item in balance_payload.get("balance_infos", ())
        if item.get("currency") == "CNY"
    ]
    if balance_payload.get("is_available") is not True or len(balances) != 1:
        raise RuntimeError("DeepSeek CNY balance is unavailable")
    if balances[0] < HARD_STOP_CNY:
        raise RuntimeError(f"DeepSeek balance is below 50 CNY: {balances[0]}")
    result = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_commit": source["base_commit"],
        "source_dirty": provenance["source"]["dirty"],
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "bundle_fingerprint": manifest["fingerprint"],
        "event_count": len(events),
        "question_count": len(questions),
        "python": sys.version.split()[0],
        "lotus": version("lotus-ai"),
        "sentence_transformers": version("sentence-transformers"),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "client_request_max_workers": config.lm_max_batch_size,
        "deepseek_balance_cny": str(balances[0]),
        "data_disk_available_gib": round(_disk_available_gib(Path("/mnt/data")), 3),
    }
    _atomic_json(paths.control / "preflight.json", result)
    _set_state(paths, "experiment-state", "prepared")
    _log(paths, "PREFLIGHT_PASSED " + json.dumps(result, sort_keys=True))
    return result


def _runtime_state(output: Path) -> tuple[str, dict[str, Any]]:
    cases = [path for path in (output / "cases").iterdir() if path.is_dir()]
    if len(cases) != 1:
        raise RuntimeError(f"expected one case directory: {cases}")
    current = json.loads((cases[0] / "checkpoints" / "current.json").read_text())
    checkpoint_id = str(current["checkpoint_id"])
    runtime_path = (
        cases[0]
        / "checkpoints"
        / "snapshots"
        / checkpoint_id
        / "driver"
        / "runtime.pkl"
    )
    return checkpoint_id, pickle.loads(runtime_path.read_bytes())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _frame_digest(frame: pd.DataFrame) -> dict[str, Any]:
    rows = [
        {str(column): _json_safe(row[column]) for column in frame.columns}
        for _, row in frame.iterrows()
    ]
    signatures = sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in rows
    )
    return {
        "row_count": len(rows),
        "multiset_digest": sha256("\n".join(signatures).encode()).hexdigest(),
    }


def _peak_rss_kib(time_path: Path) -> int | None:
    if not time_path.is_file():
        return None
    for line in time_path.read_text(errors="replace").splitlines():
        if "Maximum resident set size (kbytes)" in line:
            return int(line.rsplit(":", 1)[1].strip())
    return None


def _expected_prompt_contract(condition: ConditionSpec) -> dict[str, Any] | None:
    if condition.prompt_batch is None:
        return None
    return {
        "max_tasks": None
        if condition.prompt_batch == "all"
        else int(condition.prompt_batch)
    }


def _validate_condition(
    paths: ExperimentPaths, condition: ConditionSpec
) -> dict[str, Any]:
    directory = paths.conditions / condition.run_id
    output = directory / "output"
    trace = output / "trace" / "events.jsonl"
    monitor = _update_monitor(trace, directory / "monitor-state.json")
    if monitor["provider_contract_violations"]:
        raise RuntimeError(str(monitor["provider_contract_violations"]))
    if monitor["reasoning_tokens"] != 0:
        raise RuntimeError("reasoning tokens were observed")
    if monitor["insertion_results"] != EVENT_COUNT:
        raise RuntimeError(f"expected {EVENT_COUNT} insertion results: {monitor}")
    summary = json.loads((output / "metrics" / "summary.json").read_text())
    if (
        summary.get("completed_cases") != 1
        or summary.get("failed_cases") != 0
        or summary.get("question_count") != QUESTION_COUNT
        or summary.get("memory_system_error_question_count") != 0
        or summary.get("retrieval_system_error_count") != 0
    ):
        raise RuntimeError(f"benchmark completion mismatch: {summary}")
    scores = summary.get("scores_by_scorer") or {}
    if (scores.get(OFFICIAL_SCORER_ID) or {}).get("grade_count") != QUESTION_COUNT:
        raise RuntimeError(f"official LOCOMO grades are incomplete: {scores}")
    if (scores.get(JUDGE_SCORER_ID) or {}).get("grade_count") != JUDGED_QUESTION_COUNT:
        raise RuntimeError(f"Zep judge grades are incomplete: {scores}")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("maintenance_rule") != "rule-join-map":
        raise RuntimeError("maintenance rule drifted")
    if manifest.get("retrieval_recipe_id") != "claude-memory-declared-sem-topk:v1:listwise":
        raise RuntimeError("retrieval method is not listwise")
    if manifest.get("framework_cache_mode") != "disabled":
        raise RuntimeError("LOTUS cache must remain disabled")
    runtime = manifest.get("runtime_provenance") or {}
    execution = runtime.get("lotus_execution") or {}
    if (
        execution.get("semantic_pair_profile") != "search-filter"
        or execution.get("semantic_pair_top_k") != 20
        or not _number_equals(execution.get("semantic_pair_min_similarity"), 0.5)
        or execution.get("embedding_device") != "cuda"
        or execution.get("semantic_trace_snapshot_mode") != "compact"
        or execution.get("prompt_batching") != _expected_prompt_contract(condition)
    ):
        raise RuntimeError(f"physical execution contract drifted: {execution}")
    profiles = execution.get("semantic_pair_query_profiles")
    if not isinstance(profiles, Mapping) or len(profiles) != 1:
        raise RuntimeError(f"expected one Claude Search-Filter site: {profiles}")
    profile = next(iter(profiles.values()))
    embedding = profile.get("embedding") or {}
    if (
        profile.get("mode") != "search-filter"
        or profile.get("direction") != "left-to-right"
        or profile.get("top_k") != 20
        or not _number_equals(profile.get("min_similarity"), 0.5)
        or profile.get("embedding_device") != "cuda"
        or embedding.get("model") != "BAAI/bge-m3"
        or embedding.get("revision") != BGE_M3_REVISION
    ):
        raise RuntimeError(f"resolved Search-Filter profile drifted: {profile}")
    refresh = runtime.get("refresh")
    expected_refresh = (
        None
        if condition.refresh_every == 1
        else {"type": "count", "every": condition.refresh_every}
    )
    if refresh != expected_refresh:
        raise RuntimeError(f"refresh contract mismatch: {refresh}")
    batching = monitor["prompt_batching"]
    if condition.prompt_batch is None and batching:
        raise RuntimeError("Original emitted prompt-batching trace events")
    if condition.prompt_batch is not None and not batching:
        raise RuntimeError("Packed condition emitted no prompt-batching trace events")
    if condition.prompt_batch not in (None, "all"):
        oversized = {
            site: row
            for site, row in batching.items()
            if int(row.get("max_chunk") or 0) > int(condition.prompt_batch)
        }
        if oversized:
            raise RuntimeError(f"prompt chunk exceeded configured P: {oversized}")
    if not any(site.startswith("sem_join:") for site in monitor["candidate_pairs"]):
        raise RuntimeError("Claude Search-Filter did not exercise sem_join")

    checkpoint_id, runtime_state = _runtime_state(output)
    engine_state = (
        runtime_state
        if condition.refresh_every == 1
        else runtime_state.get("engine_state")
    )
    if condition.refresh_every > 1:
        pending = runtime_state.get("pending_rows")
        if not isinstance(pending, pd.DataFrame) or not pending.empty:
            raise RuntimeError("final refresh checkpoint contains pending rows")
    state = engine_state.get("state") if isinstance(engine_state, Mapping) else None
    if not isinstance(state, Mapping) or not all(
        isinstance(value, pd.DataFrame) for value in state.values()
    ):
        raise RuntimeError("runtime checkpoint lacks relational state")
    if "log" not in state or len(state["log"]) != EVENT_COUNT:
        raise RuntimeError("final log relation does not contain 128 rows")
    final_state = {name: _frame_digest(frame) for name, frame in sorted(state.items())}
    _atomic_json(directory / "final-state.json", final_state)
    timing = json.loads((directory / "timing.json").read_text())
    insertion = summary["insertion_wall_latency_excluding_trace_io"]
    provider = summary["actual_provider_usage"]
    result = {
        "schema_version": 1,
        **condition.to_dict(),
        "checkpoint_id": checkpoint_id,
        "condition_end_to_end_wall_seconds": float(timing["wall_seconds"]),
        "peak_rss_kib": _peak_rss_kib(directory / "time.txt"),
        "insertion_excluding_trace_mean_ms": insertion["mean_ms"],
        "insertion_excluding_trace_median_ms": insertion["median_ms"],
        "insertion_excluding_trace_p95_ms": insertion["p95_ms"],
        "insertion_excluding_trace_max_ms": insertion["max_ms"],
        "retrieval_mean_ms": summary["retrieval_wall_latency"]["mean_ms"],
        "provider_call_count": provider["provider_call_count"],
        "cache_hit_tokens": provider["cache_hit_tokens"],
        "cache_miss_tokens": provider["cache_miss_tokens"],
        "completion_tokens": provider["completion_tokens"],
        "physical_tokens": provider["total_tokens"],
        "cost_cny": monitor["cost_cny"],
        "official_locomo_score": scores[OFFICIAL_SCORER_ID]["mean_score"],
        "zep_judge_score": scores[JUDGE_SCORER_ID]["mean_score"],
        "candidate_pairs": monitor["candidate_pairs"],
        "operator_usage": monitor["operator_usage"],
        "prompt_batching": batching,
        "provider_usage_by_phase": monitor["phase_usage"],
        "final_state": final_state,
        "artifact_bytes": _directory_bytes(output),
    }
    _atomic_json(directory / "validation.json", result)
    return result


def _safe_stop(
    paths: ExperimentPaths, process: subprocess.Popen[Any], *, reason: str
) -> None:
    try:
        _log(paths, f"SAFETY_STOP reason={reason} pid={process.pid}")
        _set_state(paths, "experiment-state", "safety-stopped")
    finally:
        # A full disk or broken status file must not prevent stopping paid work.
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        else:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)


def _before_condition(paths: ExperimentPaths) -> None:
    cost = _total_cost(paths)
    if cost >= SOFT_STOP_CNY:
        raise RuntimeError(f"48 CNY stop before next condition: {cost}")
    if _disk_available_gib(Path("/mnt/data")) < MIN_DISK_GIB:
        raise RuntimeError("disk safety boundary reached")


def _condition_environment(paths: ExperimentPaths) -> dict[str, str]:
    environment = dict(os.environ)
    python_paths = [str(paths.source / "src")]
    if paths.deps is not None:
        python_paths.append(str(paths.deps))
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["HF_HOME"] = str(paths.hf_home)
    environment[SOURCE_EVIDENCE_ENV] = str(paths.control / "source-evidence.json")
    return environment


def run_experiment(root: Path) -> None:
    """Run pass one, then pass two only after pass one fully succeeds."""

    contract = _read_contract(root)
    paths = ExperimentPaths.from_contract(root, contract)
    if _state(paths, "experiment-state") != "prepared":
        raise RuntimeError("experiment must be prepared by preflight")
    conditions = tuple(_condition_from_dict(item) for item in contract["conditions"])
    _validate_continuation(contract)
    reused = (contract.get("continuation") or {}).get("reused_conditions", {})
    _set_state(paths, "experiment-state", "running")
    _log(paths, f"EXPERIMENT_START conditions={len(conditions)}")
    for condition in conditions:
        if condition.run_id in reused:
            _log(paths, f"REUSED_COMPLETE run_id={condition.run_id} origin={reused[condition.run_id]}")
            continue
        if condition.pass_index == 2:
            incomplete = [
                item.run_id
                for item in conditions
                if item.pass_index == 1
                and item.run_id not in reused
                and (
                    not (paths.conditions / item.run_id / "state").is_file()
                    or (paths.conditions / item.run_id / "state").read_text().strip()
                    != "completed"
                )
            ]
            if incomplete:
                raise RuntimeError(f"second pass cannot start: {incomplete}")
        _before_condition(paths)
        directory = paths.conditions / condition.run_id
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "state"
        if state_path.exists():
            raise RuntimeError(f"condition already has state: {condition.run_id}")
        _atomic_json(
            directory / "condition-contract.json",
            {
                "condition": condition.to_dict(),
                "experiment_contract_sha256": _sha256_file(
                    paths.control / "experiment-contract.json"
                ),
                "client_request_max_workers": CLIENT_REQUEST_MAX_WORKERS,
            },
        )
        _atomic_text(state_path, "running\n")
        _set_state(paths, "current-condition", condition.run_id)
        output = directory / "output"
        command = build_condition_command(
            condition,
            source=paths.source,
            venv=paths.venv,
            bundle=paths.bundle,
            output=output,
        )
        _atomic_json(directory / "command.json", command)
        _log(
            paths,
            f"CONDITION_START run_id={condition.run_id} "
            f"R={condition.refresh_every} P={condition.prompt_batch}",
        )
        started = time.monotonic()
        with (directory / "condition.log").open("ab") as log:
            process = subprocess.Popen(
                ("/usr/bin/time", "-v", "-o", str(directory / "time.txt"), *command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_condition_environment(paths),
                start_new_session=True,
            )
            try:
                _atomic_text(paths.control / "runner.pid", f"{process.pid}\n")
                while process.poll() is None:
                    trace = output / "trace" / "events.jsonl"
                    if trace.exists() and trace.stat().st_size:
                        monitor = _update_monitor(trace, directory / "monitor-state.json")
                        if monitor["reasoning_tokens"] != 0:
                            _safe_stop(paths, process, reason="reasoning_tokens_nonzero")
                            raise RuntimeError("reasoning tokens were observed")
                        if monitor["provider_contract_violations"]:
                            _safe_stop(paths, process, reason="provider_contract_violation")
                            raise RuntimeError(str(monitor["provider_contract_violations"]))
                        if _total_cost(paths) >= HARD_STOP_CNY:
                            _safe_stop(paths, process, reason="50_cny_hard_stop")
                            raise RuntimeError("50 CNY safety boundary reached")
                    if _directory_bytes(output) > MAX_CONDITION_BYTES:
                        _safe_stop(paths, process, reason="artifact_size_limit")
                        raise RuntimeError("condition artifact exceeded 2 GiB")
                    if _disk_available_gib(Path("/mnt/data")) < MIN_DISK_GIB:
                        _safe_stop(paths, process, reason="disk_limit")
                        raise RuntimeError("disk safety boundary reached")
                    if time.monotonic() - started > MAX_CONDITION_SECONDS:
                        _safe_stop(paths, process, reason="condition_timeout")
                        raise RuntimeError("condition exceeded its six-hour limit")
                    time.sleep(POLL_SECONDS)
            except BaseException as error:
                if process.poll() is None:
                    try:
                        _safe_stop(paths, process, reason="supervision_error")
                    except Exception as stop_error:
                        error.add_note(
                            f"Child cleanup failed: {type(stop_error).__name__}: {stop_error}"
                        )
                raise
        _atomic_json(
            directory / "timing.json",
            {"wall_seconds": time.monotonic() - started},
        )
        if process.returncode != 0:
            _atomic_text(state_path, "failed\n")
            raise RuntimeError(
                f"condition failed: {condition.run_id} status={process.returncode}"
            )
        try:
            result = _validate_condition(paths, condition)
        except Exception:
            _atomic_text(state_path, "validation-failed\n")
            raise
        _atomic_text(state_path, "completed\n")
        _log(
            paths,
            f"CONDITION_COMPLETE run_id={condition.run_id} "
            f"cost_cny={result['cost_cny']:.6f}",
        )
        if condition.run_id == "pass-1-packed-all":
            _log(paths, "PASS_COMPLETE pass=1")
    summarize(root)
    _set_state(paths, "current-condition", "none")
    _set_state(paths, "experiment-state", "completed")
    _log(paths, f"EXPERIMENT_COMPLETE total_cost_cny={_total_cost(paths)}")


def summarize(root: Path) -> dict[str, Any]:
    """Write a compact machine-readable summary of all completed conditions."""

    contract = _read_contract(root)
    paths = ExperimentPaths.from_contract(root, contract)
    results = []
    for item in contract["conditions"]:
        run_id = str(item["run_id"])
        directory = _result_directory(root, contract, run_id)
        path = directory / "validation.json"
        if path.exists():
            result = json.loads(path.read_text())
            is_reused = directory != paths.conditions / run_id
            if is_reused:
                result["cost_cny"] = float(contract["continuation"]["prior_condition_costs_cny"][run_id])
            result["result_origin"] = {
                "status": "reused-completed" if is_reused else "completed",
                "directory": str(directory),
                "source": contract["continuation"]["source"] if is_reused else contract["source"],
                "repair_version": None if is_reused else (contract.get("structured_output") or {}).get("repair_version"),
            }
            results.append(result)
    payload = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "completed_conditions": len(results),
        "expected_conditions": len(contract["conditions"]),
        "total_cost_cny": float(_total_cost(paths)),
        "continuation": contract.get("continuation"),
        "runs": results,
        "interpretation_boundary": (
            "The sweep co-varies CountRefresh R and PromptBatching P. The two passes "
            "are exploratory repeatability evidence, not a variance estimate or a "
            "claim of a universal optimum."
        ),
    }
    _atomic_json(paths.control / "sweep-summary.json", payload)
    return payload


def status(root: Path) -> None:
    """Print the authoritative, read-only experiment status."""

    contract = _read_contract(root)
    paths = ExperimentPaths.from_contract(root, contract)
    current = _state(paths, "current-condition") or "none"
    print(f"time_hkt={datetime.now(tz=HONG_KONG).isoformat(timespec='seconds')}")
    print(f"run_root={paths.root}")
    print(f"experiment_state={_state(paths, 'experiment-state') or 'unknown'}")
    print(f"current_condition={current}")
    print(
        f"total_cost_cny={_total_cost(paths):.6f} "
        f"soft_stop={SOFT_STOP_CNY} hard_stop={HARD_STOP_CNY}"
    )
    completed = failed = 0
    for item in contract["conditions"]:
        condition = _condition_from_dict(item)
        directory = _result_directory(root, contract, condition.run_id)
        state = (directory / "state").read_text().strip() if (directory / "state").exists() else "waiting"
        if directory != paths.conditions / condition.run_id:
            state = "reused-completed"
        completed += state in {"completed", "reused-completed"}
        failed += state in {"failed", "validation-failed"}
        monitor_path = directory / "monitor-state.json"
        suffix = ""
        if state == "reused-completed":
            suffix = f" events={EVENT_COUNT}/{EVENT_COUNT} questions={QUESTION_COUNT}/{QUESTION_COUNT} origin={directory}"
            print(f"{condition.run_id}: {state}{suffix}")
            continue
        if monitor_path.exists():
            monitor = json.loads(monitor_path.read_text())
            output = directory / "output"
            suffix = (
                f" events={monitor.get('insertion_results', 0)}/{EVENT_COUNT}"
                f" questions={_completed_questions(output)}/{QUESTION_COUNT}"
                f" responses={monitor.get('provider_responses', 0)}"
                f" cost_cny={float(monitor.get('cost_cny') or 0):.6f}"
            )
        print(f"{condition.run_id}: {state}{suffix}")
    print(f"progress={completed}/{len(contract['conditions'])} failed={failed}")
    print(f"data_disk_available_gib={_disk_available_gib(Path('/mnt/data')):.3f}")
    gpu = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu.stdout.strip():
        print(f"gpu={gpu.stdout.strip()}")
    log_path = paths.control / "controller.log"
    if log_path.exists():
        print("recent_controller_log:")
        for line in log_path.read_text(errors="replace").splitlines()[-12:]:
            print(line)
    if current != "none":
        condition_log = paths.conditions / current / "condition.log"
        if condition_log.exists():
            errors = [
                line
                for line in condition_log.read_text(errors="replace").splitlines()
                if any(token in line for token in ("Traceback", "Error", "Exception"))
            ]
            if errors:
                print("recent_error:")
                for line in errors[-6:]:
                    print(line)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize_parser = commands.add_parser("initialize")
    initialize_parser.add_argument("--run-root", type=Path, required=True)
    initialize_parser.add_argument("--source-dir", type=Path, required=True)
    initialize_parser.add_argument("--venv-dir", type=Path, required=True)
    initialize_parser.add_argument("--deps-dir", type=Path)
    initialize_parser.add_argument("--hf-home", type=Path, required=True)
    initialize_parser.add_argument("--dataset-path", type=Path, required=True)
    initialize_parser.add_argument("--reuse-root", type=Path)
    initialize_parser.add_argument("--reuse-condition", action="append", default=[])
    initialize_parser.add_argument(
        "--source-package-inventory", type=Path, required=True
    )
    for name in ("preflight", "run", "status", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested experiment-controller command."""

    args = _parser().parse_args(argv)
    if args.command == "initialize":
        initialize(
            root=args.run_root,
            source=args.source_dir,
            venv=args.venv_dir,
            deps=args.deps_dir,
            hf_home=args.hf_home,
            dataset=args.dataset_path,
            source_package_inventory=args.source_package_inventory,
            reuse_root=args.reuse_root,
            reuse_conditions=args.reuse_condition,
        )
        return
    try:
        if args.command == "preflight":
            print(json.dumps(preflight(args.run_root), sort_keys=True))
        elif args.command == "run":
            run_experiment(args.run_root)
        elif args.command == "status":
            status(args.run_root)
        else:
            print(json.dumps(summarize(args.run_root), sort_keys=True))
    except BaseException as error:
        if args.command == "run":
            paths = _paths(args.run_root)
            if _state(paths, "experiment-state") != "safety-stopped":
                _set_state(paths, "experiment-state", "failed")
            _log(paths, f"EXPERIMENT_FAILED {type(error).__name__}: {error}")
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
