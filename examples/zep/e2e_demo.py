"""Run a real LOTUS-backed ZepMemory insertion smoke over LOCOMO rows."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
from sys import path
import time
from typing import Any
import warnings

from dotenv import load_dotenv
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

import agent_memory as am  # noqa: E402
from agent_memory.adapters.lotus import DEFAULT_LOTUS_MODEL, LotusAdapter  # noqa: E402
from agent_memory.adapters.lotus.context import LotusExecutionConfig  # noqa: E402
from agent_memory.adapters.lotus.structured import (  # noqa: E402
    reset_structured_retry_stats,
    structured_retry_stats,
)
from agent_memory.datasets.locomo import (  # noqa: E402
    DEFAULT_LOCOMO_URL,
    ensure_locomo_dataset,
    load_locomo_rows,
)
from agent_memory.tracing.semantic import (  # noqa: E402
    append_trace_metrics,
    semantic_trace_scope,
)


LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
LOCOMO_TIMESTAMP_FORMAT = "%I:%M %p on %d %B, %Y"
DEFAULT_START_ROW = 26
DEFAULT_ROW_LIMIT = 3
PUBLIC_VIEWS = ("episodes", "entities", "facts")
USAGE_FIELDS = (
    "physical_prompt_tokens",
    "physical_completion_tokens",
    "physical_total_tokens",
    "virtual_prompt_tokens",
    "virtual_completion_tokens",
    "virtual_total_tokens",
    "cache_hits",
    "structured_retry_batches",
    "structured_retry_rows",
    "structured_failure_artifacts",
)


def default_output_dir() -> Path:
    """Return a fresh timestamped output path for one real run."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / ".memory-test" / "zep-e2e" / timestamp


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Zep insertion smoke."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-row", type=int, default=DEFAULT_START_ROW)
    parser.add_argument("--row-limit", type=int, default=DEFAULT_ROW_LIMIT)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_LOTUS_MODEL)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write real LLM request/output trace artifacts under trace/.",
    )
    return parser.parse_args()


def require_environment() -> None:
    """Load the local API key and fail before creating run artifacts if absent."""

    load_dotenv(PROJECT_ROOT / ".env")
    warnings.filterwarnings(
        "ignore",
        message="Error calculating completion cost - cost metrics will be inaccurate.*",
        category=UserWarning,
    )
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is required for the real Zep e2e demo")


def prepare_output_dir(output_dir: Path) -> None:
    """Create an empty output directory without deleting prior artifacts."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"--output-dir must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def locomo_reference_time(value: str) -> str:
    """Convert LOCOMO's timezone-free timestamp to ISO 8601 without adding a zone."""

    parsed = datetime.strptime(value.strip(), LOCOMO_TIMESTAMP_FORMAT)
    return parsed.isoformat(timespec="seconds")


def zep_log_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map one normalized LOCOMO turn to the ZepMemory source schema."""

    speaker = str(row.get("speaker", ""))
    sample_index = row.get("sample_index")
    if not isinstance(sample_index, int) or isinstance(sample_index, bool):
        raise ValueError("normalized LOCOMO row requires an integer sample_index")
    session_id = str(row.get("session_id", ""))
    session_number = session_id.removeprefix("session_")
    content = f'{speaker}: {str(row["message"])}'
    caption = str(row.get("blip_caption", "")).strip()
    if caption:
        content += f"\n(description of attached image: {caption})"
    return {
        "content": content,
        "role": speaker,
        "speaker": speaker,
        "reference_time": locomo_reference_time(str(row["timestamp"])),
        "source_description": f"LOCOMO sample {sample_index} session {session_number}",
    }


def policy_input_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Hash the normalized source rows shared with the native baseline."""

    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selected_rows(
    *,
    start_row: int,
    row_limit: int,
    sample_limit: int,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a LOCOMO slice and return both raw and Zep-shaped rows."""

    if start_row < 1:
        raise SystemExit("--start-row must be at least 1")
    if row_limit < 1:
        raise SystemExit("--row-limit must be at least 1")
    if sample_limit < 1:
        raise SystemExit("--sample-limit must be at least 1")
    dataset_path = ensure_locomo_dataset(LOCOMO_CACHE_PATH, url=DEFAULT_LOCOMO_URL)
    rows = load_locomo_rows(
        dataset_path,
        sample_limit=sample_limit,
        turn_limit=start_row + row_limit - 1,
    )
    selected = rows[start_row - 1 : start_row - 1 + row_limit]
    if len(selected) != row_limit:
        raise SystemExit(
            f"Requested {row_limit} LOCOMO rows from {start_row}, found {len(selected)}"
        )
    return dataset_path, selected, [zep_log_row(row) for row in selected]


def write_csv(path_value: Path, frame: pd.DataFrame) -> Path:
    """Write one inspectable CSV artifact."""

    path_value.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path_value, index=False)
    return path_value


def usage_snapshot() -> dict[str, int]:
    """Return current LOTUS usage and structured retry counters."""

    import lotus

    retry_stats = structured_retry_stats()
    result = {
        "physical_prompt_tokens": 0,
        "physical_completion_tokens": 0,
        "physical_total_tokens": 0,
        "virtual_prompt_tokens": 0,
        "virtual_completion_tokens": 0,
        "virtual_total_tokens": 0,
        "cache_hits": 0,
        "structured_retry_batches": retry_stats.retry_batches,
        "structured_retry_rows": retry_stats.retry_rows,
        "structured_failure_artifacts": retry_stats.failure_artifacts,
    }
    lm = lotus.settings.lm
    if lm is None:
        return result
    stats = lm.stats
    result.update(
        {
            "physical_prompt_tokens": stats.physical_usage.prompt_tokens,
            "physical_completion_tokens": stats.physical_usage.completion_tokens,
            "physical_total_tokens": stats.physical_usage.total_tokens,
            "virtual_prompt_tokens": stats.virtual_usage.prompt_tokens,
            "virtual_completion_tokens": stats.virtual_usage.completion_tokens,
            "virtual_total_tokens": stats.virtual_usage.total_tokens,
            "cache_hits": stats.cache_hits,
        }
    )
    return result


def usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Return non-negative usage changes for one add step."""

    return {field: max(0, after[field] - before[field]) for field in USAGE_FIELDS}


def public_view_counts(memory: am.ZepMemory) -> dict[str, int]:
    """Return row counts for the three baseline Zep views."""

    state = memory._runtime._state
    return {f"{name}_rows": len(state.get(name, ())) for name in PUBLIC_VIEWS}


def write_state_artifacts(memory: am.ZepMemory, output_dir: Path) -> dict[str, Path]:
    """Write current source and public view tables for success or failure analysis."""

    state = memory._runtime._state
    written = {
        "state/log": write_csv(
            output_dir / "state" / "log.csv",
            state.get("log", pd.DataFrame()),
        ),
    }
    for name in PUBLIC_VIEWS:
        written[f"views/{name}"] = write_csv(
            output_dir / "views" / f"{name}.csv",
            state.get(name, pd.DataFrame()),
        )
    return written


def write_metrics(
    rows: list[dict[str, Any]],
    *,
    model: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Write per-step and total latency/token metrics."""

    steps = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "model": model,
        "completed_adds": len(rows),
        "latency_sec": round(sum(float(row["latency_sec"]) for row in rows), 4),
    }
    for field in USAGE_FIELDS:
        summary[field] = sum(int(row[field]) for row in rows)
    return {
        "metrics/steps": write_csv(output_dir / "metrics" / "steps.csv", steps),
        "metrics/summary": write_csv(
            output_dir / "metrics" / "summary.csv",
            pd.DataFrame([summary]),
        ),
    }


def save_and_verify_checkpoint(
    memory: am.ZepMemory,
    *,
    adapter: LotusAdapter,
    output_dir: Path,
) -> dict[str, Path]:
    """Save a v2 checkpoint and verify a no-LLM restore round trip."""

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot = memory._runtime.snapshot_state()
    state_path = checkpoint_dir / "state.pkl"
    state_path.write_bytes(pickle.dumps(snapshot))
    restored = am.ZepMemory(adapter=adapter)
    restored._runtime.restore_state(pickle.loads(state_path.read_bytes()))
    for name in PUBLIC_VIEWS:
        pd.testing.assert_frame_equal(
            memory._runtime._state[name],
            restored._runtime._state[name],
        )
    metadata_path = checkpoint_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": snapshot["schema_version"],
                "policy_fingerprint": memory._runtime.policy.fingerprint,
                "round_trip_verified": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint/state": state_path,
        "checkpoint/metadata": metadata_path,
    }


def main() -> None:
    """Run sequential real Zep insertion and write inspectable artifacts."""

    args = parse_args()
    require_environment()
    reset_structured_retry_stats()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir)
    trace_dir = output_dir / "trace" if args.trace else None
    dataset_path, raw_rows, rows = selected_rows(
        start_row=args.start_row,
        row_limit=args.row_limit,
        sample_limit=args.sample_limit,
    )
    written = {
        "input/locomo_rows": write_csv(
            output_dir / "input" / "locomo_rows.csv",
            pd.DataFrame(raw_rows),
        ),
        "input/zep_rows": write_csv(
            output_dir / "input" / "zep_rows.csv",
            pd.DataFrame(rows),
        ),
    }
    input_manifest = output_dir / "input" / "manifest.json"
    input_manifest.write_text(
        json.dumps(
            {"policy_input_fingerprint": policy_input_fingerprint(rows)},
            indent=2,
        ),
        encoding="utf-8",
    )
    written["input/manifest"] = input_manifest
    print("ZepMemory real insertion e2e")
    print(f"LOCOMO cache: {dataset_path}")
    print(f"start_row: {args.start_row}")
    print(f"rows: {len(rows)}")
    print(f"model: {args.model}")
    print(f"output_dir: {output_dir}")

    adapter = LotusAdapter(
        model=args.model,
        config=LotusExecutionConfig(semantic_trace_dir=trace_dir),
    )
    memory = am.ZepMemory(adapter=adapter)
    step_metrics: list[dict[str, Any]] = []
    current_index = 0
    try:
        for current_index, row in enumerate(rows, start=1):
            print(f"add[{current_index}]: {row['content'][:100]}")
            before = usage_snapshot()
            started = time.perf_counter()
            with semantic_trace_scope(
                run_kind="zep",
                phase="add",
                add_index=current_index,
                source_description=row["source_description"],
            ):
                memory.add(row)
            metric = {
                "add_index": current_index,
                "source_description": row["source_description"],
                "latency_sec": round(time.perf_counter() - started, 4),
                **usage_delta(before, usage_snapshot()),
                **public_view_counts(memory),
            }
            step_metrics.append(metric)
            print(
                "  rows: "
                + ", ".join(
                    f"{name}={metric[f'{name}_rows']}" for name in PUBLIC_VIEWS
                )
            )
    except Exception as error:
        written.update(write_state_artifacts(memory, output_dir))
        written.update(write_metrics(step_metrics, model=args.model, output_dir=output_dir))
        failure_path = output_dir / "diagnostics" / "failure.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "add_index": current_index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"failure artifact: {failure_path}")
        raise

    written.update(write_state_artifacts(memory, output_dir))
    written.update(write_metrics(step_metrics, model=args.model, output_dir=output_dir))
    written.update(
        save_and_verify_checkpoint(memory, adapter=adapter, output_dir=output_dir)
    )
    if trace_dir is not None:
        append_trace_metrics(trace_dir, step_metrics)
        written["trace"] = trace_dir

    print("\nwrote Zep e2e artifacts:")
    for name, path_value in written.items():
        print(f"- {name}: {path_value}")


if __name__ == "__main__":
    main()
