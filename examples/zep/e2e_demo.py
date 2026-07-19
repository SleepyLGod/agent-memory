"""Run the real Zep LOCOMO storage and retrieval acceptance smoke."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
from sys import path
import time
from typing import Any
from uuid import uuid4
import warnings

from dotenv import load_dotenv
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

import agent_memory as am  # noqa: E402
from agent_memory.adapters.lotus import LotusAdapter  # noqa: E402
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
from agent_memory.memories.zep.storage import (  # noqa: E402
    GRAPHITI_BGE_M3,
    GRAPHITI_NEO4J_SCHEMA,
    GRAPHITI_NEO4J_STATEMENTS,
)
from agent_memory.storage import StorageDeployment  # noqa: E402
from agent_memory.storage.neo4j import (  # noqa: E402
    Neo4jConnector,
    SentenceTransformerCrossEncoderProvider,
    SentenceTransformerEmbeddingProvider,
)
from agent_memory.tracing.semantic import (  # noqa: E402
    append_trace_metrics,
    semantic_trace_scope,
)


LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
LOCOMO_TIMESTAMP_FORMAT = "%I:%M %p on %d %B, %Y"
DEFAULT_START_ROW = 26
DEFAULT_ROW_LIMIT = 3
DEFAULT_ZEP_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_RETRIEVAL_QUERY = "What is Caroline researching and why?"
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


# Input normalization and run configuration.
def default_output_dir() -> Path:
    """Return a fresh timestamped output path for one real run."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / ".memory-test" / "zep-e2e" / timestamp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the Zep insertion smoke."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-row", type=int, default=DEFAULT_START_ROW)
    parser.add_argument("--row-limit", type=int, default=DEFAULT_ROW_LIMIT)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_ZEP_MODEL)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--namespace",
        help="Fresh Neo4j namespace; generated automatically when omitted.",
    )
    parser.add_argument("--query", default=DEFAULT_RETRIEVAL_QUERY)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write real LLM request/output trace artifacts under trace/.",
    )
    return parser.parse_args(argv)


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
    missing = [
        name
        for name in ("AGENT_MEMORY_NEO4J_URI", "AGENT_MEMORY_NEO4J_PASSWORD")
        if not os.getenv(name)
    ]
    if missing:
        raise SystemExit(
            "Neo4j retrieval requires environment variables: " + ", ".join(missing)
        )


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


def policy_input_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
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


# Acceptance artifacts and retrieval assertions.
def write_csv(path_value: Path, frame: pd.DataFrame) -> Path:
    """Write one inspectable CSV artifact."""

    path_value.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path_value, index=False)
    return path_value


def write_json(path_value: Path, value: Any) -> Path:
    """Write one UTF-8 JSON artifact without assuming JSON-native mappings."""

    path_value.parent.mkdir(parents=True, exist_ok=True)
    path_value.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda item: dict(item) if isinstance(item, Mapping) else str(item),
        ),
        encoding="utf-8",
    )
    return path_value


def write_input_artifacts(
    output_dir: Path,
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Write the exact source rows and normalized policy input."""

    return {
        "input/locomo_rows": write_csv(
            output_dir / "input" / "locomo_rows.csv",
            pd.DataFrame(raw_rows),
        ),
        "input/zep_rows": write_csv(
            output_dir / "input" / "zep_rows.csv",
            pd.DataFrame(rows),
        ),
        "input/manifest": write_json(
            output_dir / "input" / "manifest.json",
            {"policy_input_fingerprint": policy_input_fingerprint(rows)},
        ),
    }


def print_run_configuration(
    *,
    dataset_path: Path,
    row_count: int,
    args: argparse.Namespace,
    output_dir: Path,
    namespace: str,
) -> None:
    """Print the small set of values needed to identify one acceptance run."""

    print("ZepMemory real storage and retrieval e2e")
    print(f"LOCOMO cache: {dataset_path}")
    print(f"start_row: {args.start_row}")
    print(f"rows: {row_count}")
    print(f"model: {args.model}")
    print(f"output_dir: {output_dir}")
    print(f"neo4j_namespace: {namespace}")
    print(f"retrieval_query: {args.query}")


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


def add_step_metric(
    memory: am.ZepMemory,
    *,
    add_index: int,
    row: Mapping[str, Any],
    started: float,
    before_usage: dict[str, int],
) -> dict[str, Any]:
    """Collect one insertion's latency, usage, and resulting view sizes."""

    return {
        "add_index": add_index,
        "source_description": row["source_description"],
        "latency_sec": round(time.perf_counter() - started, 4),
        **usage_delta(before_usage, usage_snapshot()),
        **public_view_counts(memory),
    }


def assert_no_semantic_usage(
    before: dict[str, int],
    after: dict[str, int],
) -> None:
    """Fail if a storage retrieval unexpectedly invokes a semantic LLM operator."""

    changed = {
        field: (before[field], after[field])
        for field in USAGE_FIELDS
        if before[field] != after[field]
    }
    if changed:
        raise RuntimeError(f"retrieval changed semantic LLM usage: {changed}")


def validate_retrieval_result(result: am.RetrievalResult) -> None:
    """Require both Zep channels to be non-empty and ranked from one."""

    if tuple(result.channels) != ("entities", "facts"):
        raise RuntimeError("Zep retrieval must return entities and facts channels")
    for name, frame in result.channels.items():
        if frame.empty:
            raise RuntimeError(f"Zep retrieval channel {name!r} is empty")
        ranks = frame["rank"].tolist()
        if ranks != list(range(1, len(frame) + 1)):
            raise RuntimeError(
                f"Zep retrieval channel {name!r} has invalid ranks: {ranks}"
            )


def assert_same_retrieval(
    before: am.RetrievalResult,
    after: am.RetrievalResult,
) -> None:
    """Verify checkpoint restore preserves result identities and ordering."""

    if tuple(before.channels) != tuple(after.channels):
        raise RuntimeError("retrieval channels changed after checkpoint restore")
    for name in before.channels:
        before_ids = before.channels[name]["record_id"].tolist()
        after_ids = after.channels[name]["record_id"].tolist()
        if before_ids != after_ids:
            raise RuntimeError(
                f"retrieval order changed after restore for {name!r}: "
                f"{before_ids!r} != {after_ids!r}"
            )


def write_retrieval_artifacts(
    result: am.RetrievalResult,
    *,
    label: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Write retrieval channels and physical search metrics."""

    written = {
        f"retrieval/{label}/{name}": write_csv(
            output_dir / "retrieval" / label / f"{name}.csv",
            frame,
        )
        for name, frame in result.channels.items()
    }
    written[f"retrieval/{label}/metrics"] = write_json(
        output_dir / "retrieval" / label / "metrics.json",
        {"query": result.query, "channels": result.metrics},
    )
    return written


# Physical deployment and checkpoint round trip.
def create_storage_deployment(namespace: str) -> StorageDeployment:
    """Create the real CPU-only Graphiti-compatible Neo4j deployment."""

    connector = Neo4jConnector(
        uri=os.environ["AGENT_MEMORY_NEO4J_URI"],
        auth=(
            os.getenv("AGENT_MEMORY_NEO4J_USER", "neo4j"),
            os.environ["AGENT_MEMORY_NEO4J_PASSWORD"],
        ),
        database=os.getenv("AGENT_MEMORY_NEO4J_DATABASE", "neo4j"),
        embedding_provider=SentenceTransformerEmbeddingProvider(GRAPHITI_BGE_M3),
        reranker_provider=SentenceTransformerCrossEncoderProvider(),
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    try:
        return StorageDeployment(
            connector=connector,
            statements=GRAPHITI_NEO4J_STATEMENTS,
            namespace=namespace,
        )
    except Exception:
        connector.close()
        raise


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


def write_failure_artifacts(
    error: Exception,
    *,
    memory: am.ZepMemory | None,
    add_index: int,
    phase: str,
    step_metrics: list[dict[str, Any]],
    model: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Preserve setup or execution diagnostics without assuming memory exists."""

    written: dict[str, Path] = {}
    if memory is not None:
        written.update(write_state_artifacts(memory, output_dir))
    written.update(write_metrics(step_metrics, model=model, output_dir=output_dir))
    written["diagnostics/failure"] = write_json(
        output_dir / "diagnostics" / "failure.json",
        {
            "add_index": add_index,
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    return written


def save_and_restore_checkpoint(
    memory: am.ZepMemory,
    *,
    adapter: LotusAdapter,
    output_dir: Path,
    storage: StorageDeployment,
) -> tuple[dict[str, Path], am.ZepMemory]:
    """Save a v2 checkpoint and verify a no-LLM restore round trip."""

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot = memory._runtime.snapshot_state()
    state_path = checkpoint_dir / "state.pkl"
    state_path.write_bytes(pickle.dumps(snapshot))
    restored = am.ZepMemory(adapter=adapter, storage=storage)
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
                "storage_bound": True,
                "storage_commit": snapshot.get("storage_commit"),
                "round_trip_verified": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return (
        {
            "checkpoint/state": state_path,
            "checkpoint/metadata": metadata_path,
        },
        restored,
    )


# End-to-end acceptance flow.
def main() -> None:
    """Run real Zep add, retrieval, and checkpoint verification."""

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
    namespace = args.namespace or f"zep-e2e-{uuid4()}"
    written = write_input_artifacts(
        output_dir,
        raw_rows=raw_rows,
        rows=rows,
    )
    print_run_configuration(
        dataset_path=dataset_path,
        row_count=len(rows),
        args=args,
        output_dir=output_dir,
        namespace=namespace,
    )
    storage: StorageDeployment | None = None
    memory: am.ZepMemory | None = None
    step_metrics: list[dict[str, Any]] = []
    current_index = 0
    current_phase = "setup_storage"
    try:
        # Setup belongs to the same ownership boundary as execution and cleanup.
        storage = create_storage_deployment(namespace)

        current_phase = "setup_adapter"
        adapter = LotusAdapter(
            model=args.model,
            config=LotusExecutionConfig(semantic_trace_dir=trace_dir),
        )
        current_phase = "setup_memory"
        memory = am.ZepMemory(adapter=adapter, storage=storage)

        # Materialize each source episode before exercising physical retrieval.
        current_phase = "add"
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
            metric = add_step_metric(
                memory,
                add_index=current_index,
                row=row,
                started=started,
                before_usage=before,
            )
            step_metrics.append(metric)
            print(
                "  rows: "
                + ", ".join(
                    f"{name}={metric[f'{name}_rows']}" for name in PUBLIC_VIEWS
                )
            )

        # Retrieval is storage-backed and must not invoke semantic LLM operators.
        current_phase = "retrieval_before_checkpoint"
        before_usage = usage_snapshot()
        before_retrieval = memory.query(args.query)
        after_usage = usage_snapshot()
        assert_no_semantic_usage(before_usage, after_usage)
        validate_retrieval_result(before_retrieval)
        written.update(
            write_retrieval_artifacts(
                before_retrieval,
                label="before_checkpoint",
                output_dir=output_dir,
            )
        )

        # Restore the same logical and physical commit before querying again.
        current_phase = "checkpoint"
        checkpoint_artifacts, restored = save_and_restore_checkpoint(
            memory,
            adapter=adapter,
            output_dir=output_dir,
            storage=storage,
        )
        written.update(checkpoint_artifacts)

        current_phase = "retrieval_after_restore"
        before_usage = usage_snapshot()
        after_retrieval = restored.query(args.query)
        after_usage = usage_snapshot()
        assert_no_semantic_usage(before_usage, after_usage)
        validate_retrieval_result(after_retrieval)
        assert_same_retrieval(before_retrieval, after_retrieval)
        written.update(
            write_retrieval_artifacts(
                after_retrieval,
                label="after_restore",
                output_dir=output_dir,
            )
        )
        written["retrieval/summary"] = write_json(
            output_dir / "retrieval" / "summary.json",
            {
                "query": args.query,
                "namespace": namespace,
                "semantic_llm_usage_changed": False,
                "checkpoint_order_verified": True,
            },
        )

        written.update(write_state_artifacts(memory, output_dir))
        written.update(
            write_metrics(step_metrics, model=args.model, output_dir=output_dir)
        )
        if trace_dir is not None:
            append_trace_metrics(trace_dir, step_metrics)
            written["trace"] = trace_dir

        print("\nwrote Zep e2e artifacts:")
        for name, path_value in written.items():
            print(f"- {name}: {path_value}")
    except Exception as error:
        written.update(
            write_failure_artifacts(
                error,
                memory=memory,
                add_index=current_index,
                phase=current_phase,
                step_metrics=step_metrics,
                model=args.model,
                output_dir=output_dir,
            )
        )
        failure_path = written["diagnostics/failure"]
        print(f"failure artifact: {failure_path}")
        raise
    finally:
        if storage is not None:
            storage.connector.close()


if __name__ == "__main__":
    main()
