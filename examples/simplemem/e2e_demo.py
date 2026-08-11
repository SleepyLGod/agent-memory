"""Real LOTUS-backed SimpleMemMemory e2e demo over LOCOMO rows."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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

from agent_memory.datasets.locomo import (  # noqa: E402
    DEFAULT_LOCOMO_URL,
    ensure_locomo_dataset,
    load_locomo_rows,
)
from agent_memory.adapters.lotus import DEFAULT_LOTUS_MODEL, LotusAdapter  # noqa: E402
from agent_memory.adapters.lotus.context import LotusExecutionConfig  # noqa: E402
from agent_memory.adapters.lotus.structured import (  # noqa: E402
    reset_structured_retry_stats,
    structured_retry_stats,
)
from agent_memory.tracing.semantic import (  # noqa: E402
    append_trace_metrics,
    semantic_trace_scope,
)
from agent_memory.memories.simplemem import (  # noqa: E402
    SimpleMemMemory,
    SimpleMemMemoryEnhanced,
    WINDOW_SIZE,
    WINDOW_SLIDE,
)
from agent_memory.memories.simplemem.storage import (  # noqa: E402
    SIMPLEMEM_BGE_M3,
    SIMPLEMEM_ENHANCED_NEO4J_STATEMENTS,
    SIMPLEMEM_NEO4J_SCHEMA,
    SIMPLEMEM_NEO4J_STATEMENTS,
)
from agent_memory.policy.retrieval import RetrievalResult  # noqa: E402
from agent_memory.storage import (  # noqa: E402
    SentenceTransformerEmbeddingProvider,
    StorageDeployment,
)
from agent_memory.storage.neo4j import (  # noqa: E402
    Neo4jConnector,
    SentenceTransformerCrossEncoderProvider,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".memory-test" / "simplemem-e2e" / "latest"
LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
DEFAULT_SAMPLE_LIMIT = 1
DEFAULT_START_ROW = 26
DEFAULT_ROW_LIMIT = 43
DEFAULT_QUERY = "Which memories are most useful for future collaboration style?"
DEFAULT_MODEL = DEFAULT_LOTUS_MODEL
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--row-limit", type=int, default=DEFAULT_ROW_LIMIT,
        help="Number of LOCOMO dialogue rows to append.",
    )
    parser.add_argument(
        "--start-row", type=int, default=DEFAULT_START_ROW,
        help="1-based LOCOMO dialogue row offset.",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT,
        help="Number of LOCOMO conversation samples.",
    )
    parser.add_argument(
        "--query", default=DEFAULT_QUERY,
        help="Retrieval query to run after maintaining memory.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LiteLLM model passed to LotusAdapter. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--policy", choices=("base", "enhanced"), default="base",
        help="SimpleMem policy variant: base (semantic-only) or enhanced (hybrid).",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Fresh Neo4j namespace; generated automatically when omitted.",
    )
    parser.add_argument(
        "--print-steps", action="store_true",
        help="Print view row counts after each add().",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="Write unified semantic trace artifacts under trace/.",
    )
    return parser.parse_args()


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"--output-dir must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, frame: Any, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def print_frame(label: str, frame: Any) -> None:
    print(f"\n--- {label} ({len(frame)} rows) ---")
    if len(frame) == 0:
        print("  (empty)")
        return
    for i, row in frame.iterrows():
        row_dict = row.to_dict()
        cols = list(row_dict.keys())
        print(f"  [{i}] {row_dict.get(cols[0], '')!r}")
        for col in cols[1:4]:
            val = row_dict.get(col, "")
            if val:
                print(f"       {col}: {val!r}")


def usage_snapshot() -> dict[str, float | int]:
    import lotus
    retry_stats = structured_retry_stats()
    snapshot: dict[str, float | int] = {
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
        return snapshot
    stats = lm.stats
    snapshot.update({
        "physical_prompt_tokens": stats.physical_usage.prompt_tokens,
        "physical_completion_tokens": stats.physical_usage.completion_tokens,
        "physical_total_tokens": stats.physical_usage.total_tokens,
        "virtual_prompt_tokens": stats.virtual_usage.prompt_tokens,
        "virtual_completion_tokens": stats.virtual_usage.completion_tokens,
        "virtual_total_tokens": stats.virtual_usage.total_tokens,
        "cache_hits": stats.cache_hits,
    })
    return snapshot


def usage_delta(before: dict[str, float | int], after: dict[str, float | int]) -> dict[str, float | int | bool]:
    delta = {field: after[field] - before[field] for field in USAGE_FIELDS}
    delta["had_structured_retry"] = bool(delta["structured_retry_batches"])
    return delta


def run_measured(*, run_kind: str, phase: str, action: Callable[[], Any], **metadata: Any) -> tuple[Any, dict[str, Any]]:
    before = usage_snapshot()
    start = time.perf_counter()
    with semantic_trace_scope(run_kind=run_kind, phase=phase, **metadata):
        result = action()
    latency_sec = time.perf_counter() - start
    after = usage_snapshot()
    metric = {
        "run_kind": run_kind,
        "phase": phase,
        "latency_sec": round(latency_sec, 4),
        **usage_delta(before, after),
        **metadata,
    }
    return result, metric


def require_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    warnings.filterwarnings(
        "ignore",
        message="Error calculating completion cost - cost metrics will be inaccurate.*",
        category=UserWarning,
    )
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is required for SimpleMemMemory e2e. "
            "Set it in .env or export it in the shell."
        )
    missing = [
        name
        for name in ("AGENT_MEMORY_NEO4J_URI", "AGENT_MEMORY_NEO4J_PASSWORD")
        if not os.getenv(name)
    ]
    if missing:
        raise SystemExit(
            "Neo4j retrieval requires environment variables: " + ", ".join(missing)
        )


def simplemem_log_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": row["message"],
        "speaker": row.get("speaker", ""),
        "timestamp": row.get("timestamp", ""),
    }


def selected_rows(*, start_row: int, row_limit: int, sample_limit: int) -> tuple[Path, list[dict[str, Any]]]:
    if start_row < 1:
        raise SystemExit("--start-row must be at least 1")
    if row_limit < 1:
        raise SystemExit("--row-limit must be at least 1")
    dataset_path = ensure_locomo_dataset(LOCOMO_CACHE_PATH, url=DEFAULT_LOCOMO_URL)
    locomo_rows = load_locomo_rows(
        dataset_path,
        sample_limit=sample_limit,
        turn_limit=start_row + row_limit - 1,
    )
    selected = locomo_rows[start_row - 1 : start_row - 1 + row_limit]
    rows = [simplemem_log_row(row) for row in selected]
    if not rows:
        raise SystemExit(f"No LOCOMO rows loaded from {dataset_path}")
    return dataset_path, rows


def main() -> None:
    args = parse_args()
    require_environment()
    reset_structured_retry_stats()
    output_dir = args.output_dir.resolve()
    reset_output_dir(output_dir)
    input_dir = output_dir / "input"
    differential_dir = output_dir / "differential"
    metrics_dir = output_dir / "metrics"
    trace_dir = output_dir / "trace" if args.trace else None

    dataset_path, rows = selected_rows(
        start_row=args.start_row,
        row_limit=args.row_limit,
        sample_limit=args.sample_limit,
    )
    namespace = args.namespace or f"simplemem-e2e-{os.getpid()}"
    print("SimpleMemMemory real e2e demo")
    print(f"LOCOMO cache: {dataset_path}")
    print(f"start_row: {args.start_row}")
    print(f"rows: {len(rows)}")
    print(f"model: {args.model}")
    print(f"policy: {args.policy}")
    print(f"output_dir: {output_dir}")
    print(f"neo4j_namespace: {namespace}")
    print(f"window size/slide: {WINDOW_SIZE}/{WINDOW_SLIDE}")
    write_csv("locomo_rows", pd.DataFrame(rows), input_dir)

    adapter = LotusAdapter(
        model=args.model,
        config=LotusExecutionConfig(
            semantic_trace_dir=trace_dir,
            structured_max_tokens=16384,
            lm_model_kwargs={"max_tokens": 16384},
        ),
    )
    connector = Neo4jConnector(
        uri=os.environ["AGENT_MEMORY_NEO4J_URI"],
        auth=(
            os.getenv("AGENT_MEMORY_NEO4J_USER", "neo4j"),
            os.environ["AGENT_MEMORY_NEO4J_PASSWORD"],
        ),
        database=os.getenv("AGENT_MEMORY_NEO4J_DATABASE", "neo4j"),
        embedding_provider=SentenceTransformerEmbeddingProvider(
            SIMPLEMEM_BGE_M3,
            device="cpu",
            dependency_extra="zep",
        ),
        reranker_provider=SentenceTransformerCrossEncoderProvider(),
        schema=SIMPLEMEM_NEO4J_SCHEMA,
    )
    statements = (
        SIMPLEMEM_ENHANCED_NEO4J_STATEMENTS
        if args.policy == "enhanced"
        else SIMPLEMEM_NEO4J_STATEMENTS
    )
    storage = StorageDeployment(
        connector=connector,
        statements=statements,
        namespace=namespace,
    )
    memory_cls = (
        SimpleMemMemoryEnhanced if args.policy == "enhanced" else SimpleMemMemory
    )
    memory = memory_cls(adapter=adapter, storage=storage)
    step_metrics: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        print(f"add[{index}]: {row.get('speaker', '')}: {row.get('content', '')[:100]}")
        _result, metric = run_measured(
            run_kind="differential",
            phase="add",
            action=lambda row=row: memory.add(row),
            step_index=index,
            speaker=row.get("speaker", ""),
        )
        metric.update({"log_rows": len(memory._runtime._state.get("log", ()))})
        step_metrics.append(metric)
        if args.print_steps and index % 10 == 0:
            facts = memory._runtime._state.get("facts", [])
            print(f"  log rows: {len(memory._runtime._state.get('log', ()))}")
            print(f"  facts rows: {len(facts)}")

    result, query_metric = run_measured(
        run_kind="differential",
        phase="differential_query",
        action=lambda: memory.query(args.query),
    )
    if isinstance(result, RetrievalResult):
        query_frame = result.channels["facts"]
    else:
        query_frame = result
    query_metric["query_result_rows"] = len(query_frame)
    phase_metrics = [query_metric]

    log_state = memory._runtime._state.get("log", [])
    facts = memory._runtime._state.get("facts", [])

    print_frame("log", log_state)
    print_frame("facts", facts)
    print_frame("query result", query_frame)

    written = {
        "input/locomo_rows": input_dir / "locomo_rows.csv",
        "differential/log": write_csv("log", log_state, differential_dir),
        "differential/facts": write_csv("facts", facts, differential_dir),
        "differential/query_result": write_csv("query_result", query_frame, differential_dir),
    }

    steps_frame = pd.DataFrame(step_metrics)
    phases_frame = pd.DataFrame(phase_metrics)
    written["metrics/steps"] = write_csv("steps", steps_frame, metrics_dir)
    written["metrics/phases"] = write_csv("phases", phases_frame, metrics_dir)

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot = memory._runtime.snapshot_state()
    (checkpoint_dir / "state.pkl").write_bytes(pickle.dumps(snapshot))
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps({
            "schema_version": snapshot["schema_version"],
            "policy_fingerprint": memory._runtime.policy.fingerprint,
        }, indent=2), encoding="utf-8",
    )
    written["checkpoint/state"] = checkpoint_dir / "state.pkl"
    written["checkpoint/metadata"] = checkpoint_dir / "metadata.json"

    if trace_dir is not None:
        append_trace_metrics(trace_dir, [*step_metrics, *phase_metrics])

    connector.close()

    print("\nwrote CSV artifacts:")
    for name, path_value in written.items():
        print(f"- {name}: {path_value}")


if __name__ == "__main__":
    main()
