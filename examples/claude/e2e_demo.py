"""Real LOTUS-backed ClaudeMemory e2e demo over LOCOMO rows.

Run with a local .env containing DEEPSEEK_API_KEY, or export the key in the
shell before running this script. The script writes local CSV artifacts under
.memory-test/ so the maintained memory tables can be inspected manually.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
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
from analyze_e2e_output import compare_public_views, print_comparison, print_frame  # noqa: E402
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
from agent_memory.policy.logical import MemorySpec, QueryExpr, UserQuery  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".memory-test" / "claude-e2e" / "latest"
LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
DEFAULT_SAMPLE_LIMIT = 1
DEFAULT_START_ROW = 26
DEFAULT_ROW_LIMIT = 7
DEFAULT_QUERY = "Which memories are most useful for future collaboration style?"
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
    """Parse CLI options for the Claude e2e demo."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV outputs. Defaults to .memory-test/claude-e2e/latest.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=DEFAULT_ROW_LIMIT,
        help="Number of LOCOMO dialogue rows to append.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=DEFAULT_START_ROW,
        help="1-based LOCOMO dialogue row offset for the demo slice.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help="Number of LOCOMO conversation samples to read before row limiting.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Retrieval query to run after maintaining ClaudeMemory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LOTUS_MODEL,
        help=f"LiteLLM model passed to LotusAdapter. Defaults to {DEFAULT_LOTUS_MODEL}.",
    )
    parser.add_argument(
        "--compare-full",
        action="store_true",
        help="Also execute full recompute queries and write comparison CSVs.",
    )
    parser.add_argument(
        "--print-steps",
        action="store_true",
        help="Print view row counts after each add().",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write unified semantic trace artifacts under trace/.",
    )
    return parser.parse_args()


def reset_output_dir(output_dir: Path) -> None:
    """Create a fresh output directory without deleting prior artifacts."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"--output-dir must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, frame: Any, output_dir: Path) -> Path:
    """Write a DataFrame-like object to a named CSV and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def usage_snapshot() -> dict[str, float | int]:
    """Return current LOTUS usage and structured retry counters."""

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
    snapshot.update(
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
    return snapshot


def usage_delta(
    before: dict[str, float | int],
    after: dict[str, float | int],
) -> dict[str, float | int | bool]:
    """Return usage delta between two snapshots."""

    delta = {field: after[field] - before[field] for field in USAGE_FIELDS}
    delta["had_structured_retry"] = bool(delta["structured_retry_batches"])
    return delta


def run_measured(
    *,
    run_kind: str,
    phase: str,
    action: Callable[[], Any],
    **metadata: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run one action and return its result plus demo-level metrics."""

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
    """Load local env and fail early when the default LOTUS model cannot run."""

    load_dotenv(PROJECT_ROOT / ".env")
    warnings.filterwarnings(
        "ignore",
        message="Error calculating completion cost - cost metrics will be inaccurate.*",
        category=UserWarning,
    )
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is required for ClaudeMemory e2e. "
            "Set it in .env or export it in the shell."
        )


def selected_rows(
    *,
    start_row: int,
    row_limit: int,
    sample_limit: int,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load LOCOMO rows and normalize them into ClaudeMemory log rows."""

    if start_row < 1:
        raise SystemExit("--start-row must be at least 1")
    if row_limit < 1:
        raise SystemExit("--row-limit must be at least 1")
    if sample_limit < 1:
        raise SystemExit("--sample-limit must be at least 1")

    dataset_path = ensure_locomo_dataset(LOCOMO_CACHE_PATH, url=DEFAULT_LOCOMO_URL)
    locomo_rows = load_locomo_rows(
        dataset_path,
        sample_limit=sample_limit,
        turn_limit=start_row + row_limit - 1,
    )
    selected = locomo_rows[start_row - 1 : start_row - 1 + row_limit]
    rows = [claude_log_row(row) for row in selected]
    if not rows:
        raise SystemExit(f"No LOCOMO rows loaded from {dataset_path}")
    return dataset_path, rows


def claude_log_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one LOCOMO row into the ClaudeMemory log schema."""

    return {
        "message": row["message"],
        "role": row.get("speaker", ""),
        "timestamp": row.get("timestamp", ""),
        "session_id": row.get("session_id", ""),
        "metadata": {
            "source": "locomo",
            "speaker": row.get("speaker", ""),
            "turn_id": row.get("turn_id", ""),
        },
    }


def run_differential(
    rows: list[dict[str, Any]],
    *,
    model: str,
    print_steps: bool,
    semantic_trace_dir: Path | None,
) -> tuple[am.ClaudeMemory, LotusAdapter, list[dict[str, Any]]]:
    """Maintain ClaudeMemory by appending rows through runtime Q' execution."""

    adapter = LotusAdapter(
        model=model,
        config=LotusExecutionConfig(
            semantic_trace_dir=semantic_trace_dir,
        ),
    )
    memory = am.ClaudeMemory(adapter=adapter)
    step_metrics: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        print(f"add[{index}]: {row['role']}: {row['message'][:100]}")
        _result, metric = run_measured(
            run_kind="differential",
            phase="add",
            action=lambda row=row: memory.add(row),
            step_index=index,
            role=row.get("role", ""),
            session_id=row.get("session_id", ""),
            turn_id=turn_id(row),
        )
        metric.update(state_row_counts(memory))
        step_metrics.append(metric)
        if print_steps:
            print_step_counts(memory)
    return memory, adapter, step_metrics


def turn_id(row: dict[str, Any]) -> str:
    """Return a readable turn id from a normalized log row."""

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("turn_id", ""))
    return ""


def state_row_counts(memory: am.ClaudeMemory) -> dict[str, int]:
    """Return compact runtime row counts."""

    state = memory._runtime._state
    return {
        "log_rows": len(state.get("log", [])),
        "topics_rows": len(state.get("topics", [])),
        "catalog_rows": len(state.get("catalog", [])),
    }


def print_step_counts(memory: am.ClaudeMemory) -> None:
    """Print compact runtime state sizes for step-by-step inspection."""

    state = memory._runtime._state
    print(
        "  rows: "
        f"log={len(state.get('log', []))}, "
        f"topics={len(state.get('topics', []))}, "
        f"catalog={len(state.get('catalog', []))}"
    )


def run_full_recompute(
    memory: am.ClaudeMemory,
    adapter: LotusAdapter,
    query_text: str,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    """Execute full view queries over the final source log state."""

    policy = am.ClaudeMemory.differentiate_policy()
    spec = am.ClaudeMemory.spec()
    log_state = memory._runtime._state["log"]
    full_state: dict[str, Any] = {}
    phase_metrics: list[dict[str, Any]] = []

    for view_name in policy.view_outputs:
        query = bind_materialized_dependencies(
            spec.views[view_name].query,
            spec=spec,
            current_view=view_name,
        )
        view_result, metric = run_measured(
            run_kind="view",
            phase=f"view_{view_name}",
            action=lambda query=query: adapter.execute(
                query,
                {
                    "log": log_state,
                    **full_state,
                },
            ),
            view_name=view_name,
        )
        full_state[view_name] = view_result
        metric[f"{view_name}_rows"] = len(view_result)
        phase_metrics.append(metric)

    retrieval_template = policy.retrieval_queries["default"]
    retrieval_query = bind_user_query(retrieval_template, query_text)
    query_result, metric = run_measured(
        run_kind="view",
        phase="view_query",
        action=lambda: adapter.execute(retrieval_query, full_state),
    )
    metric["query_result_rows"] = len(query_result)
    phase_metrics.append(metric)
    return full_state, query_result, phase_metrics


def run_full_candidates(memory: am.ClaudeMemory, adapter: LotusAdapter) -> Any:
    """Execute the topic-candidate extraction subtree over the final log state."""

    spec = am.ClaudeMemory.spec()
    candidate_query = find_first_op(spec.views["topics"].query, "sem_flat_map")
    if candidate_query is None:
        raise RuntimeError("ClaudeMemory topics view does not contain sem_flat_map")
    return adapter.execute(candidate_query, {"log": memory._runtime._state["log"]})


def bind_user_query(query: QueryExpr, text: str) -> QueryExpr:
    """Bind UserQuery placeholders for full-reference execution."""

    return QueryExpr(
        op=query.op,
        inputs=tuple(bind_user_query(item, text) for item in query.inputs),
        params={key: bind_user_query_value(value, text) for key, value in query.params.items()},
    )


def bind_user_query_value(value: Any, text: str) -> Any:
    """Recursively bind one query parameter value."""

    if isinstance(value, UserQuery):
        return text
    if isinstance(value, Mapping):
        return {key: bind_user_query_value(item, text) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(bind_user_query_value(item, text) for item in value)
    return value


def find_first_op(query: QueryExpr, op: str) -> QueryExpr | None:
    """Return the first query subtree with the requested operator."""

    if query.op == op:
        return query
    for input_query in query.inputs:
        found = find_first_op(input_query, op)
        if found is not None:
            return found
    return None


def bind_materialized_dependencies(
    query: QueryExpr,
    *,
    spec: MemorySpec,
    current_view: str,
) -> QueryExpr:
    """Bind exact upstream public-view subtrees to materialized-view inputs."""

    for name, view in spec.views.items():
        if name != current_view and query == view.query:
            return QueryExpr(op="materialized_view", params={"name": name})

    if not query.inputs:
        return query

    return QueryExpr(
        op=query.op,
        inputs=tuple(
            bind_materialized_dependencies(
                input_query,
                spec=spec,
                current_view=current_view,
            )
            for input_query in query.inputs
        ),
        params=query.params,
    )


def metrics_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a stable metrics DataFrame from metric rows."""

    return pd.DataFrame(rows)


def metrics_summary(
    *,
    step_metrics: list[dict[str, Any]],
    phase_metrics: list[dict[str, Any]],
    memory: am.ClaudeMemory,
    adapter: LotusAdapter,
    compare_full: bool,
    model: str,
) -> pd.DataFrame:
    """Summarize demo-level latency and token usage by run kind."""

    rows = [*step_metrics, *phase_metrics]
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for run_kind in ("differential", "view"):
        selected = frame[frame["run_kind"] == run_kind]
        if selected.empty and run_kind == "view" and not compare_full:
            continue
        summary_rows.append(
            {
                "model": model,
                "run_kind": run_kind,
                "metric_rows": len(selected),
                "latency_sec": round(float(selected["latency_sec"].sum()), 4),
                "physical_prompt_tokens": int(selected["physical_prompt_tokens"].sum()),
                "physical_completion_tokens": int(selected["physical_completion_tokens"].sum()),
                "physical_total_tokens": int(selected["physical_total_tokens"].sum()),
                "virtual_prompt_tokens": int(selected["virtual_prompt_tokens"].sum()),
                "virtual_completion_tokens": int(selected["virtual_completion_tokens"].sum()),
                "virtual_total_tokens": int(selected["virtual_total_tokens"].sum()),
                "cache_hits": int(selected["cache_hits"].sum()),
                "structured_retry_batches": int(selected["structured_retry_batches"].sum()),
                "structured_retry_rows": int(selected["structured_retry_rows"].sum()),
                "structured_failure_artifacts": int(
                    selected["structured_failure_artifacts"].sum()
                ),
                "cache_enabled": lotus_cache_enabled(),
                "lm_backend_retry_configured": adapter.config.lm_num_retries,
                "lm_max_batch_size": adapter.config.lm_max_batch_size,
                "lm_rate_limit": adapter.config.lm_rate_limit,
                "structured_parse_retries": adapter.config.structured_parse_retries,
            }
        )
    return pd.DataFrame(summary_rows)


def lotus_cache_enabled() -> bool:
    """Return whether LOTUS cache is enabled for this process."""

    import lotus

    return bool(lotus.settings.enable_cache)


def main() -> None:
    """Run the ClaudeMemory e2e demo and write inspectable CSV artifacts."""

    args = parse_args()
    require_environment()
    reset_structured_retry_stats()
    output_dir = args.output_dir.resolve()
    reset_output_dir(output_dir)
    input_dir = output_dir / "input"
    differential_dir = output_dir / "differential"
    view_dir = output_dir / "view"
    comparison_dir = output_dir / "comparison"
    metrics_dir = output_dir / "metrics"
    trace_dir = output_dir / "trace" if args.trace else None

    dataset_path, rows = selected_rows(
        start_row=args.start_row,
        row_limit=args.row_limit,
        sample_limit=args.sample_limit,
    )
    print("ClaudeMemory real e2e demo")
    print(f"LOCOMO cache: {dataset_path}")
    print(f"start_row: {args.start_row}")
    print(f"rows: {len(rows)}")
    print(f"model: {args.model}")
    print(f"output_dir: {output_dir}")
    write_csv("locomo_rows", pd.DataFrame(rows), input_dir)

    memory, adapter, step_metrics = run_differential(
        rows,
        model=args.model,
        print_steps=args.print_steps,
        semantic_trace_dir=trace_dir,
    )
    result, query_metric = run_measured(
        run_kind="differential",
        phase="differential_query",
        action=lambda: memory.query(args.query),
    )
    query_metric["query_result_rows"] = len(result)
    phase_metrics = [query_metric]

    log_state = memory._runtime._state["log"]
    topics = memory._runtime._state["topics"]
    catalog = memory._runtime._state["catalog"]

    print_frame("log", log_state)
    print_frame("differential topics", topics)
    print_frame("differential catalog", catalog)
    print_frame("query result", result)

    written = {
        "input/locomo_rows": input_dir / "locomo_rows.csv",
        "differential/log": write_csv("log", log_state, differential_dir),
        "differential/topics": write_csv("topics", topics, differential_dir),
        "differential/catalog": write_csv("catalog", catalog, differential_dir),
        "differential/query_result": write_csv(
            "query_result",
            result,
            differential_dir,
        ),
    }
    if trace_dir is not None:
        written["trace"] = trace_dir

    if args.compare_full:
        full_candidates, candidate_metric = run_measured(
            run_kind="view",
            phase="view_candidates",
            action=lambda: run_full_candidates(memory, adapter),
        )
        candidate_metric["candidate_rows"] = len(full_candidates)
        phase_metrics.append(candidate_metric)
        full_state, full_result, full_metrics = run_full_recompute(
            memory,
            adapter,
            args.query,
        )
        phase_metrics.extend(full_metrics)
        print_frame("view candidates", full_candidates)
        print_frame("differential topics", topics)
        print_frame("view topics", full_state["topics"])
        print_frame("differential catalog", catalog)
        print_frame("view catalog", full_state["catalog"])
        print_frame("view query result", full_result)
        comparison_summary, comparison_matches = compare_public_views(
            ivm_topics=topics,
            full_topics=full_state["topics"],
            ivm_catalog=catalog,
            full_catalog=full_state["catalog"],
        )
        print_comparison(comparison_summary, comparison_matches)
        written["view/candidates"] = write_csv(
            "candidates",
            full_candidates,
            view_dir,
        )
        written["view/topics"] = write_csv("topics", full_state["topics"], view_dir)
        written["view/catalog"] = write_csv("catalog", full_state["catalog"], view_dir)
        written["view/query_result"] = write_csv(
            "query_result",
            full_result,
            view_dir,
        )
        written["comparison/summary"] = write_csv(
            "summary",
            comparison_summary,
            comparison_dir,
        )
        for name, matches in comparison_matches.items():
            written[f"comparison/{name}"] = write_csv(name, matches, comparison_dir)

    steps_frame = metrics_frame(step_metrics)
    phases_frame = metrics_frame(phase_metrics)
    summary_frame = metrics_summary(
        step_metrics=step_metrics,
        phase_metrics=phase_metrics,
        memory=memory,
        adapter=adapter,
        compare_full=args.compare_full,
        model=args.model,
    )
    written["metrics/steps"] = write_csv("steps", steps_frame, metrics_dir)
    written["metrics/phases"] = write_csv("phases", phases_frame, metrics_dir)
    written["metrics/summary"] = write_csv("summary", summary_frame, metrics_dir)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot = memory._runtime.snapshot_state()
    (checkpoint_dir / "state.pkl").write_bytes(pickle.dumps(snapshot))
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": snapshot["schema_version"],
                "policy_fingerprint": memory._runtime.policy.fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written["checkpoint/state"] = checkpoint_dir / "state.pkl"
    written["checkpoint/metadata"] = checkpoint_dir / "metadata.json"
    if trace_dir is not None:
        append_trace_metrics(trace_dir, [*step_metrics, *phase_metrics])
        written["trace/differential/metrics"] = trace_dir / "differential" / "metrics.csv"
        if args.compare_full:
            written["trace/view/metrics"] = trace_dir / "view" / "metrics.csv"

    print("\nwrote CSV artifacts:")
    for name, path_value in written.items():
        print(f"- {name}: {path_value}")


if __name__ == "__main__":
    main()
