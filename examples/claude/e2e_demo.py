"""Real LOTUS-backed ClaudeMemory e2e demo over LOCOMO rows.

Run with a local .env containing DEEPSEEK_API_KEY, or export the key in the
shell before running this script. The script writes local CSV artifacts under
.memory-test/ so the maintained memory tables can be inspected manually.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from sys import path
from typing import Any
import warnings

from dotenv import load_dotenv
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

import agent_memory as am  # noqa: E402
from agent_memory.datasets.locomo import (  # noqa: E402
    DEFAULT_LOCOMO_URL,
    ensure_locomo_dataset,
    load_locomo_rows,
)
from agent_memory.logical import MemorySpec, QueryExpr  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".memory-test" / "claude-e2e" / "latest"
LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
DEFAULT_SAMPLE_LIMIT = 1
DEFAULT_START_ROW = 26
DEFAULT_ROW_LIMIT = 7
DEFAULT_QUERY = "Which memories are most useful for future collaboration style?"


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
        "--compare-full",
        action="store_true",
        help="Also execute full recompute queries and write comparison CSVs.",
    )
    parser.add_argument(
        "--print-steps",
        action="store_true",
        help="Print view row counts after each add().",
    )
    return parser.parse_args()


def reset_output_dir(output_dir: Path) -> None:
    """Create the output directory and remove stale artifacts from prior runs."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, frame: Any, output_dir: Path) -> Path:
    """Write a DataFrame-like object to a named CSV and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def print_frame(name: str, frame: Any) -> None:
    """Print a compact DataFrame-like object for terminal inspection."""

    print(f"\n{name}:")
    print(frame.to_string(index=False))


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


def run_differential(rows: list[dict[str, Any]], *, print_steps: bool) -> am.ClaudeMemory:
    """Maintain ClaudeMemory by appending rows through runtime Q' execution."""

    memory = am.ClaudeMemory()
    for index, row in enumerate(rows, start=1):
        print(f"add[{index}]: {row['role']}: {row['message'][:100]}")
        memory.add(row)
        if print_steps:
            print_step_counts(memory)
    return memory


def print_step_counts(memory: am.ClaudeMemory) -> None:
    """Print compact runtime state sizes for step-by-step inspection."""

    state = memory._runtime._state
    print(
        "  rows: "
        f"log={len(state.get('log', []))}, "
        f"topics={len(state.get('topics', []))}, "
        f"catalog={len(state.get('catalog', []))}"
    )


def run_full_recompute(memory: am.ClaudeMemory, query_text: str) -> tuple[dict[str, Any], Any]:
    """Execute full view queries over the final source log state."""

    policy = am.ClaudeMemory.differentiate_policy()
    spec = am.ClaudeMemory.spec()
    adapter = memory._runtime.adapter
    log_state = memory._runtime._state["log"]
    full_state: dict[str, Any] = {}

    for view_name in policy.view_execution_order:
        query = bind_materialized_dependencies(
            spec.views[view_name].query,
            spec=spec,
            current_view=view_name,
        )
        full_state[view_name] = adapter.execute(
            query,
            {
                "log": log_state,
                **full_state,
            },
        )

    retrieval_template = policy.retrieval_queries["default"]
    retrieval_query = memory._runtime._bind_user_query(retrieval_template, query_text)
    query_result = adapter.execute(retrieval_query, full_state)
    return full_state, query_result


def run_full_candidates(memory: am.ClaudeMemory) -> Any:
    """Execute the topic-candidate extraction subtree over the final log state."""

    spec = am.ClaudeMemory.spec()
    adapter = memory._runtime.adapter
    candidate_query = find_first_op(spec.views["topics"].query, "sem_flat_map")
    if candidate_query is None:
        raise RuntimeError("ClaudeMemory topics view does not contain sem_flat_map")
    return adapter.execute(candidate_query, {"log": memory._runtime._state["log"]})


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


def main() -> None:
    """Run the ClaudeMemory e2e demo and write inspectable CSV artifacts."""

    args = parse_args()
    require_environment()
    output_dir = args.output_dir.resolve()
    reset_output_dir(output_dir)
    input_dir = output_dir / "input"
    ivm_dir = output_dir / "ivm"
    full_dir = output_dir / "full"

    dataset_path, rows = selected_rows(
        start_row=args.start_row,
        row_limit=args.row_limit,
        sample_limit=args.sample_limit,
    )
    print("ClaudeMemory real e2e demo")
    print(f"LOCOMO cache: {dataset_path}")
    print(f"start_row: {args.start_row}")
    print(f"rows: {len(rows)}")
    print(f"output_dir: {output_dir}")
    write_csv("locomo_rows", pd.DataFrame(rows), input_dir)

    memory = run_differential(rows, print_steps=args.print_steps)
    result = memory.query(args.query)

    log_state = memory._runtime._state["log"]
    topics = memory._runtime._state["topics"]
    catalog = memory._runtime._state["catalog"]

    print_frame("log", log_state)
    print_frame("topics", topics)
    print_frame("catalog", catalog)
    print_frame("query result", result)

    written = {
        "input/locomo_rows": input_dir / "locomo_rows.csv",
        "ivm/log": write_csv("log", log_state, ivm_dir),
        "ivm/topics": write_csv("topics", topics, ivm_dir),
        "ivm/catalog": write_csv("catalog", catalog, ivm_dir),
        "ivm/query_result": write_csv("query_result", result, ivm_dir),
    }

    if args.compare_full:
        full_candidates = run_full_candidates(memory)
        full_state, full_result = run_full_recompute(memory, args.query)
        print_frame("full candidates", full_candidates)
        print_frame("full topics", full_state["topics"])
        print_frame("full catalog", full_state["catalog"])
        print_frame("full query result", full_result)
        written["full/candidates"] = write_csv(
            "candidates",
            full_candidates,
            full_dir,
        )
        written["full/topics"] = write_csv("topics", full_state["topics"], full_dir)
        written["full/catalog"] = write_csv("catalog", full_state["catalog"], full_dir)
        written["full/query_result"] = write_csv(
            "query_result",
            full_result,
            full_dir,
        )

    print("\nwrote CSV artifacts:")
    for name, path_value in written.items():
        print(f"- {name}: {path_value}")


if __name__ == "__main__":
    main()
