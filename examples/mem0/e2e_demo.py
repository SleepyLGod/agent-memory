"""Real LOTUS-backed Mem0Memory e2e demo over LOCOMO rows.

Run with a local .env containing DEEPSEEK_API_KEY. Writes CSV artifacts under
.memory-test/mem0-e2e/latest/ so the maintained memory tables can be inspected.
"""

from __future__ import annotations

import os
import shutil
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from sys import path
from typing import Any

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
from agent_memory.adapters.lotus import DEFAULT_LOTUS_MODEL, LotusAdapter  # noqa: E402
from agent_memory.logical import MemorySpec, QueryExpr  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".memory-test" / "mem0-e2e" / "latest"
LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
DEFAULT_SAMPLE_LIMIT = 1
DEFAULT_START_ROW = 26
DEFAULT_ROW_LIMIT = 7
DEFAULT_QUERY = "Which memories are most useful for future collaboration style?"


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
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
            val = row_dict.get(col, '')
            if val:
                print(f"       {col}: {val!r}")


def require_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    warnings.filterwarnings(
        "ignore",
        message="Error calculating completion cost.*",
        category=UserWarning,
    )
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is required. Set it in .env or export it in the shell."
        )


def mem0_log_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one LOCOMO row into the Mem0Memory log schema."""
    from agent_memory.memories.mem0 import _derive_observation_date, _now_date

    timestamp = row.get("timestamp", "")
    try:
        observation_date = _derive_observation_date(timestamp)
    except (ValueError, TypeError):
        observation_date = _now_date()

    return {
        "content": row["message"],
        "role": row.get("speaker", ""),
        "timestamp": timestamp,
        "session_id": row.get("session_id", ""),
        "metadata": {
            "source": "locomo",
            "speaker": row.get("speaker", ""),
            "turn_id": row.get("turn_id", ""),
        },
        "observation_date": observation_date,
        "current_date": _now_date(),
    }


def main() -> None:
    require_environment()
    output_dir = DEFAULT_OUTPUT_DIR.resolve()
    reset_output_dir(output_dir)

    dataset_path = ensure_locomo_dataset(LOCOMO_CACHE_PATH, url=DEFAULT_LOCOMO_URL)
    locomo_rows = load_locomo_rows(
        dataset_path,
        sample_limit=DEFAULT_SAMPLE_LIMIT,
        turn_limit=DEFAULT_START_ROW + DEFAULT_ROW_LIMIT - 1,
    )
    selected = locomo_rows[DEFAULT_START_ROW - 1 : DEFAULT_START_ROW - 1 + DEFAULT_ROW_LIMIT]
    rows = [mem0_log_row(row) for row in selected]

    print("Mem0Memory e2e demo (count_window extraction)")
    print(f"LOCOMO cache: {dataset_path}")
    print(f"start_row: {DEFAULT_START_ROW}, rows: {len(rows)}")
    print(f"model: {DEFAULT_LOTUS_MODEL}")
    print(f"output_dir: {output_dir}")

    write_csv("locomo_rows", pd.DataFrame(rows), output_dir / "input")

    # Inspect the differentiated policy
    policy = am.Mem0Memory.differentiate_policy()
    print(f"\nPolicy window plans: {list(policy.window_process_plans.keys())}")
    if "facts" in policy.window_process_plans:
        plan = policy.window_process_plans["facts"]
        print(f"  window size/slide: {plan.window_query.params}")
        print(f"  process_query.op: {plan.process_query.op}")
    print(f"View execution order: {policy.view_execution_order}")

    # Run differential maintenance
    memory = am.Mem0Memory(
        adapter=LotusAdapter(model="openrouter/deepseek/deepseek-chat")
    )
    start = time.perf_counter()
    for i, row in enumerate(rows, 1):
        role = row.get("role", "")
        msg = row.get("content", "")
        print(f"\nadd[{i}]: [{role}] {msg[:100]}")
        memory.add(row)
    latency = time.perf_counter() - start

    state = memory._runtime._state
    print(f"\nlog rows: {len(state.get('log', []))}")
    print(f"facts rows: {len(state.get('facts', []))}")
    print(f"private state keys: {sorted(k for k in state if k.startswith('_'))}")
    print(f"Latency: {latency:.2f}s")

    # Show extracted facts
    facts = state.get('facts', [])
    if len(facts):
        print_frame("facts", facts)

    # Run retrieval query
    result = memory.query(DEFAULT_QUERY)
    print_frame("query result", result)

    # Write CSV artifacts
    written = {}
    log_state = state.get("log", [])
    if len(log_state):
        written["log"] = write_csv("log", log_state, output_dir)
    if len(facts):
        written["facts"] = write_csv("facts", facts, output_dir)
    if len(result):
        written["query_result"] = write_csv("query_result", result, output_dir)
    print("\nwrote CSV artifacts:")
    for name, p in written.items():
        print(f"  {name}: {p}")

    if len(facts) == 0:
        raise SystemExit("\nFAIL: No facts extracted from the log.")

    print("\n=== PASS ===")


if __name__ == "__main__":
    main()
