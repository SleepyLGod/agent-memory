"""Real LOTUS-backed HelloWorld memory smoke over LOCOMO dialogue rows.

Run with a local .env containing DEEPSEEK_API_KEY, or export the key in the
shell before running this script.
"""

from __future__ import annotations

import os
from pathlib import Path
from sys import path
from typing import Any
import warnings

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

import agent_memory as am  # noqa: E402
from agent_memory.datasets.locomo import (  # noqa: E402
    DEFAULT_LOCOMO_URL,
    ensure_locomo_dataset,
    load_locomo_rows,
)

LOCOMO_URL = DEFAULT_LOCOMO_URL
LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
LOCOMO_SAMPLE_LIMIT = 1
LOCOMO_TURN_LIMIT = 8


class HelloWorldMemory(am.Memory):
    """Tiny policy for proving add -> view maintenance -> query execution."""

    log = am.Log(
        {
            "message": "Raw LOCOMO dialogue utterance.",
            "speaker": "Speaker name or role.",
            "session_id": "LOCOMO session identifier.",
            "turn_id": "Turn/dialogue identifier within the session.",
            "timestamp": "Session or turn timestamp when available.",
        }
    )

    helloworld_tests = (
        log
        .sem_filter(
            instruction="{message} is about LGBTQ."
        )
        .sem_map(
            output_cols={
                "memory_summary": "One-sentence memory-oriented summary of the utterance, including the relevant speaker when needed.",
            },
            instruction="Produce a concise memory summary for {speaker}'s utterance: {message}.",
        )
        .select(["memory_summary"])
    )

    retrieval_query = helloworld_tests.sem_topk(am.UserQuery(), 2)


def _ensure_locomo_dataset(cache_path: Path = LOCOMO_CACHE_PATH) -> Path:
    """Download the small official LOCOMO sample once and return its path."""

    return ensure_locomo_dataset(cache_path, url=LOCOMO_URL)


def _load_locomo_rows(
    dataset_path: Path,
    *,
    sample_limit: int = LOCOMO_SAMPLE_LIMIT,
    turn_limit: int = LOCOMO_TURN_LIMIT,
) -> list[dict[str, str]]:
    """Load a small LOCOMO dialogue slice as log rows."""

    return load_locomo_rows(
        dataset_path,
        sample_limit=sample_limit,
        turn_limit=turn_limit,
    )


def _print_frame(name: str, frame: Any) -> None:
    """Print a compact DataFrame-like object."""

    print(f"\n{name}:")
    print(frame.to_string(index=False))


def main() -> None:
    """Run the HelloWorld e2e smoke."""

    load_dotenv(PROJECT_ROOT / ".env")
    warnings.filterwarnings(
        "ignore",
        message="Error calculating completion cost - cost metrics will be inaccurate.*",
        category=UserWarning,
    )
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is required for the default DeepSeek LOTUS smoke. "
            "Set it in .env or export it in the shell."
        )

    dataset_path = _ensure_locomo_dataset()
    rows = _load_locomo_rows(dataset_path)
    if not rows:
        raise SystemExit(f"No LOCOMO rows loaded from {dataset_path}")

    print(f"LOCOMO cache: {dataset_path}")
    print(f"loaded rows: {len(rows)}")

    memory = HelloWorldMemory()
    for row in rows:
        print(f"add: {row['speaker']}: {row['message'][:100]}")
        memory.add(row)

    _print_frame("log state", memory._runtime._state["log"])
    _print_frame("helloworld_tests view", memory._runtime._state["helloworld_tests"])

    result = memory.query(
        "Which memories are most relevant to a person's preferences, plans, or relationships?"
    )
    _print_frame("query result", result)

    output_dir = Path("/private/tmp/agent-memory-helloworld")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_output_path = output_dir / "log.csv"
    view_output_path = output_dir / "helloworld_tests.csv"
    memory._runtime._state["log"].to_csv(log_output_path, index=False)
    memory._runtime._state["helloworld_tests"].to_csv(view_output_path, index=False)
    print(f"\nwrote log: {log_output_path}")
    print(f"wrote view: {view_output_path}")


if __name__ == "__main__":
    main()
