"""Real LOTUS-backed HelloWorld memory smoke over LOCOMO dialogue rows.

Run with a local .env containing DEEPSEEK_API_KEY, or export the key in the
shell before running this script.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
from sys import path
from typing import Any
from urllib.request import urlopen
import warnings

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

import agent_memory as am  # noqa: E402

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
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
            instruction="{message} is a coherent LOCOMO dialogue utterance that contains a concrete personal fact, preference, relationship, event, plan, or other memory-worthy information."
        )
        .sem_map(
            output_cols={
                "memory_summary": "One-sentence memory-oriented summary of the utterance, including the relevant speaker when needed.",
            },
            instruction="Produce a concise memory summary for {speaker}'s utterance: {message}.",
        )
        .select(["memory_summary"])
    )

    def query(self, query: str) -> Any:
        return self.helloworld_tests.sem_topk(query, 2)


def _ensure_locomo_dataset(cache_path: Path = LOCOMO_CACHE_PATH) -> Path:
    """Download the small official LOCOMO sample once and return its path."""

    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(LOCOMO_URL, timeout=60) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def _load_locomo_rows(
    dataset_path: Path,
    *,
    sample_limit: int = LOCOMO_SAMPLE_LIMIT,
    turn_limit: int = LOCOMO_TURN_LIMIT,
) -> list[dict[str, str]]:
    """Load a small LOCOMO dialogue slice as log rows."""

    with dataset_path.open(encoding="utf-8") as file:
        dataset = json.load(file)
    return _flatten_locomo_rows(dataset, sample_limit=sample_limit, turn_limit=turn_limit)


def _flatten_locomo_rows(
    dataset: Any,
    *,
    sample_limit: int,
    turn_limit: int,
) -> list[dict[str, str]]:
    """Flatten LOCOMO conversation turns into the demo log row shape."""

    rows: list[dict[str, str]] = []
    samples = dataset if isinstance(dataset, list) else [dataset]
    for sample in samples[:sample_limit]:
        if not isinstance(sample, Mapping):
            continue
        conversation = sample.get("conversation")
        for session_id, timestamp, turns in _iter_locomo_sessions(conversation):
            for turn in turns:
                row = _locomo_turn_row(turn, session_id=session_id, timestamp=timestamp)
                if row is None:
                    continue
                rows.append(row)
                if len(rows) >= turn_limit:
                    return rows
    return rows


def _iter_locomo_sessions(conversation: Any) -> Iterable[tuple[str, str, Iterable[Any]]]:
    """Yield session id, timestamp, and turns from common LOCOMO JSON shapes."""

    if isinstance(conversation, Mapping):
        for key, value in conversation.items():
            if not isinstance(value, list):
                continue
            timestamp = str(conversation.get(f"{key}_date_time", ""))
            yield str(key), timestamp, value
        return

    if isinstance(conversation, list):
        for index, item in enumerate(conversation):
            if isinstance(item, Mapping) and isinstance(item.get("dialogue"), list):
                session_id = str(item.get("session_id", f"session_{index + 1}"))
                timestamp = str(item.get("session_date_time", item.get("timestamp", "")))
                yield session_id, timestamp, item["dialogue"]
            elif isinstance(item, Mapping) and "text" in item:
                yield "session_1", "", [item]


def _locomo_turn_row(
    turn: Any,
    *,
    session_id: str,
    timestamp: str,
) -> dict[str, str] | None:
    """Convert one LOCOMO dialogue turn into a log row."""

    if not isinstance(turn, Mapping):
        return None

    message = str(turn.get("text") or turn.get("message") or turn.get("content") or "").strip()
    if not message:
        return None

    return {
        "message": message,
        "speaker": str(turn.get("speaker", "")),
        "session_id": session_id,
        "turn_id": str(turn.get("dia_id", turn.get("turn_id", ""))),
        "timestamp": str(turn.get("timestamp", timestamp)),
    }


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
