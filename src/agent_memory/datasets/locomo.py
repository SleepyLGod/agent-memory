"""LOCOMO row loading helpers for examples and audit tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

DEFAULT_LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def ensure_locomo_dataset(cache_path: Path, *, url: str = DEFAULT_LOCOMO_URL) -> Path:
    """Download the small official LOCOMO sample once and return its path."""

    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def load_locomo_rows(
    dataset_path: Path,
    *,
    sample_limit: int,
    turn_limit: int,
) -> list[dict[str, str]]:
    """Load a small LOCOMO dialogue slice as log rows."""

    with dataset_path.open(encoding="utf-8") as file:
        dataset = json.load(file)
    return flatten_locomo_rows(
        dataset,
        sample_limit=sample_limit,
        turn_limit=turn_limit,
    )


def flatten_locomo_rows(
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

    message = str(
        turn.get("text") or turn.get("message") or turn.get("content") or ""
    ).strip()
    if not message:
        return None

    return {
        "message": message,
        "speaker": str(turn.get("speaker", "")),
        "session_id": session_id,
        "turn_id": str(turn.get("dia_id", turn.get("turn_id", ""))),
        "timestamp": str(turn.get("timestamp", timestamp)),
    }
