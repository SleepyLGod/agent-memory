"""Smoke demo for count_window(...).process_window(...) memory policies."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from sys import path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

import agent_memory as am  # noqa: E402
from agent_memory.adapters.lotus import DEFAULT_LOTUS_MODEL, LotusAdapter  # noqa: E402
from agent_memory.datasets.locomo import ensure_locomo_dataset, load_locomo_rows  # noqa: E402


LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
LOCOMO_SAMPLE_LIMIT = 1
LOCOMO_ROW_LIMIT = 6


class WindowBlockMemory(am.Memory):
    """Deterministic window block policy."""

    log = am.Log(
        {
            "timestamp": "Message timestamp.",
            "speaker": "Message speaker.",
            "message": "Message body.",
        }
    )

    blocks = log.count_window(size=2, slide=1).process_window(
        lambda window: window.array_agg(
            columns=("timestamp", "speaker", "message"),
            output_col="conversation_records",
        )
    )


class WindowSelectedBlockMemory(am.Memory):
    """Count-window policy over an ordinary upstream relation."""

    log = WindowBlockMemory.log

    selected_blocks = log.select(["speaker", "message"]).count_window(
        size=2,
        slide=1,
    ).process_window(
        lambda window: window.array_agg(
            columns=("speaker", "message"),
            output_col="conversation_records",
        )
    )


class OverContextMemory(am.Memory):
    """Deterministic over-window previous-context policy."""

    log = WindowBlockMemory.log

    contextual_log = log.over(rows=(-2, -1)).array_agg(
        columns=("timestamp", "speaker", "message"),
        output_col="previous_records",
    )


class WindowSemanticInWindowMemory(am.Memory):
    """Semantic extraction inside each completed window."""

    log = WindowBlockMemory.log

    window_memories = (
        log
        .count_window(size=2, slide=1)
        .process_window(
            lambda window: window.array_agg(
                columns=("timestamp", "speaker", "message"),
                output_col="conversation_records",
            )
            .sem_flat_map(
                input_cols=["conversation_records"],
                output_cols={"memory_summary": "One durable memory from the window."},
                instruction="Extract concise durable memories from {conversation_records}.",
            )
            .select(["memory_summary"])
        )
    )


class WindowGlobalAfterMemory(am.Memory):
    """Window extraction followed by global semantic consolidation."""

    log = WindowBlockMemory.log

    topics = (
        log
        .count_window(size=2, slide=1)
        .process_window(
            lambda window: window.array_agg(
                columns=("timestamp", "speaker", "message"),
                output_col="conversation_records",
            )
        )
        .sem_flat_map(
            input_cols=["conversation_records"],
            output_cols={
                "name": "Short topic name.",
                "body": "One candidate memory topic from the window.",
            },
            instruction="Extract candidate memory topics from {conversation_records}.",
        )
        .sem_groupby(
            input_cols=["name", "body"],
            instruction="Rows belong together when they describe the same durable memory topic.",
        )
        .sem_agg(
            input_cols=["name", "body"],
            output_cols={
                "name": "Canonical topic name.",
                "body": "Merged memory body.",
            },
            instruction="Merge rows into one canonical memory with {name} and {body}.",
        )
        .select(["name", "body"])
    )


class OverSemanticMemory(am.Memory):
    """Semantic aggregation over each row's previous context frame."""

    log = WindowBlockMemory.log

    contextual_summaries = log.over(rows=(-2, -1)).sem_agg(
        input_cols=["timestamp", "speaker", "message"],
        output_cols={
            "previous_summary": "Summary of the preceding conversation context.",
        },
        instruction=(
            "Summarize the preceding conversation context using {timestamp}, "
            "{speaker}, and {message}."
        ),
    )


def _print_frame(name: str, frame: Any) -> None:
    """Print a compact DataFrame-like object."""

    print(f"\n{name}:")
    print(frame.to_string(index=False))


def _load_rows(row_limit: int) -> list[dict[str, str]]:
    """Load official LOCOMO rows for the smoke demo."""

    dataset_path = ensure_locomo_dataset(LOCOMO_CACHE_PATH)
    rows = load_locomo_rows(
        dataset_path,
        sample_limit=LOCOMO_SAMPLE_LIMIT,
        turn_limit=row_limit,
    )
    if not rows:
        raise SystemExit(f"No LOCOMO rows loaded from {dataset_path}")
    print(f"LOCOMO cache: {dataset_path}")
    print(f"loaded rows: {len(rows)}")
    return rows


def _add_rows(memory: am.Memory, rows: list[dict[str, str]]) -> None:
    """Append demo rows."""

    for row in rows:
        memory.add(row)


def _run_deterministic(output_dir: Path, rows: list[dict[str, str]]) -> None:
    """Run deterministic count-window and over-window smokes."""

    block_memory = WindowBlockMemory()
    _add_rows(block_memory, rows)

    blocks = block_memory._runtime._state["blocks"]
    _print_frame("deterministic blocks", blocks)
    for index, raw_records in enumerate(blocks["conversation_records"]):
        print(f"block[{index}]: {json.loads(raw_records)}")

    selected_memory = WindowSelectedBlockMemory()
    _add_rows(selected_memory, rows)
    selected_blocks = selected_memory._runtime._state["selected_blocks"]
    _print_frame("selected-upstream blocks", selected_blocks)

    over_memory = OverContextMemory()
    _add_rows(over_memory, rows)
    contextual_log = over_memory._runtime._state["contextual_log"]
    _print_frame("over previous context", contextual_log)

    output_dir.mkdir(parents=True, exist_ok=True)
    block_memory._runtime._state["log"].to_csv(output_dir / "log.csv", index=False)
    blocks.to_csv(output_dir / "blocks.csv", index=False)
    selected_blocks.to_csv(output_dir / "selected_blocks.csv", index=False)
    contextual_log.to_csv(output_dir / "contextual_log.csv", index=False)
    print(f"\nwrote deterministic CSVs: {output_dir}")


def _run_semantic(
    rows: list[dict[str, str]],
    *,
    model: str,
    output_dir: Path,
) -> None:
    """Run real LOTUS semantic window policies and persist their views."""

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\nsemantic window smoke skipped: DEEPSEEK_API_KEY is not set")
        return

    adapter = LotusAdapter(model=model)

    semantic_in_window = WindowSemanticInWindowMemory(adapter=adapter)
    _add_rows(semantic_in_window, rows)
    window_memories = semantic_in_window._runtime._state["window_memories"]
    _print_frame(
        "semantic-in-window memories",
        window_memories,
    )

    global_after_window = WindowGlobalAfterMemory(adapter=adapter)
    _add_rows(global_after_window, rows)
    topics = global_after_window._runtime._state["topics"]
    _print_frame(
        "global-after-window topics",
        topics,
    )

    over_semantic = OverSemanticMemory(adapter=adapter)
    _add_rows(over_semantic, rows)
    contextual_summaries = over_semantic._runtime._state["contextual_summaries"]
    _print_frame(
        "over semantic summaries",
        contextual_summaries,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    window_memories.to_csv(output_dir / "semantic_in_window.csv", index=False)
    topics.to_csv(output_dir / "global_after_window.csv", index=False)
    contextual_summaries.to_csv(output_dir / "over_semantic.csv", index=False)
    print(f"\nwrote semantic CSVs: {output_dir}")


def main() -> None:
    """Run the window smoke demo."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "agent-memory-helloworld-window",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=LOCOMO_ROW_LIMIT,
        help="number of LOCOMO rows to load",
    )
    parser.add_argument(
        "--run-semantic",
        action="store_true",
        help="also run real LOTUS semantic window policies",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LOTUS_MODEL,
        help=f"LiteLLM model for semantic mode. Defaults to {DEFAULT_LOTUS_MODEL}.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    rows = _load_rows(args.row_limit)
    _run_deterministic(args.output_dir, rows)
    if args.run_semantic:
        _run_semantic(rows, model=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
