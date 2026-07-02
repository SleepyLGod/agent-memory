"""Run LOCOMO v1 evaluation slices against the built-in ClaudeMemory policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from sys import path
import warnings

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

from agent_memory.adapters.lotus import DEFAULT_LOTUS_MODEL  # noqa: E402
from agent_memory.evaluation.claude_memory.locomo import (  # noqa: E402
    ClaudeMemoryLocomoRunConfig,
    run_claude_memory_locomo,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".memory-test" / "locomo-benchmark" / "latest"
LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"


def parse_args() -> argparse.Namespace:
    """Parse LOCOMO evaluation CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="0-based LOCOMO sample index.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=12,
        help="Number of normalized LOCOMO events to ingest.",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=5,
        help="Maximum eligible questions to evaluate.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LOTUS_MODEL,
        help=f"LiteLLM model passed to LotusAdapter. Defaults to {DEFAULT_LOTUS_MODEL}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--existing-output-dir",
        type=Path,
        default=None,
        help=(
            "Restore memory runtime state from an existing benchmark output "
            "and run questions only."
        ),
    )
    parser.add_argument(
        "--restore-csv-state",
        action="store_true",
        help=(
            "Deprecated. Display CSVs are spreadsheet-escaped; use "
            "--restore-artifact-csv-state instead."
        ),
    )
    parser.add_argument(
        "--restore-artifact-csv-state",
        action="store_true",
        help=(
            "Restore runtime state from raw state/*.jsonl artifacts instead of "
            "checkpoint/state.pkl. Requires --existing-output-dir and "
            "--trust-existing-output-dir."
        ),
    )
    parser.add_argument(
        "--trust-existing-output-dir",
        action="store_true",
        help=(
            "Allow resume/query-only modes to read local restored state. "
            "Use only for trusted local benchmark outputs."
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write semantic trace artifacts under output_dir/trace.",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Generate answers from retrieved memories and compute answer metrics.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output_dir/checkpoint instead of starting from scratch.",
    )
    return parser.parse_args()


def require_environment() -> None:
    """Load local env and fail early when LOTUS cannot call DeepSeek."""

    load_dotenv(PROJECT_ROOT / ".env")
    warnings.filterwarnings(
        "ignore",
        message="Error calculating completion cost - cost metrics will be inaccurate.*",
        category=UserWarning,
    )
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is required for LOCOMO benchmark runs. "
            "Set it in .env or export it in the shell."
        )


def main() -> None:
    """Parse CLI arguments and run the ClaudeMemory LOCOMO evaluation."""

    args = parse_args()
    require_environment()
    run_claude_memory_locomo(
        ClaudeMemoryLocomoRunConfig(
            sample_index=args.sample_index,
            row_limit=args.row_limit,
            question_limit=args.question_limit,
            model=args.model,
            output_dir=args.output_dir,
            locomo_cache_path=LOCOMO_CACHE_PATH,
            existing_output_dir=args.existing_output_dir,
            trust_existing_output_dir=args.trust_existing_output_dir,
            trace=args.trace,
            answer=args.answer,
            resume=args.resume,
            restore_csv_state=args.restore_csv_state,
            restore_artifact_csv_state=args.restore_artifact_csv_state,
        )
    )


if __name__ == "__main__":
    main()
