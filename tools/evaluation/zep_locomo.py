"""Run the formal preliminary LOCOMO benchmark against ZepMemory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from sys import path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

from agent_memory.evaluation.zep.locomo import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_QUESTION_LIMIT,
    DEFAULT_QUESTION_START,
    DEFAULT_ROW_LIMIT,
    DEFAULT_START_ROW,
    ZepLocomoRunConfig,
    run_zep_locomo,
)
from agent_memory.planner import (  # noqa: E402
    DEFAULT_GROUPED_AGG_RULE,
    GROUPED_AGG_RULES,
)

LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"


def default_output_dir() -> Path:
    """Return a fresh local artifact path."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / ".memory-test" / "zep-locomo" / timestamp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the formal Zep LOCOMO CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "benchmark"))
    parser.add_argument(
        "--locomo-path",
        type=Path,
        default=LOCOMO_CACHE_PATH,
        help="Pinned LOCOMO JSON path; downloaded and verified when absent.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--start-row", type=int)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument(
        "--question-start",
        type=int,
        help="One-based original LOCOMO question number.",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
    )
    parser.add_argument("--question-numbers", nargs="+", type=int)
    parser.add_argument(
        "--no-include-adversarial",
        dest="include_adversarial",
        action="store_false",
        default=True,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--grouped-agg-rule",
        choices=GROUPED_AGG_RULES,
        default=DEFAULT_GROUPED_AGG_RULE,
    )
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    args = parser.parse_args(argv)
    if args.question_numbers and (
        args.question_start is not None or args.question_limit is not None
    ):
        parser.error(
            "--question-numbers cannot be combined with --question-start or --question-limit"
        )
    smoke = args.mode == "smoke"
    args.start_row = (
        args.start_row
        if args.start_row is not None
        else (DEFAULT_START_ROW if smoke else 1)
    )
    args.row_limit = (
        args.row_limit
        if args.row_limit is not None
        else (DEFAULT_ROW_LIMIT if smoke else None)
    )
    if args.question_numbers:
        args.question_numbers = tuple(args.question_numbers)
        args.question_start = 1
        args.question_limit = None
    else:
        args.question_numbers = None
        args.question_start = (
            args.question_start
            if args.question_start is not None
            else (DEFAULT_QUESTION_START if smoke else 1)
        )
        args.question_limit = (
            args.question_limit
            if args.question_limit is not None
            else (DEFAULT_QUESTION_LIMIT if smoke else None)
        )
    return args


def main() -> None:
    """Run one isolated Zep LOCOMO benchmark slice."""

    args = parse_args()
    output_dir = run_zep_locomo(
        ZepLocomoRunConfig(
            dataset_path=args.locomo_path,
            output_dir=args.output_dir,
            sample_index=args.sample_index,
            start_row=args.start_row,
            row_limit=args.row_limit,
            question_start=args.question_start,
            question_limit=args.question_limit,
            question_numbers=args.question_numbers,
            include_adversarial=args.include_adversarial,
            model=args.model,
            grouped_agg_rule=args.grouped_agg_rule,
            namespace=args.namespace,
        )
    )
    print(f"run: {output_dir}")


if __name__ == "__main__":
    main()
