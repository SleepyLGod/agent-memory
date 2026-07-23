"""Compare completed memory benchmark runs after contract validation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from sys import path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

from agent_memory.evaluation.comparison import compare_benchmark_runs  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse two or more completed run directories and a fresh output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", dest="run_dirs", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.run_dirs or len(args.run_dirs) < 2:
        parser.error("--run-dir must be supplied at least twice")
    args.run_dirs = tuple(args.run_dirs)
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    """Validate contracts and write comparison artifacts."""

    args = parse_args(argv)
    return compare_benchmark_runs(args.run_dirs, args.output_dir)


if __name__ == "__main__":
    print(main())
