"""Prepare or run the pinned MemoryAgentBench canonical benchmark bundle."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from sys import path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

from agent_memory.evaluation.bundle import read_bundle, write_bundle  # noqa: E402
from agent_memory.evaluation.memory_agent_bench import (  # noqa: E402
    MEMORY_AGENT_BENCH_REVISION,
    SMOKE_SOURCES,
    chunk_text_into_sentences,
    download_memory_agent_bench,
    download_movie_entity_mapping,
    load_memory_agent_bench,
    load_movie_entity_mapping,
    memory_agent_bench_task_contracts,
)
from agent_memory.evaluation.run import (  # noqa: E402
    AGENT_MEMORY_SYSTEMS,
    run_agent_memory_bundle,
)

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / ".memory-test"
    / "datasets"
    / "memory-agent-bench"
    / MEMORY_AGENT_BENCH_REVISION
)
DEFAULT_NLTK_DATA_DIR = PROJECT_ROOT / ".memory-test" / "datasets" / "nltk"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical bundle preparation and agent-memory execution commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    prepare.add_argument(
        "--nltk-data-dir", type=Path, default=DEFAULT_NLTK_DATA_DIR
    )
    prepare.add_argument("--bundle-dir", type=Path, required=True)
    selection = prepare.add_mutually_exclusive_group()
    selection.add_argument("--smoke", action="store_true")
    selection.add_argument("--sources", nargs="+")
    prepare.add_argument("--max-cases-per-source", type=int)
    prepare.add_argument("--max-questions-per-case", type=int)

    run = commands.add_parser("run")
    run.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    run.add_argument("--bundle-dir", type=Path, required=True)
    run.add_argument("--system", choices=AGENT_MEMORY_SYSTEMS, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--namespace")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    """Prepare a canonical bundle or run one built-in policy against it."""

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    if args.command == "prepare":
        download_memory_agent_bench(args.dataset_dir)
        download_movie_entity_mapping(args.dataset_dir)
        sources = SMOKE_SOURCES if args.smoke else args.sources
        bundle = load_memory_agent_bench(
            args.dataset_dir,
            sources=sources,
            max_cases_per_source=(
                1 if args.smoke else args.max_cases_per_source
            ),
            max_questions_per_case=(
                1 if args.smoke else args.max_questions_per_case
            ),
            chunker=partial(
                chunk_text_into_sentences,
                nltk_data_dir=args.nltk_data_dir,
            ),
        )
        write_bundle(bundle, args.bundle_dir)
        return args.bundle_dir

    bundle = read_bundle(args.bundle_dir)
    if bundle.benchmark_id != "memory-agent-bench":
        raise ValueError("bundle is not the pinned MemoryAgentBench dataset")
    mapping_path = download_movie_entity_mapping(args.dataset_dir)
    contracts = memory_agent_bench_task_contracts(
        movie_entity_mapping=load_movie_entity_mapping(mapping_path)
    )
    return run_agent_memory_bundle(
        bundle=bundle,
        contracts=contracts,
        system_id=args.system,
        output_dir=args.output_dir,
        memory_provider_model_id=args.model,
        answer_model_id=args.model,
        judge_model_id=args.model,
        base_namespace=args.namespace,
    )


if __name__ == "__main__":
    print(main())
