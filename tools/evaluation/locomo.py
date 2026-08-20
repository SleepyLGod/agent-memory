"""Prepare or run the pinned LOCOMO benchmark with a built-in memory policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_memory.evaluation.bundle import read_bundle, write_bundle  # noqa: E402
from agent_memory.evaluation.locomo import (  # noqa: E402
    LOCOMO_COMMIT,
    ensure_locomo_dataset,
    locomo_bundle,
)
from agent_memory.evaluation.locomo_contracts import locomo_task_contract  # noqa: E402
from agent_memory.evaluation.run import (  # noqa: E402
    AGENT_MEMORY_SYSTEMS,
    DEFAULT_PROVIDER_MODEL_ID,
    SEMANTIC_PAIR_PROFILES,
    run_agent_memory_bundle,
)
from agent_memory.planner import GROUPED_AGG_RULES  # noqa: E402

DEFAULT_DATASET = (
    PROJECT_ROOT
    / ".memory-test"
    / "datasets"
    / "locomo"
    / LOCOMO_COMMIT
    / "locomo10.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical bundle preparation and execution commands."""

    raw_args = tuple(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--bundle-dir", type=Path, required=True)
    prepare.add_argument("--sample-index", type=int, default=0)
    prepare.add_argument("--start-row", type=int, default=1)
    prepare.add_argument("--row-limit", type=int)
    prepare.add_argument("--question-numbers", nargs="+", type=int)
    prepare.add_argument("--no-include-adversarial", action="store_true")
    prepare.add_argument("--smoke", action="store_true")

    run = commands.add_parser("run", allow_abbrev=False)
    run.add_argument("--bundle-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--system", choices=AGENT_MEMORY_SYSTEMS, required=True)
    run.add_argument("--memory-model", default=DEFAULT_PROVIDER_MODEL_ID)
    run.add_argument("--answer-model", default=DEFAULT_PROVIDER_MODEL_ID)
    run.add_argument("--judge-model", default=DEFAULT_PROVIDER_MODEL_ID)
    run.add_argument("--namespace")
    run.add_argument("--condition-id")
    run.add_argument("--maintenance-checkpoint-output-dir", type=Path)
    run.add_argument("--maintenance-only", action="store_true")
    run.add_argument(
        "--grouped-agg-rule",
        choices=GROUPED_AGG_RULES,
        default="rule-all-group",
    )
    run.add_argument(
        "--sem-topk-method",
        choices=("pairwise-naive", "pairwise-quick", "pairwise-heap", "listwise"),
        default=None,
    )
    run.add_argument("--sem-groupby-pair-batch-size", type=int)
    run.add_argument("--sem-groupby-pair-batch-retries", type=int, default=0)
    run.add_argument(
        "--semantic-pair-profile",
        choices=SEMANTIC_PAIR_PROFILES,
        default="oracle-only",
    )
    run.add_argument("--semantic-pair-top-k", type=int)
    run.add_argument("--semantic-pair-min-similarity", type=float)
    run.add_argument("--semantic-pair-profile-config", type=Path)
    run.add_argument(
        "--lotus-cache-mode",
        choices=("disabled", "memory"),
        default="disabled",
    )
    run.add_argument("--embedding-device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument(
        "--semantic-trace-snapshot-mode",
        choices=("compact", "full"),
        default="compact",
    )
    args = parser.parse_args(raw_args)
    if args.command == "run":
        if args.semantic_pair_profile_config is not None and any(
            argument == option or argument.startswith(f"{option}=")
            for option in (
                "--semantic-pair-profile",
                "--semantic-pair-top-k",
                "--semantic-pair-min-similarity",
            )
            for argument in raw_args
        ):
            parser.error(
                "--semantic-pair-profile-config is mutually exclusive with "
                "global semantic-pair profile options"
            )
        invalid = [
            option
            for option in (
                *(
                    ("--sem-topk-method",)
                    if args.system not in {"claude-memory", "mem0-enhanced"}
                    else ()
                ),
                *(
                    ("--grouped-agg-rule",)
                    if args.system in {"mem0-memory", "mem0-enhanced"}
                    else ()
                ),
            )
            if any(
                argument == option or argument.startswith(f"{option}=")
                for argument in raw_args
            )
        ]
        if invalid:
            parser.error(
                f"{', '.join(invalid)} is not valid with --system {args.system}"
            )
        if args.sem_topk_method is None:
            args.sem_topk_method = (
                "pairwise-quick"
                if args.system == "mem0-enhanced"
                else "pairwise-naive"
            )
        if (
            args.maintenance_only
            and args.maintenance_checkpoint_output_dir is not None
        ):
            parser.error(
                "--maintenance-only cannot be combined with "
                "--maintenance-checkpoint-output-dir"
            )
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    """Prepare a LOCOMO bundle or execute one built-in memory system."""

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    if args.command == "prepare":
        dataset_path = ensure_locomo_dataset(args.dataset_path)
        bundle = locomo_bundle(
            dataset_path,
            sample_index=args.sample_index,
            start_row=26 if args.smoke else args.start_row,
            row_limit=3 if args.smoke else args.row_limit,
            question_numbers=(4,) if args.smoke else args.question_numbers,
            include_adversarial=not args.no_include_adversarial,
            run_mode="integration-smoke" if args.smoke else None,
        )
        write_bundle(bundle, args.bundle_dir)
        return args.bundle_dir

    bundle = read_bundle(args.bundle_dir)
    if bundle.benchmark_id != "locomo":
        raise ValueError("bundle is not the pinned LOCOMO dataset")
    return run_agent_memory_bundle(
        bundle=bundle,
        contracts={
            "locomo": locomo_task_contract(judge_model_id=args.judge_model)
        },
        system_id=args.system,
        output_dir=args.output_dir,
        memory_provider_model_id=args.memory_model,
        answer_model_id=args.answer_model,
        judge_model_id=args.judge_model,
        base_namespace=args.namespace,
        grouped_agg_rule=args.grouped_agg_rule,
        sem_topk_method=args.sem_topk_method,
        sem_groupby_pair_batch_size=args.sem_groupby_pair_batch_size,
        sem_groupby_pair_batch_retries=args.sem_groupby_pair_batch_retries,
        semantic_pair_profile=args.semantic_pair_profile,
        semantic_pair_top_k=args.semantic_pair_top_k,
        semantic_pair_min_similarity=args.semantic_pair_min_similarity,
        semantic_pair_profile_config=args.semantic_pair_profile_config,
        lotus_cache_mode=args.lotus_cache_mode,
        embedding_device=args.embedding_device,
        semantic_trace_snapshot_mode=args.semantic_trace_snapshot_mode,
        memory_thinking_enabled=False,
        condition_id=args.condition_id or "",
        maintenance_only=args.maintenance_only,
        maintenance_checkpoint_output_dir=args.maintenance_checkpoint_output_dir,
    )


if __name__ == "__main__":
    print(main())
