"""Prepare or run the pinned LongMemEval v1 canonical benchmark bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_memory.evaluation.artifacts import BenchmarkArtifactStore  # noqa: E402
from agent_memory.evaluation.bundle import read_bundle, write_bundle  # noqa: E402
from agent_memory.evaluation.longmemeval import (  # noqa: E402
    LONGMEMEVAL_CLAUDE_PILOT_30_IDS,
    LONGMEMEVAL_CLEANED_REVISION,
    LONGMEMEVAL_SMOKE_CASE_ID,
    download_longmemeval,
    load_longmemeval,
    longmemeval_smoke_bundle,
    longmemeval_task_contract,
    write_official_hypotheses,
)
from agent_memory.evaluation.run import (  # noqa: E402
    AGENT_MEMORY_SYSTEMS,
    SEMANTIC_PAIR_PROFILES,
    run_agent_memory_bundle,
)
from agent_memory.planner import GROUPED_AGG_RULES  # noqa: E402

DEFAULT_MEMORY_MODEL_ID = "deepseek-v4-flash"
DEFAULT_PROVIDER_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_DATASET = (
    PROJECT_ROOT
    / ".memory-test"
    / "datasets"
    / "longmemeval"
    / LONGMEMEVAL_CLEANED_REVISION
    / "longmemeval_s_cleaned.json"
)

_PILOT_CONDITIONS = {
    ("rule-join-map", "pairwise-quick"): "JM-Q",
    ("rule-join-map", "listwise"): "JM-L",
    ("rule-re-group", "pairwise-quick"): "RG-Q",
    ("rule-re-group", "listwise"): "RG-L",
}


def _condition_id(args: argparse.Namespace) -> str:
    if args.condition_id:
        return str(args.condition_id)
    if args.maintenance_only and args.system in {"mem0-memory", "mem0-enhanced"}:
        return "AM-Mem0-Maintenance"
    if args.system == "mem0-memory":
        return "AM-Mem0-Base"
    if args.system == "mem0-enhanced":
        return "AM-Mem0-Enhanced"
    if args.system == "zep-memory":
        suffix = "-maintenance" if args.maintenance_only else ""
        return f"zep-memory-{args.grouped_agg_rule}{suffix}"
    if args.maintenance_only:
        return {
            "rule-join-map": "JM-M",
            "rule-re-group": "RG-M",
        }.get(
            args.grouped_agg_rule,
            f"claude-memory-{args.grouped_agg_rule}-maintenance",
        )
    return _PILOT_CONDITIONS.get(
        (args.grouped_agg_rule, args.sem_topk_method),
        f"claude-memory-{args.grouped_agg_rule}-{args.sem_topk_method}",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical bundle preparation and agent-memory execution commands."""

    raw_args = tuple(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--bundle-dir", type=Path, required=True)
    selection = prepare.add_mutually_exclusive_group()
    selection.add_argument("--question-ids", nargs="+")
    selection.add_argument(
        "--pilot-30",
        action="store_true",
        help="Select the fixed 30-case Claude pilot without reading labels.",
    )
    selection.add_argument(
        "--smoke",
        action="store_true",
        help="Select the fixed evidence-complete 8-event integration smoke.",
    )

    run = commands.add_parser("run", allow_abbrev=False)
    run.add_argument("--bundle-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--system",
        choices=AGENT_MEMORY_SYSTEMS,
        default="claude-memory",
    )
    run.add_argument("--memory-model-id", default=DEFAULT_MEMORY_MODEL_ID)
    run.add_argument("--memory-model", default=DEFAULT_PROVIDER_MODEL)
    run.add_argument("--answer-model", default=DEFAULT_PROVIDER_MODEL)
    run.add_argument("--judge-model", default=DEFAULT_PROVIDER_MODEL)
    run.add_argument("--namespace")
    run.add_argument("--condition-id")
    run.add_argument("--max-new-cases", type=int)
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
    run.add_argument(
        "--memory-thinking",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    run.add_argument("--maintenance-checkpoint-output-dir", type=Path)
    run.add_argument("--maintenance-only", action="store_true")
    args = parser.parse_args(raw_args)
    if getattr(args, "question_ids", None) is not None:
        args.question_ids = tuple(args.question_ids)
    if args.command == "run":
        invalid_options = [
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
        if invalid_options:
            if args.system == "zep-memory":
                parser.error(
                    f"{', '.join(invalid_options)} is only valid with "
                    "--system claude-memory or --system mem0-enhanced"
                )
            parser.error(
                f"{', '.join(invalid_options)} is not valid with "
                f"--system {args.system}"
            )
        if args.sem_topk_method is None:
            args.sem_topk_method = (
                "pairwise-quick"
                if args.system == "mem0-enhanced"
                else "pairwise-naive"
            )
    if (
        args.command == "run"
        and args.system in {"mem0-memory", "mem0-enhanced"}
        and args.memory_thinking != "disabled"
    ):
        parser.error(f"{args.system} requires --memory-thinking disabled")
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    """Prepare a canonical bundle or run one built-in policy against it."""

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    if args.command == "prepare":
        path = download_longmemeval(args.dataset_path)
        if args.smoke:
            bundle = longmemeval_smoke_bundle(
                load_longmemeval(path, question_ids=[LONGMEMEVAL_SMOKE_CASE_ID])
            )
        else:
            question_ids = (
                LONGMEMEVAL_CLAUDE_PILOT_30_IDS
                if args.pilot_30
                else args.question_ids
            )
            bundle = load_longmemeval(path, question_ids=question_ids)
        write_bundle(bundle, args.bundle_dir)
        return args.bundle_dir

    bundle = read_bundle(args.bundle_dir)
    if bundle.benchmark_id != "longmemeval-v1-cleaned-s":
        raise ValueError("bundle is not the pinned LongMemEval v1 dataset")
    contract = longmemeval_task_contract(judge_model_id=args.judge_model)
    output = run_agent_memory_bundle(
        bundle=bundle,
        contracts={"longmemeval-v1": contract},
        system_id=args.system,
        output_dir=args.output_dir,
        memory_model_id=args.memory_model_id,
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
        memory_thinking_enabled=args.memory_thinking == "enabled",
        condition_id=_condition_id(args),
        maintenance_only=args.maintenance_only,
        maintenance_checkpoint_output_dir=args.maintenance_checkpoint_output_dir,
        max_new_cases=args.max_new_cases,
    )
    if not args.maintenance_only:
        store = BenchmarkArtifactStore(output)
        if all(
            store.completed(case, contract.fingerprint) for case in bundle.cases
        ):
            write_official_hypotheses(bundle, output)
    return output


if __name__ == "__main__":
    print(main())
