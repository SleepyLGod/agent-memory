"""Prepare or run the pinned LongMemEval v1 canonical benchmark bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from sys import path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

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
    run_agent_memory_bundle,
)

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
    if args.maintenance_only:
        return {
            "rule-join-map": "JM-M",
            "rule-re-group": "RG-M",
        }.get(args.grouped_agg_rule, "claude-memory-maintenance")
    return _PILOT_CONDITIONS.get(
        (args.grouped_agg_rule, args.sem_topk_method),
        "claude-memory",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical bundle preparation and agent-memory execution commands."""

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

    run = commands.add_parser("run")
    run.add_argument("--bundle-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--memory-model-id", default=DEFAULT_MEMORY_MODEL_ID)
    run.add_argument("--memory-model", default=DEFAULT_PROVIDER_MODEL)
    run.add_argument("--answer-model", default=DEFAULT_PROVIDER_MODEL)
    run.add_argument("--judge-model", default=DEFAULT_PROVIDER_MODEL)
    run.add_argument("--namespace")
    run.add_argument("--condition-id")
    run.add_argument("--max-new-cases", type=int)
    run.add_argument(
        "--grouped-agg-rule",
        choices=("rule-all-group", "rule-join-map", "rule-re-group"),
        default="rule-all-group",
    )
    run.add_argument(
        "--sem-topk-method",
        choices=("pairwise-naive", "pairwise-quick", "pairwise-heap", "listwise"),
        default="pairwise-naive",
    )
    run.add_argument(
        "--memory-thinking",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    run.add_argument("--maintenance-checkpoint-output-dir", type=Path)
    run.add_argument("--maintenance-only", action="store_true")
    args = parser.parse_args(argv)
    if getattr(args, "question_ids", None) is not None:
        args.question_ids = tuple(args.question_ids)
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
    output = run_agent_memory_bundle(
        bundle=bundle,
        contracts={
            "longmemeval-v1": longmemeval_task_contract(
                judge_model_id=args.judge_model
            )
        },
        system_id="claude-memory",
        output_dir=args.output_dir,
        memory_model_id=args.memory_model_id,
        memory_provider_model_id=args.memory_model,
        answer_model_id=args.answer_model,
        judge_model_id=args.judge_model,
        base_namespace=args.namespace,
        grouped_agg_rule=args.grouped_agg_rule,
        sem_topk_method=args.sem_topk_method,
        memory_thinking_enabled=args.memory_thinking == "enabled",
        condition_id=_condition_id(args),
        maintenance_only=args.maintenance_only,
        maintenance_checkpoint_output_dir=args.maintenance_checkpoint_output_dir,
        max_new_cases=args.max_new_cases,
    )
    if not args.maintenance_only:
        write_official_hypotheses(bundle, output)
    return output


if __name__ == "__main__":
    print(main())
