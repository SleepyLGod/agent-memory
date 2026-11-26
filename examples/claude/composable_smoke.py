"""Explicitly gated, small Flash experiment with synthetic memory messages."""

import argparse
import json
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv

from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.planner.rules import DifferentialRules
from composable_memory import ComposableClaudeMemory


def main() -> None:
    """Run a small maintenance/retrieval case, saving raw evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Authorize real model calls")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=["topics", "update"], default="topics")
    args = parser.parse_args()
    if not args.run:
        parser.error("Real model calls require --run")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    args.output.mkdir(parents=True, exist_ok=False)
    model = "deepseek/deepseek-v4-flash"
    adapter = LotusAdapter(
        model=model,
        config=LotusExecutionConfig(
            lm_num_retries=0,
            lm_timeout=60,
            lm_enable_cache=False,
            structured_parse_retries=0,
            structured_max_tokens=2048,
            lm_model_kwargs={
                "max_tokens": 2048,
                "extra_body": {"thinking": {"type": "disabled"}},
            },
            semantic_trace_dir=args.output / "trace",
            semantic_trace_snapshot_mode="full",
        ),
    )
    memory = ComposableClaudeMemory(
        adapter=adapter,
        rules=DifferentialRules(
            sem_agg_rule="agg-composable",
            sem_groupby_rule="group-composable",
        ),
    )
    policy = memory._runtime.policy
    kinds = [n.execution_kind for n in policy.nodes.values()]
    assert kinds.count("semantic_composable") == 2
    messages = [
        "Remember that I am allergic to peanuts.",
        "For my dietary profile, also remember that I prefer vegetarian meals.",
        "Remember that the Atlas project's final report is due Friday.",
    ]
    question = "What are my dietary restrictions and the Atlas deadline?"
    if args.case == "update":
        messages = [
            "Remember that the Atlas project's final report is due Friday, 2026-11-06.",
            "Update that same Atlas final-report deadline: it is now Monday, 2026-11-09, not Friday.",
        ]
        question = "What is the current deadline for the Atlas final report?"
    report: dict[str, Any] = {
        "model": model,
        "synthetic": True,
        "case": args.case,
        "composable_nodes": kinds.count("semantic_composable"),
        "steps": [],
    }
    started = time.perf_counter()
    try:
        for i, message in enumerate(messages):
            memory.add(
                {
                    "message": message,
                    "role": "user",
                    "timestamp": f"2026-11-04T10:0{i}:00",
                    "session_id": "composable-review",
                    "metadata": {},
                }
            )
            report["steps"].append(
                {
                    "input": message,
                    "topics": memory._runtime._state["topics"].to_dict("records"),
                    "catalog": memory._runtime._state["catalog"].to_dict("records"),
                }
            )
        result = memory.query(question)
        report["retrieval"] = result.to_dict("records")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str)
        )
        print(json.dumps(report, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
