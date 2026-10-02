"""Bounded answer replay and frozen contradiction experiment; no memory updates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import time
from typing import Any

import pandas as pd

from agent_memory.adapters import LotusAdapter
from agent_memory.adapters.lotus.context import LotusExecutionConfig
from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.evaluation.locomo_contracts import locomo_task_contract
from agent_memory.evaluation.models import LiteLLMBenchmarkModel
from agent_memory.evaluation.trace_metrics import normalize_provider_calls, summarize_provider_calls
from agent_memory.evaluation.types import BenchmarkQuestion
from agent_memory.policy.logical import QueryExpr

MODEL = "deepseek/deepseek-flash"


def read_lines(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL evidence, failing on partial records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    """Atomically write a new experiment artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def freeze(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the first three multi-candidate groups without reading decisions."""
    results = {e["operator_call_id"]: e for e in events
               if e.get("operator") == "sem_filter" and e.get("event_type") == "operator_result"}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    identities: set[str] = set()
    for event in events:
        if event.get("operator") != "sem_filter" or event.get("event_type") != "pair_decision":
            continue
        if event["trace_id"] in identities:
            raise ValueError("duplicate occurrence trace identity")
        identities.add(event["trace_id"])
        if event["decision_source"] != "oracle" or event["direction"] != "right-to-left":
            raise ValueError("expected post-screening, directed oracle pairs")
        groups.setdefault((event["event_id"], event["right_id"]), []).append(event)
    selected = []
    for (event_id, right_id), rows in groups.items():
        if len(rows) < 2:
            continue
        instruction = results[rows[0]["operator_call_id"]]["lowered_instruction"]
        if set(re.findall(r"\{([^{}]+)\}", instruction)) != {"fact_earlier_added", "fact_later_added"}:
            raise ValueError("cannot restore every predicate input")
        frozen = []
        for row in rows[:10]:
            if results[row["operator_call_id"]]["lowered_instruction"] != instruction:
                raise ValueError("mixed predicate instructions")
            if not row["left"].startswith("fact: ") or not row["right"].startswith("fact: "):
                raise ValueError("unrecoverable fact text")
            frozen.append({"occurrence": row["trace_id"], "operator_call_id": row["operator_call_id"],
                           "pair_index": row["pair_index"], "left_id": row["left_id"],
                           "right_id": row["right_id"], "direction": row["direction"],
                           "fact_earlier_added": row["left"][6:], "fact_later_added": row["right"][6:],
                           "historical_decision": row["decision"]})
        selected.append({"event_id": event_id, "right_id": right_id,
                         "instruction": instruction, "rows": frozen})
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError("fewer than three multi-candidate groups")
    return selected


def preflight(old: Path, output: Path) -> None:
    """Freeze exact evidence and verify its original provider prompts."""
    output.mkdir(parents=True, exist_ok=False)
    import agent_memory
    from agent_memory.evaluation.provenance import build_source_evidence

    source_root = Path(agent_memory.__file__).resolve().parents[2]
    save(output / "source-evidence.json", build_source_evidence(source_root))
    trace = old / "fused/trace/events.jsonl"
    events = read_lines(trace)
    groups = freeze(events)
    calls = {(e["operator_call_id"], e["llm_item_index"]): e for e in events
             if e.get("event_type") == "llm_call" and e.get("operator") == "sem_filter"}
    hashes = {str(trace): digest(trace)}
    for group in groups:
        for row in group["rows"]:
            call = calls[(row["operator_call_id"], row["pair_index"])]
            path = old / "fused" / call["prompt_path"]
            messages = json.loads(path.read_text())
            text = "\n".join(m["content"] for m in messages)
            if not all(row[k] in text for k in ("fact_earlier_added", "fact_later_added")):
                raise ValueError("frozen text not found in original provider context")
            hashes[str(path)] = digest(path)
            row["original_messages"] = messages
    save(output / "frozen.json", groups)
    saved_answers = {}
    for mode in ("unfused", "fused"):
        questions_path = old / mode / "input/questions.jsonl"
        retrieval_path = old / mode / "cases/conv-26-2aac22fc/retrieval.jsonl"
        q, r = read_lines(questions_path), read_lines(retrieval_path)
        if len(q) != 1 or len(r) != 1 or q[0]["question_id"] != r[0]["question_id"]:
            raise ValueError("ambiguous question/retrieval identity")
        saved_answers[mode] = {"question": q[0], "context": r[0]["context"]}
        hashes.update({str(p): digest(p) for p in (questions_path, retrieval_path)})
    save(output / "answer-inputs.json", saved_answers)
    contract = locomo_task_contract(judge_model_id=MODEL)
    save(output / "manifest.json", {"model": MODEL, "input_hashes": hashes,
         "script_sha256": digest(Path(__file__)), "source_path": agent_memory.__file__,
         "dependencies": {name: version(name) for name in ("lotus-ai", "litellm", "pandas")},
         "answer_prompt_digest": contract.answer_prompt_digest, "answer_parser_id": contract.answer_parser_id,
         "order": ["answers", "pointwise", "joint"], "cache": False, "provider_retries": 0,
         "pair_parse_retries": 0, "groups": len(groups), "pairs": sum(len(g["rows"]) for g in groups),
         "note": "frozen negatives only; not full Sample 0 or contradiction recall"})
    save(output / "status.json", {"stage": "preflight_passed"})


def replay_answers(output: Path) -> None:
    contract = locomo_task_contract(judge_model_id=MODEL)
    model = LiteLLMBenchmarkModel(model_id=MODEL)
    for mode, data in json.loads((output / "answer-inputs.json").read_text()).items():
        q = {k: v for k, v in data["question"].items() if k != "case_id"}
        q["evidence_event_ids"] = tuple(q["evidence_event_ids"])
        question = BenchmarkQuestion(**q)
        prompt = contract.answer_prompt(question, data["context"])
        directory = output / "answers" / mode
        save(directory / "prompt.json", asdict(prompt))
        answer = None
        for attempt in (1, 2):
            response = model.complete(prompt, attempt=attempt)
            save(directory / f"answer-attempt-{attempt}.json", asdict(response))
            try:
                answer = contract.answer_parser(response.text)
            except ValueError as error:
                save(directory / f"parse-error-{attempt}.json", {"error": str(error)})
            else:
                break
        if answer is None:
            save(directory / "result.json", {"status": "parse_failed"})
            continue
        assert contract.deterministic_scorer is not None
        official = contract.deterministic_scorer(question, answer)
        grader = contract.additional_graders[0]
        assert grader.judge_plan is not None and grader.judge_reducer is not None
        steps = grader.judge_plan(question, answer)
        if len(steps) != 1:
            raise ValueError("expected one judge call")
        response = model.complete(steps[0].prompt, attempt=1)
        save(directory / "judge.json", {"prompt": asdict(steps[0].prompt), "response": asdict(response)})
        judged = grader.judge_reducer(question, answer, (steps[0].parse(response.text),))
        save(directory / "result.json", {"status": "completed", "answer": answer,
             "gold": question.gold_answer, "official": asdict(official), "judge": asdict(judged)})


def synthetic_preflight(fixture: Path, output: Path) -> None:
    """Freeze controlled contrasts; never pass assessment labels to the model."""
    from agent_memory.memories.zep.policy import _CONTRADICTORY_FACT_INSTRUCTION

    data = json.loads(fixture.read_text())
    if data.get("kind") != "synthetic-controlled-contrasts" or len(data["groups"]) != 6:
        raise ValueError("expected six explicitly synthetic groups")
    instruction = _CONTRADICTORY_FACT_INSTRUCTION.replace("{fact:later_added}", "{fact_later_added}").replace(
        "{fact:earlier_added}", "{fact_earlier_added}")
    groups, assessment = [], []
    for index, group in enumerate(data["groups"]):
        candidates = group["candidates"]
        if len(candidates) != 4 or any(type(c["expected"]) is not bool for c in candidates):
            raise ValueError("expected four strictly boolean-labeled candidates")
        if sum(c["expected"] for c in candidates) != 1:
            raise ValueError("expected exactly one contradiction per group")
        rows = []
        for pair, candidate in enumerate(candidates):
            if not all(isinstance(value, str) and value.strip() for value in
                       (group["new_fact"], candidate["fact"], candidate["reason"])):
                raise ValueError("empty fact or assessment reason")
            occurrence = f"synthetic-{index}-{pair}"
            rows.append({"occurrence": occurrence, "fact_earlier_added": candidate["fact"],
                         "fact_later_added": group["new_fact"]})
            assessment.append({"occurrence": occurrence, "expected": candidate["expected"],
                               "reason": candidate["reason"]})
        groups.append({"topic": group["topic"], "instruction": instruction, "rows": rows})
    output.mkdir(parents=True, exist_ok=False)
    save(output / "frozen.json", groups)
    save(output / "assessment.json", assessment)
    import agent_memory
    save(output / "manifest.json", {"model": MODEL, "input_hashes": {str(fixture): digest(fixture)},
         "script_sha256": digest(Path(__file__)), "frozen_sha256": digest(output / "frozen.json"),
         "assessment_sha256": digest(output / "assessment.json"), "source_path": agent_memory.__file__,
         "dependencies": {name: version(name) for name in ("lotus-ai", "litellm", "pandas")},
         "kind": data["kind"], "order": ["pointwise", "joint"], "pairs": 24, "groups": 6,
         "maximum_requests": 30, "cache": False, "provider_retries": 0, "pair_parse_retries": 0})
    save(output / "status.json", {"stage": "preflight_passed"})


def evaluate_group(adapter: Any, group: dict[str, Any]) -> list[bool]:
    """Execute the existing filter on frozen rows without candidate regeneration."""
    frame = pd.DataFrame([{k: row[k] for k in ("occurrence", "fact_earlier_added", "fact_later_added")}
                          for row in group["rows"]], columns=pd.Index(["occurrence", "fact_earlier_added", "fact_later_added"]))
    if frame.empty:
        return []
    source = QueryExpr(op="materialized_view", params={"name": "frozen", "columns": tuple(frame.columns)})
    query = QueryExpr(op="sem_filter", inputs=(source,), params={"instruction": group["instruction"]})
    result = adapter.execute(query, {"frozen": frame})
    kept = result["occurrence"].tolist()
    if len(kept) != len(set(kept)) or not set(kept) <= set(frame["occurrence"]):
        raise ValueError("invalid output occurrence identities")
    return [identity in kept for identity in frame["occurrence"]]


def validate_native_outputs(events: list[dict[str, Any]], directory: Path, decisions: list[bool]) -> None:
    """Reject native parser defaults rather than silently accepting malformed text."""
    raw = [json.loads((directory / e["raw_output_path"]).read_text())["output"]
           for e in events if e.get("event_type") == "llm_call" and "raw_output_path" in e]
    if len(raw) != len(decisions):
        raise ValueError("native output count mismatch")
    for output, decision in zip(raw, decisions, strict=True):
        match = re.fullmatch(r"\s*(?:Answer:\s*)?(True|False)\s*", output, flags=re.IGNORECASE)
        if match is None or (match[1].lower() == "true") != decision:
            raise ValueError("native output is not an unambiguous boolean decision")


def run(output: Path) -> None:
    if json.loads((output / "status.json").read_text())["stage"] != "preflight_passed":
        raise ValueError("refusing replay of a started experiment")
    manifest = json.loads((output / "manifest.json").read_text())
    if any(digest(Path(p)) != expected for p, expected in manifest["input_hashes"].items()):
        raise ValueError("historical evidence changed")
    if digest(Path(__file__)) != manifest["script_sha256"]:
        raise ValueError("script changed after preflight")
    if manifest.get("kind") == "synthetic-controlled-contrasts":
        for name in ("frozen", "assessment"):
            if digest(output / f"{name}.json") != manifest[f"{name}_sha256"]:
                raise ValueError("frozen synthetic evidence changed")
    else:
        save(output / "status.json", {"stage": "answer_replay"})
        replay_answers(output)
    groups = json.loads((output / "frozen.json").read_text())
    summaries = {}
    for mode in ("pointwise", "joint"):
        directory = output / mode
        save(output / "status.json", {"stage": "contradictions", "mode": mode})
        adapter = LotusAdapter(model=MODEL, config=LotusExecutionConfig(
            semantic_trace_dir=directory / "trace", lm_enable_cache=False, lm_num_retries=0,
            lm_max_batch_size=1, structured_parse_retries=0, structured_max_tokens=8192,
            prompt_batching=PromptBatching(max_tasks=10) if mode == "joint" else None,
            lm_model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}}))
        results = []
        for index, group in enumerate(groups):
            trace = directory / "trace/events.jsonl"
            before_count = len(read_lines(trace)) if trace.exists() else 0
            started = time.perf_counter()
            decisions = evaluate_group(adapter, group)
            if mode == "pointwise":
                validate_native_outputs(read_lines(trace)[before_count:], directory, decisions)
            results.append({"group": index, "decisions": decisions, "wall_seconds": time.perf_counter() - started})
            save(directory / "results.json", results)
        calls = normalize_provider_calls(read_lines(directory / "trace/events.jsonl"), output_dir=directory, include_cost=False)
        summaries[mode] = {"results": results, "usage": summarize_provider_calls(calls)}
        save(directory / "calls.json", calls)
        save(output / "summary.json", summaries)
    save(output / "status.json", {"stage": "completed"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "synthetic-preflight", "run"))
    parser.add_argument("--old", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        if args.old is None:
            parser.error("preflight requires --old")
        preflight(args.old, args.output)
    elif args.mode == "synthetic-preflight":
        if args.fixture is None:
            parser.error("synthetic-preflight requires --fixture")
        synthetic_preflight(args.fixture, args.output)
    else:
        try:
            run(args.output)
        except Exception as error:
            save(args.output / "status.json", {"stage": "failed", "error": str(error), "type": type(error).__name__})
            raise
