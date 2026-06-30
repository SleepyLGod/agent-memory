"""Analyze existing LOCOMO comparison artifacts without running any models."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for zero-cost LOCOMO artifact analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Read existing agent-memory and Native Claude Code LOCOMO artifacts "
            "and write zero-cost comparison diagnostics."
        )
    )
    parser.add_argument("--agent-run", required=True, type=Path, help="agent-memory benchmark output directory.")
    parser.add_argument("--native-run", required=True, type=Path, help="Native CC benchmark output directory.")
    parser.add_argument(
        "--native-maint-run",
        required=True,
        type=Path,
        help="Native CC maintenance/frozen-memory output directory.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for analysis CSVs and report.md.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as a list of dictionaries."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write rows to a CSV file, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_summary(path: Path) -> dict[str, str]:
    """Read a one-row summary CSV."""

    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Summary CSV has no rows: {path}")
    return rows[0]


def read_text(path: Path) -> str:
    """Read text from an existing file or return an empty string."""

    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def int_value(value: Any) -> int:
    """Convert a CSV/JSON value to int with empty values treated as zero."""

    if value in (None, ""):
        return 0
    return int(float(str(value)))


def float_value(value: Any) -> float:
    """Convert a CSV/JSON value to float with empty values treated as zero."""

    if value in (None, ""):
        return 0.0
    return float(str(value))


def bool_text(value: Any) -> bool:
    """Return true for common CSV boolean spellings."""

    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_text(value: str) -> str:
    """Normalize text for diagnostic substring checks."""

    return re.sub(r"\s+", " ", value.casefold()).strip()


def gold_probe(gold_answer: str) -> str:
    """Return a conservative gold-answer probe for diagnostic text containment."""

    return gold_answer.split(";")[0].strip()


def contains_probe(text: str, gold_answer: str) -> bool:
    """Return whether text contains the diagnostic gold-answer probe."""

    probe = normalize_text(gold_probe(gold_answer))
    if not probe:
        return False
    return probe in normalize_text(text)


def agent_memory_text(agent_run: Path) -> str:
    """Return concatenated agent-memory final memory text."""

    chunks: list[str] = []
    for relative in ("memory/topics.csv", "memory/catalog.csv"):
        path = agent_run / relative
        if not path.exists():
            continue
        for row in read_csv(path):
            chunks.extend(str(row.get(field) or "") for field in ("name", "description", "type", "body", "hook"))
    return "\n".join(chunks)


def native_memory_text(native_run: Path, native_maint_run: Path) -> str:
    """Return concatenated Native CC final markdown memory text."""

    summary_path = native_run / "metrics" / "summary.csv"
    memory_path = ""
    if summary_path.exists():
        memory_path = read_summary(summary_path).get("final_memory_path") or ""
    if not memory_path:
        maint_summary = native_maint_run / "native" / "component_eval" / "metrics" / "summary.csv"
        if maint_summary.exists():
            memory_path = read_summary(maint_summary).get("finalMemoryPath") or ""
    root = Path(memory_path) if memory_path else native_maint_run / "native" / "final_memory" / "memory"
    chunks = []
    if root.exists():
        for path in sorted(root.rglob("*.md")):
            chunks.append(read_text(path))
    return "\n".join(chunks)


def by_question(rows: Iterable[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    """Index rows by question_id."""

    return {str(row.get("question_id", "")): row for row in rows if row.get("question_id")}


def winner(agent_score: float, native_score: float) -> str:
    """Return a plain-language per-question winner label."""

    if agent_score > native_score:
        return "agent-memory"
    if native_score > agent_score:
        return "native-cc"
    return "tie"


def build_per_question_comparison(agent_run: Path, native_run: Path, native_maint_run: Path) -> list[dict[str, Any]]:
    """Build per-question score and diagnostic containment rows."""

    agent_rows = by_question(read_csv(agent_run / "metrics" / "questions.csv"))
    native_rows = by_question(read_csv(native_run / "metrics" / "questions.csv"))
    agent_question_ids = set(agent_rows)
    native_question_ids = set(native_rows)
    if agent_question_ids != native_question_ids:
        missing_in_agent = sorted(native_question_ids - agent_question_ids)
        missing_in_native = sorted(agent_question_ids - native_question_ids)
        raise ValueError(
            "Question set mismatch between runs: "
            f"missing in agent={missing_in_agent}, missing in native={missing_in_native}"
        )
    agent_mem_text = agent_memory_text(agent_run)
    native_mem_text = native_memory_text(native_run, native_maint_run)
    rows: list[dict[str, Any]] = []
    for question_id in sorted(agent_question_ids):
        agent = agent_rows[question_id]
        native = native_rows[question_id]
        gold = str(agent.get("gold_answer") or native.get("gold_answer") or "")
        agent_score = float_value(agent.get("locomo_answer_score"))
        native_score = float_value(native.get("locomo_answer_score"))
        rows.append(
            {
                "question_id": question_id,
                "category": agent.get("category") or native.get("category", ""),
                "question": agent.get("question") or native.get("question", ""),
                "gold_answer": gold,
                "agent_locomo_answer_score": agent_score,
                "native_locomo_answer_score": native_score,
                "winner": winner(agent_score, native_score),
                "score_delta_agent_minus_native": round(agent_score - native_score, 6),
                "agent_retrieved_row_count": agent.get("retrieved_row_count", ""),
                "native_retrieved_row_count": native.get("retrieved_row_count", ""),
                "agent_proxy_answer_string_hit": agent.get("proxy_answer_string_hit", ""),
                "native_proxy_answer_string_hit": native.get("proxy_answer_string_hit", ""),
                "agent_answer_f1": agent.get("answer_f1", ""),
                "native_answer_f1": native.get("answer_f1", ""),
                "agent_generated_answer": agent.get("generated_answer", ""),
                "native_generated_answer": native.get("generated_answer", ""),
                "agent_final_memory_contains_gold_probe": contains_probe(agent_mem_text, gold),
                "native_final_memory_contains_gold_probe": contains_probe(native_mem_text, gold),
                "agent_retrieved_contains_gold_probe": contains_probe(str(agent.get("retrieved_text", "")), gold),
                "native_retrieved_contains_gold_probe": contains_probe(str(native.get("retrieved_text", "")), gold),
                "agent_answer_contains_gold_probe": contains_probe(str(agent.get("generated_answer", "")), gold),
                "native_answer_contains_gold_probe": contains_probe(str(native.get("generated_answer", "")), gold),
            }
        )
    return rows


def aggregate_agent_phase_operator_cost(agent_run: Path) -> list[dict[str, Any]]:
    """Aggregate agent-memory trace usage by phase and operator."""

    trace_path = agent_run / "trace" / "events.jsonl"
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "phase": "",
            "operator": "",
            "llm_call_count": 0,
            "llm_batch_count": 0,
            "latency_sec": 0.0,
            "physical_prompt_tokens": 0,
            "physical_completion_tokens": 0,
            "physical_total_tokens": 0,
            "virtual_prompt_tokens": 0,
            "virtual_completion_tokens": 0,
            "virtual_total_tokens": 0,
            "cache_hits": 0,
            "empty_output_count": 0,
        }
    )
    seen_batches: set[tuple[str, str, str]] = set()
    if trace_path.exists():
        with trace_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {trace_path}:{line_number}") from exc
                if event.get("event_type") != "llm_call":
                    continue
                phase = str(event.get("phase") or "")
                operator = str(event.get("operator") or event.get("semantic_operator") or "")
                key = (phase, operator)
                row = groups[key]
                row["phase"] = phase
                row["operator"] = operator
                row["llm_call_count"] += 1
                batch_id = event.get("llm_batch_id") or event.get("trace_id")
                batch_id = str(batch_id) if batch_id else f"line:{line_number}"
                batch_key = (phase, operator, batch_id)
                if batch_key not in seen_batches:
                    seen_batches.add(batch_key)
                    row["llm_batch_count"] += 1
                    row["latency_sec"] += float_value(event.get("latency_sec"))
                row["physical_prompt_tokens"] += int_value(event.get("usage_physical_prompt_tokens"))
                row["physical_completion_tokens"] += int_value(event.get("usage_physical_completion_tokens"))
                row["physical_total_tokens"] += int_value(event.get("usage_physical_total_tokens"))
                row["virtual_prompt_tokens"] += int_value(event.get("usage_virtual_prompt_tokens"))
                row["virtual_completion_tokens"] += int_value(event.get("usage_virtual_completion_tokens"))
                row["virtual_total_tokens"] += int_value(event.get("usage_virtual_total_tokens"))
                row["cache_hits"] += int_value(event.get("usage_cache_hits"))

    anomalies_path = agent_run / "diagnostics" / "llm_anomalies.csv"
    if anomalies_path.exists():
        for anomaly in read_csv(anomalies_path):
            if anomaly.get("issue") != "empty_output":
                continue
            key = (anomaly.get("phase", ""), anomaly.get("operator", ""))
            row = groups[key]
            row["phase"] = key[0]
            row["operator"] = key[1]
            row["empty_output_count"] += 1

    rows = list(groups.values())
    for row in rows:
        row["latency_sec"] = round(float(row["latency_sec"]), 4)
    return sorted(rows, key=lambda item: (str(item["phase"]), str(item["operator"])))


def aggregate_native_trace_usage(native_run: Path, native_maint_run: Path) -> list[dict[str, Any]]:
    """Aggregate Native CC trace usage into maintenance and retrieval/answer rows."""

    rows: list[dict[str, Any]] = []
    maint_summary_path = native_maint_run / "native" / "component_eval" / "metrics" / "summary.csv"
    if maint_summary_path.exists():
        summary = read_summary(maint_summary_path)
        rows.append(
            {
                "phase": "maintenance",
                "source": "component_eval_summary",
                "llm_call_count": int_value(summary.get("traceLlmCalls")),
                "llm_error_count": int_value(summary.get("traceLlmErrorCalls")),
                "input_tokens": int_value(summary.get("traceLlmInputTokens")),
                "output_tokens": int_value(summary.get("traceLlmOutputTokens")),
                "cache_read_input_tokens": int_value(summary.get("traceLlmCacheReadInputTokens")),
                "cache_creation_input_tokens": int_value(summary.get("traceLlmCacheCreationInputTokens")),
                "latency_sec": round(float_value(summary.get("traceLlmLatencyMs")) / 1000, 4),
            }
        )

    trace_path = native_run / "trace" / "events.jsonl"
    aggregate = {
        "phase": "retrieval_answer",
        "source": "benchmark_trace",
        "llm_call_count": 0,
        "llm_error_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "latency_sec": 0.0,
    }
    if trace_path.exists():
        with trace_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {trace_path}:{line_number}") from exc
                event_type = event.get("event_type")
                if event_type == "llm_call_finish":
                    usage = event.get("usage") or {}
                    aggregate["llm_call_count"] += 1
                    aggregate["input_tokens"] += int_value(usage.get("input_tokens"))
                    aggregate["output_tokens"] += int_value(usage.get("output_tokens"))
                    aggregate["cache_read_input_tokens"] += int_value(usage.get("cache_read_input_tokens"))
                    aggregate["cache_creation_input_tokens"] += int_value(usage.get("cache_creation_input_tokens"))
                    aggregate["latency_sec"] += float_value(event.get("latency_ms")) / 1000
                elif event_type == "llm_call_error":
                    aggregate["llm_error_count"] += 1
    aggregate["latency_sec"] = round(float(aggregate["latency_sec"]), 4)
    rows.append(aggregate)
    return rows


def build_agent_duplicate_expansion(agent_run: Path) -> list[dict[str, Any]]:
    """Build per-question retrieved-name duplicate expansion diagnostics."""

    source = agent_run / "retrieval" / "results.csv"
    if not source.exists():
        source = agent_run / "metrics" / "questions.csv"
    rows = []
    for row in read_csv(source):
        names = [name for name in str(row.get("retrieved_names", "")).split(";") if name]
        counts = Counter(names)
        duplicated = [name for name, count in counts.items() if count > 1]
        rows.append(
            {
                "question_id": row.get("question_id", ""),
                "category": row.get("category", ""),
                "retrieved_row_count": int_value(row.get("retrieved_row_count")),
                "unique_retrieved_names": len(counts),
                "duplicate_retrieved_rows": max(0, len(names) - len(counts)),
                "duplicated_names": ";".join(sorted(duplicated)),
                "proxy_answer_string_hit": row.get("proxy_answer_string_hit", ""),
                "locomo_answer_score": row.get("locomo_answer_score", ""),
            }
        )
    return rows


def build_native_selector_outcomes(native_run: Path) -> list[dict[str, Any]]:
    """Build per-question Native CC selector outcome diagnostics."""

    rows = []
    for row in read_csv(native_run / "retrieval" / "results.csv"):
        anomaly = row.get("retrieval_anomaly_reason", "")
        retrieved_count = int_value(row.get("retrieved_row_count"))
        if anomaly:
            outcome = anomaly
        elif retrieved_count > 0:
            outcome = "retrieved"
        else:
            outcome = "clean_empty"
        rows.append(
            {
                "question_id": row.get("question_id", ""),
                "category": row.get("category", ""),
                "outcome": outcome,
                "retrieved_row_count": retrieved_count,
                "retrieved_paths": row.get("retrieved_paths", ""),
                "retrieval_anomaly_reason": anomaly,
                "selector_selected_from_trace": row.get("selector_selected_from_trace", ""),
                "selector_trace_id": row.get("selector_trace_id", ""),
                "proxy_answer_string_hit": row.get("proxy_answer_string_hit", ""),
                "locomo_answer_score": row.get("locomo_answer_score", ""),
            }
        )
    return rows


def sum_int(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    """Sum an integer field over rows."""

    return sum(int_value(row.get(field)) for row in rows)


def average_float(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    """Average a numeric field over rows."""

    if not rows:
        return 0.0
    return sum(float_value(row.get(field)) for row in rows) / len(rows)


def write_report(
    output_dir: Path,
    *,
    agent_summary: Mapping[str, str],
    native_summary: Mapping[str, str],
    per_question_rows: Sequence[Mapping[str, Any]],
    agent_cost_rows: Sequence[Mapping[str, Any]],
    native_usage_rows: Sequence[Mapping[str, Any]],
    agent_duplicate_rows: Sequence[Mapping[str, Any]],
    native_selector_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a concise Markdown report for the zero-cost analysis."""

    agent_wins = sum(1 for row in per_question_rows if row.get("winner") == "agent-memory")
    native_wins = sum(1 for row in per_question_rows if row.get("winner") == "native-cc")
    ties = sum(1 for row in per_question_rows if row.get("winner") == "tie")
    duplicate_questions = sum(1 for row in agent_duplicate_rows if int_value(row.get("duplicate_retrieved_rows")) > 0)
    native_empty = sum(1 for row in native_selector_rows if row.get("outcome") == "clean_empty")
    native_anomalies = sum(1 for row in native_selector_rows if row.get("retrieval_anomaly_reason"))
    native_retrieved = sum(1 for row in native_selector_rows if row.get("outcome") == "retrieved")

    lines = [
        "# LOCOMO Zero-Cost Comparison Analysis",
        "",
        "This report only reads existing artifacts. It does not call any model and does not rerun benchmark code.",
        "",
        "## Score Summary",
        "",
        "| metric | agent-memory | Native CC |",
        "|---|---:|---:|",
        f"| questions | {agent_summary.get('questions_evaluated', '')} | {native_summary.get('questions_evaluated', '')} |",
        f"| LOCOMO answer score mean | {agent_summary.get('locomo_answer_score_mean', '')} | {native_summary.get('locomo_answer_score_mean', '')} |",
        f"| answer F1 mean | {agent_summary.get('answer_f1_mean', '')} | {native_summary.get('answer_f1_mean', '')} |",
        f"| proxy answer string hit rate | {agent_summary.get('proxy_answer_string_hit_rate', '')} | {native_summary.get('proxy_answer_string_hit_rate', '')} |",
        "",
        "## Per-Question Outcome",
        "",
        f"- agent-memory wins: {agent_wins}",
        f"- Native CC wins: {native_wins}",
        f"- ties: {ties}",
        "",
        "See `accuracy/per_question_comparison.csv` for question-level rows.",
        "",
        "## Retrieval Shape",
        "",
        f"- agent-memory questions with duplicate retrieved rows: {duplicate_questions}",
        f"- agent-memory average duplicate retrieved rows: {average_float(agent_duplicate_rows, 'duplicate_retrieved_rows'):.3f}",
        f"- Native CC retrieved questions: {native_retrieved}",
        f"- Native CC clean empty retrievals: {native_empty}",
        f"- Native CC selector anomalies: {native_anomalies}",
        "",
        "See `retrieval/agent_duplicate_expansion.csv` and `retrieval/native_selector_outcomes.csv`.",
        "",
        "## Cost Shape",
        "",
        "### agent-memory by phase/operator",
        "",
        "| phase | operator | llm items | llm batches | total tokens | empty outputs | latency sec |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in agent_cost_rows:
        lines.append(
            "| {phase} | {operator} | {calls} | {batches} | {tokens} | {empty} | {latency} |".format(
                phase=row.get("phase", ""),
                operator=row.get("operator", ""),
                calls=row.get("llm_call_count", 0),
                batches=row.get("llm_batch_count", 0),
                tokens=row.get("physical_total_tokens", 0),
                empty=row.get("empty_output_count", 0),
                latency=row.get("latency_sec", 0),
            )
        )
    lines.extend(
        [
            "",
            "### Native CC usage",
            "",
            "| phase | llm calls | input tokens | output tokens | cache read input tokens | latency sec |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in native_usage_rows:
        lines.append(
            "| {phase} | {calls} | {inp} | {out} | {cache} | {latency} |".format(
                phase=row.get("phase", ""),
                calls=row.get("llm_call_count", 0),
                inp=row.get("input_tokens", 0),
                out=row.get("output_tokens", 0),
                cache=row.get("cache_read_input_tokens", 0),
                latency=row.get("latency_sec", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `*_contains_gold_probe` fields are diagnostic substring checks, not official LOCOMO scoring.",
            "- agent-memory token rows use LOTUS/LiteLLM visible aggregate usage from trace events.",
            "- Native CC token rows use provider-reported usage fields from Native trace artifacts.",
            "- This analysis does not change baseline artifacts.",
            "",
            "## Generated Files",
            "",
            "- `accuracy/per_question_comparison.csv`",
            "- `cost/agent_phase_operator_cost.csv`",
            "- `cost/native_phase_usage.csv`",
            "- `retrieval/agent_duplicate_expansion.csv`",
            "- `retrieval/native_selector_outcomes.csv`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run zero-cost LOCOMO artifact analysis."""

    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_summary = read_summary(args.agent_run / "metrics" / "summary.csv")
    native_summary = read_summary(args.native_run / "metrics" / "summary.csv")
    per_question_rows = build_per_question_comparison(args.agent_run, args.native_run, args.native_maint_run)
    agent_cost_rows = aggregate_agent_phase_operator_cost(args.agent_run)
    native_usage_rows = aggregate_native_trace_usage(args.native_run, args.native_maint_run)
    agent_duplicate_rows = build_agent_duplicate_expansion(args.agent_run)
    native_selector_rows = build_native_selector_outcomes(args.native_run)

    write_csv(
        output_dir / "accuracy" / "per_question_comparison.csv",
        per_question_rows,
        [
            "question_id",
            "category",
            "question",
            "gold_answer",
            "agent_locomo_answer_score",
            "native_locomo_answer_score",
            "winner",
            "score_delta_agent_minus_native",
            "agent_retrieved_row_count",
            "native_retrieved_row_count",
            "agent_proxy_answer_string_hit",
            "native_proxy_answer_string_hit",
            "agent_answer_f1",
            "native_answer_f1",
            "agent_generated_answer",
            "native_generated_answer",
            "agent_final_memory_contains_gold_probe",
            "native_final_memory_contains_gold_probe",
            "agent_retrieved_contains_gold_probe",
            "native_retrieved_contains_gold_probe",
            "agent_answer_contains_gold_probe",
            "native_answer_contains_gold_probe",
        ],
    )
    write_csv(
        output_dir / "cost" / "agent_phase_operator_cost.csv",
        agent_cost_rows,
        [
            "phase",
            "operator",
            "llm_call_count",
            "llm_batch_count",
            "latency_sec",
            "physical_prompt_tokens",
            "physical_completion_tokens",
            "physical_total_tokens",
            "virtual_prompt_tokens",
            "virtual_completion_tokens",
            "virtual_total_tokens",
            "cache_hits",
            "empty_output_count",
        ],
    )
    write_csv(
        output_dir / "cost" / "native_phase_usage.csv",
        native_usage_rows,
        [
            "phase",
            "source",
            "llm_call_count",
            "llm_error_count",
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "latency_sec",
        ],
    )
    write_csv(
        output_dir / "retrieval" / "agent_duplicate_expansion.csv",
        agent_duplicate_rows,
        [
            "question_id",
            "category",
            "retrieved_row_count",
            "unique_retrieved_names",
            "duplicate_retrieved_rows",
            "duplicated_names",
            "proxy_answer_string_hit",
            "locomo_answer_score",
        ],
    )
    write_csv(
        output_dir / "retrieval" / "native_selector_outcomes.csv",
        native_selector_rows,
        [
            "question_id",
            "category",
            "outcome",
            "retrieved_row_count",
            "retrieved_paths",
            "retrieval_anomaly_reason",
            "selector_selected_from_trace",
            "selector_trace_id",
            "proxy_answer_string_hit",
            "locomo_answer_score",
        ],
    )
    write_report(
        output_dir,
        agent_summary=agent_summary,
        native_summary=native_summary,
        per_question_rows=per_question_rows,
        agent_cost_rows=agent_cost_rows,
        native_usage_rows=native_usage_rows,
        agent_duplicate_rows=agent_duplicate_rows,
        native_selector_rows=native_selector_rows,
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
