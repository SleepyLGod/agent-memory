"""Run LOCOMO v1 benchmark slices against the built-in ClaudeMemory policy."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
from sys import path
import time
from typing import Any
import warnings

from dotenv import load_dotenv
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

import agent_memory as am  # noqa: E402
from agent_memory.adapters.lotus import DEFAULT_LOTUS_MODEL, LotusAdapter  # noqa: E402
from agent_memory.adapters.lotus.context import LotusExecutionConfig  # noqa: E402
from agent_memory.benchmarks.diagnostics import (  # noqa: E402
    LLM_ANOMALY_COLUMNS,
    build_cause_trace_rows,
    build_llm_anomaly_rows,
)
from agent_memory.benchmarks.locomo import (  # noqa: E402
    eligible_questions,
    event_to_claude_log_row,
    load_locomo_sample,
    select_events,
)
from agent_memory.benchmarks.metrics import (  # noqa: E402
    duplicate_name_count,
    duplicate_name_extra_rows,
    frame_text,
    question_metric_row,
    summarize_question_metrics,
)
from agent_memory.benchmarks.types import BenchmarkEvent, BenchmarkQuestion  # noqa: E402
from agent_memory.datasets.locomo import DEFAULT_LOCOMO_URL, ensure_locomo_dataset  # noqa: E402
from agent_memory.tracing.semantic import semantic_trace_scope  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".memory-test" / "locomo-benchmark" / "latest"
LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
ANSWER_MAX_TOKENS = 256
USAGE_FIELDS = (
    "physical_prompt_tokens",
    "physical_completion_tokens",
    "physical_total_tokens",
    "virtual_prompt_tokens",
    "virtual_completion_tokens",
    "virtual_total_tokens",
    "cache_hits",
)


def parse_args() -> argparse.Namespace:
    """Parse LOCOMO benchmark CLI arguments."""

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
        "--trace",
        action="store_true",
        help="Write semantic trace artifacts under output_dir/trace.",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Generate answers from retrieved memories and compute answer metrics.",
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


def reset_output_dir(output_dir: Path) -> None:
    """Create a clean benchmark output directory."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, frame: Any, output_dir: Path) -> Path:
    """Write a DataFrame-like object to CSV and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def usage_snapshot() -> dict[str, float | int]:
    """Return current LOTUS token/cache usage counters."""

    import lotus

    snapshot: dict[str, float | int] = {
        "physical_prompt_tokens": 0,
        "physical_completion_tokens": 0,
        "physical_total_tokens": 0,
        "virtual_prompt_tokens": 0,
        "virtual_completion_tokens": 0,
        "virtual_total_tokens": 0,
        "cache_hits": 0,
    }
    lm = lotus.settings.lm
    if lm is None:
        return snapshot
    stats = lm.stats
    snapshot.update(
        {
            "physical_prompt_tokens": stats.physical_usage.prompt_tokens,
            "physical_completion_tokens": stats.physical_usage.completion_tokens,
            "physical_total_tokens": stats.physical_usage.total_tokens,
            "virtual_prompt_tokens": stats.virtual_usage.prompt_tokens,
            "virtual_completion_tokens": stats.virtual_usage.completion_tokens,
            "virtual_total_tokens": stats.virtual_usage.total_tokens,
            "cache_hits": stats.cache_hits,
        }
    )
    return snapshot


def usage_delta(
    before: Mapping[str, float | int],
    after: Mapping[str, float | int],
) -> dict[str, float | int]:
    """Return usage counter deltas."""

    return {field: after[field] - before[field] for field in USAGE_FIELDS}


def selected_benchmark_data(
    *,
    sample_index: int,
    row_limit: int,
    question_limit: int,
) -> tuple[Path, tuple[BenchmarkEvent, ...], tuple[BenchmarkQuestion, ...]]:
    """Load and select one LOCOMO benchmark slice."""

    if row_limit < 0:
        raise SystemExit("--row-limit must be non-negative")
    if question_limit < 0:
        raise SystemExit("--question-limit must be non-negative")

    dataset_path = ensure_locomo_dataset(LOCOMO_CACHE_PATH, url=DEFAULT_LOCOMO_URL)
    sample = load_locomo_sample(dataset_path, sample_index=sample_index)
    events = select_events(sample.events, row_limit=row_limit)
    questions = eligible_questions(
        sample.questions,
        ingested_event_ids=[event.event_id for event in events],
        question_limit=question_limit,
    )
    return dataset_path, events, questions


def events_frame(events: Sequence[BenchmarkEvent]) -> pd.DataFrame:
    """Return selected benchmark events as CSV-ready rows."""

    return pd.DataFrame(
        [
            {
                "sample_id": event.sample_id,
                "event_id": event.event_id,
                "speaker": event.speaker,
                "session_id": event.session_id,
                "timestamp": event.timestamp,
                "text": event.text,
            }
            for event in events
        ]
    )


def questions_frame(questions: Sequence[BenchmarkQuestion]) -> pd.DataFrame:
    """Return selected benchmark questions as CSV-ready rows."""

    return pd.DataFrame(
        [
            {
                "question_id": question.question_id,
                "sample_id": question.sample_id,
                "question": question.question,
                "gold_answer": question.gold_answer,
                "evidence_event_ids": ";".join(question.evidence_event_ids),
                "category": question.category,
            }
            for question in questions
        ]
    )


def run_memory_ingest(
    memory: am.ClaudeMemory,
    events: Sequence[BenchmarkEvent],
    *,
    step_metrics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Append selected benchmark events into ClaudeMemory."""

    if step_metrics is None:
        step_metrics = []
    for index, event in enumerate(events, start=1):
        row = event_to_claude_log_row(event)
        before = usage_snapshot()
        start = time.perf_counter()
        with semantic_trace_scope(
            run_kind="benchmark",
            phase="add",
            add_index=index,
            sample_id=event.sample_id,
            event_id=event.event_id,
        ):
            memory.add(row)
        after = usage_snapshot()
        step_metrics.append(
            {
                "phase": "add",
                "event_id": event.event_id,
                "latency_sec": round(time.perf_counter() - start, 4),
                **usage_delta(before, after),
                **memory_row_counts(memory),
            }
        )
    return step_metrics


def create_memory(*, model: str, trace_dir: Path | None) -> am.ClaudeMemory:
    """Create the ClaudeMemory instance used by one benchmark run."""

    return am.ClaudeMemory(
        adapter=LotusAdapter(
            model=model,
            config=LotusExecutionConfig(semantic_trace_dir=trace_dir),
        )
    )


def memory_row_counts(memory: am.ClaudeMemory) -> dict[str, int]:
    """Return current public memory table sizes."""

    state = memory._runtime._state
    return {
        "log_rows": len(state.get("log", [])),
        "topics_rows": len(state.get("topics", [])),
        "catalog_rows": len(state.get("catalog", [])),
    }


def run_questions(
    memory: am.ClaudeMemory,
    questions: Sequence[BenchmarkQuestion],
    *,
    answer: bool,
    result_rows: list[dict[str, Any]] | None = None,
    metric_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run retrieval and optional answer generation for selected questions."""

    if result_rows is None:
        result_rows = []
    if metric_rows is None:
        metric_rows = []
    for question in questions:
        before = usage_snapshot()
        retrieval_start = time.perf_counter()
        with semantic_trace_scope(
            run_kind="benchmark",
            phase="retrieval",
            question_id=question.question_id,
        ):
            retrieved = memory.query(question.question)
        retrieval_latency = time.perf_counter() - retrieval_start
        after_retrieval = usage_snapshot()

        generated_answer: str | None = None
        answer_latency = 0.0
        answer_usage = {field: 0 for field in USAGE_FIELDS}
        if answer:
            before_answer = usage_snapshot()
            answer_start = time.perf_counter()
            generated_answer = generate_answer(
                question.question,
                retrieved,
                question_id=question.question_id,
            )
            answer_latency = time.perf_counter() - answer_start
            answer_usage = usage_delta(before_answer, usage_snapshot())

        metric_row = question_metric_row(
            question_id=question.question_id,
            question=question.question,
            gold_answer=question.gold_answer,
            retrieved_frame=retrieved,
            generated_answer=generated_answer,
            category=question.category,
        )
        retrieval_usage = usage_delta(before, after_retrieval)
        result_rows.append(
            {
                **metric_row,
                "category": question.category,
                "evidence_event_ids": ";".join(question.evidence_event_ids),
                "retrieved_names": retrieved_names(retrieved),
                "retrieval_latency_sec": round(retrieval_latency, 4),
                "answer_latency_sec": round(answer_latency, 4),
                **{f"retrieval_{key}": value for key, value in retrieval_usage.items()},
                **{f"answer_{key}": value for key, value in answer_usage.items()},
            }
        )
        metric_rows.append(metric_row)
    return result_rows, metric_rows


def generate_answer(question: str, retrieved: Any, *, question_id: str) -> str:
    """Generate one answer from retrieved memory rows using the configured LOTUS LM."""

    import lotus

    if lotus.settings.lm is None:
        raise RuntimeError("LOTUS LM is not configured before answer generation")

    context = frame_text(retrieved)
    messages = [
        [
            {
                "role": "system",
                "content": (
                    "Answer the benchmark question using only the retrieved memory "
                    "context. If the context is insufficient, answer 'I don't know'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved memory context:\n{context or '(empty)'}\n\n"
                    "Answer with a short factual phrase or sentence."
                ),
            },
        ]
    ]
    with semantic_trace_scope(
        semantic_operator="answer",
        phase="answer",
        question_id=question_id,
    ):
        output = lotus.settings.lm(
            messages,
            show_progress_bar=False,
            progress_bar_desc="Answering",
            max_tokens=ANSWER_MAX_TOKENS,
        )
    outputs = list(getattr(output, "outputs", ()))
    return str(outputs[0]).strip() if outputs else ""


def retrieved_names(retrieved: Any) -> str:
    """Return retrieved topic names as a semicolon-delimited string."""

    if not hasattr(retrieved, "columns") or "name" not in retrieved.columns:
        return ""
    return ";".join(str(value) for value in retrieved["name"].dropna())


def summary_frame(
    *,
    run_mode: str,
    model: str,
    sample_index: int,
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    memory: am.ClaudeMemory,
    question_metrics: Sequence[Mapping[str, Any]],
    question_results: Sequence[Mapping[str, Any]],
    step_metrics: Sequence[Mapping[str, Any]],
    llm_anomaly_rows: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build one-row benchmark summary metrics."""

    topics = memory._runtime._state.get("topics", pd.DataFrame())
    catalog = memory._runtime._state.get("catalog", pd.DataFrame())
    question_summary = summarize_question_metrics(question_metrics)
    step_frame = pd.DataFrame(step_metrics)
    result_frame = pd.DataFrame(question_results)
    ingest_latency = float(step_frame["latency_sec"].sum()) if not step_frame.empty else 0.0
    retrieval_latency = (
        float(result_frame["retrieval_latency_sec"].sum())
        if "retrieval_latency_sec" in result_frame
        else 0.0
    )
    answer_latency = (
        float(result_frame["answer_latency_sec"].sum())
        if "answer_latency_sec" in result_frame
        else 0.0
    )
    row = {
        "run_mode": run_mode,
        "input_rendering": "message_with_event_context",
        "bookkeeping_metadata_excluded_from_semantic_input": True,
        "qa_accuracy_available": run_mode == "answer",
        "official_score_available": False,
        "strict_evidence_recall_available": False,
        "model": model,
        "sample_index": sample_index,
        "events_ingested": len(events),
        "eligible_questions": len(questions),
        "topics_rows": len(topics),
        "catalog_rows": len(catalog),
        "duplicate_topic_name_count": duplicate_name_count(topics),
        "duplicate_topic_name_extra_rows": duplicate_name_extra_rows(topics),
        "latency_sec": round(ingest_latency + retrieval_latency + answer_latency, 4),
        "ingest_latency_sec": round(ingest_latency, 4),
        "retrieval_latency_sec": round(retrieval_latency, 4),
        "answer_latency_sec": round(answer_latency, 4),
        "llm_anomaly_count": "" if llm_anomaly_rows is None else len(llm_anomaly_rows),
        "llm_empty_output_count": (
            ""
            if llm_anomaly_rows is None
            else sum(row.get("issue") == "empty_output" for row in llm_anomaly_rows)
        ),
        **question_summary,
    }
    for field in USAGE_FIELDS:
        ingest_value = int(step_frame[field].sum()) if field in step_frame else 0
        retrieval_field = f"retrieval_{field}"
        answer_field = f"answer_{field}"
        retrieval_value = (
            int(result_frame[retrieval_field].sum())
            if retrieval_field in result_frame
            else 0
        )
        answer_value = (
            int(result_frame[answer_field].sum())
            if answer_field in result_frame
            else 0
        )
        row[field] = ingest_value + retrieval_value + answer_value
    return pd.DataFrame([row])


def write_memory_tables(memory: am.ClaudeMemory, output_dir: Path) -> dict[str, Path]:
    """Write runtime memory state CSVs."""

    state = memory._runtime._state
    return {
        "memory/log": write_csv("log", state.get("log", pd.DataFrame()), output_dir / "memory"),
        "memory/topics": write_csv(
            "topics",
            state.get("topics", pd.DataFrame()),
            output_dir / "memory",
        ),
        "memory/catalog": write_csv(
            "catalog",
            state.get("catalog", pd.DataFrame()),
            output_dir / "memory",
        ),
    }


def write_run_artifacts(
    *,
    output_dir: Path,
    run_mode: str,
    model: str,
    sample_index: int,
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    memory: am.ClaudeMemory | None,
    result_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    step_metrics: Sequence[Mapping[str, Any]],
    trace_dir: Path | None,
    llm_anomaly_rows: Sequence[Mapping[str, Any]] | None,
    ingested_event_ids: Iterable[str],
    include_summary: bool,
) -> dict[str, Path]:
    """Write all artifacts that are available for the current run state."""

    written: dict[str, Path] = {}
    if memory is not None:
        written.update(write_memory_tables(memory, output_dir))
    if result_rows:
        written["retrieval/results"] = write_csv(
            "results",
            pd.DataFrame(result_rows),
            output_dir / "retrieval",
        )
    if metric_rows:
        written["metrics/questions"] = write_csv(
            "questions",
            pd.DataFrame(metric_rows),
            output_dir / "metrics",
        )
    if include_summary and memory is not None:
        written["metrics/summary"] = write_csv(
            "summary",
            summary_frame(
                run_mode=run_mode,
                model=model,
                sample_index=sample_index,
                events=events,
                questions=questions,
                memory=memory,
                question_metrics=metric_rows,
                question_results=result_rows,
                step_metrics=step_metrics,
                llm_anomaly_rows=llm_anomaly_rows,
            ),
            output_dir / "metrics",
        )
    if trace_dir is not None:
        written.update(
            write_trace_diagnostics(
                output_dir=output_dir,
                trace_dir=trace_dir,
                questions=questions,
                ingested_event_ids=ingested_event_ids,
                llm_anomaly_rows=llm_anomaly_rows or [],
            )
        )
        written["trace"] = trace_dir
    return written


def write_trace_diagnostics(
    *,
    output_dir: Path,
    trace_dir: Path,
    questions: Sequence[BenchmarkQuestion],
    ingested_event_ids: Iterable[str],
    llm_anomaly_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Write trace-derived diagnostic CSV artifacts."""

    return {
        "diagnostics/cause_trace": write_csv(
            "cause_trace",
            pd.DataFrame(
                build_cause_trace_rows(
                    questions=questions,
                    ingested_event_ids=set(ingested_event_ids),
                    trace_dir=trace_dir,
                )
            ),
            output_dir / "diagnostics",
        ),
        "diagnostics/llm_anomalies": write_csv(
            "llm_anomalies",
            pd.DataFrame(llm_anomaly_rows, columns=LLM_ANOMALY_COLUMNS),
            output_dir / "diagnostics",
        ),
    }


def write_failure_metadata(
    *,
    output_dir: Path,
    error: BaseException,
    events: Sequence[BenchmarkEvent],
    questions: Sequence[BenchmarkQuestion],
    step_metrics: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    trace_dir: Path | None,
) -> Path:
    """Write one JSON failure summary without swallowing the original error."""

    completed_events = len(step_metrics)
    completed_questions = len(result_rows)
    failed_event_id = ""
    failed_question_id = ""
    failed_phase = "unknown"
    if completed_events < len(events):
        failed_phase = "add"
        failed_event_id = events[completed_events].event_id
    elif completed_questions < len(questions):
        failed_phase = "question"
        failed_question_id = questions[completed_questions].question_id

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    failure_path = diagnostics_dir / "failure.json"
    failure = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "failed_phase": failed_phase,
        "failed_event_id": failed_event_id,
        "failed_question_id": failed_question_id,
        "completed_events": completed_events,
        "total_events": len(events),
        "completed_questions": completed_questions,
        "total_questions": len(questions),
        "output_dir": str(output_dir),
        "trace_dir": "" if trace_dir is None else str(trace_dir),
    }
    failure_path.write_text(
        json.dumps(failure, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return failure_path


def completed_event_ids(step_metrics: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return event ids for successfully completed add steps."""

    return tuple(
        str(row["event_id"])
        for row in step_metrics
        if row.get("phase") == "add" and row.get("event_id")
    )


def main() -> None:
    """Run one LOCOMO benchmark slice and write CSV artifacts."""

    args = parse_args()
    require_environment()
    output_dir = args.output_dir.resolve()
    reset_output_dir(output_dir)

    dataset_path, events, questions = selected_benchmark_data(
        sample_index=args.sample_index,
        row_limit=args.row_limit,
        question_limit=args.question_limit,
    )
    trace_dir = output_dir / "trace" if args.trace else None
    run_mode = "answer" if args.answer else "retrieval_only_diagnostic"

    print("LOCOMO benchmark")
    print(f"dataset: {dataset_path}")
    print(f"sample_index: {args.sample_index}")
    print(f"events: {len(events)}")
    print(f"eligible_questions: {len(questions)}")
    print(f"run_mode: {run_mode}")
    print(f"model: {args.model}")
    print(f"output_dir: {output_dir}")

    written: dict[str, Path] = {
        "input/events": write_csv("events", events_frame(events), output_dir / "input"),
        "input/questions": write_csv(
            "questions",
            questions_frame(questions),
            output_dir / "input",
        ),
    }
    memory: am.ClaudeMemory | None = None
    step_metrics: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    try:
        memory = create_memory(model=args.model, trace_dir=trace_dir)
        run_memory_ingest(memory, events, step_metrics=step_metrics)
        result_rows, metric_rows = run_questions(
            memory,
            questions,
            answer=args.answer,
            result_rows=result_rows,
            metric_rows=metric_rows,
        )
    except Exception as error:
        try:
            llm_anomaly_rows = (
                build_llm_anomaly_rows(trace_dir=trace_dir)
                if trace_dir is not None
                else None
            )
            written.update(
                write_run_artifacts(
                    output_dir=output_dir,
                    run_mode=run_mode,
                    model=args.model,
                    sample_index=args.sample_index,
                    events=events,
                    questions=questions,
                    memory=memory,
                    result_rows=result_rows,
                    metric_rows=metric_rows,
                    step_metrics=step_metrics,
                    trace_dir=trace_dir,
                    llm_anomaly_rows=llm_anomaly_rows,
                    ingested_event_ids=completed_event_ids(step_metrics),
                    include_summary=False,
                )
            )
            written["diagnostics/failure"] = write_failure_metadata(
                output_dir=output_dir,
                error=error,
                events=events,
                questions=questions,
                step_metrics=step_metrics,
                result_rows=result_rows,
                trace_dir=trace_dir,
            )
        except Exception as artifact_error:
            print(f"warning: failed to write partial artifacts: {artifact_error}")
        print("\nwrote partial benchmark artifacts before failure:")
        for name, path_value in written.items():
            print(f"- {name}: {path_value}")
        raise

    llm_anomaly_rows = (
        build_llm_anomaly_rows(trace_dir=trace_dir)
        if trace_dir is not None
        else None
    )
    written.update(
        write_run_artifacts(
            output_dir=output_dir,
            run_mode=run_mode,
            model=args.model,
            sample_index=args.sample_index,
            events=events,
            questions=questions,
            memory=memory,
            result_rows=result_rows,
            metric_rows=metric_rows,
            step_metrics=step_metrics,
            trace_dir=trace_dir,
            llm_anomaly_rows=llm_anomaly_rows,
            ingested_event_ids=completed_event_ids(step_metrics),
            include_summary=True,
        )
    )

    print("\nwrote benchmark artifacts:")
    for name, path_value in written.items():
        print(f"- {name}: {path_value}")


if __name__ == "__main__":
    main()
