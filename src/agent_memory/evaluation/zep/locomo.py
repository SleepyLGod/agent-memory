"""Formal Zep LOCOMO benchmark orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
from time import perf_counter
from typing import Any
from urllib.request import urlopen
from uuid import uuid4

import pandas as pd

from agent_memory.evaluation.locomo import load_locomo_sample
from agent_memory.evaluation.types import BenchmarkEvent, BenchmarkQuestion
from agent_memory.evaluation.zep.answering import (
    ANSWER_PROMPT_DIGEST,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    ZEP_JUDGE_PROMPT_DIGEST,
    AnswerRecord,
    generate_answer,
    generate_zep_judge,
)
from agent_memory.evaluation.zep.scoring import (
    OfficialGrade,
    ZepJudgeGrade,
    score_official_answer,
)
from agent_memory.evaluation.zep.artifacts import ArtifactStore, PricingSnapshot
from agent_memory.policy.retrieval import RetrievalResult
from agent_memory.tracing.semantic import semantic_trace_scope

_LOCOMO_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})\s+"
    r"(?P<meridiem>am|pm)\s+on\s+(?P<day>\d{1,2})\s+"
    r"(?P<month>[A-Za-z]+),\s*(?P<year>\d{4})$",
    re.IGNORECASE,
)
_ENGLISH_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}
PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    f"{LOCOMO_COMMIT}/data/locomo10.json"
)
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_START_ROW = 26
DEFAULT_ROW_LIMIT = 3
DEFAULT_QUESTION_START = 4
DEFAULT_QUESTION_LIMIT = 1
NEO4J_IMAGE = "neo4j:5.26.2"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
USAGE_FIELDS = (
    "physical_prompt_tokens",
    "physical_completion_tokens",
    "physical_total_tokens",
    "virtual_prompt_tokens",
    "virtual_completion_tokens",
    "virtual_total_tokens",
    "cache_hits",
)


@dataclass(frozen=True)
class ZepLocomoRunConfig:
    """One isolated preliminary Zep LOCOMO run."""

    dataset_path: Path
    output_dir: Path
    sample_index: int = 0
    start_row: int = DEFAULT_START_ROW
    row_limit: int | None = DEFAULT_ROW_LIMIT
    question_start: int = DEFAULT_QUESTION_START
    question_limit: int | None = DEFAULT_QUESTION_LIMIT
    question_numbers: tuple[int, ...] | None = None
    include_adversarial: bool = True
    model: str = DEFAULT_MODEL
    grouped_agg_rule: str = "rule-re-group"
    namespace: str | None = None


@dataclass(frozen=True)
class RetrievalRecord:
    """One two-channel storage-backed retrieval result."""

    question_id: str
    category: int
    query: str
    entities: tuple[Mapping[str, Any], ...]
    facts: tuple[Mapping[str, Any], ...]
    channel_metrics: Mapping[str, Mapping[str, Any]]
    latency_ms: float

    @classmethod
    def from_result(
        cls,
        question: BenchmarkQuestion,
        result: RetrievalResult,
        *,
        latency_ms: float,
    ) -> "RetrievalRecord":
        """Build an inspectable retrieval artifact."""

        return cls(
            question_id=question.question_id,
            category=int(question.category),
            query=result.query,
            entities=tuple(_channel_records(result, "entities")),
            facts=tuple(_channel_records(result, "facts")),
            channel_metrics={
                name: dict(metrics) for name, metrics in result.metrics.items()
            },
            latency_ms=round(float(latency_ms), 3),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe artifact payload."""

        return {
            "question_id": self.question_id,
            "category": self.category,
            "query": self.query,
            "entities": [dict(row) for row in self.entities],
            "facts": [dict(row) for row in self.facts],
            "entity_ids": [str(row.get("record_id")) for row in self.entities],
            "fact_ids": [str(row.get("record_id")) for row in self.facts],
            "channel_metrics": {
                name: dict(metrics) for name, metrics in self.channel_metrics.items()
            },
            "latency_ms": self.latency_ms,
        }

    def metric_row(self) -> dict[str, Any]:
        """Return compact latency, result-count, and BFS evidence."""

        entity_metrics = self.channel_metrics.get("entities", {})
        fact_metrics = self.channel_metrics.get("facts", {})
        return {
            "question_id": self.question_id,
            "category": self.category,
            "latency_ms": self.latency_ms,
            "entity_count": len(self.entities),
            "fact_count": len(self.facts),
            "entity_latency_ms": entity_metrics.get("latency_ms", ""),
            "fact_latency_ms": fact_metrics.get("latency_ms", ""),
            "bfs_origins": ";".join(
                str(value) for value in fact_metrics.get("bfs_origins", ())
            ),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Question-level retrieval, answer, and dual-grade artifacts."""

    retrievals: tuple[RetrievalRecord, ...]
    answers: tuple[AnswerRecord, ...]
    official_grades: tuple[OfficialGrade, ...]
    zep_judge_grades: tuple[ZepJudgeGrade, ...]


def select_event_range(
    events: Sequence[BenchmarkEvent],
    *,
    start_row: int,
    row_limit: int | None,
) -> tuple[BenchmarkEvent, ...]:
    """Return a one-based contiguous event range."""

    if start_row < 1:
        raise ValueError("start_row must be one-based and at least 1")
    if row_limit is not None and row_limit < 1:
        raise ValueError("row_limit must be at least 1")
    start = start_row - 1
    selected = tuple(events[start:] if row_limit is None else events[start : start + row_limit])
    if row_limit is not None and len(selected) != row_limit:
        raise ValueError(
            f"Requested {row_limit} events from row {start_row}, found {len(selected)}"
        )
    if not selected:
        raise ValueError(f"No LOCOMO events found from row {start_row}")
    return selected


def select_questions_for_events(
    questions: Sequence[BenchmarkQuestion],
    events: Sequence[BenchmarkEvent],
    *,
    start_question: int = 1,
    question_limit: int | None = None,
    question_numbers: Sequence[int] | None = None,
    include_adversarial: bool = True,
) -> tuple[BenchmarkQuestion, ...]:
    """Select original-numbered questions and require all cited evidence."""

    if start_question < 1:
        raise ValueError("start_question must be one-based and at least 1")
    if question_limit is not None and question_limit < 1:
        raise ValueError("question_limit must be at least 1")
    normalized_numbers = tuple(question_numbers or ())
    if any(number < 1 for number in normalized_numbers):
        raise ValueError("question_numbers must contain positive one-based numbers")
    if len(set(normalized_numbers)) != len(normalized_numbers):
        raise ValueError("question_numbers must not contain duplicates")
    if normalized_numbers and (start_question != 1 or question_limit is not None):
        raise ValueError(
            "question_numbers cannot be combined with start_question or question_limit"
        )

    candidates = tuple(
        question
        for question in questions
        if include_adversarial or int(question.category) != 5
    )
    if normalized_numbers:
        by_number = {
            int(question.metadata.get("question_number", 0)): question
            for question in candidates
        }
        missing = [number for number in normalized_numbers if number not in by_number]
        if missing:
            raise ValueError(f"LOCOMO questions are unavailable: {missing}")
        selected = tuple(by_number[number] for number in normalized_numbers)
    else:
        ranged = tuple(
            question
            for question in candidates
            if int(question.metadata.get("question_number", 0)) >= start_question
        )
        selected = ranged if question_limit is None else ranged[:question_limit]
        if question_limit is not None and len(selected) != question_limit:
            raise ValueError(
                f"Requested {question_limit} questions from {start_question}, "
                f"found {len(selected)}"
            )

    ingested_ids = {event.event_id for event in events}
    for question in selected:
        missing_evidence = [
            event_id
            for event_id in question.evidence_event_ids
            if event_id not in ingested_ids
        ]
        if missing_evidence:
            number = int(question.metadata.get("question_number", 0))
            raise ValueError(
                f"LOCOMO question {number} requires un-ingested evidence: "
                + ", ".join(missing_evidence)
            )
    return selected


def validate_pinned_dataset(path: Path) -> str:
    """Require the exact LOCOMO bytes shared with the native baseline."""

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != LOCOMO_SHA256:
        raise ValueError(
            f"LOCOMO SHA-256 mismatch: expected {LOCOMO_SHA256}, got {actual}"
        )
    return actual


def build_manifest(
    config: ZepLocomoRunConfig,
    *,
    namespace: str,
    dataset_sha256: str,
    policy_input_sha256: str,
    agent_memory_commit: str,
    agent_memory_dirty: bool,
) -> dict[str, Any]:
    """Build the fixed preliminary benchmark contract."""

    pricing = PricingSnapshot.deepseek_2026_07_17()
    return {
        "benchmark_contract_version": "agent-memory-zep-locomo-preliminary-v1",
        "system": "agent-memory-zep",
        "agent_memory_commit": agent_memory_commit,
        "agent_memory_dirty": agent_memory_dirty,
        "locomo_commit": LOCOMO_COMMIT,
        "locomo_sha256": dataset_sha256,
        "policy_input_sha256": policy_input_sha256,
        "sample_index": config.sample_index,
        "start_row": config.start_row,
        "row_limit": config.row_limit,
        "question_start": config.question_start,
        "question_limit": config.question_limit,
        "question_numbers": list(config.question_numbers or ()),
        "include_adversarial": config.include_adversarial,
        "grouped_agg_rule": config.grouped_agg_rule,
        "namespace": namespace,
        "model": {
            "id": config.model,
            "temperature": GENERATION_TEMPERATURE,
            "max_tokens": GENERATION_MAX_TOKENS,
            "thinking": {"type": "disabled"},
            "application_cache": "disabled",
        },
        "storage": {
            "neo4j_image": NEO4J_IMAGE,
            "isolated_namespace": True,
        },
        "retrieval": {
            "entity_methods": ["bm25", "cosine_similarity"],
            "entity_reranker": "rrf",
            "entity_limit": 20,
            "fact_methods": ["bm25", "cosine_similarity", "bfs"],
            "fact_reranker": "cross_encoder",
            "fact_limit": 20,
            "bfs_max_depth": 3,
            "generative_llm": False,
        },
        "models": {
            "embedding": EMBEDDING_MODEL,
            "reranker": RERANKER_MODEL,
            "reranker_device": "cpu",
        },
        "scoring": {
            "official_categories": (
                [1, 2, 3, 4, 5]
                if config.include_adversarial
                else [1, 2, 3, 4]
            ),
            "zep_judge_categories": [1, 2, 3, 4],
            "shared_answer_stream": True,
        },
        "prompts": {
            "answer_sha256": ANSWER_PROMPT_DIGEST,
            "zep_judge_sha256": ZEP_JUDGE_PROMPT_DIGEST,
        },
        "pricing": pricing.to_dict(),
    }


def event_to_zep_log_row(event: BenchmarkEvent) -> dict[str, str]:
    """Map one normalized LOCOMO event to the ZepMemory source schema."""

    sample_index = event.metadata.get("sample_index")
    session_number = event.metadata.get("session_number")
    if not isinstance(sample_index, int) or isinstance(sample_index, bool):
        raise ValueError("normalized LOCOMO event requires integer sample_index metadata")
    if not isinstance(session_number, int) or isinstance(session_number, bool):
        raise ValueError("normalized LOCOMO event requires integer session_number metadata")
    content = f"{event.speaker}: {event.text}"
    caption = event.metadata.get("blip_caption")
    if isinstance(caption, str) and caption.strip():
        content += f"\n(description of attached image: {caption.strip()})"
    return {
        "content": content,
        "role": event.speaker,
        "speaker": event.speaker,
        "reference_time": _reference_time(event.timestamp),
        "source_description": (
            f"LOCOMO sample {sample_index} session {session_number}"
        ),
    }


def policy_input_fingerprint(events: Sequence[BenchmarkEvent]) -> str:
    """Hash the exact Zep source rows used for insertion."""

    payload = json.dumps(
        [event_to_zep_log_row(event) for event in events],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_questions(
    memory: Any,
    questions: Sequence[BenchmarkQuestion],
    *,
    answerer: Callable[[BenchmarkQuestion, RetrievalResult], AnswerRecord] = generate_answer,
    zep_judge: Callable[[BenchmarkQuestion, str], ZepJudgeGrade] = generate_zep_judge,
    on_question: Callable[
        [RetrievalRecord, AnswerRecord, OfficialGrade, ZepJudgeGrade | None],
        None,
    ]
    | None = None,
) -> EvaluationResult:
    """Retrieve once, answer once, then grade the same answer twice."""

    retrievals: list[RetrievalRecord] = []
    answers: list[AnswerRecord] = []
    official_grades: list[OfficialGrade] = []
    judge_grades: list[ZepJudgeGrade] = []
    for question in questions:
        before_usage = _usage_snapshot()
        started = perf_counter()
        retrieval = memory.query(question.question)
        latency_ms = (perf_counter() - started) * 1000
        after_usage = _usage_snapshot()
        if after_usage != before_usage:
            raise RuntimeError("Zep retrieval unexpectedly invoked the semantic LLM")
        if not isinstance(retrieval, RetrievalResult):
            raise TypeError("ZepMemory.query must return RetrievalResult")
        retrieval_record = RetrievalRecord.from_result(
            question,
            retrieval,
            latency_ms=latency_ms,
        )
        answer = answerer(question, retrieval)
        if answer.question_id != question.question_id:
            raise ValueError("answerer returned a mismatched question_id")
        retrievals.append(retrieval_record)
        answers.append(answer)
        official_grade = score_official_answer(question, answer.answer)
        official_grades.append(official_grade)
        judge_grade: ZepJudgeGrade | None = None
        if int(question.category) in {1, 2, 3, 4}:
            judge_grade = zep_judge(question, answer.answer)
            if judge_grade.question_id != question.question_id:
                raise ValueError("Zep judge returned a mismatched question_id")
            judge_grades.append(judge_grade)
        if on_question is not None:
            on_question(retrieval_record, answer, official_grade, judge_grade)
    return EvaluationResult(
        retrievals=tuple(retrievals),
        answers=tuple(answers),
        official_grades=tuple(official_grades),
        zep_judge_grades=tuple(judge_grades),
    )


def run_ingestion(
    memory: Any,
    events: Sequence[BenchmarkEvent],
    *,
    on_step: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Insert normalized events sequentially and measure each committed add."""

    rows: list[dict[str, Any]] = []
    for event in events:
        before = _usage_snapshot()
        started = perf_counter()
        with semantic_trace_scope(
            run_kind="zep-locomo",
            phase="insertion",
            event_id=event.event_id,
            row_number=event.metadata.get("row_number"),
        ):
            memory.add(event_to_zep_log_row(event))
        state = memory._runtime._state
        metric = {
            "event_id": event.event_id,
            "row_number": event.metadata.get("row_number", ""),
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "episodes_rows": len(state.get("episodes", ())),
            "entities_rows": len(state.get("entities", ())),
            "facts_rows": len(state.get("facts", ())),
            **_usage_delta(before, _usage_snapshot()),
        }
        rows.append(metric)
        if on_step is not None:
            on_step(metric)
    return rows


def require_environment() -> None:
    """Load local secrets and fail before opening Neo4j when required values are absent."""

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    missing = [
        name
        for name in (
            "DEEPSEEK_API_KEY",
            "AGENT_MEMORY_NEO4J_URI",
            "AGENT_MEMORY_NEO4J_PASSWORD",
        )
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            "Zep LOCOMO benchmark requires environment variables: "
            + ", ".join(missing)
        )


def ensure_pinned_dataset(path: Path) -> Path:
    """Download the pinned LOCOMO dataset once and verify its bytes."""

    if path.exists():
        validate_pinned_dataset(path)
        return path
    with urlopen(LOCOMO_URL, timeout=60) as response:
        content = response.read()
    actual = hashlib.sha256(content).hexdigest()
    if actual != LOCOMO_SHA256:
        raise ValueError(
            f"LOCOMO SHA-256 mismatch: expected {LOCOMO_SHA256}, got {actual}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def create_runtime(
    *,
    namespace: str,
    model: str,
    grouped_agg_rule: str,
    trace_dir: Path,
) -> tuple[Any, Any, Any]:
    """Create the real Graphiti-compatible storage, adapter, and ZepMemory."""

    import agent_memory as am
    from agent_memory.adapters.lotus import LotusAdapter
    from agent_memory.adapters.lotus.context import LotusExecutionConfig
    from agent_memory.memories.zep.storage import (
        GRAPHITI_BGE_M3,
        GRAPHITI_NEO4J_SCHEMA,
        GRAPHITI_NEO4J_STATEMENTS,
    )
    from agent_memory.storage import StorageDeployment
    from agent_memory.planner import DifferentialRules, PolicyDifferentiator
    from agent_memory.runtime import MemoryRuntime
    from agent_memory.storage.neo4j import (
        Neo4jConnector,
        SentenceTransformerCrossEncoderProvider,
        SentenceTransformerEmbeddingProvider,
    )

    connector = Neo4jConnector(
        uri=os.environ["AGENT_MEMORY_NEO4J_URI"],
        auth=(
            os.getenv("AGENT_MEMORY_NEO4J_USER", "neo4j"),
            os.environ["AGENT_MEMORY_NEO4J_PASSWORD"],
        ),
        database=os.getenv("AGENT_MEMORY_NEO4J_DATABASE", "neo4j"),
        embedding_provider=SentenceTransformerEmbeddingProvider(GRAPHITI_BGE_M3),
        reranker_provider=SentenceTransformerCrossEncoderProvider(),
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    try:
        storage = StorageDeployment(
            connector=connector,
            statements=GRAPHITI_NEO4J_STATEMENTS,
            namespace=namespace,
        )
        adapter = LotusAdapter(
            model=model,
            config=LotusExecutionConfig(
                semantic_trace_dir=trace_dir,
                lm_model_kwargs={
                    "extra_body": {"thinking": {"type": "disabled"}}
                },
                lm_enable_cache=False,
            ),
        )
        policy = PolicyDifferentiator(
            rules=DifferentialRules(grouped_agg_rule=grouped_agg_rule)
        ).differentiate(
            am.ZepMemory.spec(),
            statements=storage.statements,
        )
        memory = am.ZepMemory(adapter=adapter)
        memory._runtime = MemoryRuntime(
            policy,
            adapter=adapter,
            storage=storage,
        )
    except Exception:
        connector.close()
        raise
    return storage, memory, adapter


def checkpoint_round_trip(
    memory: Any,
    adapter: Any,
    storage: Any,
    question: BenchmarkQuestion,
) -> tuple[Any, bytes, dict[str, Any], dict[str, Any]]:
    """Restore one checkpoint and prove retrieval identity/order is unchanged."""

    import agent_memory as am
    from agent_memory.runtime import MemoryRuntime

    before = _query_without_semantic_usage(memory, question.question)
    snapshot = memory._runtime.snapshot_state()
    state = pickle.dumps(snapshot)
    restored = am.ZepMemory(adapter=adapter)
    restored._runtime = MemoryRuntime(
        memory._runtime.policy,
        adapter=adapter,
        storage=storage,
    )
    restored._runtime.restore_state(pickle.loads(state))
    for name in ("episodes", "entities", "facts"):
        pd.testing.assert_frame_equal(
            memory._runtime._state[name],
            restored._runtime._state[name],
        )
    after = _query_without_semantic_usage(restored, question.question)
    before_ids = _retrieval_ids(before)
    after_ids = _retrieval_ids(after)
    if before_ids != after_ids:
        raise RuntimeError(
            "retrieval result IDs or order changed after checkpoint restore"
        )
    metadata = {
        "schema_version": snapshot.get("schema_version"),
        "policy_fingerprint": restored._runtime.policy.fingerprint,
        "storage_bound": True,
        "storage_commit": snapshot.get("storage_commit"),
        "round_trip_verified": True,
    }
    validation = {
        "question_id": question.question_id,
        "query": question.question,
        "before": before_ids,
        "after": after_ids,
        "result_order_verified": True,
        "semantic_llm_usage_changed": False,
    }
    return restored, state, metadata, validation


def run_zep_locomo(config: ZepLocomoRunConfig) -> Path:
    """Run insertion, retrieval, one shared answer stream, and both scorers."""

    require_environment()
    dataset_path = ensure_pinned_dataset(config.dataset_path)
    dataset_sha256 = validate_pinned_dataset(dataset_path)
    sample = load_locomo_sample(dataset_path, sample_index=config.sample_index)
    events = select_event_range(
        sample.events,
        start_row=config.start_row,
        row_limit=config.row_limit,
    )
    questions = select_questions_for_events(
        sample.questions,
        events,
        start_question=config.question_start,
        question_limit=config.question_limit,
        question_numbers=config.question_numbers,
        include_adversarial=config.include_adversarial,
    )
    if not questions:
        raise ValueError("No eligible LOCOMO questions for the selected event range")

    namespace = config.namespace or f"zep-locomo-{uuid4()}"
    commit, dirty = _git_state()
    store = ArtifactStore.create(config.output_dir)
    store.write_manifest(
        build_manifest(
            config,
            namespace=namespace,
            dataset_sha256=dataset_sha256,
            policy_input_sha256=policy_input_fingerprint(events),
            agent_memory_commit=commit,
            agent_memory_dirty=dirty,
        )
    )
    store.write_inputs(events, questions)
    store.write_status(status="running", phase="setup")

    storage: Any | None = None
    pricing = PricingSnapshot.deepseek_2026_07_17()
    ingestion_metrics: list[dict[str, Any]] = []
    evaluation = EvaluationResult((), (), (), ())
    phase = "setup"
    try:
        storage, memory, adapter = create_runtime(
            namespace=namespace,
            model=config.model,
            grouped_agg_rule=config.grouped_agg_rule,
            trace_dir=store.trace_dir,
        )
        phase = "insertion"
        store.write_status(status="running", phase=phase)

        def record_ingestion(row: Mapping[str, Any]) -> None:
            ingestion_metrics.append(dict(row))
            store.write_ingestion_metrics(ingestion_metrics)

        run_ingestion(memory, events, on_step=record_ingestion)
        phase = "checkpoint"
        store.write_status(status="running", phase=phase)
        restored, state, metadata, validation = checkpoint_round_trip(
            memory,
            adapter,
            storage,
            questions[0],
        )
        store.write_checkpoint(
            state=state,
            metadata=metadata,
            retrieval_validation=validation,
        )
        phase = "questions"
        store.write_status(status="running", phase=phase)
        completed_retrievals: list[RetrievalRecord] = []
        completed_answers: list[AnswerRecord] = []
        completed_official: list[OfficialGrade] = []
        completed_judge: list[ZepJudgeGrade] = []

        def record_question(
            retrieval: RetrievalRecord,
            answer: AnswerRecord,
            official: OfficialGrade,
            judge: ZepJudgeGrade | None,
        ) -> None:
            nonlocal evaluation
            completed_retrievals.append(retrieval)
            completed_answers.append(answer)
            completed_official.append(official)
            if judge is not None:
                completed_judge.append(judge)
            evaluation = EvaluationResult(
                tuple(completed_retrievals),
                tuple(completed_answers),
                tuple(completed_official),
                tuple(completed_judge),
            )
            _write_evaluation_artifacts(store, evaluation)

        evaluation = evaluate_questions(
            restored,
            questions,
            on_question=record_question,
        )
        _write_evaluation_artifacts(store, evaluation)
    except Exception as error:
        store.write_ingestion_metrics(ingestion_metrics)
        _write_evaluation_artifacts(store, evaluation)
        store.write_failure(phase=phase, error=error)
        store.write_status(status="failed", phase=phase, error=error)
        try:
            store.write_metrics(pricing=pricing)
        except Exception as metrics_error:
            raise ExceptionGroup(
                "benchmark execution and metrics derivation both failed",
                [error, metrics_error],
            ) from error
        raise
    else:
        store.write_metrics(pricing=pricing)
        store.write_status(status="completed", phase="complete")
    finally:
        if storage is not None:
            storage.connector.close()
    return config.output_dir


def _write_evaluation_artifacts(
    store: ArtifactStore,
    evaluation: EvaluationResult,
) -> None:
    store.write_retrievals(evaluation.retrievals)
    store.write_retrieval_metrics(
        [record.metric_row() for record in evaluation.retrievals]
    )
    store.write_answers(evaluation.answers)
    store.write_grades(
        evaluation.official_grades,
        evaluation.zep_judge_grades,
    )


def _query_without_semantic_usage(memory: Any, query: str) -> RetrievalResult:
    before = _usage_snapshot()
    result = memory.query(query)
    after = _usage_snapshot()
    if after != before:
        raise RuntimeError("Zep retrieval unexpectedly invoked the semantic LLM")
    if not isinstance(result, RetrievalResult):
        raise TypeError("ZepMemory.query must return RetrievalResult")
    return result


def _retrieval_ids(result: RetrievalResult) -> dict[str, list[str]]:
    return {
        name: (
            [str(value) for value in frame["record_id"]]
            if "record_id" in frame.columns
            else []
        )
        for name, frame in result.channels.items()
    }


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _reference_time(value: str) -> str:
    try:
        match = _LOCOMO_TIMESTAMP_PATTERN.fullmatch(value.strip())
        if match is None:
            raise ValueError("timestamp does not match the LOCOMO format")
        month = _ENGLISH_MONTHS[match.group("month").lower()]
        hour = int(match.group("hour"))
        if not 1 <= hour <= 12:
            raise ValueError("hour must be between 1 and 12")
        hour = hour % 12 + (12 if match.group("meridiem").lower() == "pm" else 0)
        parsed = datetime(
            int(match.group("year")),
            month,
            int(match.group("day")),
            hour,
            int(match.group("minute")),
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid LOCOMO local datetime: {value}") from error
    return parsed.isoformat(timespec="seconds")


def _channel_records(
    result: RetrievalResult,
    channel: str,
) -> list[dict[str, Any]]:
    frame = result.channels.get(channel)
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def _usage_snapshot() -> dict[str, int]:
    """Return current LOTUS physical, virtual, and local-cache counters."""

    result = {field: 0 for field in USAGE_FIELDS}
    try:
        import lotus
    except ImportError:
        return result
    lm = lotus.settings.lm
    stats = getattr(lm, "stats", None) if lm is not None else None
    if stats is None:
        return result
    result.update(
        {
            "physical_prompt_tokens": int(stats.physical_usage.prompt_tokens),
            "physical_completion_tokens": int(stats.physical_usage.completion_tokens),
            "physical_total_tokens": int(stats.physical_usage.total_tokens),
            "virtual_prompt_tokens": int(stats.virtual_usage.prompt_tokens),
            "virtual_completion_tokens": int(stats.virtual_usage.completion_tokens),
            "virtual_total_tokens": int(stats.virtual_usage.total_tokens),
            "cache_hits": int(stats.cache_hits),
        }
    )
    return result


def _usage_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    return {
        field: max(0, int(after.get(field, 0)) - int(before.get(field, 0)))
        for field in USAGE_FIELDS
    }


__all__ = [
    "LOCOMO_SHA256",
    "EvaluationResult",
    "RetrievalRecord",
    "ZepLocomoRunConfig",
    "build_manifest",
    "evaluate_questions",
    "event_to_zep_log_row",
    "policy_input_fingerprint",
    "run_zep_locomo",
    "run_ingestion",
    "select_event_range",
    "select_questions_for_events",
    "validate_pinned_dataset",
]
