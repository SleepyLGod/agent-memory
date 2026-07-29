"""Pinned MemoryAgentBench parquet loading and normalization."""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.request import urlopen

from agent_memory.evaluation.bundle import BenchmarkBundle
from agent_memory.evaluation.types import BenchmarkCase, BenchmarkEvent, BenchmarkQuestion

from .infbench_prompts import (
    INF_BENCH_FLUENCY_PROMPT,
    INF_BENCH_PRECISION_PROMPT,
    INF_BENCH_RECALL_PROMPT,
)
from .tasks import MEMORY_AGENT_TASKS

MEMORY_AGENT_BENCH_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
MEMORY_AGENT_BENCH_SPLITS = {
    "Accurate_Retrieval": {
        "sha256": "56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1",
        "rows": 22,
    },
    "Conflict_Resolution": {
        "sha256": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45",
        "rows": 8,
    },
    "Long_Range_Understanding": {
        "sha256": "5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539",
        "rows": 110,
    },
    "Test_Time_Learning": {
        "sha256": "5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0",
        "rows": 6,
    },
}
MEMORY_AGENT_BENCH_SHA256 = "badf304c8e66ab531f3f96420d52164f8d18be6f010c715618c1107be17ee284"
MEMORY_AGENT_BENCH_MOVIE_MAPPING_SHA256 = (
    "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5"
)
MEMORY_AGENT_BENCH_SCORER_COMMIT = "455306dcabc3842526eb83cd4e225e5d486c5c5d"
SMOKE_SOURCES = (
    "eventqa_65536",
    "icl_banking77_5900shot_balance",
    "detective_qa",
    "factconsolidation_sh_6k",
)


def _split_url(split: str) -> str:
    return (
        "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/"
        f"{MEMORY_AGENT_BENCH_REVISION}/data/{split}-00000-of-00001.parquet"
    )


def _movie_mapping_url() -> str:
    return (
        "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/"
        f"{MEMORY_AGENT_BENCH_REVISION}/entity2id.json"
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_memory_agent_bench(directory: Path) -> Mapping[str, Path]:
    """Download all four pinned split files and verify each LFS hash."""

    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, contract in MEMORY_AGENT_BENCH_SPLITS.items():
        destination = directory / f"{split}.parquet"
        if not destination.exists():
            partial = destination.with_suffix(".parquet.partial")
            with urlopen(_split_url(split)) as response, partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            partial.replace(destination)
        actual = _sha256_file(destination)
        if actual != contract["sha256"]:
            raise ValueError(
                f"MemoryAgentBench {split} checksum mismatch: expected "
                f"{contract['sha256']}, got {actual}"
            )
        paths[split] = destination
    return paths


def download_movie_entity_mapping(directory: Path) -> Path:
    """Download and verify the official ReDial movie identity mapping."""

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "entity2id.json"
    if not destination.exists():
        partial = destination.with_suffix(".json.partial")
        with urlopen(_movie_mapping_url()) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        partial.replace(destination)
    actual = _sha256_file(destination)
    if actual != MEMORY_AGENT_BENCH_MOVIE_MAPPING_SHA256:
        raise ValueError(
            "MemoryAgentBench movie mapping checksum mismatch: expected "
            f"{MEMORY_AGENT_BENCH_MOVIE_MAPPING_SHA256}, got {actual}"
        )
    return destination


def load_movie_entity_mapping(path: Path) -> Mapping[str, int]:
    """Load a checksum-verified official movie identity mapping."""

    actual = _sha256_file(path)
    if actual != MEMORY_AGENT_BENCH_MOVIE_MAPPING_SHA256:
        raise ValueError("MemoryAgentBench movie mapping checksum mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("MemoryAgentBench movie mapping must be a JSON object")
    if not all(
        isinstance(key, str) and isinstance(item, int)
        for key, item in value.items()
    ):
        raise ValueError("MemoryAgentBench movie mapping has invalid entries")
    return value


def chunk_text_into_sentences(
    text: str,
    *,
    model_name: str = "gpt-4o-mini",
    chunk_size: int = 4096,
    nltk_data_dir: Path | None = None,
    sentence_tokenizer: Callable[[str], Sequence[str]] | None = None,
    token_encoder: Any | None = None,
) -> tuple[str, ...]:
    """Reproduce the official sentence-boundary 4096-token chunker."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    original_nltk_data_path: list[str] | None = None
    nltk_module: Any | None = None
    if sentence_tokenizer is None:
        import nltk

        nltk_module = nltk
        sentence_tokenizer = nltk.sent_tokenize
        if nltk_data_dir is not None:
            original_nltk_data_path = list(nltk.data.path)
            nltk.data.path.insert(0, str(nltk_data_dir.resolve()))
    if token_encoder is None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("install agent-memory[benchmarks] for MAB chunking") from exc
        try:
            token_encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            token_encoder = tiktoken.encoding_for_model("gpt-4o-mini")

    try:
        sentences = sentence_tokenizer(text)
    except LookupError as exc:
        raise RuntimeError(
            "MemoryAgentBench chunking requires the official NLTK punkt_tab data; "
            "download it into --nltk-data-dir"
        ) from exc
    finally:
        if original_nltk_data_path is not None and nltk_module is not None:
            nltk_module.data.path[:] = original_nltk_data_path
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(
            token_encoder.encode(sentence, allowed_special={"<|endoftext|>"})
        )
        if current_tokens + sentence_tokens > chunk_size:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return tuple(chunks)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _json_value(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_value(value.item())
        except ValueError:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _json_value(value.tolist())
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _normalize_row(
    split: str,
    row: Mapping[str, Any],
    source_row_index: int,
    source_occurrence: int,
    *,
    max_questions: int | None,
    chunker: Callable[[str], Sequence[str]],
) -> BenchmarkCase:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("MemoryAgentBench metadata must be a JSON object")
    source = metadata.get("source")
    if not isinstance(source, str):
        raise ValueError("MemoryAgentBench metadata source must be a string")
    task = MEMORY_AGENT_TASKS.get(source)
    if task is None:
        raise ValueError(f"unrecognized MemoryAgentBench source {source!r}")
    if task.split != split:
        raise ValueError(f"source {source!r} is stored in the wrong split")
    context = row.get("context")
    if not isinstance(context, str) or not context:
        raise ValueError("MemoryAgentBench context must be a non-empty string")

    case_id = f"mab:{split}:{source}:{source_occurrence}"
    chunks = tuple(chunker(context))
    if not chunks or any(not isinstance(chunk, str) or not chunk for chunk in chunks):
        raise ValueError(f"MemoryAgentBench case {case_id!r} produced an empty chunk")
    epoch = datetime(2000, 1, 1)
    events: list[BenchmarkEvent] = []
    for chunk_index, chunk in enumerate(chunks):
        timestamp = (epoch + timedelta(seconds=chunk_index)).isoformat()
        events.append(
            BenchmarkEvent(
                sample_id=case_id,
                event_id=f"{case_id}:chunk:{chunk_index}",
                speaker="user",
                text=task.format_event(chunk, timestamp),
                session_id=case_id,
                timestamp=timestamp,
                metadata={"chunk_index": chunk_index, "source": source},
            )
        )

    raw_questions = _as_list(row.get("questions"))
    raw_answers = _as_list(row.get("answers"))
    if len(raw_questions) != len(raw_answers):
        raise ValueError(f"MemoryAgentBench case {case_id!r} has misaligned QA lists")
    qa_pair_ids = _as_list(metadata.get("qa_pair_ids"))
    if qa_pair_ids and len(qa_pair_ids) != len(raw_questions):
        raise ValueError(f"MemoryAgentBench case {case_id!r} has misaligned qa_pair_ids")
    question_dates = _as_list(metadata.get("question_dates"))
    question_types = _as_list(metadata.get("question_types"))
    question_ids = _as_list(metadata.get("question_ids"))
    keypoints = _as_list(metadata.get("keypoints"))

    limit = len(raw_questions) if max_questions is None else min(max_questions, len(raw_questions))
    questions: list[BenchmarkQuestion] = []
    for question_index in range(limit):
        raw_question = raw_questions[question_index]
        if not isinstance(raw_question, str) or not raw_question:
            raise ValueError("MemoryAgentBench questions must be non-empty strings")
        question_id = (
            str(qa_pair_ids[question_index])
            if qa_pair_ids
            else f"{case_id}:question:{question_index}"
        )
        question_metadata: dict[str, Any] = {
            "source": source,
            "split": split,
            "question_index": question_index,
            "scorer": task.scorer,
        }
        if question_dates:
            question_metadata["question_date"] = _json_value(
                question_dates[question_index]
            )
        if question_types:
            question_metadata["question_type"] = _json_value(
                question_types[question_index]
            )
        if question_ids:
            question_metadata["official_question_id"] = _json_value(
                question_ids[question_index]
            )
        if keypoints:
            question_metadata["keypoints"] = _json_value(keypoints)
        questions.append(
            BenchmarkQuestion(
                question_id=question_id,
                sample_id=case_id,
                question=raw_question,
                gold_answer=_json_value(raw_answers[question_index]),
                evidence_event_ids=(),
                category=task.competence,
                metadata=question_metadata,
            )
        )

    return BenchmarkCase(
        case_id=case_id,
        task_id=task.contract_id,
        events=tuple(events),
        questions=tuple(questions),
        metadata={
            "source": source,
            "split": split,
            "source_row_index": source_row_index,
            "source_occurrence": source_occurrence,
        },
    )


def normalize_memory_agent_bench(
    split_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    sources: Sequence[str] | None = None,
    max_cases_per_source: int | None = None,
    max_questions_per_case: int | None = None,
    run_mode: str | None = None,
    chunker: Callable[[str], Sequence[str]] = chunk_text_into_sentences,
) -> BenchmarkBundle:
    """Normalize all selected official sources into inject-once/query-many cases."""

    selected = set(sources) if sources is not None else set(MEMORY_AGENT_TASKS)
    unknown = selected - set(MEMORY_AGENT_TASKS)
    if unknown:
        raise ValueError(f"unknown MemoryAgentBench sources: {sorted(unknown)}")

    occurrences: dict[str, int] = {}
    cases: list[BenchmarkCase] = []
    for split in MEMORY_AGENT_BENCH_SPLITS:
        if split not in split_rows:
            raise ValueError(f"missing MemoryAgentBench split {split!r}")
        for source_row_index, row in enumerate(split_rows[split]):
            metadata = row.get("metadata")
            source = metadata.get("source") if isinstance(metadata, dict) else None
            if source not in MEMORY_AGENT_TASKS:
                raise ValueError(f"unrecognized MemoryAgentBench source {source!r}")
            if source not in selected:
                continue
            source_occurrence = occurrences.get(source, 0)
            occurrences[source] = source_occurrence + 1
            if (
                max_cases_per_source is not None
                and source_occurrence >= max_cases_per_source
            ):
                continue
            cases.append(
                _normalize_row(
                    split,
                    row,
                    source_row_index,
                    source_occurrence,
                    max_questions=max_questions_per_case,
                    chunker=chunker,
                )
            )

    missing = selected - set(occurrences)
    if missing:
        raise ValueError(f"selected MemoryAgentBench sources are absent: {sorted(missing)}")
    return BenchmarkBundle(
        benchmark_id="memory-agent-bench",
        dataset_revision=MEMORY_AGENT_BENCH_REVISION,
        dataset_sha256=MEMORY_AGENT_BENCH_SHA256,
        cases=tuple(cases),
        metadata={
            "source": "ai-hyz/MemoryAgentBench",
            "chunking": {
                "sentence_tokenizer": "nltk.punkt_tab",
                "tokenizer_model": "gpt-4o-mini",
                "max_tokens": 4096,
            },
            "split_sha256": {
                split: contract["sha256"]
                for split, contract in MEMORY_AGENT_BENCH_SPLITS.items()
            },
            "task_contracts": {
                source: {
                    "contract_id": task.contract_id,
                    "system_prompt": task.system_prompt,
                    "query_template": task.query_template,
                    "scorer": task.scorer,
                }
                for source, task in sorted(MEMORY_AGENT_TASKS.items())
            },
            "scorer_contract": {
                "official_commit": MEMORY_AGENT_BENCH_SCORER_COMMIT,
                "movie_mapping_sha256": MEMORY_AGENT_BENCH_MOVIE_MAPPING_SHA256,
                "infbench_prompts": {
                    "fluency": INF_BENCH_FLUENCY_PROMPT,
                    "recall": INF_BENCH_RECALL_PROMPT,
                    "precision": INF_BENCH_PRECISION_PROMPT,
                },
            },
            **({"run_mode": run_mode} if run_mode is not None else {}),
        },
    )


def load_memory_agent_bench(
    directory: Path,
    **normalize_kwargs: Any,
) -> BenchmarkBundle:
    """Verify all pinned parquet files and normalize selected cases."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("install agent-memory[benchmarks] to read MAB parquet") from exc

    split_rows: dict[str, list[dict[str, Any]]] = {}
    actual_sources: set[str] = set()
    for split, contract in MEMORY_AGENT_BENCH_SPLITS.items():
        path = directory / f"{split}.parquet"
        actual = _sha256_file(path)
        if actual != contract["sha256"]:
            raise ValueError(
                f"MemoryAgentBench {split} checksum mismatch: expected "
                f"{contract['sha256']}, got {actual}"
            )
        frame = pd.read_parquet(path)
        if len(frame) != contract["rows"]:
            raise ValueError(f"MemoryAgentBench {split} row count drifted")
        rows = frame.to_dict(orient="records")
        split_rows[split] = rows
        for row in rows:
            metadata = row.get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("source"), str):
                actual_sources.add(metadata["source"])
    if actual_sources != set(MEMORY_AGENT_TASKS):
        raise ValueError(
            "MemoryAgentBench source registry drift: "
            f"missing={sorted(set(MEMORY_AGENT_TASKS) - actual_sources)}, "
            f"unexpected={sorted(actual_sources - set(MEMORY_AGENT_TASKS))}"
        )
    return normalize_memory_agent_bench(split_rows, **normalize_kwargs)


__all__ = [
    "MEMORY_AGENT_BENCH_MOVIE_MAPPING_SHA256",
    "MEMORY_AGENT_BENCH_REVISION",
    "MEMORY_AGENT_BENCH_SCORER_COMMIT",
    "MEMORY_AGENT_BENCH_SHA256",
    "MEMORY_AGENT_BENCH_SPLITS",
    "SMOKE_SOURCES",
    "chunk_text_into_sentences",
    "download_memory_agent_bench",
    "download_movie_entity_mapping",
    "load_memory_agent_bench",
    "load_movie_entity_mapping",
    "normalize_memory_agent_bench",
]
