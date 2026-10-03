"""Run the isolated Agent A-Mem storage and recovery smoke."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any, cast
from uuid import uuid4

from dotenv import load_dotenv
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import agent_memory as am  # noqa: E402
from agent_memory.adapters.lotus import LotusAdapter  # noqa: E402
from agent_memory.adapters.lotus.context import LotusExecutionConfig  # noqa: E402
from agent_memory.datasets.locomo import load_locomo_rows  # noqa: E402
from agent_memory.evaluation.locomo import ensure_locomo_dataset  # noqa: E402
from agent_memory.evaluation.pricing import PricingSnapshot  # noqa: E402
from agent_memory.evaluation.trace_metrics import (  # noqa: E402
    normalize_provider_calls,
    summarize_provider_calls,
)
from agent_memory.memories.a_mem.prompts import (  # noqa: E402
    AMEM_SOURCE_COMMIT,
    ANALYSE_CONTENT_PROMPT,
    EMBEDDING_TEXT_PROMPT,
    EVOLUTION_SYSTEM_PROMPT,
    NOTE_CONSOLIDATION_INSTRUCTION,
)
from agent_memory.memories.a_mem.storage import (  # noqa: E402
    AMEM_BGE_M3,
    AMEM_NEO4J_SCHEMA,
    AMEM_NEO4J_STATEMENTS,
)
from agent_memory.policy.retrieval import RetrievalResult  # noqa: E402
from agent_memory.storage import StorageDeployment  # noqa: E402
from agent_memory.storage.neo4j import (  # noqa: E402
    Neo4jConnector,
    SentenceTransformerEmbeddingProvider,
)
from agent_memory.tracing.semantic import semantic_trace_scope  # noqa: E402


LOCOMO_CACHE_PATH = PROJECT_ROOT / ".cache" / "agent-memory" / "locomo10.json"
MODEL = "deepseek/deepseek-v4-flash"
LOG_COLUMNS = ("content", "timestamp")
RETRIEVAL_CHANNELS = ("notes", "neighbors")
TRACE_RUN_KIND = "amem"
LOCOMO_TURN_LIMIT = 1000

# Free OpenRouter variants allow 20 requests per minute, so stay under it; LOTUS
# applies the limit as a per-request delay and caps the batch size with it.
LM_NUM_RETRIES = 12
LM_TIMEOUT_SECONDS = 120
LM_RATE_LIMIT_PER_MINUTE = 10

# One pinned LOCOMO exchange. Sample 0 is conv-26, a Caroline/Melanie dialogue, and
# the slice starts at D2:8, the first turn of the adoption-agency thread. All five
# turns advance one topic — researching agencies, then picking one, then the reason
# — which is what lets A-Mem link notes and rewrite an earlier note's context and
# tags; see EVOLUTION_SYSTEM_PROMPT in src/agent_memory/memories/a_mem/prompts.py.
SAMPLE_INDEX = 0
START_TURN_ID = "D2:8"
ROW_LIMIT = 5

# Official LOCOMO question conv-26:q88. Its answer is stated in the last ingested
# turn (D2:12), so reaching it depends on the link chain and the evolved context
# A-Mem builds across the thread rather than on any single note.
QUESTION_ID = "conv-26:q88"
QUERY = "Why did Caroline choose the adoption agency?"
GOLD_ANSWER = "because of their inclusivity and support for LGBTQ+ individuals"

RETRIEVAL_CONTRACT: dict[str, Any] = {
    "profile": "amem-bge-m3",
    "notes_method": "cosine_similarity",
    "notes_limit": 5,
    "neighbors_method": "bfs",
    "neighbors_limit": 10,
    "bfs_max_depth": 1,
    "reranker": None,
}


def main() -> None:
    """Run add, search, checkpoint restore and repeated search."""

    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = require_environment(args.model)

    trace_dir = output_dir / "trace"
    dataset_path, raw_rows, rows = selected_rows(row_limit=args.row_limit)
    write_json(output_dir / "input" / "locomo_rows.json", list(raw_rows))
    write_json(output_dir / "input" / "amem_rows.json", rows)
    print(
        f"Agent A-Mem storage and recovery smoke: {len(rows)} LOCOMO turns from "
        f"{dataset_path}, namespace {args.namespace}"
    )

    # The deployment publishes a StorageConnector protocol, but this demo owns the
    # Neo4j connections it opens and has to close them, so narrow to the concrete
    # type at the point of construction.
    connectors: list[Neo4jConnector] = []
    status = "failed"
    timings: dict[str, float] = {}
    memory: am.AMem | None = None
    observed: dict[str, dict[str, Any]] = {}
    try:
        storage = create_storage_deployment(args.namespace)
        connector = cast(Neo4jConnector, storage.connector)
        connectors.append(connector)
        adapter = LotusAdapter(
            model=args.model,
            config=LotusExecutionConfig(
                lm_model_kwargs=model_kwargs(args.model, provider),
                lm_num_retries=LM_NUM_RETRIES,
                lm_timeout=LM_TIMEOUT_SECONDS,
                lm_rate_limit=LM_RATE_LIMIT_PER_MINUTE,
                lm_enable_cache=False,
                semantic_trace_dir=trace_dir,
            ),
        )
        memory = am.AMem(adapter=adapter, storage=storage)

        insertion_started = time.perf_counter()
        for index, row in enumerate(rows, start=1):
            print(f"add[{index}]: {row['content'][:100]}")
            with semantic_trace_scope(
                run_kind=TRACE_RUN_KIND,
                phase="add",
                add_index=index,
                source_description=row["source_description"],
            ):
                memory.add({name: row[name] for name in LOG_COLUMNS})
            record_new_notes(observed, memory, add_index=index)
        timings["insertion_wall_seconds"] = round(
            time.perf_counter() - insertion_started, 4
        )
        write_state_artifacts(memory, output_dir)

        llm_before_retrieval = provider_call_count(trace_dir)
        retrieval_started = time.perf_counter()
        with semantic_trace_scope(run_kind=TRACE_RUN_KIND, phase="retrieval"):
            before = memory.query(QUERY)
        timings["retrieval_before_wall_seconds"] = round(
            time.perf_counter() - retrieval_started, 4
        )
        if not isinstance(before, RetrievalResult):
            raise TypeError("Agent A-Mem query must return RetrievalResult")
        if provider_call_count(trace_dir) != llm_before_retrieval:
            raise AssertionError("Agent A-Mem retrieval unexpectedly called the LLM")
        validate_retrieval_result(before)
        write_retrieval(output_dir / "retrieval" / "before_restore.json", before)

        checkpoint_started = time.perf_counter()
        snapshot = memory._runtime.snapshot_state()
        checkpoint_path = output_dir / "checkpoint" / "state.pkl"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(pickle.dumps(snapshot))
        timings["checkpoint_save_wall_seconds"] = round(
            time.perf_counter() - checkpoint_started, 4
        )
        write_json(
            output_dir / "checkpoint" / "metadata.json",
            {
                "schema_version": snapshot["schema_version"],
                "policy_fingerprint": snapshot["plan_fingerprint"],
                "storage_commit": snapshot.get("storage_commit"),
                "storage_bound": True,
                "bytes": checkpoint_path.stat().st_size,
            },
        )

        # Restore against a fresh connection, so the round trip proves the durable
        # state replays into a new physical handle rather than reusing the old one.
        connector.close()
        connectors.remove(connector)
        restored_storage = create_storage_deployment(args.namespace)
        connectors.append(cast(Neo4jConnector, restored_storage.connector))
        restored = am.AMem(adapter=adapter, storage=restored_storage)
        restore_started = time.perf_counter()
        restored._runtime.restore_state(pickle.loads(checkpoint_path.read_bytes()))
        timings["checkpoint_restore_wall_seconds"] = round(
            time.perf_counter() - restore_started, 4
        )
        assert_same_state(memory, restored)

        llm_before_restored_retrieval = provider_call_count(trace_dir)
        retrieval_started = time.perf_counter()
        with semantic_trace_scope(run_kind=TRACE_RUN_KIND, phase="retrieval"):
            after = restored.query(QUERY)
        timings["retrieval_after_wall_seconds"] = round(
            time.perf_counter() - retrieval_started, 4
        )
        if not isinstance(after, RetrievalResult):
            raise TypeError("restored Agent A-Mem query must return RetrievalResult")
        if provider_call_count(trace_dir) != llm_before_restored_retrieval:
            raise AssertionError(
                "restored Agent A-Mem retrieval unexpectedly called the LLM"
            )
        validate_retrieval_result(after)
        assert_same_retrieval(before, after)
        write_retrieval(output_dir / "retrieval" / "after_restore.json", after)

        evolution = evolution_summary(
            observed=observed,
            memory=restored,
            rows=rows,
            neighbors_rows=len(after.channels["neighbors"]),
        )
        write_json(output_dir / "evolution" / "summary.json", evolution)
        write_json(
            output_dir / "retrieval" / "summary.json",
            {
                "query": QUERY,
                "namespace": args.namespace,
                "channels": {
                    name: len(frame) for name, frame in after.channels.items()
                },
                "checkpoint_order_verified": True,
            },
        )
        write_state_artifacts(restored, output_dir)
        print_evolution_report(evolution)
        status = "completed"
    except Exception as error:
        write_json(
            output_dir / "diagnostics" / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        for connector in connectors:
            connector.close()
        write_metrics(output_dir, status=status, timings=timings)
        write_json(
            output_dir / "manifest.json",
            manifest(
                output_dir,
                model=args.model,
                namespace=args.namespace,
                status=status,
                rows=rows,
            ),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the isolated Agent A-Mem smoke options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row-limit",
        type=int,
        default=ROW_LIMIT,
        help=(
            "Turns to ingest from START_TURN_ID. A-Mem issues four semantic calls "
            "per note plus one evolution call per new/earlier note pair, so keep "
            "this small."
        ),
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--namespace", default=f"amem-e2e-{uuid4()}")
    return parser.parse_args(argv)


def default_output_dir() -> Path:
    """Return a fresh timestamped output path for one run."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / ".memory-test" / "a_mem-e2e" / timestamp


def model_provider(model: str) -> str:
    """Return the LiteLLM provider for ``model``, or fail when unrecognized."""

    import litellm

    try:
        _model, provider, _api_key, _api_base = litellm.get_llm_provider(model=model)
    except Exception as error:
        raise SystemExit(
            f"unrecognized LiteLLM provider for model {model!r}: {error}"
        ) from error
    return provider


def model_kwargs(model: str, provider: str) -> dict[str, Any]:
    """Return the provider-specific LOTUS model kwargs for one run.

    Retries, timeout and rate limiting are execution-owned options and must go on
    ``LotusExecutionConfig`` instead; overriding them here raises.
    """

    # litellm defaults to "constant_retry", which fires every attempt back to back
    # with no wait; a shared-pool 429 needs real backoff to clear.
    kwargs: dict[str, Any] = {"retry_strategy": "exponential_backoff_retry"}
    if provider == "deepseek":
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    elif provider == "openrouter":
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        # OpenRouter ":free" variants cap output at 512 tokens.
        kwargs["max_tokens"] = 512 if ":free" in model else 4096
    return kwargs


def require_environment(model: str = MODEL) -> str:
    """Load the local credentials and fail before creating artifacts if absent.

    Returns the LiteLLM provider so callers can pick provider-specific settings.
    """

    load_dotenv(PROJECT_ROOT / ".env")
    import litellm

    provider = model_provider(model)
    validation = litellm.validate_environment(model=model)
    if not validation["keys_in_environment"]:
        raise SystemExit(
            f"model {model!r} requires environment variable(s): "
            + ", ".join(str(name) for name in validation["missing_keys"])
        )
    missing = [
        name
        for name in ("AGENT_MEMORY_NEO4J_URI", "AGENT_MEMORY_NEO4J_PASSWORD")
        if not os.getenv(name)
    ]
    if missing:
        raise SystemExit(
            "Agent A-Mem retrieval requires environment variables: "
            + ", ".join(missing)
        )
    return provider


# Input normalization.
def locomo_source_description(row: Mapping[str, Any]) -> str:
    """Return a short human-readable provenance label for one LOCOMO turn."""

    session_number = str(row.get("session_id", "")).removeprefix("session_")
    return (
        f"LOCOMO sample {row.get('sample_index')} session {session_number} "
        f"turn {row.get('turn_id', '')}"
    )


def amem_log_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map one normalized LOCOMO turn to the AMem log schema.

    Mirrors ``A-mem/test_advanced_robust.py:240-244``; the session ``date_time``
    is passed through unchanged (``A-mem/memory_layer.py:294-296``).
    """

    speaker = str(row.get("speaker", ""))
    text = str(row.get("message", ""))
    caption = str(row.get("blip_caption", "")).strip()
    if caption:
        # A-mem/load_dataset.py:63-70 inlines the caption as "[Image: ...]".
        caption_text = f"[Image: {caption}]"
        text = f"{caption_text} {text}" if text else caption_text
    return {
        "content": f"Speaker {speaker}says : {text}",
        "timestamp": str(row.get("timestamp", "")),
    }


def selected_rows(
    *,
    row_limit: int,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the pinned LOCOMO exchange and return raw and A-Mem-shaped rows."""

    if row_limit < 1:
        raise SystemExit("--row-limit must be at least 1")
    dataset_path = ensure_locomo_dataset(LOCOMO_CACHE_PATH)
    loaded = load_locomo_rows(
        dataset_path,
        sample_limit=SAMPLE_INDEX + 1,
        turn_limit=LOCOMO_TURN_LIMIT,
    )
    start = next(
        (
            index
            for index, row in enumerate(loaded)
            if str(row.get("turn_id", "")) == START_TURN_ID
        ),
        None,
    )
    if start is None:
        raise SystemExit(f"LOCOMO turn {START_TURN_ID!r} is not in the dataset")
    selected = loaded[start : start + row_limit]
    if len(selected) != row_limit:
        raise SystemExit(
            f"Requested {row_limit} LOCOMO turns from {START_TURN_ID}, "
            f"found {len(selected)}"
        )
    rows: list[dict[str, Any]] = []
    for row in selected:
        mapped = amem_log_row(row)
        mapped["source_description"] = locomo_source_description(row)
        rows.append(mapped)
    return dataset_path, selected, rows


def policy_input_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the normalized source rows shared with the native baseline."""

    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Retrieval assertions.
def validate_retrieval_result(result: RetrievalResult) -> None:
    """Require the A-Mem channels, a non-empty note channel and valid ranks."""

    if tuple(result.channels) != RETRIEVAL_CHANNELS:
        raise RuntimeError(
            f"Agent A-Mem retrieval must return {RETRIEVAL_CHANNELS} channels, "
            f"got {tuple(result.channels)}"
        )
    for name, frame in result.channels.items():
        if frame.empty:
            if name == "notes":
                raise RuntimeError("Agent A-Mem retrieval channel 'notes' is empty")
            continue
        missing = [
            column
            for column in ("record_id", "content", "rank", "score")
            if column not in frame.columns
        ]
        if missing:
            raise RuntimeError(
                f"Agent A-Mem retrieval channel {name!r} is missing columns: {missing}"
            )
        ranks = frame["rank"].tolist()
        if ranks != list(range(1, len(frame) + 1)):
            raise RuntimeError(
                f"Agent A-Mem retrieval channel {name!r} has invalid ranks: {ranks}"
            )


def record_ids(frame: pd.DataFrame, *, channel: str) -> list[Any]:
    """Return one channel's record identity column, or fail on a broken schema."""

    if "record_id" not in frame.columns:
        raise RuntimeError(
            f"Agent A-Mem retrieval channel {channel!r} has no record_id column"
        )
    return frame["record_id"].tolist()


def assert_same_retrieval(
    before: RetrievalResult,
    after: RetrievalResult,
) -> None:
    """Verify checkpoint restore preserves result identities and ordering."""

    if tuple(before.channels) != tuple(after.channels):
        raise RuntimeError("retrieval channels changed after checkpoint restore")
    for name in before.channels:
        before_ids = record_ids(before.channels[name], channel=name)
        after_ids = record_ids(after.channels[name], channel=name)
        if before_ids != after_ids:
            raise RuntimeError(
                f"retrieval order changed after restore for {name!r}: "
                f"{before_ids!r} != {after_ids!r}"
            )


def assert_same_state(memory: am.AMem, restored: am.AMem) -> None:
    """Require the restored durable state to equal the saved state."""

    before_state = memory._runtime._state
    after_state = restored._runtime._state
    if set(before_state) != set(after_state):
        raise RuntimeError(
            "checkpoint restore changed durable state keys: "
            f"{sorted(before_state)} != {sorted(after_state)}"
        )
    for name in sorted(before_state):
        pd.testing.assert_frame_equal(before_state[name], after_state[name])


# Evolution evidence (reported, never asserted).
def text_value(value: Any) -> str:
    """Return a stripped string for one possibly-null frame cell."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def string_sequence(value: Any) -> tuple[str, ...]:
    """Return a sorted string tuple for one list-valued frame cell."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        text = text_value(value)
        return (text,) if text else ()
    try:
        entries = list(value)
    except TypeError:
        text = text_value(value)
        return (text,) if text else ()
    normalized = (text_value(entry) for entry in entries)
    return tuple(sorted(item for item in normalized if item))


def note_observations(memory: am.AMem) -> dict[str, tuple[str, tuple[str, ...], str]]:
    """Return ``{note_id: (content, tags, context)}`` for the current note view."""

    frame = memory._runtime._state.get("note", pd.DataFrame())
    observations: dict[str, tuple[str, tuple[str, ...], str]] = {}
    for record in frame.to_dict("records"):
        note_id = text_value(record.get("_row_id"))
        if not note_id:
            continue
        observations[note_id] = (
            text_value(record.get("content")),
            string_sequence(record.get("tags")),
            text_value(record.get("context")),
        )
    return observations


def record_new_notes(
    observed: dict[str, dict[str, Any]],
    memory: am.AMem,
    *,
    add_index: int,
) -> None:
    """Remember each note's analysed values at the add where it first appears."""

    for note_id, (content, tags, context) in note_observations(memory).items():
        if note_id in observed:
            continue
        observed[note_id] = {
            "index": add_index,
            "content": content,
            "tags": tags,
            "context": context,
        }


def evolution_summary(
    *,
    observed: Mapping[str, Mapping[str, Any]],
    memory: am.AMem,
    rows: Sequence[Mapping[str, Any]],
    neighbors_rows: int | None,
) -> dict[str, Any]:
    """Report link creation and neighbour rewriting without asserting them.

    A note is judged only once a later note has been added, so a freshly created
    note cannot count as preserved.
    """

    final = note_observations(memory)
    last_add_index = len(rows)
    rewritten: list[dict[str, Any]] = []
    preserved: list[str] = []
    missing: list[str] = []
    for note_id, first in observed.items():
        if first["index"] >= last_add_index:
            continue
        current = final.get(note_id)
        if current is None:
            missing.append(note_id)
            continue
        content, tags, context = current
        if (tags, context) == (first["tags"], first["context"]):
            preserved.append(note_id)
            continue
        rewritten.append(
            {
                "note_id": note_id,
                "first_seen_add": first["index"],
                "content": first["content"][:120],
                "tags_changed": list(first["tags"]) != list(tags),
                "context_changed": first["context"] != context,
                "tags_before": list(first["tags"]),
                "tags_after": list(tags),
                "context_before": first["context"],
                "context_after": context,
            }
        )
    return {
        "note_rows": len(final),
        "notes_observed": len(observed),
        "rewritten_count": len(rewritten),
        "preserved_count": len(preserved),
        "missing_count": len(missing),
        "missing_note_ids": missing,
        "rewritten": rewritten,
        "neighbors_rows": neighbors_rows,
        "links_created": None if neighbors_rows is None else neighbors_rows > 0,
        "asserted": False,
    }


def print_evolution_report(evolution: Mapping[str, Any]) -> None:
    """Print the reported, never asserted evolution evidence."""

    print("\nevolution report (reported, never asserted):")
    print(
        f"  notes: {evolution['notes_observed']} observed, "
        f"{evolution['rewritten_count']} rewritten, "
        f"{evolution['preserved_count']} preserved"
    )
    print(f"  neighbors channel rows: {evolution['neighbors_rows']}")
    if not evolution["links_created"]:
        print(
            "  warning: no RELATES_TO links were created, so the neighbors "
            "channel is empty"
        )
    if evolution["missing_count"]:
        print(
            f"  warning: {evolution['missing_count']} observed note(s) are absent "
            "from the final note view"
        )


# Physical deployment and artifacts.
def create_storage_deployment(namespace: str) -> StorageDeployment:
    """Create the real CPU-only Neo4j deployment.

    Both A-Mem channels declare ``reranker=None``, so no cross-encoder provider.
    """

    connector = Neo4jConnector(
        uri=os.environ["AGENT_MEMORY_NEO4J_URI"],
        auth=(
            os.getenv("AGENT_MEMORY_NEO4J_USER", "neo4j"),
            os.environ["AGENT_MEMORY_NEO4J_PASSWORD"],
        ),
        database=os.getenv("AGENT_MEMORY_NEO4J_DATABASE", "neo4j"),
        embedding_provider=SentenceTransformerEmbeddingProvider(AMEM_BGE_M3),
        schema=AMEM_NEO4J_SCHEMA,
    )
    try:
        return StorageDeployment(
            connector=connector,
            statements=AMEM_NEO4J_STATEMENTS,
            namespace=namespace,
        )
    except Exception:
        connector.close()
        raise


def write_state_artifacts(memory: am.AMem, output_dir: Path) -> dict[str, Path]:
    """Write the current source log and the public note view."""

    state = memory._runtime._state
    return {
        "state/log": write_json(
            output_dir / "state" / "log.json",
            state.get("log", pd.DataFrame()).to_dict("records"),
        ),
        "views/note": write_json(
            output_dir / "views" / "note.json",
            state.get("note", pd.DataFrame()).to_dict("records"),
        ),
    }


def write_retrieval(path: Path, result: RetrievalResult) -> None:
    """Write one JSON-safe retrieval artifact."""

    write_json(
        path,
        {
            "query": result.query,
            "channels": {
                name: frame.to_dict("records")
                for name, frame in result.channels.items()
            },
            "metrics": {
                name: dict(metrics) for name, metrics in result.metrics.items()
            },
        },
    )


def write_metrics(
    output_dir: Path,
    *,
    status: str,
    timings: Mapping[str, float],
) -> None:
    """Normalize provider attempts and write a reproducible metrics summary."""

    events = read_jsonl(output_dir / "trace" / "events.jsonl")
    provider_rows = normalize_provider_calls(
        events,
        output_dir=output_dir,
        pricing=PricingSnapshot.deepseek_2026_07_17(),
    )
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(provider_rows).to_csv(
        metrics_dir / "provider_usage.csv",
        index=False,
    )
    write_json(
        metrics_dir / "summary.json",
        {
            "status": status,
            **dict(timings),
            "provider_usage": summarize_provider_calls(provider_rows),
            "llm_calls": provider_call_count(output_dir / "trace"),
        },
    )


# Provenance.
def manifest(
    output_dir: Path,
    *,
    model: str,
    namespace: str,
    status: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the non-sensitive Agent A-Mem reproducibility manifest."""

    return {
        "run_mode": "integration-smoke",
        "system": "agent-amem",
        "status": status,
        "source": source_state(PROJECT_ROOT),
        "upstream_reference": "A-mem (read-only reference checkout at ../A-mem)",
        "native_amem_reference_commit": AMEM_SOURCE_COMMIT,
        "input_fingerprint": policy_input_fingerprint(rows),
        "locomo_question": {
            "question_id": QUESTION_ID,
            "question": QUERY,
            "gold_answer": GOLD_ANSWER,
            "start_turn_id": START_TURN_ID,
            "row_limit": len(rows),
        },
        "memory_model": model,
        "memory_thinking": "disabled",
        "runtime_contract_fingerprint": runtime_contract_fingerprint(model),
        "application_cache": "disabled",
        "prompts": prompt_fingerprints(),
        "embedding": {**AMEM_BGE_M3.to_dict(), "device": "cpu"},
        "retrieval": dict(RETRIEVAL_CONTRACT),
        "storage": {
            "kind": "neo4j",
            "namespace": namespace,
            "database": os.getenv("AGENT_MEMORY_NEO4J_DATABASE", "neo4j"),
            "driver": package_version("neo4j"),
        },
        "runtime": runtime_versions(),
        "pricing": PricingSnapshot.deepseek_2026_07_17().to_dict(),
    }


def prompt_fingerprints() -> dict[str, dict[str, str]]:
    """Return the four A-Mem prompt paths and their content digests."""

    prompts = (
        ("analyse_content", "ANALYSE_CONTENT_PROMPT", ANALYSE_CONTENT_PROMPT),
        ("evolution_system", "EVOLUTION_SYSTEM_PROMPT", EVOLUTION_SYSTEM_PROMPT),
        (
            "note_consolidation",
            "NOTE_CONSOLIDATION_INSTRUCTION",
            NOTE_CONSOLIDATION_INSTRUCTION,
        ),
        ("embedding_text", "EMBEDDING_TEXT_PROMPT", EMBEDDING_TEXT_PROMPT),
    )
    return {
        label: {
            "path": f"src/agent_memory/memories/a_mem/prompts.py:{constant}",
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        for label, constant, text in prompts
    }


def runtime_contract_fingerprint(model: str) -> str:
    """Hash the cross-system physical model and retrieval contract."""

    payload = {
        "memory_model": model.removeprefix("deepseek/"),
        "thinking": "disabled",
        "application_cache": "disabled",
        "embedding_model": AMEM_BGE_M3.model,
        "embedding_revision": AMEM_BGE_M3.revision,
        "embedding_dimensions": AMEM_BGE_M3.dimensions,
        "embedding_source_column": AMEM_BGE_M3.source_column,
        "embedding_device": "cpu",
        **RETRIEVAL_CONTRACT,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_state(repo: Path) -> dict[str, Any]:
    """Return commit, dirty state and uv lock digest."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {
        "commit": commit,
        "dirty": dirty,
        "lockfile": "uv.lock",
        "lockfile_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
    }


def runtime_versions() -> dict[str, Any]:
    """Return the critical isolated runtime versions."""

    return {
        "python": sys.version.split()[0],
        "agent-memory": package_version("agent-memory"),
        "neo4j": package_version("neo4j"),
        "sentence-transformers": package_version("sentence-transformers"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "lotus-ai": package_version("lotus-ai"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "litellm": package_version("litellm"),
    }


def provider_call_count(trace_dir: Path) -> int:
    """Count physical provider usage events in the semantic trace."""

    return sum(
        event.get("event_type") == "provider_usage"
        for event in read_jsonl(trace_dir / "events.jsonl")
    )


def package_version(name: str) -> str | None:
    """Return an installed package version, if present."""

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read an optional JSONL artifact."""

    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> Path:
    """Write one inspectable UTF-8 JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    main()
