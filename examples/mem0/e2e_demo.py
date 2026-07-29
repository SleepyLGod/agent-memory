"""Run the isolated Agent Mem0 Base storage and recovery smoke."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_memory.adapters.lotus import LotusAdapter  # noqa: E402
from agent_memory.adapters.lotus.context import LotusExecutionConfig  # noqa: E402
from agent_memory.evaluation.pricing import PricingSnapshot  # noqa: E402
from agent_memory.evaluation.trace_metrics import (  # noqa: E402
    normalize_provider_calls,
    summarize_provider_calls,
)
from agent_memory.memories.mem0 import Mem0Memory  # noqa: E402
from agent_memory.memories.mem0.prompts import (  # noqa: E402
    MEM0_ADDITIVE_EXTRACTION_INSTRUCTION,
    MEM0_SOURCE_COMMIT,
)
from agent_memory.memories.mem0.storage import (  # noqa: E402
    MEM0_BGE_M3,
    MEM0_QDRANT_STATEMENTS,
)
from agent_memory.policy.retrieval import RetrievalResult  # noqa: E402
from agent_memory.storage import (  # noqa: E402
    EmbeddingProvider,
    EmbeddingSpec,
    SentenceTransformerEmbeddingProvider,
    StorageDeployment,
)
from agent_memory.storage.qdrant import QdrantConnector  # noqa: E402
from agent_memory.tracing.semantic import semantic_trace_scope  # noqa: E402


MODEL = "deepseek/deepseek-v4-flash"
TOP_K = 20
THRESHOLD = 0.1
CANONICAL_EVENTS = (
    {
        "event_id": "event-0001",
        "role": "user",
        "content": "I prefer Ethiopian coffee and brew it every morning.",
        "observation_date": "2026-07-26T08:00:00+08:00",
    },
    {
        "event_id": "event-0002",
        "role": "assistant",
        "content": "I will remember your Ethiopian coffee preference.",
        "observation_date": "2026-07-26T08:00:30+08:00",
    },
    {
        "event_id": "event-0003",
        "role": "user",
        "content": "My grinder setting is 18 clicks.",
        "observation_date": "2026-07-26T08:01:00+08:00",
    },
)
QUERY = "What coffee do I prefer, and what grinder setting do I use?"


def canonical_input_fingerprint() -> str:
    """Return the cross-system canonical input fingerprint."""

    encoded = json.dumps(
        {"events": CANONICAL_EVENTS, "query": QUERY},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the isolated Agent Mem0 smoke options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--namespace", default=f"mem0-base-{uuid4()}")
    parser.add_argument("--model", default=MODEL)
    return parser.parse_args(argv)


class TracingEmbeddingProvider:
    """Trace the backend-neutral embedding boundary without persisting vectors."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        output_dir: Path,
    ) -> None:
        self.provider = provider
        self.output_dir = output_dir
        self.phase = "setup"
        self.event_id = ""

    def embed(
        self,
        spec: EmbeddingSpec,
        texts: list[str],
    ) -> list[list[float]]:
        """Delegate one real embedding call and record its input and shape."""

        trace_id = f"embedding-{uuid4()}"
        input_path = self.output_dir / "trace" / "prompts" / f"{trace_id}.json"
        write_json(input_path, texts)
        started = time.perf_counter()
        try:
            vectors = self.provider.embed(spec, texts)
        except Exception as error:
            append_jsonl(
                self.output_dir / "trace" / "events.jsonl",
                {
                    "timestamp": _timestamp(),
                    "trace_id": trace_id,
                    "event_type": "embedding_call",
                    "status": "error",
                    "phase": self.phase,
                    "event_id": self.event_id,
                    "model": spec.model,
                    "latency_seconds": time.perf_counter() - started,
                    "input_path": str(input_path.relative_to(self.output_dir)),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            raise
        append_jsonl(
            self.output_dir / "trace" / "events.jsonl",
            {
                "timestamp": _timestamp(),
                "trace_id": trace_id,
                "event_type": "embedding_call",
                "status": "success",
                "phase": self.phase,
                "event_id": self.event_id,
                "model": spec.model,
                "revision": spec.revision,
                "latency_seconds": time.perf_counter() - started,
                "input_path": str(input_path.relative_to(self.output_dir)),
                "item_count": len(vectors),
                "dimensions": len(vectors[0]) if vectors else 0,
            },
        )
        return vectors


def run(args: argparse.Namespace) -> Path:
    """Run add, search, checkpoint restore and repeated search."""

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.env_file is not None:
        load_dotenv(args.env_file)
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    write_json(
        args.output_dir / "input" / "events.json",
        list(CANONICAL_EVENTS),
    )
    write_json(args.output_dir / "input" / "query.json", {"query": QUERY})

    trace_dir = args.output_dir / "trace"
    adapter = LotusAdapter(
        model=args.model,
        config=LotusExecutionConfig(
            lm_model_kwargs={
                "extra_body": {"thinking": {"type": "disabled"}},
            },
            lm_enable_cache=False,
            semantic_trace_dir=trace_dir,
        ),
    )
    base_provider = SentenceTransformerEmbeddingProvider(
        MEM0_BGE_M3,
        device="cpu",
        dependency_extra="mem0",
    )
    embedding = TracingEmbeddingProvider(
        base_provider,
        output_dir=args.output_dir,
    )

    connectors: list[QdrantConnector] = []
    status = "failed"
    timings: dict[str, float] = {}
    memory: Mem0Memory | None = None
    try:
        connector = QdrantConnector(
            path=args.output_dir / "qdrant",
            embedding_provider=embedding,
        )
        connectors.append(connector)
        storage = StorageDeployment(
            connector=connector,
            statements=MEM0_QDRANT_STATEMENTS,
            namespace=args.namespace,
        )
        memory = Mem0Memory(adapter=adapter, storage=storage)

        insertion_started = time.perf_counter()
        for event in CANONICAL_EVENTS:
            embedding.phase = "insertion"
            embedding.event_id = event["event_id"]
            with semantic_trace_scope(
                phase="insertion",
                event_id=event["event_id"],
            ):
                memory.add(
                    {
                        "role": event["role"],
                        "content": event["content"],
                        "observation_date": event["observation_date"],
                    }
                )
        timings["insertion_wall_seconds"] = (
            time.perf_counter() - insertion_started
        )
        write_json(
            args.output_dir / "views" / "memories.json",
            memory._runtime._state["memories"].to_dict("records"),
        )

        llm_before_retrieval = provider_call_count(trace_dir)
        embedding.phase = "retrieval"
        embedding.event_id = ""
        retrieval_started = time.perf_counter()
        with semantic_trace_scope(phase="retrieval"):
            before = memory.query(QUERY)
        if not isinstance(before, RetrievalResult):
            raise TypeError("Agent Mem0 query must return RetrievalResult")
        timings["retrieval_before_wall_seconds"] = (
            time.perf_counter() - retrieval_started
        )
        if provider_call_count(trace_dir) != llm_before_retrieval:
            raise AssertionError("Agent Mem0 retrieval unexpectedly called the LLM")
        require_nonempty(before)
        write_json(
            args.output_dir / "retrieval" / "before_restore.json",
            retrieval_payload(before),
        )

        checkpoint_started = time.perf_counter()
        snapshot = memory._runtime.snapshot_state()
        checkpoint_path = args.output_dir / "checkpoint" / "state.pkl"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(pickle.dumps(snapshot))
        timings["checkpoint_save_wall_seconds"] = (
            time.perf_counter() - checkpoint_started
        )
        write_json(
            args.output_dir / "checkpoint" / "metadata.json",
            {
                "schema_version": snapshot["schema_version"],
                "plan_fingerprint": snapshot["plan_fingerprint"],
                "storage_commit": snapshot["storage_commit"],
                "bytes": checkpoint_path.stat().st_size,
            },
        )

        connector.close()
        connectors.remove(connector)
        restored_connector = QdrantConnector(
            path=args.output_dir / "qdrant",
            embedding_provider=embedding,
        )
        connectors.append(restored_connector)
        restored_storage = StorageDeployment(
            connector=restored_connector,
            statements=MEM0_QDRANT_STATEMENTS,
            namespace=args.namespace,
        )
        restored = Mem0Memory(adapter=adapter, storage=restored_storage)
        embedding.phase = "checkpoint"
        restore_started = time.perf_counter()
        restored._runtime.restore_state(snapshot)
        timings["checkpoint_restore_wall_seconds"] = (
            time.perf_counter() - restore_started
        )

        llm_before_restored_retrieval = provider_call_count(trace_dir)
        embedding.phase = "retrieval"
        retrieval_started = time.perf_counter()
        with semantic_trace_scope(phase="retrieval"):
            after = restored.query(QUERY)
        if not isinstance(after, RetrievalResult):
            raise TypeError("restored Agent Mem0 query must return RetrievalResult")
        timings["retrieval_after_wall_seconds"] = (
            time.perf_counter() - retrieval_started
        )
        if provider_call_count(trace_dir) != llm_before_restored_retrieval:
            raise AssertionError(
                "restored Agent Mem0 retrieval unexpectedly called the LLM"
            )
        require_nonempty(after)
        assert_same_results(before, after)
        write_json(
            args.output_dir / "retrieval" / "after_restore.json",
            retrieval_payload(after),
        )
        status = "completed"
        return args.output_dir
    except Exception as error:
        write_json(
            args.output_dir / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        for connector in connectors:
            connector.close()
        write_metrics(args.output_dir, status=status, timings=timings)
        write_json(
            args.output_dir / "manifest.json",
            manifest(
                args.output_dir,
                model=args.model,
                namespace=args.namespace,
                status=status,
            ),
        )


def retrieval_payload(result: RetrievalResult) -> dict[str, Any]:
    """Return one JSON-safe retrieval artifact."""

    return {
        "query": result.query,
        "channels": {
            name: frame.to_dict("records")
            for name, frame in result.channels.items()
        },
        "metrics": {
            name: dict(metrics) for name, metrics in result.metrics.items()
        },
    }


def require_nonempty(result: RetrievalResult) -> None:
    """Require the Base memory channel to contain at least one result."""

    if result.channels["memories"].empty:
        raise AssertionError("Agent Mem0 retrieval returned no memories")


def assert_same_results(
    before: RetrievalResult,
    after: RetrievalResult,
) -> None:
    """Require stable record identities and order after checkpoint restore."""

    before_ids = before.channels["memories"]["record_id"].tolist()
    after_ids = after.channels["memories"]["record_id"].tolist()
    if before_ids != after_ids:
        raise AssertionError(
            "checkpoint restore changed retrieval record IDs or order"
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
            "embedding_calls": sum(
                event.get("event_type") == "embedding_call"
                for event in events
            ),
        },
    )


def manifest(
    output_dir: Path,
    *,
    model: str,
    namespace: str,
    status: str,
) -> dict[str, Any]:
    """Return the non-sensitive Agent Mem0 reproducibility manifest."""

    return {
        "run_mode": "integration-smoke",
        "system": "agent-mem0",
        "status": status,
        "source": source_state(PROJECT_ROOT),
        "native_mem0_reference_commit": MEM0_SOURCE_COMMIT,
        "input_fingerprint": canonical_input_fingerprint(),
        "memory_model": model,
        "memory_thinking": "disabled",
        "runtime_contract_fingerprint": runtime_contract_fingerprint(model),
        "application_cache": "disabled",
        "prompt": {
            "path": (
                "src/agent_memory/memories/mem0/prompts.py:"
                "MEM0_ADDITIVE_EXTRACTION_INSTRUCTION"
            ),
            "sha256": hashlib.sha256(
                MEM0_ADDITIVE_EXTRACTION_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
        },
        "embedding": {
            **MEM0_BGE_M3.to_dict(),
            "device": "cpu",
        },
        "retrieval": {
            "profile": "mem0-base-bge-m3",
            "method": "cosine",
            "candidate_limit": 80,
            "limit": TOP_K,
            "threshold": THRESHOLD,
            "reranker": None,
        },
        "qdrant": {
            "path": str(output_dir / "qdrant"),
            "namespace": namespace,
            "version": package_version("qdrant-client"),
        },
        "runtime": runtime_versions(),
        "pricing": PricingSnapshot.deepseek_2026_07_17().to_dict(),
    }


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
        "lockfile_sha256": hashlib.sha256(
            (repo / "uv.lock").read_bytes()
        ).hexdigest(),
    }


def runtime_versions() -> dict[str, Any]:
    """Return the critical isolated runtime versions."""

    return {
        "python": sys.version.split()[0],
        "agent-memory": package_version("agent-memory"),
        "qdrant-client": package_version("qdrant-client"),
        "sentence-transformers": package_version("sentence-transformers"),
        "numpy": package_version("numpy"),
        "lotus-ai": package_version("lotus-ai"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "litellm": package_version("litellm"),
    }


def runtime_contract_fingerprint(model: str) -> str:
    """Hash the cross-system physical model and retrieval contract."""

    canonical_model = model.removeprefix("deepseek/")
    payload = {
        "memory_model": canonical_model,
        "thinking": "disabled",
        "embedding_model": MEM0_BGE_M3.model,
        "embedding_revision": MEM0_BGE_M3.revision,
        "embedding_dimensions": MEM0_BGE_M3.dimensions,
        "embedding_device": "cpu",
        "retrieval_method": "cosine",
        "candidate_limit": 80,
        "limit": TOP_K,
        "threshold": THRESHOLD,
        "reranker": None,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Append one JSON event."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, default=str))
        file.write("\n")


def write_json(path: Path, value: Any) -> None:
    """Write one inspectable UTF-8 JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    run(parse_args())
