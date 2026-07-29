"""Benchmark tracing decorator for backend-neutral embedding providers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from agent_memory.storage.embedding import EmbeddingProvider, EmbeddingSpec
from agent_memory.tracing.semantic import write_trace_event


class TracingEmbeddingProvider:
    """Record embedding inputs and latency without storing generated vectors."""

    def __init__(self, provider: EmbeddingProvider, *, trace_dir: Path) -> None:
        self._provider = provider
        self._trace_dir = trace_dir

    def embed(
        self,
        spec: EmbeddingSpec,
        texts: list[str],
    ) -> list[list[float]]:
        """Delegate one embedding call and append benchmark trace evidence."""

        trace_id = f"embedding-input-{uuid4().hex}"
        input_path = self._trace_dir / "prompts" / f"{trace_id}.json"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = input_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "embedding": spec.to_dict(),
                    "texts": texts,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, input_path)
        started = perf_counter()
        common = {
            "model": spec.model,
            "revision": spec.revision,
            "source_column": spec.source_column,
            "property_name": spec.property_name,
            "batch_size": len(texts),
            "dimensions": spec.dimensions,
            "normalize": spec.normalize,
            "device": "cpu",
            "input_path": str(input_path.relative_to(self._trace_dir.parent)),
        }
        try:
            vectors = self._provider.embed(spec, texts)
        except Exception as error:
            write_trace_event(
                self._trace_dir,
                operator="embedding",
                event_type="embedding_call",
                payload={
                    **common,
                    "status": "error",
                    "latency_ms": round((perf_counter() - started) * 1000, 3),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            raise
        write_trace_event(
            self._trace_dir,
            operator="embedding",
            event_type="embedding_call",
            payload={
                **common,
                "status": "success",
                "latency_ms": round((perf_counter() - started) * 1000, 3),
                "result_count": len(vectors),
                "result_dimensions": len(vectors[0]) if vectors else 0,
            },
        )
        return vectors


__all__ = ["TracingEmbeddingProvider"]
