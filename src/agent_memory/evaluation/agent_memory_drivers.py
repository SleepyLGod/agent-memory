"""Case-local benchmark drivers for the built-in memory policies."""

from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from agent_memory.adapters.lotus.pair_execution import (
    PAIR_LEFT_ID_COLUMN,
    PAIR_LEFT_TEXT_COLUMN,
    PAIR_RIGHT_ID_COLUMN,
    PAIR_RIGHT_TEXT_COLUMN,
    SEMANTIC_PAIR_EXECUTION_MODES,
    SemanticPairExecutionProfile,
    SemanticPairSite,
    semantic_pair_site_contract,
    semantic_pair_site_id,
)
from agent_memory.evaluation.claude_memory.bindings import event_to_claude_log_row
from agent_memory.evaluation.embedding_trace import TracingEmbeddingProvider
from agent_memory.evaluation.harness import RetrievalOutput
from agent_memory.evaluation.semantic_pair_config import SemanticPairSiteBinding
from agent_memory.evaluation.types import BenchmarkEvent, RetrievalRequest
from agent_memory.evaluation.zep.answering import format_retrieval_context
from agent_memory.policy.retrieval import RetrievalResult
from agent_memory.storage import EmbeddingSpec


BENCHMARK_STRUCTURED_MAX_TOKENS = 32_768
BENCHMARK_LM_NUM_RETRIES = 2
LOTUS_CACHE_MODES = ("disabled", "memory")
SEMANTIC_PAIR_BGE_M3 = EmbeddingSpec(
    source_column="semantic_pair_text",
    property_name="semantic_pair_embedding",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimensions=1024,
    normalize=True,
)


def build_operator_semantic_pair_profiles(
    policy: Any,
    *,
    mode: str,
    operators: tuple[str, ...],
    embedding: EmbeddingSpec,
    embedding_device: str = "cpu",
    top_k: int | None,
    min_similarity: float | None,
) -> dict[str, SemanticPairExecutionProfile]:
    """Bind one physical profile to eligible pair-shaped operator queries."""

    if mode not in SEMANTIC_PAIR_EXECUTION_MODES:
        raise ValueError(
            "semantic pair profile must be one of: "
            + ", ".join(SEMANTIC_PAIR_EXECUTION_MODES)
        )
    if mode == "oracle-only":
        if top_k is not None or min_similarity is not None:
            raise ValueError("oracle-only does not accept semantic pair bounds")
        return {}
    supported = {"sem_join", "sem_groupby"}
    unknown = sorted(set(operators) - supported)
    if unknown:
        raise ValueError(f"unsupported semantic pair operators: {unknown}")

    profiles: dict[str, SemanticPairExecutionProfile] = {}
    sites = inventory_operator_semantic_pair_sites(policy, operators=operators)
    for site in sites.values():
        direction = "left-to-right" if site.operator == "sem_join" else "symmetric"
        profile = _operator_semantic_pair_profile(
            mode=mode,
            direction=direction,
            embedding=embedding,
            embedding_device=embedding_device,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        profiles.update(
            {query_digest_value: profile for query_digest_value in site.query_digests}
        )
    if not profiles:
        raise ValueError(
            f"{mode} found no eligible " + ", ".join(operators) + " queries"
        )
    return profiles


def inventory_operator_semantic_pair_sites(
    policy: Any,
    *,
    operators: tuple[str, ...],
) -> dict[str, SemanticPairSite]:
    """Group differential query copies by their semantic predicate contract."""

    supported = {"sem_join", "sem_groupby"}
    unknown = sorted(set(operators) - supported)
    if unknown:
        raise ValueError(f"unsupported semantic pair operators: {unknown}")

    from agent_memory.tracing.semantic import query_digest

    grouped: dict[str, dict[str, Any]] = {}
    pending = [node.query for node in policy.nodes.values()]
    pending.extend(
        node.maintenance_query
        for node in policy.nodes.values()
        if node.maintenance_query is not None
    )
    visited: set[int] = set()
    while pending:
        query = pending.pop()
        if id(query) in visited:
            continue
        visited.add(id(query))
        pending.extend(query.inputs)
        if query.op not in operators:
            continue
        if query.op == "sem_groupby" and query.params.get("labels"):
            continue
        site_id = semantic_pair_site_id(query)
        contract = semantic_pair_site_contract(query)
        group = grouped.setdefault(
            site_id,
            {
                "operator": query.op,
                "contract": contract,
                "query_digests": set(),
            },
        )
        if group["contract"] != contract:
            raise RuntimeError("semantic pair site digest collision")
        group["query_digests"].add(query_digest(query))

    result: dict[str, SemanticPairSite] = {}
    for site_id, group in sorted(grouped.items()):
        contract = group["contract"]
        predicate_sha256 = site_id.split(":", 1)[1]
        result[site_id] = SemanticPairSite(
            site_id=site_id,
            operator=str(group["operator"]),
            predicate_sha256=predicate_sha256,
            instruction=str(contract["instruction"]),
            semantic_columns=tuple(contract["semantic_columns"]),
            partition_by=tuple(contract["partition_by"]),
            query_digests=tuple(sorted(group["query_digests"])),
        )
    return result


def build_site_semantic_pair_profiles(
    policy: Any,
    *,
    bindings: tuple[SemanticPairSiteBinding, ...],
    operators: tuple[str, ...],
    embedding: EmbeddingSpec,
    embedding_device: str = "cpu",
) -> tuple[
    dict[str, SemanticPairExecutionProfile],
    dict[str, SemanticPairSite],
]:
    """Resolve site bindings into the query-addressed adapter profile map."""

    sites = inventory_operator_semantic_pair_sites(policy, operators=operators)
    unknown = sorted({binding.site_id for binding in bindings} - set(sites))
    if unknown:
        raise ValueError(f"semantic pair site bindings not found in policy: {unknown}")
    profiles: dict[str, SemanticPairExecutionProfile] = {}
    for binding in bindings:
        if binding.mode == "oracle-only":
            continue
        site = sites[binding.site_id]
        direction = "left-to-right" if site.operator == "sem_join" else "symmetric"
        profile = _operator_semantic_pair_profile(
            mode=binding.mode,
            direction=direction,
            embedding=embedding,
            embedding_device=embedding_device,
            top_k=binding.top_k,
            min_similarity=binding.min_similarity,
        )
        profiles.update(
            {query_digest_value: profile for query_digest_value in site.query_digests}
        )
    return profiles, sites


def _operator_semantic_pair_profile(
    *,
    mode: str,
    direction: str,
    embedding: EmbeddingSpec,
    embedding_device: str,
    top_k: int | None,
    min_similarity: float | None,
) -> SemanticPairExecutionProfile:
    return SemanticPairExecutionProfile(
        mode=mode,
        direction=direction,
        left_id_columns=(PAIR_LEFT_ID_COLUMN,),
        right_id_columns=(PAIR_RIGHT_ID_COLUMN,),
        left_text_columns=(PAIR_LEFT_TEXT_COLUMN,),
        right_text_columns=(PAIR_RIGHT_TEXT_COLUMN,),
        embedding=embedding,
        embedding_device=embedding_device,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def _semantic_pair_embedding_contract(
    profiles: dict[str, SemanticPairExecutionProfile],
) -> tuple[EmbeddingSpec, str] | None:
    """Return the one embedding contract shared by configured pair profiles."""

    if not profiles:
        return None
    embeddings = {profile.embedding for profile in profiles.values()}
    devices = {profile.embedding_device for profile in profiles.values()}
    if None in embeddings or len(embeddings) != 1 or len(devices) != 1:
        raise ValueError(
            "semantic pair profiles must share one embedding and device"
        )
    embedding = next(iter(embeddings))
    assert embedding is not None
    return embedding, next(iter(devices))


def _validate_lotus_cache_mode(mode: str) -> None:
    if mode not in LOTUS_CACHE_MODES:
        raise ValueError(
            "lotus_cache_mode must be one of: " + ", ".join(LOTUS_CACHE_MODES)
        )


def build_mem0_semantic_pair_profiles(
    memory_type: type[Any],
    *,
    mode: str,
    embedding: EmbeddingSpec,
    embedding_device: str = "cpu",
    top_k: int | None,
    min_similarity: float | None,
) -> dict[str, SemanticPairExecutionProfile]:
    """Bind a physical profile to Mem0's unique pair-shaped semantic filter."""

    if mode not in SEMANTIC_PAIR_EXECUTION_MODES:
        raise ValueError(
            "semantic pair profile must be one of: "
            + ", ".join(SEMANTIC_PAIR_EXECUTION_MODES)
        )
    if mode == "oracle-only":
        if top_k is not None or min_similarity is not None:
            raise ValueError("oracle-only does not accept semantic pair bounds")
        return {}

    from agent_memory.planner import PolicyDifferentiator
    from agent_memory.tracing.semantic import query_digest

    policy = PolicyDifferentiator().differentiate(memory_type.spec())
    filters = {
        query_digest(node.query): node.query
        for node in policy.nodes.values()
        if node.query.op == "sem_filter"
    }
    if len(filters) != 1:
        raise ValueError(
            f"Mem0 {mode} requires exactly one semantic filter"
        )
    digest = next(iter(filters))
    return {
        digest: SemanticPairExecutionProfile(
            mode=mode,
            direction="right-to-left",
            left_id_columns=("_row_id:earlier", "_memory_ordinal:earlier"),
            right_id_columns=("_row_id:later", "_memory_ordinal:later"),
            left_text_columns=("memory:earlier",),
            right_text_columns=("memory:later",),
            embedding=embedding,
            embedding_device=embedding_device,
            top_k=top_k,
            min_similarity=min_similarity,
        )
    }


def _view_counts(memory: Any) -> dict[str, int]:
    runtime = getattr(memory, "_runtime", None)
    state = getattr(runtime, "_state", None)
    if not isinstance(state, dict):
        return {}
    return {
        f"{name}_rows": len(frame)
        for name, frame in state.items()
        if name != "log" and isinstance(frame, pd.DataFrame)
    }


def _claude_memory_shape(memory: Any) -> dict[str, int]:
    runtime = getattr(memory, "_runtime", None)
    state = getattr(runtime, "_state", None)
    if not isinstance(state, dict):
        return {}
    topics = state.get("topics")
    catalog = state.get("catalog")
    metrics = _view_counts(memory)
    if isinstance(topics, pd.DataFrame):
        names = (
            [str(value).strip() for value in topics["name"].dropna()]
            if "name" in topics.columns
            else []
        )
        counts = pd.Series(names).value_counts() if names else pd.Series(dtype=int)
        metrics.update(
            {
                "topic_unique_name_count": len(set(names)),
                "topic_duplicate_name_count": int((counts > 1).sum()),
                "topic_duplicate_extra_rows": int((counts - 1).clip(lower=0).sum()),
                "topic_body_characters": int(
                    topics["body"].fillna("").astype(str).str.len().sum()
                )
                if "body" in topics.columns
                else 0,
            }
        )
    if isinstance(catalog, pd.DataFrame):
        metrics["catalog_rows"] = len(catalog)
    return metrics


def _records(frame: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    return tuple(frame.to_dict(orient="records"))


def _generative_trace_count(trace_dir: Path) -> int:
    events_path = trace_dir / "events.jsonl"
    if not events_path.is_file():
        return 0
    count = 0
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") in {
                "llm_call",
                "llm_call_finish",
                "llm_call_error",
                "provider_usage",
            }:
                count += 1
    return count


def _provider_usage_trace_count(trace_dir: Path) -> int:
    """Count provider attempts recorded in the semantic trace."""

    events_path = trace_dir / "events.jsonl"
    if not events_path.is_file():
        return 0
    with events_path.open(encoding="utf-8") as handle:
        return sum(
            1
            for line in handle
            if line.strip() and json.loads(line).get("event_type") == "provider_usage"
        )


def _save_runtime_state(memory: Any, directory: Path) -> dict[str, Any]:
    runtime = getattr(memory, "_runtime", None)
    if runtime is None or not callable(getattr(runtime, "snapshot_state", None)):
        raise TypeError("benchmark memory does not expose runtime checkpoint state")
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = runtime.snapshot_state()
    (directory / "runtime.pkl").write_bytes(pickle.dumps(snapshot))
    return {
        "format": "agent-memory-runtime-pickle:v1",
        "schema_version": snapshot.get("schema_version"),
    }


def _restore_runtime_state(memory: Any, directory: Path) -> None:
    runtime = getattr(memory, "_runtime", None)
    if runtime is None or not callable(getattr(runtime, "restore_state", None)):
        raise TypeError("benchmark memory does not expose runtime restore state")
    snapshot = pickle.loads((directory / "runtime.pkl").read_bytes())
    if not isinstance(snapshot, dict):
        raise TypeError("benchmark runtime checkpoint must contain a mapping")
    runtime.restore_state(snapshot)


def event_to_zep_log_row(event: BenchmarkEvent) -> dict[str, str]:
    """Map a canonical benchmark event to the baseline Zep log schema."""

    if not event.timestamp:
        raise ValueError("Zep benchmark events require an explicit timestamp")
    return {
        "content": f"{event.speaker}: {event.text}",
        "role": event.speaker,
        "speaker": event.speaker,
        "reference_time": event.timestamp,
        "source_description": f"{event.sample_id} / {event.event_id}",
    }


def event_to_mem0_log_row(event: BenchmarkEvent) -> dict[str, str]:
    """Map one canonical event to the shared Native/Agent Mem0 input."""

    if not event.timestamp:
        raise ValueError("Mem0 benchmark events require an explicit timestamp")
    role = event.speaker if event.speaker in {"user", "assistant"} else "user"
    content = (
        event.text
        if event.speaker in {"user", "assistant"}
        else f"{event.speaker}: {event.text}"
    )
    caption = event.metadata.get("blip_caption")
    if isinstance(caption, str) and caption.strip():
        content += f"\n(description of attached image: {caption.strip()})"
    return {
        "role": role,
        "content": content,
        "observation_date": event.timestamp,
    }


class ClaudeMemoryDriver:
    """Execute one isolated case with the native ClaudeMemory policy."""

    system_id = "claude-memory"

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    def add(self, event: BenchmarkEvent) -> dict[str, int]:
        """Append one canonical event through ClaudeMemory's source schema."""

        self._memory.add(event_to_claude_log_row(event))
        return _claude_memory_shape(self._memory)

    def finish_session(self, session_id: str) -> dict[str, Any]:
        """ClaudeMemory maintenance has no separate session-finalization step."""

        del session_id
        return {}

    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput:
        """Run ClaudeMemory's declared semantic top-k retrieval unchanged."""

        frame = self._memory.query(request.query_text)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("ClaudeMemory.query must return a DataFrame")
        rows = _records(frame)
        context = "\n\n".join(
            f"## {row.get('name', 'Memory')}\n{row.get('body', '')}" for row in rows
        )
        return RetrievalOutput(
            context=context,
            channels={"memory": rows},
            metrics={"row_count": len(rows)},
        )

    def close(self) -> None:
        """Release no-op in-memory case resources."""

    def save_state(self, directory: Path) -> dict[str, Any]:
        """Persist the runtime snapshot inside the trusted local benchmark run."""

        return _save_runtime_state(self._memory, directory)

    def restore_state(
        self,
        directory: Path,
        completed_events: tuple[BenchmarkEvent, ...],
    ) -> None:
        """Restore runtime state without replaying completed source events."""

        del completed_events
        _restore_runtime_state(self._memory, directory)


class ZepMemoryDriver:
    """Execute one isolated case with ZepMemory and its native graph retrieval."""

    system_id = "zep-memory"

    def __init__(self, memory: Any, *, trace_dir: Path) -> None:
        self._memory = memory
        self._trace_dir = trace_dir

    def add(self, event: BenchmarkEvent) -> dict[str, int]:
        """Append one canonical event through ZepMemory's source schema."""

        self._memory.add(event_to_zep_log_row(event))
        return _view_counts(self._memory)

    def finish_session(self, session_id: str) -> dict[str, Any]:
        """ZepMemory maintenance is fully expressed by each differential add."""

        del session_id
        return {}

    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput:
        """Run the declared entity/fact retrieval DAG without an answer model."""

        before = _generative_trace_count(self._trace_dir)
        result = self._memory.query(request.query_text)
        after = _generative_trace_count(self._trace_dir)
        if after != before:
            raise RuntimeError("ZepMemory retrieval made a generative LLM call")
        if not isinstance(result, RetrievalResult):
            raise TypeError("ZepMemory.query must return RetrievalResult")
        return RetrievalOutput(
            context=format_retrieval_context(result),
            channels={
                name: _records(frame) for name, frame in result.channels.items()
            },
            metrics={**dict(result.metrics), "generative_llm_calls": 0},
        )

    def close(self) -> None:
        """Leave the factory-owned shared Neo4j connector open."""

    def save_state(self, directory: Path) -> dict[str, Any]:
        """Persist runtime and storage marker state at a session boundary."""

        return _save_runtime_state(self._memory, directory)

    def restore_state(
        self,
        directory: Path,
        completed_events: tuple[BenchmarkEvent, ...],
    ) -> None:
        """Restore runtime state and reconcile its storage-backed namespace."""

        del completed_events
        _restore_runtime_state(self._memory, directory)


class Mem0MemoryDriver:
    """Execute one isolated case with Mem0 Base storage-backed retrieval."""

    system_id = "mem0-memory"

    def __init__(self, memory: Any, *, connector: Any, trace_dir: Path) -> None:
        self._memory = memory
        self._connector = connector
        self._trace_dir = trace_dir
        self._closed = False

    def add(self, event: BenchmarkEvent) -> dict[str, int]:
        """Append one canonical event through the Mem0 logical source schema."""

        self._memory.add(event_to_mem0_log_row(event))
        return _view_counts(self._memory)

    def finish_session(self, session_id: str) -> dict[str, Any]:
        """Mem0 maintenance is fully represented by each differential add."""

        del session_id
        return {}

    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput:
        """Run indexed cosine retrieval without a generative model call."""

        before = _generative_trace_count(self._trace_dir)
        result = self._memory.query(request.query_text)
        after = _generative_trace_count(self._trace_dir)
        if after != before:
            raise RuntimeError("Mem0Memory retrieval made a generative LLM call")
        if not isinstance(result, RetrievalResult):
            raise TypeError("Mem0Memory.query must return RetrievalResult")
        frame = result.channels.get("memories")
        rows = () if frame is None else _records(frame)
        context = "\n".join(
            f"- [{row.get('attributed_to', 'unknown')}] {row.get('memory', '')}"
            for row in rows
        )
        return RetrievalOutput(
            context=context,
            channels={"memories": rows},
            metrics={**dict(result.metrics), "generative_llm_calls": 0},
        )

    def save_state(self, directory: Path) -> dict[str, Any]:
        """Persist runtime state and its published Qdrant marker."""

        return _save_runtime_state(self._memory, directory)

    def restore_state(
        self,
        directory: Path,
        completed_events: tuple[BenchmarkEvent, ...],
    ) -> None:
        """Restore runtime state and reconcile the case-local Qdrant path."""

        del completed_events
        _restore_runtime_state(self._memory, directory)

    def close(self) -> None:
        """Release the case-local embedded Qdrant filesystem owner."""

        if not self._closed:
            self._connector.close()
            self._closed = True


class Mem0MemoryEnhancedDriver(Mem0MemoryDriver):
    """Execute the Mem0 additive view with generative semantic retrieval."""

    system_id = "mem0-enhanced"

    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput:
        """Rank the materialized memory view through the declared sem_topk."""

        before = _provider_usage_trace_count(self._trace_dir)
        frame = self._memory.query(request.query_text)
        after = _provider_usage_trace_count(self._trace_dir)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Mem0MemoryEnhanced.query must return a DataFrame")
        rows = _records(frame)
        context = "\n".join(
            f"- [{row.get('attributed_to', 'unknown')}] {row.get('memory', '')}"
            for row in rows
        )
        return RetrievalOutput(
            context=context,
            channels={"memories": rows},
            metrics={
                "row_count": len(rows),
                "provider_usage_events": after - before,
            },
        )


class ClaudeMemoryDriverFactory:
    """Create one isolated ClaudeMemory runtime per benchmark case."""

    def __init__(
        self,
        *,
        model_id: str = "deepseek/deepseek-v4-flash",
        grouped_agg_rule: str = "rule-all-group",
        sem_topk_method: str = "pairwise-naive",
        sem_groupby_pair_batch_size: int | None = None,
        sem_groupby_pair_batch_retries: int = 0,
        semantic_pair_profiles: dict[str, SemanticPairExecutionProfile] | None = None,
        semantic_trace_snapshot_mode: str = "compact",
        lotus_cache_mode: str = "disabled",
        thinking_enabled: bool = True,
    ) -> None:
        from agent_memory.adapters.lotus.context import SEM_TOPK_METHODS
        from agent_memory.planner.rules import GROUPED_AGG_RULES

        if grouped_agg_rule not in GROUPED_AGG_RULES:
            raise ValueError(
                "grouped_agg_rule must be one of: " + ", ".join(GROUPED_AGG_RULES)
            )
        if sem_topk_method not in SEM_TOPK_METHODS:
            raise ValueError(
                "sem_topk_method must be one of: " + ", ".join(SEM_TOPK_METHODS)
            )
        self.model_id = model_id
        self.grouped_agg_rule = grouped_agg_rule
        self.sem_topk_method = sem_topk_method
        self.sem_groupby_pair_batch_size = sem_groupby_pair_batch_size
        self.sem_groupby_pair_batch_retries = sem_groupby_pair_batch_retries
        self.semantic_pair_profiles = dict(semantic_pair_profiles or {})
        _semantic_pair_embedding_contract(self.semantic_pair_profiles)
        self.semantic_trace_snapshot_mode = semantic_trace_snapshot_mode
        _validate_lotus_cache_mode(lotus_cache_mode)
        self.lotus_cache_mode = lotus_cache_mode
        self.thinking_enabled = thinking_enabled

    def __call__(
        self,
        case_id: str,
        state_dir: Path,
        trace_dir: Path,
    ) -> ClaudeMemoryDriver:
        del case_id, state_dir
        import agent_memory as am
        from agent_memory.adapters.lotus import LotusAdapter
        from agent_memory.adapters.lotus.context import LotusExecutionConfig
        from agent_memory.planner import DifferentialRules, PolicyDifferentiator
        from agent_memory.runtime import MemoryRuntime
        from agent_memory.storage import SentenceTransformerEmbeddingProvider

        pair_embedding_provider = None
        embedding_contract = _semantic_pair_embedding_contract(
            self.semantic_pair_profiles
        )
        if embedding_contract is not None:
            embedding, device = embedding_contract
            pair_embedding_provider = TracingEmbeddingProvider(
                SentenceTransformerEmbeddingProvider(
                    embedding,
                    device=device,
                    dependency_extra="zep or mem0",
                ),
                trace_dir=trace_dir,
            )

        adapter = LotusAdapter(
            model=self.model_id,
            config=LotusExecutionConfig(
                semantic_trace_dir=trace_dir,
                lm_num_retries=BENCHMARK_LM_NUM_RETRIES,
                lm_model_kwargs={
                    "extra_body": {
                        "thinking": {
                            "type": "enabled" if self.thinking_enabled else "disabled"
                        }
                    }
                },
                lm_enable_cache=self.lotus_cache_mode == "memory",
                structured_max_tokens=BENCHMARK_STRUCTURED_MAX_TOKENS,
                sem_topk_method=self.sem_topk_method,
                sem_groupby_pair_batch_size=self.sem_groupby_pair_batch_size,
                sem_groupby_pair_batch_retries=self.sem_groupby_pair_batch_retries,
                semantic_pair_profiles=self.semantic_pair_profiles,
                semantic_trace_snapshot_mode=self.semantic_trace_snapshot_mode,
            ),
            pair_embedding_provider=pair_embedding_provider,
        )
        memory = am.ClaudeMemory(adapter=adapter)
        policy = PolicyDifferentiator(
            rules=DifferentialRules(grouped_agg_rule=self.grouped_agg_rule)
        ).differentiate(am.ClaudeMemory.spec())
        memory._runtime = MemoryRuntime(policy, adapter=adapter)
        return ClaudeMemoryDriver(memory)


class ZepMemoryDriverFactory:
    """Create isolated ZepMemory namespaces over one shared Neo4j connector."""

    def __init__(
        self,
        *,
        connector: Any,
        base_namespace: str,
        model_id: str = "deepseek/deepseek-v4-flash",
        grouped_agg_rule: str = "rule-re-group",
        sem_groupby_pair_batch_size: int | None = None,
        sem_groupby_pair_batch_retries: int = 0,
        semantic_pair_profiles: dict[str, SemanticPairExecutionProfile] | None = None,
        embedding_device: str = "cpu",
        semantic_trace_snapshot_mode: str = "compact",
        lotus_cache_mode: str = "disabled",
        thinking_enabled: bool = True,
        neo4j_image: str,
        neo4j_image_digest: str,
    ) -> None:
        from agent_memory.planner.rules import GROUPED_AGG_RULES

        if not base_namespace:
            raise ValueError("Zep benchmark base_namespace must be non-empty")
        if grouped_agg_rule not in GROUPED_AGG_RULES:
            raise ValueError(
                "grouped_agg_rule must be one of: " + ", ".join(GROUPED_AGG_RULES)
            )
        self.connector = connector
        self.base_namespace = base_namespace
        self.model_id = model_id
        self.grouped_agg_rule = grouped_agg_rule
        self.sem_groupby_pair_batch_size = sem_groupby_pair_batch_size
        self.sem_groupby_pair_batch_retries = sem_groupby_pair_batch_retries
        self.semantic_pair_profiles = dict(semantic_pair_profiles or {})
        embedding_contract = _semantic_pair_embedding_contract(
            self.semantic_pair_profiles
        )
        if embedding_device not in {"cpu", "cuda"}:
            raise ValueError("Zep embedding_device must be 'cpu' or 'cuda'")
        if embedding_contract is not None and embedding_contract[1] != embedding_device:
            raise ValueError(
                "Zep semantic pair profile and embedding devices do not match"
            )
        self._embedding_provider = getattr(connector, "embedding_provider", None)
        if embedding_contract is not None and self._embedding_provider is None:
            raise ValueError("Zep search-filter requires an embedding provider")
        provider_device = getattr(self._embedding_provider, "device", None)
        if provider_device is not None and provider_device != embedding_device:
            raise ValueError(
                "Zep connector and configured embedding devices do not match"
            )
        self.embedding_device = embedding_device
        self.semantic_trace_snapshot_mode = semantic_trace_snapshot_mode
        _validate_lotus_cache_mode(lotus_cache_mode)
        self.lotus_cache_mode = lotus_cache_mode
        self.thinking_enabled = thinking_enabled
        self.neo4j_image = neo4j_image
        self.neo4j_image_digest = neo4j_image_digest
        self._closed = False

    @classmethod
    def from_environment(
        cls,
        *,
        base_namespace: str,
        model_id: str = "deepseek/deepseek-v4-flash",
        grouped_agg_rule: str = "rule-re-group",
        sem_groupby_pair_batch_size: int | None = None,
        sem_groupby_pair_batch_retries: int = 0,
        semantic_pair_profiles: dict[str, SemanticPairExecutionProfile] | None = None,
        embedding_device: str = "cpu",
        semantic_trace_snapshot_mode: str = "compact",
        lotus_cache_mode: str = "disabled",
        thinking_enabled: bool = True,
    ) -> ZepMemoryDriverFactory:
        """Create the pinned Graphiti-compatible deployment connector."""

        neo4j_image = os.getenv("AGENT_MEMORY_NEO4J_IMAGE")
        neo4j_image_digest = os.getenv("AGENT_MEMORY_NEO4J_IMAGE_DIGEST")
        if not neo4j_image or not neo4j_image_digest:
            raise RuntimeError(
                "Zep benchmarks require AGENT_MEMORY_NEO4J_IMAGE and "
                "AGENT_MEMORY_NEO4J_IMAGE_DIGEST"
            )

        from agent_memory.memories.zep.storage import (
            GRAPHITI_BGE_M3,
            GRAPHITI_NEO4J_SCHEMA,
        )
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
            embedding_provider=SentenceTransformerEmbeddingProvider(
                GRAPHITI_BGE_M3,
                device=embedding_device,
            ),
            reranker_provider=SentenceTransformerCrossEncoderProvider(),
            schema=GRAPHITI_NEO4J_SCHEMA,
        )
        return cls(
            connector=connector,
            base_namespace=base_namespace,
            model_id=model_id,
            grouped_agg_rule=grouped_agg_rule,
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            semantic_pair_profiles=semantic_pair_profiles,
            embedding_device=embedding_device,
            semantic_trace_snapshot_mode=semantic_trace_snapshot_mode,
            lotus_cache_mode=lotus_cache_mode,
            thinking_enabled=thinking_enabled,
            neo4j_image=neo4j_image,
            neo4j_image_digest=neo4j_image_digest,
        )

    def runtime_provenance(self) -> dict[str, Any]:
        """Return physical Neo4j evidence before benchmark insertion begins."""

        return {
            "connector": "neo4j",
            "image": self.neo4j_image,
            "image_digest": self.neo4j_image_digest,
            "server_version": self.connector.server_version(),
            "driver_version": version("neo4j"),
            "embedding_device": self.embedding_device,
            "embedding_runtime_version": version("sentence-transformers"),
        }

    def __call__(
        self,
        case_id: str,
        state_dir: Path,
        trace_dir: Path,
    ) -> ZepMemoryDriver:
        if self._closed:
            raise RuntimeError("Zep benchmark driver factory is closed")
        import agent_memory as am
        from agent_memory.adapters.lotus import LotusAdapter
        from agent_memory.adapters.lotus.context import LotusExecutionConfig
        from agent_memory.memories.zep.storage import GRAPHITI_NEO4J_STATEMENTS
        from agent_memory.planner import DifferentialRules, PolicyDifferentiator
        from agent_memory.runtime import MemoryRuntime
        from agent_memory.storage import StorageDeployment

        case_digest = sha256(case_id.encode("utf-8")).hexdigest()[:16]
        namespace = f"{self.base_namespace}-{case_digest}-{state_dir.name}"
        traced_embedding_provider = None
        if self._embedding_provider is not None:
            traced_embedding_provider = TracingEmbeddingProvider(
                self._embedding_provider,
                trace_dir=trace_dir,
            )
            self.connector.embedding_provider = traced_embedding_provider
        storage = StorageDeployment(
            connector=self.connector,
            statements=GRAPHITI_NEO4J_STATEMENTS,
            namespace=namespace,
        )
        adapter = LotusAdapter(
            model=self.model_id,
            config=LotusExecutionConfig(
                semantic_trace_dir=trace_dir,
                lm_num_retries=BENCHMARK_LM_NUM_RETRIES,
                lm_model_kwargs={
                    "extra_body": {
                        "thinking": {
                            "type": "enabled" if self.thinking_enabled else "disabled"
                        }
                    }
                },
                lm_enable_cache=self.lotus_cache_mode == "memory",
                structured_max_tokens=BENCHMARK_STRUCTURED_MAX_TOKENS,
                sem_groupby_pair_batch_size=self.sem_groupby_pair_batch_size,
                sem_groupby_pair_batch_retries=self.sem_groupby_pair_batch_retries,
                semantic_pair_profiles=self.semantic_pair_profiles,
                semantic_trace_snapshot_mode=self.semantic_trace_snapshot_mode,
            ),
            pair_embedding_provider=(
                traced_embedding_provider if self.semantic_pair_profiles else None
            ),
        )
        policy = PolicyDifferentiator(
            rules=DifferentialRules(grouped_agg_rule=self.grouped_agg_rule)
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
        return ZepMemoryDriver(
            memory,
            trace_dir=trace_dir,
        )

    def close(self) -> None:
        """Close the one factory-owned connector exactly once."""

        if not self._closed:
            self.connector.close()
            self._closed = True


class Mem0MemoryDriverFactory:
    """Create one case-local Mem0 runtime and embedded Qdrant deployment."""

    policy_name = "Mem0Memory"
    statements_name = "MEM0_QDRANT_STATEMENTS"
    driver_type = Mem0MemoryDriver

    def __init__(
        self,
        *,
        base_namespace: str,
        model_id: str = "deepseek/deepseek-v4-flash",
        sem_groupby_pair_batch_size: int | None = None,
        sem_groupby_pair_batch_retries: int = 0,
        semantic_pair_profiles: dict[str, SemanticPairExecutionProfile] | None = None,
        embedding_device: str = "cpu",
        semantic_trace_snapshot_mode: str = "compact",
        lotus_cache_mode: str = "disabled",
        thinking_enabled: bool = False,
    ) -> None:
        if not base_namespace:
            raise ValueError("Mem0 benchmark base_namespace must be non-empty")
        self.base_namespace = base_namespace
        self.model_id = model_id
        self.sem_groupby_pair_batch_size = sem_groupby_pair_batch_size
        self.sem_groupby_pair_batch_retries = sem_groupby_pair_batch_retries
        self.semantic_pair_profiles = dict(semantic_pair_profiles or {})
        if embedding_device not in {"cpu", "cuda"}:
            raise ValueError("Mem0 embedding_device must be 'cpu' or 'cuda'")
        self.embedding_device = embedding_device
        self.semantic_trace_snapshot_mode = semantic_trace_snapshot_mode
        _validate_lotus_cache_mode(lotus_cache_mode)
        self.lotus_cache_mode = lotus_cache_mode
        self.thinking_enabled = thinking_enabled
        self.sem_topk_method = "pairwise-naive"

    def runtime_provenance(self) -> dict[str, Any]:
        """Return the physical Mem0 Base storage profile."""

        from agent_memory.memories.mem0.storage import MEM0_BGE_M3
        torch = import_module("torch")
        torch_version = getattr(torch, "version", None)

        return {
            "connector": "qdrant",
            "mode": "embedded-local-single-owner",
            "driver_version": version("qdrant-client"),
            "embedding_runtime_version": version("sentence-transformers"),
            "embedding_model": MEM0_BGE_M3.model,
            "embedding_revision": MEM0_BGE_M3.revision,
            "dimensions": MEM0_BGE_M3.dimensions,
            "device": self.embedding_device,
            "torch_version": version("torch"),
            "torch_cuda_version": getattr(torch_version, "cuda", None),
            "bm25_enabled": False,
            "entity_boost_enabled": False,
            "reranker_enabled": False,
        }

    def __call__(
        self,
        case_id: str,
        state_dir: Path,
        trace_dir: Path,
    ) -> Mem0MemoryDriver:
        import agent_memory as am
        from agent_memory.adapters.lotus import LotusAdapter
        from agent_memory.adapters.lotus.context import LotusExecutionConfig
        from agent_memory.memories.mem0 import storage as mem0_storage
        from agent_memory.memories.mem0.storage import MEM0_BGE_M3
        from agent_memory.planner import PolicyDifferentiator
        from agent_memory.runtime import MemoryRuntime
        from agent_memory.storage import StorageDeployment
        from agent_memory.storage.qdrant import (
            QdrantConnector,
            SentenceTransformerEmbeddingProvider,
        )

        case_digest = sha256(case_id.encode("utf-8")).hexdigest()[:16]
        namespace = f"{self.base_namespace}-{case_digest}"
        embedding_provider = TracingEmbeddingProvider(
            SentenceTransformerEmbeddingProvider(
                MEM0_BGE_M3,
                device=self.embedding_device,
                dependency_extra="mem0",
            ),
            trace_dir=trace_dir,
        )
        connector = QdrantConnector(
            path=state_dir / "qdrant",
            embedding_provider=embedding_provider,
        )
        try:
            memory_type = getattr(am, self.policy_name)
            statements = getattr(mem0_storage, self.statements_name)
            storage = StorageDeployment(
                connector=connector,
                statements=statements,
                namespace=namespace,
            )
            adapter = LotusAdapter(
                model=self.model_id,
                config=LotusExecutionConfig(
                    semantic_trace_dir=trace_dir,
                    lm_num_retries=BENCHMARK_LM_NUM_RETRIES,
                    lm_model_kwargs={
                        "extra_body": {
                            "thinking": {
                                "type": (
                                    "enabled"
                                    if self.thinking_enabled
                                    else "disabled"
                                )
                            }
                        }
                    },
                    lm_enable_cache=self.lotus_cache_mode == "memory",
                    structured_max_tokens=BENCHMARK_STRUCTURED_MAX_TOKENS,
                    sem_topk_method=self.sem_topk_method,
                    sem_groupby_pair_batch_size=self.sem_groupby_pair_batch_size,
                    sem_groupby_pair_batch_retries=self.sem_groupby_pair_batch_retries,
                    semantic_pair_profiles=self.semantic_pair_profiles,
                    semantic_trace_snapshot_mode=self.semantic_trace_snapshot_mode,
                ),
                pair_embedding_provider=embedding_provider,
            )
            policy = PolicyDifferentiator().differentiate(
                memory_type.spec(),
                statements=storage.statements,
            )
            memory = memory_type(adapter=adapter)
            memory._runtime = MemoryRuntime(
                policy,
                adapter=adapter,
                storage=storage,
            )
        except BaseException:
            connector.close()
            raise
        return self.driver_type(
            memory,
            connector=connector,
            trace_dir=trace_dir,
        )


class Mem0MemoryEnhancedDriverFactory(Mem0MemoryDriverFactory):
    """Create storage-compatible Mem0Enhanced runtimes for benchmark reuse."""

    policy_name = "Mem0MemoryEnhanced"
    statements_name = "MEM0_ENHANCED_QDRANT_STATEMENTS"
    driver_type = Mem0MemoryEnhancedDriver

    def __init__(
        self,
        *,
        base_namespace: str,
        model_id: str = "deepseek/deepseek-v4-flash",
        sem_topk_method: str = "pairwise-quick",
        sem_groupby_pair_batch_size: int | None = None,
        sem_groupby_pair_batch_retries: int = 0,
        semantic_pair_profiles: dict[str, SemanticPairExecutionProfile] | None = None,
        embedding_device: str = "cpu",
        semantic_trace_snapshot_mode: str = "compact",
        lotus_cache_mode: str = "disabled",
        thinking_enabled: bool = False,
    ) -> None:
        from agent_memory.adapters.lotus.context import SEM_TOPK_METHODS

        if sem_topk_method not in SEM_TOPK_METHODS:
            raise ValueError(
                "sem_topk_method must be one of: " + ", ".join(SEM_TOPK_METHODS)
            )
        super().__init__(
            base_namespace=base_namespace,
            model_id=model_id,
            sem_groupby_pair_batch_size=sem_groupby_pair_batch_size,
            sem_groupby_pair_batch_retries=sem_groupby_pair_batch_retries,
            semantic_pair_profiles=semantic_pair_profiles,
            embedding_device=embedding_device,
            semantic_trace_snapshot_mode=semantic_trace_snapshot_mode,
            lotus_cache_mode=lotus_cache_mode,
            thinking_enabled=thinking_enabled,
        )
        self.sem_topk_method = sem_topk_method


__all__ = [
    "ClaudeMemoryDriver",
    "ClaudeMemoryDriverFactory",
    "Mem0MemoryDriver",
    "Mem0MemoryDriverFactory",
    "Mem0MemoryEnhancedDriver",
    "Mem0MemoryEnhancedDriverFactory",
    "SEMANTIC_PAIR_BGE_M3",
    "LOTUS_CACHE_MODES",
    "ZepMemoryDriver",
    "ZepMemoryDriverFactory",
    "build_mem0_semantic_pair_profiles",
    "build_operator_semantic_pair_profiles",
    "build_site_semantic_pair_profiles",
    "inventory_operator_semantic_pair_sites",
    "event_to_mem0_log_row",
    "event_to_zep_log_row",
]
