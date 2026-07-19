"""Opt-in Neo4j 5.26 integration tests for storage and recovery."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.memories.zep.storage import (
    GRAPHITI_NEO4J_SCHEMA,
    GRAPHITI_NEO4J_STATEMENTS,
)
from agent_memory.storage import (
    Schema,
    StatementSet,
    StorageCommit,
    StorageDeployment,
    TableDescriptor,
)
from agent_memory.storage.neo4j import (
    EmbeddingSpec,
    Neo4jConnector,
    Neo4jIdentity,
    Neo4jNodeMapping,
)
from agent_memory.storage.neo4j.sink import physical_uuid


neo4j = pytest.importorskip("neo4j")
_NEO4J_URI = os.environ.get("AGENT_MEMORY_NEO4J_URI")
pytestmark = pytest.mark.skipif(
    not _NEO4J_URI,
    reason="set AGENT_MEMORY_NEO4J_URI to run real Neo4j storage tests",
)


class _DeterministicEmbeddingProvider:
    def embed(self, spec: EmbeddingSpec, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1)] * spec.dimensions for index, _ in enumerate(texts)]


class _MinimumMemory(am.Memory):
    log = am.Log({"key": "Key.", "value": "Value."})
    minimum = log.group_by("key").min(column="value", output_col="value")


_MINIMUM_TARGET = (
    TableDescriptor.for_connector("neo4j")
    .schema(
        Schema.new_builder()
        .column("key", "STRING")
        .column("value", "BIGINT")
        .primary_key("key")
        .build()
    )
    .mapping(
        Neo4jNodeMapping(
            label="Phase3Minimum",
            identity=Neo4jIdentity(columns=("key",), kind="phase3_minimum"),
            properties={"value": "value"},
        )
    )
    .build()
)
_MINIMUM_STATEMENTS = StatementSet().add_insert(
    _MINIMUM_TARGET,
    _MinimumMemory.minimum,
)


def _driver() -> object:
    return neo4j.GraphDatabase.driver(
        _NEO4J_URI,
        auth=(
            os.environ.get("AGENT_MEMORY_NEO4J_USER", "neo4j"),
            os.environ.get("AGENT_MEMORY_NEO4J_PASSWORD", "phase3-test"),
        ),
    )


def _clear_namespace(driver: object, namespace: str) -> None:
    with driver.session(database="neo4j") as session:
        session.run(
            "MATCH ()-[r]->() WHERE r.group_id = $namespace DELETE r",
            namespace=namespace,
        ).consume()
        session.run(
            "MATCH (n) WHERE n.group_id = $namespace DETACH DELETE n",
            namespace=namespace,
        ).consume()


def test_real_neo4j_continuous_update_restore_and_continue() -> None:
    driver = _driver()
    namespace = f"phase3-runtime-{uuid4()}"
    connector = Neo4jConnector(
        driver=driver,
        database="neo4j",
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    deployment = StorageDeployment(
        connector=connector,
        statements=_MINIMUM_STATEMENTS,
        namespace=namespace,
    )
    try:
        source = _MinimumMemory(adapter=LotusAdapter(), storage=deployment)
        source.add({"key": "a", "value": 2})
        source.add({"key": "a", "value": 1})
        checkpoint = source._runtime.snapshot_state()

        continued = _MinimumMemory(adapter=LotusAdapter(), storage=deployment)
        continued._runtime.restore_state(checkpoint)
        continued.add({"key": "a", "value": 0})

        recovered = _MinimumMemory(adapter=LotusAdapter(), storage=deployment)
        recovered._runtime.restore_state(checkpoint)
        recovered.add({"key": "a", "value": -1})

        with driver.session(database="neo4j") as session:
            record = session.run(
                "MATCH (n:Phase3Minimum {group_id: $namespace}) "
                "RETURN count(n) AS count, min(n.value) AS value",
                namespace=namespace,
            ).single(strict=True)
        assert record["count"] == 1
        assert record["value"] == -1
        assert recovered._runtime.snapshot_state()["storage_commit"][
            "commit_sequence"
        ] == 3
    finally:
        _clear_namespace(driver, namespace)
        driver.close()


def test_real_neo4j_materializes_all_zep_baseline_sinks() -> None:
    driver = _driver()
    namespace = f"phase3-zep-{uuid4()}"
    connector = Neo4jConnector(
        driver=driver,
        database="neo4j",
        embedding_provider=_DeterministicEmbeddingProvider(),
        schema=GRAPHITI_NEO4J_SCHEMA,
    )
    connector.prepare(GRAPHITI_NEO4J_STATEMENTS)
    episode_id = "episode-a"
    created_at = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)
    entity_ids = ((0, 0), (0, 1))
    rows = (
        pd.DataFrame(
            [
                {
                    "episode_id": episode_id,
                    "content": "Alice likes tea.",
                    "role": "user",
                    "speaker": "Alice",
                    "reference_time": created_at,
                    "source_description": "LOCOMO sample 0, session 1",
                    "created_at": created_at,
                    "add_seq": 0,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "entity_id": entity_id,
                    "name": name,
                    "entity_type": "Entity",
                    "summary": name,
                    "mentions": json.dumps(
                        [
                            {
                                "episode_id": episode_id,
                                "entity_ordinal": ordinal,
                                "name": name,
                                "entity_type": "Entity",
                                "content": "Alice likes tea.",
                                "created_at": created_at.isoformat(),
                                "add_seq": 0,
                            }
                        ]
                    ),
                }
                for ordinal, (entity_id, name) in enumerate(
                    zip(entity_ids, ("Alice", "Tea"), strict=True)
                )
            ]
        ),
        pd.DataFrame(
            [
                {
                    "fact_id": (0, 0),
                    "source_entity_id": entity_ids[0],
                    "target_entity_id": entity_ids[1],
                    "relation_type": "LIKES",
                    "fact": "Alice likes tea.",
                    "valid_at": created_at,
                    "invalid_at": None,
                    "expired_at": None,
                    "provenance": json.dumps(
                        [
                            {
                                "episode_id": episode_id,
                                "fact_ordinal": 0,
                                "created_at": created_at.isoformat(),
                                "content": "Alice likes tea.",
                            }
                        ]
                    ),
                    "created_at": created_at,
                    "add_seq": 0,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "episode_id": episode_id,
                    "entity_ordinal": ordinal,
                    "entity_id": entity_id,
                }
                for ordinal, entity_id in enumerate(entity_ids)
            ]
        ),
    )
    commit = StorageCommit(
        plan_fingerprint="real-zep-sink-test",
        lineage_id=str(uuid4()),
        commit_sequence=1,
        source_row_count=1,
    )
    try:
        with connector.transaction(
            namespace=namespace,
            expected_commit=None,
            next_commit=commit,
        ) as transaction:
            for statement, inserted_rows in zip(
                GRAPHITI_NEO4J_STATEMENTS.statements,
                rows,
                strict=True,
            ):
                transaction.write(
                    statement,
                    inserted_rows=inserted_rows,
                    retracted_rows=inserted_rows.iloc[0:0],
                )

        with driver.session(database="neo4j") as session:
            counts = session.run(
                "MATCH (episode:Episodic {group_id: $namespace}) "
                "OPTIONAL MATCH (entity:Entity {group_id: $namespace}) "
                "WITH count(DISTINCT episode) AS episodes, "
                "count(DISTINCT entity) AS entities "
                "MATCH ()-[fact:RELATES_TO {group_id: $namespace}]->() "
                "WITH episodes, entities, count(fact) AS facts "
                "MATCH ()-[mention:MENTIONS {group_id: $namespace}]->() "
                "RETURN episodes, entities, facts, count(mention) AS mentions",
                namespace=namespace,
            ).single(strict=True)
            fact = session.run(
                "MATCH ()-[fact:RELATES_TO {group_id: $namespace}]->() "
                "RETURN fact.episodes AS episodes, "
                "size(fact.fact_embedding) AS embedding_dimensions, "
                "fact.invalid_at IS NULL AS invalid_at_is_null",
                namespace=namespace,
            ).single(strict=True)
        assert dict(counts) == {
            "episodes": 1,
            "entities": 2,
            "facts": 1,
            "mentions": 2,
        }
        assert fact["episodes"] == [
            physical_uuid(namespace, "episode", (episode_id,))
        ]
        assert fact["embedding_dimensions"] == 1024
        assert fact["invalid_at_is_null"] is True
        assert connector.read_commit(namespace=namespace) == commit
    finally:
        _clear_namespace(driver, namespace)
        driver.close()
