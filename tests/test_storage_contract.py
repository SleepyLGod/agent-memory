"""Tests for declarative storage plans and transactional materialization."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from types import MappingProxyType
from typing import Any

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter
from agent_memory.planner import PolicyDifferentiator
from agent_memory.memories import ZepMemory
from agent_memory.memories.zep.storage import GRAPHITI_NEO4J_STATEMENTS
from agent_memory.policy.schema import output_columns
from agent_memory.storage import (
    Schema,
    StatementSet,
    StorageCommit,
    StorageConflictError,
    StorageDeployment,
    TableDescriptor,
)
from agent_memory.storage.statements import InsertStatement
from agent_memory.storage.neo4j import (
    Neo4jNodeMapping,
    Neo4jRelationshipMapping,
)


def _target() -> TableDescriptor:
    schema = (
        Schema.new_builder()
        .column("value", "BIGINT")
        .primary_key("value")
        .build()
    )
    return (
        TableDescriptor.for_connector("recording")
        .schema(schema)
        .option("kind", "table")
        .build()
    )


def test_storage_descriptors_are_immutable_and_serializable() -> None:
    target = _target()

    assert target.schema.primary_key == ("value",)
    assert target.options == {"kind": "table"}
    assert isinstance(target.options, MappingProxyType)
    with pytest.raises(TypeError):
        target.options["kind"] = "node"  # type: ignore[index]
    assert target.to_dict() == {
        "connector": "recording",
        "schema": {
            "columns": [{"name": "value", "data_type": "BIGINT"}],
            "primary_key": ["value"],
        },
        "options": {"kind": "table"},
    }


@pytest.mark.parametrize(
    "option_name",
    [
        "uri",
        "url",
        "host",
        "port",
        "user",
        "username",
        "password",
        "passwd",
        "token",
        "secret",
        "auth",
        "credential",
        "credentials",
        "api_key",
        "access_key",
        "private_key",
        "tls",
        "ssl",
    ],
)
def test_table_descriptor_rejects_runtime_deployment_options(
    option_name: str,
) -> None:
    schema = _target().schema

    with pytest.raises(ValueError, match="deployment or secret"):
        TableDescriptor(
            connector="recording",
            schema=schema,
            options={option_name: "sensitive-value"},
        )


def test_table_descriptor_allows_property_mapping_with_sensitive_field_name() -> None:
    target = (
        TableDescriptor.for_connector("recording")
        .schema(_target().schema)
        .option("property.password", "password_hash")
        .build()
    )

    assert target.options["property.password"] == "password_hash"


def test_schema_builder_rejects_invalid_columns_and_primary_keys() -> None:
    with pytest.raises(ValueError, match="already exists"):
        Schema.new_builder().column("value", "BIGINT").column(
            "value", "STRING"
        )
    with pytest.raises(ValueError, match="unknown column"):
        Schema.new_builder().column("value", "BIGINT").primary_key(
            "missing"
        )
    with pytest.raises(ValueError, match="at least one column"):
        Schema.new_builder().build()


def test_statement_set_add_insert_returns_a_new_plan() -> None:
    log = am.Log({"value": "Value."})
    relation = log.select(["value"])
    original = StatementSet()

    updated = original.add_insert(_target(), relation)

    assert original.statements == ()
    assert len(updated.statements) == 1
    assert updated.statements[0].statement_id == "sink_0000"
    assert updated.statements[0].query == relation.expr
    assert json.loads(json.dumps(updated.to_dict()))["statements"][0][
        "statement_id"
    ] == "sink_0000"


def test_insert_statement_rejects_invalid_direct_construction() -> None:
    query = am.Log({"value": "Value."}).select(["value"]).expr

    with pytest.raises(ValueError, match="non-empty string"):
        InsertStatement(statement_id="", target=_target(), query=query)
    with pytest.raises(TypeError, match="TableDescriptor"):
        InsertStatement(
            statement_id="sink_0000",
            target=object(),  # type: ignore[arg-type]
            query=query,
        )
    with pytest.raises(TypeError, match="QueryExpr"):
        InsertStatement(
            statement_id="sink_0000",
            target=_target(),
            query=object(),  # type: ignore[arg-type]
        )


def test_statement_set_skips_an_existing_generated_identifier() -> None:
    relation = am.Log({"value": "Value."}).select(["value"])
    existing = InsertStatement(
        statement_id="sink_0001",
        target=_target(),
        query=relation.expr,
    )

    updated = StatementSet(statements=(existing,)).add_insert(_target(), relation)

    assert updated.statements[-1].statement_id == "sink_0002"


def test_statement_set_rejects_relation_schema_mismatch() -> None:
    log = am.Log({"other": "Other."})

    with pytest.raises(ValueError, match="exactly match target schema"):
        StatementSet().add_insert(_target(), log.select(["other"]))


def test_policy_differentiator_compiles_sink_roots_and_shares_view_nodes() -> None:
    class StoredMemory(am.Memory):
        log = am.Log({"value": "Value."})
        rows = log.select(["value"])

    statements = StatementSet().add_insert(_target(), StoredMemory.rows)
    policy = PolicyDifferentiator().differentiate(
        StoredMemory.spec(),
        statements=statements,
    )

    assert policy.sink_outputs == {"sink_0000": policy.view_outputs["rows"]}
    assert len([node for node in policy.nodes.values() if node.query.op == "select"]) == 1


def test_zep_storage_profile_binds_public_and_private_relation_roots() -> None:
    policy = PolicyDifferentiator().differentiate(
        ZepMemory.spec(),
        statements=GRAPHITI_NEO4J_STATEMENTS,
    )

    statements = GRAPHITI_NEO4J_STATEMENTS.statements
    mappings = [statement.target.mapping for statement in statements]
    assert [type(mapping) for mapping in mappings] == [
        Neo4jNodeMapping,
        Neo4jNodeMapping,
        Neo4jRelationshipMapping,
        Neo4jRelationshipMapping,
    ]
    assert [
        mapping.label for mapping in mappings if isinstance(mapping, Neo4jNodeMapping)
    ] == ["Episodic", "Entity"]
    assert [
        mapping.relationship_type
        for mapping in mappings
        if isinstance(mapping, Neo4jRelationshipMapping)
    ] == ["RELATES_TO", "MENTIONS"]
    entity_mapping = mappings[1]
    fact_mapping = mappings[2]
    mention_mapping = mappings[3]
    assert isinstance(entity_mapping, Neo4jNodeMapping)
    assert entity_mapping.embedding is not None
    assert entity_mapping.embedding.source_column == "name"
    assert entity_mapping.embedding.model == "BAAI/bge-m3"
    assert isinstance(fact_mapping, Neo4jRelationshipMapping)
    assert fact_mapping.embedding is not None
    assert fact_mapping.embedding.source_column == "fact"
    assert fact_mapping.nested_properties[0].property_name == "episodes"
    assert isinstance(mention_mapping, Neo4jRelationshipMapping)
    assert mention_mapping.identity.columns == ("episode_id", "entity_ordinal")
    assert statements[3].target.schema.primary_key == (
        "episode_id",
        "entity_ordinal",
    )
    assert statements[0].query == ZepMemory.episodes.expr
    assert statements[1].query == ZepMemory.entities.expr
    assert statements[2].query == ZepMemory.facts.expr
    assert statements[3].query == ZepMemory._episode_entities.expr
    assert policy.sink_outputs["sink_0000"] == policy.view_outputs["episodes"]
    assert policy.sink_outputs["sink_0001"] == policy.view_outputs["entities"]
    assert policy.sink_outputs["sink_0002"] == policy.view_outputs["facts"]
    episode_entities_node = policy.sink_outputs["sink_0003"]
    assert episode_entities_node not in policy.view_outputs.values()
    assert policy.nodes[episode_entities_node].output_columns == output_columns(
        ZepMemory._episode_entities.expr
    )


def test_no_storage_policy_fingerprint_is_unchanged() -> None:
    class StoredMemory(am.Memory):
        log = am.Log({"value": "Value."})
        rows = log.select(["value"])

    cached = StoredMemory.differentiate_policy()
    explicit = PolicyDifferentiator().differentiate(StoredMemory.spec())

    assert explicit.sink_outputs == {}
    assert explicit.fingerprint == cached.fingerprint


class _RecordingTransaction(AbstractContextManager["_RecordingTransaction"]):
    def __init__(
        self,
        connector: "_RecordingConnector",
        *,
        namespace: str,
        expected_commit: StorageCommit | None,
        next_commit: StorageCommit,
    ) -> None:
        self.connector = connector
        self.namespace = namespace
        self.expected_commit = expected_commit
        self.next_commit = next_commit
        self.pending: list[tuple[InsertStatement, pd.DataFrame, pd.DataFrame]] = []

    def __enter__(self) -> "_RecordingTransaction":
        return self

    def write(
        self,
        statement: InsertStatement,
        *,
        inserted_rows: pd.DataFrame,
        retracted_rows: pd.DataFrame,
    ) -> None:
        if self.connector.fail_writes:
            raise RuntimeError("storage write failed")
        self.pending.append(
            (statement, inserted_rows.copy(), retracted_rows.copy())
        )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if exc_type is not None:
            self.connector.rollbacks += 1
            return False
        if self.connector.markers.get(self.namespace) != self.expected_commit:
            self.connector.rollbacks += 1
            raise StorageConflictError("storage commit marker changed")
        self.connector.commits.append(self.pending)
        self.connector.markers[self.namespace] = self.next_commit
        return False


class _RecordingConnector:
    def __init__(
        self,
        *,
        fail_writes: bool = False,
        fail_rebuild: bool = False,
    ) -> None:
        self.fail_writes = fail_writes
        self.fail_rebuild = fail_rebuild
        self.prepared: list[StatementSet] = []
        self.namespaces: list[str] = []
        self.commits: list[
            list[tuple[InsertStatement, pd.DataFrame, pd.DataFrame]]
        ] = []
        self.markers: dict[str, StorageCommit] = {}
        self.rebuilds: list[dict[str, pd.DataFrame]] = []
        self.rollbacks = 0

    def prepare(self, statements: StatementSet) -> None:
        self.prepared.append(statements)

    def read_commit(self, *, namespace: str) -> StorageCommit | None:
        return self.markers.get(namespace)

    def transaction(
        self,
        *,
        namespace: str,
        expected_commit: StorageCommit | None,
        next_commit: StorageCommit,
    ) -> _RecordingTransaction:
        self.namespaces.append(namespace)
        return _RecordingTransaction(
            self,
            namespace=namespace,
            expected_commit=expected_commit,
            next_commit=next_commit,
        )

    def rebuild(
        self,
        *,
        namespace: str,
        statements: StatementSet,
        rows_by_statement: dict[str, pd.DataFrame],
        expected_commit: StorageCommit | None,
        next_commit: StorageCommit,
    ) -> None:
        if self.fail_rebuild:
            raise RuntimeError("storage rebuild failed")
        if self.markers.get(namespace) != expected_commit:
            raise StorageConflictError("storage commit marker changed")
        assert {item.statement_id for item in statements.statements} == set(
            rows_by_statement
        )
        self.rebuilds.append(
            {name: rows.copy() for name, rows in rows_by_statement.items()}
        )
        if next_commit.is_initial:
            self.markers.pop(namespace, None)
        else:
            self.markers[namespace] = next_commit


def _stored_memory(connector: _RecordingConnector) -> am.Memory:
    class StoredMemory(am.Memory):
        log = am.Log({"value": "Value."})
        rows = log.select(["value"])

    statements = StatementSet().add_insert(_target(), StoredMemory.rows)
    return StoredMemory(
        adapter=LotusAdapter(),
        storage=StorageDeployment(
            connector=connector,
            statements=statements,
            namespace="test",
        ),
    )


def test_storage_receives_exact_sink_changelog_after_successful_step() -> None:
    connector = _RecordingConnector()
    memory = _stored_memory(connector)

    memory.add({"value": 7})

    assert connector.namespaces == ["test"]
    assert len(connector.commits) == 1
    _, inserted, retracted = connector.commits[0][0]
    assert inserted.to_dict("records") == [{"value": 7}]
    assert retracted.empty
    assert memory._runtime._state["rows"].to_dict("records") == [{"value": 7}]


def test_storage_receives_retractions_when_a_sink_row_is_replaced() -> None:
    class EarliestMemory(am.Memory):
        log = am.Log({"key": "Key.", "value": "Value."})
        earliest = log.group_by("key").min(column="value", output_col="value")

    target = (
        TableDescriptor.for_connector("recording")
        .schema(
            Schema.new_builder()
            .column("key", "STRING")
            .column("value", "BIGINT")
            .primary_key("key")
            .build()
        )
        .build()
    )
    statements = StatementSet().add_insert(target, EarliestMemory.earliest)
    connector = _RecordingConnector()
    memory = EarliestMemory(
        adapter=LotusAdapter(),
        storage=StorageDeployment(
            connector=connector,
            statements=statements,
            namespace="test",
        ),
    )

    memory.add({"key": "k", "value": 2})
    memory.add({"key": "k", "value": 1})

    _, inserted, retracted = connector.commits[1][0]
    assert inserted.to_dict("records") == [{"key": "k", "value": 1}]
    assert retracted.to_dict("records") == [{"key": "k", "value": 2}]


def test_multiple_sink_updates_share_one_connector_transaction() -> None:
    class StoredMemory(am.Memory):
        log = am.Log({"value": "Value."})
        rows = log.select(["value"])

    statements = (
        StatementSet()
        .add_insert(_target(), StoredMemory.rows)
        .add_insert(_target(), StoredMemory.rows)
    )
    connector = _RecordingConnector()
    memory = StoredMemory(
        adapter=LotusAdapter(),
        storage=StorageDeployment(
            connector=connector,
            statements=statements,
            namespace="test",
        ),
    )

    memory.add({"value": 7})

    assert len(connector.commits) == 1
    assert len(connector.commits[0]) == 2


def test_storage_failure_rolls_back_without_committing_memory_state() -> None:
    connector = _RecordingConnector(fail_writes=True)
    memory = _stored_memory(connector)

    with pytest.raises(RuntimeError, match="storage write failed"):
        memory.add({"value": 7})

    assert connector.commits == []
    assert connector.rollbacks == 1
    assert memory._runtime._state == {}


def test_storage_bound_checkpoint_records_and_validates_commit_marker() -> None:
    connector = _RecordingConnector()
    memory = _stored_memory(connector)
    memory.add({"value": 7})

    snapshot = memory._runtime.snapshot_state()

    assert snapshot["schema_version"] == 2
    assert StorageCommit.from_dict(snapshot["storage_commit"]) == connector.markers[
        "test"
    ]
    connector.markers["test"] = StorageCommit(
        plan_fingerprint=snapshot["plan_fingerprint"],
        lineage_id="another-lineage",
        commit_sequence=1,
        source_row_count=1,
    )
    with pytest.raises(StorageConflictError, match="does not match"):
        memory._runtime.snapshot_state()


@pytest.mark.parametrize("marker_state", ["missing", "behind", "ahead"])
def test_storage_restore_rebuilds_when_marker_does_not_match(
    marker_state: str,
) -> None:
    source_connector = _RecordingConnector()
    source = _stored_memory(source_connector)
    source.add({"value": 7})
    snapshot = source._runtime.snapshot_state()
    checkpoint_commit = StorageCommit.from_dict(snapshot["storage_commit"])

    target_connector = _RecordingConnector()
    if marker_state == "behind":
        target_connector.markers["test"] = StorageCommit(
            plan_fingerprint=checkpoint_commit.plan_fingerprint,
            lineage_id=checkpoint_commit.lineage_id,
            commit_sequence=0,
            source_row_count=0,
        )
    elif marker_state == "ahead":
        target_connector.markers["test"] = StorageCommit(
            plan_fingerprint=checkpoint_commit.plan_fingerprint,
            lineage_id=checkpoint_commit.lineage_id,
            commit_sequence=2,
            source_row_count=2,
        )
    restored = _stored_memory(target_connector)

    restored._runtime.restore_state(snapshot)

    assert target_connector.markers["test"] == checkpoint_commit
    assert len(target_connector.rebuilds) == 1
    assert target_connector.rebuilds[0]["sink_0000"].to_dict("records") == [
        {"value": 7}
    ]
    assert restored._runtime._state["rows"].to_dict("records") == [{"value": 7}]


def test_storage_restore_with_matching_marker_does_not_rebuild() -> None:
    connector = _RecordingConnector()
    source = _stored_memory(connector)
    source.add({"value": 7})
    snapshot = source._runtime.snapshot_state()
    restored = _stored_memory(connector)

    restored._runtime.restore_state(snapshot)

    assert connector.rebuilds == []
    assert restored._runtime._state["rows"].to_dict("records") == [{"value": 7}]


def test_empty_storage_checkpoint_replaces_namespace_and_can_continue() -> None:
    source = _stored_memory(_RecordingConnector())
    snapshot = source._runtime.snapshot_state()
    connector = _RecordingConnector()
    connector.markers["test"] = StorageCommit(
        plan_fingerprint=snapshot["plan_fingerprint"],
        lineage_id="stale-lineage",
        commit_sequence=2,
        source_row_count=2,
    )
    restored = _stored_memory(connector)

    restored._runtime.restore_state(snapshot)
    restored.add({"value": 7})

    assert len(connector.rebuilds) == 1
    assert StorageCommit.from_dict(snapshot["storage_commit"]).is_initial
    assert connector.markers["test"].commit_sequence == 1
    assert restored._runtime._state["rows"].to_dict("records") == [{"value": 7}]


def test_storage_rebuild_failure_does_not_mutate_runtime_state() -> None:
    source_connector = _RecordingConnector()
    source = _stored_memory(source_connector)
    source.add({"value": 7})
    snapshot = source._runtime.snapshot_state()
    connector = _RecordingConnector(fail_rebuild=True)
    restored = _stored_memory(connector)

    with pytest.raises(RuntimeError, match="storage rebuild failed"):
        restored._runtime.restore_state(snapshot)

    assert restored._runtime._state == {}


def test_storage_restore_rejects_commit_source_count_mismatch() -> None:
    connector = _RecordingConnector()
    source = _stored_memory(connector)
    source.add({"value": 7})
    snapshot = source._runtime.snapshot_state()
    snapshot["storage_commit"] = {
        **snapshot["storage_commit"],
        "source_row_count": 2,
    }
    restored = _stored_memory(_RecordingConnector())

    with pytest.raises(ValueError, match="source row count"):
        restored._runtime.restore_state(snapshot)

    assert restored._runtime._state == {}


def test_fresh_storage_runtime_does_not_overwrite_existing_namespace() -> None:
    connector = _RecordingConnector()
    connector.markers["test"] = StorageCommit(
        plan_fingerprint="another-plan",
        lineage_id="another-lineage",
        commit_sequence=5,
        source_row_count=5,
    )
    memory = _stored_memory(connector)

    with pytest.raises(StorageConflictError, match="storage commit marker"):
        memory.add({"value": 7})

    assert memory._runtime._state == {}


def test_schema_v1_restore_with_storage_remains_unsupported() -> None:
    memory = _stored_memory(_RecordingConnector())

    with pytest.raises(NotImplementedError, match="schema-v1"):
        memory._runtime.restore_state({"schema_version": 1})
