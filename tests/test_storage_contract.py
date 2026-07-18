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
    StorageDeployment,
    TableDescriptor,
)
from agent_memory.storage.statements import InsertStatement


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
    assert [statement.target.options["kind"] for statement in statements] == [
        "node",
        "node",
        "relationship",
        "relationship",
    ]
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
    def __init__(self, connector: "_RecordingConnector") -> None:
        self.connector = connector
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
        if exc_type is None:
            self.connector.commits.append(self.pending)
        else:
            self.connector.rollbacks += 1
        return False


class _RecordingConnector:
    def __init__(self, *, fail_writes: bool = False) -> None:
        self.fail_writes = fail_writes
        self.namespaces: list[str] = []
        self.commits: list[
            list[tuple[InsertStatement, pd.DataFrame, pd.DataFrame]]
        ] = []
        self.rollbacks = 0

    def transaction(self, *, namespace: str) -> _RecordingTransaction:
        self.namespaces.append(namespace)
        return _RecordingTransaction(self)


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


def test_storage_bound_runtime_rejects_checkpoint_and_restore() -> None:
    memory = _stored_memory(_RecordingConnector())

    with pytest.raises(NotImplementedError, match="storage-bound checkpoint"):
        memory._runtime.snapshot_state()
    with pytest.raises(NotImplementedError, match="storage-bound checkpoint"):
        memory._runtime.restore_state({"schema_version": 2})
