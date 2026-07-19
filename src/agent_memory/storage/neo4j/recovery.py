"""Neo4j namespace commit markers and checkpoint reconciliation primitives."""

from __future__ import annotations

from typing import Any

from agent_memory.storage.connector import StorageCommit, StorageConflictError


_READ_COMMIT = """
MATCH (m:_AgentMemoryStorageCommit {group_id: $namespace})
RETURN m.plan_fingerprint AS plan_fingerprint,
       m.lineage_id AS lineage_id,
       m.commit_sequence AS commit_sequence,
       m.source_row_count AS source_row_count
"""

# Normal commits advance only from the exact marker represented by runtime state.
_COMPARE_AND_SET_COMMIT = """
MERGE (m:_AgentMemoryStorageCommit {group_id: $namespace})
WITH m,
     CASE
       WHEN $expected IS NULL THEN m.commit_sequence IS NULL
       ELSE m.plan_fingerprint = $expected.plan_fingerprint
        AND m.lineage_id = $expected.lineage_id
        AND m.commit_sequence = $expected.commit_sequence
        AND m.source_row_count = $expected.source_row_count
     END AS matches
WHERE matches
SET m.plan_fingerprint = $next.plan_fingerprint,
    m.lineage_id = $next.lineage_id,
    m.commit_sequence = $next.commit_sequence,
    m.source_row_count = $next.source_row_count
RETURN true AS marker_updated
"""

# A rebuild validates and locks the old marker before replacing its namespace.
_LOCK_REBUILD = """
MERGE (m:_AgentMemoryStorageCommit {group_id: $namespace})
WITH m,
     CASE
       WHEN $expected IS NULL THEN m.commit_sequence IS NULL
       ELSE m.plan_fingerprint = $expected.plan_fingerprint
        AND m.lineage_id = $expected.lineage_id
        AND m.commit_sequence = $expected.commit_sequence
        AND m.source_row_count = $expected.source_row_count
     END AS matches
WHERE matches
SET m.rebuild_lock = randomUUID()
RETURN true AS marker_locked
"""

_CLEAR_RELATIONSHIPS = """
MATCH ()-[r]->()
WHERE r.group_id = $namespace
DELETE r
"""

_CLEAR_NODES = """
MATCH (n)
WHERE n.group_id = $namespace
  AND NOT n:_AgentMemoryStorageCommit
DETACH DELETE n
"""

_WRITE_COMMIT = """
MERGE (m:_AgentMemoryStorageCommit {group_id: $namespace})
SET m.plan_fingerprint = $next.plan_fingerprint,
    m.lineage_id = $next.lineage_id,
    m.commit_sequence = $next.commit_sequence,
    m.source_row_count = $next.source_row_count
REMOVE m.rebuild_lock
"""

_DELETE_COMMIT = """
MATCH (m:_AgentMemoryStorageCommit {group_id: $namespace})
DELETE m
"""


def read_commit(session: Any, *, namespace: str) -> StorageCommit | None:
    """Read one namespace marker through a Neo4j session or transaction."""

    record = session.run(_READ_COMMIT, namespace=namespace).single()
    if record is None:
        return None
    return StorageCommit.from_dict(dict(record))


def compare_and_set_commit(
    tx: Any,
    *,
    namespace: str,
    expected_commit: StorageCommit | None,
    next_commit: StorageCommit,
) -> None:
    """Advance a namespace marker or fail the surrounding transaction."""

    result = tx.run(
        _COMPARE_AND_SET_COMMIT,
        namespace=namespace,
        expected=None if expected_commit is None else expected_commit.to_dict(),
        next=next_commit.to_dict(),
    ).single()
    if result is None or not result["marker_updated"]:
        raise StorageConflictError(
            f"storage commit marker changed for namespace {namespace!r}"
        )


def lock_namespace_for_rebuild(
    tx: Any,
    *,
    namespace: str,
    expected_commit: StorageCommit | None,
) -> None:
    """Lock and validate the marker before replacing a namespace."""

    result = tx.run(
        _LOCK_REBUILD,
        namespace=namespace,
        expected=None if expected_commit is None else expected_commit.to_dict(),
    ).single()
    if result is None or not result["marker_locked"]:
        raise StorageConflictError(
            f"storage commit marker changed for namespace {namespace!r}"
        )


def clear_namespace(tx: Any, *, namespace: str) -> None:
    """Delete connector-owned graph objects for one namespace."""

    tx.run(_CLEAR_RELATIONSHIPS, namespace=namespace)
    tx.run(_CLEAR_NODES, namespace=namespace)


def write_commit(
    tx: Any,
    *,
    namespace: str,
    commit: StorageCommit,
) -> None:
    """Write a marker after a locked namespace rebuild."""

    tx.run(_WRITE_COMMIT, namespace=namespace, next=commit.to_dict())


def delete_commit(tx: Any, *, namespace: str) -> None:
    """Remove the marker used to lock an empty checkpoint rebuild."""

    tx.run(_DELETE_COMMIT, namespace=namespace)


__all__ = [
    "clear_namespace",
    "compare_and_set_commit",
    "delete_commit",
    "lock_namespace_for_rebuild",
    "read_commit",
    "write_commit",
]
