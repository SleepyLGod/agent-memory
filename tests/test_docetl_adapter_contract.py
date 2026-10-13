"""Contract tests for the DocETL adapter.

Deterministic and ungated: no LLM calls, no network, no DocETL machinery.
"""

from __future__ import annotations

from agent_memory.adapters import DocetlAdapter, LotusAdapter


def test_fingerprint_is_stable_and_backend_distinct() -> None:
    """A DocETL checkpoint must never be restorable under LOTUS, or the reverse."""

    fingerprint = DocetlAdapter().maintenance_execution_fingerprint
    assert fingerprint
    assert fingerprint == DocetlAdapter().maintenance_execution_fingerprint
    assert fingerprint != LotusAdapter().maintenance_execution_fingerprint
    assert fingerprint != DocetlAdapter(model="other/model").maintenance_execution_fingerprint
