"""Pure contract tests for the isolated Agent Mem0 smoke."""

from __future__ import annotations

from examples.mem0.e2e_demo import (
    canonical_input_fingerprint,
)


def test_mem0_smoke_canonical_input_fingerprint_is_stable() -> None:
    """Native and Agent runners must share one exact input identity."""

    assert canonical_input_fingerprint() == (
        "799bf18a225b3c8121f8745572dd366c234847e823cbbcf0af107d296209b9c9"
    )
