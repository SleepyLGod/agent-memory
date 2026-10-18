"""Contract tests for the DocETL adapter.

Deterministic and ungated: no LLM calls, no network, no DocETL machinery.
"""

from __future__ import annotations

import pytest

from agent_memory.adapters import DocetlAdapter, LotusAdapter
from agent_memory.adapters.docetl import DocetlExecutionConfig
from agent_memory.adapters.docetl.context import (
    DEFAULT_DOCETL_MODEL,
    DOCETL_API_BASE_ENV,
    DOCETL_MODEL_ENV,
)


def test_fingerprint_is_stable_and_backend_distinct() -> None:
    """A DocETL checkpoint must never be restorable under LOTUS, or the reverse."""

    fingerprint = DocetlAdapter().maintenance_execution_fingerprint
    assert fingerprint
    assert fingerprint == DocetlAdapter().maintenance_execution_fingerprint
    assert fingerprint != LotusAdapter().maintenance_execution_fingerprint
    assert fingerprint != DocetlAdapter(model="other/model").maintenance_execution_fingerprint


def test_model_and_endpoint_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model string decides the provider, so the endpoint has to be settable beside it."""

    monkeypatch.setenv(DOCETL_MODEL_ENV, "openai/some-model")
    monkeypatch.setenv(DOCETL_API_BASE_ENV, "https://example.invalid/v1")

    adapter = DocetlAdapter()

    assert adapter.model == "openai/some-model"
    assert adapter.config.lm_api_base == "https://example.invalid/v1"


def test_an_explicit_argument_overrides_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The environment supplies a default, not a policy: a caller can still name both."""

    monkeypatch.setenv(DOCETL_MODEL_ENV, "openai/some-model")
    monkeypatch.setenv(DOCETL_API_BASE_ENV, "https://example.invalid/v1")

    adapter = DocetlAdapter(
        model="other/model",
        config=DocetlExecutionConfig(lm_api_base="https://other.invalid/v1"),
    )

    assert adapter.model == "other/model"
    assert adapter.config.lm_api_base == "https://other.invalid/v1"


def test_an_empty_environment_falls_back_to_the_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The literal is what a library user gets when nothing names a provider."""

    monkeypatch.delenv(DOCETL_MODEL_ENV, raising=False)
    monkeypatch.delenv(DOCETL_API_BASE_ENV, raising=False)

    adapter = DocetlAdapter()

    assert adapter.model == DEFAULT_DOCETL_MODEL
    assert adapter.config.lm_api_base is None
