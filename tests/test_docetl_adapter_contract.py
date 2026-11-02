"""The adapter's public shape: fingerprint, supported set, and config read from the environment.

No engine, no network: these cases construct an adapter and read what the runtime relies on, so a missing
guarantee fails here rather than in the middle of a policy.
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


@pytest.mark.parametrize(
    "adapter",
    [
        pytest.param(DocetlAdapter(model="other/model"), id="model"),
        pytest.param(
            DocetlAdapter(config=DocetlExecutionConfig(skip_on_error=True)),
            id="skip_on_error",
        ),
        pytest.param(
            DocetlAdapter(config=DocetlExecutionConfig(enable_cache=True)),
            id="enable_cache",
        ),
        pytest.param(
            DocetlAdapter(config=DocetlExecutionConfig(lm_api_base="https://other.invalid/v1")),
            id="lm_api_base",
        ),
        pytest.param(
            DocetlAdapter(
                config=DocetlExecutionConfig(
                    litellm_completion_kwargs={"temperature": 0}
                )
            ),
            id="completion_kwargs",
        ),
    ],
)
def test_a_setting_that_changes_a_decision_changes_the_fingerprint(
    adapter: DocetlAdapter,
) -> None:
    """A checkpoint made under one configuration must not be restored under another."""

    assert (
        adapter.maintenance_execution_fingerprint
        != DocetlAdapter().maintenance_execution_fingerprint
    )


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(DocetlExecutionConfig(max_threads=4), id="max_threads"),
        pytest.param(
            DocetlExecutionConfig(semantic_trace_dir="/tmp/trace"), id="trace_dir"
        ),
        pytest.param(
            DocetlExecutionConfig(semantic_trace_snapshot_mode="full"),
            id="snapshot_mode",
        ),
    ],
)
def test_a_setting_that_only_changes_execution_keeps_the_fingerprint(
    config: DocetlExecutionConfig,
) -> None:
    """Concurrency and tracing reach the same decisions, so a checkpoint stays compatible."""

    assert (
        DocetlAdapter(config=config).maintenance_execution_fingerprint
        == DocetlAdapter().maintenance_execution_fingerprint
    )


def test_completion_kwargs_order_does_not_change_the_fingerprint() -> None:
    """The same knobs written in a different order are the same settings."""

    first = DocetlAdapter(
        config=DocetlExecutionConfig(
            litellm_completion_kwargs={"temperature": 0, "top_p": 1}
        )
    )
    second = DocetlAdapter(
        config=DocetlExecutionConfig(
            litellm_completion_kwargs={"top_p": 1, "temperature": 0}
        )
    )

    assert first.maintenance_execution_fingerprint == second.maintenance_execution_fingerprint


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
