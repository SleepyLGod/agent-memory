"""DocETL execution context: the shared runner handle and its backend config."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reason: litellm reads a model string's prefix as the provider, which fixes both the endpoint and
#   which environment variable holds the key (`litellm.utils::get_llm_provider`), so a model literal
#   on its own cannot describe a provider — the endpoint has to move with it.
# Required by: `DocetlAdapter.model` and `DocetlExecutionConfig.lm_api_base`, which resolve these at
#   construction rather than at import, so one process can point them somewhere else per case.
DOCETL_MODEL_ENV = "AGENT_MEMORY_DOCETL_MODEL"
DOCETL_API_BASE_ENV = "AGENT_MEMORY_DOCETL_API_BASE"

# what remains when the environment names nothing, and the same literal `LotusAdapter` defaults to
DEFAULT_DOCETL_MODEL = "deepseek/deepseek-v4-pro"


def default_docetl_model() -> str:
    """Return the model to use: the environment's if it names one, else the literal default."""

    return os.getenv(DOCETL_MODEL_ENV) or DEFAULT_DOCETL_MODEL

# `DSLRunner` only needs these keys when it is used purely as a handle: `runner.run()` is never
# called, so no pipeline output path or dataset is required.
_RUNNER_BASE_CONFIG: dict[str, Any] = {
    "datasets": {},
    "operations": [],
    "pipeline": {"steps": [], "output": {"type": "memory", "path": ""}},
}

# `listwise` is the only resolver today: DocETL's verdict is a boolean, so ranking is a second call. A
#   resolver that returns a score with the verdict would change the question put to the model; such a
#   value belongs in this tuple, with its branch in `sem_join._cap_matches`.
SEM_JOIN_TOPK_METHODS = ("listwise",)


@dataclass(frozen=True)
class DocetlExecutionConfig:
    """Backend knobs for the DocETL adapter. Deliberately **not** policy semantics.

    Caching is off by default, matching the LOTUS adapter, so an answer never comes from an
    earlier run's cache and a policy behaves the same on either backend; `bypass_cache` is
    therefore set on the runner unless `enable_cache` is explicitly turned on. `skip_on_error`
    stays `False` so a failing row fails the run instead of disappearing.

    Both the model and `lm_api_base` are read from the environment when not passed, because a model
    string's prefix selects the provider, and the provider decides which endpoint and which
    credential are correct — so the two are one decision and are resolved together.

    The uniform op-level knobs (`timeout`, `num_retries_on_validate_failure`, `validate`,
    `gleaning`) and the per-op-type field layout are not exposed yet; they are wired together with
    the operations.
    """

    lm_api_base: str | None = field(
        default_factory=lambda: os.getenv(DOCETL_API_BASE_ENV)
    )
    # Reason: temperature and the reasoning switches are only pinnable through litellm, and DocETL
    #   forwards exactly this key per operation (`docetl.operations.map::MapOperation.__init__`), so the
    #   name is DocETL's rather than ours. Empty by default: a knob left unset is the adapter's choice,
    #   not a policy's.
    litellm_completion_kwargs: Mapping[str, Any] = field(default_factory=dict)
    enable_cache: bool = False
    skip_on_error: bool = False
    max_threads: int = 1
    semantic_trace_dir: Path | str | None = None
    semantic_trace_snapshot_mode: str = "compact"
    sem_join_topk_method: str = "listwise"

    def __post_init__(self) -> None:
        """Validate bounded execution settings."""

        if self.max_threads < 1:
            raise ValueError("max_threads must be positive")
        if self.semantic_trace_snapshot_mode not in {"compact", "full"}:
            raise ValueError(
                "semantic_trace_snapshot_mode must be 'compact' or 'full'"
            )
        if self.sem_join_topk_method not in SEM_JOIN_TOPK_METHODS:
            raise ValueError(
                "sem_join_topk_method must be one of: "
                + ", ".join(SEM_JOIN_TOPK_METHODS)
            )

    def trace_dir(self) -> Path | str | None:
        """Return the configured semantic trace directory."""

        return self.semantic_trace_dir

    def runner_config(self, model: str) -> dict[str, Any]:
        """Build the runner-level config.

        Only settings that have **no** per-operation home in DocETL go here —
        `default_lm_api_base` and `bypass_cache`. Everything else is applied per operation.
        """

        config: dict[str, Any] = {"default_model": model, **_RUNNER_BASE_CONFIG}
        if self.lm_api_base is not None:
            config["default_lm_api_base"] = self.lm_api_base
        if not self.enable_cache:
            config["bypass_cache"] = True
        return config


@dataclass
class DocetlExecutionContext:
    """Own one `DSLRunner` as a pure handle.

    Never calls `runner.run()`: the handle exists only to supply `runner.api` to
    operation instances we construct ourselves.
    """

    model: str = field(default_factory=default_docetl_model)
    config: DocetlExecutionConfig = field(default_factory=DocetlExecutionConfig)
    _handle: Any = field(default=None, init=False, repr=False)

    @property
    def handle(self) -> Any:
        """Return the `DSLRunner` handle, constructing it on first use."""

        if self._handle is None:
            from docetl.runner import DSLRunner

            self._handle = DSLRunner(self.config.runner_config(self.model))
        return self._handle

    def build_operation(self, op_type: str) -> Any:
        """Return the DocETL operation class for `op_type`.

        Uses `get_operation` directly: the runtime's operator registry is complete, whereas the
        `Pipeline`/YAML paths do not list every operation.
        """

        from docetl.operations import get_operation

        return get_operation(op_type)

    def usage_snapshot(self) -> dict[str, int]:
        """Return cumulative physical token counters held by the runner handle.

        Nothing calls this yet: reporting usage per operator arrives with the engine-backed
        operators. Without a handle no engine call can have happened, so zeros are *measured*, not
        assumed. DocETL updates these counters non-atomically, which is why `max_threads`
        defaults to 1.
        """

        if self._handle is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        usage = getattr(self._handle, "total_token_usage", None) or {}
        prompt = sum(
            int(entry.get("prompt_tokens", 0) or 0) for entry in usage.values()
        )
        completion = sum(
            int(entry.get("completion_tokens", 0) or 0) for entry in usage.values()
        )
        return {"prompt_tokens": prompt, "completion_tokens": completion}
