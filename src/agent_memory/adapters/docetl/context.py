"""DocETL execution context: the shared runner handle and its backend config."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reason: litellm reads a model string's prefix as the provider, which fixes both the endpoint and
#   which environment variable holds the key (`litellm.utils::get_llm_provider`), so a model literal
#   on its own cannot describe a provider, so the endpoint has to move with it.
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

    Fields are grouped by the DocETL channel each is applied through, which is the only
    classification DocETL itself defines; each channel has its single writer on
    `DocetlExecutionContext`:

    - **runner-scoped**, applied by `runner_config`: `lm_api_base`, `enable_cache`. DocETL reads
      these off the runner, so they have no per-operation home.
    - **operation-scoped defaults**, applied by `with_op_defaults`: `skip_on_error`,
      `litellm_completion_kwargs`. DocETL reads them per operation; we give every operation the
      same value so both backends behave alike. One exception: `skip_on_error` has no effect on
      `filter`, because a skipped row is a row with no answer and `strict.strict_filter` ends the
      run instead of dropping it.
    - **constructor channel**, applied by `run_operation` as `__init__` arguments: `max_threads`
      (and `model`, which sits on the context beside it).
    - **local**, never reaching DocETL: the trace settings and `sem_join_topk_method`.

    Both the model and `lm_api_base` are read from the environment when not passed, because a model
    string's prefix selects the provider, and the provider decides which endpoint and which
    credential are correct, so the two are one decision and are resolved together.
    """

    # --- runner-scoped: DocETL reads these off the runner, so no per-operation home exists.
    lm_api_base: str | None = field(
        default_factory=lambda: os.getenv(DOCETL_API_BASE_ENV)
    )
    # off by default, matching the LOTUS adapter
    enable_cache: bool = False

    # --- operation-scoped defaults: merged into every op spec by `with_op_defaults`.
    # Reason: a failing row fails the run instead of disappearing.
    skip_on_error: bool = False
    # Parameters of one model call, e.g. `temperature`; litellm forwards them to the vendor.
    litellm_completion_kwargs: Mapping[str, Any] = field(default_factory=dict)

    # --- constructor channel: passed as `__init__` arguments by `run_operation`.
    # Reason: DocETL updates its token counters non-atomically, so more than one thread would make
    #   `usage_snapshot` undercount.
    max_threads: int = 1

    # --- local: consumed by the adapter itself, never by DocETL.
    semantic_trace_dir: Path | str | None = None
    semantic_trace_snapshot_mode: str = "compact"
    sem_join_topk_method: str = "listwise"
    # Not exposed yet: the uniform op-level knobs (`timeout`, `num_retries_on_validate_failure`,
    #   `validate`, `gleaning`) and the per-op-type field layout. They are wired with the operations.

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

        Only settings that have **no** per-operation home in DocETL go here:
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
    operation instances we construct ourselves. Every native binding goes through
    `run_operation`, which is the one place the construction signature is stated.
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
        """Return the DocETL operation class for `op_type`, without constructing it.

        `op_type` is resolved against DocETL's own operation registry, so any operation it registers
        is accepted by name. To construct and run it in one step, use `run_operation`.
        """

        # Reason: `get_operation` is DocETL's registry. The `Pipeline`/YAML paths resolve only a
        #   subset of it, so they are not used here.
        from docetl.operations import get_operation

        return get_operation(op_type)

    def with_op_defaults(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the runner-wide defaults for operation-scoped settings to one op spec.

        DocETL reads `skip_on_error` and `litellm_completion_kwargs` per operation (its op
        schema), so this config acts as the runner-wide default for them: the single writer,
        and a spec that already carries a value (a test pinning one) wins.

        `skip_on_error` is passed, not enforced: an operation that refuses a lost answer
        (`strict.strict_filter`) ends the run instead of skipping the row.
        """

        merged = dict(spec)
        merged.setdefault("skip_on_error", self.config.skip_on_error)
        if self.config.litellm_completion_kwargs:
            merged.setdefault(
                "litellm_completion_kwargs",
                dict(self.config.litellm_completion_kwargs),
            )
        return merged

    def _operation(
        self, op_type: str, spec: Mapping[str, Any], operation_cls: Any
    ) -> Any:
        """Return the operation instance for one call: the one place the signature is stated.

        `self.handle` is the `runner` the operation reads `runner.api` from, `spec` is the config it
        validates on construction, and `model`/`max_threads` are this context's.

        `operation_cls` must subclass the class `op_type` resolves to. The substitute is built with
        the same four arguments and is expected to run that operation, so a class for another one
        would be constructed into an object whose hooks nothing calls.
        """

        registered = self.build_operation(op_type)
        if operation_cls is not None and not (
            isinstance(operation_cls, type) and issubclass(operation_cls, registered)
        ):
            given = getattr(operation_cls, "__name__", type(operation_cls).__name__)
            raise TypeError(
                f"{op_type!r} takes a subclass of {registered.__name__}, got {given}"
            )
        return (operation_cls or registered)(
            self.handle,
            self.with_op_defaults(spec),
            self.model,
            self.config.max_threads,
        )

    def _add_cost(self, cost: float) -> None:
        """Add one operation's returned cost to the handle.

        `execute()` returns `(output, cost)` and nothing outside adds it: DocETL does so in
        `runner._run_operation` (`runner.py:617`), the path a direct call replaces. A context that
        skipped this would spend money the handle never records.
        """

        self.handle.total_cost += cost

    def run_operation(
        self,
        op_type: str,
        spec: Mapping[str, Any],
        records: Sequence[dict[str, Any]],
        *,
        operation_cls: Any = None,
    ) -> list[dict[str, Any]]:
        """Run one DocETL operation using DocETL's own class and `execute()`.

        `op_type` is resolved against DocETL's registry (see `build_operation`); `spec` is its DocETL
        config, while `model` and `max_threads` come from this context. Returns the records the
        operation produced, and adds its cost to the handle, which is also where `usage_snapshot`
        reads the tokens.

        `execute()` is the step that calls the model, so this is where rows are sent and answers come
        back. `spec` and `records` are copied per call, so a caller can reuse either one (DocETL
        mutates only its own row copies). Runner-wide op defaults are applied by `with_op_defaults`;
        nothing here catches, so `skip_on_error` decides raise versus drop per row.

        `operation_cls` replaces the registered class with one constructed the same way (see
        `_operation`). `strict.strict_filter` is its only user: a `filter` that also refuses a row
        whose answer never arrived.
        """

        # Construction is what binds this context to the operation; `execute()` returns
        #   `(output, cost)`, and the cost is ours to add (`_add_cost`).
        operation = self._operation(op_type, spec, operation_cls)
        output, cost = operation.execute(list(records))
        self._add_cost(cost)
        return output

    def run_pair_operation(
        self,
        op_type: str,
        spec: Mapping[str, Any],
        left_records: Sequence[dict[str, Any]],
        right_records: Sequence[dict[str, Any]],
        *,
        operation_cls: Any = None,
    ) -> list[dict[str, Any]]:
        """Run one two-input DocETL operation: `execute(left_records, right_records)`.

        `equijoin` is the operation this exists for: it compares two relations rather than one
        record list (`equijoin.py:232-234`), so it answers the pairs its blocking rule admits and
        returns one joined record per pair the model kept. Everything else matches
        `run_operation`: the spec, the copied inputs and the cost added to the handle.

        `operation_cls` replaces the registered class on the same terms. `strict.strict_equijoin`
        is its only user: an `equijoin` that also refuses a pair whose answer never arrived.
        """

        operation = self._operation(op_type, spec, operation_cls)
        output, cost = operation.execute(list(left_records), list(right_records))
        self._add_cost(cost)
        return output

    def usage_snapshot(self) -> dict[str, int]:
        """Return cumulative physical token counters held by the runner handle.

        Cumulative over every call made through this context, read from DocETL's own counters. No
        engine-backed operator reports it yet, so only tests read it today.
        """

        if self._handle is None:
            # No handle means no engine call can have happened, so zero is a reading, not a default.
            return {"prompt_tokens": 0, "completion_tokens": 0}

        usage = getattr(self._handle, "total_token_usage", None) or {}
        prompt = sum(
            int(entry.get("prompt_tokens", 0) or 0) for entry in usage.values()
        )
        completion = sum(
            int(entry.get("completion_tokens", 0) or 0) for entry in usage.values()
        )
        return {"prompt_tokens": prompt, "completion_tokens": completion}
