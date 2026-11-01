"""A thin layer over two engine operations, so a lost answer fails the run.

Both operations report a lost answer as an ordinary result: `map` drops the record (`map.py:407`,
`:438`) and `equijoin` answers `False` (`equijoin.py:160-184`). Neither is a verdict about the data.
One line per operation says what this layer wraps and what it handles:

* `strict_filter` — wraps `FilterOperation.execute`, handles the **count** of answers.
* `strict_equijoin` — wraps `EquijoinOperation.compare_pair`, handles the **answer** itself.

Everything else stays where it was: the model call, its retries, the type check, blocking, the
thread pool and the join assembly are the engine's, and the pair set, `how`, `k` and the published
frame are `sem_join`'s.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# The one field `EquijoinOperation` asks the model for (`equijoin.py:169`).
MATCH_KEY = "is_match"


def strict_filter(*, index_key: str) -> Callable[..., Any]:
    """Return DocETL's `filter` with one addition: a row whose answer never arrived fails the run.

    Flow:
    1. `execute()` runs `FilterOperation.execute` unchanged: the model call, its retries, the type
       check and the cascade are the engine's.
    2. `_handle_result` receives one row's answer — it is called once per **answered** record, and an
       invalid one is dropped before it (`map.py:597-600`) — so this layer counts the calls and keeps
       the labels.
    3. `execute()` then raises when the count is short of the rows it was given, naming them.

    The count is the whole content of the wrapper: a `filter` publishes fewer rows than it receives
    by design, so the answers, not the kept rows, are what can be compared with the input.

    The counter needs no lock: the engine calls `_handle_result` from the thread that runs `execute`
    (`map.py:597-600`; the pool produces batches, not answers), and `context.py:186` builds an
    operation per call, so two runs never share an instance.

    DocETL is imported here rather than at module scope for the reason
    `DocetlExecutionContext.build_operation` gives: importing the adapter must not pull in DocETL's
    operation registry.
    """

    from docetl.operations.filter import FilterOperation

    class StrictFilter(FilterOperation):
        """`filter`, plus a check that the engine answered every row it was given."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._answered = 0
            self._answered_labels: set[Any] = set()

        def _handle_result(self, result: dict[str, Any]) -> tuple[dict | None, bool]:
            self._answered += 1
            self._answered_labels.add(result[index_key])
            return super()._handle_result(result)

        def execute(
            self, input_data: list[dict], is_build: bool = False
        ) -> tuple[list[dict], float]:
            self._answered = 0
            self._answered_labels = set()
            kept, cost = super().execute(input_data, is_build=is_build)
            self._require_an_answer_per_row(input_data)
            return kept, cost

        def _require_an_answer_per_row(self, input_data: list[dict]) -> None:
            """Raise unless every record produced an answer, naming the rows that did not."""

            if self._answered == len(input_data):
                return

            labels = [record[index_key] for record in input_data]
            # a repeated index makes the naming ambiguous, so the count stands alone in that case
            unique = len(set(labels)) == len(labels)
            missing = (
                [label for label in labels if label not in self._answered_labels]
                if unique
                else []
            )
            detail = f"; missing {missing}" if missing else ""
            raise RuntimeError(
                f"{self.config['name']} answered {self._answered} of {len(input_data)} rows: an "
                f"answer failed validation and its row was dropped{detail}"
            )

    return StrictFilter


def strict_equijoin(*, left_id_key: str, right_id_key: str) -> Callable[..., Any]:
    """Return DocETL's `equijoin` with one addition: a pair with no answer fails the run.

    Flow of `compare_pair`, the one method it replaces:
    1. Render the comparison prompt, with the engine's `strict_render`.
    2. Ask through the engine's `api.call_llm`, which owns the retries and the provider routing.
    3. Raise when no response came back; parse with the engine's `api.parse_llm_response`; raise when
       the field is not a boolean.
    4. Otherwise return the boolean and the cost, as the engine would have.

    Steps 1-4 restate the engine's own body rather than adding to it, because that body answers `False`
    for a prompt that will not render (`equijoin.py:160-164`), `False` for a response that will not
    parse (`:182-184`), and `False` for a call that never returned (`api.py:727-733`) — so once it has
    returned, the failure is no longer there to check. Its `validated` flag cannot stand in either:
    `equijoin` configures no validation, so `call_llm` sets it unconditionally (`api.py:578-580`). The
    two ids name the pair in a failure.

    The engine submits `compare_pair` to a pool (`equijoin.py:609-621`), so this method keeps no state
    between pairs: it reads the config, returns, and everything else is local. Its raise is read
    uncaught at `future.result()` (`:632`), which is how it reaches the caller.

    DocETL is imported here rather than at module scope for the reason `strict_filter` gives.
    """

    from docetl.operations.equijoin import EquijoinOperation
    from docetl.operations.utils.validation import strict_render

    class StrictEquijoin(EquijoinOperation):
        """`equijoin`, plus a check that every compared pair produced a boolean answer."""

        def compare_pair(
            self,
            comparison_prompt: str,
            model: str,
            item1: dict,
            item2: dict,
            timeout_seconds: int = 120,
            max_retries_per_timeout: int = 2,
        ) -> tuple[bool, float]:
            """Answer one pair, raising where `EquijoinOperation.compare_pair` answers `False`."""

            pair = f"left {item1.get(left_id_key)} / right {item2.get(right_id_key)}"
            prompt = strict_render(comparison_prompt, {"left": item1, "right": item2})
            response = self.runner.api.call_llm(
                model,
                "compare",
                [{"role": "user", "content": prompt}],
                {MATCH_KEY: "bool"},
                timeout_seconds=timeout_seconds,
                max_retries_per_timeout=max_retries_per_timeout,
                bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
                litellm_completion_kwargs=self.config.get(
                    "litellm_completion_kwargs", {}
                ),
                op_config=self.config,
            )
            cost = response.total_cost
            if response.response is None:
                # `call_llm` retries a timeout and reports its exhaustion this way (`api.py:727-733`)
                raise RuntimeError(
                    f"{self.config['name']} got no answer for pair {pair} after "
                    f"{max_retries_per_timeout + 1} attempts"
                )
            answer = self.runner.api.parse_llm_response(
                response.response, {MATCH_KEY: "bool"}
            )[0]
            value = answer.get(MATCH_KEY)
            if not isinstance(value, bool):
                raise RuntimeError(
                    f"{self.config['name']} answered pair {pair} with {value!r}, "
                    "which is not a boolean"
                )
            return value, cost

    return StrictEquijoin
