"""Two engine operations, wrapped so a failed answer fails the run.

Both report that failure as an ordinary result: `map` drops the record (`map.py:407`, `:438`) and
`equijoin` answers `False` (`equijoin.py:160-184`). Neither is a verdict about the data.

* `strict_filter` — wraps `execute` and `_handle_result`; checks the **shape** of each answer and
  the **count** of answers.
* `strict_equijoin` — wraps `compare_pair`; checks the **answer** itself.

Everything else stays the engine's — the model call, its retries, the type check, blocking, the
thread pool, the join assembly — and the pair set, `how`, `k` and the published frame are
`sem_join`'s.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# The one field `EquijoinOperation` asks the model for (`equijoin.py:169`).
MATCH_KEY = "is_match"


def strict_filter(*, index_key: str) -> Callable[..., Any]:
    """Return DocETL's `filter` with one addition: a row without a boolean answer fails the run.

    Flow:
    1. `execute()` runs `FilterOperation.execute` unchanged: the model call, its retries, the type
       check and the keep/drop decision are the engine's.
    2. `_handle_result` sees one row's answer, past that validation (`map.py:597-600`), and refuses
       one that is absent or not a real boolean.
    3. `execute()` then compares the answers it counted with the rows it was given, raising with the
       rows it never heard about.

    Both are needed: a lost answer leaves nothing to inspect, and the decision is read with
    `bool(result.get(...))` (`filter.py:103`), so a malformed one would pass as a verdict.

    `op_config` sets neither `cascade` nor `limit` (`specs.py:23-27`; a cascade never reaches
    `_handle_result`, `filter.py:130-131`), which is what lets the count mean one answer per row.

    The counter needs no lock (`_handle_result` runs on `execute`'s thread, `map.py:597-600`), and
    DocETL is imported inside the factory so the adapter does not pull in its operation registry.
    """

    from docetl.operations.filter import FilterOperation

    class StrictFilter(FilterOperation):
        """`filter`, plus a check that the engine answered every row it was given."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._answered = 0
            self._answered_labels: set[Any] = set()

        def _handle_result(self, result: dict[str, Any]) -> tuple[dict | None, bool]:
            # `self._filter_key` is the field the engine itself pops (`filter.py:103-104`), so checking
            #   this name cannot drift from the name it reads.
            if self._filter_key not in result:
                raise RuntimeError(
                    f"{self.config['name']} answered row {result.get(index_key)} without "
                    f"{self._filter_key!r}: an answer of the wrong shape is not a refusal"
                )
            answered = result[self._filter_key]
            if not isinstance(answered, bool):
                raise RuntimeError(
                    f"{self.config['name']} answered row {result.get(index_key)} with {answered!r}, "
                    "which is not a boolean"
                )
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
            """Raise unless every record produced exactly one boolean answer, naming the rows."""

            labels = [record[index_key] for record in input_data]
            if self._answered == len(labels):
                return

            # a repeated index makes the naming ambiguous, so the count stands alone in that case
            unique = len(set(labels)) == len(labels)
            missing = (
                [label for label in labels if label not in self._answered_labels]
                if unique
                else []
            )
            detail = f"; missing {missing}" if missing else ""
            raise RuntimeError(
                f"{self.config['name']} answered {self._answered} of {len(labels)} rows: a filter "
                f"needs exactly one boolean answer per row{detail}"
            )

    return StrictFilter


def strict_equijoin(*, left_id_key: str, right_id_key: str) -> Callable[..., Any]:
    """Return DocETL's `equijoin` with one addition: a pair without a boolean answer fails the run.

    Flow of `compare_pair`, the one method it replaces:
    1. Render the prompt with the engine's `strict_render`.
    2. Ask through the engine's `api.call_llm`, which owns the retries and the provider routing.
    3. Raise when no response came back; parse with `api.parse_llm_response`; raise when the field
       is not a boolean.
    4. Otherwise return the boolean and the cost, as the engine would have.

    The four steps restate the engine's body because it answers `False` for every failure — a prompt
    that will not render (`equijoin.py:160-164`), a response that will not parse (`:182-184`), a call
    that never returned (`api.py:727-733`) — so nothing is left to inspect afterwards. `validated`
    cannot stand in: with no validation configured it is set unconditionally (`api.py:578-580`).

    The engine submits this method to a pool (`equijoin.py:609-621`), so it keeps no state between
    pairs, and its raise surfaces uncaught at `future.result()` (`:632`). The two ids name the pair
    in a failure; DocETL is imported here for the reason `strict_filter` gives.
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
