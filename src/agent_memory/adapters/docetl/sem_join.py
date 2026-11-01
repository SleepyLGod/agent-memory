"""The `sem_join` lowering: candidate pairs, one verdict each, then assembly."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd

from agent_memory.adapters.docetl.prompt import (
    build_join_prompt,
    build_topk_prompt,
    compared_columns,
    join_slot,
)
from agent_memory.adapters.docetl.reserved import KEEP_KEY
from agent_memory.adapters.docetl.specs import op_config
from agent_memory.policy.logical import QueryExpr
from agent_memory.policy.placeholders import JoinScope
from agent_memory.policy.schema import relation_alias, require_unique_published_columns
from agent_memory.tracing.semantic import query_digest, write_pair_trace, write_trace_event

if TYPE_CHECKING:
    from agent_memory.adapters.docetl.context import DocetlExecutionContext

LEFT_ID_COLUMN = "left_id"
RIGHT_ID_COLUMN = "right_id"
GROUP_POSITION_COLUMN = "group_position"
SELECTED_COLUMN = "selected_right_ids"
# Required by: `_cap_matches`, which shows the ranking call the row its candidates were matched to.
ANCHOR_COLUMN = "left_row"

_HOWS = frozenset({"inner", "left", "right", "outer"})

# Based on: the default `structured_parse_retries` of 3
#   (`src/agent_memory/adapters/lotus/context.py::LotusExecutionConfig`).
_TOPK_RETRIES = 3

# A candidate pair, identified by row position in the left and right frames.
Pair = tuple[int, int]


class Match(NamedTuple):
    """A pair the verdict kept."""

    left_id: int
    right_id: int


@dataclass(frozen=True, slots=True)
class JoinParams:
    """Every `sem_join` parameter, read and validated once."""

    instruction: str
    how: str
    on: tuple[Any, ...]
    k: int | None
    id_columns: tuple[str, str] | None
    # This join's own inputs, as named by `alias(...)`; an earlier join's alias is not inherited.
    left_alias: str | None
    right_alias: str | None

    def scope(self, left: Iterable[str], right: Iterable[str]) -> JoinScope:
        """Return this join's scope over the two frames it compares."""

        return JoinScope.of(
            left, right, left_alias=self.left_alias, right_alias=self.right_alias
        )


def execute_sem_join(
    query: QueryExpr,
    inputs: Mapping[str, Any],
    execute: Callable[[QueryExpr, Mapping[str, Any]], Any],
    context: DocetlExecutionContext,
) -> Any:
    """Join two relations semantically, publishing the planner's declared column list."""

    params = _join_params(query)

    left = execute(query.inputs[0], inputs).reset_index(drop=True)
    right = execute(query.inputs[1], inputs).reset_index(drop=True)

    if left.empty or right.empty:
        skipped = _assemble(
            left, right, (), how=params.how, id_columns=params.id_columns
        )
        _write_skipped_pairs(left, right, skipped, params, query, context)
        return skipped

    pairs = _candidate_pairs(params, query, left, right)
    matches = _verify_pairs(pairs, left, right, params, context) if pairs else []
    _write_pair_decisions(pairs, matches, left, right, params, context)
    if params.k is not None:
        matches = _cap_matches(matches, left, right, params, context)
    return _assemble(
        left,
        right,
        matches,
        how=params.how,
        id_columns=params.id_columns,
    )


def _normalize_on(value: object) -> tuple[Any, ...]:
    """Return `on` as one tuple; a bare string or predicate must not be iterated."""

    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    raise TypeError(
        "sem_join on must be a key, a predicate, or a sequence of them; "
        f"got {type(value).__name__}"
    )


def _normalize_id_columns(value: object) -> tuple[str, str] | None:
    """Validate the optional planner-internal left/right id columns."""

    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(not isinstance(column, str) or not column for column in value)
    ):
        raise ValueError("sem_join id_columns must contain two non-empty column names")
    left_id_column, right_id_column = (str(column) for column in value)
    if left_id_column == right_id_column:
        raise ValueError("sem_join id_columns must be distinct")
    return left_id_column, right_id_column


def _join_params(query: QueryExpr) -> JoinParams:
    """Return the validated parameters; the only place this module reads `query.params`."""

    how = str(query.params.get("how", "inner")).lower()
    if how not in _HOWS:
        # Reason: the DSL stores `how` unvalidated
        #   (`src/agent_memory/policy/relation.py::RelationHandle.sem_join`), so this is its first check.
        raise ValueError(f"Unsupported sem_join how={how!r}")

    k = query.params.get("k")
    if k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 1):
        raise ValueError("sem_join k must be a positive integer")

    on = _normalize_on(query.params.get("on"))
    keys = tuple(item for item in on if isinstance(item, str))
    predicates = tuple(item for item in on if isinstance(item, Mapping))
    if len(keys) + len(predicates) != len(on) or (keys and predicates):
        raise ValueError("sem_join on must contain exact keys or predicate expressions")

    return JoinParams(
        instruction=str(query.params["instruction"]),
        how=how,
        on=on,
        k=k,
        id_columns=_normalize_id_columns(query.params.get("id_columns")),
        left_alias=relation_alias(query.inputs[0]),
        right_alias=relation_alias(query.inputs[1]),
    )


def _candidate_pairs(
    params: JoinParams,
    query: QueryExpr,
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> list[Pair]:
    """Return the pairs a model is asked about, after any deterministic restriction."""

    # `on`'s shape is validated in `_join_params`; only its effect on these two frames is decided here.
    exact_keys = tuple(str(item) for item in params.on if isinstance(item, str))
    predicates = tuple(item for item in params.on if isinstance(item, Mapping))

    if predicates:
        from agent_memory.adapters.docetl.relational import predicate_join_row_ids

        return [
            (int(left_id), int(right_id))
            for left_id, right_id in predicate_join_row_ids(
                query, left, right, on=predicates
            )
        ]

    if exact_keys:
        missing = sorted(
            set(exact_keys).difference(left.columns).union(
                set(exact_keys).difference(right.columns)
            )
        )
        if missing:
            raise ValueError(f"sem_join on columns not found on both sides: {missing}")
        right_by_key: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for right_id in right.index:
            right_by_key[_key(right.loc[right_id], exact_keys)].append(right_id)
        return [
            (left_id, right_id)
            for left_id in left.index
            for right_id in right_by_key.get(_key(left.loc[left_id], exact_keys), ())
        ]

    # Reason: without `on` every pair is a candidate. One call each, which is why `on` exists.
    return [
        (left_id, right_id) for left_id in left.index for right_id in right.index
    ]


def _key(row: pd.Series, columns: Sequence[str]) -> tuple[Any, ...]:
    """Return one row's exact-key tuple, refusing values a dict cannot key on."""

    key = tuple(row[column] for column in columns)
    try:
        hash(key)
    except TypeError as error:
        raise TypeError(
            "sem_join exact-key columns must contain hashable scalar values"
        ) from error
    return key


def _pair_text(
    frame: pd.DataFrame,
    row_id: int,
    columns: Sequence[str],
) -> str:
    """Render one side: the raw cell for one column, `column: value` lines for several."""

    row = frame.loc[row_id]
    if len(columns) == 1:
        return str(row[columns[0]])
    return "\n".join(f"{column}: {row[column]}" for column in columns)


def _verify_pairs(
    pairs: Sequence[Pair],
    left: pd.DataFrame,
    right: pd.DataFrame,
    params: JoinParams,
    context: DocetlExecutionContext,
) -> list[Match]:
    """Ask the model about each candidate pair; the verdict carries no score, so ranking is a later step."""

    scope = compared_columns(
        params.instruction, params.scope(left.columns, right.columns)
    )
    # Reason: slots are positional, so a column name never becomes a Jinja expression.
    records = [
        {
            LEFT_ID_COLUMN: left_id,
            RIGHT_ID_COLUMN: right_id,
            **{
                join_slot("left", index): str(left.loc[left_id, column])
                for index, column in enumerate(scope.left)
            },
            **{
                join_slot("right", index): str(right.loc[right_id, column])
                for index, column in enumerate(scope.right)
            },
        }
        for left_id, right_id in pairs
    ]
    config = build_verdict_config(params.instruction, scope=scope)
    kept = context.run_operation("filter", config, records)

    matches: list[Match] = []
    for record in kept:
        left_id = record.get(LEFT_ID_COLUMN)
        right_id = record.get(RIGHT_ID_COLUMN)
        if left_id is None or right_id is None:
            raise RuntimeError(
                "sem_join lost a pair's identity; the verdict returned a row without "
                "left_id/right_id"
            )
        matches.append(Match(int(left_id), int(right_id)))
    return matches


def _cap_matches(
    matches: Sequence[Match],
    left: pd.DataFrame,
    right: pd.DataFrame,
    params: JoinParams,
    context: DocetlExecutionContext,
) -> list[Match]:
    """Keep at most `k` matches per left row, ranking with a listwise call when it overflows.

    `k` caps before `how` is assembled, so a capped-out right row is later emitted as a right-only
    row, which is what `planner/rules.py::rule-join-map` depends on.
    """

    k = params.k
    if k is None:  # unreachable from `execute_sem_join`; total, so a second caller cannot crash here
        return list(matches)

    by_left: dict[int, list[Match]] = {}
    order: list[int] = []
    for match in matches:
        left_id = match.left_id
        if left_id not in by_left:
            by_left[left_id] = []
            order.append(left_id)
        by_left[left_id].append(match)

    if all(len(by_left[left_id]) <= k for left_id in order):
        # Reason: `k` is a cap, not a request for an opinion, so nothing within it is re-ranked.
        return list(matches)

    overflowing = [left_id for left_id in order if len(by_left[left_id]) > k]
    scope = compared_columns(
        params.instruction, params.scope(left.columns, right.columns)
    )
    candidates_by_left = {
        left_id: [match.right_id for match in by_left[left_id]] for left_id in overflowing
    }
    topk_config = build_topk_config(params.instruction, k=k, scope=scope)

    selected: dict[int, list[int]] = {}
    last_error: Exception | None = None
    # Reason: a malformed ranking answer is retried; only unanswered anchors are re-asked, so a retry
    #   cannot re-rank or contradict an answer already given.
    for _attempt in range(_TOPK_RETRIES + 1):
        pending = [left_id for left_id in overflowing if left_id not in selected]
        if not pending:
            break
        records = [
            {
                GROUP_POSITION_COLUMN: position,
                # Reason: the ranking question is "which of these best matches this row", so the record
                #   carries both sides; candidates alone leave the model ordering against nothing.
                ANCHOR_COLUMN: _pair_text(left, left_id, scope.left),
                "candidates": "\n".join(
                    f"{right_id}: {_pair_text(right, right_id, scope.right)}"
                    for right_id in candidates_by_left[left_id]
                ),
            }
            for position, left_id in enumerate(pending)
        ]
        mapped = context.run_operation("map", topk_config, records)
        answers = {int(record[GROUP_POSITION_COLUMN]): record for record in mapped}
        for position, left_id in enumerate(pending):
            try:
                selected[left_id] = _selection(
                    answers.get(position),
                    candidates_by_left[left_id],
                    left_id=left_id,
                    k=k,
                )
            except (ValueError, RuntimeError) as error:
                last_error = error

    missing = [left_id for left_id in overflowing if left_id not in selected]
    if missing:
        if last_error is not None:
            # Reason: surface the defect that recurred, not a count of unanswered anchors.
            raise last_error
        raise RuntimeError(
            "sem_join k selection lost a left row's ranking: "
            f"missing {len(missing)} of {len(overflowing)} (first: {missing[0]})"
        )

    capped: list[Match] = []
    for left_id in order:
        kept_ids = selected.get(left_id)
        if kept_ids is None:
            capped.extend(by_left[left_id])
            continue
        by_right = {match.right_id: match for match in by_left[left_id]}
        capped.extend(by_right[right_id] for right_id in kept_ids)
    return capped


def _selection(
    record: Mapping[str, Any] | None,
    candidates: Sequence[int],
    *,
    left_id: int,
    k: int,
) -> list[int]:
    """Return one anchor's chosen candidates, refusing only an answer a ranking cannot use.

    An empty list is a decision, not a defect: the prompt allows it, and the candidates already passed
    the verdict, so "none of these is strong enough" leaves the left row without a match.
    """

    if record is None:
        raise RuntimeError(
            "sem_join k selection lost a left row's ranking: "
            f"no answer for left row {left_id}"
        )
    chosen: list[int] = []
    for value in record.get(SELECTED_COLUMN) or ():
        try:
            right_id = int(str(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "sem_join k selection returned an id that is not a candidate: "
                f"{value!r}"
            ) from error
        if right_id not in candidates:
            raise ValueError(
                "sem_join k selection returned an id that is not a candidate: "
                f"{right_id}"
            )
        if right_id not in chosen:
            chosen.append(right_id)
    return chosen[:k]


def _assemble(
    left: pd.DataFrame,
    right: pd.DataFrame,
    matches: Sequence[Pair],
    *,
    how: str,
    id_columns: tuple[str, str] | None,
) -> pd.DataFrame:
    """Build the published frame, including the null rows `how` requires."""

    left_names, right_names, output_columns = _published_names(left, right, id_columns)
    rows = [
        _joined_row(left, right, left_id, right_id, left_names, right_names, id_columns)
        for left_id, right_id in _rows_for(how, matches, left.index, right.index)
    ]
    return pd.DataFrame(rows, columns=output_columns)


def _published_names(
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_columns: tuple[str, str] | None,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Return each side's published names, then the column list the frame carries."""

    # Required by: the node's declared schema, which suffixes only overlapping names
    #   (`src/agent_memory/policy/schema.py::semantic_join_columns`).
    overlaps = set(left.columns).intersection(right.columns)
    left_names = {
        column: f"{column}:left" if column in overlaps else str(column)
        for column in left.columns
    }
    right_names = {
        column: f"{column}:right" if column in overlaps else str(column)
        for column in right.columns
    }
    published = [*left_names.values(), *right_names.values()]
    require_unique_published_columns(published, op="sem_join")
    output_columns = [*published, *(id_columns or ())]
    if id_columns is not None:
        conflicts = sorted(set(id_columns).intersection(published))
        if conflicts:
            raise ValueError(
                f"sem_join id columns conflict with output columns: {conflicts}"
            )
    return left_names, right_names, output_columns


def _rows_for(
    how: str,
    matches: Sequence[Pair],
    left_index: Iterable[int],
    right_index: Iterable[int],
) -> list[tuple[int | None, int | None]]:
    """Return the row pairs `how` publishes: the matches, then the rows it keeps unmatched."""

    pairs: list[tuple[int | None, int | None]] = list(matches)
    matched_left = {left_id for left_id, _ in matches}
    matched_right = {right_id for _, right_id in matches}
    if how in {"left", "outer"}:
        pairs += [(left_id, None) for left_id in left_index if left_id not in matched_left]
    if how in {"right", "outer"}:
        pairs += [
            (None, right_id) for right_id in right_index if right_id not in matched_right
        ]
    return pairs


def _joined_row(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_id: int | None,
    right_id: int | None,
    left_names: Mapping[str, str],
    right_names: Mapping[str, str],
    id_columns: tuple[str, str] | None,
) -> dict[str, Any]:
    """Build one row; a side with no row id contributes nulls for all of its columns."""

    row = {out: _cell(left, left_id, column) for column, out in left_names.items()}
    row.update(
        {out: _cell(right, right_id, column) for column, out in right_names.items()}
    )
    if id_columns is not None:
        row[id_columns[0]] = left_id if left_id is not None else pd.NA
        row[id_columns[1]] = right_id if right_id is not None else pd.NA
    return row


def _cell(frame: pd.DataFrame, row_id: int | None, column: str) -> Any:
    """Return one cell, or a null for the side a `how` left unmatched."""

    return pd.NA if row_id is None else frame.loc[row_id, column]


def build_verdict_config(instruction: str, *, scope: JoinScope) -> dict[str, Any]:
    """Return the DocETL `filter` config that answers one pair at a time."""

    return op_config(
        "sem_join",
        "filter",
        build_join_prompt(instruction, scope=scope),
        {KEEP_KEY: "bool"},
    )


def build_topk_config(instruction: str, *, k: int, scope: JoinScope) -> dict[str, Any]:
    """Return the DocETL `map` config that ranks one left row's verified matches."""

    return op_config(
        "sem_join",
        "map",
        build_topk_prompt(
            instruction,
            k=k,
            ids_column=SELECTED_COLUMN,
            anchor_column=ANCHOR_COLUMN,
            scope=scope,
        ),
        {SELECTED_COLUMN: "list[string]"},
    )


def _write_skipped_pairs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    result: pd.DataFrame,
    params: JoinParams,
    query: QueryExpr,
    context: DocetlExecutionContext,
) -> None:
    """Record that no pair was compared because one side was empty."""

    write_trace_event(
        context.config.trace_dir(),
        operator="sem_join",
        event_type="sem_join_result",
        payload={
            "query_digest": query_digest(query),
            "how": params.how,
            "skipped_pairwise": True,
            "left_rows": len(left),
            "right_rows": len(right),
            "output_rows": len(result),
        },
    )


def _write_pair_decisions(
    pairs: Sequence[Pair],
    matches: Sequence[Match],
    left: pd.DataFrame,
    right: pd.DataFrame,
    params: JoinParams,
    context: DocetlExecutionContext,
) -> None:
    """Record one trace row per asked pair: what was compared, and whether it was kept."""

    if not pairs:
        return
    scope = compared_columns(
        params.instruction, params.scope(left.columns, right.columns)
    )
    kept = {(match.left_id, match.right_id) for match in matches}
    write_pair_trace(
        context.config.trace_dir(),
        operator="sem_join",
        rows=[
            {
                "operator": "sem_join",
                "left_id": left_id,
                "right_id": right_id,
                "left": _pair_text(left, left_id, scope.left),
                "right": _pair_text(right, right_id, scope.right),
                "parsed_output": (left_id, right_id) in kept,
            }
            for left_id, right_id in pairs
        ],
    )
