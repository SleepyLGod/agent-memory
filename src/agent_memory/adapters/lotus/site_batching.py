"""Opt-in, site-scoped batching of pair predicates after candidate screening."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from agent_memory.adapters.lotus.prompt_batching import PromptBatching
from agent_memory.adapters.lotus.sem_filter_batch_prompting import (
    BatchPromptingResult,
    execute_batch_prompted_sem_filter,
)

if TYPE_CHECKING:
    from agent_memory.adapters.lotus.context import LotusExecutionContext


@dataclass(frozen=True)
class PairFilterBatching:
    """Batch one pair-filter site, partitioned by stable endpoint columns."""

    group_by: tuple[str, ...]
    prompt_batching: PromptBatching
    shared_columns: tuple[str, ...] = ()
    pack_small_groups: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pack_small_groups, bool):
            raise TypeError("pack_small_groups must be boolean")
        if not isinstance(self.group_by, tuple) or not self.group_by or any(
            not isinstance(c, str) or not c for c in self.group_by
        ) or len(set(self.group_by)) != len(self.group_by):
            raise ValueError("group_by must contain distinct column names")
        if not isinstance(self.prompt_batching, PromptBatching) or self.prompt_batching.max_tasks is None:
            raise ValueError("site batching requires an explicit positive max_tasks")
        if not isinstance(self.shared_columns, tuple) or any(not isinstance(c, str) or not c for c in self.shared_columns):
            raise ValueError("shared_columns must be a tuple of column names")

    def to_dict(self) -> dict[str, Any]:
        result = {"version": "pair-filter-batching-v2", "group_by": list(self.group_by),
                  "prompt_batching": self.prompt_batching.to_dict()}
        if self.shared_columns:
            result.update(version="pair-filter-shared-context-v1", shared_columns=list(self.shared_columns))
        if self.pack_small_groups:
            result.update(version="pair-filter-packed-groups-v1", pack_small_groups=True)
        return result


def execute_site_batching(
    source: pd.DataFrame, *, identities: pd.DataFrame, instruction: str,
    context: LotusExecutionContext, config: PairFilterBatching,
) -> BatchPromptingResult:
    """Evaluate groups independently and restore original occurrence positions."""
    if len(source) != len(identities):
        raise ValueError("pair identities and predicate rows differ")
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for position, row in enumerate(identities.loc[:, list(config.group_by)].itertuples(index=False, name=None)):
        if any(value is None or value is pd.NA or isinstance(value, float) and pd.isna(value) for value in row):
            raise ValueError("site batching requires non-null endpoint identities")
        try:
            groups[row].append(position)
        except TypeError as error:
            raise ValueError("site batching requires hashable endpoint identities") from error
    partitions = tuple(tuple(positions) for positions in groups.values())
    if config.pack_small_groups:
        assert config.prompt_batching.max_tasks is not None
        partitions = _pack_partitions(partitions, config.prompt_batching.max_tasks)
    return execute_batch_prompted_sem_filter(
        source, instruction=instruction, context=context,
        prompt_batching=config.prompt_batching,
        task_partitions=partitions,
        shared_columns=config.shared_columns,
    )


def _pack_partitions(groups: tuple[tuple[int, ...], ...], limit: int) -> tuple[tuple[int, ...], ...]:
    """Pack consecutive small groups without dropping identities or splitting them."""
    packed: list[tuple[int, ...]] = []
    pending: tuple[int, ...] = ()
    for group in groups:
        if len(pending) + len(group) > limit:
            if pending:
                packed.append(pending)
            pending = ()
        if len(group) > limit:
            # The existing executor splits oversized groups using the same limit.
            packed.append(group)
        else:
            pending += group
    if pending:
        packed.append(pending)
    return tuple(packed)
