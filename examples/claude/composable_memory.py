"""Experimental Claude policy using independent grouping; baseline is unchanged."""

from agent_memory import Memory
from agent_memory.memories.claude import ClaudeMemory, _CATALOG_HOOK_INSTRUCTION
from agent_memory.policy.logical import UserQuery
from agent_memory.policy.relation import Relation

# Reuse the baseline's immutable extraction subtree and operator instructions.
# Its final select wraps aggregation, which wraps the legacy grouping carrier.
_aggregate = ClaudeMemory.topics.expr.inputs[0]
_group = _aggregate.inputs[0]
_candidates = Relation(_group.inputs[0])


class ComposableClaudeMemory(Memory):
    """Opt-in experiment, not a replacement or exact equivalent of ClaudeMemory.

    Use DifferentialRules(sem_agg_rule="agg-composable",
    sem_groupby_rule="group-composable"). Multiple membership preserves the
    possibility of multi-target matches; it does not promise legacy equivalence.
    """

    log = ClaudeMemory.log
    topics = (
        _candidates.sem_groupby(
            input_cols=_group.params["input_cols"],
            instruction=_group.params["instruction"],
            membership="multiple",
        )
        .sem_agg(
            input_cols=_aggregate.params["input_cols"],
            output_cols={
                c.name: c.description for c in _aggregate.params["output_cols"]
            },
            instruction=_aggregate.params["instruction"],
        )
        .select(["name", "description", "type", "body"])
    )

    catalog = (
        topics.sem_map(
            input_cols=["name", "description", "type", "body"],
            output_cols={"hook": "One-line relevance hook."},
            instruction=_CATALOG_HOOK_INSTRUCTION,
        )
        .assign(catalog_title=topics.col("name"))
        .select(["catalog_title", "name", "hook"])
    )

    retrieval_query = (
        topics.select(["name", "description", "type"])
        .sem_topk(UserQuery(), 5)
        .join(topics, on=["name", "description", "type"])
        .select(["name", "description", "type", "body"])
    )
