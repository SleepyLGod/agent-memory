"""Fact groups represented by their earliest occurrence.

Provenance and temporal fields retain their aggregation rules.
"""

from dataclasses import replace

from agent_memory.api import Memory
from agent_memory.memories.zep.fact_summary import ZepFactSummaryMemory
from agent_memory.memories.zep.policy import ZepMemory
from agent_memory.planner.physical import replace_query
from agent_memory.policy.aggregates import MinAggregateSpec, SemanticAggregateSpec, arg_min
from agent_memory.policy.relation import Relation, SearchRelation
from agent_memory.policy.retrieval import RetrievalQuery
from agent_memory.policy.logical import QueryExpr


_old = ZepMemory._deduplicated_facts.expr.inputs[0]
_representative = replace(_old, params={"aggregates": tuple(
    arg_min(order_by=("add_seq", "fact_ordinal"), columns=("relation_type", "fact"))
    if isinstance(spec, SemanticAggregateSpec) else spec
    for spec in _old.params["aggregates"]
    if not (isinstance(spec, MinAggregateSpec) and spec.output_col == "add_seq")
)})

_TEMPORAL_GROUP_INSTRUCTION = str(_old.inputs[0].params["instruction"]) + """
For time-specific events, use the resolved event time {valid_at} to interpret
relative dates in the fact text. Separate occurrences of an event are not
duplicates just because their wording is similar. An unknown event time is
not evidence that two occurrences are the same. For ongoing or timeless
claims, a difference in timestamps alone does not establish a different fact.
"""


def representative_query(query: QueryExpr) -> QueryExpr:
    """Replace fact synthesis, preserving provenance and temporal aggregation."""
    # Entity identity has already been replaced in the fact-summary variant.
    from agent_memory.planner.physical import walk
    candidates = [q for q in walk(query) if q.op == "agg" and q.params == _old.params]
    for candidate in candidates:
        group = candidate.inputs[0]
        temporal_group = replace(group, params={**group.params,
            "input_cols": (*group.params["input_cols"], "valid_at"),
            "instruction": _TEMPORAL_GROUP_INSTRUCTION.strip()})
        query = replace_query(query, candidate, replace(candidate, params=_representative.params,
                                                       inputs=(temporal_group,)))
    return query


class ZepRepresentativeMemory(Memory):
    """Use stable fact representatives with the existing fact-summary contract."""

    log = ZepFactSummaryMemory.log
    episodes = ZepFactSummaryMemory.episodes
    _identities = ZepFactSummaryMemory._identities
    _episode_entities = ZepFactSummaryMemory._episode_entities
    facts = Relation(representative_query(ZepFactSummaryMemory.facts.expr))
    entities = Relation(representative_query(ZepFactSummaryMemory.entities.expr))
    retrieval_query = RetrievalQuery(**{
        name: SearchRelation(representative_query(query))
        for name, query in ZepFactSummaryMemory.retrieval_query.channels.items()
    })
