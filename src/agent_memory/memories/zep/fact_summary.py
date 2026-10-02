"""Entity identities and summaries derived from incident facts."""

from dataclasses import replace

from agent_memory.api import Memory
from agent_memory.memories.zep.policy import ZepMemory
from agent_memory.policy.aggregates import SemanticAggregateSpec, sem_agg
from agent_memory.policy.expressions import case_when
from agent_memory.policy.logical import UserQuery
from agent_memory.policy.relation import Relation
from agent_memory.policy.retrieval import BFS, BM25, RRF, CosineSimilarity, CrossEncoder, RetrievalQuery
from agent_memory.planner.physical import replace_query, walk
from agent_memory.storage import StatementSet

IDENTITY_SPEC = sem_agg(
    input_cols=["name"], output_cols={"name": "Canonical entity name."},
    instruction="Choose one canonical {name} from the supplied names. Prefer the most complete name for the same entity. Never invent a name.",
)
SUMMARY_SPEC = sem_agg(
    input_cols=["summary"], output_cols={"summary": "Fact-derived entity summary."},
    instruction=(
        "Join the supplied {summary} texts with newlines, preserving their order and content. "
        "If the result is at most 2000 characters, return it unchanged. Otherwise compress it "
        "to at most 1000 characters, preserving supported names, relationships, dates and "
        "qualifiers. State facts directly. Do not invent information."
    ),
)

_old_aggregate = next(q for q in walk(ZepMemory.entities.expr) if q.op == "agg")
_identity_aggregate = replace(_old_aggregate, params={"aggregates": tuple(
    IDENTITY_SPEC if isinstance(s, SemanticAggregateSpec) else s
    for s in _old_aggregate.params["aggregates"]
)})
_identity_expr = replace_query(ZepMemory.entities.expr, _old_aggregate, _identity_aggregate)
_identity_expr = replace(_identity_expr, params={"columns": (
    "entity_id", "name", "entity_type", "mentions",
)})


class ZepFactSummaryMemory(Memory):
    """Maintain entity identity first, then summarize incident facts separately."""

    log = ZepMemory.log
    episodes = ZepMemory.episodes
    _identities = Relation(_identity_expr)
    _episode_entities = Relation(replace_query(ZepMemory._episode_entities.expr, ZepMemory.entities.expr, _identity_expr))
    facts = Relation(replace_query(ZepMemory.facts.expr, ZepMemory.entities.expr, _identity_expr))
    # Each fact contributes to its endpoints, not every entity in its message.
    _source_facts = facts.assign(entity_id=facts.col("source_entity_id"), summary=facts.col("fact")).select(["entity_id", "summary"])
    _target_facts = facts.filter(facts.col("source_entity_id") != facts.col("target_entity_id")).assign(
        entity_id=facts.col("target_entity_id"), summary=facts.col("fact"),
    ).select(["entity_id", "summary"])
    _incident_facts = _source_facts.union_by_name(_target_facts).join(_identities.select(["entity_id"]), on="entity_id")
    _summaries = _incident_facts.group_by("entity_id").agg(SUMMARY_SPEC)
    _joined = _identities.join(_summaries, on="entity_id", how="left")
    entities = _joined.assign(summary=case_when(_joined.col("summary").is_null(), "", _joined.col("summary"))).select(
        ["entity_id", "name", "entity_type", "summary", "mentions"]
    )
    _retrieved_entities = entities.search(
        UserQuery(), methods=[BM25(), CosineSimilarity()], reranker=RRF(), limit=20,
    ).select(["record_id", "name", "summary", "rank", "score"])
    retrieval_query = RetrievalQuery(
        entities=_retrieved_entities,
        facts=facts.search(UserQuery(), methods=[BM25(), CosineSimilarity(), BFS(origins=_retrieved_entities, max_depth=3)],
                           reranker=CrossEncoder(model="BAAI/bge-reranker-v2-m3"), limit=20).select(
            ["record_id", "fact", "valid_at", "invalid_at", "expired_at", "rank", "score"]),
    )


def zep_memory_type(strategy: str) -> type[Memory]:
    """Select the explicit logical variant for benchmark configuration."""
    import agent_memory as am
    if strategy == "zep-representative":
        from agent_memory.memories.zep.representative import ZepRepresentativeMemory
        return ZepRepresentativeMemory
    return ZepFactSummaryMemory if strategy == "zep-fact-summary" else am.ZepMemory


def zep_storage_statements(strategy: str) -> StatementSet:
    """Bind the selected logical views to the unchanged Graphiti targets."""
    from agent_memory.memories.zep.storage import GRAPHITI_NEO4J_STATEMENTS
    if strategy not in {"zep-fact-summary", "zep-representative"}:
        return GRAPHITI_NEO4J_STATEMENTS
    from agent_memory.memories.zep.representative import ZepRepresentativeMemory
    memory = ZepRepresentativeMemory if strategy == "zep-representative" else ZepFactSummaryMemory
    replacements = (
        (ZepMemory.entities, memory.entities),
        (ZepMemory.facts, memory.facts),
        (ZepMemory._episode_entities, memory._episode_entities),
    )
    statements = StatementSet()
    for statement in GRAPHITI_NEO4J_STATEMENTS.statements:
        relation = next((new for old, new in replacements if statement.query == old.expr), Relation(statement.query))
        statements = statements.add_insert(statement.target, relation)
    return statements
