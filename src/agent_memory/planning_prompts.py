"""Static maintenance-planning instructions and illustrative design examples."""

from __future__ import annotations

from typing import Any

from .maintenance import object_schema


AGG_INSTRUCTION = """Design a maintenance plan for the semantic aggregation below.
As records are added, the operator must keep the result up to date under the
original task. Choose a supported maintenance rule and, if needed, design auxiliary
state. Return the plan, not the aggregation result."""

GROUP_INSTRUCTION = """Design a maintenance plan for the semantic grouping or clustering below.
As records are added, the operator must keep record memberships up to date under
the original grouping criterion. Choose a supported maintenance rule and, if needed,
design auxiliary state. Return the plan, not a grouping of the records."""

_STATE_DESIGN_REQUIREMENTS = """Keep S compact and purposeful. Retain information needed for current
or later updates, at the level of detail those updates require.
Simply reorganizing historical inputs into fields does not itself
justify S. Consider the cost of reading and returning state on updates.
Specify how new information updates values, adds needed entries,
consolidates redundant entries, and makes stored information unnecessary.
Preserve distinctions required by the task."""


AGG_REQUIREMENTS = """HOW TO CHOOSE
1. Prefer agg-composable. Use the previous result and new records
   when they retain the information needed to maintain the original task.
   A concise result or omitted input details do not by themselves require S.
2. Choose agg-state only for a concrete maintenance need or benefit
   that agg-composable cannot adequately provide. Identify what S retains,
   how current or later updates use it, and why that benefit justifies
   the additional state-processing cost.
3. Avoid agg-fallback. Use it only when neither result reuse nor a
   practical state design can adequately maintain the task.
   Explain the limitation without concealing known information loss.

DESIGNING OPTIONAL STATE
Design S to complement the previous result with information needed for future updates.
You design its schema and maintenance instructions; the operator creates the first S
and updates its contents.
""" + _STATE_DESIGN_REQUIREMENTS + """
Choose a state representation suited to the task: typed JSON scalars, records,
lists, entities with properties, relationships, or knowledge graphs represented
by records and lists. Choose task-specific field names. Names can identify
semantic entities when unambiguous.
Example values: {"name":"Alice","record_count":2};
{"person":"Alice","project":"Atlas"};
{"claim":"A trial was proposed","speaker":"Alice","status":"proposal"}.
Explain what the fields mean, how to initialize S, and how to produce the original
result. Preserve repeated contributions when the task counts
them separately. The previous result and S describe the same history:
do not count them as independent observations.

OUTPUT FIELDS
For agg-state, state_schema contains value, the JSON Schema for S, and membership:null.
This membership field is reserved for grouping's record mapping and is unused here."""

GROUP_REQUIREMENTS = """HOW TO CHOOSE
1. Prefer group-composable when its record-matching procedure can
   maintain the original grouping criterion while keeping old
   memberships valid. Possible ambiguity alone does not establish
   a need for S.
2. Choose group-state only for a concrete maintenance need or benefit
   that group-composable cannot adequately provide, such as revising
   old memberships when later evidence changes their interpretation.
   Explain what evidence S retains and why processing it is worthwhile.
3. Avoid group-fallback. Use it only when neither record matching nor
   a practical state design can adequately maintain the criterion.
   Identify unresolved ambiguity rather than inventing a new criterion.

DESIGNING OPTIONAL STATE
S stores information used to maintain memberships. You design its schema and
maintenance instructions; the operator creates the first S and updates its contents.
""" + _STATE_DESIGN_REQUIREMENTS + """
S must be a JSON object containing the grouping-object array used by the member mapping.
Within S, choose task-specific fields using typed JSON scalars, records, lists,
entities with properties, relationships, or knowledge graphs represented by records
and lists. Names can identify semantic entities when unambiguous.
Example values: {"record_ids":["r1"],"project":"Atlas","owner":"Alice"};
{"name":"A. Chen","same_person_as":"Alice Chen","evidence":"explicit introduction"}.
For group-state, identify the array of objects to be grouped and the field mapping
each object to original records. This mapping uses system-provided record IDs, not
semantic names. An object representing ["r1","r3"] gives both records the same
assignments. Retain separate representations when they may need different
assignments. Evidence references are not automatically membership mappings.
Explain how to initialize S and determine memberships under the original criterion,
including when old assignments need reconsideration.

OUTPUT FIELDS
For group-state, state_schema contains value, the JSON Schema for S, and
membership:{"objects":<top-level object-array field>,"record_ids":<member-ID field>}.
Each represented object must contain a nonempty array of original record ID strings.

GROUPING CONSTRAINTS
partition_by defines independent scopes; [] means one scope. single requires
exactly one membership per record; multiple permits one or more. C is a target and
maximum nonempty group count per partition; null means no explicit target.
For composable maintenance of an existing partition, new records are locally
grouped without C, then the final group count is checked against C. Exceeding C
fails rather than merging old groups automatically."""

_SHARED_REQUIREMENTS = """OUTPUT CONTRACT
Return rule, state_schema, maintenance_instruction, and a brief rationale.
For either non-state rule, state_schema and maintenance_instruction are null.
Declare each field's type and each array's item structure. A value has one type,
optionally nullable. After validation, the schema stays fixed across updates;
only state contents change.
Declare all chosen object fields, list them in required, and set additionalProperties:false.
Use declared nullable fields for missing values. Schema references and executable code
are unsupported.

EXECUTION BOUNDARIES
Keep the original task, required output, and supplied constraints unchanged.
Use only the task, column descriptions, and any explicitly supplied Development
inputs; do not invent their contents or claim measured savings. State assumptions
only when they materially affect the plan. Plan for additions; the executor handles
withdrawals by recomputing the affected scope and rebuilding S when applicable.
Limits.max_generated_tokens_per_response is an output limit, not an input-capacity limit."""


def planning_requirements(*, grouping: bool) -> str:
    """Compose one operator's requirements with the shared output restrictions."""
    specific = GROUP_REQUIREMENTS if grouping else AGG_REQUIREMENTS
    return f"{specific}\n\n{_SHARED_REQUIREMENTS}"


def planning_examples(*, grouping: bool) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return design pairs for the target operator, not held-out task answers."""

    def example(
        task: str,
        rule: str,
        schema: Any,
        maintenance: Any,
        reason: str,
        *,
        constraints: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {
                "Original task": task,
                **({"Constraints": constraints} if constraints is not None else {}),
            },
            {
                "rule": rule,
                "state_schema": schema,
                "maintenance_instruction": maintenance,
                "rationale": reason,
            },
        )

    if grouping:
        projects = object_schema(
            {
                "items": {
                    "type": "array",
                    "items": object_schema(
                        {
                            "record_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                            "project_clues": {"type": "string"},
                        }
                    ),
                }
            }
        )
        return [
            example(
                "Group messages by the programming language explicitly named in each message. Each message names exactly one language.",
                "group-composable",
                None,
                None,
                "Each record explicitly identifies its language. Matching new records to historical records of the same language can extend existing groups without changing old memberships. This still requires historical-record matching.",
                constraints={"membership": "single", "partition_by": [], "C": None},
            ),
            example(
                "Group messages about the same project; messages about multiple projects can belong to several groups. Later messages may clarify project identities.",
                "group-state",
                {
                    "value": projects,
                    "membership": {"objects": "items", "record_ids": "record_ids"},
                },
                "Initially extract project identity clues into one item per record with its original record ID. Add items for new records; update existing clues when new evidence clarifies identity, preserving member references. Reconsider all items' project assignments; an item may belong to several project groups. Keep distinct member-level clues when identity remains uncertain, and do not combine items merely because their current groups coincide.",
                "Project identity evidence may require historical reassignment; a group name alone loses member-level clues.",
                constraints={"membership": "multiple", "partition_by": [], "C": None},
            ),
        ]

    counts = object_schema(
        {
            "counts": {
                "type": "array",
                "items": object_schema(
                    {
                        "name": {"type": "string"},
                        "record_count": {"type": "integer", "minimum": 1},
                    }
                ),
            }
        }
    )
    facts = object_schema(
        {
            "facts": {
                "type": "array",
                "items": object_schema(
                    {
                        "person": {"type": "string"},
                        "project": {"type": "string"},
                    }
                ),
            }
        }
    )
    return [
        example(
            "List all explicitly named programming languages, without duplicates.",
            "agg-composable",
            None,
            None,
            "The complete name set is already the answer.",
        ),
        example(
            "List names mentioned in at least two records; count each name once per record. Names are unambiguous.",
            "agg-state",
            {"value": counts, "membership": None},
            "Each entry records how many input records mention a name. Initialize from the first input. For each new record increment each distinct name once. Retain names below the threshold. Return names with record_count >= 2. Do not increment counts from the previous result.",
            "The result omits names seen once, but later records may make them qualify.",
        ),
        example(
            "Find people participating in both Atlas and Borealis. Input facts explicitly state participation and person names are unambiguous.",
            "agg-state",
            {"value": facts, "membership": None},
            "Initialize and retain explicit person-project participation facts, including partial matches. Add new facts, consolidate identical facts, and answer by intersecting projects per person.",
            "Partial participation facts can complete an answer later.",
        ),
    ]
