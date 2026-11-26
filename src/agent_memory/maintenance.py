"""Explicit, serializable maintenance plans; planning never runs during compilation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from jsonschema import Draft202012Validator

from .planner.serialization import stable_json, stable_value
from .policy.aggregates import SemanticAggregateSpec
from .policy.grouping import normalize_fallback
from .policy.logical import QueryExpr, _freeze_param
from .policy.relation import Relation


def object_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Describe a closed object with required fields."""
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def validate_state_schema(schema: Mapping[str, Any]) -> None:
    """Accept typed JSON trees, not references, executable code, or untyped blobs."""
    if not isinstance(schema, Mapping):
        raise TypeError("state schema must be an object")
    schema = stable_value(schema)
    json.dumps(schema, allow_nan=False)
    Draft202012Validator.check_schema(schema)
    common = {"type", "description", "enum"}
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if not kinds or any(
        not isinstance(k, str)
        or k
        not in {"object", "array", "string", "integer", "number", "boolean", "null"}
        for k in kinds
    ):
        raise ValueError("state schema requires explicit JSON types")
    concrete = [k for k in kinds if k != "null"]
    if len(concrete) > 1:
        raise ValueError("state schema supports a single type, optionally nullable")
    kind = str(concrete[0]) if concrete else "null"
    allowed = common | {
        "object": {"properties", "required", "additionalProperties"},
        "array": {"items", "minItems", "maxItems"},
        "string": {"minLength", "maxLength"},
        "integer": {"minimum", "maximum"},
        "number": {"minimum", "maximum"},
    }.get(kind, set())
    if set(schema) - allowed:
        raise ValueError("unsupported state schema keywords")
    if kind == "object":
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is not False or set(
            schema.get("required", [])
        ) != set(properties):
            raise ValueError(
                "state objects require declared fields and additionalProperties=false"
            )
        for field in properties.values():
            validate_state_schema(field)
    elif kind == "array":
        if not isinstance(schema.get("items"), dict):
            raise ValueError("state arrays require a typed item schema")
        validate_state_schema(schema["items"])


def maintenance_target(
    target: Relation | QueryExpr, aggregate_index: int = 0
) -> QueryExpr:
    """Resolve one supported semantic task, preserving its original instruction."""
    if type(aggregate_index) is not int or aggregate_index < 0:
        raise ValueError("aggregate_index must be a nonnegative integer")
    query = target.expr if isinstance(target, Relation) else target
    if query.op == "sem_groupby" and query.params.get("independent"):
        if aggregate_index != 0:
            raise ValueError("grouping has no aggregate descriptor index")
        return query
    if query.op == "agg":
        specs = query.params["aggregates"]
        if not all(isinstance(s, SemanticAggregateSpec) for s in specs):
            raise NotImplementedError(
                "maintenance plans require semantic-only aggregates"
            )
        if type(aggregate_index) is not int or not 0 <= aggregate_index < len(specs):
            raise ValueError("unknown aggregate descriptor index")
        spec = specs[aggregate_index]
        query = QueryExpr(
            op="sem_agg",
            inputs=query.inputs,
            params={
                "input_cols": spec.input_cols,
                "output_cols": spec.output_cols,
                "instruction": spec.instruction,
            },
        )
    elif query.op != "sem_agg" or aggregate_index != 0:
        raise ValueError(
            "maintenance target must be sem_agg or independent sem_groupby"
        )
    if query.inputs[0].op in {"over", "sem_groupby"}:
        raise NotImplementedError(
            "maintenance plans do not target windows or legacy grouping"
        )
    return normalize_fallback(query)


def target_key(target: Relation | QueryExpr) -> str:
    """Identify the original query independently of Agent output."""
    query = target.expr if isinstance(target, Relation) else target
    return hashlib.sha256(stable_json(query).encode()).hexdigest()


@dataclass(frozen=True)
class MaintenancePlan:
    """A frozen Agent configuration bound by the program to one semantic task.

    state_schema is null, or {value: JSON Schema, membership: binding or null}.
    A membership binding names a top-level object array and its record-ID field;
    assignment object numbers are positions in that array, not persistent IDs.
    """

    target: QueryExpr
    rule: str
    state_schema: Mapping[str, Any] | None = None
    maintenance_instruction: str | None = None
    rationale: str = ""
    aggregate_index: int = 0

    def __post_init__(self) -> None:
        query = maintenance_target(self.target, self.aggregate_index)
        group = query.op == "sem_groupby"
        prefix = "group" if group else "agg"
        if not isinstance(self.rule, str) or self.rule not in {
            f"{prefix}-fallback",
            f"{prefix}-composable",
            f"{prefix}-state",
        }:
            raise ValueError("maintenance rule is not available for this target")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be text")
        if self.rule.endswith("-state"):
            if (
                not isinstance(self.maintenance_instruction, str)
                or not self.maintenance_instruction.strip()
            ):
                raise ValueError("state maintenance requires a maintenance instruction")
            if not isinstance(self.state_schema, Mapping) or set(self.state_schema) != {
                "value",
                "membership",
            }:
                raise ValueError("state_schema must contain value and membership")
            value = self.state_schema["value"]
            validate_state_schema(value)
            binding = self.state_schema["membership"]
            if group:
                if (
                    not isinstance(binding, Mapping)
                    or set(binding) != {"objects", "record_ids"}
                    or not all(isinstance(v, str) and v for v in binding.values())
                ):
                    raise ValueError("group-state requires an explicit member binding")
                objects = value.get("properties", {}).get(binding["objects"], {})
                members = (
                    objects.get("items", {})
                    .get("properties", {})
                    .get(binding["record_ids"], {})
                )
                if (
                    value.get("type") != "object"
                    or objects.get("type") != "array"
                    or objects.get("items", {}).get("type") != "object"
                    or members.get("type") != "array"
                    or members.get("items", {}).get("type") != "string"
                ):
                    raise ValueError(
                        "member binding must reference an object array with string record IDs"
                    )
            elif binding is not None:
                raise ValueError("aggregation state does not have a member binding")
        elif self.state_schema is not None or self.maintenance_instruction is not None:
            raise ValueError("no-state rules must not declare auxiliary state")
        object.__setattr__(self, "state_schema", _freeze_param(self.state_schema))

    @property
    def key(self) -> tuple[str, int]:
        """Return the compiler binding key."""
        return target_key(self.target), self.aggregate_index

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON configuration, without allowing Agent-selected node IDs."""
        return {
            "rule": self.rule,
            "state_schema": stable_value(self.state_schema),
            "maintenance_instruction": self.maintenance_instruction,
            "rationale": self.rationale,
        }

    @classmethod
    def bind(
        cls,
        target: Relation | QueryExpr,
        config: Mapping[str, Any],
        *,
        aggregate_index: int = 0,
    ) -> MaintenancePlan:
        """Bind saved or generated configuration to an explicit authoring target."""
        if set(config) != {
            "rule",
            "state_schema",
            "maintenance_instruction",
            "rationale",
        }:
            raise ValueError(
                "maintenance plan must contain exactly the four configuration fields"
            )
        return cls(
            target=target.expr if isinstance(target, Relation) else target,
            aggregate_index=aggregate_index,
            **dict(config),
        )
