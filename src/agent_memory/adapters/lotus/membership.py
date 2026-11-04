"""Full-partition semantic membership generation, without connectivity closure."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from agent_memory.policy.grouping import GROUP_ID, RECORD_ID
from agent_memory.policy.logical import ColumnSpec, QueryExpr
from .context import LotusExecutionContext
from .structured import StructuredLMExecutor


def validate_memberships(
    rows: Sequence[Mapping[str, Any]],
    size: int,
    membership: str,
    C: int | None = None,
) -> tuple[list[int], list[int]]:
    """Reject incomplete, ambiguous or duplicated membership output."""
    seen: set[tuple[int, int]] = set()
    groups: set[int] = set()
    counts = [0] * size
    positions: list[int] = []
    group_numbers: list[int] = []
    for row in rows:
        index = row.get("record")
        group = row.get("group")
        if type(index) is not int or not 0 <= index < size:
            raise ValueError("grouping returned an unknown record")
        if type(group) is not int or group < 0:
            raise ValueError("grouping requires nonnegative integer group numbers")
        groups.add(group)
        pair = (index, group)
        if pair in seen:
            raise ValueError("grouping returned duplicate membership")
        seen.add(pair)
        counts[index] += 1
        positions.append(index)
        group_numbers.append(group)
    if any(count == 0 or (membership == "single" and count != 1) for count in counts):
        raise ValueError("grouping violates membership coverage or cardinality")
    if C is not None and len(groups) > C:
        raise ValueError("grouping returned more groups than target C")
    return positions, group_numbers


def execute_memberships(
    query: QueryExpr, source: pd.DataFrame, context: LotusExecutionContext
) -> pd.DataFrame:
    """Generate and validate complete memberships before publishing any rows."""
    columns = list(query.params["columns"])
    if source.empty:
        return pd.DataFrame(columns=columns)
    if not source.index.is_unique:
        raise ValueError("grouping requires unique upstream occurrence identities")
    partitions = tuple(query.params["partition_by"])
    C = query.params.get("C")
    parts = (
        source.groupby(list(partitions), dropna=False, sort=False, observed=True)
        if partitions
        else [(None, source)]
    )
    # Validate all partitions before spending any model calls on this input.
    if C is not None:
        sizes = parts.size() if partitions else pd.Series([len(source)])
        if (sizes[sizes > 0] < C).any():
            raise ValueError("semantic grouping partition has fewer records than target C")
    context.configure()
    result: list[pd.DataFrame] = []
    output = tuple(
        ColumnSpec(name=name, description=description)
        for name, description in (
            ("record", "Integer record number from the input."),
            (
                "group",
                "Nonnegative integer identifying a group within this input.",
            ),
        )
    )
    for partition_number, (_, part) in enumerate(parts):
        records = [
            {"record": index, "content": row}
            for index, row in enumerate(
                part.loc[:, list(query.params["input_cols"])].to_dict("records")
            )
        ]
        payload = pd.DataFrame(
            {"records": [json.dumps(records, ensure_ascii=False, default=str)]}
        )
        constraints = (
            "Return one row per membership, not one row per group. "
            "Use the supplied integer record numbers and nonnegative integer group numbers. "
            + (
                "Every record must have exactly one membership. "
                if query.params["membership"] == "single"
                else "Every record must have one or more memberships. "
            )
            + "No duplicate memberships or omitted records are allowed. "
        )
        if C is not None:
            constraints += (
                f"The target number of groups is {C}. Aim for {C} groups; "
                "fewer nonempty groups are allowed, but do not exceed the target. "
            )
        generated = StructuredLMExecutor(payload)(
            input_cols=("records",),
            output_cols=output,
            shape="array",
            instruction=(
                str(query.params["instruction"]).replace("{", "{{").replace("}", "}}")
                + "\n\n" + constraints + "\nInput records: {records}"
            ),
            format_example={"record": 0, "group": 0},
            progress_bar_desc="Semantic grouping",
            model_kwargs={},
            structured_max_tokens=context.config.structured_max_tokens,
            structured_parse_retries=context.config.structured_parse_retries,
            structured_output_transport=context.config.structured_output_transport,
            operator="sem_groupby",
        )
        positions, numbers = validate_memberships(
            generated.parsed_outputs[0], len(part), query.params["membership"], C
        )
        members = part.take(positions)
        members[RECORD_ID] = members.index.astype(str)
        # Build each snapshot-local group ID once, not once per member.
        identities = {
            group: json.dumps([partition_number, group]) for group in set(numbers)
        }
        members[GROUP_ID] = [identities[group] for group in numbers]
        result.append(members)
    return pd.concat(result, ignore_index=True).loc[:, columns]
