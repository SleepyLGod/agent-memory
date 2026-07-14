"""Aggregate descriptors for grouped relation authoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from .logical import ColumnSpec


@dataclass(frozen=True)
class SemanticAggregateSpec:
    """Serializable descriptor for one grouped semantic aggregate."""

    input_cols: tuple[str, ...] | None
    output_cols: tuple[ColumnSpec, ...]
    instruction: str


@dataclass(frozen=True)
class ArrayAggregateSpec:
    """Serializable descriptor for one grouped array aggregate."""

    columns: tuple[str, ...]
    output_col: str


@dataclass(frozen=True)
class CollectListAggregateSpec:
    """Serializable descriptor for one grouped value-list aggregate."""

    column: str
    output_col: str


@dataclass(frozen=True)
class MinAggregateSpec:
    """Serializable descriptor for one deterministic minimum aggregate."""

    columns: tuple[str, ...]
    output_col: str


AggregateSpec: TypeAlias = (
    SemanticAggregateSpec
    | ArrayAggregateSpec
    | CollectListAggregateSpec
    | MinAggregateSpec
)
ColumnOutput: TypeAlias = Sequence[str] | Mapping[str, str]


def sem_agg(
    *,
    input_cols: Sequence[str] | None = None,
    output_cols: ColumnOutput,
    instruction: str,
) -> SemanticAggregateSpec:
    """Declare a semantic aggregate function for grouped ``agg(...)``."""

    normalized_output = _normalize_output_cols(output_cols)
    if not normalized_output:
        raise ValueError("sem_agg aggregate spec requires output_cols")
    if not instruction:
        raise ValueError("sem_agg aggregate spec instruction cannot be empty")
    return SemanticAggregateSpec(
        input_cols=(
            None
            if input_cols is None
            else tuple(str(column) for column in input_cols)
        ),
        output_cols=normalized_output,
        instruction=str(instruction),
    )


def array_agg(*, columns: Sequence[str], output_col: str) -> ArrayAggregateSpec:
    """Declare an array aggregate function for grouped ``agg(...)``."""

    normalized_columns = tuple(str(column) for column in columns)
    if not normalized_columns:
        raise ValueError("array_agg aggregate spec columns cannot be empty")
    if not output_col:
        raise ValueError("array_agg aggregate spec output_col cannot be empty")
    return ArrayAggregateSpec(columns=normalized_columns, output_col=str(output_col))


def collect_list(*, column: str, output_col: str) -> CollectListAggregateSpec:
    """Declare a deterministic grouped list aggregate over one column."""

    if not column:
        raise ValueError("collect_list aggregate spec column cannot be empty")
    if not output_col:
        raise ValueError("collect_list aggregate spec output_col cannot be empty")
    return CollectListAggregateSpec(column=str(column), output_col=str(output_col))


def min(
    *,
    column: str | None = None,
    columns: Sequence[str] | None = None,
    output_col: str,
) -> MinAggregateSpec:
    """Declare a deterministic minimum aggregate for grouped ``agg(...)``."""

    normalized_columns = normalize_min_columns(column=column, columns=columns)
    if not output_col:
        raise ValueError("min aggregate spec output_col cannot be empty")
    return MinAggregateSpec(columns=normalized_columns, output_col=str(output_col))


def normalize_min_columns(
    *,
    column: str | None,
    columns: Sequence[str] | None,
) -> tuple[str, ...]:
    """Normalize the mutually exclusive scalar and composite min inputs."""

    if (column is None) == (columns is None):
        raise ValueError("min requires exactly one of column or columns")
    if column is not None:
        if not column:
            raise ValueError("min column cannot be empty")
        return (str(column),)
    if isinstance(columns, (str, bytes)):
        raise TypeError("min columns must be a sequence of column names")
    normalized = tuple(str(name) for name in columns or ())
    if not normalized or any(not name for name in normalized):
        raise ValueError("min columns cannot be empty")
    return normalized


def aggregate_output_names(spec: AggregateSpec) -> tuple[str, ...]:
    """Return output column names produced by one aggregate descriptor."""

    if isinstance(spec, SemanticAggregateSpec):
        return tuple(column.name for column in spec.output_cols)
    if isinstance(spec, (ArrayAggregateSpec, CollectListAggregateSpec, MinAggregateSpec)):
        return (spec.output_col,)
    raise TypeError(f"Unsupported aggregate spec: {type(spec).__name__}")


def normalize_aggregate_specs(specs: Sequence[AggregateSpec]) -> tuple[AggregateSpec, ...]:
    """Validate and freeze aggregate descriptors passed to grouped ``agg(...)``."""

    normalized = tuple(specs)
    if not normalized:
        raise ValueError("agg requires at least one aggregate spec")
    invalid = [
        type(spec).__name__
        for spec in normalized
        if not isinstance(
            spec,
            (
                SemanticAggregateSpec,
                ArrayAggregateSpec,
                CollectListAggregateSpec,
                MinAggregateSpec,
            ),
        )
    ]
    if invalid:
        raise TypeError(
            "agg accepts only agent_memory.sem_agg(...), agent_memory.array_agg(...), "
            "agent_memory.collect_list(...), or agent_memory.min(...) "
            f"aggregate specs; got {invalid}"
        )
    outputs = [
        output
        for spec in normalized
        for output in aggregate_output_names(spec)
    ]
    duplicates = sorted({output for output in outputs if outputs.count(output) > 1})
    if duplicates:
        raise ValueError(f"aggregate output columns must be unique: {duplicates}")
    return normalized


def _normalize_output_cols(output_cols: ColumnOutput) -> tuple[ColumnSpec, ...]:
    """Normalize output column declarations into ColumnSpec tuples."""

    if isinstance(output_cols, Mapping):
        return tuple(
            ColumnSpec(
                name=str(name),
                description=None if description is None else str(description),
            )
            for name, description in output_cols.items()
        )
    return tuple(ColumnSpec(name=str(name)) for name in output_cols)
