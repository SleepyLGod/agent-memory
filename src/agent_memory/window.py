"""Deterministic count-window helpers used by runtime and adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

WINDOW_SOURCE_INPUT = "__agent_memory_window_source"


@dataclass(frozen=True)
class CountWindowSpec:
    """Validated count-window assigner configuration."""

    size: int
    slide: int
    trigger: None = None


@dataclass(frozen=True)
class CountWindow:
    """One completed count window over an append-ordered source frame."""

    start: int
    end: int
    frame: pd.DataFrame


@dataclass(frozen=True)
class OverFrame:
    """One over-window frame for an emitted source row."""

    emit_position: int
    emit_row: pd.Series
    frame: pd.DataFrame


def parse_count_window_spec(params: Mapping[str, Any]) -> CountWindowSpec:
    """Validate and normalize count-window query parameters."""

    size = _require_positive_int(params["size"], name="size")
    slide = _require_positive_int(params.get("slide", 1), name="slide")
    trigger = params.get("trigger")
    if trigger is not None:
        raise NotImplementedError("count_window currently supports only trigger=None")
    return CountWindowSpec(size=size, slide=slide, trigger=None)


def count_window_params(params: Mapping[str, Any]) -> tuple[int, int]:
    """Validate and normalize count-window query parameters."""

    spec = parse_count_window_spec(params)
    return spec.size, spec.slide


def _require_positive_int(value: Any, *, name: str) -> int:
    """Require an integral positive count-window parameter."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"count_window {name} must be an integer")
    if value <= 0:
        raise ValueError(f"count_window {name} must be positive")
    return value


def completed_count_windows(
    source: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    next_start: int = 0,
) -> tuple[list[CountWindow], int]:
    """Return newly complete windows and the next unprocessed start offset."""

    spec = parse_count_window_spec(params)
    ordered = source.reset_index(drop=True).copy()
    row_count = len(ordered)
    windows: list[CountWindow] = []
    start = next_start
    while start + spec.size <= row_count:
        end = start + spec.size
        windows.append(
            CountWindow(
                start=start,
                end=end,
                frame=ordered.iloc[start:end].reset_index(drop=True).copy(),
            )
        )
        start += spec.slide
    return windows, start


def over_rows_params(params: Mapping[str, Any]) -> tuple[int, int]:
    """Validate first-scope over-window row bounds."""

    rows = params["rows"]
    if (
        not isinstance(rows, tuple)
        or len(rows) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in rows)
    ):
        raise TypeError("over rows must be a tuple of two integers")
    start, end = rows
    if start > end:
        raise ValueError("over rows start must be <= end")
    if end > 0:
        raise NotImplementedError("over currently supports only rows with N <= 0")
    return start, end


def over_frames(
    emit_source: pd.DataFrame,
    frame_source: pd.DataFrame,
    params: Mapping[str, Any],
) -> list[OverFrame]:
    """Return over frames for emit rows against a full append-ordered source."""

    start_offset, end_offset = over_rows_params(params)
    emit = emit_source.reset_index(drop=True).copy()
    source = frame_source.reset_index(drop=True).copy()
    if emit.empty:
        return []
    positions = _emit_positions(emit, source)
    frames: list[OverFrame] = []
    for emit_index, source_position in enumerate(positions):
        frame_start = max(0, source_position + start_offset)
        frame_end = min(len(source), source_position + end_offset + 1)
        if frame_end < frame_start:
            frame_end = frame_start
        frames.append(
            OverFrame(
                emit_position=source_position,
                emit_row=emit.iloc[emit_index].copy(),
                frame=source.iloc[frame_start:frame_end].reset_index(drop=True).copy(),
            )
        )
    return frames


def _emit_positions(emit: pd.DataFrame, source: pd.DataFrame) -> list[int]:
    """Locate append-only emit rows inside the full source state."""

    if list(emit.columns) != list(source.columns):
        raise ValueError("over emit rows must have the same columns as frame source")
    if len(emit) > len(source):
        raise ValueError("over emit rows cannot be longer than frame source")
    if len(emit) == len(source) and _frames_equal_by_value(source, emit):
        return list(range(len(source)))

    start = len(source) - len(emit)
    suffix = source.iloc[start:].reset_index(drop=True)
    if not _frames_equal_by_value(suffix, emit):
        raise NotImplementedError(
            "over differential currently requires emit rows to be an append suffix of the full source state"
        )
    return list(range(start, len(source)))


def _frames_equal_by_value(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Compare frame values without requiring identical pandas dtypes."""

    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for row_index in range(len(left)):
        for column in left.columns:
            if _normalize_cell(left.iloc[row_index][column]) != _normalize_cell(
                right.iloc[row_index][column]
            ):
                return False
    return True


def _normalize_cell(value: Any) -> Any:
    """Normalize pandas/numpy scalar values for row-value equality checks."""

    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value
