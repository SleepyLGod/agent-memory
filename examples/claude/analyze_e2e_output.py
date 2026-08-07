"""Analyze existing ClaudeMemory e2e CSV outputs without running LOTUS."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
TEXT_PREVIEW_CHARS = 180

DUPLICATE_NAME_COLUMNS = (
    "view",
    "name",
    "count",
    "row_indices",
    "row_previews",
)
STRUCTURED_NAME_COLUMNS = (
    "event_order",
    "trace_timestamp",
    "trace_id",
    "operator",
    "row_index",
    "parsed_item_index",
    "shape",
    "name",
    "description",
    "type",
    "body_preview",
    "input_preview",
    "raw_output_preview",
    "parse_retry_attempts",
    "parse_error",
    "failure_artifact",
)
PAIRWISE_DECISION_COLUMNS = (
    "event_order",
    "trace_timestamp",
    "trace_id",
    "operator",
    "left_id",
    "right_id",
    "left_name",
    "right_name",
    "left_description",
    "right_description",
    "parsed_output",
    "raw_output_preview",
    "empty_raw_output",
    "default",
    "default_false",
    "instruction_preview",
)
NAME_LINEAGE_COLUMNS = (
    "event_order",
    "trace_timestamp",
    "trace_id",
    "boundary",
    "operator",
    "output_name",
    "left_name",
    "right_name",
    "parsed_output",
    "raw_output_preview",
    "input_preview",
    "final_ivm_name_exact",
    "final_duplicate_name_exact",
)
DIAGNOSIS_SUMMARY_COLUMNS = (
    "metric",
    "value",
)
DUPLICATE_IDENTITY_COLUMNS = (
    *DUPLICATE_NAME_COLUMNS,
    "first_operator",
    "first_trace_id",
    "first_phase",
    "first_add_index",
)
JOIN_CARDINALITY_COLUMNS = (
    "side",
    "row_id",
    "name",
    "match_count",
    "matched_ids",
    "matched_names",
    "trace_ids",
)
MERGE_OUTPUT_COLLISION_COLUMNS = (
    "phase",
    "add_index",
    "operator",
    "name",
    "count",
    "trace_ids",
    "input_previews",
    "raw_output_previews",
)
FRAGMENTATION_OBSERVATION_COLUMNS = (
    "view",
    "ivm_rows",
    "full_rows",
    "row_delta_ivm_minus_full",
    "observation",
    "reference_note",
)
LLM_CALL_COLUMNS = (
    "event_order",
    "trace_id",
    "event_type",
    "operator",
    "semantic_operator",
    "operator_call_id",
    "phase",
    "add_index",
    "llm_batch_id",
    "llm_item_index",
    "llm_batch_size",
    "model",
    "latency_sec",
    "prompt_path",
    "raw_output_path",
    "error_output_path",
    "error_type",
    "error_message",
    "usage_scope",
    "physical_total_tokens",
    "virtual_total_tokens",
    "cache_hits",
    "progress_bar_desc",
)

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local e2e output analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Existing .memory-test/claude-e2e-* directory with differential/view "
            "CSVs. Legacy ivm/full directories are also supported."
        ),
    )
    return parser.parse_args()


def read_required_csv(output_dir: Path, relative_path: str) -> pd.DataFrame:
    """Read a required CSV under the e2e output directory."""

    path = output_dir / relative_path
    if not path.exists():
        raise SystemExit(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def read_required_csv_any(output_dir: Path, *relative_paths: str) -> pd.DataFrame:
    """Read the first available required CSV from candidate paths."""

    for relative_path in relative_paths:
        path = output_dir / relative_path
        if path.exists():
            return pd.read_csv(path)
    candidates = ", ".join(str(output_dir / path) for path in relative_paths)
    raise SystemExit(f"Required CSV not found. Tried: {candidates}")


def read_optional_csv(output_dir: Path, relative_path: str) -> pd.DataFrame | None:
    """Read an optional CSV under the e2e output directory."""

    path = output_dir / relative_path
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_optional_csv_any(output_dir: Path, *relative_paths: str) -> pd.DataFrame | None:
    """Read the first available optional CSV from candidate paths."""

    for relative_path in relative_paths:
        path = output_dir / relative_path
        if path.exists():
            return pd.read_csv(path)
    return None


def write_csv(name: str, frame: Any, output_dir: Path) -> Path:
    """Write a DataFrame-like object to a named CSV and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def print_frame(name: str, frame: Any) -> None:
    """Print a compact DataFrame-like object for terminal inspection."""

    print(f"\n{name}:")
    print(frame.to_string(index=False))


def text_preview(value: Any, *, limit: int = TEXT_PREVIEW_CHARS) -> str:
    """Return a one-line text preview for diagnosis CSVs."""

    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def row_text(row: pd.Series) -> str:
    """Serialize a row into deterministic text for token-overlap matching."""

    parts: list[str] = []
    for column, value in row.items():
        if pd.isna(value):
            continue
        parts.append(f"{column}: {value}")
    return "\n".join(parts)


def token_set(text: str) -> set[str]:
    """Tokenize text for deterministic diagnosis matching."""

    return set(TOKEN_PATTERN.findall(text.lower()))


def token_overlap_score(left: str, right: str) -> float:
    """Return Jaccard token overlap between two row texts."""

    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def row_name(row: pd.Series) -> str:
    """Return a row's logical name when present."""

    if "name" not in row.index or pd.isna(row["name"]):
        return ""
    return str(row["name"])


def serialized_field(text: Any, field: str) -> str:
    """Extract a simple `field: value` line from a trace row preview."""

    prefix = f"{field}:"
    for line in str(text).splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def read_trace_events(output_dir: Path) -> list[dict[str, Any]]:
    """Read unified trace events when a trace tree exists."""

    events: list[dict[str, Any]] = []
    for events_path in (
        output_dir / "trace" / "differential" / "events.jsonl",
        output_dir / "trace" / "view" / "events.jsonl",
        output_dir / "trace" / "events.jsonl",
    ):
        if not events_path.exists():
            continue
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(json.loads(line))
    return events


def trace_artifact_value(output_dir: Path, event: dict[str, Any], key: str) -> Any:
    """Read a JSON trace artifact referenced by an event path field."""

    value = event.get(key)
    if value is not None:
        return value
    path_value = event.get(f"{key}_path")
    if not path_value:
        return None
    path_text = str(path_value)
    path = output_dir / path_text
    if not path.exists() and path_text.startswith("trace/"):
        path = output_dir / path_text
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def derived_output_dir(output_dir: Path) -> Path:
    """Return the directory for derived diagnosis artifacts."""

    return output_dir / "diagnosis"


def parsed_output_items(value: Any) -> list[tuple[int, dict[str, Any]]]:
    """Normalize structured trace parsed output into named object items."""

    if isinstance(value, dict):
        return [(0, value)]
    if isinstance(value, list):
        return [(index, item) for index, item in enumerate(value) if isinstance(item, dict)]
    return []


def view_duplicate_names(view_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return duplicate logical names from one final public view."""

    if "name" not in frame.columns:
        return []

    rows: list[dict[str, Any]] = []
    names = [str(value) if not pd.isna(value) else "" for value in frame["name"]]
    for name, count in Counter(name for name in names if name).items():
        if count <= 1:
            continue
        indices = [index for index, value in enumerate(names) if value == name]
        previews = [
            text_preview(row_text(frame.iloc[index]), limit=260)
            for index in indices
        ]
        rows.append(
            {
                "view": view_name,
                "name": name,
                "count": count,
                "row_indices": "|".join(str(index) for index in indices),
                "row_previews": " || ".join(previews),
            }
        )
    return rows


def duplicate_names_frame(
    *,
    ivm_topics: pd.DataFrame,
    ivm_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Build duplicate-name diagnosis for final IVM public views."""

    rows = [
        *view_duplicate_names("topics", ivm_topics),
        *view_duplicate_names("catalog", ivm_catalog),
    ]
    return pd.DataFrame(rows, columns=DUPLICATE_NAME_COLUMNS)


def final_ivm_names(ivm_topics: pd.DataFrame, ivm_catalog: pd.DataFrame) -> set[str]:
    """Return exact final IVM topic/catalog names."""

    names: set[str] = set()
    for frame in (ivm_topics, ivm_catalog):
        if "name" in frame.columns:
            names.update(str(value) for value in frame["name"].dropna() if str(value))
    return names


def diagnosis_summary_frame(
    *,
    duplicates: pd.DataFrame,
    structured_events: pd.DataFrame,
    pairwise_decisions: pd.DataFrame,
    name_lineage: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize first duplicate evidence without semantic judgment."""

    duplicate_names = sorted(set(duplicates["name"])) if not duplicates.empty else []
    first_duplicate = {
        "name": "",
        "operator": "",
        "boundary": "",
        "trace_id": "",
        "event_order": "",
    }
    if duplicate_names and not structured_events.empty:
        duplicate_events = structured_events[structured_events["name"].isin(duplicate_names)]
        for trace_id, group in duplicate_events.groupby("trace_id", sort=False):
            name_counts = Counter(str(value) for value in group["name"])
            repeated_names = [name for name, count in name_counts.items() if count > 1]
            if repeated_names:
                row = group[group["name"] == repeated_names[0]].sort_values("event_order").iloc[0]
                first_duplicate = {
                    "name": repeated_names[0],
                    "operator": str(row.get("operator") or ""),
                    "boundary": "structured_generation",
                    "trace_id": str(trace_id),
                    "event_order": str(row.get("event_order") or ""),
                }
                break

    sem_flat_map_names = (
        structured_events.loc[structured_events["operator"] == "sem_flat_map", "name"]
        if not structured_events.empty
        else pd.Series(dtype=str)
    )
    empty_raw_count = (
        int(pairwise_decisions["empty_raw_output"].sum())
        if not pairwise_decisions.empty
        else 0
    )
    default_false_count = (
        int(pairwise_decisions["default_false"].sum())
        if not pairwise_decisions.empty
        else 0
    )
    rows = [
        {"metric": "duplicate_final_names", "value": "|".join(duplicate_names)},
        {"metric": "duplicate_final_name_count", "value": len(duplicate_names)},
        {"metric": "structured_name_event_rows", "value": len(structured_events)},
        {"metric": "distinct_structured_names", "value": structured_events["name"].nunique() if not structured_events.empty else 0},
        {"metric": "sem_flat_map_distinct_names", "value": sem_flat_map_names.nunique()},
        {"metric": "pairwise_decision_rows", "value": len(pairwise_decisions)},
        {"metric": "pairwise_empty_raw_output_count", "value": empty_raw_count},
        {"metric": "pairwise_default_false_count", "value": default_false_count},
        {"metric": "first_exact_duplicate_name", "value": first_duplicate["name"]},
        {"metric": "first_exact_duplicate_operator", "value": first_duplicate["operator"]},
        {"metric": "first_exact_duplicate_boundary", "value": first_duplicate["boundary"]},
        {"metric": "first_exact_duplicate_trace_id", "value": first_duplicate["trace_id"]},
        {"metric": "first_exact_duplicate_event_order", "value": first_duplicate["event_order"]},
    ]
    return pd.DataFrame(rows, columns=DIAGNOSIS_SUMMARY_COLUMNS)


def build_trace_diagnosis(
    output_dir: Path,
    *,
    ivm_topics: pd.DataFrame,
    ivm_catalog: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    comparison_matches: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build derived diagnosis artifacts from unified trace events."""

    events = read_trace_events(output_dir)
    duplicates = duplicate_names_frame(ivm_topics=ivm_topics, ivm_catalog=ivm_catalog)
    final_names = final_ivm_names(ivm_topics, ivm_catalog)
    duplicate_final_names = set(duplicates["name"]) if not duplicates.empty else set()
    structured_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    llm_call_rows: list[dict[str, Any]] = []

    for event_order, event in enumerate(events, start=1):
        event_type = str(event.get("event_type", ""))
        operator = str(event.get("operator", ""))
        if event_type == "structured_generation":
            parsed_value = trace_artifact_value(output_dir, event, "parsed_output")
            raw_value = trace_artifact_value(output_dir, event, "raw_output")
            for parsed_item_index, item in parsed_output_items(parsed_value):
                name = str(item.get("name") or "")
                if not name:
                    continue
                structured_rows.append(
                    {
                        "event_order": event_order,
                        "trace_timestamp": event.get("timestamp", ""),
                        "trace_id": event.get("trace_id", ""),
                        "operator": operator,
                        "row_index": event.get("row_index", event.get("group_index")),
                        "parsed_item_index": parsed_item_index,
                        "shape": event.get("shape", ""),
                        "name": name,
                        "description": item.get("description", ""),
                        "type": item.get("type", ""),
                        "body_preview": text_preview(item.get("body", ""), limit=260),
                        "input_preview": text_preview(
                            event.get("input_preview", event.get("group_preview", "")),
                            limit=260,
                        ),
                        "raw_output_preview": text_preview(raw_value or "", limit=260),
                        "parse_retry_attempts": event.get("parse_retry_attempts", 0),
                        "parse_error": event.get("parse_error", ""),
                        "failure_artifact": event.get("failure_artifact", ""),
                        "phase": event.get("phase", ""),
                        "add_index": event.get("step_index", event.get("add_index", "")),
                    }
                )
        elif event_type == "pair_decision":
            left = event.get("left", "")
            right = event.get("right", "")
            raw_output = str(trace_artifact_value(output_dir, event, "raw_output") or "")
            parsed_output = str(trace_artifact_value(output_dir, event, "parsed_output") or "")
            empty_raw_output = not raw_output.strip()
            pairwise_rows.append(
                {
                    "event_order": event_order,
                    "trace_timestamp": event.get("timestamp", ""),
                    "trace_id": event.get("trace_id", ""),
                    "operator": operator,
                    "left_id": event.get("left_id", event.get("left_unique_id", "")),
                    "right_id": event.get("right_id", event.get("right_unique_id", "")),
                    "left_name": serialized_field(left, "name"),
                    "right_name": serialized_field(right, "name"),
                    "left_description": serialized_field(left, "description"),
                    "right_description": serialized_field(right, "description"),
                    "parsed_output": parsed_output,
                    "raw_output_preview": text_preview(raw_output, limit=260),
                    "empty_raw_output": empty_raw_output,
                    "default": event.get("default", ""),
                    "default_false": empty_raw_output and parsed_output == "False",
                    "instruction_preview": text_preview(event.get("instruction", ""), limit=260),
                    "phase": event.get("phase", ""),
                    "add_index": event.get("step_index", event.get("add_index", "")),
                }
            )
        elif event_type in {"llm_call", "llm_batch_error"}:
            llm_kwargs = event.get("llm_kwargs")
            progress_bar_desc = ""
            if isinstance(llm_kwargs, dict):
                progress_bar_desc = str(llm_kwargs.get("progress_bar_desc", ""))
            llm_call_rows.append(
                {
                    "event_order": event_order,
                    "trace_id": event.get("trace_id", ""),
                    "event_type": event_type,
                    "operator": operator,
                    "semantic_operator": event.get("semantic_operator", ""),
                    "operator_call_id": event.get("operator_call_id", ""),
                    "phase": event.get("phase", ""),
                    "add_index": event.get("step_index", event.get("add_index", "")),
                    "llm_batch_id": event.get("llm_batch_id", ""),
                    "llm_item_index": event.get("llm_item_index", ""),
                    "llm_batch_size": event.get("llm_batch_size", ""),
                    "model": event.get("model", ""),
                    "latency_sec": event.get("latency_sec", ""),
                    "prompt_path": event.get("prompt_path", ""),
                    "raw_output_path": event.get("raw_output_path", ""),
                    "error_output_path": event.get("error_output_path", ""),
                    "error_type": event.get("error_type", ""),
                    "error_message": event.get("error_message", ""),
                    "usage_scope": event.get("usage_scope", ""),
                    "physical_total_tokens": event.get("usage_physical_total_tokens", ""),
                    "virtual_total_tokens": event.get("usage_virtual_total_tokens", ""),
                    "cache_hits": event.get("usage_cache_hits", ""),
                    "progress_bar_desc": progress_bar_desc,
                }
            )

    structured_events = pd.DataFrame(structured_rows)
    pairwise_decisions = pd.DataFrame(pairwise_rows)
    llm_calls = pd.DataFrame(llm_call_rows, columns=LLM_CALL_COLUMNS)
    if structured_events.empty:
        structured_events = pd.DataFrame(columns=(*STRUCTURED_NAME_COLUMNS, "trace_id", "phase", "add_index"))
    if pairwise_decisions.empty:
        pairwise_decisions = pd.DataFrame(columns=(*PAIRWISE_DECISION_COLUMNS, "trace_id", "phase", "add_index"))

    duplicate_identity_events = duplicate_identity_events_frame(
        duplicates=duplicates,
        structured_events=structured_events,
    )
    join_cardinality = join_cardinality_frame(pairwise_decisions)
    merge_collisions = merge_output_collisions_frame(structured_events)
    fragmentation = fragmentation_observations_frame(
        comparison_summary=comparison_summary,
        comparison_matches=comparison_matches,
    )
    summary = trace_summary_frame(
        duplicate_identity_events=duplicate_identity_events,
        pairwise_decisions=pairwise_decisions,
        join_cardinality=join_cardinality,
        merge_collisions=merge_collisions,
        fragmentation=fragmentation,
        llm_calls=llm_calls,
        final_names=final_names,
        duplicate_final_names=duplicate_final_names,
    )
    return {
        "duplicate_identity_events": duplicate_identity_events,
        "structured_name_events": structured_events,
        "pairwise_decisions": pairwise_decisions,
        "join_cardinality": join_cardinality,
        "merge_output_collisions": merge_collisions,
        "fragmentation_observations": fragmentation,
        "llm_calls": llm_calls,
        "summary": summary,
    }


def duplicate_identity_events_frame(
    *,
    duplicates: pd.DataFrame,
    structured_events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach first trace evidence to final duplicate identity rows."""

    rows: list[dict[str, Any]] = []
    for _index, duplicate in duplicates.iterrows():
        name = str(duplicate["name"])
        matches = structured_events[structured_events["name"] == name]
        first = matches.sort_values("event_order").head(1)
        first_row = first.iloc[0].to_dict() if not first.empty else {}
        rows.append(
            {
                **duplicate.to_dict(),
                "first_operator": first_row.get("operator", ""),
                "first_trace_id": first_row.get("trace_id", ""),
                "first_phase": first_row.get("phase", ""),
                "first_add_index": first_row.get("add_index", ""),
            }
        )
    return pd.DataFrame(rows, columns=DUPLICATE_IDENTITY_COLUMNS)


def join_cardinality_frame(pairwise_decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize sem_join match cardinality by left and right side."""

    if pairwise_decisions.empty:
        return pd.DataFrame(columns=JOIN_CARDINALITY_COLUMNS)
    selected = pairwise_decisions[
        (pairwise_decisions["operator"] == "sem_join")
        & (pairwise_decisions["parsed_output"].astype(str) == "True")
    ]
    rows: list[dict[str, Any]] = []
    for left_id, group in selected.groupby("left_id", dropna=False):
        rows.append(
            {
                "side": "left",
                "row_id": left_id,
                "name": "|".join(sorted({str(value) for value in group["left_name"] if str(value)})),
                "match_count": len(group),
                "matched_ids": "|".join(str(value) for value in group["right_id"]),
                "matched_names": "|".join(str(value) for value in group["right_name"]),
                "trace_ids": "|".join(str(value) for value in group["trace_id"]),
            }
        )
    for right_id, group in selected.groupby("right_id", dropna=False):
        rows.append(
            {
                "side": "right",
                "row_id": right_id,
                "name": "|".join(sorted({str(value) for value in group["right_name"] if str(value)})),
                "match_count": len(group),
                "matched_ids": "|".join(str(value) for value in group["left_id"]),
                "matched_names": "|".join(str(value) for value in group["left_name"]),
                "trace_ids": "|".join(str(value) for value in group["trace_id"]),
            }
        )
    return pd.DataFrame(rows, columns=JOIN_CARDINALITY_COLUMNS)


def merge_output_collisions_frame(structured_events: pd.DataFrame) -> pd.DataFrame:
    """Find same-name outputs produced by sem_map within one traced merge phase."""

    if structured_events.empty:
        return pd.DataFrame(columns=MERGE_OUTPUT_COLLISION_COLUMNS)
    selected = structured_events[
        (structured_events["operator"] == "sem_map")
        & (structured_events["name"].astype(str) != "")
    ]
    rows: list[dict[str, Any]] = []
    group_cols = ["phase", "add_index", "operator"]
    for group_key, group in selected.groupby(group_cols, dropna=False):
        for name, name_group in group.groupby("name"):
            if len(name_group) <= 1:
                continue
            rows.append(
                {
                    "phase": group_key[0],
                    "add_index": group_key[1],
                    "operator": group_key[2],
                    "name": name,
                    "count": len(name_group),
                    "trace_ids": "|".join(str(value) for value in name_group["trace_id"]),
                    "input_previews": " || ".join(str(value) for value in name_group["input_preview"]),
                    "raw_output_previews": " || ".join(str(value) for value in name_group["raw_output_preview"]),
                }
            )
    return pd.DataFrame(rows, columns=MERGE_OUTPUT_COLLISION_COLUMNS)


def fragmentation_observations_frame(
    *,
    comparison_summary: pd.DataFrame,
    comparison_matches: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Record IVM/full shape differences as observations, not failures."""

    rows: list[dict[str, Any]] = []
    for _index, row in comparison_summary.iterrows():
        view = str(row["view"])
        ivm_rows = int(row["ivm_rows"])
        full_rows = int(row["full_rows"])
        rows.append(
            {
                "view": view,
                "ivm_rows": ivm_rows,
                "full_rows": full_rows,
                "row_delta_ivm_minus_full": ivm_rows - full_rows,
                "observation": "fragmented" if ivm_rows > full_rows else "compressed" if ivm_rows < full_rows else "same_count",
                "reference_note": "full recompute is an analysis reference, not a correctness baseline",
            }
        )
    for name, matches in comparison_matches.items():
        if matches.empty:
            continue
        rows.append(
            {
                "view": name,
                "ivm_rows": "",
                "full_rows": "",
                "row_delta_ivm_minus_full": "",
                "observation": "best_match_overlap",
                "reference_note": f"min_token_overlap={matches['token_overlap_score'].min()} max_token_overlap={matches['token_overlap_score'].max()}",
            }
        )
    return pd.DataFrame(rows, columns=FRAGMENTATION_OBSERVATION_COLUMNS)


def trace_summary_frame(
    *,
    duplicate_identity_events: pd.DataFrame,
    pairwise_decisions: pd.DataFrame,
    join_cardinality: pd.DataFrame,
    merge_collisions: pd.DataFrame,
    fragmentation: pd.DataFrame,
    llm_calls: pd.DataFrame,
    final_names: set[str],
    duplicate_final_names: set[str],
) -> pd.DataFrame:
    """Summarize trace-derived diagnosis in compact key/value form."""

    one_to_many = (
        int((join_cardinality["match_count"] > 1).sum())
        if not join_cardinality.empty and "match_count" in join_cardinality
        else 0
    )
    empty_raw = (
        int(pairwise_decisions["empty_raw_output"].sum())
        if not pairwise_decisions.empty
        else 0
    )
    llm_error_count = (
        int((llm_calls["event_type"] == "llm_batch_error").sum())
        if not llm_calls.empty and "event_type" in llm_calls
        else 0
    )
    llm_operators = (
        "|".join(sorted({str(value) for value in llm_calls["operator"] if str(value)}))
        if not llm_calls.empty and "operator" in llm_calls
        else ""
    )
    rows = [
        {"metric": "final_name_count", "value": len(final_names)},
        {"metric": "duplicate_final_names", "value": "|".join(sorted(duplicate_final_names))},
        {"metric": "duplicate_identity_event_rows", "value": len(duplicate_identity_events)},
        {"metric": "sem_join_cardinality_rows_gt_1", "value": one_to_many},
        {"metric": "merge_output_collision_rows", "value": len(merge_collisions)},
        {"metric": "pairwise_empty_raw_output_count", "value": empty_raw},
        {"metric": "fragmentation_observation_rows", "value": len(fragmentation)},
        {"metric": "llm_call_rows", "value": len(llm_calls)},
        {"metric": "llm_batch_error_rows", "value": llm_error_count},
        {"metric": "llm_operators_covered", "value": llm_operators},
    ]
    return pd.DataFrame(rows, columns=DIAGNOSIS_SUMMARY_COLUMNS)


def best_match_rows(
    *,
    view_name: str,
    left_name: str,
    left: pd.DataFrame,
    right_name: str,
    right: pd.DataFrame,
    direction: str,
) -> list[dict[str, Any]]:
    """Match each left row to the highest token-overlap right row."""

    rows: list[dict[str, Any]] = []
    right_texts = [row_text(row) for _index, row in right.iterrows()]
    for left_index, left_row in left.iterrows():
        left_text = row_text(left_row)
        if right.empty:
            rows.append(
                {
                    "view": view_name,
                    "match_direction": direction,
                    "ivm_index": left_index if left_name == "ivm" else pd.NA,
                    "full_index": left_index if left_name == "full" else pd.NA,
                    "ivm_name": row_name(left_row) if left_name == "ivm" else "",
                    "full_name": row_name(left_row) if left_name == "full" else "",
                    "token_overlap_score": 0.0,
                    "ivm_text_preview": text_preview(left_text) if left_name == "ivm" else "",
                    "full_text_preview": text_preview(left_text) if left_name == "full" else "",
                }
            )
            continue

        scored = [
            (token_overlap_score(left_text, right_text), right_index, right_row, right_text)
            for (right_index, right_row), right_text in zip(right.iterrows(), right_texts)
        ]
        score, right_index, right_row, right_text = max(scored, key=lambda item: item[0])
        ivm_row = left_row if left_name == "ivm" else right_row
        full_row = right_row if right_name == "full" else left_row
        rows.append(
            {
                "view": view_name,
                "match_direction": direction,
                "ivm_index": left_index if left_name == "ivm" else right_index,
                "full_index": right_index if right_name == "full" else left_index,
                "ivm_name": row_name(ivm_row),
                "full_name": row_name(full_row),
                "token_overlap_score": round(score, 4),
                "ivm_text_preview": text_preview(left_text if left_name == "ivm" else right_text),
                "full_text_preview": text_preview(right_text if right_name == "full" else left_text),
            }
        )
    return rows


def exact_name_overlap(ivm: pd.DataFrame, full: pd.DataFrame) -> set[str]:
    """Return exact overlap of non-empty name values when both views have names."""

    if "name" not in ivm.columns or "name" not in full.columns:
        return set()
    ivm_names = {str(value) for value in ivm["name"].dropna() if str(value)}
    full_names = {str(value) for value in full["name"].dropna() if str(value)}
    return ivm_names & full_names


def compare_view(
    view_name: str,
    *,
    ivm: pd.DataFrame,
    full: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build objective summary and best-match rows for one public view."""

    ivm_columns = tuple(str(column) for column in ivm.columns)
    full_columns = tuple(str(column) for column in full.columns)
    name_overlap = exact_name_overlap(ivm, full)
    summary = {
        "view": view_name,
        "schema_equal": ivm_columns == full_columns,
        "ivm_rows": len(ivm),
        "full_rows": len(full),
        "ivm_columns": "|".join(ivm_columns),
        "full_columns": "|".join(full_columns),
        "name_exact_overlap_count": len(name_overlap),
        "name_exact_overlap": "|".join(sorted(name_overlap)),
    }
    matches = best_match_rows(
        view_name=view_name,
        left_name="ivm",
        left=ivm,
        right_name="full",
        right=full,
        direction="ivm_to_full",
    )
    matches.extend(
        best_match_rows(
            view_name=view_name,
            left_name="full",
            left=full,
            right_name="ivm",
            right=ivm,
            direction="full_to_ivm",
        )
    )
    return summary, pd.DataFrame(matches)


def compare_public_views(
    *,
    ivm_topics: pd.DataFrame,
    full_topics: pd.DataFrame,
    ivm_catalog: pd.DataFrame,
    full_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare IVM and full public views without semantic judgment."""

    topic_summary, topic_matches = compare_view(
        "topics",
        ivm=ivm_topics,
        full=full_topics,
    )
    catalog_summary, catalog_matches = compare_view(
        "catalog",
        ivm=ivm_catalog,
        full=full_catalog,
    )
    return pd.DataFrame([topic_summary, catalog_summary]), {
        "topics_matches": topic_matches,
        "catalog_matches": catalog_matches,
    }


def print_comparison(summary: pd.DataFrame, matches_by_name: dict[str, pd.DataFrame]) -> None:
    """Print deterministic comparison artifacts for terminal inspection."""

    print_frame("comparison summary", summary)
    for name, matches in matches_by_name.items():
        print_frame(f"comparison {name}", matches)


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 12) -> str:
    """Render a compact markdown table without optional tabulate dependency."""

    if frame.empty:
        return "_No rows._"

    display = frame.head(max_rows).fillna("")
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for _index, row in display.iterrows():
        values = [
            str(row[column]).replace("\n", " ").replace("|", "\\|")
            for column in display.columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    *,
    comparison_summary: pd.DataFrame,
    comparison_matches: dict[str, pd.DataFrame],
    diagnosis: dict[str, pd.DataFrame],
    metrics_summary: pd.DataFrame | None,
    metrics_steps: pd.DataFrame | None,
    metrics_phases: pd.DataFrame | None,
) -> Path:
    """Write a concise human-readable markdown report for one e2e output directory."""

    duplicate_frame = diagnosis.get("duplicate_identity_events")
    if duplicate_frame is None:
        duplicate_frame = diagnosis.get("duplicate_names", pd.DataFrame())
    join_cardinality = diagnosis.get("join_cardinality", pd.DataFrame())
    merge_collisions = diagnosis.get("merge_output_collisions", pd.DataFrame())
    fragmentation = diagnosis.get("fragmentation_observations", pd.DataFrame())
    llm_calls = diagnosis.get("llm_calls", pd.DataFrame())

    lines = [
        "# ClaudeMemory E2E Report",
        "",
        f"Output directory: `{output_dir}`",
        "",
        "## Run Status",
        "",
        markdown_table(metrics_summary) if metrics_summary is not None else "_No metrics summary found._",
        "",
        "## Identity",
        "",
        "Duplicate logical identity is the primary issue under diagnosis.",
        "",
        "### Duplicate Identity Events",
        "",
        markdown_table(duplicate_frame, max_rows=8),
        "",
        "### Join Cardinality",
        "",
        markdown_table(join_cardinality, max_rows=8),
        "",
        "### Merge Output Collisions",
        "",
        markdown_table(merge_collisions, max_rows=8),
        "",
        "## Streaming Shape",
        "",
        "Full recompute is an analysis reference, not a correctness baseline. Fragmentation is an observation.",
        "",
        markdown_table(fragmentation if not fragmentation.empty else comparison_summary, max_rows=8),
        "",
        "## Reliability",
        "",
        "Token counts come from LOTUS/LiteLLM usage counters. Failed transport attempts may not return usage.",
        "",
    ]

    if not llm_calls.empty:
        lines.extend(
            [
                "### LLM Calls",
                "",
                "`diagnosis/llm_calls.csv` indexes full prompt and raw-output artifacts.",
                "",
                markdown_table(llm_calls, max_rows=8),
                "",
            ]
        )

    if metrics_steps is not None:
        lines.extend(["### Steps", "", markdown_table(metrics_steps, max_rows=8), ""])
    if metrics_phases is not None:
        lines.extend(["### Phases", "", markdown_table(metrics_phases, max_rows=8), ""])

    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    """Print and write deterministic comparison for an existing e2e output."""

    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.exists():
        raise SystemExit(f"E2E output directory not found: {output_dir}")

    ivm_topics = read_required_csv_any(
        output_dir,
        "differential/topics.csv",
        "ivm/topics.csv",
    )
    ivm_catalog = read_required_csv_any(
        output_dir,
        "differential/catalog.csv",
        "ivm/catalog.csv",
    )
    full_topics = read_required_csv_any(output_dir, "view/topics.csv", "full/topics.csv")
    full_catalog = read_required_csv_any(
        output_dir,
        "view/catalog.csv",
        "full/catalog.csv",
    )
    metrics_summary = read_optional_csv(output_dir, "metrics/summary.csv")
    metrics_steps = read_optional_csv(output_dir, "metrics/steps.csv")
    metrics_phases = read_optional_csv(output_dir, "metrics/phases.csv")
    trace_metrics = read_optional_csv_any(
        output_dir,
        "trace/differential/metrics.csv",
        "trace/view/metrics.csv",
        "trace/metrics.csv",
    )
    if metrics_phases is None and trace_metrics is not None:
        metrics_phases = trace_metrics

    print(f"Analyzing ClaudeMemory e2e output: {output_dir}")
    print_frame("differential topics", ivm_topics)
    print_frame("view topics", full_topics)
    print_frame("differential catalog", ivm_catalog)
    print_frame("view catalog", full_catalog)

    comparison_summary, comparison_matches = compare_public_views(
        ivm_topics=ivm_topics,
        full_topics=full_topics,
        ivm_catalog=ivm_catalog,
        full_catalog=full_catalog,
    )
    diagnosis = build_trace_diagnosis(
        output_dir,
        ivm_topics=ivm_topics,
        ivm_catalog=ivm_catalog,
        comparison_summary=comparison_summary,
        comparison_matches=comparison_matches,
    )
    print_comparison(comparison_summary, comparison_matches)
    print_frame("diagnosis summary", diagnosis["summary"])
    duplicate_print = diagnosis.get(
        "duplicate_identity_events",
        diagnosis.get("duplicate_names", pd.DataFrame()),
    )
    print_frame("diagnosis duplicate_identity_events", duplicate_print)

    comparison_dir = output_dir / "comparison"
    written = {
        "comparison/summary": write_csv("summary", comparison_summary, comparison_dir),
    }
    for name, matches in comparison_matches.items():
        written[f"comparison/{name}"] = write_csv(name, matches, comparison_dir)

    diagnosis_dir = derived_output_dir(output_dir)
    for name, frame in diagnosis.items():
        prefix = "diagnosis"
        written[f"{prefix}/{name}"] = write_csv(name, frame, diagnosis_dir)

    report_path = write_report(
        output_dir,
        comparison_summary=comparison_summary,
        comparison_matches=comparison_matches,
        diagnosis=diagnosis,
        metrics_summary=metrics_summary,
        metrics_steps=metrics_steps,
        metrics_phases=metrics_phases,
    )
    written["report"] = report_path

    print("\nwrote analysis artifacts:")
    for name, path_value in written.items():
        print(f"- {name}: {path_value}")


if __name__ == "__main__":
    main()
