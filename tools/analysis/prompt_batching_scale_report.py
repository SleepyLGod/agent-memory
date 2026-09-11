"""Summarize and plot the 128-event prompt-batching scale experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import html
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


EVENT_COUNT = 128
QUESTION_COUNT = 71
EXPECTED_LABELS = (
    "original",
    "packed-1",
    "packed-2",
    "packed-4",
    "packed-8",
    "packed-16",
    "packed-32",
    "packed-64",
    "packed-128",
    "packed-all",
)
PASS_COLORS = {1: "#0F766E", 2: "#D97706"}


@dataclass(frozen=True)
class RunRow:
    """Normalized metrics for one completed condition."""

    run_id: str
    pass_index: int
    label: str
    refresh_every: int
    prompt_batch: int | str | None
    provider_calls: int
    physical_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    completion_tokens: int
    cost_cny: float
    insertion_mean_ms: float
    insertion_median_ms: float
    insertion_p95_ms: float
    insertion_max_ms: float
    condition_wall_seconds: float
    retrieval_mean_ms: float
    official_score: float
    zep_score: float
    topic_count: int | None
    catalog_count: int | None
    topic_digest: str | None
    catalog_digest: str | None
    artifact_bytes: int

    def to_csv(self) -> dict[str, Any]:
        """Return one flat CSV row with explicit units."""

        return {
            "run_id": self.run_id,
            "pass": self.pass_index,
            "condition": self.label,
            "refresh_every": self.refresh_every,
            "prompt_batch": self.prompt_batch,
            "provider_calls": self.provider_calls,
            "physical_tokens": self.physical_tokens,
            "physical_tokens_per_event": self.physical_tokens / EVENT_COUNT,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_cny": self.cost_cny,
            "cost_cny_per_event": self.cost_cny / EVENT_COUNT,
            "insertion_excluding_trace_mean_s": self.insertion_mean_ms / 1000,
            "insertion_excluding_trace_median_s": self.insertion_median_ms / 1000,
            "insertion_excluding_trace_p95_s": self.insertion_p95_ms / 1000,
            "insertion_excluding_trace_max_s": self.insertion_max_ms / 1000,
            "condition_end_to_end_wall_s": self.condition_wall_seconds,
            "condition_end_to_end_wall_s_per_event": (
                self.condition_wall_seconds / EVENT_COUNT
            ),
            "retrieval_mean_s": self.retrieval_mean_ms / 1000,
            "official_locomo_percent": self.official_score * 100,
            "zep_judge_percent": self.zep_score * 100,
            "topic_count": self.topic_count,
            "catalog_count": self.catalog_count,
            "topic_digest": self.topic_digest,
            "catalog_digest": self.catalog_digest,
            "artifact_bytes": self.artifact_bytes,
        }


def _required_number(value: Any, *, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _state_count(state: Mapping[str, Any], relation: str) -> int | None:
    row = state.get(relation)
    if not isinstance(row, Mapping):
        return None
    value = row.get("row_count")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _state_digest(state: Mapping[str, Any], relation: str) -> str | None:
    row = state.get(relation)
    if not isinstance(row, Mapping):
        return None
    value = row.get("multiset_digest")
    return str(value) if isinstance(value, str) else None


def _load_run(path: Path) -> tuple[RunRow, dict[str, Any]]:
    payload = json.loads(path.read_text())
    state = payload.get("final_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"missing final_state: {path}")
    run_id = str(payload.get("run_id") or "")
    label = str(payload.get("label") or "")
    pass_index = int(payload.get("pass_index") or 0)
    if not run_id or label not in EXPECTED_LABELS or pass_index not in (1, 2):
        raise ValueError(f"invalid run identity: {path}")
    row = RunRow(
        run_id=run_id,
        pass_index=pass_index,
        label=label,
        refresh_every=int(payload["refresh_every"]),
        prompt_batch=payload.get("prompt_batch"),
        provider_calls=int(payload["provider_call_count"]),
        physical_tokens=int(payload["physical_tokens"]),
        cache_hit_tokens=int(payload["cache_hit_tokens"]),
        cache_miss_tokens=int(payload["cache_miss_tokens"]),
        completion_tokens=int(payload["completion_tokens"]),
        cost_cny=_required_number(payload["cost_cny"], field="cost_cny"),
        insertion_mean_ms=_required_number(
            payload["insertion_excluding_trace_mean_ms"],
            field="insertion_excluding_trace_mean_ms",
        ),
        insertion_median_ms=_required_number(
            payload["insertion_excluding_trace_median_ms"],
            field="insertion_excluding_trace_median_ms",
        ),
        insertion_p95_ms=_required_number(
            payload["insertion_excluding_trace_p95_ms"],
            field="insertion_excluding_trace_p95_ms",
        ),
        insertion_max_ms=_required_number(
            payload["insertion_excluding_trace_max_ms"],
            field="insertion_excluding_trace_max_ms",
        ),
        condition_wall_seconds=_required_number(
            payload["condition_end_to_end_wall_seconds"],
            field="condition_end_to_end_wall_seconds",
        ),
        retrieval_mean_ms=_required_number(
            payload["retrieval_mean_ms"], field="retrieval_mean_ms"
        ),
        official_score=_required_number(
            payload["official_locomo_score"], field="official_locomo_score"
        ),
        zep_score=_required_number(payload["zep_judge_score"], field="zep_judge_score"),
        topic_count=_state_count(state, "topics"),
        catalog_count=_state_count(state, "catalog"),
        topic_digest=_state_digest(state, "topics"),
        catalog_digest=_state_digest(state, "catalog"),
        artifact_bytes=int(payload["artifact_bytes"]),
    )
    return row, payload


def load_completed_runs(root: Path) -> tuple[tuple[RunRow, ...], dict[str, dict[str, Any]]]:
    """Load completed runs in the immutable experiment-contract order."""

    contract_path = root / "control" / "experiment-contract.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("experiment") != "claude-prompt-batching-scale-128e":
        raise ValueError(f"not a prompt-batching scale experiment: {root}")
    rows: list[RunRow] = []
    raw: dict[str, dict[str, Any]] = {}
    continuation = contract.get("continuation") or {}
    reused = continuation.get("reused_conditions", {})
    for expected in contract.get("conditions") or ():
        run_id = str(expected["run_id"])
        directory = Path(reused[run_id]) if run_id in reused else root / "conditions" / run_id
        validation = directory / "validation.json"
        if not validation.exists():
            continue
        row, payload = _load_run(validation)
        if run_id in reused:
            row = replace(row, cost_cny=float(continuation["prior_condition_costs_cny"][run_id]))
        payload["result_origin"] = {
            "status": "reused-completed" if run_id in reused else "completed",
            "directory": str(directory),
            "source": (
                continuation.get("reused_sources", {}).get(run_id, continuation["source"])
                if run_id in reused else contract["source"]
            ),
            "repair_version": None if run_id in reused else (contract.get("structured_output") or {}).get("repair_version"),
        }
        if row.run_id != run_id:
            raise ValueError(f"run ID mismatch: {validation}")
        rows.append(row)
        raw[run_id] = payload
    return tuple(rows), raw


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _site_rows(raw_runs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, run in raw_runs.items():
        prompt = run.get("prompt_batching") or {}
        candidates = run.get("candidate_pairs") or {}
        usage = run.get("operator_usage") or {}
        for site in sorted(set(prompt) | set(candidates) | set(usage)):
            batching = prompt.get(site) or {}
            candidate = candidates.get(site) or {}
            operator = usage.get(site) or {}
            eligible = int(candidate.get("eligible") or 0)
            selected = int(candidate.get("selected") or 0)
            tasks = int(batching.get("tasks") or 0)
            prompts = int(batching.get("prompts") or 0)
            rows.append(
                {
                    "run_id": run_id,
                    "pass": run["pass_index"],
                    "condition": run["label"],
                    "site": site,
                    "ready_tasks": tasks,
                    "physical_prompts": prompts,
                    "tasks_per_prompt": tasks / prompts if prompts else None,
                    "max_chunk": int(batching.get("max_chunk") or 0),
                    "retries": int(batching.get("retries") or 0),
                    "syntax_repairs": int(batching.get("repairs") or 0),
                    "provider_responses": int(operator.get("responses") or 0),
                    "physical_tokens": int(operator.get("tokens") or 0),
                    "tokens_per_task": (
                        int(operator.get("tokens") or 0) / tasks if tasks else None
                    ),
                    "eligible_pairs": eligible,
                    "selected_pairs": selected,
                    "candidate_reduction": (
                        1 - selected / eligible if eligible else None
                    ),
                }
            )
    return rows


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#FAF7F0"/>'
        '<style>text{font-family:Arial,sans-serif;fill:#1F2937}'
        '.title{font-size:18px;font-weight:700}.axis{font-size:11px}'
        '.legend{font-size:12px}</style>'
        f"{body}</svg>\n"
    )


def _line_panels(
    rows: Sequence[RunRow],
    panels: Sequence[tuple[str, Callable[[RunRow], float], str]],
) -> str:
    width = 1180
    panel_height = 245
    height = 45 + panel_height * len(panels)
    parts = [
        '<text x="24" y="28" class="title">128-Message Batching Scale</text>',
        '<circle cx="850" cy="23" r="5" fill="#0F766E"/><text x="861" y="27" class="legend">Pass 1</text>',
        '<circle cx="935" cy="23" r="5" fill="#D97706"/><text x="946" y="27" class="legend">Pass 2</text>',
    ]
    for panel_index, (title, value, unit) in enumerate(panels):
        top = 45 + panel_index * panel_height
        left, right = 78, width - 30
        chart_top, bottom = top + 32, top + 190
        values = [value(row) for row in rows]
        maximum = max(values, default=1.0) or 1.0
        parts.extend(
            (
                f'<text x="24" y="{top + 19}" class="title">{html.escape(title)}</text>',
                f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9CA3AF"/>',
                f'<line x1="{left}" y1="{chart_top}" x2="{left}" y2="{bottom}" stroke="#9CA3AF"/>',
                f'<text x="8" y="{chart_top + 5}" class="axis">{maximum:.3g} {html.escape(unit)}</text>',
                f'<text x="42" y="{bottom + 4}" class="axis">0</text>',
            )
        )
        for pass_index in (1, 2):
            pass_rows = [row for row in rows if row.pass_index == pass_index]
            points: list[str] = []
            for index, label in enumerate(EXPECTED_LABELS):
                x = left + index * (right - left) / (len(EXPECTED_LABELS) - 1)
                matching = [row for row in pass_rows if row.label == label]
                if matching:
                    y = bottom - value(matching[0]) / maximum * (bottom - chart_top)
                    points.append(f"{x:.2f},{y:.2f}")
                    parts.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" '
                        f'fill="{PASS_COLORS[pass_index]}"/>'
                    )
                if panel_index == len(panels) - 1:
                    parts.append(
                        f'<text x="{x:.2f}" y="{bottom + 17}" class="axis" '
                        f'text-anchor="end" transform="rotate(-35 {x:.2f} {bottom + 17})">'
                        f"{html.escape(label)}</text>"
                    )
            if len(points) > 1:
                parts.append(
                    f'<polyline points="{" ".join(points)}" fill="none" '
                    f'stroke="{PASS_COLORS[pass_index]}" stroke-width="2"/>'
                )
    return _svg_document(width, height, "".join(parts))


def _pareto_svg(rows: Sequence[RunRow]) -> str:
    width, height = 1000, 620
    left, right, top, bottom = 90, 960, 55, 535
    costs = [row.cost_cny / EVENT_COUNT for row in rows]
    scores = [row.official_score * 100 for row in rows]
    max_cost = max(costs, default=1.0) or 1.0
    low_score = min(scores, default=0.0)
    high_score = max(scores, default=1.0)
    score_span = max(high_score - low_score, 1.0)
    body = [
        '<text x="24" y="28" class="title">Cost-Latency-Quality Pareto View</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9CA3AF"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#9CA3AF"/>',
        f'<text x="{(left + right) / 2}" y="590" text-anchor="middle">CNY per source event</text>',
        f'<text x="20" y="{(top + bottom) / 2}" transform="rotate(-90 20 {(top + bottom) / 2})" text-anchor="middle">Official LOCOMO (%)</text>',
    ]
    for row in rows:
        x = left + (row.cost_cny / EVENT_COUNT) / max_cost * (right - left)
        y = bottom - ((row.official_score * 100) - low_score) / score_span * (bottom - top)
        radius = 5 + min(13, math.sqrt(max(row.insertion_mean_ms, 0)) / 5)
        body.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'fill="{PASS_COLORS[row.pass_index]}" fill-opacity="0.72"/>'
        )
        body.append(
            f'<text x="{x + radius + 3:.2f}" y="{y + 4:.2f}" class="axis">'
            f"{html.escape(row.label)} p{row.pass_index}</text>"
        )
    return _svg_document(width, height, "".join(body))


def generate_report(root: Path) -> dict[str, Any]:
    """Generate bounded reports without modifying benchmark artifacts."""

    root = root.resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows, raw = load_completed_runs(root)
    csv_rows = [
        {**row.to_csv(), "result_status": raw[row.run_id]["result_origin"]["status"],
         "source_snapshot_sha256": raw[row.run_id]["result_origin"]["source"].get("source_snapshot_sha256"),
         "repair_version": raw[row.run_id]["result_origin"]["repair_version"]}
        for row in rows
    ]
    _write_csv(output / "conditions.csv", csv_rows)
    _write_csv(output / "site_metrics.csv", _site_rows(raw))

    cross_pass: dict[str, Any] = {}
    for label in EXPECTED_LABELS:
        matches = [row for row in rows if row.label == label]
        if len(matches) != 2:
            continue
        first, second = matches
        cross_pass[label] = {
            "same_source_snapshot": (
                raw[first.run_id]["result_origin"]["source"].get("source_snapshot_sha256")
                == raw[second.run_id]["result_origin"]["source"].get("source_snapshot_sha256")
            ),
            "topic_state_equal": first.topic_digest == second.topic_digest,
            "catalog_state_equal": first.catalog_digest == second.catalog_digest,
            "official_score_delta": second.official_score - first.official_score,
            "cost_cny_delta": second.cost_cny - first.cost_cny,
            "insertion_mean_ms_delta": (
                second.insertion_mean_ms - first.insertion_mean_ms
            ),
        }
    summary = {
        "schema_version": 1,
        "completed_conditions": len(rows),
        "expected_conditions": 20,
        "complete": len(rows) == 20,
        "total_cost_cny": sum(row.cost_cny for row in rows),
        "cross_pass": cross_pass,
        "result_origins": {key: value["result_origin"] for key, value in raw.items()},
        "interpretation_boundary": (
            "R and P co-vary. Each pass is shown separately; two passes are an "
            "exploratory repeatability check, not a variance estimate. Reused results "
            "retain their original source; cross-version differences are not pure run noise."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "efficiency.svg").write_text(
        _line_panels(
            rows,
            (
                (
                    "Physical tokens per source message",
                    lambda row: row.physical_tokens / EVENT_COUNT,
                    "tokens",
                ),
                (
                    "Insertion mean excluding semantic trace I/O",
                    lambda row: row.insertion_mean_ms / 1000,
                    "seconds",
                ),
                ("Actual condition cost", lambda row: row.cost_cny, "CNY"),
            ),
        ),
        encoding="utf-8",
    )
    (output / "quality.svg").write_text(
        _line_panels(
            rows,
            (
                ("Official LOCOMO", lambda row: row.official_score * 100, "%"),
                ("Zep judge C1-C4", lambda row: row.zep_score * 100, "%"),
                ("Topic count", lambda row: float(row.topic_count or 0), "rows"),
                ("Catalog count", lambda row: float(row.catalog_count or 0), "rows"),
            ),
        ),
        encoding="utf-8",
    )
    (output / "pareto.svg").write_text(_pareto_svg(rows), encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the requested report."""

    args = _parser().parse_args(argv)
    print(json.dumps(generate_report(args.run_root), sort_keys=True))


if __name__ == "__main__":
    main()
