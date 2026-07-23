"""Replay selected real benchmark prompts with DeepSeek thinking disabled."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from sys import path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(PROJECT_ROOT / "src"))

from agent_memory.adapters.lotus.sem_agg import (  # noqa: E402
    parse_structured_sem_agg_output,
)
from agent_memory.adapters.lotus.sem_flat_map import (  # noqa: E402
    parse_structured_flat_map_json,
)
from agent_memory.adapters.lotus.sem_map import parse_structured_map_json  # noqa: E402
from agent_memory.adapters.lotus.sem_topk_listwise import (  # noqa: E402
    _parse_selected_ids,
)
from agent_memory.evaluation.pricing import PricingSnapshot  # noqa: E402
from agent_memory.policy.logical import ColumnSpec  # noqa: E402

_SEMANTIC_OPERATORS = ("sem_flat_map", "sem_groupby", "sem_agg", "sem_map")


@dataclass(frozen=True)
class ReplayCase:
    """One real prompt plus the production output contract used to validate it."""

    case_id: str
    kind: str
    prompt_path: Path
    messages: tuple[Mapping[str, Any], ...]
    max_tokens: int
    response_format: Mapping[str, Any] | None
    baseline_latency_ms: float | None


def _read_json(path_: Path) -> Any:
    return json.loads(path_.read_text(encoding="utf-8"))


def _events(run_dir: Path) -> list[dict[str, Any]]:
    path_ = run_dir / "trace" / "events.jsonl"
    with path_.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_artifact(run_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("trace event is missing prompt_path")
    path_ = Path(value)
    return path_ if path_.is_absolute() else run_dir / path_


def _messages(path_: Path) -> tuple[Mapping[str, Any], ...]:
    value = _read_json(path_)
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"prompt artifact must contain a message array: {path_}")
    return tuple(dict(row) for row in value)


def _case_from_event(run_dir: Path, event: Mapping[str, Any], *, kind: str) -> ReplayCase:
    prompt_path = _resolve_artifact(run_dir, event.get("prompt_path"))
    kwargs = event.get("llm_kwargs")
    kwargs = dict(kwargs) if isinstance(kwargs, Mapping) else {}
    trace_id = str(event.get("trace_id") or prompt_path.stem)
    latency = event.get("latency_sec")
    return ReplayCase(
        case_id=f"{kind}-{sha256(trace_id.encode()).hexdigest()[:10]}",
        kind=kind,
        prompt_path=prompt_path,
        messages=_messages(prompt_path),
        max_tokens=int(kwargs.get("max_tokens") or 8192),
        response_format=(
            dict(kwargs["response_format"])
            if isinstance(kwargs.get("response_format"), Mapping)
            else None
        ),
        baseline_latency_ms=float(latency) * 1000 if latency is not None else None,
    )


def _extremes(
    run_dir: Path,
    events: Sequence[Mapping[str, Any]],
    *,
    operator: str,
    kind: str,
) -> tuple[ReplayCase, ReplayCase]:
    candidates = [
        event
        for event in events
        if event.get("event_type") == "llm_call"
        and event.get("operator") == operator
        and int(event.get("llm_item_index") or 0) == 0
        and isinstance(event.get("prompt_path"), str)
    ]
    if not candidates:
        raise ValueError(f"trace contains no real {operator} prompts")
    candidates.sort(
        key=lambda event: (
            int(event.get("usage_physical_prompt_tokens") or 0),
            str(event.get("trace_id") or ""),
        )
    )
    return (
        _case_from_event(run_dir, candidates[0], kind=f"{kind}-ordinary"),
        _case_from_event(run_dir, candidates[-1], kind=f"{kind}-heavy"),
    )


def select_replay_cases(
    *,
    operator_run_dir: Path,
    pairwise_run_dir: Path,
    listwise_run_dir: Path,
) -> tuple[ReplayCase, ...]:
    """Select deterministic ordinary/heavy requests from successful real traces."""

    selected: list[ReplayCase] = []
    operator_events = _events(operator_run_dir)
    for operator in _SEMANTIC_OPERATORS:
        selected.extend(
            _extremes(
                operator_run_dir,
                operator_events,
                operator=operator,
                kind=operator,
            )
        )
    selected.extend(
        _extremes(
            pairwise_run_dir,
            _events(pairwise_run_dir),
            operator="sem_topk",
            kind="pairwise-quick",
        )
    )
    selected.extend(
        _extremes(
            listwise_run_dir,
            _events(listwise_run_dir),
            operator="sem_topk",
            kind="listwise",
        )
    )
    answers = [
        event
        for event in operator_events
        if event.get("event_type") == "llm_call"
        and event.get("phase") == "answering"
        and isinstance(event.get("prompt_path"), str)
    ]
    if not answers:
        raise ValueError("trace contains no real LongMemEval answer request")
    selected.append(_case_from_event(operator_run_dir, answers[0], kind="answer"))
    return tuple(selected)


def _prompt_text(case: ReplayCase) -> str:
    return "\n".join(str(message.get("content") or "") for message in case.messages)


def _output_columns(case: ReplayCase) -> tuple[ColumnSpec, ...]:
    prompt = _prompt_text(case)
    markers = ("Expected JSON shape:", "Field descriptions:")
    for marker in markers:
        if marker not in prompt:
            continue
        candidate = prompt.rsplit(marker, 1)[1].lstrip()
        value, _end = json.JSONDecoder().raw_decode(candidate)
        if isinstance(value, Mapping) and value:
            return tuple(ColumnSpec(str(name), "") for name in value)
    if case.kind.startswith("sem_flat_map"):
        return tuple(
            ColumnSpec(name, "") for name in ("name", "description", "type", "body")
        )
    raise ValueError(f"cannot infer structured output columns for {case.case_id}")


def _parse_boolean(raw_output: str) -> bool:
    if "True" not in raw_output and "False" not in raw_output:
        raise ValueError("LOTUS boolean output contains neither True nor False")
    from lotus.sem_ops.postprocessors import filter_postprocess

    class _ModelName:
        @staticmethod
        def get_model_name() -> str:
            return "deepseek-v4-flash"

    parsed = filter_postprocess([raw_output], _ModelName(), default=False)
    if len(parsed.outputs) != 1 or not isinstance(parsed.outputs[0], bool):
        raise ValueError("LOTUS boolean parser did not return one decision")
    return bool(parsed.outputs[0])


def _parse_pairwise_choice(raw_output: str) -> bool:
    import re

    if re.search(r"Document\s*[12]", raw_output, flags=re.IGNORECASE) is None:
        raise ValueError("LOTUS pairwise output contains no Document 1/2 choice")
    from lotus.sem_ops.sem_topk import parse_ans_binary

    selected_first, _explanation = parse_ans_binary(raw_output)
    return bool(selected_first)


def validate_replay_output(case: ReplayCase, raw_output: str) -> Any:
    """Validate one response with the same parser contract as production execution."""

    if case.kind.startswith("sem_flat_map"):
        return parse_structured_flat_map_json(raw_output, _output_columns(case))
    if case.kind.startswith("sem_agg"):
        return parse_structured_sem_agg_output(raw_output, _output_columns(case))
    if case.kind.startswith("sem_map"):
        if case.response_format is None:
            value = raw_output.strip()
            if not value:
                raise ValueError("native sem_map output is empty")
            return value
        return parse_structured_map_json(raw_output, _output_columns(case))
    if case.kind.startswith("sem_groupby"):
        return _parse_boolean(raw_output)
    if case.kind.startswith("pairwise-quick"):
        return _parse_pairwise_choice(raw_output)
    if case.kind.startswith("listwise"):
        user_content = case.messages[-1].get("content")
        if not isinstance(user_content, str):
            raise ValueError("listwise prompt user content must be JSON text")
        request = json.loads(user_content)
        candidates = request.get("candidates")
        expected = request.get("required_count")
        if not isinstance(candidates, list) or not isinstance(expected, int):
            raise ValueError("listwise prompt is missing candidates or required_count")
        valid_ids = {
            str(candidate["id"])
            for candidate in candidates
            if isinstance(candidate, Mapping) and "id" in candidate
        }
        return _parse_selected_ids(
            raw_output,
            valid_ids=valid_ids,
            expected_count=expected,
        )
    answer = raw_output.strip()
    if not answer:
        raise ValueError("LongMemEval answer is empty")
    return answer


def _response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        value = dump()
        if isinstance(value, Mapping):
            return dict(value)
    raise TypeError("provider response must be mapping-like")


def _usage(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("usage")
    return dict(value) if isinstance(value, Mapping) else {}


def _metric_row(case: ReplayCase, payload: Mapping[str, Any], latency_ms: float) -> dict[str, Any]:
    usage = _usage(payload)
    details = usage.get("completion_tokens_details")
    details = dict(details) if isinstance(details, Mapping) else {}
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    completion = usage.get("completion_tokens")
    pricing = PricingSnapshot.deepseek_2026_07_17()
    cost = pricing.estimate_cost_usd(
        cache_hit_input_tokens=int(hit) if isinstance(hit, int) else None,
        cache_miss_input_tokens=int(miss) if isinstance(miss, int) else None,
        output_tokens=int(completion) if isinstance(completion, int) else None,
    )
    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "baseline_latency_ms": case.baseline_latency_ms,
        "latency_ms": round(latency_ms, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "completion_tokens": completion,
        "reasoning_tokens": details.get("reasoning_tokens"),
        "estimated_cost_usd": float(cost) if cost is not None else None,
    }


def replay_cases(
    cases: Sequence[ReplayCase],
    *,
    model: str,
    output_dir: Path,
    completion: Callable[..., Any] | None = None,
    parse_attempts: int = 4,
) -> list[dict[str, Any]]:
    """Replay real requests with thinking disabled and persist raw evidence."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"no-thinking output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts").mkdir()
    (output_dir / "outputs").mkdir()
    if completion is None:
        from litellm import completion as litellm_completion

        completion = litellm_completion
    rows: list[dict[str, Any]] = []
    for case in cases:
        (output_dir / "prompts" / f"{case.case_id}.json").write_text(
            json.dumps(case.messages, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        last_error: ValueError | None = None
        for attempt in range(1, parse_attempts + 1):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [dict(message) for message in case.messages],
                "temperature": 0,
                "max_tokens": case.max_tokens,
                "caching": False,
                "num_retries": 0,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
            if case.response_format is not None:
                kwargs["response_format"] = dict(case.response_format)
            started = perf_counter()
            response = completion(**kwargs)
            latency_ms = (perf_counter() - started) * 1000
            payload = _response_payload(response)
            output_path = output_dir / "outputs" / f"{case.case_id}-attempt-{attempt}.json"
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("provider response contains no choices")
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, Mapping) else None
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str):
                raise ValueError("provider response contains no text content")
            try:
                parsed = validate_replay_output(case, content)
            except ValueError as error:
                last_error = error
                continue
            row = {
                **_metric_row(case, payload, latency_ms),
                "attempt": attempt,
                "parse_success": True,
                "parsed_type": type(parsed).__name__,
                "output_path": str(output_path.relative_to(output_dir)),
            }
            rows.append(row)
            break
        else:
            assert last_error is not None
            raise ValueError(
                f"{case.case_id} failed production parsing after {parse_attempts} attempts: "
                f"{last_error}"
            ) from last_error
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse real trace selection and no-thinking output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-run-dir", type=Path, required=True)
    parser.add_argument("--pairwise-run-dir", type=Path, required=True)
    parser.add_argument("--listwise-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    """Select and replay the approved real request matrix."""

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    cases = select_replay_cases(
        operator_run_dir=args.operator_run_dir,
        pairwise_run_dir=args.pairwise_run_dir,
        listwise_run_dir=args.listwise_run_dir,
    )
    replay_cases(cases, model=args.model, output_dir=args.output_dir)
    return args.output_dir


if __name__ == "__main__":
    print(main())
