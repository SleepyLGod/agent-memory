from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.evaluation.no_thinking_replay import (
    ReplayCase,
    replay_cases,
    validate_replay_output,
)


def _case(kind: str, content: str, *, response_format: bool = True) -> ReplayCase:
    return ReplayCase(
        case_id=kind,
        kind=kind,
        prompt_path=Path("prompt.json"),
        messages=(
            {"role": "system", "content": "system"},
            {"role": "user", "content": content},
        ),
        max_tokens=1024,
        response_format={"type": "json_object"} if response_format else None,
        baseline_latency_ms=1000,
    )


@pytest.mark.parametrize(
    ("case", "raw", "expected_type"),
    [
        (
            _case(
                "sem_flat_map-ordinary",
                'Field descriptions: {"name":"n","description":"d","type":"t","body":"b"}. Use JSON null for missing values.',
            ),
            '{"rows":[{"name":"sleep","description":"routine","type":"user","body":"early"}]}',
            list,
        ),
        (
            _case("sem_agg-heavy", 'Expected JSON shape: {"name":"string","body":"string"}'),
            '{"name":"sleep","body":"early"}',
            dict,
        ),
        (
            _case("sem_map-heavy", 'Expected JSON shape: {"hook":"string"}'),
            '{"hook":"Useful for sleep questions"}',
            dict,
        ),
        (
            _case("sem_map-ordinary", "Return one hook", response_format=False),
            "Useful for sleep questions",
            str,
        ),
        (_case("sem_groupby-heavy", "boolean"), "False", bool),
        (
            _case("pairwise-quick-heavy", "ranking", response_format=False),
            "Document 2",
            bool,
        ),
        (
            _case(
                "listwise-heavy",
                json.dumps(
                    {
                        "required_count": 2,
                        "candidates": [
                            {"id": "row_0", "row": {}},
                            {"id": "row_1", "row": {}},
                        ],
                    }
                ),
            ),
            '{"selected_ids":["row_1","row_0"]}',
            tuple,
        ),
        (_case("answer", "question", response_format=False), "non-empty", str),
    ],
)
def test_replay_uses_production_output_contracts(case, raw, expected_type) -> None:
    assert isinstance(validate_replay_output(case, raw), expected_type)


def test_replay_disables_thinking_and_records_parse_usage(tmp_path: Path) -> None:
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return {
            "model": kwargs["model"],
            "choices": [{"message": {"content": "a real answer"}}],
            "usage": {
                "prompt_tokens": 10,
                "prompt_cache_hit_tokens": 8,
                "prompt_cache_miss_tokens": 2,
                "completion_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }

    rows = replay_cases(
        (_case("answer", "question", response_format=False),),
        model="deepseek/deepseek-v4-flash",
        output_dir=tmp_path,
        completion=completion,
    )

    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert calls[0]["caching"] is False
    assert rows[0]["parse_success"] is True
    assert rows[0]["reasoning_tokens"] == 0
    assert rows[0]["cache_hit_tokens"] == 8
    assert (tmp_path / "results.json").is_file()
