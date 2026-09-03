"""Tests for predicate-site physical profile configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memory.evaluation.semantic_pair_config import (
    load_semantic_pair_profile_config,
)


def test_load_semantic_pair_profile_config_is_strict_and_canonical(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bindings": [
                    {
                        "site_id": "sem_groupby:entity",
                        "mode": "search-filter",
                        "top_k": 15,
                        "min_similarity": 0.6,
                    },
                    {
                        "site_id": "sem_groupby:fact",
                        "mode": "search-filter",
                        "top_k": 10,
                        "min_similarity": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_semantic_pair_profile_config(path)

    assert [binding.site_id for binding in config.bindings] == [
        "sem_groupby:entity",
        "sem_groupby:fact",
    ]
    assert config.to_dict()["source_sha256"] == config.source_sha256
    assert config.schema_version == 1


def test_load_semantic_pair_profile_config_rejects_unknown_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "bindings": [
                    {
                        "site_id": "sem_groupby:entity",
                        "mode": "search-filter",
                        "top_k": 15,
                        "min_similarity": 0.6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be 1"):
        load_semantic_pair_profile_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("top_k", True, "positive integer"),
        ("min_similarity", True, "finite"),
        ("min_similarity", float("nan"), "finite"),
    ),
)
def test_load_semantic_pair_profile_config_rejects_invalid_bounds(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    binding = {
        "site_id": "sem_groupby:entity",
        "mode": "search-filter",
        "top_k": 15,
        "min_similarity": 0.6,
    }
    binding[field] = value
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps({"schema_version": 1, "bindings": [binding]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_semantic_pair_profile_config(path)


def test_load_semantic_pair_profile_config_rejects_duplicate_sites(
    tmp_path: Path,
) -> None:
    binding = {
        "site_id": "sem_groupby:entity",
        "mode": "search-filter",
        "top_k": 15,
        "min_similarity": 0.6,
    }
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "bindings": [binding, binding]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate semantic pair site"):
        load_semantic_pair_profile_config(path)
