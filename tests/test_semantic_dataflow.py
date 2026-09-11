"""Tests for the public semantic dataflow facade."""

from __future__ import annotations

import pandas as pd
import pytest

import agent_memory as am
from agent_memory.adapters import LotusAdapter


def _build_flow() -> am.SemanticDataflow:
    source = am.Source(
        {
            "category": "Row category.",
            "value": "Numeric value.",
        }
    )
    selected = source.filter(source.col("category") == "keep").select(
        ["category", "value"]
    )
    counts = selected.count(output_col="row_count")
    return am.SemanticDataflow(
        source=source,
        views={"selected": selected, "counts": counts},
        adapter=LotusAdapter(),
    )


def test_semantic_dataflow_applies_relation_rows_to_shared_views() -> None:
    flow = _build_flow()

    flow.apply(
        pd.DataFrame(
            [
                {"category": "keep", "value": 1},
                {"category": "drop", "value": 2},
                {"category": "keep", "value": 3},
            ]
        )
    )

    pd.testing.assert_frame_equal(
        flow.view("selected"),
        pd.DataFrame(
            [
                {"category": "keep", "value": 1},
                {"category": "keep", "value": 3},
            ]
        ),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        flow.view("counts"),
        pd.DataFrame({"row_count": [2]}),
        check_dtype=False,
    )


def test_semantic_dataflow_requires_exact_source_schema() -> None:
    flow = _build_flow()

    with pytest.raises(ValueError, match="source delta columns must exactly match"):
        flow.apply(pd.DataFrame([{"value": 1, "category": "keep"}]))


def test_semantic_dataflow_rejects_unknown_view() -> None:
    flow = _build_flow()

    with pytest.raises(KeyError, match="unknown"):
        flow.view("unknown")


def test_semantic_dataflow_view_is_a_defensive_copy() -> None:
    flow = _build_flow()
    flow.apply(pd.DataFrame([{"category": "keep", "value": 1}]))

    first = flow.view("selected")
    first.loc[0, "value"] = 999

    assert flow.view("selected").loc[0, "value"] == 1


def test_semantic_dataflow_view_copies_nested_object_values() -> None:
    flow = _build_flow()
    flow.apply(
        pd.DataFrame(
            [{"category": "keep", "value": {"evidence": ["original"]}}]
        )
    )

    first = flow.view("selected")
    first.loc[0, "value"]["evidence"].append("changed")

    assert flow.view("selected").loc[0, "value"] == {
        "evidence": ["original"]
    }


def test_semantic_dataflow_validates_declared_views() -> None:
    source = am.Source({"value": "Numeric value."})

    with pytest.raises(TypeError, match="views values must be Relation"):
        am.SemanticDataflow(
            source=source,
            views={"result": object()},  # type: ignore[dict-item]
            adapter=LotusAdapter(),
        )


@pytest.mark.parametrize("same_schema", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_semantic_dataflow_rejects_foreign_source(
    same_schema: bool, mixed: bool,
) -> None:
    source = am.Source({"value": "Numeric value."})
    foreign = am.Source(
        {"value": "Numeric value."} if same_schema else {"other": "Other value."}
    )
    view = foreign.select(["value"] if same_schema else ["other"])
    if mixed:
        view = source.union_by_name(view)

    with pytest.raises(ValueError, match="view 'result'.*declared source"):
        am.SemanticDataflow(
            source=source, views={"result": view}, adapter=LotusAdapter(),
        )


def test_semantic_dataflow_accepts_self_join_and_window_source() -> None:
    source = am.Source({"value": "Numeric value."})
    joined = source.alias("left").join(source.alias("right"), on="value")
    windowed = source.count_window(size=2).process_window(
        lambda window: window.select(["value"])
    )
    flow = am.SemanticDataflow(
        source=source, views={"joined": joined, "windowed": windowed},
        adapter=LotusAdapter(),
    )
    flow.apply(pd.DataFrame({"value": [1, 2]}))
    assert len(flow.view("joined")) == 2
    assert flow.view("windowed")["value"].tolist() == [1, 2]


def test_semantic_dataflow_owns_nested_input_after_apply() -> None:
    flow = _build_flow()
    payload = {"evidence": ["original"]}
    batch = pd.DataFrame([{"category": "keep", "value": payload}])
    flow.apply(batch)

    payload["evidence"].append("changed")
    batch.loc[0, "value"]["extra"] = ["caller"]
    assert flow.view("selected").loc[0, "value"] == {"evidence": ["original"]}

    # A later update must still consume the detached, original source row.
    flow.apply(pd.DataFrame([{"category": "keep", "value": 2}]))
    assert flow.view("selected").loc[0, "value"] == {"evidence": ["original"]}
    assert flow.view("counts").loc[0, "row_count"] == 2
