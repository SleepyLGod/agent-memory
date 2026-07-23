"""One-shot listwise lowering for semantic top-k retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pandas as pd

from agent_memory.adapters.lotus.context import LotusExecutionContext

LISTWISE_MAX_TOKENS = 1024
LISTWISE_TOPK_CONTRACT = "listwise:v1"
LISTWISE_SYSTEM_PROMPT = (
    "Rank the candidate rows by the supplied criterion. "
    "Return only a JSON object with a selected_ids array. "
    "The array must contain exactly the requested number of unique candidate IDs, "
    "ordered from most relevant to least relevant."
)


@dataclass(frozen=True)
class ListwiseTopKResult:
    """Listwise top-k output plus execution metadata."""

    frame: pd.DataFrame
    selected_ids: tuple[str, ...]
    retry_count: int


def execute_listwise_topk(
    source: pd.DataFrame,
    *,
    instruction: str,
    k: int,
    context: LotusExecutionContext,
) -> ListwiseTopKResult:
    """Rank all candidates in one structured LLM request."""

    if source.empty:
        return ListwiseTopKResult(
            frame=source.copy(),
            selected_ids=(),
            retry_count=0,
        )

    import lotus

    lm = lotus.settings.lm
    if lm is None:
        raise ValueError("listwise sem_topk requires a configured language model")

    columns = tuple(
        dict.fromkeys(
            str(column) for column in lotus.nl_expression.parse_cols(instruction)
        )
    )
    if not columns:
        raise ValueError("listwise sem_topk instruction must reference input columns")
    missing = [column for column in columns if column not in source.columns]
    if missing:
        raise ValueError(f"listwise sem_topk input columns not found: {missing}")

    expected_count = min(k, len(source))
    candidate_rows = json.loads(
        source.loc[:, list(columns)].to_json(
            orient="records",
            force_ascii=False,
            date_format="iso",
        )
    )
    candidates = [
        {"id": f"row_{index}", "row": row}
        for index, row in enumerate(candidate_rows)
    ]
    prompt = [
        {"role": "system", "content": LISTWISE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "criterion": instruction,
                    "required_count": expected_count,
                    "candidates": candidates,
                    "output_schema": {"selected_ids": ["row_id"]},
                },
                ensure_ascii=False,
            ),
        },
    ]
    valid_ids = {candidate["id"] for candidate in candidates}
    last_error: ValueError | None = None

    for attempt in range(context.config.structured_parse_retries + 1):
        output = lm(
            [prompt],
            progress_bar_desc="Listwise ranking",
            max_tokens=LISTWISE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        raw_outputs = list(getattr(output, "outputs", ()))
        raw_output = "" if not raw_outputs else str(raw_outputs[0])
        try:
            selected_ids = _parse_selected_ids(
                raw_output,
                valid_ids=valid_ids,
                expected_count=expected_count,
            )
        except ValueError as error:
            last_error = error
            continue

        positions = [int(row_id.removeprefix("row_")) for row_id in selected_ids]
        return ListwiseTopKResult(
            frame=source.iloc[positions].reset_index(drop=True),
            selected_ids=selected_ids,
            retry_count=attempt,
        )

    assert last_error is not None
    raise ValueError(
        "listwise sem_topk returned invalid structured output after "
        f"{context.config.structured_parse_retries + 1} attempt(s): {last_error}"
    ) from last_error


def _parse_selected_ids(
    raw_output: str,
    *,
    valid_ids: set[str],
    expected_count: int,
) -> tuple[str, ...]:
    """Validate one strict listwise selected-ID response."""

    if not raw_output.strip():
        raise ValueError("listwise sem_topk output is empty")
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError("listwise sem_topk output is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("listwise sem_topk output must be a JSON object")

    selected_ids = payload.get("selected_ids")
    if not isinstance(selected_ids, list) or any(
        not isinstance(row_id, str) for row_id in selected_ids
    ):
        raise ValueError("listwise sem_topk selected_ids must be a list of strings")
    if len(selected_ids) != expected_count:
        raise ValueError(
            f"listwise sem_topk selected_ids must contain exactly {expected_count} IDs"
        )
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("listwise sem_topk selected_ids must be unique")

    unknown_ids = sorted(set(selected_ids).difference(valid_ids))
    if unknown_ids:
        raise ValueError(f"listwise sem_topk selected_ids contain unknown IDs: {unknown_ids}")
    return tuple(selected_ids)
