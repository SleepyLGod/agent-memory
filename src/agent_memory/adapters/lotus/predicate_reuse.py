"""Reuse fixed text predicates independently of non-semantic row metadata."""

from collections.abc import Callable
import json

import pandas as pd


def reuse_predicate_decisions(
    source: pd.DataFrame, instruction: str, decisions: dict[str, bool],
    evaluate: Callable[[pd.DataFrame, list[int]], pd.DataFrame],
) -> tuple[pd.DataFrame, set[int]]:
    """Screening precedes this function; results always carry current row values."""
    from lotus.nl_expression import parse_cols
    from lotus.templates.task_instructions import df2multimodal_info

    columns = tuple(parse_cols(instruction))
    if not columns or any(c not in source for c in columns):
        return evaluate(source, list(range(len(source)))), set()
    # Do not guess dependencies for multimodal or nested input values.
    if any(not isinstance(v, str) and v is not None and v is not pd.NA
           and not isinstance(v, (int, float, bool))
           for row in source.loc[:, list(columns)].itertuples(index=False, name=None) for v in row):
        return evaluate(source, list(range(len(source)))), set()
    docs = df2multimodal_info(source, list(columns))
    if any(doc.get("image") for doc in docs):
        return evaluate(source, list(range(len(source)))), set()
    keys = [json.dumps([instruction, columns, doc.get("text", "")], ensure_ascii=False) for doc in docs]
    missing: dict[str, int] = {}
    reused = {i for i, key in enumerate(keys) if key in decisions}
    for i, key in enumerate(keys):
        if key not in decisions:
            missing.setdefault(key, i)
    positions = list(missing.values())
    if positions:
        pending = source.iloc[positions].copy()
        pending.index = pd.Index(positions)
        result = evaluate(pending, positions)
        if not result.index.is_unique or not set(result.index) <= set(positions):
            raise ValueError("predicate output must preserve unique input positions")
        selected = set(result.index)
        decisions.update({key: position in selected for key, position in missing.items()})
    return source.iloc[[i for i, key in enumerate(keys) if decisions[key]]].copy(), reused
