"""Shared LOCOMO answer, official scorer, and Zep judge contracts."""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Any, Mapping

from agent_memory.evaluation.harness import (
    GradeContract,
    GradeResult,
    JudgeStep,
    ModelPrompt,
    TaskContract,
)
from agent_memory.evaluation.metrics import locomo_answer_score
from agent_memory.evaluation.types import BenchmarkQuestion
from agent_memory.evaluation.zep.answering import (
    ANSWER_PROMPT_DIGEST,
    GENERATION_MAX_TOKENS,
    ZEP_JUDGE_PROMPT_DIGEST,
    build_locomo_answer_messages,
    build_locomo_zep_judge_messages,
)

LOCOMO_OFFICIAL_SCORER_ID = "locomo_official:v1"
LOCOMO_OFFICIAL_SCORER_DIGEST = sha256(
    b"locomo_answer_score:category-1-5:v1"
).hexdigest()
LOCOMO_ANSWER_PARSER_ID = "locomo-structured-or-raw-text:v1"


def _parse_json_object(response: str) -> Mapping[str, Any]:
    value = response.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*[ \t]*\r?\n?", "", value)
        value = re.sub(r"\r?\n?```[ \t]*$", "", value)
    parsed = json.loads(value.strip())
    if not isinstance(parsed, Mapping):
        raise ValueError("LOCOMO structured response must be a JSON object")
    return parsed


def _parse_answer(response: str) -> str:
    raw = response.strip()
    if not raw:
        raise ValueError("LOCOMO answer must be non-empty")
    try:
        answer = _parse_json_object(raw).get("answer")
    except (json.JSONDecodeError, ValueError):
        return raw
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return raw


def _parse_zep_judge(response: str) -> Mapping[str, str]:
    parsed = _parse_json_object(response)
    label = str(parsed.get("label", "")).strip().upper()
    reasoning = parsed.get("reasoning")
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError(f"Zep judge returned invalid label: {label!r}")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Zep judge requires non-empty reasoning")
    return {"label": label, "reasoning": reasoning.strip()}


def _answer_prompt(question: BenchmarkQuestion, context: str) -> ModelPrompt:
    return ModelPrompt(
        prompt_name="locomo.answer",
        messages=tuple(build_locomo_answer_messages(question, context)),
        prompt_digest=ANSWER_PROMPT_DIGEST,
        temperature=0,
        max_tokens=GENERATION_MAX_TOKENS,
        thinking_enabled=False,
    )


def _retrieval_query(question: BenchmarkQuestion) -> str:
    return question.question


def _official_score(question: BenchmarkQuestion, answer: str) -> GradeResult:
    score = locomo_answer_score(answer, question.gold_answer, question.category)
    return GradeResult(
        scorer_id=LOCOMO_OFFICIAL_SCORER_ID,
        score=score,
        label="correct" if score == 1 else "incorrect",
        details={"category": int(question.category)},
    )


def _zep_judge_plan(
    question: BenchmarkQuestion,
    answer: str,
) -> tuple[JudgeStep, ...]:
    return (
        JudgeStep(
            prompt=ModelPrompt(
                prompt_name="locomo.zep_judge",
                messages=tuple(build_locomo_zep_judge_messages(question, answer)),
                prompt_digest=ZEP_JUDGE_PROMPT_DIGEST,
                temperature=0,
                max_tokens=GENERATION_MAX_TOKENS,
                thinking_enabled=False,
            ),
            parse=_parse_zep_judge,
        ),
    )


def locomo_task_contract(
    *,
    judge_model_id: str = "deepseek/deepseek-v4-flash",
) -> TaskContract:
    """Return one answer contract with official and Zep grading paths."""

    if not judge_model_id:
        raise ValueError("judge_model_id must be non-empty")
    judge_scorer_id = f"locomo_zep_judge:{judge_model_id}"

    def reduce_zep_judge(
        question: BenchmarkQuestion,
        answer: str,
        results: tuple[object, ...],
    ) -> GradeResult:
        del answer
        if len(results) != 1 or not isinstance(results[0], Mapping):
            raise ValueError("Zep judge requires one parsed response")
        result = results[0]
        label = str(result.get("label", "")).upper()
        reasoning = result.get("reasoning")
        if label not in {"CORRECT", "WRONG"} or not isinstance(reasoning, str):
            raise ValueError("Zep judge reducer received invalid output")
        return GradeResult(
            scorer_id=judge_scorer_id,
            score=float(label == "CORRECT"),
            label=label,
            details={
                "category": int(question.category),
                "reasoning": reasoning,
                "judge_model_id": judge_model_id,
            },
        )

    return TaskContract(
        task_id="locomo",
        answer_prompt=_answer_prompt,
        answer_parser=_parse_answer,
        retrieval_query=_retrieval_query,
        answer_prompt_digest=ANSWER_PROMPT_DIGEST,
        scorer_id=LOCOMO_OFFICIAL_SCORER_ID,
        scorer_digest=LOCOMO_OFFICIAL_SCORER_DIGEST,
        checkpoint_boundary="session",
        memory_system_error_score=None,
        deterministic_scorer=_official_score,
        additional_graders=(
            GradeContract(
                scorer_id=judge_scorer_id,
                scorer_digest=ZEP_JUDGE_PROMPT_DIGEST,
                judge_plan=_zep_judge_plan,
                judge_reducer=reduce_zep_judge,
                applies_to=lambda question: int(question.category) in {1, 2, 3, 4},
            ),
        ),
        answer_parser_id=LOCOMO_ANSWER_PARSER_ID,
    )


__all__ = [
    "LOCOMO_OFFICIAL_SCORER_DIGEST",
    "LOCOMO_OFFICIAL_SCORER_ID",
    "LOCOMO_ANSWER_PARSER_ID",
    "locomo_task_contract",
]
