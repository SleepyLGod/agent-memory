"""Executable LongMemEval v1 task contract for the common runner."""

from __future__ import annotations

from agent_memory.evaluation.harness import (
    GradeResult,
    JudgeStep,
    ModelPrompt,
    TaskContract,
)
from agent_memory.evaluation.types import BenchmarkQuestion

from .scoring import (
    LONGMEMEVAL_ANSWER_PROMPT_DIGEST,
    LONGMEMEVAL_JUDGE_PROMPT_DIGEST,
    build_answer_prompt,
    build_judge_prompt,
    parse_judge_response,
)


def _answer_prompt(question: BenchmarkQuestion, context: str) -> ModelPrompt:
    return ModelPrompt(
        prompt_name="longmemeval.answer",
        messages=(
            {
                "role": "user",
                "content": build_answer_prompt(question, context),
            },
        ),
        prompt_digest=LONGMEMEVAL_ANSWER_PROMPT_DIGEST,
        temperature=0,
        max_tokens=8192,
        thinking_enabled=False,
    )


def _parse_answer(response: str) -> str:
    answer = response.strip()
    if not answer:
        raise ValueError("LongMemEval answer must be non-empty")
    return answer


def _retrieval_query(question: BenchmarkQuestion) -> str:
    question_date = question.metadata.get("question_date")
    if not isinstance(question_date, str) or not question_date:
        raise ValueError("LongMemEval retrieval requires question_date")
    return f"Current Date: {question_date}\nQuestion: {question.question}"


def _judge_plan(question: BenchmarkQuestion, answer: str) -> tuple[JudgeStep, ...]:
    return (
        JudgeStep(
            prompt=ModelPrompt(
                prompt_name="longmemeval.judge",
                messages=(
                    {
                        "role": "user",
                        "content": build_judge_prompt(question, answer),
                    },
                ),
                prompt_digest=LONGMEMEVAL_JUDGE_PROMPT_DIGEST,
                temperature=0,
                max_tokens=10,
                thinking_enabled=False,
            ),
            parse=parse_judge_response,
        ),
    )


def longmemeval_task_contract(
    *,
    judge_model_id: str = "deepseek/deepseek-v4-flash",
) -> TaskContract:
    """Return the official-compatible evaluator with an explicit judge model."""

    if not isinstance(judge_model_id, str) or not judge_model_id:
        raise ValueError("judge_model_id must be a non-empty string")
    scorer_id = f"longmemeval_judge:{judge_model_id}"

    def judge_reducer(
        question: BenchmarkQuestion,
        answer: str,
        results: tuple[object, ...],
    ) -> GradeResult:
        del question, answer
        if len(results) != 1 or not isinstance(results[0], bool):
            raise ValueError("LongMemEval judge requires one boolean result")
        correct = results[0]
        official_metric_model = judge_model_id in {
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "openai/gpt-4o-2024-08-06",
        }
        return GradeResult(
            scorer_id=scorer_id,
            score=float(correct),
            label="yes" if correct else "no",
            details={
                "judge_model_id": judge_model_id,
                "official_evaluator_contract": True,
                "official_metric_model": official_metric_model,
            },
        )

    return TaskContract(
        task_id="longmemeval-v1",
        answer_prompt=_answer_prompt,
        answer_parser=_parse_answer,
        retrieval_query=_retrieval_query,
        answer_prompt_digest=LONGMEMEVAL_ANSWER_PROMPT_DIGEST,
        scorer_id=scorer_id,
        scorer_digest=LONGMEMEVAL_JUDGE_PROMPT_DIGEST,
        judge_plan=_judge_plan,
        judge_reducer=judge_reducer,
    )


__all__ = ["longmemeval_task_contract"]
