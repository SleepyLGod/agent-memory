"""LongMemEval answer and official-compatible judge prompt contracts."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from agent_memory.evaluation.types import BenchmarkQuestion

LONGMEMEVAL_ANSWER_PROMPT = (
    "I will give you several history chats between you and a user. Please answer "
    "the question based on the relevant chat history.\n\n\nHistory Chats:\n\n"
    "{context}\n\nCurrent Date: {question_date}\nQuestion: {question}\nAnswer:"
)

_STANDARD_JUDGE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: "
    "{answer}\n\nModel Response: {response}\n\nIs the model response correct? "
    "Answer yes or no only."
)
_TEMPORAL_JUDGE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for the "
    "number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct. \n\nQuestion: "
    "{question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_UPDATE_JUDGE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {question}\n\nCorrect "
    "Answer: {answer}\n\nModel Response: {response}\n\nIs the model response "
    "correct? Answer yes or no only."
)
_PREFERENCE_JUDGE = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: "
    "{question}\n\nRubric: {answer}\n\nModel Response: {response}\n\nIs the "
    "model response correct? Answer yes or no only."
)
_ABSTENTION_JUDGE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is "
    "not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: "
    "{response}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)

LONGMEMEVAL_ANSWER_PROMPT_DIGEST = sha256(
    LONGMEMEVAL_ANSWER_PROMPT.encode("utf-8")
).hexdigest()
LONGMEMEVAL_JUDGE_PROMPT_DIGEST = sha256(
    json.dumps(
        [
            _STANDARD_JUDGE,
            _TEMPORAL_JUDGE,
            _UPDATE_JUDGE,
            _PREFERENCE_JUDGE,
            _ABSTENTION_JUDGE,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _question_date(question: BenchmarkQuestion) -> str:
    value = question.metadata.get("question_date")
    if not isinstance(value, str) or not value:
        raise ValueError("LongMemEval questions require question_date metadata")
    return value


def build_answer_prompt(question: BenchmarkQuestion, context: str) -> str:
    """Build the shared four-system LongMemEval answer prompt."""

    return LONGMEMEVAL_ANSWER_PROMPT.format(
        context=context,
        question_date=_question_date(question),
        question=question.question,
    )


def build_judge_prompt(question: BenchmarkQuestion, response: str) -> str:
    """Reproduce the official LongMemEval task-specific judge prompt."""

    question_type = question.metadata.get("question_type")
    values = {
        "question": question.question,
        "answer": question.gold_answer,
        "response": response,
    }
    if question.category == "abstention" or question.question_id.endswith("_abs"):
        return _ABSTENTION_JUDGE.format(**values)
    if question_type in {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    }:
        return _STANDARD_JUDGE.format(**values)
    if question_type == "temporal-reasoning":
        return _TEMPORAL_JUDGE.format(**values)
    if question_type == "knowledge-update":
        return _UPDATE_JUDGE.format(**values)
    if question_type == "single-session-preference":
        return _PREFERENCE_JUDGE.format(**values)
    raise ValueError(f"unsupported LongMemEval question type {question_type!r}")


def parse_judge_response(response: str) -> bool:
    """Apply the official evaluator's case-insensitive `yes` decision rule."""

    return "yes" in response.strip().lower()


def hypothesis_record(question: BenchmarkQuestion, answer: str) -> Mapping[str, Any]:
    """Return the official evaluator-compatible JSONL record."""

    return {"question_id": question.question_id, "hypothesis": answer}


__all__ = [
    "LONGMEMEVAL_ANSWER_PROMPT",
    "LONGMEMEVAL_ANSWER_PROMPT_DIGEST",
    "LONGMEMEVAL_JUDGE_PROMPT_DIGEST",
    "build_answer_prompt",
    "build_judge_prompt",
    "hypothesis_record",
    "parse_judge_response",
]
