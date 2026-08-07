"""Native-compatible LOCOMO context, answer, and Zep judge generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from time import perf_counter
from typing import Any

from agent_memory.evaluation.types import BenchmarkQuestion
from agent_memory.policy.retrieval import RetrievalResult
from agent_memory.tracing.semantic import semantic_trace_scope

GENERATION_MAX_TOKENS = 8192
GENERATION_TEMPERATURE = 0

_LANGUAGE_INSTRUCTION = (
    "\n\nAny extracted information should be returned in the same language as it was written in. "
    "Only output non-English text when the user has written full sentences or phrases in that non-English language. "
    "Otherwise, output English."
)

_CONTEXT_TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
of when the fact was stated.
Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.

<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
""".strip()

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful expert assistant answering questions from LOCOMO users "
    "based on the provided context."
)

ANSWER_USER_PROMPT_TEMPLATE = """
# CONTEXT:
You have access to facts and entities from a conversation.

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to timestamps when determining the answer.
3. Look for direct evidence for the event or fact in the question.
4. If memories contradict one another, prioritize the most recent valid memory.
5. Convert relative time references to specific dates, months, or years.
6. Be specific about people, places, and events.
7. Timestamps represent when an event occurred, not when it was mentioned.
8. If the context does not contain the answer, answer exactly "No information available".

Clarification:
When interpreting memories, use the timestamp to determine when the described event happened, not when someone talked about the event.

Example:
Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
Question: What day did I go to the vet?
Correct Answer: March 15, 2023

# APPROACH:
1. First, examine all memories that contain information related to the question.
2. Examine the timestamps and content of these memories carefully.
3. Look for explicit mentions of dates, times, locations, or events that answer the question.
4. If the answer requires calculation, such as converting relative time references, show your work.
5. Formulate a precise, concise answer based solely on the evidence in the memories.
6. Double-check that your answer directly addresses the question asked.
7. Ensure your final answer is specific and avoids vague time references.

Context:
{context}

Question: {question}
""".strip()

ZEP_JUDGE_SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions match a "
    "gold standard answer."
)

ZEP_JUDGE_USER_PROMPT_TEMPLATE = """
Your task is to label an answer to a question as CORRECT or WRONG. You are given:
1. A question posed by one user about another user.
2. A concise gold answer.
3. A generated answer.

The point of the question is to ask about something one user should know about
the other user based on their prior conversations. The gold answer is usually
short. A generated answer may be longer; grade generously when it clearly
contains the same essential topic as the gold answer.

For time questions, accept the same date or time period when formatting differs
or the generated answer uses an equivalent relative expression. Mark an answer
WRONG when it contradicts or does not contain the gold answer's essential
information.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return one short sentence of reasoning and exactly one label, CORRECT or WRONG.
""".strip()

ANSWER_RESPONSE_SCHEMA = {
    "properties": {
        "answer": {
            "description": "Concise answer based only on retrieved context.",
            "title": "Answer",
            "type": "string",
        }
    },
    "required": ["answer"],
    "title": "_AnswerResponse",
    "type": "object",
}

ZEP_JUDGE_RESPONSE_SCHEMA = {
    "properties": {
        "label": {
            "description": "CORRECT or WRONG.",
            "enum": ["CORRECT", "WRONG"],
            "title": "Label",
            "type": "string",
        },
        "reasoning": {
            "description": "One-sentence grading explanation.",
            "title": "Reasoning",
            "type": "string",
        },
    },
    "required": ["label", "reasoning"],
    "title": "_JudgeResponse",
    "type": "object",
}


def _prompt_digest(
    system_prompt: str,
    user_prompt_template: str,
    response_schema: Mapping[str, Any],
) -> str:
    serialized_schema = json.dumps(response_schema)
    payload = json.dumps(
        {
            "system_prompt": system_prompt,
            "user_prompt_template": (
                f"{user_prompt_template}\n\nRespond with a JSON object in the following "
                f"format:\n\n{serialized_schema}"
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


ANSWER_PROMPT_DIGEST = _prompt_digest(
    ANSWER_SYSTEM_PROMPT + _LANGUAGE_INSTRUCTION,
    ANSWER_USER_PROMPT_TEMPLATE,
    ANSWER_RESPONSE_SCHEMA,
)
ZEP_JUDGE_PROMPT_DIGEST = _prompt_digest(
    ZEP_JUDGE_SYSTEM_PROMPT + _LANGUAGE_INSTRUCTION,
    ZEP_JUDGE_USER_PROMPT_TEMPLATE,
    ZEP_JUDGE_RESPONSE_SCHEMA,
)


@dataclass(frozen=True)
class AnswerRecord:
    """One generated answer tied to one retrieval result."""

    question_id: str
    question: str
    category: int
    answer: str
    gold_answer: str | None
    evidence_event_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    latency_ms: float

    @classmethod
    def from_question(
        cls,
        question: BenchmarkQuestion,
        retrieval: RetrievalResult,
        *,
        answer: str,
        latency_ms: float,
    ) -> "AnswerRecord":
        """Build an answer artifact from the shared benchmark question."""

        gold_answer = str(question.gold_answer) if question.gold_answer else None
        return cls(
            question_id=question.question_id,
            question=question.question,
            category=int(question.category),
            answer=answer,
            gold_answer=gold_answer,
            evidence_event_ids=question.evidence_event_ids,
            entity_ids=_record_ids(retrieval, "entities"),
            fact_ids=_record_ids(retrieval, "facts"),
            latency_ms=round(float(latency_ms), 3),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe artifact payload."""

        payload = asdict(self)
        payload["evidence_event_ids"] = list(self.evidence_event_ids)
        payload["entity_ids"] = list(self.entity_ids)
        payload["fact_ids"] = list(self.fact_ids)
        return payload


def format_retrieval_context(result: RetrievalResult) -> str:
    """Format retrieval channels using the published Zep LOCOMO shape."""

    facts = "\n".join(
        f"  - {row.get('fact')} (event_time: {row.get('valid_at')})"
        for row in _records(result, "facts")
    )
    entities = "\n".join(
        f"  - {row.get('name')}: {row.get('summary')}"
        for row in _records(result, "entities")
    )
    return _CONTEXT_TEMPLATE.format(facts=facts, entities=entities)


def generate_answer(
    question: BenchmarkQuestion,
    retrieval: RetrievalResult,
) -> AnswerRecord:
    """Generate one answer from one already-computed retrieval result."""

    messages = build_locomo_answer_messages(
        question,
        format_retrieval_context(retrieval),
    )
    started = perf_counter()
    response = _generate_json(
        messages,
        response_schema=ANSWER_RESPONSE_SCHEMA,
        prompt_name="locomo.answer",
        phase="answering",
        question_id=question.question_id,
        validate=_validate_answer_response,
    )
    answer = str(response["answer"])
    return AnswerRecord.from_question(
        question,
        retrieval,
        answer=answer.strip(),
        latency_ms=(perf_counter() - started) * 1000,
    )


def build_locomo_answer_messages(
    question: BenchmarkQuestion,
    context: str,
) -> list[dict[str, str]]:
    """Build the shared LOCOMO answer prompt for any retrieval backend."""

    return _structured_messages(
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt=ANSWER_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question.question,
        ),
        response_schema=ANSWER_RESPONSE_SCHEMA,
    )


def generate_zep_judge(question: BenchmarkQuestion, answer: str) -> Any:
    """Grade one category 1-4 answer with the corrected published Zep rubric."""

    from agent_memory.evaluation.zep.scoring import ZepJudgeGrade

    if int(question.category) not in {1, 2, 3, 4}:
        raise ValueError("Zep judge only evaluates LOCOMO categories 1-4")
    messages = build_locomo_zep_judge_messages(question, answer)
    started = perf_counter()
    response = _generate_json(
        messages,
        response_schema=ZEP_JUDGE_RESPONSE_SCHEMA,
        prompt_name="locomo.zep_judge",
        phase="grading",
        question_id=question.question_id,
        validate=_validate_judge_response,
    )
    label = str(response.get("label", "")).strip().upper()
    reasoning = response.get("reasoning")
    assert isinstance(reasoning, str)
    return ZepJudgeGrade(
        question_id=question.question_id,
        category=int(question.category),
        label=label,
        is_correct=label == "CORRECT",
        reasoning=reasoning.strip(),
        latency_ms=round((perf_counter() - started) * 1000, 3),
    )


def build_locomo_zep_judge_messages(
    question: BenchmarkQuestion,
    answer: str,
) -> list[dict[str, str]]:
    """Build the corrected published Zep judge prompt."""

    return _structured_messages(
        system_prompt=ZEP_JUDGE_SYSTEM_PROMPT,
        user_prompt=ZEP_JUDGE_USER_PROMPT_TEMPLATE.format(
            question=question.question,
            gold_answer=question.gold_answer,
            generated_answer=answer,
        ),
        response_schema=ZEP_JUDGE_RESPONSE_SCHEMA,
    )


def _structured_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    response_schema: Mapping[str, Any],
) -> list[dict[str, str]]:
    schema = json.dumps(response_schema)
    return [
        {"role": "system", "content": system_prompt + _LANGUAGE_INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\nRespond with a JSON object in the following format:\n\n"
                f"{schema}"
            ),
        },
    ]


def _generate_json(
    messages: list[dict[str, str]],
    *,
    response_schema: Mapping[str, Any],
    prompt_name: str,
    phase: str,
    question_id: str,
    validate: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    import lotus

    lm: Any = lotus.settings.lm
    if lm is None:
        raise RuntimeError("LOTUS LM is not configured before benchmark generation")
    last_error: ValueError | None = None
    for attempt in range(1, 5):
        with semantic_trace_scope(
            semantic_operator=prompt_name,
            phase=phase,
            question_id=question_id,
            prompt_name=prompt_name,
            attempt=attempt,
        ):
            output = lm(
                [messages],
                show_progress_bar=False,
                progress_bar_desc=phase.capitalize(),
                max_tokens=GENERATION_MAX_TOKENS,
                temperature=GENERATION_TEMPERATURE,
                response_format={"type": "json_object"},
            )
        outputs = list(getattr(output, "outputs", ()))
        raw = str(outputs[0]).strip() if outputs else ""
        try:
            parsed = json.loads(_strip_code_fence(raw))
            if not isinstance(parsed, dict):
                raise ValueError("structured response must be a JSON object")
            _validate_required_fields(parsed, response_schema)
            if validate is not None:
                validate(parsed)
            return parsed
        except (json.JSONDecodeError, ValueError) as error:
            last_error = ValueError(f"Invalid {prompt_name} response: {error}")
    assert last_error is not None
    raise last_error


def _validate_required_fields(
    value: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> None:
    required = response_schema.get("required", ())
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"structured response is missing fields: {missing}")


def _validate_answer_response(value: Mapping[str, Any]) -> None:
    answer = value.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("LOCOMO answer response requires a non-empty answer string")


def _validate_judge_response(value: Mapping[str, Any]) -> None:
    label = str(value.get("label", "")).strip().upper()
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError(f"Zep judge returned invalid label: {label!r}")
    reasoning = value.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Zep judge requires a non-empty reasoning string")


def _strip_code_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*[ \t]*\r?\n?", "", stripped)
        stripped = re.sub(r"\r?\n?```[ \t]*$", "", stripped)
    return stripped.strip()


def _records(result: RetrievalResult, channel: str) -> list[dict[str, Any]]:
    frame = result.channels.get(channel)
    if frame is None or frame.empty:
        return []
    return frame.to_dict(orient="records")


def _record_ids(result: RetrievalResult, channel: str) -> tuple[str, ...]:
    frame = result.channels.get(channel)
    if frame is None or frame.empty or "record_id" not in frame.columns:
        return ()
    return tuple(str(value) for value in frame["record_id"])


__all__ = [
    "ANSWER_PROMPT_DIGEST",
    "ANSWER_RESPONSE_SCHEMA",
    "ANSWER_SYSTEM_PROMPT",
    "ANSWER_USER_PROMPT_TEMPLATE",
    "GENERATION_MAX_TOKENS",
    "GENERATION_TEMPERATURE",
    "ZEP_JUDGE_PROMPT_DIGEST",
    "ZEP_JUDGE_RESPONSE_SCHEMA",
    "ZEP_JUDGE_SYSTEM_PROMPT",
    "ZEP_JUDGE_USER_PROMPT_TEMPLATE",
    "AnswerRecord",
    "build_locomo_answer_messages",
    "build_locomo_zep_judge_messages",
    "format_retrieval_context",
    "generate_answer",
    "generate_zep_judge",
]
