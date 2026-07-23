"""Complete MemoryAgentBench source-to-task registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ScorerName = Literal[
    "substring_exact_match",
    "exact_match",
    "recall_at_5",
    "longmemeval_v4_flash_judge",
    "infbench_v4_flash_judge",
]

_SYSTEM = "You are a helpful assistant that can read the context and memorize it for future retrieval."

_MEMORIZE = {
    "ruler_qa": "Dialogue between User and Assistant {time_stamp}\\n<User> The following context is the documents I have read: \n{context}\n <Assistant> I have learned the documents and I will answer the question you ask.",
    "longmemeval": "Dialogue between User and Assistant \\n<User> The following context is the conversation between the user and the assistant: \n{context}\n <Assistant> I have memorized the conversation and I will answer the question you ask.",
    "eventqa": "Dialogue between User and Assistant {time_stamp}\\n<User> The following context is the book excerpt: \n{context}\n <Assistant> I have read the book excerpt and I will answer the question you ask.",
    "in_context_learning": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the examples I have learned: \n{context}\n <Assistant> I have learned the examples and I will answer the question you ask.",
    "recsys_redial": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the dialogues between a user and recommender system: \n{context}\n <Assistant> I have memorized the dialogues and I will answer the question you ask.",
    "infbench_sum": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
    "detective_qa": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
    "factconsolidation": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.",
}

_QUERY = {
    "ruler_qa": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\n Now Answer the Question: {question}",
    "longmemeval": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
    "eventqa": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
    "in_context_learning": "Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output \"label: {{label}}\" and nothing else. \n\nQuestion:{question} \n\n label:",
    "recsys_redial": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
    "infbench_sum": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
    "detective_qa": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
    "factconsolidation": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
}


@dataclass(frozen=True)
class MemoryAgentTask:
    """One official source's ingestion, query, and scoring contract."""

    source: str
    split: str
    competence: str
    task_id: str
    scorer: ScorerName
    system_prompt: str = _SYSTEM

    @property
    def contract_id(self) -> str:
        """Return the unique benchmark contract ID for this source."""

        return f"memory-agent-bench:{self.source}"

    @property
    def memorize_template(self) -> str:
        """Return the official RAG-agent ingestion template."""

        return _MEMORIZE[self.task_id]

    @property
    def query_template(self) -> str:
        """Return the official RAG-agent question template."""

        return _QUERY[self.task_id]

    def format_event(self, context: str, timestamp: str) -> str:
        """Wrap one official 4096-token chunk for memory ingestion."""

        return self.memorize_template.format(context=context, time_stamp=timestamp)

    def format_answer_prompt(self, question: str, retrieval_context: str) -> str:
        """Build the shared answer prompt with explicit retrieved memory."""

        query = self.query_template.format(question=question)
        return f"Retrieved Memory:\n{retrieval_context}\n\n{query}"


def _task(
    source: str,
    split: str,
    competence: str,
    task_id: str,
    scorer: ScorerName,
) -> MemoryAgentTask:
    return MemoryAgentTask(
        source=source,
        split=split,
        competence=competence,
        task_id=task_id,
        scorer=scorer,
    )


_TASK_LIST = [
    *(
        _task(source, "Accurate_Retrieval", "accurate_retrieval", "eventqa", "substring_exact_match")
        for source in ("eventqa_131072", "eventqa_65536", "eventqa_full")
    ),
    _task("longmemeval_s*", "Accurate_Retrieval", "accurate_retrieval", "longmemeval", "longmemeval_v4_flash_judge"),
    *(
        _task(source, "Accurate_Retrieval", "accurate_retrieval", "ruler_qa", "substring_exact_match")
        for source in ("ruler_qa1_197K", "ruler_qa2_421K")
    ),
    *(
        _task(source, "Test_Time_Learning", "test_time_learning", "in_context_learning", "exact_match")
        for source in (
            "icl_banking77_5900shot_balance",
            "icl_clinic150_7050shot_balance",
            "icl_nlu_8296shot_balance",
            "icl_trec_coarse_6600shot_balance",
            "icl_trec_fine_6400shot_balance",
        )
    ),
    _task("recsys_redial_full", "Test_Time_Learning", "test_time_learning", "recsys_redial", "recall_at_5"),
    _task("detective_qa", "Long_Range_Understanding", "long_range_understanding", "detective_qa", "exact_match"),
    _task("infbench_sum_eng_shots2", "Long_Range_Understanding", "long_range_understanding", "infbench_sum", "infbench_v4_flash_judge"),
    *(
        _task(source, "Conflict_Resolution", "conflict_resolution", "factconsolidation", "substring_exact_match")
        for source in (
            "factconsolidation_mh_262k",
            "factconsolidation_mh_32k",
            "factconsolidation_mh_64k",
            "factconsolidation_mh_6k",
            "factconsolidation_sh_262k",
            "factconsolidation_sh_32k",
            "factconsolidation_sh_64k",
            "factconsolidation_sh_6k",
        )
    ),
]

MEMORY_AGENT_TASKS = {task.source: task for task in _TASK_LIST}
if len(MEMORY_AGENT_TASKS) != len(_TASK_LIST):
    raise RuntimeError("MemoryAgentBench source registry contains duplicates")

__all__ = ["MEMORY_AGENT_TASKS", "MemoryAgentTask", "ScorerName"]
