ANALYSE_CONTENT_PROMPT = r"""Generate a structured analysis of the following content by:
    1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
    2. Extracting core themes and contextual elements
    3. Creating relevant categorical tags

    Format the response as a JSON object:
    {{
        "keywords": [
            // several specific, distinct keywords that capture key concepts and terminology
            // Order from most to least important
            // Don't include keywords that are the name of the speaker or time
            // At least three keywords, but don't be too redundant.
        ],
        "context": 
            // one sentence summarizing:
            // - Main topic/domain
            // - Key arguments/points
            // - Intended audience/purpose
        ,
        "tags": [
            // several broad categories/themes for classification
            // Include domain, format, and type tags
            // At least three tags, but don't be too redundant.
        ]
    }}

    Content for analysis:
    {content}

    Return only a single JSON object. No markdown fences and no extra prose.
    """


EVOLUTION_SYSTEM_PROMPT = r"""
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Each entry below pairs one new memory with one earlier neighbor memory. The neighbor's
id is given by its [_row_id:earlier] field, and the new memory by [_row_id:later].

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with the other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
    2.1 If strengthen, which neighbor should it be connected to? Give its neighbor_id.
        Also give the updated tags for the new memory.
    2.2 If update_neighbor, give the updated context and tags for each neighbor.
        If a neighbor's context or tags should not change, repeat the original values.
        Emit one object per neighbor, each carrying its own neighbor_id; order does not matter.
Tags should reflect the content of these memories so they can be retrieved and categorized later.

Return only a single JSON object. No markdown fences and no extra prose.
{{
    "should_evolve": true,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["<neighbor_id>", "<neighbor_id>"],
    "tags_to_update": ["tag_1", "tag_n"],
    "neighbor_updates": [
        {{"neighbor_id": "<neighbor_id>", "new_context": "new context", "new_tags": ["tag_1", "tag_n"]}}
    ]
}}
"""


NOTE_CONSOLIDATION_INSTRUCTION = """
Merge the grouped context and tags entries for one note into a single accumulated
{context} and {tags}. Use only information explicitly present in the grouped
entries; never infer beyond the evidence.

Preserve every distinct fact, detail, and tag across all entries, including
changes over time. When entries conflict or supersede one another, prefer the
newer entry: each input row carries _add_seq, and a larger _add_seq means the
entry was produced later. An entry that simply omits a fact does not remove it;
only drop a fact when a newer entry explicitly contradicts or supersedes it.
""".strip()


EMBEDDING_TEXT_PROMPT = r"""Produce the exact text to be embedded for memory retrieval.

Concatenate the following fields, IN ORDER, with no changes whatsoever:
1. content
2. keywords (join the list items with a single space; keep each keyword verbatim)
3. context
4. tags (join the list items with a single space; keep each tag verbatim)

Rules:
- Output ONLY the concatenated text. No labels, no headings, no markdown fences, no extra prose.
- The original `content` must appear word-for-word at the start, unchanged.
- Do not translate, rewrite, reorder, summarize, or add any commentary to the source material.

content: {content}
keywords: {keywords}
context: {context}
tags: {tags}
"""
