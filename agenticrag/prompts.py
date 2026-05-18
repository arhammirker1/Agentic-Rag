"""
prompts.py — Every LLM prompt used by PageIndex.

All prompts are plain strings with {format} placeholders.
Keeping them here makes them easy to inspect, customise, or translate.
"""

# ─── System messages ──────────────────────────────────────────────────────

SYS_BUILDER = (
    "You are an expert document analyst. "
    "Your task is to read document page text and produce a JSON tree index — "
    "a 'Table of Contents' optimised for LLM navigation. "
    "Always respond with valid JSON only. No prose, no markdown fences, no explanation."
)

SYS_RETRIEVER = (
    "You are a document retrieval reasoning agent. "
    "You have a JSON tree index of a document and must decide which nodes to "
    "read in order to answer a question. "
    "Always respond with valid JSON only. No prose, no markdown fences."
)

# ─── Tree Building ────────────────────────────────────────────────────────

DETECT_TOC = """\
Below are the first {n} pages of a document.

{pages}

---
Question: Does this document contain a formal Table of Contents or Index section?

Respond with JSON exactly like this:
{{"has_toc": true, "toc_page": 2}}
or
{{"has_toc": false, "toc_page": null}}

"toc_page" is the 0-based page index where the TOC appears (null if none).
"""

BUILD_FROM_TOC = """\
This document has an existing Table of Contents on page {toc_page} (0-based).

TOC text:
{toc_text}

Total pages in document: {total_pages}

Build a hierarchical JSON tree index from this TOC.
Each node must have exactly these fields:
  "title"       : string — section title
  "node_id"     : string — unique zero-padded 4-digit ID, e.g. "0001"
  "start_index" : integer — 0-based page index where section begins
  "end_index"   : integer — 0-based page index where section ends (exclusive)
  "summary"     : string — one sentence describing this section
  "nodes"       : array  — child nodes (same structure, can be [])

Return a JSON array of top-level nodes only.
"""

BUILD_NO_TOC = """\
Below are pages {start} to {end} of a document (0-based page numbers in brackets).

{pages}

---
Create a hierarchical JSON tree index for these pages.

Rules:
1. Each node = one logical section (chapter, heading, sub-heading, topic block).
2. Every node needs exactly these fields:
     "title"       : section title (string)
     "node_id"     : unique zero-padded 4-digit string starting from {next_id}
     "start_index" : 0-based page index where this section begins (integer)
     "end_index"   : 0-based page index where this section ends (exclusive, integer)
     "summary"     : one sentence summarising the section (string)
     "nodes"       : child nodes array (same structure, [] if none)
3. No node should span more than {max_pages} pages.
4. Maximum 3 levels of nesting.
5. If a page range has no clear structure, create one node for the whole range.

Return a JSON array of top-level nodes.
"""

MERGE_TREES = """\
You are combining partial JSON tree indices from consecutive page ranges
of the same document into one unified document tree.

Partial trees:
{partial}

Rules:
1. Return a single JSON array of top-level nodes.
2. Preserve all start_index / end_index values exactly.
3. Remove any duplicate or overlapping nodes.
4. Renumber node_ids sequentially starting from "0001".
5. Keep the overall section order matching the page order.

Return a JSON array only.
"""

NODE_SUMMARY = """\
Section title: "{title}"
Pages: {start} to {end}

Content:
{text}

---
Write exactly one sentence (under 30 words) summarising what this section covers.
Plain text only — no JSON, no markdown.
"""

DOC_DESCRIPTION = """\
Here is the top-level tree index of a document:

{tree}

Write a short paragraph (3-5 sentences) describing:
- What this document is about
- Who it is intended for
- The main topics or sections it covers

Plain text only.
"""

# ─── Retrieval ────────────────────────────────────────────────────────────

SELECT_NODES = """\
Document tree index:
{tree}

{history_block}

User question: "{question}"

{visited_block}

Select the 1-5 node_ids most likely to contain the answer.
Prefer leaf nodes over parent nodes.

Respond with JSON:
{{
  "reasoning": "<one sentence explaining your choice>",
  "node_ids": ["0003", "0007"]
}}
"""

CHECK_SUFFICIENT = """\
Question: "{question}"

Information collected so far:
{gathered}

---
Is this information sufficient to give a complete, accurate answer?

Respond with JSON:
{{"sufficient": true}}
or
{{"sufficient": false, "missing": "<what is still needed>"}}
"""

EXPAND_KEYWORDS = """\
You are a search keyword expansion expert.  Given a user's question, generate a
comprehensive list of keyphrases, keywords, and synonyms likely to appear
verbatim (or near-verbatim) inside a document that answers this question.

Question: "{question}"

{history_block}

Return JSON:
{{
  "keyphrases": ["<multi-word phrase 1>", "<multi-word phrase 2>"],
  "keywords":   ["<single keyword 1>", "<single keyword 2>"],
  "synonyms":   ["<synonym or domain term 1>", "<synonym 2>"]
}}

Rules:
- "keyphrases": 3-8 multi-word phrases (2+ words) an author would write when
  covering this topic.  Prefer specific technical or domain phrases.
- "keywords": 5-15 important single words (skip stop-words like "the", "is", "a").
- "synonyms": 3-8 alternative terms, abbreviations, or domain-specific vocabulary
  for the core concepts in the question.
- Think about the vocabulary an expert in this field would use.
- Include both common abbreviations and their expanded forms where relevant.
"""

FINAL_ANSWER = """\
Answer the following question based ONLY on the document sections provided below.

Question: "{question}"

Retrieved sections:
{context}

---
Give a DETAILED, thorough, and well-explained answer.
- Cite section titles and page ranges where helpful.
- If the information is not in the retrieved sections, say so clearly.
- Do not make up information.
- DO NOT merely list section names or topic labels — EXPLAIN what each point
  actually says, including specific details, numbers, and examples from the text.
- Structure your answer clearly with paragraphs or bullet points.
"""