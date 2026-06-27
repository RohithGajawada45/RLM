"""
Advanced Router — solves limitations 1 (cross-doc queries) and 2 (routing accuracy).

What's new vs the basic router:
- Returns a LIST of doc_ids, not just one. The router decides if a query
  needs one document or several.
- Returns a CONFIDENCE score (0.0–1.0). Low confidence triggers caller
  fallback behaviour (e.g. include more candidates, ask user to clarify).
- Auto-descriptions are RICHER: includes document type, entities, dates,
  and key terms — not just a vague summary.

The output format is structured:

    DOCS: <id_or_list_or_NONE>
    CONFIDENCE: <0.0-1.0>
    REASON: <one sentence>
"""

import re
from typing import List, Tuple, Optional

from rlm_core import llm_call

# ─── Rich description generation ──────────────────────────────────────────────

RICH_DESCRIBE_PROMPT = """
You are writing a ROUTING DESCRIPTION for the document below. Another LLM
will read this description (alongside a user's query) to decide whether this
document is the right one to consult.

Document filename: {filename}

Sample text from the document (first 4KB, middle 2KB, last 1KB):
---
{sample}
---

Write a structured description with EXACTLY these labeled fields. Be specific.

DOC_TYPE: <one short phrase: e.g. "SEC 10-K annual report", "research paper",
          "technical specification", "user manual", "legal contract">
SUBJECT: <who/what the document is about – entities, products, topics>
TIME_PERIOD: <dates, fiscal year, or version if applicable; "n/a" if none>
KEY_TERMS: <5–10 distinctive terms or phrases that appear prominently>
SUMMARY: <one sentence capturing the document's core purpose>

End your response after the SUMMARY line. No extra commentary.
"""


def _sample_text(
    text: str,
    head: int = 4000,
    mid: int = 2000,
    tail: int = 1000,
) -> str:
    n = len(text)

    if n <= head + mid + tail:
        return text

    head_part = text[:head]

    mid_start = (n - mid) // 2
    mid_part = text[mid_start : mid_start + mid]

    tail_part = text[-tail:]

    return (
        f"{head_part}\n\n[...middle...]\n\n{mid_part}\n\n[...end...]\n\n{tail_part}"
    )


def generate_rich_description(
    client,
    filename: str,
    text: str,
) -> str:
    """One LM call. Returns the structured multi-line description."""

    sample = _sample_text(text)

    prompt = RICH_DESCRIBE_PROMPT.format(
        filename=filename,
        sample=sample,
    )

    return llm_call(
        prompt,
        client,
        label=f"describe[{filename[:25]}]",
    ).strip()
    
    
# ─── Advanced router ──────────────────────────────────────────────────────────

ADVANCED_ROUTER_PROMPT = """
You are a document router. You will be given a catalog of available documents
and a user's question. Your job is to decide WHICH document(s) are needed to
answer the query.

=== AVAILABLE DOCUMENTS ===

{catalog}

=== USER QUERY ===

{query}

=== INSTRUCTIONS ===

Apply these rules in order:

1. If ONE document contains everything needed to answer the query:
   → DOCS: <single_doc_id>

2. If the query genuinely requires combining or comparing across multiple
   documents (look for words like "compare", "vs", "versus", "between",
   "across", "both", or distinct entities each living in distinct documents):
   → DOCS: <id1>, <id2>      (comma-separated, up to 4)

3. If no document is relevant to the query at all:
   → DOCS: NONE

Also assign a CONFIDENCE between 0.0 and 1.0:

0.9-1.0 → very sure (clear match on title/subject/entities)
0.6-0.9 → confident (good topical alignment)
0.3-0.6 → uncertain (only weak topical alignment)
0.0-0.3 → guessing (no real signal)

If your confidence is below 0.4, prefer DOCS: NONE.

Respond in EXACTLY this format. No preamble, no extra lines:

DOCS: <id_or_list_or_NONE>
CONFIDENCE: <0.0-1.0>
REASON: <one sentence>
"""


def route_query_advanced(
    query: str,
    docs_catalog: List[dict],
    client,
    verbose: bool = True,
) -> Tuple[List[str], float, str]:
    """
    Pick the best document(s) for this query.

    Args:
        query: the user's question
        docs_catalog: list of {id, title, description} dicts to choose from
        client: Azure client

    Returns:
        (doc_ids, confidence, reason)

        doc_ids is a (possibly empty) list of valid ids — empty means NONE.
    """

    if not docs_catalog:
        return [], 0.0, "No documents available."

    # Build the catalog text the router will see
    parts = []

    for d in docs_catalog:
        parts.append(
            f"DOC_ID: {d['id']}\n"
            f"TITLE: {d['title']}\n"
            f"DESCRIPTION:\n{d['description']}"
        )

    catalog_str = "\n\n---\n\n".join(parts)

    prompt = ADVANCED_ROUTER_PROMPT.format(
        catalog=catalog_str,
        query=query,
    )

    if verbose:
        print(f"  [router] choosing among {len(docs_catalog)} documents…")

    response = llm_call(
        prompt,
        client,
        label="router",
    )

    # Parse the structured response
    docs_line = re.search(
        r"DOCS:\s*([^\n]+)",
        response,
        re.IGNORECASE,
    )

    conf_line = re.search(
        r"CONFIDENCE:\s*([\d.]+)",
        response,
        re.IGNORECASE,
    )

    reason_line = re.search(
        r"REASON:\s*(.+?)(?:\n|$)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    raw_docs = docs_line.group(1).strip() if docs_line else "NONE"
    confidence = float(conf_line.group(1)) if conf_line else 0.0
    reason = (
        reason_line.group(1).strip()
        if reason_line
        else "(no reason given)"
    )

    # Parse doc list
    if raw_docs.upper() == "NONE":
        return [], confidence, reason

    valid_ids = {d["id"] for d in docs_catalog}

    parsed_ids = [
        s.strip()
        for s in raw_docs.split(",")
        if s.strip()
    ]

    accepted = [
        did
        for did in parsed_ids
        if did in valid_ids
    ]

    if not accepted:
        # Router hallucinated all ids; treat as NONE for safety
        return (
            [],
            confidence,
            f"Router returned unknown ids: {parsed_ids}. Treating as NONE.",
        )

    # If router claimed multiple but the query doesn't look comparative, the
    # confidence will reflect that. We trust the LM's call here.
    return accepted, confidence, reason