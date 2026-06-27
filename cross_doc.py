"""
Cross-Document Pipeline — solves limitation #1 (cross-doc queries).

When the advanced router picks more than one document for a single query
(e.g. "Compare Adobe's AI strategy to Tesla's"), we:

1. Reframe the original query as a per-document sub-query.
2. Run a full RLM on each picked document with that sub-query.
3. Aggregate the per-doc answers with one additional LM call.

This is the classic MapReduce shape: fan out for facts, fold to synthesize.
"""

from typing import List, Dict, Any
import re
from concurrent.futures import ThreadPoolExecutor
from rlm_core import (
    rlm, llm_call,
    get_token_usage, _enter_token_usage, _exit_token_usage,
)

# Maximum number of per-document RLMs to run concurrently. Bounded to avoid
# hitting Azure OpenAI's per-deployment tokens-per-minute (TPM) limit. The
# OpenAI Python SDK is thread-safe; the bound is purely about rate limiting.
MAX_FANOUT_WORKERS = 4

# ─── Per-doc sub-answer cache — stub implementation ────────────────────────
# Future feature: cache sub-RLM results keyed on (sub_query, doc_id) so the
# same doc-level question doesn't re-run when it appears in different
# multi-doc compositions. Not yet implemented — these are no-op stubs so
# app.py's integration hooks (delete/clear/stats endpoints) work cleanly.
# When the real implementation lands, replace these with the actual cache.

_SUB_CACHE: Dict[str, Any] = {}


def _sub_cache_size() -> int:
    """Number of entries in the per-doc sub-answer cache (currently always 0)."""
    return len(_SUB_CACHE)


def _sub_cache_clear() -> int:
    """Clear the sub-answer cache and return the number of entries evicted."""
    n = len(_SUB_CACHE)
    _SUB_CACHE.clear()
    return n


def _sub_cache_invalidate_for_doc(doc_id: str) -> int:
    """Evict any sub-cache entries that reference this document. Returns count."""
    keys_to_remove = [k for k in _SUB_CACHE if doc_id in k]
    for k in keys_to_remove:
        del _SUB_CACHE[k]
    return len(keys_to_remove)


def _slugify(title: str) -> str:
    """Derive a short label from a document title for the stage tag.
    Examples: 'adbe2024annualreport.pdf' -> 'adbe',
              'tcs_report.pdf' -> 'tcs',
              'infosys-ar-25.pdf' -> 'infosys'."""
    base = title.rsplit(".", 1)[0]
    m = re.match(r"[A-Za-z]+", base)
    return (m.group() if m else base[:8]).lower()[:12]


SUBQUERY_TEMPLATE = """\
The user's original question (which may span multiple documents) is:

"{query}"

This document is one of several being consulted. Your job: extract from THIS
document only the facts needed to address that question, then return them.

================================================================

CRITICAL — ANTI-HALLUCINATION RULES. Read carefully, follow strictly.

1. VERIFY every number, name, date, and quote before including it in FINAL.
   You MUST have seen it printed by the REPL for THIS specific document.
   Do NOT rely on memory or training-data knowledge of the company.

   ⚠️ The companies in these documents are often well-known. You may
   "remember" their financials, headcount, leadership, etc. from training
   data. Those memories are STALE and probably wrong for this filing.
   ONLY use what the REPL shows you.

2. SHOW EVIDENCE. For each fact you report, include the surrounding sentence
   exactly as the REPL printed it. The aggregator will quote these directly;
   fabricated quotes produce fabricated answers downstream.

3. IF YOU DON'T FIND IT, SAY SO — but only after you've genuinely looked.
   Before concluding "this document does not contain X", you MUST have run
   at least 3 distinct searches with different angles:
      (a) a broad term,
      (b) an exact phrase you'd expect to appear, and
      (c) a section / heading name where the topic would live.
   In your FINAL, list every search you ran (substring, regex, or sub-LM
   query) so the aggregator can audit your effort. A FINAL claiming absence
   with fewer than 3 distinct searches will be treated as UNRELIABLE.
   Inventing a plausible-looking value is FAR worse than admitting absence,
   but declaring absence prematurely is also a failure mode.

4. SEARCH SPECIFICALLY before giving up. Try exact phrases — "headcount
   worldwide", "we employed", "we had", "as of [year]", "fiscal year ended"
   — rather than single broad words that return hundreds of noisy matches.
   If the first 5 matches look like noise, refine the search; do not call
   FINAL on a hunch.

5. AT LEAST ONE EXACT QUOTE in your FINAL. If you can't quote even one
   sentence verbatim from THIS document that supports your answer, you
   haven't searched enough — keep searching.

6. PREFER AUDITED FIGURES OVER NARRATIVE ROLLUPS. For numerical questions
   (revenue, income, headcount, expenses, etc.), the canonical answer is
   the line from the company's audited Consolidated Statements — labelled
   "Revenue from operations", "Total revenues", "Total revenue", "Net
   income", and similar — typically appearing in tables under headings
   like "Consolidated Statement of Profit and Loss", "Consolidated
   Statements of Operations", or "Consolidated Income Statement".

   Rounded narrative figures from the chairman's letter, management
   commentary, marketing rollups, or press-release-style overviews (e.g.
   "we generated $30 billion in revenues") are LESS PRECISE and should
   only be quoted when no audited figure is available in this document.

   If you locate BOTH a financial-statement line AND a rounded narrative
   figure, the audited line wins. Search ANCHOR terms like "Consolidated
   Statement of Profit and Loss", "Consolidated Statements of Operations",
   "Revenue from Operations", "in ₹ crore", "in millions" — and extract
   the revenue line FROM THE STATEMENTS section, not from the overview.
   In your FINAL, also note where the figure was located (e.g. "from
   Consolidated Statement of Profit and Loss" vs "from chairman's letter").

================================================================

When you have verified evidence, call FINAL(...) with:
• The fact(s) you found, with at least one verbatim quote and its date.
• OR: a clear statement that THIS document does not contain the
  requested information."""
  
AGGREGATION_PROMPT = """
You are synthesizing answers from multiple documents.

ORIGINAL USER QUERY:
{query}

PER-DOCUMENT FINDINGS:
{findings}

Write a single coherent answer to the original query that integrates these
findings. If the query asks for a comparison, present the comparison clearly
(a small table or side-by-side bullets often works well). Cite which document
each fact came from by referring to the document title. If two documents
disagree on something, point that out explicitly.

Be precise. Use the documents' own numbers and language where possible.
Do not invent facts that are not in the findings above.

SKEPTICISM RULE — handling absence claims:
A per-doc finding of the form "this document does NOT contain information
about X" is only reliable if the sub-LM listed the searches it ran. If a
sub-LM declares absence on a topic that is plainly central to the
document's subject (e.g. AI risk in a tech company's 10-K, headcount in
an annual report) WITHOUT listing at least 3 distinct searches, mark that
finding as "absence not confirmed — searches insufficient" in your
synthesis rather than reporting it as established fact. Never write
conclusions like "Company X does not discuss Y" if the underlying sub-LM
finding is unverified by this standard.

PRESERVE SOURCE UNITS — do not rewrite numerical units:
When reporting figures, use the units AND the magnitude exactly as the
source document presents them. If a 10-K reports "Total revenue ... 21,505"
under a header "(In millions)", write the figure as "$21,505 million" —
NOT "$21.5 billion". If an annual report reports "₹1,62,990 crore", write
that — NOT "₹1.63 trillion" or "$19 billion equivalent". The source units
are the canonical form; converted/rounded restatements lose precision and
diverge from what the company itself published.

If the user explicitly asked for a same-currency comparison, you may add
an OPTIONAL follow-up note showing one chosen conversion (and state the
exchange rate used), but the primary figures in your answer must remain in
their source units. When sub-LM findings reference figures from chairman's
letters or narrative rollups AND from audited financial statements, prefer
the audited-statement figures.
"""

def run_cross_doc(
    query: str,
    doc_ids: List[str],
    doc_lookup: Dict[str, Dict[str, Any]],
    client,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fan out: run RLM on each doc with a reframed sub-query, then aggregate.

    Args:
        query: the original user query
        doc_ids: the IDs the router chose
        doc_lookup: dict mapping doc_id → {title, text, ...}
        client: Azure client

    Returns:
        {
            'final_answer': str,
            'per_doc': [{'doc_id', 'title', 'sub_answer'}],
        }
    """

    if not doc_ids:
        return {"final_answer": "(no documents to consult)", "per_doc": []}

    # Step 1 — per-document sub-RLM calls (PARALLEL)

    # Run the per-doc RLMs concurrently in a thread pool. Each worker thread
    # starts with an empty ContextVar context (Python's default), so we
    # explicitly install the parent request's TokenUsage tracker via
    # _enter_token_usage so all workers accumulate into the SAME tracker
    # instance. TokenUsage.add() is lock-protected, so concurrent writes are
    # safe. The OpenAI SDK is thread-safe per its docs.
    #
    # Each per-doc sub-RLM passes a unique stage_override (e.g. "fanout:adbe")
    # so the footer shows a per-document token breakdown instead of one
    # opaque "root_lm" bucket.
    #
    # Results from ex.map() preserve input order, matching the original
    # sequential behaviour.
    sub_query = SUBQUERY_TEMPLATE.format(query=query)
    parent_tracker = get_token_usage()

    def _process_doc(did: str):
        token = _enter_token_usage(parent_tracker)
        try:
            doc = doc_lookup.get(did)
            if not doc:
                return None

            if verbose:
                print(f"  [fan-out] running RLM on '{doc['title']}'...")

            stage_label = f"fanout:{_slugify(doc['title'])}"

            sub_answer = rlm(
                context=doc["text"],
                query=sub_query,
                client=client,
                depth=0,
                verbose=verbose,
                stage_override=stage_label,
            )

            return {
                "doc_id": did,
                "title": doc["title"],
                "sub_answer": sub_answer,
            }

        finally:
            _exit_token_usage(token)

    workers = max(1, min(MAX_FANOUT_WORKERS, len(doc_ids)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        per_doc = [r for r in ex.map(_process_doc, doc_ids) if r is not None]

    # Step 2 — aggregate findings into one final answer
    findings_text = "\n\n".join(
        f"=== From '{p['title']}' ({p['doc_id']}) ===\n{p['sub_answer']}"
        for p in per_doc
    )

    if verbose:
        print(f"  [fan-out] aggregating {len(per_doc)} sub-answers...")

    aggregator_prompt = AGGREGATION_PROMPT.format(
        query=query,
        findings=findings_text,
    )

    final_answer = llm_call(
        aggregator_prompt,
        client,
        label="aggregator",
    )

    return {
        "final_answer": final_answer,
        "per_doc": per_doc,
    }