"""
Cross-Document Pipeline — solves limitation #1 (cross-doc queries).

When the advanced router picks more than one document for a single query
(e.g. "Compare Adobe's AI strategy to Tesla's"), we:

1. Reframe the original query as a per-document sub-query.
2. Run a full RLM on each picked document with that sub-query.
3. Aggregate the per-doc answers with one additional LM call.

This is the classic MapReduce shape: fan out for facts, fold to synthesize.

Sub-Answer Cache (Layer 3):
   Caches per-doc RLM results keyed on (normalised_intent, doc_id).

   The normalisation step is the key innovation:
     "Give the total revenue of Adobe and Infosys" — for the Infosys doc —
     normalises identically to "What is the total revenue of Infosys?"
     because we strip other-company names and stopwords first, leaving
     "infosys revenue" in both cases → guaranteed exact-key cache hit.

   This means:
     Q1: "revenue of Infosys"        → RLM on infosys doc → cached
     Q2: "revenue of Adobe"          → RLM on adobe doc   → cached
     Q3: "revenue of Adobe and Infosys" → BOTH sub-answers from cache;
         only the aggregator call is new (one cheap LLM call).

Cache persistence:
   Sub-answers are persisted to cache/sub_answer_cache.pkl via
   CachePersister (cache_persist.py), which provides:
     • Background autosave every 60 s  → survives SIGKILL / kill -9
     • atexit hook                      → saves on Ctrl+C / SIGTERM
     • Load on startup                  → cache survives server restarts
"""

import hashlib
import math
import re
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from pathlib import Path

from rlm_core import (
    rlm, llm_call,
    get_token_usage, _enter_token_usage, _exit_token_usage,
    embed_text, cfg,
)
from cache_persist import CachePersister

MAX_FANOUT_WORKERS = 4

# ─── Registry: doc_id → company/entity names found in that doc ─────────────
_DOC_COMPANY_REGISTRY: Dict[str, List[str]] = {}
_DCR_LOCK = threading.Lock()


KNOWN_ALIASES = {
    "tata consultancy services": ["tcs"],
    "international business machines": ["ibm"],
    "hewlett packard": ["hp"],
    "hewlett packard enterprise": ["hpe"],
    "pricewaterhousecoopers": ["pwc"],
    "procter gamble": ["pg", "p&g"],
}


def register_doc_companies(doc_id: str, doc_title: str, doc_text_preview: str = "") -> None:
    terms = set()

    GENERIC_DOC_WORDS = {
        "annual", "report", "form", "fiscal", "year", "integrated",
        "limited", "incorporated", "corp", "company", "group",
        "holdings", "international", "document", "file", "copy",
        "financial", "statement", "statements",
    }

    SKIP_TITLE = {
        "securities","exchange","commission","united","states",
        "washington","november","december","january","february",
        "march","april","may","june","july","august",
        "september","october","annual","report","fiscal",
        "pursuant","section","form","exact","name",
        "registrant","specified","charter","delaware",
        "california","total","revenue","income",
        "operations","building","integrated",
    }

    SKIP_CAPS = {
        "FORM","PURSUANT","SECTION","SECURITIES",
        "EXCHANGE","COMMISSION","UNITED","STATES",
        "ANNUAL","REPORT","ITEM","PART",
        "TOTAL","REVENUE","INCOME",
    }

    filename_words = re.findall(r"[A-Za-z]+", doc_title)
    for w in filename_words:
        lw = w.lower()
        if len(lw) >= 3 and lw not in GENERIC_DOC_WORDS:
            terms.add(lw)
    if len(filename_words) >= 2:
        acronym = "".join(w[0] for w in filename_words).lower()
        if len(acronym) >= 2:
            terms.add(acronym)

    company_pattern = re.compile(r"\b(?:[A-Z][a-z]{2,}\s+){1,5}[A-Z][a-z]{2,}\b")
    preview = doc_text_preview[:3000]
    for m in company_pattern.finditer(preview):
        phrase = m.group().strip()
        words = phrase.split()
        full = " ".join(w.lower() for w in words)
        terms.add(full)
        for w in words:
            lw = w.lower()
            if lw not in SKIP_TITLE:
                terms.add(lw)
        acronym = "".join(w[0] for w in words).lower()
        if len(acronym) >= 2:
            terms.add(acronym)
        if full in KNOWN_ALIASES:
            terms.update(KNOWN_ALIASES[full])

    for m in re.finditer(r"\b([A-Z]{3,})\b", preview):
        word = m.group(1)
        if word not in SKIP_CAPS:
            terms.add(word.lower())

    with _DCR_LOCK:
        _DOC_COMPANY_REGISTRY[doc_id] = sorted(terms)

    print(f"[doc-registry] {doc_title}")
    print(sorted(terms))


def _normalize_query_for_doc(query: str, doc_id: str) -> str:
    """
    Produce a doc-scoped intent string from the user query.
    Strips other-document company names and stopwords, then sorts
    remaining tokens for order-independence.
    """
    text = (
        query.lower()
        .replace("&", " and ")
        .replace("/", " ")
        .replace("-", " ")
    )

    with _DCR_LOCK:
        own_terms = set(_DOC_COMPANY_REGISTRY.get(doc_id, []))
        other_terms: set = set()
        for did, terms in _DOC_COMPANY_REGISTRY.items():
            if did != doc_id:
                other_terms.update(terms)

    terms_to_strip = other_terms - own_terms
    for term in sorted(terms_to_strip, key=lambda x: (-len(x), x)):
        text = re.sub(rf'\b{re.escape(term)}\b', ' ', text)

    stopwords = {
        'what', 'is', 'are', 'was', 'were', 'the', 'of', 'give', 'me',
        'tell', 'how', 'much', 'did', 'and', 'or', 'for', 'a', 'an', 'in',
        'its', 'their', 'compare', 'vs', 'versus', 'between', 'annual',
        'generate', 'generated', 'total', 'report', 'document', 'please',
        'can', 'you', 'do', 'has', 'have', 'from', 'to', 'at', 'by',
        'per', 'with', 'about', 'on', 'this', 'that', 'these', 'those',
    }
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [t for t in tokens if t not in stopwords and len(t) >= 2]
    return ' '.join(sorted(set(tokens)))


# ─── Per-doc sub-answer cache ────────────────────────────────────────────────

SUB_CACHE_MAX_ENTRIES = 500
_SUB_CACHE: OrderedDict = OrderedDict()
_SUB_CACHE_LOCK = threading.Lock()
SUB_CACHE_SEMANTIC_THRESHOLD = 0.90

# ── Persistence ──────────────────────────────────────────────────────────────

def _set_sub_cache(new_data: OrderedDict) -> None:
    """Replace the live sub-cache contents (used by CachePersister on load)."""
    with _SUB_CACHE_LOCK:
        _SUB_CACHE.clear()
        _SUB_CACHE.update(new_data)


_sub_persister = CachePersister(
    path="cache/sub_answer_cache.pkl",
    get_cache=lambda: _SUB_CACHE,
    set_cache=_set_sub_cache,
    label="sub-cache",
)

# Load persisted cache immediately at import time, then start autosave.
# Order matters: load first so the autosave thread doesn't overwrite a
# populated cache with an empty one on the very first tick.
_sub_persister.load()
_sub_persister.start()


# ── Public save/clear helpers (called from app.py) ────────────────────────

def save_sub_cache() -> None:
    """Explicit save — called from the FastAPI shutdown event as a belt-and-
    suspenders save on top of the atexit hook."""
    _sub_persister.save()


def _sub_cache_size() -> int:
    with _SUB_CACHE_LOCK:
        return len(_SUB_CACHE)


def _sub_cache_clear() -> int:
    with _SUB_CACHE_LOCK:
        n = len(_SUB_CACHE)
        _SUB_CACHE.clear()
    # Persist the now-empty cache immediately so a restart doesn't reload
    # stale data from the old file.
    _sub_persister.save()
    return n


def _sub_cache_invalidate_for_doc(doc_id: str) -> int:
    with _SUB_CACHE_LOCK:
        keys_to_remove = [k for k, v in _SUB_CACHE.items()
                          if v.get("doc_id") == doc_id]
        for k in keys_to_remove:
            del _SUB_CACHE[k]
        n = len(keys_to_remove)
    if n:
        _sub_persister.save()
    return n


# ── Cache internals ───────────────────────────────────────────────────────

def _sub_cache_key(normalized_intent: str, doc_id: str) -> str:
    canonical = f"{normalized_intent}|{doc_id}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _sub_cache_get(query: str, doc_id: str, client=None) -> Optional[dict]:
    """
    Two-level lookup:
      A) Exact: normalize query → hash → direct dict lookup (free).
      B) Embedding fallback for near-synonyms.
    """
    normalized = _normalize_query_for_doc(query, doc_id)
    key = _sub_cache_key(normalized, doc_id)

    with _SUB_CACHE_LOCK:
        if key in _SUB_CACHE:
            _SUB_CACHE.move_to_end(key)
            entry = _SUB_CACHE[key]
            print(f"  [sub-cache] EXACT HIT intent='{normalized}' doc={doc_id[:12]}")
            return entry

    if client is None:
        return None

    with _SUB_CACHE_LOCK:
        candidates = [v for v in _SUB_CACHE.values() if v["doc_id"] == doc_id
                      and v.get("embedding") is not None]

    if not candidates:
        return None

    q_emb = embed_text(client, normalized, cfg(client).embedding_deployment)
    if q_emb is None:
        return None

    best, best_sim = None, 0.0
    for entry in candidates:
        sim = _cosine_sim(q_emb, entry["embedding"])
        if sim > best_sim:
            best_sim, best = sim, entry

    if best_sim >= SUB_CACHE_SEMANTIC_THRESHOLD:
        orig = best.get("original_query", "")[:60]
        print(f"  [sub-cache] SEMANTIC HIT sim={best_sim:.3f} intent='{normalized}' "
              f"matched='{orig}'")
        return best

    print(f"  [sub-cache] MISS intent='{normalized}' best_sim={best_sim:.3f}")
    return None


def _sub_cache_put(query: str, doc_id: str, doc_title: str,
                   sub_answer: str, client=None) -> None:
    normalized = _normalize_query_for_doc(query, doc_id)
    key = _sub_cache_key(normalized, doc_id)

    emb = None
    if client is not None:
        emb = embed_text(client, normalized, cfg(client).embedding_deployment)

    with _SUB_CACHE_LOCK:
        _SUB_CACHE[key] = {
            "sub_answer": sub_answer,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "original_query": query,
            "normalized_intent": normalized,
            "embedding": emb,
            "cached_at": time.time(),
        }
        _SUB_CACHE.move_to_end(key)
        while len(_SUB_CACHE) > SUB_CACHE_MAX_ENTRIES:
            _SUB_CACHE.popitem(last=False)

    print(f"  [sub-cache] stored intent='{normalized}' for doc='{doc_title}'")


def _cosine_sim(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if (na == 0 or nb == 0) else dot / (na * nb)


# ─── Sub-query template ─────────────────────────────────────────────────────

def _slugify(title: str) -> str:
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


# ─── Main entry point ────────────────────────────────────────────────────────

def run_cross_doc(
    query: str,
    doc_ids: List[str],
    doc_lookup: Dict[str, Dict[str, Any]],
    client,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fan out: run RLM on each doc, aggregate, with Layer-3 sub-cache.
    """
    if not doc_ids:
        return {"final_answer": "(no documents to consult)", "per_doc": []}

    sub_query = SUBQUERY_TEMPLATE.format(query=query)
    parent_tracker = get_token_usage()

    # ── Step 1: sub-cache lookup for each doc ────────────────────────────

    per_doc_results: List[Optional[dict]] = [None] * len(doc_ids)
    miss_indices: List[int] = []

    for i, did in enumerate(doc_ids):
        cached = _sub_cache_get(query, did, client=client)
        if cached is not None:
            if verbose:
                print(f"  [sub-cache] HIT doc='{cached['doc_title']}' "
                      f"intent='{cached.get('normalized_intent', '')}'")
            per_doc_results[i] = {
                "doc_id": did,
                "title": cached["doc_title"],
                "sub_answer": cached["sub_answer"],
                "from_cache": True,
            }
        else:
            miss_indices.append(i)

    # ── Step 2: RLM only for cache misses ────────────────────────────────

    def _process_doc(idx: int):
        did = doc_ids[idx]
        token = _enter_token_usage(parent_tracker)
        try:
            doc = doc_lookup.get(did)
            if not doc:
                return idx, None

            if verbose:
                print(f"  [fan-out] running RLM on '{doc['title']}'...")

            sub_answer = rlm(
                context=doc["text"],
                query=sub_query,
                client=client,
                depth=0,
                verbose=verbose,
                stage_override=f"fanout:{_slugify(doc['title'])}",
            )

            _sub_cache_put(
                query=query,
                doc_id=did,
                doc_title=doc["title"],
                sub_answer=sub_answer,
                client=client,
            )

            return idx, {
                "doc_id": did,
                "title": doc["title"],
                "sub_answer": sub_answer,
                "from_cache": False,
            }
        finally:
            _exit_token_usage(token)

    if miss_indices:
        workers = max(1, min(MAX_FANOUT_WORKERS, len(miss_indices)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for idx, result in ex.map(_process_doc, miss_indices):
                per_doc_results[idx] = result

    per_doc = [r for r in per_doc_results if r is not None]

    # ── Step 3: aggregate ─────────────────────────────────────────────────

    cache_hits = sum(1 for r in per_doc if r.get("from_cache"))
    if verbose and cache_hits:
        print(f"  [sub-cache] {cache_hits}/{len(per_doc)} sub-answers from cache "
              f"— skipped {cache_hits} RLM call(s)")

    if verbose:
        print(f"  [fan-out] aggregating {len(per_doc)} sub-answers...")

    findings_text = "\n\n".join(
        f"=== From '{p['title']}' ({p['doc_id']}) ===\n{p['sub_answer']}"
        for p in per_doc
    )

    final_answer = llm_call(
        AGGREGATION_PROMPT.format(query=query, findings=findings_text),
        client,
        label="aggregator",
    )

    return {"final_answer": final_answer, "per_doc": per_doc}