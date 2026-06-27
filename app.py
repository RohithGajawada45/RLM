"""
FastAPI backend for the multi-document RLM.

Endpoints:
    GET /                 — serves the frontend HTML
    GET /docs             — list all uploaded documents
    POST /upload          — upload a PDF/DOCX, get description back
    DELETE /docs/{id}     — remove a document
    POST /query           — ask a question, get routing decision + answer

Run with:
    uvicorn app:app --reload --port 8000
"""

import os
import time
import traceback
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ─── Local imports ────────────────────────────────────────────────
from rlm_core import (
    rlm, make_azure_client, ROOT_DEPLOYMENT, SUB_DEPLOYMENT,
    reset_token_usage, get_token_usage,
    embed_text, EMBEDDING_DEPLOYMENT,
)
from advanced_router import route_query_advanced
from cross_doc import run_cross_doc, _sub_cache_invalidate_for_doc, _sub_cache_size, _sub_cache_clear
from registry import DynamicRegistry


# ─── Configuration ────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50  # per upload
CONFIDENCE_THRESHOLD = 0.4  # below this, treat router output as NONE
KEYWORD_FALLBACK_MAX_DOCS = 3  # at most this many docs surfaced via substring search
KEYWORD_FALLBACK_CONFIDENCE = 0.5  # synthetic confidence reported when fallback fires


# ─── Result cache (zero-quality-risk repeat-query optimization) ───
# Caches the final answer for (query, doc_set, doc_versions) tuples. A re-run
# of the same query on the same library returns the same text at zero new
# token cost. Cache is invalidated automatically when any participating doc is
# replaced or deleted (because the doc id changes – registry hashes contents).
#
# Toggle via env: CACHE_ENABLED=true|false (default: true)
# Cap size via env: CACHE_MAX_ENTRIES (default: 200)

import hashlib
import json
from collections import OrderedDict

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "200"))

# ─── Semantic cache (Layer 2) ─────────────────────────────────────
# When the exact-match cache misses, fall through to a semantic lookup that
# embeds the new query and compares against stored embeddings of previously
# cached queries. If similarity to any cached entry exceeds the threshold AND
# the doc_ids match exactly, that entry is served.
#
# Why exact doc_ids match: two semantically-similar queries can route to
# different document sets ("TCS revenue" → [tcs]; "compare TCS to Infosys" →
# [tcs, infosys]). The cached answer is bound to the doc set it was computed
# different document sets ("TCS revenue" → [tcs]; "compare TCS to Infosys" →
# [tcs, infosys]). The cached answer is bound to the doc set it was computed
# from; serving the wrong one would be a real bug.
#
# Threshold tuning: 0.93 admits clear paraphrases ("What is TCS revenue?"
# vs "Give TCS revenue") while rejecting topic-near but distinct queries
# ("TCS revenue" vs "TCS profit"). Raise toward 0.95+ for stricter matching;
# lower toward 0.90 for looser matching. Every semantic hit logs its score
# so you can audit and tune.

import math

SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.93"))


def _cosine_sim(a, b) -> float:
    """Cosine similarity between two embedding vectors. Returns 0.0 on any
    structural issue (mismatched dims, zero magnitudes, missing data)."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


def _semantic_cache_lookup(query: str, doc_ids, client) -> Optional[dict]:
    """Layer 2 cache: find a semantically-similar previously-cached entry
    whose doc_ids match the current routing decision exactly. Returns the
    cache entry dict on hit, or None on miss / disabled / API failure.

    Side effects on hit:
      - moves the entry to MRU position
      - logs the similarity score and original cached query for auditability
    """

    if not (SEMANTIC_CACHE_ENABLED and CACHE_ENABLED):
        return None

    if not _RESULT_CACHE:
        return None

    sorted_doc_ids = sorted(doc_ids)

    # Pre-filter to entries with same routing decision AND a stored embedding.
    # Old entries from before this feature won't have query_embedding; skip them.
    candidates = [
        (k, v) for k, v in _RESULT_CACHE.items()
        if sorted(v.get("doc_ids", [])) == sorted_doc_ids
        and v.get("query_embedding") is not None
    ]

    if not candidates:
        return None

    # Only embed the new query if there's actually something to compare against.
    query_emb = embed_text(client, query, EMBEDDING_DEPLOYMENT)
    if query_emb is None:
        return None  # API failure – fall through to normal RLM execution

    best_key, best_entry, best_sim = None, None, 0.0

    for k, entry in candidates:
        sim = _cosine_sim(query_emb, entry["query_embedding"])
        if sim > best_sim:
            best_sim, best_key, best_entry = sim, k, entry

    if best_sim >= SEMANTIC_CACHE_THRESHOLD:
        _RESULT_CACHE.move_to_end(best_key)
        cached_q = best_entry.get("query_original", "(unknown)")
        print(
            f"[semantic-cache] HIT sim={best_sim:.3f} "
            f">= {SEMANTIC_CACHE_THRESHOLD:.2f} + cached: '{cached_q[:70]}'"
        )
        
                # Stamp the entry with match info so the caller can surface it
        best_entry = dict(best_entry)  # shallow copy so we don't mutate the cache
        best_entry["_semantic_match"] = {
            "similarity": round(best_sim, 4),
            "cached_query": cached_q,
        }

        return best_entry

    print(
        f"[semantic-cache] miss best={best_sim:.3f} "
        f"< {SEMANTIC_CACHE_THRESHOLD:.2f} (would not serve)"
    )
    return None


# OrderedDict gives us LRU semantics with move_to_end
_RESULT_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def _cache_key(query: str, doc_ids) -> str:
    """Stable key over (normalised query, sorted doc ids)."""
    canonical = f"{query.strip().lower()}||{';'.join(sorted(doc_ids))}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _cache_get(key: str):
    """Return cached entry and refresh LRU order, or None if absent/disabled."""
    if not CACHE_ENABLED or key not in _RESULT_CACHE:
        return None

    _RESULT_CACHE.move_to_end(key)
    return _RESULT_CACHE[key]


def _cache_put(key: str, payload: dict) -> None:
    """Insert into cache; evict oldest if over capacity."""
    if not CACHE_ENABLED:
        return

    _RESULT_CACHE[key] = payload
    _RESULT_CACHE.move_to_end(key)

    while len(_RESULT_CACHE) > CACHE_MAX_ENTRIES:
        _RESULT_CACHE.popitem(last=False)


def _cache_invalidate_for_doc(doc_id: str) -> int:
    """Drop any cache entries that reference this doc id. Called on doc
    upload/delete so stale answers don't survive a library change.
    Returns the number of entries evicted."""
    evicted = 0

    for key in list(_RESULT_CACHE.keys()):
        if doc_id in _RESULT_CACHE[key].get("doc_ids", []):
            del _RESULT_CACHE[key]
            evicted += 1

    return evicted


# ─── Keyword fallback for routing failures ─────────────────────────
# When the router returns NONE or low confidence, the LLM has not recognised any
# query term against any document's rich description. This is the failure mode
# for queries about entities the LLM has no training-data knowledge of (e.g. an
# internal tool name) and that didn't surface in the 7KB sample used to build
# descriptions. Before bailing out, try a cheap pure-Python substring search.

import re as _re

_STOPWORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on",
    "at", "for", "to", "with", "from", "by", "as", "that", "this", "these",
    "those", "and", "or", "but", "if", "then", "else", "than", "how", "why",
    "when", "where", "who", "which", "whom", "whose", "does", "do", "did",
    "doing", "done", "have", "has", "had", "having", "will", "would", "could",
    "should", "may", "might", "can", "tell", "me", "about", "any", "some",
    "all", "each", "every", "no", "not", "only", "very", "just", "also",
}

def _distinctive_terms(query: str) -> list:
    """Pull out query terms worth substring-searching.

    A term qualifies if it is NOT a stopword AND:
      (a) at least 4 characters, OR
      (b) length >= 3 and starts with an uppercase letter
          (catches CamelCase product names, acronyms like XYZ, etc.)
    """
    tokens = _re.findall(r"[A-Za-z][A-Za-z0-9_]+", query)
    out, seen = [], set()

    for t in tokens:
        low = t.lower()

        if low in seen:
            continue

        if low in _STOPWORDS:
            continue

        if len(t) >= 4 or (t[0].isupper() and len(t) >= 3):
            out.append(t)
            seen.add(low)

    return out


def _keyword_fallback(query: str, registry) -> tuple:
    """Substring-search every uploaded document for distinctive query terms.

    Returns (doc_ids, hits_per_doc, terms_used). doc_ids is sorted by descending
    total hit count and capped at KEYWORD_FALLBACK_MAX_DOCS. Empty tuple of
    lists if no terms were distinctive enough to search, or no doc matched.
    """
    terms = _distinctive_terms(query)

    if not terms:
        return [], {}, []

    hits = {}

    for doc_id, doc in registry.docs.items():
        text_lower = (doc.get("text") or "").lower()

        if not text_lower:
            continue

        n = sum(text_lower.count(t.lower()) for t in terms)

        if n > 0:
            hits[doc_id] = n

    if not hits:
        return [], {}, terms

    ranked = sorted(hits, key=hits.get, reverse=True)[:KEYWORD_FALLBACK_MAX_DOCS]
    return ranked, hits, terms


# ─── App lifecycle ────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Document RLM",
    description="Upload documents, ask questions, get routed answers."
)

# Mount static files (the frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Azure client and registry (shared across requests)
print("Initializing Azure client...")
_client = make_azure_client()
print(f"  Root LM: {ROOT_DEPLOYMENT}")
print(f"  Sub-LM: {SUB_DEPLOYMENT}")
print("Loading registry...")
_registry = DynamicRegistry(_client)
print(f"  Loaded {len(_registry.docs)} document(s) on startup.")


# ─── Request/response models ──────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class DocResponse(BaseModel):
    id: str
    title: str
    filename: str
    description: str
    char_count: int
    uploaded_at: str


# ─── Routes ───────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/api/docs")
def list_docs():
    return {"docs": _registry.list_for_api()}


@app.post("/api/upload")
async def upload_doc(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    # Validate extension
    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # Read bytes (with size check)
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_SIZE_MB} MB."
        )

    try:
        t0 = time.time()

        entry = _registry.add_uploaded_file(
            filename=file.filename,
            file_bytes=contents,
            title=title,
        )

        elapsed = time.time() - t0

        print(
            f"[upload] '{file.filename}' "
            f"({entry['char_count']:,} chars, described in {elapsed:.1f}s)"
        )

        return {
            "id": entry["id"],
            "title": entry["title"],
            "filename": entry["filename"],
            "description": entry["description"],
            "char_count": entry["char_count"],
            "uploaded_at": entry["uploaded_at"],
            "elapsed": round(elapsed, 1),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str):
    ok = _registry.remove(doc_id)

    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")

    evicted_main = _cache_invalidate_for_doc(doc_id)
    evicted_sub = _sub_cache_invalidate_for_doc(doc_id)

    if evicted_main:
        print(
            f"[cache] invalidated {evicted_main} main entr{'y' if evicted_main == 1 else 'ies'} "
            f"referencing deleted doc {doc_id}"
        )

    if evicted_sub:
        print(
            f"[sub-cache] invalidated {evicted_sub} sub-answer entr{'y' if evicted_sub == 1 else 'ies'} "
            f"referencing deleted doc {doc_id}"
        )

    return {"deleted": doc_id}


@app.get("/api/cache")
def cache_stats():
    """Return the result-cache state. The 'entries' array is ordered
    most-recent-first so the frontend can use it for a 'recent queries' UI."""
    # Order entries by cached_at desc (fall back to insertion order)
    entries_sorted = sorted(
        _RESULT_CACHE.items(),
        key=lambda kv: kv[1].get("cached_at", 0),
        reverse=True,
    )

    return {
        "enabled": CACHE_ENABLED,
        "max_entries": CACHE_MAX_ENTRIES,
        "size": len(_RESULT_CACHE),
        "subanswer_size": _sub_cache_size(),
        "entries": [
            {
                "key": k[:12] + "…",
                "query": v.get("query", ""),
                "cached_at": v.get("cached_at"),
                "tokens_saved": v["tokens"]["total"],
                "doc_ids": v["doc_ids"],
                "doc_titles": v.get("doc_titles", []),
                "mode": v.get("mode", ""),
            }
            for k, v in entries_sorted
        ],
    }


@app.delete("/api/cache")
def cache_clear():
    """Manually clear all cached results (both main result cache and per-doc sub-cache)."""
    main_n = len(_RESULT_CACHE)
    sub_n = _sub_cache_size()
    _RESULT_CACHE.clear()
    _sub_cache_clear()
    return {"cleared": main_n, "sub_cleared": sub_n}


@app.delete("/api/cache/{key_prefix}")
def cache_delete_one(key_prefix: str):
    """
    Delete a single cache entry by key prefix.

    Accepts the full sha256 key or any unique prefix (minimum 4 chars).
    The trailing "…" shown in /api/cache stats output is stripped automatically
    so users can paste directly from the entries list.

    Errors:
      400 - prefix too short (collision risk)
      404 - no entry matches
      409 - prefix matches multiple entries (provide more characters)
    """

    # Tolerate users pasting the truncated form "<12chars>…"
    key_prefix = key_prefix.rstrip("…").rstrip(".")

    if len(key_prefix) < 4:
        raise HTTPException(
            status_code=400,
            detail="key_prefix too short — provide at least 4 characters"
        )

    matches = [k for k in _RESULT_CACHE if k.startswith(key_prefix)]

    if len(matches) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No cache entry matches prefix '{key_prefix}'"
        )

    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Prefix '{key_prefix}' matches {len(matches)} entries; "
                f"provide more characters to disambiguate"
            )
        )

    key = matches[0]
    entry = _RESULT_CACHE.pop(key)

    saved = entry.get("tokens", {}).get("total", 0)

    print(
        f"[cache] deleted {key[:12]}… — query: "
        f"'{entry.get('query', '?')[:60]}' "
        f"(was saving ~{saved:,} tokens)"
    )

    return {
        "deleted": True,
        "key": key,
        "query": entry.get("query", ""),
        "doc_titles": entry.get("doc_titles", []),
        "tokens_freed": saved,
        "remaining": len(_RESULT_CACHE),
    }
    
@app.post("/api/query")
def run_query(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")

    reset_token_usage()  # fresh accumulator for this request

    if not _registry.docs:
        return JSONResponse({
            "answer": "No documents uploaded yet. Upload at least one PDF or DOCX.",
            "routing": {
                "doc_ids": [],
                "confidence": 0.0,
                "reason": "registry is empty",
                "mode": "no_docs"
            },
            "per_doc": [],
            "timings": {"route": 0.0, "rlm": 0.0, "total": 0.0},
        })

    total_t0 = time.time()

    # —— STAGE 1: ROUTE ———————————————————————————————
    route_t0 = time.time()
    catalog = _registry.catalog_for_router()
    doc_ids, confidence, reason = route_query_advanced(
        query=query,
        docs_catalog=catalog,
        client=_client,
        verbose=True,
    )
    route_elapsed = time.time() - route_t0

    # Confidence guardrail — but before giving up, try keyword fallback.
    # This catches the failure mode where the query is about a term the LLM
    # has no training-data knowledge of AND the rich description happens to
    # not mention it. Example: "what is XYZTool?" where XYZTool is an
    # internal-only term mentioned a few times in one of the uploaded docs.
    router_uncertain = (not doc_ids) or (confidence < CONFIDENCE_THRESHOLD)

    if router_uncertain:
        fb_ids, fb_hits, fb_terms = _keyword_fallback(query, _registry)

        if fb_ids:
            doc_ids = fb_ids
            confidence = KEYWORD_FALLBACK_CONFIDENCE

            hit_summary = ", ".join(
                f"{_registry.get(d)['title']}={fb_hits[d]}" for d in fb_ids
            )

            reason = (
                f"Router was uncertain (original confidence {confidence:.2f}); "
                f"keyword fallback found distinctive terms {fb_terms} in: {hit_summary}."
            )

            print(f"[keyword fallback] terms={fb_terms} -> docs={fb_ids}")

        elif doc_ids and confidence < CONFIDENCE_THRESHOLD:
            return _no_answer_response(
                query, doc_ids, confidence,
                f"Confidence {confidence:.2f} below threshold ({CONFIDENCE_THRESHOLD}) "
                f"and no distinctive query terms matched any uploaded document. "
                f"Original router reason: {reason}",
                route_elapsed, "low_confidence",
            )

        else:
            return _no_answer_response(
                query, doc_ids, confidence,
                f"{reason} No distinctive query terms matched any uploaded document either.",
                route_elapsed, "no_match",
            )

    # —— CACHE LOOKUP ————————————————————————————————
    # Layered cache: exact text match first (free), then semantic fallback
    # (one embedding call). Both layers require doc_ids to match the routing
    # decision, so a reworded query that routes differently never reuses an
    # unrelated answer.
    cache_key = _cache_key(query, doc_ids)
    cached = _cache_get(cache_key)
    semantic_match_info = None  # populated only if Layer 2 hits

    if cached is None:
        cached = _semantic_cache_lookup(query, doc_ids, _client)
        if cached is not None:
            semantic_match_info = cached.pop("_semantic_match", None)

    if cached is not None:
        saved = cached["tokens"]["total"]
        layer = "semantic" if semantic_match_info else "exact"

        print(f"[cache] HIT ({layer}) for query (saved ~{saved:,} tokens)")

        # Build the cached-response note so the user can see which layer hit
        if semantic_match_info:
            note = (
                f"\n\n*(served from cache – 0 new tokens · matched a "
                f"previous query at {semantic_match_info['similarity']*100:.0f}% "
                f"similarity: \"{semantic_match_info['cached_query']}\")*"
            )
        else:
            note = "\n\n*(served from cache – 0 new tokens)*"

        return {
            "answer": cached["answer"] + note,
            "routing": {
                "doc_ids": doc_ids,
                "titles": [_registry.get(d)["title"] for d in doc_ids],
                "confidence": confidence,
                "reason": reason + (f" [cached · {layer}]"),
                "mode": cached["mode"] + "_cached",
            },
            "per_doc": cached["per_doc"],
            "tokens": {
                **cached["tokens"],
                "cached": True,
                "cache_layer": layer,
                "semantic_match": semantic_match_info,
            },
            "timings": {
                "route": round(route_elapsed, 1),
                "rlm": 0.0,
                "total": round(route_elapsed, 1),
            },
        }

    # —— STAGE 2: RLM ———————————————————————————————
    rlm_t0 = time.time()

    if len(doc_ids) == 1:
        # Single-doc mode — direct RLM call
        doc = _registry.get(doc_ids[0])

        answer = rlm(
            context=doc["text"],
            query=query,
            client=_client,
            depth=0,
            verbose=True,
        )

        per_doc = [{
            "doc_id": doc_ids[0],
            "title": doc["title"],
            "sub_answer": answer
        }]

        mode = "single_doc"

    else:
        # Multi-doc fan-out
        result = run_cross_doc(
            query=query,
            doc_ids=doc_ids,
            doc_lookup=_registry.docs,
            client=_client,
            verbose=True,
        )

        answer = result["final_answer"]
        per_doc = result["per_doc"]
        mode = "fan_out"

    rlm_elapsed = time.time() - rlm_t0

    total_elapsed = time.time() - total_t0
    usage = get_token_usage()
    answer_with_footer = (answer or "") + usage.footer()

    # —— CACHE WRITE ————————————————————————————————
    # Store the final answer keyed on (query, doc_ids). Subsequent identical
    # queries against the same library will hit the cache at zero cost.
    # We also persist query text, doc titles, and a timestamp so /api/cache
    # can surface a useful "recent queries" list in the UI.
    # Embed the query so the semantic cache layer can match reworded variants
    # later. Embedding failure is non-fatal — entry still serves exact matches.
    query_embedding = None
    if SEMANTIC_CACHE_ENABLED:
        query_embedding = embed_text(_client, query, EMBEDDING_DEPLOYMENT)

    _cache_put(cache_key, {
        "query": query,
        "answer": answer_with_footer,
        "doc_ids": list(doc_ids),
        "doc_titles": [_registry.get(d)["title"] for d in doc_ids],
        "per_doc": per_doc,
        "tokens": usage.to_dict(),
        "mode": mode,
        "cached_at": time.time(),
        "query_original": query,          # alias used by semantic lookup
        "query_embedding": query_embedding,  # for semantic matching; may be None
    })

    return {
        "answer": answer_with_footer,
        "routing": {
            "doc_ids": doc_ids,
            "titles": [_registry.get(d)["title"] for d in doc_ids],
            "confidence": confidence,
            "reason": reason,
            "mode": mode,
        },
        "per_doc": per_doc,
        "tokens": usage.to_dict(),
        "timings": {
            "route": round(route_elapsed, 1),
            "rlm": round(rlm_elapsed, 1),
            "total": round(total_elapsed, 1),
        },
    }


def _no_answer_response(query, doc_ids, confidence, reason, route_elapsed, mode):
    return {
        "answer": (
            "No relevant document was found for this query. "
            "Try uploading a document on this topic, or rephrasing the question."
        ),
        "routing": {
            "doc_ids": doc_ids,
            "confidence": confidence,
            "reason": reason,
            "mode": mode,
        },
        "per_doc": [],
        "timings": {
            "route": round(route_elapsed, 1),
            "rlm": 0.0,
            "total": round(route_elapsed, 1),
        },
    }