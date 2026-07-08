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
from collections import OrderedDict

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Response, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from cross_doc import save_sub_cache

load_dotenv()

# ─── Local imports ────────────────────────────────────────────────
from rlm_core import (
    rlm, reset_token_usage, get_token_usage, embed_text, cfg,
)
from advanced_router import route_query_advanced
from cross_doc import (
    run_cross_doc, _sub_cache_invalidate_for_doc,
    _sub_cache_size, _sub_cache_clear, _sub_cache_put,
    register_doc_companies,
)
from registry import DynamicRegistry
from cache_persist import CachePersister
import user_config
from user_config import (
    SESSION_COOKIE_NAME, CredentialError,
    validate_and_build_client, create_session, get_session, clear_session,
)


# ─── Configuration ────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50
CONFIDENCE_THRESHOLD = 0.4
KEYWORD_FALLBACK_MAX_DOCS = 3
KEYWORD_FALLBACK_CONFIDENCE = 0.5


# ─── Result cache (Layer 1 exact + Layer 2 semantic) ─────────────
#
# Persisted to cache/result_cache.pkl via CachePersister so it survives
# server restarts, Ctrl+C, and (via periodic autosave) even kill -9.

import hashlib
import math

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "200"))

SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.93"))

_RESULT_CACHE: OrderedDict = OrderedDict()


def _set_result_cache(new_data: OrderedDict) -> None:
    """Replace live result cache (used by CachePersister on load)."""
    _RESULT_CACHE.clear()
    _RESULT_CACHE.update(new_data)


_result_persister = CachePersister(
    path="cache/result_cache.pkl",
    get_cache=lambda: _RESULT_CACHE,
    set_cache=_set_result_cache,
    label="result-cache",
)
# Load first, then start autosave thread.
_result_persister.load()
_result_persister.start()


# ─── Cosine similarity helper ─────────────────────────────────────

def _cosine_sim(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ─── Semantic cache lookup (Layer 2) ─────────────────────────────

def _semantic_cache_lookup(query: str, doc_ids, client) -> Optional[dict]:
    if not (SEMANTIC_CACHE_ENABLED and CACHE_ENABLED):
        return None
    if not _RESULT_CACHE:
        return None

    sorted_doc_ids = sorted(doc_ids)
    candidates = [
        (k, v) for k, v in _RESULT_CACHE.items()
        if sorted(v.get("doc_ids", [])) == sorted_doc_ids
        and v.get("query_embedding") is not None
    ]
    if not candidates:
        return None

    query_emb = embed_text(client, query, cfg(client).embedding_deployment)
    if query_emb is None:
        return None

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
        best_entry = dict(best_entry)
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


# ─── Exact cache helpers ──────────────────────────────────────────

def _cache_key(query: str, doc_ids) -> str:
    canonical = f"{query.strip().lower()}||{';'.join(sorted(doc_ids))}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _cache_get(key: str):
    if not CACHE_ENABLED or key not in _RESULT_CACHE:
        return None
    _RESULT_CACHE.move_to_end(key)
    return _RESULT_CACHE[key]


def _cache_put(key: str, payload: dict) -> None:
    if not CACHE_ENABLED:
        return
    _RESULT_CACHE[key] = payload
    _RESULT_CACHE.move_to_end(key)
    while len(_RESULT_CACHE) > CACHE_MAX_ENTRIES:
        _RESULT_CACHE.popitem(last=False)


def _cache_invalidate_for_doc(doc_id: str) -> int:
    evicted = 0
    for key in list(_RESULT_CACHE.keys()):
        if doc_id in _RESULT_CACHE[key].get("doc_ids", []):
            del _RESULT_CACHE[key]
            evicted += 1
    if evicted:
        _result_persister.save()
    return evicted


# ─── Keyword fallback for routing failures ────────────────────────

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

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Belt-and-suspenders save on graceful shutdown (SIGTERM, uvicorn --reload
    file-change restart, etc.).  For hard kills (kill -9) the atexit hooks
    registered by each CachePersister handle the save; for Ctrl+C the atexit
    hook fires too.  The periodic autosave (every 60 s) is the safety net when
    none of the above get a chance to run.
    """
    print("=" * 60)
    print("SHUTDOWN EVENT — saving caches")
    print("=" * 60)
    try:
        _result_persister.save()
        save_sub_cache()
        print("All caches saved successfully.")
    except Exception:
        traceback.print_exc()


# Registry no longer needs a client at startup: reloading persisted docs
# reuses cached descriptions (no LM call — see registry.py docstring), and
# every code path that DOES need an LM call now receives the calling
# visitor's own per-session client explicitly.
print("Loading registry...")
_registry = DynamicRegistry()
print(f"  Loaded {len(_registry.docs)} document(s) on startup.")

for _doc_id, _doc_entry in _registry.docs.items():
    register_doc_companies(
        doc_id=_doc_id,
        doc_title=_doc_entry.get("title", ""),
        doc_text_preview=_doc_entry.get("text", "")[:3000],
    )


# ─── Session dependency ───────────────────────────────────────────
#
# Every endpoint that calls Azure (upload, query) depends on this. It reads
# the visitor's session cookie, looks up their validated Azure credentials,
# and returns their per-session client. If they haven't configured working
# credentials yet, the request is rejected with 401 and the frontend sends
# them to /static/settings.html — the project simply will not run any Azure
# call on the app owner's behalf.

def require_session(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    record = get_session(session_id)
    if record is None:
        raise HTTPException(
            status_code=401,
            detail="No valid Azure credentials configured for this session. "
                   "Please add your Azure OpenAI details on the Settings page.",
        )
    return record


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


class SettingsRequest(BaseModel):
    azure_endpoint: str
    azure_api_key: str
    azure_api_version: str = "2024-12-01-preview"
    azure_root_deployment: str
    azure_sub_deployment: str
    embedding_deployment: str
    root_reasoning_effort: str = "high"
    sub_reasoning_effort: str = "high"


# ─── Settings (per-visitor Azure credentials) ─────────────────────

@app.post("/api/settings")
def save_settings(payload: SettingsRequest, response: Response):
    try:
        client = validate_and_build_client(
            endpoint=payload.azure_endpoint,
            api_key=payload.azure_api_key,
            api_version=payload.azure_api_version,
            root_deployment=payload.azure_root_deployment,
            sub_deployment=payload.azure_sub_deployment,
            embedding_deployment=payload.embedding_deployment,
            root_reasoning_effort=payload.root_reasoning_effort,
            sub_reasoning_effort=payload.sub_reasoning_effort,
        )
    except CredentialError as e:
        raise HTTPException(status_code=400, detail=str(e))

    record = create_session(client, endpoint=payload.azure_endpoint.strip())
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=record.session_id,
        httponly=True,
        samesite="lax",
        max_age=user_config.SESSION_TTL_SECONDS,
    )
    return {"ok": True, **record.masked_summary()}


@app.get("/api/settings/status")
def settings_status(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    record = get_session(session_id)
    if record is None:
        return {"configured": False}
    return record.masked_summary()


@app.delete("/api/settings")
def clear_settings(request: Request, response: Response):
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    clear_session(session_id)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"cleared": True}


# ─── Routes ───────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# Add these three lines to handle UptimeRobot's pings
@app.head("/")
def uptime_ping():
    return {"status": "alive"}

@app.get("/")
@app.head("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/api/docs")
def list_docs():
    return {"docs": _registry.list_for_api()}


@app.post("/api/upload")
async def upload_doc(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    session=Depends(require_session),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

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
            client=session.client,
        )
        register_doc_companies(
            doc_id=entry["id"],
            doc_title=entry["title"],
            doc_text_preview=entry.get("text", "")[:3000],
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
    """Clear all cached results (both main result cache and per-doc sub-cache)."""
    main_n = len(_RESULT_CACHE)
    sub_n = _sub_cache_size()
    _RESULT_CACHE.clear()
    _result_persister.save()   # persist the cleared state immediately
    _sub_cache_clear()
    return {"cleared": main_n, "sub_cleared": sub_n}


@app.delete("/api/cache/{key_prefix}")
def cache_delete_one(key_prefix: str):
    """Delete a single cache entry by key prefix."""
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
    _result_persister.save()   # persist deletion immediately
    return {
        "deleted": True,
        "key": key,
        "query": entry.get("query", ""),
        "doc_titles": entry.get("doc_titles", []),
        "tokens_freed": saved,
        "remaining": len(_RESULT_CACHE),
    }


@app.post("/api/query")
def run_query(req: QueryRequest, session=Depends(require_session)):
    client = session.client
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")

    reset_token_usage()

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

    # —— STAGE 1: ROUTE ———————————————————————————————————————————
    route_t0 = time.time()
    catalog = _registry.catalog_for_router()
    doc_ids, confidence, reason = route_query_advanced(
        query=query,
        docs_catalog=catalog,
        client=client,
        verbose=True,
    )
    route_elapsed = time.time() - route_t0

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

    # —— CACHE LOOKUP ————————————————————————————————————————————
    cache_key = _cache_key(query, doc_ids)
    cached = _cache_get(cache_key)
    semantic_match_info = None

    if cached is None:
        cached = _semantic_cache_lookup(query, doc_ids, client)
        if cached is not None:
            semantic_match_info = cached.pop("_semantic_match", None)

    if cached is not None:
        saved = cached["tokens"]["total"]
        layer = "semantic" if semantic_match_info else "exact"
        print(f"[cache] HIT ({layer}) for query (saved ~{saved:,} tokens)")

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

    # —— STAGE 2: RLM ————————————————————————————————————————————
    rlm_t0 = time.time()

    if len(doc_ids) == 1:
        doc = _registry.get(doc_ids[0])
        answer = rlm(
            context=doc["text"],
            query=query,
            client=client,
            depth=0,
            verbose=True,
        )
        per_doc = [{
            "doc_id": doc_ids[0],
            "title": doc["title"],
            "sub_answer": answer
        }]
        mode = "single_doc"
        _sub_cache_put(
            query=query,
            doc_id=doc_ids[0],
            doc_title=doc["title"],
            sub_answer=answer,
            client=client,
        )
        print(f"[sub-cache] stored answer for '{doc['title']}' — reusable by future multi-doc queries")

    else:
        result = run_cross_doc(
            query=query,
            doc_ids=doc_ids,
            doc_lookup=_registry.docs,
            client=client,
            verbose=True,
        )
        answer = result["final_answer"]
        per_doc = result["per_doc"]
        mode = "fan_out"

    rlm_elapsed = time.time() - rlm_t0
    total_elapsed = time.time() - total_t0
    usage = get_token_usage()
    answer_with_footer = (answer or "") + usage.footer()

    # —— CACHE WRITE ————————————————————————————————————————————
    query_embedding = None
    if SEMANTIC_CACHE_ENABLED:
        query_embedding = embed_text(client, query, cfg(client).embedding_deployment)

    _cache_put(cache_key, {
        "query": query,
        "answer": answer_with_footer,
        "doc_ids": list(doc_ids),
        "doc_titles": [_registry.get(d)["title"] for d in doc_ids],
        "per_doc": per_doc,
        "tokens": usage.to_dict(),
        "mode": mode,
        "cached_at": time.time(),
        "query_original": query,
        "query_embedding": query_embedding,
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
            "Try uploading a document on this topic, or rephrase the question."
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
