https://github.com/user-attachments/assets/4ac470e9-e13b-4dc7-bbb3-cb8bdf0de7ec

# 📄 Document Concierge

**A recursive-language-model document Q&A engine — upload PDFs/DOCX, ask anything, get routed, verified, cached answers.**

Document Concierge is a FastAPI web app built around a faithful implementation of the **Recursive Language Model (RLM)** algorithm (Zhang, Kraska & Khattab, 2026). Instead of stuffing documents into a context window and hoping for the best, the model is given a Python REPL, a `context` variable holding the raw document text, and the ability to recursively spawn sub-LLM calls to search, chunk, and verify — iterating until it can produce an evidence-backed final answer.

On top of the base algorithm, this project adds a full product layer: multi-document upload and routing, cross-document fan-out and synthesis, a three-layer caching system, and a four-page card-catalogue web UI.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Core Concept: What Is an RLM?](#core-concept-what-is-an-rlm)
- [Feature Overview](#feature-overview)
- [Architecture](#architecture)
- [Request Lifecycle](#request-lifecycle)
- [File Map](#file-map)
- [Setup](#setup)
- [Using the App](#using-the-app)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Caching System, In Depth](#caching-system-in-depth)
- [Anti-Hallucination Design](#anti-hallucination-design)
- [Design System](#design-system)
- [Limitations & Roadmap](#limitations--roadmap)
- [Dependencies](#dependencies)
- [Credits](#credits)

---

## Why This Exists

Basic single-document RLM implementations hit two walls in practice:

| # | Limitation | How Document Concierge fixes it |
|---|---|---|
| 1 | Queries that span multiple documents ("compare X and Y") aren't supported | An **advanced router** can select several documents at once; a **fan-out pipeline** runs RLM on each in parallel and synthesizes one answer |
| 2 | Routing accuracy depends entirely on document description quality | Each upload gets a **structured 5-field auto-description** (type, subject, time period, key terms, summary) plus a numeric **confidence score**, with a keyword-overlap fallback when the router is unsure |

Everything else in the repo — the caching layers, the anti-hallucination prompt engineering, the persistent registry — exists to make those two answers fast, cheap, and trustworthy on repeated use.

---

## Core Concept: What Is an RLM?

A standard LLM call answers a question by reading the entire context in one shot. A **Recursive Language Model** instead treats the context as *data it can program against*:

1. The root LM is given the document not as raw text in the prompt, but as a `context` variable inside a **persistent Python REPL**.
2. It writes and executes Python (regex search, slicing, chunked iteration) to explore the document.
3. It calls `llm_query(prompt)` to delegate simple sub-tasks (summarize a chunk, extract a fact) to a cheaper, non-recursive sub-LM call.
4. For sub-tasks that are themselves too complex for one call, it calls `rlm_query(context, query)` — which spawns a **full nested RLM loop** with its own REPL and its own sub-calls, up to a maximum recursion depth.
5. It keeps iterating — code, observe stdout, code again — until it calls `FINAL(answer)` or `FINAL_VAR(varname)`.

This means the model can handle documents far larger than any single context window would comfortably allow, because it never has to read the whole thing at once — it searches, slices, and delegates instead.

`rlm_core.py` is a line-for-line implementation of this loop (Algorithm 1 in the paper), including the exact system prompt from the paper's Appendix C.1, extended with financial-document search heuristics and evidence rules described below.

---

## Feature Overview

- 📤 **Multi-format upload** — PDF (via `pdftotext -layout` with a pure-Python `pypdf` fallback), DOCX (paragraphs + tables), TXT, and Markdown.
- 🧠 **Recursive Language Model core** — persistent REPL, recursive sub-RLM delegation up to depth 3, balanced-paren `FINAL()`/`FINAL_VAR()` parsing that survives numbered lists and nested parentheses in the answer text.
- 🧭 **Advanced multi-document router** — chooses one document, several documents, or none, with a 0.0–1.0 confidence score and a keyword-overlap safety net for low-confidence routes.
- 🔀 **Cross-document fan-out & aggregation** — runs RLM on each selected document concurrently (thread pool), then synthesizes one coherent, cited answer — including explicit handling of disagreements between sources and preservation of each source's original numeric units (₹ crore vs. $ million are never silently converted).
- 🧾 **Structured document auto-description** — one LM call per upload produces `DOC_TYPE`, `SUBJECT`, `TIME_PERIOD`, `KEY_TERMS`, `SUMMARY` — used by the router instead of raw document text, keeping routing fast even with many documents.
- 🗄️ **Three-layer caching** — exact-match result cache, embedding-based semantic cache, and a per-document sub-answer cache shared across queries (see [Caching System](#caching-system-in-depth)).
- 💾 **Crash-safe persistence** — every cache autosaves every 60s, on `SIGTERM`/`Ctrl+C` via `atexit`, and on graceful FastAPI shutdown; the document registry is rebuilt from disk on restart without re-spending LM calls on descriptions.
- 🛡️ **Anti-hallucination guardrails** — sub-LMs are instructed to verify every fact against REPL output (not training-data memory), quote verbatim, run a minimum number of distinct searches before declaring a fact absent, and prefer audited financial-statement figures over narrative rollups.
- 📊 **Full observability** — every response includes per-stage token usage (root LM, sub-LM, router, aggregator, describe, fallback), routing rationale, and a route/RLM/total timing breakdown.
- 🗂️ **Card-catalogue frontend** — a four-page vanilla HTML/Tailwind app (`index`, `ask`, `document`, `cache`) styled as a library archive rather than a generic AI-SaaS dashboard: folder-tab navigation, a stamped holdings count, an "intake slip" upload zone, and perforated index cards with real accession numbers derived from each document's own ID. No build step, no framework — plain HTML/CSS/JS, a shared Tailwind config, and a shared stylesheet across all four pages.

---

## Architecture

```
┌─────────────┐        HTTP/JSON        ┌───────────────────────────────────────────┐
│  Browser    │ ───────────────────────▶│              FastAPI (app.py)              │
│ (static/*)  │◀─────────────────────── │                                             │
└─────────────┘                         │  /api/upload   /api/query   /api/docs      │
                                         │  /api/cache    /api/cache/{id}             │
                                         └───────────────┬─────────────────────────────┘
                                                          │
              ┌───────────────────────────────────────────┼───────────────────────────────┐
              ▼                                           ▼                               ▼
     ┌─────────────────┐                       ┌───────────────────────┐       ┌───────────────────┐
     │  registry.py     │                       │  advanced_router.py   │       │  cross_doc.py      │
     │  DynamicRegistry  │                       │  route_query_advanced │       │  run_cross_doc      │
     │  ── uploads/ +    │                       │  ── confidence score  │       │  ── ThreadPool fan- │
     │     registry.json │                       │  ── multi-doc pick    │       │     out + aggregate │
     └────────┬─────────┘                       └───────────┬────────────┘       └─────────┬──────────┘
              │                                              │                              │
              ▼                                              ▼                              ▼
     ┌─────────────────┐                       ┌────────────────────────────────────────────────────┐
     │document_loader.py│                       │                    rlm_core.py                     │
     │ PDF/DOCX/TXT/MD   │                       │  REPLEnvironment · llm_query · rlm_query · FINAL   │
     │ extraction        │                       │  parsing · TokenUsage · Azure OpenAI client        │
     └─────────────────┘                       └────────────────────────────────────────────────────┘

                     ┌───────────────────────────────────────────────────┐
                     │                 cache_persist.py                   │
                     │  CachePersister — shared load/autosave/atexit for  │
                     │  cache/result_cache.pkl and cache/sub_answer_      │
                     │  cache.pkl                                          │
                     └───────────────────────────────────────────────────┘
```

---

## Request Lifecycle

**Upload (`POST /api/upload`)**

1. Validate extension (`.pdf`/`.docx`/`.txt`/`.md`) and size (≤ 50 MB).
2. `document_loader.load_document()` extracts text (`pdftotext -layout` → `pypdf` fallback for PDFs; `python-docx` for DOCX, including table cells).
3. `advanced_router.generate_rich_description()` makes **one LM call** producing the 5-field structured description.
4. `registry.py` persists the file to `uploads/`, writes metadata to `uploads/registry.json`, and keeps full text in memory.
5. `cross_doc.register_doc_companies()` scans the first 3KB for capitalized entity/company names (with a hand-curated alias table, e.g. `TCS` → Tata Consultancy Services) to power query normalization for the sub-answer cache.

**Query (`POST /api/query`)**

1. **Route** — `route_query_advanced()` sends the query plus every document's structured description (not raw text) to the router LM, which returns doc IDs, a confidence score, and a one-sentence reason.
2. **Confidence gate** — if confidence is below `CONFIDENCE_THRESHOLD` (default `0.4`) or no document was picked, a regex-based keyword-overlap fallback searches raw document text for distinctive query terms before giving up with a "no relevant document" response.
3. **Cache lookup** — check the exact-match cache first (SHA-256 of normalized query + sorted doc IDs), then the embedding-based semantic cache (cosine similarity ≥ `0.93`).
4. **Answer**:
   - **Single document** → run `rlm()` directly on that document's text.
   - **Multiple documents** → `run_cross_doc()`: check the per-document sub-answer cache for each doc, run RLM (in parallel, up to 4 workers) only on cache misses, then make one aggregator LLM call to synthesize a final, cited answer.
5. **Cache write** — store the result (and its query embedding) in the result cache; store each per-document sub-answer independently in the sub-answer cache for reuse by future multi-document queries.
6. **Respond** — answer text (with a token-usage footer), routing rationale, per-document findings, token breakdown by stage, and route/RLM/total timings.

---

## File Map

```
.
├── app.py                  FastAPI app — routes, upload handling, query orchestration, all 3 cache layers
├── rlm_core.py              The RLM algorithm itself: REPL, system prompt, FINAL parsing, Azure client, token accounting
├── advanced_router.py       Structured document description generation + multi-doc router with confidence scoring
├── cross_doc.py             Fan-out execution, query normalization, aggregation prompt, sub-answer cache
├── registry.py               DynamicRegistry — document CRUD, disk persistence, router catalog view
├── document_loader.py        PDF (pdftotext/pypdf) + DOCX (python-docx) + TXT/MD extraction and cleanup
├── cache_persist.py          Generic CachePersister: autosave thread, atexit hook, pickle load/save
├── requirements.txt
├── static/
│   ├── theme.js               Single shared Tailwind config (colors, type scale, radii) used by all 4 pages
│   ├── theme.css              Shared component classes: .folder-tab, .stamp, .intake-slip, .perforated, .card-lift
│   ├── index.html             Library — upload dropzone + card-catalogue document grid
│   ├── ask.html                Query box + turn history (routing, confidence, per-doc findings, token cost)
│   ├── document.html           Single-document detail view (description, metadata, delete)
│   ├── cache.html              Cache inspector — sortable table of cached queries, savings, and per-entry clear
│   └── app.js                  Upload/dropzone, query form, response rendering, token bar, all 4 pages' DOM logic
├── uploads/                   Auto-created — stores uploaded files + registry.json (git-ignored)
└── cache/                     Auto-created — result_cache.pkl + sub_answer_cache.pkl (persisted caches)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `fastapi`, `uvicorn[standard]`, `python-multipart`, `python-docx`, `openai`, `python-dotenv`, `colorama`, `pypdf`, `pydantic`.

> **Better PDF extraction (optional but recommended):** install **Poppler** so `pdftotext` is on your `PATH`. It preserves table layout, which matters a lot for financial documents (10-Ks, annual reports). `document_loader.py` uses it automatically when available and falls back to `pypdf` otherwise.
>
> - macOS: `brew install poppler`
> - Debian/Ubuntu: `apt-get install poppler-utils`

## Multi-Tenant Mode (public deployments)

If you're hosting this publicly (e.g. sharing a link on social media), you do **not**
want your own Azure key used by every visitor. As of this version, the app no longer
runs any Azure call on a shared/global key:

- Every visitor lands on **Setup** (`/static/settings.html`) and enters their **own**
  `AZURE_ENDPOINT`, `AZURE_API_KEY`, and deployment names.
- The server makes real, minimal test calls to Azure with those exact values before
  accepting them. Bad/missing values are rejected with a specific reason.
- On success, the visitor gets an HttpOnly session cookie. Their credentials live only
  in server memory for that session (never written to disk) and expire after
  `SESSION_TTL_SECONDS` of inactivity (default 4h, in `.env`).
- `/api/upload` and `/api/query` — the only endpoints that call Azure — require a valid
  session and use *that visitor's* client. If credentials aren't set or stop working,
  those endpoints return `401` and the frontend redirects back to Setup. The app simply
  will not run Azure calls on your behalf.

The `.env` file's `AZURE_*` values are now only a fallback for local single-user
development (running `uvicorn app:app` yourself, never touching the Setup page).

### 2. Configure environment

Create a `.env` file in the project root:

```env
AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_API_KEY=your-key-here
AZURE_API_VERSION=2024-12-01-preview

AZURE_ROOT_DEPLOYMENT=gpt-5-mini
AZURE_SUB_DEPLOYMENT=gpt-5-mini
EMBEDDING_DEPLOYMENT=text-embedding-3-small

ROOT_REASONING_EFFORT=high
SUB_REASONING_EFFORT=high

CACHE_ENABLED=true
CACHE_MAX_ENTRIES=200
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.93

# How long an idle visitor session (their validated Azure credentials) is
# kept in server memory before they must re-enter them. Seconds.
SESSION_TTL_SECONDS=14400
```

These `.env` values are a **local-dev fallback only** — see [Multi-Tenant Mode](#multi-tenant-mode-public-deployments)
above. In the public/hosted flow, each visitor supplies their own values on the Setup page instead.

### 3. Run

```bash
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000**.

---

## Using the App

The UI is four pages, linked by the folder-tab nav at the top of every screen:

| Page | Purpose |
|---|---|
| **Library** (`/static/index.html`) | Upload documents (drag onto the intake slip, or click to browse) and browse the card-catalogue of everything uploaded so far |
| **Ask** (`/static/ask.html`) | Submit a query and see the routing decision, per-document findings, and token cost for each turn |
| **Document** (`/static/document.html?id=...`) | A single document's own detail view — its generated description, metadata, and a delete action |
| **Cache** (`/static/cache.html`) | Inspect every cached query, tokens saved, and clear individual entries or the whole cache |

**1. Upload documents** on the Library page — each file triggers one LM call to generate its routing description; this typically takes a few seconds and only happens once per file (descriptions are reused across restarts).

**2. Ask a question** on the Ask page — the router automatically picks the mode:

| Mode | Trigger | Behavior |
|---|---|---|
| `single_doc` | The answer lives in one document | RLM runs once, on that document |
| `fan_out` | Comparative query, or entities spread across documents | RLM runs on each relevant document in parallel; results are synthesized into one answer |
| `none` | Nothing in the library is relevant | Returns a clear "no relevant document" response — no invented answers |

**3. Read the response** — each turn shows which document(s) were used and why, a confidence bar, per-document findings for fan-out queries, a token-cost breakdown by stage, and route/RLM/total timing.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend (`static/index.html`) |
| `GET` | `/api/docs` | List all uploaded documents (metadata only, no full text) |
| `POST` | `/api/upload` | Upload a file (`multipart/form-data`, fields: `file`, optional `title`) |
| `DELETE` | `/api/docs/{doc_id}` | Remove a document; invalidates any cache entries that reference it |
| `POST` | `/api/query` | Ask a question. Body: `{"query": "..."}` |
| `GET` | `/api/cache` | Inspect the result cache (size, entries, tokens saved per entry) |
| `DELETE` | `/api/cache` | Clear both the result cache and the sub-answer cache |
| `DELETE` | `/api/cache/{key_prefix}` | Delete one cache entry by (unambiguous ≥4-char) key prefix |

**Example query response:**

```json
{
  "answer": "Adobe's FY2024 total revenue was $21,505 million...",
  "routing": {
    "doc_ids": ["a1b2c3d4e5f6"],
    "titles": ["Adobe_10K_2024.pdf"],
    "confidence": 0.93,
    "reason": "The query asks specifically about Adobe's revenue, matched on TITLE and SUBJECT.",
    "mode": "single_doc"
  },
  "per_doc": [{ "doc_id": "a1b2c3d4e5f6", "title": "Adobe_10K_2024.pdf", "sub_answer": "..." }],
  "tokens": { "total": 18432, "prompt": 15210, "completion": 3222, "calls": 6, "by_stage": { "root_lm": {}, "router": {} } },
  "timings": { "route": 1.2, "rlm": 14.8, "total": 16.0 }
}
```

---

## Configuration Reference

| Setting | Default | Location | Notes |
|---|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.4` | `app.py` | Below this, keyword fallback kicks in before giving up |
| `MAX_FILE_SIZE_MB` | `50` | `app.py` | Per-file upload limit |
| Accepted file types | `.pdf`, `.docx`, `.txt`, `.md` | `app.py` | `ALLOWED_EXTENSIONS` |
| Max fan-out documents | `4` | `advanced_router.py` (router prompt) | Router won't select more than this per query |
| `MAX_FANOUT_WORKERS` | `4` | `cross_doc.py` | Thread pool size for parallel per-document RLM runs |
| `MAX_DEPTH` | `3` | `rlm_core.py` | Maximum `rlm_query` recursion depth before falling back to a flat `llm_query` |
| `MAX_ROOT_ITERS` | `30` | `rlm_core.py` | Iteration cap per RLM loop before a forced fallback answer |
| `CHUNK_SIZE_CHARS` | `200,000` | `rlm_core.py` | Reference chunk size the model is told about (metadata only — the model decides its own chunking in code) |
| `STDOUT_PREVIEW_LEN` | `4,000` | `rlm_core.py` | REPL stdout is truncated to this many chars before being fed back to the root LM |
| `CACHE_MAX_ENTRIES` | `200` | env / `app.py` | Exact-match result cache size (LRU eviction) |
| `SEMANTIC_CACHE_THRESHOLD` | `0.93` | env / `app.py` | Cosine similarity required for a semantic cache hit |
| `SUB_CACHE_MAX_ENTRIES` | `500` | `cross_doc.py` | Per-document sub-answer cache size |
| `SUB_CACHE_SEMANTIC_THRESHOLD` | `0.90` | `cross_doc.py` | Cosine similarity for sub-answer cache reuse |
| `AUTOSAVE_INTERVAL` | `60` sec | `cache_persist.py` | Background cache autosave frequency |

> **Scalability note:** every uploaded document's description is sent to the router on *every* query. This is fast and cheap up to roughly 20–30 documents; beyond that, an embedding-based pre-filter ahead of the router would be the natural next step.

---

## Caching System, In Depth

Three independent layers, each solving a different repetition pattern:

**Layer 1 — Exact result cache** (`cache/result_cache.pkl`)
Key = `sha256(normalized_query + sorted(doc_ids))`. An identical query against the identical document set is served for free, with zero new LM calls.

**Layer 2 — Semantic result cache**
When there's no exact hit, the query is embedded (`text-embedding-3-small` by default) and compared via cosine similarity against cached queries that used the *same* document set. A hit at ≥ `0.93` similarity serves the cached answer, tagged with the similarity score and the original cached question so the user knows it wasn't a fresh computation.

**Layer 3 — Per-document sub-answer cache** (`cache/sub_answer_cache.pkl`)
This is the most interesting layer. Multi-document fan-out queries are decomposed per-document, and each document's query is *normalized* — other documents' company/entity names are stripped out (via `register_doc_companies()`'s alias table), stopwords removed, tokens sorted — so that:

```
"Give the total revenue of Adobe and Infosys"  →  (for the Infosys doc)  →  "infosys revenue"
"What is the total revenue of Infosys?"        →  (for the Infosys doc)  →  "infosys revenue"
```

...normalize to the *same* cache key. So if a user already asked about Infosys revenue alone, then later asks a multi-company comparison including Infosys, that document's sub-answer is served from cache and only the (cheap) aggregation call runs fresh. This layer also has its own semantic fallback (`0.90` threshold) for near-synonymous phrasing.

**All three layers persist to disk.** `cache_persist.py` provides a generic `CachePersister`: it loads from a pickle file on startup, autosaves every 60 seconds on a background thread, registers an `atexit` hook (covers `Ctrl+C` / `SIGTERM`), and the FastAPI `shutdown` event calls `save()` explicitly as a third safety net. Only `kill -9` can outrun all three — everything else survives a restart.

Deleting a document invalidates any cache entries (both layers) that reference it, so stale answers can never be served after the source document is gone.

---

## Anti-Hallucination Design

Because the underlying models often "know" things about well-known companies and public documents from training data, the prompts are explicit about **never trusting memory over REPL output**:

- Every number, name, date, and quote must have been printed by the REPL for *this specific document* — training-data recall is explicitly flagged as untrustworthy and stale.
- Anything inside quotation marks in a `FINAL` answer must be **verbatim** from REPL output; if it can't be quoted exactly, it must be paraphrased without quotes.
- A sub-LM may only conclude "this document does not contain X" after running **at least 3 distinct searches** with different angles (broad term, exact phrase, section/heading name) — and must list them, so downstream aggregation can audit the claim rather than trust it blindly.
- The aggregator prompt applies a **skepticism rule**: an unverified absence claim is downgraded to "absence not confirmed" rather than reported as established fact.
- For financial figures, audited statement lines (e.g. "Revenue from Operations" in a Consolidated Statement of Profit and Loss) are explicitly preferred over rounded narrative figures from chairman's letters or press-release-style summaries.
- **Units are never silently converted** — a figure reported in ₹ crore stays in ₹ crore, one reported in $ millions stays in $ millions; the model may *optionally* add one clearly-labeled conversion note, but the primary figure always matches the source document's own units.
- The system prompt also includes a specific playbook for PDF-layout quirks (headings broken across lines or letter-spaced by PDF extraction) — pivoting from brittle heading-string search to content-based regex search, and a documented iteration budget so the model doesn't burn its loop retrying the same failing search pattern.

---

## Design System

The frontend is built around a **library card-catalogue / archive** aesthetic — the deliberate alternative to the generic dark-mode "AI SaaS" dashboard look most RAG demos default to. Every page shares one Tailwind config (`static/theme.js`) and one stylesheet (`static/theme.css`), so there's a single source of truth for the whole visual system instead of four copies drifting apart.

- **Palette** — parchment paper tones, a deep ledger-green primary (`#2F4B3C`, standing in for the usual violet/blue SaaS accent), and a highlighter-amber reserved specifically for citations and confidence highlights. Ink stays near-black for text, never pure black.
- **Type** — **Fraunces** (a serif with real ink-trap detailing) for headlines, **IBM Plex Sans** for body and UI text, **IBM Plex Mono** for anything that's data — accession numbers, timestamps, token counts.
- **Signature element** — the **perforated index card**: every document in the library renders with a punched top edge and a mono accession reference (`REF` + the document's own ID prefix, not an invented sequence number) instead of a plain rounded rectangle, echoing a physical library catalogue card.
- **Navigation** — reads as overlapping **folder tabs** across a manila-toned strip rather than a conventional pill/underline nav.
- **Upload zone** — styled as an **intake slip** (a hand-torn dashed top edge) rather than a generic dropzone.
- **Restraint by design** — the perforated-card treatment is intentionally kept exclusive to document cards; turn cards, the cache table, and other surfaces use the plainer shared `.glass-panel` card so the signature motif doesn't get diluted by overuse.

Markdown answers are rendered client-side with `marked.js`. Icons are Material Symbols Outlined, restyled in ink/ledger tones to sit inside the archive palette rather than swapped for a custom icon set.

---

## Limitations & Roadmap

- **No embedding pre-filter before routing** — all document descriptions go to the router on every query; fine for small-to-medium libraries, would need a vector pre-filter step to scale past ~20–30 documents.
- **Single-process, in-memory registry** — the document registry and both in-memory caches are process-local; there's no distributed/multi-worker deployment story yet (would need a shared store like Redis).
- **No authentication** — the API and UI are unauthenticated; intended for local/single-user use as shipped.
- **No streaming responses** — RLM answers are returned only after the full loop completes, which can take tens of seconds on large documents or deep fan-out queries.

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn[standard]` | Web server and routing |
| `openai` | Azure OpenAI client (chat completions + embeddings) |
| `python-docx` | DOCX text and table extraction |
| `pypdf` | Pure-Python PDF fallback |
| `python-multipart` | Multipart file upload handling |
| `python-dotenv` | `.env` config loading |
| `colorama` | Colored terminal logging of the RLM loop |
| `pydantic` | Request/response models |
| `poppler` (`pdftotext`, optional, system package) | High-fidelity, table-preserving PDF extraction |

---

## Credits

Built on the **Recursive Language Model (RLM)** algorithm from Zhang, Kraska & Khattab (2026). `rlm_core.py`'s system prompt and control loop follow the paper's Algorithm 1 and Appendix C.1 closely; this project adds the web interface, multi-document routing, fan-out aggregation, three-layer caching, and a persistent document registry on top of that core.
