# Document Concierge — Multi-Document RLM (Web)

> Upload PDFs and DOCX files, ask any question, and get answers routed to the right document using the RLM algorithm.

---

## What It Does

Document Concierge is a web app that lets you build a personal document library and query it in natural language. When you ask a question:

- The **advanced router** reads structured descriptions of every uploaded document and picks the most relevant one(s)
- If the answer spans multiple documents, it runs a **fan-out RLM** across all of them and synthesises a combined answer
- If no document is relevant, it tells you — no hallucinated answers

### Problems This Solves

| # | Limitation of basic multi-doc RLM | How this app fixes it |
|---|---|---|
| 1 | Cross-document queries not supported | Router can select multiple docs → fan-out RLM + aggregation |
| 2 | Routing accuracy depends on description quality | Confidence scoring + structured auto-descriptions (5 fields per doc) |

---

## Architecture

```
Browser  ──────────────────  FastAPI backend (app.py)
                                        │
              ┌─────────────────────────┼──────────────────────┐
              ▼                         ▼                       ▼
          /upload                    /query                  /docs
        • Save file             • advanced_router()         • List docs
        • Extract text          • Confidence score          • (registry)
        • Auto-describe (LM)    • Single-doc RLM  OR
        • Persist to              Fan-out + aggregate
          uploads/registry.json
```

---

## File Map

```
rlm_web/
├── app.py                  # FastAPI backend — routes, upload handling, query orchestration
├── rlm_core.py             # Core RLM algorithm (unchanged from base)
├── document_loader.py      # PDF + DOCX + TXT + MD text extraction
├── advanced_router.py      # Multi-doc routing with confidence scores + rich descriptions
├── cross_doc.py            # Fan-out logic and answer aggregation
├── registry.py             # Persistent document registry (JSON-backed)
├── static/
│   ├── index.html          # Single-page frontend
│   ├── style.css           # UI styles (IBM Plex + warm cream aesthetic)
│   └── app.js              # Upload, document library, and query logic
└── uploads/                # Auto-created on first run; stores files + registry.json
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install fastapi uvicorn python-multipart python-docx \
            openai python-dotenv colorama pypdf
```

> **Note:** `pypdf` is the pure-Python PDF fallback. For better accuracy on table-heavy PDFs (e.g. 10-Ks, annual reports), also install **Poppler** (`pdftotext`) — `document_loader.py` will prefer it automatically if available.

### 2. Create a `.env` file

Inside the `rlm_web/` directory, create a `.env` file:

```env
AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_API_KEY=your-key-here
AZURE_API_VERSION=2024-12-01-preview
ROOT_DEPLOYMENT=gpt-5-mini
SUB_DEPLOYMENT=gpt-5-mini
```

### 3. Run the server

```bash
cd rlm_web
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## How to Use

### Step 1 — Upload Documents

Drag and drop PDF, DOCX, TXT, or MD files onto the dropzone (or click to browse).

Each upload triggers one LM call to generate a structured description. This takes a few seconds but only happens once per file. Documents persist across server restarts.

### Step 2 — Ask a Question

Type your question in the query box. The router will automatically choose the best mode:

| Mode | When it activates | What happens |
|---|---|---|
| `single_doc` | Answer lives in one document | RLM runs on that document |
| `fan_out` | Comparative or multi-source query | RLM runs on each relevant doc, answers are synthesised |
| `none` | No relevant document found | Returns a clear no-match response |

### Step 3 — Read the Response

The response panel shows:
- Which document(s) were selected and why
- Confidence score with a visual bar
- For fan-out queries: per-document findings + a synthesised final answer
- Timing breakdown: route / RLM / total

---

## Configuration

### Confidence Threshold

Located in `app.py` as `CONFIDENCE_THRESHOLD` (default: `0.4`).

- **Raise it** (e.g. `0.6`) → stricter routing; only high-confidence matches proceed
- **Lower it** (e.g. `0.2`) → more permissive; useful when documents have overlapping topics

### Other Limits

| Setting | Default | Where to change |
|---|---|---|
| Max file size | 50 MB | `MAX_FILE_SIZE_MB` in `app.py` |
| Accepted file types | `.pdf`, `.docx`, `.txt`, `.md` | `app.py` |
| Max fan-out documents | 4 | Router prompt in `advanced_router.py` |

> **Scalability note:** All uploaded documents are sent to the router on every query. This works well up to ~20 documents. Beyond that, consider adding an embedding pre-filter (see the RLM_MultiDoc.pdf paper, limitation #3).

---

## How Document Descriptions Work

At upload time, one LM call generates a structured description for each file. The router uses these descriptions — not the full document text — to make routing decisions. This keeps routing fast and accurate.

Each description has five fields:

```
DOC_TYPE:    e.g. "SEC 10-K annual report"
SUBJECT:     Who or what the document is about (named entities)
TIME_PERIOD: Relevant dates or fiscal periods, if any
KEY_TERMS:   5–10 distinctive terms from the document
SUMMARY:     One-sentence overview
```

Better descriptions = better routing. The richer field structure is what allows the router to distinguish between documents on similar topics.

---

## Dependencies Summary

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web server |
| `openai` | Azure OpenAI client |
| `python-docx` | DOCX text extraction |
| `pypdf` | PDF text extraction (pure Python fallback) |
| `python-multipart` | File upload handling |
| `python-dotenv` | `.env` config loading |
| `colorama` | Coloured terminal output |
| `poppler` (optional) | `pdftotext` for high-accuracy PDF extraction |

---

## Project Background

This app is built on top of the RLM (Recursive Language Model) algorithm from Zhang, Kraska & Khattab (2026). The core `rlm_core.py` is unchanged from the base implementation. This project adds a web interface, multi-document routing, fan-out aggregation, and a persistent document registry on top of it.