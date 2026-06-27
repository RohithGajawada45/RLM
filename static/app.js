// Document Concierge — frontend logic
// Talks to the FastAPI backend at /api/*

const $ = (id) => document.getElementById(id);

let docs = [];

// —— On page load ————————————————————————————————————————————————————————————
window.addEventListener('DOMContentLoaded', async () => {
  setupDropzone();
  setupQueryForm();
  await refreshLibrary();
});

// —— Library ————————————————————————————————————————————————————————————————
async function refreshLibrary() {
  try {
    const res = await fetch('/api/docs');
    const data = await res.json();
    docs = data.docs || [];
  } catch (e) {
    console.error('Failed to fetch docs:', e);
    docs = [];
  }
  renderLibrary();
}

function renderLibrary() {
  const lib = $('library');
  $('meta-doc-count').textContent = docs.length;
  lib.innerHTML = '';
  if (docs.length === 0) {
    lib.innerHTML = '<div class="empty-state">No documents yet. Drop one above to start.</div>';
    return;
  }
  for (const d of docs) {
    const card = document.createElement('div');
    card.className = 'doc-card';
    card.innerHTML = `
      <div class="doc-card-top">
        <div class="doc-meta-line">
          <span class="doc-id">${escapeHtml(d.id)}</span>
          <span class="doc-dot">·</span>
          <span class="doc-size">${formatBytes(d.char_count)}</span>
          <span class="doc-dot">·</span>
          <span class="doc-date">${formatDate(d.uploaded_at)}</span>
        </div>
        <div class="doc-title">${escapeHtml(d.title)}</div>
    </div>
    <button class="doc-toggle" data-action="expand" aria-expanded="false">
      <span class="doc-toggle-text">Description</span>
      <svg viewBox="0 0 12 12" width="10" height="10" fill="currentColor" aria-hidden="true" class="doc-chevron">
        <path d="M3 4.5l3 3 3-3"/>
      </svg>
    </button>
    <div class="doc-desc">${escapeHtml(d.description)}</div>
    <button class="doc-delete" data-action="delete" aria-label="Delete document">
      <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true">
        <path d="M3 4h10M6 4V2.5h4V4M5 4l.6 9h4.8L11 4M7 7v4M9 7v4"/>
      </svg>
    </button>
  `;
    card.querySelector('[data-action="expand"]').addEventListener('click', () => {
      const isExp = card.classList.toggle('expanded');
      card.querySelector('[data-action="expand"]').setAttribute('aria-expanded', String(isExp));
    });
    card.querySelector('[data-action="delete"]').addEventListener('click', () => deleteDoc(d.id));
    lib.appendChild(card);
  }
}

async function deleteDoc(id) {
  if (!confirm('Delete this document?')) return;
  try {
    await fetch(`/api/docs/${id}`, { method: 'DELETE' });
  } catch (e) {
    alert('Delete failed: ' + e.message);
    return;
  }
  await refreshLibrary();
}

// ─── Upload ───────────────────────────────────────────────────────

function setupDropzone() {
  const zone = $('dropzone');
  const input = $('file-input');

  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => {
    handleFiles(e.target.files);
    input.value = '';
  });

  ['dragenter', 'dragover'].forEach((evt) => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach((evt) => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove('drag-over');
    });
  });

  zone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
}

function handleFiles(fileList) {
  for (const file of fileList) uploadOne(file);
}

async function uploadOne(file) {
  const queue = $('upload-queue');
  const row = document.createElement('div');
  row.className = 'upload-row';
  row.innerHTML = `
    <span class="spinner" aria-hidden="true"></span>
    <span class="upload-name">${escapeHtml(file.name)}</span>
    <span class="upload-status">reading & describing…</span>
  `;
  queue.appendChild(row);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    row.classList.add('success');
    row.innerHTML = `
      <span class="upload-check" aria-hidden="true">
        <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><path d="M6.5 11.5L-3-3 1-1 2 2 4-4 1 1z"/></svg>
            </span>
      <span class="upload-name">${escapeHtml(file.name)}</span>
      <span class="upload-status">described in ${data.elapsed}s</span>
    `;
    setTimeout(() => row.remove(), 4000);
    await refreshLibrary();
  } catch (e) {
    row.classList.add('error');
    row.innerHTML = `
      <span class="upload-x" aria-hidden="true">
        <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none"/></svg>
      </span>
      <span class="upload-name">${escapeHtml(file.name)}</span>
      <span class="upload-status">${escapeHtml(e.message)}</span>
    `;
  }
}

// ─── Query ───────────────────────────────────────────────────────

function setupQueryForm() {
  $('ask-btn').addEventListener('click', runQuery);
  $('query-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      runQuery();
    }
  });
}

async function runQuery() {
  const query = $('query-input').value.trim();
  if (!query) return;

  const btn = $('ask-btn');
  const status = $('status-bar');

  btn.disabled = true;
  status.hidden = false;
  status.className = 'status-strip active';
  status.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span class="status-text">routing…</span>`;

  // Hide previous response
  $('response-card').hidden = true;

  const tickStart = Date.now();
  const ticker = setInterval(() => {
    const elapsed = ((Date.now() - tickStart) / 1000).toFixed(1);

    // After 8s, swap phase label to suggest sub-LMs are running
    const phase = (Date.now() - tickStart) > 8000 ? 'reading documents' : 'routing';

    status.querySelector('.status-text').textContent = `${phase} · ${elapsed}s`;
  }, 200);

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });

    clearInterval(ticker);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    status.hidden = true;
    renderResponse(data);
  } catch (e) {
    clearInterval(ticker);
    status.className = 'status-strip error';
    status.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span class="status-text">Query failed: ${escapeHtml(e.message)}</span>`;
  } finally {
    btn.disabled = false;
  }
}

// ─── Render the response ─────────────────────────────────────────

function renderResponse(data) {
  const card = $('response-card');
  card.hidden = false;

  const routing = data.routing || {};
  const tokens = data.tokens || {};
  const timings = data.timings || {};
  const perDoc = data.per_doc || [];

  // Mode label
  const modeLabel = {
    'single_doc': 'Single-doc',
    'single_doc_cached': 'Single-doc · cached',
    'fan_out': 'Cross-doc fan-out',
    'fan_out_cached': 'Fan-out · cached',
    'no_match': 'No matching document',
    'low_confidence': 'Confidence below threshold',
    'no_docs': 'No documents registered',
  }[routing.mode] || routing.mode || '—';

  $('mode-value').textContent = modeLabel;

  // Documents
  const titles = routing.titles || routing.doc_ids || [];
  const docsEl = $('docs-value');

  if (titles.length === 0) {
    docsEl.innerHTML = `<span class="pill pill-none">NONE</span>`;
  } else {
    docsEl.innerHTML = titles
      .map(t => `<span class="pill">${escapeHtml(shortenTitle(t))}</span>`)
      .join('');
  }

  // Confidence
  const conf = routing.confidence ?? 0;
  const confPct = Math.round(conf * 100);

  $('conf-num').textContent = `${confPct}%`;

  const fill = $('conf-fill');
  fill.style.width = `${confPct}%`;
  fill.className =
    'conf-fill ' +
    (conf >= 0.7
      ? 'conf-high'
      : conf >= 0.4
      ? 'conf-med'
      : 'conf-low');

  // Iteration count
  const byStage = tokens.by_stage || {};

  const totalCalls = Object.values(byStage)
    .reduce((s, x) => s + (x.calls || 0), 0);

  const subjectCalls = Object.entries(byStage)
    .filter(([k]) => k !== 'router' && k !== 'aggregator')
    .reduce((s, [, x]) => s + (x.calls || 0), 0);

  $('iter-count').textContent = `${subjectCalls} · ${totalCalls}`;

  // Token bar
  renderTokenBar(tokens);

  // Cache badge
  $('cache-badge').hidden = !tokens.cached;

  // Reasoning details
  $('reason-text').textContent = routing.reason || '';

  // Per-doc findings (collapsible)
  const perDocPanel = $('per-doc-panel');
  const perDocList = $('per-doc-list');
  perDocList.innerHTML = '';

  if (routing.mode === 'fan_out' && perDoc.length > 1) {
    perDocPanel.hidden = false;
    $('perdoc-count').textContent = perDoc.length;

    perDoc.forEach((p, i) => {
      // try to find this doc's stage info from byStage
      const slug = slugify(p.title);
      const stageKey = `fanout:${slug}`;
      const stage = byStage[stageKey] || {};
      const stageTokens = (stage.prompt || 0) + (stage.completion || 0);
      const stageCalls = stage.calls || 0;

      const item = document.createElement('details');
      item.className = 'perdoc-item';
      if (i === 0) item.open = false; // start collapsed

      item.innerHTML = `
        <summary class="perdoc-item-summary">
          <span class="perdoc-item-title">${escapeHtml(p.title)}</span>
          <span class="perdoc-item-meta">
            ${stageCalls ? `<span class="perdoc-stat">${stageCalls} iter</span>` : ''}
            ${stageTokens ? `<span class="perdoc-stat">${formatNum(stageTokens)} tok</span>` : ''}
          </span>
        </summary>
        <div class="prose perdoc-item-body">${renderMarkdown(p.sub_answer || '')}</div>
      `;

      perDocList.appendChild(item);
    });

  } else {
    perDocPanel.hidden = true;
  }

  // The answer (stripped of footer, markdown-rendered)
  const cleanAnswer = stripFooter(data.answer || '');
  $('answer-text').innerHTML = renderMarkdown(cleanAnswer || '_(no answer)_');

  // Timings
  $('timings').textContent =
    `route ${timings.route ?? 0}s · rlm ${timings.rlm ?? 0}s · total ${timings.total ?? 0}s`;

  // Scroll into view
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Token-usage bar ───────────────────────────────────────────────────────────

function renderTokenBar(tokens) {
  const bar = $('token-bar');
  const legend = $('token-bar-legend');
  const totalEl = $('token-total');

  bar.innerHTML = '';
  legend.innerHTML = '';

  const byStage = tokens.by_stage || {};
  const total = tokens.total || 0;

  totalEl.textContent = total ? `${formatNum(total)} tokens` : '--';

  if (total === 0 || Object.keys(byStage).length === 0) {
    bar.innerHTML = '<div class="token-bar-empty">No token data</div>';
    return;
  }

  // Compute per-stage totals and sort to put router/aggregator at edges,
    // fanouts in the middle. (preserve insertion order otherwise)
    const stages = Object.entries(byStage).map(([k, v]) => ({
    key: k,
    label: stageLabel(k),
    total: (v.prompt || 0) + (v.completion || 0),
    calls: v.calls || 0,
    }));

    // Find the most expensive stage to emphasize
    const maxStage = stages.reduce((a, b) => (a.total > b.total ? a : b), stages[0]);

    for (const s of stages) {
    if (s.total === 0) continue;

    const pct = (s.total / total) * 100;
    const isMax = s.key === maxStage.key;

    const seg = document.createElement('div');
    seg.className = 'token-seg' + (isMax ? ' token-seg-max' : '');
    seg.style.flex = `${pct} 0 0`;
    seg.title = `${s.label}: ${formatNum(s.total)} tok · ${s.calls} call${s.calls !== 1 ? 's' : ''} (${pct.toFixed(1)}%)`;
    bar.appendChild(seg);

    const item = document.createElement('div');
    item.className = 'legend-item' + (isMax ? ' legend-item-max' : '');
    item.innerHTML = `
        <span class="legend-swatch${isMax ? ' legend-swatch-max' : ''}"></span>
        <span class="legend-name">${escapeHtml(s.label)}</span>
        <span class="legend-val">${formatNum(s.total)}</span>
        <span class="legend-pct">${pct.toFixed(0)}%</span>
    `;
    legend.appendChild(item);
    }
}

function stageLabel(key) {
  // pretty-print stage names: "fanout:tcs" -> "fanout · tcs"
  if (key.includes(':')) {
    const [a, b] = key.split(':');
    return `${a} · ${b}`;
  }
  return key;
}

// —— Footer stripping ——
// The backend appends a "**Token usage** — ..." footer to data.answer.
// We already render that info structured (token bar), so strip it before
// the markdown renderer. Also strip the cache postscript.
function stripFooter(answer) {
  if (!answer) return '';

  // Find the footer divider
  const idx = answer.indexOf('\n\n---\n\n**Token usage**');
  if (idx !== -1) return answer.slice(0, idx).trim();

  // Sometimes the footer arrives without leading newlines
  const idx2 = answer.indexOf('---\n\n**Token usage**');
  if (idx2 !== -1) return answer.slice(0, idx2).trim();

  return answer.trim();
}

// —— Markdown rendering ——
function renderMarkdown(text) {
  if (!text) return '';

  if (typeof window.marked === 'undefined' || !window.marked.parse) {
    // marked.js hasn't loaded yet (defer + race). Fall back to escaped text
    // with line breaks preserved.
    return `<p>${escapeHtml(text)
      .replace(/\n\n+/g, '</p><p>')
      .replace(/\n/g, '<br>')}</p>`;
  }

  try {
    return window.marked.parse(text, {
      breaks: true,   // newline -> <br>
      gfm: true,      // tables, strikethrough, etc.
    });
  } catch (e) {
    return `<p>${escapeHtml(text)}</p>`;
  }
}

// —— Utilities ——
function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric'
    });
  } catch {
    return iso;
  }
}

function formatBytes(charCount) {
  // We get char_count from the API; rough byte estimate
  if (charCount == null) return '';
  if (charCount < 1000) return `${charCount} ch`;
  if (charCount < 1e6) return `${(charCount / 1000).toFixed(0)}K ch`;
  return `${(charCount / 1e6).toFixed(1)}M ch`;
}

function formatNum(n) {
  if (n == null) return '0';
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)}K`;
  return `${(n / 1e6).toFixed(2)}M`;
}

function shortenTitle(t) {
  if (!t) return '';

  // Keep filename, drop extension for badges
  return t.replace(/\.(pdf|docx|txt|md)$/i, '');
}

// Match the backend's slugify in cross_doc.py so we can look up stage stats
function slugify(title) {
  if (!title) return '';

  const base = title.replace(/\.[^.]+$/, '');
  const m = base.match(/^[A-Za-z]+/);

  return (m ? m[0] : base.slice(0, 8))
    .toLowerCase()
    .slice(0, 12);
}