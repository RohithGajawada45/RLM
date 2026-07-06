// Document Concierge — shared frontend logic for all pages
// Talks to the FastAPI backend at /api/*
//
// This single file is included on every page (index.html, ask.html,
// cache.html, document.html). Each section below checks whether the
// elements it needs exist on the current page before wiring anything up,
// so it's safe to load everywhere.

const $ = (id) => document.getElementById(id);

window.addEventListener('DOMContentLoaded', () => {
  if ($('doc-grid')) initLibraryPage();
  if ($('turn-history')) initAskPage();
  if ($('cache-tbody')) initCachePage();
  if ($('doc-content')) initDocumentPage();
});

// ─── Shared utilities ──────────────────────────────────────────────

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

function formatChars(charCount) {
  if (charCount == null) return '';
  if (charCount < 1000) return `${charCount} chars`;
  if (charCount < 1e6) return `${(charCount / 1000).toFixed(1)}K chars`;
  return `${(charCount / 1e6).toFixed(1)}M chars`;
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return iso;
  }
}

// Pick an icon + accent color based on file extension, matching the
// three looks the Stitch design already established.
function docIconFor(filename) {
  const ext = (filename || '').split('.').pop().toLowerCase();
  if (ext === 'pdf') return { icon: 'picture_as_pdf', color: 'text-secondary' };
  if (ext === 'docx') return { icon: 'description', color: 'text-primary' };
  return { icon: 'code', color: 'text-fanout-seg' }; // txt, md, other
}

async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiDelete(url) {
  const res = await fetch(url, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ─── Library page (index.html) ─────────────────────────────────────

function initLibraryPage() {
  setupDropzone();
  refreshLibrary();
}

async function refreshLibrary() {
  let docs = [];
  try {
    const data = await apiGet('/api/docs');
    docs = data.docs || [];
  } catch (e) {
    console.error('Failed to fetch docs:', e);
  }
  renderLibrary(docs);
}

function renderLibrary(docs) {
  const grid = $('doc-grid');
  const empty = $('empty-state');
  const badge = $('doc-count-badge');

  if (badge) badge.textContent = docs.length;

  if (docs.length === 0) {
    grid.innerHTML = '';
    grid.classList.add('hidden');
    if (empty) empty.classList.remove('hidden');
    return;
  }

  if (empty) empty.classList.add('hidden');
  grid.classList.remove('hidden');

  grid.innerHTML = docs.map(docCardHtml).join('');

  grid.querySelectorAll('[data-action="delete"]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteDoc(btn.dataset.id);
    });
  });
  grid.querySelectorAll('[data-action="analyze"]').forEach((btn) => {
    btn.addEventListener('click', () => {
      window.location.href = `/static/document.html?id=${encodeURIComponent(btn.dataset.id)}`;
    });
  });
}

function docCardHtml(d) {
  const { icon, color } = docIconFor(d.filename);
  return `
    <div class="glass-panel rounded-xl p-[28px] flex flex-col h-full relative group">
      <div class="flex justify-between items-start mb-4">
        <div class="flex items-center gap-2 min-w-0">
          <span class="material-symbols-outlined ${color}" data-icon="${icon}">${icon}</span>
          <h4 class="font-headline-md text-headline-md text-on-surface truncate max-w-[150px]">${escapeHtml(d.title)}</h4>
        </div>
        <button class="text-text-muted hover:text-error transition-colors opacity-0 group-hover:opacity-100 p-1" data-action="delete" data-id="${escapeHtml(d.id)}" aria-label="Delete document">
          <span class="material-symbols-outlined" data-icon="delete">delete</span>
        </button>
      </div>
      <p class="font-body-sm text-body-sm text-text-muted mb-4 flex-grow line-clamp-3 whitespace-pre-line">${escapeHtml(d.description)}</p>
      <div class="mt-auto pt-4 border-t border-glass-border flex justify-between items-center">
        <span class="bg-primary/10 text-primary px-2 py-1 rounded font-data-num text-data-num">${formatChars(d.char_count)}</span>
        <button class="text-primary hover:text-primary-container text-sm font-medium transition-colors" data-action="analyze" data-id="${escapeHtml(d.id)}">Analyze</button>
      </div>
    </div>
  `;
}

async function deleteDoc(id) {
  if (!confirm('Delete this document?')) return;
  try {
    await apiDelete(`/api/docs/${encodeURIComponent(id)}`);
  } catch (e) {
    alert('Delete failed: ' + e.message);
    return;
  }
  refreshLibrary();
}

// ─── Upload ─────────────────────────────────────────────────────────

function setupDropzone() {
  const zone = $('dropzone');
  const input = $('file-input');
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => {
    handleFiles(e.target.files);
    input.value = '';
  });

  ['dragenter', 'dragover'].forEach((evt) => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add('bg-surface-container-low/50');
    });
  });

  ['dragleave', 'drop'].forEach((evt) => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove('bg-surface-container-low/50');
    });
  });

  zone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
}

function handleFiles(fileList) {
  for (const file of fileList) uploadOne(file);
}

function uploadRowHtml(id, filename) {
  return `
    <div class="glass-panel p-4 rounded-xl flex items-center justify-between" id="upload-row-${id}">
      <div class="flex items-center gap-4 min-w-0">
        <div class="w-8 h-8 rounded-full bg-surface-container-high flex items-center justify-center ripple-active shrink-0">
          <span class="material-symbols-outlined text-primary text-sm" data-icon="sync">sync</span>
        </div>
        <div class="flex flex-col min-w-0">
          <span class="font-data-num text-data-num text-on-surface upload-status">Reading &amp; describing…</span>
          <span class="font-micro-label text-micro-label text-text-muted truncate">${escapeHtml(filename)}</span>
        </div>
      </div>
    </div>
  `;
}

async function uploadOne(file) {
  const queue = $('upload-queue');
  if (!queue) return;

  const rowId = 'u' + Math.random().toString(36).slice(2, 9);
  queue.insertAdjacentHTML('beforeend', uploadRowHtml(rowId, file.name));
  const row = $(`upload-row-${rowId}`);
  const statusEl = row.querySelector('.upload-status');
  const iconEl = row.querySelector('[data-icon="sync"]');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    iconEl.textContent = 'check_circle';
    iconEl.dataset.icon = 'check_circle';
    iconEl.classList.remove('text-primary');
    iconEl.classList.add('text-[#2e7d32]');
    row.querySelector('.ripple-active').classList.remove('ripple-active');
    statusEl.textContent = `Described in ${data.elapsed}s`;
    setTimeout(() => row.remove(), 4000);
    refreshLibrary();
  } catch (e) {
    iconEl.textContent = 'error';
    iconEl.dataset.icon = 'error';
    iconEl.classList.remove('text-primary');
    iconEl.classList.add('text-error');
    row.querySelector('.ripple-active').classList.remove('ripple-active');
    statusEl.textContent = e.message;
  }
}

// ─── Ask page (ask.html) ────────────────────────────────────────────
//
// The backend has no server-side conversation memory — each /api/query
// call is independent. We keep a client-side turn history for this
// browser session only (it resets on reload) so the page can show a
// running conversation the way the design calls for.

let _turns = [];       // { id, query, time, data }
let _expandedTurnId = null;

function initAskPage() {
  $('ask-btn').addEventListener('click', runQuery);
  $('query-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      runQuery();
    }
  });
}

async function runQuery() {
  const input = $('query-input');
  const query = input.value.trim();
  if (!query) return;

  const btn = $('ask-btn');
  const status = $('status-bar');
  const statusText = $('status-text');

  btn.disabled = true;
  status.classList.remove('hidden');
  status.classList.add('flex');
  statusText.textContent = 'Routing…';
  statusText.className = 'font-data-num text-data-num text-primary';

  const tickStart = Date.now();
  const ticker = setInterval(() => {
    const elapsed = ((Date.now() - tickStart) / 1000).toFixed(1);
    const phase = (Date.now() - tickStart) > 8000 ? 'Reading documents' : 'Routing';
    statusText.textContent = `${phase}… (${elapsed}s)`;
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
    status.classList.add('hidden');
    status.classList.remove('flex');

    const id = 't' + Date.now();
    _turns.push({ id, query, time: new Date(), data });
    _expandedTurnId = id;
    input.value = '';
    renderTurns();

    const el = $(`turn-${id}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    clearInterval(ticker);
    statusText.textContent = `Query failed: ${e.message}`;
    statusText.className = 'font-data-num text-data-num text-error';
    setTimeout(() => {
      status.classList.add('hidden');
      status.classList.remove('flex');
    }, 4000);
  } finally {
    btn.disabled = false;
  }
}

function renderTurns() {
  const container = $('turn-history');
  const empty = $('turn-history-empty');

  if (_turns.length === 0) {
    if (empty) empty.classList.remove('hidden');
    return;
  }
  if (empty) empty.classList.add('hidden');

  container.querySelectorAll('[data-turn]').forEach((n) => n.remove());

  for (const turn of _turns) {
    const isExpanded = turn.id === _expandedTurnId;
    const html = isExpanded ? turnExpandedHtml(turn) : turnCollapsedHtml(turn);
    container.insertAdjacentHTML('beforeend', html);
  }

  container.querySelectorAll('[data-action="expand-turn"]').forEach((node) => {
    node.addEventListener('click', () => {
      _expandedTurnId = node.dataset.id;
      renderTurns();
      const el = $(`turn-${node.dataset.id}`);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

function turnCollapsedHtml(turn) {
  return `
    <div class="glass-panel rounded-xl p-5 opacity-80 cursor-pointer" data-turn data-action="expand-turn" data-id="${turn.id}" id="turn-${turn.id}">
      <div class="flex justify-between items-center gap-4">
        <p class="font-body-lg text-body-lg text-text-muted truncate">${escapeHtml(turn.query)}</p>
        <span class="font-data-num text-data-num text-text-muted shrink-0">${formatTime(turn.time)}</span>
      </div>
    </div>
  `;
}

const MODE_LABELS = {
  single_doc: 'Single-doc',
  single_doc_cached: 'Single-doc · cached',
  fan_out: 'Cross-doc fan-out',
  fan_out_cached: 'Fan-out · cached',
  no_match: 'No matching document',
  low_confidence: 'Confidence below threshold',
  no_docs: 'No documents registered',
};

function turnExpandedHtml(turn) {
  const data = turn.data;
  const routing = data.routing || {};
  const tokens = data.tokens || {};
  const timings = data.timings || {};
  const perDoc = data.per_doc || [];

  const modeLabel = MODE_LABELS[routing.mode] || routing.mode || '—';
  const confPct = Math.round((routing.confidence ?? 0) * 100);
  const titles = routing.titles || routing.doc_ids || [];
  const docsChip = titles.length
    ? `<span class="bg-tertiary-container/10 text-tertiary font-data-num text-data-num px-2 py-1 rounded border border-tertiary-container/20">Used ${titles.length} doc${titles.length !== 1 ? 's' : ''}: ${escapeHtml(titles.map(shortenTitle).join(', '))}</span>`
    : `<span class="bg-error-container/20 text-error font-data-num text-data-num px-2 py-1 rounded border border-error/20">No documents matched</span>`;
  const cacheChip = tokens.cached
    ? `<span class="bg-primary/10 text-primary font-data-num text-data-num px-2 py-1 rounded border border-primary/20">Cached · 0 new tokens</span>`
    : '';

  const cleanAnswer = stripFooter(data.answer || '');
  const answerHtml = renderMarkdown(cleanAnswer || '_(no answer)_');

  const perDocHtml = (routing.mode === 'fan_out' && perDoc.length > 1)
    ? perDocFindingsHtml(perDoc, tokens.by_stage || {})
    : '';

  const tokenBar = tokenBarHtml(tokens);

  return `
    <div class="glass-panel rounded-xl p-7 flex flex-col gap-6" data-turn id="turn-${turn.id}">
      <div class="flex gap-4 items-start border-b border-glass-border pb-4">
        <div class="w-8 h-8 rounded-full bg-surface-variant flex-shrink-0 flex items-center justify-center border border-glass-border">
          <span class="material-symbols-outlined text-text-muted text-sm" data-icon="person">person</span>
        </div>
        <p class="font-headline-md text-headline-md text-on-surface pt-1">${escapeHtml(turn.query)}</p>
      </div>
      <div class="flex gap-4 items-start">
        <div class="w-8 h-8 rounded-full bg-primary-container flex-shrink-0 flex items-center justify-center border border-glass-border shadow-sm shadow-accent-glow text-on-primary-container">
          <span class="material-symbols-outlined text-sm" data-icon="smart_toy">smart_toy</span>
        </div>
        <div class="flex-1 font-body-lg text-body-lg text-on-surface space-y-4 prose-turn">${answerHtml}</div>
      </div>
      <details class="group mt-4 border border-glass-border rounded-lg bg-surface-container-lowest/50" open="">
        <summary class="p-4 cursor-pointer font-data-num text-data-num text-text-muted hover:text-primary transition-colors flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-[16px]" data-icon="analytics">analytics</span>
            Diagnostic Trace
          </div>
          <span class="material-symbols-outlined transition-transform group-open:rotate-180" data-icon="expand_more">expand_more</span>
        </summary>
        <div class="p-5 border-t border-glass-border flex flex-col gap-6">
          <div class="flex flex-col gap-3">
            <div class="flex flex-wrap gap-2 items-center">
              <span class="bg-primary/10 text-primary font-data-num text-data-num px-2 py-1 rounded border border-primary/20">${escapeHtml(modeLabel)}</span>
              <span class="bg-secondary-container/10 text-secondary font-data-num text-data-num px-2 py-1 rounded border border-secondary-container/20">${confPct}% confidence</span>
              ${docsChip}
              ${cacheChip}
            </div>
            <p class="font-body-sm text-body-sm text-text-muted bg-cream p-3 rounded-md border border-glass-border">
              Rationale: ${escapeHtml(routing.reason || '—')}
            </p>
          </div>
          ${perDocHtml}
          <div class="flex flex-col gap-2">
            <div class="flex justify-between items-end mb-1">
              <h4 class="font-label-mono-bold text-label-mono-bold text-text-muted uppercase">Token Budget Overview</h4>
              <span class="font-data-num text-data-num text-on-surface">${tokenBar.totalText}</span>
            </div>
            <div class="h-3 w-full bg-surface-variant rounded-full overflow-hidden flex shadow-inner">${tokenBar.segmentsHtml}</div>
            <div class="flex justify-between font-micro-label text-micro-label text-text-muted mt-1 uppercase tracking-wider flex-wrap gap-2">
              <span>route ${timings.route ?? 0}s · rlm ${timings.rlm ?? 0}s · total ${timings.total ?? 0}s</span>
              <div class="flex gap-3 flex-wrap">${tokenBar.legendHtml}</div>
            </div>
          </div>
        </div>
      </details>
    </div>
  `;
}

function perDocFindingsHtml(perDoc, byStage) {
  const items = perDoc.map((p) => {
    const slug = slugify(p.title);
    const stage = byStage[`fanout:${slug}`] || {};
    const stageTokens = (stage.prompt || 0) + (stage.completion || 0);
    const stageCalls = stage.calls || 0;
    return `
      <details class="group/pd border border-glass-border rounded-lg bg-surface-container-low">
        <summary class="p-3 cursor-pointer font-data-num text-data-num text-on-surface flex items-center justify-between gap-2">
          <span class="flex items-center gap-2 min-w-0">
            <span class="material-symbols-outlined text-fanout-seg text-[16px]" data-icon="data_object">data_object</span>
            <span class="truncate">${escapeHtml(p.title)}</span>
          </span>
          <span class="font-micro-label text-micro-label text-text-muted shrink-0">
            ${stageCalls ? `${stageCalls} iter · ` : ''}${stageTokens ? formatNum(stageTokens) + ' tok' : ''}
          </span>
        </summary>
        <div class="p-3 border-t border-glass-border font-body-sm text-body-sm text-on-surface-variant prose-turn">${renderMarkdown(p.sub_answer || '')}</div>
      </details>
    `;
  }).join('');

  return `
    <div>
      <h4 class="font-label-mono-bold text-label-mono-bold text-text-muted mb-2 uppercase">Per-document Findings (${perDoc.length})</h4>
      <div class="space-y-2">${items}</div>
    </div>
  `;
}

// Colors cycle to match the design's router/retrieval/aggregate palette.
function stageColor(key) {
  if (key === 'router') return 'bg-router-seg';
  if (key === 'aggregator') return 'bg-agg-seg';
  return 'bg-fanout-seg';
}

function stageLabel(key) {
  if (key === 'router') return 'Router';
  if (key === 'aggregator') return 'Aggregate';
  if (key.includes(':')) {
    const [, b] = key.split(':');
    return `Retrieval · ${b}`;
  }
  return key;
}

function tokenBarHtml(tokens) {
  const byStage = tokens.by_stage || {};
  const total = tokens.total || 0;

  if (total === 0 || Object.keys(byStage).length === 0) {
    return {
      totalText: '--',
      segmentsHtml: `<div class="h-full w-full bg-surface-container-high"></div>`,
      legendHtml: '',
    };
  }

  const stages = Object.entries(byStage).map(([k, v]) => ({
    key: k,
    label: stageLabel(k),
    color: stageColor(k),
    total: (v.prompt || 0) + (v.completion || 0),
    calls: v.calls || 0,
  })).filter((s) => s.total > 0);

  const seenLabels = new Set();
  const segmentsHtml = stages.map((s) => {
    const pct = ((s.total / total) * 100).toFixed(1);
    return `<div class="h-full ${s.color} token-bar-fill" style="width: ${pct}%;" title="${escapeHtml(s.label)}: ${formatNum(s.total)} tok · ${s.calls} call${s.calls !== 1 ? 's' : ''}"></div>`;
  }).join('');

  const legendHtml = stages.filter((s) => {
    if (seenLabels.has(s.color)) return false;
    seenLabels.add(s.color);
    return true;
  }).map((s) => `<span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full ${s.color}"></span>${escapeHtml(s.label.split(' · ')[0])}</span>`).join('');

  return {
    totalText: `${formatNum(total)} total tokens`,
    segmentsHtml,
    legendHtml,
  };
}

function formatTime(d) {
  try {
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

function formatNum(n) {
  if (n == null) return '0';
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)}K`;
  return `${(n / 1e6).toFixed(2)}M`;
}

function shortenTitle(t) {
  if (!t) return '';
  return t.replace(/\.(pdf|docx|txt|md)$/i, '');
}

// Match the backend's slugify in cross_doc.py so we can look up stage stats.
function slugify(title) {
  if (!title) return '';
  const base = title.replace(/\.[^.]+$/, '');
  const m = base.match(/^[A-Za-z]+/);
  return (m ? m[0] : base.slice(0, 8)).toLowerCase().slice(0, 12);
}

// The backend appends a "**Token usage** — ..." footer to data.answer.
// We render token info separately (structured), so strip it before markdown.
function stripFooter(answer) {
  if (!answer) return '';
  const idx = answer.indexOf('\n\n---\n\n**Token usage**');
  if (idx !== -1) return answer.slice(0, idx).trim();
  const idx2 = answer.indexOf('---\n\n**Token usage**');
  if (idx2 !== -1) return answer.slice(0, idx2).trim();
  return answer.trim();
}

function renderMarkdown(text) {
  if (!text) return '';
  if (typeof window.marked === 'undefined' || !window.marked.parse) {
    return `<p>${escapeHtml(text).replace(/\n\n+/g, '</p><p>').replace(/\n/g, '<br>')}</p>`;
  }
  try {
    return window.marked.parse(text, { breaks: true, gfm: true });
  } catch (e) {
    return `<p>${escapeHtml(text)}</p>`;
  }
}

// ─── Cache page (cache.html) ────────────────────────────────────────

function initCachePage() {
  $('clear-cache-btn').addEventListener('click', clearAllCache);
  refreshCache();
}

async function refreshCache() {
  let stats;
  try {
    stats = await apiGet('/api/cache');
  } catch (e) {
    console.error('Failed to fetch cache stats:', e);
    stats = { size: 0, subanswer_size: 0, entries: [] };
  }
  renderCache(stats);
}

function renderCache(stats) {
  const entries = stats.entries || [];
  const totalSaved = entries.reduce((sum, e) => sum + (e.tokens_saved || 0), 0);

  $('cache-total-badge').textContent = `${stats.size ?? entries.length} TOTAL ENTRIES`;
  $('cache-tokens-badge').textContent = `${formatNum(totalSaved)} TOKENS SAVED`;
  $('cache-subanswer-badge').textContent = `${stats.subanswer_size ?? 0} SUB-ANSWERS CACHED`;
  $('cache-count-text').textContent = `Showing ${entries.length} ${entries.length === 1 ? 'entry' : 'entries'}`;

  const tbody = $('cache-tbody');
  const empty = $('cache-empty-state');
  const table = tbody.closest('table');

  if (entries.length === 0) {
    tbody.innerHTML = '';
    table.classList.add('hidden');
    empty.classList.remove('hidden');
    empty.classList.add('flex');
    return;
  }
  table.classList.remove('hidden');
  empty.classList.add('hidden');
  empty.classList.remove('flex');

  tbody.innerHTML = entries.map(cacheRowHtml).join('');

  tbody.querySelectorAll('[data-action="delete-cache"]').forEach((btn) => {
    btn.addEventListener('click', () => deleteCacheEntry(btn.dataset.key));
  });
}

const MODE_CHIP = {
  fan_out: { label: 'FAN-OUT', bg: 'bg-[#06B6D4]/10', text: 'text-fanout-seg', border: 'border-[#06B6D4]/20', dot: 'bg-fanout-seg' },
  single_doc: { label: 'SINGLE', bg: 'bg-primary/10', text: 'text-primary', border: 'border-primary/20', dot: 'bg-primary' },
};

function modeChipHtml(mode) {
  const chip = MODE_CHIP[mode] || { label: (mode || '—').toUpperCase(), bg: 'bg-surface-container-high', text: 'text-on-surface-variant', border: 'border-outline-variant', dot: 'bg-outline' };
  return `
    <span class="inline-flex items-center gap-1.5 px-2 py-1 rounded ${chip.bg} ${chip.text} border ${chip.border} font-micro-label text-micro-label">
      <span class="w-1.5 h-1.5 rounded-full ${chip.dot}"></span>
      ${escapeHtml(chip.label)}
    </span>
  `;
}

function cacheRowHtml(entry) {
  const docCount = (entry.doc_titles && entry.doc_titles.length) || (entry.doc_ids && entry.doc_ids.length) || 0;
  const titleTooltip = (entry.doc_titles || []).join(', ');
  // key comes back as "abcd1234…" (12 chars + ellipsis); strip the ellipsis to use as a delete prefix.
  const keyPrefix = (entry.key || '').replace(/[…\.]+$/, '');

  return `
    <tr class="hover:bg-surface/50 transition-colors duration-200 group">
      <td class="py-4 px-6">
        <p class="font-body-sm text-body-sm text-on-surface line-clamp-2">${escapeHtml(entry.query || '')}</p>
      </td>
      <td class="py-4 px-4 font-data-num text-data-num text-on-surface-variant" title="${escapeHtml(titleTooltip)}">
        ${docCount}
      </td>
      <td class="py-4 px-4">
        ${modeChipHtml(entry.mode)}
      </td>
      <td class="py-4 px-4 font-data-num text-data-num text-on-surface-variant text-right">
        +${formatNum(entry.tokens_saved || 0)} <span class="text-text-muted/60 text-[10px]">tk</span>
      </td>
      <td class="py-4 px-4 font-data-num text-data-num text-text-muted">
        ${relativeTime(entry.cached_at)}
      </td>
      <td class="py-4 px-4 text-center">
        <button class="text-text-muted hover:text-error transition-colors opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-error-container/30" data-action="delete-cache" data-key="${escapeHtml(keyPrefix)}" aria-label="Delete cache entry">
          <span class="material-symbols-outlined text-[18px]">delete</span>
        </button>
      </td>
    </tr>
  `;
}

async function deleteCacheEntry(keyPrefix) {
  try {
    await apiDelete(`/api/cache/${encodeURIComponent(keyPrefix)}`);
  } catch (e) {
    alert('Delete failed: ' + e.message);
    return;
  }
  refreshCache();
}

async function clearAllCache() {
  if (!confirm('Clear the entire cache? This removes all cached answers and sub-answers.')) return;
  try {
    await apiDelete('/api/cache');
  } catch (e) {
    alert('Clear failed: ' + e.message);
    return;
  }
  refreshCache();
}

function relativeTime(epochSeconds) {
  if (!epochSeconds) return '—';
  const diffMs = Date.now() - epochSeconds * 1000;
  const diffSec = Math.max(0, Math.floor(diffMs / 1000));
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

// ─── Document detail page (document.html?id=...) ───────────────────

async function initDocumentPage() {
  const params = new URLSearchParams(window.location.search);
  const id = params.get('id');

  if (!id) {
    showDocNotFound();
    return;
  }

  let docs = [];
  try {
    const data = await apiGet('/api/docs');
    docs = data.docs || [];
  } catch (e) {
    console.error('Failed to fetch docs:', e);
  }

  const doc = docs.find((d) => d.id === id);
  if (!doc) {
    showDocNotFound();
    return;
  }

  renderDocument(doc);
  wireDeleteFlow(doc.id);
}

function showDocNotFound() {
  $('doc-content').classList.add('hidden');
  const nf = $('doc-not-found');
  nf.classList.remove('hidden');
  nf.classList.add('flex');
}

function renderDocument(d) {
  document.title = `Document Concierge - ${d.title}`;

  const { icon } = docIconFor(d.filename);
  $('doc-icon').textContent = icon;
  $('doc-icon').dataset.weight = 'fill';

  $('doc-title').textContent = d.title;
  $('doc-filename').textContent = d.filename;

  const ext = (d.filename || '').split('.').pop().toUpperCase();
  $('doc-type-badge').textContent = ext || '—';
  $('doc-charcount-badge').textContent = formatChars(d.char_count);
  $('doc-uploaded-badge').textContent = `Uploaded ${formatDate(d.uploaded_at)}`;

  $('doc-description').textContent = d.description || '(no description)';

  $('doc-uploaded-at').textContent = formatDateTime(d.uploaded_at);
  $('doc-char-count').textContent = d.char_count != null ? d.char_count.toLocaleString() : '—';
  $('doc-id-value').textContent = d.id;
}

function wireDeleteFlow(id) {
  const btnDelete = $('btn-delete');
  const deleteWarning = $('delete-warning');
  const btnCancel = $('btn-cancel');
  const btnConfirm = $('btn-confirm-delete');

  btnDelete.addEventListener('click', () => {
    btnDelete.classList.add('hidden');
    deleteWarning.classList.remove('hidden');
    deleteWarning.classList.add('flex');
  });

  btnCancel.addEventListener('click', () => {
    deleteWarning.classList.add('hidden');
    deleteWarning.classList.remove('flex');
    btnDelete.classList.remove('hidden');
  });

  btnConfirm.addEventListener('click', async () => {
    btnConfirm.disabled = true;
    btnConfirm.textContent = 'Deleting…';
    try {
      await apiDelete(`/api/docs/${encodeURIComponent(id)}`);
      window.location.href = '/static/index.html';
    } catch (e) {
      alert('Delete failed: ' + e.message);
      btnConfirm.disabled = false;
      btnConfirm.textContent = 'Confirm';
    }
  });
}

function formatDateTime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return iso;
  }
}