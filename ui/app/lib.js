// Tiny DOM + formatting helpers shared across views. No framework, no deps.

/** Create an element from an HTML string (first root node). */
export function html(strings, ...values) {
  const tpl = document.createElement('template');
  tpl.innerHTML = String.raw({ raw: strings }, ...values).trim();
  return tpl.content.firstElementChild;
}

/** querySelector shorthand scoped to root (default: document). */
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Fetch JSON with a short timeout; returns null on any failure. */
export async function fetchJSON(url, opts = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), opts.timeout ?? 6000);
  try {
    const res = await fetch(url, { signal: ctrl.signal, ...opts });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  } finally {
    clearTimeout(t);
  }
}

export const fmt = {
  pct: (v, d = 0) => (v == null || isNaN(v) ? '—' : `${(v * (v <= 1 ? 100 : 1)).toFixed(d)}%`),
  num: (v, d = 2) => (v == null || isNaN(v) ? '—' : Number(v).toFixed(d)),
  int: (v) => (v == null || isNaN(v) ? '—' : Math.round(v).toLocaleString()),
  dbm: (v) => (v == null || isNaN(v) ? '—' : `${Number(v).toFixed(1)} dBm`),
  ago: (sec) => {
    if (sec == null) return '—';
    const s = Math.max(0, Math.floor(sec));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
  },
};

/** Set a meter bar fill (0..1) + colour by threshold. */
export function setMeter(spanEl, value, { warn = 0.7, bad = 0.9 } = {}) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  spanEl.style.width = `${v * 100}%`;
  const cls = v >= bad ? 'bg-bad' : v >= warn ? 'bg-warn' : 'bg-brand-400';
  spanEl.className = `block h-full rounded-full transition-all duration-500 ${cls}`;
}

/** Lightweight toast. */
export function toast(msg, kind = 'info') {
  const root = document.getElementById('toast-root');
  if (!root) return;
  const colors = { info: 'bg-ink-3 text-ink-fg', ok: 'bg-ok/20 text-ok', warn: 'bg-warn/20 text-warn', bad: 'bg-bad/20 text-bad' };
  const t = html`<div class="pointer-events-auto rounded-lg px-4 py-2.5 text-sm shadow-card ${colors[kind] || colors.info}">${msg}</div>`;
  root.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .4s'; }, 3200);
  setTimeout(() => t.remove(), 3700);
}

/** Draw a simple sparkline path into an inline SVG given values + bounds. */
export function sparkPath(values, w, h, min, max) {
  if (!values.length) return '';
  const lo = min ?? Math.min(...values);
  const hi = max ?? Math.max(...values);
  const span = hi - lo || 1;
  const step = w / Math.max(1, values.length - 1);
  return values
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${(h - ((v - lo) / span) * h).toFixed(1)}`)
    .join(' ');
}
