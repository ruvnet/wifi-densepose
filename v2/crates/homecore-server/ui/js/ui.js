// HOMECORE-UI shared component helpers — ADR-131 §3.3.
//
// Every panel imports from here so cards/pills/buttons/badges are
// byte-identical across the dashboard (the §3.3 "no visual seam"
// invariant). Pure DOM, no framework, no build step.

/** Hyperscript element factory. `h('div.card#x', {onClick}, ...children)`. */
export function h(spec, attrs, ...children) {
  let tag = 'div', id = null;
  const classes = [];
  spec.replace(/([.#]?[^.#]+)/g, (tok) => {
    if (tok[0] === '.') classes.push(tok.slice(1));
    else if (tok[0] === '#') id = tok.slice(1);
    else tag = tok;
    return tok;
  });
  const node = document.createElement(tag);
  if (id) node.id = id;
  if (classes.length) node.className = classes.join(' ');
  if (attrs && typeof attrs === 'object' && !(attrs instanceof Node) && !Array.isArray(attrs)) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') node.className += ' ' + v;
      else if (k === 'html') node.innerHTML = v;
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
      else node.setAttribute(k, v);
    }
  } else if (attrs != null) {
    children.unshift(attrs);
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export const txt = (s) => document.createTextNode(s == null ? '' : String(s));
export const mono = (s) => h('span.mono', String(s == null ? '' : s));
export const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); return n; };

/** Status pill. kind ∈ cyan|green|amber|red|purple|grey. */
export function pill(text, kind = 'grey') {
  return h(`span.pill.${kind}`, String(text));
}

/** Map a free-form status string to the platform colour convention. */
export function statusPill(status) {
  const s = String(status || '').toLowerCase();
  const map = {
    running: 'green', online: 'green', ok: 'green', healthy: 'green', occupied: 'green', paired: 'green', connected: 'green', valid: 'green',
    stale: 'amber', degraded: 'amber', updating: 'amber', warn: 'amber', warning: 'amber',
    failed: 'red', offline: 'red', error: 'red', veto: 'red', vetoed: 'red', unreachable: 'red', invalid: 'red',
    stopped: 'grey', absent: 'grey', unknown: 'grey', 'not trained': 'grey',
    info: 'purple', epoch: 'purple', chain: 'purple',
  };
  return pill(status, map[s] || 'grey');
}

export function card({ title, tint, accent, clickable, onClick, children = [] } = {}) {
  const cls = ['card'];
  if (tint) cls.push('tint-' + tint);
  if (clickable || onClick) cls.push('clickable');
  const node = h('.' + cls.join('.'));
  if (onClick) node.addEventListener('click', onClick);
  if (accent) node.appendChild(accentBar());
  if (title) node.appendChild(h('h2', title));
  append(node, [children]);
  return node;
}

function accentBar() {
  const b = h('div');
  b.style.height = '3px';
  b.style.borderRadius = '3px';
  b.style.margin = '-14px -10px 14px';
  b.style.background = 'linear-gradient(90deg, var(--cyan), var(--purple))';
  return b;
}

/** Section header with the cyan→purple featured gradient border (§3.3). */
export function sectionHeader(title, sub) {
  return h('.section-header', h('h1', title), sub ? h('.sub', sub) : null);
}

/** Live metric card (§4.1). */
export function metric({ icon, value, label, color = 'cyan' }) {
  return h('.metric',
    icon ? h('.ico', icon) : null,
    h(`.val${color === 'green' ? '.green' : ''}`, String(value)),
    h('.lbl', label));
}

export function button(label, { variant = 'ghost', onClick, disabled } = {}) {
  const b = h(`button.btn.${variant}`, label);
  if (disabled) b.disabled = true;
  if (onClick) b.addEventListener('click', onClick);
  return b;
}

/**
 * Progress bar with threshold colouring.
 * thresholds: [{ lt, color }] evaluated in order against the 0..1 ratio.
 */
export function bar(value, max = 1, thresholds = null) {
  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  let color = '';
  if (thresholds) {
    for (const t of thresholds) { if (ratio < t.lt) { color = t.color; break; } }
    if (!color) color = thresholds[thresholds.length - 1].color;
  }
  const fill = h('span' + (color ? '.' + color : ''));
  fill.style.width = (ratio * 100).toFixed(1) + '%';
  return h('.bar', fill);
}

/** Small inline confidence bar — amber below 0.4 (§4.5). */
export function confidenceBar(conf) {
  const c = Math.max(0, Math.min(1, conf || 0));
  const fill = h('span' + (c < 0.4 ? '.amber' : ''));
  fill.style.width = (c * 100).toFixed(0) + '%';
  return h('.conf-bar', fill);
}

/**
 * Provenance badge (§4.4 / §6) — ESP32 → SEED → COG → state machine.
 * A first-class element, never collapsed. hailo:true marks Hailo-sourced
 * inference visually distinct from CPU-only COGs (§6 invariant 5).
 */
export function provenanceBadge({ esp32, seed, cog, hailo } = {}) {
  return h('span.prov',
    esp32 ? txt(esp32) : null, esp32 ? h('span.arr', '→') : null,
    seed ? txt(seed) : null, h('span.arr', '→'),
    h(hailo ? 'span.hailo' : 'span', cog || 'cog'),
    h('span.arr', '→'), txt('homecore'));
}

/** Tiny inline SVG sparkline. */
export function sparkline(values, { w = 120, hgt = 28, color = 'var(--cyan)' } = {}) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', w); svg.setAttribute('height', hgt); svg.setAttribute('class', 'spark');
  if (!values || values.length < 2) return svg;
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const step = w / (values.length - 1);
  const pts = values.map((v, i) => `${(i * step).toFixed(1)},${(hgt - ((v - min) / span) * (hgt - 4) - 2).toFixed(1)}`).join(' ');
  const pl = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  pl.setAttribute('points', pts); pl.setAttribute('fill', 'none');
  pl.setAttribute('stroke', color); pl.setAttribute('stroke-width', '1.5');
  svg.appendChild(pl);
  return svg;
}

export function banner(text, kind = 'amber', extra) {
  return h(`.banner.${kind}`, text, extra ? txt(' ') : null, extra || null);
}

export function row(k, v) {
  return h('.row', h('span.k', k), v instanceof Node ? v : h('span.v', String(v == null ? '—' : v)));
}

export function kv(pairs) {
  const node = h('.kv');
  for (const [k, v] of pairs) {
    node.appendChild(h('span.k', k));
    node.appendChild(v instanceof Node ? v : h('span.v', String(v == null ? '—' : v)));
  }
  return node;
}

/** Collapsible section. */
export function collapsible(title, contentFn, open = false) {
  const wrap = h('.collapsible' + (open ? '.open' : ''));
  const head = h('.head', title);
  const body = h('div');
  wrap.appendChild(head); wrap.appendChild(body);
  let built = false;
  const toggle = () => {
    wrap.classList.toggle('open');
    if (wrap.classList.contains('open')) {
      if (!built) { body.appendChild(contentFn()); built = true; }
      body.classList.remove('hidden');
    } else body.classList.add('hidden');
  };
  head.addEventListener('click', toggle);
  if (open) { body.appendChild(contentFn()); built = true; } else body.classList.add('hidden');
  return wrap;
}

/** Slide-over panel (§4.4 StateChanged detail). */
export function slideover(title, content) {
  const back = h('.slideover-back');
  const panel = h('.slideover', h('span.close', { onClick: close }, '✕'), h('h2', title), content);
  function close() { back.remove(); panel.remove(); }
  back.addEventListener('click', close);
  document.body.appendChild(back);
  document.body.appendChild(panel);
  return { close };
}

/** Lag indicator (§4.1/§4.4 — broadcast channel vs 4096 capacity). */
export function lagIndicator(state, lagged) {
  const cls = state === 'open' ? (lagged ? 'warn' : '') : 'err';
  const label = state === 'open' ? (lagged ? 'WS lagging — events dropped' : 'WS live') : 'WS offline';
  return h('span.lag', h(`span.dot${cls ? '.' + cls : ''}`), h('span.t2', label));
}

export function relTime(iso) {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return String(iso);
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 0) return 'in ' + fmtDur(-s);
  if (s < 5) return 'just now';
  return fmtDur(s) + ' ago';
}
function fmtDur(s) {
  if (s < 60) return s + 's';
  if (s < 3600) return Math.round(s / 60) + 'm';
  if (s < 86400) return Math.round(s / 3600) + 'h';
  return Math.round(s / 86400) + 'd';
}

/** Loading + error wrappers panels can await. */
export function loading(label = 'Loading…') { return h('.muted-empty', label); }
export function errorCard(e) { return banner('Unavailable — ' + (e && e.message ? e.message : e), 'red'); }

/** Distinguish "not trained" (null) from "unavailable" (error) — §6 invariant 3. */
export function notTrained(prompt = 'Calibrate to enable') {
  return h('span.t3', 'Not trained ', button(prompt, { variant: 'ghost' }));
}
