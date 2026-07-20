// Minimal DOM shim — enough to *run* the HOMECORE-UI panels under Node
// without jsdom. Installs globals (document, location, localStorage,
// fetch, WebSocket) so render-smoke.mjs can execute every panel and
// assert it builds a real DOM subtree without throwing.

class ClassList {
  constructor(el) { this.el = el; this.set = new Set(); }
  add(...c) { c.forEach((x) => x && this.set.add(x)); this.sync(); }
  remove(...c) { c.forEach((x) => this.set.delete(x)); this.sync(); }
  toggle(c, force) { const has = this.set.has(c); const on = force === undefined ? !has : force; if (on) this.set.add(c); else this.set.delete(c); this.sync(); return on; }
  contains(c) { return this.set.has(c); }
  sync() { this.el._class = [...this.set].join(' '); }
}

class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.attrs = {};
    this.style = {};
    this.listeners = {};
    this._class = '';
    this.classList = new ClassList(this);
    this.parentNode = null;
    this.id = '';
    this._text = '';
    this.disabled = false;
    this.value = '';
  }
  set className(v) { this._class = v || ''; this.classList.set = new Set(String(v || '').split(/\s+/).filter(Boolean)); }
  get className() { return this._class; }
  set innerHTML(v) { this._html = v; }
  get innerHTML() { return this._html || ''; }
  set textContent(v) { this._text = v; this.children = []; }
  get textContent() { return this._text || this.children.map((c) => c.textContent || c._text || '').join(''); }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  insertBefore(c, ref) { const i = this.children.indexOf(ref); c.parentNode = this; if (i < 0) this.children.push(c); else this.children.splice(i, 0, c); return c; }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentNode = null; return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  get firstChild() { return this.children[0] || null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(t, fn) { (this.listeners[t] ||= []).push(fn); }
  removeEventListener(t, fn) { this.listeners[t] = (this.listeners[t] || []).filter((f) => f !== fn); }
  dispatch(t, detail) { (this.listeners[t] || []).forEach((fn) => fn({ detail, target: this, preventDefault() {}, stopPropagation() {} })); }
  _all() { return this.children.flatMap((c) => [c, ...(c._all ? c._all() : [])]); }
  matchesSel(sel) {
    return sel.split(/\s+/).pop().split('.').every((p, i, arr) => {
      if (i === 0 && p && !p.startsWith('.') && !p.startsWith('#')) { if (p.startsWith('.')) {} }
      return true;
    });
  }
  querySelector(sel) {
    const want = sel.replace(/^.*\s/, '');
    const cls = want.startsWith('.') ? want.slice(1) : null;
    return this._all().find((e) => (cls ? (e.classList && e.classList.contains(cls)) : e.tagName === want.toUpperCase())) || null;
  }
  querySelectorAll(sel) {
    const want = sel.replace(/^.*\s/, '');
    const cls = want.startsWith('.') ? want.slice(1) : null;
    return this._all().filter((e) => (cls ? (e.classList && e.classList.contains(cls)) : e.tagName === want.toUpperCase()));
  }
}

class TextNode { constructor(t) { this.textContent = String(t); this._text = String(t); this.nodeType = 3; this.parentNode = null; } remove() { if (this.parentNode) this.parentNode.removeChild(this); } }

// Node instanceof checks in ui.js use `instanceof Node`; expose a Node base.
globalThis.Node = El;
// TextNode must also pass `instanceof Node` (ui.js append() treats text via createTextNode).
Object.setPrototypeOf(TextNode.prototype, El.prototype);

const body = new El('body');
const documentObj = {
  createElement: (t) => new El(t),
  createElementNS: (_ns, t) => new El(t),
  createTextNode: (t) => new TextNode(t),
  getElementById: (id) => byId[id] || (byId[id] = mkRoot(id)),
  body,
  readyState: 'complete',
  addEventListener() {},
  querySelectorAll: () => [],
};
const byId = {};
function mkRoot(id) { const e = new El('div'); e.id = id; return e; }

export function install() {
  globalThis.document = documentObj;
  globalThis.EventTarget = class { constructor() { this._l = {}; } addEventListener(t, fn) { (this._l[t] ||= []).push(fn); } removeEventListener(t, fn) { this._l[t] = (this._l[t] || []).filter((f) => f !== fn); } dispatchEvent(e) { (this._l[e.type] || []).forEach((fn) => fn(e)); return true; } };
  // window with a navigable location.hash that fires `hashchange`.
  const win = new globalThis.EventTarget();
  let _hash = '';
  const loc = { host: 'localhost:8123', protocol: 'http:', get hash() { return _hash; }, set hash(v) { _hash = String(v).startsWith('#') ? String(v) : '#' + v; win.dispatchEvent({ type: 'hashchange' }); } };
  win.location = loc;
  globalThis.window = win;
  globalThis.location = loc;
  globalThis.localStorage = { _m: {}, getItem(k) { return this._m[k] ?? null; }, setItem(k, v) { this._m[k] = String(v); } };
  globalThis.fetch = () => Promise.reject(new Error('offline (test) — panels fall back to mock per §7.1'));
  globalThis.WebSocket = class { constructor() { this.readyState = 0; } send() {} close() {} };
  globalThis.CustomEvent = class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } };
  return { El, TextNode, body, document: documentObj, window: win, location: loc };
}

export { El, TextNode };
