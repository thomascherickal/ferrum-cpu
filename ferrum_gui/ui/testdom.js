// A tiny DOM stand-in for the Node tests (no jsdom, no dependencies). It
// implements only what tooltip.js, live.js and app.js use: elements with
// attributes, classes, children, text, events, simple selectors and layout
// rectangles that a test can set.
"use strict";

class ClassList {
  constructor(el) { this.el = el; this.set = new Set(); }
  add(...c) { c.forEach((x) => this.set.add(x)); }
  remove(...c) { c.forEach((x) => this.set.delete(x)); }
  contains(c) { return this.set.has(c); }
  toggle(c, on) { (on === undefined ? !this.set.has(c) : on) ? this.set.add(c) : this.set.delete(c); }
}

class Element {
  constructor(tag, doc) {
    this.tagName = tag.toUpperCase();
    this.ownerDocument = doc;
    this.attrs = {};
    this.children = [];
    this.parentNode = null;
    this.listeners = {};
    this.classList = new ClassList(this);
    this.style = {};
    this.dataset = {};
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this._text = "";
    this.rect = { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 };
  }
  get isConnected() { for (let n = this; n; n = n.parentNode) if (n === this.ownerDocument.body) return true; return false; }
  get id() { return this.attrs.id || ""; }
  set id(v) { this.attrs.id = v; }
  get className() { return [...this.classList.set].join(" "); }
  set className(v) { this.classList.set = new Set(String(v).split(/\s+/).filter(Boolean)); }
  setAttribute(k, v) {
    this.attrs[k] = String(v);
    if (k === "class") this.className = v;
    if (k.startsWith("data-")) this.dataset[k.slice(5).replace(/-(\w)/g, (_, c) => c.toUpperCase())] = String(v);
  }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  hasAttribute(k) { return k in this.attrs; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  get nextSibling() {
    const p = this.parentNode;
    return p ? p.children[p.children.indexOf(this) + 1] || null : null;
  }
  insertBefore(c, ref) {
    if (c.parentNode) c.parentNode.removeChild(c);
    c.parentNode = this;
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(c); else this.children.splice(i, 0, c);
    return c;
  }
  append(...cs) { cs.forEach((c) => this.appendChild(typeof c === "string" ? new TextNode(c) : c)); }
  get childElementCount() { return this.children.length; }
  get firstChild() { return this.children[0] || null; }
  get lastChild() { return this.children[this.children.length - 1] || null; }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); c.parentNode = null; return c; }
  replaceChildren(...cs) { this.textContent = ""; this.append(...cs); }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this.children.forEach((c) => (c.parentNode = null)); this.children = []; this._text = String(v); }
  set innerHTML(v) { this.textContent = ""; this.html = String(v); }
  get innerHTML() { return this.html || ""; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter((x) => x !== f); }
  dispatch(type, extra = {}) {
    const e = { type, target: this, key: undefined, ...extra };
    (this.listeners[type] || []).slice().forEach((f) => f(e));
    return e;
  }
  closest(sel) {
    for (let n = this; n; n = n.parentNode) if (n.matches && n.matches(sel)) return n;
    return null;
  }
  matches(sel) { return sel.split(",").some((s) => matchOne(this, s.trim())); }
  getBoundingClientRect() { return this.rect; }
  querySelectorAll(sel) {
    const out = [];
    this.walk((n) => { if (n !== this && n.matches && n.matches(sel)) out.push(n); });
    return out;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  walk(fn) { fn(this); this.children.forEach((c) => c.walk && c.walk(fn)); }
  click() { this.dispatch("click"); }
  focus() { this.ownerDocument.activeElement = this; this.dispatch("focus"); }
  blur() { if (this.ownerDocument.activeElement === this) this.ownerDocument.activeElement = null; this.dispatch("blur"); }
  showModal() { this.open = true; }
  close() { this.open = false; this.dispatch("close"); }
}

class TextNode {
  constructor(t) { this.data = String(t); this.parentNode = null; }
  get textContent() { return this.data; }
  walk() {}
}

function matchOne(el, sel) {
  if (!sel) return false;
  const m = sel.match(/^([a-zA-Z][\w-]*)?(#[\w-]+)?((?:\.[\w-]+)*)(\[[\w-]+(?:="[^"]*")?\])?$/);
  if (!m) throw new Error("testdom: unsupported selector " + sel);
  const [, tag, id, classes, attr] = m;
  if (tag && el.tagName !== tag.toUpperCase()) return false;
  if (id && el.id !== id.slice(1)) return false;
  for (const c of (classes || "").split(".").filter(Boolean)) if (!el.classList.contains(c)) return false;
  if (attr) {
    const [, k, v] = attr.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/);
    if (!el.hasAttribute(k) || (v !== undefined && el.getAttribute(k) !== v)) return false;
  }
  return true;
}

class Document {
  constructor() {
    this.listeners = {};
    this.activeElement = null;
    this.body = new Element("body", this);
    this.documentElement = new Element("html", this);
    this.documentElement.appendChild(this.body);
  }
  createElement(t) { return new Element(t, this); }
  createTextNode(t) { return new TextNode(t); }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  dispatch(type, extra = {}) { (this.listeners[type] || []).forEach((f) => f({ type, ...extra })); }
  getElementById(id) {
    let found = null;
    this.body.walk((n) => { if (!found && n.id === id) found = n; });
    return found;
  }
  querySelectorAll(sel) {
    const out = [];
    this.body.walk((n) => { if (n !== this.body && n.matches && n.matches(sel)) out.push(n); });
    return out;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

// Build a Document from the real index.html: every element becomes an Element
// with its attributes (so ids, classes and data-* work), nested as in the
// markup. Text nodes, comments and <head>/<script> contents are skipped.
// The entities a browser decodes inside an attribute value (the page uses these four and &#10;).
const unescapeAttr = (s) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#10;/g, "\n").replace(/&amp;/g, "&");
const VOID = new Set(["input", "img", "br", "hr", "meta", "link"]);
function documentFromHtml(html) {
  const doc = new Document();
  const body = html.replace(/<!--[\s\S]*?-->/g, "").replace(/^[\s\S]*?<body[^>]*>/i, "").replace(/<script[\s\S]*?<\/script>/gi, "");
  const stack = [doc.body];
  for (const m of body.matchAll(/<(\/?)([a-zA-Z][\w-]*)\b([^>]*?)(\/?)>/g)) {
    const [, close, tag, attrs] = m;
    if (close) {
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].tagName === tag.toUpperCase()) { stack.length = i; break; }
      }
      continue;
    }
    const el = doc.createElement(tag);
    for (const a of attrs.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
      el.setAttribute(a[1], a[2] === undefined ? "" : unescapeAttr(a[2]));
    }
    if (tag === "input") {
      el.value = el.getAttribute("value") || "";
      el.checked = el.hasAttribute("checked");
    }
    stack[stack.length - 1].appendChild(el);
    if (!VOID.has(tag) && !m[4]) stack.push(el);
  }
  // <select> value = the `selected` option, else the first; textarea/others start empty.
  doc.querySelectorAll("select").forEach((sel) => {
    const first = sel.querySelector("option[selected]") || sel.querySelector("option");
    sel.value = first ? first.getAttribute("value") || "" : "";
  });
  return doc;
}

module.exports = { Document, Element, documentFromHtml };
