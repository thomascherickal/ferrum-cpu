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
  append(...cs) { cs.forEach((c) => this.appendChild(typeof c === "string" ? new TextNode(c) : c)); }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); c.parentNode = null; return c; }
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
  walk(fn) { fn(this); this.children.forEach((c) => c.walk && c.walk(fn)); }
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

// Build a Document from the elements of the real index.html: every
// input/textarea/select/button gets an Element with its id, classes and
// data-* attributes, nested in its <label> when it has one.
function documentFromHtml(html) {
  const doc = new Document();
  const re = /<(\/?)(label|input|textarea|select|button|section|nav)\b([^>]*)>/g;
  const stack = [doc.body];
  let m;
  while ((m = re.exec(html))) {
    const [, close, tag, attrs] = m;
    if (close) {
      if (tag === "label" || tag === "section" || tag === "nav") stack.pop();
      continue;
    }
    const el = doc.createElement(tag);
    for (const a of attrs.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
      if (a[2] !== undefined) el.setAttribute(a[1], a[2]);
    }
    if (tag === "input" && el.getAttribute("type") === "number") el.value = el.getAttribute("value") || "";
    if (tag === "input" && el.getAttribute("type") === "text") el.value = el.getAttribute("value") || "";
    if (tag === "input" && el.getAttribute("type") === "checkbox") el.checked = el.hasAttribute("checked");
    stack[stack.length - 1].appendChild(el);
    if (tag === "label" || tag === "section" || tag === "nav") stack.push(el);
  }
  return doc;
}

module.exports = { Document, Element, documentFromHtml };
