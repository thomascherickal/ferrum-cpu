// Test helper: a small HTML reader for the accessibility tests. Unlike testdom.js it keeps text
// nodes, so a test can ask "does this label have words?". Only what index.html uses.
"use strict";
const VOID = new Set(["input", "img", "br", "hr", "meta", "link"]);

function parse(html) {
  const src = html.replace(/<!--[\s\S]*?-->/g, "").replace(/<script[\s\S]*?<\/script>/gi, "").replace(/^[\s\S]*?<body[^>]*>/i, "");
  const root = { tag: "#root", attrs: {}, children: [], parent: null };
  let cur = root;
  let last = 0;
  const text = (s) => { if (s) cur.children.push({ text: s.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">") }); };
  for (const m of src.matchAll(/<(\/?)([a-zA-Z][\w-]*)\b((?:[^>"]|"[^"]*")*?)(\/?)>/g)) {
    text(src.slice(last, m.index));
    last = m.index + m[0].length;
    const [, close, tag, attrs, selfClose] = m;
    if (close) {
      for (let n = cur; n; n = n.parent) if (n.tag === tag.toLowerCase()) { cur = n.parent; break; }
      continue;
    }
    const el = { tag: tag.toLowerCase(), attrs: {}, children: [], parent: cur };
    for (const a of attrs.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) el.attrs[a[1]] = a[2] === undefined ? "" : a[2];
    cur.children.push(el);
    if (!VOID.has(el.tag) && !selfClose) cur = el;
  }
  return root;
}

function walk(n, fn) { if (n.tag) { fn(n); n.children.forEach((c) => walk(c, fn)); } }
const all = (root, pred) => { const out = []; walk(root, (n) => { if (pred(n)) out.push(n); }); return out; };
const textOf = (n) => (n.text !== undefined ? n.text : n.children.map(textOf).join(""));
const ancestor = (n, tag) => { for (let p = n.parent; p; p = p.parent) if (p.tag === tag) return p; return null; };

module.exports = { parse, all, textOf, ancestor, walk };
