// Node test (no dependencies): demo/index.html and demo.js agree, and the page stays CSP-clean.
// Run: node slm_wasm/demo/demo.test.js
"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");

const read = (f) => fs.readFileSync(path.join(__dirname, f), "utf8");
const html = read("index.html");
const js = read("demo.js");

const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
assert.strictEqual(ids.size, [...html.matchAll(/\sid="/g)].length, "duplicate ids in index.html");

// Every control demo.js looks up exists, and none of the page's controls is missing from it.
const used = new Set([...js.matchAll(/(?:\$|number)\("([\w-]+)"/g)].map((m) => m[1]));
for (const id of used) assert(ids.has(id), `demo.js uses #${id}, which index.html lacks`);
for (const id of ["model-file", "prompt", "temperature", "top-k", "top-p", "max-tokens", "generate", "stop", "output", "stats", "error", "drop"]) {
  assert(used.has(id), `demo.js never uses #${id}`);
}

// CSP: no inline script, no inline style, no event-handler attributes, no javascript: URLs.
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)];
assert(scripts.length >= 1);
for (const [, attrs, body] of scripts) {
  assert(/\ssrc="[^"]+"/.test(attrs), "a <script> without src");
  assert.strictEqual(body.trim(), "", "inline script body");
}
assert(!/<style\b/i.test(html), "inline <style>");
assert(!/\sstyle\s*=/i.test(html), "style= attribute");
assert(!/\son[a-z]+\s*=/i.test(html), "inline event handler attribute");
assert(!/javascript:/i.test(html), "javascript: URL");
const csp = (html.match(/Content-Security-Policy"\s+content="([^"]+)"/) || [])[1];
assert(csp, "no CSP meta tag");
assert(!/unsafe-inline|unsafe-eval(?!')|\*/.test(csp.replace("'wasm-unsafe-eval'", "")), `CSP is loosened: ${csp}`);
assert(/default-src 'self'/.test(csp));

// Every file the page loads exists next to it.
for (const m of html.matchAll(/(?:src|href)="([^"#]+)"/g)) {
  assert(!/^[a-z]+:|^\/\//i.test(m[1]), `external resource ${m[1]}`);
  assert(fs.existsSync(path.join(__dirname, m[1])), `${m[1]} is missing`);
}
assert(fs.existsSync(path.join(__dirname, "worker.js")));

// Accessible labels: every form control has a <label for>, the language is set, and the
// buttons have text.
const labelled = new Set([...html.matchAll(/<label\b[^>]*\sfor="([^"]+)"/g)].map((m) => m[1]));
for (const m of html.matchAll(/<(input|textarea|select)\b([^>]*)>/g)) {
  const id = (m[2].match(/\sid="([^"]+)"/) || [])[1];
  assert(id && labelled.has(id), `<${m[1]}${m[2]}> has no <label for>`);
}
for (const f of labelled) assert(ids.has(f), `<label for="${f}"> points nowhere`);
assert(/<html lang="\w+"/.test(html), "no lang attribute");
for (const m of html.matchAll(/<button\b[^>]*>([\s\S]*?)<\/button>/g)) assert(m[1].trim(), "empty button");

console.log("demo.test.js: ok");
