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

// Run demo.js itself (absolute filename, so coverage is attributed to it) against a stub page.
const vm = require("vm");
function page(search, fetchImpl) {
  const els = {};
  const el = (id) => (els[id] ||= {
    id, textContent: "", value: "", hidden: false, disabled: false, dataset: {}, files: [],
    classList: { add(c) { this.c = c; }, remove() { this.c = null; } },
  });
  for (const [k, v] of Object.entries({ "max-tokens": "5.9", temperature: "0.5", "top-k": "3", "top-p": "1", seed: "7", prompt: "hi" })) el(k).value = v;
  const worker = { sent: [], postMessage(m) { this.sent.push(m); } };
  const ctx = {
    document: { getElementById: el }, Worker: function () { return worker; },
    URLSearchParams, location: { search }, performance, fetch: fetchImpl, Number, Math, JSON, Error, Promise,
  };
  vm.runInNewContext(js, ctx, { filename: path.join(__dirname, "demo.js") });
  return { el, worker };
}
const flush = () => new Promise((r) => setImmediate(r));
(async () => {
  let { el, worker } = page("");
  const msg = (m) => worker.onmessage({ data: m });
  // A generate request carries the parsed controls; the page disables Generate until a model loads.
  el("generate").onclick();
  assert.strictEqual(worker.sent[0].max_tokens, 5);
  assert.strictEqual(worker.sent[0].seed, 7);
  assert.strictEqual(el("stop").disabled, false);
  // Model info, chunks, a stale id, done, and a stopped run.
  msg({ type: "ready", info: '{"params":1500,"tokenizer":"bpe","vocab":9,"context":4,"quant":"F32"}' });
  assert.match(el("model-info").textContent, /^1,?500 parameters, bpe tokenizer, vocabulary 9, context 4, F32\.$/);
  el("generate").onclick();
  msg({ type: "chunk", id: 2, text: "ab" });
  msg({ type: "chunk", id: 1, text: "stale" });
  assert.strictEqual(el("output").textContent, "ab");
  assert.match(el("stats").textContent, /2 characters so far/);
  msg({ type: "done", id: 2, tokens: 4, ms: 1000, cancelled: true });
  assert.strictEqual(el("stats").textContent, "4 tokens, 4.0 tokens/s (stopped)");
  msg({ type: "done", id: 2, tokens: 0, ms: 0, cancelled: false });
  assert.strictEqual(el("stats").textContent, "0 tokens, 0.0 tokens/s");
  el("stop").onclick();
  assert.strictEqual(JSON.stringify(worker.sent.at(-1)), '{"type":"cancel"}');
  // Errors: a run's error shows; with no model the info line resets; an old run's error is ignored.
  el("model-info").dataset = {};
  msg({ type: "error", id: 2, message: "boom" });
  assert.strictEqual(el("error").textContent, "boom");
  assert.strictEqual(el("model-info").textContent, "No model loaded.");
  msg({ type: "error", id: 99, message: "old" });
  assert.strictEqual(el("error").textContent, "boom");
  worker.onerror({});
  assert.match(el("error").textContent, /see the console/);
  worker.onerror({ message: "x" });
  assert.match(el("error").textContent, /: x$/);
  // A bad number is reported, not sent.
  el("top-p").value = "abc";
  const before = worker.sent.length;
  el("generate").onclick();
  assert.strictEqual(el("error").textContent, "top-p must be a number");
  assert.strictEqual(worker.sent.length, before);
  // File picking and drag/drop load through the worker; an unreadable file reports.
  const file = { arrayBuffer: async () => new ArrayBuffer(3) };
  el("model-file").onchange({ target: { files: [] } });
  el("model-file").onchange({ target: { files: [file] } });
  await flush();
  assert.strictEqual(worker.sent.at(-1).type, "load");
  assert.strictEqual(el("model-info").textContent, "Loading...");
  let prevented = 0;
  const ev = (f) => ({ preventDefault: () => prevented++, dataTransfer: { files: f } });
  el("drop").ondragover(ev([]));
  assert.strictEqual(el("drop").classList.c, "over");
  el("drop").ondragleave();
  el("drop").ondrop(ev([]));
  el("drop").ondrop(ev([{ arrayBuffer: async () => { throw new Error("denied"); } }]));
  await flush();
  assert.strictEqual(prevented, 3);
  assert.strictEqual(el("error").textContent, "Cannot read the file: denied");
  assert.strictEqual(el("model-info").textContent, "No model loaded.");
  // ?model=URL: fetched and loaded, or the failure is shown.
  const ok = { ok: true, arrayBuffer: async () => new ArrayBuffer(1) };
  let p = page("?model=m.gguf", async () => ok);
  await flush();
  assert.strictEqual(p.worker.sent[0].type, "load");
  p = page("?model=m.gguf", async () => ({ ok: false, status: 404 }));
  await flush();
  assert.strictEqual(p.el("error").textContent, "Cannot fetch m.gguf: HTTP 404");
  p = page("?model=m.gguf", async () => { throw new Error("net"); });
  await flush();
  assert.strictEqual(p.el("error").textContent, "Cannot fetch m.gguf: net");
  console.log("demo.test.js: ok");
})().catch((e) => { console.error(e); process.exit(1); });
