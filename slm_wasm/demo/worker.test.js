// Node test (no dependencies): the worker's message protocol against a fake `Slm`.
// Run: node slm_wasm/demo/worker.test.js
"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");

const vm = require("vm");
const file = path.join(__dirname, "worker.js");
// Blank the import line (keeps line numbers) and run the rest under worker.js's own absolute
// filename, so V8 attributes coverage to the real file.
const src = fs.readFileSync(file, "utf8").replace(/^import .*$/m, "");
assert(!/^import /m.test(src), "the import line was not removed");

const busy = (ms) => { for (const t = Date.now(); Date.now() - t < ms; ); };

// Chunks "a0".."a{n-1}", each taking `ms` to produce, like a slow model.
function fakeSlm(n, ms) {
  return class {
    constructor(bytes) { assert(bytes instanceof Uint8Array); this.i = 0; this.n = 0; }
    info() { return '{"params":1}'; }
    start() { this.i = 0; this.n = 0; this.live = true; }
    next_chunk() {
      if (!this.live || this.i >= n) return undefined;
      busy(ms);
      this.n += 1;
      return "a" + this.i++;
    }
    cancel() { this.live = false; }
    tokens_generated() { return this.n; }
  };
}

// Boot a worker with `Slm`; resolves with { post(msg), out: [messages], wait(pred) }.
function boot(Slm) {
  const out = [];
  const self = { postMessage: (m) => out.push(m) };
  vm.runInNewContext(src, { self, init: async () => {}, Slm, setTimeout, performance, Uint8Array, Error, String }, { filename: file });
  const waiters = [];
  const poll = () => { for (const w of [...waiters]) if (out.some(w.pred)) { waiters.splice(waiters.indexOf(w), 1); w.ok(); } };
  const push = out.push.bind(out);
  out.push = (m) => { push(m); poll(); return out.length; };
  return {
    out,
    post: (m) => self.onmessage({ data: m }),
    wait: (pred) => new Promise((ok) => { waiters.push({ pred, ok }); poll(); }),
  };
}

const gen = { type: "generate", id: 1, prompt: "p", max_tokens: 9, temperature: 0.8, top_k: 0, top_p: 1, repetition_penalty: 1, seed: 1, stop: [] };
const text = (out, id) => out.filter((m) => m.type === "chunk" && m.id === id).map((m) => m.text).join("");

(async () => {
  // Load, generate: the chunks concatenate to the whole completion, then "done" with the count.
  let w = boot(fakeSlm(6, 0));
  w.post({ type: "load", bytes: new ArrayBuffer(4) });
  await w.wait((m) => m.type === "ready");
  assert.strictEqual(w.out[0].info, '{"params":1}');
  w.post(gen);
  await w.wait((m) => m.type === "done");
  assert.strictEqual(text(w.out, 1), "a0a1a2a3a4a5");
  const done = w.out.find((m) => m.type === "done");
  assert.deepStrictEqual([done.id, done.tokens, done.cancelled], [1, 6, false]);

  // Generating before loading is an error that carries the request id.
  w = boot(fakeSlm(1, 0));
  w.post(gen);
  await w.wait((m) => m.type === "error");
  assert.match(w.out[0].message, /load a model/);
  assert.strictEqual(w.out[0].id, 1);

  // Cancel in the middle (slow chunks make the worker yield, so the message gets in).
  w = boot(fakeSlm(20, 35));
  w.post({ type: "load", bytes: new ArrayBuffer(4) });
  await w.wait((m) => m.type === "ready");
  w.post(gen);
  await w.wait((m) => m.type === "chunk");
  w.post({ type: "cancel" });
  await w.wait((m) => m.type === "done");
  const d = w.out.find((m) => m.type === "done");
  assert.strictEqual(d.cancelled, true);
  assert(d.tokens < 20, `stopped early (${d.tokens} tokens)`);

  // A new generate replaces the run in progress; the old one reports cancelled with its id.
  w = boot(fakeSlm(20, 35));
  w.post({ type: "load", bytes: new ArrayBuffer(4) });
  await w.wait((m) => m.type === "ready");
  w.post(gen);
  await w.wait((m) => m.type === "chunk");
  w.post({ ...gen, id: 2, max_tokens: 3 });
  await w.wait((m) => m.type === "done" && m.id === 2);
  assert(w.out.some((m) => m.type === "done" && m.id === 1 && m.cancelled));
  // A cancel with no model loaded is harmless; a constructor failure becomes an error message.
  w = boot(fakeSlm(1, 0));
  w.post({ type: "cancel" });
  w.post({ type: "load", bytes: -1 });
  await w.wait((m) => m.type === "error");
  assert.strictEqual(w.out.length, 1);
  assert.strictEqual(w.out[0].id, undefined);
  w = boot(class { constructor() { throw "plain string"; } });
  w.post({ type: "load", bytes: new ArrayBuffer(1) });
  await w.wait((m) => m.type === "error");
  assert.strictEqual(w.out[0].message, "plain string");
  console.log("worker.test.js: ok");
})().catch((e) => { console.error(e); process.exit(1); });
