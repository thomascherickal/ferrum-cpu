// Test helper: runs the REAL ui scripts (escape.js ... app.js) in a Node `vm`
// against the REAL index.html parsed into the stub DOM, with a fake
// window.__TAURI__. This is what lets the Node tests exercise app.js.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { documentFromHtml } = require("./testdom.js");

const SCRIPTS = ["core/escape.js", "core/stream.js", "core/help.js", "core/tooltip.js", "core/estimate_view.js", "core/live.js", "core/a11y.js", "core/errors.js", "core/common.js", "core/settings.js", "core/mode.js", "core/shell.js", "tabs/datasets_tab.js", "tabs/prepare_tab.js", "tabs/tokenize_tab.js", "tabs/train_tab.js", "tabs/generate_tab.js", "tabs/evaluate_tab.js", "tabs/models_tab.js", "tabs/capable_tab.js", "core/gguf_budget.js", "tabs/gguf_tab.js", "tabs/finetune_tab.js", "tabs/runtime_tab.js", "tabs/export_tab.js", "tabs/serve_tab.js", "tabs/system_tab.js", "core/app.js"];
const ui = path.join(__dirname, "..");
const read = (f) => fs.readFileSync(path.join(ui, f), "utf8");

// Canvas context that accepts any drawing call.
const ctx2d = new Proxy({}, { get: () => () => {}, set: () => true });

// `backend(cmd, args)` answers invoke(); `tauri: false` simulates a plain browser.
// `storage`: an object with getItem/setItem/removeItem becomes window.localStorage; "blocked" makes
// reading window.localStorage throw, like a webview with site data disabled; absent leaves it out.
function loadApp({ backend, tauri = true, events = {}, dialogs = {}, storage, navigator } = {}) {
  const doc = documentFromHtml(read("index.html"));
  doc.createElement = ((orig) => (t) => {
    const el = orig(t);
    if (t === "canvas") el.getContext = () => ctx2d;
    return el;
  })(doc.createElement.bind(doc));
  doc.querySelectorAll("canvas").forEach((c) => { c.getContext = () => ctx2d; c.width = 900; c.height = 140; });
  const calls = [];
  const listeners = {};
  const winListeners = {};
  const win = {
    addEventListener(t, f) { (winListeners[t] = winListeners[t] || []).push(f); },
    innerWidth: 1300,
    innerHeight: 900,
  };
  if (tauri) {
    win.__TAURI__ = {
      core: { invoke: async (cmd, args) => { calls.push({ cmd, args }); return backend(cmd, args); } },
      event: { listen: async (name, cb) => { listeners[name] = cb; return () => {}; } },
      dialog: { open: async (o) => (o && o.directory ? dialogs.dir : dialogs.open) || null, save: async () => dialogs.save || null },
    };
  }
  const intervals = [];
  // Short timers (debounces) run for real, cut to 5 ms. Long ones (>= 1 s: toast lifetimes) are
  // only recorded, like intervals, so a toast never vanishes under load before a test reads it;
  // a test fires one with `timeouts[i].f()`.
  const timeouts = [];
  const sandbox = {
    window: win, document: doc, console, crypto: globalThis.crypto, Promise, Math, Number, JSON, Array, String, Object, Error, parseInt, parseFloat, isNaN,
    setTimeout: (f, ms) => ((ms || 0) >= 1000 ? -timeouts.push({ f, ms }) : setTimeout(f, Math.min(ms || 0, 5))),
    clearTimeout, setInterval: (f, ms) => { intervals.push({ f, ms }); return intervals.length; },
    clearInterval() {},
  };
  Object.assign(sandbox, win);
  sandbox.navigator = navigator;
  if (storage === "blocked") Object.defineProperty(sandbox, "localStorage", { get() { throw new Error("SecurityError: storage is blocked"); } });
  else if (storage) sandbox.localStorage = storage;
  sandbox.window = sandbox; // like a browser: window === the global object
  vm.createContext(sandbox);
  for (const f of SCRIPTS) vm.runInContext(read(f), sandbox, { filename: path.join(ui, f) });
  return { doc, win, calls, listeners, intervals, timeouts, emit: (name, payload) => listeners[name]({ payload }), fireWindow: (t) => (winListeners[t] || []).forEach((f) => f({ type: t })) };
}

const settle = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r)); };

module.exports = { loadApp, settle };
