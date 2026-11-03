// Test helper: runs the REAL ui scripts (escape.js ... app.js) in a Node `vm`
// against the REAL index.html parsed into the stub DOM, with a fake
// window.__TAURI__. This is what lets the Node tests exercise app.js.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { documentFromHtml } = require("./testdom.js");

const SCRIPTS = ["escape.js", "stream.js", "help.js", "tooltip.js", "estimate_view.js", "live.js", "a11y.js", "errors.js", "common.js", "settings.js", "shell.js", "datasets_tab.js", "prepare_tab.js", "tokenize_tab.js", "train_tab.js", "generate_tab.js", "evaluate_tab.js", "models_tab.js", "capable_tab.js", "gguf_budget.js", "gguf_tab.js", "finetune_tab.js", "runtime_tab.js", "export_tab.js", "serve_tab.js", "system_tab.js", "app.js"];
const read = (f) => fs.readFileSync(path.join(__dirname, f), "utf8");

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
  for (const f of SCRIPTS) vm.runInContext(read(f), sandbox, { filename: path.join(__dirname, f) });
  return { doc, win, calls, listeners, intervals, timeouts, emit: (name, payload) => listeners[name]({ payload }), fireWindow: (t) => (winListeners[t] || []).forEach((f) => f({ type: t })) };
}

const settle = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r)); };

module.exports = { loadApp, settle };
