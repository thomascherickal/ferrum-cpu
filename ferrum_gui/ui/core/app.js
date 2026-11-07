// Ferrum SLM Studio — vanilla JS frontend (no frameworks). This file only wires the page: the
// Tauri bindings, the shared helpers and one module per tab (shell.js, datasets_tab.js, ...), then
// the backend events and start-up. Each tab module is a plain script: window.FerrumXxx.create(env).
"use strict";

// ── Tauri bindings (gracefully degrade if opened outside Tauri) ─────────────
const T = window.__TAURI__;
const hasTauri = !!(T && T.core && typeof T.core.invoke === "function");
const invoke = hasTauri ? T.core.invoke : async () => { throw new Error("Not running inside the Tauri runtime"); };
const listen = hasTauri ? T.event.listen : async () => () => {};
const dialog = hasTauri && T.dialog ? T.dialog : null;

const $ = (id) => document.getElementById(id);
let live = null; // tooltips + live estimates (live.js), created in init()

const common = window.FerrumCommon.create({ $, doc: document, navigator: window.navigator, getLive: () => live });
// Form values, last-used folders and the Evaluate table survive a restart (settings.js); storage
// that is missing or blocked just means nothing is remembered.
const settings = window.FerrumSettings.create({ doc: document, getStorage: () => window.localStorage, win: window });
const mode = window.FerrumMode.create({ $, doc: document, getStorage: () => window.localStorage });
const openDialog = (dlg) => window.FerrumA11y.openDialog(dlg, document);
const shell = window.FerrumShell.create({ $, doc: document, invoke, dialog, toast: common.toast, errText: common.errText, settings });

// What every tab module gets: the page, the backend and the shared helpers.
const env = {
  $, doc: document, invoke, hasTauri, ...common,
  jobRunning: shell.jobRunning, stopAsked: shell.stopAsked,
  getLive: () => live, openDialog, settings, mode, navigator: window.navigator,
};

const datasetsTab = window.FerrumDatasetsTab.create(env);
const prepareTab = window.FerrumPrepareTab.create(env);
const trainTab = window.FerrumTrainTab.create(env);
const generateTab = window.FerrumGenerateTab.create(env);
const evaluateTab = window.FerrumEvaluateTab.create(env);
window.FerrumModelsTab.create(env);
const capableTab = window.FerrumCapableTab.create(env);
const getBounds = capableTab.getBounds;
const budget = window.FerrumGgufBudget.create({ $, getBounds, openDialog });
const ggufEnv = { ...env, budget, getBounds };
window.FerrumGgufTab.create(ggufEnv);
const finetuneTab = window.FerrumFinetuneTab.create(env);
const exportTab = window.FerrumExportTab.create(ggufEnv);
const serveTab = window.FerrumServeTab.create(env);
const systemTab = window.FerrumSystemTab.create(env);

settings.restore(); // after every tab is built, before the first estimate reads the fields
mode.init();
$("resetSettings").addEventListener("click", () => {
  settings.reset();
  evaluateTab.clearTable();
  common.toast("Settings reset to the defaults", "ok");
});

// ── Backend event wiring ──────────────────────────────────────────────────────
async function wireEvents() {
  await listen("engine-log", (e) => shell.termLine(e.payload, "log"));
  await listen("train-progress", (e) => trainTab.onProgress(e.payload));
  await listen("finetune-progress", (e) => finetuneTab.onProgress(e.payload));
  await listen("download-progress", (e) => datasetsTab.onDownloadProgress(e.payload));
  await listen("clean-progress", (e) => datasetsTab.onCleanProgress(e.payload));
  await listen("prepare-progress", (e) => prepareTab.onProgress(e.payload));
  await listen("tokenize-progress", (e) => trainTab.onTokenizeProgress(e.payload));
  await listen("export-progress", (e) => exportTab.onProgress(e.payload));
  await listen("gen-fragment", (e) => generateTab.onFragment(e.payload));
  await listen("serve-log", (e) => serveTab.onLog(e.payload));
}

// The connection dot: colour plus shape (filled or hollow) plus a spoken label (its data-tip
// explains both states; a native title would show a second bubble).
function setConn(on, text) {
  const dot = $("connDot");
  dot.textContent = on ? "●" : "○";
  dot.setAttribute("aria-label", text);
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  // Tooltips work everywhere; the estimates need the backend and report that they do not have it.
  live = window.FerrumLive.create({
    doc: document, invoke, help: window.FerrumHelp, tooltip: window.FerrumTooltip, view: window.FerrumEstimateView,
    tooltipEnv: { document, window },
  });
  live.start();
  if (!hasTauri) {
    $("webBanner").classList.remove("hidden");
    $("connDot").classList.remove("ok");
    setConn(false, "Backend not connected (running outside Tauri)");
    shell.termLine("Ferrum SLM Studio — backend unavailable (open via `cargo tauri dev`).", "sys");
    return;
  }
  $("connDot").classList.add("ok");
  setConn(true, "Backend connected");
  await wireEvents();
  systemTab.restart();
  serveTab.restore(); // a server started before the page was reloaded is still running
  trainTab.runPreflight();
  shell.termLine("Ferrum SLM Studio ready. Verbose engine logs and shell output appear here.", "sys");
})();
