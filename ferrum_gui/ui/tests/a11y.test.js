// Tests of a11y.js (tab keys, the modal dialog helper) and of the accessibility behaviour wired
// into the real page: the tab bar's roving tabindex and keys, toasts, progress bars, the
// connection dot and the dialogs' focus handling, all against the stub DOM. A screen reader and
// a real focus ring were NOT exercised.
// Run with: node ui/tests/a11y.test.js
"use strict";
const assert = require("node:assert");
const { nextTabIndex, openDialog } = require("../core/a11y.js");
const { Document } = require("./testdom.js");
const { loadApp, settle } = require("./testapp.js");

// ── nextTabIndex ──
assert.strictEqual(nextTabIndex("ArrowRight", 0, 4), 1);
assert.strictEqual(nextTabIndex("ArrowRight", 3, 4), 0);
assert.strictEqual(nextTabIndex("ArrowLeft", 0, 4), 3);
assert.strictEqual(nextTabIndex("ArrowLeft", 2, 4), 1);
assert.strictEqual(nextTabIndex("Home", 2, 4), 0);
assert.strictEqual(nextTabIndex("End", 0, 4), 3);
for (const k of ["a", "Enter", "Tab", " ", "ArrowDown", "ArrowUp"]) assert.strictEqual(nextTabIndex(k, 1, 4), -1, k);

// ── openDialog on its own ──
{
  const doc = new Document();
  const opener = doc.createElement("button"); doc.body.appendChild(opener);
  const dlg = doc.createElement("dialog"); doc.body.appendChild(dlg);
  const [a, b] = ["button", "button"].map(() => dlg.appendChild(doc.createElement("button")));
  const gone = doc.createElement("button"); gone.disabled = true; dlg.appendChild(gone);
  opener.focus();
  openDialog(dlg, doc);
  assert.ok(dlg.open);
  // Tab on the last control wraps to the first; Shift+Tab on the first wraps to the last.
  b.focus();
  let e = dlg.dispatch("keydown", { key: "Tab", shiftKey: false, preventDefault() { this.stopped = true; } });
  assert.ok(e.stopped); assert.strictEqual(doc.activeElement, a);
  e = dlg.dispatch("keydown", { key: "Tab", shiftKey: true, preventDefault() { this.stopped = true; } });
  assert.ok(e.stopped); assert.strictEqual(doc.activeElement, b);
  // Tab in the middle is left to the browser.
  a.focus();
  e = dlg.dispatch("keydown", { key: "Tab", shiftKey: false, preventDefault() { this.stopped = true; } });
  assert.ok(!e.stopped);
  // Escape closes it and hands focus back to the opener.
  dlg.dispatch("keydown", { key: "Escape", preventDefault() {} });
  assert.ok(!dlg.open);
  assert.strictEqual(doc.activeElement, opener);
  assert.strictEqual((dlg.listeners.keydown || []).length, 0, "listeners removed on close");
  // Closed by a button: focus returns too, and a second open works.
  opener.focus();
  openDialog(dlg, doc);
  a.focus();
  dlg.close();
  assert.strictEqual(doc.activeElement, opener);
}

// ── The real page ──
const HW = { cpuModel: "T", ferrumThreads: 4, logicalCores: 4, physicalCores: 4, ramTotalMb: 8000, ramAvailableMb: 4000, gpus: [] };
const REPORT = { cpu: "T", cores: 4, threads: 4, memTotal: 8e9, memAvail: 4e9, memBwGbps: 3.5, gemmGflops: 60, inferInt4: 1e9, inferInt8: 5e8, inferF32: 1e8,
  trainChinchilla: 2e6, trainFixed1b: 1e6, testEval: 3e6, loadInt4: 5e9, loadInt8: 2e9, loadF32: 1e9, finetuneLo: 1e5, finetuneHi: 1e6, targetToks: 3,
  trainHours: 24, evalTokens: 1e7, fixedTrainTokens: 1e9, chinchillaRatio: 20, finetuneTokensLo: 1e7, finetuneTokensHi: 1e6, loadFraction: 0.9 };
const BIG = { architecture: "llama", runnable: true, version: 3, numTensors: 1, modelDim: 1, nLayers: 1, nHeads: 1, nKvHeads: 1, vocabSize: 1, tokenizer: "t", estInt4Mb: 1, estInt8Mb: 1, estF32Mb: 1, availMb: 1, note: "", paramCount: 9e9 };
const backend = async (cmd, args) => {
  switch (cmd) {
    case "list_datasets": return [];
    case "hardware_info": return HW;
    case "preflight_train": return { params: 1, peakRamMb: 1, ramAvailableMb: 9, ramLimitMb: 9, fitsInRam: true, seconds: 1, llamaCpp: { runs: true, reason: "" }, problems: [] };
    case "estimate_job": return { kind: args.kind, seconds: 5, lowSeconds: 3, highSeconds: 8, threadsUsed: 4, notes: [] };
    case "system_stats": return { cpuTotal: 12.5, memPercent: 40.4, ferrumThreads: 4, memUsed: 3e9, memTotal: 8e9, perCore: [10, 20] };
    case "capability_report": return REPORT;
    case "gguf_info": return BIG;
    default: throw new Error("no fake for " + cmd);
  }
};

(async () => {
  const app = loadApp({ backend });
  await settle(8);
  const $ = (id) => app.doc.getElementById(id);
  const tabs = app.doc.querySelectorAll(".tab");
  const state = () => tabs.map((t) => `${t.getAttribute("aria-selected")}/${t.getAttribute("tabindex")}`);

  // Tab bar: clicking moves aria-selected and the roving tabindex with the visible state.
  assert.strictEqual(state()[0], "true/0");
  $("tab-models").dispatch("click");
  assert.strictEqual($("tab-models").getAttribute("aria-selected"), "true");
  assert.strictEqual($("tab-models").getAttribute("tabindex"), "0");
  assert.strictEqual(state().filter((s) => s === "true/0").length, 1);
  assert.strictEqual(state().filter((s) => s === "false/-1").length, tabs.length - 1);
  assert.ok($("panel-models").classList.contains("active") && !$("panel-datasets").classList.contains("active"));

  // Arrow keys focus and activate the neighbour; Home/End jump; others do nothing.
  const press = (id, key) => { const e = $(id).dispatch("keydown", { key, preventDefault() { this.stopped = true; } }); return e; };
  let e = press("tab-models", "ArrowRight");
  assert.ok(e.stopped);
  assert.strictEqual(app.doc.activeElement, $("tab-gguf"));
  assert.ok($("panel-gguf").classList.contains("active"));
  assert.strictEqual($("tab-gguf").getAttribute("aria-selected"), "true");
  press("tab-gguf", "End");
  assert.strictEqual(app.doc.activeElement, $("tab-capable"));
  press("tab-capable", "ArrowRight"); // wraps
  assert.strictEqual(app.doc.activeElement, $("tab-datasets"));
  press("tab-datasets", "ArrowLeft"); // wraps back
  assert.strictEqual(app.doc.activeElement, $("tab-capable"));
  press("tab-capable", "Home");
  assert.strictEqual(app.doc.activeElement, $("tab-datasets"));
  e = press("tab-datasets", "x");
  assert.ok(!e.stopped);
  // Moving to Train by key also runs the Train tab's own click listener (the pre-flight).
  const before = app.calls.filter((c) => c.cmd === "preflight_train").length;
  $("tab-prepare").focus();
  press("tab-prepare", "ArrowRight");
  await new Promise((r) => setTimeout(r, 400));
  assert.ok(app.calls.filter((c) => c.cmd === "preflight_train").length > before, "pre-flight did not follow a keyboard tab switch");

  // Toasts: an error is announced at once; the others rely on the polite container.
  $("dsLoadFile").dispatch("click"); // empty path -> inline error (a role=alert line), no toast
  await settle();
  assert.ok(/File path is required/.test($("errDsSource").textContent));
  $("miInspect").dispatch("click"); // empty -> inline error only
  $("ggRun").dispatch("click"); await settle(); // validation error only
  $("miPath").value = "/m.gguf";
  $("miInspect").dispatch("click"); await settle(); // model_info has no fake -> error toast
  const toast = $("toasts").children[$("toasts").children.length - 1];
  assert.ok(toast.classList.contains("err"));
  assert.strictEqual(toast.getAttribute("role"), "alert");

  // Progress bars: the width and aria-valuenow move together (system monitor bars here).
  await settle(8);
  assert.strictEqual($("sysCpuBar").style.width, "12.5%");
  assert.strictEqual($("sysCpuBar").parentNode.getAttribute("aria-valuenow"), "13");
  assert.strictEqual($("sysMemBar").parentNode.getAttribute("aria-valuenow"), "40");

  // The connection dot: filled and spoken when connected.
  assert.strictEqual($("connDot").textContent, "●");
  assert.strictEqual($("connDot").getAttribute("aria-label"), "Backend connected");
  const plain = loadApp({ tauri: false });
  await settle(6);
  const dot = plain.doc.getElementById("connDot");
  assert.strictEqual(dot.textContent, "○");
  assert.ok(/not connected/.test(dot.getAttribute("aria-label")));

  // Dialogs open through the page's own wiring: the Capable report, then the over-budget
  // warning. Esc closes each and focus goes back to the button that opened it.
  $("capCheck").focus();
  $("capCheck").dispatch("click"); await settle(6);
  assert.ok($("capDialog").open);
  $("capDialogClose").focus();
  $("capDialog").dispatch("keydown", { key: "Escape", preventDefault() {} });
  assert.ok(!$("capDialog").open);
  assert.strictEqual(app.doc.activeElement, $("capCheck"));

  $("ggPath").value = "/g.gguf"; $("ggPrompt").value = "hi";
  $("ggRun").focus();
  $("ggRun").dispatch("click"); await settle(6);
  assert.ok($("ggWarnDialog").open);
  $("ggWarnProceed").focus();
  $("ggWarnDialog").dispatch("keydown", { key: "Tab", shiftKey: false, preventDefault() {} }); // last -> first
  assert.strictEqual(app.doc.activeElement, $("ggWarnCancel"));
  $("ggWarnDialog").dispatch("keydown", { key: "Escape", preventDefault() {} });
  assert.ok(!$("ggWarnDialog").open);
  assert.strictEqual(app.doc.activeElement, $("ggRun"));
  await settle();
  assert.strictEqual($("ggStatus").textContent, "cancelled"); // Esc means "do not proceed"

  console.log("a11y.test.js: all assertions passed");
})().catch((err) => { console.error(err); process.exit(1); });
