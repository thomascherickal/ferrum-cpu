// Tests of errors.js and of how the page shows a structured error ({ code, message, hint }):
// message + hint, "Show full message" for long ones, "Copy details" with the clipboard API and
// its fallback, toasts, and the busy spinner. Stub DOM and a stand-in clipboard; the real
// clipboard and webview were NOT exercised.
// Run with: node ui/tests/errors.test.js
"use strict";
const assert = require("node:assert");
const E = require("../core/errors.js");
const { loadApp, settle } = require("./testapp.js");

// ── normalize ──
const RUST = { code: "not_found", message: "model path: /x does not exist", hint: "Check the path." };
assert.deepStrictEqual(E.normalize(RUST), RUST);
assert.deepStrictEqual(E.normalize("plain text"), { code: "error", message: "plain text", hint: "" });
assert.deepStrictEqual(E.normalize(new Error("Corpus is required")), { code: "error", message: "Corpus is required", hint: "" });
assert.deepStrictEqual(E.normalize({ message: "m", code: 5, hint: null }), { code: "error", message: "m", hint: "" });
assert.deepStrictEqual(E.normalize(undefined), { code: "error", message: "", hint: "" });
assert.deepStrictEqual(E.normalize({ foo: 1 }), { code: "error", message: "[object Object]", hint: "" });
assert.strictEqual(E.text(RUST), RUST.message);
assert.ok(E.isCancelled({ code: "cancelled", message: "x", hint: "" }));
assert.ok(E.isCancelled("export cancelled; no file was written"));
assert.ok(!E.isCancelled(RUST) && !E.isCancelled(new Error("nope")));

// ── truncate ──
assert.deepStrictEqual(E.truncate("short"), { shown: "short", truncated: false });
assert.deepStrictEqual(E.truncate("a".repeat(300)), { shown: "a".repeat(300), truncated: false });
const t = E.truncate("a".repeat(301));
assert.ok(t.truncated && t.shown === "a".repeat(300) + "…");
const emoji = E.truncate("x".repeat(9) + "😀".repeat(5), 10); // the cut would fall inside a surrogate pair
assert.ok(emoji.truncated && emoji.shown === "x".repeat(9) + "…", JSON.stringify(emoji.shown));

// ── details ──
const d = E.details(RUST, new Date("2026-01-02T03:04:05Z"));
assert.strictEqual(d, "Ferrum SLM Studio error\ncode: not_found\nmessage: model path: /x does not exist\nhint: Check the path.\ntime: 2026-01-02T03:04:05.000Z");
assert.ok(E.details("bare").includes("hint: -"));

(async () => {
  // ── copy ──
  {
    let got = null;
    assert.strictEqual(await E.copy("abc", { navigator: { clipboard: { writeText: async (s) => { got = s; } } }, document: null }), true);
    assert.strictEqual(got, "abc");
    // Clipboard API refused -> the textarea fallback.
    const { Document } = require("./testdom.js");
    const doc = new Document();
    let copied = null;
    doc.execCommand = (c) => { copied = { c, text: doc.body.children[doc.body.children.length - 1].value }; return true; };
    const denied = { clipboard: { writeText: async () => { throw new Error("denied"); } } };
    assert.strictEqual(await E.copy("xyz", { navigator: denied, document: doc }), true);
    assert.deepStrictEqual(copied, { c: "copy", text: "xyz" });
    assert.strictEqual(doc.body.children.length, 0, "the helper textarea is removed");
    // No clipboard API at all, and a failing execCommand.
    doc.execCommand = () => false;
    assert.strictEqual(await E.copy("q", { navigator: undefined, document: doc }), false);
    assert.strictEqual(await E.copy("q", { navigator: {}, document: { createElement() { throw new Error("no dom"); } } }), false);
  }

  // ── The page ──
  const HW = { cpuModel: "T", ferrumThreads: 4, logicalCores: 4, physicalCores: 4, ramTotalMb: 8000, ramAvailableMb: 4000, gpus: [] };
  const LONG = "x".repeat(450);
  let release = null;
  const errors = {
    gguf_info: { code: "model", message: "cannot open /g.gguf: bad magic", hint: "The file may be damaged." },
    model_info: { code: "internal", message: LONG, hint: "Use Copy details." },
    capability_report: "plain string from Tauri",
  };
  const backend = async (cmd, args) => {
    switch (cmd) {
      case "list_datasets": return [];
      case "hardware_info": return HW;
      case "preflight_train": return { params: 1, peakRamMb: 1, ramAvailableMb: 9, ramLimitMb: 9, fitsInRam: true, seconds: 1, llamaCpp: { runs: true, reason: "" }, problems: [] };
      case "estimate_job": return { kind: args.kind, seconds: 5, lowSeconds: 3, highSeconds: 8, threadsUsed: 4, notes: [] };
      case "system_stats": return { cpuTotal: 1, memPercent: 1, ferrumThreads: 4, memUsed: 1, memTotal: 2, perCore: [1] };
      case "run_gguf": await new Promise((r) => { release = r; }); throw { code: "memory", message: "estimated resident memory (3 GB) exceeds 90% of available (2 GB)", hint: "Pick a smaller model." };
      default:
        if (errors[cmd]) throw errors[cmd];
        throw new Error("no fake for " + cmd);
    }
  };
  let written = null;
  const navigator = { clipboard: { writeText: async (s) => { written = s; } } };
  const app = loadApp({ backend, navigator });
  await settle(8);
  const $ = (id) => app.doc.getElementById(id);
  const more = (id) => $(id).nextSibling;
  const btn = (box, label) => box.children.find((c) => c.tagName === "BUTTON" && c.textContent === label);

  // A structured error: message in the line, hint beneath, Copy details next to it.
  $("ggPath").value = "/g.gguf";
  $("ggInspect").dispatch("click"); await settle(6);
  assert.strictEqual($("errGguf").textContent, "cannot open /g.gguf: bad magic");
  const box = more("errGguf");
  assert.ok(box.classList.contains("errmore") && !box.classList.contains("hidden"));
  assert.strictEqual(box.children[0].textContent, "The file may be damaged.");
  assert.ok(btn(box, "Show full message").classList.contains("hidden"), "a short message needs no expander");
  btn(box, "Copy details").dispatch("click"); await settle(6);
  assert.ok(/^Ferrum SLM Studio error\ncode: model\nmessage: cannot open \/g.gguf: bad magic\nhint: The file may be damaged\.\ntime: /.test(written), written);
  assert.ok(Array.from($("toasts").children).some((t) => t.textContent === "Error details copied"));
  // The error toast carries the message only (no "[object Object]").
  assert.ok(Array.from($("toasts").children).some((t) => t.classList.contains("err") && t.textContent === "Inspect failed: cannot open /g.gguf: bad magic"));

  // Clearing the line (a new attempt) hides the companion and empties the hint.
  $("ggPath").value = "";
  $("ggInspect").dispatch("click"); await settle(4);
  assert.strictEqual($("errGguf").textContent, "GGUF file is required");
  assert.strictEqual(box.children[0].textContent, "", "a plain Error has no hint");

  // A long message: cut with an ellipsis, the rest behind "Show full message".
  $("miPath").value = "/m.bin";
  $("miInspect").dispatch("click"); await settle(6);
  const mbox = more("errModel");
  assert.strictEqual($("errModel").textContent, "x".repeat(300) + "…");
  const expand = btn(mbox, "Show full message");
  assert.ok(!expand.classList.contains("hidden"));
  expand.dispatch("click");
  assert.strictEqual($("errModel").textContent, LONG);
  assert.strictEqual(expand.textContent, "Show less");
  expand.dispatch("click");
  assert.strictEqual($("errModel").textContent.length, 301);
  // The toast is cut too.
  const errToasts = Array.from($("toasts").children).filter((t) => t.classList.contains("err"));
  assert.ok(errToasts.some((t) => t.textContent.length <= 241 && t.textContent.endsWith("…")));
  // A toast leaves when its timer fires: 15 s for an error, 5 s for the rest.
  assert.deepStrictEqual([...new Set(app.timeouts.map((t) => t.ms))].sort((x, y) => x - y), [5000, 15000]);
  const before = $("toasts").children.length;
  app.timeouts.forEach((t) => t.f());
  assert.strictEqual($("toasts").children.length, before - app.timeouts.length);

  // Copy falls back when the clipboard API is missing, and says so when nothing works.
  const bare = loadApp({ backend, navigator: undefined });
  await settle(8);
  bare.doc.execCommand = () => false;
  const b$ = (id) => bare.doc.getElementById(id);
  b$("miPath").value = "/m.bin";
  b$("miInspect").dispatch("click"); await settle(6);
  const bb = b$("errModel").nextSibling.children.find((c) => c.textContent === "Copy details");
  bb.dispatch("click"); await settle(6);
  assert.ok(Array.from(b$("toasts").children).some((t) => t.classList.contains("err") && /Could not copy/.test(t.textContent)));

  // A plain string from Tauri itself is shown as is, with no hint.
  $("capCheck").dispatch("click"); await settle(6);
  assert.strictEqual($("capStatus").textContent, "error");
  assert.ok(Array.from($("toasts").children).some((t) => t.textContent === "Capability check failed: plain string from Tauri"));

  // The busy spinner shows while run_gguf runs and goes away with the error.
  $("ggPath").value = "/g.gguf"; $("ggPrompt").value = "hi";
  assert.ok($("ggSpin").classList.contains("hidden"));
  $("ggRun").dispatch("click"); await settle(6);
  assert.ok(!$("ggSpin").classList.contains("hidden"), "spinner while the model loads and runs");
  assert.ok($("ggRun").disabled);
  release(); await settle(8);
  assert.ok($("ggSpin").classList.contains("hidden"));
  assert.strictEqual(app.doc.getElementById("errGguf").textContent, "estimated resident memory (3 GB) exceeds 90% of available (2 GB)");
  assert.strictEqual(more("errGguf").children[0].textContent, "Pick a smaller model.");

  // The Capable check shows a spinner too.
  assert.ok($("capSpin").classList.contains("hidden"));

  console.log("errors.test.js: all assertions passed");
})().catch((e) => { console.error(e); process.exit(1); });
