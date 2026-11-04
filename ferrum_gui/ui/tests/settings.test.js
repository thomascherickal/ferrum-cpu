// Tests of settings.js (remembered form values, folders and the Evaluate table) and of how the
// page uses it. Storage is a stand-in object; a real webview's localStorage was NOT exercised.
// Run with: node ui/tests/settings.test.js
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const S = require("../core/settings.js");
const { documentFromHtml } = require("./testdom.js");
const { loadApp, settle } = require("./testapp.js");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const memory = (initial = {}) => {
  const m = { data: { ...initial }, sets: 0 };
  m.getItem = (k) => (k in m.data ? m.data[k] : null);
  m.setItem = (k, v) => { m.sets++; m.data[k] = String(v); };
  m.removeItem = (k) => { delete m.data[k]; };
  return m;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const stored = (m) => JSON.parse(m.data[S.KEY]);

// ── parse: junk in, empty state out ──
const EMPTY = { fields: {}, folders: {}, evalRuns: [] };
for (const bad of ["", "{", "null", "[]", "42", '"x"', '{"version":2,"fields":{"a":"b"}}', '{"fields":{"a":"b"}}']) {
  assert.deepStrictEqual(S.parse(bad), EMPTY, bad);
}
{
  const p = S.parse(JSON.stringify({
    version: 1,
    fields: { a: "x", b: true, c: 5, d: null, e: { z: 1 }, f: "y".repeat(5000) },
    folders: { open: "/data", save: 7, dir: "", bogus: "/x" },
    evalRuns: [[{ label: "held-out" }], "junk", [], [1], [{ label: "b" }]],
  }));
  assert.deepStrictEqual(p.fields, { a: "x", b: true });
  assert.deepStrictEqual(p.folders, { open: "/data" });
  assert.deepStrictEqual(p.evalRuns, [[{ label: "held-out" }], [{ label: "b" }]]);
  const many = S.parse(JSON.stringify({ version: 1, evalRuns: Array.from({ length: 80 }, (_, i) => [{ i }]) }));
  assert.strictEqual(many.evalRuns.length, 50);
  assert.strictEqual(many.evalRuns[49][0].i, 79, "keeps the latest runs");
}
assert.strictEqual(S.dirOf("/a/b/c.txt"), "/a/b");
assert.strictEqual(S.dirOf("C:\\data\\c.txt"), "C:\\data");
assert.strictEqual(S.dirOf("/c.txt"), "/");
assert.strictEqual(S.dirOf("c.txt"), "");

// ── What is saved: never contents or secrets ──
{
  const doc = documentFromHtml(html);
  const ids = S.savedFields(doc).map((e) => e.id);
  assert.ok(ids.length >= 90, "only " + ids.length + " saved fields");
  for (const id of ["trCorpus", "trEpochs", "trLr", "genSeed", "genTemp", "ggQuant", "exQuant", "sysOn", "dpInputs", "trCosine", "svModel", "svHost", "svPort", "svMaxTokens"]) assert.ok(ids.includes(id), id);
  for (const id of ["dsRaw", "dsPreview", "evText", "dsHfToken", "svToken", "svShowToken", "trForce", "ggForce", "exForce", "ftForce", "evForce"]) assert.ok(!ids.includes(id), id + " must not be saved");
  // Every field that could hold file contents or a secret is accounted for: a new textarea or
  // token-like field has to be named here (and in settings.js) on purpose.
  const textareas = Array.from(doc.querySelectorAll("textarea")).map((e) => e.id).sort();
  const contentOrSecret = ["dsRaw", "dsPreview", "evText", "rxModelfile"];
  const typedSettings = ["dpInputs", "genSeed", "genStop", "ggPrompt", "ggStop"];
  assert.deepStrictEqual(textareas, [...contentOrSecret, ...typedSettings].sort());
  for (const id of contentOrSecret) assert.ok(S.isNever(id));
  for (const id of ids) assert.ok(!/(token|secret|password|apikey|key)$/i.test(id), id + " looks like a secret");
}

// ── The module on a stub page ──
function page(storage) {
  const doc = documentFromHtml(html);
  const winL = {};
  const win = { addEventListener: (t, f) => (winL[t] = winL[t] || []).push(f) };
  const settings = S.create({ doc, getStorage: () => storage, win });
  return { doc, settings, $: (id) => doc.getElementById(id), pagehide: () => (winL.pagehide || []).forEach((f) => f()), win };
}
(async () => {
  // Round trip: change fields, save, a new page restores them.
  {
    const m = memory();
    const a = page(m);
    a.settings.restore();
    const defaultEpochs = a.$("trEpochs").value;
    a.$("trEpochs").value = "7";
    a.$("trCorpus").value = "/data/train.txt";
    a.$("trCosine").checked = !a.$("trCosine").checked;
    a.$("genStop").value = "END\n---";
    a.$("evText").value = "SECRET HELD-OUT TEXT";
    a.$("dsRaw").value = "WHOLE DOWNLOADED BOOK";
    a.$("dsHfToken").value = "hf_SECRET";
    a.$("trForce").checked = true;
    a.$("exQuant").value = "f16";
    a.$("trEpochs").dispatch("input");
    assert.strictEqual(m.sets, 0, "debounced: nothing written yet");
    await sleep(450);
    assert.strictEqual(m.sets, 1);
    const raw = m.data[S.KEY];
    for (const secret of ["SECRET", "WHOLE", "hf_"]) assert.ok(!raw.includes(secret), secret + " leaked into storage");
    assert.strictEqual(stored(m).version, 1);
    assert.strictEqual(stored(m).fields.trForce, undefined);

    const b = page(m);
    b.settings.restore();
    assert.strictEqual(b.$("trEpochs").value, "7");
    assert.strictEqual(b.$("trCorpus").value, "/data/train.txt");
    assert.strictEqual(b.$("trCosine").checked, a.$("trCosine").checked);
    assert.strictEqual(b.$("genStop").value, "END\n---");
    assert.strictEqual(b.$("exQuant").value, "f16");
    assert.strictEqual(b.$("evText").value, "");
    assert.strictEqual(b.$("dsRaw").value, "");
    assert.strictEqual(b.$("trForce").checked, false);

    // Reset: storage emptied, fields back at their defaults, no re-save of the defaults.
    const sets = m.sets;
    b.settings.reset();
    assert.ok(!(S.KEY in m.data));
    assert.strictEqual(b.$("trEpochs").value, defaultEpochs);
    assert.strictEqual(b.$("trCorpus").value, "");
    await sleep(450);
    assert.strictEqual(m.sets, sets, "reset must not write the defaults back");
    assert.ok(!(S.KEY in m.data));
  }

  // Flush on pagehide, without waiting for the debounce.
  {
    const m = memory();
    const a = page(m);
    a.settings.restore();
    a.$("trEpochs").value = "9";
    a.$("trEpochs").dispatch("change");
    a.pagehide();
    assert.strictEqual(stored(m).fields.trEpochs, "9");
  }

  // Folders: kept per dialog kind, only well-formed ones, and saved.
  {
    const m = memory();
    const a = page(m);
    a.settings.restore();
    assert.strictEqual(a.settings.folder("open"), undefined);
    a.settings.setFolder("open", "/home/me/data");
    a.settings.setFolder("save", "");
    a.settings.setFolder("bogus", "/x");
    a.settings.setFolder("dir", 5);
    a.pagehide();
    assert.deepStrictEqual(stored(m).folders, { open: "/home/me/data" });
    assert.strictEqual(page(m).settings.folder("open"), "/home/me/data");
  }

  // Bad stored values are ignored field by field; a select keeps its default for an unknown option.
  {
    const m = memory({
      [S.KEY]: JSON.stringify({ version: 1, fields: { trEpochs: 5, trCosine: "yes", exQuant: "no-such-type", trCorpus: "/ok.txt", gone: "x" } }),
    });
    const a = page(m);
    const defQuant = a.$("exQuant").value, defEpochs = a.$("trEpochs").value, defCos = a.$("trCosine").checked;
    a.settings.restore();
    assert.strictEqual(a.$("trEpochs").value, defEpochs);
    assert.strictEqual(a.$("trCosine").checked, defCos);
    assert.strictEqual(a.$("exQuant").value, defQuant);
    assert.strictEqual(a.$("trCorpus").value, "/ok.txt");
  }

  // Corrupted JSON and storage that throws: no exception, defaults, and saving does not fail.
  {
    const bad = memory({ [S.KEY]: "{not json" });
    const a = page(bad);
    const def = a.$("trEpochs").value;
    a.settings.restore();
    assert.strictEqual(a.$("trEpochs").value, def);
    a.$("trEpochs").value = "3"; a.settings.save();
    assert.strictEqual(stored(bad).fields.trEpochs, "3", "a good save replaces the junk");

    const throws = { getItem() { throw new Error("denied"); }, setItem() { throw new Error("quota"); }, removeItem() { throw new Error("denied"); } };
    const b = page(throws);
    b.settings.restore();
    b.$("trEpochs").value = "4"; b.settings.save(); b.settings.setFolder("open", "/x"); b.settings.addEvalRun([{ label: "x" }]);
    b.settings.reset();
    assert.strictEqual(b.$("trEpochs").value, def);

    const none = page(null); // no storage at all
    none.settings.restore(); none.settings.save(); none.settings.reset();
    const blocked = page(undefined);
    blocked.settings.restore();
    const getter = S.create({ doc: documentFromHtml(html), getStorage() { throw new Error("SecurityError"); } });
    getter.restore(); getter.save(); getter.reset();
    assert.deepStrictEqual(getter.evalRuns(), []);
  }

  // ── The real page ──
  const EVAL = [{ label: "held-out", model: "m.gguf", numPredictions: 1234, crossEntropy: 1.5, bitsPerToken: 2.16, bitsPerByte: 1, perplexity: 4.5, uniformBaseline: 300, seconds: 1, report: { x: 1 } }];
  const backend = async (cmd, args) => {
    switch (cmd) {
      case "list_datasets": return [];
      case "hardware_info": return { cpuModel: "T", ferrumThreads: 4, logicalCores: 4, physicalCores: 4, ramTotalMb: 8000, ramAvailableMb: 4000, gpus: [] };
      case "preflight_train": return { params: 1, peakRamMb: 1, ramAvailableMb: 9, ramLimitMb: 9, fitsInRam: true, seconds: 1, llamaCpp: { runs: true, reason: "" }, problems: [] };
      case "estimate_job": return { kind: args.kind, seconds: 5, lowSeconds: 3, highSeconds: 8, threadsUsed: 4, notes: [] };
      case "system_stats": return { cpuTotal: 1, memPercent: 1, ferrumThreads: 4, memUsed: 1, memTotal: 2, perCore: [1] };
      case "evaluate_slm": return EVAL;
      default: throw new Error("no fake for " + cmd);
    }
  };
  {
    const m = memory();
    const a = loadApp({ backend, storage: m, dialogs: { open: "/home/me/corpora/a.txt", dir: "/home/me/out", save: "/home/me/o/x.json" } });
    await settle(8);
    const $ = (id) => a.doc.getElementById(id);
    // Editing a field is remembered.
    $("trEpochs").value = "12"; $("trEpochs").dispatch("input");
    $("evModel").value = "/m/x.gguf"; $("evText").value = "held out words"; $("evCtx").value = "0";
    $("evRun").dispatch("click"); await settle(6);
    assert.strictEqual($("evTable").querySelector("tbody").children.length, 1);
    $("dsFileBrowse").dispatch("click"); await settle(4);
    assert.strictEqual($("dsFile").value, "/home/me/corpora/a.txt");
    a.fireWindow("pagehide");
    const saved = stored(m);
    assert.strictEqual(saved.fields.trEpochs, "12");
    assert.strictEqual(saved.fields.evText, undefined);
    assert.strictEqual(saved.folders.open, "/home/me/corpora");
    assert.strictEqual(saved.evalRuns.length, 1);

    // A new start: fields, the table and the file dialog's starting folder come back.
    let dialogOpts = null;
    const b = loadApp({ backend, storage: m, dialogs: { open: "/home/me/corpora/b.txt" } });
    b.win.__TAURI__.dialog.open = async (o) => { dialogOpts = o; return "/home/me/corpora/b.txt"; };
    await settle(8);
    const $b = (id) => b.doc.getElementById(id);
    assert.strictEqual($b("trEpochs").value, "12");
    assert.strictEqual($b("evModel").value, "/m/x.gguf");
    assert.strictEqual($b("evText").value, "");
    assert.strictEqual($b("evTable").querySelector("tbody").children.length, 1);
    assert.ok($b("evTable").querySelector("tbody").children[0].innerHTML.includes("held-out"));
    // The estimates and pre-flight were asked with the restored values, not the defaults.
    const pf = b.calls.filter((c) => c.cmd === "preflight_train");
    assert.ok(pf.length && pf[0].args.params.epochs === 12, JSON.stringify(pf[0] && pf[0].args));
    $b("dsFileBrowse").dispatch("click"); await settle(4);
    assert.deepStrictEqual(JSON.parse(JSON.stringify(dialogOpts)), { multiple: false, directory: false, defaultPath: "/home/me/corpora" });

    // Save JSON after a restart works for the restored run.
    $b("evJsonPath").value = "/o/r.json";
    b.calls.length = 0;
    // (save_corpus has no fake: it must at least be attempted with the restored report)
    $b("evSaveJson").dispatch("click"); await settle(4);
    assert.ok(b.calls.some((c) => c.cmd === "save_corpus" && JSON.parse(c.args.text).x === 1));

    // Reset settings: link wipes the memory, the form and the table.
    $b("resetSettings").dispatch("click");
    assert.ok(Array.from($b("toasts").children).some((t) => /Settings reset/.test(t.textContent)));
    await settle(4);
    assert.strictEqual($b("trEpochs").value, "100");
    assert.strictEqual($b("evTable").querySelector("tbody").children.length, 0);
    assert.ok(!(S.KEY in m.data));
    // Clear table forgets the stored runs too.
    const mc = memory({ [S.KEY]: JSON.stringify({ version: 1, evalRuns: [EVAL] }) });
    const c = loadApp({ backend, storage: mc });
    await settle(8);
    assert.strictEqual(c.doc.getElementById("evTable").querySelector("tbody").children.length, 1);
    c.doc.getElementById("evClear").dispatch("click");
    assert.strictEqual(c.doc.getElementById("evTable").querySelector("tbody").children.length, 0);
    c.fireWindow("pagehide");
    assert.deepStrictEqual(stored(mc).evalRuns, []);
  }

  // A page with blocked or absent storage still starts and works.
  for (const storage of ["blocked", undefined]) {
    const a = loadApp({ backend, storage });
    await settle(8);
    const $ = (id) => a.doc.getElementById(id);
    assert.ok($("connDot").classList.contains("ok"));
    $("trEpochs").value = "3"; $("trEpochs").dispatch("input");
    a.fireWindow("pagehide");
    $("resetSettings").dispatch("click"); await settle(3);
    assert.strictEqual($("trEpochs").value, "100");
  }

  console.log("settings.test.js: all assertions passed");
})().catch((e) => { console.error(e); process.exit(1); });
