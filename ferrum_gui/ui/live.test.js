// Functional test of the tooltip + estimate wiring (live.js) against the REAL
// index.html structure (parsed into a stub DOM) and a fake Tauri backend.
// Run with: node ui/live.test.js
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { documentFromHtml } = require("./testdom.js");
const help = require("./help.js");
const tooltip = require("./tooltip.js");
const view = require("./estimate_view.js");
const Live = require("./live.js");

const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const settle = () => new Promise((r) => setImmediate(r));

const HW = {
  cpuModel: "Test CPU", ferrumThreads: 8, logicalCores: 8, physicalCores: 4, ramTotalMb: 16000, ramAvailableMb: 11468,
  gpus: [{ name: "Test GPU", memoryMb: 4096, cores: null, usedByFerrum: false }],
};

function build(overrides = {}) {
  const doc = documentFromHtml(html);
  for (const id of ["hwLine", "hwSystem"]) { // the page has these spans; the stub parser only builds controls
    const e = doc.createElement("span"); e.id = id; doc.body.appendChild(e);
  }
  const timers = []; // fake setTimeout queue
  const intervals = [];
  const calls = [];
  const state = { hw: { ...HW }, fail: null, estimate: null };
  const invoke = async (cmd, args) => {
    calls.push({ cmd, args: JSON.parse(JSON.stringify(args || {})) });
    if (state.fail === cmd) throw new Error(state.failMsg || "boom");
    if (cmd === "hardware_info") return state.hw;
    if (cmd === "estimate_job") {
      if (state.estimate) return state.estimate(args);
      const secs = 100 * (args.params.epochs || 1);
      return { kind: args.kind, seconds: secs, lowSeconds: secs * 0.6, highSeconds: secs * 1.6, peakRamMb: 450, ramLimitMb: 8000,
        ramAvailableMb: state.hw.ramAvailableMb, fitsInRam: true, threadsUsed: 8, usesGpu: false,
        notes: ["Rates measured on a 4-core Core i5-1135G7; run the Capable check to scale them to this machine."] };
    }
    throw new Error("unexpected command " + cmd);
  };
  const live = Live.create({
    doc, invoke, help, tooltip, view,
    setTimeout: (f, ms) => { timers.push({ f, ms, live: true }); return timers.length; },
    clearTimeout: (id) => { if (id && timers[id - 1]) timers[id - 1].live = false; },
    setInterval: (f, ms) => { intervals.push({ f, ms }); return intervals.length; },
    tooltipEnv: { document: doc, viewport: () => ({ width: 1300, height: 900 }), setTimeout: (f) => f(), clearTimeout() {} },
    ...overrides,
  });
  doc.getElementById("ferrum-tooltip") || null;
  const pending = () => timers.filter((t) => t.live);
  const flushTimers = async () => { const t = pending(); timers.length = 0; t.forEach((x) => x.f()); await settle(); await settle(); };
  const estimateCalls = () => calls.filter((c) => c.cmd === "estimate_job");
  return { doc, live, calls, state, timers, intervals, pending, flushTimers, estimateCalls };
}
const bubbleOf = (doc) => doc.getElementById("ferrum-tooltip");
// What app.js's tab handler does, then the click that live.js listens to.
const gotoTab = (doc, name, from = "datasets") => {
  doc.getElementById("panel-" + from).classList.remove("active");
  doc.getElementById("panel-" + name).classList.add("active");
  doc.getElementById("tab-" + name).dispatch("click");
};
const show = (doc, live, id) => { const el = doc.getElementById(id); el.rect = { left: 100, top: 100, right: 220, bottom: 130, width: 120, height: 30 }; el.dispatch("focus"); return bubbleOf(doc).textContent; };

(async () => {
  // 0. The parameter objects live.js builds from the default form values are the
  // contract with the Rust `JobParams`; src/../tests/ui_contract.rs checks the same
  // file field by field. Regenerate with UPDATE_FIXTURE=1 after changing PARAMS.
  {
    const { doc } = build();
    const current = {};
    for (const id of Object.keys(Live.PARAMS).sort()) {
      current[id] = { kind: help.helpFor(id).kind, params: Live.PARAMS[id](doc) };
    }
    const file = path.join(__dirname, "live_params.fixture.json");
    if (process.env.UPDATE_FIXTURE) fs.writeFileSync(file, JSON.stringify(current, null, 2) + "\n");
    assert.deepStrictEqual(JSON.parse(fs.readFileSync(file, "utf8")), current, "live_params.fixture.json is stale; run UPDATE_FIXTURE=1 node ui/live.test.js");
  }

  // 1. Startup: every control with a help entry gets a tooltip description; hardware line is filled.
  {
    const { doc, live, calls, intervals, estimateCalls } = build();
    await live.start();
    const controls = doc.querySelectorAll("input,textarea,select,button");
    const withHelp = controls.filter((c) => help.helpFor(c.id));
    assert.strictEqual(withHelp.length, controls.length, "every control has help");
    for (const c of withHelp) {
      const d = doc.getElementById(c.getAttribute("aria-describedby"));
      assert.ok(d && d.textContent.length > 40, `#${c.id} has a screen-reader description`);
    }
    assert.strictEqual(doc.getElementById("hwLine").textContent, view.hardwareLine(HW));
    assert.strictEqual(doc.getElementById("hwSystem").textContent, view.hardwareLine(HW));
    assert.ok(doc.getElementById("hwLine").textContent.includes("Test GPU") && /not used by Ferrum/.test(doc.getElementById("hwLine").textContent));
    assert.deepStrictEqual(intervals.map((i) => i.ms), [5000], "hardware polled every 5 s");
    assert.strictEqual(calls[0].cmd, "hardware_info");
    // Visible tab = Datasets: its non-instant buttons were estimated, instant ones were not.
    const kinds = estimateCalls().map((c) => c.args.kind).sort();
    assert.deepStrictEqual(kinds, ["clean_text", "download", "download", "download", "load_file"]);
    assert.ok(!estimateCalls().some((c) => c.args.kind === "instant"));
  }

  // 1b. The Prepare tab: its button asks for a "prepare" estimate with the input list (one path per
  // line, blanks dropped) and the output folder, and re-asks when the list changes.
  {
    const { doc, live, flushTimers, estimateCalls } = build();
    await live.start();
    gotoTab(doc, "prepare");
    await flushTimers();
    const first = estimateCalls().filter((c) => c.args.kind === "prepare");
    assert.strictEqual(first.length, 1);
    assert.deepStrictEqual(first[0].args.params, { inputPaths: [], outDir: "" });
    doc.getElementById("dpInputs").value = "/data/a.txt\n\n  /data/dir \n";
    doc.getElementById("dpOut").value = "/out";
    doc.getElementById("dpInputs").dispatch("input");
    await flushTimers();
    const last = estimateCalls().filter((c) => c.args.kind === "prepare").pop();
    assert.deepStrictEqual(last.args.params, { inputPaths: ["/data/a.txt", "/data/dir"], outDir: "/out" });
    const text = show(doc, live, "dpRun");
    assert.ok(text.startsWith("Prepare") && text.includes("Estimated time: about"), text);
    // The Stop and "use the splits" buttons are instant: no estimate is requested for them.
    assert.ok(!estimateCalls().some((c) => c.args.kind === "instant"));
  }

  // 2. A button tooltip says what it does AND an estimate; fields only help text.
  {
    const { doc, live, flushTimers, estimateCalls, calls } = build();
    await live.start();
    doc.getElementById("tab-train").dispatch("click");
    // app.js's own tab handler switches the panel; emulate it.
    doc.getElementById("panel-datasets").classList.remove("active");
    doc.getElementById("panel-train").classList.add("active");
    await flushTimers();
    const train = estimateCalls().filter((c) => c.args.kind === "train");
    assert.strictEqual(train.length, 1);
    assert.deepStrictEqual(train[0].args.params, {
      corpusPath: "", contextLen: 16, embedDim: 32, numHeads: 4, numBlocks: 2, hiddenDim: 64, epochs: 100, batchSize: 16,
      vocabSize: 512, threads: 0, windowStride: 0, maxTokens: 0, valFraction: 0,
    });
    let text = show(doc, live, "trStart");
    assert.ok(text.startsWith("Train"), text);
    assert.ok(text.includes("What it does: Trains a new transformer"), text);
    assert.ok(text.includes("Estimated time: about 3 h on your 8 CPU threads"), text); // 10000 s
    assert.ok(text.includes("(likely"), text);
    assert.ok(text.includes("11.2 GB free RAM; GPU detected but not used by Ferrum"), text);
    assert.ok(text.includes("Memory: needs about 450 MB RAM (you have 11.2 GB free)."), text);
    assert.ok(text.includes("New to AI?"), text);
    // The estimate comes right after "What it does".
    assert.ok(text.indexOf("Estimated time") < text.indexOf("New to AI?") && text.indexOf("What it does") < text.indexOf("Estimated time"));
    // A field shows help text only, no estimate.
    text = show(doc, live, "trEpochs");
    assert.ok(text.includes("What it is:") && !text.includes("Estimated time"), text);
    // An instant button.
    text = show(doc, live, "termClear");
    assert.ok(text.includes("Estimated time: instant"), text);
    assert.strictEqual(calls.filter((c) => c.args && c.args.kind === "instant").length, 0);
  }

  // 3. Changing a field re-estimates after a debounce and refreshes the open tooltip.
  {
    const { doc, live, flushTimers, pending, estimateCalls, state } = build();
    await live.start();
    gotoTab(doc, "train");
    await flushTimers();
    const before = estimateCalls().length;
    let text = show(doc, live, "trStart");
    assert.ok(text.includes("about 3 h"), text);
    const epochs = doc.getElementById("trEpochs");
    for (const v of ["2", "20", "200"]) { epochs.value = v; epochs.dispatch("input"); }
    assert.strictEqual(pending().length, 1, "rapid edits collapse into one pending estimate");
    assert.strictEqual(pending()[0].ms, 300);
    assert.strictEqual(estimateCalls().length, before, "nothing is sent before the debounce fires");
    await flushTimers();
    assert.strictEqual(estimateCalls().length, before + 1);
    assert.strictEqual(estimateCalls().pop().args.params.epochs, 200);
    text = bubbleOf(doc).textContent;
    assert.ok(text.includes("about 5.5 h"), "open tooltip shows the new estimate: " + text); // 20000 s
    // change events (selects, checkboxes) also schedule.
    doc.getElementById("trQat").dispatch("change");
    assert.strictEqual(pending().length, 1);
    // Blank / garbage numbers never become NaN or null.
    epochs.value = ""; doc.getElementById("trLr").value = "abc"; doc.getElementById("trTokens").value = "5e7";
    await flushTimers();
    const p = estimateCalls().pop().args.params;
    assert.strictEqual(p.epochs, 0);
    assert.strictEqual(p.maxTokens, 50000000);
    assert.ok(!JSON.stringify(p).includes("null") && !JSON.stringify(p).includes("NaN"));
    assert.ok(state);
  }

  // 4. Hardware changes: the line and the tooltips follow free RAM; small wobbles do not re-estimate.
  {
    const { doc, live, state, estimateCalls, intervals, flushTimers } = build();
    await live.start();
    gotoTab(doc, "train");
    await flushTimers();
    let n = estimateCalls().length;
    show(doc, live, "trStart");
    state.hw = { ...HW, ramAvailableMb: 11468 + 20 }; // +0.17%: ignored
    await intervals[0].f(); await settle();
    assert.strictEqual(estimateCalls().length, n, "no re-estimate for a tiny RAM change");
    state.hw = { ...HW, ramAvailableMb: 2048, gpus: [] }; // big drop, GPU gone
    await intervals[0].f(); await settle();
    assert.ok(estimateCalls().length > n, "re-estimate after a real change");
    assert.ok(doc.getElementById("hwLine").textContent.includes("2.0 GB free"), doc.getElementById("hwLine").textContent);
    assert.ok(doc.getElementById("hwLine").textContent.includes("no GPU detected"));
    const text = bubbleOf(doc).textContent;
    assert.ok(text.includes("2.0 GB free RAM; GPU not used"), text);
  }

  // 5. Failures are shown, not swallowed or thrown.
  {
    const { doc, live, state, flushTimers } = build();
    state.fail = "hardware_info"; state.failMsg = "no backend";
    await live.start();
    assert.strictEqual(doc.getElementById("hwLine").textContent, "Hardware unavailable (no backend)");
    state.fail = "estimate_job"; state.failMsg = "disk on fire";
    doc.getElementById("panel-datasets").classList.remove("active");
    doc.getElementById("panel-train").classList.add("active");
    await flushTimers(); // tab switch not clicked; trigger explicitly
    await live.estimateVisible();
    assert.ok(show(doc, live, "trStart").includes("Estimated time: not available (disk on fire)"));
  }

  // 6. An older, slower response never overwrites a newer one.
  {
    const slow = [];
    const { doc, live, state, estimateCalls } = build();
    state.estimate = (args) => new Promise((res) => slow.push(() => res({ kind: args.kind, seconds: 10 * (args.params.epochs || 1), notes: [], threadsUsed: 1 })));
    doc.getElementById("panel-datasets").classList.remove("active");
    doc.getElementById("panel-train").classList.add("active");
    const started = live.start(); // blocks on the first (slow) estimate
    await settle();
    assert.strictEqual(slow.length, 1, "train request #1 is in flight");
    doc.getElementById("trEpochs").value = "5";
    const second = live.estimateVisible();
    await settle();
    assert.strictEqual(slow.length, 2);
    slow[1](); // the newest request answers first...
    await settle();
    slow[0](); // ...then the stale one
    await Promise.all([started, second]);
    assert.strictEqual(estimateCalls().length, 2);
    const text = show(doc, live, "trStart");
    assert.ok(text.includes("about 50 seconds"), text); // epochs 5 -> 50 s, not the stale 1000 s
  }

  // 7. Backend text is never parsed as markup.
  {
    const evil = "<img src=x onerror=alert(1)>";
    const { doc, live, state, flushTimers } = build();
    state.hw = { ...HW, gpus: [{ name: evil, memoryMb: 1, cores: 1 }] };
    state.estimate = (a) => ({ kind: a.kind, seconds: null, notes: [evil] });
    doc.getElementById("panel-datasets").classList.remove("active");
    doc.getElementById("panel-train").classList.add("active");
    await live.start();
    show(doc, live, "trStart");
    let imgs = 0;
    for (const root of [bubbleOf(doc), doc.getElementById("hwLine"), doc.getElementById("ferrum-descriptions")]) {
      root.walk((n) => { if (n.tagName === "IMG") imgs++; });
    }
    assert.strictEqual(imgs, 0);
    assert.ok(doc.getElementById("hwLine").textContent.includes(evil));
    assert.ok(bubbleOf(doc).textContent.includes(evil));
    assert.ok(!doc.getElementById("hwLine").innerHTML);
    assert.ok(flushTimers);
  }

  // 8. Dynamically created buttons (dataset catalog) can be attached.
  {
    const { doc, live } = build();
    await live.start();
    const btn = doc.createElement("button");
    btn.dataset.help = "dsCatalogDownload";
    doc.getElementById("panel-datasets").appendChild(btn);
    btn.rect = { left: 10, top: 10, right: 90, bottom: 40, width: 80, height: 30 };
    live.attach(btn);
    assert.ok(btn.getAttribute("aria-describedby"));
    btn.dispatch("focus");
    assert.ok(bubbleOf(doc).textContent.includes("Download (catalog item)"));
    // An element without a help entry is left alone.
    const plain = doc.createElement("button");
    live.attach(plain);
    assert.strictEqual(plain.getAttribute("aria-describedby"), null);
  }

  console.log("live.test.js: all assertions passed");
})().catch((e) => { console.error(e); process.exit(1); });
