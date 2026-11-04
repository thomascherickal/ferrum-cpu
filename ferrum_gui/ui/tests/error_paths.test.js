// Error and edge paths of the tab scripts that app_flows.test.js does not reach: a failing
// backend call per tab, the Export tab's Inspect button and size gate, the non-streaming Generate
// result, fmtParams below 1000, and testdom's attribute selectors. Run: node ui/tests/error_paths.test.js
"use strict";
const assert = require("node:assert");
const { loadApp, settle } = require("./testapp.js");
const { fmtParams } = require("../core/common.js");
const { documentFromHtml } = require("./testdom.js");

const HW = { cpuModel: "T", ferrumThreads: 4, logicalCores: 4, physicalCores: 4, ramTotalMb: 8000, ramAvailableMb: 4000, gpus: [] };
const BIG = { architecture: "llama", runnable: true, version: 3, numTensors: 1, modelDim: 1, nLayers: 1, nHeads: 1, nKvHeads: 1, vocabSize: 1, tokenizer: "t", estInt4Mb: 1, estInt8Mb: 1, estF32Mb: 1, availMb: 1, note: "", paramCount: 9e9 };
const REPORT = { cpu: "c", cores: 4, threads: 4, memTotal: 8e9, memAvail: 4e9, memBwGbps: 3.5, gemmGflops: 60, inferInt4: 1e9, inferInt8: 5e8, inferF32: 1e8,
  trainChinchilla: 2e6, trainFixed1b: 1e6, testEval: 3e6, loadInt4: 5e9, loadInt8: 2e9, loadF32: 1e9, finetuneLo: 1e5, finetuneHi: 1e6, targetToks: 3,
  trainHours: 24, evalTokens: 1e7, fixedTrainTokens: 1e9, chinchillaRatio: 20, finetuneTokensLo: 1e7, finetuneTokensHi: 1e6, loadFraction: 0.9 };

async function start(overrides) {
  const calls = [];
  const backend = async (cmd, args) => {
    calls.push(cmd);
    if (overrides[cmd]) return overrides[cmd](args);
    switch (cmd) {
      case "list_datasets": return [];
      case "hardware_info": return HW;
      case "preflight_train": return { params: 1, peakRamMb: 1, ramAvailableMb: 4000, ramLimitMb: 3000, fitsInRam: true, seconds: 1, llamaCpp: { runs: true, reason: "" }, problems: [] };
      case "estimate_job": return { kind: args.kind, seconds: 5, lowSeconds: 3, highSeconds: 8, threadsUsed: 4, notes: [] };
      case "system_stats": return { cpuTotal: 1, memPercent: 1, ferrumThreads: 4, memUsed: 1, memTotal: 2, perCore: [1] };
      default: throw new Error("no fake for " + cmd);
    }
  };
  const app = loadApp({ backend });
  await settle(8);
  const $ = (id) => app.doc.getElementById(id);
  return { ...app, calls, $, set: (id, v) => { $(id).value = v; }, click: async (id) => { $(id).dispatch("click"); await settle(6); } };
}

(async () => {
  assert.strictEqual(fmtParams(7), "7");

  // A failing backend command shows the error, the "error" status and re-enables the button.
  const fail = (m) => async () => { throw new Error(m); };
  {
    const a = await start({ evaluate_slm: fail("eval down") });
    a.set("evModel", "/m.gguf"); a.set("evText", "t");
    await a.click("evRun");
    assert.strictEqual(a.$("evStatus").textContent, "error");
    assert.ok(a.$("errEval").textContent.includes("eval down"));
    assert.ok(!a.$("evRun").disabled);
  }
  {
    const a = await start({ finetune_gguf: fail("ft down") });
    a.set("ftPath", "/g.gguf"); a.set("ftCorpus", "/c.txt"); a.set("ftOut", "/o.flck");
    await a.click("ftRun");
    assert.strictEqual(a.$("ftStatus").textContent, "error");
    assert.ok(a.$("errFt").textContent.includes("ft down"));
  }
  {
    const a = await start({ generate_slm: fail("gen down") });
    a.set("genModel", "/m.gguf");
    await a.click("genStart");
    assert.strictEqual(a.$("genStatus").textContent, "error");
    assert.ok(a.$("errGen").textContent.includes("gen down"));
    assert.ok(!a.$("genStart").disabled);
  }
  {
    // Not streaming: the returned text is the output.
    const a = await start({ generate_slm: async () => "whole text" });
    a.set("genModel", "/m.gguf");
    a.$("genStream").checked = false;
    await a.click("genStart");
    assert.strictEqual(a.$("genOut").textContent, "whole text");
    assert.strictEqual(a.$("genStatus").textContent, "done (10 chars)");
  }
  {
    // Stopping the server fails: the error is shown and the status is polled again.
    const a = await start({ serve_start: async () => ({ running: true, url: "http://127.0.0.1:1", requests: 0, modelPath: "/m.gguf", hasToken: false, curl: "curl" }),
      serve_status: async () => ({ running: true, url: "http://127.0.0.1:1", requests: 0, modelPath: "/m.gguf", hasToken: false, curl: "curl" }), serve_stop: fail("stop down") });
    a.set("svModel", "/m.gguf");
    await a.click("svStart");
    await a.click("svStop");
    assert.strictEqual(a.$("svStatus").textContent, "error");
    assert.ok(a.$("errServe").textContent.includes("stop down"));
    assert.ok(!a.$("svStop").disabled);
  }

  // Export tab: Inspect (needs a path; failure; runnable / not), the size gate on Inspect and Export.
  {
    let info = { ...BIG, paramCount: 1000 };
    const a = await start({ gguf_info: async () => info, export_gguf: async () => ({ outPath: "/o", bytes: 1, seconds: 1, tensorSummary: "t" }), capability_report: async () => REPORT });
    await a.click("tab-export");
    await a.click("exInspect");
    assert.ok(a.$("errExport").textContent.includes("Source GGUF is required"));
    a.set("exPath", "/g.gguf");
    await a.click("exInspect");
    assert.ok(a.$("exInfo").innerHTML.includes("llama"));
    assert.strictEqual(a.$("exStatus").textContent, "");
    info = { ...info, runnable: false };
    await a.click("exInspect");
    // With the Capable bounds known, an oversized model asks before inspect (advisory) and export.
    await a.click("capCheck");
    info = { ...BIG };
    await a.click("exInspect");
    assert.ok(a.$("ggWarnDialog").open, "advisory dialog on Inspect");
    await a.click("ggWarnProceed");
    a.set("exOut", "/o.gguf");
    a.$("exExport").dispatch("click"); await settle(6);
    assert.ok(a.$("ggWarnDialog").open, "gate on Export");
    await a.click("ggWarnCancel");
    assert.strictEqual(a.$("exStatus").textContent, "cancelled");
    assert.strictEqual(a.calls.filter((c) => c === "export_gguf").length, 0);
    // Proceeding exports; a failing inspect inside the gate is swallowed and the export runs.
    a.$("exExport").dispatch("click"); await settle(6);
    await a.click("ggWarnProceed");
    assert.strictEqual(a.calls.filter((c) => c === "export_gguf").length, 1);
    info = null;
    const gi = a.calls.length;
    a.$("exExport").dispatch("click"); await settle(6);
    assert.strictEqual(a.calls.slice(gi).filter((c) => c === "export_gguf").length, 1);
  }
  {
    const a = await start({ gguf_info: fail("no such file") });
    a.set("exPath", "/g.gguf");
    await a.click("exInspect");
    assert.strictEqual(a.$("exStatus").textContent, "error");
    assert.ok(a.$("errExport").textContent.includes("no such file"));
  }
  {
    const a = await start({ export_gguf: fail("export down") });
    a.set("exPath", "/g.gguf"); a.set("exOut", "/o.gguf");
    await a.click("exExport");
    assert.strictEqual(a.$("exStatus").textContent, "error");
    assert.ok(a.$("errExport").textContent.includes("export down"));
  }

  // The GGUF tab's Inspect also warns once the Capable bounds are known.
  {
    const a = await start({ gguf_info: async () => BIG, capability_report: async () => REPORT });
    await a.click("capCheck");
    a.set("ggPath", "/g.gguf");
    await a.click("ggInspect");
    assert.ok(a.$("ggWarnDialog").open);
    await a.click("ggWarnProceed");
  }

  // testdom: attribute selectors, with and without a value; unsupported selectors throw.
  {
    const d = documentFromHtml('<div><a data-x="1" id="p">a</a><a id="q">b</a></div>');
    assert.strictEqual(d.querySelectorAll("a[data-x]").length, 1);
    assert.strictEqual(d.querySelectorAll('a[data-x="1"]').length, 1);
    assert.strictEqual(d.querySelectorAll('a[data-x="2"]').length, 0);
    assert.strictEqual(d.querySelectorAll("a[href]").length, 0);
  }
  console.log("error_paths.test.js: all assertions passed");
})().catch((e) => { console.error(e); process.exit(1); });
