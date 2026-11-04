// Smoke test: the real app.js loads against the real index.html, wires the
// tooltips and live estimates, and degrades cleanly without Tauri.
// Run with: node ui/tests/app_smoke.test.js
"use strict";
const assert = require("node:assert");
const { loadApp, settle } = require("./testapp.js");

const HW = { cpuModel: "T", ferrumThreads: 4, logicalCores: 4, physicalCores: 2, ramTotalMb: 8000, ramAvailableMb: 4000, gpus: [] };
const backend = async (cmd, args) => {
  switch (cmd) {
    case "list_datasets": return [
      { id: "a", name: "Evil <b>name</b>", source: "huggingface", approxMb: 5, format: "txt", description: "<img src=x onerror=alert(1)>" },
      { id: "b", name: "Second", source: "kaggle", approxMb: 9, format: "csv", description: "d" },
    ];
    case "hardware_info": return HW;
    case "estimate_job": return { kind: args.kind, seconds: 30, lowSeconds: 18, highSeconds: 48, threadsUsed: 4, notes: [] };
    case "system_stats": return { cpuTotal: 10, memPercent: 20, ferrumThreads: 4, memUsed: 1e9, memTotal: 8e9, perCore: [1, 2] };
    default: throw new Error("unexpected " + cmd);
  }
};

(async () => {
  // With a backend.
  {
    const app = loadApp({ backend });
    await settle(8);
    const { doc, calls } = app;
    assert.ok(doc.getElementById("connDot").classList.contains("ok"));
    assert.ok(doc.getElementById("hwLine").textContent.startsWith("4 CPU threads (2 cores) · RAM 7.8 GB total"), doc.getElementById("hwLine").textContent);
    assert.ok(doc.getElementById("hwLine").textContent.includes("no GPU detected"));
    assert.ok(doc.getElementById("trStart").getAttribute("aria-describedby"));
    assert.strictEqual(doc.getElementById("ferrum-tooltip").getAttribute("role"), "tooltip");
    // Catalog buttons are created at runtime and still get tooltips.
    const buttons = doc.getElementById("dsCatalog").querySelectorAll("button");
    assert.strictEqual(buttons.length, 2);
    for (const b of buttons) {
      assert.strictEqual(b.dataset.help, "dsCatalogDownload");
      assert.ok(b.getAttribute("aria-describedby"));
    }
    // Hardware is polled on a 5 s interval next to the system monitor's.
    assert.ok(app.intervals.some((i) => i.ms === 5000));
    assert.ok(calls.some((c) => c.cmd === "estimate_job" && c.args.kind === "download"));
  }

  // Without Tauri (plain browser): the banner shows, tooltips still work, estimates say why not.
  {
    const app = loadApp({ tauri: false });
    await settle(8);
    const { doc } = app;
    assert.ok(!doc.getElementById("webBanner").classList.contains("hidden"));
    assert.ok(doc.getElementById("trStart").getAttribute("aria-describedby"));
    assert.ok(doc.getElementById("hwLine").textContent.startsWith("Hardware unavailable"));
    const desc = doc.getElementById(doc.getElementById("dsDownload").getAttribute("aria-describedby")).textContent;
    assert.ok(desc.includes("not available (Not running inside the Tauri runtime)"), desc);
  }

  console.log("app_smoke.test.js: all assertions passed");
})().catch((e) => { console.error(e); process.exit(1); });
