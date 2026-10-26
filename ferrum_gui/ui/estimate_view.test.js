// Node test for estimate_view.js. Run with: node ui/estimate_view.test.js
"use strict";
const assert = require("node:assert");
const V = require("./estimate_view.js");

// ── durations ───────────────────────────────────────────────────────────────
const f = V.formatDuration;
assert.strictEqual(f(0.2), "under 1 second");
assert.strictEqual(f(1), "about 1 second");
assert.strictEqual(f(45), "about 45 seconds");
assert.strictEqual(f(59.7), "about 1 min", "rounds up into the next unit, never '60 seconds'");
assert.strictEqual(f(240), "about 4 min");
assert.strictEqual(f(270), "about 4.5 min");
assert.strictEqual(f(1500), "about 25 min");
assert.strictEqual(f(3590), "about 1 h");
assert.strictEqual(f(9000), "about 2.5 h");
assert.strictEqual(f(86400 * 9), "about 9 days");
assert.strictEqual(f(86400 * 800), "about 2 years");
for (const bad of [null, undefined, NaN, -5, Infinity, "3"]) assert.strictEqual(f(bad), "unknown", String(bad));
// Monotone: a longer time never prints a smaller number in the same unit family.
let last = 0;
for (let s = 0.5; s < 1e9; s *= 1.37) {
  const d = V.duration(s);
  const factor = { "under 1 second": 0, second: 1, seconds: 1, min: 60, h: 3600, days: 86400, years: 365 * 86400 }[d.unit];
  const approx = (Number(d.value) || 0) * factor;
  assert.ok(approx >= last * 0.9, `${s}: ${d.value} ${d.unit} after ${last}`);
  last = approx;
}

assert.strictEqual(V.formatRange(120, 360), "2 to 6 min");
assert.strictEqual(V.formatRange(50, 150), "50 seconds to 2.5 min");
assert.strictEqual(V.formatRange(0.1, 3), "under 1 second to 3 seconds");
assert.strictEqual(V.formatRange(0.1, 0.4), "");
assert.strictEqual(V.formatRange(null, 5), "");

assert.strictEqual(V.formatMb(450.4), "450 MB");
assert.strictEqual(V.formatMb(11468.8), "11.2 GB");
assert.strictEqual(V.formatMb(undefined), "unknown");

// ── hardware line ───────────────────────────────────────────────────────────
// The JSON this project's dev machine really returned from hardware_info.
const real = {
  cpuModel: "11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz", ferrumThreads: 8, logicalCores: 8, physicalCores: 4,
  ramAvailableMb: 5502, ramTotalMb: 15716,
  gpus: [
    { cores: null, memoryMb: 2048, name: "NVIDIA GeForce MX330", usedByFerrum: false },
    { cores: null, memoryMb: null, name: "Intel GPU (device 0x9a49)", usedByFerrum: false },
  ],
};
assert.strictEqual(
  V.hardwareLine(real),
  "8 CPU threads (4 cores) · RAM 15.3 GB total, 5.4 GB free · GPU: NVIDIA GeForce MX330 (2.0 GB memory, cores unknown); Intel GPU (device 0x9a49) (memory unknown, cores unknown) - detected, not used by Ferrum",
);
assert.strictEqual(
  V.hardwareLine({ logicalCores: 1, physicalCores: 1, ramTotalMb: null, ramAvailableMb: null, gpus: [] }),
  "1 CPU thread · RAM unknown · no GPU detected (Ferrum is CPU-only)",
);
assert.ok(V.hardwareLine({ logicalCores: 4, ramTotalMb: 8000, ramAvailableMb: null, gpus: [{ name: "X", memoryMb: 4096, cores: 1536 }] })
  .includes("X (4.0 GB memory, 1536 cores)"));
assert.strictEqual(V.hardwareLine(null), "Detecting your hardware...");

// ── estimate sections ───────────────────────────────────────────────────────
const hw = { ramAvailableMb: 11468.8, gpus: [] };
const train = {
  kind: "train", seconds: 240, lowSeconds: 150, highSeconds: 390, threadsUsed: 8, peakRamMb: 450, ramLimitMb: 8600,
  ramAvailableMb: 11468.8, fitsInRam: true, usesGpu: false,
  notes: ["Rates measured on a 4-core Core i5-1135G7; run the Capable check to scale them to this machine.",
    "500000 parameters, about 1000000 tokens trained.", "GPU: not used by Ferrum (CPU only)."],
};
let secs = V.estimateSections(train, hw);
assert.deepStrictEqual(secs[0], {
  label: "Estimated time",
  text: "about 4 min on your 8 CPU threads (likely 2.5 to 6.5 min); 11.2 GB free RAM; GPU not used",
});
assert.deepStrictEqual(secs[1], { label: "Memory", text: "needs about 450 MB RAM (you have 11.2 GB free).", tone: undefined });
assert.ok(secs[2].text.startsWith("Rates measured on") && secs[2].text.includes("500000 parameters") && secs[2].text.endsWith("not a promise."));
assert.ok(!secs[2].text.includes("GPU: not used"), "boilerplate GPU note is not repeated");

// A GPU on the machine is mentioned honestly.
secs = V.estimateSections(train, { ...hw, gpus: [{ name: "G" }] });
assert.ok(secs[0].text.endsWith("GPU detected but not used by Ferrum"));

// Too little RAM: warning tone and a plain-language consequence.
secs = V.estimateSections({ ...train, peakRamMb: 9000, ramLimitMb: 8600, fitsInRam: false }, hw);
assert.strictEqual(secs[1].tone, "bad");
assert.ok(/more than it may use/.test(secs[1].text) && secs[1].text.includes("8.4 GB"), secs[1].text);

// Instant, unknown time, missing RAM info.
assert.deepStrictEqual(V.estimateSections({ kind: "instant", seconds: 0, notes: [] }, hw), [{ label: "Estimated time", text: "instant" }]);
secs = V.estimateSections({ kind: "train", seconds: null, notes: ["Rates measured on x", "Fill in the model size to get an estimate."] }, hw);
assert.strictEqual(secs[0].tone, "warn");
assert.ok(secs[0].text.includes("Fill in the model size"));
secs = V.estimateSections({ kind: "gguf_run", seconds: 0.4, lowSeconds: 0.2, highSeconds: 1, peakRamMb: 100, fitsInRam: true, notes: [] }, { gpus: [] });
assert.ok(secs[0].text.startsWith("under 1 second;") && secs[0].text.includes("free RAM unknown"), secs[0].text);
assert.ok(secs[1].text.includes("free RAM unknown"));
assert.deepStrictEqual(V.estimateSections(null, hw), []);

console.log("estimate_view.test.js: all assertions passed");
