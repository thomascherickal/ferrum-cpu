// Node test for the HTML escaping helpers and card/row builders. Run with:
//   node ui/escape.test.js
"use strict";
const assert = require("node:assert");
const { esc, card, catalogMeta, evalRow, modelInfo, ggufInfo, exportInfo, preflightView, preflightWhy } = require("./escape.js");

const EVIL = [
  "<img src=x onerror=alert(1)>",
  '"><script>alert(1)</script>',
  "</script><svg onload=alert(1)>",
  `a & b "q" 'q' <b>`,
];

// esc(): exact output, null-safety, numbers.
assert.strictEqual(esc(`<a href="x">'&'</a>`), "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;");
assert.strictEqual(esc(null), "");
assert.strictEqual(esc(undefined), "");
assert.strictEqual(esc(42), "42");
assert.strictEqual(esc("plain ✓ café"), "plain ✓ café");
for (const e of EVIL) assert.ok(!/[<>"']/.test(esc(e)), "esc leaked: " + e);

// Builders: strip the markup the builders emit themselves; no raw `<` may
// remain inside text positions, i.e. the payload never becomes a tag.
const TAGS = /<\/?(div|strong|span|br|b|td)(\s+class="[a-z -]+")?>/g;
function noInjected(html, what) {
  const rest = html.replace(TAGS, "");
  assert.ok(!rest.includes("<"), `${what}: raw '<' survived: ${rest}`);
  assert.ok(!/<(img|script|svg)/i.test(html), `${what}: injected tag`);
}
for (const e of EVIL) {
  noInjected(card(e, e), "card");
  noInjected(catalogMeta({ name: e, source: e, approxMb: e, format: e, description: e }), "catalogMeta");
  const r = { model: e, label: e, numPredictions: 1234, crossEntropy: 1, bitsPerToken: 1, bitsPerByte: 1.5, perplexity: 2, uniformBaseline: 50, seconds: 0.25 };
  noInjected(evalRow(r, true), "evalRow");
  noInjected(modelInfo({ path: e, bytes: 10, format: e, name: e, task: e, inputDim: e, outputDim: e, tokenizer: e, merges: e, layers: e }), "modelInfo");
  const g = { architecture: e, runnable: false, version: e, numTensors: e, modelDim: e, nLayers: e, nHeads: e, nKvHeads: e,
              vocabSize: e, tokenizer: e, estInt4Mb: 1, estInt8Mb: 2, estF32Mb: 4, availMb: null, note: e };
  noInjected(ggufInfo(g), "ggufInfo");
  noInjected(exportInfo(g), "exportInfo");
  const pf = { params: 1000, peakRamMb: 10, ramLimitMb: 20, ramAvailableMb: 30, fitsInRam: true, seconds: 1, llamaCpp: { runs: false, reason: e }, problems: [e] };
  noInjected(preflightView(pf), "preflightView");
}

// Pre-flight: RAM units, the blocked-Start reason, and the "fits" / "unknown" cases.
const pf0 = { params: null, peakRamMb: null, ramLimitMb: null, ramAvailableMb: null, fitsInRam: true, llamaCpp: { runs: true, reason: "r" }, problems: [] };
assert.ok(preflightView(pf0).includes("unknown"));
assert.strictEqual(preflightWhy(pf0), "");
assert.strictEqual(preflightWhy(null), "");
const tight = { ...pf0, peakRamMb: 2048, ramLimitMb: 512, ramAvailableMb: 700, fitsInRam: false };
assert.ok(preflightView(tight).includes("2.00 GB of 512 MB usable (700 MB free)"), preflightView(tight));
assert.ok(preflightWhy(tight).startsWith("This run needs about 2.00 GB but the engine allows 512 MB"), preflightWhy(tight));
assert.ok(preflightWhy(tight).includes("try anyway"));
assert.strictEqual(preflightWhy({ ...tight, fitsInRam: true }), "", "fits: nothing to explain");
assert.strictEqual(preflightWhy({ ...tight, peakRamMb: null }), "", "no peak known: never block");

// Spot check: the attack string is visible as text, not markup.
assert.ok(card("Name", EVIL[0]).includes("&lt;img src=x onerror=alert(1)&gt;"));
assert.ok(ggufInfo({ architecture: "llama", runnable: true, version: 3, numTensors: 1, modelDim: 1, nLayers: 1, nHeads: 1,
  nKvHeads: 1, vocabSize: 1, tokenizer: "bpe", estInt4Mb: 1, estInt8Mb: 1, estF32Mb: 1, availMb: 2048, note: "ok" }).includes("2.00 GB"));

console.log("escape.test.js: all assertions passed");
