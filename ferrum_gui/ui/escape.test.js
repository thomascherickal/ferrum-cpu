// Node test for the HTML escaping helpers and card/row builders. Run with:
//   node ui/escape.test.js
"use strict";
const assert = require("node:assert");
const { esc, card, catalogMeta, evalRow, modelInfo, ggufInfo, exportInfo } = require("./escape.js");

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
  const r = { model: e, numPredictions: 1234, crossEntropy: 1, bitsPerToken: 1, perplexity: 2, uniformBaseline: 50 };
  noInjected(evalRow(r, true), "evalRow");
  noInjected(modelInfo({ path: e, bytes: 10, format: e, name: e, task: e, inputDim: e, outputDim: e, tokenizer: e, merges: e, layers: e }), "modelInfo");
  const g = { architecture: e, runnable: false, version: e, numTensors: e, modelDim: e, nLayers: e, nHeads: e, nKvHeads: e,
              vocabSize: e, tokenizer: e, estInt4Mb: 1, estInt8Mb: 2, estF32Mb: 4, availMb: null, note: e };
  noInjected(ggufInfo(g), "ggufInfo");
  noInjected(exportInfo(g), "exportInfo");
}

// Spot check: the attack string is visible as text, not markup.
assert.ok(card("Name", EVIL[0]).includes("&lt;img src=x onerror=alert(1)&gt;"));
assert.ok(ggufInfo({ architecture: "llama", runnable: true, version: 3, numTensors: 1, modelDim: 1, nLayers: 1, nHeads: 1,
  nKvHeads: 1, vocabSize: 1, tokenizer: "bpe", estInt4Mb: 1, estInt8Mb: 1, estF32Mb: 1, availMb: 2048, note: "ok" }).includes("2.00 GB"));

console.log("escape.test.js: all assertions passed");
