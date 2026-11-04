// Node test for the help data. Run with: node ui/tests/help.test.js
// Guards completeness (every field and button in index.html has a long enough
// help entry), safety (no markup in the data) and that the job kinds agree with
// the Rust enum.
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { HELP, KINDS, helpFor, isButton } = require("../core/help.js");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");

// Every input / textarea / select / button of the page, with its id.
const elements = [];
for (const m of html.matchAll(/<(input|textarea|select|button)\b([^>]*)>/g)) {
  const id = (m[2].match(/\bid="([^"]+)"/) || [])[1];
  assert.ok(id, `<${m[1]} ${m[2].trim()}> has no id, so it cannot have a tooltip`);
  elements.push({ tag: m[1], id });
}
assert.ok(elements.length >= 140, "expected the whole page, got " + elements.length);
const ids = elements.map((e) => e.id);
assert.strictEqual(new Set(ids).size, ids.length, "duplicate ids in index.html");

// Completeness and minimum quality.
const FIELD_MIN = 200;
const BUTTON_MIN = 40;
for (const { tag, id } of elements) {
  const e = helpFor(id);
  assert.ok(e, `no help entry for #${id}`);
  assert.ok(e.title && e.title.length >= 3, `#${id}: title`);
  if (tag === "button") {
    assert.ok(isButton(e), `#${id} is a button but its entry has no 'does'`);
    assert.ok(e.does.length >= BUTTON_MIN, `#${id}: 'does' too short (${e.does.length})`);
    assert.ok(KINDS.includes(e.kind), `#${id}: unknown kind ${e.kind}`);
  } else {
    assert.ok(!isButton(e), `#${id} is a field but its entry looks like a button`);
    for (const k of ["what", "how", "beginner"]) {
      assert.ok(typeof e[k] === "string" && e[k].length >= 20, `#${id}: '${k}' missing or tiny`);
    }
    const total = e.what.length + e.how.length + e.beginner.length;
    assert.ok(total >= FIELD_MIN, `#${id}: help is only ${total} chars (need ${FIELD_MIN})`);
  }
}

// No orphan entries: every key is an element id, or a button built at runtime.
const RUNTIME_KEYS = new Set(["dsCatalogDownload"]);
for (const key of Object.keys(HELP)) {
  assert.ok(ids.includes(key) || RUNTIME_KEYS.has(key), `help entry '${key}' matches no element`);
}

// The data is plain text: no markup characters anywhere.
(function scan(o, where) {
  for (const [k, v] of Object.entries(o)) {
    if (typeof v === "string") assert.ok(!/[<>]/.test(v), `${where}.${k} contains < or >`);
    else if (v && typeof v === "object") scan(v, where + "." + k);
  }
})(HELP, "HELP");

// Job kinds agree with the Rust enum (snake_case of the variant names).
const rs = fs.readFileSync(path.join(__dirname, "..", "..", "src", "sizing", "estimate.rs"), "utf8");
const body = rs.match(/pub enum JobKind \{([\s\S]*?)\n\}/)[1];
const variants = [...body.matchAll(/^\s*([A-Z]\w*),/gm)].map((m) => m[1].replace(/(?<!^)([A-Z])/g, "_$1").toLowerCase());
assert.deepStrictEqual([...variants].sort(), [...KINDS].sort(), "KINDS differs from JobKind in src/sizing/estimate.rs");
for (const k of KINDS) {
  assert.ok(Object.values(HELP).some((e) => e.kind === k), `no button uses the kind '${k}'`);
}

// Beginner terms are explained where they first matter.
const TERMS = [
  ["trContext", /token/i], ["trEpochs", /epoch/i], ["trBatch", /batch|stack/i], ["trLr", /learning rate/i],
  ["evText", /perplexity/i], ["optLower", /vocabulary/i], ["trContext", /context/i], ["genTemp", /dial|creativ|adventurous/i],
  ["ggTopK", /rank/i], ["ggTopP", /90 percent|candidates/i], ["trCkpt", /save game/i], ["ggQuant", /quantization/i],
  ["trModel", /GGUF/], ["ftPath", /fine-tuning/i], ["trVocab", /BPE|byte pair/i], ["trEmbed", /embedding/i],
  ["trHeads", /attention/i], ["trBlocks", /block/i], ["trHidden", /neurons|thinking/i], ["trSeed", /repeatable|identical/i],
];
for (const [id, re] of TERMS) {
  const e = HELP[id];
  assert.ok(re.test(e.what + " " + e.how + " " + e.beginner), `#${id} should explain ${re}`);
  assert.ok(re.test(e.beginner) || re.test(e.what), `#${id}: ${re} should be in the plain-language parts`);
}

// Claims about defaults must match the page: spot-check a few against index.html.
const def = (id) => (html.match(new RegExp(`id="${id}"[^>]*value="([^"]*)"`)) || [])[1];
assert.strictEqual(def("trContext"), "16");
assert.match(HELP.trContext.what, /Default 16/);
assert.strictEqual(def("trEmbed"), "32");
assert.match(HELP.trEmbed.what, /Default 32/);
assert.strictEqual(def("trEpochs"), "100");
assert.match(HELP.trEpochs.what, /Default 100/);
assert.strictEqual(def("trLr"), "0.01");
assert.match(HELP.trLr.what, /Default 0\.01/);
assert.strictEqual(def("trBatch"), "16");
assert.match(HELP.trBatch.what, /Default 16/);
assert.strictEqual(def("trVocab"), "512");
assert.match(HELP.trVocab.what, /Default 512/);
assert.strictEqual(def("trSeed"), "1337");
assert.strictEqual(def("genChars"), "200");
assert.strictEqual(def("genTemp"), "0.8");
assert.strictEqual(def("ggMax"), "64");
assert.strictEqual(def("ftEpochs"), "3");
assert.strictEqual(def("ftLr"), "0.0001");
assert.strictEqual(def("ftBatch"), "8");
assert.strictEqual(def("ftSeq"), "64");
assert.strictEqual(def("dsMaxMb"), "8");
assert.strictEqual(def("sysInterval"), "1.5");

console.log(`help.test.js: all assertions passed (${elements.length} elements, ${Object.keys(HELP).length} entries)`);
