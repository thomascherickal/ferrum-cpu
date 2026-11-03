// Guard: every_element_has_a_30_word_tooltip. Scans index.html and fails, listing each offender,
// when an element a user can see or use has no tool-tip of at least MIN_WORDS words (words =
// whitespace-separated pieces, counted by tooltip.js `wordCount`). A tip comes from one of two
// places: the help.js entry of a control (input, select, textarea, button; its section texts,
// not the title or the "What it does:" labels) or a `data-tip` attribute on any element.
// Run with: node ui/tooltips_guard.test.js
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { parse, all, textOf } = require("./testhtml.js");
const { wordCount, contentFromHelp, MIN_WORDS } = require("./tooltip.js");
const { helpFor } = require("./help.js");
const { ERR_TIPS } = require("./common.js");

const CONTROLS = new Set(["input", "select", "textarea", "button"]);
// Elements that always show something, even before they have text.
const ALWAYS = new Set(["img", "canvas", "pre", "table", "h1", "h2", "h3", "legend", "th"]);
// Inline formatting is part of the text around it and shares that block's tip.
const INLINE = new Set(["b", "strong", "em", "i", "code", "br"]);
// Page background and wrappers whose parts carry their own tips (documented in docs/gui-tooltips.md).
const STRUCTURAL_IDS = new Set(["topbar", "miniStats", "tabs", "content", "svRunning", "terminal", "capDialog", "ggWarnDialog"]);

const ancestors = function* (n) { for (let p = n.parent; p && p.tag !== "#root"; p = p.parent) yield p; };
const directText = (n) => n.children.some((c) => c.text !== undefined && c.text.trim());
const helpWords = (id) => { const e = id ? helpFor(id) : null; return e ? wordCount(contentFromHelp(e).sections.map((s) => s.text).join(" ")) : 0; };
// Words of the element's own tip (0 = none).
function tipWords(n) {
  if (CONTROLS.has(n.tag) && helpWords(n.attrs.id)) return helpWords(n.attrs.id);
  return n.attrs["data-tip"] === undefined ? 0 : wordCount(n.attrs["data-tip"]);
}

// Why `n` needs no tip of its own, or null when it does. `n` is any parsed element.
function exempt(n) {
  if (n.tag === "option") return "covered by its <select>";
  if (n.attrs["aria-hidden"] === "true") return "decorative (aria-hidden)";
  if (INLINE.has(n.tag)) return "inline text, shares the tip of its block";
  if (STRUCTURAL_IDS.has(n.attrs.id) || n.attrs.role === "tabpanel") return "background / wrapper";
  for (const a of ancestors(n)) if (a.attrs.role === "progressbar") return "covered by its progress bar";
  if (n.tag === "label") {
    const control = all(n, (c) => CONTROLS.has(c.tag))[0];
    if (control && tipWords(control) >= MIN_WORDS) return "covered by the control it wraps";
    return null;
  }
  const needs = CONTROLS.has(n.tag) || ALWAYS.has(n.tag) || n.attrs.id || n.attrs.role || n.attrs["aria-label"] || directText(n);
  return needs ? null : "structural wrapper";
}

// Every element of `html` that needs a tip and lacks one: [{ what, words }].
function missingTips(html) {
  const out = [];
  for (const n of all(parse(html), (x) => x.tag !== "#root")) {
    if (exempt(n)) continue;
    const words = tipWords(n);
    if (words < MIN_WORDS) out.push({ what: `<${n.tag}${n.attrs.id ? " #" + n.attrs.id : ""}> "${textOf(n).replace(/\s+/g, " ").trim().slice(0, 40)}"`, words });
  }
  return out;
}

// ── The rule itself: 29 words fail, 30 pass, for a data-tip and for a wrapped label ──
{
  const words = (n) => Array.from({ length: n }, (_, i) => "w" + i).join(" ");
  assert.strictEqual(wordCount(words(30)), 30);
  assert.strictEqual(wordCount("  a\tb\n c  "), 3);
  assert.strictEqual(wordCount(""), 0);
  assert.strictEqual(missingTips(`<body><h2 data-tip="${words(29)}">T</h2></body>`).length, 1, "29 words must fail");
  assert.deepStrictEqual(missingTips(`<body><h2 data-tip="${words(30)}">T</h2></body>`), [], "30 words must pass");
  assert.strictEqual(missingTips(`<body><label>X <input id="x" data-tip="${words(29)}" /></label></body>`).length, 2, "the input and its label both fail");
  assert.deepStrictEqual(missingTips(`<body><label>X <input id="x" data-tip="${words(30)}" /></label></body>`), []);
  assert.deepStrictEqual(missingTips(`<body><div class="row"><div><select id="s" data-tip="${words(30)}"><option>a</option></select></div></div></body>`), [], "wrappers and options are exempt");
  assert.strictEqual(missingTips(`<body><div id="out"></div><span>text</span></body>`).length, 2, "an id or own text needs a tip");
}

// ── The real page ──
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const missing = missingTips(html);
assert.deepStrictEqual(missing, [], `${missing.length} element(s) without a ${MIN_WORDS}-word tool-tip:\n` + missing.map((m) => `  ${m.what} (${m.words} words)`).join("\n"));

// Controls built at runtime (not in index.html): the catalog's Download buttons
// (datasets_tab.js, help entry dsCatalogDownload) and the two buttons of every error line
// (common.js ERR_TIPS). No other tab module creates a control.
assert.ok(helpWords("dsCatalogDownload") >= MIN_WORDS, "dsCatalogDownload help is too short");
for (const [k, v] of Object.entries(ERR_TIPS)) assert.ok(wordCount(v) >= MIN_WORDS, `ERR_TIPS.${k} has ${wordCount(v)} words`);

// A help entry and a data-tip on one control would be two texts for one bubble.
const root = parse(html);
for (const n of all(root, (x) => CONTROLS.has(x.tag) && x.attrs["data-tip"] !== undefined)) {
  assert.ok(!helpFor(n.attrs.id), `#${n.attrs.id} has both a help entry and a data-tip`);
}
// Tips are plain text in an attribute: no markup or quotes that would need escaping.
for (const n of all(root, (x) => x.attrs["data-tip"] !== undefined)) assert.ok(!/[<>&"]/.test(n.attrs["data-tip"]), `data-tip of <${n.tag} #${n.attrs.id}> contains markup`);
// Native title tool-tips would show a second bubble next to ours.
assert.deepStrictEqual(all(root, (x) => x.attrs.title !== undefined).map((x) => x.tag + "#" + x.attrs.id), [], "use data-tip, not title");

const tipped = all(root, (x) => x.tag !== "#root" && !exempt(x)).length;
console.log(`tooltips_guard.test.js: every_element_has_a_30_word_tooltip passed (${tipped} elements with their own tip)`);
