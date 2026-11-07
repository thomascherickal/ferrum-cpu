// Node test for tooltip.js. Run with: node ui/tests/tooltip.test.js
"use strict";
const assert = require("node:assert");
const { placeTooltip, contentFromHelp, plainText, createTooltips, HOVER_MS } = require("../core/tooltip.js");
const { Document } = require("./testdom.js");
const { HELP } = require("../core/help.js");

const rect = (left, top, width, height) => ({ left, top, width, height, right: left + width, bottom: top + height });
const VP = { width: 1000, height: 700 };

// ── placeTooltip (pure) ─────────────────────────────────────────────────────
{
  const a = rect(100, 100, 80, 24);
  let p = placeTooltip(a, { width: 300, height: 200 }, VP);
  assert.deepStrictEqual(p, { left: 100, top: 132, placement: "below" });

  // No room below -> above.
  p = placeTooltip(rect(100, 600, 80, 24), { width: 300, height: 200 }, VP);
  assert.strictEqual(p.placement, "above");
  assert.strictEqual(p.top, 600 - 8 - 200);

  // Right edge: slide left so the bubble stays inside.
  p = placeTooltip(rect(900, 100, 80, 24), { width: 300, height: 100 }, VP);
  assert.strictEqual(p.left, 1000 - 300 - 8);

  // Anchor scrolled off to the left.
  assert.strictEqual(placeTooltip(rect(-50, 100, 80, 24), { width: 300, height: 100 }, VP).left, 8);

  // Taller than the space on both sides: pick the roomier side and clamp inside.
  p = placeTooltip(rect(100, 300, 80, 24), { width: 300, height: 600 }, VP);
  assert.ok(p.top >= 8 && p.top + 600 <= 700 - 8 + 1e-9, JSON.stringify(p));

  // Bigger than the whole viewport: never negative, pinned to the margin.
  p = placeTooltip(rect(100, 100, 80, 24), { width: 2000, height: 2000 }, VP);
  assert.deepStrictEqual([p.left, p.top], [8, 8]);

  // Property: whenever the bubble fits the viewport, the result is fully inside it.
  let seed = 42;
  const rnd = (n) => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
  for (let i = 0; i < 2000; i++) {
    const tip = { width: 50 + rnd(500), height: 30 + rnd(500) };
    const an = rect(rnd(1200) - 100, rnd(900) - 100, 10 + rnd(200), 10 + rnd(50));
    const r = placeTooltip(an, tip, VP);
    assert.ok(r.left >= 8 && r.left + tip.width <= VP.width - 8, "x " + JSON.stringify([an, tip, r]));
    assert.ok(r.top >= 8 && r.top + tip.height <= VP.height - 8, "y " + JSON.stringify([an, tip, r]));
  }
}

// ── content ─────────────────────────────────────────────────────────────────
{
  const field = contentFromHelp(HELP.trEpochs);
  assert.deepStrictEqual(field.sections.map((s) => s.label), ["What it is", "How it works"]);
  const begField = contentFromHelp(HELP.trEpochs, "beginner");
  assert.deepStrictEqual(begField.sections.map((s) => s.label), ["New to AI?", "What it is"]);
  const button = contentFromHelp(HELP.trStart);
  assert.strictEqual(button.sections[0].label, "What it does");
  assert.ok(!button.sections.some((s) => s.label === "New to AI?"));
  const begButton = contentFromHelp(HELP.trStart, "beginner");
  assert.deepStrictEqual(begButton.sections.map((s) => s.label), ["New to AI?", "What it does"]);
  // "expert" and an unset mode both give the Advanced view.
  assert.deepStrictEqual(contentFromHelp(HELP.trEpochs, "expert").sections.map((s) => s.label), ["What it is", "How it works"]);
  // No `beginner` text: Beginner mode keeps How it works, so nothing is lost.
  assert.deepStrictEqual(contentFromHelp({ title: "T", what: "w", how: "h" }, "beginner").sections.map((s) => s.label), ["What it is", "How it works"]);
  assert.strictEqual(contentFromHelp(null), null);
  assert.strictEqual(plainText(null), "");
  const t = plainText(field);
  assert.ok(t.startsWith("Epochs.") && t.includes("What it is: ") && !t.includes(".."), t);
}

// ── behaviour with a stub DOM ───────────────────────────────────────────────
function setup() {
  const doc = new Document();
  const timers = [];
  const env = {
    document: doc,
    setTimeout: (f, ms) => { timers.push({ f, ms }); return timers.length; },
    clearTimeout: (id) => { if (id) timers[id - 1] = null; },
    viewport: () => VP,
  };
  const tips = createTooltips(env);
  const label = doc.createElement("label");
  const input = doc.createElement("input");
  label.appendChild(input);
  doc.body.appendChild(label);
  input.rect = rect(200, 100, 120, 30);
  tips.bubble.rect = rect(0, 0, 320, 180);
  const fire = () => { const t = timers.filter(Boolean).pop(); timers.length = 0; t && t.f(); return t; };
  return { doc, tips, label, input, fire, timers };
}

{
  const { doc, tips, input, label, fire } = setup();
  assert.strictEqual(tips.bubble.getAttribute("role"), "tooltip");
  assert.ok(tips.bubble.classList.contains("hidden"));
  let content = { title: "Epochs", sections: [{ label: "What it is", text: "A pass." }] };
  tips.attach(input, () => content);

  // Screen readers: aria-describedby points at a persistent description node.
  const descId = input.getAttribute("aria-describedby");
  const desc = doc.getElementById(descId);
  assert.ok(desc && desc.textContent === "Epochs. What it is: A pass.", desc && desc.textContent);

  // Hover shows after the delay, not before; leaving hides.
  label.dispatch("mouseenter");
  assert.ok(tips.bubble.classList.contains("hidden"));
  const t = fire();
  assert.strictEqual(t.ms, HOVER_MS);
  assert.ok(!tips.bubble.classList.contains("hidden") && tips.current() === input);
  assert.strictEqual(tips.bubble.textContent, "EpochsWhat it is: A pass.");
  assert.strictEqual(tips.bubble.style.left, "200px");
  assert.strictEqual(tips.bubble.style.top, "138px"); // below the input + 8
  assert.strictEqual(tips.bubble.getAttribute("data-placement"), "below");
  label.dispatch("mouseleave");
  assert.ok(tips.bubble.classList.contains("hidden"));

  // Keyboard focus shows immediately; blur hides.
  input.dispatch("focus");
  assert.ok(tips.isVisible());
  input.dispatch("blur");
  assert.ok(!tips.isVisible());

  // Escape dismisses and the tooltip does not come back until focus leaves.
  input.dispatch("focus");
  doc.dispatch("keydown", { key: "Tab" });
  assert.ok(tips.isVisible(), "other keys must not dismiss");
  doc.dispatch("keydown", { key: "Escape" });
  assert.ok(!tips.isVisible());
  label.dispatch("mouseenter"); fire();
  assert.ok(!tips.isVisible(), "dismissed tooltip stays dismissed while the pointer/focus is still there");
  input.dispatch("blur");
  input.dispatch("focus");
  assert.ok(tips.isVisible(), "re-shows after leaving and coming back");

  // Live update: changing the content refreshes the open bubble and the description.
  content = { title: "Epochs", sections: [{ label: "Estimated time", text: "about 4 min" }] };
  tips.refreshTarget(input);
  assert.ok(tips.bubble.textContent.includes("about 4 min"));
  assert.ok(desc.textContent.includes("about 4 min"));
  // ...and refreshing a target that is not shown does not open it.
  input.dispatch("blur");
  content = { title: "Epochs", sections: [{ label: null, text: "again" }] };
  tips.refreshAll();
  assert.ok(!tips.isVisible() && desc.textContent === "Epochs. again.");

  // The bubble repositions above when the anchor is near the bottom.
  input.rect = rect(200, 600, 120, 30);
  input.dispatch("focus");
  assert.strictEqual(tips.bubble.getAttribute("data-placement"), "above");
  tips.hide();
}

// Untrusted text is never parsed as markup.
{
  const { tips, input } = setup();
  const evil = "<img src=x onerror=alert(1)>";
  tips.attach(input, () => ({ title: evil, sections: [{ label: evil, text: evil, tone: "warn" }] }));
  input.dispatch("focus");
  let imgs = 0;
  tips.bubble.walk((n) => { if (n.tagName === "IMG") imgs++; });
  assert.strictEqual(imgs, 0);
  assert.ok(tips.bubble.textContent.includes(evil));
  assert.ok(!tips.bubble.innerHTML, "no innerHTML was used");
  assert.ok(tips.store.textContent.includes(evil));
}

// No content (e.g. no help entry) -> nothing shown.
{
  const { tips, input } = setup();
  tips.attach(input, () => null);
  input.dispatch("focus");
  assert.ok(!tips.isVisible());
  assert.strictEqual(tips.store.textContent, "");
}

// Attaching twice replaces the content getter instead of adding listeners.
{
  const { tips, input, doc } = setup();
  tips.attach(input, () => ({ title: "A", sections: [] }));
  tips.attach(input, () => ({ title: "B", sections: [] }));
  assert.strictEqual(doc.getElementById("ferrum-descriptions").children.length, 1);
  assert.strictEqual(input.listeners.focus.length, 1);
  input.dispatch("focus");
  assert.strictEqual(tips.bubble.textContent, "B");
}

// Inside a modal <dialog> the bubble must live in the dialog (the top layer sits above any z-index
// outside it); back in the page it returns to <body>; a closed dialog does not capture it.
{
  const { doc, tips, input } = setup();
  const dlg = doc.createElement("dialog");
  const btn = doc.createElement("button");
  dlg.appendChild(btn);
  doc.body.appendChild(dlg);
  btn.rect = rect(300, 200, 80, 24);
  tips.attach(btn, () => ({ title: "Proceed", sections: [{ label: "What it does", text: "Goes on." }] }));
  tips.attach(input, () => ({ title: "Epochs", sections: [] }));

  input.dispatch("focus");
  assert.strictEqual(tips.bubble.parentNode, doc.body, "page controls keep the bubble in <body>");
  input.dispatch("blur");

  dlg.showModal();
  btn.dispatch("focus");
  assert.ok(tips.isVisible());
  assert.strictEqual(tips.bubble.parentNode, dlg, "the bubble is inside the open dialog, in the top layer");
  assert.ok(dlg.querySelector("#ferrum-tooltip"), "found through the dialog");
  assert.strictEqual(tips.bubble.style.left, "300px", "still positioned by viewport coordinates");
  btn.dispatch("blur");
  dlg.close();

  input.dispatch("focus");
  assert.strictEqual(tips.bubble.parentNode, doc.body, "back in <body> once the dialog is closed");
  input.dispatch("blur");

  // A control in a dialog that is not open never pulls the bubble into it.
  btn.dispatch("focus");
  assert.strictEqual(tips.bubble.parentNode, doc.body);
  tips.hide();
}

// The stylesheet keeps the bubble out of the layout (fixed) and above the page, which is what
// lets it sit inside a dialog and still be placed by viewport coordinates.
{
  const css = require("node:fs").readFileSync(require("node:path").join(__dirname, "..", "styles.css"), "utf8");
  const rule = css.match(/\n\.tip \{([^}]*)\}/)[1];
  assert.ok(/position:\s*fixed/.test(rule) && /z-index:\s*1000/.test(rule), rule);
  assert.ok(!/(^|[;\s])(transform|filter|perspective)\s*:/.test(rule), "a transform would make it relative to its parent");
}

// data-tip content: one plain section, no title; the bubble and the description skip the title.
{
  const { MIN_WORDS, wordCount, contentFromTip } = require("../core/tooltip.js");
  assert.strictEqual(MIN_WORDS, 30);
  assert.strictEqual(wordCount("one  two\nthree\t four "), 4);
  assert.strictEqual(wordCount(null), 0);
  assert.strictEqual(contentFromTip(""), null);
  assert.strictEqual(contentFromTip(null), null);
  const c = contentFromTip("  Shows the training loss.  ");
  assert.deepStrictEqual(c, { title: "", sections: [{ label: null, text: "Shows the training loss." }] });
  assert.strictEqual(plainText(c), "Shows the training loss.");
  assert.deepStrictEqual(contentFromTip("Tip.", "  "), { title: "", sections: [{ label: null, text: "Tip." }] }, "blank live text adds nothing");
  assert.strictEqual(plainText(contentFromTip("Tip.", "4 CPU threads")), "Tip. Now: 4 CPU threads.");

  const { doc, tips } = setup();
  const h2 = doc.createElement("h2");
  doc.body.appendChild(h2);
  h2.rect = rect(10, 10, 300, 30);
  tips.attach(h2, () => c);
  assert.strictEqual(doc.getElementById(h2.getAttribute("aria-describedby")).textContent, "Shows the training loss.");
  h2.dispatch("focus");
  assert.strictEqual(tips.bubble.textContent, "Shows the training loss.");
  assert.ok(!tips.bubble.querySelector(".tip-title"), "no empty title line");
  tips.hide();
}

console.log("tooltip.test.js: all assertions passed");
