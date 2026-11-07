// Unit tests for ui/core/mode.js (Beginner and Expert modes, presets, log dock collapse).
// Run with: node ui/tests/mode.test.js
"use strict";
const assert = require("node:assert");
const { Document } = require("./testdom.js");
const Mode = require("../core/mode.js");

function makeEnv(initialStore = {}) {
  const doc = new Document();
  const store = { ...initialStore };
  const storage = {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
  };

  const sel = doc.createElement("select");
  sel.id = "uiMode";
  for (const v of ["beginner", "expert"]) {
    const opt = doc.createElement("option");
    opt.setAttribute("value", v);
    sel.appendChild(opt);
  }
  doc.body.appendChild(sel);

  const tabDs = doc.createElement("button");
  tabDs.id = "tab-datasets";
  tabDs.className = "tab active";
  tabDs.addEventListener("click", () => {
    doc.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tabDs.classList.add("active");
  });
  doc.body.appendChild(tabDs);

  const tabFt = doc.createElement("button");
  tabFt.id = "tab-finetune";
  tabFt.className = "tab adv-only";
  tabFt.addEventListener("click", () => {
    doc.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tabFt.classList.add("active");
  });
  doc.body.appendChild(tabFt);

  const term = doc.createElement("section");
  term.id = "terminal";
  const btnLog = doc.createElement("button");
  btnLog.id = "btnToggleLog";
  term.appendChild(btnLog);
  doc.body.appendChild(term);

  for (const id of ["trContext", "trEmbed", "trHeads", "trBlocks", "trHidden", "trVocab", "trBatch", "trEpochs", "trLr"]) {
    const inp = doc.createElement("input");
    inp.id = id;
    doc.body.appendChild(inp);
  }

  const $ = (id) => doc.getElementById(id);
  const mode = Mode.create({ $, doc, getStorage: () => storage });
  return { doc, $, store, mode };
}

// 1. Default initialization is beginner mode and collapses log dock
{
  const { doc, $, mode } = makeEnv();
  mode.init();
  assert.strictEqual(mode.getMode(), "beginner");
  assert.strictEqual(doc.body.getAttribute("data-mode"), "beginner");
  assert.strictEqual($("uiMode").value, "beginner");
  assert.strictEqual(mode.isLogCollapsed(), true);
  assert.ok($("terminal").classList.contains("collapsed"));
  assert.strictEqual($("btnToggleLog").textContent, "Show log");
  assert.strictEqual($("btnToggleLog").getAttribute("aria-expanded"), "false");
}

// 2. Switching to expert mode updates attribute, storage, and expands log
{
  const { doc, $, store, mode } = makeEnv();
  mode.init();
  $("uiMode").value = "expert";
  $("uiMode").dispatch("change");
  assert.strictEqual(mode.getMode(), "expert");
  assert.strictEqual(doc.body.getAttribute("data-mode"), "expert");
  assert.strictEqual(store[Mode.MODE_KEY], "expert");
  assert.strictEqual(mode.isLogCollapsed(), false);
  assert.ok(!$("terminal").classList.contains("collapsed"));
  assert.strictEqual($("btnToggleLog").textContent, "Hide log");
  assert.strictEqual($("btnToggleLog").getAttribute("aria-expanded"), "true");
}

// 3. Switching back to beginner while on an .adv-only tab returns to #tab-datasets
{
  const { $, mode } = makeEnv({ [Mode.MODE_KEY]: "expert" });
  mode.init();
  assert.strictEqual(mode.getMode(), "expert");
  $("tab-finetune").click();
  assert.ok($("tab-finetune").classList.contains("active"));
  mode.setMode("beginner");
  assert.ok($("tab-datasets").classList.contains("active"));
  assert.ok(!$("tab-finetune").classList.contains("active"));
}

// 4. Legacy "advanced" stored value normalizes to "expert", and manual log toggle works
{
  const { $, mode } = makeEnv({ [Mode.MODE_KEY]: "advanced" });
  mode.init();
  assert.strictEqual(mode.getMode(), "expert");
  assert.strictEqual(mode.isLogCollapsed(), false);
  $("btnToggleLog").click();
  assert.strictEqual(mode.isLogCollapsed(), true);
  $("btnToggleLog").click();
  assert.strictEqual(mode.isLogCollapsed(), false);
  assert.strictEqual(Mode.normalizeMode("advanced"), "expert");
  assert.strictEqual(Mode.normalizeMode("bogus"), "beginner");
}

// 5. Blocked storage degrades gracefully to beginner mode
{
  const doc = new Document();
  const $ = (id) => doc.getElementById(id);
  const mode = Mode.create({
    $,
    doc,
    getStorage: () => {
      throw new Error("blocked");
    },
  });
  mode.init();
  assert.strictEqual(mode.getMode(), "beginner");
  mode.setMode("expert");
  assert.strictEqual(mode.getMode(), "expert");
}

// 6. Presets populate Train fields and reject unknown presets
{
  const { doc, $, mode } = makeEnv();
  assert.strictEqual(mode.applyPreset("small"), true);
  assert.strictEqual($("trContext").value, "64");
  assert.strictEqual($("trEmbed").value, "128");
  assert.strictEqual($("trHeads").value, "4");
  assert.strictEqual($("trBlocks").value, "4");
  assert.strictEqual($("trHidden").value, "256");
  assert.strictEqual($("trVocab").value, "1024");
  assert.strictEqual(mode.applyPreset("medium"), true);
  assert.strictEqual($("trEmbed").value, "256");
  assert.strictEqual(mode.applyPreset("tiny"), true);
  assert.strictEqual($("trEmbed").value, "64");
  assert.strictEqual(mode.applyPreset("unknown"), false);
  assert.strictEqual(Mode.applyPreset(null, "tiny"), false);
}

// 7. Tooltip cards follow the mode (plan Task 4)
{
  const { contentFromHelp } = require("../core/tooltip.js");
  const { HELP } = require("../core/help.js");
  const labels = (m) => contentFromHelp(HELP.trEpochs, m).sections.map((s) => s.label);
  assert.deepStrictEqual(labels("beginner"), ["New to AI?", "What it is"]);
  assert.deepStrictEqual(labels("expert"), ["What it is", "How it works"]);
  assert.deepStrictEqual(labels(undefined), ["What it is", "How it works"]);
}

