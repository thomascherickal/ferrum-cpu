// Static accessibility checks over index.html: every form control has an accessible name, every
// button has text or an aria-label, the tab bar and dialogs carry their ARIA roles, status and
// error elements are live regions, and ids are unique.
// Run with: node ui/a11y_html.test.js
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { parse, all, textOf, ancestor } = require("./testhtml.js");

const root = parse(fs.readFileSync(path.join(__dirname, "index.html"), "utf8"));
const byId = new Map();
all(root, (n) => n.attrs.id).forEach((n) => {
  assert.ok(!byId.has(n.attrs.id), "duplicate id " + n.attrs.id);
  byId.set(n.attrs.id, n);
});
const words = (n) => textOf(n).replace(/\s+/g, " ").trim();
const label = (n, what) => `<${n.tag}${n.attrs.id ? " #" + n.attrs.id : ""}> ${what}`;

// ── Form controls: a <label for>, a wrapping <label> with words, aria-label or aria-labelledby ──
const controls = all(root, (n) => ["input", "select", "textarea"].includes(n.tag) && n.attrs.type !== "hidden");
assert.ok(controls.length >= 90, "found only " + controls.length + " form controls");
const labelFor = new Map(all(root, (n) => n.tag === "label" && n.attrs.for).map((l) => [l.attrs.for, l]));
for (const c of controls) {
  let name = (c.attrs["aria-label"] || "").trim();
  if (!name && c.attrs["aria-labelledby"]) {
    name = c.attrs["aria-labelledby"].split(/\s+/).map((id) => (byId.has(id) ? words(byId.get(id)) : "")).join(" ").trim();
  }
  if (!name && c.attrs.id && labelFor.has(c.attrs.id)) name = words(labelFor.get(c.attrs.id));
  if (!name) { const l = ancestor(c, "label"); if (l) name = words(l); }
  assert.ok(name, label(c, "has no accessible name (wrap it in a <label>, or add aria-label)"));
}

// ── Buttons: visible text or aria-label ──
const buttons = all(root, (n) => n.tag === "button");
assert.ok(buttons.length >= 50, "found only " + buttons.length + " buttons");
for (const b of buttons) assert.ok(words(b) || (b.attrs["aria-label"] || "").trim(), label(b, "has no text or aria-label"));
for (const b of buttons) assert.ok(b.attrs.type === undefined || ["button", "submit"].includes(b.attrs.type), label(b, "odd type"));

// ── Images and canvases ──
for (const i of all(root, (n) => n.tag === "img")) assert.ok("alt" in i.attrs, label(i, "has no alt"));
for (const c of all(root, (n) => n.tag === "canvas")) {
  assert.strictEqual(c.attrs.role, "img", label(c, "needs role=img"));
  assert.ok((c.attrs["aria-label"] || "").trim(), label(c, "needs an aria-label"));
}

// ── Tab bar: tablist > tab (aria-controls a tabpanel, labelled by the tab) ──
const tablist = all(root, (n) => n.attrs.role === "tablist");
assert.strictEqual(tablist.length, 1, "exactly one tablist");
assert.ok((tablist[0].attrs["aria-label"] || "").trim(), "the tablist needs a label");
const tabs = all(tablist[0], (n) => n.tag === "button");
const panels = all(root, (n) => n.tag === "section" && /\bpanel\b/.test(n.attrs.class || ""));
assert.strictEqual(tabs.length, panels.length, "one panel per tab");
assert.ok(tabs.length >= 10);
let selected = 0;
for (const t of tabs) {
  assert.strictEqual(t.attrs.role, "tab", label(t, "role=tab"));
  const name = t.attrs["data-tab"];
  assert.strictEqual(t.attrs.id, "tab-" + name);
  assert.strictEqual(t.attrs["aria-controls"], "panel-" + name, label(t, "aria-controls"));
  const isActive = /\bactive\b/.test(t.attrs.class || "");
  assert.strictEqual(t.attrs["aria-selected"], isActive ? "true" : "false", label(t, "aria-selected"));
  assert.strictEqual(t.attrs.tabindex, isActive ? "0" : "-1", label(t, "roving tabindex"));
  if (isActive) selected++;
  const p = byId.get("panel-" + name);
  assert.ok(p && p.tag === "section", "panel-" + name);
  assert.strictEqual(p.attrs.role, "tabpanel", label(p, "role=tabpanel"));
  assert.strictEqual(p.attrs["aria-labelledby"], "tab-" + name, label(p, "aria-labelledby"));
}
assert.strictEqual(selected, 1, "exactly one tab starts selected");

// ── Dialogs ──
for (const d of all(root, (n) => n.tag === "dialog")) {
  const ref = d.attrs["aria-labelledby"];
  assert.ok(ref && byId.has(ref) && words(byId.get(ref)), label(d, "needs aria-labelledby pointing at its heading"));
  assert.ok(d.attrs.id);
}

// ── Live regions: toasts, every status line, every error line, progress bars ──
assert.strictEqual(byId.get("toasts").attrs["aria-live"], "polite", "#toasts must be a polite live region");
const statusIds = [...byId.keys()].filter((id) => /Status$/.test(id));
assert.ok(statusIds.length >= 9, "status lines: " + statusIds);
for (const id of statusIds) assert.strictEqual(byId.get(id).attrs.role, "status", "#" + id + " needs role=status");
const errIds = [...byId.keys()].filter((id) => /^err[A-Z]/.test(id));
assert.ok(errIds.length >= 10, "error lines: " + errIds);
for (const id of errIds) assert.strictEqual(byId.get(id).attrs.role, "alert", "#" + id + " needs role=alert");
for (const bar of all(root, (n) => /\bprogress\b/.test(n.attrs.class || "") || /\bmeter\b/.test(n.attrs.class || ""))) {
  assert.strictEqual(bar.attrs.role, "progressbar", "progress/meter needs role=progressbar");
  assert.ok((bar.attrs["aria-label"] || "").trim(), "progressbar needs an aria-label");
  for (const k of ["aria-valuemin", "aria-valuemax", "aria-valuenow"]) assert.ok(k in bar.attrs, "progressbar needs " + k);
}

// ── No information by colour alone: the connection dot has a text label and a shape change ──
const dot = byId.get("connDot");
assert.ok((dot.attrs["aria-label"] || "").trim(), "#connDot needs an aria-label");
assert.strictEqual(dot.attrs.role, "img");

console.log(`a11y_html.test.js: all assertions passed (${controls.length} controls, ${buttons.length} buttons, ${tabs.length} tabs)`);
