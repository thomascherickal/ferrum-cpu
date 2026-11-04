// Accessibility checks over styles.css: WCAG contrast ratios of the colours the page really uses
// (a pure function over the CSS variables), a visible focus ring, no removed outlines, reduced
// motion, and no colour-only status marks.
// Run with: node ui/tests/a11y_css.test.js
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const css = fs.readFileSync(path.join(__dirname, "..", "styles.css"), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");

// ── WCAG 2.x relative luminance and contrast ratio (pure) ──
function luminance(hex) {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.replace(/./g, "$&$&") : h;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

// The function itself, against the values the WCAG definition gives.
assert.strictEqual(contrast("#000000", "#ffffff").toFixed(2), "21.00");
assert.strictEqual(contrast("#ffffff", "#ffffff").toFixed(2), "1.00");
assert.strictEqual(contrast("#777777", "#ffffff").toFixed(2), "4.48"); // the textbook near-miss
assert.strictEqual(contrast("#fff", "#000"), contrast("#000000", "#ffffff"));

// ── Reading the stylesheet ──
const vars = {};
for (const m of css.match(/:root\s*\{([^}]*)\}/)[1].matchAll(/(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{3,6})/g)) vars[m[1]] = m[2];
function rule(selector) {
  const re = new RegExp("(?:^|[}\\s])" + selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\s*\\{([^}]*)\\}");
  const m = css.match(re);
  assert.ok(m, "no rule for " + selector);
  return Object.fromEntries(m[1].split(";").map((d) => d.split(/:(.+)/).map((x) => (x || "").trim())).filter(([k]) => k));
}
// "var:--fg", "#fff", or "rule:selector:property" (a var() / hex inside it is resolved).
function colour(spec) {
  if (spec.startsWith("#")) return spec;
  if (spec.startsWith("var:")) { assert.ok(vars[spec.slice(4)], "unknown " + spec); return vars[spec.slice(4)]; }
  const [, sel, prop] = spec.match(/^rule:(.+):([\w-]+)$/);
  const raw = rule(sel)[prop];
  assert.ok(raw, `${sel} has no ${prop}`);
  const v = raw.match(/var\((--[\w-]+)/);
  if (v) return vars[v[1]];
  const hex = raw.match(/#[0-9a-fA-F]{3,6}\b/);
  assert.ok(hex, `${sel} ${prop}: ${raw}`);
  return hex[0];
}

const BG = ["var:--bg", "var:--bg2", "var:--bg3"];
const TEXT = 4.5, UI = 3;
const checks = [];
const need = (min, fg, bgs, why) => bgs.forEach((bg) => checks.push({ min, fg, bg, why }));

// Body, secondary and accent text on every surface it sits on.
need(TEXT, "var:--fg", BG, "body text");
need(TEXT, "var:--muted", BG, "secondary text");
need(TEXT, "var:--accent2", BG, "accent text (active tab, titles, gauges)");
need(TEXT, "var:--ok", BG, "success text");
need(TEXT, "var:--warn", BG, "warning text");
need(TEXT, "var:--err", BG, "error text and tooltips");
// Specific controls.
need(TEXT, "rule:button.primary:color", ["rule:button.primary:background"], "primary button");
need(TEXT, "var:--fg", ["var:--accent-strong-hover"], "text on the hovered primary button (white)");
need(TEXT, "#ffffff", ["var:--accent-strong", "var:--accent-strong-hover"], "white on the primary button");
need(TEXT, "#04210f", ["var:--accent"], "dataset badge text");
need(TEXT, "rule:.banner:color", ["rule:.banner:background"], "banner");
need(TEXT, "rule:.term-out .stderr:color", ["#0b1120"], "terminal stderr");
need(TEXT, "rule:.term-out .log:color", ["#0b1120"], "terminal log");
need(TEXT, "rule:.prompt:color", ["#0b1120"], "terminal prompt");
need(TEXT, "var:--fg", ["#0b1120"], "terminal text");
need(TEXT, "var:--muted", ["#0b1120"], "terminal system lines");
// Non-text: field borders and the focus ring must show against what is behind them.
need(UI, "var:--field-line", ["var:--bg", "var:--bg2"], "form-field border");
need(UI, "var:--focus", [...BG, "#0b1120"], "focus ring");
need(UI, "var:--accent", ["var:--bg2"], "progress fill on its track");

for (const c of checks) {
  const r = contrast(colour(c.fg), colour(c.bg));
  assert.ok(r >= c.min, `${c.why}: ${c.fg} on ${c.bg} is ${r.toFixed(2)}:1, needs ${c.min}:1`);
}

// ── Focus, motion, colour ──
assert.ok(/:focus-visible\s*\{[^}]*outline:\s*2px solid var\(--focus\)/.test(css), "no visible :focus-visible ring");
assert.ok(!/outline\s*:\s*(none|0)\b/.test(css), "a rule removes the focus outline");
const motion = css.match(/@media \(prefers-reduced-motion: reduce\)\s*\{([\s\S]*?)\}\s*\}/);
assert.ok(motion && /animation:\s*none/.test(motion[1]) && /transition:\s*none/.test(motion[1]), "reduced motion is not honoured");
assert.ok(/\.toast\.err::before\s*\{[^}]*content:\s*"[^"]+"/.test(css) && /\.toast\.ok::before\s*\{[^}]*content:\s*"[^"]+"/.test(css), "toast kinds need a symbol, not only a colour");

console.log(`a11y_css.test.js: all assertions passed (${checks.length} contrast pairs)`);
