// Regression guard for the strict Content-Security-Policy: no inline script or
// style in the page, no inline handlers, no style attributes written from JS,
// no eval, and the CSP and window capabilities are not loosened.
// Run with: node ui/csp.test.js
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");

assert.ok(!/\sstyle\s*=/.test(html), "index.html has a style= attribute (blocked by the CSP)");
assert.ok(!/<style[\s>]/i.test(html), "index.html has an inline <style>");
assert.ok(!/\son[a-z]+\s*=/i.test(html), "index.html has an inline event handler");
assert.ok(!/javascript:/i.test(html), "index.html has a javascript: URL");
for (const m of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
  assert.ok(/\bsrc="[\w.-]+\.js"/.test(m[1]), "script without a local src: " + m[1]);
  assert.strictEqual(m[2].trim(), "", "inline script body");
}
for (const m of html.matchAll(/<(?:link|img)\b[^>]*(?:href|src)="([^"]+)"/g)) {
  assert.ok(!/^[a-z]+:\/\//i.test(m[1]), "external resource " + m[1]);
}
// Every script the page loads exists.
for (const m of html.matchAll(/<script src="([^"]+)"/g)) assert.ok(fs.existsSync(path.join(__dirname, m[1])), m[1]);

for (const f of fs.readdirSync(__dirname).filter((f) => f.endsWith(".js") && !f.endsWith(".test.js") && !f.startsWith("test"))) {
  const js = fs.readFileSync(path.join(__dirname, f), "utf8");
  assert.ok(!/setAttribute\(\s*["']style["']/.test(js), `${f}: setAttribute("style")`);
  assert.ok(!/\.cssText\s*=/.test(js), `${f}: cssText`);
  assert.ok(!/\beval\s*\(|new Function\s*\(/.test(js), `${f}: eval`);
  assert.ok(!/document\.write|insertAdjacentHTML|outerHTML\s*=/.test(js), `${f}: raw HTML insertion`);
}
// The tooltip module may only use innerHTML-free rendering.
assert.ok(!/innerHTML/.test(fs.readFileSync(path.join(__dirname, "tooltip.js"), "utf8").replace(/\/\/.*$/gm, "")), "tooltip.js uses innerHTML");
assert.ok(!/innerHTML/.test(fs.readFileSync(path.join(__dirname, "live.js"), "utf8").replace(/\/\/.*$/gm, "")), "live.js uses innerHTML");

const conf = JSON.parse(fs.readFileSync(path.join(root, "tauri.conf.json"), "utf8"));
const csp = conf.app.security.csp;
for (const d of ["default-src 'self'", "script-src 'self'", "style-src 'self'", "object-src 'none'", "base-uri 'none'", "form-action 'none'"]) {
  assert.ok(csp.includes(d), "CSP lost '" + d + "'");
}
assert.ok(!/unsafe-inline|unsafe-eval|\*/.test(csp), "CSP was loosened: " + csp);

const caps = JSON.parse(fs.readFileSync(path.join(root, "capabilities", "default.json"), "utf8"));
assert.deepStrictEqual([...caps.permissions].sort(), ["core:event:allow-listen", "core:event:allow-unlisten", "dialog:allow-open", "dialog:allow-save"]);

console.log("csp.test.js: all assertions passed");
