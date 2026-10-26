// Regression guard: every backend command the UI calls is registered in
// src/lib.rs, every registered command is called by the UI, and every command
// has a #[tauri::command] definition. A typo in either side fails here instead
// of at runtime in a window nobody can click-test.
// Run with: node ui/commands.test.js
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const ui = __dirname;
const src = path.join(ui, "..", "src");
const used = new Set();
for (const f of fs.readdirSync(ui).filter((f) => f.endsWith(".js") && !f.endsWith(".test.js") && !f.startsWith("test"))) {
  for (const m of fs.readFileSync(path.join(ui, f), "utf8").matchAll(/\binvoke\(\s*"([a-z_]+)"/g)) used.add(m[1]);
}

const lib = fs.readFileSync(path.join(src, "lib.rs"), "utf8");
const handler = lib.match(/generate_handler!\[([\s\S]*?)\]/)[1];
const registered = new Set([...handler.matchAll(/(?:\w+::)*(\w+),/g)].map((m) => m[1]));

// The Rust side: #[tauri::command] fns across src/*.rs.
const defined = new Set();
for (const f of fs.readdirSync(src).filter((f) => f.endsWith(".rs"))) {
  const text = fs.readFileSync(path.join(src, f), "utf8");
  for (const m of text.matchAll(/#\[tauri::command\]\s*pub\s+(?:async\s+)?fn\s+(\w+)/g)) defined.add(m[1]);
}

assert.ok(used.size >= 15, "found only " + used.size + " invoke() calls");
for (const c of used) assert.ok(registered.has(c), `UI calls '${c}' but lib.rs does not register it`);
for (const c of registered) {
  assert.ok(defined.has(c), `'${c}' is registered but no #[tauri::command] fn has that name`);
  assert.ok(used.has(c), `'${c}' is registered but the UI never calls it`);
}
for (const c of defined) assert.ok(registered.has(c), `#[tauri::command] '${c}' is not registered in lib.rs`);
for (const c of ["hardware_info", "estimate_job"]) assert.ok(used.has(c) && registered.has(c), c);

console.log(`commands.test.js: all assertions passed (${used.size} commands)`);
