// Every UI script also loads under Node (module.exports), not only through testapp.js's vm
// context. Requiring them covers that export branch and checks each exposes an API object.
"use strict";
const assert = require("node:assert");

const MODULES = ["common.js", "settings.js", "shell.js", "datasets_tab.js", "prepare_tab.js", "tokenize_tab.js", "train_tab.js", "generate_tab.js", "evaluate_tab.js", "models_tab.js", "capable_tab.js", "gguf_budget.js", "gguf_tab.js", "finetune_tab.js", "runtime_tab.js", "export_tab.js", "serve_tab.js", "system_tab.js"];
for (const f of MODULES) {
  const api = require("./" + f);
  assert(api && typeof api === "object" && Object.keys(api).length > 0, `${f} exports nothing`);
}
console.log("modules.test.js: all assertions passed");
