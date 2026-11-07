// Every UI script also loads under Node (module.exports), not only through testapp.js's vm
// context. Requiring them covers that export branch and checks each exposes an API object.
"use strict";
const assert = require("node:assert");

const MODULES = ["core/common.js", "core/settings.js", "core/mode.js", "core/shell.js", "tabs/datasets_tab.js", "tabs/prepare_tab.js", "tabs/tokenize_tab.js", "tabs/train_tab.js", "tabs/generate_tab.js", "tabs/evaluate_tab.js", "tabs/models_tab.js", "tabs/capable_tab.js", "core/gguf_budget.js", "tabs/gguf_tab.js", "tabs/finetune_tab.js", "tabs/runtime_tab.js", "tabs/export_tab.js", "tabs/serve_tab.js", "tabs/system_tab.js"];
for (const f of MODULES) {
  const api = require("../" + f);
  assert(api && typeof api === "object" && Object.keys(api).length > 0, `${f} exports nothing`);
}
console.log("modules.test.js: all assertions passed");
