// The Models tab: inspect a saved Ferrum model.
"use strict";
(function (global) {
  function create(env) {
    const { $, invoke, toast, setErr, clearErr, reqStr, errText } = env;
    const { modelInfo } = global.FerrumEscape;

    $("miInspect").addEventListener("click", async () => {
      clearErr("errModel");
      let path;
      try { path = reqStr("miPath", "Model file"); } catch (e) { setErr("errModel", e); return; }
      try {
        const i = await invoke("model_info", { path });
        $("miInfo").innerHTML = modelInfo(i);
        toast("Model loaded", "ok");
      } catch (e) { setErr("errModel", e); toast("Inspect failed: " + errText(e), "error"); }
    });
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumModelsTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
