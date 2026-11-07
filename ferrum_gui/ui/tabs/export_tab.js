// The Export tab: write a loaded, optionally fine-tuned, GGUF back out; the Ollama / LM Studio
// block at its end is runtime_tab.js.
"use strict";
(function (global) {
  function create(env) {
    const { $, invoke, toast, setErr, clearErr, reqStr, jobRunning, stopAsked, budget, getBounds, errText, isCancelled, busy } = env;
    const { exportInfo } = global.FerrumEscape;

    $("exInspect").addEventListener("click", async () => {
      clearErr("errExport");
      let path;
      try { path = reqStr("exPath", "Source GGUF"); } catch (e) { setErr("errExport", e); return; }
      $("exStatus").textContent = "inspecting…";
      busy("exSpin", true);
      try {
        const i = await invoke("gguf_info", { path });
        $("exInfo").innerHTML = exportInfo(i);
        $("exStatus").textContent = "";
        busy("exSpin", false);
        toast(i.runnable ? "GGUF inspected" : "Inspected — architecture not exportable", i.runnable ? "ok" : "error");
        const crossed = budget.check(i.paramCount);
        if (crossed.length) await budget.confirm(i.paramCount, crossed); // advisory; result ignored
      } catch (e) { busy("exSpin", false); $("exStatus").textContent = "error"; setErr("errExport", e); toast("Inspect failed: " + errText(e), "error"); }
    });

    $("exExport").addEventListener("click", async () => {
      clearErr("errExport");
      let params;
      try {
        const resume = $("exResume").value.trim();
        params = {
          modelPath: reqStr("exPath", "Source GGUF"),
          outPath: reqStr("exOut", "Output file"),
          quant: $("exQuant").value,
          resume: resume === "" ? null : resume,
          force: $("exForce").checked,
          verbose: $("exVerbose").checked,
        };
      } catch (e) { setErr("errExport", e); return; }

      // Budget gate (advisory + confirm), mirroring the GGUF Run button.
      if (getBounds()) {
        try {
          const info = await invoke("gguf_info", { path: params.modelPath });
          const crossed = budget.check(info.paramCount);
          if (crossed.length) {
            const ok = await budget.confirm(info.paramCount, crossed);
            if (!ok) { $("exStatus").textContent = "cancelled"; return; }
          }
        } catch (_) { /* inspection failed; the export itself will error clearly */ }
      }

      $("exResult").textContent = "";
      $("exStatus").textContent = "starting… (CPU: a large model can take minutes)";
      busy("exSpin", true);
      $("exExport").disabled = true;
      jobRunning("exCancel", "export", true);
      try {
        const r = await invoke("export_gguf", { params });
        $("exStatus").textContent = "done";
        $("exResult").textContent =
          `Wrote ${r.bytes.toLocaleString()} bytes → ${r.outPath} in ${r.seconds.toFixed(1)}s\n` +
          `tensors: ${r.tensorSummary}\n` +
          `The file runs in llama.cpp / ollama / LM Studio as-is.`;
        toast("Export complete", "ok");
      } catch (e) {
        if (stopAsked.export && isCancelled(e)) {
          $("exStatus").textContent = "cancelled; nothing was written";
        } else {
          $("exStatus").textContent = "error";
          setErr("errExport", e);
          toast("Export failed: " + errText(e), "error");
        }
      } finally { busy("exSpin", false); $("exExport").disabled = false; jobRunning("exCancel", "export", false); }
    });

    // Run a Ferrum model in Ollama / LM Studio (the fieldset at the end of the Export tab).
    global.FerrumRuntimeTab.create({ $, invoke, toast, setErr, clearErr, errText, isCancelled, jobRunning, stopAsked });

    return { onProgress: (text) => { $("exStatus").textContent = text; } };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumExportTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
