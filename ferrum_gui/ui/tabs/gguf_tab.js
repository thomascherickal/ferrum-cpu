// The GGUF tab: inspect an imported Llama/Qwen GGUF and run it.
"use strict";
(function (global) {
  function create(env) {
    const { $, invoke, toast, setErr, clearErr, reqInt, reqStr, samplerParams, jobRunning, stopAsked, budget, getBounds, errText, isCancelled, busy } = env;
    const { ggufInfo } = global.FerrumEscape;

    $("ggInspect").addEventListener("click", async () => {
      clearErr("errGguf");
      let path;
      try { path = reqStr("ggPath", "GGUF file"); } catch (e) { setErr("errGguf", e); return; }
      $("ggStatus").textContent = "inspecting…";
      busy("ggSpin", true);
      try {
        const i = await invoke("gguf_info", { path });
        $("ggInfo").innerHTML = ggufInfo(i);
        $("ggStatus").textContent = "";
        busy("ggSpin", false);
        toast(i.runnable ? "GGUF inspected" : "Inspected — architecture not runnable", i.runnable ? "ok" : "error");
        const crossedI = budget.check(i.paramCount);
        if (crossedI.length) {
          await budget.confirm(i.paramCount, crossedI); // advisory on inspect; result ignored
        } else if (!getBounds()) {
          toast("Tip: run the Capable check to flag oversized models.", "info");
        }
      } catch (e) { busy("ggSpin", false); $("ggStatus").textContent = "error"; setErr("errGguf", e); toast("Inspect failed: " + errText(e), "error"); }
    });

    $("ggRun").addEventListener("click", async () => {
      clearErr("errGguf");
      let params;
      try {
        const seed = $("ggSeed").value.trim();
        const ids = $("ggIds").value.trim();
        const resume = $("ggResume").value.trim();
        params = {
          modelPath: reqStr("ggPath", "GGUF file"),
          prompt: $("ggPrompt").value,
          quant: $("ggQuant").value,
          maxNew: reqInt("ggMax", "Max new tokens"),
          ...samplerParams("gg"),
          genSeed: seed === "" ? null : parseInt(seed, 10),
          ids: ids === "" ? null : ids,
          force: $("ggForce").checked,
          resume: resume === "" ? null : resume,
          raw: !$("ggChatFmt").checked,
          stop: $("ggStop").value.split("\n").map((l) => l.replace(/\r$/, "")).filter((l) => l !== ""),
        };
        if (!params.prompt.trim() && !params.ids) throw new Error("Enter a prompt (or token IDs)");
      } catch (e) { setErr("errGguf", e); return; }

      // Budget gate: if the model exceeds this machine's estimated limits, confirm.
      if (getBounds()) {
        try {
          const info = await invoke("gguf_info", { path: params.modelPath });
          const crossed = budget.check(info.paramCount);
          if (crossed.length) {
            const ok = await budget.confirm(info.paramCount, crossed);
            if (!ok) { $("ggStatus").textContent = "cancelled"; return; }
          }
        } catch (_) { /* inspection failed; fall through to run, which will error clearly */ }
      }

      $("ggOut").textContent = "";
      $("ggStatus").textContent = "loading & generating… (CPU: this can take a while)";
      busy("ggSpin", true);
      $("ggRun").disabled = true;
      jobRunning("ggCancel", "run", true);
      try {
        const r = await invoke("run_gguf", { params });
        $("ggOut").textContent = r.text;
        $("ggStatus").textContent =
          `${r.stopped ? "stopped" : "done"} — ${r.generated} tokens in ${r.seconds.toFixed(1)}s (${r.tokensPerSec.toFixed(1)} tok/s, prompt ${r.promptTokens} tok)`;
      } catch (e) {
        if (stopAsked.run && isCancelled(e)) {
          $("ggStatus").textContent = "stopped before any text was produced";
        } else {
          $("ggStatus").textContent = "error";
          setErr("errGguf", e);
          toast("Run failed: " + errText(e), "error");
        }
      } finally { busy("ggSpin", false); $("ggRun").disabled = false; jobRunning("ggCancel", "run", false); }
    });
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumGgufTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
