// The Generate tab: sampling controls, streamed output reconciled with the returned text (G1).
"use strict";
(function (global) {
  function create(env) {
    const { $, invoke, toast, setErr, clearErr, reqInt, reqStr, samplerParams, jobRunning, stopAsked } = env;
    let generating = false;

    $("genStart").addEventListener("click", async () => {
      clearErr("errGen");
      let params;
      try {
        const seedNum = $("genSeedNum").value.trim();
        params = {
          modelPath: reqStr("genModel", "Model"),
          seedText: $("genSeed").value,
          numChars: reqInt("genChars", "Chars"),
          ...samplerParams("gen"),
          maxTokens: reqInt("genMaxTokens", "Max tokens", 0),
          stop: $("genStop").value.split("\n").map((l) => l.replace(/\r$/, "")).filter((l) => l !== ""),
          genSeed: seedNum === "" ? null : parseInt(seedNum, 10),
          stream: $("genStream").checked,
          verbose: $("genVerbose").checked,
        };
      } catch (e) { setErr("errGen", String(e)); return; }

      $("genOut").textContent = "";
      $("genStatus").textContent = "generating…";
      $("genStart").disabled = true;
      generating = params.stream;
      jobRunning("genCancel", "generate", true);
      try {
        const out = await invoke("generate_slm", { params });
        const stopped = stopAsked.generate; // Stop was pressed (the text so far is returned)
        // Stop accepting late gen-fragment events *before* reconciling, then set the
        // display from the authoritative return value so a dropped/late stream tail
        // can never leave the output truncated (G1).
        generating = false;
        if (params.stream) {
          $("genOut").textContent = global.streamContinuation(out, params.seedText);
        } else {
          $("genOut").textContent = out;
        }
        $("genStatus").textContent = `${stopped ? "stopped" : "done"} (${out.length} chars)`;
      } catch (e) {
        $("genStatus").textContent = "error";
        setErr("errGen", String(e));
        toast("Generation failed: " + e, "error");
      } finally { generating = false; $("genStart").disabled = false; jobRunning("genCancel", "generate", false); }
    });

    return { onFragment: (text) => { if (generating) $("genOut").textContent += text; } };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumGenerateTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
