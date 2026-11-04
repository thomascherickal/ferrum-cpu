// The Capable tab: the machine capability report (cards plus the detail dialog). It keeps the
// bounds the GGUF and Export tabs check model sizes against.
"use strict";
(function (global) {
  function create(env) {
    const { $, invoke, toast, openDialog, errText, busy } = env;
    const { esc, card } = global.FerrumEscape;
    const { fmtBytes, fmtParams } = global.FerrumCommon;
    let capBounds = null; // {inferInt4/8/F32, trainChinchilla, trainFixed1b, testEval, loadInt4, finetuneHi}

    function renderCapReport(r) {
      capBounds = {
        inferInt4: r.inferInt4, inferInt8: r.inferInt8, inferF32: r.inferF32,
        trainChinchilla: r.trainChinchilla, trainFixed1b: r.trainFixed1b, testEval: r.testEval,
        loadInt4: r.loadInt4, finetuneHi: r.finetuneHi,
      };
      // Summary cards on the panel.
      $("capCards").innerHTML =
        card("CPU", r.cpu) +
        card("Cores / threads", `${r.cores} / ${r.threads}`) +
        card("Memory", `${fmtBytes(r.memAvail)} free / ${fmtBytes(r.memTotal)}`) +
        card("Mem bandwidth", r.memBwGbps.toFixed(1) + " GB/s") +
        card("GEMM throughput", r.gemmGflops.toFixed(1) + " GFLOP/s") +
        card("Load @int4 ceiling", fmtParams(r.loadInt4)) +
        card("Infer @int8 ceiling", fmtParams(r.inferInt8));
      // Detailed four-range table inside the dialog.
      const rng = (lo, hi) => {
        const l = fmtParams(lo), h = fmtParams(hi);
        return l === h ? l : `${l} – ${h}`;
      };
      const row = (label, val, note) =>
        `<tr><td>${esc(label)}</td><td>${esc(val)}</td><td class="muted">${esc(note)}</td></tr>`;
      const trainLo = Math.min(r.trainChinchilla, r.trainFixed1b);
      const trainHi = Math.max(r.trainChinchilla, r.trainFixed1b);
      $("capDialogBody").innerHTML = `
        <p class="muted">${esc(r.cpu)} · ${esc(r.cores)} cores · ${esc(r.memBwGbps.toFixed(1))} GB/s · ${esc(r.gemmGflops.toFixed(1))} GFLOP/s</p>
        <table class="data">
          <thead><tr><th>Capability</th><th>Range (params)</th><th>Basis</th></tr></thead>
          <tbody>
            ${row("Load", rng(r.loadF32, r.loadInt4),
                  `f32 → int4 (int8: ${fmtParams(r.loadInt8)}); fits in ${Math.round(r.loadFraction * 100)}% of free RAM`)}
            ${row("Train (scratch)", rng(trainLo, trainHi),
                  `Chinchilla ${r.chinchillaRatio}× vs fixed ${fmtParams(r.fixedTrainTokens)}-token corpus; < ${r.trainHours} h, RAM-capped @16 B/param`)}
            ${row("Fine-tune", rng(r.finetuneLo, r.finetuneHi),
                  `${fmtParams(r.finetuneTokensLo)} → ${fmtParams(r.finetuneTokensHi)} token corpus; < ${r.trainHours} h, RAM-capped @16 B/param`)}
            ${row("Inference", rng(r.inferF32, r.inferInt4),
                  `f32 → int4 (int8: ${fmtParams(r.inferInt8)}); ≥ ${r.targetToks} tok/s, bandwidth-bound, RAM-capped`)}
            ${row("Eval pass", fmtParams(r.testEval),
                  `${fmtParams(r.evalTokens)} tokens, < ${r.trainHours} h`)}
          </tbody>
        </table>
        <p class="capfoot">Ranges from a live micro-benchmark: each capability spans its natural axis
          (precision f32 → int4, or corpus size). Decode is bandwidth-bound; train/fine-tune/eval are
          compute-bound (≈6·N·T train, 2·N·T eval) and memory-capped. Real throughput varies with
          architecture, context length, and other load.</p>`;
    }

    $("capCheck").addEventListener("click", async () => {
      $("capStatus").textContent = "benchmarking…";
      busy("capSpin", true);
      $("capCheck").disabled = true;
      try {
        const r = await invoke("capability_report");
        renderCapReport(r);
        $("capStatus").textContent = "";
        openDialog($("capDialog"));
        toast("Capability check complete", "ok");
      } catch (e) {
        $("capStatus").textContent = "error";
        toast("Capability check failed: " + errText(e), "error");
      } finally { $("capCheck").disabled = false; busy("capSpin", false); }
    });

    return { getBounds: () => capBounds };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumCapableTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
