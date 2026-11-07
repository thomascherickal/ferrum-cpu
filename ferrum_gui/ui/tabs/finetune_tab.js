// The Fine-tune tab: AdamW-fine-tune an imported GGUF into a `.flck` checkpoint.
"use strict";
(function (global) {
  function create(env) {
    const { $, invoke, setBar, toast, setErr, clearErr, num, int, reqInt, reqNum, reqStr, jobRunning, errText, busy } = env;
    const { drawLoss } = global.FerrumCommon;
    let lossHistory = [];
    const drawChart = () => drawLoss($, "ftChart", lossHistory);

    $("ftRun").addEventListener("click", async () => {
      clearErr("errFt");
      let params;
      try {
        const resume = $("ftResume").value.trim();
        params = {
          modelPath: reqStr("ftPath", "GGUF file"),
          corpusPath: reqStr("ftCorpus", "Training corpus"),
          outPath: reqStr("ftOut", "Output checkpoint"),
          epochs: reqInt("ftEpochs", "Epochs"),
          lr: reqNum("ftLr", "Learning rate", 0),
          batchSize: reqInt("ftBatch", "Batch size"),
          seqLen: reqInt("ftSeq", "Sequence length"),
          warmup: Math.max(0, int("ftWarmup") || 0),
          clip: Math.max(0, num("ftClip") || 0),
          weightDecay: Math.max(0, num("ftWd") || 0),
          dropout: Math.max(0, num("ftDropout") || 0),
          qat: $("ftQat").checked,
          seed: Math.max(0, int("ftSeed") || 0),
          threads: Math.max(0, int("ftThreads") || 0),
          resume: resume === "" ? null : resume,
          force: $("ftForce").checked,
          verbose: $("ftVerbose").checked,
          maxSeconds: null,
          sample: $("ftSample").checked,
          temp: Math.max(0, num("ftTemp") ?? 0.8),
          sampleTokens: Math.max(1, int("ftSampleTokens") || 48),
        };
        // Empty = no budget; anything else goes to the backend, which refuses what the CLI refuses.
        const secs = $("ftMaxSeconds").value.trim();
        if (secs !== "") params.maxSeconds = Number(secs);
      } catch (e) { setErr("errFt", e); return; }

      lossHistory = []; drawChart();
      setBar("ftBar", 0);
      $("ftResult").textContent = "";
      $("ftStatus").textContent = "fine-tuning… (CPU: this can take a while)";
      busy("ftSpin", true);
      $("ftRun").disabled = true;
      jobRunning("ftCancel", "finetune", true);
      try {
        const r = await invoke("finetune_gguf", { params });
        $("ftStatus").textContent = r.stopped ? `stopped at step ${r.steps}` : "done";
        $("ftResult").textContent = (r.stopped ? `Stopped at step ${r.steps}; the checkpoint so far was saved and can be resumed.\n` : "") +
          `Saved ${r.bytes.toLocaleString()} bytes → ${r.outPath}\n` +
          `epochs=${r.epochs}  first loss=${r.firstLoss.toFixed(6)}  final loss=${r.finalLoss.toFixed(6)}\n` +
          (r.sample != null ? `── sample ──\n${r.sample}\n────────────\n` : "") +
          `tokens=${r.tokens.toLocaleString()}  steps=${r.steps}  time=${r.seconds.toFixed(2)}s\n` +
          `Run it: GGUF tab → set "Fine-tune checkpoint" to ${r.outPath}`;
        toast(r.stopped ? `Fine-tune stopped at step ${r.steps}; checkpoint saved` : "Fine-tune complete", "ok");
        if (!$("ggPath").value) $("ggPath").value = params.modelPath;
        if (!$("ggResume").value) $("ggResume").value = r.outPath;
      } catch (e) {
        $("ftStatus").textContent = "error";
        setErr("errFt", e);
        toast("Fine-tune failed: " + errText(e), "error");
      } finally { busy("ftSpin", false); $("ftRun").disabled = false; jobRunning("ftCancel", "finetune", false); }
    });

    function onProgress(p) {
      setBar("ftBar", Math.round((p.epoch / p.total) * 100));
      $("ftStatus").textContent = `epoch ${p.epoch}/${p.total} — loss ${p.loss.toFixed(4)} (ppl ${p.ppl.toFixed(2)})`;
      lossHistory.push(p.loss);
      if (lossHistory.length > 1000) lossHistory.shift();
      drawChart();
    }
    return { onProgress };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumFinetuneTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
