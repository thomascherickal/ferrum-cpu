// The Train tab: the form as `train_slm` parameters, the pre-flight check that gates Start, the
// loss chart and the "Tokenize corpus" block (tokenize_tab.js). Everything it needs comes in
// through `env`.
"use strict";
(function (global) {
  function create(env) {
    const { $, doc, invoke, hasTauri, setBar, toast, setErr, clearErr, errText, isCancelled, num, int, reqInt, reqNum, reqStr, jobRunning, stopAsked } = env;
    const { preflightView, preflightWhy } = global.FerrumEscape;
    const { drawLoss } = global.FerrumCommon;

    let lossHistory = [];
    const drawChart = () => drawLoss($, "trChart", lossHistory);

    // Pre-flight (preflight_train): parameter count, estimated peak RAM, llama.cpp verdict. It
    // follows the form (300 ms after the last change); Start is blocked while the run would exceed
    // what the engine allows, unless "I know, try anyway" is ticked.
    let preflight = null;
    let preflightSeq = 0;
    let preflightTimer = null;
    let trRunning = false;
    function updateStartGate() {
      const why = preflightWhy(preflight);
      const blocked = !!why && !$("trForce").checked;
      $("trForceLabel").classList.toggle("hidden", !why);
      $("trPreflightWhy").textContent = !why ? "" : blocked ? why : "RAM check skipped: the run may be slow, swap, or be killed by the system.";
      $("trStart").disabled = trRunning || blocked;
    }
    async function runPreflight() {
      if (!hasTauri) return;
      const n = ++preflightSeq;
      try {
        const pf = await invoke("preflight_train", { params: global.FerrumLive.PARAMS.trStart(doc) });
        if (n !== preflightSeq) return; // a newer request superseded this one
        preflight = pf;
        $("trPreflight").innerHTML = preflightView(pf);
      } catch (e) {
        if (n !== preflightSeq) return;
        preflight = null;
        $("trPreflight").textContent = "Pre-flight unavailable: " + errText(e);
      }
      updateStartGate();
    }
    function schedulePreflight() {
      clearTimeout(preflightTimer);
      preflightTimer = setTimeout(runPreflight, 300);
    }
    $("panel-train").querySelectorAll("input,select").forEach((el) => {
      el.addEventListener("input", schedulePreflight);
      el.addEventListener("change", schedulePreflight);
    });
    if ($("trPreset")) {
      $("trPreset").addEventListener("change", () => {
        const v = $("trPreset").value;
        if (v !== "custom" && global.FerrumMode) {
          global.FerrumMode.applyPreset(doc, v);
          schedulePreflight();
        }
      });
    }
    if ($("btnGoGenerate")) {
      $("btnGoGenerate").addEventListener("click", () => {
        const path = $("trModel").value.trim();
        if (path) $("genModel").value = path;
        $("tab-generate").click();
      });
    }
    if ($("btnGoEvaluate")) {
      $("btnGoEvaluate").addEventListener("click", () => {
        const path = $("trModel").value.trim();
        if (path) $("evModel").value = path;
        $("tab-evaluate").click();
      });
    }
    $("trForce").addEventListener("change", updateStartGate);
    // A finished token file becomes the Train corpus, and the pre-flight is asked again.
    const tokenizeTab = global.FerrumTokenizeTab.create({
      $, invoke, setBar, toast, setErr, clearErr, errText, isCancelled, jobRunning, stopAsked,
      useAsCorpus: (path) => { $("trCorpus").value = path; schedulePreflight(); },
    });
    $("tab-train").addEventListener("click", schedulePreflight);

    $("trStart").addEventListener("click", async () => {
      clearErr("errTrain");
      if (preflight && preflightWhy(preflight) && !$("trForce").checked) {
        setErr("errTrain", preflightWhy(preflight));
        return;
      }
      let params;
      try {
        params = {
          method: "transformer",
          corpusPath: reqStr("trCorpus", "Corpus"),
          modelPath: reqStr("trModel", "Model output"),
          contextLen: reqInt("trContext", "Context"),
          embedDim: reqInt("trEmbed", "Embed dim"),
          numHeads: reqInt("trHeads", "Heads"),
          numBlocks: reqInt("trBlocks", "Blocks"),
          hiddenDim: reqInt("trHidden", "Hidden"),
          epochs: reqInt("trEpochs", "Epochs"),
          lr: reqNum("trLr", "Learning rate", 0),
          batchSize: reqInt("trBatch", "Batch"),
          vocabSize: Math.max(0, int("trVocab") || 0),
          seed: Math.max(0, int("trSeed") || 0),
          threads: Math.max(0, int("trThreads") || 0),
          verbose: $("trVerbose").checked,
          windowStride: Math.max(0, int("trStride") || 0),
          maxTokens: Math.max(0, Math.floor(Number($("trTokens").value) || 0)),
          weightDecay: Math.max(0, num("trWd") || 0),
          dropout: Math.max(0, num("trDropout") || 0),
          gradClip: Math.max(0, num("trClip") || 0),
          warmupSteps: Math.max(0, int("trWarmup") || 0),
          cosineLr: $("trCosine").checked,
          weightTying: $("trTie").checked,
          qat: $("trQat").checked,
          tokenizerPath: $("trTokenizer").value.trim(),
          checkpointPath: $("trCkpt").value.trim(),
          checkpointEvery: Math.max(0, int("trCkptEvery") || 0),
          resume: $("trResume").checked,
          valFraction: Math.max(0, num("trVal") || 0),
          patience: Math.max(0, int("trPatience") || 0),
          force: $("trForce").checked,
          quant: $("trQuant").value,
        };
        if (params.resume && !params.checkpointPath)
          throw new Error("Resume needs a checkpoint file");
        // Client-side guards mirroring the backend, for instant feedback.
        if (params.embedDim % params.numHeads !== 0)
          throw new Error(`Embed dim (${params.embedDim}) must be divisible by heads (${params.numHeads})`);
        if (params.vocabSize !== 0 && params.vocabSize < 256)
          throw new Error("Vocab must be 0 (char-level) or ≥ 256 (BPE)");
      } catch (e) { setErr("errTrain", e); return; }

      lossHistory = []; drawChart();
      setBar("trBar", 0);
      $("trResult").textContent = "";
      $("trStatus").textContent = "training…";
      trRunning = true;
      updateStartGate();
      jobRunning("trCancel", "train", true);
      try {
        const r = await invoke("train_slm", { params });
        $("trStatus").textContent = r.stopped ? `stopped at step ${r.stoppedAtStep}` : "done";
        $("trResult").textContent = (r.stopped
          ? `Stopped at step ${r.stoppedAtStep}; the model trained so far was saved` +
            (params.checkpointPath ? ` and the checkpoint written (tick Resume to continue).\n` : ".\n")
          : "") +
          `Saved ${r.bytes.toLocaleString()} bytes → ${r.modelPath}\n` +
          `method=${r.method}  final loss=${r.finalLoss.toFixed(6)}  time=${r.seconds.toFixed(2)}s\n` +
          `input_dim=${r.inputDim}  output_dim=${r.outputDim}  tokenizer=${r.tokenizer}  layers=${r.layers}`;
        toast(r.stopped ? `Training stopped at step ${r.stoppedAtStep}; model saved` : "Training complete", "ok");
        if (!$("genModel").value) $("genModel").value = r.modelPath;
        if (!$("evModel").value) $("evModel").value = r.modelPath;
        if (!$("miPath").value) $("miPath").value = r.modelPath;
        if (!$("rxPath").value && r.tokenizer === "byte-level BPE") $("rxPath").value = r.modelPath;
      } catch (e) {
        $("trStatus").textContent = "error";
        setErr("errTrain", e);
        toast("Training failed: " + errText(e), "error");
      } finally { trRunning = false; updateStartGate(); jobRunning("trCancel", "train", false); }
    });

    function onProgress(p) {
      // total is null under a token budget (epoch count unknown up front).
      if (p.total) setBar("trBar", Math.round((p.epoch / p.total) * 100));
      $("trStatus").textContent = `epoch ${p.epoch}${p.total ? "/" + p.total : ""} — loss ${p.loss.toFixed(6)}`;
      lossHistory.push(p.loss);
      if (lossHistory.length > 1000) lossHistory.shift();
      drawChart();
    }

    return { onProgress, onTokenizeProgress: tokenizeTab.onProgress, runPreflight };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumTrainTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
