// The Train tab's "Tokenize corpus" block: reads the form (the tokenizer comes from the Train
// tab's Tokenizer file field), calls `tokenize_corpus`, shows progress and the summary, and
// offers the token file to Train as its corpus. Everything it needs from the page comes in
// through `env`, so the Node tests can drive it against the stub DOM. Text from the backend only
// reaches the page through textContent.
"use strict";
(function (global) {
  const fmtInt = (n) => Number(n).toLocaleString("en-US");
  const fmtMb = (b) => (b / 1e6).toFixed(1) + " MB";

  // The form as `TokenizeParams`. Throws an Error with a plain message.
  function readParams($) {
    const input = $("tkInput").value.trim();
    if (!input) throw new Error("Text file or folder is required");
    const tokenizerPath = $("trTokenizer").value.trim();
    if (!tokenizerPath) throw new Error("Fill in the Tokenizer file field above (an existing tokenizer made by an earlier training run)");
    const outPath = $("tkOut").value.trim();
    if (!outPath) throw new Error("Token file to write is required");
    return { input, tokenizerPath, outPath, docSep: $("tkSep").value };
  }

  function resultText(r) {
    return [
      `Wrote ${fmtInt(r.tokens)} tokens from ${fmtInt(r.documents)} documents (${r.files} file(s), vocabulary ${r.vocabSize}) in ${r.seconds.toFixed(1)} s`,
      `${fmtInt(r.bytesIn)} bytes of text -> ${fmtInt(r.bytesOut)} bytes -> ${r.outPath}`,
      `same line as the command line: ${r.summary}`,
    ].join("\n");
  }

  function create(env) {
    const { $, invoke, setBar, toast, setErr, clearErr, jobRunning, stopAsked, useAsCorpus, errText, isCancelled } = env;

    function onProgress(p) {
      if (p.bytesTotal > 0) setBar("tkBar", Math.min(100, Math.round((p.bytesDone / p.bytesTotal) * 100)));
      $("tkStatus").textContent = `file ${p.file}/${p.files} - ${fmtMb(p.bytesDone)} of ${fmtMb(p.bytesTotal)} - ${fmtInt(p.tokens)} tokens`;
    }

    $("tkRun").addEventListener("click", async () => {
      clearErr("errTokenize");
      let params;
      try { params = readParams($); } catch (e) { setErr("errTokenize", e.message); return; }
      $("tkResult").textContent = "";
      setBar("tkBar", 0);
      $("tkStatus").textContent = "starting...";
      $("tkRun").disabled = true;
      jobRunning("tkCancel", "tokenize", true);
      try {
        const r = await invoke("tokenize_corpus", { params });
        setBar("tkBar", 100);
        $("tkStatus").textContent = "done";
        $("tkResult").textContent = resultText(r);
        useAsCorpus(r.outPath);
        toast(`Tokenized ${fmtInt(r.tokens)} tokens; the Train corpus now points at the token file`, "ok");
      } catch (e) {
        if (stopAsked.tokenize && isCancelled(e)) {
          $("tkStatus").textContent = "stopped; no file was written";
        } else {
          $("tkStatus").textContent = "error";
          setErr("errTokenize", e);
          toast("Tokenize failed: " + errText(e), "error");
        }
      } finally { $("tkRun").disabled = false; jobRunning("tkCancel", "tokenize", false); }
    });

    return { onProgress };
  }

  const api = { create, readParams, resultText };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumTokenizeTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
