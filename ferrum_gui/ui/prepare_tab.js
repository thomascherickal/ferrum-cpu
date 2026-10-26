// The Prepare tab: reads the form, calls `prepare_dataset`, shows progress and the
// summary. Everything it needs from the page comes in through `env`, so the Node
// tests can drive it against the stub DOM. Text from the backend only reaches the
// page through textContent.
"use strict";
(function (global) {
  const fmtInt = (n) => Number(n).toLocaleString("en-US");
  const fmtMb = (b) => (b / 1e6).toFixed(1) + " MB";

  // The form as `prepare_dataset` parameters. Throws an Error with a plain message.
  function readParams($) {
    const inputs = $("dpInputs").value.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!inputs.length) throw new Error("Add at least one input file or folder");
    const outDir = $("dpOut").value.trim();
    if (!outDir) throw new Error("Output folder is required");
    const whole = (id, name) => {
      const v = Number($(id).value);
      if (!Number.isInteger(v) || v < 0) throw new Error(name + " must be a whole number, 0 or more");
      return v;
    };
    const frac = (id, name) => {
      const v = Number($(id).value);
      if (!Number.isFinite(v) || v < 0 || v >= 1) throw new Error(name + " must be at least 0 and below 1");
      return v;
    };
    const valFrac = frac("dpVal", "Validation share");
    const testFrac = frac("dpTest", "Test share");
    if (valFrac + testFrac >= 1) throw new Error("Validation share and test share must add up to less than 1");
    return {
      inputs, outDir, valFrac, testFrac,
      seed: whole("dpSeed", "Split seed"),
      minChars: whole("dpMin", "Min chars"),
      maxChars: whole("dpMax", "Max chars"),
      docSep: $("dpSep").value,
      dedup: $("dpDedup").checked,
      nfkc: $("dpNfkc").checked,
      lowercase: $("dpLower").checked,
      stripGutenberg: $("dpGutenberg").checked,
      lossy: $("dpLossy").checked,
    };
  }

  // The summary shown after a run (also what the tests compare).
  function resultText(r) {
    const split = (name, s) => `${name}: ${fmtInt(s.documents)} documents, ${fmtInt(s.chars)} characters, ${fmtInt(s.bytes)} bytes -> ${s.path}`;
    const rejected = Object.entries(r.rejected).filter(([, n]) => n > 0).map(([k, n]) => `${k.replace("_", " ")}=${fmtInt(n)}`).join("  ") || "none";
    return [
      `Prepared ${r.files} file(s) in ${r.seconds.toFixed(1)} s`,
      split("train", r.train), split("val", r.val), split("test", r.test),
      `rejected: ${rejected}`,
      r.lossyRepaired > 0 ? `repaired invalid UTF-8 in ${fmtInt(r.lossyRepaired)} kept document(s)` : null,
      `manifest: ${r.manifestPath}`,
      `same line as the command line: ${r.summary}`,
    ].filter(Boolean).join("\n");
  }

  function create(env) {
    const { $, invoke, toast, setErr, clearErr, jobRunning, stopAsked } = env;
    let last = null; // the result of the latest finished run

    function onProgress(p) {
      if (p.bytesTotal > 0) $("dpBar").style.width = Math.min(100, Math.round((p.bytesDone / p.bytesTotal) * 100)) + "%";
      $("dpStatus").textContent = `file ${p.file}/${p.files} - ${fmtMb(p.bytesDone)} of ${fmtMb(p.bytesTotal)} - ${fmtInt(p.documents)} documents read`;
    }

    $("dpRun").addEventListener("click", async () => {
      clearErr("errPrepare");
      let params;
      try { params = readParams($); } catch (e) { setErr("errPrepare", e.message); return; }
      $("dpResult").textContent = "";
      $("dpUse").classList.add("hidden");
      $("dpBar").style.width = "0%";
      $("dpStatus").textContent = "starting...";
      $("dpRun").disabled = true;
      jobRunning("dpCancel", "prepare", true);
      try {
        const r = await invoke("prepare_dataset", { params });
        last = r;
        $("dpBar").style.width = "100%";
        $("dpStatus").textContent = "done";
        $("dpResult").textContent = resultText(r);
        $("dpUse").classList.toggle("hidden", r.train.documents === 0);
        toast(`Prepared: train=${r.train.documents} val=${r.val.documents} test=${r.test.documents}`, "ok");
      } catch (e) {
        if (stopAsked.prepare && /cancelled/.test(String(e))) {
          $("dpStatus").textContent = "stopped; no files were changed";
        } else {
          $("dpStatus").textContent = "error";
          setErr("errPrepare", String(e));
          toast("Prepare failed: " + e, "error");
        }
      } finally { $("dpRun").disabled = false; jobRunning("dpCancel", "prepare", false); }
    });

    // Hand the splits to the Train and Evaluate tabs (only splits that have documents).
    $("dpUse").addEventListener("click", () => {
      if (!last) return;
      $("trCorpus").value = last.train.path;
      if (last.val.documents > 0) $("evFile").value = last.val.path;
      if (last.test.documents > 0) $("evTestFile").value = last.test.path;
      toast("Train and Evaluate now point at the prepared splits", "ok");
    });

    return { onProgress };
  }

  const api = { create, readParams, resultText };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumPrepareTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
