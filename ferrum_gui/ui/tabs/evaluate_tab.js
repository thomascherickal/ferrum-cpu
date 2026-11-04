// The Evaluate tab: perplexity rows appended to a table, and the JSON report of the latest run.
"use strict";
(function (global) {
  function create(env) {
    const { $, doc, invoke, toast, setErr, clearErr, reqInt, reqStr, settings, errText, busy } = env;
    const { evalRow } = global.FerrumEscape;
    let lastEval = [];

    // Append result rows to the table (the HTML comes from the escaping builder in escape.js).
    function addRows(rows) {
      const html = rows.map((r) => evalRow(r, r.perplexity < r.uniformBaseline * 0.5)); // all or none
      for (const h of html) {
        const tr = doc.createElement("tr");
        tr.innerHTML = h;
        $("evTable").querySelector("tbody").appendChild(tr);
      }
    }
    // The table of the last session, from settings. A stored row that does not fit is skipped.
    for (const run of settings.evalRuns()) {
      try { addRows(run); lastEval = run; } catch (_) { /* stale or hand-edited row */ }
    }

    $("evRun").addEventListener("click", async () => {
      clearErr("errEval");
      let params;
      try {
        params = {
          modelPath: reqStr("evModel", "Model"),
          text: $("evText").value.trim() || null,
          textPath: $("evFile").value.trim() || null,
          testPath: $("evTestFile").value.trim() || null,
          lossy: $("evLossy").checked,
          quant: $("evQuant").value,
          ctx: reqInt("evCtx", "Window tokens", 0),
          resume: $("evResume").value.trim() || null,
          force: $("evForce").checked,
        };
        if (!params.text && !params.textPath) throw new Error("Provide held-out text or a file");
      } catch (e) { setErr("errEval", e); return; }

      $("evStatus").textContent = "evaluating…"; $("evRun").disabled = true; busy("evSpin", true);
      try {
        const rows = await invoke("evaluate_slm", { params });
        addRows(rows);
        lastEval = rows;
        settings.addEvalRun(rows);
        $("evStatus").textContent = rows.length > 1 ? `added ${rows.length} rows` : "added";
        toast(rows.map((r) => `${r.label} perplexity ${r.perplexity.toFixed(3)}`).join(" · "), "ok");
      } catch (e) {
        $("evStatus").textContent = "error";
        setErr("errEval", e);
        toast("Evaluation failed: " + errText(e), "error");
      } finally { $("evRun").disabled = false; busy("evSpin", false); }
    });
    function clearTable() { $("evTable").querySelector("tbody").textContent = ""; lastEval = []; }
    $("evClear").addEventListener("click", () => { clearTable(); settings.clearEval(); });

    // Save the latest run as JSON: one row is exactly `eval --json`; a held-out + test pair is
    // {"held_out": {...}, "test": {...}} with the same keys in each.
    $("evSaveJson").addEventListener("click", async () => {
      clearErr("errEvalJson");
      try {
        if (!lastEval.length) throw new Error("Run an evaluation first");
        const path = reqStr("evJsonPath", "JSON output path");
        const report = lastEval.length === 1
          ? lastEval[0].report
          : Object.fromEntries(lastEval.map((r) => [r.label.replace("-", "_"), r.report]));
        const n = await invoke("save_corpus", { text: JSON.stringify(report, null, 2) + "\n", path });
        toast(`Saved ${n.toLocaleString()} bytes → ${path}`, "ok");
      } catch (e) { setErr("errEvalJson", e); }
    });

    return { clearTable };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumEvaluateTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
