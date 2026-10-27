// The Evaluate tab: perplexity rows appended to a table, and the JSON report of the latest run.
"use strict";
(function (global) {
  function create(env) {
    const { $, doc, invoke, toast, setErr, clearErr, reqInt, reqStr } = env;
    const { evalRow } = global.FerrumEscape;
    let lastEval = [];

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
      } catch (e) { setErr("errEval", String(e)); return; }

      $("evStatus").textContent = "evaluating…"; $("evRun").disabled = true;
      try {
        const rows = await invoke("evaluate_slm", { params });
        for (const r of rows) {
          const tr = doc.createElement("tr");
          tr.innerHTML = evalRow(r, r.perplexity < r.uniformBaseline * 0.5);
          $("evTable").querySelector("tbody").appendChild(tr);
        }
        lastEval = rows;
        $("evStatus").textContent = rows.length > 1 ? `added ${rows.length} rows` : "added";
        toast(rows.map((r) => `${r.label} perplexity ${r.perplexity.toFixed(3)}`).join(" · "), "ok");
      } catch (e) {
        $("evStatus").textContent = "error";
        setErr("errEval", String(e));
        toast("Evaluation failed: " + e, "error");
      } finally { $("evRun").disabled = false; }
    });
    $("evClear").addEventListener("click", () => { $("evTable").querySelector("tbody").textContent = ""; });

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
      } catch (e) { setErr("errEvalJson", String(e)); }
    });
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumEvaluateTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
