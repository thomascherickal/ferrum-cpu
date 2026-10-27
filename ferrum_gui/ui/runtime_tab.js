// The Export tab's "run it in Ollama or LM Studio" buttons: preview (starts nothing), export to
// Ollama, import into LM Studio, write the Modelfile only. Everything it needs from the page comes
// in through `env`, so the Node tests can drive it against the stub DOM. Text from the backend only
// reaches the page through textContent / value.
"use strict";
(function (global) {
  // The form as `RuntimeParams` for one target. Throws an Error with a plain message.
  function readParams($, target, withModelfileOut) {
    const modelPath = $("rxPath").value.trim();
    if (!modelPath) throw new Error("Ferrum model (.gguf) is required");
    const modelfileOut = withModelfileOut ? $("rxModelfileOut").value.trim() : "";
    if (withModelfileOut && !modelfileOut) throw new Error("Modelfile to write is required");
    const name = $("rxName").value.trim();
    if (target === "ollama" && !name) throw new Error("Ollama model name is required");
    return {
      modelPath, target, name: target === "ollama" ? name : "",
      repo: target === "lmstudio" ? $("rxRepo").value.trim() : "",
      bin: $(target === "ollama" ? "rxOllamaBin" : "rxLmsBin").value.trim(),
      modelfileOut,
    };
  }

  const NAMES = { ollama: "Ollama", lmstudio: "LM Studio" };

  // Text for one preview: the tool that would run and the commands, or why there is no tool.
  function previewText(r) {
    return [
      `${NAMES[r.target]}: ` + (r.tool ? `tool ${r.tool}` : `tool not found (${r.toolError})`),
      ...r.commands.map((c) => "  " + c),
    ].join("\n");
  }

  // Text after a run (or a Modelfile written): what was told to the user, the commands, the output.
  function runText(r) {
    return [
      r.summary,
      ...r.commands.map((c) => (r.ran ? "ran: " : "would run: ") + c),
      r.output && r.output.trim() ? "output: " + r.output.trim() : null,
    ].filter(Boolean).join("\n");
  }

  function create(env) {
    const { $, invoke, toast, setErr, clearErr, jobRunning, stopAsked, errText, isCancelled } = env;
    const buttons = ["rxPreview", "rxOllama", "rxLms", "rxWriteModelfile"];
    const busy = (on) => buttons.forEach((b) => { $(b).disabled = on; });

    async function run(target, withModelfileOut) {
      clearErr("errRuntime");
      let params;
      try { params = readParams($, target, withModelfileOut); } catch (e) { setErr("errRuntime", e.message); return; }
      $("rxResult").textContent = "";
      $("rxStatus").textContent = withModelfileOut ? "writing..." : `running ${NAMES[target]} (an Ollama create can take a while)...`;
      busy(true);
      jobRunning("rxCancel", "runtime_export", true);
      try {
        const r = await invoke("export_to_runtime", { params });
        if (r.modelfile) $("rxModelfile").value = r.modelfile;
        $("rxResult").textContent = runText(r);
        $("rxStatus").textContent = "done";
        toast(r.summary, "ok");
      } catch (e) {
        if (stopAsked.runtime_export && isCancelled(e)) {
          $("rxStatus").textContent = "stopped";
        } else {
          $("rxStatus").textContent = "error";
          setErr("errRuntime", e);
          toast("Export failed: " + errText(e), "error");
        }
      } finally { busy(false); jobRunning("rxCancel", "runtime_export", false); }
    }

    $("rxOllama").addEventListener("click", () => run("ollama", false));
    $("rxLms").addEventListener("click", () => run("lmstudio", false));
    $("rxWriteModelfile").addEventListener("click", () => run("ollama", true));

    $("rxPreview").addEventListener("click", async () => {
      clearErr("errRuntime");
      let ollama, lmstudio;
      try { ollama = readParams($, "ollama", false); lmstudio = readParams($, "lmstudio", false); } catch (e) { setErr("errRuntime", e.message); return; }
      $("rxStatus").textContent = "looking...";
      busy(true);
      try {
        const a = await invoke("preview_runtime_export", { params: ollama });
        const b = await invoke("preview_runtime_export", { params: lmstudio });
        $("rxModelfile").value = a.modelfile || "";
        $("rxResult").textContent = previewText(a) + "\n" + previewText(b) + "\nNothing was run.";
        $("rxStatus").textContent = "";
      } catch (e) {
        $("rxStatus").textContent = "error";
        setErr("errRuntime", e);
      } finally { busy(false); }
    });
  }

  const api = { create, readParams, previewText, runText };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumRuntimeTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
