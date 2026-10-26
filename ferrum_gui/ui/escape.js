// Pure HTML-escaping helpers and the card/row builders that turn untrusted
// strings (GGUF metadata, dataset catalog entries, error text) into markup.
// Shared by the browser UI (app.js) and the Node test (escape.test.js), no DOM
// or Tauri dependencies. Loaded as a plain <script> in the browser and
// `require`d in Node. Every interpolated value goes through esc().
"use strict";
(function (global) {
  const ENT = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, (c) => ENT[c]);
  }

  const card = (k, v) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`;

  // Dataset catalog entry -> the inner HTML of its description block.
  function catalogMeta(d) {
    return `<strong>${esc(d.name)}</strong> <span class="badge">${esc(d.source)}</span> ` +
      `<span class="muted">~${esc(d.approxMb)} MB · ${esc(d.format)}</span><br>` +
      `<span class="muted">${esc(d.description)}</span>`;
  }

  // Evaluate-tab table row (the <td>s of one <tr>).
  function evalRow(r, good) {
    return `<td>${esc(r.model)}</td><td>${esc(r.label)}</td><td>${esc(r.numPredictions.toLocaleString())}</td>` +
      `<td>${esc(r.crossEntropy.toFixed(4))}</td><td>${esc(r.bitsPerToken.toFixed(4))}</td>` +
      `<td>${esc(r.bitsPerByte.toFixed(4))}</td>` +
      `<td><b>${esc(r.perplexity.toFixed(4))}</b></td><td>${esc(r.uniformBaseline.toFixed(0))}</td>` +
      `<td>${esc(r.seconds.toFixed(2))}</td>` +
      `<td class="${good ? "verdict-good" : "verdict-weak"}">${good ? "learned" : "weak"}</td>`;
  }

  // Models tab: Ferrum model_info.
  function modelInfo(i) {
    return card("File", i.path) + card("Size", i.bytes.toLocaleString() + " bytes") +
      card("Format", i.format) + card("Name", i.name) + card("Task", i.task) +
      card("Input dim", i.inputDim) + card("Output dim", i.outputDim) +
      card("Tokenizer", i.tokenizer) + card("BPE merges", i.merges) + card("Layers", i.layers);
  }

  const gb = (mb) => (mb / 1024).toFixed(2);
  const availRam = (i) => (i.availMb == null ? "unknown" : gb(i.availMb) + " GB");
  const shape = (i) => `dim ${i.modelDim} · ${i.nLayers}L · ${i.nHeads}h/${i.nKvHeads}kv · vocab ${i.vocabSize}`;

  // GGUF tab: gguf_info.
  function ggufInfo(i) {
    return card("Architecture", i.architecture + (i.runnable ? " ✓" : " ✗ not runnable")) +
      card("GGUF", "v" + i.version + " · " + i.numTensors + " tensors") +
      card("Shape", shape(i)) + card("Tokenizer", i.tokenizer) +
      card("Resident (int4/int8/f32)", `${gb(i.estInt4Mb)} / ${gb(i.estInt8Mb)} / ${gb(i.estF32Mb)} GB`) +
      card("Available RAM", availRam(i)) + card("Note", i.note);
  }

  // Export tab: gguf_info.
  function exportInfo(i) {
    return card("Architecture", i.architecture + (i.runnable ? " ✓" : " ✗ not exportable")) +
      card("GGUF", "v" + i.version + " · " + i.numTensors + " tensors") +
      card("Shape", shape(i)) + card("Tokenizer", i.tokenizer) +
      card("Resident f32 (export loads f32)", gb(i.estF32Mb) + " GB") +
      card("Available RAM", availRam(i));
  }

  // Train tab pre-flight: preflight_train -> cards (numbers and the llama.cpp verdict) + problems.
  const mbText = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(2) + " GB" : Math.round(mb) + " MB");
  function preflightView(pf) {
    const ram = pf.peakRamMb == null ? "unknown"
      : `${mbText(pf.peakRamMb)} of ${pf.ramLimitMb == null ? "?" : mbText(pf.ramLimitMb)} usable` +
        (pf.ramAvailableMb == null ? "" : ` (${mbText(pf.ramAvailableMb)} free)`);
    const lc = pf.llamaCpp || { runs: false, reason: "" };
    const verdict = `<span class="${lc.runs ? "verdict-good" : "verdict-weak"}">${lc.runs ? "Yes" : "No"}</span>`;
    return card("Parameters", pf.params == null ? "unknown" : pf.params.toLocaleString()) +
      `<div class="card"><div class="k">Peak RAM (estimate)</div><div class="v ${pf.fitsInRam ? "" : "verdict-weak"}">${esc(ram)}</div></div>` +
      `<div class="card"><div class="k">Runs in llama.cpp?</div><div class="v">${verdict}</div></div>` +
      `<div class="card"><div class="k">About</div><div class="v">${esc(lc.reason)}</div></div>` +
      (pf.corpus ? card("Corpus", `Token file: ${pf.corpus.tokens.toLocaleString("en-US")} tokens, ${pf.corpus.documents.toLocaleString("en-US")} documents, vocabulary ${pf.corpus.vocab}`) +
        card("Corpus and RAM", pf.corpus.note) : "") +
      (pf.problems || []).map((m) => `<div class="card"><div class="k">Fix first</div><div class="v verdict-weak">${esc(m)}</div></div>`).join("");
  }
  // Why Start is blocked ("" when it is not): the run needs more RAM than the engine allows.
  function preflightWhy(pf) {
    if (!pf || pf.fitsInRam || pf.peakRamMb == null) return "";
    return `This run needs about ${mbText(pf.peakRamMb)} but the engine allows ${pf.ramLimitMb == null ? "less" : mbText(pf.ramLimitMb)} ` +
      `(three quarters of your ${pf.ramAvailableMb == null ? "free" : mbText(pf.ramAvailableMb)} RAM, at most 16 GB). ` +
      `Lower batch, context, embed dim or blocks, or tick "I know, try anyway".`;
  }

  const api = { esc, card, catalogMeta, evalRow, modelInfo, ggufInfo, exportInfo, preflightView, preflightWhy };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumEscape = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
