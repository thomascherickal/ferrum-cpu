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

  const api = { esc, card, catalogMeta, evalRow, modelInfo, ggufInfo, exportInfo };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumEscape = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
