// Ferrum SLM Studio — vanilla JS frontend (no frameworks).
"use strict";

// ── Tauri bindings (gracefully degrade if opened outside Tauri) ─────────────
const T = window.__TAURI__;
const hasTauri = !!(T && T.core && typeof T.core.invoke === "function");
const invoke = hasTauri ? T.core.invoke : async () => { throw new Error("Not running inside the Tauri runtime"); };
const listen = hasTauri ? T.event.listen : async () => () => {};
const dialog = hasTauri && T.dialog ? T.dialog : null;

const { esc, card, catalogMeta, evalRow, modelInfo, ggufInfo, exportInfo, preflightView, preflightWhy } = window.FerrumEscape;
const $ = (id) => document.getElementById(id);
let live = null; // tooltips + live estimates (live.js), created in init()
const bySel = (s) => Array.from(document.querySelectorAll(s));

// ── Toasts & errors ─────────────────────────────────────────────────────────
function toast(msg, kind = "info") {
  const el = document.createElement("div");
  el.className = "toast " + (kind === "error" ? "err" : kind === "ok" ? "ok" : "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 5000);
}
function setErr(id, msg) { const e = $(id); if (e) e.textContent = msg || ""; }
function clearErr(id) { setErr(id, ""); }

// ── Number/validation helpers ───────────────────────────────────────────────
function num(id) { return parseFloat($(id).value); }
function int(id) { return parseInt($(id).value, 10); }
function reqInt(id, name, min = 1) {
  const v = int(id);
  if (!Number.isFinite(v) || v < min) throw new Error(`${name} must be an integer ≥ ${min}`);
  return v;
}
function reqNum(id, name, min = 0, allowEq = false) {
  const v = num(id);
  if (!Number.isFinite(v) || (allowEq ? v < min : v <= min)) {
    throw new Error(`${name} must be a number ${allowEq ? "≥" : ">"} ${min}`);
  }
  return v;
}
// The sampler fields shared by the Generate and GGUF tabs (ids differ by prefix).
function samplerParams(prefix) {
  const topP = reqNum(prefix + "TopP", "Top-p", 0, true);
  if (topP > 1) throw new Error("Top-p must be a number between 0 and 1");
  return {
    temp: reqNum(prefix + "Temp", "Temperature", 0, true),
    topK: reqInt(prefix + "TopK", "Top-k", 0),
    topP,
    repPenalty: reqNum(prefix + "Rep", "Repetition penalty", 0),
  };
}
function reqStr(id, name) {
  const v = $(id).value.trim();
  if (!v) throw new Error(`${name} is required`);
  return v;
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
bySel(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    bySel(".tab").forEach((b) => b.classList.remove("active"));
    bySel(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("panel-" + btn.dataset.tab).classList.add("active");
  });
});

// ── File dialogs (Browse buttons) ────────────────────────────────────────────
async function pickOpen(targetId) {
  if (!dialog) { toast("File dialog unavailable; type the path manually.", "info"); return; }
  try {
    const file = await dialog.open({ multiple: false, directory: false });
    if (file) $(targetId).value = typeof file === "string" ? file : file.path || file;
  } catch (e) { toast("dialog error: " + e, "error"); }
}
async function pickSave(targetId) {
  if (!dialog) { toast("File dialog unavailable; type the path manually.", "info"); return; }
  try {
    const file = await dialog.save({});
    if (file) $(targetId).value = typeof file === "string" ? file : file.path || file;
  } catch (e) { toast("dialog error: " + e, "error"); }
}
async function pickDir(targetId) {
  if (!dialog) { toast("File dialog unavailable; type the path manually.", "info"); return; }
  try {
    const dir = await dialog.open({ multiple: false, directory: true });
    if (dir) $(targetId).value = typeof dir === "string" ? dir : dir.path || dir;
  } catch (e) { toast("dialog error: " + e, "error"); }
}
bySel(".browse").forEach((b) => b.addEventListener("click", () => pickOpen(b.dataset.target)));
bySel(".browse-save").forEach((b) => b.addEventListener("click", () => pickSave(b.dataset.target)));
bySel(".browse-dir").forEach((b) => b.addEventListener("click", () => pickDir(b.dataset.target)));
// "Add file/folder": append the picked paths as new lines of a list (the Prepare inputs).
async function pickAdd(targetId, directory) {
  if (!dialog) { toast("File dialog unavailable; type the path manually.", "info"); return; }
  try {
    const picked = await dialog.open({ multiple: true, directory });
    const paths = (Array.isArray(picked) ? picked : picked ? [picked] : []).map((p) => (typeof p === "string" ? p : p.path || p));
    if (!paths.length) return;
    const box = $(targetId);
    box.value = (box.value.trim() ? box.value.replace(/\n*$/, "\n") : "") + paths.join("\n");
  } catch (e) { toast("dialog error: " + e, "error"); }
}
bySel(".browse-add").forEach((b) => b.addEventListener("click", () => pickAdd(b.dataset.target, false)));
bySel(".browse-add-dir").forEach((b) => b.addEventListener("click", () => pickAdd(b.dataset.target, true)));

// ── Job control: a Stop button per long job, shown while it runs ─────────────
const stopAsked = {}; // kind -> the user pressed Stop for the job running now
function wireStop(btnId, kind) {
  $(btnId).addEventListener("click", async () => {
    stopAsked[kind] = true;
    $(btnId).disabled = true; // asked once; the job ends at its next check
    try { await invoke("cancel_job", { kind }); } catch (e) { toast("Stop failed: " + e, "error"); }
  });
}
function jobRunning(btnId, kind, on) {
  stopAsked[kind] = false;
  $(btnId).classList.toggle("hidden", !on);
  $(btnId).disabled = false;
}
wireStop("trCancel", "train");
wireStop("genCancel", "generate");
wireStop("ggCancel", "run");
wireStop("ftCancel", "finetune");
wireStop("exCancel", "export");
wireStop("dpCancel", "prepare");
wireStop("rxCancel", "runtime_export");
wireStop("tkCancel", "tokenize");

// ── Terminal (bottom dock) ───────────────────────────────────────────────────
const termOut = $("termOut");
function termLine(text, cls = "") {
  const div = document.createElement("div");
  div.className = "line " + cls;
  div.textContent = text;
  termOut.appendChild(div);
  // Keep last ~2000 lines.
  while (termOut.childElementCount > 2000) termOut.removeChild(termOut.firstChild);
  termOut.scrollTop = termOut.scrollHeight;
}
$("termClear").addEventListener("click", () => { termOut.textContent = ""; });

// ── Datasets ─────────────────────────────────────────────────────────────────
let cleanedCorpus = "";

$("dsDownload").addEventListener("click", async () => {
  clearErr("errDsSource");
  try {
    const url = reqStr("dsUrl", "Source URL");
    const maxBytes = Math.max(1, int("dsMaxMb") || 8) * 1024 * 1024;
    $("dsDownload").disabled = true; toast("Downloading…");
    const text = await invoke("download_text", { url, maxBytes });
    $("dsRaw").value = text;
    toast(`Downloaded ${text.length.toLocaleString()} chars`, "ok");
  } catch (e) { setErr("errDsSource", String(e)); }
  finally { $("dsDownload").disabled = false; }
});

$("dsLoadFile").addEventListener("click", async () => {
  clearErr("errDsSource");
  try {
    const path = reqStr("dsFile", "File path");
    const text = await invoke("read_text_file", { path, maxBytes: null });
    $("dsRaw").value = text;
    toast(`Loaded ${text.length.toLocaleString()} chars`, "ok");
  } catch (e) { setErr("errDsSource", String(e)); }
});

// ── Dataset catalog (HuggingFace / Kaggle) ───────────────────────────────────
function destDir() {
  const d = $("dsDestDir").value.trim();
  if (!d) throw new Error("Choose a download folder first");
  return d;
}

// After a dataset lands on disk, point the local-file loader at it so it flows
// straight into the clean pipeline.
function afterDownload(res) {
  const mb = (res.bytes / (1024 * 1024)).toFixed(2);
  $("dsFile").value = res.path;
  toast(`Downloaded ${mb} MB from ${res.source} → ${res.path}. Click “Load file”.`, "ok");
}

function renderCatalog(list) {
  const box = $("dsCatalog");
  box.textContent = "";
  if (!list.length) { box.textContent = "No datasets listed."; return; }
  for (const d of list) {
    const card = document.createElement("div");
    card.className = "catalog-item";
    const meta = document.createElement("div");
    meta.innerHTML = catalogMeta(d);
    const btn = document.createElement("button");
    btn.textContent = "Download";
    btn.dataset.help = "dsCatalogDownload";
    if (live) live.attach(btn);
    btn.addEventListener("click", async () => {
      clearErr("errDsCatalog");
      try {
        btn.disabled = true; toast(`Downloading ${d.name}…`);
        const res = await invoke("download_dataset", { id: d.id, destDir: destDir() });
        afterDownload(res);
      } catch (e) { setErr("errDsCatalog", String(e)); }
      finally { btn.disabled = false; }
    });
    card.appendChild(meta);
    card.appendChild(btn);
    box.appendChild(card);
  }
}

async function loadCatalog() {
  try {
    renderCatalog(await invoke("list_datasets"));
  } catch (e) { setErr("errDsCatalog", String(e)); }
}

$("dsCatalogLoad").addEventListener("click", loadCatalog);

$("dsHfDownload").addEventListener("click", async () => {
  clearErr("errDsCatalog");
  try {
    const repo = reqStr("dsHfRepo", "HuggingFace repo");
    const file = reqStr("dsHfFile", "File");
    const hfToken = $("dsHfToken").value.trim() || null;
    $("dsHfDownload").disabled = true; toast("Downloading from HuggingFace…");
    const res = await invoke("download_hf_file", { repo, file, revision: null, destDir: destDir(), hfToken });
    afterDownload(res);
  } catch (e) { setErr("errDsCatalog", String(e)); }
  finally { $("dsHfDownload").disabled = false; }
});

$("dsKgDownload").addEventListener("click", async () => {
  clearErr("errDsCatalog");
  try {
    const ownerSlug = reqStr("dsKgRepo", "Kaggle owner/slug");
    const file = reqStr("dsKgFile", "File");
    $("dsKgDownload").disabled = true; toast("Downloading from Kaggle…");
    const res = await invoke("download_kaggle_file", { ownerSlug, file, destDir: destDir() });
    afterDownload(res);
  } catch (e) { setErr("errDsCatalog", String(e)); }
  finally { $("dsKgDownload").disabled = false; }
});

// Populate the catalog on startup (best-effort).
loadCatalog();

$("dsClean").addEventListener("click", async () => {
  clearErr("errDsClean");
  try {
    const raw = $("dsRaw").value;
    if (!raw.trim()) throw new Error("No raw text — download, load, or paste first");
    const opts = {
      stripGutenberg: $("optGutenberg").checked,
      lowercase: $("optLower").checked,
      collapseWhitespace: $("optCollapse").checked,
      normalizePunctuation: $("optPunct").checked,
      stripControlChars: $("optControl").checked,
      maxChars: Math.max(0, int("optMaxChars") || 0),
    };
    const r = await invoke("clean_text", { raw, opts });
    cleanedCorpus = r.cleaned;
    $("dsPreview").value = r.preview;
    $("dsStats").innerHTML =
      `chars <b>${esc(r.chars.toLocaleString())}</b> · words <b>${esc(r.words.toLocaleString())}</b> · ` +
      `lines <b>${esc(r.lines.toLocaleString())}</b> · bytes <b>${esc(r.bytes.toLocaleString())}</b> · ` +
      `char-vocab <b>${esc(r.uniqueChars)}</b>`;
    toast("Cleaned. char-level vocab = " + r.uniqueChars, "ok");
  } catch (e) { setErr("errDsClean", String(e)); }
});

$("dsSave").addEventListener("click", async () => {
  clearErr("errDsSave");
  try {
    if (!cleanedCorpus) throw new Error("Clean a corpus first");
    const path = reqStr("dsOut", "Output path");
    const n = await invoke("save_corpus", { text: cleanedCorpus, path });
    toast(`Saved ${n.toLocaleString()} bytes → ${path}`, "ok");
    // Convenience: prefill the Train corpus field.
    if (!$("trCorpus").value) $("trCorpus").value = path;
  } catch (e) { setErr("errDsSave", String(e)); }
});

// ── Prepare (clean + split; the tab lives in prepare_tab.js) ─────────────────────
const prepareTab = window.FerrumPrepareTab.create({ $, invoke, toast, setErr, clearErr, jobRunning, stopAsked });

// ── Train ─────────────────────────────────────────────────────────────────────

let lossHistory = [];
function drawChart() {
  const c = $("trChart"), ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1e293b"; ctx.fillRect(0, 0, w, h);
  if (lossHistory.length < 2) return;
  const max = Math.max(...lossHistory), min = Math.min(...lossHistory);
  const pad = 8, span = (max - min) || 1;
  ctx.strokeStyle = "#ea580c"; ctx.lineWidth = 2; ctx.beginPath();
  lossHistory.forEach((v, i) => {
    const x = pad + (i / (lossHistory.length - 1)) * (w - 2 * pad);
    const y = pad + (1 - (v - min) / span) * (h - 2 * pad);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#94a3b8"; ctx.font = "11px monospace";
  ctx.fillText("loss " + max.toFixed(4), pad, 14);
  ctx.fillText(min.toFixed(4), pad, h - 4);
}

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
    const pf = await invoke("preflight_train", { params: window.FerrumLive.PARAMS.trStart(document) });
    if (n !== preflightSeq) return; // a newer request superseded this one
    preflight = pf;
    $("trPreflight").innerHTML = preflightView(pf);
  } catch (e) {
    if (n !== preflightSeq) return;
    preflight = null;
    $("trPreflight").textContent = "Pre-flight unavailable: " + String(e);
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
$("trForce").addEventListener("change", updateStartGate);
// "Tokenize corpus" (the block in the Train form; its code is in tokenize_tab.js): a finished token
// file becomes the Train corpus, and the pre-flight is asked again.
const tokenizeTab = window.FerrumTokenizeTab.create({
  $, invoke, toast, setErr, clearErr, jobRunning, stopAsked,
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
    };
    if (params.resume && !params.checkpointPath)
      throw new Error("Resume needs a checkpoint file");
    // Client-side guards mirroring the backend, for instant feedback.
    if (params.embedDim % params.numHeads !== 0)
      throw new Error(`Embed dim (${params.embedDim}) must be divisible by heads (${params.numHeads})`);
    if (params.vocabSize !== 0 && params.vocabSize < 256)
      throw new Error("Vocab must be 0 (char-level) or ≥ 256 (BPE)");
  } catch (e) { setErr("errTrain", String(e)); return; }

  lossHistory = []; drawChart();
  $("trBar").style.width = "0%";
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
    setErr("errTrain", String(e));
    toast("Training failed: " + e, "error");
  } finally { trRunning = false; updateStartGate(); jobRunning("trCancel", "train", false); }
});

// ── Generate ──────────────────────────────────────────────────────────────────
let generating = false;
$("genStart").addEventListener("click", async () => {
  clearErr("errGen");
  let params;
  try {
    const seedNum = $("genSeedNum").value.trim();
    params = {
      modelPath: reqStr("genModel", "Model"),
      seedText: $("genSeed").value,
      numChars: reqInt("genChars", "Chars"),
      ...samplerParams("gen"),
      maxTokens: reqInt("genMaxTokens", "Max tokens", 0),
      stop: $("genStop").value.split("\n").map((l) => l.replace(/\r$/, "")).filter((l) => l !== ""),
      genSeed: seedNum === "" ? null : parseInt(seedNum, 10),
      stream: $("genStream").checked,
      verbose: $("genVerbose").checked,
    };
  } catch (e) { setErr("errGen", String(e)); return; }

  $("genOut").textContent = "";
  $("genStatus").textContent = "generating…";
  $("genStart").disabled = true;
  generating = params.stream;
  jobRunning("genCancel", "generate", true);
  try {
    const out = await invoke("generate_slm", { params });
    const stopped = stopAsked.generate; // Stop was pressed (the text so far is returned)
    // Stop accepting late gen-fragment events *before* reconciling, then set the
    // display from the authoritative return value so a dropped/late stream tail
    // can never leave the output truncated (G1).
    generating = false;
    if (params.stream) {
      $("genOut").textContent = streamContinuation(out, params.seedText);
    } else {
      $("genOut").textContent = out;
    }
    $("genStatus").textContent = `${stopped ? "stopped" : "done"} (${out.length} chars)`;
  } catch (e) {
    $("genStatus").textContent = "error";
    setErr("errGen", String(e));
    toast("Generation failed: " + e, "error");
  } finally { generating = false; $("genStart").disabled = false; jobRunning("genCancel", "generate", false); }
});

// ── Evaluate ──────────────────────────────────────────────────────────────────
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
      const tr = document.createElement("tr");
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
let lastEval = [];
$("evSaveJson").addEventListener("click", async () => {
  clearErr("errEvalJson");
  try {
    if (!lastEval.length) throw new Error("Run an evaluation first");
    const path = reqStr("evJsonPath", "JSON output path");
    const doc = lastEval.length === 1
      ? lastEval[0].report
      : Object.fromEntries(lastEval.map((r) => [r.label.replace("-", "_"), r.report]));
    const n = await invoke("save_corpus", { text: JSON.stringify(doc, null, 2) + "\n", path });
    toast(`Saved ${n.toLocaleString()} bytes → ${path}`, "ok");
  } catch (e) { setErr("errEvalJson", String(e)); }
});

// ── Models ────────────────────────────────────────────────────────────────────
$("miInspect").addEventListener("click", async () => {
  clearErr("errModel");
  let path;
  try { path = reqStr("miPath", "Model file"); } catch (e) { setErr("errModel", String(e)); return; }
  try {
    const i = await invoke("model_info", { path });
    $("miInfo").innerHTML = modelInfo(i);
    toast("Model loaded", "ok");
  } catch (e) { setErr("errModel", String(e)); toast("Inspect failed: " + e, "error"); }
});

// ── GGUF (import & run Llama/Qwen) ──────────────────────────────────────────────

// Compare a GGUF's parameter count against the cached capability bounds.
// Returns a list of human-readable descriptions for every bound it exceeds.
function checkGgufBudget(paramCount) {
  if (!capBounds || !Number.isFinite(paramCount) || paramCount <= 0) return [];
  const checks = [
    ["loading at int4 (fits in RAM)", capBounds.loadInt4],
    ["inference at int4 (≥ 3 tok/s)", capBounds.inferInt4],
    ["inference at int8 (≥ 3 tok/s)", capBounds.inferInt8],
    ["inference at f32 (≥ 3 tok/s)", capBounds.inferF32],
    ["training, compute-optimal (< 24 h)", capBounds.trainChinchilla],
    ["training, fixed corpus (< 24 h)", capBounds.trainFixed1b],
    ["evaluation pass (< 24 h)", capBounds.testEval],
  ];
  return checks
    .filter(([, bound]) => Number.isFinite(bound) && bound > 0 && paramCount > bound)
    .map(([label, bound]) => `${label}: limit ${fmtParams(bound)}`);
}

// Show the warning dialog; resolves true if the user chooses Proceed, and
// false on Cancel or any dismissal (Esc/backdrop close the native <dialog>,
// which fires its "close" event — the single resolution point).
function confirmGgufWarning(paramCount, crossed) {
  return new Promise((resolve) => {
    // Keyed on the load-check label in checkGgufBudget: crossing that line
    // means "may not fit in RAM", which is a different risk than "slow".
    const footer = crossed.some((c) => c.includes("fits in RAM"))
      ? "You can still proceed — but this exceeds the fits-in-RAM ceiling: expect heavy swapping or a failed load."
      : "You can still proceed — expect slower than the targets above.";
    $("ggWarnBody").innerHTML =
      `<p>This model has <strong>${esc(fmtParams(paramCount))}</strong> parameters, which exceeds
       the estimated limits of this machine:</p><ul>` +
      crossed.map((c) => `<li>${esc(c)}</li>`).join("") +
      `</ul><p class="muted">${footer}</p>`;
    const dlg = $("ggWarnDialog");
    let proceed = false; // default for Esc / backdrop dismissal
    const onProceed = () => { proceed = true; dlg.close(); };
    const onCancel = () => { proceed = false; dlg.close(); };
    const onClose = () => {
      $("ggWarnProceed").removeEventListener("click", onProceed);
      $("ggWarnCancel").removeEventListener("click", onCancel);
      resolve(proceed);
    };
    $("ggWarnProceed").addEventListener("click", onProceed);
    $("ggWarnCancel").addEventListener("click", onCancel);
    dlg.addEventListener("close", onClose, { once: true });
    dlg.showModal();
  });
}

$("ggInspect").addEventListener("click", async () => {
  clearErr("errGguf");
  let path;
  try { path = reqStr("ggPath", "GGUF file"); } catch (e) { setErr("errGguf", String(e)); return; }
  $("ggStatus").textContent = "inspecting…";
  try {
    const i = await invoke("gguf_info", { path });
    $("ggInfo").innerHTML = ggufInfo(i);
    $("ggStatus").textContent = "";
    toast(i.runnable ? "GGUF inspected" : "Inspected — architecture not runnable", i.runnable ? "ok" : "error");
    const crossedI = checkGgufBudget(i.paramCount);
    if (crossedI.length) {
      await confirmGgufWarning(i.paramCount, crossedI); // advisory on inspect; result ignored
    } else if (!capBounds) {
      toast("Tip: run the Capable check to flag oversized models.", "info");
    }
  } catch (e) { $("ggStatus").textContent = "error"; setErr("errGguf", String(e)); toast("Inspect failed: " + e, "error"); }
});

$("ggRun").addEventListener("click", async () => {
  clearErr("errGguf");
  let params;
  try {
    const seed = $("ggSeed").value.trim();
    const ids = $("ggIds").value.trim();
    const resume = $("ggResume").value.trim();
    params = {
      modelPath: reqStr("ggPath", "GGUF file"),
      prompt: $("ggPrompt").value,
      quant: $("ggQuant").value,
      maxNew: reqInt("ggMax", "Max new tokens"),
      ...samplerParams("gg"),
      genSeed: seed === "" ? null : parseInt(seed, 10),
      ids: ids === "" ? null : ids,
      force: $("ggForce").checked,
      resume: resume === "" ? null : resume,
      raw: !$("ggChatFmt").checked,
    };
    if (!params.prompt.trim() && !params.ids) throw new Error("Enter a prompt (or token IDs)");
  } catch (e) { setErr("errGguf", String(e)); return; }

  // Budget gate: if the model exceeds this machine's estimated limits, confirm.
  if (capBounds) {
    try {
      const info = await invoke("gguf_info", { path: params.modelPath });
      const crossed = checkGgufBudget(info.paramCount);
      if (crossed.length) {
        const ok = await confirmGgufWarning(info.paramCount, crossed);
        if (!ok) { $("ggStatus").textContent = "cancelled"; return; }
      }
    } catch (_) { /* inspection failed; fall through to run, which will error clearly */ }
  }

  $("ggOut").textContent = "";
  $("ggStatus").textContent = "loading & generating… (CPU: this can take a while)";
  $("ggRun").disabled = true;
  jobRunning("ggCancel", "run", true);
  try {
    const r = await invoke("run_gguf", { params });
    $("ggOut").textContent = r.text;
    $("ggStatus").textContent =
      `${r.stopped ? "stopped" : "done"} — ${r.generated} tokens in ${r.seconds.toFixed(1)}s (${r.tokensPerSec.toFixed(1)} tok/s, prompt ${r.promptTokens} tok)`;
  } catch (e) {
    if (stopAsked.run && /cancelled/.test(String(e))) {
      $("ggStatus").textContent = "stopped before any text was produced";
    } else {
      $("ggStatus").textContent = "error";
      setErr("errGguf", String(e));
      toast("Run failed: " + e, "error");
    }
  } finally { $("ggRun").disabled = false; jobRunning("ggCancel", "run", false); }
});

// ── Fine-tune (GGUF) ────────────────────────────────────────────────────────────
let ftLossHistory = [];
function drawFtChart() {
  const c = $("ftChart"), ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1e293b"; ctx.fillRect(0, 0, w, h);
  if (ftLossHistory.length < 2) return;
  const max = Math.max(...ftLossHistory), min = Math.min(...ftLossHistory);
  const pad = 8, span = (max - min) || 1;
  ctx.strokeStyle = "#ea580c"; ctx.lineWidth = 2; ctx.beginPath();
  ftLossHistory.forEach((v, i) => {
    const x = pad + (i / (ftLossHistory.length - 1)) * (w - 2 * pad);
    const y = pad + (1 - (v - min) / span) * (h - 2 * pad);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#94a3b8"; ctx.font = "11px monospace";
  ctx.fillText("loss " + max.toFixed(4), pad, 14);
  ctx.fillText(min.toFixed(4), pad, h - 4);
}

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
      verbose: false,
    };
  } catch (e) { setErr("errFt", String(e)); return; }

  ftLossHistory = []; drawFtChart();
  $("ftBar").style.width = "0%";
  $("ftResult").textContent = "";
  $("ftStatus").textContent = "fine-tuning… (CPU: this can take a while)";
  $("ftRun").disabled = true;
  jobRunning("ftCancel", "finetune", true);
  try {
    const r = await invoke("finetune_gguf", { params });
    $("ftStatus").textContent = r.stopped ? `stopped at step ${r.steps}` : "done";
    $("ftResult").textContent = (r.stopped ? `Stopped at step ${r.steps}; the checkpoint so far was saved and can be resumed.\n` : "") +
      `Saved ${r.bytes.toLocaleString()} bytes → ${r.outPath}\n` +
      `epochs=${r.epochs}  first loss=${r.firstLoss.toFixed(6)}  final loss=${r.finalLoss.toFixed(6)}\n` +
      `tokens=${r.tokens.toLocaleString()}  steps=${r.steps}  time=${r.seconds.toFixed(2)}s\n` +
      `Run it: GGUF tab → set "Fine-tune checkpoint" to ${r.outPath}`;
    toast(r.stopped ? `Fine-tune stopped at step ${r.steps}; checkpoint saved` : "Fine-tune complete", "ok");
    if (!$("ggPath").value) $("ggPath").value = params.modelPath;
    if (!$("ggResume").value) $("ggResume").value = r.outPath;
  } catch (e) {
    $("ftStatus").textContent = "error";
    setErr("errFt", String(e));
    toast("Fine-tune failed: " + e, "error");
  } finally { $("ftRun").disabled = false; jobRunning("ftCancel", "finetune", false); }
});

// ── Export (write a GGUF back out) ──────────────────────────────────────────────
$("exInspect").addEventListener("click", async () => {
  clearErr("errExport");
  let path;
  try { path = reqStr("exPath", "Source GGUF"); } catch (e) { setErr("errExport", String(e)); return; }
  $("exStatus").textContent = "inspecting…";
  try {
    const i = await invoke("gguf_info", { path });
    $("exInfo").innerHTML = exportInfo(i);
    $("exStatus").textContent = "";
    toast(i.runnable ? "GGUF inspected" : "Inspected — architecture not exportable", i.runnable ? "ok" : "error");
    const crossed = checkGgufBudget(i.paramCount);
    if (crossed.length) await confirmGgufWarning(i.paramCount, crossed); // advisory; result ignored
  } catch (e) { $("exStatus").textContent = "error"; setErr("errExport", String(e)); toast("Inspect failed: " + e, "error"); }
});

$("exExport").addEventListener("click", async () => {
  clearErr("errExport");
  let params;
  try {
    const resume = $("exResume").value.trim();
    params = {
      modelPath: reqStr("exPath", "Source GGUF"),
      outPath: reqStr("exOut", "Output file"),
      quant: $("exQuant").value,
      resume: resume === "" ? null : resume,
      force: $("exForce").checked,
      verbose: false,
    };
  } catch (e) { setErr("errExport", String(e)); return; }

  // Budget gate (advisory + confirm), mirroring the GGUF Run button.
  if (capBounds) {
    try {
      const info = await invoke("gguf_info", { path: params.modelPath });
      const crossed = checkGgufBudget(info.paramCount);
      if (crossed.length) {
        const ok = await confirmGgufWarning(info.paramCount, crossed);
        if (!ok) { $("exStatus").textContent = "cancelled"; return; }
      }
    } catch (_) { /* inspection failed; the export itself will error clearly */ }
  }

  $("exResult").textContent = "";
  $("exStatus").textContent = "starting… (CPU: a large model can take minutes)";
  $("exExport").disabled = true;
  jobRunning("exCancel", "export", true);
  try {
    const r = await invoke("export_gguf", { params });
    $("exStatus").textContent = "done";
    $("exResult").textContent =
      `Wrote ${r.bytes.toLocaleString()} bytes → ${r.outPath} in ${r.seconds.toFixed(1)}s\n` +
      `tensors: ${r.tensorSummary}\n` +
      `The file runs in llama.cpp / ollama / LM Studio as-is.`;
    toast("Export complete", "ok");
  } catch (e) {
    if (stopAsked.export && /cancelled/.test(String(e))) {
      $("exStatus").textContent = "cancelled; nothing was written";
    } else {
      $("exStatus").textContent = "error";
      setErr("errExport", String(e));
      toast("Export failed: " + e, "error");
    }
  } finally { $("exExport").disabled = false; jobRunning("exCancel", "export", false); }
});

// Run a Ferrum model in Ollama / LM Studio (the fieldset at the end of the Export tab).
window.FerrumRuntimeTab.create({ $, invoke, toast, setErr, clearErr, jobRunning, stopAsked });

// ── System monitor ──────────────────────────────────────────────────────────
let sysTimer = null;
function fmtBytes(b) {
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; let n = b;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(1) + " " + u[i];
}
async function pollStats() {
  if (!hasTauri) return;
  try {
    const s = await invoke("system_stats");
    $("miniCpu").textContent = s.cpuTotal.toFixed(0) + "%";
    $("miniMem").textContent = s.memPercent.toFixed(0) + "%";
    $("miniThreads").textContent = s.ferrumThreads;
    $("sysCpu").textContent = s.cpuTotal.toFixed(1) + "%";
    $("sysCpuBar").style.width = Math.min(100, s.cpuTotal) + "%";
    $("sysMem").textContent = s.memPercent.toFixed(1) + "%";
    $("sysMemBar").style.width = Math.min(100, s.memPercent) + "%";
    $("sysMemAbs").textContent = `${fmtBytes(s.memUsed)} / ${fmtBytes(s.memTotal)}`;
    $("sysThreads").textContent = s.ferrumThreads;
    // DOM construction: an inline style="" attribute in markup would need CSP 'unsafe-inline'.
    $("sysCores").replaceChildren(...s.perCore.map((v, i) => {
      const core = document.createElement("div");
      core.className = "core";
      core.append(`core ${i} — ${v.toFixed(0)}%`);
      const bar = document.createElement("div");
      bar.className = "cbar";
      const fill = document.createElement("div");
      fill.style.width = Math.min(100, v) + "%";
      bar.appendChild(fill);
      core.appendChild(bar);
      return core;
    }));
  } catch (e) { /* transient; ignore */ }
}
function restartMonitor() {
  if (sysTimer) clearInterval(sysTimer);
  if (!$("sysOn").checked) return;
  const sec = Math.max(0.5, num("sysInterval") || 1.5);
  pollStats();
  sysTimer = setInterval(pollStats, sec * 1000);
}
$("sysOn").addEventListener("change", restartMonitor);
$("sysInterval").addEventListener("change", restartMonitor);

// ── Capable (machine capability) ─────────────────────────────────────────────
let capBounds = null; // {inferInt4/8/F32, trainChinchilla, trainFixed1b, testEval, loadInt4, finetuneHi}

function fmtParams(n) {
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + " K";
  return n.toFixed(0);
}

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
  $("capCheck").disabled = true;
  try {
    const r = await invoke("capability_report");
    renderCapReport(r);
    $("capStatus").textContent = "";
    $("capDialog").showModal();
    toast("Capability check complete", "ok");
  } catch (e) {
    $("capStatus").textContent = "error";
    toast("Capability check failed: " + e, "error");
  } finally { $("capCheck").disabled = false; }
});

// ── Backend event wiring ──────────────────────────────────────────────────────
async function wireEvents() {
  await listen("engine-log", (e) => termLine(e.payload, "log"));
  await listen("train-progress", (e) => {
    const p = e.payload;
    // total is null under a token budget (epoch count unknown up front).
    if (p.total) $("trBar").style.width = Math.round((p.epoch / p.total) * 100) + "%";
    $("trStatus").textContent = `epoch ${p.epoch}${p.total ? "/" + p.total : ""} — loss ${p.loss.toFixed(6)}`;
    lossHistory.push(p.loss);
    if (lossHistory.length > 1000) lossHistory.shift();
    drawChart();
  });
  await listen("finetune-progress", (e) => {
    const p = e.payload;
    const pct = Math.round((p.epoch / p.total) * 100);
    $("ftBar").style.width = pct + "%";
    $("ftStatus").textContent = `epoch ${p.epoch}/${p.total} — loss ${p.loss.toFixed(4)} (ppl ${p.ppl.toFixed(2)})`;
    ftLossHistory.push(p.loss);
    if (ftLossHistory.length > 1000) ftLossHistory.shift();
    drawFtChart();
  });
  await listen("prepare-progress", (e) => prepareTab.onProgress(e.payload));
  await listen("tokenize-progress", (e) => tokenizeTab.onProgress(e.payload));
  await listen("export-progress", (e) => { $("exStatus").textContent = e.payload; });
  await listen("gen-fragment", (e) => { if (generating) $("genOut").textContent += e.payload; });
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  // Tooltips work everywhere; the estimates need the backend and report that they do not have it.
  live = window.FerrumLive.create({
    doc: document, invoke, help: window.FerrumHelp, tooltip: window.FerrumTooltip, view: window.FerrumEstimateView,
    tooltipEnv: { document, window },
  });
  live.start();
  if (!hasTauri) {
    $("webBanner").classList.remove("hidden");
    $("connDot").classList.remove("ok");
    $("connDot").title = "Backend not connected (running outside Tauri)";
    termLine("Ferrum SLM Studio — backend unavailable (open via `cargo tauri dev`).", "sys");
    return;
  }
  $("connDot").classList.add("ok");
  $("connDot").title = "Backend connected";
  await wireEvents();
  restartMonitor();
  runPreflight();
  termLine("Ferrum SLM Studio ready. Verbose engine logs and shell output appear here.", "sys");
})();
