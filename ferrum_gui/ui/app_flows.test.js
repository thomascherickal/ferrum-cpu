// Functional tests of app.js: the real script, the real index.html (stub DOM),
// a fake Tauri backend. Each test is a user flow: fill the form, click the
// button, check the command that was sent and what the page shows.
// What this CANNOT see (no display here): layout, the real webview, the CSP in
// a running window. Run with: node ui/app_flows.test.js
"use strict";
const assert = require("node:assert");
const { loadApp, settle } = require("./testapp.js");

const HW = { cpuModel: "T", ferrumThreads: 4, logicalCores: 4, physicalCores: 4, ramTotalMb: 8000, ramAvailableMb: 4000, gpus: [] };
const EVIL = "<img src=x onerror=alert(1)>";

// A backend whose answers a test can override per command.
function makeBackend(overrides = {}) {
  const calls = [];
  const backend = async (cmd, args) => {
    calls.push({ cmd, args: JSON.parse(JSON.stringify(args === undefined ? null : args)) });
    if (overrides[cmd]) return overrides[cmd](args);
    switch (cmd) {
      case "list_datasets": return [];
      case "hardware_info": return HW;
      case "estimate_job": return { kind: args.kind, seconds: 5, lowSeconds: 3, highSeconds: 8, threadsUsed: 4, notes: [] };
      case "system_stats": return { cpuTotal: 12.5, memPercent: 40, ferrumThreads: 4, memUsed: 3e9, memTotal: 8e9, perCore: [10, 20] };
      default: throw new Error("no fake for " + cmd);
    }
  };
  return { backend, calls, of: (cmd) => calls.filter((c) => c.cmd === cmd) };
}

async function start(overrides, opts = {}) {
  const b = makeBackend(overrides);
  const app = loadApp({ backend: b.backend, ...opts });
  await settle(8);
  const $ = (id) => app.doc.getElementById(id);
  const set = (id, v) => { $(id).value = v; };
  const click = async (id) => { $(id).dispatch("click"); await settle(6); };
  return { ...app, ...b, $, set, click };
}

// The exact `params` objects each flow sent; pinned to ui/command_params.fixture.json,
// which src/commands.rs-side tests (ipc_tests.rs) deserialize into the Rust types.
const captured = {};

const TRAINED = { modelPath: "/m/out.gguf", method: "transformer", seconds: 1.5, finalLoss: 2.25, inputDim: 8, outputDim: 300, tokenizer: "byte-level BPE", layers: 5, bytes: 999 };

(async () => {
  // ── Tabs ────────────────────────────────────────────────────────────────
  {
    const a = await start();
    await a.click("tab-train");
    assert.ok(a.$("panel-train").classList.contains("active") && !a.$("panel-datasets").classList.contains("active"));
    assert.ok(a.$("tab-train").classList.contains("active") && !a.$("tab-datasets").classList.contains("active"));
    await a.click("tab-capable");
    assert.ok(a.$("panel-capable").classList.contains("active") && !a.$("panel-train").classList.contains("active"));
  }

  // ── Train ───────────────────────────────────────────────────────────────
  {
    const a = await start({ train_slm: async () => TRAINED });
    await a.click("tab-train");
    await a.click("trStart");
    assert.strictEqual(a.$("errTrain").textContent, "Error: Corpus is required");
    assert.strictEqual(a.of("train_slm").length, 0, "nothing is sent for an invalid form");

    a.set("trCorpus", "/data/c.txt"); a.set("trModel", "/m/out.gguf");
    a.set("trHeads", "3"); // 32 % 3 != 0
    await a.click("trStart");
    assert.ok(/divisible by heads/.test(a.$("errTrain").textContent), a.$("errTrain").textContent);
    a.set("trHeads", "4"); a.set("trVocab", "100");
    await a.click("trStart");
    assert.ok(/Vocab must be 0/.test(a.$("errTrain").textContent));

    a.set("trVocab", "512"); a.set("trTokens", "5e7"); a.$("trCosine").checked = true; a.$("trVal").value = "0.1";
    await a.click("trStart");
    assert.strictEqual(a.$("errTrain").textContent, "");
    const sent = a.of("train_slm")[0].args.params;
    captured.train_slm = sent;
    assert.deepStrictEqual(sent, {
      method: "transformer", corpusPath: "/data/c.txt", modelPath: "/m/out.gguf", contextLen: 16, embedDim: 32, numHeads: 4,
      numBlocks: 2, hiddenDim: 64, epochs: 100, lr: 0.01, batchSize: 16, vocabSize: 512, seed: 1337, threads: 0, verbose: false,
      windowStride: 0, maxTokens: 50000000, weightDecay: 0, dropout: 0, gradClip: 0, warmupSteps: 0, cosineLr: true,
      weightTying: false, qat: true, tokenizerPath: "", checkpointPath: "", checkpointEvery: 0, resume: false,
      valFraction: 0.1, patience: 0,
    });
    assert.strictEqual(a.$("trStatus").textContent, "done");
    assert.ok(a.$("trResult").textContent.includes("Saved 999 bytes → /m/out.gguf"), a.$("trResult").textContent);
    assert.ok(a.$("trResult").textContent.includes("final loss=2.250000"));
    // The trained model flows into the other tabs.
    assert.deepStrictEqual(["genModel", "evModel", "miPath"].map((i) => a.$(i).value), ["/m/out.gguf", "/m/out.gguf", "/m/out.gguf"]);
    assert.strictEqual(a.$("trStart").disabled, false, "button re-enabled");

    // Progress events drive the bar and status (and tolerate a token budget: total null).
    a.emit("train-progress", { epoch: 2, total: 4, loss: 1.5 });
    assert.strictEqual(a.$("trBar").style.width, "50%");
    assert.strictEqual(a.$("trStatus").textContent, "epoch 2/4 — loss 1.500000");
    a.emit("train-progress", { epoch: 3, total: null, loss: 1.25 });
    assert.strictEqual(a.$("trStatus").textContent, "epoch 3 — loss 1.250000");
    a.emit("engine-log", "hello from the engine");
    assert.ok(a.$("termOut").textContent.includes("hello from the engine"));
  }

  // Backend errors are shown in the form, not thrown; the button comes back.
  {
    const a = await start({ train_slm: async () => { throw "corpus /data/c.txt is empty"; } });
    a.set("trCorpus", "/data/c.txt"); a.set("trModel", "/m/o.gguf");
    await a.click("trStart");
    assert.strictEqual(a.$("errTrain").textContent, "corpus /data/c.txt is empty");
    assert.strictEqual(a.$("trStatus").textContent, "error");
    assert.strictEqual(a.$("trStart").disabled, false);
    assert.ok(a.$("toasts").children.some((t) => t.textContent.startsWith("Training failed")));
  }

  // ── Datasets ────────────────────────────────────────────────────────────
  {
    const a = await start({
      clean_text: async () => ({ cleaned: "hello", preview: "hello", chars: 5, bytes: 5, lines: 1, words: 1, uniqueChars: 4 }),
      save_corpus: async () => 5,
      download_text: async () => "downloaded text",
      read_text_file: async () => "file text",
    });
    await a.click("dsClean");
    assert.ok(a.$("errDsClean").textContent.includes("No raw text"));
    await a.click("dsSave");
    assert.ok(a.$("errDsSave").textContent.includes("Clean a corpus first"));

    a.set("dsRaw", "Hello  there");
    a.$("optLower").checked = true;
    await a.click("dsClean");
    assert.deepStrictEqual(a.of("clean_text")[0].args, {
      raw: "Hello  there",
      opts: { stripGutenberg: true, lowercase: true, collapseWhitespace: true, normalizePunctuation: true, stripControlChars: true, maxChars: 0 },
    });
    captured.clean_text = a.of("clean_text")[0].args;
    assert.strictEqual(a.$("dsPreview").value, "hello");
    assert.ok(a.$("dsStats").innerHTML.includes("chars <b>5</b>"), a.$("dsStats").innerHTML);

    await a.click("dsSave");
    assert.ok(a.$("errDsSave").textContent.includes("Output path is required"));
    a.set("dsOut", "/out/c.txt");
    await a.click("dsSave");
    assert.deepStrictEqual(a.of("save_corpus")[0].args, { text: "hello", path: "/out/c.txt" });
    assert.strictEqual(a.$("trCorpus").value, "/out/c.txt", "Train corpus prefilled");

    await a.click("dsDownload");
    assert.ok(a.$("errDsSource").textContent.includes("Source URL is required"));
    a.set("dsUrl", "https://example.com/a.txt"); a.set("dsMaxMb", "2");
    await a.click("dsDownload");
    assert.deepStrictEqual(a.of("download_text")[0].args, { url: "https://example.com/a.txt", maxBytes: 2 * 1024 * 1024 });
    assert.strictEqual(a.$("dsRaw").value, "downloaded text");
    a.set("dsFile", "/x.txt");
    await a.click("dsLoadFile");
    assert.strictEqual(a.$("dsRaw").value, "file text");
    assert.strictEqual(a.$("dsDownload").disabled, false);
  }

  // Dataset catalog: untrusted names are escaped, downloads fill the file field.
  {
    const a = await start({
      list_datasets: async () => [{ id: "x", name: EVIL, source: "huggingface", approxMb: 5, format: "txt", description: EVIL }],
      download_dataset: async () => ({ path: "/d/x.txt", bytes: 2 * 1024 * 1024, source: "huggingface" }),
      download_hf_file: async () => ({ path: "/d/f.txt", bytes: 1024 * 1024, source: "huggingface" }),
      download_kaggle_file: async () => ({ path: "/d/k.csv", bytes: 1024 * 1024, source: "kaggle" }),
    });
    const item = a.$("dsCatalog").querySelector(".catalog-item");
    assert.ok(item, "catalog rendered");
    const meta = item.children[0].innerHTML;
    assert.ok(!meta.includes("<img") && meta.includes("&lt;img"), meta);
    const btn = item.querySelector("button");
    btn.dispatch("click"); await settle(6);
    assert.ok(a.$("errDsCatalog").textContent.includes("Choose a download folder"));
    a.set("dsDestDir", "/d");
    btn.dispatch("click"); await settle(6);
    assert.deepStrictEqual(a.of("download_dataset")[0].args, { id: "x", destDir: "/d" });
    assert.strictEqual(a.$("dsFile").value, "/d/x.txt");
    // HF / Kaggle forms.
    a.set("dsHfRepo", "o/n"); a.set("dsHfFile", "f.txt"); a.set("dsHfToken", "  ");
    await a.click("dsHfDownload");
    assert.deepStrictEqual(a.of("download_hf_file")[0].args, { repo: "o/n", file: "f.txt", revision: null, destDir: "/d", hfToken: null });
    assert.strictEqual(a.$("dsFile").value, "/d/f.txt");
    a.set("dsKgRepo", "o/d"); a.set("dsKgFile", "k.csv");
    await a.click("dsKgDownload");
    assert.deepStrictEqual(a.of("download_kaggle_file")[0].args, { ownerSlug: "o/d", file: "k.csv", destDir: "/d" });
    assert.strictEqual(a.$("dsFile").value, "/d/k.csv");
  }

  // ── Generate (streaming reconciliation) ────────────────────────────────
  {
    let release;
    const a = await start({ generate_slm: () => new Promise((res) => { release = res; }) });
    await a.click("tab-generate");
    await a.click("genStart");
    assert.ok(a.$("errGen").textContent.includes("Model is required"));
    a.set("genModel", "/m.gguf"); a.set("genSeed", "the q"); a.set("genSeedNum", "7");
    a.$("genStart").dispatch("click"); await settle(3);
    assert.strictEqual(a.$("genStatus").textContent, "generating…");
    assert.strictEqual(a.$("genStart").disabled, true, "button disabled while running");
    a.emit("gen-fragment", "uick "); a.emit("gen-fragment", "brown");
    assert.strictEqual(a.$("genOut").textContent, "uick brown");
    const sent = a.of("generate_slm")[0].args.params;
    captured.generate_slm = sent;
    assert.deepStrictEqual(sent, {
      modelPath: "/m.gguf", seedText: "the q", numChars: 200, temp: 0.8, topK: 0, topP: 1, repPenalty: 1, maxTokens: 0, stop: [],
      genSeed: 7, stream: true, verbose: false,
    });
    release("the quick brown fox"); // the authoritative result has a longer tail than was streamed
    await settle(6);
    assert.strictEqual(a.$("genOut").textContent, "uick brown fox");
    assert.strictEqual(a.$("genStatus").textContent, "done (19 chars)");
    a.emit("gen-fragment", "LATE"); // late fragments after completion are ignored
    assert.strictEqual(a.$("genOut").textContent, "uick brown fox");
    // Blank seed means random (null).
    a.set("genSeedNum", "");
    a.$("genStream").checked = false;
    await a.click("genStart");
    const sent2 = a.of("generate_slm")[1].args.params;
    assert.strictEqual(sent2.genSeed, null);
    assert.strictEqual(sent2.stream, false);

    // G7: greedy decoding and every sampling control reach the backend.
    a.set("genTemp", "0"); a.set("genMaxTokens", "30"); a.set("genTopK", "40"); a.set("genTopP", "0.9"); a.set("genRep", "1.2");
    a.set("genStop", "\\n\r\n\nEND\n");
    await a.click("genStart");
    assert.strictEqual(a.$("errGen").textContent, "");
    assert.deepStrictEqual(a.of("generate_slm")[2].args.params, {
      modelPath: "/m.gguf", seedText: "the q", numChars: 200, temp: 0, topK: 40, topP: 0.9, repPenalty: 1.2, maxTokens: 30,
      stop: ["\\n", "END"], genSeed: null, stream: false, verbose: false,
    });
    // Out-of-range values are refused before any call.
    const n = a.of("generate_slm").length;
    for (const [id, v, msg] of [["genTemp", "-1", /Temperature/], ["genTopP", "1.5", /Top-p/], ["genRep", "0", /Repetition/], ["genTopK", "-2", /Top-k/], ["genMaxTokens", "-1", /Max tokens/]]) {
      const keep = a.$(id).value;
      a.set(id, v);
      await a.click("genStart");
      assert.ok(msg.test(a.$("errGen").textContent), id + ": " + a.$("errGen").textContent);
      a.set(id, keep);
    }
    assert.strictEqual(a.of("generate_slm").length, n);
  }

  // GGUF tab: temperature 0 (greedy) is accepted.
  {
    const a = await start({ run_gguf: async () => ({ text: "x", promptTokens: 1, generated: 1, seconds: 1, tokensPerSec: 1 }) });
    a.set("ggPath", "/g.gguf"); a.set("ggPrompt", "hi"); a.set("ggTemp", "0");
    await a.click("ggRun");
    assert.strictEqual(a.of("run_gguf")[0].args.params.temp, 0);
    a.set("ggTopP", "2");
    await a.click("ggRun");
    assert.ok(/Top-p/.test(a.$("errGguf").textContent));
    assert.strictEqual(a.of("run_gguf").length, 1);
  }

  // ── Evaluate ────────────────────────────────────────────────────────────
  {
    const a = await start({ evaluate_slm: async () => ({ model: EVIL, numPredictions: 1234, crossEntropy: 2.5, bitsPerToken: 3.6, perplexity: 12.25, uniformBaseline: 300 }) });
    await a.click("tab-evaluate");
    await a.click("evRun");
    assert.ok(a.$("errEval").textContent.includes("Model is required"));
    a.set("evModel", "/m.gguf");
    await a.click("evRun");
    assert.ok(a.$("errEval").textContent.includes("Provide held-out text or a file"));
    a.set("evText", "  held out  ");
    await a.click("evRun");
    assert.deepStrictEqual(a.of("evaluate_slm")[0].args, { params: { modelPath: "/m.gguf", text: "held out", textPath: null } });
    captured.evaluate_slm = a.of("evaluate_slm")[0].args.params;
    const rows = a.$("evTable").querySelector("tbody").children;
    assert.strictEqual(rows.length, 1);
    assert.ok(rows[0].innerHTML.includes("&lt;img") && !rows[0].innerHTML.includes("<img"), rows[0].innerHTML);
    assert.ok(rows[0].innerHTML.includes("learned"), "12.25 is far below half of 300");
    await a.click("evRun");
    assert.strictEqual(a.$("evTable").querySelector("tbody").children.length, 2);
    await a.click("evClear");
    assert.strictEqual(a.$("evTable").querySelector("tbody").children.length, 0);
  }

  // ── Models ──────────────────────────────────────────────────────────────
  {
    const a = await start({ model_info: async () => ({ path: EVIL, bytes: 1000, format: "GGUF v3", name: EVIL, task: "x", inputDim: 1, outputDim: 2, tokenizer: EVIL, merges: 3, layers: 4 }) });
    await a.click("miInspect");
    assert.ok(a.$("errModel").textContent.includes("Model file is required"));
    a.set("miPath", "/m.gguf");
    await a.click("miInspect");
    assert.deepStrictEqual(a.of("model_info")[0].args, { path: "/m.gguf" });
    assert.ok(a.$("miInfo").innerHTML.includes("&lt;img") && !a.$("miInfo").innerHTML.includes("<img"));
  }

  // ── GGUF run / inspect ──────────────────────────────────────────────────
  {
    const info = { architecture: "llama", runnable: true, version: 3, numTensors: 10, modelDim: 8, nLayers: 2, nHeads: 2, nKvHeads: 2, vocabSize: 32, tokenizer: "t", estInt4Mb: 1, estInt8Mb: 2, estF32Mb: 4, availMb: 4000, note: "n", paramCount: 1000 };
    const a = await start({
      gguf_info: async () => info,
      run_gguf: async () => ({ text: "hi there", promptTokens: 3, generated: 4, seconds: 2, tokensPerSec: 2 }),
    });
    await a.click("tab-gguf");
    await a.click("ggInspect");
    assert.ok(a.$("errGguf").textContent.includes("GGUF file is required"));
    a.set("ggPath", "/g.gguf");
    await a.click("ggInspect");
    assert.ok(a.$("ggInfo").innerHTML.includes("llama"));
    await a.click("ggRun");
    assert.ok(a.$("errGguf").textContent.includes("Enter a prompt"));
    a.set("ggPrompt", "hello"); a.set("ggQuant", "int4"); a.set("ggSeed", "5"); a.set("ggTopK", "40"); a.$("ggChatFmt").checked = false;
    await a.click("ggRun");
    captured.run_gguf = a.of("run_gguf")[0].args.params;
    assert.deepStrictEqual(a.of("run_gguf")[0].args.params, {
      modelPath: "/g.gguf", prompt: "hello", quant: "int4", maxNew: 64, temp: 0.8, topK: 40, topP: 1, repPenalty: 1,
      genSeed: 5, ids: null, force: false, resume: null, raw: true,
    });
    assert.strictEqual(a.$("ggOut").textContent, "hi there");
    assert.ok(a.$("ggStatus").textContent.startsWith("done — 4 tokens in 2.0s"), a.$("ggStatus").textContent);
    // Token IDs alone are a valid prompt; invalid numbers are rejected before any call.
    a.set("ggPrompt", ""); a.set("ggIds", "1 2 3");
    await a.click("ggRun");
    assert.strictEqual(a.of("run_gguf")[1].args.params.ids, "1 2 3");
    a.set("ggMax", "0");
    const n = a.of("run_gguf").length;
    await a.click("ggRun");
    assert.ok(a.$("errGguf").textContent.includes("Max new tokens must be an integer"));
    assert.strictEqual(a.of("run_gguf").length, n);
  }

  // ── Fine-tune ───────────────────────────────────────────────────────────
  {
    const a = await start({ finetune_gguf: async () => ({ outPath: "/o.flck", epochs: 3, firstLoss: 3.5, finalLoss: 2.5, seconds: 12.5, tokens: 999, steps: 30, bytes: 512 }) });
    await a.click("tab-finetune");
    await a.click("ftRun");
    assert.ok(a.$("errFt").textContent.includes("GGUF file is required"));
    a.set("ftPath", "/g.gguf"); a.set("ftCorpus", "/c.txt"); a.set("ftOut", "/o.flck");
    await a.click("ftRun");
    captured.finetune_gguf = a.of("finetune_gguf")[0].args.params;
    assert.deepStrictEqual(a.of("finetune_gguf")[0].args.params, {
      modelPath: "/g.gguf", corpusPath: "/c.txt", outPath: "/o.flck", epochs: 3, lr: 0.0001, batchSize: 8, seqLen: 64, warmup: 0,
      clip: 1, weightDecay: 0, dropout: 0, qat: false, seed: 1337, threads: 0, resume: null, force: false, verbose: false,
    });
    assert.ok(a.$("ftResult").textContent.includes("Saved 512 bytes → /o.flck"));
    assert.strictEqual(a.$("ggResume").value, "/o.flck", "the GGUF tab is prefilled with the checkpoint");
    a.emit("finetune-progress", { epoch: 1, total: 3, loss: 3.5, ppl: 33.1 });
    assert.strictEqual(a.$("ftBar").style.width, "33%");
    assert.ok(a.$("ftStatus").textContent.includes("ppl 33.10"));
  }

  // ── Export ──────────────────────────────────────────────────────────────
  {
    const a = await start({ export_gguf: async () => ({ outPath: "/out.gguf", bytes: 2048, seconds: 1.25, tensorSummary: "3× Q8_0, 2× F32" }) });
    await a.click("tab-export");
    a.set("exPath", "/g.gguf");
    await a.click("exExport");
    assert.ok(a.$("errExport").textContent.includes("Output file is required"));
    a.set("exOut", "/out.gguf"); a.set("exResume", "/t.flck");
    await a.click("exExport");
    captured.export_gguf = a.of("export_gguf")[0].args.params;
    assert.deepStrictEqual(a.of("export_gguf")[0].args.params, { modelPath: "/g.gguf", outPath: "/out.gguf", quant: "q8_0", resume: "/t.flck", force: false, verbose: false });
    assert.ok(a.$("exResult").textContent.includes("tensors: 3× Q8_0, 2× F32"));
    a.emit("export-progress", "encoding + writing (q8_0)…");
    assert.strictEqual(a.$("exStatus").textContent, "encoding + writing (q8_0)…");
  }

  // ── Stop buttons (G9): shown while a job runs, send cancel_job, report the stop ──
  {
    const hidden = (a, id) => a.$(id).classList.contains("hidden");
    const pending = () => { let release; const p = new Promise((res) => { release = res; }); return { p, release: (v) => release(v) }; };

    // Train: the Stop button appears, cancel_job("train") is sent, "stopped at step N" is shown.
    {
      const t = pending();
      const a = await start({ train_slm: () => t.p, cancel_job: async () => true });
      a.set("trCorpus", "/c.txt"); a.set("trModel", "/m.gguf");
      assert.ok(hidden(a, "trCancel"), "no Stop button before a job runs");
      a.$("trStart").dispatch("click"); await settle(4);
      assert.ok(!hidden(a, "trCancel"), "Stop button while training");
      await a.click("trCancel");
      assert.deepStrictEqual(a.of("cancel_job")[0].args, { kind: "train" });
      assert.strictEqual(a.$("trCancel").disabled, true, "asked once");
      t.release({ ...TRAINED, stopped: true, stoppedAtStep: 7 });
      await settle(6);
      assert.strictEqual(a.$("trStatus").textContent, "stopped at step 7");
      assert.ok(a.$("trResult").textContent.includes("Stopped at step 7"), a.$("trResult").textContent);
      assert.ok(a.$("trResult").textContent.includes("Saved 999 bytes"));
      assert.ok(hidden(a, "trCancel") && !a.$("trStart").disabled);
      assert.ok(a.$("toasts").children.some((x) => x.textContent.includes("stopped at step 7")));
      // A second start while one runs: the backend's message is shown as is.
      const b = await start({ train_slm: async () => { throw "a training job is already running"; } });
      b.set("trCorpus", "/c.txt"); b.set("trModel", "/m.gguf");
      await b.click("trStart");
      assert.strictEqual(b.$("errTrain").textContent, "a training job is already running");
      assert.ok(hidden(b, "trCancel"));
    }

    // Generate: Stop keeps the text so far and says "stopped".
    {
      const t = pending();
      const a = await start({ generate_slm: () => t.p, cancel_job: async () => true });
      a.set("genModel", "/m.gguf"); a.set("genSeed", "ab");
      a.$("genStart").dispatch("click"); await settle(3);
      assert.ok(!hidden(a, "genCancel"));
      await a.click("genCancel");
      assert.deepStrictEqual(a.of("cancel_job")[0].args, { kind: "generate" });
      t.release("abcd");
      await settle(6);
      assert.strictEqual(a.$("genStatus").textContent, "stopped (4 chars)");
      assert.strictEqual(a.$("genOut").textContent, "cd");
      assert.ok(hidden(a, "genCancel") && !a.$("genStart").disabled);
    }

    // GGUF run: stopped result, and a cancel before any text is not an error.
    {
      const t = pending();
      const a = await start({ run_gguf: () => t.p, cancel_job: async () => true });
      a.set("ggPath", "/g.gguf"); a.set("ggPrompt", "hi");
      a.$("ggRun").dispatch("click"); await settle(3);
      await a.click("ggCancel");
      assert.deepStrictEqual(a.of("cancel_job")[0].args, { kind: "run" });
      t.release({ text: "hi the", promptTokens: 1, generated: 2, seconds: 1, tokensPerSec: 2, stopped: true });
      await settle(6);
      assert.ok(a.$("ggStatus").textContent.startsWith("stopped — 2 tokens"), a.$("ggStatus").textContent);
      assert.ok(hidden(a, "ggCancel"));
      const t2 = pending();
      const b = await start({ run_gguf: () => t2.p, cancel_job: async () => true });
      b.set("ggPath", "/g.gguf"); b.set("ggPrompt", "hi");
      b.$("ggRun").dispatch("click"); await settle(3);
      await b.click("ggCancel");
      t2.p.catch(() => {});
      t2.release(Promise.reject("cancelled before any text was produced"));
      await settle(6);
      assert.strictEqual(b.$("ggStatus").textContent, "stopped before any text was produced");
      assert.strictEqual(b.$("errGguf").textContent, "");
    }

    // Fine-tune: stopped result keeps the checkpoint.
    {
      const t = pending();
      const a = await start({ finetune_gguf: () => t.p, cancel_job: async () => true });
      a.set("ftPath", "/g.gguf"); a.set("ftCorpus", "/c.txt"); a.set("ftOut", "/o.flck");
      a.$("ftRun").dispatch("click"); await settle(3);
      assert.ok(!hidden(a, "ftCancel"));
      await a.click("ftCancel");
      assert.deepStrictEqual(a.of("cancel_job")[0].args, { kind: "finetune" });
      t.release({ outPath: "/o.flck", epochs: 1, firstLoss: 3, finalLoss: 2.5, seconds: 1, tokens: 99, steps: 12, bytes: 64, stopped: true });
      await settle(6);
      assert.strictEqual(a.$("ftStatus").textContent, "stopped at step 12");
      assert.ok(a.$("ftResult").textContent.includes("checkpoint so far was saved"));
      assert.strictEqual(a.$("ggResume").value, "/o.flck", "the stopped checkpoint is offered on the GGUF tab");
      assert.ok(hidden(a, "ftCancel"));
    }

    // Export: a cancel is reported as cancelled, not as a failure.
    {
      const t = pending();
      const a = await start({ export_gguf: () => t.p, cancel_job: async () => true });
      a.set("exPath", "/g.gguf"); a.set("exOut", "/out.gguf");
      a.$("exExport").dispatch("click"); await settle(3);
      assert.ok(!hidden(a, "exCancel"));
      await a.click("exCancel");
      assert.deepStrictEqual(a.of("cancel_job")[0].args, { kind: "export" });
      t.p.catch(() => {});
      t.release(Promise.reject("export cancelled; no file was written"));
      await settle(6);
      assert.strictEqual(a.$("exStatus").textContent, "cancelled; nothing was written");
      assert.strictEqual(a.$("errExport").textContent, "");
      assert.ok(hidden(a, "exCancel") && !a.$("exExport").disabled);
    }
  }

  // ── Capable ─────────────────────────────────────────────────────────────
  {
    const report = { cpu: EVIL, cores: 4, threads: 4, memTotal: 8e9, memAvail: 4e9, memBwGbps: 3.5, gemmGflops: 60, inferInt4: 1e9, inferInt8: 5e8, inferF32: 1e8,
      trainChinchilla: 2e6, trainFixed1b: 1e6, testEval: 3e6, loadInt4: 5e9, loadInt8: 2e9, loadF32: 1e9, finetuneLo: 1e5, finetuneHi: 1e6, targetToks: 3,
      trainHours: 24, evalTokens: 1e7, fixedTrainTokens: 1e9, chinchillaRatio: 20, finetuneTokensLo: 1e7, finetuneTokensHi: 1e6, loadFraction: 0.9 };
    const a = await start({ capability_report: async () => report });
    await a.click("tab-capable");
    await a.click("capCheck");
    assert.ok(a.$("capCards").innerHTML.includes("&lt;img") && !a.$("capCards").innerHTML.includes("<img"));
    assert.ok(a.$("capDialog").open, "report dialog opened");
    assert.strictEqual(a.$("capCheck").disabled, false);
    // Capable ran, so the bounds are known: running an oversized GGUF asks first.
    const big = { architecture: "llama", runnable: true, version: 3, numTensors: 1, modelDim: 1, nLayers: 1, nHeads: 1, nKvHeads: 1, vocabSize: 1, tokenizer: "t", estInt4Mb: 1, estInt8Mb: 1, estF32Mb: 1, availMb: 1, note: "", paramCount: 9e9 };
    const b = await start({ capability_report: async () => report, gguf_info: async () => big, run_gguf: async () => ({ text: "x", promptTokens: 1, generated: 1, seconds: 1, tokensPerSec: 1 }) });
    await b.click("capCheck");
    b.set("ggPath", "/g.gguf"); b.set("ggPrompt", "hi");
    b.$("ggRun").dispatch("click"); await settle(6);
    assert.ok(b.$("ggWarnDialog").open, "warning dialog is shown for a 9B model");
    assert.ok(b.$("ggWarnBody").innerHTML.includes("9.00 B"), b.$("ggWarnBody").innerHTML);
    await b.click("ggWarnCancel");
    assert.strictEqual(b.$("ggStatus").textContent, "cancelled");
    assert.strictEqual(b.of("run_gguf").length, 0, "cancel means the model is never loaded");
    b.$("ggRun").dispatch("click"); await settle(6);
    await b.click("ggWarnProceed");
    await settle(6);
    assert.strictEqual(b.of("run_gguf").length, 1, "proceed runs it");
  }

  // ── Browse buttons use the native dialogs ───────────────────────────────
  {
    const a = await start({}, { dialogs: { open: "/picked/file.txt", save: "/picked/save.gguf", dir: "/picked/dir" } });
    await a.click("trCorpusBrowse");
    assert.strictEqual(a.$("trCorpus").value, "/picked/file.txt");
    await a.click("trModelBrowse");
    assert.strictEqual(a.$("trModel").value, "/picked/save.gguf");
    await a.click("dsDestDirBrowse");
    assert.strictEqual(a.$("dsDestDir").value, "/picked/dir");
  }

  // ── Plain browser (no Tauri) ────────────────────────────────────────────
  {
    const a = await start({}, { tauri: false });
    assert.ok(!a.$("webBanner").classList.contains("hidden"));
    a.set("trCorpus", "/c.txt"); a.set("trModel", "/m.gguf");
    await a.click("trStart");
    assert.ok(a.$("errTrain").textContent.includes("Not running inside the Tauri runtime"));
    assert.strictEqual(a.$("trStart").disabled, false);
  }

  // Pin the captured arguments to the fixture the Rust tests read.
  {
    const file = require("node:path").join(__dirname, "command_params.fixture.json");
    const sorted = Object.fromEntries(Object.keys(captured).sort().map((k) => [k, captured[k]]));
    assert.strictEqual(Object.keys(sorted).length, 7, "every flow captured its arguments");
    if (process.env.UPDATE_FIXTURE) require("node:fs").writeFileSync(file, JSON.stringify(sorted, null, 2) + "\n");
    assert.deepStrictEqual(JSON.parse(require("node:fs").readFileSync(file, "utf8")), sorted, "command_params.fixture.json is stale; run UPDATE_FIXTURE=1 node ui/app_flows.test.js");
  }

  console.log("app_flows.test.js: all assertions passed");
})().catch((e) => { console.error(e); process.exit(1); });
