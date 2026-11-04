// Functional tests of app.js: the real script, the real index.html (stub DOM),
// a fake Tauri backend. Each test is a user flow: fill the form, click the
// button, check the command that was sent and what the page shows.
// What this CANNOT see (no display here): layout, the real webview, the CSP in
// a running window. Run with: node ui/tests/app_flows.test.js
"use strict";
const assert = require("node:assert");
const { loadApp, settle } = require("./testapp.js");

const PF_OK = { params: 64000, peakRamMb: 300, ramAvailableMb: 4000, ramLimitMb: 3000, fitsInRam: true, seconds: 5, llamaCpp: { runs: true, reason: "Byte-level BPE models are saved as gpt2." }, problems: [] };
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
      case "preflight_train": return PF_OK;
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

// The exact `params` objects each flow sent; pinned to ui/tests/command_params.fixture.json,
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

  // ── Tool-tips: hover a tab and a heading, focus an output area, Escape, runtime buttons ──
  {
    const a = await start();
    const bubble = a.$("ferrum-tooltip");
    const shown = () => !bubble.classList.contains("hidden");
    // Hover waits for the show delay: a real timer here, capped at 5 ms by testapp.js, so each
    // step yields to the timer queue (bounded, not a fixed sleep).
    const tick = () => new Promise((r) => setTimeout(r, 1));
    const hover = async (el) => { el.dispatch("mouseenter"); for (let i = 0; i < 200 && !shown(); i++) await tick(); };
    const descOf = (el) => a.doc.getElementById(el.getAttribute("aria-describedby")).textContent;

    await hover(a.$("tab-train"));
    assert.ok(shown(), "hovering a tab shows its tip");
    assert.ok(bubble.textContent.includes("Shows the Train page") && bubble.textContent.includes("pre-flight check"), bubble.textContent);
    a.$("tab-train").dispatch("mouseleave");
    assert.ok(!shown());

    const h2 = a.$("panel-datasets").querySelector("h2");
    await hover(h2);
    assert.ok(shown(), "hovering a heading shows its tip");
    assert.strictEqual(bubble.textContent, h2.getAttribute("data-tip"), "a data-tip shows as it is, with no title");
    assert.strictEqual(descOf(h2), h2.getAttribute("data-tip"), "and screen readers get the same text");
    h2.dispatch("mouseleave");

    // Keyboard: the focusable engine log shows its tip at once; Escape hides it.
    a.$("termOut").focus();
    assert.ok(shown() && bubble.textContent === a.$("termOut").getAttribute("data-tip"));
    a.doc.dispatch("keydown", { key: "Escape" });
    assert.ok(!shown());
    a.$("termOut").blur();

    // The buttons each error line gets at runtime have tips too.
    await a.click("evRun"); // no model: an inline error
    const box = a.$("errEval").nextSibling;
    const copy = box.children.find((c) => c.textContent === "Copy details");
    const expand = box.children.find((c) => c.textContent === "Show full message");
    const { ERR_TIPS } = require("../core/common.js");
    assert.strictEqual(descOf(copy), ERR_TIPS.copy);
    assert.strictEqual(descOf(expand), ERR_TIPS.expand);
    copy.focus();
    assert.strictEqual(bubble.textContent, ERR_TIPS.copy);
    copy.blur();

    // Without live.js (no getLive, or it answers null) the buttons still carry their data-tip.
    const { Document } = require("./testdom.js");
    globalThis.FerrumErrors = require("../core/errors.js"); // common.js reads it from the global, as in the page
    for (const getLive of [undefined, () => null]) {
      const doc = new Document();
      const row = doc.createElement("div");
      const err = doc.createElement("span");
      err.id = "e";
      row.appendChild(err);
      doc.body.appendChild(row);
      require("../core/common.js").create({ $: (id) => doc.getElementById(id), doc, getLive }).setErr("e", "boom");
      const b = row.children[1].children.find((c) => c.textContent === "Copy details");
      assert.strictEqual(b.getAttribute("data-tip"), ERR_TIPS.copy);
      assert.strictEqual(b.getAttribute("aria-describedby"), null);
    }

    // No native title bubble competes with ours.
    assert.strictEqual(a.$("connDot").getAttribute("title"), null);
    assert.strictEqual(a.$("hwLine").getAttribute("title"), null);
  }

  // ── Train ───────────────────────────────────────────────────────────────
  {
    const a = await start({ train_slm: async () => TRAINED });
    await a.click("tab-train");
    await a.click("trStart");
    assert.strictEqual(a.$("errTrain").textContent, "Corpus is required");
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
      valFraction: 0.1, patience: 0, force: false, quant: "int8",
    });
    assert.strictEqual(a.$("trStatus").textContent, "done");
    // The precision select offers the CLI's three values and is sent as chosen.
    a.set("trQuant", "int4");
    await a.click("trStart");
    assert.strictEqual(a.of("train_slm").pop().args.params.quant, "int4");
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

  // download_progress_updates_bar: download-progress events move the one shared bar; the finished
  // command settles it at 100 %; a failure puts it back to idle.
  {
    let a;
    const mb = 1e6;
    a = await start({
      download_dataset: async () => {
        assert.strictEqual(a.$("dsProgress").textContent, "starting...");
        a.emit("download-progress", { id: "x", bytes: 1 * mb, total: 4 * mb });
        assert.strictEqual(a.$("dsBar").style.width, "25%");
        assert.strictEqual(a.$("dsProgress").textContent, "x: 1.0 MB of 4.0 MB");
        a.emit("download-progress", { id: "x", bytes: 3 * mb, total: 4 * mb });
        assert.strictEqual(a.$("dsBar").style.width, "75%");
        a.emit("download-progress", { id: "x", bytes: 4 * mb, total: 4 * mb });
        return { path: "/d/x.txt", bytes: 4 * mb, source: "huggingface" };
      },
      download_hf_file: async () => {
        // No Content-Length: the status counts bytes and the bar stays where it was.
        a.emit("download-progress", { id: "o/n/f", bytes: 2 * mb, total: null });
        assert.strictEqual(a.$("dsProgress").textContent, "o/n/f: 2.0 MB");
        assert.strictEqual(a.$("dsBar").style.width, "0%");
        throw "download failed: boom";
      },
      list_datasets: async () => [{ id: "x", name: "X", source: "huggingface", approxMb: 5, format: "txt", description: "d" }],
    });
    a.set("dsDestDir", "/d");
    a.$("dsCatalog").querySelector("button").dispatch("click"); await settle(6);
    assert.strictEqual(a.$("dsBar").style.width, "100%");
    assert.strictEqual(a.$("dsProgress").textContent, "downloaded 3.81 MB from huggingface");
    a.set("dsHfRepo", "o/n"); a.set("dsHfFile", "f");
    await a.click("dsHfDownload");
    assert.strictEqual(a.$("dsProgress").textContent, "idle");
    assert.ok(a.$("errDsCatalog").textContent.includes("boom"));
  }

  // one_bar_one_job: the Source URL download moves the bar like the others, and while any
  // download or Clean runs, every button that would start another one is disabled (gap L9).
  {
    let a;
    const mb = 1e6;
    const others = () => ["dsDownload", "dsHfDownload", "dsKgDownload", "dsClean"].map((id) => a.$(id).disabled)
      .concat([a.$("dsCatalog").querySelector("button").disabled]);
    a = await start({
      download_text: async (args) => {
        assert.deepStrictEqual(others(), [true, true, true, true, true]);
        a.emit("download-progress", { id: args.url, bytes: 1 * mb, total: 2 * mb });
        assert.strictEqual(a.$("dsBar").style.width, "50%");
        assert.strictEqual(a.$("dsProgress").textContent, "https://e.x/t.txt: 1.0 MB of 2.0 MB");
        return "abc";
      },
      download_kaggle_file: async () => {
        assert.deepStrictEqual(others(), [true, true, true, true, true]);
        throw "download failed: boom";
      },
      list_datasets: async () => [{ id: "x", name: "X", source: "huggingface", approxMb: 5, format: "txt", description: "d" }],
    });
    assert.deepStrictEqual(others(), [false, false, false, false, false]);
    a.set("dsUrl", "https://e.x/t.txt");
    await a.click("dsDownload");
    assert.strictEqual(a.$("dsBar").style.width, "100%");
    assert.strictEqual(a.$("dsProgress").textContent, "downloaded 0.0 MB");
    assert.deepStrictEqual(others(), [false, false, false, false, false]);
    // A failure frees the buttons too; a form error before the start leaves them as they were.
    a.set("dsDestDir", "/d"); a.set("dsKgRepo", "o/d"); a.set("dsKgFile", "k.csv");
    await a.click("dsKgDownload");
    assert.strictEqual(a.$("dsProgress").textContent, "idle");
    assert.deepStrictEqual(others(), [false, false, false, false, false]);
    a.set("dsUrl", "");
    await a.click("dsDownload");
    assert.ok(a.$("errDsSource").textContent.includes("Source URL is required"));
    assert.deepStrictEqual(others(), [false, false, false, false, false]);
  }

  // clean_progress_updates_bar: clean-progress events fill the bar against the raw text's size.
  {
    let a;
    a = await start({
      clean_text: async () => {
        a.emit("clean-progress", { bytes: 6, documents: 1 });
        assert.strictEqual(a.$("dsBar").style.width, "50%");
        assert.strictEqual(a.$("dsProgress").textContent, "cleaning: 0.0 MB - 1 documents");
        a.emit("clean-progress", { bytes: 12, documents: 2 });
        return { cleaned: "hello", preview: "hello", chars: 5, bytes: 5, lines: 1, words: 1, uniqueChars: 4 };
      },
    });
    a.set("dsRaw", "twelve bytes");
    await a.click("dsClean");
    assert.strictEqual(a.$("dsBar").style.width, "100%");
    assert.strictEqual(a.$("dsProgress").textContent, "cleaned 0.0 MB");
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
    for (const [id, v, msg] of [["genTemp", "-1", /Temperature/], ["genTopP", "1.5", /Top-p/], ["genTopP", "0", /Top-p/], ["genRep", "0", /Repetition/], ["genTopK", "-2", /Top-k/], ["genMaxTokens", "-1", /Max tokens/]]) {
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
    const report = (extra = {}) => ({ perplexity: 12.25, cross_entropy_nats: 2.5, bits_per_token: 3.6, bits_per_byte: 1.9, tokens: 1234, bytes: 2000, seconds: 0.5, ...extra });
    const row = (label, extra = {}) => ({ model: EVIL, label, numPredictions: 1234, crossEntropy: 2.5, bitsPerToken: 3.6, bitsPerByte: 1.9, perplexity: 12.25, uniformBaseline: 300, bytes: 2000, seconds: 0.5, report: report(extra), ...extra });
    let answer = [row("held-out")];
    const a = await start({ evaluate_slm: async () => answer, save_corpus: async (args) => args.text.length });
    await a.click("tab-evaluate");
    await a.click("evRun");
    assert.ok(a.$("errEval").textContent.includes("Model is required"));
    a.set("evModel", "/m.gguf");
    await a.click("evRun");
    assert.ok(a.$("errEval").textContent.includes("Provide held-out text or a file"));
    a.set("evText", "  held out  ");
    await a.click("evRun");
    assert.deepStrictEqual(a.of("evaluate_slm")[0].args, { params: {
      modelPath: "/m.gguf", text: "held out", textPath: null, testPath: null, lossy: false, quant: "f32", ctx: 0, resume: null, force: false,
    } });
    captured.evaluate_slm = a.of("evaluate_slm")[0].args.params;
    const rows = a.$("evTable").querySelector("tbody").children;
    assert.strictEqual(rows.length, 1);
    assert.ok(rows[0].innerHTML.includes("&lt;img") && !rows[0].innerHTML.includes("<img"), rows[0].innerHTML);
    assert.ok(rows[0].innerHTML.includes("learned"), "12.25 is far below half of 300");
    assert.ok(rows[0].innerHTML.includes("<td>1.9000</td>") && rows[0].innerHTML.includes("<td>0.50</td>"), rows[0].innerHTML);
    assert.strictEqual(a.$("evStatus").textContent, "added");

    // Save JSON: refuses before a path is given; one row is written as the CLI's `eval --json`.
    await a.click("evSaveJson");
    assert.ok(a.$("errEvalJson").textContent.includes("JSON output path is required"));
    a.set("evJsonPath", "/out/eval.json");
    await a.click("evSaveJson");
    assert.strictEqual(a.$("errEvalJson").textContent, "");
    const saved = a.of("save_corpus")[0].args;
    assert.strictEqual(saved.path, "/out/eval.json");
    assert.deepStrictEqual(JSON.parse(saved.text), report());
    assert.ok(saved.text.endsWith("}\n"));
    assert.deepStrictEqual(Object.keys(JSON.parse(saved.text)).sort(),
      ["bits_per_byte", "bits_per_token", "bytes", "cross_entropy_nats", "perplexity", "seconds", "tokens"]);

    // Every option reaches the backend; a held-out + test pair adds two rows and saves both.
    a.set("evFile", "/h.txt"); a.set("evText", ""); a.set("evTestFile", "/t.txt"); a.$("evLossy").checked = true;
    a.set("evQuant", "int8"); a.set("evCtx", "64"); a.set("evResume", "/ck.flck"); a.$("evForce").checked = true;
    answer = [row("held-out"), row("test", { perplexity: 200 })];
    await a.click("evRun");
    assert.deepStrictEqual(a.of("evaluate_slm")[1].args.params, {
      modelPath: "/m.gguf", text: null, textPath: "/h.txt", testPath: "/t.txt", lossy: true, quant: "int8", ctx: 64, resume: "/ck.flck", force: true,
    });
    assert.strictEqual(a.$("evTable").querySelector("tbody").children.length, 3);
    assert.strictEqual(a.$("evStatus").textContent, "added 2 rows");
    assert.ok(a.$("evTable").querySelector("tbody").children[2].innerHTML.includes("weak"));
    await a.click("evSaveJson");
    const pair = JSON.parse(a.of("save_corpus")[1].args.text);
    assert.deepStrictEqual(Object.keys(pair), ["held_out", "test"]);
    assert.strictEqual(pair.test.perplexity, 200);
    // A bad window is refused before any call.
    const n = a.of("evaluate_slm").length;
    a.set("evCtx", "-1");
    await a.click("evRun");
    assert.ok(a.$("errEval").textContent.includes("Window tokens"));
    assert.strictEqual(a.of("evaluate_slm").length, n);
    await a.click("evClear");
    assert.strictEqual(a.$("evTable").querySelector("tbody").children.length, 0);

    // Nothing to save before a run succeeds.
    const b = await start({});
    b.set("evJsonPath", "/x.json");
    await b.click("evSaveJson");
    assert.ok(b.$("errEvalJson").textContent.includes("Run an evaluation first"));
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

  // Models tab on a non-model Ferrum file (gap L12): the kind and the CLI's fields, escaped.
  {
    const a = await start({ model_info: async () => ({ path: "/t.gguf", bytes: 10, format: "GGUF v3 (ferrum-tokenizer)", name: "", task: "", inputDim: 0, outputDim: 0, tokenizer: "", merges: 0, layers: 0, kind: "BPE tokenizer file", details: [["Vocab", "257"], ["Merges", EVIL]] }) });
    a.set("miPath", "/t.gguf");
    await a.click("miInspect");
    const html = a.$("miInfo").innerHTML;
    assert.ok(html.includes("BPE tokenizer file") && html.includes("257") && html.includes("Merges"), html);
    assert.ok(!html.includes("Layers") && !html.includes("<img") && html.includes("&lt;img"), html);
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
      genSeed: 5, ids: null, force: false, resume: null, raw: true, stop: [],
    });
    assert.strictEqual(a.$("ggOut").textContent, "hi there");
    assert.ok(a.$("ggStatus").textContent.startsWith("done — 4 tokens in 2.0s"), a.$("ggStatus").textContent);
    // Token IDs alone are a valid prompt; invalid numbers are rejected before any call.
    a.set("ggPrompt", ""); a.set("ggIds", "1 2 3");
    await a.click("ggRun");
    assert.strictEqual(a.of("run_gguf")[1].args.params.ids, "1 2 3");
    // Stop strings: one per line, CR and empty lines dropped, escapes left for the backend.
    a.set("ggStop", "\\n\r\n\nEND\n");
    await a.click("ggRun");
    assert.deepStrictEqual(a.of("run_gguf")[2].args.params.stop, ["\\n", "END"]);
    a.set("ggStop", "");
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

  // ── Prepare (G3): the form, the exact arguments, progress, result, errors, stop ──
  {
    const PREPARED = {
      outDir: "/out", manifestPath: "/out/manifest.json", files: 2, seconds: 0.4,
      train: { path: "/out/train.txt", documents: 90, chars: 9000, bytes: 9100 },
      val: { path: "/out/val.txt", documents: 5, chars: 500, bytes: 510 },
      test: { path: "/out/test.txt", documents: 0, chars: 0, bytes: 0 },
      rejected: { duplicate: 2, empty: 0, invalid_utf8: 0, too_long: 0, too_short: 1 }, lossyRepaired: 3,
      summary: "prepared a b -> /out  (train=90 val=5 test=0)  rejected duplicate=2 too_short=1  lossy_repaired=3",
    };
    const a = await start({ prepare_dataset: async () => PREPARED });
    await a.click("tab-prepare");
    assert.ok(a.$("panel-prepare").classList.contains("active"));
    // The defaults are the CLI's.
    assert.deepStrictEqual(["dpVal", "dpTest", "dpSeed", "dpMin", "dpMax", "dpSep"].map((i) => a.$(i).value), ["0.05", "0.05", "42", "1", "0", "<|endoftext|>"]);
    await a.click("dpRun");
    assert.strictEqual(a.$("errPrepare").textContent, "Add at least one input file or folder");
    a.set("dpInputs", "/data/a.txt\n\n  /data/folder  \n"); await a.click("dpRun");
    assert.strictEqual(a.$("errPrepare").textContent, "Output folder is required");
    a.set("dpOut", "/out"); a.set("dpVal", "0.6"); a.set("dpTest", "0.5"); await a.click("dpRun");
    assert.ok(a.$("errPrepare").textContent.includes("add up to less than 1"), a.$("errPrepare").textContent);
    a.set("dpVal", "0.05"); a.set("dpTest", "1"); await a.click("dpRun");
    assert.ok(a.$("errPrepare").textContent.startsWith("Test share must be at least 0 and below 1"));
    a.set("dpTest", "0.05"); a.set("dpSeed", "-3"); await a.click("dpRun");
    assert.ok(a.$("errPrepare").textContent.startsWith("Split seed must be a whole number"));
    assert.strictEqual(a.of("prepare_dataset").length, 0, "nothing is sent for an invalid form");

    a.set("dpSeed", "7"); a.set("dpMin", "50"); a.set("dpMax", "20000");
    for (const id of ["dpDedup", "dpNfkc", "dpLossy"]) a.$(id).checked = true;
    await a.click("dpRun");
    assert.strictEqual(a.$("errPrepare").textContent, "");
    const sent = a.of("prepare_dataset")[0].args.params;
    captured.prepare_dataset = sent;
    assert.deepStrictEqual(sent, {
      inputs: ["/data/a.txt", "/data/folder"], outDir: "/out", valFrac: 0.05, testFrac: 0.05, seed: 7, minChars: 50, maxChars: 20000,
      docSep: "<|endoftext|>", dedup: true, nfkc: true, lowercase: false, stripGutenberg: false, lossy: true,
    });
    const text = a.$("dpResult").textContent;
    assert.ok(text.includes("Prepared 2 file(s)") && text.includes("train: 90 documents, 9,000 characters, 9,100 bytes -> /out/train.txt"), text);
    assert.ok(text.includes("rejected: duplicate=2  too short=1") && text.includes("repaired invalid UTF-8 in 3") && text.includes("manifest: /out/manifest.json"), text);
    assert.ok(text.includes("(train=90 val=5 test=0)"), "the CLI summary line is shown");
    assert.strictEqual(a.$("dpStatus").textContent, "done");
    assert.strictEqual(a.$("dpBar").style.width, "100%");
    assert.ok(!a.$("dpUse").classList.contains("hidden"));
    assert.strictEqual(a.$("dpRun").disabled, false, "button re-enabled");

    // Progress events drive the bar and status.
    a.emit("prepare-progress", { file: 1, files: 2, path: "/data/a.txt", bytesDone: 5000000, bytesTotal: 20000000, documents: 4096 });
    assert.strictEqual(a.$("dpBar").style.width, "25%");
    assert.strictEqual(a.$("dpStatus").textContent, "file 1/2 - 5.0 MB of 20.0 MB - 4,096 documents read");

    // "Use the splits": train always, val/test only when they have documents.
    a.set("evFile", ""); a.set("evTestFile", "");
    await a.click("dpUse");
    assert.deepStrictEqual(["trCorpus", "evFile", "evTestFile"].map((i) => a.$(i).value), ["/out/train.txt", "/out/val.txt", ""]);
  }
  {
    // The browse buttons append to the list (several paths at once, a folder too).
    const a = await start({}, { dialogs: { open: ["/p/one.txt", "/p/two.txt"], dir: "/p/dir" } });
    await a.click("dpAddFile");
    assert.strictEqual(a.$("dpInputs").value, "/p/one.txt\n/p/two.txt");
    await a.click("dpAddDir");
    assert.strictEqual(a.$("dpInputs").value, "/p/one.txt\n/p/two.txt\n/p/dir");
    await a.click("dpOutBrowse");
    assert.strictEqual(a.$("dpOut").value, "/p/dir", "the output field takes a single folder");
  }
  {
    // Errors are shown as the backend wrote them (path and byte offset included); Stop is wired.
    const bad = "/data/a.txt: invalid UTF-8 at byte 14 (tick 'Replace invalid UTF-8 bytes' to repair them instead)";
    const a = await start({ prepare_dataset: async () => { throw bad; } });
    a.set("dpInputs", "/data/a.txt"); a.set("dpOut", "/out");
    await a.click("dpRun");
    assert.strictEqual(a.$("errPrepare").textContent, bad);
    assert.strictEqual(a.$("dpStatus").textContent, "error");
    assert.strictEqual(a.$("dpRun").disabled, false);
    assert.ok(a.$("dpUse").classList.contains("hidden"));

    let release;
    const a2 = await start({ prepare_dataset: () => new Promise((_, rej) => { release = rej; }), cancel_job: async () => true });
    a2.set("dpInputs", "/data/a.txt"); a2.set("dpOut", "/out");
    assert.ok(a2.$("dpCancel").classList.contains("hidden"));
    a2.$("dpRun").dispatch("click"); await settle(4);
    assert.ok(!a2.$("dpCancel").classList.contains("hidden") && a2.$("dpRun").disabled);
    await a2.click("dpCancel");
    assert.deepStrictEqual(a2.of("cancel_job")[0].args, { kind: "prepare" });
    release("prepare cancelled; no output files were changed");
    await settle(6);
    assert.strictEqual(a2.$("dpStatus").textContent, "stopped; no files were changed");
    assert.strictEqual(a2.$("errPrepare").textContent, "");
    assert.ok(a2.$("dpCancel").classList.contains("hidden") && !a2.$("dpRun").disabled);
  }

  // ── Run in Ollama / LM Studio (G4): preview, the three run buttons, errors, Stop ──
  {
    const PV = (target) => ({ target, ran: false, modelfile: target === "ollama" ? "FROM \"/m/m.gguf\"\nPARAMETER num_ctx 64\n" : null,
      commands: target === "ollama" ? ["/bin/ollama create ferrum-model -f \"<temporary Modelfile>\""] : ["/bin/lms import -y -l --user-repo ferrum/m /m/m.gguf", "/bin/lms ls --json"],
      tool: target === "ollama" ? "/bin/ollama" : null, toolError: target === "ollama" ? null : "cannot find 'lms'", written: null, output: "", key: null, summary: "nothing was run" });
    const RAN = { ...PV("ollama"), ran: true, commands: ["/bin/ollama create ferrum-model -f /tmp/x.Modelfile"], output: "success\n", summary: "exported to Ollama as 'ferrum-model'  (try: ollama run ferrum-model)" };
    const a = await start({
      preview_runtime_export: async (args) => PV(args.params.target),
      export_to_runtime: async (args) => (args.params.target === "lmstudio"
        ? { ...PV("lmstudio"), ran: true, key: "ferrum/m", tool: "/bin/lms", toolError: null, summary: "imported into LM Studio as modelKey 'ferrum/m'  (try: lms load ferrum/m)" }
        : args.params.modelfileOut ? { ...PV("ollama"), written: args.params.modelfileOut, summary: "wrote " + args.params.modelfileOut + "; nothing was run" } : RAN),
    });
    await a.click("tab-export");
    for (const id of ["rxPreview", "rxOllama", "rxLms", "rxWriteModelfile"]) {
      await a.click(id);
      assert.strictEqual(a.$("errRuntime").textContent, "Ferrum model (.gguf) is required", id);
    }
    assert.strictEqual(a.of("export_to_runtime").length + a.of("preview_runtime_export").length, 0, "nothing is sent for an invalid form");
    a.set("rxPath", "/m/m.gguf");
    await a.click("rxWriteModelfile");
    assert.strictEqual(a.$("errRuntime").textContent, "Modelfile to write is required");
    a.set("rxName", "  "); await a.click("rxOllama");
    assert.strictEqual(a.$("errRuntime").textContent, "Ollama model name is required");
    a.set("rxName", "ferrum-model");

    // Preview: both targets, nothing run, the Modelfile shown verbatim.
    await a.click("rxPreview");
    const pv = a.of("preview_runtime_export");
    assert.deepStrictEqual(pv.map((c) => c.args.params.target), ["ollama", "lmstudio"]);
    captured.preview_runtime_export = pv[0].args.params;
    assert.strictEqual(a.$("rxModelfile").value, "FROM \"/m/m.gguf\"\nPARAMETER num_ctx 64\n");
    const text = a.$("rxResult").textContent;
    assert.ok(text.includes("Ollama: tool /bin/ollama") && text.includes("LM Studio: tool not found (cannot find 'lms')") && text.includes("/bin/lms ls --json") && text.endsWith("Nothing was run."), text);
    assert.strictEqual(a.of("export_to_runtime").length, 0);

    // Export to Ollama: the exact arguments, the result and the tool's output.
    a.set("rxOllamaBin", " /bin/ollama ");
    await a.click("rxOllama");
    const sent = a.of("export_to_runtime")[0].args.params;
    captured.export_to_runtime = sent;
    assert.deepStrictEqual(sent, { modelPath: "/m/m.gguf", target: "ollama", name: "ferrum-model", repo: "", bin: "/bin/ollama", modelfileOut: "" });
    assert.ok(a.$("rxResult").textContent.includes("exported to Ollama as 'ferrum-model'") && a.$("rxResult").textContent.includes("ran: /bin/ollama create") && a.$("rxResult").textContent.includes("output: success"), a.$("rxResult").textContent);
    assert.strictEqual(a.$("rxStatus").textContent, "done");
    assert.ok(a.$("rxOllama").disabled === false && a.$("rxCancel").classList.contains("hidden"));

    // LM Studio: its own tool field and repo, none of the Ollama fields.
    a.set("rxLmsBin", "/bin/lms"); a.set("rxRepo", "me/m");
    await a.click("rxLms");
    assert.deepStrictEqual(a.of("export_to_runtime")[1].args.params, { modelPath: "/m/m.gguf", target: "lmstudio", name: "", repo: "me/m", bin: "/bin/lms", modelfileOut: "" });
    assert.ok(a.$("rxResult").textContent.includes("modelKey 'ferrum/m'"));

    // Write the Modelfile only: no tool involved, and the result says nothing ran.
    a.set("rxModelfileOut", "/o/x.Modelfile");
    await a.click("rxWriteModelfile");
    const wm = a.of("export_to_runtime")[2].args.params;
    assert.strictEqual(wm.modelfileOut, "/o/x.Modelfile");
    assert.ok(a.$("rxResult").textContent.startsWith("wrote /o/x.Modelfile; nothing was run") && a.$("rxResult").textContent.includes("would run:"), a.$("rxResult").textContent);
  }
  {
    // The refusal text of the backend is shown as it is (the Manual pointer included).
    const refusal = "/m/l.gguf: architecture 'llama': Ferrum exports only its own gpt2 models; for a llama GGUF write a Modelfile by hand (Manual 9.5)";
    const a = await start({ export_to_runtime: async () => { throw refusal; }, preview_runtime_export: async () => { throw refusal; } });
    a.set("rxPath", "/m/l.gguf");
    await a.click("rxOllama");
    assert.strictEqual(a.$("errRuntime").textContent, refusal);
    assert.strictEqual(a.$("rxStatus").textContent, "error");
    assert.strictEqual(a.$("rxOllama").disabled, false);
    await a.click("rxPreview");
    assert.strictEqual(a.$("errRuntime").textContent, refusal);

    let release;
    const b = await start({ export_to_runtime: () => new Promise((_, rej) => { release = rej; }), cancel_job: async () => true });
    b.set("rxPath", "/m/m.gguf");
    assert.ok(b.$("rxCancel").classList.contains("hidden"));
    b.$("rxLms").dispatch("click"); await settle(4);
    assert.ok(!b.$("rxCancel").classList.contains("hidden") && b.$("rxLms").disabled);
    await b.click("rxCancel");
    assert.deepStrictEqual(b.of("cancel_job")[0].args, { kind: "runtime_export" });
    release("export cancelled; nothing was run");
    await settle(6);
    assert.strictEqual(b.$("rxStatus").textContent, "stopped");
    assert.strictEqual(b.$("errRuntime").textContent, "");
    assert.ok(b.$("rxCancel").classList.contains("hidden") && !b.$("rxLms").disabled);
  }
  {
    // Training a BPE model offers it to this tab; a character-level one is not offered.
    const bpe = await start({ train_slm: async () => TRAINED });
    bpe.set("trCorpus", "/c.txt"); bpe.set("trModel", "/m/out.gguf");
    await bpe.click("trStart");
    assert.strictEqual(bpe.$("rxPath").value, "/m/out.gguf");
    const chars = await start({ train_slm: async () => ({ ...TRAINED, tokenizer: "character-level" }) });
    chars.set("trCorpus", "/c.txt"); chars.set("trModel", "/m/out.gguf");
    await chars.click("trStart");
    assert.strictEqual(chars.$("rxPath").value, "");
  }

  // ── Tokenize corpus (Train tab): form, arguments, progress, result, errors, Stop, preflight ──
  {
    const DONE = { outPath: "/d/c.tok.gguf", tokens: 6362444, documents: 24823, files: 2, vocabSize: 1024, bytesIn: 20190677, bytesOut: 12725336, seconds: 1.5,
      summary: "tokenized /d/corpus -> /d/c.tok.gguf: 6362444 tokens, 24823 documents, vocab 1024 (20190677 bytes in, 12725336 bytes out, 1.5s)" };
    const a = await start({ tokenize_corpus: async () => DONE });
    await a.click("tab-train");
    await a.click("tkRun");
    assert.strictEqual(a.$("errTokenize").textContent, "Text file or folder is required");
    a.set("tkInput", "/d/corpus"); await a.click("tkRun");
    assert.ok(a.$("errTokenize").textContent.startsWith("Fill in the Tokenizer file field above"), a.$("errTokenize").textContent);
    a.set("trTokenizer", "/d/tok.gguf"); await a.click("tkRun");
    assert.strictEqual(a.$("errTokenize").textContent, "Token file to write is required");
    assert.strictEqual(a.of("tokenize_corpus").length, 0, "nothing is sent for an invalid form");
    a.set("tkOut", "/d/c.tok.gguf");
    const preflights = a.of("preflight_train").length;
    await a.click("tkRun");
    await new Promise((r) => setTimeout(r, 40)); await settle(10); // the pre-flight debounce runs for real (5 ms in the stub)
    const sent = a.of("tokenize_corpus")[0].args.params;
    captured.tokenize_corpus = sent;
    assert.deepStrictEqual(sent, { input: "/d/corpus", tokenizerPath: "/d/tok.gguf", outPath: "/d/c.tok.gguf", docSep: "<|endoftext|>" });
    const text = a.$("tkResult").textContent;
    assert.ok(text.includes("Wrote 6,362,444 tokens from 24,823 documents (2 file(s), vocabulary 1024)") && text.includes("-> /d/c.tok.gguf") && text.includes("tokenized /d/corpus"), text);
    assert.strictEqual(a.$("tkStatus").textContent, "done");
    assert.strictEqual(a.$("trCorpus").value, "/d/c.tok.gguf", "Train now points at the token file");
    assert.ok(a.of("preflight_train").length > preflights, "the pre-flight is asked again for the new corpus");
    assert.strictEqual(a.of("preflight_train").slice(-1)[0].args.params.corpusPath, "/d/c.tok.gguf");
    assert.strictEqual(a.of("preflight_train").slice(-1)[0].args.params.tokenizerPath, "/d/tok.gguf");
    assert.strictEqual(a.$("tkRun").disabled, false);
    a.emit("tokenize-progress", { file: 1, files: 2, path: "/d/corpus/a.txt", bytesDone: 10000000, bytesTotal: 40000000, tokens: 2500000 });
    assert.strictEqual(a.$("tkBar").style.width, "25%");
    assert.strictEqual(a.$("tkStatus").textContent, "file 1/2 - 10.0 MB of 40.0 MB - 2,500,000 tokens");
  }
  {
    // The pre-flight of a token file: the corpus cards, in plain text, and the same Start rules.
    const PF = { ...PF_OK, corpus: { tokens: 6362444, documents: 24823, vocab: 1024, ramScalesWithCorpus: false, note: "Token file: the ids stay on disk." } };
    const a = await start({ preflight_train: async () => PF });
    await a.click("tab-train");
    await settle(8);
    const html = a.$("trPreflight").innerHTML;
    assert.ok(html.includes("Token file: 6,362,444 tokens, 24,823 documents, vocabulary 1024") && html.includes("the ids stay on disk"), html);
    assert.strictEqual(a.$("trStart").disabled, false);
  }
  {
    // Errors as the backend wrote them, and Stop.
    const bad = "/d/a.txt: invalid UTF-8 at byte 9 (the Prepare tab's 'Replace invalid UTF-8 bytes' repairs it)";
    const a = await start({ tokenize_corpus: async () => { throw bad; } });
    a.set("tkInput", "/d/a.txt"); a.set("tkOut", "/d/o.tok.gguf"); a.set("trTokenizer", "/d/tok.gguf");
    await a.click("tkRun");
    assert.strictEqual(a.$("errTokenize").textContent, bad);
    assert.strictEqual(a.$("tkStatus").textContent, "error");
    assert.strictEqual(a.$("trCorpus").value, "", "a failed run does not change the corpus");

    let release;
    const b = await start({ tokenize_corpus: () => new Promise((_, rej) => { release = rej; }), cancel_job: async () => true });
    b.set("tkInput", "/d/a.txt"); b.set("tkOut", "/d/o.tok.gguf"); b.set("trTokenizer", "/d/tok.gguf");
    assert.ok(b.$("tkCancel").classList.contains("hidden"));
    b.$("tkRun").dispatch("click"); await settle(4);
    assert.ok(!b.$("tkCancel").classList.contains("hidden") && b.$("tkRun").disabled);
    await b.click("tkCancel");
    assert.deepStrictEqual(b.of("cancel_job")[0].args, { kind: "tokenize" });
    release("tokenize cancelled; no file was written");
    await settle(6);
    assert.strictEqual(b.$("tkStatus").textContent, "stopped; no file was written");
    assert.strictEqual(b.$("errTokenize").textContent, "");
    assert.ok(b.$("tkCancel").classList.contains("hidden") && !b.$("tkRun").disabled);
  }

  // ── Train pre-flight (G8): numbers, the llama.cpp badge, and a blocked Start ──
  {
    const a = await start();
    await a.click("tab-train");
    await settle(8);
    assert.ok(a.of("preflight_train").length >= 1, "pre-flight runs when the app starts");
    // The same arguments as the Train button's estimate.
    assert.strictEqual(a.of("preflight_train")[0].args.params.contextLen, 16);
    assert.strictEqual(a.of("preflight_train")[0].args.params.vocabSize, 512);
    const html = a.$("trPreflight").innerHTML;
    assert.ok(html.includes("64,000") && html.includes("Yes") && html.includes("300 MB of 2.93 GB usable"), html);
    assert.ok(a.$("trForceLabel").classList.contains("hidden"));
    assert.strictEqual(a.$("trStart").disabled, false);
  }
  {
    // The form drives the pre-flight: changing a field re-asks (debounced); vocab 0 says "No".
    let answer = PF_OK;
    const a = await start({ preflight_train: async () => answer });
    const n = a.of("preflight_train").length;
    answer = { ...PF_OK, llamaCpp: { runs: false, reason: EVIL }, problems: [EVIL] };
    a.set("trVocab", "0");
    a.$("trVocab").dispatch("input");
    await new Promise((r) => setTimeout(r, 40)); // the debounce timer runs for real (5 ms in the stub)
    await settle(10);
    assert.ok(a.of("preflight_train").length > n);
    assert.strictEqual(a.of("preflight_train").slice(-1)[0].args.params.vocabSize, 0);
    const html = a.$("trPreflight").innerHTML;
    assert.ok(html.includes("No") && html.includes("&lt;img") && !html.includes("<img"), html);
    assert.ok(html.includes("Fix first"));
  }
  {
    // Over the RAM limit: Start is disabled with the reason, a click sends nothing, the box lifts it.
    const tight = { ...PF_OK, peakRamMb: 5000, ramLimitMb: 3000, ramAvailableMb: 4000, fitsInRam: false };
    const a = await start({ preflight_train: async () => tight, train_slm: async () => TRAINED });
    a.set("trCorpus", "/c.txt"); a.set("trModel", "/m.gguf");
    assert.strictEqual(a.$("trStart").disabled, true, "Start is disabled when the run does not fit");
    assert.ok(!a.$("trForceLabel").classList.contains("hidden"), "the override box appears");
    assert.ok(a.$("trPreflightWhy").textContent.includes("needs about 4.88 GB"), a.$("trPreflightWhy").textContent);
    assert.ok(a.$("trPreflightWhy").textContent.includes("try anyway"));
    await a.click("trStart"); // even a stray click is refused
    assert.strictEqual(a.of("train_slm").length, 0);
    assert.ok(a.$("errTrain").textContent.includes("needs about"));
    a.$("trForce").checked = true;
    a.$("trForce").dispatch("change");
    assert.strictEqual(a.$("trStart").disabled, false, "ticking the box re-enables Start");
    assert.ok(a.$("trPreflightWhy").textContent.startsWith("RAM check skipped"));
    await a.click("trStart");
    assert.strictEqual(a.of("train_slm").length, 1);
    assert.strictEqual(a.of("train_slm")[0].args.params.force, true, "the override reaches the engine");
    assert.strictEqual(a.$("trStart").disabled, false);
    // Unticking blocks it again; a failing pre-flight never blocks (the engine still checks).
    a.$("trForce").checked = false;
    a.$("trForce").dispatch("change");
    assert.strictEqual(a.$("trStart").disabled, true);
    const b = await start({ preflight_train: async () => { throw "boom"; } });
    assert.ok(b.$("trPreflight").textContent.includes("Pre-flight unavailable: boom"));
    assert.strictEqual(b.$("trStart").disabled, false);
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

  // ── Serve tab: validation, the token, start / poll / stop, a server that dies, restore ──
  {
    let status = { running: false, url: null, requests: 0, auth: false };
    const copied = [];
    const a = await start({
      serve_start: async (args) => { status = { running: true, url: "http://127.0.0.1:41234", requests: 0, auth: args.params.token !== null }; return status; },
      serve_stop: async () => { status = { running: false, url: null, requests: status.requests, auth: false }; return status; },
      serve_status: async () => status,
    }, { navigator: { clipboard: { writeText: async (t) => { copied.push(t); } } } });
    await a.click("tab-serve");
    assert.ok(a.$("panel-serve").classList.contains("active"));
    assert.ok(a.$("svStop").classList.contains("hidden") && a.$("svRunning").classList.contains("hidden"));
    assert.strictEqual(a.$("svToken").getAttribute("type"), "password");

    // Invalid forms send nothing.
    await a.click("svStart");
    assert.strictEqual(a.$("errServe").textContent, "Model is required");
    a.set("svModel", "/m/m.gguf"); a.set("svPort", "70000");
    await a.click("svStart");
    assert.strictEqual(a.$("errServe").textContent, "Port must be at most 65535");
    a.set("svPort", "0"); a.set("svMaxBatch", "0");
    await a.click("svStart");
    assert.strictEqual(a.$("errServe").textContent, "Max batch must be an integer ≥ 1");
    a.set("svMaxBatch", "64");
    assert.strictEqual(a.of("serve_start").length, 0);

    // Generate: 48 hex digits, different each time, shown so it can be copied.
    await a.click("svGenToken");
    const tok = a.$("svToken").value;
    assert.match(tok, /^[0-9a-f]{48}$/);
    await a.click("svGenToken");
    assert.notStrictEqual(a.$("svToken").value, tok);
    assert.ok(a.$("svShowToken").checked && a.$("svToken").getAttribute("type") === "text");
    a.$("svShowToken").checked = false; a.$("svShowToken").dispatch("change");
    assert.strictEqual(a.$("svToken").getAttribute("type"), "password");
    a.set("svToken", "0123456789abcdef".repeat(3)); // a fixed one, so the pinned fixture is stable
    const token = a.$("svToken").value;

    // Start: the exact arguments, then the address, locked form and curl example (without the token).
    await a.click("svStart");
    const sent = a.of("serve_start")[0].args.params;
    captured.serve_start = sent;
    assert.deepStrictEqual(sent, { modelPath: "/m/m.gguf", host: "127.0.0.1", port: 0, maxBatch: 64, maxChars: 65536, maxBody: 8388608, maxTokens: 4096, token });
    assert.strictEqual(a.$("errServe").textContent, "");
    assert.strictEqual(a.$("svUrl").textContent, "http://127.0.0.1:41234");
    assert.ok(!a.$("svRunning").classList.contains("hidden") && !a.$("svStop").classList.contains("hidden"));
    assert.ok(a.$("svStart").disabled && a.$("svModel").disabled && a.$("svToken").disabled && a.$("svPort").disabled);
    assert.strictEqual(a.$("svStatus").textContent, "running");
    const curl = a.$("svCurl").textContent;
    assert.ok(curl.startsWith("curl -s -H 'Content-Type: application/json' -H \"Authorization: Bearer $FERRUM_TOKEN\" http://127.0.0.1:41234/v1/generate -d '"), curl);
    assert.ok(!curl.includes(token), "the token is never written into the example");

    // Log lines arrive as events and are shown as text; the page keeps the last 500.
    a.emit("serve-log", "ferrum-serve: auth ON");
    a.emit("serve-log", EVIL);
    const lines = a.$("svLog").children;
    assert.strictEqual(lines.length, 2);
    assert.strictEqual(lines[1].textContent, EVIL);
    assert.strictEqual(lines[1].children.length, 0, "no markup from the log");
    for (let i = 0; i < 600; i++) a.emit("serve-log", "line " + i);
    assert.strictEqual(a.$("svLog").children.length, 500);
    assert.strictEqual(a.$("svLog").lastChild.textContent, "line 599");

    // The counter follows the poll; copy buttons use the clipboard.
    status = { ...status, requests: 7 };
    await a.intervals[a.intervals.length - 1].f(); await settle(4);
    assert.strictEqual(a.$("svRequests").textContent, "7");
    await a.click("svCopyUrl");
    await a.click("svCopyCurl");
    assert.deepStrictEqual(copied, ["http://127.0.0.1:41234", curl]);

    // Stop: unlocked again, the count reported.
    await a.click("svStop");
    assert.strictEqual(a.of("serve_stop").length, 1);
    assert.ok(a.$("svRunning").classList.contains("hidden") && a.$("svStop").classList.contains("hidden"));
    assert.ok(!a.$("svStart").disabled && !a.$("svModel").disabled && !a.$("svToken").disabled);
    assert.strictEqual(a.$("svStatus").textContent, "stopped (7 requests answered)");

    // No token: the example has no Authorization header, and sends token: null.
    a.set("svToken", "");
    await a.click("svStart");
    assert.strictEqual(a.of("serve_start")[1].args.params.token, null);
    assert.ok(!a.$("svCurl").textContent.includes("Authorization"));

    // The server dies by itself: the next poll unlocks the form and says so.
    status = { running: false, url: "http://127.0.0.1:41234", requests: 9, auth: false };
    await a.intervals[a.intervals.length - 1].f(); await settle(4);
    assert.ok(a.$("svRunning").classList.contains("hidden") && !a.$("svStart").disabled);
    assert.strictEqual(a.$("svStatus").textContent, "stopped");
    assert.ok(a.$("toasts").textContent.includes("stopped on its own"));
  }
  {
    // A refused start shows the structured error and leaves the form usable.
    const a = await start({ serve_start: async () => { throw { code: "busy", message: "a model server is already running", hint: "Stop it first." }; } });
    a.set("svModel", "/m/m.gguf");
    await a.click("svStart");
    assert.strictEqual(a.$("errServe").textContent, "a model server is already running");
    assert.strictEqual(a.$("svStatus").textContent, "error");
    assert.ok(!a.$("svStart").disabled && a.$("svRunning").classList.contains("hidden"));
    // A server that was already running when the page loaded is shown.
    const b = await start({ serve_status: async () => ({ running: true, url: "http://127.0.0.1:8080", requests: 3, auth: true }) });
    assert.strictEqual(b.$("svUrl").textContent, "http://127.0.0.1:8080");
    assert.strictEqual(b.$("svRequests").textContent, "3");
    assert.ok(b.$("svStart").disabled && !b.$("svStop").classList.contains("hidden"));
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
    assert.strictEqual(Object.keys(sorted).length, 12, "every flow captured its arguments");
    if (process.env.UPDATE_FIXTURE) require("node:fs").writeFileSync(file, JSON.stringify(sorted, null, 2) + "\n");
    assert.deepStrictEqual(JSON.parse(require("node:fs").readFileSync(file, "utf8")), sorted, "command_params.fixture.json is stale; run UPDATE_FIXTURE=1 node ui/tests/app_flows.test.js");
  }

  console.log("app_flows.test.js: all assertions passed");
})().catch((e) => { console.error(e); process.exit(1); });
