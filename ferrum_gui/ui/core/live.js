// Wires tooltips and live estimates into the page. Everything it needs from the
// outside (document, Tauri `invoke`, timers, the help/tooltip/view modules) is
// passed in, so live.test.js drives it against a stub DOM and a fake backend.
//
//  - Every input/textarea/select/button with a help entry gets a tooltip; any
//    other element (heading, status line, output area...) gets one from its
//    `data-tip` attribute; with `data-tip-live` its current text is appended.
//  - Buttons also show an estimate (time, RAM) for the visible tab. It is
//    recomputed 300 ms after a field changes, when the tab changes, and when
//    the hardware (free RAM) changes, which is polled every 5 s.
//  - Text from the backend only ever reaches the page through textContent.
"use strict";
(function (global) {
  const DEBOUNCE_MS = 300;
  const HARDWARE_MS = 5000;

  // Form values -> `estimate_job` parameters, per button. Unparsable numbers
  // become 0 (never NaN/null: the backend expects numbers).
  const num = (doc, id) => { const e = doc.getElementById(id); const v = e ? parseFloat(e.value) : NaN; return Number.isFinite(v) ? v : 0; };
  const int = (doc, id) => Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.floor(num(doc, id))));
  const str = (doc, id) => { const e = doc.getElementById(id); return e && e.value ? String(e.value).trim() : ""; };
  const lines = (doc, id) => { const e = doc.getElementById(id); return e && e.value ? String(e.value).split("\n").map((s) => s.trim()).filter(Boolean) : []; };
  const len = (doc, id) => { const e = doc.getElementById(id); return e && e.value ? e.value.length : 0; };

  const PARAMS = {
    dsDownload: (d) => ({ maxMb: num(d, "dsMaxMb") }),
    dsLoadFile: (d) => ({ textPath: str(d, "dsFile") }),
    dsClean: (d) => ({ textChars: len(d, "dsRaw") }),
    dpRun: (d) => ({ inputPaths: lines(d, "dpInputs"), outDir: str(d, "dpOut") }),
    tkRun: (d) => ({ inputPaths: lines(d, "tkInput"), outDir: str(d, "tkOut") }),
    trStart: (d) => ({
      corpusPath: str(d, "trCorpus"), tokenizerPath: str(d, "trTokenizer"), contextLen: int(d, "trContext"), embedDim: int(d, "trEmbed"),
      numHeads: int(d, "trHeads"), numBlocks: int(d, "trBlocks"), hiddenDim: int(d, "trHidden"),
      epochs: int(d, "trEpochs"), batchSize: int(d, "trBatch"), vocabSize: int(d, "trVocab"),
      threads: int(d, "trThreads"), windowStride: int(d, "trStride"), maxTokens: int(d, "trTokens"),
      valFraction: num(d, "trVal"),
    }),
    genStart: (d) => ({ modelPath: str(d, "genModel"), numChars: int(d, "genChars"), maxNew: int(d, "genMaxTokens") }),
    evRun: (d) => ({
      modelPath: str(d, "evModel"), textPath: str(d, "evFile"), textChars: len(d, "evText"),
      testPath: str(d, "evTestFile"), quant: str(d, "evQuant"), resumePath: str(d, "evResume"),
    }),
    miInspect: (d) => ({ modelPath: str(d, "miPath") }),
    ggInspect: (d) => ({ modelPath: str(d, "ggPath") }),
    ggRun: (d) => ({
      modelPath: str(d, "ggPath"), quant: str(d, "ggQuant"), maxNew: int(d, "ggMax"),
      promptChars: len(d, "ggPrompt"), resumePath: str(d, "ggResume"),
    }),
    ftRun: (d) => ({
      modelPath: str(d, "ftPath"), corpusPath: str(d, "ftCorpus"), epochs: int(d, "ftEpochs"),
      batchSize: int(d, "ftBatch"), seqLen: int(d, "ftSeq"), threads: int(d, "ftThreads"), resumePath: str(d, "ftResume"),
    }),
    exInspect: (d) => ({ modelPath: str(d, "exPath") }),
    rxPreview: (d) => ({ modelPath: str(d, "rxPath") }),
    rxOllama: (d) => ({ modelPath: str(d, "rxPath"), target: "ollama" }),
    rxLms: (d) => ({ modelPath: str(d, "rxPath"), target: "lmstudio" }),
    rxWriteModelfile: (d) => ({ modelPath: str(d, "rxPath"), target: "modelfile" }),
    svStart: (d) => ({ modelPath: str(d, "svModel") }),
    exExport: (d) => ({ modelPath: str(d, "exPath"), quant: str(d, "exQuant"), resumePath: str(d, "exResume") }),
  };

  function create(env) {
    const { doc, invoke, help, tooltip, view } = env;
    const setT = env.setTimeout || setTimeout;
    const clearT = env.clearTimeout || clearTimeout;
    const setI = env.setInterval || setInterval;
    const tips = tooltip.createTooltips(env.tooltipEnv || { document: doc });

    let hardware = null;
    let hardwareError = null;
    const estimates = {}; // button id -> Estimate | { error } | undefined (pending)
    const requestSeq = {};
    let debounce = null;

    const helpKey = (el) => el.id || (el.dataset && el.dataset.help) || "";

    function content(el) {
      const entry = help.helpFor(helpKey(el));
      const base = tooltip.contentFromHelp(entry);
      if (!base || !help.isButton(entry) || entry.kind === undefined) return base;
      const id = helpKey(el);
      let extra;
      if (entry.kind === "instant") extra = [{ label: "Estimated time", text: "instant" }];
      else if (estimates[id] && estimates[id].error) extra = [{ label: "Estimated time", text: "not available (" + estimates[id].error + ")", tone: "warn" }];
      else if (estimates[id]) extra = view.estimateSections(estimates[id], hardware);
      else extra = [{ label: "Estimated time", text: "calculating..." }];
      const sections = base.sections.slice();
      sections.splice(1, 0, ...extra); // right after "What it does"
      return { title: base.title, sections };
    }

    function attach(el) {
      if (help.helpFor(helpKey(el))) tips.attach(el, () => content(el));
      else if (el.getAttribute("data-tip")) {
        tips.attach(el, () => tooltip.contentFromTip(el.getAttribute("data-tip"), el.hasAttribute("data-tip-live") ? el.textContent : ""));
      }
    }

    function renderHardware() {
      const text = hardware ? view.hardwareLine(hardware) : hardwareError ? "Hardware unavailable (" + hardwareError + ")" : view.hardwareLine(null);
      for (const id of ["hwLine", "hwSystem"]) {
        const e = doc.getElementById(id);
        if (e) e.textContent = text; // #hwLine's tip shows it in full (data-tip-live)
      }
    }

    function visibleButtons() {
      const panel = doc.querySelector(".panel.active");
      return panel ? Array.from(panel.querySelectorAll("button")) : [];
    }

    async function estimateButton(btn) {
      const id = helpKey(btn);
      const entry = help.helpFor(id);
      if (!entry || entry.kind === "instant") return;
      const params = PARAMS[id] ? PARAMS[id](doc) : {};
      const n = (requestSeq[id] = (requestSeq[id] || 0) + 1);
      try {
        const est = await invoke("estimate_job", { kind: entry.kind, params });
        if (requestSeq[id] !== n) return; // a newer request superseded this one
        estimates[id] = est;
      } catch (e) {
        if (requestSeq[id] !== n) return;
        estimates[id] = { error: String(e && e.message ? e.message : e) };
      }
      tips.refreshTarget(btn);
    }

    function estimateVisible() {
      return Promise.all(visibleButtons().map(estimateButton));
    }

    function schedule() {
      clearT(debounce);
      debounce = setT(() => { debounce = null; estimateVisible(); }, DEBOUNCE_MS);
    }

    async function refreshHardware() {
      let before = hardware && hardware.ramAvailableMb;
      try {
        hardware = await invoke("hardware_info");
        hardwareError = null;
      } catch (e) {
        hardwareError = String(e && e.message ? e.message : e);
      }
      renderHardware();
      tips.refreshAll(); // descriptions mention the free RAM
      const now = hardware && hardware.ramAvailableMb;
      // Re-estimate only when something worth re-estimating changed.
      if (before === undefined || before === null || !now || Math.abs(now - before) / Math.max(before, 1) >= 0.01) {
        await estimateVisible();
      }
    }

    function start() {
      Array.from(doc.querySelectorAll("input,textarea,select,button,[data-tip]")).forEach(attach);
      Array.from(doc.querySelectorAll("input,textarea,select")).forEach((f) => {
        f.addEventListener("input", schedule);
        f.addEventListener("change", schedule);
      });
      Array.from(doc.querySelectorAll(".tab")).forEach((t) => t.addEventListener("click", schedule));
      renderHardware();
      const first = refreshHardware();
      setI(refreshHardware, HARDWARE_MS);
      return first;
    }

    return {
      start, attach, schedule, estimateVisible, refreshHardware, tips,
      hardware: () => hardware, estimates: () => estimates,
    };
  }

  const api = { create, PARAMS, DEBOUNCE_MS, HARDWARE_MS };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumLive = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
