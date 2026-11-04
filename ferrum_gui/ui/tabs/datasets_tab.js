// The Datasets tab: download or load raw text, the HuggingFace / Kaggle catalog, clean, save.
// Everything it needs from the page comes in through `env`. Text from the backend only reaches
// the page through textContent, `esc` or the escaped builders in escape.js.
"use strict";
(function (global) {
  function create(env) {
    const { $, doc, invoke, toast, setErr, clearErr, setBar, int, reqStr, getLive } = env;
    const { esc, catalogMeta } = global.FerrumEscape;
    let cleanedCorpus = "";
    let cleanTotal = 0; // bytes of the raw text being cleaned, to turn clean-progress into a fraction
    const fmtMb = (b) => (b / 1e6).toFixed(1) + " MB";
    function utf8Len(str) { // the backend counts bytes, not characters
      let n = 0;
      for (const ch of str) { const c = ch.codePointAt(0); n += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4; }
      return n;
    }

    // One bar serves downloads and Clean; the backend's `download-progress` and `clean-progress`
    // events (wired in app.js) move it, and the command finishing settles it. One job at a time:
    // while one runs, every button that would start another is disabled, so two runs never
    // write over each other's progress.
    const BAR_BUTTONS = ["dsDownload", "dsHfDownload", "dsKgDownload", "dsClean"];
    let catalogButtons = [];
    let barBusy = false;
    function lockBar(on) {
      barBusy = on;
      for (const id of BAR_BUTTONS) $(id).disabled = on;
      for (const b of catalogButtons) b.disabled = on;
    }
    // Returns true so a handler can remember that it owns the bar.
    function barStart(text) { lockBar(true); setBar("dsBar", 0); $("dsProgress").textContent = text; return true; }
    function barEnd(ok, text) { lockBar(false); if (ok) setBar("dsBar", 100); $("dsProgress").textContent = ok ? text : "idle"; }
    function onDownloadProgress(p) {
      if (p.total > 0) setBar("dsBar", Math.min(100, Math.round((p.bytes / p.total) * 100)));
      $("dsProgress").textContent = `${p.id}: ${fmtMb(p.bytes)}` + (p.total > 0 ? ` of ${fmtMb(p.total)}` : "");
    }
    function onCleanProgress(p) {
      if (cleanTotal > 0) setBar("dsBar", Math.min(100, Math.round((p.bytes / cleanTotal) * 100)));
      $("dsProgress").textContent = `cleaning: ${fmtMb(p.bytes)} - ${Number(p.documents).toLocaleString("en-US")} documents`;
    }

    $("dsDownload").addEventListener("click", async () => {
      clearErr("errDsSource");
      let started = false;
      try {
        const url = reqStr("dsUrl", "Source URL");
        const maxBytes = Math.max(1, int("dsMaxMb") || 8) * 1024 * 1024;
        toast("Downloading…"); started = barStart("starting...");
        const text = await invoke("download_text", { url, maxBytes });
        $("dsRaw").value = text;
        barEnd(true, `downloaded ${fmtMb(utf8Len(text))}`);
        toast(`Downloaded ${text.length.toLocaleString()} chars`, "ok");
      } catch (e) { if (started) barEnd(false); setErr("errDsSource", e); }
    });

    $("dsLoadFile").addEventListener("click", async () => {
      clearErr("errDsSource");
      try {
        const path = reqStr("dsFile", "File path");
        const text = await invoke("read_text_file", { path, maxBytes: null });
        $("dsRaw").value = text;
        toast(`Loaded ${text.length.toLocaleString()} chars`, "ok");
      } catch (e) { setErr("errDsSource", e); }
    });

    // ── Dataset catalog (HuggingFace / Kaggle) ──
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
      barEnd(true, `downloaded ${mb} MB from ${res.source}`);
      toast(`Downloaded ${mb} MB from ${res.source} → ${res.path}. Click “Load file”.`, "ok");
    }

    function renderCatalog(list) {
      const box = $("dsCatalog");
      box.textContent = "";
      catalogButtons = [];
      if (!list.length) { box.textContent = "No datasets listed."; return; }
      for (const d of list) {
        const card = doc.createElement("div");
        card.className = "catalog-item";
        const meta = doc.createElement("div");
        meta.innerHTML = catalogMeta(d);
        const btn = doc.createElement("button");
        btn.textContent = "Download";
        btn.dataset.help = "dsCatalogDownload";
        btn.disabled = barBusy;
        catalogButtons.push(btn);
        const live = getLive();
        if (live) live.attach(btn);
        btn.addEventListener("click", async () => {
          clearErr("errDsCatalog");
          let started = false;
          try {
            const dir = destDir();
            toast(`Downloading ${d.name}…`); started = barStart("starting...");
            const res = await invoke("download_dataset", { id: d.id, destDir: dir });
            afterDownload(res);
          } catch (e) { if (started) barEnd(false); setErr("errDsCatalog", e); }
        });
        card.appendChild(meta);
        card.appendChild(btn);
        box.appendChild(card);
      }
    }

    async function loadCatalog() {
      try {
        renderCatalog(await invoke("list_datasets"));
      } catch (e) { setErr("errDsCatalog", e); }
    }

    $("dsCatalogLoad").addEventListener("click", loadCatalog);

    $("dsHfDownload").addEventListener("click", async () => {
      clearErr("errDsCatalog");
      let started = false;
      try {
        const repo = reqStr("dsHfRepo", "HuggingFace repo");
        const file = reqStr("dsHfFile", "File");
        const hfToken = $("dsHfToken").value.trim() || null;
        const dir = destDir();
        toast("Downloading from HuggingFace…"); started = barStart("starting...");
        const res = await invoke("download_hf_file", { repo, file, revision: null, destDir: dir, hfToken });
        afterDownload(res);
      } catch (e) { if (started) barEnd(false); setErr("errDsCatalog", e); }
    });

    $("dsKgDownload").addEventListener("click", async () => {
      clearErr("errDsCatalog");
      let started = false;
      try {
        const ownerSlug = reqStr("dsKgRepo", "Kaggle owner/slug");
        const file = reqStr("dsKgFile", "File");
        const dir = destDir();
        toast("Downloading from Kaggle…"); started = barStart("starting...");
        const res = await invoke("download_kaggle_file", { ownerSlug, file, destDir: dir });
        afterDownload(res);
      } catch (e) { if (started) barEnd(false); setErr("errDsCatalog", e); }
    });

    // Populate the catalog on startup (best-effort).
    loadCatalog();

    $("dsClean").addEventListener("click", async () => {
      clearErr("errDsClean");
      let started = false;
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
        cleanTotal = utf8Len(raw);
        started = barStart("starting...");
        const r = await invoke("clean_text", { raw, opts });
        barEnd(true, `cleaned ${fmtMb(cleanTotal)}`);
        cleanedCorpus = r.cleaned;
        $("dsPreview").value = r.preview;
        $("dsStats").innerHTML =
          `chars <b>${esc(r.chars.toLocaleString())}</b> · words <b>${esc(r.words.toLocaleString())}</b> · ` +
          `lines <b>${esc(r.lines.toLocaleString())}</b> · bytes <b>${esc(r.bytes.toLocaleString())}</b> · ` +
          `char-vocab <b>${esc(r.uniqueChars)}</b>`;
        toast("Cleaned. char-level vocab = " + r.uniqueChars, "ok");
      } catch (e) { if (started) barEnd(false); setErr("errDsClean", e); }
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
      } catch (e) { setErr("errDsSave", e); }
    });

    return { onDownloadProgress, onCleanProgress };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumDatasetsTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
