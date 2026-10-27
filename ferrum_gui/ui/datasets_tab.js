// The Datasets tab: download or load raw text, the HuggingFace / Kaggle catalog, clean, save.
// Everything it needs from the page comes in through `env`. Text from the backend only reaches
// the page through textContent, `esc` or the escaped builders in escape.js.
"use strict";
(function (global) {
  function create(env) {
    const { $, doc, invoke, toast, setErr, clearErr, int, reqStr, getLive } = env;
    const { esc, catalogMeta } = global.FerrumEscape;
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
      toast(`Downloaded ${mb} MB from ${res.source} → ${res.path}. Click “Load file”.`, "ok");
    }

    function renderCatalog(list) {
      const box = $("dsCatalog");
      box.textContent = "";
      if (!list.length) { box.textContent = "No datasets listed."; return; }
      for (const d of list) {
        const card = doc.createElement("div");
        card.className = "catalog-item";
        const meta = doc.createElement("div");
        meta.innerHTML = catalogMeta(d);
        const btn = doc.createElement("button");
        btn.textContent = "Download";
        btn.dataset.help = "dsCatalogDownload";
        const live = getLive();
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
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumDatasetsTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
