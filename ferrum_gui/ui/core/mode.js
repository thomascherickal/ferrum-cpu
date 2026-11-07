// Beginner and Expert UI mode controller, model presets, and log-dock collapse.
// Loaded as a plain <script> in the browser and `require`d by the Node tests.
"use strict";
(function (global) {
  const MODE_KEY = "ferrum.ui.mode";
  const DEFAULT_MODE = "beginner";
  const VALID_MODES = ["beginner", "expert"];

  const PRESETS = {
    tiny: {
      trContext: 32,
      trEmbed: 64,
      trHeads: 4,
      trBlocks: 2,
      trHidden: 128,
      trVocab: 512,
      trBatch: 16,
      trEpochs: 50,
      trLr: 0.005,
    },
    small: {
      trContext: 64,
      trEmbed: 128,
      trHeads: 4,
      trBlocks: 4,
      trHidden: 256,
      trVocab: 1024,
      trBatch: 16,
      trEpochs: 50,
      trLr: 0.003,
    },
    medium: {
      trContext: 128,
      trEmbed: 256,
      trHeads: 8,
      trBlocks: 6,
      trHidden: 512,
      trVocab: 2048,
      trBatch: 16,
      trEpochs: 50,
      trLr: 0.001,
    },
  };

  function normalizeMode(mode) {
    if (mode === "advanced") return "expert";
    return VALID_MODES.includes(mode) ? mode : DEFAULT_MODE;
  }

  function applyPreset(doc, name) {
    const p = PRESETS[name];
    if (!p || !doc) return false;
    for (const [id, val] of Object.entries(p)) {
      const el = doc.getElementById(id);
      if (el) {
        el.value = String(val);
        if (typeof el.dispatch === "function") el.dispatch("input");
      }
    }
    return true;
  }

  function create(env) {
    const { $, doc, getStorage } = env;
    let currentMode = DEFAULT_MODE;
    let logCollapsed = false;

    function readStoredMode() {
      try {
        const s = getStorage && getStorage();
        const v = s && s.getItem(MODE_KEY);
        if (v === "advanced") return "expert";
        return VALID_MODES.includes(v) ? v : DEFAULT_MODE;
      } catch (_) {
        return DEFAULT_MODE;
      }
    }

    function writeStoredMode(m) {
      try {
        const s = getStorage && getStorage();
        if (s) s.setItem(MODE_KEY, m);
      } catch (_) {}
    }

    function setLogCollapsed(collapsed) {
      logCollapsed = !!collapsed;
      const term = $("terminal");
      const btn = $("btnToggleLog");
      if (term) term.classList.toggle("collapsed", logCollapsed);
      if (btn) {
        btn.textContent = logCollapsed ? "Show log" : "Hide log";
        btn.setAttribute("aria-expanded", String(!logCollapsed));
      }
    }

    function toggleLog() {
      setLogCollapsed(!logCollapsed);
    }

    function setMode(mode, opts = {}) {
      const next = normalizeMode(mode);
      currentMode = next;
      if (doc && doc.body) {
        doc.body.setAttribute("data-mode", next);
      }
      const sel = $("uiMode");
      if (sel && sel.value !== next) sel.value = next;

      // In beginner mode, if an advanced-only tab is currently active, switch to Datasets.
      if (next === "beginner" && doc) {
        const activeTab = doc.querySelector(".tab.active");
        if (activeTab && activeTab.classList.contains("adv-only")) {
          const fallback = $("tab-datasets");
          if (fallback) fallback.click();
        }
      }

      if (!opts.keepLog) {
        setLogCollapsed(next === "beginner");
      }
      if (!opts.skipSave) {
        writeStoredMode(next);
      }
    }

    function init() {
      const initial = readStoredMode();
      setMode(initial, { skipSave: true });
      const sel = $("uiMode");
      if (sel) {
        sel.addEventListener("change", () => setMode(sel.value));
      }
      const btnLog = $("btnToggleLog");
      if (btnLog) {
        btnLog.addEventListener("click", () => toggleLog());
      }
    }

    return {
      init,
      getMode: () => currentMode,
      setMode,
      isLogCollapsed: () => logCollapsed,
      setLogCollapsed,
      toggleLog,
      applyPreset: (name) => applyPreset(doc, name),
    };
  }

  const api = { MODE_KEY, DEFAULT_MODE, VALID_MODES, PRESETS, normalizeMode, applyPreset, create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumMode = api;
  }
})(typeof window !== "undefined" ? window : globalThis);

