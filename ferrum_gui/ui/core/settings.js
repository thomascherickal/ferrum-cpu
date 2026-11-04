// Remembers the form between runs: every tab's field values, the folder each file dialog last
// used and the Evaluate table, in localStorage under one versioned key. Storage can be missing,
// blocked, full or hold junk (private windows, cleared data, an older build), so every access is
// in try/catch and the page works the same without it. File contents and secrets are never
// stored: only field values the user typed or picked (paths, numbers, switches, short texts) and
// the evaluation result rows (re-rendered through the escaping builders on load).
"use strict";
(function (global) {
  const KEY = "ferrum-slm-studio:settings";
  const VERSION = 1;
  const SAVE_DELAY_MS = 400;
  const MAX_TEXT = 4096;      // a longer value is not a setting; it is dropped, not cut
  const MAX_EVAL_RUNS = 50;
  const FOLDER_KINDS = ["open", "save", "dir"];

  // Fields that must never be written down: pasted, downloaded or generated text (file contents),
  // the HuggingFace token, and the "try anyway" switches that skip a safety check.
  const NEVER = new Set(["dsRaw", "dsPreview", "evText", "rxModelfile", "dsHfToken"]);
  const isNever = (id) => NEVER.has(id) || /Force$|Token$/.test(id);

  // The fields worth remembering on a page: text, number and select inputs, checkboxes and the
  // remaining textareas, each with an id.
  function savedFields(doc) {
    return Array.from(doc.querySelectorAll("input,select,textarea")).filter((el) => {
      const type = (el.getAttribute("type") || "text").toLowerCase();
      return el.id && !isNever(el.id) && ["text", "number", "checkbox"].includes(el.tagName === "INPUT" ? type : "text");
    });
  }

  const isCheckbox = (el) => el.tagName === "INPUT" && (el.getAttribute("type") || "").toLowerCase() === "checkbox";
  const read = (el) => (isCheckbox(el) ? !!el.checked : String(el.value));

  // Put a stored value into a field if it has the right type (and, for a select, is an option).
  function write(el, v) {
    if (isCheckbox(el)) { if (typeof v === "boolean") el.checked = v; return; }
    if (typeof v !== "string" || v.length > MAX_TEXT) return;
    if (el.tagName === "SELECT" && !Array.from(el.querySelectorAll("option")).some((o) => (o.getAttribute("value") || "") === v)) return;
    el.value = v;
  }

  // Stored text -> a clean state, or the empty state for anything unusable (junk, other version).
  function parse(text) {
    const empty = () => ({ fields: {}, folders: {}, evalRuns: [] });
    let o;
    try { o = JSON.parse(text); } catch (_) { return empty(); }
    if (!o || typeof o !== "object" || Array.isArray(o) || o.version !== VERSION) return empty();
    const plain = (x) => (x && typeof x === "object" && !Array.isArray(x) ? x : {});
    const fields = {};
    for (const [id, v] of Object.entries(plain(o.fields))) {
      if (typeof v === "boolean" || (typeof v === "string" && v.length <= MAX_TEXT)) fields[id] = v;
    }
    const folders = {};
    for (const k of FOLDER_KINDS) {
      const v = plain(o.folders)[k];
      if (typeof v === "string" && v && v.length <= MAX_TEXT) folders[k] = v;
    }
    const evalRuns = (Array.isArray(o.evalRuns) ? o.evalRuns : [])
      .filter((run) => Array.isArray(run) && run.length && run.every((r) => r && typeof r === "object" && !Array.isArray(r)))
      .slice(-MAX_EVAL_RUNS);
    return { fields, folders, evalRuns };
  }

  // The directory part of a path (either separator); "" when there is none.
  function dirOf(p) {
    const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    return i > 0 ? p.slice(0, i) : i === 0 ? p.slice(0, 1) : "";
  }

  // `getStorage()` answers the Storage or throws / returns null when there is none.
  function create({ doc, getStorage, win }) {
    const store = () => { try { return getStorage() || null; } catch (_) { return null; } };
    const s = store();
    let raw = null;
    try { raw = s ? s.getItem(KEY) : null; } catch (_) { /* blocked: start empty */ }
    const state = raw ? parse(raw) : parse("");
    const fields = savedFields(doc);
    const defaults = Object.fromEntries(fields.map((el) => [el.id, read(el)]));
    let timer = null;
    let quiet = false; // true while reset() announces the defaults: that must not save them again

    function save() {
      clearTimeout(timer);
      timer = null;
      const out = { version: VERSION, fields: Object.fromEntries(fields.map((el) => [el.id, read(el)])), folders: state.folders, evalRuns: state.evalRuns };
      try { const st = store(); if (st) st.setItem(KEY, JSON.stringify(out)); } catch (_) { /* full or blocked: keep working */ }
    }
    function scheduleSave() {
      if (quiet) return;
      clearTimeout(timer);
      timer = setTimeout(save, SAVE_DELAY_MS);
    }

    // Apply the stored values and start watching the fields. Call once, after the page is built.
    function restore() {
      for (const el of fields) if (el.id in state.fields) write(el, state.fields[el.id]);
      for (const el of fields) {
        el.addEventListener("input", scheduleSave);
        el.addEventListener("change", scheduleSave);
      }
      if (win && win.addEventListener) win.addEventListener("pagehide", save);
    }

    // Forget everything: storage, folders, the Evaluate table; the form returns to its defaults.
    function reset() {
      try { const st = store(); if (st) st.removeItem(KEY); } catch (_) { /* nothing to remove */ }
      clearTimeout(timer);
      state.folders = {};
      state.evalRuns = [];
      for (const el of fields) {
        const d = defaults[el.id];
        if (isCheckbox(el)) el.checked = d; else el.value = d;
      }
      // Tell the live estimates and the pre-flight that the fields changed.
      quiet = true;
      for (const el of fields) for (const type of ["input", "change"]) {
        if (typeof el.dispatchEvent === "function") el.dispatchEvent(new global.Event(type, { bubbles: true }));
        else if (typeof el.dispatch === "function") el.dispatch(type);
      }
      quiet = false;
    }

    return {
      restore, save, scheduleSave, reset,
      folder: (kind) => state.folders[kind] || undefined,
      setFolder(kind, path) {
        if (!FOLDER_KINDS.includes(kind) || typeof path !== "string" || !path || path.length > MAX_TEXT || state.folders[kind] === path) return;
        state.folders[kind] = path;
        scheduleSave();
      },
      evalRuns: () => state.evalRuns.slice(),
      addEvalRun(rows) { state.evalRuns = state.evalRuns.concat([rows]).slice(-MAX_EVAL_RUNS); scheduleSave(); },
      clearEval() { state.evalRuns = []; scheduleSave(); },
      savedIds: () => fields.map((el) => el.id),
    };
  }

  const api = { create, parse, dirOf, isNever, savedFields, KEY, VERSION };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumSettings = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
