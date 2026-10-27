// Command errors in the webview. A failed backend command rejects with `{ code, message, hint }`
// (src/error.rs); anything else that reaches a catch block (a plain string from Tauri itself, an
// Error thrown by a field check, a page running outside Tauri) is accepted too and read the
// same way. Pure functions plus the clipboard helper, no DOM of their own.
"use strict";
(function (global) {
  const LIMIT = 300; // characters shown inline before "Show full message"

  // Any thrown value -> { code, message, hint } (code "error" and an empty hint when unknown).
  function normalize(e) {
    if (e && typeof e === "object" && typeof e.message === "string") {
      return {
        code: typeof e.code === "string" && e.code ? e.code : "error",
        message: e.message,
        hint: typeof e.hint === "string" ? e.hint : "",
      };
    }
    return { code: "error", message: e == null ? "" : String(e), hint: "" };
  }

  const text = (e) => normalize(e).message;
  const isCancelled = (e) => { const n = normalize(e); return n.code === "cancelled" || /cancelled/.test(n.message); };

  // The message cut to `limit` characters (never inside a surrogate pair) with an ellipsis.
  function truncate(message, limit = LIMIT) {
    if (message.length <= limit) return { shown: message, truncated: false };
    let end = limit;
    const c = message.charCodeAt(end - 1);
    if (c >= 0xd800 && c <= 0xdbff) end--; // do not split a pair
    return { shown: message.slice(0, end).trimEnd() + "…", truncated: true };
  }

  // What "Copy details" puts on the clipboard: everything needed to report the problem.
  function details(e, when = new Date()) {
    const n = normalize(e);
    return [
      "Ferrum SLM Studio error",
      "code: " + n.code,
      "message: " + n.message,
      "hint: " + (n.hint || "-"),
      "time: " + when.toISOString(),
    ].join("\n");
  }

  // Copy `str` with the async clipboard API, else a hidden textarea and execCommand("copy").
  // Resolves to whether one of them worked.
  async function copy(str, env) {
    const nav = env.navigator;
    if (nav && nav.clipboard && typeof nav.clipboard.writeText === "function") {
      try { await nav.clipboard.writeText(str); return true; } catch (_) { /* denied: fall back */ }
    }
    const doc = env.document;
    try {
      const ta = doc.createElement("textarea");
      ta.value = str;
      ta.setAttribute("readonly", "");
      ta.className = "sr-only";
      doc.body.appendChild(ta);
      if (ta.select) ta.select();
      const ok = typeof doc.execCommand === "function" && doc.execCommand("copy");
      ta.remove();
      return !!ok;
    } catch (_) { return false; }
  }

  const api = { normalize, text, isCancelled, truncate, details, copy, LIMIT };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumErrors = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
