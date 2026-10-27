// Helpers every tab module shares: toasts and inline errors, form-field readers and number
// formatting, and the loss chart. `create({ $, doc })` binds the DOM-touching ones to a page;
// the formatters are pure.
"use strict";
(function (global) {
  function fmtBytes(b) {
    const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; let n = b;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(1) + " " + u[i];
  }

  function fmtParams(n) {
    if (!Number.isFinite(n) || n <= 0) return "—";
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + " M";
    if (n >= 1e3) return (n / 1e3).toFixed(0) + " K";
    return n.toFixed(0);
  }

  // Redraw a loss curve on the canvas `id` from `history` (an array of numbers).
  function drawLoss($, id, history) {
    const c = $(id), ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#1e293b"; ctx.fillRect(0, 0, w, h);
    if (history.length < 2) return;
    const max = Math.max(...history), min = Math.min(...history);
    const pad = 8, span = (max - min) || 1;
    ctx.strokeStyle = "#ea580c"; ctx.lineWidth = 2; ctx.beginPath();
    history.forEach((v, i) => {
      const x = pad + (i / (history.length - 1)) * (w - 2 * pad);
      const y = pad + (1 - (v - min) / span) * (h - 2 * pad);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#94a3b8"; ctx.font = "11px monospace";
    ctx.fillText("loss " + max.toFixed(4), pad, 14);
    ctx.fillText(min.toFixed(4), pad, h - 4);
  }

  function create({ $, doc }) {
    // ── Toasts & errors ──
    function toast(msg, kind = "info") {
      const el = doc.createElement("div");
      el.className = "toast " + (kind === "error" ? "err" : kind === "ok" ? "ok" : "");
      el.textContent = msg;
      $("toasts").appendChild(el);
      setTimeout(() => el.remove(), 5000);
    }
    function setErr(id, msg) { const e = $(id); if (e) e.textContent = msg || ""; }
    function clearErr(id) { setErr(id, ""); }

    // ── Number/validation helpers ──
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
    return { toast, setErr, clearErr, num, int, reqInt, reqNum, reqStr, samplerParams };
  }

  const api = { create, fmtBytes, fmtParams, drawLoss };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumCommon = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
