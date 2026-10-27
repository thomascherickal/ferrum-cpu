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

  // Set a progress bar's fill (`pct` 0-100) and the aria-valuenow of its progressbar host.
  function setBar($, id, pct) {
    const bar = $(id);
    bar.style.width = pct + "%";
    if (bar.parentNode && bar.parentNode.setAttribute) bar.parentNode.setAttribute("aria-valuenow", String(Math.round(pct)));
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

  function create({ $, doc, navigator: nav }) {
    const E = global.FerrumErrors;
    const env = { navigator: nav };
    // ── Toasts & errors ──
    function toast(msg, kind = "info") {
      const el = doc.createElement("div");
      el.className = "toast " + (kind === "error" ? "err" : kind === "ok" ? "ok" : "");
      el.textContent = msg.length > 240 ? msg.slice(0, 239) + "…" : msg;
      // The container is a polite live region; an error is also announced at once and stays
      // longer, so a slow reader does not lose it.
      if (kind === "error") el.setAttribute("role", "alert");
      $("toasts").appendChild(el);
      setTimeout(() => el.remove(), kind === "error" ? 15000 : 5000);
    }
    // An error line: the message (cut when long, with a button for the rest), the hint beneath it
    // and a "Copy details" button. The line itself holds only the message; the rest sits in a
    // companion block right after it, built on first use. `err` is anything a catch block
    // received (errors.js reads it).
    const mores = new Map(); // error line id -> { box, hint, expand, copy, state }
    function moreFor(el) {
      if (mores.has(el.id)) return mores.get(el.id);
      const mk = (tag, cls, label) => {
        const x = doc.createElement(tag);
        x.className = cls;
        if (label) x.textContent = label;
        return x;
      };
      const m = {
        box: mk("div", "errmore hidden"),
        hint: mk("span", "hint-line"),
        expand: mk("button", "ghost small hidden", "Show full message"),
        copy: mk("button", "ghost small", "Copy details"),
        state: null,
      };
      m.hint.setAttribute("aria-live", "polite");
      for (const b of [m.expand, m.copy]) b.setAttribute("type", "button");
      m.expand.addEventListener("click", () => {
        m.state.open = !m.state.open;
        paint(el, m);
      });
      m.copy.addEventListener("click", async () => {
        const ok = await E.copy(E.details(m.state.err), { navigator: env.navigator, document: doc });
        toast(ok ? "Error details copied" : "Could not copy; select the message and copy it by hand", ok ? "ok" : "error");
      });
      m.box.append(m.hint, m.expand, m.copy);
      if (el.parentNode && el.parentNode.insertBefore) el.parentNode.insertBefore(m.box, el.nextSibling);
      mores.set(el.id, m);
      return m;
    }
    function paint(el, m) {
      const n = E.normalize(m.state.err);
      const cut = E.truncate(n.message);
      el.textContent = m.state.open ? n.message : cut.shown;
      m.hint.textContent = n.hint;
      m.expand.classList.toggle("hidden", !cut.truncated);
      m.expand.textContent = m.state.open ? "Show less" : "Show full message";
    }
    function setErr(id, err) {
      const el = $(id);
      if (!el) return;
      const m = moreFor(el);
      if (err === "" || err == null) {
        m.state = null;
        el.textContent = "";
        m.hint.textContent = "";
        m.box.classList.add("hidden");
        return;
      }
      m.state = { err, open: false };
      paint(el, m);
      m.box.classList.remove("hidden");
    }
    function clearErr(id) { setErr(id, ""); }
    // A spinner next to a status line while a long call runs.
    function busy(spinId, on) { $(spinId).classList.toggle("hidden", !on); }

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
    return { toast, setErr, clearErr, busy, errText: E.text, isCancelled: E.isCancelled, setBar: (id, pct) => setBar($, id, pct), num, int, reqInt, reqNum, reqStr, samplerParams };
  }

  const api = { create, fmtBytes, fmtParams, drawLoss, setBar };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumCommon = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
