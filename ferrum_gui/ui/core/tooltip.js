// Accessible tooltips for the Ferrum UI. No dependencies; loaded as a plain
// <script> in the browser and `require`d by the Node tests.
//
// - Shows on hover (after a short delay) and immediately on keyboard focus;
//   hides on mouse-leave, blur, scrolling, resizing, and Escape (WCAG 1.4.13).
// - The visible bubble is one shared <div role="tooltip">. Every target also
//   gets a visually hidden description node and `aria-describedby`, so screen
//   readers always have the (live-updated) text, not only while it is shown.
// - Content comes from a help.js entry (`contentFromHelp`) or from a plain
//   `data-tip` attribute (`contentFromTip`, no title). Every tip has at least
//   MIN_WORDS words (`wordCount`); tooltips_guard.test.js enforces it.
// - All text is set with textContent, never innerHTML.
// - CSP: the page forbids inline styles. Only CSSOM (`element.style.left = ...`)
//   is used for positioning; it is allowed, unlike a style="" attribute.
// - `placeTooltip` is pure and unit-tested.
"use strict";
(function (global) {
  const HOVER_MS = 350;
  const GAP = 8;
  const MARGIN = 8;
  const MIN_WORDS = 30;

  // Words in `text`: its whitespace-separated pieces. The one counter the guard test uses.
  const wordCount = (text) => String(text == null ? "" : text).split(/\s+/).filter(Boolean).length;

  // A `data-tip` attribute value -> renderable content (null when empty). `current`, when
  // given and not empty, is the element's live text, appended as a "Now" section.
  function contentFromTip(text, current) {
    const t = String(text == null ? "" : text).trim();
    if (!t) return null;
    const now = String(current == null ? "" : current).trim();
    return { title: "", sections: now ? [{ label: null, text: t }, { label: "Now", text: now }] : [{ label: null, text: t }] };
  }

  // Where to put a tooltip of size `tip` next to `anchor`, inside `vp`.
  // Prefers below the anchor, then above, then the side with more room; always
  // clamped so the bubble stays inside the viewport (never negative).
  function placeTooltip(anchor, tip, vp, gap = GAP, margin = MARGIN) {
    const maxLeft = Math.max(margin, vp.width - tip.width - margin);
    const left = Math.min(Math.max(anchor.left, margin), maxLeft);
    const below = vp.height - anchor.bottom - gap - margin;
    const above = anchor.top - gap - margin;
    let placement = "below";
    if (tip.height > below && (tip.height <= above || above > below)) placement = "above";
    let top = placement === "below" ? anchor.bottom + gap : anchor.top - gap - tip.height;
    const maxTop = Math.max(margin, vp.height - tip.height - margin);
    top = Math.min(Math.max(top, margin), maxTop);
    return { left, top, placement };
  }

  // Help entry (field or button) -> renderable content.
  function contentFromHelp(entry) {
    if (!entry) return null;
    const sections = [];
    if (typeof entry.does === "string") {
      sections.push({ label: "What it does", text: entry.does });
      if (entry.how) sections.push({ label: "How it works", text: entry.how });
    } else {
      sections.push({ label: "What it is", text: entry.what });
      sections.push({ label: "How it works", text: entry.how });
    }
    if (entry.beginner) sections.push({ label: "New to AI?", text: entry.beginner });
    return { title: entry.title, sections };
  }

  // Plain-text rendering for screen readers (and tests): one sentence per part.
  function plainText(content) {
    if (!content) return "";
    const stop = (t) => (/[.!?]$/.test(t) ? t : t + ".");
    const parts = content.title ? [stop(content.title)] : [];
    for (const s of content.sections) parts.push(stop(s.label ? `${s.label}: ${s.text}` : s.text));
    return parts.join(" ");
  }

  function createTooltips(env = {}) {
    const doc = env.document || document;
    const setT = env.setTimeout || setTimeout;
    const clearT = env.clearTimeout || clearTimeout;
    const viewport = env.viewport || (() => ({ width: window.innerWidth, height: window.innerHeight }));
    const win = env.window || (typeof window !== "undefined" ? window : null);

    const bubble = doc.createElement("div");
    bubble.id = "ferrum-tooltip";
    bubble.className = "tip hidden";
    bubble.setAttribute("role", "tooltip");
    bubble.setAttribute("aria-hidden", "true"); // the sr-only description carries the text
    const store = doc.createElement("div"); // visually hidden descriptions
    store.className = "sr-only";
    store.id = "ferrum-descriptions";
    doc.body.appendChild(bubble);
    doc.body.appendChild(store);

    const targets = new Map(); // element -> { getContent, descNode }
    let current = null; // element whose tooltip is visible
    let dismissed = null; // element dismissed with Esc until the pointer/focus leaves
    let timer = null;
    let seq = 0;

    function render(content) {
      bubble.textContent = "";
      if (content.title) {
        const h = doc.createElement("div");
        h.className = "tip-title";
        h.textContent = content.title;
        bubble.appendChild(h);
      }
      for (const s of content.sections) {
        const p = doc.createElement("p");
        p.className = "tip-sec" + (s.tone ? " tip-" + s.tone : "");
        if (s.label) {
          const b = doc.createElement("b");
          b.textContent = s.label + ": ";
          p.appendChild(b);
        }
        p.appendChild(doc.createTextNode(s.text));
        bubble.appendChild(p);
      }
    }

    function position(el) {
      const r = el.getBoundingClientRect();
      const t = bubble.getBoundingClientRect();
      const pos = placeTooltip(r, { width: t.width, height: t.height }, viewport());
      bubble.style.left = pos.left + "px";
      bubble.style.top = pos.top + "px";
      bubble.setAttribute("data-placement", pos.placement);
    }

    // A modal <dialog> lives in the browser's top layer, above everything outside it whatever the
    // z-index, so a bubble kept in <body> would render behind it. Inside an open dialog the
    // bubble is moved into that dialog (position: fixed still places it by viewport
    // coordinates); everywhere else it lives in <body>.
    function placeInTopLayer(el) {
      const dlg = el.closest ? el.closest("dialog") : null;
      const host = dlg && dlg.open ? dlg : doc.body;
      if (bubble.parentNode !== host) host.appendChild(bubble);
    }

    function show(el) {
      const t = targets.get(el);
      if (!t || dismissed === el) return;
      const content = t.getContent();
      if (!content) return;
      render(content);
      placeInTopLayer(el);
      bubble.classList.remove("hidden");
      current = el;
      position(el);
    }

    function hide() {
      clearT(timer);
      timer = null;
      current = null;
      bubble.classList.add("hidden");
    }

    function leave(el) {
      if (dismissed === el) dismissed = null;
      if (current === el || timer) hide();
    }

    // Update one target's screen-reader description (and the bubble if visible).
    function refreshTarget(el) {
      const t = targets.get(el);
      if (!t) return;
      const content = t.getContent();
      t.descNode.textContent = plainText(content);
      if (current === el && content) {
        render(content);
        position(el);
      }
    }

    function attach(el, getContent) {
      if (targets.has(el)) { targets.get(el).getContent = getContent; refreshTarget(el); return; }
      const descNode = doc.createElement("span");
      descNode.id = "ferrum-desc-" + (++seq);
      store.appendChild(descNode);
      targets.set(el, { getContent, descNode });
      el.setAttribute("aria-describedby", descNode.id);
      descNode.textContent = plainText(getContent());
      const label = el.closest ? el.closest("label") : null;
      for (const node of label && label !== el ? [el, label] : [el]) {
        node.addEventListener("mouseenter", () => {
          clearT(timer);
          timer = setT(() => show(el), HOVER_MS);
        });
        node.addEventListener("mouseleave", () => leave(el));
      }
      el.addEventListener("focus", () => show(el));
      el.addEventListener("blur", () => leave(el));
    }

    function refreshAll() {
      for (const el of targets.keys()) refreshTarget(el);
    }

    doc.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && current) {
        dismissed = current;
        hide();
      }
    });
    if (win && win.addEventListener) {
      win.addEventListener("resize", hide);
      win.addEventListener("scroll", hide, true);
    }

    return {
      attach, refreshTarget, refreshAll, show, hide,
      isVisible: () => current !== null,
      current: () => current,
      bubble, store,
    };
  }

  const api = { placeTooltip, contentFromHelp, contentFromTip, plainText, wordCount, createTooltips, HOVER_MS, MIN_WORDS };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumTooltip = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
