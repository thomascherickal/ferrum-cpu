// Keyboard and focus helpers: which tab an arrow key moves to, and a modal dialog that remembers
// the element that opened it, keeps Tab inside, closes on Escape and hands focus back on close.
// No dependencies; a plain <script> in the browser and `require`d by the Node tests.
"use strict";
(function (global) {
  // The tab a key moves to in a horizontal tablist of `n` tabs with `i` current, or -1 when the
  // key is not a tab-navigation key (WAI-ARIA tabs pattern).
  function nextTabIndex(key, i, n) {
    switch (key) {
      case "ArrowRight": return (i + 1) % n;
      case "ArrowLeft": return (i - 1 + n) % n;
      case "Home": return 0;
      case "End": return n - 1;
      default: return -1;
    }
  }

  const FOCUSABLE = "button,input,select,textarea,a";

  // Show `dlg` modally. Tab wraps inside it, Escape closes it, and on close focus returns to
  // the element that was focused when it opened (if it is still there).
  function openDialog(dlg, doc) {
    const opener = doc.activeElement || null;
    const focusable = () => Array.from(dlg.querySelectorAll(FOCUSABLE)).filter((el) => !el.disabled);
    const onKey = (e) => {
      if (e.key === "Escape") {
        if (e.preventDefault) e.preventDefault();
        dlg.close();
      } else if (e.key === "Tab") {
        const items = focusable();
        if (!items.length) return;
        const first = items[0], last = items[items.length - 1], at = doc.activeElement;
        if (e.shiftKey && (at === first || !items.includes(at))) { if (e.preventDefault) e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && (at === last || !items.includes(at))) { if (e.preventDefault) e.preventDefault(); first.focus(); }
      }
    };
    const onClose = () => {
      dlg.removeEventListener("keydown", onKey);
      dlg.removeEventListener("close", onClose);
      if (opener && typeof opener.focus === "function" && opener.isConnected !== false) opener.focus();
    };
    dlg.addEventListener("keydown", onKey);
    dlg.addEventListener("close", onClose);
    dlg.showModal();
  }

  const api = { nextTabIndex, openDialog };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumA11y = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
