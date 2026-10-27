// The page frame: tab switching, the Browse buttons, the Stop buttons of the long jobs and the
// terminal dock. Everything it needs comes in through `env`.
"use strict";
(function (global) {
  function create(env) {
    const { $, doc, invoke, dialog, toast } = env;
    const bySel = (s) => Array.from(doc.querySelectorAll(s));

    // ── Tabs: a tablist with roving tabindex; arrows, Home and End move and activate ──
    const tabs = bySel(".tab");
    tabs.forEach((btn) => {
      btn.addEventListener("click", () => {
        tabs.forEach((b) => {
          const on = b === btn;
          b.classList.toggle("active", on);
          b.setAttribute("aria-selected", String(on));
          b.setAttribute("tabindex", on ? "0" : "-1");
        });
        bySel(".panel").forEach((p) => p.classList.remove("active"));
        $("panel-" + btn.dataset.tab).classList.add("active");
      });
      btn.addEventListener("keydown", (e) => {
        const i = global.FerrumA11y.nextTabIndex(e.key, tabs.indexOf(btn), tabs.length);
        if (i < 0) return;
        if (e.preventDefault) e.preventDefault();
        tabs[i].focus();
        tabs[i].click(); // so every click listener (pre-flight, estimates) sees the switch
      });
    });

    // ── File dialogs (Browse buttons) ──
    const asPath = (p) => (typeof p === "string" ? p : p.path || p);
    // Open the dialog `kind` with `opts`, hand the result to `use`; no dialog means type the path.
    async function pick(kind, opts, use) {
      if (!dialog) { toast("File dialog unavailable; type the path manually.", "info"); return; }
      try {
        const picked = await dialog[kind](opts);
        if (picked) use(picked);
      } catch (e) { toast("dialog error: " + e, "error"); }
    }
    const pickOne = (kind, opts) => (targetId) => pick(kind, opts, (f) => { $(targetId).value = asPath(f); });
    const pickOpen = pickOne("open", { multiple: false, directory: false });
    const pickSave = pickOne("save", {});
    const pickDir = pickOne("open", { multiple: false, directory: true });
    bySel(".browse").forEach((b) => b.addEventListener("click", () => pickOpen(b.dataset.target)));
    bySel(".browse-save").forEach((b) => b.addEventListener("click", () => pickSave(b.dataset.target)));
    bySel(".browse-dir").forEach((b) => b.addEventListener("click", () => pickDir(b.dataset.target)));
    // "Add file/folder": append the picked paths as new lines of a list (the Prepare inputs).
    function pickAdd(targetId, directory) {
      return pick("open", { multiple: true, directory }, (picked) => {
        const paths = (Array.isArray(picked) ? picked : [picked]).map(asPath);
        if (!paths.length) return;
        const box = $(targetId);
        box.value = (box.value.trim() ? box.value.replace(/\n*$/, "\n") : "") + paths.join("\n");
      });
    }
    bySel(".browse-add").forEach((b) => b.addEventListener("click", () => pickAdd(b.dataset.target, false)));
    bySel(".browse-add-dir").forEach((b) => b.addEventListener("click", () => pickAdd(b.dataset.target, true)));

    // ── Job control: a Stop button per long job, shown while it runs ──
    const stopAsked = {}; // kind -> the user pressed Stop for the job running now
    function wireStop(btnId, kind) {
      $(btnId).addEventListener("click", async () => {
        stopAsked[kind] = true;
        $(btnId).disabled = true; // asked once; the job ends at its next check
        try { await invoke("cancel_job", { kind }); } catch (e) { toast("Stop failed: " + e, "error"); }
      });
    }
    function jobRunning(btnId, kind, on) {
      stopAsked[kind] = false;
      $(btnId).classList.toggle("hidden", !on);
      $(btnId).disabled = false;
    }
    wireStop("trCancel", "train");
    wireStop("genCancel", "generate");
    wireStop("ggCancel", "run");
    wireStop("ftCancel", "finetune");
    wireStop("exCancel", "export");
    wireStop("dpCancel", "prepare");
    wireStop("rxCancel", "runtime_export");
    wireStop("tkCancel", "tokenize");

    // ── Terminal (bottom dock) ──
    const termOut = $("termOut");
    function termLine(text, cls = "") {
      const div = doc.createElement("div");
      div.className = "line " + cls;
      div.textContent = text;
      termOut.appendChild(div);
      // Keep last ~2000 lines.
      while (termOut.childElementCount > 2000) termOut.removeChild(termOut.firstChild);
      termOut.scrollTop = termOut.scrollHeight;
    }
    $("termClear").addEventListener("click", () => { termOut.textContent = ""; });

    return { jobRunning, stopAsked, termLine };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumShell = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
