// Pure formatting of hardware and estimate objects (as returned by the Rust
// commands `hardware_info` and `estimate_job`) into plain text for tooltips and
// the "Your hardware" line. No DOM, no Tauri; tested by estimate_view.test.js.
// Every number shown is an estimate and is labelled as one.
"use strict";
(function (global) {
  const isNum = (n) => typeof n === "number" && Number.isFinite(n);
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

  // Seconds -> { value, unit } with a friendly unit, e.g. { value: "4.5", unit: "min" }.
  function duration(sec) {
    if (!isNum(sec) || sec < 0) return null;
    if (sec < 1) return { value: "", unit: "under 1 second" };
    const half = (x) => (x < 10 ? String(Math.round(x * 2) / 2) : String(Math.round(x)));
    if (sec < 59.5) return { value: String(Math.round(sec)), unit: Math.round(sec) === 1 ? "second" : "seconds" };
    if (sec < 3570) return { value: half(sec / 60), unit: "min" };
    if (sec < 84600) return { value: half(sec / 3600), unit: "h" };
    if (sec < 364.5 * 86400) return { value: half(sec / 86400), unit: "days" };
    return { value: half(sec / (365 * 86400)), unit: "years" };
  }
  const words = (d) => (d.value ? `${d.value} ${d.unit}` : d.unit);

  function formatDuration(sec) {
    const d = duration(sec);
    if (!d) return "unknown";
    return d.value ? `about ${words(d)}` : d.unit;
  }

  // "2 to 6 min" (same unit) or "50 s to 2 min".
  function formatRange(lo, hi) {
    const a = duration(lo), b = duration(hi);
    if (!a || !b || (!a.value && !b.value)) return "";
    if (!a.value) return `${a.unit} to ${words(b)}`;
    return a.unit === b.unit ? `${a.value} to ${words(b)}` : `${words(a)} to ${words(b)}`;
  }

  function formatMb(mb) {
    if (!isNum(mb)) return "unknown";
    if (mb >= 1024) return (mb / 1024).toFixed(1) + " GB";
    return Math.round(mb) + " MB";
  }

  function gpuText(g) {
    const mem = isNum(g.memoryMb) ? formatMb(g.memoryMb) + " memory" : "memory unknown";
    const cores = isNum(g.cores) ? plural(g.cores, "core") : "cores unknown";
    return `${g.name} (${mem}, ${cores})`;
  }

  // The "Your hardware" line.
  function hardwareLine(hw) {
    if (!hw) return "Detecting your hardware...";
    const cores = isNum(hw.physicalCores) && hw.physicalCores !== hw.logicalCores
      ? `${plural(hw.logicalCores, "CPU thread")} (${plural(hw.physicalCores, "core")})`
      : plural(hw.logicalCores, "CPU thread");
    const ram = isNum(hw.ramTotalMb)
      ? `RAM ${formatMb(hw.ramTotalMb)} total, ${isNum(hw.ramAvailableMb) ? formatMb(hw.ramAvailableMb) + " free" : "free unknown"}`
      : "RAM unknown";
    const gpus = Array.isArray(hw.gpus) && hw.gpus.length
      ? `GPU: ${hw.gpus.map(gpuText).join("; ")} - detected, not used by Ferrum`
      : "no GPU detected (Ferrum is CPU-only)";
    return `${cores} · ${ram} · ${gpus}`;
  }

  // Notes that explain the estimate, without the boilerplate every estimate carries.
  const BOILERPLATE = [/^Rates measured on/, /^Scaled by your Capable/, /^GPU: not used/];
  function userNotes(est) {
    return (est.notes || []).filter((n) => !BOILERPLATE.some((re) => re.test(n)));
  }

  // Estimate -> tooltip sections: [{ label, text, tone? }].
  function estimateSections(est, hw) {
    if (!est) return [];
    const out = [];
    const free = hw && isNum(hw.ramAvailableMb) ? formatMb(hw.ramAvailableMb) + " free RAM" : "free RAM unknown";
    const gpuNote = hw && Array.isArray(hw.gpus) && hw.gpus.length ? "GPU detected but not used by Ferrum" : "GPU not used";
    if (est.kind === "instant") {
      return [{ label: "Estimated time", text: "instant" }];
    }
    if (!isNum(est.seconds)) {
      const why = userNotes(est)[0] || "fill in the fields above first.";
      out.push({ label: "Estimated time", text: `cannot be estimated yet - ${why}`, tone: "warn" });
    } else {
      const t = formatDuration(est.seconds);
      const range = formatRange(est.lowSeconds, est.highSeconds);
      const threads = isNum(est.threadsUsed) && est.threadsUsed > 0 ? ` on your ${plural(est.threadsUsed, "CPU thread")}` : "";
      out.push({
        label: "Estimated time",
        text: `${t}${est.seconds < 1 ? "" : threads}${range && est.seconds >= 1 ? ` (likely ${range})` : ""}; ${free}; ${gpuNote}`,
      });
    }
    if (isNum(est.peakRamMb)) {
      const have = isNum(est.ramAvailableMb) ? `you have ${formatMb(est.ramAvailableMb)} free` : "free RAM unknown";
      const over = est.fitsInRam === false;
      const lim = isNum(est.ramLimitMb) ? ` The app allows this job ${formatMb(est.ramLimitMb)}.` : "";
      out.push({
        label: "Memory",
        text: `needs about ${formatMb(est.peakRamMb)} RAM (${have}).` + (over ? ` That is more than it may use, so it will probably be refused or swap to disk.${lim}` : ""),
        tone: over ? "bad" : undefined,
      });
    }
    const notes = userNotes(est).slice(0, 4);
    const basis = (est.notes || []).find((n) => BOILERPLATE.slice(0, 2).some((re) => re.test(n)));
    if (notes.length || basis) {
      out.push({ label: "Estimate based on", text: [basis, ...notes].filter(Boolean).join(" ") + " This is an estimate, not a promise." });
    }
    return out;
  }

  const api = { duration, formatDuration, formatRange, formatMb, hardwareLine, userNotes, estimateSections };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumEstimateView = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
