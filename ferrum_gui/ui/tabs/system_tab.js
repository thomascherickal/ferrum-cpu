// The System monitor: CPU, memory and per-core bars polled from the backend.
"use strict";
(function (global) {
  function create(env) {
    const { $, doc, invoke, hasTauri, setBar, num } = env;
    const { fmtBytes } = global.FerrumCommon;
    let sysTimer = null;

    async function pollStats() {
      if (!hasTauri) return;
      try {
        const s = await invoke("system_stats");
        $("miniCpu").textContent = s.cpuTotal.toFixed(0) + "%";
        $("miniMem").textContent = s.memPercent.toFixed(0) + "%";
        $("miniThreads").textContent = s.ferrumThreads;
        $("sysCpu").textContent = s.cpuTotal.toFixed(1) + "%";
        setBar("sysCpuBar", Math.min(100, s.cpuTotal));
        $("sysMem").textContent = s.memPercent.toFixed(1) + "%";
        setBar("sysMemBar", Math.min(100, s.memPercent));
        $("sysMemAbs").textContent = `${fmtBytes(s.memUsed)} / ${fmtBytes(s.memTotal)}`;
        $("sysThreads").textContent = s.ferrumThreads;
        // DOM construction: an inline style="" attribute in markup would need CSP 'unsafe-inline'.
        $("sysCores").replaceChildren(...s.perCore.map((v, i) => {
          const core = doc.createElement("div");
          core.className = "core";
          core.append(`core ${i} — ${v.toFixed(0)}%`);
          const bar = doc.createElement("div");
          bar.className = "cbar";
          const fill = doc.createElement("div");
          fill.style.width = Math.min(100, v) + "%";
          bar.appendChild(fill);
          core.appendChild(bar);
          return core;
        }));
      } catch (e) { /* transient; ignore */ }
    }
    function restartMonitor() {
      if (sysTimer) clearInterval(sysTimer);
      if (!$("sysOn").checked) return;
      const sec = Math.max(0.5, num("sysInterval") || 1.5);
      pollStats();
      sysTimer = setInterval(pollStats, sec * 1000);
    }
    $("sysOn").addEventListener("change", restartMonitor);
    $("sysInterval").addEventListener("change", restartMonitor);

    return { restart: restartMonitor };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumSystemTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
