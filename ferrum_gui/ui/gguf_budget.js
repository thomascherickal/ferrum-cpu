// The "this model is bigger than this machine can handle" check shared by the GGUF and Export
// tabs: compares a parameter count with the Capable tab's bounds and asks before going on.
"use strict";
(function (global) {
  // Descriptions of every bound `paramCount` exceeds (empty without bounds).
  function crossedBounds(capBounds, paramCount) {
    if (!capBounds || !Number.isFinite(paramCount) || paramCount <= 0) return [];
    const { fmtParams } = global.FerrumCommon;
    const checks = [
      ["loading at int4 (fits in RAM)", capBounds.loadInt4],
      ["inference at int4 (≥ 3 tok/s)", capBounds.inferInt4],
      ["inference at int8 (≥ 3 tok/s)", capBounds.inferInt8],
      ["inference at f32 (≥ 3 tok/s)", capBounds.inferF32],
      ["training, compute-optimal (< 24 h)", capBounds.trainChinchilla],
      ["training, fixed corpus (< 24 h)", capBounds.trainFixed1b],
      ["evaluation pass (< 24 h)", capBounds.testEval],
    ];
    return checks
      .filter(([, bound]) => Number.isFinite(bound) && bound > 0 && paramCount > bound)
      .map(([label, bound]) => `${label}: limit ${fmtParams(bound)}`);
  }

  // `getBounds()` answers the Capable tab's latest bounds (null until it ran).
  function create({ $, getBounds }) {
    const { esc } = global.FerrumEscape;
    const { fmtParams } = global.FerrumCommon;

    const check = (paramCount) => crossedBounds(getBounds(), paramCount);

    // Show the warning dialog; resolves true if the user chooses Proceed, and
    // false on Cancel or any dismissal (Esc/backdrop close the native <dialog>,
    // which fires its "close" event — the single resolution point).
    function confirm(paramCount, crossed) {
      return new Promise((resolve) => {
        // Keyed on the load-check label in crossedBounds: crossing that line
        // means "may not fit in RAM", which is a different risk than "slow".
        const footer = crossed.some((c) => c.includes("fits in RAM"))
          ? "You can still proceed — but this exceeds the fits-in-RAM ceiling: expect heavy swapping or a failed load."
          : "You can still proceed — expect slower than the targets above.";
        $("ggWarnBody").innerHTML =
          `<p>This model has <strong>${esc(fmtParams(paramCount))}</strong> parameters, which exceeds
           the estimated limits of this machine:</p><ul>` +
          crossed.map((c) => `<li>${esc(c)}</li>`).join("") +
          `</ul><p class="muted">${footer}</p>`;
        const dlg = $("ggWarnDialog");
        let proceed = false; // default for Esc / backdrop dismissal
        const onProceed = () => { proceed = true; dlg.close(); };
        const onCancel = () => { proceed = false; dlg.close(); };
        const onClose = () => {
          $("ggWarnProceed").removeEventListener("click", onProceed);
          $("ggWarnCancel").removeEventListener("click", onCancel);
          resolve(proceed);
        };
        $("ggWarnProceed").addEventListener("click", onProceed);
        $("ggWarnCancel").addEventListener("click", onCancel);
        dlg.addEventListener("close", onClose, { once: true });
        dlg.showModal();
      });
    }

    return { check, confirm };
  }

  const api = { create, crossedBounds };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumGgufBudget = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
