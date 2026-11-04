// The Serve tab: start and stop the HTTP server for one model, show where it listens, a request
// counter, a copyable curl example and the server's log. The token is read from the field when
// Start is pressed and goes nowhere else: it is not stored, logged or shown back.
"use strict";
(function (global) {
  const MAX_LOG_LINES = 500;
  const POLL_MS = 1000;
  // Every control that only matters before the server starts; locked while it runs.
  const FORM_IDS = ["svModel", "svModelBrowse", "svHost", "svPort", "svMaxBatch", "svMaxChars", "svMaxBody", "svMaxTokens", "svToken", "svGenToken"];

  // A curl command that asks the server for a few tokens. A token is never written into it: the
  // command reads $FERRUM_TOKEN, so the copied text can be pasted or shared safely.
  function curlFor(url, auth) {
    return "curl -s -H 'Content-Type: application/json'" + (auth ? ' -H "Authorization: Bearer $FERRUM_TOKEN"' : "") +
      " " + url + "/v1/generate -d '{\"requests\":[{\"prompt\":\"Once upon a time\",\"max_tokens\":32}]}'";
  }

  // 24 random bytes from the system's secure generator, as 48 hex digits.
  function newToken(cryptoApi) {
    if (!cryptoApi || typeof cryptoApi.getRandomValues !== "function") throw new Error("this window has no secure random generator");
    const bytes = new Uint8Array(24);
    cryptoApi.getRandomValues(bytes);
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  function create(env) {
    const { $, doc, invoke, toast, setErr, clearErr, reqInt, reqStr, errText, busy } = env;
    const E = global.FerrumErrors;
    let running = false;
    let timer = null;

    function logLine(text) {
      const box = $("svLog");
      const div = doc.createElement("div");
      div.className = "line";
      div.textContent = String(text);
      box.appendChild(div);
      while (box.childElementCount > MAX_LOG_LINES) box.removeChild(box.firstChild);
      box.scrollTop = box.scrollHeight;
    }

    // Show a status answer: the form is locked and the address, counter and example appear while
    // the server runs.
    function render(st) {
      running = !!st.running;
      $("svStart").disabled = running;
      $("svStop").classList.toggle("hidden", !running);
      $("svRunning").classList.toggle("hidden", !running);
      for (const id of FORM_IDS) $(id).disabled = running;
      $("svUrl").textContent = running ? st.url : "";
      $("svRequests").textContent = String(st.requests || 0);
      $("svCurl").textContent = running ? curlFor(st.url, st.auth) : "";
    }

    function stopPolling() { if (timer !== null) { clearInterval(timer); timer = null; } }
    async function poll() {
      let st;
      try { st = await invoke("serve_status"); } catch (_) { return; }
      const was = running;
      render(st);
      if (was && !running) {
        stopPolling();
        $("svStatus").textContent = "stopped";
        toast("The server stopped on its own; the log says why", "error");
      }
    }
    function startPolling() { stopPolling(); timer = setInterval(poll, POLL_MS); }

    $("svShowToken").addEventListener("change", () => {
      $("svToken").setAttribute("type", $("svShowToken").checked ? "text" : "password");
    });
    $("svGenToken").addEventListener("click", () => {
      try {
        $("svToken").value = newToken(global.crypto);
        $("svShowToken").checked = true;
        $("svToken").setAttribute("type", "text");
        toast("Token generated. Copy it now: the Studio does not save it.", "ok");
      } catch (e) { toast("Could not generate a token: " + errText(e), "error"); }
    });

    $("svStart").addEventListener("click", async () => {
      clearErr("errServe");
      let params;
      try {
        const port = reqInt("svPort", "Port", 0);
        if (port > 65535) throw new Error("Port must be at most 65535");
        const token = $("svToken").value;
        params = {
          modelPath: reqStr("svModel", "Model"),
          host: reqStr("svHost", "Host"),
          port,
          maxBatch: reqInt("svMaxBatch", "Max batch"),
          maxChars: reqInt("svMaxChars", "Max characters"),
          maxBody: reqInt("svMaxBody", "Max request body"),
          maxTokens: reqInt("svMaxTokens", "Max tokens"),
          token: token === "" ? null : token,
        };
      } catch (e) { setErr("errServe", e); return; }
      $("svStatus").textContent = "loading the model…";
      busy("svSpin", true);
      $("svStart").disabled = true;
      try {
        const st = await invoke("serve_start", { params });
        render(st);
        startPolling();
        $("svStatus").textContent = "running";
        toast("Server started at " + st.url, "ok");
      } catch (e) {
        $("svStatus").textContent = "error";
        setErr("errServe", e);
        toast("Could not start the server: " + errText(e), "error");
      } finally { busy("svSpin", false); $("svStart").disabled = running; }
    });

    $("svStop").addEventListener("click", async () => {
      $("svStop").disabled = true;
      $("svStatus").textContent = "stopping…";
      busy("svSpin", true);
      try {
        const st = await invoke("serve_stop");
        stopPolling();
        render(st);
        $("svStatus").textContent = "stopped (" + st.requests + " requests answered)";
      } catch (e) {
        $("svStatus").textContent = "error";
        setErr("errServe", e);
        toast("Stop failed: " + errText(e), "error");
        await poll();
      } finally { busy("svSpin", false); $("svStop").disabled = false; }
    });

    async function copyText(text, what) {
      const ok = await E.copy(text, { navigator: env.navigator, document: doc });
      toast(ok ? what + " copied" : "Could not copy; select the text and copy it by hand", ok ? "ok" : "error");
    }
    $("svCopyUrl").addEventListener("click", () => copyText($("svUrl").textContent, "URL"));
    $("svCopyCurl").addEventListener("click", () => copyText($("svCurl").textContent, "curl example"));

    // On start-up: show a server that is already running (the page was reloaded under it).
    async function restore() {
      try {
        const st = await invoke("serve_status");
        if (st.running) { render(st); startPolling(); $("svStatus").textContent = "running"; }
      } catch (_) { /* no backend, or nothing to show */ }
    }

    return { onLog: logLine, restore };
  }

  const api = { create, curlFor, newToken, MAX_LOG_LINES };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.FerrumServeTab = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
