// Browser demo for slm_wasm: pick a Ferrum .gguf, type a prompt, watch the text stream in.
// The model runs in worker.js (a Web Worker), so the page stays responsive. See Manual 11.3.
const $ = (id) => document.getElementById(id);
const worker = new Worker("worker.js", { type: "module" });
let run = 0; // id of the generation whose output is shown; older ones are ignored
let t0 = 0;

function fail(message) { $("error").textContent = message; $("error").hidden = !message; }
function busy(on) { $("generate").disabled = on || !$("model-info").dataset.loaded; $("stop").disabled = !on; }

function stats(tokens, ms, tail) { // the final readout, from the model's own token count
  const rate = ms > 0 ? ((tokens / ms) * 1000).toFixed(1) : "0.0";
  $("stats").textContent = `${tokens} tokens, ${rate} tokens/s${tail}`;
}

async function load(source) {
  fail("");
  $("model-info").textContent = "Loading...";
  delete $("model-info").dataset.loaded;
  try {
    const bytes = await source.arrayBuffer();
    worker.postMessage({ type: "load", bytes }, [bytes]);
  } catch (e) {
    $("model-info").textContent = "No model loaded.";
    fail(`Cannot read the file: ${e.message}`);
  }
}

worker.onmessage = ({ data: m }) => {
  if (m.type === "ready") {
    const i = JSON.parse(m.info);
    $("model-info").textContent = `${i.params.toLocaleString()} parameters, ${i.tokenizer} tokenizer, vocabulary ${i.vocab}, context ${i.context}, ${i.quant}.`;
    $("model-info").dataset.loaded = "1";
    busy(false);
  } else if (m.type === "error" && (m.id === undefined || m.id === run)) {
    if (!$("model-info").dataset.loaded) $("model-info").textContent = "No model loaded.";
    fail(m.message);
    busy(false);
  } else if (m.id !== run) {
    return;
  } else if (m.type === "chunk") {
    $("output").textContent += m.text;
    $("stats").textContent = `${$("output").textContent.length} characters so far`;
  } else if (m.type === "done") {
    stats(m.tokens, m.ms, m.cancelled ? " (stopped)" : "");
    busy(false);
  }
};
worker.onerror = (e) => { fail(`The worker failed: ${e.message || "see the console"}`); busy(false); };

function number(id, int) {
  const v = Number($(id).value);
  if (!Number.isFinite(v)) throw new Error(`${id} must be a number`);
  return int ? Math.trunc(v) : v;
}

$("generate").onclick = () => {
  fail("");
  try {
    run += 1;
    $("output").textContent = "";
    $("stats").textContent = "";
    t0 = performance.now();
    busy(true);
    worker.postMessage({
      type: "generate", id: run, prompt: $("prompt").value,
      max_tokens: number("max-tokens", true), temperature: number("temperature"),
      top_k: number("top-k", true), top_p: number("top-p"), repetition_penalty: 1,
      seed: number("seed", true) >>> 0, stop: [],
    });
  } catch (e) {
    fail(e.message);
    busy(false);
  }
};
$("stop").onclick = () => worker.postMessage({ type: "cancel" });

$("model-file").onchange = (e) => { if (e.target.files[0]) load(e.target.files[0]); };
const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) load(e.dataTransfer.files[0]);
};

// ?model=URL loads a model from a URL on the same site (or one that allows CORS).
const url = new URLSearchParams(location.search).get("model");
if (url) fetch(url).then((r) => (r.ok ? load(r) : fail(`Cannot fetch ${url}: HTTP ${r.status}`)), (e) => fail(`Cannot fetch ${url}: ${e.message}`));
