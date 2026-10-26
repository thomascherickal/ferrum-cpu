// Web Worker for slm_wasm: loads the model and generates off the page's thread.
//
// Start it as a module worker next to the built `pkg/` (scripts/build_wasm.sh puts it there):
//   const w = new Worker("worker.js", { type: "module" });
//
// Page -> worker
//   { type: "load", bytes: ArrayBuffer }                         load a .gguf (transfer the buffer)
//   { type: "generate", id, prompt, max_tokens, temperature, top_k,
//     top_p, repetition_penalty, seed, stop: [string] }          start; replaces a run in progress
//   { type: "cancel" }                                           stop the run in progress
// Worker -> page
//   { type: "ready", info }          info is the JSON string from Slm.info()
//   { type: "chunk", id, text }      new text, in order; the chunks concatenate to the completion
//   { type: "done", id, tokens, ms, cancelled }   (a replaced run reports cancelled too; check id)
//   { type: "error", message, id? }  id is set for a failed "generate"
// `id` is whatever number or string the page sent with "generate"; use it to ignore a stale run.

import init, { Slm } from "./pkg/slm_wasm.js";

const SLICE_MS = 30; // work this long, then let a queued "cancel" or "generate" message in
let slm = null;
let run = 0; // bumped by every start and cancel; a loop stops when it sees a newer number

const ready = init();
const yieldToMessages = () => new Promise((r) => setTimeout(r, 0));

async function generate(m) {
  const mine = ++run;
  const t0 = performance.now();
  const say = (msg) => self.postMessage({ ...msg, id: m.id });
  slm.start(m.prompt, m.max_tokens, m.temperature, m.top_k, m.top_p,
    m.repetition_penalty, m.seed, m.stop || []);
  let text = "";
  let sliceStart = t0;
  for (let c; run === mine && (c = slm.next_chunk()) !== undefined; ) {
    text += c;
    if (performance.now() - sliceStart >= SLICE_MS) {
      say({ type: "chunk", text });
      text = "";
      await yieldToMessages();
      sliceStart = performance.now();
    }
  }
  const cancelled = run !== mine;
  if (text && !cancelled) say({ type: "chunk", text });
  say({ type: "done", tokens: slm.tokens_generated(), ms: performance.now() - t0, cancelled });
}

self.onmessage = async ({ data: m }) => {
  try {
    await ready;
    if (m.type === "load") {
      slm = new Slm(new Uint8Array(m.bytes));
      self.postMessage({ type: "ready", info: slm.info() });
    } else if (m.type === "generate") {
      if (!slm) throw new Error("load a model first");
      await generate(m);
    } else if (m.type === "cancel") {
      run++;
      if (slm) slm.cancel();
    }
  } catch (e) {
    self.postMessage({ type: "error", message: String((e && e.message) || e), id: m.id });
  }
};
