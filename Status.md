# Project Status

_Last updated: 2026-10-26 · version 0.3.0 · branch `feat/finish-remaining`_

This page is the single place for **where Ferrum stands**: what was just
reorganized, what the engine can realistically do on laptop hardware, what is
implemented and tested, what is broken or missing, what comes next, and how the
project got here. For *how to use* Ferrum, see [Manual.md](Manual.md). For
executed walk-throughs, see [Samples.md](Samples.md). For the overview, see
[ReadMe.md](ReadMe.md).

---

## Contents

1. [Snapshot](#1-snapshot)
2. [The 2026-10-03 restructuring](#2-the-2026-10-03-restructuring)
3. [Capability assessment](#3-capability-assessment)
4. [Implemented and tested](#4-implemented-and-tested)
5. [Known issues](#5-known-issues)
6. [Roadmap](#6-roadmap)
7. [Backward compatibility](#7-backward-compatibility)
8. [Development history](#8-development-history)
9. [Metrics](#9-metrics)

---

## 1. Snapshot

| | |
|---|---|
| **Scope** | A pure-Rust engine for training, running, serving, and evaluating small transformer SLMs on the CPU. Models are GPT-2-architecture GGUF files that run in llama.cpp, Ollama, and LM Studio. Also GGUF import, fine-tuning, and export for Llama/Qwen |
| **Date and branch** | 2026-10-26 · version 0.3.0 · `feat/finish-remaining` (not yet merged to `master`) |
| **Build** | `cargo build --workspace` clean; `cargo clippy --workspace --all-targets -- -D warnings` clean; `ferrum_gui` builds separately (`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test` pass from inside `ferrum_gui/`; `node --check` passes on every `ui/*.js`) |
| **Tests** | **1071 passed, 0 failed, 2 ignored** under `cargo test --workspace --release` (665 `ferrum_core` unit, 93 cross-crate integration in `tests/`, 218 `slm_cli`, 79 `slm_server`, 14 `slm_wasm`, 2 doc-tests; the two ignored are exploration helpers in `tests/tests/probe_gguf.rs`). `slm_cli/tests/tiny_e2e.rs` drives prepare, train, info, eval, generate, resume and the quantized exports through the real binary with a 14,336-parameter model (under half a second for its 11 tests, including graceful stop, resume and the metrics CSV). `slm_cli/tests/tiny_gguf_e2e.rs` does the same for the imported-GGUF path with a synthetic tiny llama (untied and tied head): finetune-gguf, export-gguf `--resume`, `run-gguf` text equality between the exported file and base plus checkpoint, the `finetune-gguf` stop file and resume, and the `export-ollama --modelfile-only` refusal and Modelfile (4 tests, no external tool). `slm_server/tests/tiny_http_e2e.rs` starts the real `ferrum-serve` binary on a library-trained tiny model and checks every route against `ferrum_core`, the `--max-*` flags (zero and malformed values exit 1) and SIGINT shutdown (3 tests; helpers in `slm_server/tests/common/`). `tests/tests/test_tiny_library.rs` covers the library API with a tiny model: bit-identical save/load at f32, int8 and int4, `evaluate` against `score_rolling`, and checkpoint resume at an epoch boundary equal to uninterrupted training (4 tests). The `slm_wasm` unit tests now run `TransformerSLMModel` natively on a library-trained model and match the library's forward pass, the KV-cached path, and `score_rolling` (3 tests; the browser build itself is still not run). Six more cover the `Slm` text API natively on tiny BPE (f32, int8) and character-level models: `encode`, `decode`, `generate` (greedy, sampled with top-k/top-p/repetition penalty, stop strings, zero tokens) and `score` equal the library's results for the same seed, `info`, and the error paths. Streaming: `GenerativeSLM::token_stream` and `stream_next` (new in `ferrum_core`, additive) are the step-wise form of `generate_tokens_with`, which now runs on them; three `ferrum_core` tests check that the pulled fragments equal `generate_tokens` on BPE, character-level, EOS and partial-UTF-8 models across budgets that re-prime the KV cache, with and without stop strings, and two `slm_wasm` tests that `start`/`next_chunk` add up to `generate` and that `cancel` stops. **121** `ferrum_gui` backend tests pass (111 unit, among them 10 headless IPC tests through Tauri's mock runtime, and 10 integration, among them the estimate-versus-wall-time regression tests), and so do the Node tests `ui/stream.test.js`, `ui/escape.test.js`, `ui/help.test.js` (every field and button in `index.html` has a help entry of the required length; the job kinds match the Rust enum) and `ui/tooltip.test.js` (positioning, hover/focus/Esc behaviour, no markup injection), `ui/estimate_view.test.js` (formatting), `ui/live.test.js` (the wiring against the real `index.html` structure and a fake backend: tooltips update when a field or the hardware changes, debouncing, stale responses, errors, no markup injection), `ui/app_smoke.test.js` (the real `app.js` loads with and without Tauri), `ui/app_flows.test.js` (every tab's flow through the real `app.js`: the exact arguments sent, results, errors, progress events, streaming reconciliation, the over-budget dialog; the arguments are pinned to a fixture that Rust deserializes), `ui/commands.test.js` (every `invoke()` name is registered in `lib.rs` and vice versa) and `ui/csp.test.js` (no inline script or style, CSP not loosened), all against a small dependency-free DOM stand-in. Python: **36** pass and **18** skip of 54 (the LightEval-dependent ones, because LightEval is not installed on this machine; 48 passed with it on 2026-10-25, before the suite grew to 53 tests; the run emits no `ResourceWarning`). `slm_cli/tests/tokenize_cli.rs` checks that `tokenize` writes exactly the ids `encode_docs` gives (across the 1 MiB read chunk, for directories, custom separators and the error paths), `slm_cli/tests/token_train_cli.rs` that `train tokens.gguf` gives the same model bytes as `train corpus.txt` (serial, threaded, with knobs, across a resume) and `tests/tests/test_token_training.rs` the same at library level plus validation and refusals. `tests/tests/test_bench_script.rs` runs `scripts/bench_slm.sh` from another directory with a relative corpus path (2 tests). The sibling **Ferrum-ML** project is outside this repository and was not re-run in this pass. |
| **Safety** | `#![forbid(unsafe_code)]` in every crate root, plus a repo-wide scan test |
| **Dependencies** | `ferrum_core`: two pure-Rust Unicode tables, nothing else. `slm_cli`: `sha2`, `serde_json`. `slm_server`: `axum`, `tokio`, `serde`, `serde_json`, `sha2`. `slm_wasm`: `wasm-bindgen`. `ferrum_gui`: Tauri and a few helper crates (`tauri-plugin-dialog`, `sysinfo`, `ureq`, `serde`, `serde_json`). `rayon`, `ndarray`, and `faer` were measured and not adopted ([Manual §2.2.1](Manual.md#221-what-is-used-and-what-was-tried-and-not-kept)) |
| **Commits** | 74 commits on `feat/finish-remaining` ahead of `master` (`git log --oneline master..HEAD`) at `3ba1bc8`, before the documentation commits |

---

## 2. The 2026-10-03 restructuring

An evaluation on 2026-10-03 (summarized in §3) found two kinds of problem: a
training pipeline that could not run at real scale, and a repository carrying a
second product (tabular ML) plus several dead-end features. Both were
addressed the same day. The work after it (GGUF-only models, serving, data
operations, and evaluation) is in §8 and §7.

### 2.1 Training-pipeline fixes

| Defect found | Fix | Evidence |
|---|---|---|
| **Stride-1 windows.** One "epoch" trained on every token about `context_len` times. A 6.9M model on 10M tokens needed about 32 days per epoch. | Configurable window stride, defaulting to `context_len` (non-overlapping) with a random phase each epoch. The legacy positional API keeps stride 1. | A test pins steps per epoch at each stride; on a real corpus one epoch went from about 12,000 steps to 186 |
| **No token or step budget** | `max_tokens` / `--tokens 5e7` stops exactly on budget, mid-epoch if needed, through a new step-capped epoch core (`train_transformer_steps` with a per-step hook) | Tests: budget stops at the exact step across epoch boundaries |
| **BPE trainer was O(merges × corpus)**: 437 KB took 49 s, and 1 GB would take over 30 h | Count each distinct word once and update pair counts incrementally using a lazy max-heap | 437 KB: **49 s → 0.09 s**; 22 MB: 0.33 s; an oracle test proves merges are **identical** to the old algorithm |
| **Tokenizer retrained every run** | `tokenizer_path` / `--tokenizer`: trained and saved once, reused afterwards | Test: second run on different text reuses the saved merges |
| **Schedule, clipping, tying, validation, and checkpoints existed but were unreachable** | All wired into `TransformerConfig`, the CLI (`--cosine --warmup --clip --tie --no-qat --val --patience --checkpoint --checkpoint-every --resume`), and the GUI Train tab | End-to-end: a run killed at 25 s resumed at step 300, reused its tokenizer, stopped exactly at the 1,465-step budget; loss 5.23 → 3.44 |
| **QAT forced on** | `qat` field / `--no-qat` | Test |
| **GUI dropped weight decay and dropout** | Restored, alongside all the new options | GUI tests pass |
| **Checkpoint writes could corrupt on a crash** | Atomic writes (temporary file, then rename) | — |
| **Resume could silently mix tokenizers** | Resume refuses a checkpoint whose vocabulary or context differs | Test |

### 2.2 What moved out

The repository was narrowed to *training and releasing SLMs*. Everything else
went to one of two sibling folders next to the repository.

| Item | Went to | Why |
|---|---|---|
| Tabular MLP stack: `train_cli`, the CSV loader (`CsvDataset`, split helpers), the trainable MLP (`Net`), 10 datasets, 10 pretrained models, download/train scripts, and their tests | **`../Ferrum-ML`** as crate `ferrum_ml` + `tabular_wasm`, depending on `ferrum_core` by path | A working, useful ML demo that has nothing to do with SLMs |
| `TabularModel` browser binding | **Ferrum-ML** `tabular_wasm` | Same |
| One-hot MLP SLM trainer (`GenerativeSLM::train`), embedding-MLP trainer (`train_embedded`), hex-CSV builder, one-hot inference and scoring paths, and their 6 tests | **`../Ferrum-Junk`** (`removed_slm_paths.rs`, `removed_slm_tests.rs`) | Teaching baselines strictly dominated by the transformer, and of no use on their own |
| `web/` launcher (linked to three external repos not present here, ran one-hot models only) | **Ferrum-Junk** `web/` | Broken links, superseded model family |
| GitHub Pages `deploy.yml` (trained tabular models, published `web/`) | **Ferrum-Junk** `github-workflows/` | Deployed only the removed pieces |
| GUI embedded shell (`run_terminal`, `term_cwd`) and the Tabular tab | **Ferrum-Junk** `gui/` (fragments) | Arbitrary command execution from the webview, unrelated to SLMs. The docked panel remains as a read-only engine log. |
| GUI Train-tab method selector (embedded / one-hot) | removed | Only the transformer path remains |
| 26 overlapping Markdown files (root docs, `docs/`, `manual/`, crate READMEs) | Consolidated into **ReadMe.md, Manual.md, Status.md** (and later **Samples.md**). The agent design specs, plans, and work logs are archived in **Ferrum-Junk** `docs-archive/`. | One coherent document set instead of 26 drifting ones |

**Renames:** `tabular_wasm` → **`slm_wasm`** (now only `TransformerSLMModel`);
`ferrum_core::csv` → **`ferrum_core::meta`** (now only `TaskType`,
`ModelMetadata`, `Normalizer`, which the model format needs).

**A latent test bug surfaced.** Ferrum-ML's trained-model tests used a path
relative to the working directory that never resolved under `cargo test`, so
they had **always skipped silently**. Once anchored to the crate directory,
two failed. They probed the housing regressor with every feature set to 0.5
(longitude 0.5, latitude 0.5), far outside the data. They now probe at each
feature's training-range midpoint, and all of them pass. The model itself was fine:
retrained, it predicts $417k against an actual $453k.

---

## 3. Capability assessment

Reference machine: Intel i5-1135G7 (4 cores / 8 threads), 16 GB RAM (15 GiB reported), GeForce
MX330 (2 GB). Ferrum never uses the GPU, so the MX330 contributes nothing.
Compute is the binding limit; RAM is not.

**Measured kernel rate:** 40 GFLOP/s on a 2048² matmul at 8 threads and 14 at one thread (medians of three, 2026-10-26, low-memory build, desktop session running; the tile commit recorded about 50 on a quieter machine). The kernels use no explicit SIMD
(`unsafe` is forbidden). `target-cpu=native` speeds up the large-matmul benchmark but not the reference
model ([Manual §13.3](Manual.md#133-kernel-throughput-synthetic-1b-class-shapes)).

### 3.1 Training from scratch (native transformer)

| Goal | Reachable size |
|---|---|
| Heavily over-trained, minutes to hours | 0.1M–2M parameters |
| Compute-optimal (≈20 tokens/param) in 24 h | ≈4M |
| Compute-optimal in a week | ≈12M |
| Compute-optimal in a month | ≈27M (extrapolated) |
| Fits in RAM but impractical | ≈400M (batch 8, context 256) |

The 24 h and week rows were re-derived on 2026-10-26 from short runs of four configurations ([Manual §6.7](Manual.md#67-sizing-a-run-for-your-machine)): they interpolate between measured sizes, and the month row extrapolates from the largest one (20.6M). The 400M row is the 2026-10-03 estimate and was not re-measured.

### 3.2 External models (GGUF)

| | Verdict on this machine |
|---|---|
| Load 1–3B (supported quants) | Yes: about 0.5 GB int4 plus the f32 embedding; 2–3.5 GB peak during import |
| Run 1B | Yes, slowly: about 6.7 tokens/s int4 decode, about 25 s prefill for 512 tokens (estimated from the GEMM rate) |
| Fine-tune (full AdamW, f32, 16 B/param) | SmolLM2-135M comfortable (≈2.2 GB, about 4M tokens/day); Qwen2.5-0.5B tight (≈8 GB + activations); **1.5B and up: no** |
| Train 1B | **No.** About 16 GB of optimizer state, and about 172 days per 100M tokens at the measured GEMM rate |

### 3.3 Overall verdict

The engineering quality is high. A std-only engine (two Unicode tables), no
`unsafe`, gradient-checked backprop, determinism across thread counts, GGUF
files that run in llama.cpp, Ollama, and LM Studio, and a byte-exact GGUF round
trip are all real and tested. Before 2026-10-03 the training **pipeline** was
the weak point. With the §2.1 fixes, 1–10M-parameter models trained on hundreds
of millions of tokens are realistic on a laptop. A TinyStories-class model
(1–3M parameters, about 470M tokens) is a 2–4 day job: about 1.7 days for 1.3M parameters at the measured rate, and
about 4 days for 3M by interpolation.

---

## 4. Implemented and tested

| Subsystem | Module(s) | State |
|---|---|---|
| Tensors and kernels | `tensor`, `ops` | ✅ Fused cache-tiled `linear_forward` with a 4×8 register tile; packed `qlinear` int8/int4 with column-split decode GEMV |
| Parallelism | `parallel` | ✅ Persistent `std` worker pool; row and column splits; deterministic; a failed worker spawn (e.g. RLIMIT_NPROC) degrades to fewer workers or serial; a panicking kernel is reported and does not kill the pool; serial on wasm32 |
| Layers | `layer/`, `model` | ✅ Linear (+ packed `QWeight`), LayerNorm, Embedding, causal MHA, GELU FFN, `KvCache` |
| Quantization | `quant` | ✅ GGUF Q8_0-exact QAT (32-blocks, f16 scale); int8 per-tensor and per-channel for the imported stack; in-memory int8/int4 (split-half) |
| Transformer training | `train_transformer/` | ✅ Full backprop, Adam/AdamW, data-parallel epochs, dropout, weight tying, clipping, LR schedules, window stride, step-capped epochs with hooks, GGUF checkpoints, graceful stop (`stop::StopCondition`: `--max-seconds`, `<model>.stop`, cancel flag) |
| SLM pipeline | `slm/` | ✅ Config-driven training (budget, validation + early stop, tokenizer reuse, checkpoint/resume, `-v` progress lines); KV-cached generation with half-window re-prime, top-k/top-p/repetition/stop strings, `generate_tokens` and its live callback variant `generate_tokens_with`; streaming (also with `--stop`/`--max-tokens`); `score` / `score_rolling` / `evaluate` |
| Tokenizer | `tokenizer/` | ✅ Byte-level BPE, incremental trainer, GPT-2 regex pre-tokenizer, rank-based encode, `<|endoftext|>` always reserved |
| Data tools | `dataset`, `slm_cli` `prepare` | ✅ Streaming cleaner and splitter with a hash manifest; `clean_corpus`, `corpus_stats`, `validate_for_training` |
| Token files | `token_file/`, `train_transformer/source.rs`, `slm_cli` `tokenize` | ✅ `ferrum-tokens` GGUF (I16 or I32 ids) written in one pass and read by `Read`+`Seek`; `train tokens.gguf` streams windows and is bit-identical to in-memory training (serial and threaded, resume, budget); `info` describes the file |
| Model format | `meta`, `loader/` | ✅ GGUF v3 only: `gpt2` layout (runs in llama.cpp) and the generic `ferrum` layout; F32/Q8_0/Q4_0; bounds-checked; checkpoints, tokenizer files and token files are GGUF too |
| GGUF reader | `gguf/` | ✅ F32/F16/Q8_0/Q8_1/Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K; Q2_K/Q3_K (read-only); streamed `open`; IQ* rejected |
| GGUF tokenizer | `gguf_tokenizer` | ✅ BPE exact; SPM decode exact, encode greedy |
| Llama/Qwen runner | `llm` | ✅ RMSNorm, RoPE, GQA, SwiGLU; cached decode equals full forward |
| Llama/Qwen fine-tuning | `llm_train/` | ✅ Gradient-checked backprop, AdamW, schedules, QAT, dropout, `.flck` checkpoints, threaded epochs |
| GGUF writer | `gguf_write/` | ✅ Byte-exact GGUF v3; encoders verified as inverses of the decoders; atomic writes |
| Chat templates | `chat_format` | ✅ ChatML / Llama-3 / Llama-2 / Zephyr detection, rendering, and stop tokens |
| CLI | `slm_cli` | ✅ train · run · generate · eval (`--json`) · info · prepare · tokenize · run-gguf · finetune-gguf · export-gguf · export-ollama · export-lmstudio; strict flag parsing |
| Serving | `slm_server` | ✅ `ferrum-serve`: health, info, log-likelihood, rolling log-likelihood, generate; limits, one model thread, survives model panics |
| Evaluation | `eval/lighteval/` | ✅ LightEval 0.13.0 model class and tasks (held-out perplexity, TinyStories cloze, BLiMP, ARC-easy) over the server |
| Data operations | `dvc.yaml`, `params.yaml` | ✅ fetch → prepare → train → eval, reproducible; `lighteval` stage opt-in |
| Runtime compatibility | `tests/`, `scripts/verify_runtimes.sh` | ✅ llama.cpp b11521 and Ollama v0.40.2 required in CI; LM Studio 0.4.18 checked locally |
| Browser | `slm_wasm` | ✅ `Slm` text API (`encode`, `decode`, `generate`, `score`, `info`, and pull streaming `start`/`next_chunk`/`cancel`) for BPE and character-level models, equal to the library and CLI for the same seed (native tests); `TransformerSLMModel` for KV-cached token-id inference, sampling helpers and attention maps; builds for wasm32. `slm_wasm/demo/` holds a browser demo (`index.html`, `demo.js`, `style.css`: file picker or drop, prompt, sampling controls, streaming output, token count, tokens/s, errors; strict CSP) and `worker.js`, a Web Worker wrapper. `build_wasm.sh` copies `pkg/` next to it. Node tests without dependencies: `demo/worker.test.js` (the message protocol against a fake `Slm`) and `demo/demo.test.js` (every id `demo.js` uses exists, labels, no inline script, style or handler, CSP not loosened; both run in CI). `slm_wasm/tests/node.rs` is a `wasm-bindgen-test` suite (6 tests on a tiny model built inside the test, untrained because the trainer reads the clock and `wasm32-unknown-unknown` has none: load/info, encode/decode against native ids, `generate` against the library and the texts a native run gave, streaming and cancel, `score`, `JsError` paths) that runs the compiled module under Node; it is wasm32-only (an empty file natively) and CI installs `wasm-bindgen-cli` 0.2.122 for it. **It has compiled for wasm32 here (`cargo test --target wasm32-unknown-unknown --no-run`, clippy `--tests`) but could not be run here, because `wasm-bindgen-cli` is not installed: it ran only on CI.** Its first CI run (commit `76930f3`) failed all six tests, because the trainer calls `Instant::now`, which `wasm32-unknown-unknown` lacks; the model is therefore built without training (`b2bd168`). On that commit's CI run all six pass, including the generated texts and the score taken from a native run, so wasm and native agree on this model. The demo, the generated JS bindings and the real worker were **not** run in a browser or Node here (no `wasm-bindgen` CLI on this machine, and it was not installed to keep to the project directory), so the page is unverified end to end; `node --check` and the two Node tests pass |
| GUI | `ferrum_gui` | ✅ Backend tests and builds. Tabs: Datasets, Train, Generate, Evaluate, Models, GGUF, Fine-tune, Export, System, Capable, plus engine log. |
| CI | `.github/workflows/ci.yml` | ✅ Build/test (`--locked`), rustfmt, clippy, rustdoc (`-D warnings`), JS syntax, MSRV 1.87 `cargo check`, cargo-deny, wasm32 (build, clippy including tests, the demo's Node tests, and `wasm-bindgen-test` under Node), Python tests (without LightEval), GUI fmt/clippy/test (`ferrum_gui/Cargo.lock` is committed), runtimes job. The `rustdoc` step ran locally; the MSRV job and the `runtimes` job have not run on GitHub, and no 1.87 toolchain was installed here, so 1.87 rests on code analysis (`is_multiple_of` needs 1.87, `sha2` 1.85; clippy's `incompatible_msrv` finds no other newer std API). The MSRV covers the shipped crates: two `slm_cli` test helpers use `File::lock` (std 1.89), so `cargo test` needs 1.89. |

---

## 5. Known issues

| Severity | Issue |
|---|---|
| 🟡 | **`train corpus.txt` still reads the whole corpus.** A corpus larger than RAM goes through `tokenize` and a token file instead ([Manual §6.11](Manual.md#611-corpora-larger-than-ram-tokenize-and-token-files)), which streams windows from disk with memory O(batch × context). Its limits: the tokenizer is still trained from text in memory (train it on a sample); `tokenize` holds one document at a time, so a corpus with no separators is one document; the shuffled window order costs 8 bytes per window (small at the default stride, 8 bytes per token at `--stride 1`); `--val` reads the held-out token tail into memory and splits by tokens, not characters. Verified bit-identical to in-memory training and run on a 20 MB corpus; a corpus beyond RAM was not run on this machine. |
| 🟡 | **Training is reproducible per shard count, not per machine.** `--threads 0` (auto, the CLI default) takes the number of data-parallel shards from `FERRUM_NUM_THREADS` or the core count, and a different shard count trains different weights (the reference model's perplexity differs between 1 and 8 shards). The DVC pipeline pins the shard count (`train.threads: 8` in `params.yaml`), so its `dvc.lock` hashes reproduce on any machine; a hand-run `train` at auto reproduces them only on a machine with the same core count. Generation and evaluation are identical at any thread count. |
| 🟡 | **The server handles one request at a time.** One dedicated model thread owns the model, because `GenerativeSLM` is not `Send` or `Sync`. Each matmul still uses every core, but a long batch delays other clients. It binds to loopback by default; an optional bearer token (`FERRUM_TOKEN`) protects every route but `/health`, and a non-loopback `--host` needs one (or `--allow-unauthenticated`). There is no TLS, so the token travels in clear text. |
| 🟡 | **LightEval needs two workarounds on 0.13.0.** `FerrumModel` skips `super().__init__()`, and rolling log-likelihood is not cached. The module is pinned to that version, to `xxhash<4`, and to `datasets<5`. The litellm and TGI backends cannot run log-likelihood tasks. The Python tests run in CI without LightEval (35 pass / 18 skip); the LightEval-dependent ones run only locally. |
| 🟡 | **Quantized files can differ from llama.cpp late in a generation.** llama.cpp quantizes activations in Q8_0 matmuls. f32 files match exactly; Q8_0 files are compared over the first 8 tokens; Q4_0 files are checked only for loading and generating. |
| 🟡 | **Character-level models and Ferrum-ML tabular models do not run in llama.cpp, Ollama, or LM Studio.** They are GGUF, but use a generic Ferrum layout. The export commands refuse them. |
| 🟡 | **Only the verified tool versions are tested.** llama.cpp b11521, Ollama v0.40.2, LM Studio 0.4.18. Runtime tests skip a missing tool, but fail under `FERRUM_REQUIRE_RUNTIMES=1`; the CI `runtimes` job sets it (llama.cpp and Ollama required, `FERRUM_SKIP_LMSTUDIO=1` as the one exemption; LM Studio is required locally before a release). Newer versions may behave differently. The `runtimes` job has not run on GitHub yet, and the LM Studio commands in [Samples §7](Samples.md#7-serving-and-running-your-model-in-llamacpp-ollama-and-lm-studio) were not re-run in the 2026-10-26 documentation pass. |
| 🟡 | **Models are small and CPU-bound.** The reachable sizes in §3 come from short runs of four configurations; the 33.8M configuration could not be re-run because the training memory guard refused it with the RAM available at that moment ([Manual §6.7](Manual.md#67-sizing-a-run-for-your-machine)). |
| 🟡 | **The Capable tab's training estimates are rough.** They ignore activation memory and attention FLOPs, and they scale a single-thread GEMM rate by 8 hyperthreads on 4 physical cores. Treat them as loose upper bounds. |
| 🟡 | **The GUI is not click-tested after the 2026-10-03 changes, or after the GGUF change, or after the tooltip and estimate wiring.** The backend compiles and its tests pass; the tooltip, estimate and hardware-line code is tested against a stub DOM and a fake backend only, never in a real webview, so the real layout (bubble position, the line in the top bar, tooltips of buttons inside a modal dialog, which may render behind it) and the CSP behaviour of `element.style` were not seen. |
| 🟢 | **GUI dataset downloads (G2):** a download streams into `<dest>.part` and is renamed to `dest` only when complete. A body over the 512 MiB cap, an empty body, or a mid-stream error is an error that removes the `.part` file and leaves any existing `dest` untouched (previously an over-cap file was silently truncated and reported as success). There is no sha256 check on the GUI path, because `ferrum_gui` has no hashing dependency (`scripts/fetch.sh` verifies for the CLI path). Tested with in-memory readers; not exercised against a live server. |
| 🟢 | **GUI security (G1):** untrusted strings (GGUF metadata, dataset catalog, errors) are HTML-escaped in `ui/escape.js` (tested by `ui/escape.test.js`) and `tauri.conf.json` sets a Content-Security-Policy (`default-src 'self'`, no inline script or style). The CSP was not exercised in a running webview on this machine, so it is not click-tested either. |
| 🟡 | **GUI time and RAM estimates are calibrated on one machine.** `estimate_job` uses rates measured on the 4-core i5-1135G7 (Manual §10.3 lists them); a machine with a different CPU is scaled only after the Capable check in a release build, and thread scaling beyond 8 threads, AMD compute-unit detection (`/sys/class/kfd`) and non-NVIDIA GPU names were never exercised on real hardware here (fixtures only). The regression test `tests/regression_estimate.rs` compares each estimate with a measured wall time on tiny models (estimated/measured in one run on 2026-10-26: train 1 epoch 0.93, train 30 epochs 1.01 (the source of the small-model floor, so not independent), generate 5.6, evaluate 1.1, clean 2.6, GGUF run 0.53, fine-tune 0.81, export 0.25); it only guards a 0.05x to 20x band. The RAM figure for training is the engine's own guard, which sits about 130 MB under the measured peak for tiny runs and about 1.5x over it for larger ones. |
| 🟢 | **GUI greedy decoding and sampling controls (G7):** the Generate tab takes temperature 0 (exact argmax, like `--temp 0`), top-k, top-p, repetition penalty, a token budget and stop strings (one per line, `\n` `\t` `\\` unescaped); the GGUF tab takes temperature 0. The backend runs `GenerativeSLM::generate_tokens_with` when a token budget or a stop string is set and `generate_stream_with` otherwise, the CLI's own choice, and validates like the CLI (`temp >= 0`, `top-p` in [0, 1], positive penalty). Tested headlessly against `generate_tokens` / `generate_with` on a tiny BPE model (same text, same streamed fragments, seed-independent at temperature 0), through the mock-runtime IPC path, and with the stub-DOM flow tests. GGUF runs have no stop strings (`LlamaModel::generate` stops on token ids only; the CLI has none either). Not click-tested. |
| 🟢 | **GUI job control (G9):** Train, Generate, GGUF, Fine-tune and Export have a Stop button while they run (`cancel_job`; `src/jobs.rs` keeps one slot per kind, so a second start of the same kind fails at once with "a training job is already running"). Train and Fine-tune stop after the current optimizer step through `StopCondition` (`TransformerConfig.stop`, `finetune_epoch_until`) and still save the model and checkpoint (the result carries `stopped`, the UI says "stopped at step N"); Generate and GGUF run stop after the current fragment or token and keep the text so far; Export stops between phases and writes nothing. The GUI fine-tune checkpoint is now written through a temp file and a rename (it was a plain write). Not cancellable: Evaluate (one scoring call; it only refuses a concurrent second run), the encode-and-write phase of an export, and the GGUF load. Generate cancels by unwinding out of the text callback (`resume_unwind`, caught in `run_generation`) because the three decode loops have no stop hook. Tested headlessly: a training and a fine-tune job cancelled mid-run through IPC (stopped result, loadable model, written checkpoint that resumes, no `.tmp` left, second start refused), export cancelled at each phase, generation cancelled on both paths, and the Stop buttons against the stub DOM. Not click-tested. |
| 🟢 | **GUI Evaluate parity (G6):** the Evaluate tab reports bits per byte, token count and seconds, takes an optional test file next to the held-out text (one row each), a lossy-UTF-8 option, and saves the latest run as JSON through the save dialog with the keys of `eval --json`. A Llama/Qwen `.gguf` is detected from its architecture and scored with `LlamaModel::score_rolling` exactly as `eval-gguf` does (BOS prefix, window `min(512, context)`, optional precision, fine-tune checkpoint and the 90% RAM guard); the report is formatted in `ferrum_gui` because `slm_cli`'s lives in that binary crate. Tested against `GenerativeSLM::evaluate` and `score_rolling` on tiny models (identical bits), through IPC with a fine-tune checkpoint, and with the stub-DOM flows. The scoring call itself is not cancellable. Not click-tested. |
| 🟢 | **GUI path hardening (G17):** every command that reads, writes or creates a path from the webview (`save_corpus`, `read_text_file`, train/generate/evaluate, `model_info`, `gguf_info`, `run_gguf`, fine-tune, export, the three downloads) first runs it through `src/paths.rs`: it refuses empty, NUL-containing and over-4096-byte paths, anything under `/dev`, `/proc` or `/sys` (after symlinks and `..` are resolved), and existing directories, FIFOs and device nodes where a file is expected. Typed relative paths still work. This is not a sandbox: ordinary files anywhere the user can write stay reachable, and there is a check-then-use window. The window capability now grants only `core:event:allow-listen`, `core:event:allow-unlisten`, `dialog:allow-open` and `dialog:allow-save` (was `core:default` and `dialog:default`); `withGlobalTauri` stays on because `ui/app.js` reads `window.__TAURI__` and importing the API otherwise needs a bundler. Unit-tested; the trimmed capability file builds, but it was not click-tested in a running window. |
| 🟡 | **A matmul crate is an open decision.** `ndarray` gave +34% to +40% train tokens/s in a trial but would be the engine's first hot-path dependency (and slowed single-token generation in its plain form); `faer` failed the determinism rule. The shipped kernel is std-only with a 4×8 register tile (+20%). See [Manual §2.2.1](Manual.md#221-what-is-used-and-what-was-tried-and-not-kept). |
| 🟢 | Resume may not replay the exact remaining batches of the interrupted epoch (marked `ponytail:` in `slm/training_loop.rs`), so a resumed run is close to, not bit-identical with, an uninterrupted one; it is exact when the last checkpoint falls on an epoch boundary. An interrupt (`SIGINT`) does not save on exit; the last periodic checkpoint is what resumes. |
| 🟢 | A graceful stop checks the clock and the stop file once per optimizer step, so a very slow step delays the stop. A stopped `train` without `--checkpoint` saves the partial model but cannot resume (the message says so). |
| 🟢 | GGUF loads expand Q8_0/Q4_0 weights to f32 in RAM (about 4x/8x the file size; marked `ponytail:` in `loader/decode.rs`) until a packed Q8_0 kernel exists. Loading a Ferrum model parses the file bytes in place (`GenerativeSLM::load` does not copy them). |
| 🟢 | Exporting a model whose tokenizer has no `<|endoftext|>` token (only a hand-built one can lack it; every tokenizer Ferrum trains reserves it) is refused with an error that says to add it with `add_special_token`. Before, the GGUF had no `bos`/`eos` ids and llama.cpp defaulted them to token 11 (byte `0x0B`). The same refusal applies to re-exporting a foreign `gpt2` GGUF whose tokenizer has no EOS. |
| 🟢 | GGUF import of Llama/Qwen models is lossy (dequantize and re-quantize to Ferrum's per-row grid) and not bit-exact to llama.cpp. RoPE type is hard-wired to `Norm` on import (`Neox` for qwen2); no RoPE scaling. |
| 🟢 | No DVC remote is configured. `data/` and `models/` live only on the machine that ran `dvc repro`. |
| 🟢 | `--tie` and LM-head sharing apply to Ferrum's own models only. |

---

## 6. Roadmap

### 6.1 Done since the 2026-10-03 evaluation

Scoring API and fast evaluation, EOS as a special token, strict CLI,
linear-time generation, `prepare` with manifests, DVC pipeline, `ferrum-serve`,
the LightEval module, GGUF-only files, GPT-2-compatible models, the Ollama and
LM Studio export commands, hostile-input hardening, the 4×8 matmul tile, and
splitting the large `ferrum_core` and `slm_cli` files into directory modules.
Of the eight-upgrades branch, **F1** (chat templates), **F2** (sampling knobs
on the GGUF run path), and **F8** (Q2_K/Q3_K read) are done, as is Q5_0/Q5_1
read support. The pre-tokenized token file (`tokenize`, `train tokens.gguf`)
is done as well.

### 6.2 Still open

These are directions, not commitments.

| ID | Feature | State |
|---|---|---|
| F3 | `qwen3` architecture (per-head q/k RMSNorm, explicit head_dim) | Not started |
| F7 | Perplexity evaluation for GGUF models (`eval-gguf`, Evaluate tab) | **`eval-gguf` done** (`LlamaModel::score_rolling`; Manual §9.3); the GUI Evaluate tab scores imported GGUFs too (G6) |
| F5 | Streamed GGUF export (two-pass, checkpoint overlay without loading it whole) | Not started |
| F6 | Safe-SIMD decode (autovectorization-friendly `qlinear` loops) | Not started |
| F4 | GUI CI boot smoke test (`xvfb-run`) | The CI GUI job exists; the boot smoke step is outstanding |

**P0: none open.** The pre-tokenized token file, streamed from disk with EOS
between documents, is done ([Manual §6.11](Manual.md#611-corpora-larger-than-ram-tokenize-and-token-files)).
What it unblocks is **A3**, a full-size DVC profile (the 2 GB TinyStories train
split): the data side is ready (`tokenize`, `train tokens.gguf`); the profile and
the download are not started.

**P1, high payoff**

- **Llama architecture from scratch.** `LlamaModel` (RoPE, RMSNorm, SwiGLU,
  GQA, AdamW, GGUF export) can only be loaded from a file today. Adding
  `LlamaModel::init(cfg, rng)` gives a complete path: train from scratch,
  export GGUF, run in llama.cpp. Every other piece exists. Once it lands, the
  native GPT-2-style stack becomes a candidate for retirement.
- **Faster kernels.** The 4×8 register tile is done. Still open: packed-B
  micro-kernels, no full-weight `Arc` copy per parallel matmul (`ops.rs`), and
  the decision on a matmul crate (§5). Likely 2–5× without `unsafe`, which
  raises every bound in §3. A `-C target-cpu=native` build already runs the
  2048² GEMM about 1.7× faster (32.7 to 54.9 GFLOP/s) but does not speed up the
  reference model, so the small-matmul path is the one to work on
  ([Manual §13.3](Manual.md#133-kernel-throughput-synthetic-1b-class-shapes)).
- **Gradient accumulation** (effective batch not limited by RAM) and
  activation recomputation for longer contexts.
- **Distillation from a GGUF teacher** (logit KD), the best-known quality
  lever below 10M parameters.
- **SFT support**: chat template plus prompt-loss masking for native models.
- **LoRA adapters**, so 0.5B-class fine-tuning fits in 16 GB.
- **Metrics** beyond the per-step CSV: validation loss and perplexity columns, and a
  `--metrics-every N` thinning for very long runs (ETA, gradient norm and `--metrics-csv` ship).
- **Size-aware defaults.** `train --preset tiny|small|medium` ships (the §6.7
  starting sizes; explicit flags override). The bare defaults (context 16, embed 32,
  lr 0.01) stay toy settings, because many tests rely on them.
- **A checkpoint-on-interrupt handler**, so `SIGINT` saves before exiting.

**Also open:** exact SPM encode (unigram Viterbi), IQ* quants, finer import
quantization (per-block scales).

---

## 7. Backward compatibility

This release has deliberate breaking changes. Each one is listed with what you
must do.

### 7.1 Breaking changes

| Change | What breaks | What to do |
|---|---|---|
| **FINF is removed.** Every file Ferrum writes is GGUF v3: models, training checkpoints, and tokenizer files. | `.bin` model files (FINF v4 and v5) no longer load. Checkpoints written before this change (magic `FCKP`) fail with a "pre-GGUF checkpoint" error. Old tokenizer files in the `256;257...` text form are rejected. Model file names are now `.gguf`. | Retrain. Committed models in this repo and in Ferrum-ML (10 tabular models) were converted before the reader was deleted. |
| **The Transformer is GPT-2-compatible.** The feed-forward activation changes from ReLU to GELU (tanh approximation), and the LM head loses its bias. | A model trained before this change computes a different function. Its weights do not carry over. The parameter count falls by `vocab` (the removed head bias): the 1,024-token reference config goes from 508,576 to 507,552 parameters (arithmetic from the removed `[vocab]` bias; the earlier build is not runnable any more). Tests that pinned exact losses or generated text were re-derived. | Retrain. There is no converter. |
| **The tokenizer changed.** Pre-tokenization now uses the GPT-2 regex llama.cpp uses (`tokenizer.ggml.pre = "gpt-2"`), and vocabulary and merges are stored in GPT-2 bytes-to-unicode form. | Token ids differ from the old tokenizer for the same text. Tokenizers saved before this change cannot be reused with `--tokenizer`. | Let `train` build a new tokenizer file. |
| **`<|endoftext|>` is always reserved.** BPE training registers it as a special token, counted in the vocabulary size, so every model has a real `eos_token_id`. | The reserved id is counted in `--vocab`. The literal text `<|endoftext|>` now encodes to one id. | None for new models. Older tokenizers had no specials, and `eos_id()` was `None` for them; they no longer load as GGUF tokenizer files anyway. |
| **Export refuses a tokenizer without `<|endoftext|>`.** `save`/`to_gguf` of a BPE transformer whose tokenizer has no EOS token returns an error. | Hand-built tokenizers without EOS exported before (with no `bos`/`eos` ids, which llama.cpp read as token 11). Models trained by Ferrum are unaffected and export byte-identically. | Call `add_special_token("<|endoftext|>")` (or `add_standard_special_tokens`) and grow the model vocabulary to match. |
| **QAT simulates Q8_0 exactly.** Fake-quantization works in 32-element blocks with an f16 scale. | Int8 weights differ from the old per-channel int8 snapping, so a model trained before this change gives different int8 numbers. | Retrain. A new int8 file reloads bit-identically. |
| **The CLI is strict.** Each subcommand accepts only its own flags. | Scripts with a misspelled or unsupported flag, a value that does not parse, or a `--resume` path that does not exist now stop with an error. Before, they ran with a silent default. | Fix the command. The error names the flag and lists the valid ones. |
| **`temperature 0` is exact argmax.** The old code clamped it to 0.01. | Greedy output can differ from runs made with the old clamp. | None. `--temp 0` is now reproducible and matches llama.cpp at `--temp 0`. |
| **Perplexity covers every token.** `evaluate()` and training `--val` are rebuilt on `score_rolling`. The old code skipped the first window. | Perplexity, bits/token, and `num_predictions` are not comparable with numbers from before this change. | Re-evaluate old models if you need a comparison, in a Ferrum version that can load them. For new work, compare bits per byte. |
| **Other behaviour changes.** | `info` prints the parameter count and labels int4 files. `-v` on `train` prints a progress line every 10 steps instead of per-op logs (the old output is behind `--debug`). With `--val`, a tokenizer trained in the same run uses only the train portion. A resumed run restores the best validation score and reports the correct epoch and step. `--tie` shrinks the parameter count and the file by `vocab × embed`. | None. |
| **Character-level models** raise an error for an unknown character in `score` and `score_rolling`, instead of using id 0. `evaluate()` reports the count. | Callers that relied on silent id 0 now get an error. | Clean the text, or use a BPE model. |
| **The DVC `train` stage pins `--threads`.** `params.yaml` gains `train.threads: 8`, and `dvc.yaml` passes it. | Before, the stage used auto, so `dvc repro` on a machine with another core count trained different weights. The reference weights are unchanged on an 8-thread machine (same `models/slm.gguf` md5 as before). | Set `train.threads` in `params.yaml` if you want another shard count; changing it reruns `train` and changes the weights. |
| **Dependencies were admitted.** The zero-dependency rule is replaced by "dependencies need a measured reason". | `ferrum_core` now builds with a few crates. `deny.toml` and `cargo deny check` cover them. | None. |
| **`TransformerConfig` gained `log_every_steps`** (default 0 = off). | Struct literals already need `..TransformerConfig::default()`. | None. |
| **`TransformerConfig` gained `stop: StopCondition`** (default never stops) for graceful stop (`--max-seconds`, `<model>.stop`, a cancel flag). `train_transformer_steps`' `StepHook` now returns `Result<bool>` (`Ok(true)` = continue). | A struct literal without `..TransformerConfig::default()` no longer compiles (only `slm_cli` and `ferrum_gui` had one; both fixed). A custom `StepHook` must return `Ok(true)`. | Add `..TransformerConfig::default()`; change `Ok(())` to `Ok(true)` in a hook. |
| **`LlamaModel::generate_cancellable`** (new, additive): `generate` plus a `cancelled()` poll before each prompt and sampled token; `generate` is now `generate_cancellable` with `|| false`. | Nothing: results are bit-for-bit those of `generate`. | None. |
| **`MetaValue` gained a `Nums(NumArray)` variant.** The GGUF reader now returns every non-empty numeric or bool metadata array (`tokenizer.ggml.token_type`, `.scores`, ...) as `MetaValue::Nums` at its on-disk width (1-8 bytes per element) instead of a `MetaValue::Array` of scalars (32 bytes per element), so a hostile 400 MB file no longer reaches 12.8 GB of RAM; an array whose count times element size exceeds the bytes left in the file is rejected before anything is allocated. A 100 MB u8 array loaded with `train_transformer info` peaked at 3.34 GB RSS before and 0.30 GB after. | Code that matches `MetaValue::Array` to read a numeric array no longer sees it, and an exhaustive `match` on `MetaValue` needs the new arm. Writing is unchanged: `Nums` and the equivalent `Array` of scalars produce identical bytes, and every GGUF Ferrum writes is byte-identical to before. | Read arrays with `MetaValue::array_len`, `array_get`, or `as_i32_slice`, `as_f32_slice`, `as_u32_slice`, `as_u8_slice`. String arrays and empty arrays are still `MetaValue::Array`. |
| **A fifth GGUF kind, `ferrum-tokens`.** `tokenize` writes a pre-tokenized corpus and `train` accepts it in place of a text corpus (detected by content). | Nothing: it is additive. `info` and `train` recognise it; other programs see an unknown architecture. The GGUF reader now sizes the integer types I16 (25) and I32 (26) (they cannot be dequantized). `GenerativeSLM` gained `evaluate_ids`, `train_transformer_tokens` and `train_transformer_tokens_validated`. | None. |

### 7.2 What still works

- The imported Llama/Qwen stack (`llm`, `llm_train`, `gguf*`, `chat_format`) is
  untouched, and so are `run-gguf`, `finetune-gguf`, and `export-gguf`.
- The positional `train_transformer*` functions keep stride 1 and their API.
  Their numeric results changed for the reasons above (GELU, tokenizer).
- `TransformerConfig` is extended only by fields with defaults. Its default
  stride is `context_len`. Set `window_stride: 1` (`--stride 1`) for full overlap.
- `generate`, `generate_with`, `generate_until`, and `generate_stream` keep their
  signatures. The new token-budget generator is `generate_tokens`; `generate_tokens_with` adds a text callback (new, `generate_tokens` is unchanged).
- The Ferrum GGUF `ferrum.format` is 1.

### 7.3 Earlier changes that still hold

- **Removed APIs** from earlier releases stay removed: `GenerativeSLM::train`,
  `train_with_callback`, `train_embedded`, `train_embedded_with_callback`,
  `slm::build_csv_dataset`, `ferrum_core::{CsvDataset, CsvRow,
  fit_normalizer_with_target, train_val_split, Net, EmbedT, train_epoch,
  accuracy}` (the last six live in `ferrum_ml`), and the `ferrum_core::csv`
  module (now `meta`). Legacy one-hot MLP model files can no longer be read at
  all (they were FINF).
- `TransformerConfig` gained fields. Struct literals need
  `..TransformerConfig::default()`. Its **default stride is `context_len`**, so
  config-driven runs take far fewer steps per epoch than before. Set
  `window_stride: 1` (`--stride 1`) to reproduce an old run.
- `train` and `finetune-gguf` reject a non-positive or non-finite `--lr` with an
  error instead of saving a garbage model.
- **Training stops on a diverged loss.** `train` returns the new
  `InferError::Diverged` ("the loss became NaN at step N; lower --lr or set
  --clip") as soon as a step's loss is NaN or infinite, before the update, and
  writes no model. The probability clamp keeps NaN (`loss::clamp_prob`), so
  `evaluate()` no longer reports a finite perplexity for a NaN model.
- **Char-level `generate` accepts seeds shorter than the context** (library,
  CLI and GUI): the window is left-padded with token 0 (cached path primes a
  partial window), as BPE models already did. Output for seeds at least
  `context_len` long is unchanged.
- **GGUF writers add `general.size_label`** (e.g. `507K`, `124M`, `1.5B`; the
  parameter count, llama.cpp style) so LM Studio shows a size; older files
  without it load unchanged. `train` prints a one-line warning when it writes a
  plain `ferrum`-architecture GGUF (`--vocab 0`) that llama.cpp / Ollama /
  LM Studio cannot load, and `export-ollama` / `export-lmstudio` refuse such
  files with the fix (retrain with a BPE vocabulary).
- **Hostile-input hardening (review pass).** Behaviour changes, all errors instead of
  aborts or silent garbage: `GenerativeSLM::from_bytes` rejects a token-ID model whose
  metadata (`input_dim`, `output_dim`, char table or BPE vocabulary) disagrees with its
  layers; training refuses models above 2^29 parameters or 4096 blocks and runs whose
  estimated peak (weights, optimizer, activations) exceeds the smaller of 16 GB and 3/4 of
  available RAM; imported-model `generate` stops at the model's context window (a prompt
  that fills it is an error); `--temp` (finite, >= 0), `--top-p` ([0, 1]),
  `--rep`/`--repetition-penalty` (> 0), `--dropout` ([0, 1)), `--weight_decay`, `--clip`
  and `--tokens` are validated before anything loads or trains; `--resume` names a missing
  checkpoint up front for `run-gguf`, `finetune-gguf` and `export-gguf`; `prepare` refuses
  a non-regular file named as input (and skips FIFOs/devices inside directories),
  writes `*.part` files and renames them into place only after all input was read (a failed
  run keeps the previous outputs), fails on an empty train split, and names the path and
  operation in I/O errors;
  `ferrum-serve` caps stop strings (16 x 1024 bytes) and in-flight POSTs (32).
  Greedy decoding (`temperature <= 0`) returns the first maximum on exact ties
  (llama.cpp / `torch.argmax`), and `generate` on a char-level model with an empty seed
  starts from token 0 like the BPE path.
- **Deferred-minors pass.** Behaviour changes: `generate` on a char-level model
  errors on a seed character outside the vocabulary (`character 'X' is not in the model
  vocabulary`, as scoring already did) instead of silently using token 0 (a CR in the
  seed is still tolerated); `generate_tokens` checks stop strings against the bytes
  flushed at the token budget or EOS (a stop of `U+FFFD` now fires on a dangling lead
  byte); BPE models build their tokenizer once per model instead of per call
  (bit-identical output). `run-gguf --resume` and `export-gguf --resume` apply the
  checkpoint's weights with `ferrum_core::apply_checkpoint_weights` instead of
  building a `LlamaTrainer` (peak on SmolLM2-135M with a 1.96 GB `.flck`: 5.77 GB ->
  2.70 GB, identical outputs; the printed estimate is now model + checkpoint + half a
  model). `train` and `run` reject a missing `--resume` file even when the model
  already exists. Training output is unchanged bit for bit (head bias and
  export-clone removals).
- **Fine-tune checkpoints no longer carry placeholder biases (2026-10-26).**
  `finetune-gguf` used to train a zero-initialised bias on every projection, but a
  GGUF can only hold the q/k/v biases its source file has, so `export-gguf --resume`
  dropped them and the exported file was not the fine-tuned model (found by the
  end-to-end run, defect D1). Now a bias the source file lacks is not a parameter:
  its slot in the `.flck` is an empty tensor (indices and checkpoint version are
  unchanged), nothing is trained for it, and the export is lossless (the logits
  of `export-gguf --resume` equal those of base + `run-gguf --resume`). Layers with real
  q/k/v biases (qwen2) still train and export them. **Action:** a `.flck` written
  by an earlier build is refused with a message naming the cause ("written by an
  older Ferrum that trained placeholder biases"); re-run `finetune-gguf`. Fine-tune
  numerics change (the removed biases no longer move or enter the gradient-clip
  norm); no golden hash or `dvc.lock` output depends on them.
- **Tied-embedding fine-tunes keep the head tied (2026-10-26).** Verified bug, found
  by a synthetic llama GGUF without `output.weight` (like SmolLM2): the loader built a
  separate LM head, `finetune-gguf` trained it independently of `token_embd`, and
  `export-gguf --resume` (which writes `output.weight` only if the source had it)
  dropped it, so the exported logits differed from base + `run-gguf --resume`. Now
  `LlamaModel::tied_head` marks such a model: the head is not a parameter (its slot in
  the `.flck` is an empty tensor, so indices and checkpoint version are unchanged),
  its gradient is added into `token_embd`, and the head is re-derived from the
  embedding after every update, restore, QAT snap and checkpoint load. The export is
  lossless and still has no `output.weight`. Untied models are unchanged. **Action:**
  a `.flck` written by an earlier build for a tied model is refused ("separately
  trained LM head"); re-run `finetune-gguf`. Tied fine-tune numerics change (one shared
  parameter, one fewer `vocab x dim` of optimizer state). Covered by library tests
  only (no CLI test, no real SmolLM2 run).
- **Module splits.** `slm`, `train_transformer`, `gguf`, `gguf_write`, `llm_train`,
  `loader`, `layer`, and `tokenizer` in `ferrum_core` are directories now (and
  `slm_cli` and `slm_server` are split into modules). Public paths are unchanged
  (`mod.rs` re-exports them); model files, checkpoints, and the `dvc.lock` hashes
  are bit-identical.
- **End-to-end defect fixes (D3, D5, D6, D16, D19, D20, D22, D33, D34).** Additive or
  stricter only: `ferrum_core` gains `CheckpointInfo` / `TransformerNet::checkpoint_info`;
  `LlamaTrainer` now returns `InferError::Diverged` (was `DimMismatch("non-finite loss")`)
  for a non-finite loss; model files, tokenizer files and training checkpoints are written
  atomically (temp file + rename) with byte-identical content (`dvc.lock` model md5
  `e387fa48...` reproduced); `--epochs 0`, `--batch 0`, `--threads` above 1024 and
  `ferrum-serve` limits of 0 are errors; output paths are checked before work starts.
  The GUI already rejected 0 epochs and 0 batch and was not changed.
- The GUI `train_slm` command ignores the removed `momentum` field. New fields
  are optional and default sensibly, so older front-ends keep working.

---

## 8. Development history

| Date | Milestone |
|---|---|
| 2026-05-30 | Repository converted into an independent edge generative SLM library; installation and deployment docs |
| 2026-06-11 | Trainable transformer, KV cache, Adam, hardened model parsing; compile and latent-bug fixes |
| 2026-06-12 | `train_transformer` CLI; **byte-level BPE** integrated end to end; matmul parallelized across threads, then a **persistent worker pool**; first benchmarks |
| 2026-06-21–22 | Major progress: **GGUF compatibility**: reader, k-quants, tokenizer import, Llama/Qwen runner, gradient-checked `llm_train` |
| 2026-06-23 | Deep project review (1B feasibility pushback); documentation brought up to date |
| 2026-06-23–24 | **Capable** module: machine micro-benchmarks, capability report, GGUF over-budget warning |
| 2026-07-01 | **Full AdamW GGUF fine-tuning** (CLI + GUI); lint tooling (clippy, rustfmt, cargo-deny, JS syntax) gating CI; SPM/BPE speedups; **GGUF export**: `GgufBuilder`, all nine encoders, quantization policy, `export-gguf` |
| 2026-07-02 | Export hardening and audit fixes; **GUI Export tab**; **Capable v2** (load/train/fine-tune/run parameter ranges) |
| 2026-07-03 | Eight-upgrades design and plan; **F2** sampling knobs; **F1** chat templates |
| 2026-10-03 | Evaluation for SLM training on 2 GB VRAM / 16 GB RAM hardware; **training-pipeline fixes** (stride, budgets, incremental BPE, tokenizer reuse, schedule/clip/tie/validation/checkpoint wiring); **restructuring** into Ferrum / Ferrum-ML / Ferrum-Junk; documentation consolidated into three files |
| 2026-10-09 | Design and plans for data operations, serving, evaluation, speed, and samples; **GPT-2-compatible transformer** (GELU, no head bias), **exact Q8_0 QAT**, llama.cpp's GPT-2 pre-tokenizer, EOS always reserved; public `score`/`score_rolling` and batched evaluation; strict CLI; LightEval module and tasks; `#![forbid(unsafe_code)]` everywhere with a repo-wide scan; documentation drafts |
| 2026-10-10 | **GGUF-only files** (FINF removed; checkpoints and tokenizer files are GGUF); GPT-2 GGUF codec; `ferrum-serve` with its routes and limits; `export-ollama` / `export-lmstudio`; `generate_tokens` with budget, stops, and EOS; runtime compatibility tests against llama.cpp, Ollama, and LM Studio |
| 2026-10-11 | `prepare`, cleaning primitives, `eval --json`, the DVC pipeline; linear-time generation; sampling flags and `--max-tokens`/`--stop` on the CLI; `train --quant`, `info` parameter counts, progress lines, resume fixes |
| 2026-10-24 | Hostile-input hardening and fuzzing of every loader; the **4×8 register-tile matmul** (+17–20% train tokens/s, bit-identical); `rayon`, `ndarray`, `faer` trials recorded; runtime export refusals; Q5_0/Q5_1 read support |
| 2026-10-25 | CLI and server split into modules; CI `runtimes` job and `scripts/verify_runtimes.sh`; `tokio` 1.53.3 (RUSTSEC-2026-0386); LightEval fixes and real-run results; deferred-minors pass; `ferrum_core` large files split into directory modules; coverage passes |
| 2026-10-26 | Documentation: drafts executed and merged; `Samples.md` added; every command in the docs run against the build |

**Design decisions worth keeping** (from the archived specs):

- `#![forbid(unsafe_code)]` stays, by user decision, even where `unsafe` would
  buy speed (mmap, intrinsics).
- The engine stays std-only apart from two Unicode tables. A dependency needs
  a benchmark gain, a determinism test, and a wasm32 build; the matmul crates
  that were measured are not in the tree.
- Chat templates are fingerprinted, not interpreted: there is no Jinja engine,
  and single-turn only until asked.
- Capability estimates are grounded in a live micro-benchmark, not static
  tables.
- GGUF export uses one quantization policy shared by every writer path, and
  the encoders are tested as exact inverses of the decoders.
- Every file Ferrum writes is GGUF, so there is no private format to keep
  compatible.

---

## 9. Metrics

_As of commit `3ba1bc8` (2026-10-25), plus the documentation commits that follow it, excluding `target/`._

| Metric | Value |
|---|---:|
| Workspace crates | 5 (`ferrum_core`, `slm_cli`, `slm_server`, `slm_wasm`, `tests`) + `ferrum_gui` (excluded) |
| `ferrum_core` source files / lines | 95 / 29,651 (including unit tests) |
| `ferrum_core` `pub fn` | 329 |
| Other Rust source (files / lines) | `slm_cli` 25 / 7,152; `slm_server` 11 / 2,408; `slm_wasm` 3 / 1,081; `tests` 17 / 3,600; `ferrum_gui` 6 / 3,252 |
| External dependencies (`ferrum_core`) | 2 (`unicode-general-category`, `unicode-normalization`) |
| `unsafe` blocks | 0 (forbidden) |
| Tests | 1071 pass, 0 fail, 2 ignored (workspace) + 121 `ferrum_gui` + 35 Python (18 skipped) |
| Parameters, reference model | 507,552 |
| Train tokens/s, reference model, 8 threads (start of review / before the tile / now) | 5,895 / 6,141 / 6,725 |
| Train tokens/s, 1 / 2 / 4 / 8 threads (a separate set of runs from the row above) | 2,900 / 6,058 / 6,764 / 7,090 |
| Eval seconds, 30 KB held-out (start of review / before the tile / now) | 138.9 / 1.76 / 1.65 |
| Generate ms per KB of output (start of review / before the tile / now) | 6,873 / 100 / 101 |
| Square GEMM 2048², 8 threads / 1 thread | 40.3 / 13.6 GFLOP/s |
| Markdown documents | 5 (`ReadMe`, `Manual`, `Status`, `Samples`, `glossary`) |

Speed rows: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 4 cores / 8 threads, 2026-10-26, medians of three runs from [Samples §9](Samples.md#9-benchmarking). Line counts are `find ... -name '*.rs' | xargs cat | wc -l` on the working tree, excluding `target/`.
