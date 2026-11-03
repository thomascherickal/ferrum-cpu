<p align="center"><img src="docs/assets/logo.png" width="96" alt="Ferrum logo"></p>

# Project Status

_Last updated: 2026-11-03 · version 0.3.0 · branch `feat/coverage-100`_

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
| **Date and branch** | 2026-11-03 · version 0.3.0 · `feat/coverage-100` (`git rev-list --count master..HEAD` is 32 at `9966e1c`, before the commits of the final fix round; the earlier `feat/finish-remaining` work is already on `master`) |
| **Build** | `cargo build --workspace` clean; `cargo clippy --workspace --all-targets -- -D warnings` clean; `ferrum_gui` builds separately (`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test` pass from inside `ferrum_gui/`; `node --check` passes on every `ui/*.js`) |
| **Tests** | **1211 passed, 0 failed, 2 ignored** under `cargo test --workspace --release` (the final fix round added the text-against-token-file resume refusal test in `slm_cli/tests/token_train_cli.rs` (both directions) and earlier tests of that round; 4 more than the 1204 before the polish commit: the cached-BPE half-character cancel test, the window-boundary re-prime test, the `test_cli_matrix.rs` matcher test and the `prepare()` commit-failure cleanup test; the 1204 were 6 more than the 1198 before the prepare commit marker, Task 2.4: six `slm_cli` tests of `commit_outputs` and of byte-identical reruns; the 1198 were 9 more than the 1189 before the cooperative cancel flag, Task 2.3: eight cancel tests in `ferrum_core`'s `slm::token_stream` and `slm::stream` (cancel before the first token, after k tokens, mid multi-byte character, held-back stop text, and no-flag bit-identity, over every decode loop) and the `SamplingParams` equality test; the GUI's IPC generate test gained a mid-run `cancel_job`, and `ferrum_gui` still has 193 tests; the 1189 were 5 more than the 1184 before corpus identity in checkpoints, Task 2.2: the `corpus_id` checkpoint codec, the two resume-corpus tests in `ferrum_core`, `file_id` in `slm_cli` and `resume_on_other_corpus_exits_1`; the 1184 were 7 more than the 1177 before exact mid-epoch resume, Task 2.1: the `ferrum.ckpt.epoch_rng` codec, the drifting-epoch resume and the old-checkpoint reshuffle in `ferrum_core`, and two mid-epoch resume tests in `tests/tests/test_tiny_library.rs`; the 1177 were 29 more than the 1148 before Task 1.4, which added tests to `slm_cli`, `slm_server` and `tests/tests/test_cli_matrix.rs`; the per-crate split was not recomputed; the two ignored are exploration helpers in `tests/tests/probe_gguf.rs`; `loader::gpt2::tests::loads_foreign_gpt2_gguf_when_present` is not ignored: it asserts that the real GPT-2 file under `.ferrum-dev/` loads and continues like llama.cpp when it exists and that the check fails with a read error when it does not, instead of skipping with a print). `slm_cli/tests/tiny_e2e.rs` drives prepare, train, info, eval, generate, resume and the quantized exports through the real binary with a 14,336-parameter model (0.53-0.54 s for its 11 tests in three runs on 2026-11-02, including graceful stop, resume at an epoch boundary and mid-epoch, and the metrics CSV). `slm_cli/tests/tiny_gguf_e2e.rs` does the same for the imported-GGUF path with a synthetic tiny llama (untied and tied head): finetune-gguf, export-gguf `--resume`, `run-gguf` text equality between the exported file and base plus checkpoint, the `finetune-gguf` stop file and resume, and the `export-ollama --modelfile-only` refusal and Modelfile (4 tests, no external tool). `slm_server/tests/tiny_http_e2e.rs` starts the real `ferrum-serve` binary on a library-trained tiny model and checks every route against `ferrum_core`, the `--max-*` flags (zero and malformed values exit 1) and SIGINT shutdown (3 tests; helpers in `slm_server/tests/common/`). `tests/tests/test_tiny_library.rs` covers the library API with a tiny model: bit-identical save/load at f32, int8 and int4, `evaluate` against `score_rolling`, and checkpoint resume equal to uninterrupted training at an epoch boundary and mid-epoch (serial, `threads = 2`, and from a token file; 6 tests). The logic of `TransformerSLMModel` now lives in `slm_wasm/src/token_model.rs` (`TokenModel`, plain Rust returning `Result<_, String>`); `lib.rs` only forwards each call and maps the `String` to a `JsValue`, so the JS API and error strings are unchanged. Its 14 native tests (plus the tests in `text.rs`) match the library's forward pass and the KV-cached path, check the error strings, the sampling helpers (including empty, NaN and all-zero input; only the empty input used to panic) and the attention-map shape, and the 3 older `lib.rs` tests still cover `score_rolling` through the wrapper; `text.rs` gained tests for the non-text-model refusal and a failing stream step. `sample_from_probs` now compares `r < cumsum`, so a zero-probability leading entry is never drawn at `r == 0` (temperature 0 is then exactly argmax). The browser build itself is still not run. Six more cover the `Slm` text API natively on tiny BPE (f32, int8) and character-level models: `encode`, `decode`, `generate` (greedy, sampled with top-k/top-p/repetition penalty, stop strings, zero tokens) and `score` equal the library's results for the same seed, `info`, and the error paths. Streaming: `GenerativeSLM::token_stream` and `stream_next` (new in `ferrum_core`, additive) are the step-wise form of `generate_tokens_with`, which now runs on them; three `ferrum_core` tests check that the pulled fragments equal `generate_tokens` on BPE, character-level, EOS and partial-UTF-8 models across budgets that re-prime the KV cache, with and without stop strings, and two `slm_wasm` tests that `start`/`next_chunk` add up to `generate` and that `cancel` stops. **204** `ferrum_gui` backend tests pass (191 unit, 11 more than the 193 before progress events for downloads and *Clean*, Task 2.5: the throttle, the download stream and *Clean* progress tests and one IPC test, plus two Node flows in `ui/app_flows.test.js`; among them `train_resume_other_corpus_is_refused`; among them 21 headless IPC tests through Tauri's mock runtime (`ipc_tests.rs`, `ipc_errors.rs`: bad arguments for every argument of every command, each refusal of fine-tune, train and GGUF run, and the command-matrix guard `every_command_has_an_ipc_test`, which reads the `generate_handler!` list in `lib.rs` and fails for a command no IPC test names), and 13 integration, among them the estimate-versus-wall-time regression tests; downloads are tested against a local HTTP server and hand-built GGUFs, no network and no trained model), and so do the Node tests `ui/stream.test.js`, `ui/escape.test.js`, `ui/help.test.js` (every field and button in `index.html` has a help entry of the required length; the job kinds match the Rust enum) and `ui/tooltip.test.js` (positioning, hover/focus/Esc behaviour, no markup injection, `wordCount` and the title-less `data-tip` bubble), `ui/tooltips_guard.test.js` (`every_element_has_a_30_word_tooltip`: every control, heading, status line, progress bar, error line, table header, chart and output area of `index.html` has a tool-tip of at least 30 words, from its `help.js` entry or a `data-tip`; 351 elements on 2026-11-03, including the 12 hint paragraphs; the runtime-built buttons' tips are checked too; 29 words fail and 30 pass on synthetic markup; no native `title`), `ui/estimate_view.test.js` (formatting), `ui/live.test.js` (the wiring against the real `index.html` structure and a fake backend: tooltips update when a field or the hardware changes, debouncing, stale responses, errors, no markup injection), `ui/app_smoke.test.js` (the real `app.js` loads with and without Tauri), `ui/app_flows.test.js` (every tab's flow through the real `app.js`: the exact arguments sent, results, errors, progress events, streaming reconciliation, the over-budget dialog; the arguments are pinned to a fixture that Rust deserializes), `ui/commands.test.js` (every `invoke()` name is registered in `lib.rs` and vice versa) `ui/csp.test.js` (no inline script or style, CSP not loosened) and the accessibility tests `ui/a11y_html.test.js`, `ui/a11y_css.test.js` and `ui/a11y.test.js` (accessible names, roles, contrast ratios, tab keys, dialog focus) and `ui/settings.test.js` (remembered settings, storage that throws, junk JSON) and `ui/errors.test.js` (structured errors, copy details, long messages, the spinner) and `ui/icons.test.js` (the bundle icons' PNG, ICO and ICNS structure), `ui/modules.test.js` (every tab script also loads under Node) and `ui/error_paths.test.js` (a failing backend call per tab, the Export tab's Inspect and size gate, non-streaming Generate), all against a small dependency-free DOM stand-in. Python: **46** pass and **18** skip of 64 (`test_stubbed.py` adds 10 that run `ferrum_model.py`, `ferrum_tasks.py` and the `make_task_data.py` usage guard against fake `lighteval` modules; the skipped are the LightEval-dependent ones, because LightEval is not installed on this machine; 48 passed with it on 2026-10-25, before the suite grew to 53 tests; the run emits no `ResourceWarning`). The how-to for exact resume, corpus identity, cancel and the `prepare` marker is `docs/resume-cancel-and-prepare.md` (commands run on 2026-11-03). `slm_cli/tests/tokenize_cli.rs` checks that `tokenize` writes exactly the ids `encode_docs` gives (across the 1 MiB read chunk, for directories, custom separators and the error paths), `slm_cli/tests/token_train_cli.rs` that `train tokens.gguf` gives the same model bytes as `train corpus.txt` (serial, threaded, with knobs, across a resume) and `tests/tests/test_token_training.rs` the same at library level plus validation and refusals. `tests/tests/test_bench_script.rs` runs `scripts/bench_slm.sh` from another directory with a relative corpus path (2 tests). `tests/tests/test_coverage_script.rs` checks `scripts/coverage.sh` with a fake `cargo` and `pgrep` (unknown or non-numeric flag exits 2, refuses to run beside another cargo, always passes `--release`, the low-memory variables and the line gate, still writes missing.txt below the floor, and the JS guard is not fooled by `testapp.js` for `app.js`; 6 tests). The sibling **ferrum-ml** project is outside this repository and was not re-run in this pass. |
| **Safety** | `#![forbid(unsafe_code)]` in every crate root, plus a repo-wide scan test |
| **Dependencies** | `ferrum_core`: two pure-Rust Unicode tables, nothing else. `slm_cli`: `sha2`, `serde_json`. `slm_server`: `axum`, `tokio`, `serde`, `serde_json`, `sha2`. `slm_wasm`: `wasm-bindgen`. `ferrum_gui`: Tauri and a few helper crates (`tauri-plugin-dialog`, `sysinfo`, `ureq`, `serde`, `serde_json`), `slm_cli` and `slm_server` by path (the latter brings `axum`, `axum-core`, `matchit` and a few more small crates into its lockfile; `tokio`, `hyper` and `tower` were already there through Tauri), and `tokio` declared directly for the server's shutdown signal. `rayon`, `ndarray`, and `faer` were measured and not adopted ([Manual §2.2.1](Manual.md#221-what-is-used-and-what-was-tried-and-not-kept)) |
| **Commits** | `master` has 314 commits (`git log --oneline master | wc -l` at `575d4cf`); `feat/coverage-100` is 32 commits ahead of it at `9966e1c` (`git rev-list --count master..HEAD`; `master` is `575d4cf`): the coverage work, the checkpoint, cancel and `prepare` features, the polish commit and the final fixes |

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
| Tabular MLP stack: `train_cli`, the CSV loader (`CsvDataset`, split helpers), the trainable MLP (`Net`), 10 datasets, 10 pretrained models, download/train scripts, and their tests | **`../ferrum-ml`** as crate `ferrum_ml` + `tabular_wasm`, depending on `ferrum_core` by path | A working, useful ML demo that has nothing to do with SLMs |
| `TabularModel` browser binding | **ferrum-ml** `tabular_wasm` | Same |
| One-hot MLP SLM trainer (`GenerativeSLM::train`), embedding-MLP trainer (`train_embedded`), hex-CSV builder, one-hot inference and scoring paths, and their 6 tests | **`../ferrum-junk`** (`removed_slm_paths.rs`, `removed_slm_tests.rs`) | Teaching baselines strictly dominated by the transformer, and of no use on their own |
| `web/` launcher (linked to three external repos not present here, ran one-hot models only) | **ferrum-junk** `web/` | Broken links, superseded model family |
| GitHub Pages `deploy.yml` (trained tabular models, published `web/`) | **ferrum-junk** `github-workflows/` | Deployed only the removed pieces |
| GUI embedded shell (`run_terminal`, `term_cwd`) and the Tabular tab | **ferrum-junk** `gui/` (fragments) | Arbitrary command execution from the webview, unrelated to SLMs. The docked panel remains as a read-only engine log. |
| GUI Train-tab method selector (embedded / one-hot) | removed | Only the transformer path remains |
| 26 overlapping Markdown files (root docs, `docs/`, `manual/`, crate READMEs) | Consolidated into **ReadMe.md, Manual.md, Status.md** (and later **Samples.md**). The agent design specs, plans, and work logs are archived in **ferrum-junk** `docs-archive/`. | One coherent document set instead of 26 drifting ones |

**Renames:** `tabular_wasm` → **`slm_wasm`** (now only `TransformerSLMModel`);
`ferrum_core::csv` → **`ferrum_core::meta`** (now only `TaskType`,
`ModelMetadata`, `Normalizer`, which the model format needs).

**A latent test bug surfaced.** ferrum-ml's trained-model tests used a path
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
| Data tools | `dataset`, `slm_cli` `prepare` | ✅ Streaming cleaner and splitter with a hash manifest (the pipeline is the `slm_cli::prepare` library API, `prepare(&PrepareOptions, progress)`, which the CLI and the GUI both call; `slm_cli` has a `lib.rs` for this because the pipeline needs `sha2` and `serde_json`, which `ferrum_core` may not depend on); `clean_corpus`, `corpus_stats`, `validate_for_training` |
| Token files | `token_file/`, `train_transformer/source.rs`, `slm_cli` `tokenize` | ✅ `ferrum-tokens` GGUF (I16 or I32 ids) written in one pass and read by `Read`+`Seek`; `train tokens.gguf` streams windows and is bit-identical to in-memory training (serial and threaded, resume at an epoch boundary or mid-epoch, budget); `info` describes the file |
| Model format | `meta`, `loader/` | ✅ GGUF v3 only: `gpt2` layout (runs in llama.cpp) and the generic `ferrum` layout; F32/Q8_0/Q4_0; bounds-checked; checkpoints, tokenizer files and token files are GGUF too |
| GGUF reader | `gguf/` | ✅ F32/F16/Q8_0/Q8_1/Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K; Q2_K/Q3_K (read-only); streamed `open`; IQ* rejected |
| GGUF tokenizer | `gguf_tokenizer` | ✅ BPE exact; SPM decode exact, encode greedy |
| Llama/Qwen runner | `llm` | ✅ RMSNorm, RoPE, GQA, SwiGLU; cached decode equals full forward |
| Llama/Qwen fine-tuning | `llm_train/` | ✅ Gradient-checked backprop, AdamW, schedules, QAT, dropout, `.flck` checkpoints, threaded epochs |
| GGUF writer | `gguf_write/` | ✅ Byte-exact GGUF v3; encoders verified as inverses of the decoders; atomic writes |
| Chat templates | `chat_format` | ✅ ChatML / Llama-3 / Llama-2 / Zephyr detection, rendering, and stop tokens |
| CLI | `slm_cli` | ✅ train · run · generate · eval (`--json`) · info · prepare · tokenize · run-gguf · finetune-gguf · export-gguf · export-ollama · export-lmstudio; strict flag parsing |
| Serving | `slm_server` | ✅ `ferrum-serve`: health, info, log-likelihood, rolling log-likelihood, generate; limits, one model thread, survives model panics. Also a library: `bind` (guard, load, bind; real address, banner, request counter and log) then `Bound::serve`, which the binary and the GUI Serve tab share |
| Evaluation | `eval/lighteval/` | ✅ LightEval 0.13.0 model class and tasks (held-out perplexity, TinyStories cloze, BLiMP, ARC-easy) over the server |
| Data operations | `dvc.yaml`, `params.yaml` | ✅ fetch → prepare → train → eval, reproducible; `lighteval` stage opt-in |
| Runtime compatibility | `tests/`, `scripts/verify_runtimes.sh` | ✅ llama.cpp b11521 and Ollama v0.40.2 required in CI; LM Studio 0.4.18 checked locally |
| Browser | `slm_wasm` | ✅ `Slm` text API (`encode`, `decode`, `generate`, `score`, `info`, and pull streaming `start`/`next_chunk`/`cancel`) for BPE and character-level models, equal to the library and CLI for the same seed (native tests); `TransformerSLMModel` for KV-cached token-id inference, sampling helpers and attention maps; builds for wasm32. `slm_wasm/demo/` holds a browser demo (`index.html`, `demo.js`, `style.css`: file picker or drop, prompt, sampling controls, streaming output, token count, tokens/s, errors; strict CSP) and `worker.js`, a Web Worker wrapper. `build_wasm.sh` copies `pkg/` next to it. Node tests without dependencies: `demo/worker.test.js` (the message protocol against a fake `Slm`; it now runs `worker.js` itself in a `vm` context) and `demo/demo.test.js` (every id `demo.js` uses exists, labels, no inline script, style or handler, CSP not loosened; it now also runs `demo.js` against a stub page and Worker: generate, chunks, stale ids, errors, file pick, drop, `?model=`); both run in CI. `slm_wasm/tests/node.rs` is a `wasm-bindgen-test` suite (6 tests on a tiny model built inside the test, untrained because the trainer reads the clock and `wasm32-unknown-unknown` has none: load/info, encode/decode against native ids, `generate` against the library and the texts a native run gave, streaming and cancel, `score`, `JsError` paths) that runs the compiled module under Node; it is wasm32-only (an empty file natively) and CI installs `wasm-bindgen-cli` 0.2.122 for it. **It has compiled for wasm32 here (`cargo test --target wasm32-unknown-unknown --no-run`, clippy `--tests`) but could not be run here, because `wasm-bindgen-cli` is not installed: it ran only on CI.** Its first CI run (commit `76930f3`) failed all six tests, because the trainer calls `Instant::now`, which `wasm32-unknown-unknown` lacks; the model is therefore built without training (`b2bd168`). On that commit's CI run all six pass, including the generated texts and the score taken from a native run, so wasm and native agree on this model. The demo, the generated JS bindings and the real worker were **not** run in a browser or Node here (no `wasm-bindgen` CLI on this machine, and it was not installed to keep to the project directory), so the page is unverified end to end; `node --check` and the two Node tests pass |
| GUI | `ferrum_gui` | ✅ Backend tests and builds. Tabs: Datasets, Prepare, Train, Generate, Evaluate, Models, GGUF, Fine-tune, Export, Serve, System, Capable, plus engine log. Every control and informative element has a tool-tip of at least 30 words (help entries for controls, `data-tip` for the rest, `ui/tooltips_guard.test.js`; tutorial `docs/gui-tooltips.md`); not click-tested in a window. |
| CI | `.github/workflows/ci.yml` | ✅ Build/test (`--locked`), rustfmt, clippy, rustdoc (`-D warnings`), JS syntax, MSRV 1.87 `cargo check`, cargo-deny, wasm32 (build, clippy including tests, the demo's Node tests, and `wasm-bindgen-test` under Node), Python tests (without LightEval), GUI fmt/clippy/test (`ferrum_gui/Cargo.lock` is committed), runtimes job. `.github/workflows/release.yml` (separate, tags and manual runs only) builds the Linux bundles (the G13 row of the known-issues table). The `rustdoc` step ran locally; the MSRV job and the `runtimes` job have not run on GitHub, and no 1.87 toolchain was installed here, so 1.87 rests on code analysis (`is_multiple_of` needs 1.87, `sha2` 1.85; clippy's `incompatible_msrv` finds no other newer std API). The MSRV covers the shipped crates: two `slm_cli` test helpers use `File::lock` (std 1.89), so `cargo test` needs 1.89. |

---

## 5. Known issues

| Severity | Issue |
|---|---|
| 🟡 | **`train corpus.txt` still reads the whole corpus.** A corpus larger than RAM goes through `tokenize` and a token file instead ([Manual §6.11](Manual.md#611-corpora-larger-than-ram-tokenize-and-token-files)), which streams windows from disk with memory O(batch × context). Its limits: the tokenizer is still trained from text in memory (train it on a sample); `tokenize` holds one document at a time, so a corpus with no separators is one document; the shuffled window order costs 8 bytes per window (small at the default stride, 8 bytes per token at `--stride 1`); `--val` reads the held-out token tail into memory and splits by tokens, not characters. Verified bit-identical to in-memory training and run on a 20 MB corpus; a corpus beyond RAM was not run on this machine. |
| 🟡 | **Training is reproducible per shard count, not per machine.** `--threads 0` (auto, the CLI default) takes the number of data-parallel shards from `FERRUM_NUM_THREADS` or the core count, and a different shard count trains different weights (the reference model's perplexity differs between 1 and 8 shards). The DVC pipeline pins the shard count (`train.threads: 8` in `params.yaml`), so its `dvc.lock` hashes reproduce on any machine; a hand-run `train` at auto reproduces them only on a machine with the same core count. Generation and evaluation are identical at any thread count. |
| 🟡 | **The server handles one request at a time.** One dedicated model thread owns the model, because `GenerativeSLM` is not `Send` or `Sync`. Each matmul still uses every core, but a long batch delays other clients. It binds to loopback by default; an optional bearer token (`FERRUM_TOKEN`) protects every route but `/health`, and a non-loopback `--host` needs one (or `--allow-unauthenticated`). There is no TLS, so the token travels in clear text. |
| 🟡 | **LightEval needs two workarounds on 0.13.0.** `FerrumModel` skips `super().__init__()`, and rolling log-likelihood is not cached. The module is pinned to that version, to `xxhash<4`, and to `datasets<5`. The litellm and TGI backends cannot run log-likelihood tasks. The Python tests run in CI without LightEval (46 pass / 18 skip); the LightEval-dependent ones run only locally. |
| 🟡 | **Quantized files can differ from llama.cpp late in a generation.** llama.cpp quantizes activations in Q8_0 matmuls. f32 files match exactly; Q8_0 files are compared over the first 8 tokens; Q4_0 files are checked only for loading and generating. |
| 🟡 | **Character-level models and ferrum-ml tabular models do not run in llama.cpp, Ollama, or LM Studio.** They are GGUF, but use a generic Ferrum layout. The export commands refuse them. |
| 🟡 | **Only the verified tool versions are tested.** llama.cpp b11521, Ollama v0.40.2, LM Studio 0.4.18. Runtime tests skip a missing tool, but fail under `FERRUM_REQUIRE_RUNTIMES=1`; the CI `runtimes` job sets it (llama.cpp and Ollama required, `FERRUM_SKIP_LMSTUDIO=1` as the one exemption; LM Studio is required locally before a release). Newer versions may behave differently. The `runtimes` job has not run on GitHub yet, and the LM Studio commands in [Samples §7](Samples.md#7-serving-and-running-your-model-in-llamacpp-ollama-and-lm-studio) Part D were re-run on 2026-11-03 against LM Studio v0.4.26 (same greedy text as llama.cpp; `export-lmstudio` itself still not run). |
| 🟡 | **Models are small and CPU-bound.** The reachable sizes in §3 come from short runs of four configurations; the 33.8M configuration could not be re-run because the training memory guard refused it with the RAM available at that moment ([Manual §6.7](Manual.md#67-sizing-a-run-for-your-machine)). |
| 🟡 | **The Capable tab's training estimates are rough.** They ignore activation memory and attention FLOPs, and they scale a single-thread GEMM rate by 8 hyperthreads on 4 physical cores. Treat them as loose upper bounds. |
| 🟡 | **The GUI is not click-tested after the 2026-10-03 changes, or after the GGUF change, or after the tooltip and estimate wiring, or after the 2026-11-03 tool-tips on every element (124 `data-tip` texts; hover, focus and Escape are tested on the stub DOM only).** The backend compiles and its tests pass; the tooltip, estimate and hardware-line code is tested against a stub DOM and a fake backend only, never in a real webview, so the real layout (bubble position, the line in the top bar, tooltips of buttons inside a modal dialog: the bubble is now moved into the open dialog so it cannot render behind it, tested against the stub DOM only) and the CSP behaviour of `element.style` were not seen. |
| 🟢 | **`prepare` output is never a manifest over a partial set (Task 2.4):** `commit_outputs` removes the old `manifest.json` first, renames the splits and renames the new manifest last, so a present manifest means a complete set ([Manual §5.2](Manual.md)). Ceiling (`ponytail:`): the split files are not rolled back, so a crash can leave a mix of old and new splits, but without a manifest. Tested with a stub `rename` (order, failure on the third, first and manifest rename, unremovable old manifest); successful outputs are byte-identical. `dvc repro prepare` was NOT run: `.venv/bin/dvc` does not run on this machine. Coverage of the new lines: UNVERIFIED (coverage runs only at noon). |
| 🟢 | **GUI dataset downloads (G2):** a download streams into `<dest>.part` and is renamed to `dest` only when complete. A body over the 512 MiB cap, an empty body, or a mid-stream error is an error that removes the `.part` file and leaves any existing `dest` untouched (previously an over-cap file was silently truncated and reported as success). There is no sha256 check on the GUI path, because `ferrum_gui` has no hashing dependency (`scripts/fetch.sh` verifies for the CLI path). Tested with in-memory readers; not exercised against a live server. |
| 🟢 | **GUI security (G1):** untrusted strings (GGUF metadata, dataset catalog, errors) are HTML-escaped in `ui/escape.js` (tested by `ui/escape.test.js`) and `tauri.conf.json` sets a Content-Security-Policy (`default-src 'self'`, no inline script or style). The CSP was not exercised in a running webview on this machine, so it is not click-tested either. |
| 🟡 | **GUI time and RAM estimates are calibrated on one machine.** `estimate_job` uses rates measured on the 4-core i5-1135G7 (Manual §10.3 lists them); a machine with a different CPU is scaled only after the Capable check in a release build, and thread scaling beyond 8 threads, AMD compute-unit detection (`/sys/class/kfd`) and non-NVIDIA GPU names were never exercised on real hardware here (fixtures only). The regression test `tests/regression_estimate.rs` compares each estimate with a measured wall time on tiny models (estimated/measured in one run on 2026-10-26: train 1 epoch 0.93, train 30 epochs 1.01 (the source of the small-model floor, so not independent), generate 5.6, evaluate 1.1, clean 2.6, GGUF run 0.53, fine-tune 0.81, export 0.25); it only guards a 0.05x to 20x band. The RAM figure for training is the engine's own guard, which sits about 130 MB under the measured peak for tiny runs and about 1.5x over it for larger ones. |
| 🟢 | **GUI greedy decoding and sampling controls (G7):** the Generate tab takes temperature 0 (exact argmax, like `--temp 0`), top-k, top-p, repetition penalty, a token budget and stop strings (one per line, `\n` `\t` `\\` unescaped); the GGUF tab takes temperature 0. The backend runs `GenerativeSLM::generate_tokens_with` when a token budget or a stop string is set and `generate_stream_with` otherwise, the CLI's own choice, and validates like the CLI (`temp >= 0`, `top-p` in [0, 1], positive penalty). Tested headlessly against `generate_tokens` / `generate_with` on a tiny BPE model (same text, same streamed fragments, seed-independent at temperature 0), through the mock-runtime IPC path, and with the stub-DOM flow tests. GGUF runs have no stop strings (`LlamaModel::generate` stops on token ids only; the CLI has none either). Not click-tested. |
| 🟢 | **GUI job control (G9):** Prepare, Train, Generate, GGUF, Fine-tune and Export have a Stop button while they run (`cancel_job`; `src/jobs.rs` keeps one slot per kind, so a second start of the same kind fails at once with "a training job is already running"). Train and Fine-tune stop after the current optimizer step through `StopCondition` (`TransformerConfig.stop`, `finetune_epoch_until`) and still save the model and checkpoint (the result carries `stopped`, the UI says "stopped at step N"); Generate and GGUF run stop after the current fragment or token and keep the text so far; Export stops between phases and writes nothing. The GUI fine-tune checkpoint is now written through a temp file and a rename (it was a plain write). Not cancellable: Evaluate (one scoring call; it only refuses a concurrent second run), the encode-and-write phase of an export, and the GGUF load. Generate cancels through the engine's cooperative cancel flag (`SamplingParams.cancel`, checked by every decode loop before each token; §7.0.4); the old `resume_unwind` out of the text callback is gone. Tested headlessly: a training and a fine-tune job cancelled mid-run through IPC (stopped result, loadable model, written checkpoint that resumes, no `.tmp` left, second start refused), export cancelled at each phase, generation cancelled on both paths, and the Stop buttons against the stub DOM. Not click-tested. |
| 🟢 | **GUI dataset preparation (G3):** the new Prepare tab runs `train_transformer prepare` with the same code: `prepare_dataset` calls `slm_cli::prepare::prepare` (the pipeline moved out of the CLI binary into a library target of `slm_cli`; it needs `sha2` and `serde_json`, so it cannot live in `ferrum_core`, and `ferrum_gui` depends on `slm_cli` by path, which adds `slm_cli` and the `sha2` 0.11 stack (8 packages) to its lockfile; `cargo deny` still passes). Several files and folders in, an output folder, validation and test shares, seed, minimum and maximum characters, document separator, dedup, NFKC, lowercase, Gutenberg, lossy; it streams from disk, so a corpus never goes through the window. For the same options the splits and the manifest are byte-identical to the CLI's (tested against the library call, and the dvc `prepare` output is byte-identical before and after the move), and the manifest `args` are the equivalent command line. Progress events carry bytes and documents; Stop ends the run between documents and leaves earlier outputs untouched (a single huge document is read to its end first, marked `ponytail:`). Errors show the CLI messages with the flag names put into the tab's wording. Tested headlessly (library call, IPC, job slot, cancel mid-run, errors) and against the stub DOM; the estimate (`prepare`, 18 MB/s, measured) has a wall-time regression test. Not click-tested. |
| 🟢 | **GUI tokenize and train from a token file:** the Train tab has *Tokenize corpus* (`tokenize_corpus` over `slm_cli::tokenize`, which the `tokenize` subcommand now wraps; progress events, Stop, no file on a stop) and accepts a token file as its corpus, detected by content like the CLI (`train_slm` opens it with `slm_cli::token_corpus`, the code `train` uses, and calls `train_transformer_tokens[_validated]`). The pre-flight shows the token count, documents and the file's own vocabulary, which decides the model, and says the RAM figure does not depend on the corpus size; the time estimate uses the token count with no tokenizing time. Tested: the ids equal `encode_docs` and the CLI function's file bytes; training from the token file gives a model **byte-identical (same sha256)** to training from the text on a tiny corpus (the in-memory `--val` split differs from the token-file one, as in Manual §6.11, so equality is not claimed with validation); errors (missing or wrong tokenizer, Vocab 0). A corpus larger than RAM was not run through the GUI. Not click-tested. Moving the tokenizer pieces into the library also moved `check_writable`/`check_distinct_paths` (`slm_cli::outputs`) and `sha256_hex` (`slm_cli::hash`); the bin re-exports them. |
| 🟢 | **GUI export to Ollama and LM Studio (G4):** the Export tab has *Show Modelfile and commands*, *Export to Ollama*, *Import into LM Studio* and *Write Modelfile only*, calling `slm_cli::runtime_export` (moved into the `slm_cli` library next to `prepare`, so the CLI and the GUI run one implementation): the program lookup (field, `FERRUM_OLLAMA`/`FERRUM_LMS`, `PATH`, default location), the Modelfile generator, the refusal of every non-`gpt2` model with the Manual 9.5 pointer for a llama GGUF, `--modelfile-only`, the temporary Modelfile. The library gained detail-returning forms (`ollama_run`, `lmstudio_run`, `ollama_preview`, `lmstudio_preview`, `write_modelfile`, `default_lmstudio_repo`) that the old functions now wrap, so CLI behaviour and messages are unchanged. The tab shows the exact Modelfile and the commands that ran or would run; a preview starts nothing. Stop works before a program starts and while LM Studio's listing is awaited, not inside a running `ollama create`. **Never run against a real Ollama or LM Studio from the GUI tests** (fake scripts in temp dirs); the real-tool run of this tab was not done, so the commands are unchanged CLI code but the tab itself is untested against the real programs. Not click-tested. |
| 🟢 | **GUI Evaluate parity (G6):** the Evaluate tab reports bits per byte, token count and seconds, takes an optional test file next to the held-out text (one row each), a lossy-UTF-8 option, and saves the latest run as JSON through the save dialog with the keys of `eval --json`. A Llama/Qwen `.gguf` is detected from its architecture and scored with `LlamaModel::score_rolling` exactly as `eval-gguf` does (BOS prefix, window `min(512, context)`, optional precision, fine-tune checkpoint and the 90% RAM guard); the report is formatted in `ferrum_gui` because `slm_cli`'s lives in that binary crate. Tested against `GenerativeSLM::evaluate` and `score_rolling` on tiny models (identical bits), through IPC with a fine-tune checkpoint, and with the stub-DOM flows. The scoring call itself is not cancellable. Not click-tested. |
| 🟢 | **GUI training pre-flight (G8):** the Train tab shows the parameter count, the estimated peak RAM against the engine's limit (three quarters of the free RAM, at most 16 GB) and the time estimate, and a "Runs in llama.cpp?" verdict from the vocab setting (0 = character level, Ferrum-only layout: No; 256 and up: gpt2 GGUF: Yes), from `preflight_train` (`src/preflight.rs`, built on `TransformerNet::check_size` and the existing estimate module, so the figures match the Train tooltip and the engine's own refusal). Start is disabled with an explanation when the run would not fit, unless "I know, try anyway" is ticked; that sends `force`, which sets the new `TransformerConfig::skip_memory_check`. The verdict looks only at the vocab setting: it does not inspect a reused tokenizer file. Tested against the engine's own functions, through IPC, and with the stub-DOM flows. Not click-tested; the panel layout was never seen. |
| 🟢 | **GUI code layout (X1/G15):** `ferrum_gui/src/commands.rs` (2,083 lines, about 1,500 of them code) is now the `src/commands/` directory: `text`, `train`, `generate`, `gguf`, `guard`, `finetune`, `export`, `system`, with each module's tests beside it, and `ui/app.js` (978 lines) only wires the page while one plain-script module per tab (`shell.js`, `datasets_tab.js`, `train_tab.js`, ...) holds the logic, the way `prepare_tab.js` already did. No command name, argument struct or behaviour changed: the generate-handler list and both fixtures are unchanged and the 152 backend tests of that time and all Node tests passed unchanged. Checked by tests and stub DOM only, not click-tested in a window. |
| 🟢 | **GUI accessibility (G12):** the tab bar is a WAI-ARIA tablist with roving tabindex and arrow, Home and End keys; toasts, status lines, inline errors and progress bars are live regions or progressbars; every field has an accessible name; a high-contrast `:focus-visible` ring replaces the removed outlines; both dialogs trap Tab, close on Esc and return focus; `prefers-reduced-motion` is honoured; the CSS variables are contrast-checked (the primary button, the error red and the accent text were darkened or lightened to pass 4.5:1) and nothing relies on colour alone. Covered by `ui/a11y_html.test.js`, `ui/a11y_css.test.js` and `ui/a11y.test.js` against the stub DOM. Not tested: a real screen reader, the real focus ring and layout, and the engine log's `role=log` behaviour in a webview. |
| 🟢 | **GUI remembered settings (G11):** every tab's field values, the folder each Browse dialog last used and the Evaluate table are kept in `localStorage` (one versioned key, try/catch everywhere, so it works with storage blocked, full or damaged) and restored on start; *Reset settings* in the top bar clears them. File contents (raw/preview/held-out/Modelfile text boxes), the Hugging Face token and the "try anyway" switches are never stored; the Evaluate table is stored as result rows and re-rendered through the escaping builder. Covered by `ui/settings.test.js` against the stub DOM and a stand-in storage; a real webview's `localStorage` was not exercised. |
| 🟢 | **GUI structured errors and progress (G10):** every command now fails with `{code, message, hint}` (`ferrum_gui/src/error.rs`; the code is derived from the message, the inner functions still return strings) instead of a bare string. The UI shows the message and the hint, a *Copy details* button, a *Show full message* expander for messages over 300 characters, and a spinner on the long GGUF run, inspect, export, fine-tune, evaluate, generate and Capable calls. Client-side field errors no longer start with `Error: `. **Progress (Task 2.5, closes G10):** `download_dataset`, `download_hf_file` and `download_kaggle_file` emit `download-progress {id, bytes, total}` (`total` is the server's Content-Length, `null` when absent) and `clean_text` emits `clean-progress {bytes, documents}` (bytes of text consumed, non-empty lines), at most one event per 100 ms (`ferrum_gui/src/progress.rs`, driven by an injected clock in the tests) plus a final event that always goes out and carries the final total; bytes never decrease. `clean_text` now cleans line-aligned 256 KiB chunks and a test shows the result equals `clean_corpus` for every chunk size and option mix; it is `async` and takes an `AppHandle`, which the webview does not send, so command names, arguments, results and error strings are unchanged and both fixtures are unchanged. The Datasets tab shows both in one progress bar (the existing `.progress` component). `download_text` (the Source URL box) has no progress events. Covered by `error.rs` tests, the headless IPC tests and `ui/errors.test.js`; the clipboard, the spinner and the progress bar were not seen in a window (not click-tested). With the Gutenberg strip on, *Clean* counts bytes after the strip against the raw total, so the bar jumps to 100 % at the end (documented in Manual). |
| 🟡 | **Mobile is untested (G16).** Manual §10.5 used to tick Android and iOS for every feature; nothing proved it, so it now says *untested*: Tauri 2 supports mobile and the crate keeps the entry point, but no mobile project exists and none was built or run. |
| 🟡 | **The GUI boot smoke test only proves that the window opens (CI5).** The CI `gui` job builds the app, runs it under `xvfb-run` and `dbus-run-session` for 8 s and requires a window titled *Ferrum SLM Studio* and a live process. First run without a session bus (run 40435051750, `4d8dca4`): the process stayed up for 8 s (and for 20 s in run 40435907713), but `xwininfo -root -tree` listed only `0x200001 "ferrum_gui": ("ferrum_gui" "Ferrum_gui") 10x10+10+10` (GDK's client-leader window) and never the app window; stderr held `dbind-WARNING: AT-SPI: Error retrieving accessibility bus address: org.freedesktop.DBus.Error.ServiceUnknown` and `Gtk-DEBUG: Failed to get the GNOME session proxy: The name org.gnome.SessionManager is not owned`. With `dbus-run-session` (run 40436627701, `3d2cb2e`) the window appeared after 1 s and the step passed. It does not click, read the page or check that `ui/` loaded, so the real layout, the tabs and the CSP in a running window are still unseen. |
| 🟢 | **GUI path hardening (G17):** every command that reads, writes or creates a path from the webview (`save_corpus`, `read_text_file`, train/generate/evaluate, `model_info`, `gguf_info`, `run_gguf`, fine-tune, export, the three downloads) first runs it through `src/paths.rs`: it refuses empty, NUL-containing and over-4096-byte paths, anything under `/dev`, `/proc` or `/sys` (after symlinks and `..` are resolved), and existing directories, FIFOs and device nodes where a file is expected. Typed relative paths still work. This is not a sandbox: ordinary files anywhere the user can write stay reachable, and there is a check-then-use window. The window capability now grants only `core:event:allow-listen`, `core:event:allow-unlisten`, `dialog:allow-open` and `dialog:allow-save` (was `core:default` and `dialog:default`); `withGlobalTauri` stays on because `ui/app.js` reads `window.__TAURI__` and importing the API otherwise needs a bundler. Unit-tested; the trimmed capability file builds, but it was not click-tested in a running window. |
| 🟢 | **GUI Serve tab (G5):** the tab starts and stops the `ferrum-serve` server for one model inside the app. Decision: **(a) embed the library**, not a sidecar binary. `slm_server` already took a shutdown future; it gained `bind()` (the CLI's exposure guard, model load and socket bind, returning the real address for `--port 0`, the startup banner, a request counter and an optional per-request log) and `Bound::serve()`, and `cli::run` is now `bind` + print the banner + `serve`, so the binary behaves as before (its 79 tests pass unchanged, plus 2 new). The GUI adds `serve_start`, `serve_stop` and `serve_status` (`src/serve.rs`, one `Job::Serve` slot, `serve-log` events). The form is checked like the CLI (limits of at least 1, no non-loopback host without a token; `--allow-unauthenticated` has no button) and the token is only in memory while the server runs: not in settings, events, logs or status, and the curl example reads `$FERRUM_TOKEN`. Stop is graceful, waits up to 10 s for a stuck request, and returns when the model is released. **Dependency effect:** `ferrum_gui/Cargo.lock` gains `axum`, `axum-core`, `matchit`, `serde_path_to_error`, `serde_urlencoded` and a few other small crates (`tokio`, `hyper`, `hyper-util`, `tower` were already there through Tauri), `tokio` is declared directly (feature `sync`) for the shutdown channel, and the lock moved `tokio` 1.52.3 to 1.54.1 because `cargo deny --manifest-path ferrum_gui/Cargo.toml check` failed on RUSTSEC-2026-0386 (an unsound `AsyncFd` constructor, not used by Ferrum) once `slm_server` was a dependency; both `cargo deny` runs pass. Tested headlessly on 127.0.0.1 only: the server started on port 0 with a tiny model and called over a real `std::net::TcpStream` (`/health`, `/v1/info`, `/v1/generate`, 401 with and without the token, 404, the counter, the banner and request lines in the events, a second start refused, stop closing the port, a restart), validation and start-up errors, an estimate with a wall-time regression test, and the stub-DOM flow. **Not click-tested:** the tab's layout, the real clipboard and `crypto.getRandomValues` in the webview, and serving a model larger than a tiny one (the load time and RAM estimate for it are assumptions, marked as such). |
| 🟢 | **GUI icons and a Linux release workflow (G13):** `docs/assets/logo.png` (the 512x512 Ferrum logo, since 2026-11-03; `ferrum_gui/icons/icon.svg` is the older vector source, kept) is the source, also shown at the top of every Markdown doc and, downscaled to 128x128, in the GUI header (`ui/logo.png`); `cargo tauri icon` generated `icon.ico`, `icon.icns` and the PNGs, and `tauri.conf.json` lists them, so Windows and macOS bundles no longer lack their icon formats. `ui/icons.test.js` checks the files exist and that the PNG, ICO and ICNS structures are well formed. `.github/workflows/release.yml` (tags `v*` and `workflow_dispatch`) builds the AppImage and the deb with a pinned, cached `tauri-cli` 2.11.3 and uploads them to the run; on a tag whose name equals the app version it also creates a **draft** GitHub release. Run by hand on `master` (run 40439676923) it succeeded on the first try: AppImage 83,884,536 bytes, deb 5,999,602 bytes, 11 min 14 s cold. **Not done:** code signing and an updater (out of scope: a key, a notarization account and the updater plugin are needed; the bundles are unsigned), the draft-release step has never run for real (it needs a tag, and a tag would publish a draft release), neither bundle was installed or started, and no Windows or macOS bundle was built (how to do it by hand is in Manual §10.5). |
| 🟡 | **A matmul crate is an open decision.** `ndarray` gave +34% to +40% train tokens/s in a trial but would be the engine's first hot-path dependency (and slowed single-token generation in its plain form); `faer` failed the determinism rule. The shipped kernel is std-only with a 4×8 register tile (+20%). See [Manual §2.2.1](Manual.md#221-what-is-used-and-what-was-tried-and-not-kept). |
| 🟢 | An interrupt (`SIGINT`) does not save on exit; the last periodic checkpoint is what resumes. |
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
is done as well. A resume from a checkpoint written inside an epoch now replays the rest of
that epoch's window order (`ferrum.ckpt.epoch_rng`), so it is bit-identical to an
uninterrupted run. Checkpoints also record the corpus (`ferrum.ckpt.corpus_id`), and a
resume on a different corpus is refused. The GUI has a tool-tip of at least
30 words on every control and informative element, kept by a guard test
(`docs/gui-tooltips.md`).

### 6.2 Still open

These are directions, not commitments.

| ID | Feature | State |
|---|---|---|
| F3 | `qwen3` architecture (per-head q/k RMSNorm, explicit head_dim) | Not started |
| F7 | Perplexity evaluation for GGUF models (`eval-gguf`, Evaluate tab) | **`eval-gguf` done** (`LlamaModel::score_rolling`; Manual §9.3); the GUI Evaluate tab scores imported GGUFs too (G6) |
| F5 | Streamed GGUF export (two-pass, checkpoint overlay without loading it whole) | Not started |
| F6 | Safe-SIMD decode (autovectorization-friendly `qlinear` loops) | Not started |
| F4 | GUI CI boot smoke test (`xvfb-run`) | **Added** as the *Boot smoke test* step of the CI `gui` job (build, run under `xvfb-run` for 8 s, require the process is still up, then stop it); no clicks. Its outcome on the runner is recorded in §5 |

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
- **Faster kernels.** The 4×8 register tile is done, and matmuls of 20 million
  multiply-adds or more now borrow their operands on scoped threads instead of
  copying them for the pool (512² GEMM -15%; not measurable on the reference
  model, whose matmuls are mostly smaller; Manual §2.2.1). Still open:
  packed-B micro-kernels, `Arc`-held weights for the small pooled matmuls
  (needs the public `Vec<f32>` weight fields to change), and the decision on a
  matmul crate (§5). Likely 2–5× without `unsafe`, which
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

### 7.0 `slm_wasm` sampling helper

`TransformerSLMModel.sample_from_probs` now draws with `r < cumulative` instead of `r <= cumulative`, so a zero-probability leading entry is never returned at `random_value == 0` (temperature 0 is exactly argmax), and an empty `probs` returns 0 instead of underflowing. Draws at exact cumulative boundaries can differ from earlier builds; the JS API and error strings are unchanged.

### 7.0.1 `GgufBuilder::tensor` refuses a duplicate name in release builds

The duplicate-name check in `GgufBuilder::tensor` was a `debug_assert!`, so release builds wrote a malformed file silently; it is now an `assert!` and panics with `duplicate tensor name '<name>'`. No caller in the workspace pushes a duplicate (the debug-built CI suite already enforced it). The private helpers behind `llama_gguf_bytes` (`add_tensor`, `add_weight_2d`, `add_linear`) no longer return a `Result`: `plan_tensor_type` only picks a block type when the block axis divides the tensor, so the encode step could not fail; the public `llama_gguf_bytes` and `write_llama_gguf` signatures and error strings are unchanged.

### 7.0.2 Checkpoints record where their epoch began

A `ferrum-checkpoint` written by the config-driven trainer (`train`, `train tokens.gguf`, `GenerativeSLM::train_transformer_config*`, `train_transformer_tokens*`) carries a new optional key, `ferrum.ckpt.epoch_rng`: a `u64` array `[rng, step, epoch]`, the RNG state the current epoch drew its window order from, the step it began after, and its 1-based number (the step and number are stored because epochs with a stride above 1 are not all the same length). A resume rebuilds that epoch's order and skips the steps already done, so a run resumed from a mid-epoch checkpoint is bit-identical to an uninterrupted one, serial or with a fixed `--threads`, in memory or from a token file. Older checkpoints, which lack the key, still load and resume as before (the rest of the partly done epoch is reshuffled) and print `note: the checkpoint (step N) predates ferrum.ckpt.epoch_rng; ...`. Checkpoints written through `TransformerNet::save_checkpoint` outside that loop (the positional API) carry no key and are byte-identical to before (a net loaded from a config-loop checkpoint keeps that key if it is then trained through the positional API and saved again, so resume it with the config-driven trainer, not the positional one); the model GGUFs and every result of a run that never resumes are unchanged. The checkpoint layout version stays 1; an older build reads only the keys it knows, so it should load a new checkpoint and resume it with the old reshuffle (not run).

### 7.0.3 Checkpoints record the corpus

`TransformerConfig` gains `corpus_id: Option<String>` (default `None`; callers use `..TransformerConfig::default()`). A checkpoint written with an id carries the optional string key `ferrum.ckpt.corpus_id`. `ferrum_core` treats it as opaque; `train` (and the GUI Train tab) set it to `sha256:<hex>` of the corpus file (text or token file, hashed in 64 KiB chunks by `slm_cli::hash::file_id`) whenever the run writes a checkpoint. On resume the ids are compared only when both sides have one; a mismatch is refused with an error naming both ids and the word `corpus`, before any weight is loaded into the run, and the old `note: the corpus is not checked ...` line is gone. A checkpoint without the key, or a run without an id (the library's positional API, a run without `--checkpoint`), resumes as before; a run without an id keeps the checkpoint's. The ids are of the file, so a checkpoint trained from a text corpus is refused on the token file made from it, and the reverse (the error text says so). Because the id is part of the checkpoint, a text-corpus checkpoint and a token-file checkpoint of the same run now differ by that key (the models stay byte-identical). Checkpoints written through the positional API are byte-identical to before. The checkpoint layout version stays 1. Coverage of the new lines: UNVERIFIED (coverage runs only at noon).

### 7.0.4 A cooperative cancel flag in every decode loop

`SamplingParams` gains `cancel: Option<Arc<AtomicBool>>` (default `None`) and `is_cancelled()`; `Generation` gains `cancelled: bool`. Every decode loop of `GenerativeSLM` (`generate*`, `generate_until`, `generate_tokens[_with]`, `token_stream`/`stream_next`) and `LlamaModel::generate[_cancellable]` loads the flag once per token, before sampling, and on a set flag returns `Ok` with exactly the text already streamed (a half UTF-8 character or a held-back stop prefix is dropped); `stream_next` returns `None` and `Generation.cancelled` is `true` with `stop_reason` `MaxTokens`. A cancelled run draws no further RNG values; with the flag `None` or unset, output and RNG draws are bit-identical to before (tested on the tiny BPE and character models and on both non-transformer fallbacks). The check is an atomic load, never the clock, so it runs on `wasm32-unknown-unknown`. `SamplingParams` no longer derives `PartialEq`: it is implemented by hand over the four sampling fields, so the flag is ignored by `==`. `ferrum_gui`'s Generate tab cancels through this flag instead of unwinding out of the text callback; its command, arguments and errors are unchanged. `ferrum-serve` does not use the flag yet. **Source break:** a `SamplingParams { .. }` or `Generation { .. }` literal that lists every field no longer compiles; the in-repo ones (`slm_cli`, `slm_server`, `slm_wasm`, `ferrum_gui`, tests, `Manual.md`) now end in `..SamplingParams::default()`. The sibling ferrum-ml has no `SamplingParams` or `Generation` use (grepped, not re-built). Coverage of the new lines: UNVERIFIED (coverage runs only at noon).

### 7.0.5 `prepare` removes the old manifest before it renames

`train_transformer prepare` (and `slm_cli::prepare::prepare`) now deletes an existing `manifest.json` at the start of the commit step and renames the new one last. Successful runs write the same bytes and print the same line; the only visible change is that a run that fails during the final renames leaves no manifest (before, it could leave the previous run's manifest over mixed files). Arguments, errors and progress are unchanged. No action for callers.

### 7.1 Breaking changes

| Change | What breaks | What to do |
|---|---|---|
| **FINF is removed.** Every file Ferrum writes is GGUF v3: models, training checkpoints, and tokenizer files. | `.bin` model files (FINF v4 and v5) no longer load. Checkpoints written before this change (magic `FCKP`) fail with a "pre-GGUF checkpoint" error. Old tokenizer files in the `256;257...` text form are rejected. Model file names are now `.gguf`. | Retrain. Committed models in this repo and in ferrum-ml (10 tabular models) were converted before the reader was deleted. |
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
| **`TransformerConfig` gained `corpus_id`** (`Option<String>`, default `None`). | Struct literals already need `..TransformerConfig::default()`. A checkpoint written with an id carries `ferrum.ckpt.corpus_id` and is refused on a different corpus (also text against its token file); one without the key, or a run without an id, resumes as before (§7.0.3). | None. |
| **`TransformerConfig` gained `stop: StopCondition`** (default never stops) for graceful stop (`--max-seconds`, `<model>.stop`, a cancel flag). `train_transformer_steps`' `StepHook` now returns `Result<bool>` (`Ok(true)` = continue). | A struct literal without `..TransformerConfig::default()` no longer compiles (only `slm_cli` and `ferrum_gui` had one; both fixed). A custom `StepHook` must return `Ok(true)`. | Add `..TransformerConfig::default()`; change `Ok(())` to `Ok(true)` in a hook. |
| **`TransformerConfig` gained `skip_memory_check`** (default `false`): skips the up-front refusal of a training run that needs more than 16 GB or three quarters of the free RAM. | A struct literal without `..TransformerConfig::default()` no longer compiles (`slm_cli`'s full literal in `args.rs` was updated; the sibling ferrum-ml was not re-built). Default behaviour is unchanged; the CLI never sets it. | Add `..TransformerConfig::default()`. |
| **`slm_cli` is a library as well as a binary** (new `src/lib.rs`: `slm_cli::prepare`, `slm_cli::runtime_export`, `slm_cli::tokenize`, `slm_cli::token_corpus`, plus `outputs` and `hash` helpers), so the GUI shares the prepare pipeline. `PrepareOptions` has a `stop` field and `prepare` returns a `PrepareReport` instead of printing. | Nothing for CLI users: `train_transformer prepare` writes the same bytes and prints the same line. Code that built `slm_cli`'s private `PrepareOptions` does not exist outside the crate. | None. |
| **`SamplingParams` gained `cancel` and `Generation` gained `cancelled`** (§7.0.4): a cooperative cancel flag checked once per token. | A struct literal of either without `..Default` no longer compiles; `==` on `SamplingParams` ignores the flag. Output without a set flag is bit-identical. | Add `..SamplingParams::default()`; read `g.cancelled`. |
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
| 2026-10-03 | Evaluation for SLM training on 2 GB VRAM / 16 GB RAM hardware; **training-pipeline fixes** (stride, budgets, incremental BPE, tokenizer reuse, schedule/clip/tie/validation/checkpoint wiring); **restructuring** into Ferrum / ferrum-ml / ferrum-junk; documentation consolidated into three files |
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
| `ferrum_core` source files / lines | 108 / 33,282 (`ferrum_core/src`, including unit tests) |
| `ferrum_core` `pub fn` | 329 |
| Other Rust source (files / lines) | `slm_cli` 25 / 7,152; `slm_server` 11 / 2,408; `slm_wasm` 3 / 1,081; `tests` 17 / 3,600; `ferrum_gui` 6 / 3,252 |
| External dependencies (`ferrum_core`) | 2 (`unicode-general-category`, `unicode-normalization`) |
| `unsafe` blocks | 0 (forbidden) |
| Tests | 1131 pass, 0 fail, 2 ignored (workspace) + 162 `ferrum_gui` + 36 Python (18 skipped) |
| Parameters, reference model | 507,552 |
| Train tokens/s, reference model, 8 threads (start of review / before the tile / now) | 5,895 / 6,141 / 6,725 |
| Train tokens/s, 1 / 2 / 4 / 8 threads (a separate set of runs from the row above) | 2,900 / 6,058 / 6,764 / 7,090 |
| Eval seconds, 30 KB held-out (start of review / before the tile / now) | 138.9 / 1.76 / 1.65 |
| Generate ms per KB of output (start of review / before the tile / now) | 6,873 / 100 / 101 |
| Square GEMM 2048², 8 threads / 1 thread | 40.3 / 13.6 GFLOP/s |
| Square GEMM 512², 8 threads, previous build / scoped-thread build (2026-10-27) | 54.2 / 64.0 GFLOP/s |
| Markdown documents | 5 (`ReadMe`, `Manual`, `Status`, `Samples`, `glossary`) |
| Line coverage, workspace (`scripts/coverage.sh --rust`, 2026-11-03 noon run, commit 9fe7320) | 99.70 % (92 of 30,480 lines missed; functions 98.54 %, regions 98.80 %; `ferrum_gui/src/(main\|run).rs` and `slm_wasm/src/lib.rs` excluded; per-file misses in `docs/coverage-results.md`). The CI floor is 99 (the nearest whole percent the run met) |
| Line coverage, `ferrum_gui` Rust (`--gui`, 2026-11-03 noon run) | 98.75 % (76 of 6,102 lines missed; functions 91.74 %, regions 98.15 %; `main.rs` and `run.rs` excluded). The CI floor is 98 |
| Line coverage, UI and demo JavaScript (`--js`, 2026-11-03 noon run, Node 24) | `ferrum_gui/ui/*.js` 100.00 % lines (branches 93.59 %, functions 98.31 %); `slm_wasm/demo/` 98.06 % lines (`worker.js` lines 63-65 uncovered). The CI floor is 98 over both directories |
| Line coverage, `eval/lighteval/*.py` (`--py`, coverage.py, 2026-11-03 noon run) | 100 % of statements (153 of 153) with LightEval absent, so the 18 LightEval-dependent tests skip and fake `lighteval` modules cover `ferrum_model.py` and `ferrum_tasks.py`. The CI floor is 100 |

Speed rows: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 4 cores / 8 threads, 2026-10-26, medians of three runs from [Samples §9](Samples.md#9-benchmarking). Line counts are `find ... -name '*.rs' | xargs cat | wc -l` on the working tree, excluding `target/`.
