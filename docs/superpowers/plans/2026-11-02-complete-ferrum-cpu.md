# Completing ferrum-cpu: Coverage, Gaps, Roadmap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **One agent at a time** (CLAUDE.md working rule 2): subagent-driven means strictly sequential, never parallel.

**Goal:** Take ferrum-cpu from "feature-complete for 0.3.0 with 98.19 % line coverage" to "every module in `Status.md` §6 either shipped or explicitly declined, 100 % line coverage gated in CI, and a functional (real binary) plus integration test for every user-facing surface".

**Architecture:** Four phases. **Phase 0** builds the measuring stick (one coverage script, a CI gate that ratchets). **Phase 1** closes the 524 missed lines file by file, deleting dead code where a missed line is unreachable. **Phase 2** closes the known-issue gaps that `Status.md` §5 and the `ponytail:` markers name. **Phase 3** lists the roadmap features (`Status.md` §6.2) at interface-and-test level. Each one gets its own detailed plan, written just before it runs, because each is a subsystem of its own (writing-plans scope rule).

**Tech Stack:** Rust 2021 (MSRV 1.87, stable toolchain), `cargo-llvm-cov` (installed, `~/.cargo/bin/cargo-llvm-cov`), Node 24 built-in test coverage (`node --test --experimental-test-coverage`, verified working on `ferrum_gui/ui/escape.test.js`), Python stdlib `trace`.

**Spec:** There is no separate design doc. The sources are `Status.md` (§4 implemented, §5 known issues, §6 roadmap, §7 compatibility), the `CLAUDE.md` hard constraints, and the baseline measured below. Read `Status.md` §5–§6 before any Phase 2 or 3 task.

## Baseline (measured 2026-11-02 on this machine, `master` at `5108c26`)

`cargo llvm-cov --workspace --release --summary-only` (low-memory build flags):

| Metric | Covered |
|---|---|
| Lines | **98.19 %** (524 of 29,025 missed) |
| Functions | 96.50 % (112 of 3,198 missed) |
| Regions | 97.74 % (1,292 of 57,123 missed) |
| Branches | not measured (needs nightly) |

There are 79 files with a missed line. The worst ten by missed lines: `slm_wasm/src/lib.rs` 130, `slm_cli/src/info.rs` 20, `slm_cli/src/eval_gguf.rs` 14, `ferrum_core/src/token_file/reader.rs` 14, `ferrum_core/src/slm/stream.rs` 14, `ferrum_core/src/loss.rs` 13, `ferrum_core/src/slm/train.rs` 12, `slm_wasm/src/text.rs` 11, `ferrum_core/src/layer/transformer.rs` 11, `ferrum_core/src/gguf/mod.rs` 11. The full per-line list comes from `cargo llvm-cov report --release --show-missing-lines`. Phase 0 Task 0.1 writes it to `target/coverage/missing.txt`.

Under llvm-cov, 1,088 tests pass and 2 are ignored, against 1,090 in the normal suite. The difference is the 2 doc-tests, which llvm-cov on stable does not run. Nobody should chase this gap.

**Not measured yet:** `ferrum_gui` Rust (heavy Tauri build), the `ferrum_gui/ui/*.js` and `slm_wasm/demo/*.js` code, and `eval/lighteval/*.py`. Phase 0 measures them. No number for them may appear in a doc before that run.

## Global Constraints

Copied from `CLAUDE.md`. Every task's requirements include these.

- **GGUF only.** Every file Ferrum writes is GGUF v3. A new file kind is a new GGUF kind, never a custom binary format. BPE SLMs stay `general.architecture = "gpt2"` and keep running in llama.cpp.
- **`ferrum_core` has two external dependencies** (`unicode-general-category`, `unicode-normalization`). Add no others. Benchmarks are std-only (`harness = false`). A new dependency anywhere needs four things: a benchmark gain (or more code replaced than added), a passing determinism test, a working `wasm32-unknown-unknown` build of `ferrum_core`, and `cargo deny check`. **Any task that needs a new dependency stops and asks the user** (see "Decisions for the user").
- **`#![forbid(unsafe_code)]`** in every crate root. No SIMD intrinsics, and no direct call to a dependency's `unsafe fn`.
- **Deterministic at any thread count.** Results are bit-identical under any `FERRUM_NUM_THREADS`. The data-parallel trainer is the one documented exception. `cargo test -p slm_cli --release --test determinism` must pass after every kernel or trainer change.
- **WASM-portable.** `ferrum_core` and `slm_wasm` build for `wasm32-unknown-unknown`, with no threads or fs on that path.
- **Backward compatibility.** `TransformerConfig` (and `SamplingParams`) grow by new fields plus `Default`. The positional `train_transformer*` functions keep stride 1 and their exact results. Record every compat change in `Status.md` §7.
- **Scoring semantics.** `evaluate()`/`--val` use `score_rolling`, `temperature == 0` is exact argmax, and `<|endoftext|>` is always reserved.
- **Low-memory machine (15 GB).** Prefix every cargo command with `CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`. Run one cargo command at a time, and **never run `llvm-cov` together with tests**.
- **Stay inside the project.** Scratch files go in the session scratchpad. Tools are installed with `--root <scratchpad>`, never into `~/.cargo` or `~/.cache`.
- **One commit per completed feature.** A feature is complete when it has tests, a green full suite, docs, and `Status.md` (snapshot, test counts, known issues, roadmap, §7) updated in the same commit. Use Conventional subjects. Push after every green suite, fast-forward merge to `master`, no PRs, no merge commits.
- **No invented output.** Every number written into a doc comes from a run on this machine. Say plainly what was not run.
- **Docs are fixed:** `ReadMe.md`, `Manual.md`, `Status.md`, `Samples.md`, `glossary.md`, `docs/SLM-edge-use-cases.md`. Add no others. (Plans under `docs/superpowers/plans/` are agent records, as this file is.)
- **New CLI flags** go in the per-subcommand `*_FLAGS` tables in `slm_cli/src/args.rs`.
- **Deliberate simplifications** are marked `// ponytail:`, naming the ceiling and the upgrade path.
- **`ferrum_gui` is outside the workspace.** Build and test it from `ferrum_gui/`. Use `node --check` for every `ui/*.js`. After changing what the UI sends, refresh the fixtures with `UPDATE_FIXTURE=1 node ui/live.test.js` and `UPDATE_FIXTURE=1 node ui/app_flows.test.js`.
- **`ferrum_core` public API changes can break `../ferrum-ml`.** Rebuild it (`cd ../ferrum-ml && cargo test --release`) after any public API change, and commit there separately.

### What "100 % coverage" means here

1. **Gate:** `cargo llvm-cov --workspace --release --fail-under-lines 100 --ignore-filename-regex '<EXCLUDE>'`. Lines are gated. Functions and regions are reported but not gated, because generic monomorphs and `?` regions produce unreachable regions on stable. Branch coverage needs nightly, so it is neither gated nor claimed.
2. **Exclusions** (stable has no `#[coverage(off)]`) live in one regex, in `scripts/coverage.sh`, and each one is named here with its reason:
   - `ferrum_gui/src/main.rs`. This is the Tauri entry point. The CI boot smoke test runs it. It is excluded from Task 0.1 on.
   - `slm_wasm/src/lib.rs`. It is excluded **only from Task 1.1 on**, in the same commit that leaves it holding nothing but `#[wasm_bindgen]` shims that forward to native-tested functions. The Node `wasm-bindgen-test` suite (`slm_wasm/tests/node.rs`) runs them. Excluding it earlier would hide 130 lines and let Task 1.1 be skipped.
   - Nothing else. Each `test_support.rs` is **not** excluded: a missed helper line is dead test code and gets deleted.
3. **Dead code is deleted, not tested.** When a missed line cannot be reached from any public input, remove it, and say so in the commit message. Candidates spotted in the baseline: the `arch`/`name` fields `slm_cli/src/runtime_export.rs:10` marks as read only by tests, and the `rng.rs:65-71` block.
4. **JS** gate: `node --test --experimental-test-coverage --test-coverage-lines=100` over `ferrum_gui/ui/*.test.js`, run from `ferrum_gui/`, and over `slm_wasm/demo/*.test.js`, with `--test-coverage-exclude='**/*.test.js'`.
   - **Known blind spot (checked 2026-11-02):** `ui/testapp.js:54` loads `app.js` and the `*_tab.js` modules with `vm.runInContext(read(f), sandbox, { filename: f })`. Because `filename` is a bare name, V8 coverage attributes nothing to those files. A run over `ui/app_flows.test.js` listed only `testapp.js` and `testdom.js`.
   - Task 0.1 therefore passes `filename: path.join(__dirname, f)`.
   - The script also fails when any `ui/*.js` or `demo/*.js` (other than tests) is **missing** from the report. Node lists only the files that were loaded, so a file no test touches would otherwise show nothing instead of 0 %.
5. **Python** gate: stdlib `python3 -m trace --count --missing --coverdir <scratch>` over the unittest run. Code that needs LightEval and is skipped without it is reported but not gated. It is checked locally with LightEval installed.

## Review Focus

These are the failure modes most likely to bite, with no current test. Each line names the test that pins it and the task that owns it.

1. **Coverage gate run on a debug build, or concurrently with `cargo test`.** The OOM risk on 15 GB is real. `scripts/coverage.sh` refuses to start while another `cargo` process runs (`pgrep -x cargo`) and always passes `--release`. The test `coverage_script_refuses_concurrent_cargo` goes in `tests/tests/test_coverage_script.rs` (Task 0.1).
2. **Deleting a "dead" line that is reachable through a hostile file.** Loader, GGUF and token-file branches look dead but guard malformed input. Before deleting any missed line under `gguf/`, `loader/`, `token_file/`, `tokenizer/gguf_io.rs`, `ckpt.rs`, write a malformed-input test that reaches it. Delete the line only if no byte sequence reaches it (Tasks 1.2, 1.3).
3. **A resumed run on a different corpus.** Today it trains silently on the wrong data. After Task 2.2 it must refuse with the two corpus ids in the message. The test is `resume_refuses_a_different_corpus` (Task 2.2).
4. **Cancelling generation mid-UTF-8 sequence.** A stop hook must not emit half a code point. The test `cancel_mid_multibyte_keeps_valid_utf8` uses a model whose byte tokens split "é" (Task 2.3).
5. **`prepare` interrupted between renames.** It must leave no `manifest.json`, so the output is detectably incomplete, and the next run must still succeed. The test is `failed_rename_leaves_no_manifest` (Task 2.4).

---

## Phase 0 — The measuring stick

### Task 0.1: One coverage script, CI gate that ratchets

**Files:**
- Create: `scripts/coverage.sh`, `tests/tests/test_coverage_script.rs`
- Modify: `.github/workflows/ci.yml` (new job `coverage`), `CLAUDE.md` (Commands block), `Manual.md` §13.6 (Reproducing), `Status.md` (header branch, §1 snapshot, §9 metrics)

**Interfaces:**
- Produces: `scripts/coverage.sh [--rust|--gui|--js|--py|--all] [--fail-under N]`. It writes `target/coverage/{summary.txt,missing.txt,gui-summary.txt,js.txt,py.txt}` and exits non-zero when lines fall below N (default `100`). Later tasks run `scripts/coverage.sh --rust --fail-under 100`.

- [ ] **Step 1: Write the failing tests** in `tests/tests/test_coverage_script.rs`. Run the script with `bash` from `CARGO_MANIFEST_DIR/..`, the same pattern as `test_bench_script.rs`. Replace `cargo` with a fake on `PATH`. Every test also needs a fake `pgrep`, which prints nothing by default, because the real `pgrep -x cargo` finds the `cargo test` running the suite:
  - `coverage_script_rejects_unknown_flag`: `--bogus` exits 2, and stderr contains `usage: scripts/coverage.sh`.
  - `coverage_script_refuses_concurrent_cargo`: with a fake `pgrep` that prints a pid, the script exits 1 with `another cargo process is running`.
  - `coverage_script_passes_release_and_low_memory_env`: a fake `cargo` that records its argv and env into a temp file. Assert that `llvm-cov`, `--release` and `--fail-under-lines 100` are present, along with `CARGO_BUILD_JOBS=3`.
- [ ] **Step 2: Run them.** `cargo test -p integration_tests --release --test test_coverage_script` gives FAIL (no script).
- [ ] **Step 3: Write `scripts/coverage.sh`** (`set -euo pipefail`; exports the three low-memory variables; `EXCLUDE='ferrum_gui/src/main\.rs'`; Task 1.1 adds `slm_wasm/src/lib\.rs`).
  - `--rust` runs `cargo llvm-cov --workspace --release --ignore-filename-regex "$EXCLUDE" --fail-under-lines "$N" --summary-only`, then `cargo llvm-cov report --release --show-missing-lines` into `missing.txt`.
  - `--gui` does the same inside `ferrum_gui/` (no `--release`, since the crate's dev profile already optimizes the engine).
  - `--js` runs the Node command from "What 100 % means" item 4, after the `filename: path.join(__dirname, f)` fix in `ui/testapp.js`. It fails when a non-test `ui/*.js` or `demo/*.js` is absent from the report. Check this works by confirming that `app.js` and each `*_tab.js` appear in `js.txt`.
  - `--py` runs `trace`.
  - `--all` runs them in sequence, never in parallel.
- [ ] **Step 4: Run the tests.** They PASS.
- [ ] **Step 5: Measure the unmeasured baselines.** Run `scripts/coverage.sh --gui --fail-under 0`, then `--js --fail-under 0`, then `--py --fail-under 0`, one at a time. Copy the four totals into `Status.md` §9 as measured on that date. These numbers fix the size of Tasks 1.5 and 1.6.
- [ ] **Step 6: Add the CI job `coverage`** (ubuntu, `taiki-e/install-action` for `cargo-llvm-cov`). The floor is set from the **first CI run** of this job, not from the local 98.19 %, because some covered lines depend on the environment:
  - `loader/gpt2.rs:632-633` runs only when `.ferrum-dev` exists.
  - Tool-gated runtime branches depend on whether the tools are installed.
  - Pool and shard paths depend on the core count.
  - `memory.rs` and GUI hardware detection read `/proc` and `/sys`.

  Run the job once with `--fail-under 0`, then set the floor to its reported total, rounded down. Tasks 1.1–1.4 raise the floor in their own commits, and Task 1.4 sets it to 100. Add the JS gate at its measured floor too. **A Phase 1 task is done when the CI `coverage` job passes at the new floor**, not when the local script does.
- [ ] **Step 7: Fix the stale `Status.md` header.** It names branch `feat/finish-remaining`. The work is on `master`. Correct the header and §1 "Date and branch"/"Commits" rows.
- [ ] **Step 8: Run the full suite**, then the coverage script on its own. Commit `build(coverage): coverage script and a ratcheting CI gate`.

---

## Phase 1 — 100 % line coverage

Each task follows the same loop:
1. Read the file's missed lines in `target/coverage/missing.txt`.
2. For each line, decide whether it is reachable. If it is, write the test that reaches it, named for the behaviour, not the line. If it is not, delete it.
3. Run the crate's tests, then `scripts/coverage.sh --rust --fail-under <new floor>` on its own.
4. Commit, with the floor raised in `ci.yml` in the same commit.

The test names below are the decisions. The implementer writes the assertions from the code at the given lines.

### Task 1.1: `slm_wasm` — thin shims, native-tested logic (130 + 11 missed)

**Files:**
- Create: `slm_wasm/src/token_model.rs` (the logic of `TransformerSLMModel`: `new`, `predict_next`, `prime`, `predict_next_cached`, attention weights, `sample_from_probs`, `entropy`, `top_k_indices`, as plain Rust returning `Result<_, String>`)
- Modify: `slm_wasm/src/lib.rs` (each `#[wasm_bindgen]` method becomes a one-line forward, mapping `String` errors to `JsError`/`JsValue` exactly as today), `slm_wasm/src/text.rs` (lines 77-80, 166-168)

**Interfaces:**
- Produces: `pub(crate) struct TokenModel` with the same method names and argument types as `TransformerSLMModel`. The JS-visible API and its error strings do not change, and `slm_wasm/tests/node.rs` must pass unchanged.

- [ ] Write tests in `token_model.rs`. Each must FAIL before the move and PASS after.
  - `predict_next_matches_library_forward`: probabilities equal `GenerativeSLM`'s forward pass for the same ids. This uses the existing `test_support.rs` model.
  - `prime_then_cached_equals_full_forward`
  - `predict_next_rejects_out_of_vocab_id`
  - `predict_next_rejects_empty_context`
  - `sample_from_probs_temperature_zero_is_argmax`
  - `sample_from_probs_bad_input_never_panics` (empty, NaN, all zeros)
  - `top_k_indices_k_larger_than_vocab`
  - `entropy_of_one_hot_is_zero`
  - `attention_weights_shape_is_heads_x_len`
- [ ] Write tests in `text.rs` for its missed lines. Read the lines and name each test for the error it reaches.
- [ ] In the same commit, add `slm_wasm/src/lib\.rs` to `EXCLUDE` in `scripts/coverage.sh`. Its reason is already recorded under "What 100 % means".
- [ ] Move the logic, then run `cargo test -p slm_wasm --release` (native). Build the wasm32 target. Run the Node suite with the runner installed in the scratchpad (`cargo install wasm-bindgen-cli --version 0.2.122 --locked --root <scratchpad>/wbcli`, then `PATH=<scratchpad>/wbcli/bin:$PATH CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo test -p slm_wasm --target wasm32-unknown-unknown --release --test node`). It must show 6 passed.
- [ ] Run the full suite. Run coverage alone and raise the floor. Commit `test(wasm): token-model logic moves out of the bindings and is tested natively`.

### Task 1.2: `ferrum_core` file formats and loaders

Files (missed lines from the baseline):
- `token_file/reader.rs` 14, `token_file/writer.rs` 6
- `gguf/mod.rs` 11, `gguf/tensors.rs` 9, `gguf/llama.rs` 5, `gguf/header.rs`, `gguf/cursor.rs`, `gguf/num_array.rs`
- `loader/decode.rs` 5, `loader/gpt2.rs` 5, `loader/mod.rs` 4, `loader/test_support.rs` 5
- `gguf_write/llama.rs` 4, `gguf_write/builder.rs` 3
- `tokenizer/gguf_io.rs` 3, `gguf_tokenizer.rs` 5
- `meta.rs` 7, `ckpt.rs`

- [ ] Write one malformed-input test per reachable missed line, in the file's own `tests` module, through the public entry point (`Gguf::open`, `TokenFileReader::open`, `GenerativeSLM::load`, ...):
  - token-file reader: `token_file_reader_truncated_tensor`, `token_file_reader_wrong_dtype`, `token_file_reader_seek_past_end`
  - token-file writer: `token_file_writer_io_error_on_finish` (a writer that fails on the Nth write)
  - GGUF reader: `gguf_open_tensor_offset_overflow`, `gguf_open_misaligned_data_section`
  - loader: `loader_rejects_q8_0_with_ragged_row`
  - meta: `meta_roundtrip_every_task_type`
- [ ] Review `loader/gpt2.rs:632-633`. That is the optional foreign-GPT-2 test path under `.ferrum-dev`: it is covered only when the file exists. Make the test always take one branch, by replacing the skip-with-print with a `#[ignore]`d test plus a covered helper, and record the choice in the commit.
- [ ] Delete what no input reaches. Run `cargo test -p ferrum_core --release` and the fuzz-style hostile-input tests that already exist in these modules. Run coverage alone and raise the floor. Commit `test(core): every file-format error path has a test`.

### Task 1.3: `ferrum_core` engine, training and generation

Files (missed lines from the baseline):
- generation: `slm/stream.rs` 14, `slm/token_stream.rs`, `slm/generate.rs`, `slm/sampling.rs`, `slm/scoring.rs`
- SLM training: `slm/train.rs` 12, `slm/train_tokens.rs` 8, `slm/persist.rs` 8, `slm/corpus.rs`, `slm/training_loop.rs`, `slm/cached.rs`
- numerics and layers: `loss.rs` 13, `rng.rs` 9, `optim.rs` 9, `ops.rs` 8, `quant.rs`, `activation.rs`, `layer/transformer.rs` 11, `layer/linear.rs` 6, `layer/norm.rs`, `layer/embedding.rs`
- runtime: `parallel.rs` 8, `stop.rs` 7, `verbose.rs`
- transformer trainer: `train_transformer/{backward,epoch,export,optimizer,threaded,checkpoint}.rs`
- other: `tokenizer/{encode,train,test_support}.rs`, `llm.rs` 4, `dataset.rs`

- [ ] Tests by behaviour:
  - parallel: `parallel_pool_with_one_worker_matches_serial`, `parallel_worker_panic_reported_once`
  - stop: `stop_condition_stop_file_accessor_and_clear` (the `stop.rs:80-82` accessor)
  - loss: `loss_handles_all_masked_targets`
  - persist: `persist_save_to_unwritable_dir_errors`
  - SLM training: `train_budget_zero_tokens_is_an_error`, `train_tokens_rejects_vocab_mismatch`
  - stream: `stream_reprime_at_exact_window_boundary`
  - sampling: `sampling_top_p_one_equals_disabled`
  - optimizer: `optimizer_adamw_zero_decay_equals_adam`
  - trainer: `threaded_epoch_single_shard_matches_serial`
- [ ] For `rng.rs:65-71`, either test it through its public caller or delete it.
- [ ] Run `cargo test -p ferrum_core --release` and `cargo test -p slm_cli --release --test determinism`. Run coverage alone and raise the floor. Commit `test(core): engine paths reach 100 % line coverage`.

### Task 1.4: `slm_cli` and `slm_server`, gate at 100

Files (missed lines from the baseline):
- `slm_cli`: `info.rs` 20, `eval_gguf.rs` 14, `runtime_export.rs` 10, `finetune.rs` 8, `prepare/input.rs` 8, `tokenize.rs` 7, `token_corpus.rs` 6, `run_gguf.rs` 5, `export.rs` 5, `prepare/mod.rs` 5, `main.rs` 4, `train.rs` 4, `generate.rs`, `outputs.rs`, `prepare/split.rs`, `stop.rs`, `memory.rs`, `eval.rs`, `cmd_prepare.rs`
- `slm_server`: `model_thread.rs` 4, `serve.rs` 3, `cli.rs`, `auth.rs`, `http.rs`

- [ ] Tests:
  - info: `info_reports_dominant_quant_for_every_ggml_type`. This is table-driven over each type in `info.rs:15-29`: build one tiny GGUF per type with `GgufBuilder`, then assert the printed label.
  - eval-gguf: `eval_gguf_with_checkpoint_and_precision` and `eval_gguf_refuses_over_ram_budget` (binary-level, in `tests/tiny_gguf_e2e.rs`).
  - token corpus: `token_corpus_rejects_non_token_gguf`.
  - server: `server_model_thread_survives_two_panics`.
- [ ] Remove the dead `arch`/`name` fields in `runtime_export.rs`, along with their `ponytail:` line, if nothing outside tests reads them.
- [ ] **Functional matrix guard:** add `tests/tests/test_cli_matrix.rs` with two tests.
  - `every_subcommand_has_a_binary_test`: read the canonical names from the match arms of `spec_for` in `slm_cli/src/args.rs`, taking the first quoted string of each arm: `train`, `run`, `generate`, `run-gguf`, `finetune-gguf`, `export-gguf`, `export-ollama`, `export-lmstudio`, `prepare`, `tokenize`, `eval`, `eval-gguf`, `info`. Assert that each appears as a quoted argument in some `slm_cli/tests/*.rs`. All 13 do today (checked 2026-11-02). The guard keeps a new subcommand from landing without one.
  - `every_route_has_an_http_e2e_test`: read the `.route("…"` paths in `slm_server/src/http.rs` and check that each occurs in `slm_server/tests/tiny_http_e2e.rs`.
- [ ] Set the CI floor to **100** in this commit. Run the full suite, then coverage alone. It must exit 0 at 100. Commit `test(cli,server): 100 % line coverage, gated in CI`.

### Task 1.5: `ferrum_gui` Rust to 100 %

Its size is set by Task 0.1 Step 5. Do not start before that number exists.

- [ ] Work file by file from `target/coverage/gui-missing.txt`, using the same loop as above. Headless IPC tests use Tauri's mock runtime (`tauri` feature `test`, already a dev-dependency).
- [ ] **Command matrix guard:** add the test `every_command_has_an_ipc_test` in `ferrum_gui/src/ipc_tests.rs`. It reads the `generate_handler!` list in `lib.rs` (the same source `ui/commands.test.js` uses) and asserts that each command name occurs in an IPC test.
- [ ] Add `scripts/coverage.sh --gui --fail-under 100` to the CI `gui` job. Run the GUI checks from CLAUDE.md Commands. Commit `test(gui): 100 % backend line coverage, gated in CI`.

### Task 1.6: UI JavaScript and Python to 100 %

- [ ] From `target/coverage/js.txt`, add assertions to the existing `ui/*.test.js` and `slm_wasm/demo/*.test.js` against the stub DOM. Add no test framework: the files keep running as `node file.test.js` and under `node --test`.
- [ ] From `target/coverage/py.txt`, add `unittest` cases to `eval/lighteval/test_*.py` for the uncovered lines that do not need LightEval.
- [ ] Raise both CI gates to 100. Commit `test(ui,eval): 100 % line coverage for the UI scripts and the Python module`.

---

## Phase 2 — Known-issue gaps (Status.md §5, `ponytail:` markers)

Every task here changes behaviour. Each task writes three kinds of test:
- **unit**, in the module
- **functional**, through the real binary or the IPC path
- **integration**, across crates, in `tests/tests/` when more than one crate is involved

Each task also updates `Status.md` §5 (remove or rewrite the issue), §7 if a format changes, and the matching `Manual.md` section.

### Task 2.1: Exact mid-epoch resume

The marker is `ferrum_core/src/slm/training_loop.rs:187`: "a resumed run reshuffles its partially-done epoch".

**Files:** `ferrum_core/src/train_transformer/checkpoint.rs`, `ferrum_core/src/slm/training_loop.rs`, `ferrum_core/src/ckpt.rs`; Test: same modules plus `tests/tests/test_tiny_library.rs`

**Interfaces:**
- Produces: a new checkpoint key `ferrum.ckpt.epoch_rng` (u64 array, the RNG state at the start of the current epoch). A checkpoint without it resumes as today, with the reshuffle and a note. This is backward compatible; record it in §7.

- [ ] Unit: `checkpoint_roundtrips_epoch_rng`, and `old_checkpoint_without_epoch_rng_still_loads`, which uses a checkpoint written by the current code and committed as test bytes.
- [ ] Integration: `resume_mid_epoch_is_bit_identical`, in `test_tiny_library.rs`. Train N steps with a checkpoint at step N/2 + 3 (not an epoch boundary), resume, and compare the model bytes with an uninterrupted run, both serial and with `threads = 2`. Add `resume_mid_epoch_is_bit_identical_from_token_file` next to it: `train tokens.gguf` shuffles its windows in `train_transformer/source.rs`, and `Status.md` claims token-file training is bit-identical to in-memory training across a resume.
- [ ] Implement it: store the epoch-start RNG state. On resume, regenerate that epoch's permutation and skip the `partial` windows already done. Remove the `ponytail:` marker.
- [ ] Functional: extend the resume case in `slm_cli/tests/tiny_e2e.rs` to compare the sha256 of the interrupted and the uninterrupted model.
- [ ] Docs: `Status.md` §5 row "Resume may not replay…" goes, §7 gains a note, and `Manual.md` §6.6 changes. Commit `feat(train): exact mid-epoch resume`.

### Task 2.2: Corpus identity in checkpoints

The marker is `slm_cli/src/resume.rs:34`.

**Interfaces:**
- Produces: `TransformerConfig.corpus_id: Option<String>` (Default `None`). It is stored as `ferrum.ckpt.corpus_id`, and resume compares it when both sides have one. `slm_cli` fills it with `sha256:<hex>` of the corpus file (text or token file) using the existing `slm_cli::hash::sha256_hex`. `ferrum_core` treats it as opaque, so no hashing dependency enters the core.

- [ ] Unit (core): `resume_refuses_a_different_corpus`. The error contains both ids and the word `corpus`. Also `resume_without_corpus_id_on_either_side_proceeds`.
- [ ] Functional: `resume_on_other_corpus_exits_1` in `tiny_e2e.rs`, with the binary.
- [ ] GUI: the Train tab passes the same id (`train_slm` in `ferrum_gui/src/commands/train.rs`), and the IPC test `train_resume_other_corpus_is_refused` checks it.
- [ ] Implement, remove the marker and the "not checked" note, and add the docs and §7 entries. Commit `feat(train): checkpoints record the corpus and resume checks it`.

### Task 2.3: A cancel hook in the decode loops

The marker is `ferrum_gui/src/commands/generate.rs:131`: the GUI cancels by `resume_unwind`.

**Interfaces:**
- Produces: `SamplingParams.cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>` (Default `None`; `PartialEq` ignores it, so implement `PartialEq` by hand over the other fields). Every decode loop (`generate_with`, `generate_stream_with`, `generate_tokens_with`, and the stream types `token_stream`/`stream_next`) check it once per token. On cancel they stop at the last complete UTF-8 boundary and return what they have, with `Ok`. `Generation` (returned by `generate_tokens_with`, built by `GenStream::into_generation`) gains `cancelled: bool`. `stream_next` returns `Ok(None)` once cancelled.

- [ ] Unit:
  - `cancel_before_first_token_returns_seed_only`
  - `cancel_after_k_tokens_returns_k_tokens`
  - `cancel_mid_multibyte_keeps_valid_utf8`
  - `no_cancel_flag_is_bit_identical_to_today` (same text and same RNG draws as `generate_tokens` on the tiny BPE and char models)
- [ ] GUI: replace `catch_unwind`/`resume_unwind` in `run_generation` with the flag. The existing IPC cancel tests must pass unchanged. Remove the marker.
- [ ] Server (optional consumer): none now. Note it in Manual §9.7 as possible.
- [ ] Struct literals: run `grep -rn 'SamplingParams {' --include=*.rs .` over the workspace, `ferrum_gui` and `../ferrum-ml`. Confirm that `SamplingParams` implements `Default` (it derives only `Clone, Debug, PartialEq` today, `slm/sampling.rs:17`, so check its constructors). Then fix every literal that lacks `..Default::default()` in this commit, and record it in `Status.md` §7 if it is a public break.
- [ ] Wasm: `Slm::start`/`next_chunk` go through `stream_next`. Build `slm_wasm` for wasm32 and run the Node `wasm-bindgen-test` suite (6 passed). The cancel check reads only the `AtomicBool`, never the clock (`Instant::now` traps on `wasm32-unknown-unknown`). That is why this hook is a plain flag and not a `StopCondition`.
- [ ] Commit `feat(generate): cooperative cancel in every decode loop`.

### Task 2.4: `prepare` output is never half-written

The marker is `slm_cli/src/prepare/mod.rs:340`.

- [ ] Unit: `failed_rename_leaves_no_manifest`. Factor the rename loop into `fn commit_outputs(out: &Path, rename: &dyn Fn(&Path, &Path) -> std::io::Result<()>) -> Result<(), String>`, and pass a `rename` that fails on the third call. (A directory pre-created at `out/test.txt` would be refused earlier by `slm_cli::outputs::check_writable`, which rejects directories, so it never reaches the renames.) Assert that the call errors, that `out/manifest.json` does not exist, and that a second run after removing the obstacle succeeds with byte-identical outputs.
- [ ] Implement: remove an existing `out/manifest.json` **first**, rename the data files, and rename the manifest **last**. A present manifest then means a complete set. Document "the manifest is the commit marker" in Manual §5.2, and keep the `ponytail:` marker rewritten to that guarantee.
- [ ] Functional: the DVC `prepare` output stays byte-identical. Run `.venv/bin/dvc repro prepare` if the venv exists, or say it was not run. Commit `fix(prepare): the manifest is written last and marks a complete output`.

### Task 2.5: GUI progress events for downloads and *Clean* (rest of G10)

**Interfaces:**
- Produces: the events `download-progress {id, bytes, total: Option<u64>}` and `clean-progress {bytes, documents}`, emitted at most every 100 ms. The UI shows them in the existing progress bar component.

- [ ] Rust: `download_emits_monotonic_progress_and_a_final_event` (in-memory reader, the same pattern as the existing `datasets` tests). `clean_emits_progress`.
- [ ] JS: the flow `download_progress_updates_bar` in `ui/app_flows.test.js`. Refresh the fixtures with `UPDATE_FIXTURE=1`.
- [ ] Commit `feat(gui): progress for downloads and Clean`.

### Task 2.6: Docs that run

- [ ] Re-run every command in `Samples.md` that the 2026-11-02 path move touched (§7 Ollama/LM Studio output paths), paste the real output, and mark anything not re-run.
- [ ] Commit `docs(samples): re-run after the directory move`.

---

## Phase 3 — Roadmap modules (Status.md §6.2)

Each item below is a subsystem. **Before starting one, write its own plan** with this skill, using the item below as its spec seed. Each one lands under the same rules: unit, functional and integration tests, 100 % on the new lines (the gate enforces it), and docs plus `Status.md` in the commit. They are ordered by dependency.

| # | Module | Files (main) | Interface seed | Tests that prove it |
|---|---|---|---|---|
| 3.1 | **Gradient accumulation** | `train_transformer/epoch.rs`, `threaded.rs`, `slm/config.rs`, `slm_cli/src/args.rs` (`--accum N`) | `TransformerConfig.grad_accum: usize` (Default 1, meaning today's behaviour bit-for-bit) | `accum_1_is_bit_identical`; `accum_k_batch_b_equals_batch_kb` (within f32 tolerance, same seed); determinism test at 1 and 8 threads; CLI e2e |
| 3.2 | **Metrics columns** (val loss, perplexity, `--metrics-every N`) | `slm/metrics_log.rs`, `slm_cli` train | Two CSV columns, appended (old readers keep working) | `metrics_csv_has_val_columns_only_with_val`; `metrics_every_thins_rows` |
| 3.3 | **Packed Q8_0 matmul** (no f32 expansion on load) | `ops.rs` (`qlinear` family), `loader/decode.rs` (the `ponytail:` at line 99) | Q8_0 tensors stay packed in `QWeight`; f32 expansion only on request | `q8_0_packed_forward_equals_expanded` (bit-identical); RAM test: loaded model ≤ file size × 1.3; llama.cpp compat test still passes; `cargo bench --bench gemm` before and after in the commit message |
| 3.4 | **F6 safe-SIMD decode** (autovectorization only, no intrinsics) | `ops.rs` | No API change | Bit-identical outputs; determinism test; a bench row in Manual §13.4 from a real run, kept only if faster |
| 3.5 | **`LlamaModel::init`** (train a Llama from scratch) | `llm.rs`, `llm_train/`, `slm_cli` (`train-llama` or `train --arch llama`; choose in its plan) | `LlamaModel::init(cfg: &LlamaConfig, rng: &mut Rng) -> LlamaModel` | Gradient check on a fresh init; train tiny → export GGUF → `run-gguf` equals in-memory; llama.cpp greedy continuation equals Ferrum's (runtime compat test) |
| 3.6 | **LoRA adapters** | `llm_train/lora.rs` (new) | `LoraConfig { rank, alpha, targets }`; adapters saved as GGUF kind `ferrum-lora` | Merged-weights forward equals adapter forward; gradient check; RAM test that a 0.5B fine-tune plan fits 16 GB (estimate only, run on SmolLM2-135M) |
| 3.7 | **SFT** (chat template + prompt-loss masking, native models) | `slm/train.rs`, `chat_format.rs` | `TransformerConfig.loss_mask: LossMask` (Default `All`) | Masked tokens contribute zero gradient (gradient check); `All` bit-identical to today |
| 3.8 | **Distillation from a GGUF teacher** | `slm/distill.rs` (new) | `distill(student_cfg, teacher: &LlamaModel or GenerativeSLM, corpus, T, alpha)` | KL term zero when student = teacher; tiny run lowers KL; vocab mismatch refused |
| 3.9 | **F3 `qwen3` architecture** | `gguf/llama.rs`, `llm.rs` | Per-head q/k RMSNorm, explicit `head_dim` | Synthetic tiny qwen3 GGUF forward equals a hand-computed reference; llama.cpp compat if a qwen3 file is available locally (else say not run) |
| 3.10 | **F5 streamed GGUF export** | `gguf_write/` | Two-pass writer, checkpoint overlay without loading whole | Byte-identical to the in-memory export; peak RAM bounded (test with a counting allocator stand-in: track bytes via the writer, not a global allocator) |
| 3.11 | **Exact SPM encode** (unigram Viterbi) | `gguf_tokenizer.rs` | `encode_spm_exact` replaces greedy | Ids equal llama.cpp `llama-tokenize` on a fixture set (recorded from a real run, committed) |
| 3.12 | **IQ\* quants (read)** | `gguf/dequant.rs` | Decoders for IQ4_NL/IQ4_XS first | Dequant equals ggml reference values recorded from llama.cpp |
| 3.13 | **Per-block import quantization** | `gguf_write/quant.rs`, import path | Policy option | Error vs source lower than per-row; round trip |
| 3.14 | **Size-aware defaults** remainder | `slm/config.rs` | Already shipped as `--preset`; only close the item in Status if nothing remains | — |
| 3.15 | **A3 full-size DVC profile** (TinyStories 2 GB) | `params.yaml` (a second profile), `dvc.yaml` | `dvc repro -P full` or a params file switch | Runs only with the download present; a small "profile parses" test always runs; the real run's numbers go in Status §3 |

---

## Not planned (with the reason)

- **Concurrent serving.** `GenerativeSLM` is not `Send`/`Sync`. Concurrency needs a model-sharing design (immutable weights + per-request KV cache) of its own. It goes in a separate design spec first.
- **Retiring the native GPT-2 stack after `LlamaModel::init`.** The `gpt2` GGUF that runs in llama.cpp is a stated selling point.
- **Mobile builds (G16).** No Android or iOS SDK on this machine, so nothing could be run, and rule 6 forbids claiming it.
- **Code signing and an updater.** These need keys and accounts the project does not hold.
- **A matmul crate (ndarray/faer).** This is an open user decision (below). `faer` already failed determinism.

## Decisions for the user (each blocks only its own item)

1. **Checkpoint on `SIGINT`** (Status §6.2 P1). Catching a signal needs either `unsafe` (forbidden) or a dependency (`ctrlc` in `slm_cli`), which goes through the dependency rules. Is `ctrlc` acceptable? Without it, the current stop file and `--max-seconds` stay the graceful path.
2. **Real-webview click tests for the GUI.** `tauri-driver` + WebDriver would be new dev tooling, and the only way to retire the "not click-tested" rows in §5. Is that worth adopting?
3. **Python coverage tool.** Stdlib `trace` is the default here. `coverage.py` would be a new pinned dev dependency, with better reports.
4. **The matmul crate decision** (Status §5, ndarray +34–40 % train tokens/s, first hot-path dependency).
5. **Branch coverage.** It needs a nightly toolchain in CI. Should the gate stay at lines only?
