<p align="center"><img src="../../assets/logo.png" width="96" alt="Ferrum logo"></p>

# Closing Remaining Functionality & Front-End Parity Gaps — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **One agent at a time** (CLAUDE.md working rule 2): subagent-driven means strictly sequential, never parallel.

**Goal:** Close the user-visible front-end parity and functionality gaps identified in the 2026-11-07 repository analysis (`docs/GAP-REPORT.md` H4, H5, M14, M15, M16, L17, plus M5/L11 residue), bringing `ferrum_gui`, `slm_cli`, and `slm_server` to full parity without violating any `CLAUDE.md` hard constraint.

**Architecture:** Seven independent, modular tasks ordered by user impact. Each task touches a single concern, adds unit + functional + integration tests (per rules 12, 13, 19, 20), refreshes GUI IPC fixtures where applicable, updates `MANUAL.md` and `STATUS.md` in the same commit, and passes the full workspace and `ferrum_gui` test suites before committing (rules 1, 5, 11).

**Tech Stack:** Rust 2021 (MSRV 1.87, `#![forbid(unsafe_code)]`), Tauri v2 mock-runtime IPC tests (`ferrum_gui`), dependency-free vanilla ES modules (`ferrum_gui/ui/`), Node 24 test suite (`ferrum_gui/ui/tests/*.test.js`).

**Spec & Sources:**
- `docs/GAP-REPORT.md` (2026-11-07 update: H4, H5, M5, M14, M15, M16, L11, L17)
- `STATUS.md` (§5 known issues, §6.2 roadmap, §7 compatibility)
- Companion UX plan: `docs/superpowers/plans/2026-11-07-GUI-BEGINNER-ADVANCED-MODE.md` (covers H5's Beginner/Advanced mode and guided workflow in detail; Task 2 below covers the backend/CLI preset sharing and optional train flags).

## Global Constraints (from `CLAUDE.md`)

- **GGUF only** for all persisted artifacts.
- **`ferrum_core` has two external dependencies** (`unicode-general-category`, `unicode-normalization`); add no others.
- **`#![forbid(unsafe_code)]`** across all crates.
- **Deterministic at any thread count** (`cargo test -p slm_cli --release --test determinism`).
- **Low-memory machine (15 GB):** Always build/test with `CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`, one cargo command at a time, and never run `llvm-cov` alongside `cargo test`.
- **Fixture discipline in `ferrum_gui`:** Whenever a tab module (`*_tab.js`) or `live.js` changes the payload sent to `invoke(...)`, run `UPDATE_FIXTURE=1 node ui/tests/live.test.js` and `UPDATE_FIXTURE=1 node ui/tests/app_flows.test.js` before `cargo test` in `ferrum_gui/`.
- **Tooltip guard discipline:** Every interactive element (`input`, `select`, `textarea`, `button`) added to `ferrum_gui/ui/index.html` must have a matching entry in `ferrum_gui/ui/core/help.js` with `>= 30` words across `what`/`how`/`beginner`, verified by `node ui/tests/tooltips_guard.test.js`, `node ui/tests/help.test.js`, and `node ui/tests/a11y_html.test.js`.

---

## Task 1: GUI Serve Tab Parity with `ferrum-serve` (Gap H4)

**Problem:** `slm_server` supports serving imported `llama`/`qwen2` GGUFs with `--quant f32|int8|int4`, `--force` (skip the `/proc/meminfo` RAM guard), and `--repetition-penalty <f>` (`slm_server/src/cli.rs:41-54`). However, `ferrum_gui/src/serving/serve.rs` (`ServeParams`, lines 34–46, and `to_config`, lines 68–80) hard-codes `quant: Quant::Int4`, `force: false`, and `repetition_penalty: 1.0`. Users serving an imported GGUF from the GUI Serve tab cannot select `int8`/`f32`, override the RAM guard, or set a default repetition penalty.

**Files:**
- Modify: `ferrum_gui/src/serving/serve.rs`
- Modify: `ferrum_gui/ui/index.html` (Serve tab panel `#panel-serve`, lines 504–545)
- Modify: `ferrum_gui/ui/tabs/serve_tab.js`
- Modify: `ferrum_gui/ui/core/help.js` (add tooltip entries for `svQuant`, `svRepPenalty`, `svForce`)
- Modify: `ferrum_gui/src/ipc_tests.rs`, `ferrum_gui/ui/tests/app_flows.test.js`, `ferrum_gui/ui/tests/settings.test.js`
- Modify: `docs/GUI-TOOLTIPS.md`, `MANUAL.md` §10.3, `STATUS.md` §5

**Interfaces:**
- Extend `ServeParams` in `ferrum_gui/src/serving/serve.rs` with `#[serde(default)]` fields so existing callers remain backward-compatible:
  ```rust
  #[serde(default)]
  pub quant: Option<String>,
  #[serde(default)]
  pub force: bool,
  #[serde(default = "default_rep_penalty")]
  pub repetition_penalty: f32,
  ```
- Parse `quant` (`"int4"` default, `"int8"`, `"f32"`, case-insensitive; reject unknown values with a clear error before binding). Validate `repetition_penalty` (`is_finite() && > 0.0`, matching `slm_server/src/cli.rs:144`).

**Steps:**
- [ ] **Step 1: Write failing Rust unit & IPC tests** in `ferrum_gui/src/serving/serve.rs` (`#[cfg(test)]`) and `ferrum_gui/src/ipc_tests.rs`:
  - `to_config_maps_quant_force_and_rep_penalty`: verifies `"int8"` -> `Quant::Int8`, `"f32"` -> `Quant::F32`, `None` -> `Quant::Int4`, `force: true`, and `repetition_penalty: 1.15`.
  - `to_config_rejects_invalid_quant_and_nonpositive_rep_penalty`: verifies `"int2"` and `repetition_penalty: 0.0` return `Err`.
- [ ] **Step 2: Implement `ServeParams` and `to_config` changes** in `ferrum_gui/src/serving/serve.rs`.
- [ ] **Step 3: Add UI controls** in `ferrum_gui/ui/index.html` (`#panel-serve`):
  - `<select id="svQuant">` (`int4 (default, imported GGUF)`, `int8 (imported GGUF)`, `f32 (imported GGUF)`)
  - `<input id="svRepPenalty" type="number" value="1.0" min="0.1" max="10" step="0.05">`
  - `<input type="checkbox" id="svForce">` (`Skip RAM check (--force, imported GGUF)`)
- [ ] **Step 4: Wire `ferrum_gui/ui/tabs/serve_tab.js`** to pass `quant`, `repetitionPenalty`, and `force` in `serve_start`, and add help entries in `ferrum_gui/ui/core/help.js`.
- [ ] **Step 5: Refresh fixtures & run JS/GUI test suites:**
  - `cd ferrum_gui && UPDATE_FIXTURE=1 node ui/tests/app_flows.test.js`
  - `for f in $(find ui -name '*.js'); do node --check "$f" || exit 1; done && for t in ui/tests/*.test.js; do node "$t" || exit 1; done`
  - `CARGO_BUILD_JOBS=3 cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt -- --check`
- [ ] **Step 6: Update `MANUAL.md` §10.3, `docs/GUI-TOOLTIPS.md`, and `STATUS.md` §5.** Commit: `feat(gui): expose quant, repetition penalty, and force on the Serve tab`.

---

## Task 2: Shared Model Size Presets & Train Tab CLI Parity (Gap H5)

**Problem:** `slm_cli/src/preset.rs` defines `--preset tiny|small|medium` (`tiny`: context 64, embed 64, heads 4, blocks 2, hidden 256, vocab 512, lr 1e-2; `small`: 128/128/4/4/512/2048/3e-3; `medium`: 256/256/8/6/1024/4096/1e-3), but `preset.rs` is private to the `slm_cli` binary (`slm_cli/src/main.rs`), and the GUI Train tab has neither a preset selector nor `--max-seconds` / `--metrics-csv`.

**Files:**
- Modify: `slm_cli/src/lib.rs` (export `pub mod preset;`), `slm_cli/src/preset.rs`
- Modify: `ferrum_gui/src/commands/train.rs` (accept optional `max_seconds: Option<u64>` and `metrics_csv: Option<String>` in `TrainParams`, validate `metrics_csv` path via `check_writable`)
- Modify: `ferrum_gui/ui/index.html` (`#trPreset` select, `#trMaxSeconds`, `#trMetricsCsv`)
- Modify: `ferrum_gui/ui/tabs/train_tab.js`, `ferrum_gui/ui/core/help.js`
- Modify: `ferrum_gui/src/ipc_tests.rs`, `ferrum_gui/ui/tests/app_flows.test.js`
- Modify: `MANUAL.md` §10.3, `docs/GUI-TOOLTIPS.md`, `STATUS.md` §5

**Interfaces:**
- In `slm_cli::preset`:
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub struct PresetSpec {
      pub name: &'static str,
      pub context_length: usize,
      pub embed_dim: usize,
      pub num_heads: usize,
      pub num_blocks: usize,
      pub hidden_dim: usize,
      pub vocab_size: usize,
      pub learning_rate: f32,
  }
  pub fn lookup(name: &str) -> Result<PresetSpec, String>;
  ```
- In `ferrum_gui/src/commands/train.rs`: `TrainParams` gains `#[serde(default)] pub max_seconds: Option<u64>` and `#[serde(default)] pub metrics_csv: Option<String>`.

**Steps:**
- [ ] **Step 1: Expose `pub mod preset` from `slm_cli/src/lib.rs`** and add a unit test in `ferrum_gui` verifying that the GUI preset table matches `slm_cli::preset::lookup` for `"tiny"`, `"small"`, and `"medium"`.
- [ ] **Step 2: Add `max_seconds` and `metrics_csv` to `TrainParams`** in `ferrum_gui/src/commands/train.rs`, wiring `max_seconds` into `StopCondition` and `metrics_csv` into `TransformerConfig.metrics_csv` (validated with `paths::clean` and writable-parent check). Add unit and IPC tests (`train_slm_writes_metrics_csv`, `train_slm_rejects_zero_max_seconds`).
- [ ] **Step 3: Add `#trPreset`, `#trMaxSeconds`, and `#trMetricsCsv`** to `ferrum_gui/ui/index.html` and wire `trPreset` in `ferrum_gui/ui/tabs/train_tab.js` so choosing `tiny`, `small`, or `medium` populates `trContext`, `trEmbed`, `trHeads`, `trBlocks`, `trHidden`, `trVocab`, and `trLr` and calls `schedulePreflight()`, while editing any of those 7 inputs switches `trPreset` to `custom`.
- [ ] **Step 4: Add help entries in `ferrum_gui/ui/core/help.js`**, refresh fixtures (`UPDATE_FIXTURE=1`), and run all workspace + `ferrum_gui` Rust and Node tests.
- [ ] **Step 5: Update `MANUAL.md` §10.3, `docs/GUI-TOOLTIPS.md`, and `STATUS.md` §5.** Commit: `feat(gui): add model size presets, max-seconds, and metrics-csv to Train tab`.

---

## Task 3: Fine-Tune BOS Alignment & Fine-Tune / Export Verbose Controls (Gaps M14, L17)

**Problem:**
1. `slm_cli/src/finetune.rs:97` calls `tok.prefix_bos()` (which checks `add_bos_token` in the GGUF metadata; `false` on GPT-2/Qwen2/SmolLM2, `true` on Llama), whereas `ferrum_gui/src/commands/finetune.rs:130` calls `tok.bos()` unconditionally. For any model where `add_bos_token == false`, the GUI prepends a BOS token that the CLI does not, shifting the token count by 1 and causing `--resume` across CLI and GUI to fail or train on different windows.
2. Both `ferrum_gui/ui/tabs/finetune_tab.js:32` and `ferrum_gui/ui/tabs/export_tab.js:37` hard-code `verbose: false` because `#panel-finetune` and `#panel-export` have no `Verbose` checkbox, even though `FinetuneParams` and `ExportParams` in Rust already accept `verbose: bool`.

**Files:**
- Modify: `ferrum_gui/src/commands/finetune.rs` (line 130: `tok.bos()` -> `tok.prefix_bos()`)
- Modify: `ferrum_gui/ui/index.html` (add `#ftVerbose` in `#panel-finetune` and `#exVerbose` in `#panel-export`)
- Modify: `ferrum_gui/ui/tabs/finetune_tab.js`, `ferrum_gui/ui/tabs/export_tab.js`, `ferrum_gui/ui/core/help.js`
- Modify: `ferrum_gui/src/ipc_tests.rs`, `ferrum_gui/ui/tests/app_flows.test.js`
- Modify: `MANUAL.md` §9.2 & §10.3, `docs/GUI-TOOLTIPS.md`, `STATUS.md` §5 & §7

**Steps:**
- [ ] **Step 1: Write a regression test** in `ferrum_gui/src/commands/finetune.rs` (`finetune_respects_add_bos_token_false`) using a tiny GGUF fixture with `tokenizer.ggml.add_bos_token = false` and `bos_token_id = 1`: assert that the checkpoint saved by `finetune_inner` can be resumed by the CLI `finetune-gguf --resume` (same window count and corpus id).
- [ ] **Step 2: Change `tok.bos()` to `tok.prefix_bos()`** in `ferrum_gui/src/commands/finetune.rs:130`.
- [ ] **Step 3: Add `#ftVerbose` and `#exVerbose` checkboxes** to `ferrum_gui/ui/index.html`, wire them in `finetune_tab.js` and `export_tab.js`, and add tooltip entries in `help.js`.
- [ ] **Step 4: Refresh fixtures (`UPDATE_FIXTURE=1`) and run all tests** (workspace + `ferrum_gui` + Node).
- [ ] **Step 5: Record the BOS alignment in `STATUS.md` §5 (closing the open BOS item) and §7.** Commit: `fix(gui): align fine-tune BOS handling with CLI and expose verbose toggles`.

---

## Task 4: Live Token Streaming on the GUI GGUF Tab (Gap M15)

**Problem:** The Generate tab (for Ferrum's own GPT-2 models) streams tokens live via `gen-fragment` Tauri events and updates a token progress bar (`ferrum_gui/src/commands/generate.rs`). In contrast, `run_gguf` (`ferrum_gui/src/commands/gguf.rs:344-396`) runs `LlamaModel` generation synchronously inside `blocking` and returns only when all `max_new_tokens` have been decoded, leaving the user staring at `Running GGUF model (CPU)...` with no live text or token progress bar on slow CPU runs.

**Files:**
- Modify: `ferrum_gui/src/commands/gguf.rs` (`run_gguf` and `run_gguf_inner`)
- Modify: `ferrum_gui/ui/index.html` (`#panel-gguf`: add `#ggufStream` checkbox and `#ggufProgress` progress bar)
- Modify: `ferrum_gui/ui/tabs/gguf_tab.js`, `ferrum_gui/ui/core/help.js`
- Modify: `ferrum_gui/src/ipc_tests.rs`, `ferrum_gui/ui/tests/app_flows.test.js`
- Modify: `MANUAL.md` §10.3, `docs/GUI-TOOLTIPS.md`, `STATUS.md` §5

**Interfaces:**
- Emit `gguf-fragment` events `{ text: String, generated: usize, budget: usize }` from the per-token decode loop in `run_gguf` (throttled or per UTF-8 boundary, matching `generate.rs`).
- Listen for `gguf-fragment` in `gguf_tab.js` to append to `#ggufOutput` and update `#ggufProgress`.

**Steps:**
- [ ] **Step 1: Write unit & IPC tests** in `ferrum_gui/src/commands/gguf.rs` and `ipc_tests.rs` verifying that `run_gguf` emits `gguf-fragment` events and that the concatenated fragments equal the returned `GgufRunReport.output`.
- [ ] **Step 2: Implement per-token streaming callback in `ferrum_gui/src/commands/gguf.rs`** (respecting `StopMatcher` holdback so held-back partial stop strings are never emitted prematurely).
- [ ] **Step 3: Add `#ggufStream` and `#ggufProgress`** in `ferrum_gui/ui/index.html` and wire `listen('gguf-fragment', ...)` in `ferrum_gui/ui/tabs/gguf_tab.js`.
- [ ] **Step 4: Add help entry in `help.js`, refresh fixtures, and run all tests.**
- [ ] **Step 5: Update `MANUAL.md` §10.3, `docs/GUI-TOOLTIPS.md`, and `STATUS.md` §5.** Commit: `feat(gui): stream tokens and progress live on the GGUF tab`.

---

## Task 5: Windows `.exe` Tool Discovery in `runtime_export` (Gap M16)

**Problem:** `slm_cli/src/runtime_export.rs` (`find_tool`, lines 79–105) searches `PATH` directories for bare `name` (`"ollama"` or `"lms"`). On Windows (`cfg!(windows)` or when `PATHEXT`/`.exe` applies), binaries on `PATH` are named `ollama.exe` and `lms.exe`, so `PATH` lookup misses them unless `--bin`, `FERRUM_OLLAMA`/`FERRUM_LMS`, or the fallback `%LOCALAPPDATA%` path hits (`STATUS.md` §7.0.13).

**Files:**
- Modify: `slm_cli/src/runtime_export.rs`
- Modify: `STATUS.md` §5 & §7.0.13

**Interfaces:**
- Extract a pure helper `fn find_in_path_dirs(name: &str, path_var: &OsStr, is_windows: bool) -> Option<PathBuf>` so unit tests on Linux can test both `is_windows = false` and `is_windows = true` deterministically without needing a Windows host.
- When `is_windows` is true, check `<dir>/<name>.exe` first, then `<dir>/<name>`.

**Steps:**
- [ ] **Step 1: Write unit tests** in `slm_cli/src/runtime_export.rs`:
  - `find_in_path_dirs_finds_exe_when_windows_flag_set`
  - `find_in_path_dirs_prefers_exe_over_bare_file_on_windows`
- [ ] **Step 2: Implement `find_in_path_dirs`** and call it from `find_tool` with `cfg!(windows)`.
- [ ] **Step 3: Run `cargo test -p slm_cli --release` and full workspace checks.**
- [ ] **Step 4: Update `STATUS.md` §7.0.13.** Commit: `fix(cli): check .exe suffix when scanning PATH for ollama and lms on Windows`.

---

## Task 6: Narrow `dvc.yaml` Source Dependencies (Gap M5 Residue)

**Problem:** `dvc.yaml` lists the entire `ferrum_core/src` and `slm_cli/src` directories as dependencies of the `prepare`, `train`, and `eval` stages. Consequently, any comment, doc, or unrelated module edit in `ferrum_core/src` or `slm_cli/src` marks `dvc.lock` stale (`docs/GAP-REPORT.md` M5).

**Files:**
- Modify: `dvc.yaml`, `dvc.lock`
- Modify: `tests/tests/` (or `slm_cli/tests/`): add a test verifying every `deps:` path in `dvc.yaml` exists in the repository.
- Modify: `MANUAL.md` §9.9, `STATUS.md` §5

**Steps:**
- [ ] **Step 1: Write a test** (`dvc_yaml_deps_exist_in_repo`) that parses `deps:` entries from `dvc.yaml` and asserts every referenced file/directory exists.
- [ ] **Step 2: Narrow `dvc.yaml` stage `deps:`** to the specific modules each stage uses (e.g., `prepare` depends on `ferrum_core/src/dataset.rs`, `slm_cli/src/prepare/`, `slm_cli/src/cmd_prepare.rs`, `scripts/fetch.sh`, `data/raw/tiny_shakespeare.txt`), and refresh `dvc.lock` for `prepare` with `XDG_CACHE_HOME`/`XDG_CONFIG_HOME`/`XDG_DATA_HOME` pointed inside the session scratchpad (rule 3).
- [ ] **Step 3: Run workspace tests and update `STATUS.md` §5 and `docs/GAP-REPORT.md` M5.** Commit: `build(dvc): narrow stage source dependencies in dvc.yaml`.

---

## Task 7: Complete `#![warn(missing_docs)]` Across `ferrum_core` (Gap L11 Residue)

**Problem:** 14 modules in `ferrum_core` enforce `#![warn(missing_docs)]` (`05420aa`), leaving ~218 undocumented public items across the remaining modules (`slm/`, `train_transformer/`, `loader/`, `gguf/`, `gguf_write/`, `llm_train/`, `layer/`, `tokenizer/`, `ckpt.rs`, `meta.rs`, `llm.rs`, `gguf_tokenizer.rs`).

**Files:**
- Modify: Remaining `ferrum_core/src/**/*.rs` modules and `ferrum_core/src/lib.rs` (enable `#![warn(missing_docs)]` at crate root once all submodules pass).
- Modify: `STATUS.md` §5, `docs/GAP-REPORT.md` L11

**Steps:**
- [ ] **Step 1: Document public items module-by-module**, adding `#![warn(missing_docs)]` to each completed module (`ckpt`, `meta`, `gguf_tokenizer`, `llm`, `layer`, `tokenizer`, `gguf`, `gguf_write`, `loader`, `slm`, `train_transformer`, `llm_train`).
- [ ] **Step 2: Promote `#![warn(missing_docs)]` to `ferrum_core/src/lib.rs`** and remove redundant inner attributes.
- [ ] **Step 3: Verify with `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`** and `cargo test --workspace --release` (ensuring any new doc-tests compile and pass).
- [ ] **Step 4: Update `STATUS.md` and `docs/GAP-REPORT.md` L11.** Commit: `docs(core): document all public items and enable warn(missing_docs) crate-wide`.


---

## Remaining work after the 2026-11-07 reconciliation (`main` at the logo/favicon commit)

Verified by grepping `ferrum_gui/ui/index.html` and the Rust sources on 2026-11-07; Tasks 1-7 above are all still open except where noted.

1. **H4, Serve tab:** `quant`, `force` and `repetition_penalty` controls (hard-coded `Int4`, no force, `1.0`).
2. **H5, Train tab:** `--max-seconds` and `--metrics-csv` controls (the `#trPreset` selector landed in `92988f2`).
3. **M14 / L17:** Fine-tune BOS parity with the CLI (`prefix_bos()`), `Verbose` checkboxes on Fine-tune and Export (no `ftVerbose`/`exVerbose` ids exist).
4. **M15:** streaming and a progress bar for the GGUF tab's `run_gguf`.
5. **M16:** `.exe` lookup for `ollama`/`lms` on Windows in `slm_cli::runtime_export::find_tool` (now `slm_cli/src/runtime_export/`).
6. **GUI beginner-mode leftovers:** mode-aware `tooltip.js` (Task 4), help and tooltip-guard entries for the new controls, a real click-test of Beginner/Expert mode and the new logo/favicon.
7. **M5 / L11 residue** as listed in Tasks 6-7 above.
8. **Housekeeping:** `STATUS.md` test counts after every suite run; `METRICS.md` is generated by `scripts/update_metrics.sh` (pre-commit), do not edit it by hand.
