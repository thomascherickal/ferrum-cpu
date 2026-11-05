<p align="center"><img src="assets/logo.png" width="96" alt="Ferrum logo"></p>

# Ferrum audit report

Date 2026-11-05 · commit audited `c7f34e8` (`main`, clean tree) · work on branch
`audit-2026-11-05` (not pushed, not merged).

The user asked for this report by name: "Check the repo for gaps and errors. Add tests to
cover them if found and create a report." Items that `docs/gap-report.md` already records as
open by choice were not reported again: M5, M6 (LM Studio), M9, M11, M12, the M13 residue
and the rest of L11.

## Method

1. **Build health.** Every check in the CLAUDE.md Commands section was run on the untouched
   tree, one cargo command at a time, with `CARGO_BUILD_JOBS=3
   CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`.
2. **Test gaps (rules 19 and 20).**
   - Listed every `src/` Rust file that has no `#[cfg(test)]` or `#[test]` and checked where
     it is exercised.
   - Matched every JS module under `ferrum_gui/ui/core` and `ui/tabs` to a test that names it.
   - Grepped non-test code for `unwrap`, `expect`, `panic!` and `unreachable!`, and read each
     hit.
   - Read the trust boundaries: the CLI parser and `validate.rs`, the server's request
     checks, the token-file reader, the `Embedding` id conversion and the wasm id API.
   - Ran a probe script of 155 `train_transformer` invocations, each under `timeout 30`.
     They covered every numeric train flag with `0, 1, 3, 2^64-1` or `0, -1, nan, inf,
     1e30`, the generate sampling flags, an empty prompt and an empty stop string, empty and
     truncated model files, a directory given as a model, and `prepare` with bad fractions.
     The script looked for exit 101, `panicked at` and timeouts.
3. **Docs against code.**
   - Checked every flag in the `slm_cli/src/args.rs` tables against `Manual.md`, in both
     directions.
   - Checked the server flags and routes against Manual §9.7.
   - Checked the Status.md test counts and commit count against real runs.
   - Ran a script over every relative link and `#anchor` in the tracked `*.md` files
     (`docs/superpowers/` excluded).
   - Checked that every `src/…rs` and `ui/…` path named in the docs exists.
   - Checked that every test name the docs quote exists.
   - Checked the CLAUDE.md Layout table against the tree.
4. **Scripts and CI.**
   - Ran `bash -n` on every script and read each one.
   - Grepped `.github/workflows/*.yml` for removed files, old `ui/` paths and Ollama
     leftovers.
5. **Constraints.** Read `tests/tests/test_no_unsafe.rs` (its scope and the crate-root
   `forbid` check), and checked that every crate root, build script, bench and example has
   `#![forbid(unsafe_code)]`. The determinism test ran as part of the full suite.

## Findings

| ID | Severity | Area | Evidence | Status and recommended action |
|---|---|---|---|---|
| A1 | Medium | `ferrum_core` / `slm_wasm` trust boundary | `ferrum_core/src/layer/embedding.rs:121` converted an id with `input.data[..].round() as usize`. The cast saturates, so `-1.0`, `-0.6` and `NaN` became token 0 and the forward pass returned a distribution for the wrong token. `slm_wasm/src/token_model.rs:107` (`prime`) did the same. Through `TransformerSLMModel.predict_next` / `prime`, a JS caller's bad `Float32Array` got a wrong answer instead of an error. Run on 2026-11-05, both regression tests failed with the old lines put back. With only the `prime` check removed, the wasm test failed at its `prime` assert ("prime took -1"). | **Fixed in `a6fbf9f`**: both places now refuse a negative or NaN id: `Embedding` with its existing out-of-range error, and `prime` with "Token id … is not a non-negative number". Regression tests `layer::embedding::tests::forward_rejects_negative_and_nan_ids` and `token_model::tests::predict_next_and_prime_reject_negative_and_nan_ids`. Recorded in Status §7.0.20 and Manual §11.3. |
| A2 | Low | Docs: Manual §10.6 | The GUI architecture map still showed the layout from before the `ui/` and `src/` regrouping: `ui/` holding `app.js` and the tab modules flat, and `src/jobs.rs`, `src/error.rs`, `src/preflight.rs`, `src/capable.rs`, `src/hardware.rs`, `src/estimate.rs`, `src/tokenize.rs`, `src/runtime_export.rs`, `src/prepare.rs`, `src/datasets.rs`, `src/paths.rs`, `src/serve.rs`. None of these exist now; they live in `ui/core`, `ui/tabs`, `ui/tests` and `src/{commands,sizing,data,serving,infra}`. The same section also named `src/estimate.rs` (cost-model constants) and `src/hardware.rs` (test table). | **Fixed in `f47cb9a`**: the map was rewritten from `ls` of the tree, and both paths now point to `src/sizing/`. |
| A3 | Low | Docs: Status §4 | The G9, G8 and G17 rows named `src/jobs.rs`, `src/preflight.rs` and `src/paths.rs`. | **Fixed in `f47cb9a`**: they now name `src/infra/jobs.rs`, `src/sizing/preflight.rs` and `src/infra/paths.rs`. |
| A4 | Low | Docs: Status snapshot | One Tests cell gave two current counts for `ferrum_gui`: "`ferrum_gui` has **211**" and, further on, "**205** `ferrum_gui` backend tests pass". The run on 2026-11-05 gave 211. | **Fixed in this report's commit**: the second count is now worded as the count at H1. |
| A5 | Low | Docs: gap-report totals | The Total row said **29**, but the tables hold 3 High, 13 Medium and 14 Low rows, which is 30. L14 was added without updating the total. | **Fixed in this report's commit**: the total now counts the rows, including the rows this audit adds. |
| A6 | Low | Rule 19: unit test per file | These `src/` files have no unit test of their own:<br>• `ferrum_core`: `layer/matmul.rs`, `train_transformer/param.rs`, `train_transformer/block.rs`, `train_transformer/kernels.rs`, `llm_train/blocks.rs`, `llm_train/buffers.rs`<br>• `slm_cli` binary modules: `eval_gguf.rs`, `finetune.rs`, `generate.rs`, `run_gguf.rs`, `train.rs`, `usage.rs`, `token_train.rs`<br>• `ferrum_gui`: `commands/train.rs`, `run.rs`<br>Other tests do exercise them: the whole-network finite-difference check in `train_transformer/test_support.rs`, the `llm_train` gradient tests, the binary e2e tests in `slm_cli/tests/`, and the GUI IPC tests. Not counted: `token_file/reader.rs` and `writer.rs` (covered by `token_file/tests.rs`), the `test_support.rs` files, and the re-export-only `mod.rs`, `main.rs` and `lib.rs`. | **Partly fixed in `6b19605`**: unit tests added for `layer/matmul.rs` (both helpers against a plain loop, below and above the pool threshold, bit-exact; empty dimensions) and `train_transformer/param.rs` (constructors, `weights_only`, `zero_grad`, AdamW decay skips rank-1 params). **Closed in `ff7dcc3`, `1a53c42` and `47af81a`** (2026-11-05): every `src` file now holds its own `#[cfg(test)]` module, 27 files in all, because the scan also counted the re-export `mod.rs`, `lib.rs` and `main.rs` files and the test-only helper files left out above. 56 workspace tests (26 `ferrum_core`, among them `kernels::ln_bwd`, the attention backward and the Llama block backward against finite differences and the `buffers.rs` layout; 25 `slm_cli`, among them `usage.rs` naming every command and flag; 5 `slm_server`) and 22 `ferrum_gui` tests. The one exception is `ferrum_gui/build.rs`: a build script is not a test target, so a test inside it would never run; the `desktop` cfg it sets is asserted from `run.rs`. The work found one bug, fixed with a regression test: a failed save in the GUI's Train tab left verbose mode on (`Status.md` §7.0.21). |
| A7 | Low | Docs: CLAUDE.md Layout | The `docs/` row does not list `docs/gui-tooltips.md` (a rule-14 tutorial) or `docs/coverage-results.md` (rule 10). The `scripts/` row does not list `coverage.sh`. This report, `docs/audit-report.md`, is in neither the Layout table nor rule 9's list of allowed docs. | **Fixed in `d0669da`**: the four names are in the Layout table and `docs/audit-report.md` is in rule 9 (the user asked for this report on 2026-11-05). The audit agent had left it open because CLAUDE.md is the user's rule file. |

## What passed clean

| Check | Result (2026-11-05, on `c7f34e8`) |
|---|---|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` | exit 0 |
| `cargo build -p slm_wasm --target wasm32-unknown-unknown --release` and its clippy | exit 0, exit 0 |
| `cargo deny check`, and `cargo deny --manifest-path ferrum_gui/Cargo.toml check` | exit 0 and exit 0 (`advisories ok, bans ok, licenses ok, sources ok`) |
| `cargo test --workspace --release --no-fail-fast` (baseline) | **1304 passed, 0 failed, 2 ignored**, as Status said |
| `ferrum_gui`: `cargo fmt -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` | exit 0, exit 0, **211 passed** |
| `node --check` on every `ferrum_gui/ui/**/*.js` | 49 files, all pass |
| `ferrum_gui/ui/tests/*.test.js` | 19 of 19 files pass |
| `slm_wasm/demo/*.test.js` | 2 of 2 pass |
| `python3 -m unittest discover -s eval/lighteval -p 'test_*.py'` | Ran 64, OK (skipped=18), as Status said |
| CLI flags: `args.rs` tables against Manual | every table flag is documented. Every Manual `--flag` missing from the tables belongs to another tool (`ferrum-serve`, llama.cpp, `lms`, `run.sh`, cargo, `coverage.sh`) or is the deliberate typo `--contxt` in the unknown-flag example |
| Server routes and flags against Manual §9.7 | `/health`, `/v1/info`, `/v1/loglikelihood`, `/v1/loglikelihood_rolling`, `/v1/generate` and all 12 server flags (with `--force`) are named in the Manual |
| Relative links and anchors in tracked `*.md` | none broken |
| CI workflows | no reference to a removed file, an old flat `ui/` path or a removed Ollama test. The demo tests run in CI (`ci.yml:235-236`) |
| Scripts | `bash -n` passes for all six. Five start with `set -euo pipefail`. `verify_runtimes.sh` uses only `set -u`, by design: it records each check in a `failed` flag and exits with that flag at the end (0, 1, or 2 for usage), as its header says, so one runtime failing does not stop the others |
| CLI probe (155 invocations, each under `ulimit -v 4000000` and `timeout 30`) | exit codes: 62 × 0, 88 × 1, 5 × 124. There was no other exit code (no 101 panic, no 134 abort, no 137 or 139) and no `panicked at` or `memory allocation` line. The five timeouts were genuinely long requests (`--epochs 99999999999`, `--epochs 2^64-1`, `--tokens 1e30`, `--chars` / `--max-tokens 2^64-1`) that train or generate as asked; they are not hangs |
| `unwrap`/`expect` in non-test code | 21 hits. Each one states a local invariant (an F32 encode of any length, the shard count, a cache restored after every step, worker-thread joins); none sits on external input |
| Hostile files | `tests/tests/test_hostile_inputs.rs` mutation-fuzzes every loader under `ulimit -v`. `token_file/tests.rs` covers truncated, mistyped and out-of-vocabulary token files |
| `#![forbid(unsafe_code)]` | present in every crate root, build script, bench and example. `test_no_unsafe` walks the whole repository, `ferrum_gui` included |
| Determinism | `slm_cli/tests/determinism.rs` ran in the full suite and passed |

## Not run, and why

- **shellcheck**: not installed on this machine. The scripts were read and passed
  `bash -n`.
- **Coverage** (`scripts/coverage.sh`): not run, by rule 10.
- **MSRV check** (`cargo +1.87 check --workspace --locked`): not run. Only the stable toolchain is installed (`rustup toolchain list`); CI runs this check.
- **wasm under Node** (`wasm-bindgen-test-runner`): not installed; CI runs it.
- **Runtime compatibility with `FERRUM_REQUIRE_RUNTIMES=1`, LM Studio and Ollama**: not run.
  The suite's runtime tests ran in their default skip-or-run mode.
- **GUI in a window, `cargo tauri build`, `dvc repro`, the LightEval-dependent Python tests**
  (LightEval is not installed): not run.
- **Hardware other than Linux x86_64**: none was available.

## Totals

| | Count |
|---|---|
| Findings | 7 (High 0, Medium 1, Low 6) |
| Fixed | 7 (A1 to A7; A6 in three follow-up commits) |
| Partly fixed | 0 |
| Open | none (`ferrum_gui/build.rs` cannot hold an executed test, see A6) |
| Tests added | 7 in the audit: 1 + 1 regression tests (A1), 2 `matmul` and 3 `param` unit tests (A6); then 56 workspace and 22 `ferrum_gui` tests closing A6, one of them the regression test of `Status.md` §7.0.21 |
| Suite after the changes | **1311 passed, 0 failed, 2 ignored** (`cargo test --workspace --release --no-fail-fast`, 7 more than 1304); `ferrum_gui` 211 passed; clippy (workspace and wasm32), fmt, doc and the wasm32 build clean |
| Suite after closing A6 | **1367 passed, 0 failed, 2 ignored** (56 more than 1311); `ferrum_gui` **233 passed** (22 more than 211), its fmt and clippy clean, `node --check` and the 19 `ui/tests` pass; workspace fmt and clippy clean; doc and the wasm32 build not re-run (no change outside `#[cfg(test)]` code there, except the `usage.rs` text constant and the GUI fix) |
