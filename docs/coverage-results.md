<p align="center"><img src="assets/logo.png" width="96" alt="Ferrum logo"></p>

# Coverage results

One entry per `scripts/coverage.sh` run. Local runs stopped on 2026-11-04: CI measures coverage on every push (CLAUDE.md rule 10); the entries below are the past local runs. Numbers are copied from the run's `target/coverage/summary.txt`.

## 2026-11-02

Runs before the once-a-day rule existed (all `--rust`, branch `feat/coverage-100`, commit 67c4d74 and its parent state):

| Run | Lines missed | Line cover | Functions | Regions |
|---|---|---|---|---|
| Task 1.1 baseline | 384 of 29,045 | 98.68 % | 97.44 % | 98.09 % |
| Task 1.2, first run | 319 of 29,120 | 98.90 % | 97.60 % | 98.22 % |
| Task 1.2, fix round 1 (tests of the previous run's gaps) | 303 of 29,214 | 98.96 % | 97.98 % | 98.30 % |

Files in the Task 1.2 list that were not 100.00 % in the last run: `gguf/cursor.rs` 1 line, `gguf/tensors.rs` 2, `gguf_tokenizer.rs` 3, `loader/decode.rs` 1, `loader/gpt2.rs` 4 (the ignored foreign-GPT-2 test). All others at 100.00 %.

How the summary counts lines (found by recomputing it from `llvm-cov export` JSON, matching all five files exactly): per function group (same source location), line coverage is the best single instantiation, and `ferrum_core` is compiled twice (the library, used by the other crates' tests, and its own unit-test build). A function whose lines are split between the two builds, with neither build covering all of them, leaves lines missed even though the merged per-line view shows none. Fix: make the unit tests (the build that sees the whole function) cover every line of the function themselves.

Noon run of 2026-11-02: skipped on the user's instruction.

## 2026-11-03

Noon run, `scripts/coverage.sh --all --fail-under 0`, branch `feat/coverage-100`, commit 9fe7320, alone on the machine (40 min). Numbers are copied from `target/coverage/`.

| Part | Lines | Functions | Regions |
|---|---|---|---|
| Rust workspace (`--rust`) | 99.70 % (92 of 30,480 missed) | 98.54 % | 98.80 % |
| `ferrum_gui` Rust (`--gui`) | 98.75 % (76 of 6,102 missed) | 91.74 % | 98.15 % |
| `ferrum_gui/ui/*.js` (`--js`) | 100.00 % | 98.31 % | branches 93.59 % |
| `slm_wasm/demo/*.js` (`--js`) | 98.06 % (`worker.js` 95.38 %, lines 63-65) | 100 % | branches 98.46 % |
| `eval/lighteval/*.py` (`--py`) | 100 % (153 of 153 statements; LightEval absent, 18 tests skip) | | |

Rust files with missed lines (workspace): `slm_cli/src/tokenize.rs` 7, `slm_cli/src/finetune.rs` 7, `slm_cli/src/runtime_export.rs` 6, `slm_cli/src/prepare/input.rs` 5, `slm_cli/src/eval_gguf.rs` 5, `ferrum_core/src/parallel.rs` 5, `ferrum_core/src/chat_format.rs` 5, `slm_server/src/model_thread.rs` 4, `slm_cli/src/run_gguf.rs` 4, `slm_cli/src/prepare/mod.rs` 4, `slm_cli/src/export.rs` 4, `ferrum_core/src/stop.rs` 4, `slm_cli/src/main.rs` 3, `ferrum_core/src/slm/persist.rs` 3, `ferrum_core/src/slm/generate.rs` 3, `slm_server/src/serve.rs` 2, `slm_cli/src/train.rs` 2, `ferrum_core/src/slm/training_loop.rs` 2, `ferrum_core/src/slm/train.rs` 2, `ferrum_core/src/loss.rs` 2, `slm_server/src/http.rs` 1, `slm_cli/src/token_corpus.rs` 1, `slm_cli/src/outputs.rs` 1, `slm_cli/src/generate.rs` 1, `ferrum_core/src/verbose.rs` 1, `ferrum_core/src/train_transformer/checkpoint.rs` 1, `ferrum_core/src/train_transformer/backward.rs` 1, `ferrum_core/src/slm/train_tokens.rs` 1, `ferrum_core/src/slm/stream.rs` 1, `ferrum_core/src/slm/corpus.rs` 1, `ferrum_core/src/slm/cached.rs` 1, `ferrum_core/src/meta.rs` 1, `ferrum_core/src/dataset.rs` 1.

`ferrum_gui` files with missed lines: `commands/gguf.rs` 12, `evaluate.rs` 10, `commands/export.rs` 9, `commands/text.rs` 7, `datasets.rs` 5, `capable.rs` 5, `commands/train.rs` 4, `test_support.rs` 3, `serve.rs` 3, `jobs.rs` 3, `estimate_cmd.rs` 3, `commands/generate.rs` 3, `hardware.rs` 2, `preflight.rs` 1, `paths.rs` 1, `ipc_errors.rs` 1, `error.rs` 1, `commands/system.rs` 1, `commands/mod.rs` 1, `commands/finetune.rs` 1.

CI floors set from this run (CLAUDE.md rule 15): Rust 99, GUI 98, JS 98, Python 100. The 100 % goal is not met yet; the per-file lists above are the work list for the next round. The causes of the remaining lines were not analysed in this run.

No further local runs are scheduled (user decision, 2026-11-04); CI's `coverage` job is the gate on every push.
