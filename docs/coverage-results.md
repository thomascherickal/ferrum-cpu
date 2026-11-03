<p align="center"><img src="assets/logo.png" width="96" alt="Ferrum logo"></p>

# Coverage results

One entry per `scripts/coverage.sh` run. Runs happen once a day, at noon (CLAUDE.md rule 10). Numbers are copied from the run's `target/coverage/summary.txt`.

## 2026-11-02

Runs before the once-a-day rule existed (all `--rust`, branch `feat/coverage-100`, commit 67c4d74 and its parent state):

| Run | Lines missed | Line cover | Functions | Regions |
|---|---|---|---|---|
| Task 1.1 baseline | 384 of 29,045 | 98.68 % | 97.44 % | 98.09 % |
| Task 1.2, first run | 319 of 29,120 | 98.90 % | 97.60 % | 98.22 % |
| Task 1.2, fix round 1 (tests of the previous run's gaps) | 303 of 29,214 | 98.96 % | 97.98 % | 98.30 % |

Files in the Task 1.2 list that were not 100.00 % in the last run: `gguf/cursor.rs` 1 line, `gguf/tensors.rs` 2, `gguf_tokenizer.rs` 3, `loader/decode.rs` 1, `loader/gpt2.rs` 4 (the ignored foreign-GPT-2 test). All others at 100.00 %.

How the summary counts lines (found by recomputing it from `llvm-cov export` JSON, matching all five files exactly): per function group (same source location), line coverage is the best single instantiation, and `ferrum_core` is compiled twice (the library, used by the other crates' tests, and its own unit-test build). A function whose lines are split between the two builds, with neither build covering all of them, leaves lines missed even though the merged per-line view shows none. Fix: make the unit tests (the build that sees the whole function) cover every line of the function themselves.

Noon run of 2026-11-02: skipped on the user's instruction. Next run: noon 2026-11-03.

After the last run, tests were added for exactly the gaps found (see the fix-round commit); they have not been measured by coverage yet. Expected from the analysis: `cursor.rs` unknown scalar type, `tensors.rs` Q5_0/Q5_1 dequant, `gguf_tokenizer.rs` empty token table and the whitespace run of `pretokenize`, `decode.rs` the never-called closure, `gpt2.rs` the foreign-file test now always running.
