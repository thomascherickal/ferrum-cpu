<p align="center"><img src="assets/logo.png" width="96" alt="Ferrum logo"></p>

# Ferrum functionality gap report

Date 2026-11-03 · commit `e4c457a` (`master`, clean tree) · read-only scan; nothing was fixed.

**Method.** (1) Spec `2026-10-09-dataops-serve-eval-design.md` and the 2026-10-09 plans were checked against HEAD by reading code and by the older `.superpowers/sdd/spec-audit.md`; the 2026-11-02 plan's Phases 0-2 were taken as done and spot-checked (corpus id, cancel flag, prepare marker, progress events). (2) `Status.md` §5/§6 rows were compared with code and `git log`. (3) `rg` over the tree for `ponytail:`, `TODO`, `FIXME`, `todo!`, `unimplemented!`, `#[ignore]` (target/, node_modules/ excluded). (4) Mechanical cross-checks: every `--flag` in `Manual.md` against the `*_FLAGS` tables in `slm_cli/src/args.rs` (both directions, plus per-section), server flags/routes against Manual §9.7, `generate_handler!` against `ui/*.js` calls, and `<cmd> --help` of the already-built `target/release/train_transformer` (built 2026-11-03 14:55, four minutes before HEAD, docs-only commit after). (5) Old inputs `.superpowers/sdd/{gap-ledger,spec-audit,e2e-report,deferred-minors,task-catalog,plan}.md` re-verified item by item at HEAD. (6) Front-end parity matrix from the Rust/JS/wasm sources. A few cheap runs of the built binary in the scratchpad (`prepare`, `train` on a 150-line toy corpus, `info`, `eval`); no cargo build/test, no coverage, no GUI window, no network, no Ollama/LM Studio, no `dvc`.

| Severity | Count |
|---|---|
| Critical (documented claim the code does not meet) | 0 |
| High (feature in one front end, missing in another, user-visible) | 2 open (H2 closed) |
| Medium (declared ceiling with no plan) | 13 |
| Low (polish) | 13 |
| **Total** | **29** (28 open) |

No Critical finding: every Manual flag table, default, server route and server flag that was cross-checked matches the code (see "Mechanical cross-checks" below). The Critical column is empty because nothing met the bar, not because the check was skipped.

## Findings

### High

| ID | Severity | Gap | Evidence | Tracked in Status §5/§6 or plan Phase 3? | Suggested next step |
|---|---|---|---|---|---|
| H1 | High | The GUI Train tab cannot choose the saved precision: every model is saved int8. The CLI offers `--quant f32\|int8\|int4`, `--preset`, `--metrics-csv`, `--max-seconds`, `--sample`, `--debug`; none exists in the GUI. Manual §10.3 does not say this. | `ferrum_gui/src/commands/train.rs:295` calls `slm.save()`, which is int8 (`ferrum_core/src/slm/persist.rs:44-48`); `TrainParams` (train.rs:13-65) and `ui/train_tab.js` have no quant field (`rg -i quant ferrum_gui/ui/train_tab.js` is empty); CLI `args.rs:42-45`, `slm_cli/src/train.rs:35,271`. Same gap was in the old `e2e-report.md` §4 and is not closed. | no | Add a precision select (f32/int8/int4) using `to_bytes_*` as `slm_cli/src/train.rs:271` does; state the other missing flags in Manual §10.3. |
| H2 | High | **CLOSED in `2a72d02`** (optional `repetition_penalty`, server flag `--repetition-penalty`). `ferrum-serve` ignores repetition penalty: `/v1/generate` has no such field and the model thread hard-codes 1.0, while CLI `generate --repetition-penalty`, GUI Generate/GGUF and wasm `Slm.generate` all expose it. | `slm_server/src/model_thread.rs:88` (`repetition_penalty: 1.0`); request fields in Manual §9.7 table are `prompt,max_tokens,stop,temperature,top_k,top_p,seed` only; `ferrum_core/src/slm/sampling.rs:32` has the field. | no | Add an optional `repetition_penalty` to `GenRequest` (validated like the GUI: finite, > 0), default 1.0; document it in §9.7. |
| H3 | High | `ferrum-serve` and the wasm `Slm` load only Ferrum's own models (`GenerativeSLM`); an imported Llama/Qwen GGUF can be run (CLI `run-gguf`, GUI GGUF tab), scored (`eval-gguf`, Evaluate tab) and fine-tuned, but not served. Manual §9.7 does not state that the server is Ferrum-model-only. | `slm_server/src/model_thread.rs:126` (`spawn_with(bytes, GenerativeSLM::from_bytes)`); `slm_wasm/src/text.rs:75` (`from_bytes`); no `LlamaModel` reference in `slm_server/src` or `slm_wasm/src`. | no | Either state the limit in Manual §9.7/§11.3, or add a `LlamaModel` backend for `/v1/loglikelihood*` and `/v1/generate`. |

### Medium

| ID | Severity | Gap | Evidence | Tracked? | Suggested next step |
|---|---|---|---|---|---|
| M1 | Medium | The server does not end a generation when the client disconnects; `SamplingParams.cancel` exists but is not wired to the HTTP layer. (Session gap 4: confirmed.) | `Manual.md:1668-1671` ("the server does not wire that up yet"); `rg cancel slm_server/src` finds nothing. | partly: Manual §9.7 only; not in Status §5/§6 | Add a Status §5 row; wire a drop guard on the response future to the flag. |
| M2 | Medium | Resume does not detect a changed `--batch` (nor `--stride`, `--lr`, `--cosine`, `--warmup`, `--clip`); only vocab/context/architecture and the corpus id are compared, so a "bit-identical" resume (Manual §6.6, "same flags") silently becomes a different run. (Session gap 8: confirmed.) | No `batch` key in `ferrum_core/src/train_transformer/checkpoint.rs` or `ferrum_core/src/ckpt.rs` (`rg batch` in both: no hit); Manual §6.6 prints notes only for `--embed/--heads/--hidden/--blocks/--seed`. | no | Record batch size and stride in the checkpoint and refuse or warn on mismatch, as for the corpus id. |
| M3 | Medium | `finetune-gguf` checkpoints (`.flck`) record no corpus identity and no mid-epoch RNG state, so `finetune-gguf --resume` on a different corpus is accepted and a resumed fine-tune is not bit-identical; both guarantees exist only for `train`. | `rg -i corpus ferrum_core/src/llm_train/checkpoint.rs` is empty; `ferrum.ckpt.epoch_rng` appears only under `train_transformer/` and `slm/`; CLI `slm_cli/src/finetune.rs` has no id. | no | Reuse the `corpus_id` mechanism in the llama checkpoint; or state in Manual §9.2 that it is train-only. |
| M4 | Medium | Arm, Windows (incl. Windows-on-Arm / Snapdragon) and macOS (incl. Apple silicon) builds are in no CI job or test; release workflow builds Linux only. (Session gap 11: confirmed.) | `rg 'aarch64\|arm\|windows\|macos' .github/workflows/*.yml` finds nothing; every job is `runs-on: ubuntu-*` (`ci.yml:16,35,62,79,104,151,165,207,291`, `release.yml:25`). Status G13 records "no Windows or macOS bundle was built" but not Arm. | partly (Status §5 G13 row: Windows/macOS GUI bundles; Arm not mentioned) | Add a `cargo check --workspace` matrix on `aarch64-unknown-linux-gnu` (cross) and a macOS/Windows runner; add a Status row. |
| M5 | Medium | The DVC pipeline cannot run here and is untested; `dvc.lock` may be stale. Root cause found: the `.venv/bin/dvc` script has a shebang to `/home/thomas/Projects/Ferrum/ferrum-cpu/.venv/bin/python3` (capital `F`, the pre-rename path), so it fails with "required file not found". The venv's Python imports dvc fine. `dvc.lock` was last refreshed in `3c5a91d` (2026-10-26); 33 commits since then touched `ferrum_core/src` or `slm_cli/src`, which are `dvc.yaml` deps. (Session gap 2: confirmed, cause identified.) | `head -1 .venv/bin/dvc`; `.venv/bin/dvc --version` -> "cannot execute: required file not found"; `.venv/bin/python3 -c 'import dvc'` prints the module path; `git log 3c5a91d..HEAD -- ferrum_core/src slm_cli/src \| wc -l` = 33. `dvc repro` was not run (it would write XDG caches). | partly: Status §5 rows say "`dvc repro prepare` was NOT run"; the broken shebang and the lock staleness are not stated | Recreate `.venv` (or `pip install --force-reinstall dvc` to rewrite the shebang), then `dvc status`/`dvc repro` with the XDG_* variables inside the repo; refresh `dvc.lock`. Narrowing `dvc.yaml` deps (old task A2) would stop the lock going stale on every source edit. |
| M6 | Medium | The only end-to-end test of `export-ollama`/`export-lmstudio` against real tools (`slm_cli/tests/runtime_export_cli.rs`) needs a model file outside the repository (`../.ferrum-dev/gpt2gguf/gpt2-q8_0.gguf`, untracked) and is skipped on plain CI; the CI `runtimes` job runs only `test_runtime_compat`. Its `ponytail:` comment still says "until Ferrum's own GPT-2 GGUF writer (G1) merges", which happened long ago. Status says the `runtimes` job "has not run on GitHub yet". | `runtime_export_cli.rs:8-10,19,28-32`; `ci.yml:341-342`; `Status.md` §5 (Only the verified tool versions are tested). | partly (runtimes job: Status §5) | Swap the fixture for a Ferrum-trained tiny GPT-2 GGUF (the comment's own plan), add the test to the `runtimes` job, fix the comment. |
| M7 | Medium | Five `ponytail:` ceilings are mentioned in neither Status nor Manual: unbounded server job queue, tokenizer state cap (64 B/state byte + 64 MiB), O(n²) re-decode in the no-block BPE fallback, a vocab x dim scratch per tied-head backward call, and the Prepare tab's folder walk on every form change. (All other markers are listed below.) | See the ponytail table below: `slm_server/src/model_thread.rs:134`, `ferrum_core/src/tokenizer/state.rs:37`, `ferrum_core/src/slm/stream.rs:378`, `ferrum_core/src/llm_train/passes.rs:113`, `ferrum_gui/src/estimate_cmd.rs:76`. | no | One Status §5 line listing them, or accept as internal (the server one matters if the server is ever exposed to many clients: bounded in practice by `MAX_INFLIGHT = 32`, `limits.rs:6`). |
| M8 | Medium | No SIGINT/Ctrl-C checkpoint; stop file and `--max-seconds` are the graceful paths; the `ctrlc` dependency decision is still open. (Session gap 9: confirmed.) | `Manual.md:794-795`; `Status.md` §5 (🟢 "An interrupt does not save on exit") and §6.2 ("A checkpoint-on-interrupt handler"). | yes: Status §5, §6.2 | Decide on `ctrlc` or close the item as "won't do". |
| M9 | Medium | No concurrent serving: one model thread, one request at a time, because `GenerativeSLM` is not `Send`/`Sync`. (Session gap 10: confirmed.) | `Manual.md:1603-1605`; `Status.md` §5 (server row); `slm_server/src/model_thread.rs`. | yes: Status §5 | None unless multi-client use is wanted. |
| M10 | Medium | No stop strings for imported GGUFs: CLI `run-gguf` has no `--stop`/`--max-tokens`, the GUI GGUF tab has no stop box; Ferrum models have them in CLI, GUI, server and wasm. | `RUN_GGUF_FLAGS` (`args.rs:71-83`) has none; `Manual.md:1855` ("no stop strings: the library and `run-gguf` stop on token ids only"); `Status.md` §5 G7 row. | yes: Status §5 (G7) | Add a stop-string path to `LlamaModel::generate` (old task N4). |
| M11 | Medium | Plan `2026-11-02` Phase 3, items 3.1-3.15, are planned and not built by decision: gradient accumulation, metrics columns, packed Q8_0 matmul, safe-SIMD decode, `LlamaModel::init`, LoRA, SFT, distillation, `qwen3`, streamed GGUF export, exact SPM encode, IQ* quants, per-block import quantization, size-aware defaults remainder, full-size DVC profile. Not re-analysed. | `docs/superpowers/plans/2026-11-02-complete-ferrum-cpu.md:302-322`; Status §6.2. | yes: plan Phase 3, Status §6.2 | None here. |
| M12 | Medium | No DVC remote: `data/` and `models/` exist only where `dvc repro` ran. | `.dvc/config` (no remote section); Status §5 last rows. | yes: Status §5 | Pick a remote or document the policy. |
| M13 | Medium | No RoPE scaling on GGUF import (Llama 3.x long context, YaRN); RoPE type hard-wired (`Norm`, `Neox` for qwen2). | Status §5 ("no RoPE scaling"). | yes: Status §5 | Add to §6.2 if wanted. |

### Low

| ID | Severity | Gap | Evidence | Tracked? | Suggested next step |
|---|---|---|---|---|---|
| L1 | Low | Manual does not document `finetune-gguf --force`, `--temp`, `--max` (the last two apply with `--sample`) or `export-gguf --force`; all four are accepted flags. Every other Manual flag matches the tables. | `FINETUNE_FLAGS` `args.rs:85-103`, `EXPORT_FLAGS` `args.rs:105-109`; Manual §9.2 table (`Manual.md:1372-1387`) and §9.3 (`Manual.md:1405-1450`) lack them; `finetune-gguf --help` lists them. `--force` is read at `slm_cli/src/finetune.rs:64` and `export.rs:72`. | no | Add three rows. |
| L2 | Low | Manual §9.7 limits table is broken: two rows ("POST requests in flight", auth, malformed JSON, model error) follow a prose paragraph and render outside the table. | `Manual.md:1632-1640`. | no | Remove the blank line/paragraph between the rows. |
| L3 | Low | Manual §10.3 tab table has no Serve row (the tab is described in prose at `Manual.md:1922`, and in §10.1). The index.html has 12 tabs, the table 11 plus the engine log. | `rg -o 'data-tab=' ferrum_gui/ui/index.html` vs `Manual.md:1840-1860`. | no | Add a row. |
| L4 | Low | `Status.md` snapshot is stale: it says branch `feat/coverage-100` (only `master` exists: `git branch -a`), "master is `575d4cf`" and "314 commits" (`git rev-list --count HEAD` = 349). | `Status.md:5,34,38`. | no | Refresh the snapshot in the next feature commit. |
| L5 | Low | Range of `top_p` differs across front ends: server requires (0, 1], CLI/GUI/wasm accept [0, 1]. | `slm_server/src/http.rs:209`; `ferrum_gui/src/commands/generate.rs:50`; `slm_wasm/src/text.rs:50`. | no | Pick one range. |
| L6 | Low | wasm `Slm` seed is `u32`; CLI `--gen-seed` is `u64`, so seeds above 4294967295 cannot be reproduced in the browser. Documented ("0 to 4294967295"). | `slm_wasm/src/lib.rs:79-90`; `Manual.md:2339`. | yes, in the Manual | None. |
| L7 | Low | After a cancel `StopReason` stays `MaxTokens`; a cancel is visible only in `Generation.cancelled`. (Session gap 5: confirmed.) | `ferrum_core/src/slm/generate.rs:9-12` (doc says so). | yes: Status §7.0.4 (field) | Add a `Cancelled` variant at the next breaking release, or leave. |
| L8 | Low | `download_text` (Source URL box) emits no progress events. (Session gap 6: confirmed.) | Status §5 G10 row; `Manual.md:1850`; tooltip in `ui/index.html:55`. | yes: Status §5 | None. |
| L9 | Low | One shared progress bar: events carry `id` but the UI writes every event to the same bar, and downloads have no job slot, so two concurrent downloads overwrite each other. (Session gap 7: confirmed.) | `ferrum_gui/ui/datasets_tab.js:18-24`; no `jobs.start` in `ferrum_gui/src/datasets.rs`. | partly (one bar is stated, the overlap is not) | Disable the download buttons while one runs, or filter by id. |
| L10 | Low | `Samples.md:220` says "DVC is not installed ... `dvc: command not found`"; on this machine dvc is installed but its script is broken (M5). | `Samples.md:220`; `head -1 .venv/bin/dvc`. | no | Reword after M5 is fixed. |
| L11 | Low | `#![warn(missing_docs)]` is not enabled in `ferrum_core`; the old catalog counted 86 of 317 public items undocumented (not recounted). | `rg missing_docs ferrum_core slm_cli slm_server` is empty; `.superpowers/sdd/task-catalog.md` D2. | no | Enable per module as the docs are written. |
| L12 | Low | The GUI Models tab (`model_info`) rejects Ferrum checkpoints, tokenizer files and token files ("not a valid model"), while the CLI `info` describes all of them. | `ferrum_gui/src/commands/gguf.rs:29-60` (`GenerativeSLM::from_bytes`); `slm_cli/src/info.rs:1,83,116`. | no | Reuse the CLI's kind detection. |
| L13 | Low | Conflicting statements on `export-lmstudio`: Status §5 and Samples §7 say it was not run; the old ledger records `slm_cli/tests/runtime_export_cli.rs` (`export_lmstudio_then_generate`, line 330) passing under `FERRUM_REQUIRE_RUNTIMES=1` on 2026-10-25 (8 runtime tests). Not re-run at HEAD (LM Studio daemon; I did not touch it). (Session gap 1: partly contradicted.) | `.superpowers/sdd/gap-ledger.md` ("runtime_export_cli 7 passed", log 2026-10-25 15:14); `Samples.md:884,1034`; `Status.md` §5. | partly | Re-run once with `FERRUM_REQUIRE_RUNTIMES=1`, then state one version in Status and Samples. |

## Session-known gaps, one line each

1. `export-lmstudio` never run end to end: partly contradicted, see L13. 2. DVC cannot run: confirmed, root cause is a stale shebang (M5). 3. GUI never click-tested: confirmed (`Status.md` §5 🟡 rows; CI boot smoke test only proves the window opens; no change to report). 4. Server cancel on disconnect: M1. 5. `StopReason` after cancel: L7. 6. `download_text` no progress: L8. 7. Shared progress bar: L9. 8. Resume ignores batch size: M2. 9. No SIGINT checkpoint: M8. 10. No concurrent serving: M9. 11. Arm/Windows/macOS: M4. 12. Text-vs-token-file cross-resume refused: documented at `Manual.md:781-783` and `docs/resume-cancel-and-prepare.md:73`, closed as a doc gap. 13. The two injected-writer refactors (`41954aa` `slm_cli/src/tokenize.rs`, `28d71b7` `ferrum_core/src/slm/metrics_log.rs`, `training_loop.rs`): diffs read, the file/sink path is unchanged (`CsvLog.file` was a `File`, is now `Box<dyn Write>` over the same `File`); behaviour looks unchanged by reading, but the suite was not re-run here.

## Mechanical cross-checks

- **CLI flags, Manual vs `args.rs`.** All 65 distinct flag names in the `*_FLAGS` tables appear in `Manual.md`. Flags in the Manual but in no CLI table are the server's (`--host --port --max-batch --max-body --token --allow-unauthenticated`, all present in `slm_server/src/cli.rs:15-23`), LightEval's (`--max-samples --wikitext` in `eval/lighteval/run.sh:17,19`), cargo/tool flags, `--help/--version/--verbose` (accepted everywhere, `Manual.md:620-638`) and `--contxt` (an example typo). Per-section check: the only code flags missing from their command's Manual text are in L1. `train_transformer <cmd> --help` for `finetune-gguf`, `export-gguf`, `eval-gguf`, `run-gguf` matches `args.rs`.
- **Server.** Routes in `slm_server/src/http.rs:44-53` (`/health`, `/v1/info`, `/v1/loglikelihood`, `/v1/loglikelihood_rolling`, `/v1/generate`) equal Manual §9.7's five; flags and defaults match; `deny_unknown_fields` and limits as documented. Defects: L2, L5, H3 (H2 closed).
- **GUI.** All 29 commands in `generate_handler!` (`ferrum_gui/src/lib.rs:69-99`) are called from some `ui/*.js` (non-test); no UI call names a command that is not registered. Parity gaps: H1, L12.
- **Subcommands.** `spec_for` (`args.rs:154-172`) has train, run, generate, run-gguf, finetune-gguf, export-gguf, export-ollama, export-lmstudio, prepare, tokenize, eval, eval-gguf, info (plus aliases); all are in the Manual and `CLAUDE.md` Layout. `tests/tests/test_cli_matrix.rs` guards test coverage of each.
- **Defaults.** Manual §6.2 train defaults (context 16, embed 32, heads 4, blocks 2, hidden 64, vocab 512, epochs 100, batch 16, lr 0.01, seed 1337) equal `TransformerConfig::default()` (`ferrum_core/src/slm/config.rs:95-115`).

## `ponytail:`, `TODO`, `FIXME`, `todo!`, `unimplemented!`, `#[ignore]`

`TODO`, `FIXME`, `todo!`, `unimplemented!`: none in the repo. `#[ignore]`: 2, both exploration helpers in `tests/tests/probe_gguf.rs:9,50` (match Status "2 ignored"). `ponytail:` markers (18 in code, 1 in a test header, 3 more in `Status.md` prose, 1 in `CLAUDE.md`):

| File:line | Ceiling declared | Mentioned in Status/Manual? |
|---|---|---|
| `slm_server/src/model_thread.rs:134` | unbounded job queue; bounded channel + 503 if exposed to many clients | no (M7) |
| `slm_server/src/limits.rs:6` | `MAX_INFLIGHT = 32` fixed; make it a `Limits` field | value yes (`Manual.md:1632`), the fixed-ness no |
| `slm_cli/src/prepare/mod.rs:153` | split files not rolled back after a failed rename | yes (Status §5) |
| `slm_cli/src/prepare/mod.rs:229` | cancel polled between documents; one huge document is read to its end | yes (Status §5 G3) |
| `slm_cli/src/prepare/input.rs:282`, `slm_cli/src/outputs.rs:140` | root bypasses permission checks (test only) | n/a |
| `slm_cli/src/tokenize.rs:129` | same as above for `tokenize` | yes (Status §5, "one document at a time") |
| `slm_cli/src/tokenize.rs:145` | id cast cannot truncate (vocab capped at `i32::MAX`) | n/a (invariant) |
| `slm_cli/src/main.rs:179` | only `info` is left after `spec_for` | n/a (invariant) |
| `slm_cli/tests/runtime_export_cli.rs:8` | test model is llama.cpp's GPT-2 124M | no, and stale (M6) |
| `slm_wasm/src/text.rs:83,89`; `ferrum_core/src/llm.rs:393`; `ferrum_core/src/slm/token_stream.rs:159`; `ferrum_gui/src/commands/finetune.rs:146`; `ferrum_core/src/parallel.rs:289` | cannot fail / a failed spawn just means fewer helpers | n/a (invariants) |
| `ferrum_core/src/slm/stream.rs:378` | O(n²) re-decode in the no-transformer-block BPE fallback (legacy family) | no (M7) |
| `ferrum_core/src/llm_train/passes.rs:113` | vocab x dim scratch per tied-head backward call | no (M7) |
| `ferrum_core/src/loader/decode.rs:99` | Q8_0/Q4_0 expanded to f32 (4x/8x RAM) until a packed kernel exists | yes (Status §5; plan 3.3) |
| `ferrum_core/src/tokenizer/state.rs:37` | vocab built from state capped at 64 B/state byte + 64 MiB | no (M7) |
| `ferrum_core/src/tokenizer/train.rs:138` | EOS always reserved | yes (Status §5 EOS row) |
| `ferrum_gui/src/estimate_cmd.rs:76` | Prepare estimate walks folders on every form change | no (M7) |

## Requirements: specs and 2026-10-09 plans

Spot-checked at HEAD against the spec's sections: scoring API and `evaluate()` on `score_rolling` (§4.1), EOS special token (§4.2), strict parser with unknown-flag, bad-value and missing `--resume` errors (§4.3, `slm_cli/tests/cli_strict.rs`), generation flags (`GENERATE_FLAGS`), `prepare` flags and manifest (§5.1; `PREPARE_FLAGS` equals the spec's list), `eval --json` (§5.2), `params.yaml` keys (§5.3), the five server routes, limits and defaults (§6.1), `run.sh` options (§6.2), ten Samples (`Samples.md` has sections 1-10). All implemented. Deliberate deviations already ledgered: `generate_until` named `generate_tokens`; `lighteval` stage lives in `eval/lighteval/dvc.yaml` (frozen stage broke plain `dvc repro`); rayon/gemm/ndarray/faer not adopted (spec §4.5, Manual §2.2.1). The 2026-11-02 plan's Task 2.3 note "GUI passes the corpus id" is implemented (`ferrum_gui/src/commands/train.rs:127,186,251`). Nothing in the spec or the plans is unimplemented apart from the Phase 3 items (M11).

## Status.md §5 / §6 accuracy at HEAD

| Row | Verdict |
|---|---|
| §5 whole-corpus read, shard-count reproducibility, one-request server, LightEval workarounds, quantized-file drift, char-level not in llama.cpp, verified tool versions, small models, Capable estimates, GUI not click-tested, GUI estimates, mobile untested, boot smoke test, matmul crate decision | still open (code and CI unchanged; not re-run) |
| §5 🟢 rows (prepare marker, downloads, security, greedy/sampling, job control, prepare tab, tokenize, export tab, Evaluate, pre-flight, layout, a11y, settings, errors, paths, Serve tab, icons) | accurate where checkable by reading (e.g. job slots in `ferrum_gui/src/jobs.rs`, `check_epochs` in `validate.rs:91-98`) |
| §5 "An interrupt does not save on exit" | still open (M8) |
| §5 "`export-lmstudio` itself still not run" | disputed, see L13 |
| §5 "`runtimes` job has not run on GitHub yet" | not verifiable offline |
| §1 snapshot (branch, commit counts) | wrong, see L4; the **1211 passed / 2 ignored** claim not re-run (the 2 ignored match `rg '#\[ignore'`) |
| §6.2 F3 (qwen3), F5 (streamed export), F6 (safe SIMD), P1 list (Llama from scratch, kernels, accumulation, KD, SFT, LoRA, metrics columns, interrupt handler) | still open (all in plan Phase 3) |
| §6.2 F7 (`eval-gguf`) and F4 (boot smoke test) | closed: `f2c9b65` (CLI), `d3bd9e2` (GUI), CI `gui` job (`ci.yml:207-290`) |
| §6.2 "A3 full-size DVC profile" | still open (plan 3.15) |

## Front-end parity matrix

yes / no / partial, with the file that decides it. "CLI" is `train_transformer`/`slm_cli`; "GUI" is `ferrum_gui`; "wasm" is `Slm` (text) / `TokenModel` (`TransformerSLMModel`, token-id).

| Feature | CLI | GUI | Server | wasm `Slm` | wasm `TokenModel` |
|---|---|---|---|---|---|
| train (from scratch) | yes (`train.rs`) | partial: no `--quant/--preset/--metrics-csv/--max-seconds/--sample/--debug` (`commands/train.rs`, H1) | no | no | no |
| resume (train) | yes (`--resume`) | yes (`resume` checkbox, `train_tab.js:95`) | n/a | n/a | n/a |
| resume (finetune / apply `.flck`) | yes (`finetune-gguf`, `run-gguf`, `export-gguf`, `eval-gguf --resume`) | yes (`resume` in finetune/run/export/evaluate params) | no | no | no |
| exact mid-epoch resume | yes (train only, Manual §6.6; not finetune, M3) | yes (same engine) | n/a | n/a | n/a |
| cancel / stop | partial: stop file and `--max-seconds` on train/finetune; Ctrl-C kills without saving (M8) | yes: Stop for Prepare, Train, Generate, GGUF, Fine-tune, Export, Tokenize (`jobs.rs`, `cancel_job`); not Evaluate or an `ollama create` | partial: server shutdown only; no cancel on disconnect (M1) | yes (`cancel()`, `lib.rs:142`) | no |
| corpus identity (`corpus_id`) | yes (train: `train.rs:48`); no finetune (M3) | yes (train: `commands/train.rs:127`); no finetune | n/a | n/a | n/a |
| progress events | text lines (`-v`, epoch lines) | yes: `train-progress`, `finetune-progress`, `export-progress`, `prepare-progress`, `tokenize-progress`, `download-progress`, `clean-progress`, `serve-log`; none for `download_text` (L8) | no | partial: pull `next_chunk`, `tokens_generated` | no |
| stop sequences | yes `generate --stop` (repeatable); no `run-gguf` | yes Generate tab; no GGUF tab (M10) | yes (`stop`, 16 x 1024 B) | yes | no |
| precision / quantization | yes: `train --quant/--no-qat`, `run-gguf/eval-gguf/export-gguf --quant` | partial: QAT checkbox and int8 save only for Train (H1); precision for GGUF run, Evaluate, Export, f32 Fine-tune | no (serves the file as is; `/v1/info` reports `quant`) | reads f32/int8/int4 files | reads f32/int8/int4 files |
| eval (Ferrum model) | yes `eval [--json]` | yes Evaluate tab (bpb, JSON save) | yes `/v1/loglikelihood*` | partial: `score()` only | no |
| eval-gguf (imported) | yes | yes (same Evaluate tab) | no (H3) | no | no |
| run-gguf (imported) | yes | yes GGUF tab | no (H3) | no | no |
| finetune-gguf | yes | yes Fine-tune tab (no `--sample/--temp/--max`) | no | no | no |
| export-gguf | yes (`--quant`, `--resume`, `--force`) | yes Export tab | no | no | no |
| export-ollama / export-lmstudio | yes | yes (Export tab; fake-script tested only) | no | no | no |
| prepare | yes | yes Prepare tab (byte-identical outputs, tested) | no | no | no |
| tokenize / token files | yes `tokenize`; `train` accepts a token file | yes Tokenize button + Train accepts a token file | no | no | no |
| sampling: temperature | yes (0 = greedy) | yes | yes | yes | partial: `sample_from_probs(temperature)` only |
| sampling: top-k | yes | yes | yes | yes | partial: `top_k_indices` helper only |
| sampling: top-p | yes | yes | yes ((0, 1], L5) | yes | no |
| sampling: repetition penalty | yes (`--repetition-penalty`; `--rep` for run-gguf) | yes | yes (H2, closed) | yes | no |
| info / inspect | yes `info` (models, checkpoints, tokenizer, token files, imported GGUF) | partial: `model_info` only Ferrum models, `gguf_info` imported (L12) | yes `/v1/info` | yes `info()` | yes `metadata()` |
| serve | `ferrum-serve` binary (separate) | yes Serve tab (in-process library) | yes | no | no |

## Closed since older reports

| Item | Source | Closing commit |
|---|---|---|
| `export-gguf --resume` dropped fine-tuned biases (D1) | e2e-report | `f659a0e` |
| NaN loss swallowed by `max(1e-12)` (C1); divergence guard (C2) | task-catalog, gap-ledger | `e10803d`, `9219c81` |
| `prepare` destroyed previous splits / manifest over partial set (D2) | e2e-report | `750a3dc` (manifest last; residual ceiling stays, marked `ponytail:`) |
| `--epochs 0`, `--batch 0`, `--threads` > 1024, unwritable outputs checked early (D3, D5, D16, D22) | e2e-report | `0c238a6`, `14f51d3`, `829512a` (`check_writable`) |
| Resume on a different corpus accepted (D6) | e2e-report | `67ef93f` |
| Mid-epoch resume not bit-identical (D8) | e2e-report | `716c8c9` |
| `run-gguf` chatter on stdout, `--ids` parse, stop reason (D10, D11, D18) | e2e-report | `5777e51` (verified at HEAD: `--ids "a b"` -> "`'a' is not a token id`"; `run_gguf.rs:197-201`) |
| Llama GGUF refusal advice (D21) | e2e-report | `runtime_export.rs:15-20` now special-cases `llama` |
| Server `--max-tokens` default and unknown JSON fields (D23, D24) | e2e-report | `4032819` |
| `--version`, `<cmd> --help`, extra positionals (D17, D32) | e2e-report | `34b32db` (positional error verified: "`unexpected argument 'extra' for 'info'`") |
| Broken pipe panic (D31) | e2e-report | `15e6a6a` (verified: `info \| head -1` prints no panic) |
| `info` for checkpoints / imported GGUFs (D15) | e2e-report | `c074a8b` (checkpoint and tokenizer file verified) |
| `lighteval/run.sh` deleting results (D28); `bench_slm.sh` relative path (D29) | e2e-report | `run.sh:34` keeps `results.prev`; `bench_slm.sh:7` |
| BPE models unusable in the browser (W1); no demo (W2); wasm tests (W3) | task-catalog, e2e-report | `0f14a90`, `slm_wasm/demo/`, `slm_wasm/tests/node.rs` |
| Bearer token and non-loopback guard (S1) | task-catalog | `f954ba3` |
| GUI: greedy and sampling knobs (G7), cancel (G9), prepare (G3), export (G4), evaluate parity (G6), pre-flight (G8), Serve tab (G5), progress for downloads and Clean (G10 rest) | task-catalog | `06aab85` (progress), `27774d8` (prepare), `a208d11` (export), `d3bd9e2` (evaluate), `e791956` (serve); G7, G8, G9 are marked done in Status §5 (commit hashes not looked up) |
| `eval-gguf` (L6), `--max-seconds`/stop file (L1), `--preset` (C6), metrics CSV (C7) | task-catalog | `f2c9b65`, `4d55f01`, `cdf7829`, Status §7 |
| Flaky `mem_bandwidth` test | gap-ledger | `0e4b631` |
| Char-level model export refused / warned; `general.size_label` | gap-ledger | `f32a3bc` |
| Deferred-minors list (all items FIX/ALREADY FIXED) | deferred-minors | commits listed in that file; none re-opened |
| Python tests not in CI (CI2); `--locked`/doc/MSRV (CI3); `ferrum_gui/Cargo.lock` committed (CI4) | task-catalog | `git ls-files ferrum_gui/Cargo.lock` lists it; `ci.yml`; `40dd36d` |

Still open from those reports: matmul crate decision, packed Q8_0, RoPE scaling, DVC remote, GUI click test, SIGINT save (all tracked above).

## Not verified

- No test, clippy, fmt or doc build was run (one cargo command at a time on a 15 GB machine was not needed for a read-only scan); the Status test counts (1211 / 0 / 2) are quoted, not re-measured. Rule 10: no coverage run, and no coverage topics here.
- `dvc status`/`dvc repro` (M5): not run; the lock staleness is inferred from `git log`, not from a DVC hash check.
- GUI behaviour in a window; Tauri commands were read, not driven.
- Real Ollama/LM Studio/llama.cpp runs (L13, M6); I did not start them (rule 3 standing permission covers only Ferrum's own models, and none were needed for a read-only scan).
- `../ferrum-ml`: out of scope. I did not grep it for `ferrum_core` public APIs it depends on (the task allowed a note only); every public-API change recorded in Status §7 is a potential break for it.
- Windows, macOS, Arm: nothing could be built here (M4).
- `ferrum-serve` serving a real client over HTTP; routes and limits were read, not exercised.
- The 2026-10-09 plans' per-task checklists (hundreds of steps) were not walked one by one; the older `spec-audit.md` was re-verified on its GAP/DEFERRED rows only.
