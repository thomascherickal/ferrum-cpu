# Ferrum

A pure-Rust engine for training, running, serving, and evaluating small causal
Transformer language models (SLMs) on the CPU, plus GGUF import, fine-tuning,
and export for Llama/Qwen checkpoints. Ferrum's own SLMs are GPT-2-architecture
GGUF files that run in llama.cpp, Ollama, and LM Studio. Overview in
`ReadMe.md`, usage in `Manual.md`, walk-throughs in `Samples.md`, current state
/ known issues / roadmap / compatibility record in `Status.md`.

> `AGENTS.md` is a symlink to `CLAUDE.md`, so the two cannot drift. Edit
> `CLAUDE.md` only; never replace the symlink with a regular file.

## Working rules

Every rule the user has given for this repo. They override defaults and
skills. When the user states a new rule, add it here.

1. **Commit after every completed feature.** A feature is complete when its
   tests are in, the full suite passes (`cargo test --workspace --release`,
   clippy `-D warnings`, fmt, and for GUI work the checks in Commands), and its
   docs are updated. Then make one commit with a Conventional-style subject.
   Never commit a half-done feature or a red tree, and never batch several
   finished features into one commit. Keep features modular: one concern per
   module and per commit, with a small public surface.
2. **Run only one agent at a time.** No parallel subagents. Finish or stop the
   running agent before launching the next. Before relaunching, check that an
   agent is truly active (file mtimes, running processes); after a crash or
   restart launch a fresh agent with explicit state instead of resuming it.
   Tasks cut off when an agent stops are not dropped: relaunch them and finish
   them before starting new work.
3. **Stay inside the project.** Never touch anything outside the project
   directory (no `~/.cache`). Exception, standing permission from the user
   (2026-11-03): `~/.ollama` and `~/.lmstudio` may be used by the Ollama and LM
   Studio runs (export commands, `verify_runtimes.sh`, Samples); touch only the
   models Ferrum itself creates there and remove only those. Scratch files go
   in the session scratchpad.
4. **Low-memory machine (15 GB).** Build with
   `CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`,
   run one cargo command at a time, and do not run `llvm-cov` together with
   tests.
5. **Tests run after every added feature**, the whole suite, not just the new
   tests. Report the exact pass/fail/ignored counts; fix or disclose any failure.
6. **No invented output.** Every number and command output in the docs comes
   from a real run on this machine. Say plainly what was not run.
7. **Push after every green suite, merge when a feature is done.** After each
   full-suite run with no failures, push the branch (or `main`); push at
   least once a day. As soon as a feature is complete, fast-forward merge it to
   `main` and push. No PRs, no merge commits. "Finish completely" also means
   resolving the remaining Minor review findings in one polish commit first.
8. **`ferrum_gui` is outside the workspace.** Build and test it from its crate
   dir; use `node --check` for the UI JavaScript.
9. **Docs stay in sync with the code.** Update `Status.md` (snapshot, test
   counts, known issues, roadmap, section 7) in the feature's commit. Only the
   docs listed under Conventions exist (plus `docs/coverage-results.md`, rule
   10, the tutorials of rule 14, and reports the user asks for by name:
   `docs/ferrum-capabilities.md`, `docs/gap-report.md`); add no others.
10. **Coverage runs once a day, at noon, no more.** This limits LOCAL runs of
    `scripts/coverage.sh`; the CI `coverage` job runs on every push as the
    gate. Run it locally (alone, never beside tests) only in the noon slot (about 12:00, local
    time), never on demand in between. Record every run in
    `docs/coverage-results.md`: date, commit, the totals, and the per-file lines
    that are not 100 %. Between runs, find gaps by reasoning from the last
    results file and the unit tests, not by rerunning coverage. (No run on
    2026-11-02, by the user's instruction.)
11. **Run the tests and fix every failure before every commit.** The full
    `cargo test --workspace --release` (plus clippy and fmt) is green before a
    commit is made; a red or untested tree is never committed.

12. **Clean engineering.** Modular code with high cohesion and low coupling:
    one concern per module, a small public surface, dependencies pointing one
    way.
13. **Tests by tier.** Every new feature lands with a unit test. When a
    functionality is complete, add functional tests for it; when a module is
    complete, add integration tests for the module. New tests use stubs or
    fakes instead of full or tiny trained models, so the suite stays fast.
14. **Docs with tutorials.** When a new functionality is complete, document it
    with a tutorial in the `docs/` folder. This supersedes the "add no other
    docs" limit in rule 9 and Conventions for tutorials placed in `docs/`.

15. **Coverage floor.** The CI coverage floors stay in the 95 %+ range and are
    raised toward 100 % as measured runs allow; reach 100 % wherever possible.
    A floor is only raised to a value a real run has met. (`GEMINI.md` is a
    symlink to this file, like `AGENTS.md`.)

16. **Ask only when there is no clear choice.** When a question has a
    **Recommended** option, take it without asking. Ask the user only when no
    option is marked Recommended, or when the choice is genuinely ambiguous or
    confusing. Never fill a real gap with an unstated assumption: if you took a
    Recommended option or ruled on a detail, say so in one line. Subagents
    report real doubts back (NEEDS_CONTEXT) and the controller relays them
    only under this same test. (This replaces the earlier "ask, never assume"
    rule of 2026-11-03; `GEMINI.md` and `AGENTS.md` are symlinks to this file,
    so the rule applies to all three.)

17. **Hand-off file.** After every agent result (and at every task boundary)
    refresh `.superpowers/HANDOFF.md` (git-ignored): the goal, what is done
    and pushed, the running or paused agents, uncommitted files, the next
    steps in order, and every open question for the user. The usage skill
    cannot read the real 5-hour percentage, so the file is kept current
    instead of written at 98 %. At the start of every session, if
    `.superpowers/HANDOFF.md` exists, read it first, continue from its next
    steps, and mark it consumed.

18. **The default branch is `main`, never `master`.** The repository's default
    branch was renamed from `master` to `main` on 2026-11-03. Use `main` for
    every merge, push, checkout, diff base and CI/release reference; do not
    recreate or mention `master` as the integration branch in new work.

## Hard constraints

Do not break these; they are the product's selling points.

- **GGUF only.** Every file Ferrum writes — models, training checkpoints,
  tokenizer files — is GGUF v3. FINF is removed; do not reintroduce a custom
  binary format. Ferrum's own BPE SLMs are `general.architecture = "gpt2"` and
  must keep running in llama.cpp (the runtime compatibility test guards this).
  Checkpoints are `ferrum-checkpoint`, tokenizer files `ferrum-tokenizer`,
  pre-tokenized corpora (`tokenize`; I16/I32 `tokens` tensor streamed by
  `Read`+`Seek`) `ferrum-tokens`. A new file Ferrum writes is a new GGUF kind,
  never a custom binary format.
  Non-transformer models (tabular MLPs, character-level SLMs) use the generic
  `ferrum` layout and do not run in llama.cpp.
- **`ferrum_core` has two external dependencies**, both pure-Rust Unicode tables:
  `unicode-general-category` (the GPT-2 pre-tokenizer's `\p{L}`/`\p{N}`) and
  `unicode-normalization` (NFKC for `prepare`). Add no others — std only; its
  threading and matmul kernels are hand-written. Benchmarks are std-only too
  (`harness = false`, no Criterion). The other crates pull their own
  (`slm_cli`: sha2/serde_json; `slm_server`: axum/tokio/serde/serde_json/sha2;
  `slm_wasm`: wasm-bindgen; `ferrum_gui`: Tauri and a few helpers, see its `Cargo.toml`). A new dependency anywhere
  needs a benchmark that shows a gain (or that it replaces more code than it
  adds), a passing determinism test, a working `wasm32-unknown-unknown` build of
  `ferrum_core`, and `cargo deny check`. `rayon`, `ndarray`, and `faer` were
  measured and not adopted (`Manual.md` §2.2.1).
- **`#![forbid(unsafe_code)]`** in every crate root, including examples,
  benches, and build scripts. A repo-wide test (`test_no_unsafe`) scans for
  `unsafe`. No SIMD intrinsics. Never call a dependency's `unsafe fn` directly.
- **Deterministic at any thread count.** Matmuls split across the persistent
  worker pool in `parallel.rs`; results must stay bit-for-bit identical
  regardless of `FERRUM_NUM_THREADS`, and generation and evaluation are
  identical at any thread count. The one exception is the data-parallel
  trainer: `train --threads 0` (auto) takes its shard count from
  `FERRUM_NUM_THREADS` or the core count, and a different shard count changes
  the trained weights. Training is reproducible for a fixed `--threads`, and
  the DVC pipeline pins it (`train.threads` in `params.yaml`), so `dvc.lock`
  reproduces on any machine.
  `cargo test -p slm_cli --release --test determinism` enforces the rest.
- **WASM-portable.** `ferrum_core` and `slm_wasm` must compile and run for
  `wasm32-unknown-unknown` (no threads/fs assumptions on that path). CI builds
  `slm_wasm` for wasm32.
- **Backward compatibility.** `TransformerConfig` is extended with new fields +
  `Default`; callers use `..TransformerConfig::default()`. The positional
  `train_transformer*` functions keep stride 1 and their exact results. Record
  any compat change in `Status.md` §7. FINF files do not load (deliberate).
- **Tests first.** Every feature lands with its unit tests in the same commit
  (Working rules 1 and 5).
- **Scoring semantics.** `evaluate()` and `--val` use `score_rolling`, which
  scores every token. `temperature == 0` is exact argmax. `<|endoftext|>` is
  always a reserved special token.

## Layout

| Path | What |
|---|---|
| `ferrum_core/` | The engine library; everything calls into it. The large modules are directories (`slm/`, `train_transformer/`, `loader/`, `gguf/`, `gguf_write/`, `llm_train/`, `layer/`, `tokenizer/`): `mod.rs` re-exports the public API, submodules hold one concern each with its own tests, shared fixtures live in a `#[cfg(test)] test_support.rs`. `token_file/` (the `ferrum-tokens` writer and streaming reader) is a small directory too. `dataset` (cleaning), `ckpt`, `meta`, `ops`, `parallel`, `quant` and `llm` are single files. `examples/library_tour.rs` is the program behind `Samples.md` §10. |
| `slm_cli/` | Binary `train_transformer`: train, run, generate, eval, info, prepare, tokenize, run-gguf, eval-gguf, finetune-gguf, export-gguf, export-ollama, export-lmstudio. It is also a library (`src/lib.rs`: `slm_cli::prepare`, the prepare pipeline with progress and cancel, `slm_cli::runtime_export`, the Ollama / LM Studio export, and `slm_cli::tokenize` / `token_corpus`, the token-file writer and opener) that `ferrum_gui` depends on by path, so front ends share the logic instead of copying it (it lives here, not in `ferrum_core`, because it needs `sha2`/`serde_json`). `tests/tokenize_cli.rs` and `tests/token_train_cli.rs` cover token files; `tests/tiny_e2e.rs` and `tests/tiny_gguf_e2e.rs` drive the BPE pipeline and the imported-GGUF fine-tune/export path end to end through the real binary with tiny models |
| `slm_server/` | Binary `ferrum-serve` (axum): `/health`, `/v1/info`, `/v1/loglikelihood`, `/v1/loglikelihood_rolling`, `/v1/generate`, for Ferrum models and imported `llama`/`qwen2` GGUFs (`load.rs` picks the kind by `general.architecture` and runs the RAM guard, `backend.rs` dispatches, `imported.rs` holds the `LlamaModel` path). It is also a library: `bind` (exposure guard, model load, socket bind; real address, startup banner, request counter and log) then `Bound::serve`, which the binary and `ferrum_gui`'s Serve tab share. `tests/tiny_http_e2e.rs` serves a tiny model from the real binary, `tests/imported_e2e.rs` a synthetic tiny llama (and SmolLM2-135M when `.ferrum-dev/models/` has it) |
| `slm_wasm/` | wasm-bindgen browser bindings (`Slm` text API, `TransformerSLMModel` token-id model); native-callable logic in `text.rs` and `token_model.rs` (`lib.rs` is one-line forwards, excluded from coverage), unit tests compare it with the library; `demo/` is the browser demo page + Web Worker with dependency-free Node tests (`node slm_wasm/demo/*.test.js`); `tests/node.rs` is the `wasm-bindgen-test` suite (wasm32 only, runs on CI) |
| `eval/lighteval/` | Python: `FerrumModel`, `ferrum_tasks.py`, `run.sh`; pinned `lighteval==0.13.0` |
| `dvc.yaml`, `params.yaml`, `dvc.lock`, `.dvc/` | fetch → prepare → train → eval (the `lighteval` stage is opt-in in `eval/lighteval/dvc.yaml`) |
| `scripts/` | `fetch.sh`, `bench_slm.sh`, `build_wasm.sh`, `verify_runtimes.sh` |
| `tests/` | Cross-crate integration/regression tests (`tests/tests/*.rs`), incl. `test_runtime_compat.rs` and `test_tiny_library.rs` (tiny-model library end to end) `test_bench_script.rs` (`scripts/bench_slm.sh` path handling) and `test_cli_matrix.rs` (every subcommand has a binary test, every server route an HTTP e2e test) |
| `docs/` | `ferrum-capabilities.md` (capability estimates by RAM tier and processor), `gap-report.md` (functionality gaps and the front-end parity matrix), `resume-cancel-and-prepare.md` (tutorial: exact resume, corpus identity, cancel, the prepare commit marker), `serving-imported-models.md` (tutorial: serving an imported Llama/Qwen GGUF with `ferrum-serve`), `SLM-edge-use-cases.md` (a standalone article on what small language models do on edge devices; it makes no claims about Ferrum), plus the agent-skill config in `docs/agents/` and the plans/specs in `docs/superpowers/` |
| `ferrum_gui/` | Tauri app "Ferrum SLM Studio" — **excluded from the workspace**. `src/serve.rs` runs the `slm_server` library in-process for the Serve tab; `docs/assets/logo.png` is the source of every bundle icon (`cargo tauri icon docs/assets/logo.png -o <dir>`; `icons/icon.svg` is legacy); `.github/workflows/release.yml` builds the Linux AppImage and deb (tags `v*` and `workflow_dispatch`, draft release on a tag) |

This repo's directory is `ferrum-cpu` inside the workspace folder `ferrum`
(`/home/thomas/Projects/ferrum/`; the GitHub repo is `ferrum-cpu`).
Sibling folders `../ferrum-ml` (tabular MLP demos, depends on `ferrum_core` by
path `../../ferrum-cpu/ferrum_core`; its models are GGUF), `../ferrum-junk`
(archive of removed features) and `../ferrum-gpu` (plan for a GPU edition, no
code yet: `../ferrum-gpu/PLAN.md`) are separate repos. Changing a `ferrum_core`
public API can break `ferrum-ml`.

## Two Transformer stacks — don't mix them up

They share only the low-level kernels (`ops`, `parallel`, `quant`).

| | Ferrum's own | Imported |
|---|---|---|
| Modules | `train_transformer`, `slm`, `tokenizer`, `meta`, `loader` | `gguf`, `gguf_tokenizer`, `llm`, `llm_train`, `gguf_write`, `chat_format` |
| Architecture | GPT-2: learned positions, LayerNorm, GELU MLP, dense MHA, no head bias | RoPE, RMSNorm, SwiGLU, GQA |
| Training | from scratch, Adam/AdamW, int8 QAT (= Q8_0) | AdamW fine-tuning (f32) |
| Ships as | `gpt2` GGUF (F32/Q8_0/Q4_0), tokenizer inside | GGUF v3 |

## Commands

Run from the repo root unless noted. This mirrors CI (`.github/workflows/ci.yml`), which also passes `--locked` to
cargo and checks the workspace on the MSRV (`rust-version = "1.87"`).

```bash
cargo build --workspace
cargo test --workspace --release          # CI runs it without --release
cargo test -p slm_cli --release --test determinism
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo build -p slm_wasm --target wasm32-unknown-unknown --release
cargo clippy -p slm_wasm --target wasm32-unknown-unknown -- -D warnings
cargo deny check
cargo bench --bench gemm                  # matmul/decode microbenchmark
scripts/coverage.sh [--rust|--gui|--js|--py|--all] [--fail-under N]   # line coverage -> target/coverage/; run alone, never beside cargo test (missing lines: missing.txt, gui-missing.txt)
scripts/bench_slm.sh data/prepared/train.txt   # train/eval/generate speed of the 507,552-param reference
bash scripts/build_wasm.sh                # slm_wasm → slm_wasm/pkg/ + demo/pkg/ (needs wasm-bindgen 0.2.122)
# optional, CI only unless you install wasm-bindgen-cli 0.2.122: the wasm module under Node
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo test -p slm_wasm --target wasm32-unknown-unknown --release --test node
```

On a 15 GB machine build with fewer jobs and no LTO:
`CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`.

Data, serving, and evaluation:

```bash
train_transformer prepare --input corpus.txt --out data/prepared --dedup --nfkc
cargo run --release -p slm_server -- --model models/slm.gguf      # ferrum-serve, 127.0.0.1:8080
python3 -m venv .venv && .venv/bin/pip install -r requirements-dataops.txt
.venv/bin/dvc repro && .venv/bin/dvc status
python3 -m unittest discover -s eval/lighteval -p 'test_*.py'    # CI runs this; skips what needs lighteval
eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt --out metrics/lighteval.json --max-samples 200
```

Runtime compatibility tests skip a tool that is absent, unless
`FERRUM_REQUIRE_RUNTIMES=1` is set (then a missing tool fails the test; CI's
`runtimes` job sets it, with `FERRUM_SKIP_LMSTUDIO=1` as its one exemption).
`export-ollama`/`export-lmstudio` find their tool by flag (`--bin`), then env
(`FERRUM_OLLAMA`, `FERRUM_LMS`), then `PATH`, then the default install path;
the tests and `scripts/verify_runtimes.sh` find llama.cpp through
`FERRUM_LLAMA_BIN`. Export commands refuse non-gpt2 GGUFs (for a llama GGUF the Ollama route is a hand-written Modelfile, Manual §9.5). The Ollama and LM
Studio checks need `FERRUM_VERIFY_RUNTIMES=1`.

```bash
FERRUM_REQUIRE_RUNTIMES=1 cargo test -p integration_tests --release --test test_runtime_compat -- --nocapture --test-threads 1
```

**`ferrum_gui`** is outside the workspace, so `cargo … -p ferrum_gui` from the
root fails. Work inside the crate:

```bash
cd ferrum_gui
cargo fmt -- --check && cargo clippy --all-targets -- -D warnings && cargo test
cargo tauri dev                           # run the app
cargo tauri build --bundles appimage,deb  # what .github/workflows/release.yml runs (it downloads linuxdeploy: CI only)
for f in ui/*.js; do node --check "$f"; done && for t in ui/*.test.js; do node "$t" || exit 1; done
```

After changing what `live.js` or a tab module (`*_tab.js`) sends to the backend, refresh the pinned
arguments with `UPDATE_FIXTURE=1 node ui/live.test.js` and
`UPDATE_FIXTURE=1 node ui/app_flows.test.js`; `cargo test` then checks them against
the Rust types. `cargo test` also needs no display (Tauri's mock runtime).

The UI (`ui/index.html`, `app.js` plus one `*_tab.js` module per tab, `styles.css`) is static — `cargo build`
won't catch JS errors; `node --check` is the gate. Tabs are generic: a
`<button class="tab" data-tab="X">` auto-wires to `<section class="panel" id="panel-X">`.

## Conventions

- Rust 2021, rustfmt defaults (`rustfmt.toml`, Unix newlines); clippy clean with `-D warnings`.
- New CLI flags go in the per-subcommand `*_FLAGS` tables in `slm_cli/src/args.rs`;
  the parser rejects anything not listed.
- Deliberate simplifications are marked with a `// ponytail:` comment naming the
  ceiling and the upgrade path.
- Docs are `ReadMe.md`, `Manual.md`, `Status.md`, `Samples.md`, and
  `glossary.md`, plus the article `docs/SLM-edge-use-cases.md`. Don't add other top-level docs; update `Status.md` (snapshot,
  test counts, known issues, roadmap, §7 compatibility) when shipping a
  feature. Every command in `Samples.md` is executed and its output pasted from
  the run; no invented numbers. `glossary.md` is a beginner-facing guide to LLM
  terms, not a domain glossary for agents.
- Git: solo repo, linear history; see Working rules 1 and 7. Each commit builds
  and passes its tests on its own. Conventional-style subjects
  (`feat(gguf): …`, `fix(gui): …`, `docs(status): …`).

## Agent skills

### Issue tracker

GitHub Issues via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root (created lazily). See `docs/agents/domain.md`.
