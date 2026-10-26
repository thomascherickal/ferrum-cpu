# Ferrum

A pure-Rust engine for training, running, serving, and evaluating small causal
Transformer language models (SLMs) on the CPU, plus GGUF import, fine-tuning,
and export for Llama/Qwen checkpoints. Ferrum's own SLMs are GPT-2-architecture
GGUF files that run in llama.cpp, Ollama, and LM Studio. Overview in
`ReadMe.md`, usage in `Manual.md`, walk-throughs in `Samples.md`, current state
/ known issues / roadmap / compatibility record in `Status.md`.

> `CLAUDE.md` and `AGENTS.md` are identical copies. Edit both together
> (`cp CLAUDE.md AGENTS.md`).

## Hard constraints

Do not break these; they are the product's selling points.

- **GGUF only.** Every file Ferrum writes — models, training checkpoints,
  tokenizer files — is GGUF v3. FINF is removed; do not reintroduce a custom
  binary format. Ferrum's own BPE SLMs are `general.architecture = "gpt2"` and
  must keep running in llama.cpp (the runtime compatibility test guards this).
  Checkpoints are `ferrum-checkpoint`, tokenizer files `ferrum-tokenizer`.
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
  the trained weights. Training is reproducible for a fixed `--threads`.
  `cargo test -p slm_cli --release --test determinism` enforces the rest.
- **WASM-portable.** `ferrum_core` and `slm_wasm` must compile and run for
  `wasm32-unknown-unknown` (no threads/fs assumptions on that path). CI builds
  `slm_wasm` for wasm32.
- **Backward compatibility.** `TransformerConfig` is extended with new fields +
  `Default`; callers use `..TransformerConfig::default()`. The positional
  `train_transformer*` functions keep stride 1 and their exact results. Record
  any compat change in `Status.md` §7. FINF files do not load (deliberate).
- **Tests first.** Every feature lands with its unit tests in the same commit.
  Every commit builds and passes its tests.
- **Scoring semantics.** `evaluate()` and `--val` use `score_rolling`, which
  scores every token. `temperature == 0` is exact argmax. `<|endoftext|>` is
  always a reserved special token.

## Layout

| Path | What |
|---|---|
| `ferrum_core/` | The engine library; everything calls into it. The large modules are directories (`slm/`, `train_transformer/`, `loader/`, `gguf/`, `gguf_write/`, `llm_train/`, `layer/`, `tokenizer/`): `mod.rs` re-exports the public API, submodules hold one concern each with its own tests, shared fixtures live in a `#[cfg(test)] test_support.rs`. `dataset` (cleaning), `ckpt`, `meta`, `ops`, `parallel`, `quant` and `llm` are single files. `examples/library_tour.rs` is the program behind `Samples.md` §10. |
| `slm_cli/` | Binary `train_transformer`: train, run, generate, eval, info, prepare, run-gguf, finetune-gguf, export-gguf, export-ollama, export-lmstudio |
| `slm_server/` | Binary `ferrum-serve` (axum): `/health`, `/v1/info`, `/v1/loglikelihood`, `/v1/loglikelihood_rolling`, `/v1/generate` |
| `slm_wasm/` | wasm-bindgen browser bindings (`TransformerSLMModel`) |
| `eval/lighteval/` | Python: `FerrumModel`, `ferrum_tasks.py`, `run.sh`; pinned `lighteval==0.13.0` |
| `dvc.yaml`, `params.yaml`, `dvc.lock`, `.dvc/` | fetch → prepare → train → eval (the `lighteval` stage is opt-in in `eval/lighteval/dvc.yaml`) |
| `scripts/` | `fetch.sh`, `bench_slm.sh`, `build_wasm.sh`, `verify_runtimes.sh` |
| `tests/` | Cross-crate integration/regression tests (`tests/tests/*.rs`), incl. `test_runtime_compat.rs` |
| `ferrum_gui/` | Tauri app "Ferrum SLM Studio" — **excluded from the workspace** |

Sibling folders `../Ferrum-ML` (tabular MLP demos, depends on `ferrum_core` by
path; its models are GGUF) and `../Ferrum-Junk` (archive of removed features)
are separate repos. Changing a `ferrum_core` public API can break `Ferrum-ML`.

## Two Transformer stacks — don't mix them up

They share only the low-level kernels (`ops`, `parallel`, `quant`).

| | Ferrum's own | Imported |
|---|---|---|
| Modules | `train_transformer`, `slm`, `tokenizer`, `meta`, `loader` | `gguf`, `gguf_tokenizer`, `llm`, `llm_train`, `gguf_write`, `chat_format` |
| Architecture | GPT-2: learned positions, LayerNorm, GELU MLP, dense MHA, no head bias | RoPE, RMSNorm, SwiGLU, GQA |
| Training | from scratch, Adam/AdamW, int8 QAT (= Q8_0) | AdamW fine-tuning (f32) |
| Ships as | `gpt2` GGUF (F32/Q8_0/Q4_0), tokenizer inside | GGUF v3 |

## Commands

Run from the repo root unless noted. This mirrors CI (`.github/workflows/ci.yml`).

```bash
cargo build --workspace
cargo test --workspace --release          # CI runs it without --release
cargo test -p slm_cli --release --test determinism
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build -p slm_wasm --target wasm32-unknown-unknown --release
cargo clippy -p slm_wasm --target wasm32-unknown-unknown -- -D warnings
cargo deny check
cargo bench --bench gemm                  # matmul/decode microbenchmark
scripts/bench_slm.sh data/prepared/train.txt   # train/eval/generate speed of the 507,552-param reference
bash scripts/build_wasm.sh                # slm_wasm → slm_wasm/pkg/ (needs wasm-bindgen 0.2.122)
```

On a 15 GB machine build with fewer jobs and no LTO:
`CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`.

Data, serving, and evaluation:

```bash
train_transformer prepare --input corpus.txt --out data/prepared --dedup --nfkc
cargo run --release -p slm_server -- --model models/slm.gguf      # ferrum-serve, 127.0.0.1:8080
python3 -m venv .venv && .venv/bin/pip install -r requirements-dataops.txt
.venv/bin/dvc repro && .venv/bin/dvc status
python3 -m unittest eval/lighteval/test_ferrum_model.py           # skips without lighteval
eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt --out metrics/lighteval.json --max-samples 200
```

Runtime compatibility tests skip a tool that is absent, unless
`FERRUM_REQUIRE_RUNTIMES=1` is set (then a missing tool fails the test; CI's
`runtimes` job sets it, with `FERRUM_SKIP_LMSTUDIO=1` as its one exemption).
`export-ollama`/`export-lmstudio` find their tool by flag (`--bin`), then env
(`FERRUM_OLLAMA`, `FERRUM_LMS`), then `PATH`, then the default install path;
the tests and `scripts/verify_runtimes.sh` find llama.cpp through
`FERRUM_LLAMA_BIN`. Export commands refuse non-gpt2 GGUFs. The Ollama and LM
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
node --check ui/app.js && node --check ui/stream.js && node ui/stream.test.js
```

The UI (`ui/index.html`, `app.js`, `styles.css`) is static — `cargo build`
won't catch JS errors; `node --check` is the gate. Tabs are generic: a
`<button class="tab" data-tab="X">` auto-wires to `<section class="panel" id="panel-X">`.

## Conventions

- Rust 2021, rustfmt defaults (`rustfmt.toml`, Unix newlines); clippy clean with `-D warnings`.
- New CLI flags go in the per-subcommand `*_FLAGS` tables in `slm_cli/src/args.rs`;
  the parser rejects anything not listed.
- Deliberate simplifications are marked with a `// ponytail:` comment naming the
  ceiling and the upgrade path.
- Docs are `ReadMe.md`, `Manual.md`, `Status.md`, `Samples.md`, and
  `glossary.md`. Don't add other top-level docs; update `Status.md` (snapshot,
  test counts, known issues, roadmap, §7 compatibility) when shipping a
  feature. Every command in `Samples.md` is executed and its output pasted from
  the run; no invented numbers. `glossary.md` is a beginner-facing guide to LLM
  terms, not a domain glossary for agents.
- Git: solo repo, linear history. Feature branches are fast-forward merged into
  `master`, pushed, then deleted — no PRs, no merge commits. Each commit should
  build and pass its tests on its own. Conventional-style subjects
  (`feat(gguf): …`, `fix(gui): …`, `docs(status): …`).

## Agent skills

### Issue tracker

GitHub Issues via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root (created lazily). See `docs/agents/domain.md`.
