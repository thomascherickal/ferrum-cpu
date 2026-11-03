<p align="center"><img src="docs/assets/logo.png" width="160" alt="Ferrum logo"></p>

# Ferrum

**A pure-Rust engine for training, running, serving, and evaluating small
causal Transformer language models (SLMs) on the CPU, with no GPU. Models you
train are GPT-2-architecture GGUF files that also run in llama.cpp, Ollama, and
LM Studio.**

Ferrum is `#![forbid(unsafe_code)]` throughout. The project contains the
tensors, the hand-written and gradient-checked backpropagation, a byte-level BPE
tokenizer, int8 quantization-aware training (QAT), a GGUF model format, a
multi-core matmul engine, a corpus-preparation tool, an HTTP inference server,
and a LightEval module. There is no NumPy, no BLAS, and no CUDA. The core also
compiles to WebAssembly, so a model you train can run on a server, a laptop, a
Raspberry Pi, or in a browser tab.

Ferrum keeps its dependencies few. The engine, `ferrum_core`, is `std`-only
apart from two pure-Rust Unicode tables, and a crate is admitted anywhere only
where a measurement shows a gain (see the table below and
[Manual §2](Manual.md#2-why-rust-and-what-few-dependencies-buy)). It is built
for small models. It is not a runtime for frontier LLMs.

| Property | What it means |
|---|---|
| **Few dependencies, each with a reason** | `ferrum_core` has two (`unicode-general-category` and `unicode-normalization`, pure-Rust tables) and hand-writes its threading and matmul kernels. The CLI adds `sha2` and `serde_json`, the server `axum`/`tokio`/`serde`, the browser crate `wasm-bindgen`, and the GUI Tauri plus a few helper crates. A crate is admitted only if it is faster on a benchmark and keeps results deterministic and `wasm32` builds working. `rayon`, `ndarray`, and `faer` were measured and none is in the tree ([Manual §2.2.1](Manual.md#221-what-is-used-and-what-was-tried-and-not-kept)). |
| **CPU only, deterministic** | Every matmul splits across a worker pool. Training, generation, and evaluation give bit-for-bit identical results at any pool size (override with `FERRUM_NUM_THREADS`), and a test enforces this. One knob changes the trained weights: the number of data-parallel training workers, `--threads` (auto = the core count). Fix it to reproduce a run on another machine. |
| **GGUF only** | Every file Ferrum writes is GGUF v3: models, training checkpoints, tokenizer files, and pre-tokenized corpora. The old FINF `.bin` format is removed. |
| **Runs in llama.cpp, Ollama, and LM Studio** | A trained BPE SLM is a `gpt2`-architecture GGUF. The same file gives the same greedy continuation in Ferrum and in llama.cpp (a test checks this). Models can be as small as about 0.1M parameters. |
| **Quantization-aware** | QAT simulates GGUF Q8_0 exactly, so an int8 file reloads bit-identical to the model you trained. int4 (Q4_0) export is a lossy option. |
| **Byte-level BPE with a GPT-2 pre-tokenizer** | Any UTF-8 text round-trips. llama.cpp tokenizes prompts exactly as Ferrum does. `<|endoftext|>` is always a reserved special token. |
| **A real data pipeline** | `train_transformer prepare` cleans, dedups, and splits a corpus and writes a hash manifest. DVC reproduces fetch, prepare, train, and eval from `params.yaml`. |
| **Corpora larger than RAM** | `train_transformer tokenize` writes a pre-tokenized GGUF token file (2 bytes per token); `train tokens.gguf` streams windows from disk and gives the same model, bit for bit, as training on the text ([Manual §6.11](Manual.md#611-corpora-larger-than-ram-tokenize-and-token-files)). |
| **Serving and standard evaluation** | `ferrum-serve` exposes log-likelihood and generation over HTTP, on loopback by default, with an optional bearer token (`FERRUM_TOKEN`) that a non-loopback bind requires. A LightEval module evaluates Ferrum models through it. |
| **Llama/Qwen GGUF in and out** | Import Llama/Qwen checkpoints, run them, fine-tune them with AdamW, and export them back to GGUF. |

One limit to know. Ferrum's tabular models (the ferrum-ml MLP classifiers) and
character-level models are also GGUF files. They do **not** run in llama.cpp,
because llama.cpp has no MLP-classifier architecture and no character-level
tokenizer. Only BPE transformer SLMs run there.

---

## Contents

1. [Workspace layout](#workspace-layout)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Run your model elsewhere](#run-your-model-elsewhere)
5. [How big a model can I train?](#how-big-a-model-can-i-train)
6. [Documentation](#documentation)
7. [Building and testing](#building-and-testing)
8. [License](#license)

---

## Workspace layout

| Path | Kind | What it is |
|---|---|---|
| `ferrum_core` | library | The engine: tensors, layers, the transformer trainer, tokenizer, quantization, the GGUF model codecs, and the GGUF reader/writer plus Llama/Qwen runner and trainer |
| `slm_cli` | binary `train_transformer` | Train, generate, evaluate, inspect, `prepare` and `tokenize` data; `export-ollama`, `export-lmstudio`; `run-gguf`, `eval-gguf`, `finetune-gguf`, `export-gguf` for external checkpoints |
| `slm_server` | binary `ferrum-serve` | HTTP inference server (axum): log-likelihood, rolling log-likelihood, generation |
| `slm_wasm` | cdylib + rlib | `wasm-bindgen` bindings that run a trained SLM in the browser (`Slm` text API, `TransformerSLMModel`), plus a no-framework demo page and Web Worker in `slm_wasm/demo/` |
| `eval/lighteval/` | Python package | `FerrumModel` and custom tasks for Hugging Face LightEval 0.13.0, plus `run.sh` |
| `dvc.yaml`, `params.yaml`, `dvc.lock` | DVC pipeline | fetch, prepare, train, eval (and an opt-in `lighteval` stage in `eval/lighteval/dvc.yaml`) |
| `scripts/` | shell | `fetch.sh` (pinned download with sha256 check), `bench_slm.sh`, `build_wasm.sh`, `verify_runtimes.sh` |
| `tests` | integration tests | Cross-crate integration and regression tests, including the llama.cpp/Ollama/LM Studio compatibility test |
| `ferrum_gui` | Tauri app (built separately) | **Ferrum SLM Studio**, a desktop GUI; see [Manual §10](Manual.md#10-the-gui-ferrum-slm-studio) |

`ferrum_gui` is deliberately **outside** the Cargo workspace because it needs
heavy system WebView libraries. Build it from its own directory.

**Sibling folders.** Two folders next to this repository hold code that used
to live here (see [Status §2](Status.md#2-the-2026-10-03-restructuring)):

- `../ferrum-ml` holds the machine-learning demos that are not about SLMs: a
  tabular MLP classifier/regressor, its CSV loader, ten datasets, pretrained
  models (now GGUF), and a browser binding. It builds on `ferrum_core` by path.
- `../ferrum-junk` is an archive of removed features.
- `../ferrum-gpu` holds the plan (`PLAN.md`) for a GPU-capable edition of this engine. There is no code yet.

---

## Installation

### Requirements

| Requirement | Version | Notes |
|---|---|---|
| Rust + Cargo | 1.87 or newer, edition 2021 | Install via [rustup](https://rustup.rs). The dependency tree needs 1.85 (`sha2`), and the code uses `is_multiple_of`, stable since 1.87. Built and tested with 1.96.0; older toolchains are untested. |
| OS | Linux, macOS, or Windows | Pure CPU code with no platform-specific paths |
| `curl` | any | Only for `scripts/fetch.sh` (Samples 1 and 2) |
| GNU coreutils | Linux | The samples use `sha256sum`, `md5sum`, and `timeout`; they are written for Linux |
| Python 3.10+ | optional | Only for DVC and LightEval (Samples 2 and 8) |
| `wasm32-unknown-unknown` | optional | Only for browser builds (`slm_wasm`) |
| WebView system libraries | optional | Only for the desktop GUI; see [Manual §10.2](Manual.md#102-building-and-launching) |
| llama.cpp, Ollama, LM Studio | optional | Only to run Ferrum models there (Sample 7) |

You need only Rust to train, evaluate, generate, and serve. DVC and Python are
optional.

### Build and verify

```bash
git clone https://github.com/thomascherickal/Ferrum
cd Ferrum
cargo build --workspace --release   # LTO + 1 codegen unit; use it for real training
cargo test  --workspace --release   # all unit + integration tests should pass
```

Debug builds are several times slower. Use `--release` for any real training or
for running GGUF models.

### Install the CLI and the server, or use the library

```bash
cargo install --path slm_cli        # installs `train_transformer`
cargo install --path slm_server     # installs `ferrum-serve`
```

```toml
[dependencies]
ferrum_core = { git = "https://github.com/thomascherickal/Ferrum" }
```

`ferrum_core` forbids `unsafe` code. Its dependencies are the few listed in
[Manual §2](Manual.md#2-why-rust-and-what-few-dependencies-buy); `cargo deny check`
covers their licences.

---

## Quick start

### Prepare, train, generate, evaluate

```bash
# 1. Download the pinned TinyStories file (22,502,601 bytes, sha256 is checked).
scripts/fetch.sh \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt \
  6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d \
  data/raw/tinystories.txt

# 2. Clean, dedup, and split it. Writes train/val/test.txt and manifest.json.
train_transformer prepare --input data/raw/tinystories.txt --out data/prepared \
    --dedup --nfkc --min-chars 50 --max-chars 20000

# 3. Train the 507,552-parameter reference model.
train_transformer train data/prepared/train.txt models/slm.gguf \
    --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 \
    --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1 \
    --val 0.02 --patience 3 --tokenizer models/tokenizer.gguf

# 4. Use it.
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 100 --temp 0.7
train_transformer eval     models/slm.gguf data/prepared/test.txt --json metrics/test.json
train_transformer info     models/slm.gguf
```

Interrupted? Add `--checkpoint run.fckp --checkpoint-every 500` to the training
command, then repeat the same command with `--resume run.fckp`. To stop on
purpose, pass `--max-seconds N` or create the file `<model>.stop`; the run
checkpoints and exits cleanly ([Manual §6.6](Manual.md#66-validation-checkpoints-and-resuming)).
[Samples.md](Samples.md) walks through every step with real output, and
[Manual §6](Manual.md#6-training-an-slm) explains every flag.

The CLI is strict. An unknown flag, a value that does not parse, or a missing
`--resume` file stops the command with an error that names the problem, and so
does an extra positional argument (`info m.gguf extra`). `train_transformer
--version` prints the version, and `train_transformer <command> --help` prints
that command's flags.

### Reproduce it with DVC

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dataops.txt
.venv/bin/dvc repro          # fetch -> prepare -> train -> eval
.venv/bin/dvc metrics show
```

Edit `params.yaml` and run `dvc repro` again to rerun only the stages that
changed. Training is deterministic, so the same inputs give the same
`models/slm.gguf` on any machine: the pipeline pins the data-parallel shard count (`train.threads` in `params.yaml`; [Manual §6.8](Manual.md#68-multi-threading-and-determinism)).

### Use the library

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

let corpus = std::fs::read_to_string("corpus.txt")?;
let cfg = TransformerConfig {
    context_len: 64, embed_dim: 64, num_heads: 4, num_blocks: 2, hidden_dim: 128,
    vocab_size: 512, weight_tying: true, lr: 0.003, batch_size: 16,
    max_tokens: 5_000_000, cosine_lr: true, warmup_steps: 100, grad_clip: 1.0,
    ..TransformerConfig::default()
};
let mut rng = Rng::new(1337);
let slm = GenerativeSLM::train_transformer_config_threaded(&corpus, &cfg, 0, &mut rng, |ep, loss| {
    println!("epoch {ep}: loss {loss:.4}");
})?;
slm.save("model.gguf")?;                                   // int8 GGUF (gpt2 layout)
let s = slm.score("The cat sat on the", " mat")?;           // log p(continuation | context)
println!("logprob {:.3} over {} tokens", s.logprob, s.n_tokens);
```


### Run, fine-tune, and export a Llama/Qwen GGUF model

```bash
train_transformer run-gguf      qwen2-0_5b.gguf "Explain rust ownership" --quant int4 --max 64
train_transformer finetune-gguf smollm2-135m.gguf corpus.txt tuned.flck --epochs 1 --lr 1e-4
train_transformer export-gguf   smollm2-135m.gguf tuned.gguf --resume tuned.flck --quant q8_0
```

Supported architectures are `llama` and `qwen2`. See
[Manual §9](Manual.md#9-gguf-ferrum-models-other-runtimes-serving-and-external-models).

---

## Run your model elsewhere

A Ferrum BPE SLM is a normal `gpt2` GGUF. Use it with other runtimes directly,
or with a one-command export.

```bash
# llama.cpp
llama-completion -m models/slm.gguf -p "Once upon a time" -n 16 --temp 0 -no-cnv --no-warmup </dev/null 2>/dev/null

# One-command exports (Ferrum gpt2 models only: each refuses anything else)
train_transformer export-ollama   models/slm.gguf ferrum-tiny   # writes a Modelfile, runs `ollama create`
train_transformer export-lmstudio models/slm.gguf               # links it into LM Studio with `lms import -l`
```

Versions verified so far: llama.cpp b11521, Ollama v0.40.2, LM Studio 0.4.18.
Each tool is found from a `--bin` flag (export commands), then the env var
(`FERRUM_LLAMA_BIN`, `FERRUM_OLLAMA`, `FERRUM_LMS`), then `PATH`, then the default
install location (`~/.local/opt/llama.cpp-b11521`, `~/.local/bin/ollama`,
`~/.lmstudio/bin/lms`).
The exact commands, and what to expect from each tool, are in
[Samples.md §7](Samples.md#7-serving-and-running-your-model-in-llamacpp-ollama-and-lm-studio)
and [Manual §9.5](Manual.md#95-running-ferrum-models-in-llamacpp-ollama-and-lm-studio).

Serve a model over HTTP with `ferrum-serve --model models/slm.gguf` and
evaluate it with LightEval: [Manual §9.7](Manual.md#97-serving-ferrum-serve) and
[Manual §9.8](Manual.md#98-evaluating-with-lighteval).

---

## How big a model can I train?

Measured on an Intel i5-1135G7 (4 cores / 8 threads, 15 GiB RAM) on 2026-10-26 with int8 QAT,
batch 16, and 8 threads. Compute is the binding limit. RAM is not, and a 2 GB
GPU makes no difference because Ferrum never uses it.

| Goal | Reachable size |
|---|---|
| Heavily over-trained, minutes to hours | 0.1M–2M parameters |
| Compute-optimal (about 20 tokens/param) in 24 h | about 4M |
| Compute-optimal in a week | about 12M |
| Compute-optimal in a month | about 27M (extrapolated) |
| Fine-tune an imported GGUF (full AdamW, f32) | about 135M comfortably; 0.5B is tight |

The sizes come from short runs of four configurations, after the speed work in
this release ([Manual §6.7](Manual.md#67-sizing-a-run-for-your-machine)). A
TinyStories-class model (1–3M parameters on about 470M tokens) is a 2–4 day
job. The [capability report](docs/ferrum-capabilities.md) estimates what 8 to 128 GB machines and recent Intel, Ryzen and Snapdragon processors can do. [Manual §6.7](Manual.md#67-sizing-a-run-for-your-machine) has the throughput
table, and [Status §3](Status.md#3-capability-assessment) has the assessment.

Two reference sizes are used in the docs:

| Config | Flags | Parameters |
|---|---|---|
| Reference 500K | `--context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024` | 507,552 |
| Smallest supported reference | `--context 64 --embed 64 --heads 4 --blocks 2 --hidden 128 --vocab 512 --tie` | 103,936 |

`--vocab N` is a ceiling: a small corpus supplies fewer BPE merges, and `train`
then prints `vocab: 282 (requested 512, limited by the corpus)`. The counts
above assume a corpus large enough to fill the vocabulary, as the TinyStories
reference does ([Manual §6.2](Manual.md#6-training-an-slm)).

---

## Documentation

| Document | Read it for |
|---|---|
| **[Manual.md](Manual.md)** | Everything about *using* Ferrum, from first principles through corpus preparation, training, generation, evaluation, the GGUF format, running in other runtimes, DVC, serving, LightEval, the GUI, and deployment, to the API reference, benchmarks, FAQ, and an honest critique |
| **[Samples.md](Samples.md)** | Ten executed walk-throughs with real commands and output: quick start, DVC, data cleaning, training controls, evaluation, generation, serving and other runtimes, LightEval, benchmarking, and the library API |
| **[Status.md](Status.md)** | Where the project stands: what is implemented and tested, known issues, the roadmap, the history, and the backward-compatibility record (§7) |
| **[glossary.md](glossary.md)** | A beginner's guide to LLM terms |
| `cargo doc -p ferrum_core --open` | Rustdoc for every public item |

---

## Building and testing

```bash
cargo build --workspace                       # everything except the GUI
cargo test  --workspace --release             # unit + integration tests
cargo test  -p slm_cli --release --test determinism   # 1 thread vs N threads, identical output
cargo bench --bench gemm                      # matmul/decode microbenchmark
cargo clippy --workspace --all-targets -- -D warnings
cargo deny check                              # licences and advisories for the dependencies
bash scripts/build_wasm.sh                    # slm_wasm -> slm_wasm/pkg/ and the demo (needs wasm-bindgen)
python3 -m http.server -d slm_wasm/demo 8000  # the browser demo: http://localhost:8000/ (Manual §11.3)
(cd ferrum_gui && cargo tauri dev)            # the desktop app

train_transformer prepare --input corpus.txt --out data/prepared   # clean + split + manifest
cargo run --release -p slm_server -- --model models/slm.gguf       # ferrum-serve on 127.0.0.1:8080
.venv/bin/dvc repro                                                # whole data-to-metrics pipeline
python3 -m unittest eval/lighteval/test_ferrum_model.py            # LightEval module unit tests
eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt --out metrics/lighteval.json
```

Runtime compatibility tests skip a tool that is not installed. Set
`FERRUM_REQUIRE_RUNTIMES=1` and a missing tool becomes a test **failure**
instead. CI and pre-release runs set it. Tool discovery is described above.

CI (`.github/workflows/ci.yml`) runs build and test, rustfmt, clippy, JS syntax
checks, cargo-deny, a wasm32 build of `slm_wasm`, the LightEval module's import
check, and the GUI crate's own format, lint, and test jobs. A `runtimes` job
runs the compatibility tests with `FERRUM_REQUIRE_RUNTIMES=1`: llama.cpp b11521
and Ollama v0.40.2 are required there, and `FERRUM_SKIP_LMSTUDIO=1` is its one
exemption (LM Studio is required locally before a release). Python and DVC are
otherwise not in CI.

---

## License

MIT. See [LICENSE](LICENSE). The default corpus (TinyStories V2) is
CDLA-Sharing-1.0. Ferrum downloads it and never commits it.
