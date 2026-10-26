# The Ferrum Manual

This manual covers everything about *using* Ferrum. It starts from first
principles for readers who have never trained a model. It then walks through
preparing data, training, generating, evaluating, running external models, the
GUI, and deployment. It ends with the complete reference.

For a one-page overview and installation, see [ReadMe.md](ReadMe.md). For
executed walk-throughs with real output, see [Samples.md](Samples.md). For what
is implemented, tested, and planned, see [Status.md](Status.md).

**How to read it.**

- New to all of this? Read Part I in order, then §6.10, a complete worked
  walkthrough, and try [Samples.md](Samples.md).
- Know the basics already? Go to Part II and read only the chapter for the
  task in front of you.
- Integrating the library? Part III is the reference.

The manual separates *excitement* from *honesty* on purpose. Several chapters
explain what Ferrum and small models can do. [§17](#17-limits-an-honest-critique)
is about what they cannot do, and never will. [§3](#3-data-decides-garbage-in-garbage-out)
argues the point beginners miss most often: the **data**, not the model,
usually decides whether a project succeeds.

---

## Contents

**Part I — Foundations**
1. [Generative AI, SLMs, and Transformers from zero](#1-generative-ai-slms-and-transformers-from-zero)
2. [Why Rust, and what few dependencies buy](#2-why-rust-and-what-few-dependencies-buy)
3. [Data decides: garbage in, garbage out](#3-data-decides-garbage-in-garbage-out)

**Part II — Using Ferrum**

4. [How Ferrum is built](#4-how-ferrum-is-built)
5. [Preparing a corpus](#5-preparing-a-corpus)
6. [Training an SLM](#6-training-an-slm)
7. [Generating text](#7-generating-text)
8. [Evaluating a model](#8-evaluating-a-model)
9. [GGUF: Ferrum models, other runtimes, serving, and external models](#9-gguf-ferrum-models-other-runtimes-serving-and-external-models)
10. [The GUI (Ferrum SLM Studio)](#10-the-gui-ferrum-slm-studio)
11. [Deploying models](#11-deploying-models)

**Part III — Reference**

12. [Library API reference](#12-library-api-reference)
13. [Performance and benchmarks](#13-performance-and-benchmarks)
14. [Working with any language](#14-working-with-any-language)
15. [Use cases and applications](#15-use-cases-and-applications)
16. [FAQ](#16-faq)
17. [Limits: an honest critique](#17-limits-an-honest-critique)

[Sources](#sources)

---

# Part I — Foundations

## 1. Generative AI, SLMs, and Transformers from zero

> **Who this is for:** someone who has heard "AI", "language model", or
> "transformer" but could not confidently explain them. No maths required.

### 1.1 The single idea: guess the next piece

Shown *"The cat sat on the ____"*, you think "mat". You are not remembering
that sentence. You are predicting what is likely to come next. A **language
model** is a program trained on large amounts of text to do exactly this: given
some text, predict the next small piece. Append the prediction, predict again,
and repeat, and the machine *generates* new text one piece at a time. This loop
is **autoregressive generation**.

Because each piece depends on the one before, generation is *inherently
sequential*. That one fact explains why text generation is hard to speed up
with more CPU cores ([§13](#13-performance-and-benchmarks)).

**Generative AI** is the umbrella term for systems that *produce* content, as
opposed to systems that *classify* it.

### 1.2 Training

A fresh model's internal numbers (**parameters**, or **weights**) are random,
so its guesses are gibberish. **Training** repeatedly does the following:

1. Show the model real text and hide what comes next.
2. Let it guess.
3. Score how wrong it was. This score is the **loss**.
4. Nudge every weight slightly in the direction that would have reduced the
   loss (**backpropagation** and **gradient descent**, which Ferrum writes out
   by hand).

One pass over the training text is an **epoch**. Falling loss means the model
is getting less wrong.

> **Key intuition:** a model is not a database that looks up answers. It is a
> large adjustable function tuned so that plausible text comes out. That is
> also why it can be confidently wrong ([§17](#17-limits-an-honest-critique)).

### 1.3 Tokens and byte-level BPE

Computers see numbers, not letters. A **tokenizer** chops text into
**tokens** and gives each one an ID. A token can be a word, part of a word, a
character, or a byte.

- One token per character is simple but wasteful: the model takes many tiny
  steps to produce one word.
- One token per word is compact but breaks on words it has never seen.
- **Byte-Pair Encoding (BPE)** is the compromise used by GPT-2, GPT-4, and
  most modern models. It starts from a small alphabet and *learns* to merge the
  most frequent adjacent pairs into new tokens.

Ferrum uses **byte-level BPE**, which starts from the 256 possible byte values.
Every file in every language is made of those 256 values, so **any text can be
tokenized** (English, emoji, Chinese, Arabic, code) and there is no "unknown
character" error.

### 1.4 The Transformer

The 2017 paper *Attention Is All You Need* introduced the **Transformer**, the
architecture behind nearly every modern language model. Its key mechanism is
**self-attention**: while processing one token, the model looks at every other
token in its context and weights the relevant ones. In *"The animal didn't
cross the street because **it** was too tired"*, attention lets "it" draw on
"animal".

| Term | Plain-language meaning |
|---|---|
| Self-attention | Every token looks at every other token and decides what is relevant |
| Multi-head attention | Several attention "viewpoints" in parallel, each catching a different relationship |
| Embedding | A lookup table that turns token IDs into lists of numbers |
| Positional encoding | Tells the model the *order* of tokens, since attention sees them all at once |
| Feed-forward network (FFN) | A small neural network applied after attention |
| Layer normalization | Keeps numbers in a healthy range so training stays stable |
| Block / layer | One round of attention + FFN + normalization; stacking blocks learns deeper patterns |
| Context window | How many tokens the model can see at once |

Ferrum implements two Transformer flavours by hand. One is its own design,
which you train from your text. The other is the Llama/Qwen design it imports
from downloaded files ([§4](#4-how-ferrum-is-built)).

### 1.5 LLMs and SLMs

**Large language models** (ChatGPT, Claude, Gemini, Llama) have billions to
trillions of parameters, cost millions to train, and usually run in the cloud.
**Small language models** use the same architecture kept deliberately small:
generally under about 10 billion parameters, often far smaller. That lets them
run on laptops, phones, and microcontrollers. Recent industry research argues
SLMs are "sufficiently powerful, inherently more suitable, and necessarily more
economical" for the many narrow, repetitive jobs that make up most real AI
pipelines. The advantages: lower latency, less memory, lower cost, easier
deployment, and private offline operation.

> **Where Ferrum sits:** at the *very small* end. Ferrum trains models from
> about 0.1M to about 30M parameters on a laptop CPU, and runs or fine-tunes
> small open models that others trained. That makes it excellent for learning,
> experimentation, and narrow tasks, and unsuitable as a ChatGPT replacement.

### 1.6 Training and inference

**Training** teaches the model: slow, done once, compute-heavy. **Inference**
uses the trained model to generate or predict: fast and repeated. Ferrum does
both on the CPU in one tool. It can also run and fine-tune models someone else
trained.

### 1.7 The journey of one sentence

```text
"the quick brown fox"
   │ (1) TOKENIZER  → [412, 88, 1290, 77]
   │ (2) EMBEDDING  → a vector of numbers per token
   │ (3) TRANSFORMER BLOCKS → attention + FFN, several times
   │ (4) OUTPUT     → a probability for every possible next token
   │                  " jumps" 71%, " ran" 9%, " sat" 4%, …
   │ (5) SAMPLING   → pick one (temperature = how adventurous)
   └──► append " jumps" and repeat from (1)
```

**Temperature** controls how adventurous sampling is. Low values (0.1–0.3)
play it safe and repeat learned patterns. High values (0.8 and up) give more
variety and more risk.

---

## 2. Why Rust, and what few dependencies buy

### 2.1 Rust in one minute

**Rust** compiles to native machine code like C and C++, but its compiler
enforces ownership rules that make whole categories of memory bugs impossible
to compile. You get C-level speed with guarantees C cannot offer, and no
garbage collector pausing the program.

### 2.2 The three guarantees

- **Memory safety without a garbage collector.** There are no surprise pauses
  while tokens are streaming.
- **`#![forbid(unsafe_code)]` in every crate.** The compiler rejects the code if
  it uses Rust's `unsafe` escape hatch, and a repo-wide test (`test_no_unsafe`)
  scans every source file for `unsafe` and checks that every crate root
  forbids it. The cost is concrete: reading a downloaded model cannot use
  memory-mapping (which needs `unsafe`), so Ferrum streams files with ordinary
  safe reads instead. Dependencies may use `unsafe` inside themselves; Ferrum
  calls only their safe APIs, which rules out `unsafe fn` entry points such as
  `matrixmultiply::sgemm`.
- **Few dependencies, each with a reason.** `ferrum_core`, the engine, depends
  on two pure-Rust Unicode-table crates and nothing else. It stays `std`-only
  for threading, linear algebra, parsing, and file formats. The other crates add
  what their job needs (§2.2.1). The rule for a new crate:
  1. a benchmark shows a gain (or the crate replaces more code than it adds),
  2. results stay bit-for-bit identical at 1 thread and at N threads (a test
     enforces this), and
  3. `ferrum_core` and `slm_wasm` still build and run on
     `wasm32-unknown-unknown`.

  A crate that fails any of the three is rejected. `deny.toml` lists the
  allowed licences, and `cargo deny check` runs in CI.

### 2.2.1 What is used, and what was tried and not kept

The dependencies as they are in each `Cargo.toml`:

| Crate(s) | Used by | Used for |
|---|---|---|
| `unicode-general-category`, `unicode-normalization` | `ferrum_core` | The GPT-2 pre-tokenizer's `\p{L}`/`\p{N}` classes, and NFKC for `prepare --nfkc`. Both are pure Rust tables and build for `wasm32`. |
| `sha2`, `serde_json` | `slm_cli`, `slm_server` | sha256 in `prepare` manifests and the server's `model_sha256`; `manifest.json`, `eval --json`, and the server's JSON. Hashes are correctness-critical, and a hand-written one is more code to audit. |
| `axum`, `tokio`, `serde` | `slm_server` only | The HTTP server. The engine does not depend on them. |
| `wasm-bindgen` | `slm_wasm` only | Browser bindings. |
| `tauri` and friends | `ferrum_gui` only | The desktop GUI, outside the workspace. |

Three matmul or threading crates were benchmarked against the hand-written
kernel and worker pool on the 507,552-parameter reference model. None is in
the tree. All rows are the median of three runs of `scripts/bench_slm.sh` and
`cargo bench --bench gemm` at 8 threads on the machine in
[Samples §9](Samples.md#9-benchmarking), measured on 2026-10-24 with the low-memory
build settings (`CARGO_PROFILE_RELEASE_LTO=off`, 16 codegen units).

| Candidate | Train tokens/s | Eval s | Generate ms/token | GEMM 2048³ GFLOP/s | Outcome |
|---|---:|---:|---:|---:|---|
| Starting point: previous kernel, hand-written pool | 6,726 | 1.63 | 0.292 | 35.2 | |
| `rayon` pool instead of the hand-written pool | 6,836 (+1.6%) | 1.49 | 0.293 | 33.0 | Rejected: no gain. |
| `ndarray` (`matrixmultiply` backend) | 9,429 (+40%) | 1.44 | 0.459 | 176.9 | Not adopted: it clears the 10% bar but makes generation 57% slower, and it would be the engine's first dependency on the hot path. A hybrid (`ndarray` for large matmuls, own kernel for single-token decode) gave 9,036 (+34%) at 0.301 ms/token. Whether to take it is an open decision ([Status §6](Status.md#6-roadmap)). |
| `faer` | 8,368 (+24%) | 1.50 | 0.235 | 158.2 | Rejected: it failed the thread-count determinism rule. |
| **A 4×8 register tile in the existing kernel (kept)** | **8,056 (+20%)** | 1.48 | 0.304 | 49.8 | Kept. Safe `std` Rust, and bit-identical results to the kernel it replaced. |

| Benefit | Why it matters |
|---|---|
| Small, readable engine | The forward and backward pass, the tokenizer, and the GGUF codecs are all in the repo |
| Few crates to vet | `cargo deny check` covers the whole tree; the engine crate has a two-line dependency list |
| Runs anywhere Rust runs | Servers, laptops, Raspberry Pi, browsers |
| Reproducibility | A determinism test blocks any change that makes results depend on the thread count |

### 2.3 Why Rust fits CPU inference

Rust has no global interpreter lock, so it can safely spread work across every
core, which Ferrum does for every matrix multiply. Its efficiency suits edge
and resource-constrained systems. The wider industry is moving the same way:
Burn, Candle, mistral.rs, and MicroFlow all serve or run models in Rust.

### 2.4 One codebase, many destinations

Because the engine is pure Rust, it compiles to **WebAssembly**. A model
trained on your laptop can run entirely in a visitor's browser, with no server,
no API key, and no data leaving their machine ([§11.3](#113-in-the-browser-webassembly)).

### 2.5 The trade-offs, honestly

- Rust is harder to learn than Python. *Using* Ferrum requires no Rust.
- Admitting a crate costs something: more code you did not write, a larger
  build, and a licence and advisory list to watch. The rule above keeps that
  cost to crates that pay for themselves, and for the engine that is two
  Unicode tables.
- Ferrum does not call OpenBLAS or MKL, so it is **not the fastest possible CPU
  engine**. It trades peak speed for transparency, safety, and portability.
- The AI ecosystem still lives mostly in Python, so ready-made Rust resources
  are fewer. (Ferrum's LightEval module and DVC pipeline use Python tools from
  outside the engine.)

---

## 3. Data decides: garbage in, garbage out

> If you read only one chapter before training anything, read this one.

### 3.1 The model is a mirror

A language model sees nothing but its training text. It cannot learn what is
not there, and it cannot un-learn what is. **Garbage In, Garbage Out (GIGO)**
was first recorded in 1957 and is usually credited to IBM's George Fuechsel.
Machine learning amplifies it. A buggy program fails predictably. A model
trained on bad data fails in ways baked invisibly into millions of numbers, and
it fails confidently.

### 3.2 Why data matters even more for small models

A giant LLM can average away noise across trillions of tokens. A small model
cannot. Every token must earn its place, and junk consumes scarce capacity.
Microsoft's *Textbooks Are All You Need* showed a 1.3B model trained on curated
"textbook-quality" data beating models about 10× larger trained on about 100×
more data. **Quality beats volume**, and with Ferrum's tiny models quality is
the only lever you have.

### 3.3 What good data is

| Quality | What goes wrong without it |
|---|---|
| **Relevant** to your task and domain | Train on Shakespeare, get Shakespeare |
| **Representative**, including the right *language* | An English corpus cannot speak French |
| **Clean**: no boilerplate, markup, OCR errors | The model learns `<div>` tags and page numbers |
| **Correct** | Errors are learned and repeated confidently |
| **Consistent** in format | Capacity is wasted on pointless variation |
| **Sufficient and diverse** | The model memorizes instead of generalizing |
| **De-duplicated** | Repeats are over-weighted and encourage memorization |
| **Balanced** | Biased data in, biased output out |
| **Valid UTF-8** | Mojibake in, mojibake out |
| **Legally and ethically yours** | Legal and privacy risk |

A useful test: *if a careful human read this corpus, would they learn the right
thing from it?*

### 3.4 The asymmetry

Good data cannot turn a tiny model into a genius. But bad data guarantees
failure at *any* size. Great data is necessary but not sufficient, and bad data
is sufficient on its own to ruin a model. Spend your effort accordingly.
[§5](#5-preparing-a-corpus) shows the tools Ferrum gives you.

---

# Part II — Using Ferrum

## 4. How Ferrum is built

### 4.1 The workspace

| Part | What it is |
|---|---|
| `ferrum_core` | The engine library. Everything else calls into it. |
| `slm_cli` | The `train_transformer` command-line tool: train, run, generate, eval, info, prepare, run-gguf, finetune-gguf, export-gguf, export-ollama, export-lmstudio |
| `slm_server` | The `ferrum-serve` HTTP server: log-likelihood, rolling log-likelihood, and generation routes ([§9.7](#97-serving-ferrum-serve)) |
| `slm_wasm` | Browser bindings for trained SLMs |
| `eval/lighteval/` | Python: a LightEval 0.13.0 model class and custom tasks that evaluate a Ferrum model through the server ([§9.8](#98-evaluating-with-lighteval)) |
| `dvc.yaml`, `params.yaml` | The fetch → prepare → train → eval pipeline ([§9.9](#99-dvc-pipeline)) |
| `ferrum_gui` | **Ferrum SLM Studio**, a point-and-click desktop app ([§10](#10-the-gui-ferrum-slm-studio)) |
| `tests` | Integration and regression tests |

### 4.2 Two Transformer stacks

The single most common source of over-claiming is mixing these two up:

| | Ferrum's own (`train_transformer`) | Imported (`llm`) |
|---|---|---|
| Positions | learned absolute embedding | RoPE (rotary) |
| Norm | LayerNorm | RMSNorm |
| FFN | GELU MLP (tanh approximation, as GPT-2) | SwiGLU |
| Attention | dense multi-head | grouped-query (GQA) |
| Training | full backprop, Adam/AdamW, QAT, from scratch | gradient-checked backprop, AdamW fine-tuning (f32) |
| Source | trained from your text | imported from a GGUF file |
| Tokenizer | Ferrum byte-level BPE (stored in the model) | the checkpoint's own (BPE exact, SPM approximate) |
| Ships as | `gpt2`-architecture GGUF (runs in llama.cpp, Ollama, LM Studio, and in Ferrum natively and in WASM) | GGUF |

They share the low-level kernels, worker pool, and quantization grid, and
nothing above that.

### 4.3 Architecture at a glance

```text
Tensor ──► ops (matmul, qlinear, softmax, layernorm)  ──► parallel (persistent worker pool)
   │
   └──► Layer trait: Linear[+packed QWeight] · ActivationLayer · LayerNorm · Embedding
                     · Flatten · TransformerBlock (causal MHA + FFN) · KvCache
Sequential ──► ordered pipeline of layers

tokenizer          ByteBpeTokenizer (byte-level BPE with the GPT-2 pre-tokenizer; char-level fallback)
quant              int8/int4 fake-quant for QAT; packed in-memory QWeight
train_transformer  TransformerNet: backprop, data-parallel epochs, stride, step budgets
slm                GenerativeSLM: config-driven training pipeline, generate, evaluate
meta / loader      ModelMetadata + the GGUF codecs for Ferrum models (GPT-2 layout, generic layout)
dataset            clean_corpus, corpus_stats, validate_for_training, prepare's cleaning steps

── imported architecture ──
gguf / gguf_tokenizer   std-only GGUF v2/v3 reader + the checkpoint's own tokenizer
llm                     LlamaModel: RMSNorm, RoPE, GQA, SwiGLU, KV-cached decode
llm_train               LlamaTrainer: gradient-checked backprop, AdamW, .flck checkpoints
gguf_write              GgufBuilder + block encoders → GGUF v3 export
chat_format             chat-template detection and rendering for instruct models
```

---

## 5. Preparing a corpus

Any UTF-8 text file is a corpus. For a good one, use `prepare`: a repeatable,
streaming cleaner and splitter that records exactly what it did
([Samples §3](Samples.md#3-cleaning-your-own-corpus) runs it on a deliberately messy input).
The same cleaning steps are available as library functions and in the GUI's
**Datasets** tab ([§10.3](#103-the-tabs)).

### 5.1 `train_transformer prepare`

```text
train_transformer prepare --input <file|dir> [--input ...] --out <dir>
    [--doc-sep S] [--lossy] [--nfkc] [--lowercase]
    [--strip-gutenberg] [--min-chars N] [--max-chars N] [--dedup]
    [--val-frac F] [--test-frac F] [--seed N]
```

| Flag | Default | Meaning |
|---|---|---|
| `--input PATH` | required, repeatable | A file or a directory. Directories are walked in sorted order and **every** file in them is read, whatever its extension (hidden files and anything under `--out` are skipped), so a stray `.bin` counts as an invalid-UTF-8 document. A file is one document unless it contains the separator. |
| `--out DIR` | required | Output directory. An input that is one of this run's own outputs is refused. Splits and manifest are written to `*.part` files and renamed into place only after every input was read and validated, so a failed run leaves the previous outputs untouched and a fresh directory empty. |
| `--doc-sep S` | `<\|endoftext\|>` | The string that separates documents inside a file. |
| `--lossy` | off | Replace invalid UTF-8 bytes with U+FFFD. Without it, invalid UTF-8 is an error that names the file and byte offset. |
| `--nfkc` | off | Unicode NFKC normalization. |
| `--lowercase` | off | Lowercase every document. |
| `--strip-gutenberg` | off | Remove the Project Gutenberg header and footer of each book. Handles several books in one file; each book becomes its own document. |
| `--min-chars N` | 1 | Drop documents shorter than `N` characters. |
| `--max-chars N` | 0 (off) | Drop documents longer than `N` characters. |
| `--dedup` | off | Drop exact duplicate documents (a 128-bit content hash; only the hashes are held in memory). |
| `--val-frac F` | 0.05 | Share of documents for the validation split. |
| `--test-frac F` | 0.05 | Share of documents for the test split. |
| `--seed N` | 42 | Seed for the split. |

What it does to each document, in order:

1. Decode UTF-8 (an error, or replacement with `--lossy`). Every line ending
   (CRLF, and the lone CR of old Mac files) becomes `\n`.
2. Split out Gutenberg books if asked.
3. Strip the byte-order mark, zero-width characters (U+200B to U+200D, U+2060,
   U+FEFF), and control characters (NUL included) except newline and tab. They
   are removed, not replaced by a space, so the surrounding text is joined:
   `hello\0world` becomes `helloworld`.
4. Apply NFKC if asked.
5. Fold curly quotes, dashes, and the ellipsis to ASCII, collapse runs of three
   or more newlines, and lowercase if asked.
6. Collapse runs of spaces and trim the document.
7. Apply the length filter, then the duplicate filter.

Each input is read in fixed 1 MiB chunks, so a large file is not loaded whole.

**The split is by document, not by line.** A document goes to `val`, `test`, or
`train` by `H(seed, document) mod 10,000` against the fractions. Three
consequences: the same document always lands in the same split, no document
appears in two splits, and appending new documents later does not move the old
ones. With very few documents a split can be empty. An empty **train** split is an
error (exit 1, nothing written): `train split is empty: N documents found, need at
least 3 (one file is one document unless it contains the separator ...; use
--doc-sep or add more files)`. A single file without `<|endoftext|>` separators is
one document, so split it with `--doc-sep` or give `prepare` more files.

**Outputs.** `out/train.txt`, `out/val.txt`, `out/test.txt` (documents joined by
`\n<|endoftext|>\n`) and `out/manifest.json`.

### 5.2 The manifest

`manifest.json` makes a run auditable.

| Field | Content |
|---|---|
| `ferrum_version` | The version that ran `prepare` |
| `args` | The full argument list |
| `inputs` | For each input: `path`, `bytes`, `sha256` |
| `outputs` | For `train`, `val`, `test`: `path`, `bytes`, `sha256`, `documents`, `chars` |
| `rejected` | Counts by reason: `empty`, `too_short`, `too_long`, `duplicate`, `invalid_utf8`. With `--lossy`, `invalid_utf8` counts the documents that contained invalid bytes; they are kept, with U+FFFD in place of the bad bytes. |
| `train_stats` | `corpus_stats` of the train split (`bytes`, `chars`, `lines`, `unique_chars`, `words`) |

If the hashes in two manifests match, the inputs and outputs are byte-identical.
DVC tracks `manifest.json` in git (not in the DVC cache), so a change in the
data shows up in a `git diff`.

### 5.3 Library helpers

The cleaning steps live in `ferrum_core::dataset` as small functions you can
combine: `strip_invisible`, `gutenberg_bodies`, `squeeze_spaces`,
`normalize_nfkc`, `clean_corpus`, `corpus_stats`, and `validate_for_training`.

1. **`clean_corpus(raw, &CleanOptions)`**: strip Gutenberg boilerplate, fold
   punctuation, drop control characters, collapse whitespace, optionally
   lowercase, optionally cap the size. `CleanOptions::default()` strips
   Gutenberg, folds punctuation, drops control characters, and collapses
   whitespace, and does **not** lowercase and does not cap. `prepare` calls it
   with Gutenberg stripping handled separately and adds the other steps above.
2. **`corpus_stats`**: characters, bytes, lines, words, and unique characters.
   A suspiciously low unique-character count points to a corrupt or one-note
   file.
3. **`validate_for_training`**: fails fast, with a reason, if the corpus cannot
   train (for example, it is shorter than the context window).

The GUI **Datasets** tab can also download corpora directly from a URL, from
Hugging Face (including gated files with a token), or from Kaggle. It ships a
small catalog: TinyStories, TinyStories V2 (GPT-4), WikiText-2, Shakespeare,
and Poetry Foundation. A download is written to `<file>.part` and renamed only when
complete; a file over the 512 MiB cap fails with an error instead of being
truncated, and a failed download leaves no partial file.

### 5.4 The default corpus

The DVC pipeline uses the TinyStories V2 (GPT-4) validation file, about 22 MB.
Stories are already separated by `<|endoftext|>`.

| | |
|---|---|
| URL | `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt` |
| Size | 22,502,601 bytes |
| sha256 | `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d` |
| Licence | CDLA-Sharing-1.0 (downloaded, never committed) |

`scripts/fetch.sh <url> <sha256> <out>` downloads with `curl` and fails, leaving
nothing behind, if the checksum does not match. On success it prints only
`curl`'s progress meter.

**Hold text back.** Keep a separate held-out file in the same style that the
model never trains on. `prepare` makes `test.txt` for this. Alternatively,
`--val F` on `train` splits off the last fraction automatically
([§6.6](#66-validation-checkpoints-and-resuming)).

**Checklist before you train**

- [ ] The right content, in the domain and language you want
- [ ] Cleaned of boilerplate, markup, and control characters
- [ ] No large passages repeated many times
- [ ] Comfortably longer than the context window, with real variety
- [ ] Inspected: you looked at `corpus_stats` or `manifest.json` and *read a sample*
- [ ] Held-out text set aside

---

## 6. Training an SLM

### 6.1 How training works

Ferrum trains a causal Transformer by cutting the token stream into
**windows** of `context_len` tokens. In each window the model predicts the next
token at *every* position, so one window yields `context_len` training targets.
Minibatches of windows are drawn from a fresh random shuffle each epoch.

- **Window stride** is the distance between window starts. The default is
  `stride = context_len`: windows do not overlap, and **one epoch sees each
  token once**. With `--stride 1` every token starts a window (the legacy
  behaviour), so an epoch trains on each token about `context_len` times and
  costs that much more. With a stride above 1, the window grid gets a random
  offset each epoch, so boundaries differ between epochs.
- **Epochs or a token budget.** Train for `--epochs N` passes, or set
  `--tokens N` (for example `5e7`) to stop after exactly that many training
  tokens (`steps × batch × context`), even partway through an epoch. A budget
  is the right unit for real runs. "Compute-optimal" training is about 20
  tokens per parameter.
- **Quantization-aware training (QAT)** is on by default. Each step runs
  forward and backward against int8-snapped weights while Adam updates
  full-precision masters (a straight-through estimator), so the int8 file you
  ship behaves like the model you trained.

Training is int8 QAT, multi-threaded, and deterministic for a fixed seed,
configuration, and `--threads`. The model is saved as an int8 `gpt2`-architecture
GGUF file (`--quant` picks f32 or int4 instead). If the model file already
exists, `train` loads it instead of retraining; pass `--force` to retrain.

### 6.2 Command-line reference

```text
train_transformer train <corpus.txt> <model.gguf> [options]
train_transformer run   <corpus.txt> <model.gguf> <seed text> [options]   # train if needed, then generate
```

`train_transformer --help` prints the same flags. Every subcommand uses a strict
parser with its own allow-list of flags:

| Mistake | What happens |
|---|---|
| Unknown flag | Error naming the subcommand and listing the valid flags, for example `unknown flag --contxt for 'train' (valid: --context --embed ...)` |
| Value that does not parse (`--context 12x`) | Error naming the flag and the value; never a silent default |
| Flag with no value (`--temp` at the end) | `--temp needs a value` |
| `--resume` path that does not exist | Error before anything trains; use `--checkpoint` to start a new run |
| Output path that cannot be written (model, `--checkpoint`, `--tokenizer`, `finetune-gguf` `.flck`, `export-gguf` output, `eval --json`) | Error `cannot write <path>: <reason>` in well under a second, before any training or evaluation. Missing parent directories are created first, the same for all of these commands. |

A flag given twice takes its last value, except the repeatable `--stop` and
`--input`. A negative number is read as a value (`--temp -1`, which is then
rejected by the range check). `--verbose` and `-v` are accepted everywhere.

**Model file name.** Model files are GGUF. Use `.gguf`: `model.gguf`.

**Architecture**

| Flag | Default | Meaning |
|---|---|---|
| `--context N` | 16 | Context window in tokens |
| `--embed N` | 32 | Embedding width (must be divisible by `--heads`) |
| `--heads N` | 4 | Attention heads per block |
| `--blocks N` | 2 | Transformer blocks |
| `--hidden N` | 64 | FFN hidden width (typically 4× `--embed`) |
| `--vocab N` | 512 | `0` = character-level; `≥ 256` = byte-level BPE of that size (the end-of-text token counts toward it) |

**Schedule and data**

| Flag | Default | Meaning |
|---|---|---|
| `--epochs N` | 100 | Training epochs (ignored when `--tokens` is set) |
| `--tokens N` | 0 (off) | Token budget, e.g. `5e7`; stops mid-epoch if need be. The banner then shows `Epochs  : budget`. |
| `--stride N` | 0 (= context) | Window stride; `1` = legacy full overlap |
| `--batch N` | 16 | Sequences per optimizer step |
| `--tokenizer PATH` | none | Reuse a saved BPE tokenizer (a GGUF file); trained and saved there if missing |
| `--val F` | none | Hold out the last fraction `F` for validation; keep the best epoch. A tokenizer trained in the same run sees only the train portion. |
| `--patience N` | 0 | With `--val`: stop after `N` epochs without improvement (0 = never) |

**Optimization and output**

| Flag | Default | Meaning |
|---|---|---|
| `--lr F` | 0.01 | Adam learning rate (peak rate when `--cosine` is on) |
| `--cosine` | off | Linear warmup, then cosine decay to 0 over the whole run |
| `--warmup N` | 0 | Warmup steps for `--cosine` |
| `--clip F` | 0 (off) | Global gradient-norm clipping threshold |
| `--weight_decay F` | 0 | AdamW decoupled weight decay |
| `--dropout F` | 0 | FFN-hidden dropout during training |
| `--tie` | off | Tie the LM head to the token embedding. The parameter count drops by `vocab × embed`, and the file shrinks by the same amount, because a tied model has no `output.weight` tensor. |
| `--no-qat` | off | Plain f32 training. **QAT is int8 only** (Q8_0). There is no int4 QAT. |
| `--quant f32\|int8\|int4` | `int8` | Precision of the saved file. `int8` is GGUF Q8_0, `int4` is Q4_0 (converted after training, so lossy). |

**Run control**

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint PATH` | none | Write training state (weights + Adam moments + RNG) to a GGUF checkpoint |
| `--checkpoint-every N` | 0 | Checkpoint every `N` optimizer steps (always at the end) |
| `--resume PATH` | none | Resume from that checkpoint and keep writing to it. A resumed run reports the right epoch and step, and with `--val` it restores the best validation score. |
| `--threads N` | 0 (auto) | Data-parallel training workers; `1` = serial. See [§6.8](#68-multi-threading-and-determinism). |
| `--seed N` | 1337 | RNG seed |
| `--force` | off | Retrain even if the model file exists |
| `--sample` | off | Print a short sample after training |
| `--verbose`, `-v` | off | Prints a progress line every 10 optimizer steps: `[train] step S/T  tokens/s R  elapsed Xs`. It does not print per-op logs. |
| `--debug` | off | Streams the engine's internal trace. Very verbose; use it to debug, not to watch progress. |

The sampling flags `run` shares with `generate` are in [§7](#7-generating-text).
`prepare` is in [§5.1](#51-train_transformer-prepare), `eval` in
[§8](#8-evaluating-a-model), and the export commands in
[§9.6](#96-one-command-exports).

### 6.3 Choosing a tokenizer

- **`--vocab 0`** (character-level): one token per distinct character. This is
  the simplest and most transparent option, and works well for tiny corpora.
  Character-level generation needs a seed at least `context_len` characters
  long.
- **`--vocab ≥ 256`** (byte-level BPE): subword tokens pack more text into the
  same context window, handle any language, and usually win on real text. Start
  with 512–4096. Larger vocabularies need more data to learn. Values from 1 to
  255 are rejected because the 256-byte base is irreducible.
- **Train once, reuse.** `--tokenizer tok.gguf` writes the learned merge list on
  the first run and loads it on every later run, so tokenizer training is
  skipped and the vocabulary stays stable across experiments and resumes. The
  trainer counts each distinct word once and updates pair counts
  incrementally, so it is fast: 22 MB trains at vocab 4096 in about 0.3 s.
- Merge learning is deterministic. Ties break by pair order, so the same corpus
  and vocab size always give the same tokenizer.

### 6.4 Epochs, stride, and budgets in practice

| You want | Use |
|---|---|
| A quick experiment on a small file | `--epochs 20` (stride = context) |
| A real run sized to compute | `--tokens <20 × params>`, e.g. `--tokens 3e7` for 1.5M params |
| To reproduce a pre-2026-10 run exactly | `--stride 1` (legacy overlapping windows) |

With the default stride, `--epochs` means what it says: passes over the
corpus. Example CLI commands from older docs that used `--epochs 200` with
stride 1 trained on each token thousands of times. Do not copy them.

### 6.5 Optimization and regularization

- **Learning rate.** Small models tolerate `3e-3`–`1e-2`. Above about 5M
  parameters, start near `1e-3`–`3e-3` and use `--cosine --warmup 100–500`.
- **`--clip 1`** prevents rare exploding steps; it is cheap insurance on
  longer runs.
- **`--weight_decay 0.01–0.1` and `--dropout 0.1`** help when held-out
  perplexity lags training perplexity, which is the signature of
  memorization.
- **`--tie`** shares the embedding and output matrices. It saves parameters
  and usually helps small models.
- **`--no-qat`** trains in plain fp32. QAT costs a little speed. Keep it on if
  you will ship int8.

### 6.6 Validation, checkpoints, and resuming

**Validation.** `--val 0.05` holds out the last 5% of the corpus (the
tokenizer is fit on the training part only). After each epoch the model is
scored on it, and the weights with the **lowest validation loss are the ones
saved**, which are not necessarily the last epoch's. `--patience N` stops
early.

**Checkpoints.** `--checkpoint run.fckp --checkpoint-every 500` writes the
full training state every 500 steps and at the end: weights, Adam moments,
step count, and RNG. Writes are atomic (temporary file, then rename), so a
crash mid-write never corrupts the checkpoint.

**Resume.** Re-run the *same* command with `--resume run.fckp`. Training
continues from the saved step toward the same budget or epoch count, and the
LR schedule picks up where it stopped. The checkpoint fixes the architecture.
If its vocabulary or context does not match the run (for example, a different
tokenizer), training refuses rather than silently mixing them. Use
`--tokenizer` so the vocabulary is guaranteed identical. One detail: a resumed
run restarts its partially-finished epoch with a fresh shuffle instead of
replaying the exact remaining batches, so a resumed run ends close to, but not
bit-identical with, an uninterrupted one ([Samples §4](Samples.md#4-training-controls)).
A mid-run interrupt does not save on exit; the checkpoint on disk is the last
periodic one.

```bash
train_transformer train big.txt model.gguf --context 128 --embed 128 --heads 4 --blocks 4 \
    --hidden 512 --vocab 2048 --lr 0.003 --tokens 5e7 --cosine --warmup 200 --clip 1 --tie \
    --tokenizer tok.gguf --checkpoint run.fckp --checkpoint-every 500
# …power cut at step 9,000…
train_transformer train big.txt model.gguf <same flags> --resume run.fckp
```

### 6.7 Sizing a run for your machine

Measured on 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4 cores / 8 threads, 15 GiB) on 2026-10-26, with int8 QAT, batch 16,
8 threads, and the first 8 MB of the prepared TinyStories training split. Each row
is one short `train` run (16 to 200 steps); tokens/s is the tokens trained divided
by the "Trained in" time, and the peak is the process's resident set from
`/usr/bin/time -v`, which includes training the BPE tokenizer. The binary was the
low-memory build (no LTO), so a shipped build is a little faster, and the desktop
session was running.

| Params | Config (T = context, C = width, L = layers, H = heads, F = FFN hidden, V = vocabulary) | Sec/step | Tokens/s | Peak RAM | Tokens/day |
|---|---|---|---|---|---|
| 0.17M | T64, C64, L2, H4, F256, V512 | 0.06 | 16,113 | 176 MB | 1.4B |
| 1.33M | T128, C128, L4, H4, F512, V2048 | 0.64 | 3,202 | 297 MB | 277M |
| 6.90M | T256, C256, L6, H8, F1024, V4096 | 6.28 | 652 | 1.4 GB | 56M |
| 20.59M | T256, C384, L8, H8, F1536, V8192 | 15.27 | 268 | 3.0 GB | 23M |

The 33.8M configuration (T256, C512, L8) was not re-run. The memory check in
`train` estimated 6.2 GB for it and refused to start, because only 6.1 GB (three
quarters of the available RAM) was allowed at that moment.

**RAM is dominated by activations**, not by the 16 bytes per parameter of
weights and optimizer state. RAM is rarely the limit; compute is.

| Goal | Reachable size | Basis |
|---|---|---|
| Heavily over-trained, minutes to hours | 0.1M to 2M | The 1.33M model sees 277M tokens a day, about 207 tokens per parameter |
| Compute-optimal (20 tokens/param) in 24 h | about 4M | Interpolated between the 1.33M and 6.9M runs |
| Compute-optimal in a week | about 12M | Interpolated between the 6.9M and 20.6M runs |
| Compute-optimal in a month | about 27M | Extrapolated; the largest run measured is 20.6M |

The 1.33M model reaches 20 tokens per parameter in about 2.3 h, the 6.9M model
in about 2.5 days, and the 20.6M model in about 17.8 days. Tokens per day fall
roughly in proportion to the parameter count while the tokens needed grow in
proportion to it, so the time to a compute-optimal model grows a little less
than with the square of the size.

**Starting points by size**

| Target | `--context` | `--embed` | `--heads` | `--blocks` | `--hidden` | `--vocab` | `--lr` |
|---|---|---|---|---|---|---|---|
| ~0.2M (minutes) | 64 | 64 | 4 | 2 | 256 | 512 | 1e-2 |
| ~1.5M (hours) | 128 | 128 | 4 | 4 | 512 | 2048 | 3e-3 |
| ~7M (a day or two) | 256 | 256 | 8 | 6 | 1024 | 4096 | 1e-3 |

All rows used `--lr 3e-3` and `--seed 1`; the learning rate does not change the speed.
Throughput is
measured; the presets are reasonable defaults, not tuned optima. A GPU is never
used, so a small discrete GPU (for example, a 2 GB MX330) adds nothing.

### 6.8 Multi-threading and determinism

There are two independent knobs:

- **`--threads N`** splits each minibatch across worker threads (data-parallel
  training). Gradients are reduced in a fixed order, so a given thread count is
  reproducible, and `--threads 1` is bit-identical to the serial trainer.
  Changing the shard count can change the low bits of the weights, because
  floating-point addition is not associative.
- **`FERRUM_NUM_THREADS=N`** sizes the matmul worker pool used by training,
  generation, and evaluation. The pool splits work by rows or columns without
  changing any per-element arithmetic, so it never changes a result. It does
  set the default shard count of the data-parallel trainer, though: with
  `--threads 0` (auto), the number of shards is `FERRUM_NUM_THREADS` or the
  core count. A different shard count trains different weights (the
  reference model reaches a different perplexity at 1 and at 8 shards), so
  pass `--threads N` explicitly to reproduce a run on another machine. The
  DVC pipeline does this for you: `train.threads` in `params.yaml` is passed
  to `train`. Only
  training is affected; `generate` and `eval` give identical output at any
  `FERRUM_NUM_THREADS`.

### 6.9 Library equivalents

Every CLI flag maps onto a field of `TransformerConfig`:

```rust
pub struct TransformerConfig {
    pub context_len: usize,    // 16
    pub embed_dim: usize,      // 32
    pub num_heads: usize,      // 4
    pub num_blocks: usize,     // 2
    pub hidden_dim: usize,     // 64
    pub epochs: usize,         // 100
    pub lr: f32,               // 0.01
    pub batch_size: usize,     // 16
    pub vocab_size: usize,     // 512   (0 = char-level, >= 256 = BPE)
    pub weight_decay: f32,     // 0.0
    pub dropout: f32,          // 0.0
    pub window_stride: usize,  // 0     (= context_len)
    pub max_tokens: u64,       // 0     (train for `epochs`)
    pub qat: bool,             // true
    pub grad_clip: f32,        // 0.0   (off)
    pub cosine_lr: bool,       // false
    pub warmup_steps: u64,     // 0
    pub weight_tying: bool,    // false
    pub tokenizer_path: String,   // ""  (train in memory)
    pub checkpoint_path: String,  // ""  (no checkpoints)
    pub checkpoint_every: u64,    // 0   (only at the end)
    pub resume: bool,             // false
    pub log_every_steps: u64,     // 0     (progress line every N steps; CLI -v = 10)
}
```

```rust
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig, ValidationConfig};

let cfg = TransformerConfig { max_tokens: 2_000_000, cosine_lr: true, ..TransformerConfig::default() };
let mut rng = Rng::new(7);

// Plain (threads: 0 = auto, 1 = serial):
let slm = GenerativeSLM::train_transformer_config_threaded(&corpus, &cfg, 0, &mut rng, |ep, loss| {})?;

// With validation + early stopping (returns the best-by-validation model):
let val = ValidationConfig { val_fraction: 0.05, patience: 3 };
let slm = GenerativeSLM::train_transformer_config_validated(&corpus, &cfg, 0, &val, &mut rng, |p| {
    println!("epoch {} train {:.3} val ppl {:.2}{}", p.epoch, p.train_loss, p.val.perplexity,
             if p.is_best { " *" } else { "" });
})?;

// Train once, then load from disk on later runs:
let (slm, was_loaded) = GenerativeSLM::load_or_train("model.gguf", &corpus, &cfg, &mut rng, |_, _| {})?;
```

The older positional functions `GenerativeSLM::train_transformer(corpus,
context, embed, heads, blocks, hidden, epochs, lr, batch, vocab, rng)` and the
`*_with_callback` / `*_threaded_with_callback` variants still work. They keep
**stride 1** so existing code reproduces exactly.

For full control, drive `TransformerNet` directly ([§12.3](#123-training-primitives)).

### 6.10 A worked walkthrough

This uses a few sentences as the corpus, which is enough to see the pipeline
end to end. Real models want megabytes ([Samples §1](Samples.md#1-quick-start-a-500k-slm-on-tinystories)
trains a real one). Every output below is from a real run
([Samples](Samples.md) lists the machine).

```bash
cargo build -p slm_cli --release
export PATH="$PWD/target/release:$PATH"

cat > corpus.txt <<'EOF'
the quick brown fox jumps over the lazy dog. the town wakes slowly each morning, and the baker opens his shop while the streets are still quiet. children walk to school past the old stone bridge, and the river runs down to the sea carrying leaves and small boats. by noon the market is full of voices, and the fox watches from the hill. in the evening the lamps are lit one by one, the shops close, and the streets grow quiet again. the quick brown fox goes home to sleep, and the town sleeps too.
EOF
cat > heldout.txt <<'EOF'
the lazy dog wakes slowly each morning, and the baker walks to the old stone bridge while the streets are quiet. children carry small boats to the river, and by noon the market is full of leaves and voices. in the evening the lamps are lit and the town sleeps.
EOF

train_transformer train corpus.txt model.gguf --epochs 60 --context 12 --embed 32 --heads 4 \
    --blocks 2 --hidden 64 --vocab 300 --seed 7 --stride 1
```

```text
╔══════════════════════════════════════════════════════════╗
  ferrum transformer SLM trainer (int8 QAT)
  Corpus  : corpus.txt  (498 chars)
  Context : 12   Embed: 32   Hidden: 64
  Heads   : 4   Blocks: 2
  Tokenizer: byte-level BPE (vocab 300)
  Epochs  : 60   LR: 0.01   Batch: 16   Seed: 7
  Threads : 8 (data-parallel minibatch training)
╚══════════════════════════════════════════════════════════╝

  epoch     1/60   loss = 4.383340
  epoch     3/60   loss = 2.058480
  epoch     6/60   loss = 0.695912
  epoch     9/60   loss = 0.403406
  epoch    12/60   loss = 0.309260
  epoch    15/60   loss = 0.273127
  epoch    18/60   loss = 0.276861
  epoch    21/60   loss = 0.259686
  epoch    24/60   loss = 0.231830
  epoch    27/60   loss = 0.226736
  epoch    30/60   loss = 0.223918
  epoch    33/60   loss = 0.223233
  epoch    36/60   loss = 0.240042
  epoch    39/60   loss = 0.239465
  epoch    42/60   loss = 0.218993
  epoch    45/60   loss = 0.216633
  epoch    48/60   loss = 0.202404
  epoch    51/60   loss = 0.195243
  epoch    54/60   loss = 0.199890
  epoch    57/60   loss = 0.187257
  epoch    60/60   loss = 0.188075

Trained in 5.80s.
  vocab = 300 BPE tokens   context = 12   output_dim = 300
Saved 50264 bytes → model.gguf (int8 GGUF)
Reload check: OK (6 layers).
exit=0
```

(`--stride 1` is used here only because the corpus is a few hundred
characters. On real data, leave the default.)

```bash
train_transformer info model.gguf
```

```text
Model     : model.gguf  (50264 bytes)
Format    : GGUF v3 (gpt2, int8)
Params    : 36736
Name      : GenerativeSLM Transformer
Task      : TransformerSLM
Input dim : 12
Output dim: 300
Tokenizer : byte-level BPE (300 tokens, 43 merges)
Layers    : 6
```

```bash
train_transformer generate model.gguf "the quick brown" --chars 60 --temp 0.2 --gen-seed 1
train_transformer eval model.gguf heldout.txt
train_transformer eval model.gguf corpus.txt
```

```text
the quick brown fox jumps over the lazy dog. the town wakes slowly each mor

Model        : model.gguf
Held-out text: heldout.txt  (261 chars)
Predictions  : 151
Cross-entropy: 2.3796 nats/token
Bits/token   : 3.4331
Perplexity   : 10.8007
Bits/byte    : 1.9862
Seconds      : 0.00

Model        : model.gguf
Held-out text: corpus.txt  (498 chars)
Predictions  : 274
Cross-entropy: 0.0416 nats/token
Bits/token   : 0.0600
Perplexity   : 1.0425
Bits/byte    : 0.0330
Seconds      : 0.02
```

A model that learned nothing scores a perplexity equal to its vocabulary size
(300 here), so a held-out 10.8 means it learned real structure. The gap between
1.04 on seen text and 10.8 on unseen text is the textbook sign of memorization
on a tiny corpus. More data, regularization, and a right-sized model close it.
Iterate in this order: **more and better data → right-size the model →
regularize → tune `--vocab` → tune `--lr` and the budget**. Re-run `eval` after
each change.

---

## 7. Generating text

```text
train_transformer generate <model.gguf> <seed text> [--chars N] [--max-tokens N] [--temp F] [--gen-seed N]
                           [--top-k N] [--top-p F] [--repetition-penalty F] [--stop S]... [--stream]
```

`run` takes the same flags after its training flags. [Samples §6](Samples.md#6-generation)
runs each of them.

| Flag | Default | Meaning |
|---|---|---|
| `--chars N` | 200 | Characters to generate (characters even for BPE models) |
| `--max-tokens N` | off | Token budget instead of a character count. With `--max-tokens` or `--stop` set, `--chars` is ignored, and a `--stop` run without `--max-tokens` uses the `--chars` value as its token budget. |
| `--temp F` | 0.8 | Sampling temperature. **`0` is exact argmax (greedy).** 0.1–0.3 for faithful, 0.7+ for varied. |
| `--gen-seed N` | time-based | Fix it for reproducible output |
| `--top-k N` | 0 (off) | Sample only from the `N` most likely tokens |
| `--top-p F` | 1.0 (off) | Nucleus sampling, in `[0, 1]` |
| `--repetition-penalty F` | 1.0 (off) | Penalize tokens already generated; must be positive |
| `--stop S` | none | Stop when `S` appears. Repeatable. The output ends before the first match. |
| `--stream` | off | Print the completion live (with `--max-tokens` or `--stop` the completion is printed whole at the end) |

- The output is the **seed followed by** the continuation.
- Temperature `0` is exact argmax and picks the first maximum on an exact tie
  (the llama.cpp rule). The old behaviour clamped `0` to `0.01`; that clamp is
  gone.
- If the tokenizer has an end-of-text token (every newly trained model does),
  generation stops when the model emits it. The CLI does not print the stop
  reason.
- BPE models encode the seed and sample subword tokens. Character-level models
  need every seed character to be in the vocabulary.
- Streaming is UTF-8-safe: a partial multi-byte character is held back until
  it completes, so a `U+FFFD` placeholder is never shown.
- With the same `--gen-seed`, the output is identical at any `FERRUM_NUM_THREADS`.

### 7.1 Generation cost

A per-block **KV cache** means each new token attends to the cached keys and
values instead of recomputing the window. Learned absolute positions forbid
sliding the cache one token at a time. When the cache is full, Ferrum
**re-primes** from the last `context_len / 2` tokens. That is one re-prime per
`context_len / 2` generated tokens. The cost per generated token is therefore
bounded, and generating many times the window length takes linear time. Output
text is decoded incrementally, not re-decoded in full on every step.

The library adds `generate_tokens`, which takes a token budget and a list of
stop strings and returns the continuation only:

```rust
use ferrum_core::{Rng, SamplingParams};

let p = SamplingParams { temperature: 0.7, top_k: 40, top_p: 0.9, repetition_penalty: 1.1 };
let g = slm.generate_tokens("Once upon a time", 100, &["\n\n".to_string()], &p, &mut Rng::new(1))?;
println!("{} ({} tokens, {:?})", g.text, g.n_tokens, g.stop_reason);
```

`stop_reason` is `MaxTokens`, `Stop`, or `Eos`. The older `generate`,
`generate_with`, `generate_until`, and `generate_stream` are unchanged:

```rust
let p = SamplingParams { temperature: 0.7, top_k: 40, top_p: 0.9, repetition_penalty: 1.1 };
let text  = slm.generate_with("Once upon a time", 200, &p, &mut Rng::new(1))?;
let reply = slm.generate_continuation("Q: hi\nA:", 80, 0.5, &mut Rng::new(1))?; // without the seed
let line  = slm.generate_until("Q: hi\nA:", 200, "\n", &p, &mut Rng::new(1))?;  // stop at newline
let full  = slm.generate_stream("the", 100, 0.7, &mut Rng::new(1), |frag| print!("{frag}"))?;
```

`format!("{seed}{continuation}") == generate(seed, …)`, and the concatenated
stream equals the continuation.

---

## 8. Evaluating a model

### 8.1 Held-out perplexity: the honesty check

Training loss shows the model is *fitting*. Perplexity on unseen text shows
whether it *generalizes*.

```bash
train_transformer eval model.gguf heldout.txt --json metrics/heldout.json
```

```text
train_transformer eval <model.gguf> <text.txt> [--json <path>]
```

`--json` writes `{"perplexity", "cross_entropy_nats", "bits_per_token", "bits_per_byte", "tokens", "bytes", "seconds"}`.
DVC tracks this file as a metric.

```rust
let e = slm.evaluate(&std::fs::read_to_string("heldout.txt")?)?;
println!("ppl {:.3}  bits/token {:.3}  CE {:.3} nats  ({} predictions)",
         e.perplexity, e.bits_per_token, e.cross_entropy, e.num_predictions);
```

- **Every token is scored.** The first token is conditioned on a start
  position (BOS, which is the EOS id), so `num_predictions` equals the token
  count. Earlier versions skipped the first window, so perplexities from before
  this change are not comparable ([Status §7](Status.md#7-backward-compatibility)).
- Texts longer than the context window use sliding windows with stride
  `context_len / 2`. Each token is scored exactly once, with at least
  `context_len / 2` tokens of history (except in the first window).
- Lower is better, and **1.0 is perfect**. A model that learned nothing scores
  about the vocabulary size.
- Each token's probability is floored at 1e-12, which is a per-token floor of
  −27.63 nats.
- A large train-versus-held-out gap means **memorization**, which points to a
  data problem rather than a model that is too small.
- Evaluation uses the shipped weights and the same forward path as generation.
- For a character-level model, an unknown character is an error in the scoring
  API, and `evaluate()` reports how many it met.
- To compare models with **different tokenizers**, compare **bits per byte**
  (`bits_per_byte` in the JSON). Per-token numbers are not comparable across
  vocabularies.

**Speed.** Evaluation runs one padded forward pass per window instead of one
pass per token. On the 507,552-parameter reference model it scores a 30 KB
held-out file in about 1.5 s; the 1.1 MB test split of the default corpus takes
about a minute ([Samples §5](Samples.md#5-evaluation) and
[§9](Samples.md#9-benchmarking)).

### 8.2 Other checks

| Axis | How |
|---|---|
| **Seed fidelity** | `generate(seed, …)` always starts with `seed`. If not, the file is corrupt. |
| **Tokenizer compression** | `text.len() / tok.encode(text).len()` bytes per token; higher packs more text per step |
| **Size** | `to_bytes()` (f32) vs `to_bytes_quantized()` (int8, ≈4×) vs `to_bytes_quantized_int4()` (≈8×). Matrices whose row length is not a multiple of 32 stay f32, so ratios are slightly below 4×/8×. |
| **Quantization fidelity** | An int8 file reloads bit-identically to the QAT model that trained it, so `evaluate()` gives the same numbers before and after saving. int4's grid is coarser ([Samples §5](Samples.md#5-evaluation)). |
| **Speed** | `time train_transformer generate …`; compare `FERRUM_NUM_THREADS=1` vs auto. Outputs are identical, only wall-clock changes. |
| **Reproducibility** | Same seed, same output; reloaded model matches in-memory model; same corpus and vocab give the same BPE merges. |
| **Standard benchmarks** | LightEval through `ferrum-serve` ([§9.8](#98-evaluating-with-lighteval)) |

### 8.3 The scoring API

```rust
let s = slm.score("The cat sat on the", " mat")?;   // log p(continuation | context)
let r = slm.score_rolling(&text)?;                   // every token, the first conditioned on BOS
// Score { logprob: f64, n_tokens: usize, greedy: bool }
```

`score` follows the lm-eval-harness convention. Trailing whitespace in the
context moves to the front of the continuation. The continuation tokens are the
suffix of the joint encoding after the context's length. An empty continuation
scores `0.0` over `0` tokens. `greedy` is true when every scored token was the
argmax. The server and LightEval use this API.

---

## 9. GGUF: Ferrum models, other runtimes, serving, and external models

GGUF is the one file format Ferrum uses. Everything it writes is GGUF v3: the
models you train, training checkpoints, and tokenizer files. §9.1 to §9.4 cover
the **imported** Llama/Qwen checkpoints: Ferrum runs them, fine-tunes them, and
writes them back out. §9.5 to §9.9 cover your own models: running them in other
tools, serving them, evaluating them, and the DVC pipeline. The file layout is in
[§12.7](#127-the-ferrum-gguf-format).

**Formats read:** `F32, F16, Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1` and the k-quants
`Q2_K, Q3_K, Q4_K, Q5_K, Q6_K`. The `IQ*` families are rejected with a clear
message. **Architectures:** `llama`, `qwen2`.

### 9.1 Running (`run-gguf`)

```text
train_transformer run-gguf <model.gguf> [prompt] [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--quant int4\|int8\|f32` | int4 | In-memory precision: int4 = least RAM, int8 = fastest decode, f32 = no second quantization |
| `--max N` | — | Maximum new tokens |
| `--temp F` | 0.8 | Temperature |
| `--top-k N` / `--top-p F` / `--rep F` | 0 / 1.0 / 1.0 | Sampling knobs (defaults = off) |
| `--gen-seed N` | — | Generation seed |
| `--raw` | — | Disable automatic chat-template formatting |
| `--ids "1 2 3"` | — | Raw prompt token IDs (only if the file has no tokenizer) |
| `--resume ckpt.flck` | — | Overlay fine-tuned weights (forces f32) |
| `--force` | — | Load even if the memory estimate exceeds available RAM |

- **Chat templates.** For instruct models, Ferrum detects the template from
  `tokenizer.chat_template` (ChatML, Llama-3, Llama-2 `[INST]`, Zephyr) or
  falls back by architecture. It wraps your prompt and stops at the
  template's end-of-turn token. It handles single turns only. Use `--raw` for
  base models.
- **Memory guard.** Before loading, Ferrum estimates the resident footprint
  and refuses with a warning if it will not fit (override with `--force`).
- **Expectations.** Decode is memory-bandwidth-bound: each token streams every
  weight once. A ~1B model decodes at about **6.7 tokens/s at int4** after about
  **25 s of prefill** for a 512-token prompt (estimates from the benchmark in §13). That is a patient demo, not a
  chatbot ([§13.4](#134-quantized-decode-and-the-gguf-runner)).
- **Fidelity.** Import dequantizes and re-quantizes onto Ferrum's per-row
  grid. That is lossy and **not bit-exact to llama.cpp**. `--quant f32` avoids
  the second quantization.

### 9.2 Fine-tuning (`finetune-gguf`)

```text
train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--epochs N` | 3 | Passes over the corpus |
| `--lr F` | 1e-4 | AdamW learning rate |
| `--batch N` | 8 | Sequences per step |
| `--seq N` | 64 | Window length (capped at the model's context) |
| `--warmup N` | 0 | Warmup steps (cosine schedule) |
| `--clip F` | 1.0 | Gradient clipping |
| `--weight_decay F`, `--dropout F` | 0 | Regularization |
| `--qat` | off | int8 quantization-aware fine-tuning |
| `--threads N`, `--seed N` | auto, — | Parallelism and determinism |
| `--resume ckpt.flck` | — | Continue a previous fine-tune |
| `--sample` | — | Print a sample afterwards |

The model loads at f32, because training needs full-precision masters.
Full-parameter AdamW needs about **16 bytes per parameter** for the training
state, but two things add to that. Each worker thread holds a full gradient
copy, so the tool caps `--threads` to keep training under half of the free
RAM. The checkpoint save serializes weights and both moments twice, so the
peak is about **12 bytes per f32 weight byte**: SmolLM2-135M measured 6.4 GB
(3 threads) and 9.6 GB with 8 uncapped threads. Qwen2.5-0.5B (about 8 GB of
training state) and 1.5B and above do not fit. Speed is about 6 tokens/s on
135M with 3 threads, so use a small corpus. The output `.flck` checkpoint is
applied with `run-gguf --resume` or exported with `export-gguf --resume`.
Only parameters the source file has are trained: a bias the GGUF lacks (every
bias on a Llama model; all but q/k/v on Qwen2) stays absent, so the exported file
is exactly the fine-tuned model. A model with tied embeddings (no `output.weight`,
like SmolLM2) keeps its LM head tied to the embedding while training and stays
tied on export. A `.flck` from before these rules is refused; re-run the fine-tune.

### 9.3 Exporting (`export-gguf`)

```bash
train_transformer export-gguf in.gguf out.gguf --quant q8_0                       # re-quantize
train_transformer export-gguf base.gguf tuned.gguf --resume tuned.flck --quant q6_k  # ship a fine-tune
```

- Output types are `f32 | f16 | q8_0 | q8_1 | q4_0 | q4_1 | q4_k | q5_k | q6_k`
  (default `q8_0`). Q2_K and Q3_K are read-only.
- The source's hyperparameters and tokenizer carry over verbatim, so the file
  runs unchanged in llama.cpp, ollama, and LM Studio.
- Norms and biases stay f32. A matrix whose rows are not block-aligned for
  the chosen quant (32 for legacy formats, 256 for k-quants) falls back to
  f16. The per-type summary printed after export shows what was emitted.
- Writes are atomic. Re-quantizing an already-quantized source is lossy. The
  lossless paths are `f16`/`f32`, or exporting fine-tuned f32 masters.

### 9.4 Library API

```rust
use ferrum_core::{Gguf, GgufQuant, GgufTokenizer, LlamaTrainer, QKind, Rng};

let gguf  = Gguf::open("model.gguf")?;           // streamed: tensors read on demand
let tok   = GgufTokenizer::from_gguf(&gguf)?;    // the checkpoint's own tokenizer
let model = gguf.load_llama(QKind::Int4)?;       // or Int8; load_llama_prec(None) = f32
let out   = model.generate(&tok.encode("Hello"), 32, 0.7, &mut Rng::new(1))?;
println!("{}", tok.decode(&out));

let f32m = Gguf::open("model.gguf")?.load_llama_prec(None)?;
let mut trainer = LlamaTrainer::new(f32m)?;      // AdamW, lr 1e-4; errors if any weight is quantized
trainer.set_lr(3e-4);
trainer.set_grad_clip(Some(1.0));
let loss = trainer.train_step(&tok.encode("training text"))?;   // one AdamW step, returns the loss

trainer.model.write_gguf(&gguf, GgufQuant::Q4K, "out.gguf")?;  // export the fine-tuned model
```

### 9.5 Running Ferrum models in llama.cpp, Ollama, and LM Studio

A Ferrum BPE transformer SLM is a `gpt2`-architecture GGUF. Other runtimes load
it without changes. The same f32 file gives the same greedy continuation in
Ferrum and in llama.cpp; the test `llama_cpp_greedy_matches_ferrum_f32` checks
this. Versions the commands were verified with:

| Tool | Version | Verified how |
|---|---|---|
| llama.cpp | b11521 | `llama-completion`, `llama-tokenize` |
| Ollama | v0.40.2 (bundled llama-server) | `ollama create`, `/api/generate` |
| LM Studio | 0.4.18 (llama.cpp-linux-x86_64-avx2 2.23.1) | `lms import`, `lms load`, `/v1/completions` |

A stock GPT-2 124M Q8_0 GGUF gave identical 16-token greedy output in all three
tools, which fixed the method. [Samples §7](Samples.md#7-serving-and-running-your-model-in-llamacpp-ollama-and-lm-studio)
runs every command below on the reference model.

**llama.cpp**

```bash
llama-completion -m models/slm.gguf -p "Once upon a time" -n 16 --temp 0 -no-cnv --no-warmup </dev/null 2>/dev/null
llama-tokenize   -m models/slm.gguf -p "Once upon a time" --ids --log-disable
```

**Ollama**

```bash
OLLAMA_HOST=127.0.0.1:11545 OLLAMA_MODELS=./ollama-models ollama serve &
printf 'FROM "%s"\nTEMPLATE "{{ .Prompt }}"\nPARAMETER num_predict 16\n' "$PWD/models/slm.gguf" > Modelfile
OLLAMA_HOST=127.0.0.1:11545 OLLAMA_MODELS=./ollama-models ollama create ferrum-tiny -f Modelfile
curl -s http://127.0.0.1:11545/api/generate \
  -d '{"model":"ferrum-tiny","prompt":"Once upon a time","raw":true,"stream":false,"options":{"temperature":0,"num_predict":16}}'
```

Use `raw: true`. A Ferrum model has no chat template.

**LM Studio**

```bash
lms import -y -l --user-repo ferrum-test/slm models/slm.gguf
lms ls --json                       # find the entry whose path matches; its modelKey is KEY
lms load KEY --gpu off --identifier ferrum-tiny -y
lms server start -p 1234
curl -s http://127.0.0.1:1234/v1/completions \
  -d '{"model":"ferrum-tiny","prompt":"Once upon a time","max_tokens":17,"temperature":0,"repeat_penalty":1.0}'
```

Two LM Studio details. `max_tokens` N returns N−1 visible tokens, so ask for 17
to see 16. And its default repeat penalty (1.1) changes the output, so set
`repeat_penalty` to 1.0 when you compare with Ferrum.

**What to expect**

- **f32:** greedy output equals Ferrum's.
- **Q8_0:** llama.cpp quantizes activations in its Q8_0 matmuls, which can flip
  near-ties late in a generation. The compatibility test compares Ferrum's
  output with llama.cpp's and asserts equality only over the first 8 tokens.
- **Q4_0:** the test only checks that llama.cpp loads the file and generates.
- **Prompts** are tokenized the same way as in Ferrum, because the file declares
  `tokenizer.ggml.pre = "gpt-2"` and Ferrum uses the same pre-tokenizer.

**Finding the tools.** The export commands (§9.6) look for `ollama` and `lms` in
this order: the `--bin` flag, the environment variable (`FERRUM_OLLAMA`,
`FERRUM_LMS`), `PATH`, then the default install location
(`~/.local/bin/ollama`, `~/.lmstudio/bin/lms`). If none has it, the error lists
all four places. The compatibility tests and `scripts/verify_runtimes.sh` also
find llama.cpp this way, through `FERRUM_LLAMA_BIN` (the directory holding
`llama-completion`), then `PATH`, then `~/.local/opt/llama.cpp-b11521`. They skip
a missing tool, but fail instead when `FERRUM_REQUIRE_RUNTIMES=1` is set.
`FERRUM_VERIFY_RUNTIMES=1` also runs the Ollama and LM Studio checks in
`test_runtime_compat` (it starts those tools).

**What does not run there.** Character-level models (no BPE vocabulary) and
Ferrum-ML tabular MLPs are valid GGUF files, but llama.cpp has no architecture
for them. Use `train_transformer` or the Ferrum library for those.

### 9.6 One-command exports

```text
train_transformer export-ollama   <model.gguf> <name> [--bin P] [--modelfile-only out.Modelfile]
train_transformer export-lmstudio <model.gguf> [--repo ferrum/<stem>] [--bin P]
```

| Command | What it does |
|---|---|
| `export-ollama` | Writes a Modelfile and runs `ollama create <name> -f Modelfile`. `--modelfile-only out.Modelfile` writes the Modelfile and stops. `--bin` names the `ollama` binary. |
| `export-lmstudio` | Runs `lms import -y -l` on the file, then looks the import up with `lms ls --json` and prints its `modelKey`. `-l` links the file instead of copying it. `--repo` sets the `publisher/name` (default `ferrum/<file stem>`). `--bin` names `lms`. |

The Modelfile contains:

```text
FROM "<absolute path to the GGUF>"
TEMPLATE "{{ .Prompt }}"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.8
PARAMETER num_ctx <context_length of the model>
```

Rules for both commands:

- A GGUF that llama.cpp cannot run (generic layout, character-level, tabular) is
  **refused before any tool runs**. The error says why and how to fix it.
- A missing tool gives an error that lists the four places searched (§9.5).
- If the Ollama server is not running, `ollama create` fails and the error says
  to start it with `ollama serve`. `ollama create` talks to `OLLAMA_HOST`.
- After export you can use `ollama run <name>` or load the model in LM Studio.
- `train` prints a one-line warning when it writes a plain `ferrum`-architecture
  GGUF (`--vocab 0`) that these tools cannot load.

### 9.7 Serving: `ferrum-serve`

```text
ferrum-serve --model <path.gguf> [--host 127.0.0.1] [--port 8080]
             [--max-batch 64] [--max-chars 65536] [--max-body 8388608] [--max-tokens 4096]
```

The server binds to loopback by default and prints the model and address when
it is ready. Unknown flags are errors. It holds one loaded model on **one
dedicated model thread** and handles **one request at a time**, because
`GenerativeSLM` is not `Send` or `Sync`. Each matmul still uses every core.

| Route | Request | Response |
|---|---|---|
| `GET /health` | none | `"ok"` |
| `GET /v1/info` | none | `{params, context_len, vocab, quant, tokenizer, model_sha256}` |
| `POST /v1/loglikelihood` | `{"requests":[{"context","continuation"}]}` | `{"results":[{"logprob","n_tokens","greedy"}]}` |
| `POST /v1/loglikelihood_rolling` | `{"texts":[...]}` | `{"results":[{"logprob","n_tokens"}]}` |
| `POST /v1/generate` | `{"requests":[{"prompt","max_tokens","stop","temperature","top_k","top_p","seed"}]}` | `{"results":[{"text","n_tokens"}]}` |

`generate` defaults: `temperature` 0 (greedy), `max_tokens` 64, `top_k` 0, `top_p` 1,
`seed` 0. `temperature` must be finite and at least 0, `top_p` in (0, 1].
POST requests must send `Content-Type: application/json`; without it the server
answers `400`, and `curl -d` does not set it for you.

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8080/v1/loglikelihood \
  -d '{"requests":[{"context":"The cat sat on the","continuation":" mat"}]}'
```

**Limits.** They are checked before any compute.

| Limit | Default | Error |
|---|---|---|
| Body size (`--max-body`) | 8 MiB | `413` |
| Batch length (`--max-batch`) | 64 | `400` |
| Characters per text (`--max-chars`) | 65,536 | `400` |
| `max_tokens` per generate request (`--max-tokens`) | 4,096 | `400` |
| Stop strings per generate request | 16 of at most 1,024 bytes | `400` |
| POST requests in flight | 32 | `503`, before the body is read |
| Malformed JSON or a missing field | | `400` |
| Model error (for example an unknown character in a character-level model) | | `422` |

Errors have the form `{"error": "..."}`. An empty `requests` array returns `200`
with `{"results": []}`. A model error, or a panic in the model thread, does not
stop the server. Ctrl-C shuts it down.

There is no authentication and no TLS. Do not bind it to a public address. The
routes are Ferrum's own; there are no OpenAI-compatible routes.

### 9.8 Evaluating with LightEval

`eval/lighteval/` lets Hugging Face LightEval **0.13.0** (Python 3.10 or newer)
evaluate a Ferrum model through `ferrum-serve`.

```bash
python3 -m venv eval/lighteval/.venv
eval/lighteval/.venv/bin/pip install -r eval/lighteval/requirements.txt
eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt \
    --out metrics/lighteval.json --max-samples 200
```

`run.sh` builds and starts `ferrum-serve`, waits for it to report its address
and answer `/health`, runs
`lighteval custom ferrum ferrum_model.py "<tasks>" --custom-tasks ferrum_tasks.py`,
stops the server, and writes a summary:
`{"tasks": {"<task>|0": {"<metric>": value}}, "lighteval_version": "0.13.0", "model": ..., "max_samples": ...}`.

| Option | Meaning |
|---|---|
| `--model`, `--test`, `--out` | Required: model file, test split, summary path |
| `--max-samples N` | Cap samples per task |
| `--wikitext` | Add `wikitext:103:document_level` |
| `--port P` | Server port (default 8765) |

`FERRUM_SERVE_WAIT` (seconds, default 60) caps the wait for the server to load the
model and answer `/health`. `FERRUM_LIGHTEVAL_VENV` names a different virtual
environment. `run.sh` stops the server (and LightEval) on every exit path,
including Ctrl-C and `kill`. Each run uses its own LightEval sample cache under
`eval/lighteval/out/cache`, because LightEval's default cache is keyed by the
model *config*, not the weights, and would answer a second model from the first
one's results. The `blimp` and `arc` tasks download their datasets from Hugging
Face on first use. `run.sh` caches them in `eval/lighteval/.hf-cache` (git-ignored,
kept between runs), never in `~/.cache/huggingface`; set `FERRUM_HF_CACHE` to
choose another directory.

**Tasks.**

| Task | What it measures | Metrics |
|---|---|---|
| `ferrum_heldout` | Rolling log-likelihood of the test documents | word and byte perplexity, bits per byte |
| `tinystories_cloze` | Multiple choice: the true next sentence versus 3 sentences from other stories (fixed seed) | `loglikelihood_acc` (reported as `acc`) |
| `blimp:anaphor_gender_agreement`, `blimp:anaphor_number_agreement`, `blimp:regular_plural_subject_verb_agreement_1` | Grammatical versus ungrammatical sentence pairs | accuracy |
| `arc:easy` | Science questions | accuracy |

A 500K-parameter model will score near chance on `arc:easy` (25% for four
choices). That is expected. [Samples §8](Samples.md#8-testing-with-lighteval)
shows a real run of the reference model.

`FerrumModel` reads the server address from `FERRUM_URL` (default
`http://127.0.0.1:8080`; `run.sh` sets it) and uses only the Python standard
library to talk to it.

**Two LightEval 0.13.0 workarounds.**

1. `FerrumModel` does not call `super().__init__()` (it raises in 0.13.0). It
   sets `self.config` and the sample cache directly.
2. `loglikelihood_rolling` is not cached, because 0.13.0's perplexity cache
   drops results. `requirements.txt` also pins `xxhash<4`, because xxhash 4.x
   crashes the details logger, and `datasets<5`.

LightEval's litellm and TGI backends cannot run log-likelihood tasks, which is
why Ferrum plugs in as a custom model class.

### 9.9 DVC pipeline

`dvc repro` reproduces data, model, and metrics from `params.yaml`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dataops.txt
.venv/bin/dvc repro
.venv/bin/dvc dag
.venv/bin/dvc metrics show
```

| Stage | Command (from `dvc.yaml`) | Outputs |
|---|---|---|
| `fetch` | `scripts/fetch.sh <url> <sha256> data/raw/tinystories.txt` | `data/raw/tinystories.txt` |
| `prepare` | `train_transformer prepare ... --dedup --nfkc` | `data/prepared/{train,val,test}.txt`, `manifest.json` |
| `train` | `train_transformer train data/prepared/train.txt models/slm.gguf ... --force` | `models/slm.gguf`, `models/tokenizer.gguf` |
| `eval` | `train_transformer eval ... --json` on `val.txt` and `test.txt` | `metrics/val.json`, `metrics/test.json` |
| `lighteval` (opt-in, in `eval/lighteval/dvc.yaml`) | `eval/lighteval/run.sh ...` | `metrics/lighteval.json` |

`params.yaml` holds the knobs:

| Section | Keys |
|---|---|
| `fetch` | `url`, `sha256` |
| `prepare` | `min_chars`, `max_chars`, `val_frac`, `test_frac`, `seed` |
| `model` | `context`, `embed`, `heads`, `blocks`, `hidden`, `vocab` (default: the 507,552-parameter reference) |
| `train` | `tokens`, `lr`, `warmup`, `clip`, `val`, `patience`, `quant`, `seed` |

Edit a value, run `dvc repro`, then `dvc params diff` and `dvc metrics diff` to
see what changed. Stages whose inputs did not change are skipped. The
`lighteval` stage is kept out of the root pipeline, so a plain `dvc repro` does
not need the LightEval environment; run it with
`dvc repro eval/lighteval/dvc.yaml:lighteval`.

Notes:

- `data/` and `models/` are git-ignored and tracked by DVC. `dvc.lock` is
  committed. `.dvc/` is initialised with **no remote**; a remote is out of scope.
- Training is deterministic for a fixed `--threads` (the pipeline leaves it at
  auto), so `dvc repro --force train` gives the same `models/slm.gguf` sha256 on
  the same machine. On a machine with a different core count the data-parallel
  shard count changes, and so do the trained weights
  ([§6.8](#68-multi-threading-and-determinism)).
- `train` always reads the whole prepared train split into memory.
- Metrics files use `cache: false`, so they live in git.

---

## 10. The GUI (Ferrum SLM Studio)

### 10.1 What it is

Ferrum SLM Studio is a cross-platform **Tauri 2** desktop (and mobile) app.
Its interface is plain HTML, CSS, and vanilla JavaScript with no framework and
no Node build step, and every button calls straight into `ferrum_core`. It puts
the whole pipeline in one window: data preparation, training, streaming
generation, evaluation, model inspection, GGUF run, fine-tune, and export, plus
a system monitor and a live engine log.

### 10.2 Building and launching

**Prerequisites**

1. Rust and Cargo, plus the Tauri CLI: `cargo install tauri-cli --version "^2"`.
2. System WebView libraries:
   - **Linux (Debian/Ubuntu):** `libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev build-essential curl wget file libssl-dev`
   - **macOS:** Xcode command-line tools.
   - **Windows:** WebView2 runtime (preinstalled on Windows 11) and MSVC build tools.

```bash
cd ferrum_gui
cargo tauri dev      # development window
cargo tauri build    # installers / binaries for this OS
```

For real release artifacts, replace the placeholder icons with `cargo tauri
icon path/to/logo.png`.

**Snap-polluted environments.** Launching from a terminal inside a snap app
(for example, the VS Code snap) can crash with `undefined symbol:
__libc_pthread_init`, because the snap's GTK paths leak in. Strip them:

```bash
env -u GTK_PATH -u GTK_EXE_PREFIX -u LOCPATH -u GIO_MODULE_DIR \
    -u GTK_IM_MODULE_FILE -u GSETTINGS_SCHEMA_DIR -u XDG_DATA_HOME \
    XDG_DATA_DIRS=/usr/local/share:/usr/share ./target/release/ferrum_gui
```

**Mobile:** `cargo tauri android init && cargo tauri android dev`, or `cargo
tauri ios init && cargo tauri ios dev` (macOS only).

### 10.3 The tabs

| Tab | What you do there |
|---|---|
| **Datasets** | Download from a URL, Hugging Face, or Kaggle (or pick a catalog corpus), clean with `clean_corpus`, preview statistics, save |
| **Train** | Train a transformer SLM: architecture, epochs or a **token budget**, stride, LR/cosine/warmup, clip, weight decay, dropout, tie, QAT, tokenizer file, checkpoint/resume, validation fraction and patience; live loss chart |
| **Generate** | Prompt, characters, temperature, optional streaming |
| **Evaluate** | Score a model on held-out text; each run adds a table row (perplexity, CE, bits/token, uniform baseline) |
| **Models** | Inspect or reload a `.gguf` model: format, task, dimensions, tokenizer, layers |
| **GGUF** | Inspect and run a Llama/Qwen checkpoint: precision, top-k/top-p/repetition penalty, chat format, optional fine-tune overlay |
| **Fine-tune** | AdamW fine-tune an imported GGUF on a corpus into a `.flck` checkpoint |
| **Export** | Re-quantize or export a (fine-tuned) GGUF with phase-by-phase progress and a per-type summary |
| **System** | Per-core CPU, memory, and Ferrum's thread count (also as top-bar gauges) |
| **Capable** | A 1–2 s micro-benchmark that estimates the parameter ranges this machine can load, train, fine-tune, and run. Treat the numbers as rough upper bounds ([Status §5](Status.md#5-known-issues)). |
| **Engine log** (docked) | The engine's `--verbose` trace, streamed live when *Verbose* is ticked |

Every form validates input and shows plain-language errors. If the Rust
backend is unavailable (for example, in a plain web preview), a banner says so.

### 10.4 A first run, click by click

1. **Datasets:** pick *TinyStories (validation)* or load a local file, tick the
   cleaning options, click **Clean & preview**, then **Save corpus**.
2. **Train:** browse to the corpus and an output path. For a first model use
   context 64, embed 64, heads 4, blocks 2, hidden 256, vocab 1024, LR 0.003,
   and a token budget of `2e6`. Tick **Cosine LR** and set a checkpoint file.
   Click **Train** and watch the loss chart.
3. **Generate:** browse to the model, type a prompt, set temperature 0.7, tick
   **stream**.
4. **Evaluate:** add held-out text and compare the perplexity row against the
   baseline column.
5. **Models:** inspect the file you just made.
6. **GGUF → Fine-tune → Export:** the separate "someone else's model" path
   ([§9](#9-gguf-ferrum-models-other-runtimes-serving-and-external-models)).

### 10.5 Platform support

| Feature | Desktop | Android / iOS | Web |
|---|:-:|:-:|:-:|
| Datasets, Train, Generate, Evaluate, Models | ✅ | ✅ | ❌¹ |
| GGUF run, fine-tune, export | ✅ | ✅² | ❌¹ |
| Engine log | ✅ | ✅ | ❌¹ |
| System monitor | ✅ | partial³ | ❌¹ |

¹ Tauri's Rust backend does not run in a plain browser. To run a *model* in a
browser, use `slm_wasm` ([§11.3](#113-in-the-browser-webassembly)).
² Limited by device RAM. ³ Limited by `sysinfo`'s mobile backends.

### 10.6 Architecture

```text
ui/               index.html, styles.css, app.js (vanilla), escape.js (+ escape.test.js), stream.js (+ stream.test.js)
src/lib.rs        Tauri builder, shared state, verbose → event sink, command registry
src/commands.rs   #[tauri::command]s calling ferrum_core directly
src/capable.rs    machine micro-benchmark + capability bounds
src/datasets.rs   corpus catalog + HTTP / Hugging Face / Kaggle downloads
src/paths.rs      validation of every file path the webview sends (no empty / NUL / over-long paths, no /dev /proc /sys, no directory or FIFO as a file)
tauri.conf.json   window + bundle config (frontendDist = "ui") + Content-Security-Policy (no inline script or style)
capabilities/     Tauri v2 permissions (event listen/unlisten + dialog open/save only)
```

Heavy commands run on blocking tasks and stream events back to the UI
(`engine-log`, `train-progress`, `train-done`, `gen-fragment`,
`finetune-progress`, `export-progress`), so the window stays responsive. Under
a token budget, `train-progress` sends `total: null` because the epoch count is
not known in advance.

---

## 11. Deploying models

A trained model is one self-contained `.gguf` file (GGUF v3, `gpt2`
architecture). It carries weights, metadata, and, for BPE models, the
tokenizer. There is nothing else to ship, and the same file runs in Ferrum
(native and WASM), in llama.cpp, in Ollama, and in LM Studio
([§9.5](#95-running-ferrum-models-in-llamacpp-ollama-and-lm-studio)).

### 11.1 Native (server, desktop, edge box)

```bash
cargo build --release -p slm_cli
scp target/release/train_transformer model.gguf user@host:/opt/ferrum/
ssh user@host /opt/ferrum/train_transformer generate /opt/ferrum/model.gguf "prompt" --chars 200
```

To embed inference in your own service, depend on `ferrum_core` and call
`GenerativeSLM::load` and `generate`, with no model server, or run
`ferrum-serve` ([§9.7](#97-serving-ferrum-serve)) for an HTTP interface. Set
`FERRUM_NUM_THREADS` to bound CPU use per process.

### 11.2 Embedded and constrained targets

- int8 models are typically tens of kilobytes for sub-megabyte parameter
  counts. Use `to_bytes_quantized_int4()` to halve that again.
- Build with the workspace `release` profile (LTO, one codegen unit).
- On single-core or timing-sensitive targets, set `FERRUM_NUM_THREADS=1` for
  fully serial, predictable execution.

### 11.3 In the browser (WebAssembly)

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.122
bash scripts/build_wasm.sh        # → slm_wasm/pkg/{slm_wasm.js, slm_wasm_bg.wasm}
```

Serve `pkg/` and your `model.gguf` from any static host. There is no backend,
and the visitor's input never leaves their machine.

```js
import init, { TransformerSLMModel } from "./pkg/slm_wasm.js";
await init();
const bytes = new Uint8Array(await (await fetch("model.gguf")).arrayBuffer());
const slm = new TransformerSLMModel(bytes);
const meta = JSON.parse(slm.metadata());          // class_names, input_dim, tokenizer_state, …
let probs = slm.prime(new Float32Array(contextIds)); // KV-cached: prime once…
const next = slm.sample_from_probs(probs, 0.7, Math.random());
probs = slm.predict_next_cached(next);            // …then one token at a time
const attn = slm.get_last_attention_weights();    // [heads × T × T] for visualization
```

Other methods: `predict_next` (full forward), `num_heads`, `vocab_size`,
`context_len`, `num_layers`, `cached_len`, `entropy`, and `top_k_indices`.
WASM runs single-threaded but is otherwise the same engine.

### 11.4 Air-gapped deployment

Train on a connected machine, then carry the single `model.gguf` and the static
binary in on physical media. Nothing is fetched at runtime, so there is nothing
to update in the field.

### 11.5 Versioning and compatibility

- FINF `.bin` files do not load. Retrain, or convert with an older Ferrum
  release. Status §7 has the full list of breaking changes.
- Models are GGUF v3 with `ferrum.format = 1`. A future format change will bump
  that number.
- `TransformerConfig` grows only by fields with defaults. Use
  `..TransformerConfig::default()`.
- The file fully determines architecture, vocabulary, and tokenizer, so there
  is no version skew between a model and its config.

---

# Part III — Reference

## 12. Library API reference

### 12.1 Tensors, ops, and parallelism

`Tensor` is a flat `Vec<f32>` plus a shape (`Tensor::row`, `Tensor::matrix`,
`matrix_dims()`). The `ops` module provides matmul, a fused cache-tiled
`linear_forward`, packed `qlinear` (int8/int4), `softmax_rows`, and
`argmax_rows`.

The matmul kernels split output rows across a **persistent worker pool**
(`parallel`). It is built only on `std` threads, channels, and `Arc`, with no
`unsafe`. Workers are spawned once and reused, so generation pays no per-call
thread-creation cost. Single-token decode (`m = 1`) cannot be split by rows,
so the quantized path splits the GEMV **by output column** (`run_1d`). The
worker count comes from `std::thread::available_parallelism()`; override it
with `FERRUM_NUM_THREADS` and query it with `ferrum_core::num_threads()`. Small
workloads and `wasm32` run serially. Splitting never changes per-element
arithmetic, so results are bit-identical at any thread count.

### 12.2 Layers

All layers implement `Layer` (`forward`, `name`, `as_any`) and compose into
`Sequential`.

| Layer | Forward |
|---|---|
| `Linear` | `y = xW + b`; may hold a packed `Arc<QWeight>` consumed by `qlinear` without f32 expansion |
| `ActivationLayer` | element-wise ReLU, Softmax, … |
| `LayerNorm` | per-row normalization with learned γ/β |
| `Embedding` | token + positional lookup |
| `Flatten` | `[T, D]` → `[1, T·D]` |
| `TransformerBlock` | causal multi-head self-attention + FFN |

`KvCache` stores past keys and values per block for O(context)-per-token
generation.

### 12.3 Training primitives

- **`TransformerNet`**: the trainable causal transformer.
  - Setters: `set_qat`, `set_weight_decay`, `set_dropout`, `set_grad_clip`,
    `set_lr_schedule`, `set_weight_tying`, `set_window_stride`.
  - `save_checkpoint(&rng)` and `load_checkpoint(bytes)` serialize the full
    training state.
  - `to_inference()` returns a `Sequential`.
- **Epoch drivers**:
  - `train_transformer_epoch(net, tokens, batch, &adam, rng)` runs serially.
  - `train_transformer_epoch_threaded(…, threads)` shards each minibatch
    across threads.
  - `train_transformer_steps(…, threads, max_steps, &mut on_step)` runs one
    epoch capped at `max_steps` optimizer steps and calls
    `on_step(&net, &rng)` after each step (the hook used for budgets and
    checkpoints).
- **Optimizers**: `Sgd` (optional momentum), `Adam`, AdamW decay
  (`Adam::with_weight_decay`), `clip_grad_norm`, and `LrSchedule::{warmup_cosine,
  warmup_linear}` with `LrDecay`.
- **Losses**: `softmax_cross_entropy`, `mse`.

### 12.4 `GenerativeSLM`

`GenerativeSLM` wraps a `Sequential`, a `Normalizer`, and `ModelMetadata`.

| Group | Methods |
|---|---|
| Train | `train_transformer_config`, `train_transformer_config_threaded`, `train_transformer_config_validated`, `load_or_train`; legacy positional `train_transformer[_with_callback / _threaded_with_callback]` |
| Generate | `generate`, `generate_with`, `generate_continuation[_with]`, `generate_stream[_with]`, `generate_until`, `generate_stream_until`, `generate_tokens` (token budget, stop strings, returns the stop reason) |
| Evaluate | `evaluate(text) -> Evaluation { num_predictions, cross_entropy, bits_per_token, perplexity }`; `score(context, continuation)` and `score_rolling(text)` return `Score { logprob, n_tokens, greedy }` (§8.3) |
| Persist | `to_bytes` (f32), `to_bytes_quantized` (int8), `to_bytes_quantized_int4`, `save` (int8), `save_int4`, `load`, `from_bytes` |

`SamplingParams { temperature, top_k, top_p, repetition_penalty }`;
`SamplingParams::with_temperature(t)` reproduces temperature-only sampling.

### 12.5 The BPE tokenizer

```rust
use ferrum_core::ByteBpeTokenizer;

let mut tok = ByteBpeTokenizer::train(corpus, 4096)?;   // learns up to vocab-256 merges
let ids  = tok.encode("lowest news");
assert_eq!(tok.decode(&ids), "lowest news");
let state = tok.encode_state();                          // "a,b;c,d;…" (+ "\n" + specials)
let tok2  = ByteBpeTokenizer::from_state(&state)?;
let [bos, eos, pad, unk] = tok.add_standard_special_tokens();
let framed = tok.encode_framed("text", true, true);      // <bos> … <eos>
```

Text is pre-split with the GPT-2 regex that llama.cpp uses
(`tokenizer.ggml.pre = "gpt-2"`), so merges never span a word boundary and
llama.cpp tokenizes prompts exactly as Ferrum does. Encoding is rank-based,
applying the earliest-learned merge first. `<|endoftext|>` is always a
reserved special token, counted in the vocabulary size; the literal string
encodes to that one id.

### 12.6 Quantization

Symmetric int8: `value ≈ i8 × scale`, `scale = max|value| / 127`, per tensor
or **per channel** (one scale per output row).

- `fake_quantize_int8` and `fake_quantize_int8_per_channel` snap values onto
  the grid in place (used by QAT).
- The weight-snapping grid used by Ferrum's own models is GGUF **Q8_0**
  (blocks of 32 along each weight row with an f16 scale, §12.7). Matrices whose
  row length is not a multiple of 32 stay f32, and non-finite tensors are left
  untouched. The per-tensor and per-channel helpers above serve the imported
  Llama stack.
- `QWeight` holds int8 or int4 weights packed in memory. int4 uses a
  **split-half** layout: byte `b`'s low nibble is column `b` and its high
  nibble is column `half + b`. That makes each nibble lane unit-stride, so the
  decode loop autovectorizes like int8. An interleaved layout ran several
  times slower.

### 12.7 The Ferrum GGUF format

Every file Ferrum writes is GGUF v3. There are four kinds.

| Kind | `general.architecture` | Runs in llama.cpp |
|---|---|---|
| BPE transformer SLM (what `train` writes) | `gpt2` | Yes |
| Generic Ferrum model (character-level SLMs, Ferrum-ML tabular MLPs) | `ferrum` | No |
| Training checkpoint | `ferrum-checkpoint` (`ferrum-llama-checkpoint` for Llama fine-tunes) | No |
| Reusable tokenizer file | `ferrum-tokenizer` | No |

#### The GPT-2 layout

Ferrum's own transformer matches GPT-2: learned absolute positions, pre-LayerNorm
(epsilon 1e-5, with bias), dense multi-head attention with biases, a **GELU**
feed-forward block (the tanh approximation llama.cpp uses as `ggml_gelu`), and
an LM head **without a bias**.

Metadata that llama.cpp reads:

| Key | Type | Value |
|---|---|---|
| `general.architecture` | string | `gpt2` |
| `gpt2.context_length`, `gpt2.embedding_length`, `gpt2.block_count`, `gpt2.feed_forward_length`, `gpt2.attention.head_count` | u32 | model dimensions |
| `gpt2.attention.layer_norm_epsilon` | f32 | 1e-5 |
| `tokenizer.ggml.model` | string | `gpt2` |
| `tokenizer.ggml.pre` | string | `gpt-2` |
| `tokenizer.ggml.tokens`, `.merges`, `.token_type` | arrays | the vocabulary in GPT-2 bytes-to-unicode form |
| `tokenizer.ggml.bos_token_id`, `.eos_token_id` | u32 | both the id of `<|endoftext|>` |
| `tokenizer.ggml.add_bos_token`, `.add_eos_token` | bool | false |
| `general.file_type` | u32 | 0 all F32, 7 mostly Q8_0, 2 mostly Q4_0 |
| `general.quantization_version` | u32 | 2 |

Tensors (GGML dims `[ne0, ne1]`):

| Tensor | Dims |
|---|---|
| `token_embd.weight` | `[n_embd, n_vocab]` |
| `position_embd.weight` | `[n_embd, n_ctx]` |
| `blk.N.attn_norm.{weight,bias}` | `[n_embd]` |
| `blk.N.attn_qkv.{weight,bias}` | `[n_embd, 3·n_embd]` and `[3·n_embd]`, ordered Q, K, V |
| `blk.N.attn_output.{weight,bias}` | `[n_embd, n_embd]` and `[n_embd]` |
| `blk.N.ffn_norm.{weight,bias}` | `[n_embd]` |
| `blk.N.ffn_up.{weight,bias}` | `[n_embd, n_ff]` and `[n_ff]` |
| `blk.N.ffn_down.{weight,bias}` | `[n_ff, n_embd]` and `[n_embd]` |
| `output_norm.{weight,bias}` | `[n_embd]` |
| `output.weight` | `[n_embd, n_vocab]`; **omitted when the head is tied** (llama.cpp falls back to `token_embd`) |

Ferrum stores weights as row-major `[in, out]`, so they are written transposed.
This is why `--tie` shrinks the file as well as the parameter count.

#### Ferrum extras

Keys in the `ferrum.` namespace are ignored by llama.cpp.

| Key | Type | Meaning |
|---|---|---|
| `ferrum.format` | u32 | 1 |
| `ferrum.quant` | string | `f32`, `int8`, or `int4` |
| `ferrum.metadata` | string | `ModelMetadata` JSON: task, class names, dimensions |
| `ferrum.normalizer` | string | Normalizer state (empty for SLMs) |

`train_transformer info` reads these. A GPT-2 GGUF that Ferrum did not write
also loads, with its metadata rebuilt from the `gpt2.*` keys.

#### Quantization rules

| Type | Used for |
|---|---|
| F32 | `quant = f32`; all 1-D tensors (biases, LayerNorm); `position_embd`; and any matrix whose row length (`ne0`) is not a multiple of 32 (stored exactly, never as F16) |
| Q8_0 | `int8`: blocks of 32 values along each weight row, one f16 scale per block |
| Q4_0 | `int4`: same block size, 4 bits per value; lossy |

- Q8_0 and Q4_0 need the row length (`ne0`, the input width) to be a multiple of
  32. Keep `--embed`, `--hidden`, and `--vocab` multiples of 32 so every matrix
  quantizes. The 507,552-parameter reference is built that way.
- `position_embd` stays F32 when quantizing, following llama-quantize.
- Because ineligible tensors are F32, an int8 file reloads bit-identically for
  **every** tensor.
- Loaded Ferrum GGUF weights are dequantized to f32 in memory. The packed
  int8/int4 kernels are for imported Llama models.

#### QAT equals Q8_0

Quantization-aware training simulates **Q8_0 exactly**: 32-element blocks along
each weight row, the scale computed as `max|x| / 127` and rounded to f16, and
values rounded to int8. Because the training step and the file writer share one
encoder, an int8 file reloads **bit-identically** to the model you trained.
`evaluate()` returns the same numbers before and after saving. **QAT is int8
only.** int4 export is lossy and is applied after training.

#### Tokenizer

The tokenizer is byte-level BPE. Text is pre-split with the GPT-2 regex that
llama.cpp uses for `tokenizer.ggml.pre = "gpt-2"`, and tokens and merges are
stored in the GPT-2 bytes-to-unicode form. `<|endoftext|>` is registered as a
special token during BPE training and **is always reserved**, so every model has
a real `eos_token_id`. The literal string `<|endoftext|>` in text encodes to
that one id.

#### The generic codec (not runnable in llama.cpp)

Character-level SLMs and Ferrum-ML tabular classifiers do not fit the GPT-2
layout. They use a generic layer list: `ferrum.layer_count`,
`ferrum.layer.{i}.type` (`linear`, `activation`, `flatten`, `embedding`,
`layernorm`, `transformer_block`), per-layer dimension keys, and `l{i}.*`
tensors, with matrices in F32, Q8_0, or Q4_0 under the same 32-wide rule. These
are valid GGUF, so general GGUF tools can read their metadata, but llama.cpp
cannot run them: it has no MLP-classifier architecture and no character-level
tokenizer.

#### Checkpoints and the tokenizer file

A training checkpoint is a GGUF container with `ferrum.ckpt.version = 1`, the
scalar settings (`vocab`, `context`, `embed`, `heads`, `hidden`, `blocks`,
`qat`, `tie`, `step`, `rng`), and three f32 tensors per parameter (`p{i}.data`,
`p{i}.m`, `p{i}.v`: the weights and the two Adam moments). A tokenizer file holds
`tokenizer.ferrum.state`. Both are for Ferrum only. Checkpoints written before
this change (magic `FCKP`) do not load; the error tells you to retrain. Old
tokenizer files in the `256;257...` text form are also rejected.

Files are written atomically. The loader bounds-checks every dimension before it
allocates, and a file that is valid GGUF but not a Ferrum model (for example a
Llama GGUF), or a truncated file, gives a `Format` error rather than a panic.

### 12.8 GGUF reader, tokenizer, runner, trainer, writer

| Module | Key API | Notes |
|---|---|---|
| `gguf` (directory: `format`, `cursor`, `header`, `tensors`, `dequant`, `llama`) | `Gguf::from_path` (in memory), `Gguf::open` (streamed via `Mutex<File>`, no mmap), `load_llama(QKind)`, `load_llama_prec(None)` | GGUF v2/v3; checked offsets, EOF guards, rejects nested arrays and absurd counts |
| `gguf_tokenizer` | `GgufTokenizer::from_gguf`, `encode`, `decode` | BPE exact; SPM decode exact, encode greedy longest-match |
| `llm` | `LlamaModel::{forward, generate}`, `LlamaConfig`, `RopeType` | RMSNorm, RoPE (Norm and Neox), GQA + KV cache, SwiGLU; cached decode verified row-for-row against the full forward |
| `llm_train` | `LlamaTrainer::{new, set_lr, set_lr_schedule, set_grad_clip, train_step, train_batch, finetune_epoch_threaded, save_checkpoint, load_checkpoint_into}` | finite-difference-checked backward pass; AdamW; `.flck` checkpoints; rejects quantized weights |
| `gguf_write` | `GgufBuilder`, `write_llama_gguf`, `LlamaModel::write_gguf`, `GgufQuant` | byte-exact GGUF v3; encoders are exact inverses of the decoders (verified round trip) |
| `chat_format` | `ChatFormat::{ChatMl, Llama3, Llama2Inst, Zephyr}`, `detect`, `render`, `stop_token` | single-turn instruct formatting |

---

## 13. Performance and benchmarks

All numbers were measured on this project on 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4 cores / 8 threads, 15 GiB) on
2026-10-26. They are medians of three runs of release builds made with the low-memory
settings (no LTO, 16 codegen units), CPU only, with the desktop session running, so
treat them as indicative; a shipped build and an idle machine are somewhat faster.
The logs behind them, and the earlier builds used for the before and after
comparison, are described in [Samples §9](Samples.md#9-benchmarking).

### 13.1 Determinism (verified)

- `cargo test -p slm_cli --release --test determinism` trains, generates, and
  evaluates through the CLI at `FERRUM_NUM_THREADS=1` and `4` (with `--threads`
  fixed) and requires identical bytes.
- The same training config produced byte-identical models twice (Samples 1, and
  again in Sample 2's lock check). Changing the data-parallel shard count changes
  low bits of the weights ([§6.8](#68-multi-threading-and-determinism)); the
  matmul pool size alone does not.

### 13.2 End-to-end scaling

Measured with `scripts/bench_slm.sh` on the 507,552-parameter reference model
(`FERRUM_NUM_THREADS` sets the matmul pool and, with `--threads` auto, the
number of training workers):

| `FERRUM_NUM_THREADS` | Train tokens/s (range of 3) | Speedup | Eval s | Generate ms/token | Held-out perplexity |
|---|---|---|---|---|---|
| 1 | 2,900 (2,799 to 3,322) | 1.00× | 2.28 | 0.256 | 81.70 |
| 2 | 6,058 (5,351 to 6,256) | 2.09× | 1.71 | 0.260 | 76.46 |
| 4 | 6,764 (5,857 to 7,810) | 2.33× | 1.91 | 0.309 | 78.70 |
| 8 | 7,090 (4,918 to 7,407) | 2.44× | 1.55 | 0.291 | 74.24 |

Small models scale poorly. Much of a training step is serial (softmax, small
per-head matmuls, the optimizer), and generation is a chain of small dependent
matmuls, so eight threads give 2.4× on training, 1.5× on
evaluation, and nothing on generation. The one-thread row runs the serial trainer; two workers
overlap some of its serial work, which is why the step from 1 to 2 threads is
larger than 2×. Larger models, where big matmuls dominate, approach the kernel
ceiling below. For faster generation, prefer a BPE vocabulary (fewer steps per
character) and a smaller network over more cores.

### 13.3 Kernel throughput (synthetic ~1B-class shapes)

| Square GEMM | 8 threads (GFLOP/s) | 1 thread (GFLOP/s) | Speedup |
|---|---|---|---|
| 256² | 69.4 | 21.6 | 3.2× |
| 512² | 51.4 | 17.7 | 2.9× |
| 1024² | 42.0 | 14.2 | 3.0× |
| 2048² | 40.3 | 13.6 | 3.0× |

The cache-tiled kernel with a 4×8 register tile holds about 40 GFLOP/s at 2048²
on 8 threads, and 14 on one. The kernels use no explicit SIMD (`unsafe` is
forbidden, so there are no intrinsics) and rely on what the compiler
auto-vectorizes. Building with `RUSTFLAGS="-C target-cpu=native"` raised the 2048² rate from 32.7 to 54.9 GFLOP/s (median of three back-to-back runs of each build), but it did not speed up the reference model: training ran at 6,004 tokens/s against 6,542 for the default build (medians of three, same perplexity), because its matmuls are small. A native build also ties the binary to one CPU, so the shipped build does not use it.

**f32 decode GEMV** (`m = 1`) runs serially at any thread count, at 8.0 GB/s (2048×2048), 7.5 (2048×8192), 5.2 (8192×2048), 4.6 (2048×32000)
of weights streamed on 8 threads. A synthesized f32 1B decode step reaches
**1.2 tokens/s** at 8 threads and 0.9 at one thread.

### 13.4 Quantized decode and the GGUF runner

| Synthesized 1B decode step | f32 (8 threads) | int4 (8 threads) | int4 (1 thread) |
|---|---|---|---|
| ms/token | 807 | **150** | 253 |
| tokens/s | 1.24 | **6.68** | 3.95 |
| Bytes streamed per token | 3.48 GB | 0.44 GB | 0.44 GB |

- **int4** is the right default for the GGUF runner: about 5.4× f32 on 8 threads, at half
  int8's RAM.
- **int8** is faster per call on three of the four shapes measured (for example
  `ffn up`: 1002 µs int8 against 1335 µs int4, at 8 threads), because there is no nibble
  unpack. Choose it when RAM is ample.
- The column split spreads the `m = 1` GEMV across cores for both.
- Net result for a 1B model: about 6.7 tokens/s decode. Prefill is compute-bound:
  at the measured 40 GFLOP/s, 2 FLOPs per parameter per token comes to about
  25 s for 512 tokens and 102 s for 2,048 (an estimate from the GEMM rate, not a measurement).

### 13.5 Feasibility at 1B and beyond

Two rules of thumb set the limits. A forward pass costs about 2·N FLOPs per
token, and decode streams every weight once per token. A training step costs
about 6·N·T FLOPs and needs about 16 bytes per parameter resident.

| | 1B model on this machine |
|---|---|
| **Load** | Yes for supported quants: about 0.5 GB int4 plus the f32 token embedding; 2–3.5 GB peak during import. 7B F16 sources do not fit the no-mmap path. |
| **Run** | Yes, slowly (above) |
| **Fine-tune / train** | **No.** Optimizer state alone is about 16 GB, and 100M tokens through 1B parameters would take about 172 days at the measured 40 GFLOP/s. |

### 13.6 Reproducing

```bash
cargo build --release -p slm_cli
cargo bench --bench gemm                          # auto threads
FERRUM_NUM_THREADS=1 cargo bench --bench gemm     # serial
cargo bench --bench gemm -- 512 1024 4096         # custom GEMM sizes
scripts/bench_slm.sh data/prepared/train.txt      # train / eval / generate speed
for t in 1 2 4 8; do FERRUM_NUM_THREADS=$t target/release/train_transformer \
  generate model.gguf "a long enough seed text here" --chars 2000 --gen-seed 42 > out_$t.txt; done
md5sum out_*.txt                                  # identical
```

---

## 14. Working with any language

The byte-level tokenizer starts from all 256 byte values, and every UTF-8 text
is made of them. So **any language, script, or emoji round-trips exactly**,
with no `<UNK>` token. This is the same design GPT-2 introduced and GPT-4 still
uses.

- Train on any language. Use BPE (`--vocab ≥ 256`) rather than
  character-level for scripts with large alphabets.
- Mixed scripts, code, and emoji-heavy chat logs all encode cleanly.
  Streaming never shows a half-character.
- Imported GGUF models bring *their own* tokenizer. Their language coverage is
  whatever they were trained on.

**Caveats.** The model learns only what is in the corpus: English text does
not teach French. In some scripts one visible character is several bytes, so a
short corpus may have fewer tokens than expected and hit the "corpus must be
longer than the context window" error. The fix is more text, a smaller
context, or a smaller vocabulary. Languages without spaces benefit from a
larger vocabulary. Right-to-left scripts are stored correctly; how they
display is up to your terminal or app.

---

## 15. Use cases and applications

Ferrum fits wherever a **small, self-contained, predictable model must run
without a GPU, a Python runtime, or a network connection**.

| # | Use case | Why Ferrum fits |
|---|---|---|
| 1 | Offline text generation on edge devices | One `.gguf`, CPU only, about 3× smaller via int8 (30% of the f32 file for the reference model), nothing to install |
| 2 | Running and fine-tuning small open models offline | GGUF import with its own tokenizer, AdamW fine-tune, export back to GGUF; private but slow (a few tokens/s at 1B) |
| 3 | Domain autocomplete (shell history, logs, code, templates) | BPE captures flags, paths, and identifiers; deterministic suggestions; millisecond latency |
| 4 | Privacy-preserving on-device modeling | Nothing leaves the machine; no telemetry; auditable code |
| 5 | In-browser demos (WASM) | `slm_wasm` runs models client-side with no backend or API key |
| 6 | Reproducible research and teaching | Readable forward and backward pass; bit-for-bit reproducible |
| 7 | Embedded and resource-constrained systems | Kilobyte-scale int8/int4 models; serial, predictable timing |
| 8 | Air-gapped and field deployments | One file plus a static binary; nothing to fetch or update |
| 9 | Cheap, horizontally scaled CPU inference | Embed `ferrum_core` in a Rust service with no model server |
| 10 | Rapid prototyping | Compare tokenizers, contexts, and sizes in minutes with identical APIs |

(Tabular classification and regression now live in the sibling **Ferrum-ML**
project.)

**A starter project.** Train a small BPE transformer on your own shell history
and you have a private, offline command autocomplete. That exercises every
strength above.

**The decision rule:** use CPU-bound inference when the model is small, the
task is narrow, and the value lies in being local, private, offline, cheap,
deterministic, or embeddable. Do not use it when the value depends on a large
model's breadth, real-time responses from a big model, or reliable open-ended
reasoning.

| Situation | CPU-bound inference? |
|---|:-:|
| Keyword spotting, sensor anomaly detection on a device | ✅ Excellent |
| Private, offline autocomplete or classification | ✅ Excellent |
| Teaching how models work, reproducible research | ✅ Excellent |
| A small open model offline for a private demo | ✅ Workable (slow) |
| Browser "try it yourself" demo | ✅ Good |
| Interactive chat with a ChatGPT-class model | ❌ Use a GPU or cloud |
| High-throughput serving of a large model | ❌ Use accelerators |
| Reliable facts or logic from a tiny model | ❌ Wrong tool entirely |

This niche is real and growing under the names *edge AI* and *TinyML*. One
documented ESP32-S3 deployment ran inference in about 113 ms at about 5.8 mA,
enough for a week on battery.

---

## 16. FAQ

**Does it need a GPU?** No, and it never uses one, even if one is present.

**What dependencies does it have?** `ferrum_core` has two pure-Rust Unicode-table crates
(`unicode-general-category`, `unicode-normalization`); `slm_cli` adds `sha2` and `serde_json`,
`slm_server` adds `axum`/`tokio`/`serde`/`serde_json`/`sha2`. `slm_wasm` uses `wasm-bindgen`, and the GUI uses Tauri plus a few helper crates
([§2.2.1](#221-what-is-used-and-what-was-tried-and-not-kept)).

**Character-level or BPE?** BPE (`--vocab 512–4096`) for real text,
multilingual text, or anything long. Character-level (`--vocab 0`) for tiny
corpora and maximum transparency.

**Where does the tokenizer live?** Inside the model file (the `tokenizer.ggml.*`
keys). With `--tokenizer`, it is also saved to disk as its own GGUF file so
later runs reuse it.

**Is BPE compatible with QAT?** Yes. Tokenization changes only the token
stream. QAT acts on the weights.

**How big are the models?** int8 files are about one byte per parameter plus
small f32 tensors, so a 1M-parameter model is about 1 MB. int4 is about half
that.

**Why does one epoch take so long?** If you passed `--stride 1`, each epoch
trains on every token about `context_len` times. Use the default stride, and
prefer `--tokens` to `--epochs` for real runs.

**Training crashed after hours. Did I lose everything?** Not if you used
`--checkpoint`. Re-run with `--resume <file>`.

**Can it run models I downloaded?** Llama/Qwen GGUFs, yes
([§9.1](#91-running-run-gguf)). Other architectures are rejected.

**Can llama.cpp, Ollama, or LM Studio run the models I train?** Yes, if they are
BPE models (`--vocab 256` or more): they are `gpt2`-architecture GGUF files
([§9.5](#95-running-ferrum-models-in-llamacpp-ollama-and-lm-studio)).
Character-level models are not.

**Can I get the probabilities of a text, or serve a model?** Yes: the
`score` library API ([§8.3](#83-the-scoring-api)) and `ferrum-serve`
([§9.7](#97-serving-ferrum-serve)).

**Can I fine-tune them?** Yes, with full AdamW at f32. That works up to about
135M parameters comfortably on 16 GB. 1B is out of reach (about 16 GB of
optimizer state alone).

**Can it write GGUF?** Yes: `export-gguf` produces files that run in
llama.cpp, ollama, and LM Studio.

**Does `--chars` count characters or tokens?** Characters, for both
tokenizers.

**"Corpus must be longer than the context window"?** BPE compresses, so a
short or repetitive corpus may have too few tokens. Use more text, a smaller
`--context`, or a smaller `--vocab`.

**Is it deterministic?** Yes. Training, tokenizer learning, and generation are
deterministic for a fixed seed and configuration, and identical across matmul
thread counts.

**Will my old models load?** No. FINF `.bin` files, `FCKP` checkpoints, and
old tokenizer files no longer load (deliberate; retrain). See
[Status §7](Status.md#7-backward-compatibility).

**Where did the tabular trainer and the web playground go?** The tabular
trainer is in `../Ferrum-ML`. The old one-hot web playground is archived in
`../Ferrum-Junk`.

**What can it not do?** See the next chapter.

---

## 17. Limits: an honest critique

> This is deliberately the harshest chapter. A tool whose limits you
> understand is one you can trust. Limits are labelled **(Design)** for
> deliberate trade-offs and **(Fundamental)** for what no engineering on a
> project like this can remove.

### 17.1 A teaching-and-edge engine, not an intelligence

Ferrum is not a drop-in replacement for large GPU-trained LLMs. Even when it
runs a downloaded billion-parameter model, it does so at a few tokens per
second. That proves capability; it is not a usable chatbot.

### 17.2 Limits chosen on purpose (Design)

- **The models are small.** Ferrum models range from kilobytes to tens of
  megabytes. Commercial models are gigabytes to terabytes trained on much of
  the internet. A small corpus is memorized more than generalized (seen-text
  perplexity of about 1.0 against about 3.7 unseen in §6.10). Do not expect
  long-form writing, factual Q&A, conversation, or complex
  instruction-following.
- **CPU only is a hard ceiling.** Compute-optimal training tops out around
  5M parameters per day on a laptop, and generation barely speeds up with more
  cores.
- **Writing the kernels ourselves costs peak speed.** Without explicit SIMD or a tuned
  BLAS, much of a modern CPU's vector throughput goes unused.
- **You supply the intelligence ceiling.** Ferrum ships no pretrained weights
  of its own. Quality is bounded by your data and patience
  ([§3](#3-data-decides-garbage-in-garbage-out)).
- **The GUI is verified by compilation and backend tests, not on every
  machine.** Build it yourself before relying on it.

### 17.3 Limits shared by all small models

- **More hallucination, less reasoning.** Smaller models show fewer reasoning
  capabilities and more hallucination, due to scale. Never wire a small
  generative model to anything consequential without a human or a hard rule
  checking its output.
- **Brittle logic.** Rule-following, multi-step reasoning, and cross-step
  consistency are weak. Do not expect arithmetic or long constraint chains.

### 17.4 What language models will never achieve (Fundamental)

- **They predict; they do not understand.** You can read every line of Ferrum
  and confirm there is no understanding module, only weighted guesses about
  the next token. It can find something *probable*; it cannot *know* that it
  is true.
- **Hallucination can be reduced, not eliminated.** Treat every
  factual-sounding output as unverified.
- **No grounding, memory, or goals.** There are no senses, no live
  information, no memory beyond frozen weights and a short context window,
  and no ability to explain *why*.
- **No substitute for thinking.** The fluency is real; the reliability is not.

### 17.5 The fair counterpoint

Most Design limits are the *reasons* to choose Ferrum:

- "Too small to be ChatGPT" is the same property as "small enough to audit
  and run on a Raspberry Pi".
- "CPU-only" is the same property as "no driver stack, runs offline
  anywhere".
- "You train it yourself" is the same property as "your data never leaves
  your machine".
- "Writes its own maths" is the same property as "nothing to vet".

> **Verdict:** Ferrum will never be intelligent, a knowledge source, or
> trustworthy without verification, and that is fine. It was built to be
> small, transparent, private, and yours. Judge it as a precise tool, not a
> substitute brain, and it excels.

---

## Sources

- Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762
- NVIDIA Research, *Small Language Models Are the Future of Agentic AI* — https://research.nvidia.com/labs/lpr/slm-agents/
- *Small Language Models for Agentic Systems: A Survey* (arXiv 2510.03847) — https://arxiv.org/pdf/2510.03847
- Hugging Face LLM Course, *Byte-Pair Encoding tokenization* — https://huggingface.co/learn/llm-course/en/chapter6/5
- A. Karpathy, *minbpe* — https://github.com/karpathy/minbpe
- Gunasekar et al., *Textbooks Are All You Need* (arXiv 2306.11644) — https://arxiv.org/abs/2306.11644
- Xu et al., *Data-Centric AI in the Age of Large Language Models* (arXiv 2406.14473) — https://arxiv.org/abs/2406.14473
- TechTarget, *What is garbage in, garbage out (GIGO)?* — https://www.techtarget.com/searchsoftwarequality/definition/garbage-in-garbage-out
- Nagle, Redman & Sammon, *Only 3% of Companies' Data Meets Basic Quality Standards*, HBR (2017) — https://hbr.org/2017/09/only-3-of-companies-data-meets-basic-quality-standards
- Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference* (arXiv 1806.08342) — https://arxiv.org/pdf/1806.08342
- NVIDIA Technical Blog, *Improving INT8 Accuracy Using Quantization Aware Training* — https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/
- Markaicode, *Rust for ML: Building High-Performance Inference Engines in 2025* — https://markaicode.com/rust-ml-Building-high-performance-inference-engines-2025/
- Carnelos et al., *MicroFlow: An Efficient Rust-Based Inference Engine for TinyML* (arXiv 2409.19432) — https://arxiv.org/abs/2409.19432
- Talent500, *What Is TinyML?* — https://talent500.com/blog/what-is-tinyml-introduction/
- *Deploying TinyML for energy-efficient object detection…*, Nature Scientific Reports (2025) — https://www.nature.com/articles/s41598-025-27818-9
- *On the Fundamental Limits of LLMs at Scale* (arXiv 2511.12869) — https://arxiv.org/pdf/2511.12869
- Huang et al., *A Survey on Hallucination in Large Language Models* (arXiv 2311.05232) — https://arxiv.org/pdf/2311.05232
- Tauri — https://tauri.app/
