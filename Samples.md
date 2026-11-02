# Ferrum Samples

Ten walk-throughs. Each one has a goal, what you need, numbered steps with exact
commands, the output from a real run, and what to look at. Run every command
from the repository root. Scratch files go in `samples/` (git-ignored).

Every output block in this file was pasted from a run of the commands shown
above it, on this machine, with the build at the commit named below. Nothing
was typed in by hand. Where a block is shortened, the omission is marked in the
block itself. Where a tool could not be run, the sample says so.

| | |
|---|---|
| Machine | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 4 cores / 8 threads, 15 GiB RAM |
| Date | 2026-10-26 |
| Build | `cargo build --release --workspace` at commit `3ba1bc8` with `CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16` (the low-memory settings; the shipped profile uses LTO and one codegen unit, so absolute speeds here are a little lower than a shipped build) |
| Toolchain | rustc 1.96.0 |
| Other work | The desktop session was running (browser, editor), so absolute speeds are noisy by several percent |

Throughput numbers differ on other machines. Compare ratios, not absolute speeds.

All samples use the release build and put it on your `PATH` for the shell session:

```bash
cargo build --release --workspace
export PATH="$PWD/target/release:$PATH"
mkdir -p samples models metrics
```

The commands are written for Linux (they use `sha256sum`, `md5sum`, and `timeout` from GNU coreutils).

## Contents

1. [Quick start: a 500K SLM on TinyStories](#1-quick-start-a-500k-slm-on-tinystories)
2. [The full DVC pipeline](#2-the-full-dvc-pipeline)
3. [Cleaning your own corpus](#3-cleaning-your-own-corpus)
4. [Training controls](#4-training-controls)
5. [Evaluation](#5-evaluation)
6. [Generation](#6-generation)
7. [Serving, and running your model in llama.cpp, Ollama, and LM Studio](#7-serving-and-running-your-model-in-llamacpp-ollama-and-lm-studio)
8. [Testing with LightEval](#8-testing-with-lighteval)
9. [Benchmarking](#9-benchmarking)
10. [The Rust library API](#10-the-rust-library-api)

For the full flag reference see [Manual §6.2](Manual.md#62-command-line-reference); for what
the numbers mean see [Manual §8](Manual.md#8-evaluating-a-model).

---

## 1. Quick start: a 500K SLM on TinyStories

**Goal.** Go from nothing to a trained 507,552-parameter model that writes short stories.

**You need.** Rust, `curl`, and about 45 MB of disk (the raw file plus the prepared splits). No Python and no DVC.

**Steps.**

1. Download the pinned TinyStories V2 file. The script checks the sha256 and fails, leaving nothing behind, on a mismatch. On success it prints only `curl`'s progress meter, so the check below shows the result.

```bash
scripts/fetch.sh \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt \
  6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d \
  data/raw/tinystories.txt
ls -l data/raw/tinystories.txt && sha256sum data/raw/tinystories.txt
```

```text
-rw-rw-r-- 1 thomas thomas 22502601 Oct 26 10:42 data/raw/tinystories.txt
6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d  data/raw/tinystories.txt
```

2. Clean and split it.

```bash
train_transformer prepare --input data/raw/tinystories.txt --out data/prepared \
    --dedup --nfkc --min-chars 50 --max-chars 20000
```

```text
prepared data/raw/tinystories.txt → data/prepared  (train=24823 val=1384 test=1423)  rejected empty=1
```

3. Train the reference config. The budget is 2 million tokens, so this takes about six minutes on the machine above.

```bash
time train_transformer train data/prepared/train.txt models/slm.gguf \
    --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 \
    --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1 \
    --val 0.02 --patience 3 --tokenizer models/tokenizer.gguf
```

```text
╔══════════════════════════════════════════════════════════╗
  ferrum transformer SLM trainer (int8 QAT)
  Corpus  : data/prepared/train.txt  (20190672 chars)
  Context : 128   Embed: 96   Hidden: 320
  Heads   : 4   Blocks: 3
  Tokenizer: byte-level BPE (vocab 1024)
  Epochs  : budget   LR: 0.005   Batch: 16   Seed: 1337
  Threads : 8 (data-parallel minibatch training)
  Budget  : 2000000 tokens (overrides --epochs)
╚══════════════════════════════════════════════════════════╝

  epoch     1   loss = 3.413080   val ppl = 18.690  *best

Trained in 344.05s.
  vocab = 1024 BPE tokens   context = 128   output_dim = 1024
Saved 620320 bytes → models/slm.gguf (int8 GGUF)
Reload check: OK (7 layers).

real	5m44.107s
user	24m26.231s
sys	1m11.317s
```

4. Inspect the model.

```bash
train_transformer info models/slm.gguf
```

```text
Model     : models/slm.gguf  (620320 bytes)
Format    : GGUF v3 (gpt2, int8)
Params    : 507552
Name      : GenerativeSLM Transformer
Task      : TransformerSLM
Input dim : 128
Output dim: 1024
Tokenizer : byte-level BPE (1024 tokens, 767 merges)
Layers    : 7
```

5. Generate a story.

```bash
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 80 --temp 0.7 --gen-seed 1
```

```text
Once upon a time, in the forest, there was a little boy named Tim. He lived in a small yard. Tim loved to play with his friends with his dad. One day, Tim said, "No, Tom! I am mean to help others?"
Tim said, "Yes, I want to go to play with the water."
Tim was very happy and played together. Spot was
```

**Sizing a config to a parameter target.** Parameter counts for three sizes
(and the first one without `--tie`). Each row was trained for one step
(`--tokens 1`) and read from `info`.

```bash
for cfg in "64 4 2 128 512 --tie" "96 4 3 320 1024" "192 6 4 768 2048"; do
  set -- $cfg
  train_transformer train data/prepared/train.txt samples/size.gguf \
      --context 128 --embed $1 --heads $2 --blocks $3 --hidden $4 --vocab $5 $6 \
      --tokens 1 --force >/dev/null
  train_transformer info samples/size.gguf | grep -i param
done
```

```text
Params    : 108032
Params    : 507552
Params    : 2590848
```

| Target | `--context` | `--embed` | `--heads` | `--blocks` | `--hidden` | `--vocab` | Parameters |
|---|---|---|---|---|---|---|---|
| about 100K (with `--tie`) | 128 | 64 | 4 | 2 | 128 | 512 | 108,032 (140,800 without `--tie`) |
| about 500K | 128 | 96 | 4 | 3 | 320 | 1024 | 507,552 |
| about 2.6M | 128 | 192 | 6 | 4 | 768 | 2048 | 2,590,848 |

The same 64-wide model with `--context 64` has 103,936 parameters tied (136,704
untied), because the position table shrinks by `64 × 64`.

Per block the parameters are `4·e² + 4·e + 2·e·h + h + e + 4·e` (attention,
feed-forward, and two LayerNorms; `e` = embed, `h` = hidden). Add the token
embedding (`vocab × e`), the position embedding (`context × e`), the final
LayerNorm (`2·e`), and the head (`e × vocab`, no bias, and none when `--tie` is
on). For the reference: `3 × 99,488 + 98,304 + 12,288 + 192 + 98,304 = 507,552`,
which is what `info` printed.

**What to look at.**

- The banner shows the token budget, not an epoch count. 2 million tokens is
  less than one pass over the 20 MB corpus, so one epoch line is printed. With
  `--val`, that line also shows the validation perplexity.
- `info` reports the parameter count and the quantization (int8) of a `gpt2`-architecture GGUF. It also reads a `.fckp` training checkpoint (this is `samples/run.fckp` from section 4, "Training controls"; that run took 977 optimizer steps) and an imported llama/qwen GGUF (architecture, layers, dims, vocab, quantization, context):

  ```text
  $ train_transformer info samples/run.fckp
  Model     : samples/run.fckp  (6097088 bytes)
  Format    : GGUF v3 (ferrum-checkpoint)
  Kind      : training checkpoint; continue it with train --resume
  Vocab     : 1024
  Context   : 128
  Embed     : 96
  Heads     : 4
  Hidden    : 320
  Blocks    : 3
  Step      : 977
  ```
- Training is deterministic. The model written by this run is byte-identical
  to the one the committed `dvc.lock` records (sha256
  `63249ca483b9c94a9dec49e8ee0674977d2a8941cfa067cff905b135c082892f`, the same
  value `ferrum-serve` reports in Sample 7).
- Keep `--embed`, `--hidden`, and `--vocab` multiples of 32 so every matrix can be stored as Q8_0.
- The file is `models/slm.gguf`, a `gpt2` GGUF. Sample 7 runs it in other tools.

---

## 2. The full DVC pipeline

**Goal.** Know what `dvc repro` runs, and check that a run on your machine matches the hashes the repository records in `dvc.lock`.

**You need.** Sample 1's prerequisites. To run DVC itself, also Python 3.10 or newer.

**Not run for this document.** DVC is not installed on the machine used for this documentation pass (`dvc: command not found`), so no `dvc` command below was run and no DVC output is shown. Step 4 was run: it checks the files from Sample 1 against `dvc.lock`, which was written by a real `dvc repro` (commit `b28aef8` regenerated it from a full repro).

**Steps.**

1. The stage graph, read from `dvc.yaml`. This is a drawing of the file, not the output of `dvc dag`.

```text
fetch -> prepare -> train -> eval
              \        \
               +--------+--> lighteval   (opt-in, eval/lighteval/dvc.yaml)
```

2. Each stage runs one of the commands from Sample 1, with its numbers taken from `params.yaml`:

| Stage | What it runs | Outputs |
|---|---|---|
| `fetch` | `scripts/fetch.sh` with `fetch.url` and `fetch.sha256` | `data/raw/tinystories.txt` |
| `prepare` | `train_transformer prepare --min-chars 50 --max-chars 20000 --val-frac 0.05 --test-frac 0.05 --seed 42 --dedup --nfkc` | `data/prepared/{train,val,test}.txt`, `manifest.json` |
| `train` | `train_transformer train` with the `model` and `train` sections of `params.yaml` and `--force` | `models/slm.gguf`, `models/tokenizer.gguf` |
| `eval` | `train_transformer eval ... --json` on `val.txt` and `test.txt` | `metrics/val.json`, `metrics/test.json` |

3. To run it with DVC (not run here), install the pinned version and reproduce:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dataops.txt
.venv/bin/dvc repro
.venv/bin/dvc metrics show
.venv/bin/dvc exp run -S train.tokens=4e6    # reruns only train and eval
.venv/bin/dvc status                         # reports nothing stale
```

4. Check the files from Sample 1 against `dvc.lock`. The `awk` line lists the `md5` that the lock file records for every `data/` and `models/` path:

```bash
awk '/- path:/{p=$3} /^ +md5:/{print $2"  "p}' dvc.lock | sort -u | grep -E "  (data|models)/" | md5sum -c
```

```text
data/prepared/train.txt: OK
data/prepared/manifest.json: OK
data/prepared/test.txt: OK
data/raw/tinystories.txt: OK
models/tokenizer.gguf: OK
models/slm.gguf: OK
data/prepared/val.txt: OK
```

5. The metrics files are committed (`cache: false`), so they are in the checkout:

```bash
cat metrics/val.json metrics/test.json
```

```text
{
  "bits_per_byte": 1.3135587291353807,
  "bits_per_token": 4.19795560836792,
  "bytes": 1129150,
  "cross_entropy_nats": 2.9098010063171387,
  "perplexity": 18.353145599365234,
  "seconds": 51.530224827,
  "tokens": 353316
}
{
  "bits_per_byte": 1.3049572422288735,
  "bits_per_token": 4.181272983551025,
  "bytes": 1154992,
  "cross_entropy_nats": 2.898237705230713,
  "perplexity": 18.14214515686035,
  "seconds": 52.60895892,
  "tokens": 360468
}
```

**What to look at.**

- All seven files match, including the model and tokenizer that Sample 1 trained. Training is deterministic, so the same inputs, flags, and `--threads` give the same bytes.
- `dvc.lock` records the hash of every input and output. It is committed; `data/` and `models/` are not, and there is no DVC remote.
- The `seconds` field in the metrics files is the time of the DVC run, so it differs from the time you measure. Everything else in them matches Sample 5.
- The data-parallel trainer splits each batch across `--threads` workers (auto = the core count), and a different worker count trains different weights. The pipeline therefore pins it (`train.threads: 8` in `params.yaml`, passed as `--threads 8`), so this lock file reproduces on any machine ([Manual §6.8](Manual.md#68-multi-threading-and-determinism)). This machine has 8 logical CPUs, which is why the pinned value equals what auto picked here.
- `eval/lighteval/dvc.yaml` holds the opt-in `lighteval` stage, kept out of the root pipeline so a plain `dvc repro` does not need the LightEval environment (Sample 8).

---

## 3. Cleaning your own corpus

**Goal.** Turn messy text files into clean train/val/test splits and read the manifest that records what happened.

**You need.** Sample 1's prerequisites.

**Steps.**

1. Build a messy input. It has a byte-order mark, Windows line endings, zero-width characters, the same document in two files (a duplicate), a two-book Gutenberg file, a too-short file, a too-long file, and a file with an invalid byte. One file is one document, so duplicates must be in separate files (or separated by `<|endoftext|>`).

```bash
mkdir -p samples/raw
printf '\xef\xbb\xbfThe sun rose over the quiet town and the baker opened his shop.\n' > samples/raw/bom.txt
printf 'The river ran down to the sea.\r\nIt carried leaves and small boats.\r\n' > samples/raw/crlf.txt
printf 'The mountain stood tall\xe2\x80\x8b and silent\xe2\x80\x8c in the cold morning air.\n' > samples/raw/zerowidth.txt
printf 'A fox came out of the wood at night.\n' > samples/raw/dup1.txt
cp samples/raw/dup1.txt samples/raw/dup2.txt
printf 'Hi.\n' > samples/raw/tiny.txt
yes 'and the story went on and on ' | head -n 20 | tr -d '\n' > samples/raw/long.txt
printf 'Preface text.\n*** START OF THE PROJECT GUTENBERG EBOOK ONE ***\nBook one is about a lighthouse keeper who counted the waves each night.\n*** END OF THE PROJECT GUTENBERG EBOOK ONE ***\nlicence text\n*** START OF THE PROJECT GUTENBERG EBOOK TWO ***\nBook two is about a girl who found a map inside an old clock.\n*** END OF THE PROJECT GUTENBERG EBOOK TWO ***\n' > samples/raw/books.txt
printf 'This line has a bad byte \xff in the middle of the sentence.\n' > samples/raw/badbyte.txt
```

2. Run `prepare` without `--lossy`. It stops and names the file and byte offset.

```bash
train_transformer prepare --input samples/raw --out samples/prepared
```

```text
error: samples/raw/badbyte.txt: invalid UTF-8 at byte 25 (pass --lossy to replace invalid bytes)
exit=1
```

3. Run it with the cleaning options.

```bash
train_transformer prepare --input samples/raw --out samples/prepared \
    --lossy --dedup --nfkc --strip-gutenberg --min-chars 20 --max-chars 300 --val-frac 0.2 --test-frac 0.2
cat samples/prepared/manifest.json
```

```text
prepared samples/raw → samples/prepared  (train=6 val=0 test=1)  rejected duplicate=1 too_long=1 too_short=1  lossy_repaired=1
```
```json
{
  "args": [
    "--input",
    "samples/raw",
    "--out",
    "samples/prepared",
    "--lossy",
    "--dedup",
    "--nfkc",
    "--strip-gutenberg",
    "--min-chars",
    "20",
    "--max-chars",
    "300",
    "--val-frac",
    "0.2",
    "--test-frac",
    "0.2"
  ],
  "ferrum_version": "0.3.0",
  "inputs": [
    {
      "bytes": 58,
      "path": "samples/raw/badbyte.txt",
      "sha256": "88b5859ab34a9afc93c1edae0a854d418591cae4926e720fd64f9027d7c360c3"
    },
    {
      "bytes": 67,
      "path": "samples/raw/bom.txt",
      "sha256": "9f9635bce0cf8da624b04cb91e5fa727fd30070e6e93b7258b4fa06a027dedd6"
    },
    {
      "bytes": 353,
      "path": "samples/raw/books.txt",
      "sha256": "38316bb99dff327c527789737484cefe91e753a7676724058c7b64c9455167af"
    },
    {
      "bytes": 68,
      "path": "samples/raw/crlf.txt",
      "sha256": "7e6d796691ebf470714c1117a88617fff75e55038ac6b8b986634fa0bd9d98c9"
    },
    {
      "bytes": 37,
      "path": "samples/raw/dup1.txt",
      "sha256": "a3818f72024a97f5fa7d3eef3fd8a3afa3c7c44c24cb52a3496a74377c66a8bd"
    },
    {
      "bytes": 37,
      "path": "samples/raw/dup2.txt",
      "sha256": "a3818f72024a97f5fa7d3eef3fd8a3afa3c7c44c24cb52a3496a74377c66a8bd"
    },
    {
      "bytes": 580,
      "path": "samples/raw/long.txt",
      "sha256": "52265600bdf99c66ac5922b46480f65a49314666a31a2c36d350e1dfa001e81b"
    },
    {
      "bytes": 4,
      "path": "samples/raw/tiny.txt",
      "sha256": "c7e13072a15d4bec0f88822de02bc825d5c6055ac8cf928913d301e336a28c1e"
    },
    {
      "bytes": 66,
      "path": "samples/raw/zerowidth.txt",
      "sha256": "0980ce36a79d4fc963ca523addb4dcae921dd4cc4cd29a6f83b9f2b6cc11a7d6"
    }
  ],
  "lossy_repaired": 1,
  "outputs": {
    "test": {
      "bytes": 63,
      "chars": 63,
      "documents": 1,
      "path": "samples/prepared/test.txt",
      "sha256": "f156ae94d92da9d09638622b3aa74c05ed30e7b5867a6c8dfa0253ea06d67021"
    },
    "train": {
      "bytes": 426,
      "chars": 349,
      "documents": 6,
      "path": "samples/prepared/train.txt",
      "sha256": "81c3afa34fecdc508975deed9fcaf40fc7e426474a2d44abf4f811a8abc43c5d"
    },
    "val": {
      "bytes": 0,
      "chars": 0,
      "documents": 0,
      "path": "samples/prepared/val.txt",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }
  },
  "rejected": {
    "duplicate": 1,
    "empty": 0,
    "invalid_utf8": 0,
    "too_long": 1,
    "too_short": 1
  },
  "train_stats": {
    "bytes": 426,
    "chars": 349,
    "lines": 7,
    "unique_chars": 31,
    "words": 73
  }
}
```

The splits that came out (the documents are separated by `<|endoftext|>`; the bad byte became U+FFFD):

```bash
for s in train val test; do echo "--- $s"; cat samples/prepared/$s.txt; done
```

```text
--- train
This line has a bad byte � in the middle of the sentence.
<|endoftext|>
Book one is about a lighthouse keeper who counted the waves each night.
<|endoftext|>
Book two is about a girl who found a map inside an old clock.
<|endoftext|>
The river ran down to the sea.
It carried leaves and small boats.
<|endoftext|>
A fox came out of the wood at night.
<|endoftext|>
The mountain stood tall and silent in the cold morning air.--- val
--- test
```

**What to look at.**

- `rejected` has one counter per reason: `empty`, `too_short` (under `--min-chars`), `too_long`, `duplicate`, and `invalid_utf8`. Here `tiny.txt` is too short, `long.txt` too long, and `dup2.txt` a duplicate. With `--lossy`, the document that held the bad byte is kept, not dropped (its line is in `train.txt` above), and it is counted under the top-level `lossy_repaired` (also shown in the summary line), not under `rejected`. `lossy_repaired` appears only when it is non-zero.
- The BOM, the zero-width characters, and the CRLF endings are gone from the kept documents, and the Gutenberg file became two documents without its preface or licence text.
- `inputs` and `outputs` list the sha256 of every file. If two manifests carry the same hashes, the runs are byte-identical.
- `train_stats` is `corpus_stats` of the train split.
- The split is by document and stable: rerun after appending a file and the old documents stay in their old splits. With only eight documents the validation split is empty here; real corpora have thousands.

---

## 4. Training controls

**Goal.** Use a token budget, cosine learning rate, validation with early stopping, and checkpoints; interrupt a run and resume it; see the strict CLI reject mistakes; train from a pre-tokenized token file.

**You need.** Sample 1's data.

**Steps.**

1. Train with a validation split, progress lines, and a checkpoint. `timeout -s INT 40` interrupts it after 40 seconds. The interrupt kills the process outright (it does not save on exit), so the checkpoint on disk is the last periodic one: `--checkpoint-every 50` writes one every 50 steps.

```bash
timeout -s INT 40 train_transformer train data/prepared/train.txt samples/run.gguf \
    --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 \
    --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1 \
    --val 0.02 --patience 3 --tokenizer samples/tok.gguf \
    --checkpoint samples/run.fckp --checkpoint-every 50 -v
```

```text
╔══════════════════════════════════════════════════════════╗
  ferrum transformer SLM trainer (int8 QAT)
  Corpus  : data/prepared/train.txt  (20190672 chars)
  Context : 128   Embed: 96   Hidden: 320
  Heads   : 4   Blocks: 3
  Tokenizer: byte-level BPE (vocab 1024)
  Epochs  : budget   LR: 0.005   Batch: 16   Seed: 1337
  Threads : 8 (data-parallel minibatch training)
  Budget  : 2000000 tokens (overrides --epochs)
╚══════════════════════════════════════════════════════════╝

[train] step 10/977  tokens/s 6658  elapsed 3s
[train] step 20/977  tokens/s 7130  elapsed 6s
[train] step 30/977  tokens/s 6934  elapsed 9s
[train] step 40/977  tokens/s 6567  elapsed 12s
[train] step 50/977  tokens/s 6415  elapsed 16s
[train] step 60/977  tokens/s 6350  elapsed 19s
[train] step 70/977  tokens/s 6251  elapsed 23s
[train] step 80/977  tokens/s 5840  elapsed 28s
[train] step 90/977  tokens/s 5452  elapsed 34s
exit=124
```

2. Resume with the same flags plus `--resume`. The run continues from the last checkpoint (step 50), keeps writing to it, and finishes the 2 million token budget.

```bash
train_transformer train data/prepared/train.txt samples/run.gguf \
    --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 \
    --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1 \
    --val 0.02 --patience 3 --tokenizer samples/tok.gguf \
    --checkpoint samples/run.fckp --checkpoint-every 50 -v --resume samples/run.fckp
```

```text
note: reusing samples/tok.gguf; if it was trained without --val it has seen the validation text
╔══════════════════════════════════════════════════════════╗
  ferrum transformer SLM trainer (int8 QAT)
  Corpus  : data/prepared/train.txt  (20190672 chars)
  Context : 128   Embed: 96   Hidden: 320
  Heads   : 4   Blocks: 3
  Tokenizer: byte-level BPE (vocab 1024)
  Epochs  : budget   LR: 0.005   Batch: 16   Seed: 1337
  Threads : 8 (data-parallel minibatch training)
  Budget  : 2000000 tokens (overrides --epochs)
  Resume  : samples/run.fckp
╚══════════════════════════════════════════════════════════╝

[train] step 60/977  tokens/s 3833  elapsed 5s
[train] step 70/977  tokens/s 4106  elapsed 10s
[train] step 80/977  tokens/s 4110  elapsed 15s
... (87 lines omitted) ...
[train] step 960/977  tokens/s 6309  elapsed 295s
[train] step 970/977  tokens/s 6313  elapsed 298s
  epoch     1   loss = 3.305283   val ppl = 18.910  *best

Trained in 351.33s.
  vocab = 1024 BPE tokens   context = 128   output_dim = 1024
Saved 620320 bytes → samples/run.gguf (int8 GGUF)
Reload check: OK (7 layers).
exit=0
```

3. Try two mistakes.

```bash
train_transformer train data/prepared/train.txt samples/x.gguf --contxt 8
train_transformer train data/prepared/train.txt samples/x.gguf --resume samples/does-not-exist.fckp
```

```text
error: unknown flag --contxt for 'train' (valid: --context --embed --heads --blocks --hidden --epochs --lr --batch --vocab --seed --threads --weight_decay --dropout --stride --tokens --cosine --warmup --clip --tie --no-qat --quant --debug --tokenizer --checkpoint --checkpoint-every --resume --val --patience --force --sample)
exit=1
error: checkpoint samples/does-not-exist.fckp does not exist (--resume needs an existing .fckp file)
exit=1
```

4. Train from a pre-tokenized file, for corpora larger than RAM ([Manual §6.11](Manual.md#611-corpora-larger-than-ram-tokenize-and-token-files)). A three-document corpus shows the equivalence first. The first `train` creates the tokenizer; `tokenize` streams the text into a `ferrum-tokens` GGUF holding the ids `train` would see; `train` on that file streams windows from disk. The two models must be the same bytes.

```bash
mkdir -p samples/tokens
cat > samples/tokens/corpus.txt <<'EOF'
the quick brown fox jumps over the lazy dog. the town wakes slowly each morning, and the baker opens his shop while the streets are still quiet.
<|endoftext|>
children walk to school past the old stone bridge, and the river runs down to the sea carrying leaves and small boats. by noon the market is full of voices, and the fox watches from the hill.
<|endoftext|>
in the evening the lamps are lit one by one, the shops close, and the streets grow quiet again. the quick brown fox goes home to sleep, and the town sleeps too.
EOF
FLAGS="--epochs 20 --context 12 --embed 32 --heads 4 --blocks 2 --hidden 64 --vocab 300 --seed 7 --stride 1 --threads 2 --tokenizer samples/tokens/tok.gguf"
train_transformer train samples/tokens/corpus.txt samples/tokens/text.gguf $FLAGS
train_transformer tokenize samples/tokens/corpus.txt --tokenizer samples/tokens/tok.gguf --out samples/tokens/corpus.tok.gguf
train_transformer info samples/tokens/corpus.tok.gguf
train_transformer train samples/tokens/corpus.tok.gguf samples/tokens/tokens.gguf $FLAGS
sha256sum samples/tokens/text.gguf samples/tokens/tokens.gguf
```

```text
$ train_transformer train samples/tokens/corpus.txt samples/tokens/text.gguf $FLAGS
╔══════════════════════════════════════════════════════════╗
  ferrum transformer SLM trainer (int8 QAT)
  Corpus  : samples/tokens/corpus.txt  (526 chars)
  Context : 12   Embed: 32   Hidden: 64
  Heads   : 4   Blocks: 2
  Tokenizer: byte-level BPE (vocab 300)
  Epochs  : 20   LR: 0.01   Batch: 16   Seed: 7
  Threads : 2 (data-parallel minibatch training)
╚══════════════════════════════════════════════════════════╝

  epoch     1/20   loss = 4.438987
  epoch     2/20   loss = 3.073753
...

Trained in 1.27s.
  vocab = 300 BPE tokens   context = 12   output_dim = 300
Saved 50264 bytes → samples/tokens/text.gguf (int8 GGUF)
Reload check: OK (6 layers).

$ train_transformer tokenize samples/tokens/corpus.txt --tokenizer samples/tokens/tok.gguf --out samples/tokens/corpus.tok.gguf
tokenized samples/tokens/corpus.txt → samples/tokens/corpus.tok.gguf: 279 tokens, 3 documents, vocab 300 (526 bytes in, 1006 bytes out, 0.0s)

$ train_transformer info samples/tokens/corpus.tok.gguf
Model     : samples/tokens/corpus.tok.gguf  (1006 bytes)
Format    : GGUF v3 (ferrum-tokens)
Kind      : pre-tokenized corpus; train on it with train samples/tokens/corpus.tok.gguf ...
Tokens    : 279
Documents : 3
Vocab     : 300
EOS id    : 299
Tokenizer : sha256 75c0facf5f33572558831c074f6271e307ed808f0cf9249f6dc7d69e28fc574c

$ train_transformer train samples/tokens/corpus.tok.gguf samples/tokens/tokens.gguf $FLAGS
╔══════════════════════════════════════════════════════════╗
  ferrum transformer SLM trainer (int8 QAT)
  Corpus  : samples/tokens/corpus.tok.gguf  (279 tokens, 3 documents, streamed from disk)
  Context : 12   Embed: 32   Hidden: 64
  Heads   : 4   Blocks: 2
  Tokenizer: byte-level BPE (vocab 300)
  Epochs  : 20   LR: 0.01   Batch: 16   Seed: 7
  Threads : 2 (data-parallel minibatch training)
╚══════════════════════════════════════════════════════════╝

  epoch     1/20   loss = 4.438987
  epoch     2/20   loss = 3.073753
...

Trained in 1.24s.
  vocab = 300 BPE tokens   context = 12   output_dim = 300
Saved 50264 bytes → samples/tokens/tokens.gguf (int8 GGUF)
Reload check: OK (6 layers).

$ sha256sum samples/tokens/text.gguf samples/tokens/tokens.gguf
4c58479b2c5cbfa9c5e4ae220622035b5abfad97b2a2dcf2b60dc247106c6818  samples/tokens/text.gguf
4c58479b2c5cbfa9c5e4ae220622035b5abfad97b2a2dcf2b60dc247106c6818  samples/tokens/tokens.gguf
```

Then the same on real data: the 20 MB TinyStories train split from Sample 1 and the DVC pipeline's tokenizer (`models/tokenizer.gguf`, md5 `c2ebc991dc85594541ad9852e92db99b`, only read), with the DVC model shape and a 3e5-token budget. `/usr/bin/time` reports wall time and peak resident memory (the `grep` keeps only those lines; `--force` retrains over the existing file).

```bash
ARCH="--context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 --lr 0.005 --warmup 50 --clip 1 --seed 1337 --threads 8 --tokens 3e5 --tokenizer models/tokenizer.gguf"
mkdir -p samples/big
/usr/bin/time -f "tokenize: %es wall, peak RSS %M KB" train_transformer tokenize data/prepared/train.txt --tokenizer models/tokenizer.gguf --out samples/big/train.tok.gguf 2>&1 | grep -E "^(tokenized|tokenize:|train from)"
/usr/bin/time -f "train from text: %es wall, peak RSS %M KB" train_transformer train data/prepared/train.txt samples/big/text.gguf $ARCH --force 2>&1 | grep -E "^(tokenized|tokenize:|train from)"
/usr/bin/time -f "train from token file: %es wall, peak RSS %M KB" train_transformer train samples/big/train.tok.gguf samples/big/tok.gguf $ARCH --force 2>&1 | grep -E "^(tokenized|tokenize:|train from)"
sha256sum samples/big/text.gguf samples/big/tok.gguf
```

```text
tokenized data/prepared/train.txt → samples/big/train.tok.gguf: 6306484 tokens, 24823 documents, vocab 1024 (20190677 bytes in, 12613416 bytes out, 1.5s)
tokenize: 1.46s wall, peak RSS 7232 KB
train from text: 33.46s wall, peak RSS 178760 KB
train from token file: 36.60s wall, peak RSS 134040 KB
04a43505dbde369f125fd67bf81f1c9b56744a406c553117576fba7df93fee64  samples/big/text.gguf
04a43505dbde369f125fd67bf81f1c9b56744a406c553117576fba7df93fee64  samples/big/tok.gguf
```

**What to look at.**

- The step counter after `--resume` continues from the checkpoint (it restarts at step 50 plus the first progress interval). The output above was recorded with a build that restarted a partial epoch with a fresh shuffle: the uninterrupted run in Sample 1 ended at validation perplexity 18.690 and this resumed run at 18.910. Since then a checkpoint records where its epoch began (`ferrum.ckpt.epoch_rng`) and a resume replays the rest of that epoch, so a mid-epoch resume is bit-identical to an uninterrupted run in the tests (`slm_cli/tests/tiny_e2e.rs` compares the sha256 of both models); this sample was not re-run with that build, so its numbers still show the old behaviour.
- With `--val`, the best validation score is restored on resume.
- `-v` prints one line per 10 steps. `--debug` is the old per-op trace.
- The CLI never falls back to a default silently. The second error names `.fckp` because `train` checkpoints are `.fckp` (`run-gguf`, `export-gguf` and `finetune-gguf` say `.flck`).
- Step 4: the token file is 1006 bytes for 279 tokens (a 300-token vocabulary uses 2 bytes per id, plus the header), and the model trained from it has the same sha256 as the model trained from the text. On the 20 MB split the two models are again identical. The token-file run needs about 45 MB less resident memory (the 20 MB of text and the 25 MB of ids the in-memory path holds), and `tokenize` needs 7 MB for the whole 20 MB. The 133 MB left is the model's activations, which do not depend on the corpus. Wall time showed no cost from streaming: run in the order text then token file it read 33.46 s and 36.60 s, and run twice in the reverse order (token file first) the token file took 34.36 s and 36.36 s and the text 38.02 s and 37.93 s, so the spread is run order and machine noise, not the source. The 20 MB corpus fits in RAM here; the point of the token file is the corpus that does not, which this machine cannot demonstrate without a multi-gigabyte download, so that case is untested at scale (the memory use is O(batch), asserted in the tests).
- The first line of the resume output is a reminder that a reused tokenizer file may have seen the validation text. The run used `--tokenizer samples/tok.gguf`, created by the first command with `--val`, so it did not.

---

## 5. Evaluation

**Goal.** Measure perplexity and bits per byte, write them as JSON, and compare f32, int8, and int4 files.

**You need.** Samples 1 and 3 done; `data/prepared/val.txt` and `test.txt` exist.

**Steps.**

1. Evaluate the Sample 1 model on both splits. Scoring 360,000 tokens takes about a minute.

```bash
mkdir -p metrics
train_transformer eval models/slm.gguf data/prepared/val.txt  --json metrics/val.json
train_transformer eval models/slm.gguf data/prepared/test.txt --json metrics/test.json
cat metrics/test.json
```

```text
Model        : models/slm.gguf
Held-out text: data/prepared/val.txt  (1129150 chars)
Predictions  : 353316
Cross-entropy: 2.9098 nats/token
Bits/token   : 4.1980
Perplexity   : 18.3531
Bits/byte    : 1.3136
Seconds      : 63.84
Model        : models/slm.gguf
Held-out text: data/prepared/test.txt  (1154992 chars)
Predictions  : 360468
Cross-entropy: 2.8982 nats/token
Bits/token   : 4.1813
Perplexity   : 18.1421
Bits/byte    : 1.3050
Seconds      : 67.16
{
  "bits_per_byte": 1.3049572422288735,
  "bits_per_token": 4.181272983551025,
  "bytes": 1154992,
  "cross_entropy_nats": 2.898237705230713,
  "perplexity": 18.14214515686035,
  "seconds": 67.162741499,
  "tokens": 360468
}
```

2. Train the same config at three precisions. Each run takes about six minutes.

```bash
for q in f32 int8 int4; do
  train_transformer train data/prepared/train.txt samples/slm-$q.gguf \
      --context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024 \
      --tokens 2e6 --lr 5e-3 --cosine --warmup 50 --clip 1 --tokenizer samples/tok-$q.gguf \
      --quant $q --force
  train_transformer eval samples/slm-$q.gguf data/prepared/test.txt --json samples/test-$q.json
done
ls -l samples/slm-*.gguf
```

The training banners are the same as in Sample 1. The numbers that matter:

```text
Trained in 345.63s.
Saved 2064160 bytes → samples/slm-f32.gguf (f32 GGUF)
Model        : samples/slm-f32.gguf
Held-out text: data/prepared/test.txt  (1154992 chars)
Predictions  : 360498
Cross-entropy: 2.9217 nats/token
Bits/token   : 4.2152
Perplexity   : 18.5738
Bits/byte    : 1.3157
Seconds      : 70.03
Trained in 311.39s.
Saved 620320 bytes → samples/slm-int8.gguf (int8 GGUF)
Model        : samples/slm-int8.gguf
Held-out text: data/prepared/test.txt  (1154992 chars)
Predictions  : 360498
Cross-entropy: 2.9217 nats/token
Bits/token   : 4.2152
Perplexity   : 18.5738
Bits/byte    : 1.3157
Seconds      : 70.86
Trained in 372.65s.
Saved 374560 bytes → samples/slm-int4.gguf (int4 GGUF)
Model        : samples/slm-int4.gguf
Held-out text: data/prepared/test.txt  (1154992 chars)
Predictions  : 360498
Cross-entropy: 2.9409 nats/token
Bits/token   : 4.2428
Perplexity   : 18.9325
Bits/byte    : 1.3243
Seconds      : 72.37
-rw-rw-r-- 1 thomas thomas 2064160 Oct 26 11:10 samples/slm-f32.gguf
-rw-rw-r-- 1 thomas thomas  374560 Oct 26 11:24 samples/slm-int4.gguf
-rw-rw-r-- 1 thomas thomas  620320 Oct 26 11:16 samples/slm-int8.gguf
```

| Precision | File size (bytes) | Perplexity | Bits per byte |
|---|---:|---:|---:|
| f32 | 2,064,160 | 18.5738 | 1.3157 |
| int8 (Q8_0) | 620,320 | 18.5738 | 1.3157 |
| int4 (Q4_0) | 374,560 | 18.9325 | 1.3243 |

The int8 file is 30% of the f32 size and the int4 file 18%. They are not exactly a quarter and an eighth because the tokenizer, the metadata, the 1-D tensors (biases and LayerNorm), and the position embedding stay in f32. int4 raises the perplexity by 1.9%.

**What to look at.**

- Every token is scored, the first conditioned on BOS (the EOS id), so `Predictions` equals the token count and `tokens` in the JSON is the token count of the file.
- **Perplexity** is per token. **Bits per byte** divides the total loss by the file's bytes, so it compares models with different tokenizers. Per-token numbers do not.
- QAT is int8 only, and the model that gets saved is the one training saw: its weights snapped to the Q8_0 grid. So the f32 file holds the same snapped values as 4-byte floats, and the f32 and int8 files score **identically** (int8 loses nothing, at 30% of the size). int4 is converted after training and is lossy. (`--no-qat` trains in plain f32 and saves unsnapped weights; then int8 is a lossy conversion too.)
- The three runs train their own tokenizers (no `--val`), so their numbers differ slightly from Sample 1's, which trained its tokenizer on the train portion only.
- Matrices whose row length is not a multiple of 32 are not quantized (they stay F32, exactly). That is why the reference config uses multiples of 32.

---

## 6. Generation

**Goal.** Use the sampling controls, stop sequences, the end-of-text token, and a fixed seed.

**You need.** `models/slm.gguf` from Sample 1.

**Steps.**

1. Greedy output (`--temp 0` is exact argmax), then sampled output.

```bash
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 60 --temp 0
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 60 \
    --temp 0.8 --top-k 40 --top-p 0.9 --repetition-penalty 1.2 --gen-seed 7
```

```text
Once upon a time, there was a little girl named Lily. She loved to play with her friends. One day, she found a big, a big, red ball. The ball was very happy.
The ball was very happy. She said, "In't know what to do. I have to play
```

```text
Once upon a time, there was a little boy named Tim. Tim loved to play with his toys. One day, Tim was a big box. He wanted to build a funny day, came todge toy car inside.
Tim looked around and saw the ground. The ball was very sad. Tim
```

2. Stop sequences. `--stop` is repeatable. The output ends before the first match.

```bash
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 200 --temp 0.7 --gen-seed 3 \
    --stop "." --stop "!"
```

```text
Once upon a time, there was a little girl named Lily
```

3. End of text. The model emits `<|endoftext|>` at the end of a story, and generation stops there. The CLI does not print why it stopped; the story below ends well short of the 400-token budget.

```bash
train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 400 --temp 0.7 --gen-seed 5
```

```text
Once upon a time, there was a big dog named Tom. Spot had a big spot all day long rock.
One day, while he went for a new friend, Jane. Tim and his mom went to the park, and Sam were very picking. They were sad, and not like to play with his friends.
As they put the red ball in the ball down. They flew down and started to get the ball. They could not find cream. The ball was happy. It was so happy! The ball was very happy and had a fun day. Tim was happy to have fun.
```

4. The same seed gives the same text at any thread count.

```bash
FERRUM_NUM_THREADS=1 train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 60 --temp 0.8 --gen-seed 9 | md5sum
FERRUM_NUM_THREADS=8 train_transformer generate models/slm.gguf "Once upon a time" --max-tokens 60 --temp 0.8 --gen-seed 9 | md5sum
```

```text
3053024777f0a8d9cb9768d41c737eb9  -
3053024777f0a8d9cb9768d41c737eb9  -
```

**What to look at.**

- `--temp 0` always gives the same text and matches llama.cpp at `--temp 0` (Sample 7).
- `--max-tokens` is a token budget. `--chars` is a character count. Either, or `--stop`, ignores the other.
- Generation past the context window works: the cache is rebuilt from the last half window once per half window of new tokens.
- Identical hashes show that thread count does not change results.
- This model is 500K parameters trained for six minutes. It writes fluent-looking sentences that do not hold together (the sampled text above wanders). That is expected at this size ([Manual §17](Manual.md#17-limits-an-honest-critique)).

---

## 7. Serving, and running your model in llama.cpp, Ollama, and LM Studio

**Goal.** Run the model from three other tools, check they agree with Ferrum, then serve it over HTTP and call every route.

**You need.** `models/slm.gguf` from Sample 1. The tools are optional; skip any you do not have. The commands were verified with llama.cpp b11521, Ollama v0.40.2, and LM Studio 0.4.18. In this run llama.cpp and Ollama were executed; **LM Studio was not** (see Part D).

Set the tool locations (the defaults of a standard install):

```bash
LLAMA="$HOME/.local/opt/llama.cpp-b11521"
OLLAMA="$HOME/.local/bin/ollama"
LMS="$HOME/.lmstudio/bin/lms"
PROMPT="Once upon a time"
```

**Part A. Ferrum's own greedy output (the reference).**

```bash
train_transformer generate models/slm.gguf "$PROMPT" --max-tokens 16 --temp 0
```

```text
Once upon a time, there was a little girl named Lily. She loved to play with her friends
```

**Part B. llama.cpp.**

```bash
"$LLAMA/llama-completion" -m models/slm.gguf -p "$PROMPT" -n 16 --temp 0 -no-cnv --no-warmup </dev/null 2>/dev/null
"$LLAMA/llama-tokenize" -m models/slm.gguf -p "$PROMPT" --ids --log-disable
```

```text
Once upon a time, there was a little girl named Lily. She loved to play with her friends


[424, 434, 258, 396]
```

The text is the same as Part A.

**Part C. Ollama.** Run a private server on its own port and model directory.

```bash
export OLLAMA_HOST=127.0.0.1:11545 OLLAMA_MODELS="$PWD/samples/ollama-models"
"$OLLAMA" serve >samples/ollama.log 2>&1 &
OLLAMA_PID=$!
sleep 3
printf 'FROM "%s"\nTEMPLATE "{{ .Prompt }}"\nPARAMETER num_predict 16\n' "$PWD/models/slm.gguf" > samples/Modelfile
"$OLLAMA" create ferrum-tiny -f samples/Modelfile
curl -s http://127.0.0.1:11545/api/generate \
  -d '{"model":"ferrum-tiny","prompt":"Once upon a time","raw":true,"stream":false,"options":{"temperature":0,"num_predict":16}}'
```

```text
FROM "/home/thomas/Projects/ferrum/ferrum-cpu/models/slm.gguf"
TEMPLATE "{{ .Prompt }}"
PARAMETER num_predict 16
gathering model components gathering model components gathering model components 
copying file sha256:63249ca483b9c94a9dec49e8ee0674977d2a8941cfa067cff905b135c082892f 100% 
parsing GGUF 
using existing layer sha256:63249ca483b9c94a9dec49e8ee0674977d2a8941cfa067cff905b135c082892f 
creating new layer sha256:b507b9c2f6ca642bffcd06665ea7c91f235fd32daeefdf875a0f938db05fb315 
creating new layer sha256:fd7db000477862386258ffa820f1ac8e1ada4ab5330031bb0d67527bc192b0e7 
writing manifest 
success 
{"model":"ferrum-tiny","created_at":"2026-10-26T05:37:52.694940613Z","response":", there was a little girl named Lily. She loved to play with her friends","done":true,"done_reason":"length","total_duration":1439947173,"load_duration":1396701646,"prompt_eval_count":4,"prompt_eval_cached_count":0,"prompt_eval_duration":1088000,"eval_count":16,"eval_duration":40093000}
```

The `response` field is Part A's continuation. (Ollama's progress bar writes terminal control codes; they are removed from the block.)

Leave the Ollama server running for Part E. It is stopped there.

**Part D. LM Studio.** *Not run in this pass.* The LM Studio install on the documentation machine is a personal one, and `lms import` writes into it, so the commands were not executed here and no output is pasted. They were verified when they were pinned (LM Studio 0.4.18, identical 16-token greedy output to llama.cpp), and `scripts/verify_runtimes.sh` re-checks them against a real install.

```bash
"$LMS" import -y -l --user-repo ferrum-test/slm models/slm.gguf
KEY=$("$LMS" ls --json | python3 -I -c 'import json,sys; print([m["modelKey"] for m in json.load(sys.stdin) if "ferrum-test" in m.get("path","")][0])')
"$LMS" load "$KEY" --gpu off --identifier ferrum-tiny -y
"$LMS" server start -p 1234
curl -s http://127.0.0.1:1234/v1/completions \
  -d '{"model":"ferrum-tiny","prompt":"Once upon a time","max_tokens":17,"temperature":0,"repeat_penalty":1.0}'
"$LMS" server stop
"$LMS" unload ferrum-tiny
```

`max_tokens` 17 shows 16 visible tokens. `repeat_penalty` 1.0 matters because LM Studio's default (1.1) changes the text. `lms ls --json` entries carry `modelKey` and `path`; that is what the `python3` line reads, and what `export-lmstudio` reads.

**Part E. One-command exports.** Both commands accept only Ferrum's own `gpt2` models: they refuse a GGUF that llama.cpp cannot run (a character-level or tabular model), and an imported llama GGUF (write its Modelfile by hand, Manual §9.5), before they start any tool. They find the tools as described in [Manual §9.5](Manual.md#95-running-ferrum-models-in-llamacpp-ollama-and-lm-studio), or take `--bin`. The Ollama export talks to the server from Part C, because `OLLAMA_HOST` is still exported.

```bash
train_transformer export-ollama models/slm.gguf ferrum-export --modelfile-only samples/ferrum-export.Modelfile
cat samples/ferrum-export.Modelfile
train_transformer export-ollama models/slm.gguf ferrum-export
"$OLLAMA" run --nowordwrap ferrum-export "Once upon a time"
kill $OLLAMA_PID
```

```text
wrote samples/ferrum-export.Modelfile
FROM "/home/thomas/Projects/ferrum/ferrum-cpu/models/slm.gguf"
TEMPLATE "{{ .Prompt }}"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.8
PARAMETER num_ctx 128
exported to Ollama as 'ferrum-export'  (try: ollama run ferrum-export)
```

`ollama run` has no token limit here (the Modelfile sets none), so the model writes until it emits end-of-text. With `--nowordwrap` the text is printed as it streams (the second run, after restarting the server on the same model directory, is shown):

```text
NAME                    ID              SIZE      MODIFIED           
ferrum-export:latest    9c12cb510eb2    620 KB    About a minute ago    
ferrum-tiny:latest      cb3935332b54    620 KB    About a minute ago    
⠙ ⠙ ⠸ ⠸ ⠼ ⠦ ⠧ ⠧ ⠏ ⠋ ⠙ ⠙ ⠹ , there was a bright. It would fix the tiny blue car.
One day, they were playing, and the park to play with the tir and befish. They played with the ball and had a lot of fun. They played together every day.
Sue was very happy and laughed. They thanked the ball for the ball and had a lot. They played on the ball and played with the ball together. They played together, and had fun together every day.
```

`export-lmstudio` was not run (see Part D). Its refusal path does not touch LM Studio, so it can be shown. Exporting a character-level model (`train ... --vocab 0`) fails for both commands before any tool is used:

```bash
train_transformer train data/prepared/val.txt samples/char.gguf --vocab 0 --context 16 --embed 32 \
    --heads 4 --blocks 1 --hidden 64 --epochs 1 --force
train_transformer export-ollama samples/char.gguf ferrum-char
train_transformer export-lmstudio samples/char.gguf
```

```text
error: /home/thomas/Projects/ferrum/ferrum-cpu/samples/char.gguf: architecture 'ferrum' is not runnable in llama.cpp (Ollama / LM Studio run only 'gpt2' GGUF files; character-level, generic and tabular Ferrum models have no llama.cpp architecture). To run a Ferrum SLM there, retrain with a BPE vocabulary (`train ... --vocab 1024`; `--vocab 0` is character-level) to get a 'gpt2' GGUF
error: /home/thomas/Projects/ferrum/ferrum-cpu/samples/char.gguf: architecture 'ferrum' is not runnable in llama.cpp (Ollama / LM Studio run only 'gpt2' GGUF files; character-level, generic and tabular Ferrum models have no llama.cpp architecture). To run a Ferrum SLM there, retrain with a BPE vocabulary (`train ... --vocab 1024`; `--vocab 0` is character-level) to get a 'gpt2' GGUF
```

**Part F. Serve over HTTP.**

```bash
ferrum-serve --model models/slm.gguf &
sleep 2
```

Every route. POST routes need `Content-Type: application/json`; `curl -d` alone sends a form content type and gets a `400`.

```bash
J='Content-Type: application/json'
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/info
curl -s -H "$J" http://127.0.0.1:8080/v1/loglikelihood -d '{"requests":[{"context":"The cat sat on the","continuation":" mat"},{"context":"The cat sat on the","continuation":" moon"}]}'
curl -s -H "$J" http://127.0.0.1:8080/v1/loglikelihood_rolling -d '{"texts":["Tom had a red ball. He liked to play with it."]}'
curl -s -H "$J" http://127.0.0.1:8080/v1/generate -d '{"requests":[{"prompt":"Once upon a time","max_tokens":40,"stop":["."],"temperature":0}]}'
```

```text
"ok"
{"params":507552,"context_len":128,"vocab":1024,"quant":"int8","tokenizer":"bpe","model_sha256":"63249ca483b9c94a9dec49e8ee0674977d2a8941cfa067cff905b135c082892f"}
{"results":[{"greedy":false,"logprob":-7.325810936843296,"n_tokens":2},{"greedy":false,"logprob":-6.774459765918736,"n_tokens":2}]}
{"results":[{"logprob":-28.802501869155233,"n_tokens":13}]}
{"results":[{"n_tokens":9,"text":", there was a little girl named Lily"}]}
```

`params` equals the count `info` printed in Sample 1, and `model_sha256` equals `sha256sum models/slm.gguf` (`63249ca4...`, shown in Sample 1's notes).

An oversized batch (65 items, over the default limit of 64) is rejected before any compute:

```bash
python3 -I -c 'import json; print(json.dumps({"requests":[{"context":"a","continuation":"b"}]*65}))' \
  | curl -s -i -H "$J" http://127.0.0.1:8080/v1/loglikelihood -d @-
kill $!
```

```text
HTTP/1.1 400 Bad Request
content-type: application/json
content-length: 44
date: Mon, 26 Oct 2026 05:39:52 GMT

{"error":"batch of 65 exceeds max_batch 64"}
```

**What to look at.**

- Greedy outputs from Ferrum and llama.cpp should be the same text for an f32 file (the test `llama_cpp_greedy_matches_ferrum_f32` checks this). For int8 or int4 files, tiny numeric differences in llama.cpp can change tokens late in a run. Here the int8 file matched over the 16 tokens shown, in llama.cpp and in Ollama.
- `llama-tokenize` ids equal Ferrum's ids for the same text, because both use the GPT-2 pre-tokenizer.
- LightEval and your own scripts use `/v1/loglikelihood`. A higher `logprob` is a more likely continuation. `greedy` is true when every token was the model's top choice.
- The server binds to `127.0.0.1` by default and has no authentication. Limits (body 8 MiB, batch 64, 65,536 characters per text, 4,096 tokens per generate request) are checked first: 413 for a large body, 400 for the others, 422 for a model error.
- The server handles one request at a time, on one dedicated model thread (the model is not `Send` or `Sync`). Each matmul still uses every core.

---

## 8. Testing with LightEval

**Goal.** Score the model with Hugging Face LightEval and read the results.

**You need.** Python 3.10 or newer. `models/slm.gguf` and `data/prepared/test.txt` from Sample 1. Disk space for CPU PyTorch: not measured here, because the environment was not rebuilt for this document.

**Steps.**

1. Install the pinned versions in their own environment.

```bash
python3 -m venv eval/lighteval/.venv
eval/lighteval/.venv/bin/pip install -r eval/lighteval/requirements.txt
```

`requirements.txt` pins `lighteval==0.13.0`, `xxhash<4` (xxhash 4.x crashes the details logger), and `datasets<5`, and uses the CPU PyTorch index.

2. Run it. `run.sh` starts `ferrum-serve`, waits for `/health`, runs LightEval, and stops the server.

```bash
eval/lighteval/run.sh --model models/slm.gguf --test data/prepared/test.txt \
    --out metrics/lighteval.json --max-samples 200
cat metrics/lighteval.json
```

**Not re-run for this document.** The environment above was not built on the machine used for this documentation pass, so the command above was not run here. Two real results exist from earlier runs on this same machine (2026-10-25), and both are reproduced below.

The first is a 20-document smoke run (`--max-samples 20`). Its raw LightEval results file is still on disk under `eval/lighteval/out/results/` (a later `run.sh` moves them to `out/results.prev/` and keeps them for one more run), and `summarize.py` (the last step of `run.sh`) turns it into `metrics/lighteval.json`. This is that step, run now on that file:

```text
$ python3 eval/lighteval/summarize.py eval/lighteval/out/results metrics/lighteval.json models/slm.gguf 20
{
  "arc:easy|0": {
    "acc": 0.25,
    "acc_stderr": 0.09933992677987828
  },
  "blimp:_average|0": {
    "acc": 0.6,
    "acc_stderr": 0.10667735554134312
  },
  "blimp:anaphor_gender_agreement|0": {
    "acc": 0.8,
    "acc_stderr": 0.09176629354822471
  },
  "blimp:anaphor_number_agreement|0": {
    "acc": 0.55,
    "acc_stderr": 0.11413288653790232
  },
  "blimp:regular_plural_subject_verb_agreement_1|0": {
    "acc": 0.45,
    "acc_stderr": 0.11413288653790232
  },
  "ferrum_heldout|0": {
    "bits_per_byte": 1.3590188681805733,
    "bits_per_byte_stderr": 0.0015930526727733984,
    "byte_perplexity": 2.5651067528643994,
    "byte_perplexity_stderr": 0.0028179954733378617,
    "word_perplexity": 110.84596989946888,
    "word_perplexity_stderr": 0.660634398130358
  },
  "tinystories_cloze|0": {
    "acc": 0.3,
    "acc_stderr": 0.10513149660756935
  }
}
```

The second is the full run (no `--max-samples`, all 1,423 test documents), recorded in the work log when it finished on 2026-10-25 (the log is not part of the repository) and kept in [Manual §9.8](Manual.md#98-evaluating-with-lighteval). Its raw output was not kept, so only the summary numbers survive:

| Task | Metrics | Value |
|---|---|---|
| `ferrum_heldout` | word perplexity, byte perplexity, bits per byte | 103.59, 2.531, 1.3397 |
| `tinystories_cloze` | `acc` | 0.2776 |
| `blimp:anaphor_gender_agreement` | `acc` | 0.807 |
| `blimp:anaphor_number_agreement` | `acc` | 0.521 |
| `blimp:regular_plural_subject_verb_agreement_1` | `acc` | 0.669 |
| `arc:easy` | `acc` | 0.2508 |

The full run took about five minutes. The smoke run is small enough that its accuracies move by several points (the standard errors in the file are around 0.1). The held-out bits per byte (1.3397 full, against 1.3050 from `eval` on the whole test file in Sample 5) differ because LightEval scores the 1,423 documents one at a time, each starting from the start-of-text token, while `eval` scores the file as one text.

3. Or run it through DVC. The stage lives in `eval/lighteval/dvc.yaml`, outside the root pipeline, so a plain `dvc repro` never needs the LightEval environment.

```bash
.venv/bin/dvc repro eval/lighteval/dvc.yaml:lighteval
```

Not run here (DVC is not installed on this machine). The stage runs the same `run.sh` command as step 2 and writes `metrics/lighteval.json`; its dependencies are `models/slm.gguf`, `data/prepared/test.txt`, and the `eval/lighteval` folder.

**What to look at.**

- Result keys in the JSON are `<task>|0`, for example `tinystories_cloze|0`.
- `ferrum_heldout` is rolling log-likelihood of whole test documents.
- `tinystories_cloze` gives the model a story prefix and four sentences: the real next sentence and three from other stories (fixed seed). It must choose the real one. Chance is 25%.
- `blimp` compares a grammatical sentence with a minimally different ungrammatical one. Chance is 50%.
- `arc:easy` is a science question set. **A 500K model scores near chance (25%). That is expected.** Do not read it as a failure.
- Two LightEval 0.13.0 workarounds are built in: `FerrumModel` does not call `super().__init__()` (it raises in 0.13.0), and rolling log-likelihood is not cached (0.13.0's perplexity cache drops results).
- LightEval's litellm and TGI backends cannot run log-likelihood tasks, which is why Ferrum is a custom model class.

---

## 9. Benchmarking

**Goal.** Measure training, evaluation, and generation speed, thread scaling, and the before/after of this release.

**You need.** Sample 1's data. Every table below states the machine and date it was measured on.

**Steps.**

1. The end-to-end benchmark script, at auto threads and at one thread. It builds the CLI, trains the reference config on the first 300 KB of the corpus for 200,000 tokens, scores the last 30 KB, and times 4,096 tokens of generation (the generation time includes process start and model load).

```bash
scripts/bench_slm.sh data/prepared/train.txt
FERRUM_NUM_THREADS=1 scripts/bench_slm.sh data/prepared/train.txt
```

One run of each command:

```text
== threads: auto
train: 6468 tokens/s (30.9 s)
Predictions  : 9468
Perplexity   : 74.2377
eval: 2.26 s
generate: 0.347 ms/token (4096 max tokens, 13068 bytes out)
== threads: 1
train: 2659 tokens/s (75.2 s)
Predictions  : 9468
Perplexity   : 81.6963
eval: 2.55 s
generate: 0.312 ms/token (4096 max tokens, 12876 bytes out)
```

The first block is `auto` threads (8 here) and the second is one thread. `train` is the 200,000-token budget divided by the wall time of the whole `train` command, so it includes the BPE tokenizer training and the model write.

2. The matmul microbenchmark.

```bash
cargo bench --bench gemm
```

This prints the square-GEMM rates, the f32 decode GEMV rates, a synthesized f32 1B-class decode step, and the same for int8 and int4. Machine and date: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 4 cores / 8 threads, 2026-10-26. The banner line is omitted; it says the benchmark program is std-only (no Criterion). This is the third of three runs at auto threads; the table takes the median of three at 8 threads and three at one thread (`FERRUM_NUM_THREADS=1 cargo bench --bench gemm`):

```text
threads = 8   (FERRUM_NUM_THREADS = unset)

== Square GEMM   C[m×n] = A[m×k]·B[k×n]   (compute-bound; m≥2 ⇒ parallel) ==
  [  256×256  ]·[  256×256  ]      0.473 ms     70.9 GFLOP/s   (1306 iters)
  [  512×512  ]·[  512×512  ]      5.223 ms     51.4 GFLOP/s   (159 iters)
  [ 1024×1024 ]·[ 1024×1024 ]     51.083 ms     42.0 GFLOP/s   (21 iters)
  [ 2048×2048 ]·[ 2048×2048 ]    426.034 ms     40.3 GFLOP/s   (3 iters)

== Decode GEMV   c[1×n] = a[1×k]·W[k×n]   (m=1 ⇒ SERIAL on one core; bandwidth-bound) ==
  attn q/k/v/o proj    W[ 2048×2048 ]     2101.8 µs/call     8.0 GB/s   (292 iters)
  ffn up               W[ 2048×8192 ]    10620.8 µs/call     6.3 GB/s   (76 iters)
  ffn down             W[ 8192×2048 ]    13021.4 µs/call     5.2 GB/s   (64 iters)
  output logits        W[ 2048×32000]    59830.1 µs/call     4.4 GB/s   (18 iters)

== Synthesized decode step   d_model=2048, d_ff=8192, layers=16, vocab=32000 ==
     874.4 ms/token     1.14 tok/s      4.0 GB/s effective   (3 iters)
  weights streamed/token ≈ 3.48 GB  (this is the f32 footprint; int8 ⇒ ¼, int4 ⇒ ⅛)

  NOTE: estimate. Excludes attention score·V matmuls, layernorm, softmax, and
  sampling; one layer's weights are replayed, so they may stay warmer in cache
  than a true 16-layer model. The f32 §2 GEMV does not scale with threads
  (m=1, serial); the quantized §4 below does, via the column-split (Opt#2).

== Quantized decode GEMV   c[1×n] = a[1×k]·Wq[k×n]   (int8/int4 packed; m=1 column-split) ==
  attn q/k/v/o proj int8   4.20 MB packed      258.6 µs/call    16.3 GB/s   (1485 iters)
  attn q/k/v/o proj int4   2.11 MB packed      352.1 µs/call     6.0 GB/s   (1719 iters)
  ffn up           int8  16.79 MB packed     1007.6 µs/call    16.7 GB/s   (469 iters)
  ffn up           int4   8.40 MB packed     1352.2 µs/call     6.2 GB/s   (431 iters)
  ffn down         int8  16.81 MB packed     1598.0 µs/call    10.5 GB/s   (246 iters)
  ffn down         int4   8.42 MB packed     1745.6 µs/call     4.8 GB/s   (304 iters)
  output logits    int8  65.54 MB packed     3196.5 µs/call    20.5 GB/s   (208 iters)
  output logits    int4  32.78 MB packed     4974.1 µs/call     6.6 GB/s   (135 iters)

== Synthesized int4 decode step   d_model=2048, d_ff=8192, layers=16, vocab=32000 ==
     149.8 ms/token     6.68 tok/s      2.9 GB/s effective   (10 iters)
  weights streamed/token ≈ 0.44 GB int4  (vs the f32 figure in §3; same caveats apply)
```

| Square GEMM | 8 threads (GFLOP/s) | 1 thread (GFLOP/s) | Speedup |
|---|---|---|---|
| 256² | 69.4 | 21.6 | 3.2× |
| 512² | 51.4 | 17.7 | 2.9× |
| 1024² | 42.0 | 14.2 | 3.0× |
| 2048² | 40.3 | 13.6 | 3.0× |

The benchmark gained a block of the reference model's own matmul shapes (the fused QKV projection, MLP up and down, the LM head, and four sequences at once), which are small enough for per-call overhead to show. Matmuls of 20 million multiply-adds or more (here the 512² and larger squares) now run on scoped threads that borrow their operands instead of copying them for the worker pool ([Manual §2.2.1](Manual.md#221-what-is-used-and-what-was-tried-and-not-kept)). The first two blocks of the third of three runs of that build (same machine, 2026-10-27, a browser and an IDE open, so the rates are below the 2026-10-26 ones above); the median of three interleaved runs against the previous build was 4.95 ms before and 4.20 ms after for 512², and unchanged within noise for the other rows:

```text
threads = 8   (FERRUM_NUM_THREADS = unset)

== Square GEMM   C[m×n] = A[m×k]·B[k×n]   (compute-bound; m≥2 ⇒ parallel) ==
  [  256×256  ]·[  256×256  ]      0.515 ms     65.2 GFLOP/s   (1223 iters)
  [  512×512  ]·[  512×512  ]      4.503 ms     59.6 GFLOP/s   (162 iters)
  [ 1024×1024 ]·[ 1024×1024 ]     48.079 ms     44.7 GFLOP/s   (19 iters)
  [ 2048×2048 ]·[ 2048×2048 ]    434.290 ms     39.6 GFLOP/s   (3 iters)

== Reference-model matmuls   (507,552-param SLM: context 128, embed 96, hidden 320, vocab 1024) ==
  [  128×96   ]·[   96×288  ]      0.128 ms     55.2 GFLOP/s   (4635 iters)
  [  128×96   ]·[   96×320  ]      0.143 ms     54.9 GFLOP/s   (4591 iters)
  [  128×320  ]·[  320×96   ]      0.138 ms     56.8 GFLOP/s   (4649 iters)
  [  128×96   ]·[   96×1024 ]      0.462 ms     54.4 GFLOP/s   (1513 iters)
  [  512×96   ]·[   96×320  ]      0.546 ms     57.6 GFLOP/s   (1222 iters)
```

The f32 decode GEMV (`m = 1`) runs on one core whatever the thread count and streams 8.0 GB/s (2048×2048), 7.5 (2048×8192), 5.2 (8192×2048), 4.6 (2048×32000) at 8 threads, against 7.2 GB/s (2048×2048), 3.8 (2048×8192), 4.1 (8192×2048), 3.1 (2048×32000) at one thread (medians). The synthesized 1B-class f32 decode step takes 807 ms/token at 8 threads (1.24 tokens/s); with int4 weights it takes 150 ms/token (6.68 tokens/s) at 8 threads and 253 ms/token (3.95 tokens/s) at one thread.

3. Thread scaling.

```bash
for t in 1 2 4 8; do
  echo "threads $t"
  FERRUM_NUM_THREADS=$t scripts/bench_slm.sh data/prepared/train.txt
done
```

Machine and date: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 4 cores / 8 threads, 2026-10-26. Median of three runs per row. The "range" is the lowest and highest of the three; the desktop session was running, so the spread is wide.

| `FERRUM_NUM_THREADS` | Train tokens/s (range of 3) | Speedup | Eval s | Generate ms/token | Held-out perplexity |
|---|---|---|---|---|---|
| 1 | 2,900 (2,799 to 3,322) | 1.00× | 2.28 | 0.256 | 81.70 |
| 2 | 6,058 (5,351 to 6,256) | 2.09× | 1.71 | 0.260 | 76.46 |
| 4 | 6,764 (5,857 to 7,810) | 2.33× | 1.91 | 0.309 | 78.70 |
| 8 | 7,090 (4,918 to 7,407) | 2.44× | 1.55 | 0.291 | 74.24 |

`FERRUM_NUM_THREADS` sizes the matmul worker pool and, because `--threads` is auto in this script, also the number of data-parallel training workers. One thread therefore runs the serial trainer, and the step from 1 to 2 is more than 2× because two workers also overlap the serial parts of a step (softmax, the optimizer). The perplexity column differs per row for the same reason: a different worker count trains different weights. Evaluation gains little (about 1.5× at 8 threads) and generation nothing, because they are mostly serial.

4. Before and after this release. Older commits were built from clean exports into a scratch directory and run through the same script, side by side with the current build.

Three builds of the CLI, each run through the same benchmark with the same corpus, interleaved (old, middle, new, three rounds), median of three. Machine and date: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 4 cores / 8 threads, 2026-10-26.

- **Start of review:** commit `067150e` (2026-10-09), built from a clean export. Its CLI still wrote the old model format and had no batched evaluation; the script was adapted only for its flag names (`model.bin`, `--tokenizer tok.bpe`, `--chars 4096`).
- **Before the 4×8 tile:** `df66dad` (2026-10-24), the commit before the register-tiled matmul. It has `scripts/bench_slm.sh` unchanged.
- **This branch:** the checkout the other samples used.

| Reference model, auto threads | Start of review (`067150e`, 2026-10-09) | Before the 4×8 tile (`df66dad`, 2026-10-24) | This branch (`HEAD`) |
|---|---|---|---|
| Train tokens/s | 5,895 | 6,141 | 6,725 |
| Eval seconds (30 KB held-out) | 138.9 | 1.76 | 1.65 |
| Generate ms per KB of output | 6,873 | 100 | 101 |

Generation is compared per kilobyte of output because the start-of-review CLI counted characters, not tokens (it produced 4,119 bytes for `--chars 4096`, the current one about 13 KB for 4,096 tokens); both timings include process start and model load. The old evaluation also skipped the first context window of tokens, so it scored 9,340 predictions where the current one scores 9,468. Evaluation is 84× faster than at the start of the review and generation 68× faster per kilobyte. Training throughput is 1.10× the pre-tile build's and 1.14× the start-of-review build's. The tile commit quotes +17% to +19% from an A/B run on a quieter machine; the spread between runs here is as wide as the difference. The start-of-review build is an older implementation that writes the old model format, so its column compares whole pipelines, not the kernel alone.

**What to look at.**

- For a fixed model, generation and evaluation give identical results at every `FERRUM_NUM_THREADS`, and so does training when `--threads` is fixed (the `determinism` test checks this). In the scaling table the model itself changes with the thread count, because `--threads` is auto there, so its perplexity and output length change too. Only time changes when you hold `--threads` fixed.
- Eval is much faster than at the start of the review because it scores whole windows in one forward pass instead of one pass per token.
- Small models scale poorly with threads. Much of a training step is serial, and generation is a chain of small dependent matmuls. The matmul benchmark scales better (about 3× at 8 threads) because it is one large operation.
- Absolute numbers are from a busy desktop and a low-memory build; compare ratios, and re-run on an idle machine for headline figures.

---

## 10. The Rust library API

**Goal.** Train, score, generate, save, and load from Rust.

**You need.** Rust only. The program is committed as `ferrum_core/examples/library_tour.rs`.

**The program.**

```rust
//! Train, score, generate, save, and load with the library. Samples.md §10 runs it:
//! `cargo run --release -p ferrum_core --example library_tour`

#![forbid(unsafe_code)]

use ferrum_core::{GenerativeSLM, Rng, SamplingParams, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A small repetitive corpus; documents are separated by <|endoftext|>.
    let story = "Tom had a red ball. He kicked it to the park. His dog ran after it.";
    let corpus = vec![story; 200].join("\n<|endoftext|>\n");

    let cfg = TransformerConfig {
        context_len: 64,
        embed_dim: 64,
        num_heads: 4,
        num_blocks: 2,
        hidden_dim: 128,
        vocab_size: 512,
        lr: 0.01,
        batch_size: 8,
        max_tokens: 200_000,
        ..TransformerConfig::default()
    };
    let mut rng = Rng::new(1);
    let slm = GenerativeSLM::train_transformer_config_threaded(
        &corpus,
        &cfg,
        0,
        &mut rng,
        |ep, loss| {
            println!("epoch {ep}: loss {loss:.4}");
        },
    )?;

    // Score a continuation, and a whole text.
    let s = slm.score("Tom had a red", " ball")?;
    println!(
        "score: logprob {:.3}, {} tokens, greedy {}",
        s.logprob, s.n_tokens, s.greedy
    );
    let r = slm.score_rolling(story)?;
    println!(
        "rolling: logprob {:.3} over {} tokens",
        r.logprob, r.n_tokens
    );

    // Generate: greedy, with a stop string.
    let greedy = SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let g = slm.generate_tokens("Tom had", 30, &[".".to_string()], &greedy, &mut Rng::new(2))?;
    println!(
        "generated: {:?} ({} tokens, {:?})",
        g.text, g.n_tokens, g.stop_reason
    );

    // Save as GGUF and load it back.
    slm.save("samples/library_tour.gguf")?;
    let back = GenerativeSLM::load("samples/library_tour.gguf")?;
    let s2 = back.score("Tom had a red", " ball")?;
    println!("reloaded score: logprob {:.3}", s2.logprob);
    Ok(())
}
```

```bash
mkdir -p samples
time cargo run --release -p ferrum_core --example library_tour
```

The training callback prints one line per epoch (44 here); the first four epoch lines and everything from the last epoch on are shown. The time includes compiling the example.

```text
epoch 1: loss 4.6551
epoch 2: loss 2.2797
epoch 3: loss 1.7246
epoch 4: loss 1.2433
... (39 lines omitted) ...
epoch 44: loss 0.0454
score: logprob -0.003, 1 tokens, greedy true
rolling: logprob -32.553 over 19 tokens
generated: " a red ball" (4 tokens, Stop)
reloaded score: logprob -0.003

real	0m33.544s
user	0m51.757s
sys	0m11.483s
```

**What to look at.**

- `score` returns the log-probability of the continuation given the context, the number of tokens scored, and whether all of them were the model's top choice.
- `score_rolling` scores every token, the first conditioned on BOS (the EOS id), so the count equals the token count; long texts use sliding windows. The whole-story score is poor next to the one-token score because the story's first tokens come right after the start marker, a position this tiny model saw rarely in training.
- `generate_tokens` returns only the continuation, without the stop string, and the reason it stopped (`Stop` here).
- A model saved with `save` is a `gpt2` GGUF. The reloaded model gives the same score, because the int8 file reloads bit-identically to a QAT model.
