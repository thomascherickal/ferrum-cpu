<p align="center"><img src="assets/logo.png" width="96" alt="Ferrum logo"></p>

# Ferrum capability report: RAM tiers and processors

What Ferrum (`ferrum-cpu`) can create, train, run, fine-tune and export on machines from 8 GB to 128 GB of RAM, on Intel Core i3/i5/i7, the five latest Ryzen chips and the five latest Snapdragon chips. Written 2026-11-03.

## 1. How to read this report

Only one machine was measured: an **11th Gen Core i5-1135G7 (4 cores / 8 threads, 15 GiB)**, with the numbers recorded in [Manual §6.7](../MANUAL.md#67-sizing-a-run-for-your-machine), [§9.2](../MANUAL.md#92-fine-tuning-finetune-gguf) and [§13](../MANUAL.md#13-performance-and-benchmarks). **Nothing in this report was run on any other processor.** Every other number is an **estimate** computed from the measured anchor by the formulas in §2. No new benchmark was run for this report (pure reasoning, as requested).

Treat the estimates as order-of-magnitude. Three things make them uncertain, and each is marked where it matters:

- **Scaling beyond 8 threads is unmeasured.** Ferrum's own cost model assumes training speed is flat beyond 8 threads (Manual §10, `estimate.rs`). Two columns are therefore given: a **base** estimate (flat beyond 8 threads) and an **upper** estimate (75 % parallel efficiency per core).
- **Per-core speed (IPC) for other chips is an assumption**, not a measurement (§2, A1).
- **Chip specifications** for the Ryzen and Snapdragon rows come from web sources (listed at the end); Intel rows and a few Ryzen details are from memory and marked *(unverified)*.

Ferrum never uses a GPU or NPU. The kernels use no SIMD intrinsics (`unsafe` is forbidden) and rely on compiler auto-vectorization, so a chip's AI engines add nothing; only its CPU cores and memory matter.

## 2. Method

**Measured anchor (i5-1135G7, 4C/8T):**

| Quantity | Value | Source |
|---|---|---|
| GEMM, 8 threads / 1 thread (2048²) | 40.3 / 13.6 GFLOP/s | Manual §13.3 |
| Training, 20.6M-param model | 268 tokens/s, peak 3.0 GB | Manual §6.7 |
| Decode, 1B int4 | 6.68 tokens/s (0.44 GB streamed per token) | Manual §13.4 |
| Fine-tune, SmolLM2-135M | about 6 tokens/s, peak 6.4 GB | Manual §9.2 |
| Prepare / Tokenize | 18 / 13 MB/s of input, one thread | Manual §10 |
| RAM refusal | a run is refused above 3/4 of available RAM; a GGUF run above 9/10 | Manual §6.7, `guard.rs` |

**Assumptions (A1-A4):**

- **A1, per-core speed.** `f = (boost GHz / 4.2) × IPC factor`. IPC factor: 1.0 for the anchor (Willow Cove), 1.2 for Intel 12th/13th gen, 1.35 for AMD Zen 5, 1.4 for Qualcomm Oryon. These factors are my assumptions; the same-binary SSE2/NEON 128-bit baseline code gives every chip the same vector width.
- **A2, thread scaling.** Speed-up over one thread: 1.0 at 1, 1.8 at 2, 2.5 at 4, 3.0 at 8 or more threads (the 1 and 8 values are measured, the others interpolated; flat beyond 8 as in `estimate.rs`).
- **A3, kernel rate.** *Base* `13.6 × f × speedup(threads)` GFLOP/s. *Upper* `13.6 × f × core-equivalents × 0.75`. Core-equivalents count performance cores as 1 and efficiency cores as 0.5 (Intel) or 0.8 (Oryon "Performance" cores).
- **A4, decode.** The measured anchor streams only 0.44 GB × 6.68 = 2.9 GB/s, far below DRAM bandwidth, so decode is kernel-bound, not bandwidth-bound. Estimate `6.68 × (kernel rate / 40)` tokens/s for a 1B int4 model, capped at `0.7 × memory bandwidth / 0.44 GB`.

**RAM rules, taken from the code and Manual:**

- Available RAM ≈ installed − 3 GB (≤ 16 GB) or − 4 GB (above): the desktop session and OS.
- **Train from scratch:** peak RAM is dominated by activations. The §6.7 rows show about 0.15-0.18 GB per million parameters at context 256, batch 16 (20.6M → 3.0 GB; 33.8M → 6.2 GB estimated). Largest size ≈ `0.75 × available / 0.16 GB` M parameters. A smaller batch or context trades speed for RAM.
- **Run a GGUF model:** about 0.5 byte per parameter at int4 plus the f32 token embedding (`gguf_resident_bytes`); the import peak is 2-3.5 GB for 1B parameters (Manual §13.5), about 3.5 bytes per parameter used here as a conservative bound. Sources in F16 that cannot be read fully into RAM do not fit (the 7B F16 case in §13.5).
- **Fine-tune a GGUF (AdamW f32):** the checkpoint save makes the peak about 12× the f32 weight bytes, i.e. 48 bytes per parameter (SmolLM2-135M: 6.4 GB measured at 3 threads). Largest size ≈ `available / 48 B`.

## 3. Processors

Specification columns for Ryzen and Snapdragon are from the cited sources; "IPC" is assumption A1.

| Processor | Cores / threads | Boost GHz | Memory bandwidth | Max RAM | Kernel GFLOP/s base / upper | 1B int4 decode tok/s base / upper |
|---|---|---|---|---|---|---|
| **Core i5-1135G7 (measured)** | 4 / 8 | 4.2 | LPDDR4x ~68 GB/s *(unverified)* | 32 GB *(unverified)* | **40.8 / 40.8 (measured 40.3)** | **6.7 (measured 6.68)** |
| Core i3-1315U *(unverified)* | 6 (2P+4E) / 8 | 4.5 | ~76 GB/s | - | 53 / 53 | 8.8 / 8.8 |
| Core i5-1335U *(unverified)* | 10 (2P+8E) / 12 | 4.6 | ~76 GB/s | - | 54 / 80 | 9.0 / 13.4 |
| Core i7-13700H *(unverified)* | 14 (6P+8E) / 20 | 5.0 | ~89 GB/s | - | 58 / 146 | 9.7 / 24.3 |
| **Ryzen AI Max+ PRO 495** | 16 / 32 | 5.0-5.2 (sources differ) | 256-bit LPDDR5X-8533, ~273 GB/s | 192 GB | 66 / 262 | 11.0 / 43.8 |
| **Ryzen AI Max PRO 490** | 12 / 24 | ~5.0 *(unverified)* | ~273 GB/s | 192 GB | 66 / 197 | 11.0 / 32.9 |
| **Ryzen AI 9 HX 475** | 12 / 24 | 5.2 | ~137 GB/s *(unverified)* | 128 GB *(unverified)* | 68 / 205 | 11.4 / 34.2 |
| **Ryzen AI 7 450** | 8 / 16 | 5.1 | ~137 GB/s *(unverified)* | 128 GB *(unverified)* | 67 / 134 | 11.2 / 22.3 |
| **Ryzen AI 5 440** | 6 / 12 | 4.8 | ~137 GB/s *(unverified)* | 128 GB *(unverified)* | 63 / 94 | 10.5 / 15.8 |
| **Snapdragon X2 Elite Extreme** | 18 (12P+6) / 18 | ~5.0 *(unverified)* | 228 GB/s | 128 GB | 68 / 265 | 11.4 / 44.3 |
| **Snapdragon X2 Elite (18-core)** | 18 / 18 | 4.7 | 152 GB/s | - | 64 / 249 | 10.7 / 41.6 |
| **Snapdragon X2 Elite (12-core)** | 12 / 12 | 4.7 | 152 GB/s | - | 64 / 192 | 10.7 / 32.0 |
| **Snapdragon X2 Plus (10-core)** | 10 (6P+4) / 10 | ~4.0 *(unverified)* | 128 GB/s (sources say 128 or 152) | - | 54 / 120 | 9.1 / 20.0 |
| **Snapdragon X2 Plus (6-core)** | 6 / 6 | ~4.0 *(unverified)* | 128 GB/s | - | 51 / 82 | 8.5 / 13.6 |

What this shows: because small-model training and single-request decode barely use extra cores, **all fourteen CPUs land within about 1.6× of each other on the base estimate** (51-68 GFLOP/s against the anchor's 40). Extra cores only pay off, if at all, on large matmuls (fine-tuning, big GGUF prefill), and that scaling is the unmeasured part. Memory bandwidth is never the limit at Ferrum's current kernel speed: even the weakest chip's cap (about 108 tokens/s for a 1B int4 model) is far above the kernel-bound estimate.

## 4. What the RAM tier allows (RAM-bound, same for every CPU)

Largest model, by the RAM rules of §2 (estimates):

| Installed RAM | Available (est.) | Train from scratch (T256, batch 16) | Run GGUF, import-peak bound | Run GGUF, resident int4 only | Fine-tune GGUF (AdamW f32) |
|---|---|---|---|---|---|
| 8 GB | 5 GB | ~23M | ~1.4B | ~9B | ~100M (SmolLM2-135M tight) |
| 16 GB | 13 GB | ~60M | ~3.7B | ~23B | ~270M |
| 32 GB | 28 GB | ~130M | ~8B | ~50B | ~580M (Qwen2.5-0.5B fits) |
| 64 GB | 60 GB | ~280M | ~17B | ~108B | ~1.25B |
| 128 GB | 124 GB | ~580M | ~35B | ~223B | ~2.6B |

"Import-peak bound" is the safe size for models you load from a GGUF file today (the 2-3.5 GB peak at 1B is the only measurement); "resident int4 only" is what stays in RAM after load. Run speed, not RAM, is the practical limit above about 3B (§5).

## 5. What the CPU allows (speed-bound)

Compute-optimal training (about 20 tokens per parameter) needs `6 × 20 × N²` FLOPs. The anchor's measured/interpolated answers (Status §3, Manual §6.7) scale with the square root of the kernel rate:

| Processor class | Rate vs anchor (base) | Compute-optimal in 24 h | in a week | in a month |
|---|---|---|---|---|
| i5-1135G7 (measured/interpolated) | 1.0× | ~4M | ~12M | ~27M (extrapolated) |
| Core i3/i5/i7 (12th-13th gen) | 1.3-1.4× | ~4.6-4.8M | ~14M | ~31-33M |
| Ryzen AI 5/7/9 HX 4xx, Max PRO | 1.6-1.7× | ~5.0-5.2M | ~15-16M | ~34-35M |
| Snapdragon X2 Plus | 1.3× | ~4.5-4.7M | ~13.5-14M | ~30-31M |
| Snapdragon X2 Elite / Extreme | 1.6-1.7× | ~5.1-5.2M | ~15M | ~34-35M |

Even at the upper estimate (3-6× the anchor's kernel rate) the compute-optimal size grows only by the square root, to roughly 10M in 24 h and 30M in a week. **On every CPU in this report, the training limit is time, not RAM:** a 128 GB machine could hold a ~580M model but would need years to train it compute-optimally. More RAM buys longer contexts, bigger batches and bigger imported models, not bigger from-scratch SLMs.

**Inference speed, 1B int4:** 6.7 tokens/s measured on the anchor; 8-11 tokens/s base and 13-44 tokens/s upper on the others (§3). Decode time scales with model size: a 3B int4 model is about 3× slower, an 8B about 8×, so 8B runs at roughly 1-1.4 tokens/s on the base estimate (16-32 GB tiers). Prefill (prompt reading) is compute-bound: at 40 GFLOP/s about 25 s for 512 tokens at 1B (Manual §13.4).

**Fine-tuning speed:** about 6 tokens/s on a 135M model at 3 threads (anchor); this scales as `1 / parameters`, so a 500M model is about 1.6 tokens/s. Use small corpora: 1 million tokens at 6 tokens/s is about two days per epoch on the anchor, about 1 day on the faster chips (base estimate).

**Prepare / tokenize:** 18 / 13 MB/s on one thread on the anchor, so roughly 20-30 MB/s on the faster chips (single thread, scaled by `f`). RAM does not grow with the corpus.

## 6. Integrated matrix (RAM tier × processor class)

Each cell is the best Ferrum can do there (estimates). "Train" is the largest from-scratch model that is both RAM-feasible and compute-optimal in a week (base estimate); "Run" is the largest imported model at int4 with its decode speed; "Fine-tune" is the largest imported model that fits.

Processor classes: **A** = i3/i5 and Snapdragon X2 Plus (kernel ~50 GFLOP/s base); **B** = Ryzen AI 5/7/9, Snapdragon X2 Elite 12-core (~64-68 base); **C** = Ryzen AI Max, Snapdragon X2 Elite 18-core / Extreme, i7-H (~66-68 base, up to ~260 if multi-core scaling holds).

| RAM | Train from scratch (week) | Run GGUF at int4 | Fine-tune GGUF | Export (`export-gguf`, Ollama, LM Studio) |
|---|---|---|---|---|
| **8 GB** | ~12-14M (all classes; RAM allows ~23M) | ≤ ~1.4B at ~8-11 tok/s (1B); 3B only if the import fits the 9/10 guard | ≤ ~100M (SmolLM2-135M refused or tight) | Ferrum gpt2 models: yes (small). Imported ≤ 1B: yes |
| **16 GB** | ~12-15M (RAM allows ~60M) | ≤ ~3.7B at ~3-4 tok/s | ≤ ~270M (SmolLM2-135M yes) | Imported ≤ 3B: yes |
| **32 GB** | ~14-16M (RAM allows ~130M) | ≤ ~8B at ~1-1.4 tok/s (A/B); up to ~5 tok/s for class C upper | ≤ ~580M (Qwen2.5-0.5B yes) | Imported ≤ 8B: yes |
| **64 GB** | ~14-16M (RAM allows ~280M) | ≤ ~17B at ~0.5-0.7 tok/s (base) | ≤ ~1.25B (1B yes, hours-days per epoch) | Imported ≤ 17B: yes |
| **128 GB** | ~14-16M (RAM allows ~580M) | ≤ ~35B at ~0.2-0.3 tok/s (base); needs the F16 source read path for very large files | ≤ ~2.6B (days per epoch) | Imported ≤ 35B: yes, slow |

Export peaks: the export commands reuse the GUI's RAM guard (Manual §10); no separate peak was measured for this report, so the export column is only the load-size bound. Ferrum's own `gpt2` SLMs export to F32/Q8_0/Q4_0 GGUF and run in llama.cpp, Ollama and LM Studio (Manual §9.5); that path is limited by training, not by export.

## 7. Practical guidance by tier

- **8 GB:** train SLMs up to ~20M parameters (the `small`/`medium` presets and below); run ≤ 1B GGUF models at int4; fine-tune only very small models. Close other applications: the refusal guard uses a fraction of free RAM.
- **16 GB:** the sweet spot for Ferrum: train up to the 20-60M range if time allows, run ~3B at int4, fine-tune SmolLM2-135M-class models.
- **32-64 GB:** adds headroom for 7-8B inference (slow), Qwen2.5-0.5B fine-tuning (32 GB) and 1B fine-tuning (64 GB).
- **128 GB:** only useful for very large imported models at very low speed; training time, not RAM, caps from-scratch SLMs.
- **CPU choice:** per core, newer Ryzen and Snapdragon chips are about 1.3-1.7× the measured i5-1135G7 on the base estimate. More than 8 threads are unmeasured, so do not buy cores for Ferrum on this report alone. If a multi-core run is possible, `FERRUM_NUM_THREADS` and `train --threads` control the split, and results are bit-identical at any thread count (except `--threads 0` auto, which fixes the shard count from the core count).
- **Platforms:** Snapdragon X2 machines run Windows on Arm; building Ferrum for `aarch64-pc-windows-msvc` is not covered by any CI job or test in this repository, so Arm support is **not verified** here. Ryzen and Intel on Linux/Windows build with the standard `cargo build --release`.

## 8. What was not done

- No benchmark was run on any chip in §3 other than the i5-1135G7 anchor (and only from the recorded 2026-10-26 measurements).
- The IPC factors, thread scaling beyond 8 threads, the import-peak bound at 3-35B, the export RAM peak, and all Arm and Windows-on-Arm behaviour are unmeasured assumptions.
- Intel, a few Ryzen and Snapdragon details are marked *(unverified)* where the sources disagreed or were not found.
- Model sizes above 1B were not loaded; limits come from the measured 1B import and the formulas in §2.

## 9. Sources

- AMD Ryzen AI Max 400 series: [Neowin](https://www.neowin.net/news/amd-details-specs-of-new-ryzen-ai-max-400-series-apus/), [TechPowerUp](https://www.techpowerup.com/349218/amd-launches-the-ryzen-ai-max-400-series-processors-strix-halo-gets-a-memory-upgrade), [TweakTown](https://www.tweaktown.com/news/111752/amd-launches-the-ryzen-ai-max-pro-400-series-of-cpus-up-to-16-cores-with-192gb-of-unified-memory/index.html), [ServeTheHome](https://www.servethehome.com/amd-reveals-ryzen-ai-max-pro-400-series-192gb-ram-for-ai-systems/).
- Ryzen AI 9 HX 475, AI 7 450, AI 5 440: [Notebookcheck HX 475](https://www.notebookcheck.net/AMD-Ryzen-AI-9-HX-475-Processor-Benchmarks-and-Specs.1197731.0.html), [AI 7 450](https://www.notebookcheck.net/AMD-Ryzen-AI-7-450-Processor-Benchmarks-and-Specs.1197762.0.html), [AI 5 PRO 440](https://www.notebookcheck.net/AMD-Ryzen-AI-5-PRO-440-Processor-Benchmarks-and-Specs.1198884.0.html), [Windows Central](https://www.windowscentral.com/hardware/cpus/amd-ryzen-ai-pro-400-series-announcement-mwc-barcelona).
- Snapdragon X2 Elite / Extreme / Plus: [HotHardware](https://hothardware.com/reviews/qualcomm-snapdragon-x2-elite-architecture-deep-dive), [Notebookcheck](https://www.notebookcheck.net/Qualcomm-Snapdragon-X2-Elite-Extreme-Analysis-Benchmarks-Efficiency-Serious-rival-for-Apple-and-a-problem-for-AMD-Intel.1266974.0.html), [NanoReview](https://nanoreview.net/en/cpu/qualcomm-snapdragon-x2-elite-extreme-x2e-96-100), [How-To Geek](https://www.howtogeek.com/i-tested-all-three-snapdragon-x2-processors-and-the-entry-level-one-surprised-me-most/).
- Ferrum numbers: [Manual](../MANUAL.md) §6.7, §9.2, §10, §13; [Status](../STATUS.md) §3.
