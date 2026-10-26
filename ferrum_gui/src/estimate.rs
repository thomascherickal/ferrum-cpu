//! Hardware-aware time and RAM estimates for every GUI action that does work.
//!
//! [`estimate`] is a pure function: job kind + numbers in, [`Estimate`] out. It
//! never touches the disk or the machine (the caller resolves file sizes and
//! model facts, see `estimate_cmd`), so it is unit-tested with plain numbers.
//!
//! **Everything here is an estimate.** The rates are *measured* on one reference
//! machine (11th Gen Intel Core i5-1135G7, 4 cores / 8 threads, 15 GiB, release
//! build, 2026-10-26; sources in the constants' comments) and carried to other
//! machines in two ways: with a live benchmark from the Capable tab
//! ([`Throughput`]) the rates are scaled by how much faster or slower this CPU
//! is; without one they are used as they are ("a 2021 mobile quad-core"), which
//! is conservative for a modern desktop. Thread scaling is interpolated from
//! the measured 1/2/4/8-thread points and assumed flat beyond 8 threads
//! (unmeasured). GPUs are **never** used by Ferrum and never enter a formula.

use crate::hardware::HardwareInfo;
use ferrum_core::TransformerNet;
use serde::{Deserialize, Serialize};

// ── Measured reference constants (see module docs) ───────────────────────────

/// Single-thread GEMM rate of the capability micro-benchmark on the reference
/// machine: median of 5 runs of `capable::measure_gemm_gflops` built with `-O`.
pub const REF_GEMM_SINGLE_GFLOPS: f64 = 17.1;
/// Memory-stream rate of `capable::measure_mem_bandwidth` there (same runs).
pub const REF_MEM_BW_GBPS: f64 = 3.76;

/// Training speed, 8 threads, as effective GFLOP/s of `6*M + 12*L*T*C` FLOPs per
/// token, where `M` counts matmul parameters: `R(M) = 28 * M / (M + 120k)`.
/// Fitted to the four rows of Manual §6.7 (0.17M..6.9M params: 14.4, 20, 22.8
/// and 25.7 GFLOP/s; the formula gives 14.7, 21.5, 25.1 and 27.4) and checked
/// against a fresh run (18.5, 26 and 44 GFLOP/s for the 0.17M, 1.3M and 6.9M models).
const TRAIN_RMAX_GFLOPS: f64 = 28.0;
const TRAIN_M0: f64 = 120_000.0;
/// No Ferrum model trains or evaluates slower than this many effective GFLOP/s
/// at any thread count: a 7k-matmul-parameter model measured 7.4 GFLOP/s on one
/// thread (30 epochs of a 3 KB corpus in 0.44 s, 2026-10-26), and the
/// formula above would otherwise say 0.7 for it. Used as a floor, rounded down.
const TRAIN_FLOOR_GFLOPS: f64 = 6.0;
/// Speed-up relative to 8 threads: 2,900 / 6,058 / 6,764 / 7,090 tokens/s at
/// 1 / 2 / 4 / 8 threads (Manual §13.2).
const TRAIN_THREAD_SCALE: [(f64, f64); 4] = [(1.0, 0.409), (2.0, 0.854), (4.0, 0.954), (8.0, 1.0)];
/// Held-out evaluation speed relative to 8 threads (Manual §13.2: 2.28 / 1.71 /
/// 1.91 / 1.55 s, smoothed to be monotone).
const EVAL_THREAD_SCALE: [(f64, f64); 4] = [(1.0, 0.68), (2.0, 0.85), (4.0, 0.92), (8.0, 1.0)];
/// Generation of a small model is a chain of small dependent matmuls at about
/// 3 GFLOP/s whatever the thread count (Manual §13.2: 0.256-0.291 ms/token for
/// the 0.5M model; a fresh run gave 3.2-5 GFLOP/s).
const GEN_GFLOPS: f64 = 3.0;
/// Bytes of English text per BPE token by vocabulary size, measured by
/// evaluating `Manual.md` (127,924 bytes) with tokenizers trained on it.
const BYTES_PER_TOKEN: [(f64, f64); 6] = [
    (0.0, 1.005),
    (300.0, 1.307),
    (512.0, 1.813),
    (1024.0, 2.332),
    (2048.0, 2.849),
    (4096.0, 3.333),
];
/// Llama/Qwen-class BPE vocabularies: the usual ~4 characters per token for
/// English. An assumption, not measured here.
const LLAMA_BYTES_PER_TOKEN: f64 = 4.0;
/// BPE training plus encoding the corpus, ms per KB of text (measured on a
/// 84-336 KB corpus: 1.3 + 1.3 ms/KB, independent of the vocabulary size).
const TOKENIZE_MS_PER_KB: f64 = 2.6;
/// Imported-GGUF load / export throughput in millions of parameters per second
/// (fresh runs on a 29.4M-parameter llama: load 113-197, export 72-74).
const LOAD_MPARAMS_PER_S: f64 = 110.0;
const EXPORT_MPARAMS_PER_S: f64 = 70.0;
/// GGUF prefill GFLOP/s (8 threads, int4): 8.3 measured on a 29.4M-parameter
/// llama; 40 for a 1B model is the figure of Manual §13.4 (derived from the
/// GEMM rate, not measured). Interpolated in log(parameters).
const PREFILL_POINTS: [(f64, f64); 2] = [(29.4e6, 8.3), (1.0e9, 40.0)];
/// f32 prefill is slower than int4 (75 vs 125-146 tokens/s measured on the 29.4M model).
const PREFILL_F32_FACTOR: f64 = 0.6;
/// GEMM thread scaling relative to 8 threads: 14.2 vs 42 GFLOP/s (Manual §13.3).
const GEMM_THREAD_SCALE: [(f64, f64); 2] = [(1.0, 0.34), (8.0, 1.0)];
/// Decode streams the non-embedding weights once per token at this effective
/// rate with 8 threads, by model size (interpolated in log(parameters)):
/// 0.8 GB/s for a 29.4M-parameter int4 model (14.7 MB in 18.5 ms, measured) and
/// 2.94 GB/s for a 1B int4 model (0.44 GB in 150 ms, Manual §13.4).
const DECODE_BYTES_PER_S: [(f64, f64); 2] = [(29.4e6, 0.8e9), (1.0e9, 2.94e9)];
/// Decode speed relative to 8 threads: 3.95 vs 6.68 tokens/s (Manual §13.4).
const DECODE_THREAD_SCALE: [(f64, f64); 2] = [(1.0, 0.59), (8.0, 1.0)];
/// Full-parameter fine-tune: GFLOP/s of `6 * M * token-passes` on one thread,
/// measured 4.83 on a 1.05M llama (3 threads: 10.25, 8: 16.46) and 2.33 derived
/// from Manual §9.2 (6 token-passes/s on 135M with 3 threads = 4.9 GFLOP/s).
/// Interpolated in log(parameters).
const FT_POINTS: [(f64, f64); 2] = [(1.05e6, 4.83), (135e6, 2.33)];
/// Fine-tune speed-up over one thread: 1 / 3 / 8 threads = 1 / 2.1 / 3.4.
const FT_THREAD_SCALE: [(f64, f64); 3] = [(1.0, 1.0), (3.0, 2.1), (8.0, 3.4)];
/// Peak fine-tune RAM is ~12x the f32 weights (Manual §9.2, measured 6.4 GB on 135M).
const FT_PEAK_FACTOR: usize = 12;
/// Cleaning text, bytes/s: 115 MB/s measured on a 1.6 MB text, rounded down.
const CLEAN_BYTES_PER_S: f64 = 100e6;
/// `prepare` (clean, NFKC, dedup, hash, split), bytes of input per second on one thread: 22.5 MB
/// of TinyStories took 0.62 s plain, 0.93 s with dedup + NFKC and 1.18 s with lowercase too
/// (release, 2026-10-27); the slowest is rounded down. Its RAM stays near 7 MB on that input.
const PREPARE_BYTES_PER_S: f64 = 18e6;
/// Reading a file from disk, bytes/s. Assumed, not measured.
const DISK_BYTES_PER_S: f64 = 200e6;
/// Download speed assumption, bytes/s, and the range shown around it.
const NET_BYTES_PER_S: f64 = 5e6;
/// The app refuses training above 3/4 of available RAM or 16 GiB (`check_train_memory`).
const TRAIN_RAM_FRACTION: f64 = 0.75;
const TRAIN_RAM_CAP_MB: f64 = 16.0 * 1024.0;
/// Every GGUF guard in the app uses 90% of available RAM.
const GGUF_RAM_FRACTION: f64 = 0.9;

// ── Public types ─────────────────────────────────────────────────────────────

/// What a GUI button does, as far as the estimate is concerned.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    /// Tabs, refresh, clear, browse, close, save: no real work.
    Instant,
    Download,
    LoadFile,
    CleanText,
    Train,
    Generate,
    Evaluate,
    ModelInfo,
    GgufInfo,
    GgufRun,
    GgufFinetune,
    GgufExport,
    Capability,
    /// The Prepare tab: clean and split text files into train/val/test.
    Prepare,
}

/// Facts about a Ferrum SLM file (`generate`, `evaluate`), read from its GGUF header.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct SlmFacts {
    pub params: u64,
    /// Parameters that cost matmul FLOPs (everything but the embedding tables).
    pub matmul_params: u64,
    pub vocab: usize,
    pub file_bytes: u64,
}

/// Facts about an imported Llama/Qwen GGUF, read from its header.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct GgufFacts {
    pub params: u64,
    /// Elements of `token_embd.weight` (kept f32; a lookup, not a matmul).
    pub embed_params: u64,
    /// `false` when the head is tied to the embedding.
    pub has_output_head: bool,
    pub n_layers: usize,
    pub context_len: usize,
    pub file_bytes: u64,
}

/// Everything the estimate reads. Missing fields mean "not filled in yet".
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct JobParams {
    // Train tab
    pub context_len: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub hidden_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub vocab_size: usize,
    pub threads: usize,
    pub window_stride: usize,
    pub max_tokens: u64,
    pub val_fraction: f32,
    // Other tabs
    pub num_chars: usize,
    pub max_new: usize,
    pub prompt_chars: usize,
    pub seq_len: usize,
    pub quant: String,
    pub max_mb: f64,
    /// Characters of pasted/typed text (clean, evaluate).
    pub text_chars: usize,
    // Paths: only `estimate_cmd` reads these, to fill in the sizes and facts below.
    pub corpus_path: String,
    pub model_path: String,
    pub text_path: String,
    /// The Evaluate tab's optional second (test) file.
    pub test_path: String,
    pub resume_path: String,
    /// The Prepare tab's input files and folders, and its output folder.
    pub input_paths: Vec<String>,
    pub out_dir: String,
    // Resolved by `estimate_cmd` (tests set them directly).
    pub corpus_bytes: Option<u64>,
    pub text_bytes: Option<u64>,
    pub test_bytes: Option<u64>,
    pub resume_bytes: Option<u64>,
    /// Bytes of every file under `input_paths`.
    pub input_bytes: Option<u64>,
    pub slm: Option<SlmFacts>,
    pub gguf: Option<GgufFacts>,
}

/// A live benchmark (Capable tab) used to scale the reference rates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Throughput {
    pub gemm_single_gflops: f64,
    pub mem_bw_gbps: f64,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Estimate {
    pub kind: JobKind,
    /// Expected wall time; `None` when it cannot be estimated (see `notes`).
    pub seconds: Option<f64>,
    pub low_seconds: Option<f64>,
    pub high_seconds: Option<f64>,
    pub peak_ram_mb: Option<f64>,
    /// The RAM the app allows this job to use (its own guard), if known.
    pub ram_limit_mb: Option<f64>,
    pub ram_available_mb: Option<f64>,
    /// `false` only when the peak is known to exceed the limit.
    pub fits_in_ram: bool,
    /// Threads the estimate assumed.
    pub threads_used: usize,
    /// Always `false`: Ferrum does not use a GPU.
    pub uses_gpu: bool,
    pub notes: Vec<String>,
}

// ── Small pure helpers ───────────────────────────────────────────────────────

/// Piecewise-linear interpolation through `points` (ascending x), clamped at the ends.
pub fn interp(points: &[(f64, f64)], x: f64) -> f64 {
    let (first, last) = (points[0], points[points.len() - 1]);
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    let i = points.iter().position(|p| p.0 >= x).unwrap_or(1).max(1);
    let (a, b) = (points[i - 1], points[i]);
    a.1 + (b.1 - a.1) * (x - a.0) / (b.0 - a.0)
}

/// [`interp`] with the x axis on a log scale (parameter counts).
fn interp_log(points: &[(f64, f64)], x: f64) -> f64 {
    let logged: Vec<_> = points.iter().map(|p| (p.0.ln(), p.1)).collect();
    interp(&logged, x.max(1.0).ln())
}

/// Bytes of English text per BPE token for a vocabulary of `vocab` (0 = characters).
pub fn bytes_per_token(vocab: usize) -> f64 {
    interp(&BYTES_PER_TOKEN, vocab as f64)
}

/// Matmul-only parameters of a Ferrum net: everything except the token and position tables.
fn matmul_params(n: u64, vocab: usize, embed: usize, ctx: usize) -> f64 {
    n.saturating_sub((vocab as u64 + ctx as u64).saturating_mul(embed as u64)) as f64
}

/// Training GFLOP/s at 8 threads for `m` matmul parameters.
pub fn train_gflops_8(m: f64) -> f64 {
    TRAIN_RMAX_GFLOPS * m / (m + TRAIN_M0)
}

/// (compute speed, bandwidth speed) of this machine relative to the reference.
fn machine_speed(bench: Option<&Throughput>) -> (f64, f64) {
    match bench {
        Some(b) if b.gemm_single_gflops > 0.0 && b.mem_bw_gbps > 0.0 => (
            (b.gemm_single_gflops / REF_GEMM_SINGLE_GFLOPS).clamp(0.25, 4.0),
            (b.mem_bw_gbps / REF_MEM_BW_GBPS).clamp(0.25, 4.0),
        ),
        _ => (1.0, 1.0),
    }
}

/// Threads that actually run: the request (0 = Ferrum's pool), capped by the logical cores.
fn effective_threads(requested: usize, hw: &HardwareInfo) -> usize {
    let want = if requested == 0 {
        hw.ferrum_threads
    } else {
        requested
    };
    want.min(hw.logical_cores).max(1)
}

fn mb(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0)
}

fn quant_bytes_per_param(q: &str) -> f64 {
    match q {
        "int4" | "q4" => 0.5,
        "int8" | "q8" => 1.0,
        _ => 4.0,
    }
}

/// Builder for the result envelope.
struct Draft<'a> {
    kind: JobKind,
    hw: &'a HardwareInfo,
    threads: usize,
    notes: Vec<String>,
}

impl<'a> Draft<'a> {
    fn new(kind: JobKind, hw: &'a HardwareInfo, bench: Option<&Throughput>) -> Self {
        let mut notes = Vec::new();
        notes.push(match bench {
            Some(b) => {
                let (s, _) = machine_speed(Some(b));
                format!("Scaled by your Capable benchmark ({s:.2}x the reference CPU).")
            }
            None => "Rates measured on a 4-core Core i5-1135G7; run the Capable check to scale them to this machine.".into(),
        });
        Draft {
            kind,
            hw,
            threads: 1,
            notes,
        }
    }

    fn note(&mut self, s: impl Into<String>) {
        self.notes.push(s.into());
    }

    /// Finish with a time (`ratio` = (low, high) multipliers) and optional RAM figures.
    fn done(
        self,
        seconds: Option<f64>,
        ratio: (f64, f64),
        peak_bytes: Option<f64>,
        limit_mb: Option<f64>,
    ) -> Estimate {
        let mut notes = self.notes;
        let avail = self.hw.ram_available_mb.map(|m| m as f64);
        let peak_ram_mb = peak_bytes.map(mb);
        let limit = limit_mb.or(avail);
        let fits = match (peak_ram_mb, limit) {
            (Some(p), Some(l)) => p <= l,
            _ => true,
        };
        if avail.is_none() && peak_ram_mb.is_some() {
            notes.push(
                "Available RAM is unknown on this system, so the RAM check is skipped.".into(),
            );
        }
        notes.push("GPU: not used by Ferrum (CPU only).".into());
        Estimate {
            kind: self.kind,
            seconds,
            low_seconds: seconds.map(|s| s * ratio.0),
            high_seconds: seconds.map(|s| s * ratio.1),
            peak_ram_mb,
            ram_limit_mb: limit,
            ram_available_mb: avail,
            fits_in_ram: fits,
            threads_used: self.threads,
            uses_gpu: false,
            notes,
        }
    }
}

// ── The estimate ─────────────────────────────────────────────────────────────

/// Estimate time and RAM for one job. Pure; see the module docs for the model.
pub fn estimate(
    kind: JobKind,
    p: &JobParams,
    hw: &HardwareInfo,
    bench: Option<&Throughput>,
) -> Estimate {
    let mut d = Draft::new(kind, hw, bench);
    let (speed, bw_speed) = machine_speed(bench);
    d.threads = effective_threads(p.threads, hw);
    match kind {
        JobKind::Instant => {
            let mut e = d.done(Some(0.0), (1.0, 1.0), None, None);
            e.notes.clear();
            e
        }
        JobKind::Capability => {
            d.note("A fixed micro-benchmark of about 1 to 2 seconds (Manual 10.3); one core, then a short matmul.");
            d.done(Some(1.5), (0.7, 2.0), None, None)
        }
        JobKind::GgufInfo => {
            d.note("Reads only the file header.");
            d.done(Some(0.05), (0.5, 20.0), None, None)
        }
        JobKind::ModelInfo | JobKind::LoadFile => {
            let bytes = match kind {
                JobKind::ModelInfo => p.slm.as_ref().map(|s| s.file_bytes),
                _ => p.text_bytes,
            };
            d.note(format!(
                "Reading the file at an assumed {:.0} MB/s.",
                DISK_BYTES_PER_S / 1e6
            ));
            let secs = 0.02 + bytes.unwrap_or(0) as f64 / DISK_BYTES_PER_S;
            d.done(Some(secs), (0.5, 5.0), None, None)
        }
        JobKind::CleanText => {
            let n = if p.text_chars > 0 {
                p.text_chars as f64
            } else {
                0.0
            };
            d.note(format!(
                "Cleaning at about {:.0} MB/s of text.",
                CLEAN_BYTES_PER_S / 1e6
            ));
            d.done(
                Some(0.01 + n / CLEAN_BYTES_PER_S / speed),
                (0.5, 5.0),
                None,
                None,
            )
        }
        JobKind::Download => {
            let size = if p.max_mb > 0.0 {
                d.note("Stops at the Max MB cap; a smaller file finishes sooner.");
                p.max_mb * 1e6
            } else {
                d.note("File size unknown: assuming 50 MB.");
                50e6
            };
            d.note(format!(
                "Depends on your connection: assumes {:.0} MB/s (40 Mbit/s); a slow link can be 5x slower.",
                NET_BYTES_PER_S / 1e6
            ));
            d.done(Some(size / NET_BYTES_PER_S), (0.25, 5.0), None, None)
        }
        JobKind::Train => train(d, p, speed),
        JobKind::Generate => generate(d, p, speed),
        JobKind::Evaluate => evaluate(d, p, speed),
        JobKind::GgufRun => gguf_run(d, p, speed, bw_speed),
        JobKind::GgufFinetune => finetune(d, p, speed),
        JobKind::GgufExport => export(d, p),
        JobKind::Prepare => prepare(d, p, speed),
    }
}

fn prepare(mut d: Draft, p: &JobParams, speed: f64) -> Estimate {
    let Some(bytes) = p.input_bytes else {
        d.note("Add an input file or folder to get an estimate.");
        return d.done(None, (1.0, 1.0), None, None);
    };
    d.threads = 1;
    let secs = 0.05 + bytes as f64 / (PREPARE_BYTES_PER_S * speed);
    d.note(format!(
        "{:.1} MB of input at about {:.0} MB/s on one thread (cleaning, hashing and splitting are serial); NFKC, lowercase and dedup make it slower.",
        bytes as f64 / 1e6,
        PREPARE_BYTES_PER_S / 1e6 * speed
    ));
    d.note("Streams in 1 MiB chunks, so RAM does not grow with the corpus; a file without the document separator is one document and is held whole (a few times over while cleaning).");
    // 7 MB measured on 22.5 MB; the dedup set holds 16 bytes per kept document.
    let peak = 16e6 + bytes as f64 / 30.0;
    let limit = d.hw.ram_available_mb.map(|a| a as f64 * GGUF_RAM_FRACTION);
    d.done(Some(secs), (0.5, 2.5), Some(peak), limit)
}

fn train(mut d: Draft, p: &JobParams, speed: f64) -> Estimate {
    let none = |mut d: Draft, why: &str| {
        d.note(why);
        d.done(None, (1.0, 1.0), None, None)
    };
    if p.context_len == 0 || p.embed_dim == 0 || p.num_blocks == 0 || p.hidden_dim == 0 {
        return none(d, "Fill in the model size to get an estimate.");
    }
    if p.num_heads == 0 || !p.embed_dim.is_multiple_of(p.num_heads) {
        return none(
            d,
            "The embedding dimension must be divisible by the number of heads.",
        );
    }
    if p.batch_size == 0 || (p.epochs == 0 && p.max_tokens == 0) {
        return none(d, "Set epochs (or a token budget) and a batch size.");
    }
    // Character-level models (vocab 0) have one token per distinct character; assume plain ASCII.
    let vocab = if p.vocab_size == 0 { 128 } else { p.vocab_size };
    let n = match TransformerNet::check_size(
        vocab,
        p.context_len,
        p.embed_dim,
        p.hidden_dim,
        p.num_blocks,
    ) {
        Ok(n) => n as u64,
        Err(e) => return none(d, &format!("This model is too large to train: {e}")),
    };
    let corpus = p.corpus_bytes.unwrap_or_else(|| {
        d.note("Corpus not found yet: assuming 100 KB of text.");
        100_000
    }) as f64;
    let bpt = bytes_per_token(p.vocab_size);
    let val = f64::from(p.val_fraction).clamp(0.0, 0.5);
    let tokens = corpus / bpt * (1.0 - val);
    let ctx = p.context_len as f64;
    let stride = if p.window_stride == 0 {
        ctx
    } else {
        p.window_stride as f64
    };
    let per_epoch = tokens * ctx / stride;
    let (total, how) = if p.max_tokens > 0 {
        (p.max_tokens as f64, "token budget".to_string())
    } else {
        (
            per_epoch * p.epochs as f64,
            format!("{} epoch(s)", p.epochs),
        )
    };
    d.threads = effective_threads(p.threads, d.hw).min(p.batch_size);
    let m = matmul_params(n, vocab, p.embed_dim, p.context_len);
    let flops_tok = 6.0 * m + 12.0 * (p.num_blocks * p.context_len * p.embed_dim) as f64;
    let scale = interp(&TRAIN_THREAD_SCALE, d.threads as f64);
    let rate = (train_gflops_8(m) * scale).max(TRAIN_FLOOR_GFLOPS) * 1e9 * speed / flops_tok;
    let tokenize = if p.vocab_size >= 256 {
        corpus / 1024.0 * TOKENIZE_MS_PER_KB / 1000.0 / speed
    } else {
        0.0
    };
    let secs = total / rate + tokenize;
    d.note(format!(
        "{n} parameters, about {} tokens trained ({how}), {} tokens/s on {} thread(s); tokenizer {tokenize:.1} s.",
        total.round(), rate.round(), d.threads
    ));
    if d.threads > 8 {
        d.note("Speed-up beyond 8 threads was not measured; assumed flat.");
    }
    let windows =
        ((tokens as usize).saturating_sub(p.context_len + 1)) / stride.max(1.0) as usize + 1;
    let peak = TransformerNet::estimate_train_peak_bytes(
        vocab,
        p.context_len,
        p.embed_dim,
        p.num_heads,
        p.hidden_dim,
        p.num_blocks,
        windows,
        p.batch_size,
        d.threads,
    )
    .ok()
    .map(|b| b as f64);
    let limit =
        d.hw.ram_available_mb
            .map(|a| (a as f64 * TRAIN_RAM_FRACTION).min(TRAIN_RAM_CAP_MB));
    let mut est = d.done(Some(secs), (0.6, 1.6), peak, limit);
    if !est.fits_in_ram {
        est.notes.push("Training will be refused: it needs more than 3/4 of your free RAM. Lower batch, context, embed or blocks.".into());
    }
    est
}

fn slm_or_default(d: &mut Draft, p: &JobParams) -> SlmFacts {
    p.slm.clone().unwrap_or_else(|| {
        d.note("No model chosen yet: assuming a 0.5M-parameter model.");
        SlmFacts {
            params: 507_552,
            matmul_params: 396_960,
            vocab: 1024,
            file_bytes: 0,
        }
    })
}

fn generate(mut d: Draft, p: &JobParams, speed: f64) -> Estimate {
    let s = slm_or_default(&mut d, p);
    // A token budget (`max_new`) replaces the character budget, as in `generate_tokens`.
    let tokens = if p.max_new > 0 {
        p.max_new as f64
    } else {
        p.num_chars as f64 / bytes_per_token(s.vocab)
    };
    let per_token = 2.0 * s.matmul_params as f64 / (GEN_GFLOPS * 1e9 * speed);
    d.note(format!(
        "About {} tokens at {:.2} ms each; generation is serial, so extra cores do not help.",
        tokens.round(),
        per_token * 1000.0
    ));
    d.threads = 1;
    let secs = 0.05 + s.file_bytes as f64 / DISK_BYTES_PER_S + tokens * per_token;
    d.note("RAM shown is an upper bound (weights as f32).");
    let limit = d.hw.ram_available_mb.map(|a| a as f64 * GGUF_RAM_FRACTION);
    d.done(Some(secs), (0.5, 2.0), Some(s.params as f64 * 4.0), limit)
}

fn evaluate(mut d: Draft, p: &JobParams, speed: f64) -> Estimate {
    // The text to score: pasted or a file, plus the optional test file.
    let bytes = match (p.text_bytes, p.text_chars) {
        (Some(b), _) if b > 0 => b as f64,
        (_, c) if c > 0 => c as f64,
        _ => {
            d.note("No held-out text yet: assuming 100 KB.");
            100_000.0
        }
    } + p.test_bytes.unwrap_or(0) as f64;
    if p.slm.is_none() && p.gguf.is_some() {
        return evaluate_gguf(d, p, bytes, speed);
    }
    let s = slm_or_default(&mut d, p);
    let tokens = bytes / bytes_per_token(s.vocab);
    d.threads = effective_threads(0, d.hw);
    let gflops = (train_gflops_8(s.matmul_params as f64)
        * interp(&EVAL_THREAD_SCALE, d.threads as f64))
    .max(TRAIN_FLOOR_GFLOPS);
    let rate = gflops * 1e9 * speed / (6.0 * s.matmul_params as f64).max(1.0);
    d.note(format!(
        "About {} tokens scored at {} tokens/s (scoring is not batched).",
        tokens.round(),
        rate.round()
    ));
    let limit = d.hw.ram_available_mb.map(|a| a as f64 * GGUF_RAM_FRACTION);
    d.done(
        Some(0.05 + tokens / rate),
        (0.6, 1.6),
        Some(s.params as f64 * 4.0),
        limit,
    )
}

/// `eval-gguf`: load the model, then one forward pass over every token (2 FLOPs per weight, at
/// the prefill rate); a fine-tune checkpoint forces f32 and adds its read.
fn evaluate_gguf(mut d: Draft, p: &JobParams, bytes: f64, speed: f64) -> Estimate {
    let g = p.gguf.clone().unwrap_or_default();
    let resume = !p.resume_path.trim().is_empty();
    let bpp = if resume {
        4.0
    } else {
        quant_bytes_per_param(&p.quant)
    };
    let non_embd = g.params.saturating_sub(g.embed_params) as f64;
    let resident = non_embd * bpp + g.embed_params as f64 * 4.0;
    // Applying a checkpoint holds the model, the checkpoint bytes and half a model of scratch.
    let peak = if resume {
        resident * 1.5 + p.resume_bytes.unwrap_or(0) as f64
    } else {
        resident
    };
    let tokens = bytes / LLAMA_BYTES_PER_TOKEN;
    d.threads = effective_threads(p.threads, d.hw);
    let mut gflops = interp_log(&PREFILL_POINTS, g.params as f64)
        * interp(&GEMM_THREAD_SCALE, d.threads as f64)
        * speed;
    if bpp >= 4.0 {
        gflops *= PREFILL_F32_FACTOR;
    }
    let load = g.params as f64 / (LOAD_MPARAMS_PER_S * 1e6)
        + p.resume_bytes.unwrap_or(0) as f64 / DISK_BYTES_PER_S;
    let score = 2.0 * non_embd * tokens / (gflops * 1e9);
    d.note(format!(
        "Imported GGUF: load {load:.1} s + scoring about {} tokens ({:.1} s); ~4 characters per token is assumed.",
        tokens.round(),
        score
    ));
    let limit = d.hw.ram_available_mb.map(|a| a as f64 * GGUF_RAM_FRACTION);
    d.done(Some(load + score), (0.4, 2.5), Some(peak), limit)
}

fn gguf_run(mut d: Draft, p: &JobParams, speed: f64, bw_speed: f64) -> Estimate {
    let Some(g) = p.gguf.clone() else {
        d.note("Choose a GGUF file to get an estimate.");
        return d.done(None, (1.0, 1.0), None, None);
    };
    let bpp = if p.resume_path.trim().is_empty() {
        quant_bytes_per_param(&p.quant)
    } else {
        4.0
    };
    let non_embd = g.params.saturating_sub(g.embed_params) as f64;
    let resident = non_embd * bpp + g.embed_params as f64 * 4.0;
    let streamed = non_embd * bpp
        + if g.has_output_head {
            0.0
        } else {
            g.embed_params as f64 * bpp
        };
    let load = g.params as f64 / (LOAD_MPARAMS_PER_S * 1e6);
    let prompt_tokens = p.prompt_chars as f64 / LLAMA_BYTES_PER_TOKEN + 8.0;
    d.threads = effective_threads(p.threads, d.hw);
    let t = d.threads as f64;
    let mut gflops =
        interp_log(&PREFILL_POINTS, g.params as f64) * interp(&GEMM_THREAD_SCALE, t) * speed;
    if bpp >= 4.0 {
        gflops *= PREFILL_F32_FACTOR;
    }
    let prefill = 2.0 * non_embd * prompt_tokens / (gflops * 1e9);
    let bw = interp_log(&DECODE_BYTES_PER_S, g.params as f64)
        * interp(&DECODE_THREAD_SCALE, t)
        * bw_speed;
    let per_tok = streamed / bw;
    let decode = p.max_new as f64 * per_tok;
    d.note(format!(
        "Load {load:.1} s + prompt {prefill:.1} s (about {} tokens) + up to {} new tokens at {:.1} tokens/s.",
        prompt_tokens.round(), p.max_new, 1.0 / per_tok
    ));
    d.note("Generation stops early at an end-of-text token; ~4 characters per token is assumed.");
    let limit = d.hw.ram_available_mb.map(|a| a as f64 * GGUF_RAM_FRACTION);
    d.done(
        Some(load + prefill + decode),
        (0.4, 2.5),
        Some(resident),
        limit,
    )
}

fn finetune(mut d: Draft, p: &JobParams, speed: f64) -> Estimate {
    let Some(g) = p.gguf.clone() else {
        d.note("Choose a GGUF file to get an estimate.");
        return d.done(None, (1.0, 1.0), None, None);
    };
    let corpus = p.corpus_bytes.unwrap_or_else(|| {
        d.note("Corpus not found yet: assuming 100 KB of text.");
        100_000
    }) as f64;
    let tokens = corpus / LLAMA_BYTES_PER_TOKEN + 1.0;
    let ctx = if g.context_len == 0 {
        p.seq_len
    } else {
        g.context_len
    };
    let seq = p.seq_len.min(ctx).max(2) as f64;
    if tokens < seq || p.epochs == 0 || p.batch_size == 0 {
        d.note("The corpus is shorter than the sequence length, or epochs/batch are empty: fine-tuning would fail.");
        return d.done(None, (1.0, 1.0), None, None);
    }
    // Every epoch trains on every overlapping window of `seq` tokens.
    let passes = (tokens - seq + 1.0) * seq * p.epochs as f64;
    let non_embd = g.params.saturating_sub(g.embed_params) as f64;
    let head = if g.has_output_head {
        0.0
    } else {
        g.embed_params as f64
    };
    d.threads = effective_threads(p.threads, d.hw).min(p.batch_size);
    let gflops = interp_log(&FT_POINTS, g.params as f64)
        * interp(&FT_THREAD_SCALE, d.threads as f64)
        * speed;
    let train = 6.0 * (non_embd + head) * passes / (gflops * 1e9);
    let load = g.params as f64 / (LOAD_MPARAMS_PER_S * 1e6);
    d.note(format!(
        "{} epoch(s) over about {} windows of {} tokens, {gflops:.1} GFLOP/s on {} thread(s).",
        p.epochs,
        (tokens - seq + 1.0).round(),
        seq,
        d.threads
    ));
    d.note("Rates were measured on a 1M model and derived from a 135M model (3 threads); other sizes are interpolated.");
    d.note("Peak RAM is about 12x the f32 weights (measured); the app's own check only counts 4x.");
    let limit = d.hw.ram_available_mb.map(|a| a as f64 * GGUF_RAM_FRACTION);
    let peak = (g.params as usize)
        .saturating_mul(4)
        .saturating_mul(FT_PEAK_FACTOR) as f64;
    d.done(Some(load + train), (0.4, 2.5), Some(peak), limit)
}

fn export(mut d: Draft, p: &JobParams) -> Estimate {
    let Some(g) = p.gguf.clone() else {
        d.note("Choose a GGUF file to get an estimate.");
        return d.done(None, (1.0, 1.0), None, None);
    };
    let est_model = (g.params as usize).saturating_mul(4);
    let out = match p.quant.as_str() {
        "f32" => est_model,
        "f16" => est_model / 2,
        _ => est_model / 3,
    };
    let write_peak = est_model.saturating_add(out);
    let resume = !p.resume_path.trim().is_empty();
    let peak = if resume {
        est_model.saturating_mul(4).max(write_peak)
    } else {
        write_peak
    };
    let mut secs = g.params as f64 / (EXPORT_MPARAMS_PER_S * 1e6);
    if resume {
        secs += p.resume_bytes.unwrap_or(0) as f64 / DISK_BYTES_PER_S;
        d.note("Applying a fine-tune checkpoint needs weights + gradients + Adam moments (4x).");
    }
    d.note(format!(
        "Load, re-encode and write at about {EXPORT_MPARAMS_PER_S:.0} million parameters/s."
    ));
    let limit = d.hw.ram_available_mb.map(|a| a as f64 * GGUF_RAM_FRACTION);
    d.done(Some(secs), (0.5, 2.0), Some(peak as f64), limit)
}

#[cfg(test)]
mod tests;
