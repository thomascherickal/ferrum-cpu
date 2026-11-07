//! The Train tab's pre-flight panel: before Start, the parameter count, the
//! estimated peak RAM against the free RAM, and whether the saved model will
//! run in llama.cpp.
//!
//! Everything is computed from the form values with the engine's own functions
//! (`TransformerNet::check_size` and `estimate_train_peak_bytes` through the
//! [`crate::sizing::estimate`] module, so the numbers match the Train button's tooltip
//! and the engine's refusal). Nothing is allocated or trained.

use crate::infra::error::CmdError;
use crate::sizing::estimate::{estimate, JobKind, JobParams, Throughput};
use crate::sizing::estimate_cmd::resolve;
use crate::sizing::hardware::{self, HardwareInfo};
use crate::AppState;
use ferrum_core::TransformerNet;
use serde::Serialize;
use tauri::State;

/// Will the saved model run in llama.cpp, Ollama and LM Studio?
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LlamaCpp {
    pub runs: bool,
    /// One plain sentence saying why.
    pub reason: String,
}

/// What the corpus is, when it is a token file (`tokenize`): its ids stay on disk.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CorpusInfo {
    pub tokens: u64,
    pub documents: u64,
    pub vocab: usize,
    /// Always `false` for a token file: only the current minibatch's windows are in memory.
    pub ram_scales_with_corpus: bool,
    pub note: String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Preflight {
    /// Trainable parameters of the net (`None` when the sizes are invalid or too large).
    pub params: Option<u64>,
    pub peak_ram_mb: Option<f64>,
    pub ram_available_mb: Option<f64>,
    /// What the engine allows a training run to use (3/4 of the free RAM, at most 16 GiB).
    pub ram_limit_mb: Option<f64>,
    /// `false` only when the peak is known to exceed the limit: the engine would refuse to start.
    pub fits_in_ram: bool,
    pub seconds: Option<f64>,
    pub llama_cpp: LlamaCpp,
    /// Set when the corpus is a token file.
    pub corpus: Option<CorpusInfo>,
    /// Why the run cannot start for a reason other than RAM.
    pub problems: Vec<String>,
}

/// The llama.cpp verdict from the vocabulary setting alone: 0 = character level (the generic
/// `ferrum` layout, which only Ferrum reads), 256 or more = byte-level BPE (`gpt2` GGUF).
fn llama_cpp(vocab: usize) -> LlamaCpp {
    match vocab {
        0 => LlamaCpp {
            runs: false,
            reason: "Vocab 0 trains a character-level model, saved in Ferrum's own layout: only Ferrum can load it, not llama.cpp, Ollama or LM Studio. Use a vocab of 256 or more (for example 512) for a byte-level BPE model that they all run.".into(),
        },
        1..=255 => LlamaCpp {
            runs: false,
            reason: "A vocab between 1 and 255 is not valid: use 0 (character level) or 256 and up (byte-level BPE).".into(),
        },
        _ => LlamaCpp {
            runs: true,
            reason: "Byte-level BPE models are saved as a gpt2-architecture GGUF with the tokenizer inside, which llama.cpp, Ollama and LM Studio load as they are.".into(),
        },
    }
}

/// The checks the Train button runs before it starts, as readable sentences.
fn problems(p: &JobParams) -> Vec<String> {
    let mut out = Vec::new();
    if p.context_len == 0 || p.embed_dim == 0 || p.num_blocks == 0 || p.hidden_dim == 0 {
        out.push("Context, embed dim, blocks and hidden must each be at least 1.".into());
    }
    if p.num_heads == 0 || !p.embed_dim.is_multiple_of(p.num_heads) {
        out.push(format!(
            "Embed dim ({}) must be divisible by heads ({}).",
            p.embed_dim, p.num_heads
        ));
    }
    if p.batch_size == 0 || (p.epochs == 0 && p.max_tokens == 0) {
        out.push("Set epochs (or a token budget) and a batch size of at least 1.".into());
    }
    if p.corpus_tokens.is_some() {
        // A token file is BPE-tokenized and its vocabulary is the tokenizer's.
        if p.vocab_size == 0 {
            out.push("A token file is BPE-tokenized: set Vocab to 256 or more (the model uses the token file's own vocabulary).".into());
        }
        if p.tokenizer_path.trim().is_empty() {
            out.push(
                "A token file needs the Tokenizer file field: the file Tokenize corpus used."
                    .into(),
            );
        }
    } else if (1..256).contains(&p.vocab_size) {
        out.push("Vocab must be 0 (character level) or at least 256 (byte-level BPE).".into());
    }
    out
}

/// Pure core of [`preflight_train`]: the form values (with corpus facts already resolved) and the
/// hardware in, the panel out.
pub fn preflight(p: &JobParams, hw: &HardwareInfo, bench: Option<&Throughput>) -> Preflight {
    let problems = problems(p);
    let mut out = Preflight {
        params: None,
        peak_ram_mb: None,
        ram_available_mb: hw.ram_available_mb.map(|m| m as f64),
        ram_limit_mb: None,
        fits_in_ram: true,
        seconds: None,
        llama_cpp: llama_cpp(p.corpus_vocab.unwrap_or(p.vocab_size)),
        corpus: p.corpus_tokens.map(|tokens| CorpusInfo {
            tokens,
            documents: p.corpus_documents.unwrap_or(0),
            vocab: p.corpus_vocab.unwrap_or(0),
            ram_scales_with_corpus: false,
            note: "Token file: the ids stay on disk and only the current minibatch's windows are read, so the RAM figure does not depend on how large the corpus is.".into(),
        }).or_else(|| {
            let bytes = p.corpus_bytes.filter(|&b| p.vocab_size >= 256 && b >= ferrum_core::STREAM_CORPUS_THRESHOLD)?;
            let est_tokens = (bytes as f64 / crate::sizing::estimate::bytes_per_token(p.vocab_size)).round() as u64;
            Some(CorpusInfo {
                tokens: est_tokens,
                documents: 0,
                vocab: p.vocab_size,
                ram_scales_with_corpus: false,
                note: "Large text corpus (≥ 8 MB): streamed directly from disk through a temporary token file so RAM does not grow with the corpus size.".into(),
            })
        }),
        problems,
    };
    if !out.problems.is_empty() {
        return out;
    }
    // Character-level models have one token per distinct character; the estimate assumes
    // plain ASCII (128), as `estimate::train` does.
    let vocab = p
        .corpus_vocab
        .unwrap_or(if p.vocab_size == 0 { 128 } else { p.vocab_size });
    match TransformerNet::check_size(
        vocab,
        p.context_len,
        p.embed_dim,
        p.hidden_dim,
        p.num_blocks,
    ) {
        Ok(n) => out.params = Some(n as u64),
        Err(e) => {
            out.problems
                .push(format!("This model is too large to train: {e}"));
            return out;
        }
    }
    let est = estimate(JobKind::Train, p, hw, bench);
    out.peak_ram_mb = est.peak_ram_mb;
    out.ram_limit_mb = est.ram_limit_mb;
    out.fits_in_ram = est.fits_in_ram;
    out.seconds = est.seconds;
    out
}

/// Pre-flight for the Train tab: parameter count, estimated peak RAM against the free RAM, and
/// the llama.cpp badge, from the form values. Reads only file sizes; trains nothing.
#[tauri::command]
pub async fn preflight_train(
    state: State<'_, AppState>,
    params: JobParams,
) -> Result<Preflight, CmdError> {
    let bench = *crate::infra::lock(&state.bench);
    crate::infra::error::blocking_ok(move || {
        let mut params = params;
        resolve(JobKind::Train, &mut params);
        preflight(&params, &hardware::detect(), bench.as_ref())
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sizing::hardware::GpuInfo;

    fn hw(avail_mb: Option<u64>) -> HardwareInfo {
        HardwareInfo {
            cpu_model: "test cpu".into(),
            physical_cores: Some(2),
            logical_cores: 4,
            ferrum_threads: 4,
            ram_total_mb: Some(16_000),
            ram_available_mb: avail_mb,
            gpus: vec![GpuInfo {
                name: "GPU".into(),
                memory_mb: None,
                cores: None,
                used_by_ferrum: false,
            }],
        }
    }

    /// The Train tab's defaults, with a 100 KB corpus.
    fn defaults() -> JobParams {
        JobParams {
            context_len: 16,
            embed_dim: 32,
            num_heads: 4,
            num_blocks: 2,
            hidden_dim: 64,
            epochs: 100,
            batch_size: 16,
            vocab_size: 512,
            corpus_bytes: Some(100_000),
            ..JobParams::default()
        }
    }

    #[test]
    fn the_default_form_fits_and_reports_the_engines_parameter_count() {
        let p = defaults();
        let r = preflight(&p, &hw(Some(8_000)), None);
        let want = TransformerNet::check_size(512, 16, 32, 64, 2).unwrap() as u64;
        assert_eq!(r.params, Some(want));
        assert!(r.fits_in_ram && r.problems.is_empty(), "{r:?}");
        assert!(r.peak_ram_mb.is_some_and(|m| m > 0.0));
        assert!(r.seconds.is_some_and(|s| s > 0.0));
        assert_eq!(r.ram_available_mb, Some(8_000.0));
        // The limit is the engine's: three quarters of the free RAM.
        assert_eq!(r.ram_limit_mb, Some(6_000.0));
        // The same peak the engine computes for these dimensions.
        let windows = ((100_000.0 / ferrum_core_bpt(512)) as usize).saturating_sub(16 + 1) / 16 + 1;
        let engine =
            TransformerNet::estimate_train_peak_bytes(512, 16, 32, 4, 64, 2, windows, 16, 4)
                .unwrap() as f64
                / (1024.0 * 1024.0);
        let got = r.peak_ram_mb.unwrap();
        assert!((got - engine).abs() / engine < 0.05, "{got} vs {engine}");
    }

    fn ferrum_core_bpt(vocab: usize) -> f64 {
        crate::sizing::estimate::bytes_per_token(vocab)
    }

    #[test]
    fn a_run_bigger_than_the_free_ram_does_not_fit() {
        let mut p = defaults();
        p.embed_dim = 1024;
        p.hidden_dim = 4096;
        p.num_blocks = 12;
        p.num_heads = 16;
        p.context_len = 512;
        p.batch_size = 64;
        let small = preflight(&p, &hw(Some(1_000)), None);
        assert!(!small.fits_in_ram, "{small:?}");
        assert!(small.peak_ram_mb.unwrap() > small.ram_limit_mb.unwrap());
        // The same job on a machine with plenty of RAM is fine (up to the 16 GiB engine cap).
        let big = preflight(&p, &hw(Some(1_000_000)), None);
        assert_eq!(
            big.fits_in_ram,
            big.peak_ram_mb.unwrap() <= 16.0 * 1024.0,
            "{big:?}"
        );
        // Unknown RAM is never a reason to block.
        assert!(preflight(&p, &hw(None), None).fits_in_ram);
    }

    #[test]
    fn the_llama_cpp_badge_follows_the_vocab_setting() {
        let verdict = |v: usize| {
            let mut p = defaults();
            p.vocab_size = v;
            preflight(&p, &hw(Some(8_000)), None)
        };
        let char_level = verdict(0);
        assert!(!char_level.llama_cpp.runs);
        assert!(char_level.llama_cpp.reason.contains("character-level"));
        assert!(
            char_level.problems.is_empty(),
            "vocab 0 is valid, just not portable"
        );
        assert!(char_level.params.is_some());
        for v in [256, 512, 4096] {
            let r = verdict(v);
            assert!(
                r.llama_cpp.runs && r.llama_cpp.reason.contains("gpt2"),
                "{v}"
            );
        }
        let bad = verdict(100);
        assert!(!bad.llama_cpp.runs);
        assert!(bad.problems.iter().any(|m| m.contains("Vocab must be 0")));
        assert_eq!(bad.params, None, "an invalid form gets no numbers");
    }

    #[test]
    fn a_token_file_corpus_uses_its_vocabulary_and_says_ram_does_not_scale() {
        let mut p = defaults();
        p.corpus_tokens = Some(5_000_000_000);
        p.corpus_documents = Some(1_000_000);
        p.corpus_vocab = Some(1024);
        p.tokenizer_path = "tok.gguf".into();
        let r = preflight(&p, &hw(Some(8_000)), None);
        let want = TransformerNet::check_size(1024, 16, 32, 64, 2).unwrap() as u64;
        assert_eq!(
            r.params,
            Some(want),
            "the file's vocabulary, not the Vocab field (512)"
        );
        let c = r.corpus.clone().unwrap();
        assert_eq!(
            (c.tokens, c.documents, c.vocab, c.ram_scales_with_corpus),
            (5_000_000_000, 1_000_000, 1024, false)
        );
        assert!(c
            .note
            .contains("does not depend on how large the corpus is"));
        assert!(r.problems.is_empty() && r.fits_in_ram, "{r:?}");
        // 5 billion tokens or 5 thousand: the same RAM.
        p.corpus_tokens = Some(5_000);
        let small = preflight(&p, &hw(Some(8_000)), None);
        assert_eq!(small.peak_ram_mb, r.peak_ram_mb);
        assert!(r.seconds.unwrap() > small.seconds.unwrap());
        // A token file is BPE: Vocab 0 and a missing tokenizer file are problems the run would hit.
        p.vocab_size = 0;
        p.tokenizer_path = String::new();
        let bad = preflight(&p, &hw(Some(8_000)), None);
        assert_eq!(bad.problems.len(), 2, "{bad:?}");
        assert!(
            bad.problems[0].contains("BPE-tokenized") && bad.problems[1].contains("Tokenizer file")
        );
        // The llama.cpp verdict follows the file's vocabulary: it is BPE, so it runs.
        assert!(bad.llama_cpp.runs);
        // Small text corpora (< 8 MiB) have no corpus card; large BPE text corpora (>= 8 MiB)
        // report that they are streamed from disk.
        assert!(preflight(&defaults(), &hw(None), None).corpus.is_none());
        let mut big_text = defaults();
        big_text.corpus_bytes = Some(ferrum_core::STREAM_CORPUS_THRESHOLD + 1024);
        let big_pf = preflight(&big_text, &hw(Some(8_000)), None);
        let big_c = big_pf.corpus.unwrap();
        assert!(!big_c.ram_scales_with_corpus);
        assert!(big_c.note.contains("streamed directly from disk"));
    }

    #[test]
    fn invalid_forms_list_their_problems_instead_of_numbers() {
        let mut p = defaults();
        p.num_heads = 3; // 32 is not divisible by 3
        let r = preflight(&p, &hw(Some(8_000)), None);
        assert!(
            r.problems.iter().any(|m| m.contains("divisible by heads")),
            "{r:?}"
        );
        assert!(r.params.is_none() && r.peak_ram_mb.is_none() && r.fits_in_ram);

        let mut p = defaults();
        p.epochs = 0;
        assert!(!preflight(&p, &hw(None), None).problems.is_empty());
        p.max_tokens = 1_000_000; // a token budget replaces the epochs
        assert!(preflight(&p, &hw(None), None).problems.is_empty());

        let mut p = defaults();
        p.num_blocks = 0;
        assert!(preflight(&p, &hw(None), None)
            .problems
            .iter()
            .any(|m| m.contains("at least 1")));

        // Absurd sizes are an error message, not an allocation or a panic.
        let mut p = defaults();
        p.embed_dim = 1 << 20;
        p.hidden_dim = 1 << 20;
        p.num_heads = 4;
        let r = preflight(&p, &hw(Some(8_000)), None);
        assert!(r.problems.iter().any(|m| m.contains("too large")), "{r:?}");
        assert!(r.params.is_none());
    }

    #[test]
    fn json_uses_the_names_the_ui_reads() {
        let v = serde_json::to_value(preflight(&defaults(), &hw(Some(8_000)), None)).unwrap();
        for k in [
            "params",
            "peakRamMb",
            "ramAvailableMb",
            "ramLimitMb",
            "fitsInRam",
            "seconds",
            "llamaCpp",
            "problems",
        ] {
            assert!(v.get(k).is_some(), "{k} missing in {v}");
        }
        assert_eq!(v["llamaCpp"]["runs"], true);
    }
}
