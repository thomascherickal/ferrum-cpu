//! `info`: describe a GGUF file (a Ferrum model, an imported llama/qwen model, a checkpoint, or a token file).

use crate::args::*;
use ferrum_core::{GenerativeSLM, Gguf, GgufTokenizer, MetaValue, TransformerNet};

/// The quantization of an imported file: the ggml type holding most of its weights.
fn dominant_quant(g: &Gguf) -> &'static str {
    let mut by_type = std::collections::BTreeMap::<u32, usize>::new();
    for t in &g.tensors {
        let n = by_type.entry(t.ggml_type).or_default();
        *n = n.saturating_add(t.num_elements());
    }
    match by_type.into_iter().max_by_key(|&(_, n)| n).map(|(t, _)| t) {
        Some(0) => "f32",
        Some(1) => "f16",
        Some(2) => "q4_0",
        Some(3) => "q4_1",
        Some(6) => "q5_0",
        Some(7) => "q5_1",
        Some(8) => "q8_0",
        Some(9) => "q8_1",
        Some(10) => "q2_k",
        Some(11) => "q3_k",
        Some(12) => "q4_k",
        Some(13) => "q5_k",
        Some(14) => "q6_k",
        Some(30) => "bf16",
        Some(_) => "other",
        None => "unknown",
    }
}

/// An imported (llama/qwen-style) GGUF, read from its header only.
fn info_imported(path: &str, g: &Gguf, arch: &str, bytes: u64) {
    let n = |k: &str| g.meta(&format!("{arch}.{k}")).and_then(MetaValue::as_usize);
    let show = |v: Option<usize>| v.map_or("?".to_string(), |v| v.to_string());
    let params = g
        .tensors
        .iter()
        .fold(0usize, |a, t| a.saturating_add(t.num_elements()));
    let vocab = g
        .tensor("token_embd.weight")
        .filter(|t| t.dims.len() == 2)
        .map(|t| t.dims[1] as usize)
        .or_else(|| n("vocab_size"));
    let heads = n("attention.head_count");
    println!("Model     : {path}  ({bytes} bytes)");
    println!(
        "Format    : GGUF v{} ({arch}, {})",
        g.version,
        dominant_quant(g)
    );
    println!("Params    : {params}");
    println!("Layers    : {}", show(n("block_count")));
    println!("Dim       : {}", show(n("embedding_length")));
    println!(
        "Heads     : {} ({} KV)",
        show(heads),
        show(n("attention.head_count_kv").or(heads))
    );
    println!("Vocab     : {}", show(vocab));
    println!("Context   : {}", show(n("context_length")));
    match GgufTokenizer::from_gguf(g) {
        Ok(t) => println!("Tokenizer : {:?} ({} tokens)", t.model(), t.vocab_size()),
        Err(_) => println!("Tokenizer : none in file (use run-gguf --ids)"),
    }
    if matches!(arch, "llama" | "qwen2") {
        println!("Run it    : train_transformer run-gguf {path} \"<prompt>\"");
    } else {
        println!("Note      : architecture '{arch}' is not runnable (llama / qwen2 only)");
    }
}

pub(crate) fn cmd_info(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.is_empty() {
        return Err("usage: train_transformer info <model.gguf>".into());
    }
    let model_path = &args.positional[0];
    // The header alone says what kind of file this is; anything unreadable falls through to
    // the Ferrum loader below, which reports the precise error.
    if let Ok(g) = Gguf::open(model_path) {
        let size = std::fs::metadata(model_path).map_or(0, |m| m.len());
        match g.architecture() {
            Some("gpt2" | "ferrum") | None => {}
            Some("ferrum-checkpoint") => {
                let c = TransformerNet::checkpoint_info(std::fs::read(model_path)?)?;
                println!("Model     : {model_path}  ({size} bytes)");
                println!("Format    : GGUF v{} (ferrum-checkpoint)", g.version);
                println!("Kind      : training checkpoint; continue it with train --resume");
                println!("Vocab     : {}", c.vocab);
                println!("Context   : {}", c.context);
                println!("Embed     : {}", c.embed);
                println!("Heads     : {}", c.heads);
                println!("Hidden    : {}", c.hidden);
                println!("Blocks    : {}", c.blocks);
                println!("Step      : {}", c.step);
                return Ok(());
            }
            Some("ferrum-tokens") => {
                let f = ferrum_core::TokenFile::open(model_path)?;
                println!("Model     : {model_path}  ({size} bytes)");
                println!("Format    : GGUF v{} (ferrum-tokens)", g.version);
                println!(
                    "Kind      : pre-tokenized corpus; train on it with train {model_path} ..."
                );
                println!("Tokens    : {}", f.len());
                println!("Documents : {}", f.documents());
                println!("Vocab     : {}", f.vocab_size());
                match f.eos_id() {
                    Some(e) => println!("EOS id    : {e}"),
                    None => println!("EOS id    : none"),
                }
                println!("Tokenizer : sha256 {}", f.tokenizer_sha256());
                return Ok(());
            }
            Some("ferrum-llama-checkpoint") => {
                println!("Model     : {model_path}  ({size} bytes)");
                println!("Format    : GGUF v{} (ferrum-llama-checkpoint)", g.version);
                println!("Kind      : fine-tune checkpoint; apply it with run-gguf --resume");
                return Ok(());
            }
            Some(arch) => {
                info_imported(model_path, &g, arch, size);
                return Ok(());
            }
        }
    }
    let bytes =
        std::fs::read(model_path).map_err(|e| format!("cannot read model {model_path}: {e}"))?;
    let slm = GenerativeSLM::from_bytes(&bytes)?;
    let m = &slm.meta;
    println!("Model     : {model_path}  ({} bytes)", bytes.len());
    println!("Format    : {}", ferrum_core::loader::gguf_format(&bytes)?);
    println!("Params    : {}", slm.model.num_params());
    println!("Name      : {}", m.dataset_name);
    println!("Task      : {:?}", m.task);
    println!("Input dim : {}", m.input_dim);
    println!("Output dim: {}", m.output_dim);
    if m.tokenizer_state.is_empty() {
        println!(
            "Tokenizer : character-level ({} chars)",
            m.class_names.len()
        );
    } else {
        let merges = m
            .tokenizer_state
            .split(';')
            .filter(|s| !s.is_empty())
            .count();
        println!(
            "Tokenizer : byte-level BPE ({} tokens, {} merges)",
            m.output_dim, merges
        );
    }
    println!("Layers    : {}", slm.model.len());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// run-gguf: import & run an external llama/qwen2 GGUF checkpoint (G-CLI)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{GgufBuilder, MetaValue};

    fn open_with(tag: &str, tensors: &[(&str, u32, u64)]) -> Gguf {
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        for &(name, ty, n) in tensors {
            b.tensor(name, &[n], ty, vec![0; 4096]);
        }
        let path =
            std::env::temp_dir().join(format!("ferrum_info_{tag}_{}.gguf", std::process::id()));
        std::fs::write(&path, b.into_bytes()).unwrap();
        let g = Gguf::open(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        g
    }

    #[test]
    fn info_reports_dominant_quant_for_every_ggml_type() {
        for (ty, label) in [
            (0, "f32"),
            (1, "f16"),
            (2, "q4_0"),
            (3, "q4_1"),
            (6, "q5_0"),
            (7, "q5_1"),
            (8, "q8_0"),
            (9, "q8_1"),
            (10, "q2_k"),
            (11, "q3_k"),
            (12, "q4_k"),
            (13, "q5_k"),
            (14, "q6_k"),
            (30, "bf16"),
            (99, "other"),
        ] {
            // The type holding most elements wins, however many small tensors disagree.
            let g = open_with(&format!("t{ty}"), &[("small", 0, 1), ("big", ty, 256)]);
            assert_eq!(
                dominant_quant(&g),
                if ty == 0 { "f32" } else { label },
                "type {ty}"
            );
        }
        assert_eq!(dominant_quant(&open_with("none", &[])), "unknown");
    }
}
