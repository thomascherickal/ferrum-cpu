//! `info`: describe a GGUF file.

use crate::args::*;
use ferrum_core::GenerativeSLM;

pub(crate) fn cmd_info(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.is_empty() {
        return Err("usage: train_transformer info <model.gguf>".into());
    }
    let model_path = &args.positional[0];
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
