//! `export-gguf`: re-quantize / convert a GGUF.

use crate::args::*;
use crate::memory::*;
use crate::validate::*;

pub(crate) fn cmd_export_gguf(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use ferrum_core::{Gguf, GgufQuant};

    if args.positional.len() < 2 {
        return Err("usage: train_transformer export-gguf <in.gguf> <out.gguf> \
                    [--quant q8_0|q4_0|q4_1|q8_1|q4_k|q5_k|q6_k|f16|f32] \
                    [--resume tuned.flck] [--force]"
            .into());
    }
    require_resume_file(args.value("resume"), ".flck")?;
    let in_path = &args.positional[0];
    let out_path = &args.positional[1];
    check_writable(out_path)?;
    let quant_name = args.value("quant").unwrap_or("q8_0");
    let quant =
        GgufQuant::from_str(quant_name).ok_or_else(|| format!("unknown --quant '{quant_name}'"))?;

    println!("Opening {in_path} (streamed)…");
    let g = Gguf::open(in_path).map_err(|e| format!("cannot open GGUF {in_path}: {e}"))?;
    println!(
        "  GGUF v{}   architecture = {}",
        g.version,
        g.architecture().unwrap_or("?")
    );

    // Whitespace-only --resume means "none".
    let resume = args
        .value("resume")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty());

    // Export loads f32 masters AND builds the whole encoded output in memory
    // before the atomic write, so peak ≈ model + output buffer (which for a
    // lossless f32 target is another full model's worth). Applying a --resume
    // checkpoint has its own, separate transient peak: the checkpoint bytes
    // plus one decoded parameter on top of the model, freed before the write.
    let est_model = estimate_resident_bytes(&g, None);
    let est_out = match quant {
        GgufQuant::F32 => est_model,
        GgufQuant::F16 => est_model / 2,
        _ => est_model / 3, // quantized outputs are smaller; /3 is conservative
    };
    let write_peak = est_model.saturating_add(est_out);
    let est = match resume {
        Some(ckpt) => resume_peak_bytes(est_model, file_len(ckpt)).max(write_peak),
        None => write_peak,
    };
    if resume.is_some() {
        println!(
            "  estimated peak ≈ {:.2} GB  (checkpoint apply: model + checkpoint bytes + one parameter)",
            est as f64 / 1e9
        );
    } else {
        println!(
            "  estimated peak ≈ {:.2} GB  (f32 model {:.2} + output buffer {:.2})",
            est as f64 / 1e9,
            est_model as f64 / 1e9,
            est_out as f64 / 1e9
        );
    }
    if let Some(avail) = available_memory_bytes() {
        if let Some(e) = memory_error(
            "peak memory",
            est,
            avail,
            args.has("force"),
            "pass --force to attempt it anyway.",
        ) {
            return Err(e.into());
        }
    }

    println!("Loading weights (f32)…");
    let mut model = g
        .load_llama_prec(None)
        .map_err(|e| format!("cannot load model: {e}"))?;

    // Optional: apply a fine-tune checkpoint's f32 masters before export.
    if let Some(ckpt) = resume {
        println!("Applying fine-tune checkpoint {ckpt}…");
        let bytes = std::fs::read(ckpt)?;
        ferrum_core::apply_checkpoint_weights(&mut model, bytes)?;
    }

    println!("Writing {out_path}  (--quant {quant_name})…");
    ferrum_core::write_llama_gguf(&model, &g, quant, out_path)
        .map_err(|e| format!("export failed: {e}"))?;

    // Per-type tensor summary, read back from the written file — this both
    // shows what was actually emitted (block-alignment fallbacks appear as
    // F16 counts) and proves the output re-opens.
    let gout = Gguf::open(out_path).map_err(|e| format!("written file failed to re-open: {e}"))?;
    let mut counts: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for t in &gout.tensors {
        *counts.entry(t.ggml_type).or_default() += 1;
    }
    let summary: Vec<String> = counts
        .iter()
        .map(|(ty, n)| format!("{n}× {}", ggml_type_name(*ty)))
        .collect();
    println!("  tensors: {}", summary.join(", "));

    let sz = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    println!("Done: {out_path}  ({:.2} MB)", sz as f64 / 1e6);
    Ok(())
}

/// Human name for the GGML tensor-type ids the exporter can emit.
pub(crate) fn ggml_type_name(t: u32) -> String {
    match t {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        6 => "Q5_0".into(),
        7 => "Q5_1".into(),
        8 => "Q8_0".into(),
        9 => "Q8_1".into(),
        12 => "Q4_K".into(),
        13 => "Q5_K".into(),
        14 => "Q6_K".into(),
        other => format!("type{other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_type_names_cover_every_exportable_type_and_fall_back() {
        let want = [
            (0, "F32"),
            (1, "F16"),
            (2, "Q4_0"),
            (3, "Q4_1"),
            (6, "Q5_0"),
            (7, "Q5_1"),
            (8, "Q8_0"),
            (9, "Q8_1"),
            (12, "Q4_K"),
            (13, "Q5_K"),
            (14, "Q6_K"),
            (4, "type4"),
            (999, "type999"),
        ];
        for (id, name) in want {
            assert_eq!(ggml_type_name(id), name, "id {id}");
        }
    }
}
