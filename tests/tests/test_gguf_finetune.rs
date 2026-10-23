//! End-to-end test of the GGUF import → **fine-tune** → checkpoint → resume →
//! run pipeline that `slm_cli finetune-gguf` / `run-gguf --resume` drive: write a
//! synthetic `llama` GGUF, `load_llama_prec(None)`, wrap it in a `LlamaTrainer`,
//! run a few AdamW epochs (loss must fall), save a checkpoint, reload it over a
//! fresh base model, and confirm the resumed weights match and still generate.

use ferrum_core::{Adam, Gguf, LlamaTrainer, Rng, SamplingParams};

#[path = "synth/mod.rs"]
mod synth;
use synth::synth_llama;

#[test]
fn gguf_finetune_checkpoint_resume_pipeline() {
    let gguf = synth_llama();
    let gpath = std::env::temp_dir().join(format!("ferrum_ft_{}.gguf", std::process::id()));
    let cpath = std::env::temp_dir().join(format!("ferrum_ft_{}.flck", std::process::id()));
    std::fs::write(&gpath, &gguf).unwrap();

    // A learnable periodic token stream over the in-vocab ids (matches how the
    // CLI tokenizes a corpus, minus the tokenizer step).
    let tokens: Vec<usize> = (0..120).map(|i| 1 + (i % 6)).collect();
    let (seq, batch) = (8usize, 8usize);

    // 1. Import f32 and wrap in a trainer (the finetune-gguf load path).
    let g = Gguf::open(gpath.to_str().unwrap()).unwrap();
    let model = g.load_llama_prec(None).unwrap();
    let mut tr = LlamaTrainer::new(model).unwrap();
    tr.set_optimizer(Adam::new(0.01));
    tr.set_grad_clip(Some(1.0));

    // 2. Fine-tune: loss must fall.
    let mut rng = Rng::new(1337);
    let first = tr.finetune_epoch(&tokens, seq, batch, &mut rng).unwrap();
    let mut last = first;
    for _ in 0..12 {
        last = tr.finetune_epoch(&tokens, seq, batch, &mut rng).unwrap();
    }
    assert!(
        last < first * 0.8,
        "fine-tune did not reduce loss: {first:.4} → {last:.4}"
    );
    let trained = tr.model_snapshot();
    let trained_steps = tr.step_count();

    // 3. Save a checkpoint (the artifact finetune-gguf writes).
    let ckpt = tr.save_checkpoint(&rng);
    std::fs::write(&cpath, &ckpt).unwrap();

    // 4. Resume over a *fresh* base model (the run-gguf --resume path): reload the
    //    base GGUF f32, apply the checkpoint, and confirm the weights + step count
    //    match the trained model bit-for-bit.
    let base2 = g.load_llama_prec(None).unwrap();
    let mut tr2 = LlamaTrainer::new(base2).unwrap();
    let bytes = std::fs::read(&cpath).unwrap();
    let _resumed_rng = tr2.load_checkpoint_into(&bytes).unwrap();
    assert_eq!(
        tr2.step_count(),
        trained_steps,
        "resumed step counter mismatch"
    );
    let resumed = tr2.model_snapshot();
    assert_eq!(resumed.len(), trained.len());
    for (a, b) in trained.iter().zip(&resumed) {
        assert_eq!(a, b, "resumed weights differ from the trained model");
    }

    // 5. The resumed model generates in-vocab, deterministic tokens.
    let params = SamplingParams::with_temperature(0.8);
    let prompt = [1usize, 2, 3];
    let stops: Vec<usize> = Some(9).into_iter().collect();
    let out = tr2
        .model
        .generate(&prompt, 8, &params, &stops, &mut Rng::new(7))
        .unwrap();
    assert!(out.iter().all(|&t| t < tr2.model.cfg.vocab_size));
    let again = tr2
        .model
        .generate(&prompt, 8, &params, &stops, &mut Rng::new(7))
        .unwrap();
    assert_eq!(
        out, again,
        "generation must be deterministic for a fixed seed"
    );

    let _ = std::fs::remove_file(&gpath);
    let _ = std::fs::remove_file(&cpath);
}
