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
