//! The wasm module itself, run under Node by `wasm-bindgen-test-runner` (CI job `wasm`):
//!
//! ```text
//! CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!   cargo test -p slm_wasm --target wasm32-unknown-unknown
//! ```
//!
//! On a native target this file is empty, so `cargo test --workspace` is unaffected. The tiny
//! model is trained inside the test from a fixed seed (nothing to download or commit). The
//! tokenizer ids are integer-only and pinned to the values a native run gives; the generated
//! texts and the score were also taken from a native run, so a mismatch means wasm and native
//! numerics (libm) differ.

#![cfg(target_arch = "wasm32")]

use ferrum_core::{GenerativeSLM, Rng, SamplingParams, TransformerConfig};
use slm_wasm::Slm;
use wasm_bindgen_test::*;

/// A tiny int8 BPE model (as the browser would fetch it) and the library view of the same bytes.
fn model() -> (Vec<u8>, GenerativeSLM) {
    let cfg = TransformerConfig {
        context_len: 16,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 2,
        hidden_dim: 32,
        epochs: 3,
        vocab_size: 280,
        ..TransformerConfig::default()
    };
    let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(20);
    let m = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(3), |_, _| {})
        .unwrap();
    let bytes = m.to_bytes_quantized().unwrap();
    let lib = GenerativeSLM::from_bytes(&bytes).unwrap();
    (bytes, lib)
}

fn sampling(t: f32, k: usize, p: f32) -> SamplingParams {
    SamplingParams {
        temperature: t,
        top_k: k,
        top_p: p,
        repetition_penalty: 1.0,
    }
}

#[wasm_bindgen_test]
fn loads_and_describes_the_model() {
    let (bytes, lib) = model();
    let slm = Slm::new(&bytes).unwrap();
    let info = slm.info();
    assert!(info.contains("\"tokenizer\":\"bpe\""), "{info}");
    assert!(info.contains("\"quant\":\"int8\""), "{info}");
    assert!(
        info.contains(&format!("\"vocab\":{}", lib.meta.output_dim)),
        "{info}"
    );
    assert!(info.contains("\"context\":16"), "{info}");
}

#[wasm_bindgen_test]
fn encode_and_decode_match_the_native_values() {
    let (bytes, _) = model();
    let slm = Slm::new(&bytes).unwrap();
    let ids = slm.encode("the cat sat on the mat.").unwrap();
    assert_eq!(ids, [258, 269, 264, 263, 259, 272, 46]);
    assert_eq!(slm.decode(&ids).unwrap(), "the cat sat on the mat.");
}

#[wasm_bindgen_test]
fn generate_matches_the_native_text_and_the_library() {
    let (bytes, lib) = model();
    let slm = Slm::new(&bytes).unwrap();
    let cases = [
        (0.0, 0, 1.0, 1, "...... the...... the. the...... the."),
        (
            0.8,
            5,
            0.9,
            7,
            ". the the mat the mat log log. sat on on. on mat mat. mat mat mat the.. the",
        ),
    ];
    for (t, k, p, seed, native) in cases {
        let got = slm
            .generate("the cat", 24, t, k, p, 1.0, seed, vec![])
            .unwrap();
        let want = lib
            .generate_tokens(
                "the cat",
                24,
                &[],
                &sampling(t, k, p),
                &mut Rng::new(seed as u64),
            )
            .unwrap()
            .text;
        assert_eq!(got, want, "wasm Slm vs the library inside wasm");
        assert_eq!(got, native, "wasm vs the text a native run gave");
    }
}

#[wasm_bindgen_test]
fn streaming_adds_up_to_generate_and_cancel_stops() {
    let (bytes, _) = model();
    let mut slm = Slm::new(&bytes).unwrap();
    let whole = slm
        .generate("the cat", 24, 0.8, 5, 0.9, 1.0, 7, vec![])
        .unwrap();
    slm.start("the cat", 24, 0.8, 5, 0.9, 1.0, 7, vec![])
        .unwrap();
    let mut streamed = String::new();
    while let Some(c) = slm.next_chunk().unwrap() {
        assert!(!c.is_empty());
        streamed.push_str(&c);
    }
    assert_eq!(streamed, whole);
    assert_eq!(slm.next_chunk().unwrap(), None);

    slm.start("the cat", 24, 0.8, 5, 0.9, 1.0, 7, vec![])
        .unwrap();
    assert!(slm.next_chunk().unwrap().is_some());
    slm.cancel();
    assert_eq!(slm.next_chunk().unwrap(), None);
    assert!(slm.tokens_generated() < 24);
}

#[wasm_bindgen_test]
fn score_matches_the_library() {
    let (bytes, lib) = model();
    let slm = Slm::new(&bytes).unwrap();
    let got = slm.score("the cat sat").unwrap();
    assert!((got - lib.score_rolling("the cat sat").unwrap().logprob).abs() < 1e-9);
    assert!((got - -14.723644989356476).abs() < 1e-3, "{got}");
}

#[wasm_bindgen_test]
fn bad_input_is_a_js_error_not_a_trap() {
    let (bytes, _) = model();
    assert!(Slm::new(b"not a model").is_err());
    assert!(Slm::new(&[]).is_err());
    let slm = Slm::new(&bytes).unwrap();
    assert!(slm
        .generate("the", 5, -1.0, 0, 1.0, 1.0, 1, vec![])
        .is_err());
    assert!(slm.generate("the", 5, 0.8, 0, 2.0, 1.0, 1, vec![]).is_err());
}
