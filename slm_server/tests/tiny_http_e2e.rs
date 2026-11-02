//! A tiny trained model served by the real `ferrum-serve` binary over TCP: every route is
//! checked against the library on the same model file, plus the numeric limit flags and a clean
//! SIGINT shutdown.
mod common;

use common::{http, sigint_and_wait, spawn, BIN};
use ferrum_core::{GenerativeSLM, Rng, SamplingParams, TransformerConfig};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::Command;

/// A model file in the temp dir, removed on drop.
struct ModelFile {
    path: PathBuf,
    bytes: Vec<u8>,
}

impl Drop for ModelFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl ModelFile {
    /// Train a BPE model (vocab 270, 2 blocks, context 16) with the library and save it.
    fn new(tag: &str) -> ModelFile {
        let cfg = TransformerConfig {
            context_len: 16,
            embed_dim: 16,
            num_heads: 2,
            num_blocks: 2,
            hidden_dim: 32,
            epochs: 3,
            vocab_size: 270,
            ..TransformerConfig::default()
        };
        let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(20);
        let m = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(3), |_, _| {})
            .unwrap();
        let bytes = m.to_bytes_quantized().unwrap();
        let path = std::env::temp_dir().join(format!(
            "ferrum-serve-tiny-{}-{tag}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        ModelFile { path, bytes }
    }

    /// The model exactly as the server loads it.
    fn reference(&self) -> GenerativeSLM {
        GenerativeSLM::from_bytes(&self.bytes).unwrap()
    }
}

fn json_of(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|_| panic!("not JSON: {body:?}"))
}

fn post(addr: &str, path: &str, body: &Value) -> (u16, Value) {
    let (s, b) = http(addr, "POST", path, &body.to_string());
    (s, json_of(&b))
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..SamplingParams::default()
    }
}

#[test]
fn tiny_model_over_http_matches_the_library_and_shuts_down_cleanly() {
    let mf = ModelFile::new("api");
    let reference = mf.reference();
    let (mut child, _err, addr) = spawn(&mf.path, &["--max-body", "4096"]);

    assert_eq!(http(&addr, "GET", "/health", ""), (200, "\"ok\"".into()));

    let (s, b) = http(&addr, "GET", "/v1/info", "");
    assert_eq!(s, 200);
    let info = json_of(&b);
    let sha: String = Sha256::digest(&mf.bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(info["model_sha256"], sha);
    assert_eq!(info["params"], reference.model.num_params());
    assert_eq!(info["context_len"], 16);
    assert_eq!(info["vocab"], 270);
    assert_eq!(info["quant"], "int8");
    assert_eq!(info["tokenizer"], "bpe");

    // loglikelihood: finite, equal to the library, and the likely continuation wins.
    let (s, v) = post(
        &addr,
        "/v1/loglikelihood",
        &json!({"requests": [
            {"context": "the cat sat on", "continuation": " the mat."},
            {"context": "the cat sat on", "continuation": " zebra"}]}),
    );
    assert_eq!(s, 200, "{v}");
    let r = v["results"].as_array().unwrap();
    let want = reference.score("the cat sat on", " the mat.").unwrap();
    let got = r[0]["logprob"].as_f64().unwrap();
    assert!(got.is_finite() && got < 0.0);
    assert!(
        (got - want.logprob).abs() < 1e-6,
        "{got} vs {}",
        want.logprob
    );
    assert_eq!(r[0]["n_tokens"], want.n_tokens);
    assert!(got > r[1]["logprob"].as_f64().unwrap());

    // rolling: a text longer than the context window, scored whole.
    let text = "the dog sat on the log. ".repeat(6);
    let (s, v) = post(
        &addr,
        "/v1/loglikelihood_rolling",
        &json!({"texts": [text.clone()]}),
    );
    assert_eq!(s, 200, "{v}");
    let want = reference.score_rolling(&text).unwrap();
    assert!(want.n_tokens > 16);
    let got = v["results"][0]["logprob"].as_f64().unwrap();
    assert!(got.is_finite() && (got - want.logprob).abs() < 1e-6);
    assert_eq!(v["results"][0]["n_tokens"], want.n_tokens);

    // generate: greedy equals the library; a sampled request with a seed equals it too.
    let (s, v) = post(
        &addr,
        "/v1/generate",
        &json!({"requests": [{"prompt": "the cat", "max_tokens": 20}]}),
    );
    assert_eq!(s, 200, "{v}");
    let want = reference
        .generate_tokens("the cat", 20, &[], &greedy(), &mut Rng::new(0))
        .unwrap();
    assert_eq!(v["results"][0]["text"], want.text);
    assert_eq!(v["results"][0]["n_tokens"], want.n_tokens);
    let sampled = SamplingParams {
        temperature: 0.8,
        top_k: 5,
        ..greedy()
    };
    let want = reference
        .generate_tokens("the cat", 15, &[], &sampled, &mut Rng::new(7))
        .unwrap();
    let (_, v) = post(
        &addr,
        "/v1/generate",
        &json!({"requests": [{"prompt": "the cat", "max_tokens": 15,
            "temperature": 0.8, "top_k": 5, "seed": 7}]}),
    );
    assert_eq!(v["results"][0]["text"], want.text);

    // Bad requests are errors with a JSON body, and the server keeps answering.
    let (s, b) = http(&addr, "POST", "/v1/generate", "{not json");
    assert_eq!(s, 400);
    assert!(json_of(&b)["error"].is_string());
    let huge = json!({"requests": [{"prompt": "x".repeat(8000)}]}).to_string();
    let (s, b) = http(&addr, "POST", "/v1/generate", &huge);
    assert_eq!(s, 413);
    assert!(json_of(&b)["error"].is_string());
    assert_eq!(http(&addr, "GET", "/health", "").0, 200);

    let status = sigint_and_wait(&mut child);
    assert!(status.success(), "{status:?}");
}

#[test]
fn limit_flags_reach_the_running_server() {
    let mf = ModelFile::new("limits");
    let (mut child, _err, addr) = spawn(
        &mf.path,
        &[
            "--max-batch",
            "2",
            "--max-chars",
            "40",
            "--max-tokens",
            "10",
            "--max-body",
            "1500",
        ],
    );
    let gen = |reqs: Vec<Value>| post(&addr, "/v1/generate", &json!({ "requests": reqs }));
    let one = |prompt: &str, n: u32| json!({"prompt": prompt, "max_tokens": n});

    // Exactly at each limit is fine, one over is a 400 naming the limit.
    assert_eq!(gen(vec![one("a", 10), one("b", 10)]).0, 200);
    let (s, v) = gen(vec![one("a", 1), one("b", 1), one("c", 1)]);
    assert_eq!(s, 400);
    assert!(v["error"].as_str().unwrap().contains("max_batch"), "{v}");
    assert_eq!(gen(vec![one(&"p".repeat(40), 1)]).0, 200);
    let (s, v) = gen(vec![one(&"p".repeat(41), 1)]);
    assert_eq!(s, 400);
    assert!(v["error"].as_str().unwrap().contains("max_chars"), "{v}");
    let (s, v) = gen(vec![one("a", 11)]);
    assert_eq!(s, 400);
    assert!(v["error"].as_str().unwrap().contains("max_tokens"), "{v}");
    // A body past --max-body is refused before it is parsed.
    let big = json!({"requests": [one(&"p".repeat(2000), 1)]}).to_string();
    assert_eq!(http(&addr, "POST", "/v1/generate", &big).0, 413);
    assert_eq!(http(&addr, "GET", "/health", "").0, 200);

    assert!(sigint_and_wait(&mut child).success());
}

#[test]
fn zero_or_malformed_numeric_flags_exit_one_before_the_model_is_read() {
    for flag in ["--max-batch", "--max-chars", "--max-body", "--max-tokens"] {
        let o = Command::new(BIN)
            .args(["--model", "/nonexistent/m.gguf", flag, "0"])
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&o.stderr);
        assert_eq!(o.status.code(), Some(1), "{flag} 0");
        assert!(
            err.contains(&format!("{flag} must be at least 1 (got 0)")),
            "{err}"
        );
        assert!(!err.contains("cannot read"), "{err}");
        for bad in ["abc", "1.5", "-1", "99999999999999999999999"] {
            let o = Command::new(BIN)
                .args(["--model", "/nonexistent/m.gguf", flag, bad])
                .output()
                .unwrap();
            assert_eq!(o.status.code(), Some(1), "{flag} {bad}");
            let err = String::from_utf8_lossy(&o.stderr);
            assert!(
                err.starts_with("error: ") && !err.contains("cannot read"),
                "{err}"
            );
        }
    }
    let o = Command::new(BIN)
        .args(["--model", "/nonexistent/m.gguf", "--port", "70000"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1));
}
