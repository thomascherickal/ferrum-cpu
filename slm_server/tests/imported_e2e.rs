//! Imported Llama/Qwen GGUFs served by the real `ferrum-serve` binary: a synthetic tiny llama
//! (stub weights, fast) on every route against `LlamaModel` + `GgufTokenizer` on the same
//! file, the `--quant`/`--force` flags and the startup refusals, plus one check on the real
//! SmolLM2-135M when `../.ferrum-dev/models/smollm2-135m-q8_0.gguf` exists (skipped loudly
//! otherwise).
mod common;

use common::llama::{tiny_llama, tiny_llama_bos_eos, BOS, EOS};
use common::{http, sigint_and_wait, spawn, BIN};
use ferrum_core::{
    Gguf, GgufBuilder, GgufTokenizer, LlamaModel, MetaValue, QKind, Rng, SamplingParams,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

const DEV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../.ferrum-dev");

/// A GGUF in the temp dir, removed on drop.
struct File(PathBuf);

impl File {
    fn new(tag: &str, bytes: &[u8]) -> File {
        let p = std::env::temp_dir().join(format!(
            "ferrum-serve-imported-{}-{tag}.gguf",
            std::process::id()
        ));
        std::fs::write(&p, bytes).unwrap();
        File(p)
    }
}

impl Drop for File {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The library side: the model and tokenizer exactly as the server loads them.
struct Reference {
    model: LlamaModel,
    tok: GgufTokenizer,
}

impl Reference {
    fn new(path: &Path, prec: Option<QKind>) -> Reference {
        let g = Gguf::open(path.to_str().unwrap()).unwrap();
        Reference {
            tok: GgufTokenizer::from_gguf(&g).unwrap(),
            model: g.load_llama_prec(prec).unwrap(),
        }
    }

    /// Generation as documented: `prefix_bos` + the prompt, cut from the left (BOS kept) to
    /// leave room for `max` tokens; EOS stops and is dropped.
    fn generate(&self, prompt: &str, max: usize, params: &SamplingParams, seed: u64) -> Value {
        let mut ids: Vec<usize> = self.tok.prefix_bos().into_iter().collect();
        let n_bos = ids.len();
        ids.extend(self.tok.encode(prompt));
        let ctx = self.model.cfg.context_len;
        let budget = if max >= ctx {
            ctx - 1
        } else {
            (ctx - max).clamp(1, ctx - 1)
        };
        let keep = budget.saturating_sub(n_bos).max(1);
        if ids.len() > n_bos + keep {
            ids.drain(n_bos..ids.len() - keep);
        }
        let eos: Vec<usize> = self.tok.eos().into_iter().collect();
        let mut out = self
            .model
            .generate(&ids, max, params, &eos, &mut Rng::new(seed))
            .unwrap();
        if out.last().is_some_and(|t| eos.contains(t)) {
            out.pop();
        }
        json!({"text": self.tok.decode(&out), "n_tokens": out.len()})
    }

    /// Scoring as documented: `prefix_bos()` (as for generation) + the text.
    fn ids(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<usize> = self.tok.prefix_bos().into_iter().collect();
        ids.extend(self.tok.encode(text));
        ids
    }

    fn rolling(&self, text: &str) -> Value {
        let ids = self.ids(text);
        let (s, _) = self
            .model
            .score_from(&ids, 1, self.model.cfg.context_len)
            .unwrap();
        json!({"logprob": s.logprob, "n_tokens": s.n_tokens})
    }

    fn loglikelihood(&self, context: &str, continuation: &str) -> Value {
        let ids = self.ids(&format!("{context}{continuation}"));
        let from = self.ids(context).len();
        let (s, greedy) = self
            .model
            .score_from(&ids, from, self.model.cfg.context_len)
            .unwrap();
        json!({"logprob": s.logprob, "n_tokens": s.n_tokens, "greedy": greedy})
    }
}

fn post(addr: &str, path: &str, body: &Value) -> (u16, Value) {
    let (s, b) = http(addr, "POST", path, &body.to_string());
    (s, serde_json::from_str(&b).unwrap())
}

fn results(addr: &str, path: &str, body: Value) -> Vec<Value> {
    let (s, v) = post(addr, path, &body);
    assert_eq!(s, 200, "{path}: {v}");
    v["results"].as_array().unwrap().clone()
}

/// Score objects agree: equal keys and counts, and log-probabilities equal up to JSON's float
/// parsing (serde_json reads floats to within an ULP or so, not always exactly).
fn same_scores(got: &[Value], want: &[Value]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        let (mut g, mut w) = (g.clone(), w.clone());
        let (a, b) = (g["logprob"].take(), w["logprob"].take());
        let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
        assert_eq!(g, w);
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

#[test]
fn tiny_llama_over_http_matches_the_library() {
    let bytes = tiny_llama_bos_eos();
    let f = File::new("f32", &bytes);
    let r = Reference::new(&f.0, None);
    assert_eq!(
        (r.tok.bos(), r.tok.eos()),
        (Some(BOS as usize), Some(EOS as usize))
    );
    let (mut child, _err, addr) = spawn(&f.0, &["--quant", "f32"]);

    let (s, info) = http(&addr, "GET", "/v1/info", "");
    let info: Value = serde_json::from_str(&info).unwrap();
    assert_eq!(s, 200);
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        info,
        json!({"params": 1832, "context_len": 16, "vocab": 32, "quant": "f32",
               "tokenizer": "bpe", "model_sha256": sha})
    );

    // Greedy and seeded sampling, with max_tokens and the context cap.
    let sampled = SamplingParams {
        temperature: 0.9,
        top_k: 8,
        top_p: 0.95,
        repetition_penalty: 1.2,
        ..SamplingParams::default()
    };
    let got = results(
        &addr,
        "/v1/generate",
        json!({"requests": [
            {"prompt": "ab", "max_tokens": 6},
            {"prompt": "the", "max_tokens": 99},
            {"prompt": "ab", "max_tokens": 8, "temperature": 0.9, "top_k": 8, "top_p": 0.95,
             "repetition_penalty": 1.2, "seed": 5},
        ]}),
    );
    assert_eq!(got[0], r.generate("ab", 6, &greedy(), 0));
    assert_eq!(got[1], r.generate("the", 99, &greedy(), 0));
    assert_eq!(got[2], r.generate("ab", 8, &sampled, 5));
    assert!(
        got[1]["n_tokens"].as_u64().unwrap() <= 12,
        "context 16 minus BOS + 3"
    );

    // A stop string cuts the text where it first appears.
    let free = r.generate("ab", 10, &greedy(), 0);
    let text = free["text"].as_str().unwrap();
    let ch = text
        .chars()
        .nth(1)
        .expect("the tiny model generates at least two characters");
    let cut = text.find(ch).unwrap();
    let got = results(
        &addr,
        "/v1/generate",
        json!({"requests": [{"prompt": "ab", "max_tokens": 10, "stop": [ch.to_string()]}]}),
    );
    assert_eq!(got[0]["text"], text[..cut]);
    assert!(got[0]["n_tokens"].as_u64() < free["n_tokens"].as_u64());

    // Loglikelihoods, BOS first; the long text spans several context windows.
    let long = "abcdefghij".repeat(4);
    let got = results(
        &addr,
        "/v1/loglikelihood_rolling",
        json!({"texts": ["abc", long, ""]}),
    );
    same_scores(&got, &[r.rolling("abc"), r.rolling(&long), r.rolling("")]);
    assert_eq!(got[1]["n_tokens"], 40);
    let got = results(
        &addr,
        "/v1/loglikelihood",
        json!({"requests": [{"context": "ab", "continuation": "cd"},
                            {"context": "", "continuation": "abc"}]}),
    );
    assert_eq!(
        got,
        [r.loglikelihood("ab", "cd"), r.loglikelihood("", "abc")]
    );

    // A prompt longer than the context is cut from the left, not refused, and the rest of
    // the batch is untouched.
    let long = "abcdefghijklmnopqrstuvwxyz".repeat(2);
    let got = results(
        &addr,
        "/v1/generate",
        json!({"requests": [{"prompt": long, "max_tokens": 4}, {"prompt": "ab", "max_tokens": 3}]}),
    );
    assert_eq!(got[0], r.generate(&long, 4, &greedy(), 0));
    assert_eq!(got[1], r.generate("ab", 3, &greedy(), 0));
    assert!(sigint_and_wait(&mut child).success());
}

#[test]
fn bos_follows_add_bos_token_and_an_empty_context_without_one_is_a_422() {
    let id = ("tokenizer.ggml.bos_token_id", MetaValue::U32(BOS));
    for add in [true, false] {
        let f = File::new(
            if add { "bos-on" } else { "bos-off" },
            &tiny_llama(&[
                id.clone(),
                ("tokenizer.ggml.add_bos_token", MetaValue::Bool(add)),
            ]),
        );
        let r = Reference::new(&f.0, None);
        assert_eq!(r.tok.prefix_bos().is_some(), add);
        let (mut child, _err, addr) = spawn(&f.0, &["--quant", "f32"]);
        let got = results(
            &addr,
            "/v1/loglikelihood_rolling",
            json!({"texts": ["abc"]}),
        );
        same_scores(&got, &[r.rolling("abc")]);
        assert_eq!(got[0]["n_tokens"], if add { 3 } else { 2 });
        let (s, v) = post(
            &addr,
            "/v1/loglikelihood",
            &json!({"requests": [{"context": "", "continuation": "abc"}]}),
        );
        if add {
            assert_eq!(s, 200, "{v}");
            same_scores(
                &v["results"].as_array().unwrap().clone(),
                &[r.loglikelihood("", "abc")],
            );
        } else {
            assert_eq!(s, 422, "{v}");
            assert!(
                v.to_string()
                    .contains("context is empty and the model prepends no BOS token"),
                "{v}"
            );
        }
        assert!(sigint_and_wait(&mut child).success());
    }
}

#[test]
fn default_int8_is_reported_and_matches_the_library_at_int8() {
    let f = File::new("int8", &tiny_llama(&[]));
    let r = Reference::new(&f.0, Some(QKind::Int8));
    let (mut child, _err, addr) = spawn(&f.0, &["--force"]);
    let (_, info) = http(&addr, "GET", "/v1/info", "");
    let info: Value = serde_json::from_str(&info).unwrap();
    assert_eq!(info["quant"], "int8");
    let got = results(
        &addr,
        "/v1/generate",
        json!({"requests": [{"prompt": "abc", "max_tokens": 5}]}),
    );
    assert_eq!(got[0], r.generate("abc", 5, &greedy(), 0));
    // No BOS in this file: the first token is not scored.
    let got = results(
        &addr,
        "/v1/loglikelihood_rolling",
        json!({"texts": ["abc"]}),
    );
    same_scores(&got, &[r.rolling("abc")]);
    assert_eq!(got[0]["n_tokens"], 2);
    assert!(sigint_and_wait(&mut child).success());
}

#[test]
fn unservable_files_and_bad_flags_are_refused_at_startup() {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("phi3".into()));
    let phi = File::new("phi3", &b.into_bytes());
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("llama".into()));
    let notok = File::new("notok", &b.into_bytes());
    let llama = File::new("flags", &tiny_llama(&[]));
    for (path, extra, want) in [
        (&phi.0, vec![], "architecture 'phi3' is not supported"),
        (&notok.0, vec![], "has no tokenizer"),
        (
            &llama.0,
            vec!["--quant", "fp8"],
            "--quant must be int4|int8|f32 (got 'fp8')",
        ),
    ] {
        let o = Command::new(BIN)
            .arg("--model")
            .arg(path)
            .args(&extra)
            .args(["--port", "0"])
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&o.stderr);
        assert_eq!(o.status.code(), Some(1), "{err}");
        assert!(err.contains(want), "{err}");
    }
}

/// The real SmolLM2-135M (Q8_0, `add_bos_token = false` but a BOS id): greedy generation and
/// both loglikelihoods equal the library at the default int8.
#[test]
fn smollm2_135m_over_http_matches_the_library_when_present() {
    let path = Path::new(DEV).join("models/smollm2-135m-q8_0.gguf");
    if !path.is_file() {
        eprintln!(
            "SKIPPED smollm2_135m_over_http_matches_the_library_when_present: {} is absent",
            path.display()
        );
        return;
    }
    let r = Reference::new(&path, Some(QKind::Int8));
    assert!(r.tok.prefix_bos().is_none() && r.tok.bos().is_some());
    let (mut child, _err, addr) = spawn(&path, &[]);
    let got = results(
        &addr,
        "/v1/generate",
        json!({"requests": [{"prompt": "Hello, my name is", "max_tokens": 8}]}),
    );
    assert_eq!(got[0], r.generate("Hello, my name is", 8, &greedy(), 0));
    let generated = got[0].clone();
    let text = "The capital of France is Paris.";
    let got = results(&addr, "/v1/loglikelihood_rolling", json!({"texts": [text]}));
    same_scores(&got, &[r.rolling(text)]);
    let rolling = got[0].clone();
    let got = results(
        &addr,
        "/v1/loglikelihood",
        json!({"requests": [{"context": "The capital of France is", "continuation": " Paris"}]}),
    );
    same_scores(
        &got,
        &[r.loglikelihood("The capital of France is", " Paris")],
    );
    eprintln!(
        "smollm2: generate {generated} / rolling {rolling} / loglikelihood {}",
        got[0]
    );
    assert!(sigint_and_wait(&mut child).success());
}
