//! Ferrum GGUFs against llama.cpp (plan p0g Task G5): the tokenizer produces the same ids,
//! greedy generation agrees, and quantized files load and run. Ollama and LM Studio are
//! covered by `slm_cli/tests/runtime_export_cli.rs`.
//!
//! Tools are found via `FERRUM_LLAMA_BIN` (dir holding `llama-completion` and
//! `llama-tokenize`), then `~/.local/opt/llama.cpp-b11521`. A missing tool skips the test,
//! unless `FERRUM_REQUIRE_RUNTIMES=1`, which turns the skip into a failure.
use ferrum_core::{GenerativeSLM, Rng, SamplingParams, TransformerConfig};
use std::path::{Path, PathBuf};

/// Pure core of [`llama_dir`]: `explicit` is `$FERRUM_LLAMA_BIN`, `home` is `$HOME`.
fn find_llama(explicit: Option<&str>, home: Option<&str>) -> Result<PathBuf, String> {
    let mut cands: Vec<PathBuf> = explicit.map(PathBuf::from).into_iter().collect();
    cands.extend(home.map(|h| Path::new(h).join(".local/opt/llama.cpp-b11521")));
    cands
        .into_iter()
        .find(|d| d.join("llama-completion").is_file() && d.join("llama-tokenize").is_file())
        .ok_or_else(|| {
            "llama.cpp not found ($FERRUM_LLAMA_BIN, ~/.local/opt/llama.cpp-b11521)".into()
        })
}

/// `Some(dir)` when llama.cpp is available; otherwise skip (print + `None`), or panic under
/// `FERRUM_REQUIRE_RUNTIMES=1`.
fn llama_dir() -> Option<PathBuf> {
    skip_or_fail(
        find_llama(
            std::env::var("FERRUM_LLAMA_BIN").ok().as_deref(),
            std::env::var("HOME").ok().as_deref(),
        ),
        std::env::var("FERRUM_REQUIRE_RUNTIMES").ok().as_deref() == Some("1"),
    )
}

fn skip_or_fail<T>(tool: Result<T, String>, require: bool) -> Option<T> {
    match tool {
        Ok(t) => Some(t),
        Err(e) if require => panic!("FERRUM_REQUIRE_RUNTIMES=1 but a runtime is missing: {e}"),
        Err(e) => {
            eprintln!("skip: {e}");
            None
        }
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_compat_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn greedy() -> SamplingParams {
    SamplingParams::with_temperature(0.0)
}

/// The 0.1M reference config (103,936 params at vocab 512 tied), trained on a repetitive story
/// so its distribution is peaked.
fn tiny_model() -> GenerativeSLM {
    let story = "The cat sat on the mat. It's 2024, isn't it?  Yes!\n<|endoftext|>\n".repeat(120);
    let cfg = TransformerConfig {
        context_len: 64,
        embed_dim: 64,
        num_heads: 4,
        num_blocks: 2,
        hidden_dim: 128,
        vocab_size: 512,
        weight_tying: true,
        epochs: 40,
        lr: 0.01,
        batch_size: 8,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(&story, &cfg, &mut Rng::new(11), |_, _| {}).unwrap()
}

/// Text llama.cpp generates after `prompt` (greedy, `n` tokens). Its async logger sometimes
/// drops the tail of tiny models' output, so take the longest of a few runs.
fn llama_complete(dir: &Path, model: &Path, prompt: &str, n: usize) -> String {
    let out_file = model.with_extension("out.txt");
    let run = || {
        let st = std::process::Command::new(dir.join("llama-completion"))
            .args(["-m", model.to_str().unwrap(), "-p", prompt, "-n"])
            .arg(n.to_string())
            .args(["--temp", "0", "-no-cnv", "--no-warmup"])
            .stdin(std::process::Stdio::null())
            .stdout(std::fs::File::create(&out_file).unwrap())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(st.success(), "llama-completion failed on {model:?}");
        std::fs::read_to_string(&out_file).unwrap()
    };
    let text = (0..6).map(|_| run()).max_by_key(|t| t.len()).unwrap();
    text.replace("[end of text]", "")
        .strip_prefix(prompt)
        .unwrap_or_else(|| panic!("llama output does not echo the prompt: {text:?}"))
        .to_string()
}

/// One side must be a prefix of the other (the shorter may have lost its tail), and the
/// llama.cpp side must have printed something.
fn assert_prefix_compatible(llama: &str, ferrum: &str) {
    assert!(!llama.trim().is_empty(), "llama.cpp generated nothing");
    let (a, b) = (llama.trim_end(), ferrum.trim_end());
    assert!(
        b.starts_with(a) || a.starts_with(b),
        "llama.cpp {a:?} vs Ferrum {b:?}"
    );
}

#[test]
fn tokenizer_matches_llama_cpp() {
    let Some(dir) = llama_dir() else { return };
    let slm = tiny_model();
    let work = tmp("tok");
    let f = work.join("m.gguf");
    std::fs::write(&f, slm.to_bytes().unwrap()).unwrap();
    let strings = [
        "The cat sat on the mat.",
        "It's isn't we'll they've I'm",
        "year 2024, 1234567 and 3.14",
        "wait...  what?!  ok;;  fine",
        "two  spaces   and\nnewlines\n\nhere",
        "héllo wörld — ünïcode ☃",
        "a<|endoftext|>b",
        "<|endoftext|>",
        "",
    ];
    for s in strings {
        let want = slm.encode_text(s).unwrap();
        let out = std::process::Command::new(dir.join("llama-tokenize"))
            .args(["-m", f.to_str().unwrap(), "-p", s, "--ids", "--log-disable"])
            .output()
            .unwrap();
        let txt = String::from_utf8_lossy(&out.stdout);
        let got: Vec<usize> = txt
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        assert_eq!(
            got, want,
            "tokenization of {s:?} differs (llama.cpp printed {txt:?})"
        );
    }
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn llama_cpp_greedy_matches_ferrum_f32() {
    let Some(dir) = llama_dir() else { return };
    let slm = tiny_model();
    let work = tmp("f32");
    let f = work.join("m.gguf");
    std::fs::write(&f, slm.to_bytes().unwrap()).unwrap();
    let prompt = "The cat sat";
    let want = slm
        .generate_tokens(prompt, 16, &[], &greedy(), &mut Rng::new(1))
        .unwrap();
    let got = llama_complete(&dir, &f, prompt, 16);
    assert_prefix_compatible(&got, &want.text);
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn llama_cpp_runs_q8_0_and_q4_0() {
    let Some(dir) = llama_dir() else { return };
    let slm = tiny_model();
    let work = tmp("quant");
    let (q8, q4) = (work.join("q8.gguf"), work.join("q4.gguf"));
    std::fs::write(&q8, slm.to_bytes_quantized().unwrap()).unwrap();
    std::fs::write(&q4, slm.to_bytes_quantized_int4().unwrap()).unwrap();
    let prompt = "The cat sat";
    // Q8_0: llama.cpp quantizes activations in its matmuls, which can flip near-ties late in a
    // tiny model, so equality is asserted on the first 8 tokens only.
    let reloaded = GenerativeSLM::from_bytes(&std::fs::read(&q8).unwrap()).unwrap();
    let want = reloaded
        .generate_tokens(prompt, 8, &[], &greedy(), &mut Rng::new(1))
        .unwrap();
    assert_prefix_compatible(&llama_complete(&dir, &q8, prompt, 8), &want.text);
    // Q4_0: llama.cpp must load it and generate.
    assert!(!llama_complete(&dir, &q4, prompt, 16).trim().is_empty());
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn reference_507k_config_loads_and_generates_in_llama_cpp() {
    let Some(dir) = llama_dir() else { return };
    // Pseudo-random words give a vocabulary rich enough for 1024 BPE tokens.
    let mut x = 12345u64;
    let mut words = Vec::new();
    for _ in 0..4000 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let len = 2 + (x >> 60) as usize % 6;
        let w: String = (0..len)
            .map(|i| (b'a' + ((x >> (8 + 5 * i)) % 26) as u8) as char)
            .collect();
        words.push(w);
    }
    let corpus = words.join(" ");
    let cfg = TransformerConfig {
        context_len: 128,
        embed_dim: 96,
        num_heads: 4,
        num_blocks: 3,
        hidden_dim: 320,
        vocab_size: 1024,
        max_tokens: 128 * 8 * 4,
        batch_size: 8,
        ..TransformerConfig::default()
    };
    let slm = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(2), |_, _| {})
        .unwrap();
    assert_eq!(slm.model.num_params(), 507_552);
    let work = tmp("ref");
    let f = work.join("ref.gguf");
    std::fs::write(&f, slm.to_bytes_quantized().unwrap()).unwrap();
    assert!(!llama_complete(&dir, &f, "abc", 8).trim().is_empty());
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn find_llama_prefers_explicit_dir_then_home() {
    let d = tmp("find");
    assert!(find_llama(Some(d.to_str().unwrap()), None).is_err());
    for n in ["llama-completion", "llama-tokenize"] {
        std::fs::write(d.join(n), b"").unwrap();
    }
    assert_eq!(find_llama(Some(d.to_str().unwrap()), None).unwrap(), d);
    let home = tmp("home");
    let opt = home.join(".local/opt/llama.cpp-b11521");
    std::fs::create_dir_all(&opt).unwrap();
    for n in ["llama-completion", "llama-tokenize"] {
        std::fs::write(opt.join(n), b"").unwrap();
    }
    assert_eq!(find_llama(None, Some(home.to_str().unwrap())).unwrap(), opt);
    assert!(find_llama(None, None)
        .unwrap_err()
        .contains("llama.cpp not found"));
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn require_runtimes_turns_skip_into_failure() {
    assert_eq!(skip_or_fail(Ok(5), true), Some(5));
    assert_eq!(skip_or_fail::<u8>(Err("x".into()), false), None);
    let r = std::panic::catch_unwind(|| skip_or_fail::<u8>(Err("no llama".into()), true));
    assert!(
        r.is_err(),
        "a missing tool must fail under FERRUM_REQUIRE_RUNTIMES=1"
    );
}
