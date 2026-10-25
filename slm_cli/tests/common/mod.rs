//! Helpers shared by the runtime-facing integration tests (E1 here, G5 later).
#![allow(dead_code)] // each test crate uses a different subset

use ferrum_core::{GgufBuilder, MetaValue};
use std::path::PathBuf;

/// Pure core of [`require_or_skip`]: `require` is the value of `FERRUM_REQUIRE_RUNTIMES=1`.
pub fn require_or_skip_with<T>(tool: Result<T, String>, require: bool) -> Option<T> {
    match tool {
        Ok(t) => Some(t),
        Err(e) if require => panic!("FERRUM_REQUIRE_RUNTIMES=1 but a runtime is missing: {e}"),
        Err(e) => {
            eprintln!("skip: {e}");
            None
        }
    }
}

/// `Some(tool)` when found. When missing: print `skip: …` and return `None`,
/// or panic if `FERRUM_REQUIRE_RUNTIMES=1`.
pub fn require_or_skip<T>(tool: Result<T, String>) -> Option<T> {
    let v = std::env::var("FERRUM_REQUIRE_RUNTIMES").ok();
    require_or_skip_with(tool, is_required(v.as_deref()))
}

/// Whether the value of `FERRUM_REQUIRE_RUNTIMES` turns a skip into a failure.
pub fn is_required(v: Option<&str>) -> bool {
    v == Some("1")
}

/// Locate a tool: `$env`, then `PATH`, then `~/<home_rel>`.
pub fn tool(env: &str, name: &str, home_rel: &str) -> Result<PathBuf, String> {
    let mut cands: Vec<PathBuf> = std::env::var_os(env)
        .map(PathBuf::from)
        .into_iter()
        .collect();
    if let Some(p) = std::env::var_os("PATH") {
        cands.extend(std::env::split_paths(&p).map(|d| d.join(name)));
    }
    if let Some(h) = std::env::var_os("HOME") {
        cands.push(PathBuf::from(h).join(home_rel));
    }
    cands
        .into_iter()
        .find(|c| c.is_file())
        .ok_or_else(|| format!("{name} not found (${env}, PATH, ~/{home_rel})"))
}

/// Held while a test touches Ollama, LM Studio or their ports. Serialises threads of this
/// process (mutex) *and* concurrent test processes (advisory `flock` on a file in the temp
/// dir), so two `cargo test` runs never fight over the same runtime. Released on drop.
pub struct RuntimeLock {
    _thread: std::sync::MutexGuard<'static, ()>,
    _file: std::fs::File,
}

static THREAD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn runtime_lock_path() -> PathBuf {
    std::env::temp_dir().join("ferrum-runtime-tests.lock")
}

fn open_lock_file() -> std::fs::File {
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(runtime_lock_path())
        .expect("open runtime lock file")
}

/// Block until this process owns the runtime lock.
pub fn runtime_lock() -> RuntimeLock {
    let thread = THREAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let file = open_lock_file();
    file.lock().expect("flock runtime lock"); // blocks; the OS drops it if we die
    RuntimeLock {
        _thread: thread,
        _file: file,
    }
}

/// `true` if some other holder (this or another process) currently owns the lock.
pub fn runtime_lock_is_held_elsewhere() -> bool {
    matches!(
        open_lock_file().try_lock(),
        Err(std::fs::TryLockError::WouldBlock)
    )
}

pub fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    let mut o = Vec::new();
    for &x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

/// A tiny but complete `llama`-architecture GGUF (2 layers, dim 8, all F32
/// tensors). Mirrors `tiny_llama` in `tests/tests/test_gguf_export.rs`; kept
/// as a minimal duplicate here since that helper lives in a different crate.
pub fn tiny_llama_gguf_bytes() -> Vec<u8> {
    tiny_llama_gguf_with_tokens(&(0..32).map(|i| format!("t{i}")).collect::<Vec<_>>())
}

/// Same model with a caller-supplied (32-entry) vocabulary, so a corpus can
/// actually be tokenized.
pub fn tiny_llama_gguf_with_tokens(tokens: &[String]) -> Vec<u8> {
    tiny_llama_gguf(Some(tokens), &[])
}

/// The tiny llama model with an optional vocabulary (`None`: no tokenizer metadata at all) and
/// extra metadata keys (chat template, BOS id, ...).
pub fn tiny_llama_gguf(tokens: Option<&[String]>, extra: &[(&str, MetaValue)]) -> Vec<u8> {
    const GGML_F32: u32 = 0;
    let (dim, n_layers, n_heads, ffn, vocab) = (8usize, 2usize, 2usize, 16usize, 32usize);
    let head_dim = dim / n_heads;
    let gen = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((seed * 131 + i * 17) % 101) as f32 / 500.0) - 0.1)
            .collect()
    };

    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("llama".into()));
    b.meta("llama.embedding_length", MetaValue::U32(dim as u32));
    b.meta("llama.block_count", MetaValue::U32(n_layers as u32));
    b.meta("llama.attention.head_count", MetaValue::U32(n_heads as u32));
    b.meta(
        "llama.attention.head_count_kv",
        MetaValue::U32(n_heads as u32),
    );
    b.meta("llama.feed_forward_length", MetaValue::U32(ffn as u32));
    // `extra` may override the 16-token default context window.
    let ctx = extra
        .iter()
        .find(|(k, _)| *k == "llama.context_length")
        .map_or(MetaValue::U32(16), |(_, v)| v.clone());
    b.meta("llama.context_length", ctx);
    b.meta(
        "llama.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-5),
    );
    if let Some(tokens) = tokens {
        b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
        b.meta(
            "tokenizer.ggml.tokens",
            MetaValue::Array(
                tokens
                    .iter()
                    .map(|t| MetaValue::String(t.clone()))
                    .collect(),
            ),
        );
    }
    for (k, v) in extra.iter().filter(|(k, _)| *k != "llama.context_length") {
        b.meta(k, v.clone());
    }

    let t = |b: &mut GgufBuilder, name: &str, dims: &[u64], seed: usize| {
        let n: usize = dims.iter().product::<u64>() as usize;
        b.tensor(name, dims, GGML_F32, f32_bytes(&gen(seed, n)));
    };
    t(&mut b, "token_embd.weight", &[dim as u64, vocab as u64], 1);
    for i in 0..n_layers {
        let p = format!("blk.{i}");
        t(
            &mut b,
            &format!("{p}.attn_norm.weight"),
            &[dim as u64],
            10 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_q.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            20 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_k.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            30 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_v.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            40 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_output.weight"),
            &[(n_heads * head_dim) as u64, dim as u64],
            50 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_norm.weight"),
            &[dim as u64],
            60 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_gate.weight"),
            &[dim as u64, ffn as u64],
            70 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_up.weight"),
            &[dim as u64, ffn as u64],
            80 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_down.weight"),
            &[ffn as u64, dim as u64],
            90 + i,
        );
    }
    t(&mut b, "output_norm.weight", &[dim as u64], 200);
    t(&mut b, "output.weight", &[dim as u64, vocab as u64], 201);
    b.into_bytes()
}

pub fn letter_vocab() -> Vec<String> {
    let mut v: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
    v.push("\u{120}".into());
    v.extend((0..5).map(|i| format!("<f{i}>")));
    v
}
