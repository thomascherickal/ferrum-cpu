//! Shared test fixtures (compiled only for tests).
// The integration tests include this file by path and each uses only part of it.
#![allow(dead_code)]

/// A valid `llama` GGUF (gpt2-style tokenizer metadata, F32 weights,
/// deterministic values, context length 128) of the given size. Returns the
/// file's bytes. The default test size is `(8, 2, 2, 16, 32)`.
pub fn tiny_llama_gguf(
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    ffn: usize,
    vocab: usize,
) -> Vec<u8> {
    Tiny {
        dim,
        n_layers,
        n_heads,
        ffn,
        vocab,
        ..Tiny::default()
    }
    .build()
}

/// A tiny llama GGUF and the ways a test breaks it on purpose.
pub struct Tiny {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn: usize,
    /// Embedding rows (and output rows).
    pub vocab: usize,
    /// Tokenizer entries; `None` = as many as `vocab`, `Some(0)` = no tokenizer at all. More
    /// entries than rows makes a token id the weights cannot embed.
    pub tok_count: Option<usize>,
    /// `false`: a header that opens but has no tensors to load.
    pub with_tensors: bool,
    /// `tokenizer.chat_template`, when the file should look like an instruct model.
    pub chat_template: Option<&'static str>,
    /// Special pieces stored at the given token ids (e.g. `<|im_end|>`), so chat markers resolve.
    pub pieces: &'static [(usize, &'static str)],
    /// `tokenizer.ggml.add_bos_token`, when the file states it.
    pub add_bos: Option<bool>,
}

impl Default for Tiny {
    fn default() -> Self {
        Tiny {
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            ffn: 16,
            vocab: 32,
            tok_count: None,
            with_tensors: true,
            chat_template: None,
            pieces: &[],
            add_bos: None,
        }
    }
}

impl Tiny {
    pub fn build(&self) -> Vec<u8> {
        let Tiny {
            dim,
            n_layers,
            n_heads,
            ffn,
            vocab,
            with_tensors,
            ..
        } = *self;
        let tok_count = self.tok_count.unwrap_or(vocab);
        use ferrum_core::{GgufBuilder, MetaValue};
        const GGML_F32: u32 = 0;
        let head_dim = dim / n_heads;
        let gen = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((seed * 131 + i * 17) % 101) as f32 / 500.0) - 0.1)
                .collect()
        };
        let f32_bytes = |xs: &[f32]| -> Vec<u8> {
            let mut o = Vec::with_capacity(xs.len() * 4);
            for &x in xs {
                o.extend_from_slice(&x.to_le_bytes());
            }
            o
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
        b.meta("llama.context_length", MetaValue::U32(128));
        b.meta(
            "llama.attention.layer_norm_rms_epsilon",
            MetaValue::F32(1e-5),
        );
        if tok_count > 0 {
            b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
            // A usable tokenizer: single-character tokens for "hi abcdefg" first, filler after,
            // no merges, BOS = 0, EOS = 9 (so text fine-tuning and prompts work on the fixture).
            let chars = ["h", "i", " ", "a", "b", "c", "d", "e", "f", "g"];
            let piece = |i: usize| match self.pieces.iter().find(|(id, _)| *id == i) {
                Some((_, p)) => p.to_string(),
                None => chars.get(i).map_or(format!("t{i}"), |c| c.to_string()),
            };
            b.meta(
                "tokenizer.ggml.tokens",
                MetaValue::Array(
                    (0..tok_count)
                        .map(|i| MetaValue::String(piece(i)))
                        .collect(),
                ),
            );
            b.meta("tokenizer.ggml.merges", MetaValue::Array(vec![]));
            b.meta("tokenizer.ggml.bos_token_id", MetaValue::U32(0));
            b.meta("tokenizer.ggml.eos_token_id", MetaValue::U32(9));
            if let Some(f) = self.add_bos {
                b.meta("tokenizer.ggml.add_bos_token", MetaValue::Bool(f));
            }
            if let Some(t) = self.chat_template {
                b.meta("tokenizer.chat_template", MetaValue::String(t.into()));
            }
        }
        if !with_tensors {
            return b.into_bytes();
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
            let qkv = (n_heads * head_dim) as u64;
            t(
                &mut b,
                &format!("{p}.attn_q.weight"),
                &[dim as u64, qkv],
                20 + i,
            );
            t(
                &mut b,
                &format!("{p}.attn_k.weight"),
                &[dim as u64, qkv],
                30 + i,
            );
            t(
                &mut b,
                &format!("{p}.attn_v.weight"),
                &[dim as u64, qkv],
                40 + i,
            );
            t(
                &mut b,
                &format!("{p}.attn_output.weight"),
                &[qkv, dim as u64],
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
}

/// A local HTTP server for download tests: answers up to `connections` requests with `status`
/// (e.g. `"200 OK"`) and `body`, and sends each raw request (request line and headers) on the
/// returned channel. Returns its base URL (`http://127.0.0.1:<port>`).
pub fn http_server(
    status: &'static str,
    body: Vec<u8>,
    connections: usize,
) -> (String, std::sync::mpsc::Receiver<String>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a local port");
    let base = format!("http://{}", listener.local_addr().expect("local address"));
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for _ in 0..connections {
            let Ok((mut s, _)) = listener.accept() else {
                return;
            };
            let mut req = Vec::new();
            let mut buf = [0u8; 1024];
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                match s.read(&mut buf) { Ok(n) if n > 0 => req.extend_from_slice(&buf[..n]), _ => break }
            }
            let _ = tx.send(String::from_utf8_lossy(&req).into_owned());
            let head = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = s.write_all(head.as_bytes());
            let _ = s.write_all(&body);
        }
    });
    (base, rx)
}

/// Whether the tests run as root, for which permission bits deny nothing.
#[cfg(unix)]
pub fn running_as_root() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            let uid = s.lines().find(|l| l.starts_with("Uid:"))?;
            Some(uid.split_whitespace().nth(2) == Some("0"))
        })
        .unwrap_or(false)
}

/// Sets `path` to `mode` and puts `restore` back when dropped, even on a panic, so a failed
/// assertion cannot leave an undeletable scratch directory behind. `None` (nothing changed) when
/// running as root, where the mode would deny nothing: the caller skips its check.
#[cfg(unix)]
pub fn restrict(path: &std::path::Path, mode: u32, restore: u32) -> Option<Restore> {
    use std::os::unix::fs::PermissionsExt;
    (!running_as_root()).then(|| { std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap(); Restore(path.to_path_buf(), restore) })
}

#[cfg(unix)]
pub struct Restore(std::path::PathBuf, u32);

#[cfg(unix)]
impl Drop for Restore {
    fn drop(&mut self) {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(self.1));
    }
}

/// The `.rs` files under `src/<group>` (recursively) that name `crate::<other>` for any of
/// `others`, as `path: crate::other` lines: the group-layering tests' scanner. (The needle is
/// built here, so a test that lists the groups does not match itself.)
pub fn sources_using(group: &str, others: &[&str]) -> Vec<String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(group);
    let mut files = Vec::new();
    walk(&root, &mut files);
    assert!(!files.is_empty(), "no sources under {}", root.display());
    let mut hits = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        for o in others {
            let needle = format!("crate::{o}");
            if text.contains(&needle) {
                hits.push(format!("{}: {needle}", f.display()));
            }
        }
    }
    hits
}
