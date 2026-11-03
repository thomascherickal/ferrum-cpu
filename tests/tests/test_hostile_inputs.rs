//! Hostile-input hardening: mutation-fuzz every untrusted-file loader (GGUF parse/import,
//! GPT-2 / generic model codecs, training checkpoints, tokenizer files and state strings).
//!
//! Contract: a malformed file returns `Err` - it never panics, aborts, hangs or asks for
//! absurd memory. The fuzz loop runs in a child process of this test binary under
//! `ulimit -v` (1 GiB of address space), so a hostile count that slips past a guard and
//! sizes an allocation aborts the child instead of eating the machine; the parent then
//! reports the last case that was running.
//!
//! `FERRUM_FUZZ_N=<cases per seed file>` raises the iteration count for a longer local run.
use ferrum_core::{
    from_bytes, from_gguf, ByteBpeTokenizer, GenerativeSLM, Gguf, GgufTokenizer, LlamaTrainer,
    QKind, Rng, TransformerConfig, TransformerNet,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[path = "synth/mod.rs"]
mod synth;

const CHILD_ENV: &str = "FERRUM_FUZZ_CHILD";

/// Values that stress counts, sizes and offsets.
const U64S: [u64; 14] = [
    0,
    1,
    2,
    7,
    0xFF,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    1 << 32,
    (1 << 32) + 5,
    1 << 40,
    1 << 62,
    1 << 63,
    u64::MAX,
];

struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// One deterministic mutation of `seed` (focused on the header region, where the counts are).
fn mutate(seed: &[u8], r: &mut Xs) -> Vec<u8> {
    let mut b = seed.to_vec();
    let hdr = b.len().min(if r.below(4) == 0 { b.len() } else { 4096 });
    for _ in 0..1 + r.below(3) {
        let pos = r.below(hdr.max(1));
        match r.below(10) {
            0 | 1 => {
                let v = U64S[r.below(U64S.len())];
                for (i, x) in v.to_le_bytes().iter().enumerate() {
                    if let Some(s) = b.get_mut(pos + i) {
                        *s = *x;
                    }
                }
            }
            2 => {
                let v = U64S[r.below(U64S.len())] as u32;
                for (i, x) in v.to_le_bytes().iter().enumerate() {
                    if let Some(s) = b.get_mut(pos + i) {
                        *s = *x;
                    }
                }
            }
            3 => {
                if let Some(s) = b.get_mut(pos) {
                    *s ^= 1 << r.below(8);
                }
            }
            4 => b.truncate(pos),
            5 => {
                let n = r.below(16);
                if pos < b.len() {
                    b.drain(pos..(pos + n).min(b.len()));
                }
            }
            6 => {
                let at = pos.min(b.len());
                for _ in 0..r.below(16) {
                    b.insert(at, r.next() as u8);
                }
            }
            // Same-length digit run -> '9's (numbers inside JSON metadata / tokenizer state).
            7 => {
                if let Some(start) = (pos..b.len()).find(|&i| b[i].is_ascii_digit()) {
                    let mut i = start;
                    while i < b.len() && b[i].is_ascii_digit() {
                        b[i] = b'9';
                        i += 1;
                    }
                }
            }
            // Structural character injection (JSON / state-string delimiters).
            9 => {
                let at = (0..b.len())
                    .cycle()
                    .skip(pos)
                    .take(b.len())
                    .find(|&i| b[i].is_ascii_graphic());
                if let Some(i) = at {
                    b[i] = b"\"[]{},:;\n\\<"[r.below(10)];
                }
            }
            // A single digit run -> '0's.
            _ => {
                if let Some(start) = (pos..b.len()).find(|&i| b[i].is_ascii_digit()) {
                    let mut i = start;
                    while i < b.len() && b[i].is_ascii_digit() {
                        b[i] = b'0';
                        i += 1;
                    }
                }
            }
        }
    }
    b
}

fn tiny_cfg(vocab: usize) -> TransformerConfig {
    TransformerConfig {
        context_len: 8,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        epochs: 1,
        vocab_size: vocab,
        ..TransformerConfig::default()
    }
}

fn train(vocab: usize) -> GenerativeSLM {
    let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(20);
    GenerativeSLM::train_transformer_config(&corpus, &tiny_cfg(vocab), &mut Rng::new(3), |_, _| {})
        .unwrap()
}

/// Every seed file the fuzzer mutates, by name.
fn seeds() -> Vec<(&'static str, Vec<u8>)> {
    let bpe = train(270);
    let chr = train(0);
    let llama = synth::synth_llama();
    let model = Gguf::parse(llama.clone())
        .unwrap()
        .load_llama_prec(None)
        .unwrap();
    let flck = LlamaTrainer::new(model)
        .unwrap()
        .save_checkpoint(&Rng::new(1));
    let net = TransformerNet::new(270, 8, 16, 2, 32, 1, &mut Rng::new(1)).unwrap();
    let tok = ByteBpeTokenizer::train(&"the cat sat on the mat. ".repeat(20), 280).unwrap();
    vec![
        ("gpt2-f32", bpe.to_bytes().unwrap()),
        ("gpt2-int8", bpe.to_bytes_quantized().unwrap()),
        ("gpt2-int4", bpe.to_bytes_quantized_int4().unwrap()),
        ("generic-char", chr.to_bytes().unwrap()),
        ("llama", llama),
        ("fckp", net.save_checkpoint(&Rng::new(1))),
        ("flck", flck),
        ("tokenizer", tok.to_gguf_bytes()),
    ]
}

/// Record which loader is running (read back by the parent when the child aborts).
fn stage(name: &str) {
    if let Ok(log) = std::env::var("FERRUM_FUZZ_LOG") {
        let _ = std::fs::write(format!("{log}.stage"), name);
    }
}

/// Run every loader on `bytes`; an `Err` is the expected outcome, a panic is a bug.
fn exercise(bytes: &[u8], flck_base: &dyn Fn() -> LlamaTrainer) {
    stage("parse+import");
    let _ = Gguf::parse(bytes.to_vec()).map(|g| {
        if let Ok(t) = GgufTokenizer::from_gguf(&g) {
            stage("tokenizer-use");
            let ids = t.encode("Hello, wörld! 123  <|endoftext|>\n");
            let _ = t.decode(&ids);
            let _ = t.decode(&[0, 1, usize::MAX, 99_999]);
            let _ = t.token_id("h");
        }
        stage("load_llama");
        if let Ok(m) = g.load_llama_prec(None) {
            stage("llama-use");
            let p = ferrum_core::SamplingParams::with_temperature(0.0);
            let _ = m.generate(&[1, 2], 3, &p, &[], &mut Rng::new(1));
            let _ = m.forward_tokens(&[0, 1, 2]);
        }
        let _ = g.load_llama_prec(Some(QKind::Int4));
        for t in g.tensors.clone() {
            let _ = g.dequantize(&t);
        }
    });
    stage("from_bytes");
    let _ = from_bytes(bytes);
    stage("from_gguf");
    let _ = from_gguf(bytes);
    stage("slm::from_bytes");
    let _ = GenerativeSLM::from_bytes(bytes);
    stage("load_checkpoint");
    let _ = TransformerNet::load_checkpoint(bytes);
    stage("tokenizer");
    let _ = ByteBpeTokenizer::from_gguf_bytes(bytes);
    stage("flck");
    let _ = flck_base().load_checkpoint_into(bytes);
    // Model-in-the-middle: a model that did load must not panic when used.
    stage("use-model");
    if let Ok(m) = GenerativeSLM::from_bytes(bytes) {
        stage("use:score");
        let _ = m.score("the cat", " sat");
        stage("use:rolling");
        let _ = m.score_rolling("the cat sat on the mat");
        let mut rng = Rng::new(1);
        let p = ferrum_core::SamplingParams::with_temperature(0.0);
        stage("use:generate_tokens");
        let _ = m.generate_tokens("the", 4, &[], &p, &mut rng);
        stage("use:generate");
        let _ = m.generate("the", 4, 0.0, &mut rng);
    }
}

fn state_strings() -> Vec<String> {
    let mut v: Vec<String> = [
        "",
        ";",
        ",",
        "1,2",
        "256,256",
        "0,0;256,256;257,257;258,258;259,259;260,260;261,261;262,262;263,263;264,264;265,265;266,266;267,267;268,268;269,269;270,270;271,271;272,272;273,273;274,274;275,275;276,276;277,277;278,278;279,279;280,280;281,281;282,282;283,283;284,284;285,285;286,286;287,287;288,288;289,289;290,290;291,291;292,292;293,293;294,294;295,295",
        "4294967295,0",
        "-1,2",
        "a,b",
        "\n",
        "\n,,,",
        "1,1\n<|endoftext|>,<|endoftext|>",
        "\n<|endoftext|>\n<|endoftext|>",
    ]
    .map(String::from)
    .to_vec();
    v.push(format!("{}\n{}", "0,1;".repeat(5000), "x,".repeat(5000)));
    v.push(",".repeat(100_000));
    v.push(";".repeat(100_000));
    v
}

/// The fuzz loop proper (runs in the memory-limited child).
fn fuzz_child() {
    let n: usize = std::env::var("FERRUM_FUZZ_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let log = std::env::var("FERRUM_FUZZ_LOG").unwrap_or_default();
    let synth_bytes = synth::synth_llama();
    let flck_base = move || {
        let m = Gguf::parse(synth_bytes.clone())
            .unwrap()
            .load_llama_prec(None)
            .unwrap();
        LlamaTrainer::new(m).unwrap()
    };
    std::panic::set_hook(Box::new(|_| {}));
    let mut failures = Vec::new();
    for (name, seed) in seeds() {
        let mut r = Xs(0x9E37_79B9_7F4A_7C15 ^ name.len() as u64);
        for case in 0..n {
            let bytes = mutate(&seed, &mut r);
            if !log.is_empty() {
                let _ = std::fs::write(&log, format!("{name} case {case} len {}", bytes.len()));
                let _ = std::fs::write(format!("{log}.bin"), &bytes);
            }
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| exercise(&bytes, &flck_base))) {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                failures.push(format!("{name} case {case}: panic: {msg}"));
                if failures.len() >= 10 {
                    break;
                }
            }
        }
    }
    for s in state_strings() {
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| {
            if let Ok(t) = ByteBpeTokenizer::from_state(&s) {
                let _ = t.encode_docs("the cat <|endoftext|> sat");
                let _ = t.decode(&[0, 1, 2, 300, usize::MAX]);
            }
        })) {
            let msg = p.downcast_ref::<String>().cloned().unwrap_or_default();
            failures.push(format!("state {:?}: panic: {msg}", &s[..s.len().min(40)]));
        }
    }
    // Text cleaners: random mixes of invisible / combining / marker / multi-byte pieces.
    let pieces = [
        "a",
        " ",
        "\n",
        "\r\n",
        "\u{feff}",
        "\u{200b}",
        "\u{200d}",
        "\u{301}",
        "\u{ff21}",
        "\u{fb01}",
        "\u{1f600}",
        "\t",
        "é",
        "\u{0}",
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***",
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***",
        "<|endoftext|>",
        "\"",
        "\\",
    ];
    let mut tr = Xs(77);
    for case in 0..n * 4 {
        let text: String = (0..tr.below(60))
            .map(|_| pieces[tr.below(pieces.len())])
            .collect();
        let opts = ferrum_core::CleanOptions {
            strip_gutenberg: tr.below(2) == 0,
            lowercase: tr.below(2) == 0,
            collapse_whitespace: tr.below(2) == 0,
            normalize_punctuation: tr.below(2) == 0,
            strip_control_chars: tr.below(2) == 0,
            max_chars: if tr.below(2) == 0 {
                None
            } else {
                Some(tr.below(40))
            },
        };
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| {
            let c = ferrum_core::clean_corpus(&text, &opts);
            let _ = ferrum_core::corpus_stats(&c);
            let _ = ferrum_core::validate_for_training(&c, tr.below(10));
            let _ = ferrum_core::squeeze_spaces(&ferrum_core::strip_invisible(&text));
            let _ = ferrum_core::normalize_nfkc(&text);
            for b in ferrum_core::gutenberg_bodies(&text) {
                let _ = ferrum_core::clean_corpus(b, &opts);
            }
        })) {
            let msg = p.downcast_ref::<String>().cloned().unwrap_or_default();
            failures.push(format!("text case {case} {text:?}: panic: {msg}"));
            break;
        }
    }
    if !failures.is_empty() {
        eprintln!("FUZZ FAILURES:\n{}", failures.join("\n"));
        std::process::exit(3);
    }
}

#[test]
fn mutated_files_and_states_never_panic_or_exhaust_memory() {
    if std::env::var_os(CHILD_ENV).is_some() {
        fuzz_child();
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let log = std::env::temp_dir().join(format!("ferrum_fuzz_{}.log", std::process::id()));
    // On Linux `ulimit -v` caps the child's address space (KiB); glibc arenas stay small with one
    // thread. Other platforms have no usable `ulimit -v` (macOS refuses it, Windows has no sh),
    // so there the child runs uncapped and only the panic/crash check applies.
    #[cfg(target_os = "linux")]
    let mut cmd = {
        let mut c = std::process::Command::new("sh");
        c.args([
            "-c",
            "ulimit -v 1048576; exec \"$0\" --exact mutated_files_and_states_never_panic_or_exhaust_memory --test-threads 1 --nocapture",
        ]);
        c
    };
    #[cfg(not(target_os = "linux"))]
    let mut cmd = {
        let mut c = std::process::Command::new(&exe);
        c.args([
            "--exact",
            "mutated_files_and_states_never_panic_or_exhaust_memory",
            "--test-threads",
            "1",
            "--nocapture",
        ]);
        c
    };
    #[cfg(target_os = "linux")]
    cmd.arg(&exe);
    let out = cmd
        .env(CHILD_ENV, "1")
        .env("FERRUM_FUZZ_LOG", &log)
        .env("MALLOC_ARENA_MAX", "1")
        .output()
        .expect("spawn fuzz child");
    let last = std::fs::read_to_string(&log).unwrap_or_default()
        + " stage "
        + &std::fs::read_to_string(format!("{}.stage", log.display())).unwrap_or_default();
    let _ = std::fs::remove_file(format!("{}.stage", log.display()));
    let _ = std::fs::remove_file(&log);
    // Keep the offending input next to the log for reproduction when set.
    if !out.status.success() {
        if let Some(keep) = std::env::var_os("FERRUM_FUZZ_KEEP") {
            let _ = std::fs::copy(format!("{}.bin", log.display()), keep);
        }
    }
    let _ = std::fs::remove_file(format!("{}.bin", log.display()));
    assert!(
        out.status.success(),
        "fuzz child failed ({:?}); last case: {last}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(25)
            .collect::<Vec<_>>()
            .join("\n")
    );
}
