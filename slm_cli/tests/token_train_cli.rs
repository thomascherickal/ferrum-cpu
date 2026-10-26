//! `tokenize` then `train tokens.gguf` through the real binary gives the same model bytes as
//! `train corpus.txt` with the same flags (tiny model, a 3 KB generated corpus).
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

const ARCH: [&str; 10] = [
    "--context",
    "16",
    "--embed",
    "16",
    "--heads",
    "2",
    "--blocks",
    "2",
    "--hidden",
    "32",
];

struct Env(PathBuf);

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tt(args: &[&str]) -> (bool, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .unwrap();
    (
        o.status.success(),
        String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr),
    )
}

fn sha(p: &str) -> String {
    Sha256::digest(std::fs::read(p).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn corpus() -> String {
    let words: Vec<&str> =
        "the cat dog sat ran on a mat log hill and then it saw big small red old"
            .split(' ')
            .collect();
    let mut state = 12345u64;
    let mut next = move |n: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize % n
    };
    (0..60)
        .map(|_| {
            let body: Vec<&str> = (0..8 + next(7)).map(|_| words[next(words.len())]).collect();
            format!("{}.", body.join(" "))
        })
        .collect::<Vec<_>>()
        .join("<|endoftext|>")
}

impl Env {
    /// A temp dir holding the corpus, a tokenizer (made by a first `train`) and the token file.
    fn new(tag: &str) -> Env {
        let dir = std::env::temp_dir().join(format!("ferrum_tok_cli_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let env = Env(dir);
        std::fs::write(env.s("c.txt"), corpus()).unwrap();
        let (ok, out) = tt(&[
            &[
                "train",
                &env.s("c.txt"),
                &env.s("seed.gguf"),
                "--vocab",
                "300",
                "--epochs",
                "1",
                "--tokenizer",
                &env.s("tok.gguf"),
            ][..],
            &ARCH,
        ]
        .concat());
        assert!(ok, "{out}");
        let (ok, out) = tt(&[
            "tokenize",
            &env.s("c.txt"),
            "--tokenizer",
            &env.s("tok.gguf"),
            "--out",
            &env.s("c.tok.gguf"),
        ]);
        assert!(ok, "{out}");
        env
    }

    fn s(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_string()
    }

    /// `train <input> <model>` with the tiny architecture and the shared tokenizer.
    fn train(&self, input: &str, model: &str, extra: &[&str]) -> (bool, String) {
        let (i, m, t) = (self.s(input), self.s(model), self.s("tok.gguf"));
        let mut a = vec![
            "train",
            &i,
            &m,
            "--tokenizer",
            &t,
            "--vocab",
            "300",
            "--seed",
            "7",
        ];
        a.extend(ARCH);
        a.extend(extra);
        tt(&a)
    }

    /// Train both ways with the same flags and return the two model hashes.
    fn pair(&self, tag: &str, extra: &[&str]) -> (String, String) {
        let (a, b) = (format!("{tag}_text.gguf"), format!("{tag}_tok.gguf"));
        let (ok, out) = self.train("c.txt", &a, extra);
        assert!(ok, "{out}");
        let (ok, out) = self.train("c.tok.gguf", &b, extra);
        assert!(ok, "{out}");
        assert!(out.contains("streamed from disk"), "{out}");
        (sha(&self.s(&a)), sha(&self.s(&b)))
    }
}

#[test]
fn a_token_file_trains_the_same_model_as_the_text() {
    let e = Env::new("same");
    for threads in ["1", "2"] {
        let (a, b) = e.pair(
            &format!("t{threads}"),
            &["--epochs", "3", "--threads", threads],
        );
        assert_eq!(a, b, "threads {threads}");
    }
    // Other knobs: stride, budget, cosine schedule, dropout, tied head, clip.
    let (a, b) = e.pair(
        "knobs",
        &[
            "--epochs",
            "9",
            "--threads",
            "2",
            "--stride",
            "3",
            "--batch",
            "4",
            "--tokens",
            "6000",
            "--cosine",
            "--warmup",
            "5",
            "--dropout",
            "0.1",
            "--tie",
            "--clip",
            "1",
        ],
    );
    assert_eq!(a, b);
}

#[test]
fn resuming_a_token_file_run_matches_the_text_run() {
    let e = Env::new("resume");
    let mut hashes = Vec::new();
    for (input, tag) in [("c.txt", "text"), ("c.tok.gguf", "tok")] {
        let (ck, model) = (e.s(&format!("{tag}.fckp")), format!("{tag}_r.gguf"));
        let (ok, out) = e.train(
            input,
            &format!("{tag}_half.gguf"),
            &["--epochs", "2", "--threads", "2", "--checkpoint", &ck],
        );
        assert!(ok, "{out}");
        let (ok, out) = e.train(
            input,
            &model,
            &["--epochs", "5", "--threads", "2", "--resume", &ck],
        );
        assert!(ok, "{out}");
        hashes.push((sha(&e.s(&model)), sha(&ck)));
    }
    assert_eq!(hashes[0], hashes[1]);
}

#[test]
fn validation_from_a_token_file_trains_and_reports_perplexity() {
    let e = Env::new("val");
    let (ok, out) = e.train(
        "c.tok.gguf",
        "v.gguf",
        &["--epochs", "3", "--val", "0.2", "--threads", "1"],
    );
    assert!(ok, "{out}");
    assert!(out.contains("val ppl"), "{out}");
    assert!(Path::new(&e.s("v.gguf")).exists());
}

#[test]
fn existing_models_sampling_and_misuse_are_handled() {
    let e = Env::new("misuse");
    let (ok, out) = e.train("c.tok.gguf", "m.gguf", &["--epochs", "1", "--sample"]);
    assert!(ok && out.contains("── sample ──"), "{out}");
    let (ok, out) = e.train("c.tok.gguf", "m.gguf", &["--epochs", "1"]);
    assert!(ok && out.contains("already exists"), "{out}");

    let (i, m) = (e.s("c.tok.gguf"), e.s("n.gguf"));
    // The tokenizer is mandatory, and must be the one the tokens were made with.
    let (ok, out) = tt(&["train", &i, &m]);
    assert!(!ok && out.contains("needs --tokenizer"), "{out}");
    std::fs::write(
        e.s("other.txt"),
        "completely different text for another tokenizer. ".repeat(40),
    )
    .unwrap();
    let (ok, out) = tt(&[
        "train",
        &e.s("other.txt"),
        &e.s("o.gguf"),
        "--vocab",
        "280",
        "--epochs",
        "1",
        "--tokenizer",
        &e.s("other.gguf"),
    ]);
    assert!(ok, "{out}");
    let (ok, out) = tt(&["train", &i, &m, "--tokenizer", &e.s("other.gguf")]);
    assert!(
        !ok && out.contains("is not the tokenizer") && out.contains("re-run `tokenize`"),
        "{out}"
    );
    let (ok, out) = tt(&[
        "train",
        &i,
        &m,
        "--tokenizer",
        &e.s("tok.gguf"),
        "--vocab",
        "0",
    ]);
    assert!(!ok && out.contains("--vocab 0"), "{out}");
    // A model file is a GGUF but not a token file.
    let (ok, out) = tt(&["train", &e.s("m.gguf"), &m, "--tokenizer", &e.s("tok.gguf")]);
    assert!(!ok && out.contains("not a ferrum-tokens file"), "{out}");
    // A truncated token file is reported, not trained on.
    let bytes = std::fs::read(&i).unwrap();
    std::fs::write(e.s("cut.tok.gguf"), &bytes[..bytes.len() - 40]).unwrap();
    let (ok, out) = tt(&[
        "train",
        &e.s("cut.tok.gguf"),
        &m,
        "--tokenizer",
        &e.s("tok.gguf"),
    ]);
    assert!(!ok && out.contains("past"), "{out}");
    assert!(!Path::new(&m).exists());
    // `run` needs text.
    let (ok, out) = tt(&["run", &i, &m, "hello", "--tokenizer", &e.s("tok.gguf")]);
    assert!(!ok && out.contains("token file"), "{out}");
}
