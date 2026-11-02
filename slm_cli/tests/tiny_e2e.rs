//! The whole BPE pipeline through the real `train_transformer` binary with a tiny model
//! (about 14k parameters, a 3 KB generated corpus): prepare, train, info, eval, generate,
//! determinism, resume, quantized exports, character-level and model reuse.
//! Each test builds its own model in its own temp dir (training takes milliseconds).
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_train_transformer");
const VOCAB: f64 = 300.0;
const ARCH: [&str; 12] = [
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
    "--threads",
    "2",
];

/// A unique temp dir, removed on drop (also when an assertion panics).
struct Env {
    dir: PathBuf,
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn run_env(envs: &[(&str, &str)], args: &[&str]) -> Output {
    let mut c = Command::new(BIN);
    c.args(args);
    for (k, v) in envs {
        c.env(k, v);
    }
    let o = c.output().unwrap();
    assert!(
        o.status.success(),
        "{args:?} failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    o
}

fn run(args: &[&str]) -> String {
    String::from_utf8(run_env(&[], args).stdout).unwrap()
}

fn sha(p: &Path) -> String {
    Sha256::digest(std::fs::read(p).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// About 3.3 KB: 60 short documents over an 18-word vocabulary, from a fixed LCG.
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
    let docs: Vec<String> = (0..60)
        .map(|_| {
            let len = 8 + next(7);
            let body: Vec<&str> = (0..len).map(|_| words[next(words.len())]).collect();
            format!("{}.", body.join(" "))
        })
        .collect();
    docs.join("<|endoftext|>")
}

impl Env {
    /// Temp dir with the corpus written and `prepare`d into `prep/`.
    fn new(tag: &str) -> Env {
        let dir =
            std::env::temp_dir().join(format!("ferrum_tiny_e2e_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let env = Env { dir };
        std::fs::write(env.p("corpus.txt"), corpus()).unwrap();
        let out = run(&[
            "prepare",
            "--input",
            &env.s("corpus.txt"),
            "--out",
            &env.s("prep"),
            "--val-frac",
            "0.1",
            "--test-frac",
            "0.1",
        ]);
        assert!(out.contains("prepared"), "{out}");
        env
    }

    fn p(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn s(&self, name: &str) -> String {
        self.p(name).to_str().unwrap().to_string()
    }

    /// `train` on the prepared train split; returns stdout. `extra` follows the fixed flags
    /// (`--seed 7 --epochs 3`), so it can override them.
    fn train(&self, model: &str, extra: &[&str]) -> String {
        let (train, model) = (self.s("prep/train.txt"), self.s(model));
        let mut a = vec!["train", &train, &model];
        a.extend(ARCH);
        a.extend(["--seed", "7", "--epochs", "3"]);
        a.extend(extra);
        run(&a)
    }

    /// The standard BPE model (vocab 300, int8) as `name`.
    fn bpe(&self, name: &str) -> String {
        self.train(name, &["--vocab", "300"]);
        self.s(name)
    }

    fn eval_json(&self, model: &str, envs: &[(&str, &str)]) -> serde_json::Value {
        let out = self.s("eval.json");
        let test = self.s("prep/test.txt");
        run_env(envs, &["eval", model, &test, "--json", &out]);
        serde_json::from_slice(&std::fs::read(out).unwrap()).unwrap()
    }
}

fn info_field(model: &str, key: &str) -> String {
    run(&["info", model])
        .lines()
        .find_map(|l| l.strip_prefix(key))
        .unwrap_or_else(|| panic!("no {key} line"))
        .trim_start_matches([' ', ':'])
        .trim()
        .to_string()
}

fn generate(model: &str, extra: &[&str]) -> String {
    let mut a = vec!["generate", model, "the cat"];
    a.extend(extra);
    run(&a)
}

fn perplexity(j: &serde_json::Value) -> f64 {
    j["perplexity"].as_f64().unwrap()
}

#[test]
fn prepare_train_info_eval_generate() {
    let e = Env::new("pipeline");
    for f in ["train.txt", "val.txt", "test.txt", "manifest.json"] {
        let len = std::fs::metadata(e.p("prep").join(f)).unwrap().len();
        assert!(len > 0, "{f} is empty");
    }
    let out = e.train("m.gguf", &["--vocab", "300"]);
    assert!(out.contains("Reload check: OK"), "{out}");
    let m = e.s("m.gguf");

    assert_eq!(info_field(&m, "Params"), "14336");
    assert_eq!(info_field(&m, "Format"), "GGUF v3 (gpt2, int8)");
    assert!(info_field(&m, "Tokenizer").contains("300 tokens"));

    // Better than guessing uniformly over the vocabulary, and the JSON is self-consistent.
    let j = e.eval_json(&m, &[]);
    let ppl = perplexity(&j);
    assert!(ppl.is_finite() && ppl > 1.0 && ppl < VOCAB, "{j}");
    let ce = j["cross_entropy_nats"].as_f64().unwrap();
    assert!((ce.exp() - ppl).abs() < 1e-3 * ppl, "{j}");
    assert!(j["tokens"].as_u64().unwrap() > 50);

    // Greedy and seeded sampling repeat exactly.
    let greedy = ["--temp", "0", "--max-tokens", "20"];
    let g = generate(&m, &greedy);
    assert!(g.starts_with("the cat"), "{g:?}");
    assert_eq!(g, generate(&m, &greedy));
    let sampled = ["--temp", "0.9", "--gen-seed", "5", "--max-tokens", "20"];
    assert_eq!(generate(&m, &sampled), generate(&m, &sampled));

    // A stop string is excluded and cuts the output where it first appears.
    let free = generate(&m, &["--temp", "0", "--max-tokens", "30"]);
    let cont = free.trim_end().strip_prefix("the cat").unwrap().to_string();
    let stop: String = cont.chars().skip(2).take(3).collect();
    assert_eq!(stop.chars().count(), 3, "{free:?}");
    let cut = generate(&m, &["--temp", "0", "--max-tokens", "30", "--stop", &stop]);
    let at = cont.find(&stop).unwrap();
    assert_eq!(cut.trim_end(), format!("the cat{}", &cont[..at]), "{cut:?}");

    // --chars bounds the length.
    let short = generate(&m, &["--temp", "0", "--chars", "3"]);
    let long = generate(&m, &["--temp", "0", "--chars", "30"]);
    assert!(short.len() < long.len(), "{short:?} vs {long:?}");
}

#[test]
fn training_twice_is_byte_identical() {
    let e = Env::new("determinism");
    e.bpe("a.gguf");
    e.bpe("b.gguf");
    assert_eq!(sha(&e.p("a.gguf")), sha(&e.p("b.gguf")));
    // A different seed really changes the weights, so the equality above means something.
    e.train("c.gguf", &["--vocab", "300", "--seed", "8"]);
    assert_ne!(sha(&e.p("a.gguf")), sha(&e.p("c.gguf")));
}

#[test]
fn output_is_identical_across_matmul_thread_counts() {
    let e = Env::new("threads");
    let m = e.bpe("m.gguf");
    let gen = |t: &str| {
        run_env(
            &[("FERRUM_NUM_THREADS", t)],
            &[
                "generate",
                &m,
                "the cat",
                "--temp",
                "0.8",
                "--gen-seed",
                "3",
                "--max-tokens",
                "25",
            ],
        )
        .stdout
    };
    assert_eq!(gen("1"), gen("3"));
    let strip = |mut j: serde_json::Value| {
        j.as_object_mut().unwrap().remove("seconds");
        j
    };
    assert_eq!(
        strip(e.eval_json(&m, &[("FERRUM_NUM_THREADS", "1")])),
        strip(e.eval_json(&m, &[("FERRUM_NUM_THREADS", "3")]))
    );
}

#[test]
fn resume_from_an_epoch_checkpoint_finishes_the_run() {
    let e = Env::new("resume");
    // The reference: three epochs in one go (its checkpoint gives the step count).
    let full_ck = e.s("full.fckp");
    e.train("full.gguf", &["--vocab", "300", "--checkpoint", &full_ck]);
    // One epoch with a checkpoint, then continue to epoch three from it.
    let ck = e.s("ck.fckp");
    e.train(
        "part.gguf",
        &["--vocab", "300", "--epochs", "1", "--checkpoint", &ck],
    );
    assert!(e.p("ck.fckp").is_file());
    let out = e.train("resumed.gguf", &["--vocab", "300", "--resume", &ck]);
    assert!(out.contains("epoch     3/3"), "{out}");
    let resumed = e.s("resumed.gguf");
    assert!(perplexity(&e.eval_json(&resumed, &[])) < VOCAB);
    assert!(!generate(&resumed, &["--temp", "0", "--max-tokens", "5"]).is_empty());
    // At an epoch boundary no batch is replayed, so the result equals the uninterrupted run.
    assert_eq!(sha(&e.p("full.gguf")), sha(&e.p("resumed.gguf")));

    // Mid-epoch: a token budget stops the run after N / 2 + 3 of its N steps; the resumed run
    // replays the rest of that epoch's window order and ends on the same bytes.
    let n: u64 = info_field(&full_ck, "Step").parse().unwrap();
    let stop = n / 2 + 3;
    let mid = e.s("mid.fckp");
    let budget = (stop * 16 * 16).to_string(); // batch 16 x context 16 tokens per step
                                               // Three equal epochs: N / 3 steps each, and the stop is not a multiple of that.
    assert_ne!(stop % (n / 3), 0, "the stop must fall inside an epoch");
    e.train(
        "mid.gguf",
        &["--vocab", "300", "--checkpoint", &mid, "--tokens", &budget],
    );
    assert_eq!(info_field(&mid, "Step"), stop.to_string());
    let out = e.train("mid_resumed.gguf", &["--vocab", "300", "--resume", &mid]);
    assert!(out.contains("epoch     3/3"), "{out}");
    assert_eq!(sha(&e.p("full.gguf")), sha(&e.p("mid_resumed.gguf")));
}

#[test]
fn a_stop_file_or_time_budget_stops_cleanly_and_resume_finishes_the_run() {
    let e = Env::new("stop");
    // One optimizer step per epoch, so a stop after step 1 is an epoch boundary.
    let one = ["--vocab", "300", "--batch", "4096"];
    e.train("full.gguf", &one);

    // The stop file is `<model>.stop`; it is consumed, the checkpoint is written, exit is 0.
    let ck = e.s("ck.fckp");
    std::fs::write(e.p("part.gguf.stop"), b"").unwrap();
    let mut a = one.to_vec();
    a.extend(["--checkpoint", &ck]);
    let out = e.train("part.gguf", &a);
    assert!(
        out.contains(&format!(
            "stopped early at step 1: stop file found; resume with --resume {ck}"
        )),
        "{out}"
    );
    assert!(e.p("ck.fckp").is_file() && e.p("part.gguf").is_file());
    assert!(!e.p("part.gguf.stop").exists(), "stop file must be removed");

    let at_stop = sha(&e.p("ck.fckp")); // `--resume` keeps writing to this file
    let mut r = one.to_vec();
    r.extend(["--resume", &ck]);
    e.train("resumed.gguf", &r);
    assert_eq!(sha(&e.p("full.gguf")), sha(&e.p("resumed.gguf")));

    // A time budget crossed during the first step stops the same way.
    let ck2 = e.s("ck2.fckp");
    let mut t = one.to_vec();
    t.extend(["--checkpoint", &ck2, "--max-seconds", "0.000001"]);
    let out = e.train("timed.gguf", &t);
    assert!(
        out.contains("stopped early at step 1: time budget reached"),
        "{out}"
    );
    assert_eq!(at_stop, sha(&e.p("ck2.fckp")));

    // A budget the run never reaches changes nothing; nonsense is refused up front.
    e.train(
        "roomy.gguf",
        &[&one[..], &["--max-seconds", "3600"]].concat(),
    );
    assert_eq!(sha(&e.p("full.gguf")), sha(&e.p("roomy.gguf")));
    let bad = Command::new(BIN)
        .args([
            "train",
            &e.s("prep/train.txt"),
            &e.s("bad.gguf"),
            "--max-seconds",
            "0",
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("--max-seconds"));
}

#[test]
fn checkpoint_every_writes_a_checkpoint_with_optimizer_state() {
    let e = Env::new("ckevery");
    let ck = e.s("ck.fckp");
    e.train(
        "m.gguf",
        &[
            "--vocab",
            "300",
            "--checkpoint",
            &ck,
            "--checkpoint-every",
            "2",
        ],
    );
    let size = std::fs::metadata(&ck).unwrap().len();
    // Weights plus Adam state: far bigger than the int8 model.
    assert!(size > std::fs::metadata(e.p("m.gguf")).unwrap().len());
    // `info` reads the checkpoint header.
    let out = run(&["info", &ck]);
    assert!(
        out.contains("Format    : GGUF v3 (ferrum-checkpoint)"),
        "{out}"
    );
    for want in [
        "Context   : 16",
        "Embed     : 16",
        "Heads     : 2",
        "Hidden    : 32",
        "Blocks    : 2",
    ] {
        assert!(out.contains(want), "missing {want:?}: {out}");
    }
    assert!(out.contains("Step      : "), "{out}");
    assert!(out.contains("train --resume"), "{out}");
}

#[test]
fn f32_int8_and_int4_exports_load_and_agree() {
    let e = Env::new("quant");
    let mut ppl = Vec::new();
    let mut size = Vec::new();
    for q in ["f32", "int8", "int4"] {
        let name = format!("{q}.gguf");
        e.train(&name, &["--vocab", "300", "--quant", q]);
        let m = e.s(&name);
        assert_eq!(info_field(&m, "Format"), format!("GGUF v3 (gpt2, {q})"));
        ppl.push(perplexity(&e.eval_json(&m, &[])));
        size.push(std::fs::metadata(&m).unwrap().len());
        assert!(!generate(&m, &["--temp", "0", "--max-tokens", "5"]).is_empty());
    }
    assert!(size[2] < size[1] && size[1] < size[0], "{size:?}");
    for p in &ppl[1..] {
        assert!((p / ppl[0] - 1.0).abs() < 0.10, "perplexities {ppl:?}");
    }
}

#[test]
fn character_level_model_trains_and_generates() {
    let e = Env::new("chars");
    e.train("c.gguf", &["--vocab", "0"]);
    let m = e.s("c.gguf");
    assert_eq!(info_field(&m, "Format"), "GGUF v3 (ferrum, int8)");
    assert!(info_field(&m, "Tokenizer").starts_with("character-level"));
    let ppl = perplexity(&e.eval_json(&m, &[]));
    assert!(ppl.is_finite() && ppl > 1.0 && ppl < 24.0, "{ppl}"); // 24 characters
    let args = ["--temp", "0", "--chars", "20"];
    let g = generate(&m, &args);
    assert!(
        g.starts_with("the cat") && g.len() > "the cat".len(),
        "{g:?}"
    );
    assert_eq!(g, generate(&m, &args));
}

#[test]
fn existing_model_is_reused_and_force_retrains() {
    let e = Env::new("reuse");
    let m = e.bpe("m.gguf");
    let before = sha(Path::new(&m));
    // A different config would give a different model; reuse ignores it and leaves the file alone.
    let out = e.train("m.gguf", &["--vocab", "280", "--epochs", "1"]);
    assert!(out.contains("already exists"), "{out}");
    assert!(!out.contains("epoch "), "{out}");
    assert_eq!(sha(Path::new(&m)), before);
    let out = e.train("m.gguf", &["--vocab", "280", "--epochs", "1", "--force"]);
    assert!(out.contains("epoch     1/1"), "{out}");
    assert_ne!(sha(Path::new(&m)), before);
}

#[test]
fn training_output_tells_the_truth_about_the_run() {
    let e = Env::new("honest");
    // The banner names what is really trained and saved.
    let out = e.train("a.gguf", &["--vocab", "300"]);
    assert!(out.contains("trainer (int8 QAT)"), "{out}");
    let out = e.train("b.gguf", &["--vocab", "300", "--no-qat", "--quant", "f32"]);
    assert!(!out.contains("int8 QAT"), "{out}");
    assert!(
        out.contains("trainer (fp32, no QAT, saved as f32)"),
        "{out}"
    );
    let out = e.train("c.gguf", &["--vocab", "300", "--quant", "int4"]);
    assert!(out.contains("trainer (int8 QAT, saved as int4)"), "{out}");

    // A vocab the corpus cannot fill is reported with the real size.
    let out = e.train("d.gguf", &["--vocab", "5000", "--force"]);
    let real = info_field(&e.s("d.gguf"), "Output dim");
    assert!(
        out.contains(&format!(
            "vocab: {real} (requested 5000, limited by the corpus)"
        )),
        "{real}: {out}"
    );
    // Nothing to explain when the request was met.
    assert!(!e.train("e.gguf", &["--vocab", "300"]).contains("requested"));

    // Early stopping says so, and which epoch's weights were kept.
    let out = e.train(
        "f.gguf",
        &[
            "--vocab",
            "300",
            "--epochs",
            "300",
            "--val",
            "0.3",
            "--patience",
            "1",
        ],
    );
    let line = out
        .lines()
        .find(|l| l.starts_with("early stop at epoch "))
        .unwrap_or_else(|| panic!("no early stop line: {out}"));
    assert!(
        line.contains(", best epoch ") && line.ends_with(" kept"),
        "{line}"
    );
}

#[test]
fn metrics_csv_and_verbose_progress_leave_the_trained_model_bit_identical() {
    let e = Env::new("metrics");
    let plain = e.train("plain.gguf", &["--vocab", "300", "--batch", "4"]);
    assert!(
        !plain.contains("Metrics :") && !plain.contains("eta"),
        "{plain}"
    );
    let csv = e.s("m/run.csv");
    let out = e.train(
        "csv.gguf",
        &[
            "--vocab",
            "300",
            "--batch",
            "4",
            "--metrics-csv",
            &csv,
            "-v",
        ],
    );
    assert_eq!(
        sha(&e.p("plain.gguf")),
        sha(&e.p("csv.gguf")),
        "metrics changed training"
    );
    assert!(out.contains(&format!("Metrics : {csv}")), "{out}");
    // `-v` now shows the gradient norm and an ETA; the total step count is in the first line.
    let first = out
        .lines()
        .find(|l| l.starts_with("[train] step 10/"))
        .unwrap_or_else(|| panic!("no progress line: {out}"));
    assert!(
        first.contains("grad_norm ") && first.contains("eta "),
        "{first}"
    );
    let steps: usize = first["[train] step 10/".len()..]
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();

    let text = std::fs::read_to_string(&csv).unwrap();
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some("step,epoch,loss,lr,grad_norm,tokens_per_sec,elapsed")
    );
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split(',').map(|x| x.parse().unwrap()).collect())
        .collect();
    assert_eq!(rows.len(), steps);
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r.len(), 7);
        assert!(r.iter().all(|x| x.is_finite()), "{r:?}");
        assert_eq!(r[0] as usize, i + 1);
        assert!(
            r[2] > 0.0 && r[4] > 0.0,
            "loss and grad norm are positive: {r:?}"
        );
    }
    assert_eq!(rows.last().unwrap()[1], 3.0);

    // An unwritable CSV path fails before any training.
    let dir = e.s("");
    let bad = Command::new(BIN)
        .args([
            "train",
            &e.s("prep/train.txt"),
            &e.s("x.gguf"),
            "--metrics-csv",
            &dir,
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("cannot write"));
    assert!(!e.p("x.gguf").exists());
}
