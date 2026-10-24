//! Bit-identical outputs at any thread count (a hard project invariant).
use std::process::Command;

fn run(threads: &str, args: &[&str]) -> Vec<u8> {
    let out = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .env("FERRUM_NUM_THREADS", threads)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

#[test]
fn cli_outputs_identical_across_thread_counts() {
    let dir = std::env::temp_dir().join(format!("ferrum_det_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("c.txt");
    std::fs::write(
        &corpus,
        "the cat sat on the mat. the dog sat on the log. ".repeat(80),
    )
    .unwrap();
    let c = corpus.to_str().unwrap();
    // `--threads` (data-parallel gradient shards) legitimately changes the
    // summation order, so it stays fixed at 1; the invariant is about the matmul
    // pool size, FERRUM_NUM_THREADS.
    let mut models = Vec::new();
    for threads in ["1", "4"] {
        let m = dir.join(format!("m{threads}.gguf"));
        let ms = m.to_str().unwrap();
        run(
            threads,
            &[
                "train",
                c,
                ms,
                "--context",
                "32",
                "--embed",
                "32",
                "--heads",
                "4",
                "--blocks",
                "2",
                "--hidden",
                "128",
                "--epochs",
                "2",
                "--vocab",
                "300",
                "--threads",
                "1",
            ],
        );
        models.push(std::fs::read(&m).unwrap());
    }
    assert!(
        models[0] == models[1],
        "trained model bytes differ between 1 and 4 threads"
    );
    let m = dir.join("m1.gguf");
    let ms = m.to_str().unwrap();
    let gen = |t| {
        run(
            t,
            &[
                "generate",
                ms,
                "the cat",
                "--max-tokens",
                "40",
                "--gen-seed",
                "7",
                "--temp",
                "0.8",
            ],
        )
    };
    let (g1, g4) = (gen("1"), gen("4"));
    assert!(!g1.is_empty());
    assert!(g1 == g4, "generate differs between 1 and 4 threads");
    // `eval` prints its wall-clock "Seconds" line; everything else must match.
    let eval = |t| {
        String::from_utf8(run(t, &["eval", ms, c]))
            .unwrap()
            .lines()
            .filter(|l| !l.starts_with("Seconds"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let (e1, e4) = (eval("1"), eval("4"));
    assert!(!e1.is_empty());
    assert!(e1 == e4, "eval differs between 1 and 4 threads");
    let _ = std::fs::remove_dir_all(&dir);
}
