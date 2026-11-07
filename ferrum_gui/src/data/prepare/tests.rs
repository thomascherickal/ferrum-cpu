use super::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_gui_prep_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn params(inputs: &[&Path], out: &Path) -> PrepareParams {
    PrepareParams {
        inputs: inputs.iter().map(|p| p.to_string_lossy().into()).collect(),
        out_dir: out.to_string_lossy().into(),
        val_frac: 0.1,
        test_frac: 0.1,
        seed: 7,
        min_chars: 10,
        max_chars: 0,
        doc_sep: DOC_SEP.into(),
        dedup: true,
        nfkc: true,
        lowercase: false,
        strip_gutenberg: false,
        lossy: false,
    }
}

/// Two files and a folder: many short documents, some duplicates, a too-short one.
fn corpus(dir: &Path) -> (PathBuf, PathBuf) {
    let docs: Vec<String> = (0..300)
        .map(|i| format!("Document number {i} says: the quick brown fox \u{FB01}nds a lazy dog."))
        .collect();
    let a = dir.join("a.txt");
    let mut text = docs[..150].join(DOC_SEP);
    text.push_str(DOC_SEP);
    text.push_str(&docs[0]); // a duplicate
    text.push_str(DOC_SEP);
    text.push_str("tiny"); // too short
    std::fs::write(&a, text).unwrap();
    let sub = dir.join("more");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("b.txt"), docs[150..].join(DOC_SEP)).unwrap();
    std::fs::write(sub.join(".hidden"), "ignored").unwrap();
    (a, sub)
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap()
}

#[test]
fn flags_in_the_manifest_are_the_equivalent_cli_command() {
    let p = PrepareParams {
        inputs: vec!["a.txt".into(), "d".into()],
        out_dir: "out".into(),
        doc_sep: "\n\n".into(),
        lowercase: true,
        lossy: true,
        max_chars: 500,
        ..params(&[], Path::new("out"))
    };
    let want = "--input a.txt --input d --out out --doc-sep \n\n --lossy --nfkc --lowercase \
                --min-chars 10 --max-chars 500 --dedup --val-frac 0.1 --test-frac 0.1 --seed 7";
    assert_eq!(cli_args(&p).join(" "), want);
    // The default separator is left out, as a CLI user would.
    let q = PrepareParams {
        doc_sep: DOC_SEP.into(),
        lossy: false,
        lowercase: false,
        nfkc: false,
        dedup: false,
        ..p
    };
    assert!(!cli_args(&q).iter().any(|a| a == "--doc-sep"));
}

#[test]
fn the_gui_writes_what_the_cli_library_writes_for_the_same_options() {
    let d = tmp("same");
    let (a, sub) = corpus(&d);
    let out = d.join("out");
    let p = params(&[&a, &sub], &out);

    // The reference: the library the CLI calls, with options written out by hand.
    let args: Vec<String> = [
        "--input",
        &a.to_string_lossy(),
        "--input",
        &sub.to_string_lossy(),
        "--out",
        &out.to_string_lossy(),
        "--nfkc",
        "--min-chars",
        "10",
        "--max-chars",
        "0",
        "--dedup",
        "--val-frac",
        "0.1",
        "--test-frac",
        "0.1",
        "--seed",
        "7",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let o = PrepareOptions {
        inputs: vec![a.to_string_lossy().into(), sub.to_string_lossy().into()],
        out: out.to_string_lossy().into(),
        doc_sep: DOC_SEP.into(),
        lossy: false,
        nfkc: true,
        lowercase: false,
        strip_gutenberg: false,
        min_chars: 10,
        max_chars: 0,
        dedup: true,
        val_frac: 0.1,
        test_frac: 0.1,
        seed: 7,
        args,
        stop: StopCondition::none(),
    };
    let want = prepare(&o, &mut |_| {}).unwrap();
    let files = ["train.txt", "val.txt", "test.txt", "manifest.json"];
    let want_bytes: Vec<Vec<u8>> = files.iter().map(|f| read(&out, f)).collect();
    assert_eq!(
        want.documents.iter().sum::<u64>(),
        300,
        "302 documents; a duplicate and a too-short one dropped"
    );
    assert_eq!(want.rejected["duplicate"], 1);
    assert_eq!(want.rejected["too_short"], 1);

    // The GUI path, into the same folder: every file byte for byte, including the manifest.
    std::fs::remove_dir_all(&out).unwrap();
    let got = do_prepare(&p, StopCondition::none(), &mut |_| {}).unwrap();
    for (f, w) in files.iter().zip(&want_bytes) {
        assert_eq!(
            &read(&out, f),
            w,
            "{f} differs from the CLI library's output"
        );
    }
    assert_eq!(got.summary, want.summary(&o));
    assert_eq!(got.files, 2);
    assert_eq!(
        [got.train.documents, got.val.documents, got.test.documents],
        want.documents
    );
    assert_eq!(got.train.path, out.join("train.txt").to_string_lossy());
    assert_eq!(got.train.bytes, read(&out, "train.txt").len() as u64);
    assert_eq!(got.rejected, want.rejected);
    assert_eq!(
        got.manifest_path,
        out.join("manifest.json").to_string_lossy()
    );
    assert_eq!(got.lossy_repaired, 0);
    // NFKC folded the "fi" ligature in the splits.
    assert!(!String::from_utf8(read(&out, "train.txt"))
        .unwrap()
        .contains('\u{FB01}'));
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn errors_name_the_problem_in_the_guis_words() {
    let d = tmp("errors");
    let out = d.join("out");
    let f = d.join("one.txt");
    std::fs::write(&f, "   \n  ").unwrap();

    // Nothing usable in the input: the empty-train-split message.
    let e = do_prepare(&params(&[&f], &out), StopCondition::none(), &mut |_| {}).unwrap_err();
    assert!(e.starts_with("train split is empty"), "{e}");
    assert!(
        e.contains("change the document separator") && !e.contains("--doc-sep"),
        "{e}"
    );
    assert!(
        !out.join("train.txt").exists(),
        "a failed run writes nothing"
    );

    // Invalid UTF-8 names the file and the byte, and points at the checkbox, not at a flag.
    let bad = d.join("bad.txt");
    std::fs::write(&bad, b"fine text here\xff and more").unwrap();
    let e = do_prepare(&params(&[&bad], &out), StopCondition::none(), &mut |_| {}).unwrap_err();
    assert!(
        e.contains(&*bad.to_string_lossy()) && e.contains("invalid UTF-8 at byte 14"),
        "{e}"
    );
    assert!(
        e.contains("tick 'Replace invalid UTF-8 bytes'") && !e.contains("--lossy"),
        "{e}"
    );
    let ok = PrepareParams {
        lossy: true,
        val_frac: 0.0,
        test_frac: 0.0,
        ..params(&[&bad], &out)
    };
    let mut noop = |_| {};
    let r = do_prepare(&ok, StopCondition::none(), &mut noop).unwrap();
    assert_eq!((r.lossy_repaired, r.train.documents), (1, 1));
    assert!(r.summary.ends_with("lossy_repaired=1"), "{}", r.summary);

    // Form problems are caught before anything runs.
    let none = PrepareParams {
        inputs: vec!["  ".into()],
        ..params(&[], &out)
    };
    assert!(do_prepare(&none, StopCondition::none(), &mut noop)
        .unwrap_err()
        .contains("at least one input"));
    let missing = params(&[&d.join("nope.txt")], &out);
    assert!(do_prepare(&missing, StopCondition::none(), &mut noop)
        .unwrap_err()
        .contains("does not exist"));
    let dev = params(&[Path::new("/dev/null")], &out);
    assert!(do_prepare(&dev, StopCondition::none(), &mut noop)
        .unwrap_err()
        .contains("not allowed"));
    let into_file = params(&[&f], &f);
    assert!(do_prepare(&into_file, StopCondition::none(), &mut noop)
        .unwrap_err()
        .contains("not a directory"));
    let fr = PrepareParams {
        val_frac: 0.6,
        test_frac: 0.6,
        ..params(&[&f], &out)
    };
    assert!(do_prepare(&fr, StopCondition::none(), &mut noop)
        .unwrap_err()
        .contains("sum to less than 1"));
    // An input that is one of the run's own outputs is refused.
    let own = params(&[&out.join("train.txt")], &out);
    std::fs::write(out.join("train.txt"), "old output").unwrap();
    assert!(do_prepare(&own, StopCondition::none(), &mut noop)
        .unwrap_err()
        .contains("inside --out"));
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn progress_streams_and_a_cancel_mid_run_leaves_the_previous_outputs_untouched() {
    let d = tmp("cancel");
    let big = d.join("big.txt");
    let docs: Vec<String> = (0..5000)
        .map(|i| format!("a document of text number {i}"))
        .collect();
    std::fs::write(&big, docs.join(DOC_SEP)).unwrap();
    let out = d.join("out");
    let p = params(&[&big], &out);

    let mut seen = Vec::new();
    do_prepare(&p, StopCondition::none(), &mut |x| seen.push(x)).unwrap();
    assert!(
        seen.len() >= 4,
        "start, every 2048 documents, end: {seen:?}"
    );
    assert!(seen.windows(2).all(|w| w[0].bytes_done <= w[1].bytes_done));
    let last = seen.last().unwrap();
    assert_eq!((last.bytes_done, last.documents), (last.bytes_total, 5000));
    let before = read(&out, "train.txt");

    // Cancel as soon as the first mid-file report arrives.
    let stop = StopCondition::none();
    let flag = stop.cancel_flag();
    let e = do_prepare(&p, stop, &mut |x| {
        if x.documents >= 2048 {
            flag.store(true, Ordering::Relaxed);
        }
    })
    .unwrap_err();
    assert_eq!(e, slm_cli::prepare::CANCELLED);
    assert_eq!(
        read(&out, "train.txt"),
        before,
        "the previous output is untouched"
    );
    assert!(!out.join("train.txt.part").exists() && !out.join("manifest.json.part").exists());
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn the_folder_walk_skips_hidden_files_and_the_output_folder() {
    let d = tmp("walk");
    let (_, sub) = corpus(&d);
    let out = d.join("more").join("out"); // inside the input folder
    let r = do_prepare(&params(&[&sub], &out), StopCondition::none(), &mut |_| {}).unwrap();
    assert_eq!(r.files, 1, "only b.txt: .hidden is skipped");
    // A second run does not feed the first run's outputs back in.
    let again = do_prepare(&params(&[&sub], &out), StopCondition::none(), &mut |_| {}).unwrap();
    assert_eq!(again.files, 1);
    assert_eq!(again.train.documents, r.train.documents);
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn a_form_with_only_inputs_and_an_output_folder_gets_the_cli_defaults() {
    let p: PrepareParams =
        serde_json::from_str(r#"{"inputs": ["a.txt"], "outDir": "out"}"#).unwrap();
    assert_eq!(
        (p.val_frac, p.test_frac, p.seed, p.min_chars),
        (0.05, 0.05, 42, 1)
    );
    assert_eq!((p.max_chars, p.doc_sep.as_str()), (0, DOC_SEP));
    assert!(!(p.dedup || p.nfkc || p.lowercase || p.strip_gutenberg || p.lossy));
}
