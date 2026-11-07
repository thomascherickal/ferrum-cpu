use super::*;

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_prep_lib_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn opts(inputs: Vec<String>, out: &Path) -> PrepareOptions {
    PrepareOptions {
        inputs,
        out: out.to_string_lossy().into(),
        doc_sep: "<|endoftext|>".into(),
        lossy: false,
        nfkc: false,
        lowercase: false,
        strip_gutenberg: false,
        min_chars: 1,
        max_chars: 0,
        dedup: false,
        val_frac: 0.0,
        test_frac: 0.0,
        seed: 1,
        args: vec![],
        stop: StopCondition::none(),
    }
}

fn run(o: &PrepareOptions) -> Result<PrepareReport, String> {
    prepare(o, &mut |_| {})
}

#[test]
fn rejected_summary_lists_nonzero_reasons_in_key_order() {
    assert!(write_manifest(Path::new("/nonexistent_dir_12345/m.json"), "").is_err());
    let mut r: BTreeMap<String, u64> = BTreeMap::new();
    r.insert("duplicate".into(), 0);
    r.insert("empty".into(), 0);
    assert_eq!(rejected_summary(&r), "none");
    r.insert("duplicate".into(), 3);
    r.insert("too_short".into(), 1);
    assert_eq!(rejected_summary(&r), "duplicate=3 too_short=1");
    assert_eq!(rejected_summary(&BTreeMap::new()), "none");
}

#[test]
fn every_rejection_reason_is_counted_in_the_manifest() {
    let d = tmpdir("reasons");
    let sep = "<|endoftext|>";
    let mut data = b"bad \xff bytes here".to_vec();
    for doc in [
        "   ",           // empty
        "tiny",          // too_short
        &"x".repeat(60), // too_long
        "a perfectly fine doc",
        "a perfectly fine doc", // duplicate
    ] {
        data.extend_from_slice(sep.as_bytes());
        data.extend_from_slice(doc.as_bytes());
    }
    std::fs::write(d.join("in.txt"), &data).unwrap();
    let mk = |lossy| PrepareOptions {
        lossy,
        min_chars: 5,
        max_chars: 40,
        dedup: true,
        ..opts(
            vec![d.join("in.txt").to_string_lossy().into()],
            &d.join("out"),
        )
    };
    // Invalid UTF-8 without --lossy is a clean error naming the flag.
    let e = run(&mk(false)).unwrap_err();
    assert!(e.contains("--lossy"), "{e}");
    let rep = run(&mk(true)).unwrap();
    let m: Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("out/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(
        m, rep.manifest,
        "the report carries the manifest as written"
    );
    let r = &m["rejected"];
    assert_eq!(r["invalid_utf8"], 0);
    assert_eq!(m["lossy_repaired"], 1);
    assert_eq!(r["empty"], 1);
    assert_eq!(r["too_short"], 1);
    assert_eq!(r["too_long"], 1);
    assert_eq!(r["duplicate"], 1);
    // The lossy doc (replacement char) and one "fine" doc are kept: 2 train docs.
    assert_eq!(m["outputs"]["train"]["documents"], 2);
    assert_eq!(rep.documents, [2, 0, 0]);
    assert_eq!(rep.lossy_repaired, 1);
    assert_eq!(rep.rejected["empty"], 1);
    assert_eq!(rep.manifest_path, d.join("out/manifest.json"));
    assert!(rep.summary(&mk(true)).ends_with(
        "(train=2 val=0 test=0)  rejected duplicate=1 empty=1 too_long=1 too_short=1  lossy_repaired=1"
    ));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn empty_train_split_and_bad_fractions_are_clean_errors_and_write_nothing() {
    let d = tmpdir("errors");
    std::fs::write(d.join("in.txt"), "   ").unwrap();
    let o = opts(
        vec![d.join("in.txt").to_string_lossy().into()],
        &d.join("out"),
    );
    let o_ok = PrepareOptions {
        val_frac: 0.5,
        test_frac: 0.5,
        ..opts(vec![], &d.join("out"))
    };
    assert!(run(&o_ok).unwrap_err().contains("sum to less than 1"));
    // The only document is empty, so nothing is left for train.
    let e = run(&o).unwrap_err();
    assert!(
        e.starts_with("train split is empty: 0 documents found"),
        "{e}"
    );
    let names: Vec<_> = std::fs::read_dir(d.join("out"))
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert!(
        names.is_empty(),
        "a failed run leaves nothing behind: {names:?}"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn progress_is_reported_per_file_and_a_stop_ends_the_run_with_nothing_written() {
    let d = tmpdir("progress");
    for n in 0..3 {
        std::fs::write(d.join(format!("f{n}.txt")), format!("document number {n}")).unwrap();
    }
    let inputs = vec![d.to_string_lossy().into()];
    let out = d.join("out");
    let mut seen = Vec::new();
    run_with(&opts(inputs.clone(), &out), &mut seen);
    // One report at the start of each file, plus the closing one.
    assert_eq!(seen.len(), 4);
    assert_eq!((seen[0].file, seen[0].files, seen[0].bytes_done), (1, 3, 0));
    let last = seen.last().unwrap();
    assert_eq!((last.bytes_done, last.documents), (last.bytes_total, 3));
    assert!(last.bytes_total > 0);

    // A stop requested before the first document: error, no files, previous outputs kept.
    std::fs::write(out.join("train.txt"), "previous").unwrap();
    let o = opts(inputs, &out);
    o.stop
        .cancel_flag()
        .store(true, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(run(&o).unwrap_err(), CANCELLED);
    assert_eq!(
        std::fs::read_to_string(out.join("train.txt")).unwrap(),
        "previous"
    );
    assert!(!out.join("train.txt.part").exists());
    let _ = std::fs::remove_dir_all(&d);
}

fn run_with(o: &PrepareOptions, seen: &mut Vec<PrepareProgress>) {
    prepare(o, &mut |p| seen.push(p)).unwrap();
}

#[test]
fn a_progress_report_comes_every_progress_every_documents() {
    let d = tmpdir("every");
    let n = PROGRESS_EVERY as usize + 1;
    let text = (0..n)
        .map(|i| format!("doc {i}"))
        .collect::<Vec<_>>()
        .join("<|endoftext|>");
    std::fs::write(d.join("c.txt"), text).unwrap();
    let mut seen = Vec::new();
    run_with(
        &opts(
            vec![d.join("c.txt").to_string_lossy().into()],
            &d.join("out"),
        ),
        &mut seen,
    );
    // The start of the file, the report at document PROGRESS_EVERY, and the closing one.
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[1].documents, PROGRESS_EVERY);
    assert_eq!(seen[2].documents, n as u64);
    let _ = std::fs::remove_dir_all(&d);
}

/// Four `*.part` files in `d`, as `build` leaves them before the commit.
fn stage(d: &Path, tag: &str) {
    for n in OUTPUTS {
        std::fs::write(part(d, n), format!("{tag} {n}")).unwrap();
    }
}

#[test]
fn commit_renames_the_manifest_last() {
    let d = tmpdir("commit_order");
    stage(&d, "new");
    let order = std::cell::RefCell::new(Vec::new());
    commit_outputs(&d, &|from, to| {
        order.borrow_mut().push(to.file_name().unwrap().to_owned());
        std::fs::rename(from, to)
    })
    .unwrap();
    assert_eq!(
        *order.borrow(),
        ["train.txt", "val.txt", "test.txt", "manifest.json"]
    );
    assert_eq!(
        std::fs::read_to_string(d.join("manifest.json")).unwrap(),
        "new manifest.json"
    );
}

#[test]
fn failed_rename_leaves_no_manifest() {
    let d = tmpdir("failed_rename");
    stage(&d, "old");
    commit_outputs(&d, &|f, t| std::fs::rename(f, t)).unwrap();
    stage(&d, "new");
    let calls = std::cell::Cell::new(0);
    let err = commit_outputs(&d, &|from, to| {
        calls.set(calls.get() + 1);
        if calls.get() == 3 {
            Err(std::io::Error::other("obstacle"))
        } else {
            std::fs::rename(from, to)
        }
    })
    .unwrap_err();
    assert!(
        err.contains("cannot rename") && err.contains("obstacle"),
        "{err}"
    );
    assert!(
        !d.join("manifest.json").exists(),
        "stale or partial manifest left"
    );
    // The obstacle is gone: a second commit of the same staged files succeeds.
    stage(&d, "new");
    commit_outputs(&d, &|f, t| std::fs::rename(f, t)).unwrap();
    for n in OUTPUTS {
        assert_eq!(
            std::fs::read_to_string(d.join(n)).unwrap(),
            format!("new {n}")
        );
        assert!(!part(&d, n).exists());
    }
}

#[test]
fn a_failure_on_the_first_rename_does_not_keep_the_previous_manifest() {
    let d = tmpdir("first_fails");
    std::fs::write(d.join("manifest.json"), "previous").unwrap();
    stage(&d, "new");
    let r = commit_outputs(&d, &|_, _| Err(std::io::Error::other("nope")));
    assert!(r.is_err());
    assert!(!d.join("manifest.json").exists());
}

#[test]
fn a_failing_manifest_rename_leaves_no_manifest() {
    let d = tmpdir("manifest_fails");
    stage(&d, "new");
    let r = commit_outputs(&d, &|from, to| {
        if to.ends_with("manifest.json") {
            Err(std::io::Error::other("nope"))
        } else {
            std::fs::rename(from, to)
        }
    });
    assert!(r.is_err());
    assert!(d.join("test.txt").exists() && !d.join("manifest.json").exists());
    // The splits renamed before the failure hold the new data.
    for n in ["train.txt", "val.txt", "test.txt"] {
        assert_eq!(
            std::fs::read_to_string(d.join(n)).unwrap(),
            format!("new {n}")
        );
    }
}

#[test]
fn a_commit_failure_inside_prepare_removes_every_part_file_and_keeps_old_outputs() {
    let d = tmpdir("commit_fails");
    let input = d.join("in.txt");
    std::fs::write(&input, "alpha one\n<|endoftext|>beta two\n").unwrap();
    let out = d.join("out");
    std::fs::create_dir_all(out.join("manifest.json")).unwrap(); // remove_file cannot delete it
    std::fs::write(out.join("train.txt"), "old train").unwrap();
    let o = opts(vec![input.to_string_lossy().into()], &out);
    let err = prepare(&o, &mut |_| {}).unwrap_err();
    assert!(err.starts_with("cannot remove"), "{err}");
    for n in OUTPUTS {
        assert!(!part(&out, n).exists(), "{n}.part left behind");
    }
    assert_eq!(
        std::fs::read_to_string(out.join("train.txt")).unwrap(),
        "old train"
    );
}

#[test]
fn an_unremovable_old_manifest_is_an_error() {
    let d = tmpdir("manifest_dir");
    std::fs::create_dir(d.join("manifest.json")).unwrap();
    stage(&d, "new");
    let err = commit_outputs(&d, &|f, t| std::fs::rename(f, t)).unwrap_err();
    assert!(err.starts_with("cannot remove"), "{err}");
}

#[test]
fn two_runs_give_byte_identical_outputs() {
    let d = tmpdir("rerun");
    let input = d.join("in.txt");
    std::fs::write(
        &input,
        "alpha one\n<|endoftext|>beta two\n<|endoftext|>gamma three\n",
    )
    .unwrap();
    let out = d.join("out");
    let mut o = opts(vec![input.to_string_lossy().into()], &out);
    o.val_frac = 0.3;
    o.test_frac = 0.3;
    run(&o).unwrap();
    let read = || OUTPUTS.map(|n| std::fs::read(out.join(n)).unwrap());
    let first = read();
    run(&o).unwrap();
    assert_eq!(first, read());
}
