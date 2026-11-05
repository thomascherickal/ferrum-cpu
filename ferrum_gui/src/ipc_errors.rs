//! Mock-runtime IPC tests for the failure paths: bad arguments for every command, and each
//! refusal a command makes before or while it works. Tiny hand-built GGUFs stand in for models.

use crate::ipc_tests::{app, call, handler_commands, msg, tmp};
use crate::test_support::Tiny;
use serde_json::{json, Value};

/// A command and its arguments in call order, each with a valid value. Every command with
/// arguments is listed (a test checks the list against the registry), so each argument's
/// "cannot be decoded" arm runs.
fn arg_table() -> Vec<(&'static str, Vec<(&'static str, Value)>)> {
    let p = |name: &'static str| (name, vec![("params", json!({}))]);
    vec![
        (
            "download_text",
            vec![("url", json!("http://x")), ("maxBytes", json!(1))],
        ),
        ("clean_text", vec![("raw", json!("x")), ("opts", json!({}))]),
        (
            "save_corpus",
            vec![("text", json!("x")), ("path", json!("p"))],
        ),
        (
            "read_text_file",
            vec![("path", json!("p")), ("maxBytes", json!(1))],
        ),
        p("train_slm"),
        p("generate_slm"),
        p("evaluate_slm"),
        p("prepare_dataset"),
        p("preview_runtime_export"),
        p("export_to_runtime"),
        p("tokenize_corpus"),
        p("serve_start"),
        ("model_info", vec![("path", json!("p"))]),
        ("gguf_info", vec![("path", json!("p"))]),
        p("run_gguf"),
        p("finetune_gguf"),
        p("export_gguf"),
        ("cancel_job", vec![("kind", json!("train"))]),
        (
            "estimate_job",
            vec![("kind", json!("train")), ("params", json!({}))],
        ),
        p("preflight_train"),
        (
            "download_dataset",
            vec![("id", json!("x")), ("destDir", json!("d"))],
        ),
        (
            "download_hf_file",
            vec![
                ("repo", json!("o/n")),
                ("file", json!("f")),
                ("revision", json!("main")),
                ("destDir", json!("d")),
                ("hfToken", json!("t")),
            ],
        ),
        (
            "download_kaggle_file",
            vec![
                ("ownerSlug", json!("o/n")),
                ("file", json!("f")),
                ("destDir", json!("d")),
            ],
        ),
    ]
}

/// Commands that take nothing from the webview.
const NO_ARGS: [&str; 6] = [
    "serve_stop",
    "serve_status",
    "system_stats",
    "hardware_info",
    "capability_report",
    "list_datasets",
];

#[test]
fn every_argument_of_every_command_is_checked_before_it_runs() {
    let (_app, w) = app();
    let table = arg_table();
    let mut known: Vec<&str> = table.iter().map(|(c, _)| *c).chain(NO_ARGS).collect();
    known.sort_unstable();
    let mut registered = handler_commands(include_str!("lib.rs"));
    registered.sort_unstable();
    assert_eq!(
        known, registered,
        "a new command needs its arguments listed here"
    );
    for (cmd, args) in &table {
        for bad in 0..args.len() {
            // Valid values before the argument under test, then a boolean in its place.
            let mut body = serde_json::Map::new();
            for (name, v) in &args[..bad] {
                body.insert((*name).into(), v.clone());
            }
            body.insert(args[bad].0.into(), json!(true));
            let e = call(&w, cmd, Value::Object(body)).unwrap_err();
            let text = e.to_string();
            assert!(
                text.contains(args[bad].0),
                "{cmd}: argument {} not named in {text}",
                args[bad].0
            );
        }
    }
}

#[test]
fn commands_without_arguments_answer_through_ipc() {
    let (_app, w) = app();
    assert_eq!(
        call(&w, "serve_status", json!({})).unwrap()["running"],
        false
    );
    // Stopping a server that is not running is not an error.
    let stopped = call(&w, "serve_stop", json!({})).unwrap();
    assert_eq!(
        (stopped["running"].clone(), stopped["requests"].clone()),
        (json!(false), json!(0))
    );
    let sets = call(&w, "list_datasets", json!({})).unwrap();
    assert!(sets
        .as_array()
        .unwrap()
        .iter()
        .any(|d| d["id"] == "tinystories-valid"));
    let cap = call(&w, "capability_report", json!({})).unwrap();
    assert!(cap["memBwGbps"].as_f64().unwrap() > 0.0, "{cap}");
    assert!(cap["loadInt8"].as_f64().is_some(), "{cap}");
}

fn ft_params(dir: &std::path::Path, model: &str, corpus: &str, out: &str) -> Value {
    json!({"params": {
        "modelPath": dir.join(model), "corpusPath": dir.join(corpus), "outPath": dir.join(out),
        "epochs": 1, "lr": 0.001, "batchSize": 2, "seqLen": 8, "warmup": 0, "clip": 1.0,
        "weightDecay": 0, "dropout": 0, "qat": false, "seed": 1, "threads": 1, "resume": null,
        "force": false, "verbose": false}})
}

#[test]
fn finetune_refuses_bad_input_and_resumes_from_its_own_checkpoint() {
    let (_app, w) = app();
    let dir = tmp("ft_errors");
    let write = |name: &str, bytes: &[u8]| std::fs::write(dir.join(name), bytes).unwrap();
    write("base.gguf", &Tiny::default().build());
    write(
        "corpus.txt",
        "hi abc hi abc hi abc hi abc hi abc hi abc ".as_bytes(),
    );
    let fails = |params: Value, want: &str| {
        let e = call(&w, "finetune_gguf", params).unwrap_err();
        assert!(msg(&e).contains(want), "wanted {want:?}, got {e}");
    };
    let with = |key: &str, v: Value| {
        let mut p = ft_params(&dir, "base.gguf", "corpus.txt", "out.flck");
        p["params"][key] = v;
        p
    };
    fails(
        with("resume", json!(dir.join("none.flck"))),
        "resume checkpoint path",
    );
    fails(with("epochs", json!(0)), "epochs must be");
    fails(with("lr", json!(0)), "learning rate");

    write("junk.gguf", b"not a gguf");
    fails(
        ft_params(&dir, "junk.gguf", "corpus.txt", "out.flck"),
        "cannot open",
    );
    write("bare.gguf", &{
        let mut b = ferrum_core::GgufBuilder::new();
        b.meta(
            "general.architecture",
            ferrum_core::MetaValue::String("llama".into()),
        );
        b.into_bytes()
    });
    fails(
        ft_params(&dir, "bare.gguf", "corpus.txt", "out.flck"),
        "no tokenizer",
    );
    write(
        "empty.gguf",
        &Tiny {
            with_tensors: false,
            ..Tiny::default()
        }
        .build(),
    );
    fails(
        ft_params(&dir, "empty.gguf", "corpus.txt", "out.flck"),
        "cannot load model",
    );
    write(
        "wide.gguf",
        &Tiny {
            vocab: 4,
            tok_count: Some(10),
            ..Tiny::default()
        }
        .build(),
    );
    fails(
        ft_params(&dir, "wide.gguf", "corpus.txt", "out.flck"),
        "fine-tuning failed",
    );

    write("binary.txt", &[0xff, 0xfe, 0xfd]);
    fails(
        ft_params(&dir, "base.gguf", "binary.txt", "out.flck"),
        "cannot read corpus",
    );
    write("blank.txt", b"  \n ");
    fails(
        ft_params(&dir, "base.gguf", "blank.txt", "out.flck"),
        "is empty",
    );
    write("short.txt", b"hi");
    fails(
        ft_params(&dir, "base.gguf", "short.txt", "out.flck"),
        "supply more text",
    );
    fails(
        ft_params(&dir, "base.gguf", "corpus.txt", "no_such_dir/out.flck"),
        "cannot write",
    );

    // A good run, then a resume with a warmup schedule and the automatic thread count.
    let first = call(
        &w,
        "finetune_gguf",
        ft_params(&dir, "base.gguf", "corpus.txt", "a.flck"),
    )
    .unwrap();
    assert_eq!(first["stopped"], false);
    let mut again = ft_params(&dir, "base.gguf", "corpus.txt", "b.flck");
    again["params"]["resume"] = json!(dir.join("a.flck"));
    again["params"]["warmup"] = json!(2);
    again["params"]["threads"] = json!(0);
    let second = call(&w, "finetune_gguf", again).unwrap();
    assert!(
        second["steps"].as_u64() > first["steps"].as_u64(),
        "{first} {second}"
    );
    // A checkpoint that is not one cannot be resumed from; an unreadable one says so.
    let mut bad = ft_params(&dir, "base.gguf", "corpus.txt", "c.flck");
    bad["params"]["resume"] = json!(dir.join("junk.gguf"));
    fails(bad, "cannot resume from");
    #[cfg(unix)]
    {
        let locked = dir.join("locked.flck");
        write("locked.flck", b"x");
        if let Some(_restore) = crate::test_support::restrict(&locked, 0o000, 0o600) {
            let mut p = ft_params(&dir, "base.gguf", "corpus.txt", "d.flck");
            p["params"]["resume"] = json!(locked);
            fails(p, "cannot read checkpoint");
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn finetune_resume_on_another_corpus_or_batch_is_refused() {
    let (_app, w) = app();
    let dir = tmp("ft_other_corpus");
    std::fs::write(dir.join("base.gguf"), Tiny::default().build()).unwrap();
    std::fs::write(
        dir.join("a.txt"),
        "hi abc hi abc hi abc hi abc hi abc hi abc ",
    )
    .unwrap();
    std::fs::write(
        dir.join("b.txt"),
        "hi abd hi abd hi abd hi abd hi abd hi abd ",
    )
    .unwrap();
    call(
        &w,
        "finetune_gguf",
        ft_params(&dir, "base.gguf", "a.txt", "a.flck"),
    )
    .unwrap();
    let ckpt = dir.join("a.flck");
    let before = std::fs::read(&ckpt).unwrap();
    // Resuming in place on another corpus is refused, naming the corpus, and writes nothing.
    let mut p = ft_params(&dir, "base.gguf", "b.txt", "a.flck");
    p["params"]["resume"] = json!(ckpt);
    let e = call(&w, "finetune_gguf", p.clone()).unwrap_err();
    assert!(
        msg(&e).contains("cannot resume")
            && msg(&e).contains("corpus")
            && msg(&e).contains("sha256:"),
        "{e}"
    );
    p["params"]["corpusPath"] = json!(dir.join("a.txt"));
    p["params"]["batchSize"] = json!(3);
    let e = call(&w, "finetune_gguf", p.clone()).unwrap_err();
    assert!(
        msg(&e).contains("--batch is 3 but the checkpoint was trained with batch 2"),
        "{e}"
    );
    assert_eq!(std::fs::read(&ckpt).unwrap(), before);
    // The corpus and batch it was started with resume.
    p["params"]["batchSize"] = json!(2);
    call(&w, "finetune_gguf", p).unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

fn train_params(dir: &std::path::Path, corpus: &str) -> Value {
    json!({"params": {
        "method": "transformer", "corpusPath": dir.join(corpus), "modelPath": dir.join("m.gguf"),
        "contextLen": 8, "embedDim": 16, "numHeads": 2, "numBlocks": 1, "hiddenDim": 32,
        "epochs": 1, "lr": 0.01, "batchSize": 4, "vocabSize": 0, "seed": 1, "threads": 1,
        "verbose": false}})
}

#[test]
fn train_refuses_bad_forms_and_trains_character_level_with_validation() {
    let (_app, w) = app();
    let dir = tmp("train_errors");
    std::fs::write(dir.join("c.txt"), "abc abd abe ".repeat(60)).unwrap();
    std::fs::write(dir.join("blank.txt"), " \n").unwrap();
    let fails = |corpus: &str, key: &str, v: Value, want: &str| {
        let mut p = train_params(&dir, corpus);
        p["params"][key] = v.clone();
        let e = call(&w, "train_slm", p).unwrap_err();
        assert!(
            msg(&e).contains(want),
            "{key}={v}: wanted {want:?}, got {e}"
        );
    };
    fails("c.txt", "contextLen", json!(0), "context length must be");
    fails("c.txt", "epochs", json!(0), "epochs must be");
    fails("c.txt", "batchSize", json!(0), "batch size must be");
    fails("c.txt", "hiddenDim", json!(0), "hidden width must be");
    fails("c.txt", "lr", json!(0), "learning rate must be");
    fails("c.txt", "numBlocks", json!(0), "transformer blocks must be");
    fails("c.txt", "vocabSize", json!(100), "vocab must be 0");
    fails(
        "c.txt",
        "method",
        json!("mlp"),
        "unknown training method: mlp",
    );
    fails("blank.txt", "epochs", json!(1), "is empty");
    // The engine's own refusal reaches the UI as "training failed".
    fails("c.txt", "valFraction", json!(0.999), "training failed");

    // Character level, with a held-out split: the result says so and a model file exists.
    let mut p = train_params(&dir, "c.txt");
    p["params"]["valFraction"] = json!(0.2);
    let r = call(&w, "train_slm", p).unwrap();
    assert_eq!(r["tokenizer"], "character-level");
    assert!(r["bytes"].as_u64().unwrap() > 0);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn train_saves_the_chosen_precision_and_refuses_an_unknown_one() {
    let (_app, w) = app();
    let dir = tmp("train_quant");
    std::fs::write(dir.join("c.txt"), "abc abd abe ".repeat(60)).unwrap();
    let mut sizes = vec![];
    // Absent = int8 (what every older caller got), then each value explicitly.
    for (quant, want) in [
        (None, "int8"),
        (Some("f32"), "f32"),
        (Some("int8"), "int8"),
        (Some("int4"), "int4"),
    ] {
        let mut p = train_params(&dir, "c.txt");
        if let Some(q) = quant {
            p["params"]["quant"] = json!(q);
        }
        let r = call(&w, "train_slm", p).unwrap();
        let bytes = std::fs::read(dir.join("m.gguf")).unwrap();
        assert_eq!(ferrum_core::gguf_quant(&bytes).unwrap(), want);
        assert_eq!(r["bytes"].as_u64().unwrap(), bytes.len() as u64);
        sizes.push(bytes.len());
    }
    assert_eq!(sizes[0], sizes[2], "absent and int8 write the same file");
    assert!(sizes[1] > sizes[2] && sizes[2] > sizes[3], "{sizes:?}");
    // An unknown value is refused before anything is trained or written.
    std::fs::remove_file(dir.join("m.gguf")).unwrap();
    let mut p = train_params(&dir, "c.txt");
    p["params"]["quant"] = json!("int2");
    let e = call(&w, "train_slm", p).unwrap_err();
    assert!(
        msg(&e).contains("quant must be int4 | int8 | f32 (got 'int2')"),
        "{e}"
    );
    assert!(!dir.join("m.gguf").exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn train_resume_other_corpus_is_refused() {
    let (_app, w) = app();
    let dir = tmp("train_other_corpus");
    std::fs::write(dir.join("a.txt"), "abc abd abe ".repeat(60)).unwrap();
    std::fs::write(dir.join("b.txt"), "abc abd abf ".repeat(60)).unwrap();
    let ckpt = dir.join("r.fckp");
    let mut p = train_params(&dir, "a.txt");
    p["params"]["checkpointPath"] = json!(ckpt);
    call(&w, "train_slm", p).unwrap();
    let before = std::fs::read(&ckpt).unwrap();
    // Resuming on another corpus is refused, naming the corpus, and leaves the checkpoint alone.
    let mut p = train_params(&dir, "b.txt");
    p["params"]["checkpointPath"] = json!(ckpt);
    p["params"]["resume"] = json!(true);
    p["params"]["epochs"] = json!(2);
    let e = call(&w, "train_slm", p.clone()).unwrap_err();
    assert!(
        msg(&e).contains("training failed")
            && msg(&e).contains("corpus")
            && msg(&e).contains("sha256:"),
        "{e}"
    );
    assert_eq!(std::fs::read(&ckpt).unwrap(), before);
    // The corpus it was started with resumes.
    p["params"]["corpusPath"] = json!(dir.join("a.txt"));
    call(&w, "train_slm", p).unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn train_resume_with_another_batch_or_stride_is_refused_and_a_new_lr_only_noted() {
    let (_app, w) = app();
    let dir = tmp("train_other_settings");
    std::fs::write(dir.join("a.txt"), "abc abd abe ".repeat(60)).unwrap();
    let ckpt = dir.join("r.fckp");
    let mut p = train_params(&dir, "a.txt");
    p["params"]["checkpointPath"] = json!(ckpt);
    call(&w, "train_slm", p.clone()).unwrap();
    let before = std::fs::read(&ckpt).unwrap();
    p["params"]["resume"] = json!(true);
    p["params"]["epochs"] = json!(2);
    let mut other = p.clone();
    other["params"]["batchSize"] = json!(2);
    let e = msg(&call(&w, "train_slm", other).unwrap_err()).to_string();
    assert!(e.contains("--batch is 2") && e.contains("batch 4"), "{e}");
    let mut other = p.clone();
    other["params"]["windowStride"] = json!(3);
    let e = msg(&call(&w, "train_slm", other).unwrap_err()).to_string();
    assert!(e.contains("--stride is 3"), "{e}");
    assert_eq!(std::fs::read(&ckpt).unwrap(), before);
    // A new learning rate resumes (the CLI prints a note; the GUI log shows the same line).
    p["params"]["lr"] = json!(0.005);
    call(&w, "train_slm", p).unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

fn run_params(model: &std::path::Path, extra: Value) -> Value {
    let mut p = json!({"params": {"modelPath": model, "prompt": "hi", "quant": "f32", "maxNew": 3,
        "temp": 0, "genSeed": 1, "ids": null, "force": false, "resume": null, "raw": true}});
    for (k, v) in extra.as_object().unwrap() {
        p["params"][k] = v.clone();
    }
    p
}

#[test]
fn gguf_run_stop_strings_cut_the_text_and_need_a_tokenizer() {
    let (_app, w) = app();
    let dir = tmp("gguf_stop");
    let model = dir.join("m.gguf");
    std::fs::write(&model, Tiny::default().build()).unwrap();
    let run = |extra: Value| {
        let r = call(&w, "run_gguf", run_params(&model, extra)).unwrap();
        r["text"].as_str().unwrap().to_string()
    };
    let more = json!({"maxNew": 12});
    let full = run(more.clone());
    let cont = &full["hi".len()..];
    assert!(cont.chars().count() >= 3, "{full:?}");
    // No stop, an empty line and a stop that never appears are all the same run.
    assert_eq!(run(json!({"maxNew": 12, "stop": []})), full);
    assert_eq!(run(json!({"maxNew": 12, "stop": ["", "zzzz"]})), full);
    let ch: String = cont.chars().nth(2).unwrap().to_string();
    let want = format!("hi{}", &cont[..cont.find(&ch).unwrap()]);
    assert_eq!(run(json!({"maxNew": 12, "stop": [ch.clone()]})), want);
    // The earliest match wins whatever the order of the lines.
    let first: String = cont.chars().next().unwrap().to_string();
    let want = format!("hi{}", &cont[..cont.find(&first).unwrap()]);
    assert_eq!(
        run(json!({"maxNew": 12, "stop": [ch.clone(), first.clone()]})),
        want
    );
    assert_eq!(run(json!({"maxNew": 12, "stop": [first, ch]})), want);

    // Without a tokenizer a stop string cannot be matched: refused before loading.
    let bare = dir.join("bare.gguf");
    std::fs::write(
        &bare,
        Tiny {
            tok_count: Some(0),
            ..Tiny::default()
        }
        .build(),
    )
    .unwrap();
    let e = call(
        &w,
        "run_gguf",
        run_params(&bare, json!({"ids": "1 2", "stop": ["a"]})),
    )
    .unwrap_err();
    assert!(msg(&e).contains("stop strings need a tokenizer"), "{e}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn gguf_info_and_run_cover_files_without_a_tokenizer_and_instruct_templates() {
    let (_app, w) = app();
    let dir = tmp("gguf_run");
    let write = |name: &str, bytes: Vec<u8>| {
        std::fs::write(dir.join(name), bytes).unwrap();
        dir.join(name)
    };
    // An architecture Ferrum cannot run, with no tokenizer: described, not runnable.
    let mut b = ferrum_core::GgufBuilder::new();
    b.meta(
        "general.architecture",
        ferrum_core::MetaValue::String("mamba".into()),
    );
    let mamba = write("mamba.gguf", b.into_bytes());
    let info = call(&w, "gguf_info", json!({"path": mamba})).unwrap();
    assert_eq!(info["runnable"], false);
    assert!(
        info["note"]
            .as_str()
            .unwrap()
            .contains("'mamba' is not runnable"),
        "{info}"
    );
    assert_eq!(info["tokenizer"], "none in file (use token IDs)");

    // Without a tokenizer a run needs token ids, and answers with ids.
    let bare = write(
        "bare.gguf",
        Tiny {
            tok_count: Some(0),
            ..Tiny::default()
        }
        .build(),
    );
    let e = call(&w, "run_gguf", run_params(&bare, json!({}))).unwrap_err();
    assert!(msg(&e).contains("no tokenizer"), "{e}");
    let e = call(&w, "run_gguf", run_params(&bare, json!({"ids": "x y"}))).unwrap_err();
    assert!(msg(&e).contains("empty prompt"), "{e}");
    let r = call(
        &w,
        "run_gguf",
        run_params(&bare, json!({"ids": "1 2 junk 3"})),
    )
    .unwrap();
    assert_eq!(r["promptTokens"], 3);
    let ids: Vec<&str> = r["text"].as_str().unwrap().split(' ').collect();
    assert_eq!(ids.len() as u64, r["generated"].as_u64().unwrap());
    assert!(ids.iter().all(|i| i.parse::<usize>().is_ok()), "{r}");

    // An instruct template wraps the prompt and adds the template's stop token.
    let chat = write(
        "chat.gguf",
        Tiny {
            chat_template: Some("<|im_start|>user {{ prompt }}<|im_end|>"),
            pieces: &[(10, "<|im_start|>"), (11, "<|im_end|>")],
            ..Tiny::default()
        }
        .build(),
    );
    let raw = call(&w, "run_gguf", run_params(&chat, json!({}))).unwrap();
    let wrapped = call(&w, "run_gguf", run_params(&chat, json!({"raw": false}))).unwrap();
    assert!(
        wrapped["promptTokens"].as_u64() > raw["promptTokens"].as_u64(),
        "{raw} {wrapped}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The fixtures above shape every call in this file; these pin them to the command types.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::{FinetuneParams, GgufRunParams, TrainParams};

    #[test]
    fn param_fixtures_deserialize_into_the_command_types() {
        let d = std::path::Path::new("/d");
        let ft = ft_params(d, "m.gguf", "c.txt", "o.flck");
        let p: FinetuneParams = serde_json::from_value(ft["params"].clone()).unwrap();
        assert_eq!(p.out_path, "/d/o.flck");
        let tr: TrainParams =
            serde_json::from_value(train_params(d, "c.txt")["params"].clone()).unwrap();
        assert_eq!(
            (tr.corpus_path.as_str(), tr.model_path.as_str()),
            ("/d/c.txt", "/d/m.gguf")
        );
        let run = run_params(&d.join("m.gguf"), json!({"maxNew": 9, "raw": false}));
        let r: GgufRunParams = serde_json::from_value(run["params"].clone()).unwrap();
        assert_eq!(
            (r.max_new, r.raw, r.prompt.as_str()),
            (9, false, "hi"),
            "overrides kept the rest"
        );
    }

    #[test]
    fn arg_table_lists_each_command_once_with_distinct_arguments() {
        let table = arg_table();
        let mut names: Vec<&str> = table.iter().map(|(c, _)| *c).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), table.len(), "a command listed twice");
        assert!(
            names.iter().all(|n| !NO_ARGS.contains(n)),
            "a command in both lists"
        );
        for (cmd, args) in &table {
            assert!(!args.is_empty(), "{cmd} belongs in NO_ARGS");
            let mut a: Vec<&str> = args.iter().map(|(n, _)| *n).collect();
            a.sort_unstable();
            a.dedup();
            assert_eq!(a.len(), args.len(), "{cmd}: an argument listed twice");
        }
    }
}
