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
        use std::os::unix::fs::PermissionsExt;
        let locked = dir.join("locked.flck");
        write("locked.flck", b"x");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let mut p = ft_params(&dir, "base.gguf", "corpus.txt", "d.flck");
        p["params"]["resume"] = json!(locked);
        fails(p, "cannot read checkpoint");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
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

fn run_params(model: &std::path::Path, extra: Value) -> Value {
    let mut p = json!({"params": {"modelPath": model, "prompt": "hi", "quant": "f32", "maxNew": 3,
        "temp": 0, "genSeed": 1, "ids": null, "force": false, "resume": null, "raw": true}});
    for (k, v) in extra.as_object().unwrap() {
        p["params"][k] = v.clone();
    }
    p
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
