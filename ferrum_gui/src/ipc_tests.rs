//! Mock-runtime IPC tests: the real command registry driven through Tauri's
//! own invoke path (argument deserialization, camelCase names, result
//! serialization), with no window system. Complements the unit tests, which
//! call the Rust functions directly.

use super::configure;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tauri::ipc::{CallbackFn, InvokeBody, InvokeResponseBody};
use tauri::test::{mock_builder, mock_context, noop_assets, INVOKE_KEY};
use tauri::webview::InvokeRequest;
use tauri::{App, Listener, WebviewWindow, WebviewWindowBuilder};

pub(crate) fn app() -> (
    App<tauri::test::MockRuntime>,
    WebviewWindow<tauri::test::MockRuntime>,
) {
    let app = configure(mock_builder())
        .build(mock_context(noop_assets()))
        .expect("mock app");
    let w = WebviewWindowBuilder::new(&app, "main", Default::default())
        .build()
        .expect("mock window");
    (app, w)
}

pub(crate) fn call(
    w: &WebviewWindow<tauri::test::MockRuntime>,
    cmd: &str,
    body: Value,
) -> Result<Value, Value> {
    tauri::test::get_ipc_response(
        w,
        InvokeRequest {
            cmd: cmd.into(),
            callback: CallbackFn(0),
            error: CallbackFn(1),
            url: "tauri://localhost".parse().unwrap(),
            body: InvokeBody::Json(body),
            headers: Default::default(),
            invoke_key: INVOKE_KEY.to_string(),
        },
    )
    .map(|r| match r {
        InvokeResponseBody::Json(s) => serde_json::from_str(&s).unwrap(),
        InvokeResponseBody::Raw(_) => unreachable!(),
    })
}

/// The text of a failed command: errors reach the webview as `{ code, message, hint }`.
pub(crate) fn msg(e: &Value) -> &str {
    e["message"]
        .as_str()
        .expect("an error object with a message")
}

#[test]
fn hardware_info_and_system_stats_answer_through_ipc() {
    let (_app, w) = app();
    let hw = call(&w, "hardware_info", json!({})).unwrap();
    assert!(hw["logicalCores"].as_u64().unwrap() >= 1);
    assert!(hw["gpus"].is_array());
    let s = call(&w, "system_stats", json!({})).unwrap();
    assert_eq!(s["available"], true);
    assert!(s["memTotal"].as_u64().unwrap() > 0);
}

#[test]
fn estimate_job_takes_the_ui_argument_shape_and_returns_camel_case() {
    let (_app, w) = app();
    let e = call(
        &w,
        "estimate_job",
        json!({"kind": "train", "params": {
            "contextLen": 16, "embedDim": 32, "numHeads": 4, "numBlocks": 2, "hiddenDim": 64,
            "epochs": 3, "batchSize": 16, "vocabSize": 0, "threads": 0, "windowStride": 0,
            "maxTokens": 0, "valFraction": 0.0, "corpusPath": ""
        }}),
    )
    .unwrap();
    assert_eq!(e["kind"], "train");
    assert!(e["seconds"].as_f64().unwrap() > 0.0);
    assert_eq!(e["usesGpu"], false);
    assert!(e["notes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|n| n.as_str().unwrap().contains("Corpus not found")));
    // An unknown job kind is a clear deserialization error, not a panic.
    let err = call(
        &w,
        "estimate_job",
        json!({"kind": "nonsense", "params": {}}),
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("nonsense") || err.to_string().contains("variant"),
        "{err}"
    );
    // Missing kind.
    assert!(call(&w, "estimate_job", json!({"params": {}})).is_err());
}

#[test]
fn preflight_train_takes_the_ui_argument_shape_and_answers_in_camel_case() {
    let (_app, w) = app();
    let dir = tmp("preflight");
    let corpus = dir.join("c.txt");
    std::fs::write(&corpus, "the quick brown fox. ".repeat(500)).unwrap();
    let form = |vocab: u64, heads: u64| {
        json!({"params": {
            "contextLen": 16, "embedDim": 32, "numHeads": heads, "numBlocks": 2, "hiddenDim": 64,
            "epochs": 3, "batchSize": 16, "vocabSize": vocab, "threads": 0, "windowStride": 0,
            "maxTokens": 0, "valFraction": 0.0, "corpusPath": corpus.to_string_lossy()
        }})
    };
    let pf = call(&w, "preflight_train", form(512, 4)).unwrap();
    // 512x32 embeddings, 16x32 positions, two blocks, final norm and head: the engine's count.
    let want = ferrum_core::TransformerNet::check_size(512, 16, 32, 64, 2).unwrap();
    assert_eq!(pf["params"], want);
    assert!(pf["peakRamMb"].as_f64().unwrap() > 0.0);
    assert_eq!(pf["llamaCpp"]["runs"], true);
    assert_eq!(pf["problems"], json!([]));
    assert!(pf["ramAvailableMb"].as_f64().unwrap() > 0.0, "{pf}");
    // Character level: valid, but not portable.
    let chars = call(&w, "preflight_train", form(0, 4)).unwrap();
    assert_eq!(chars["llamaCpp"]["runs"], false);
    assert!(chars["params"].as_u64().is_some());
    // An invalid form lists the problem and gives no numbers.
    let bad = call(&w, "preflight_train", form(512, 3)).unwrap();
    assert!(
        bad["problems"][0].as_str().unwrap().contains("divisible"),
        "{bad}"
    );
    assert!(bad["params"].is_null());
    // The UI also sends it before the form is filled in.
    assert!(call(&w, "preflight_train", json!({"params": {}})).is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn clean_text_and_file_commands_round_trip_through_ipc() {
    let (app, w) = app();
    let progress = collect(&app, "clean-progress");
    let r = call(
        &w,
        "clean_text",
        json!({"raw": "Hello\u{a0}\u{201c}World\u{201d}   \n\n\n\nbye", "opts": {
            "stripGutenberg": false, "lowercase": true, "collapseWhitespace": true,
            "normalizePunctuation": true, "stripControlChars": true, "maxChars": null
        }}),
    )
    .unwrap();
    assert!(
        r["cleaned"]
            .as_str()
            .unwrap()
            .starts_with("hello \"world\""),
        "{r}"
    );
    assert_eq!(r["words"], 3);
    // The final event carries the totals (one non-empty line, then another after the gap).
    let events = progress.lock().unwrap().clone();
    let last: Value = serde_json::from_str(events.last().expect("a clean-progress event")).unwrap();
    assert_eq!(last, json!({"bytes": 28, "documents": 2}), "{events:?}");
    // Errors come back as `{ code, message, hint }`; the message is the text the UI shows.
    let e = call(
        &w,
        "clean_text",
        json!({"raw": "   ", "opts": {
        "stripGutenberg": false, "lowercase": false, "collapseWhitespace": false,
        "normalizePunctuation": false, "stripControlChars": false, "maxChars": null}}),
    )
    .unwrap_err();
    assert!(msg(&e).contains("nothing left"), "{e}");

    let dir = std::env::temp_dir().join(format!("ferrum_ipc_{}", std::process::id()));
    let path = dir.join("sub/c.txt").to_string_lossy().into_owned();
    assert_eq!(
        call(&w, "save_corpus", json!({"text": "abc", "path": path})).unwrap(),
        3
    );
    assert_eq!(
        call(
            &w,
            "read_text_file",
            json!({"path": path, "maxBytes": null})
        )
        .unwrap(),
        "abc"
    );
    // The path guard applies through IPC too.
    assert!(call(
        &w,
        "read_text_file",
        json!({"path": "/proc/cpuinfo", "maxBytes": null})
    )
    .is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn every_registered_command_is_reachable_and_unknown_ones_are_rejected() {
    let (_app, w) = app();
    // A command that does not exist.
    assert!(call(&w, "no_such_command", json!({})).is_err());
    // Arguments of the wrong shape are errors, not panics.
    assert!(call(&w, "clean_text", json!({"raw": 5})).is_err());
    assert!(call(&w, "model_info", json!({})).is_err());
}

pub(crate) fn tmp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_ipc_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Collect the payloads of one event for the rest of the test.
pub(crate) fn collect(app: &App<tauri::test::MockRuntime>, event: &str) -> Arc<Mutex<Vec<String>>> {
    let got = Arc::new(Mutex::new(Vec::new()));
    let sink = got.clone();
    app.listen(event, move |e| {
        sink.lock().unwrap().push(e.payload().to_string())
    });
    got
}

#[test]
fn tiny_model_train_generate_evaluate_inspect_end_to_end_with_events() {
    let (app, w) = app();
    let progress = collect(&app, "train-progress");
    let done = collect(&app, "train-done");
    let frags = collect(&app, "gen-fragment");
    let dir = tmp("model");
    let corpus = dir.join("c.txt");
    std::fs::write(
        &corpus,
        "the quick brown fox jumps over the lazy dog. ".repeat(60),
    )
    .unwrap();
    let model = dir.join("m.gguf").to_string_lossy().into_owned();

    // Exactly the parameter object app.js sends from the Train tab.
    let r = call(
        &w,
        "train_slm",
        json!({"params": {
            "method": "transformer", "corpusPath": corpus.to_string_lossy(), "modelPath": model,
            "contextLen": 8, "embedDim": 16, "numHeads": 2, "numBlocks": 1, "hiddenDim": 32,
            "epochs": 3, "lr": 0.01, "batchSize": 8, "vocabSize": 300, "seed": 1, "threads": 1,
            "verbose": false, "windowStride": 0, "maxTokens": 0, "weightDecay": 0, "dropout": 0,
            "gradClip": 0, "warmupSteps": 0, "cosineLr": false, "weightTying": false, "qat": true,
            "tokenizerPath": "", "checkpointPath": "", "checkpointEvery": 0, "resume": false,
            "valFraction": 0, "patience": 0
        }}),
    )
    .unwrap();
    assert!(
        r["bytes"].as_u64().unwrap() > 0 && r["modelPath"] == json!(model),
        "{r}"
    );
    assert_eq!(r["tokenizer"], "byte-level BPE");
    assert_eq!(
        progress.lock().unwrap().len(),
        3,
        "one train-progress per epoch"
    );
    assert_eq!(done.lock().unwrap().len(), 1);
    let p: Value = serde_json::from_str(&progress.lock().unwrap()[0]).unwrap();
    assert_eq!(
        (p["epoch"].as_u64(), p["total"].as_u64()),
        (Some(1), Some(3))
    );

    // Inspect, estimate from the real file, generate (streamed), evaluate.
    let info = call(&w, "model_info", json!({"path": model})).unwrap();
    // A tiny corpus supplies fewer merges than the 300 requested.
    assert!(
        info["tokenizer"]
            .as_str()
            .unwrap()
            .starts_with("byte-level BPE ("),
        "{info}"
    );
    let est = call(
        &w,
        "estimate_job",
        json!({"kind": "generate", "params": {"modelPath": model, "numChars": 100}}),
    )
    .unwrap();
    assert!(est["seconds"].as_f64().unwrap() > 0.0 && est["peakRamMb"].as_f64().unwrap() > 0.0);

    let out = call(
        &w,
        "generate_slm",
        json!({"params": {"modelPath": model, "seedText": "the quick", "numChars": 40, "temp": 0.8, "genSeed": 7, "stream": true, "verbose": false}}),
    )
    .unwrap();
    let out = out.as_str().unwrap();
    assert!(out.starts_with("the quick"));
    let streamed: String = frags
        .lock()
        .unwrap()
        .iter()
        .map(|p| serde_json::from_str::<String>(p).unwrap())
        .collect();
    assert_eq!(
        format!("the quick{streamed}"),
        out,
        "fragments reassemble to the returned text"
    );
    // Same gen seed, same text (determinism through the whole IPC path).
    let again = call(
        &w,
        "generate_slm",
        json!({"params": {"modelPath": model, "seedText": "the quick", "numChars": 40, "temp": 0.8, "genSeed": 7, "stream": false, "verbose": false}}),
    )
    .unwrap();
    assert_eq!(again.as_str().unwrap(), out);

    // G7: greedy (temp 0) with every sampling control, through the IPC path; the seed is irrelevant.
    let greedy = |seed: u64, stop: Value| {
        call(
            &w,
            "generate_slm",
            json!({"params": {"modelPath": model, "seedText": "the ", "numChars": 40, "temp": 0, "genSeed": seed,
                "stream": false, "verbose": false, "topK": 5, "topP": 0.9, "repPenalty": 1.1, "maxTokens": 12, "stop": stop}}),
        )
        .unwrap()
    };
    let g1 = greedy(1, json!([]));
    assert_eq!(g1, greedy(2, json!([])), "temp 0 ignores the seed");
    assert!(g1.as_str().unwrap().starts_with("the "));
    let bad = call(
        &w,
        "generate_slm",
        json!({"params": {"modelPath": model, "seedText": "x", "numChars": 4, "temp": -1, "genSeed": 1, "stream": false, "verbose": false}}),
    )
    .unwrap_err();
    assert!(msg(&bad).contains("temperature"), "{bad}");

    // cancel_job ends a streamed generation through the engine's cancel flag: the text so far
    // comes back (no unwinding), it is exactly the streamed fragments, and the slot is freed.
    let before = frags.lock().unwrap().len();
    let (w2, m2) = (w.clone(), model.clone());
    let long = std::thread::spawn(move || {
        call(
            &w2,
            "generate_slm",
            json!({"params": {"modelPath": m2, "seedText": "the ", "numChars": 1_000_000, "temp": 0.8,
                "genSeed": 3, "stream": true, "verbose": false}}),
        )
    });
    wait_for(&frags, before + 1);
    assert_eq!(
        call(&w, "cancel_job", json!({"kind": "generate"})).unwrap(),
        true
    );
    let part = long.join().unwrap().unwrap();
    let part = part.as_str().unwrap();
    assert!(part.starts_with("the ") && part.chars().count() < 1_000_004);
    let streamed: String = frags.lock().unwrap()[before..]
        .iter()
        .map(|p| serde_json::from_str::<String>(p).unwrap())
        .collect();
    assert_eq!(format!("the {streamed}"), part);
    assert_eq!(
        call(&w, "cancel_job", json!({"kind": "generate"})).unwrap(),
        false
    );

    let ev = call(&w, "evaluate_slm", json!({"params": {"modelPath": model, "text": "the lazy dog. the quick brown fox.", "textPath": null}})).unwrap();
    let ev = &ev.as_array().unwrap()[0];
    assert!(ev["perplexity"].as_f64().unwrap() > 0.0 && ev["numPredictions"].as_u64().unwrap() > 0);
    // G6: the row carries the CLI's numbers (bits per byte, bytes, seconds) and `eval --json`'s keys,
    // and a test file adds a second row.
    assert!(ev["bitsPerByte"].as_f64().unwrap() > 0.0 && ev["bytes"].as_u64().unwrap() > 0);
    assert_eq!(ev["label"], "held-out");
    let report = ev["report"].as_object().unwrap();
    for k in [
        "perplexity",
        "cross_entropy_nats",
        "bits_per_token",
        "bits_per_byte",
        "tokens",
        "bytes",
        "seconds",
    ] {
        assert!(report.contains_key(k), "{k}");
    }
    let test_file = dir.join("test.txt");
    std::fs::write(&test_file, "the quick brown fox. the lazy dog.").unwrap();
    let pair = call(
        &w,
        "evaluate_slm",
        json!({"params": {"modelPath": model, "text": "the lazy dog.", "textPath": null,
        "testPath": test_file.to_string_lossy(), "lossy": true}}),
    )
    .unwrap();
    let labels: Vec<_> = pair
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["label"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(labels, ["held-out", "test"]);
    let _ = std::fs::remove_dir_all(dir);
}

/// Block until `got` holds `n` payloads (a job that never reports fails the test, not hangs it).
pub(crate) fn wait_for(got: &Arc<Mutex<Vec<String>>>, n: usize) {
    let t0 = std::time::Instant::now();
    while got.lock().unwrap().len() < n {
        assert!(
            t0.elapsed().as_secs() < 120,
            "no event arrived within 120 s"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn no_temp_files(dir: &std::path::Path) {
    for e in std::fs::read_dir(dir).unwrap() {
        let name = e.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !name.ends_with(".tmp") && !name.ends_with(".part"),
            "{name}"
        );
    }
}

#[test]
fn a_running_training_job_is_cancelled_saves_a_loadable_model_and_refuses_a_second_start() {
    let (app, w) = app();
    let progress = collect(&app, "train-progress");
    let done = collect(&app, "train-done");
    let dir = tmp("cancel_train");
    let corpus = dir.join("c.txt");
    std::fs::write(
        &corpus,
        "the quick brown fox jumps over the lazy dog. ".repeat(60),
    )
    .unwrap();
    let model = dir.join("m.gguf").to_string_lossy().into_owned();
    let ckpt = dir.join("run.fckp").to_string_lossy().into_owned();
    // Far more epochs than the test will ever wait for: only a cancel ends this run.
    let params = json!({"params": {
        "method": "transformer", "corpusPath": corpus.to_string_lossy(), "modelPath": model,
        "contextLen": 8, "embedDim": 16, "numHeads": 2, "numBlocks": 1, "hiddenDim": 32,
        "epochs": 1_000_000, "lr": 0.01, "batchSize": 8, "vocabSize": 300, "seed": 1, "threads": 1,
        "verbose": false, "checkpointPath": ckpt
    }});
    // Nothing is running yet.
    assert_eq!(
        call(&w, "cancel_job", json!({"kind": "train"})).unwrap(),
        false
    );

    let (w2, p2) = (w.clone(), params.clone());
    let job = std::thread::spawn(move || call(&w2, "train_slm", p2));
    wait_for(&progress, 1);

    // A second training job is refused while the first runs.
    let e = call(&w, "train_slm", params.clone()).unwrap_err();
    assert_eq!(msg(&e), "a training job is already running", "{e}");
    // Other kinds have their own slot, and an unknown kind is a clear error.
    assert!(call(&w, "cancel_job", json!({"kind": "nonsense"})).is_err());
    assert_eq!(
        call(&w, "cancel_job", json!({"kind": "export"})).unwrap(),
        false
    );

    assert_eq!(
        call(&w, "cancel_job", json!({"kind": "train"})).unwrap(),
        true
    );
    let r = job.join().unwrap().unwrap();
    assert_eq!(r["stopped"], true, "{r}");
    assert!(r["stoppedAtStep"].as_u64().unwrap() >= 1, "{r}");
    assert!(r["bytes"].as_u64().unwrap() > 0);
    // The Done event carries the same stopped result.
    wait_for(&done, 1);
    let d: Value = serde_json::from_str(&done.lock().unwrap()[0]).unwrap();
    assert_eq!(
        (d["stopped"].clone(), d["modelPath"].clone()),
        (json!(true), json!(model))
    );
    // The model is complete and loads; the checkpoint was written for a resume; no temp files.
    assert!(ferrum_core::GenerativeSLM::load(&model).is_ok());
    let info = call(&w, "model_info", json!({"path": model})).unwrap();
    assert!(info["bytes"].as_u64().unwrap() > 0);
    assert!(std::fs::metadata(&ckpt).unwrap().len() > 0);
    no_temp_files(&dir);

    // The slot is free again: cancel finds nothing and a fresh job can start.
    assert_eq!(
        call(&w, "cancel_job", json!({"kind": "train"})).unwrap(),
        false
    );
    let again = {
        let mut q = params.clone();
        q["params"]["epochs"] = json!(1);
        call(&w, "train_slm", q).unwrap()
    };
    assert_eq!(again["stopped"], false);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_running_finetune_is_cancelled_and_its_checkpoint_resumes() {
    let (app, w) = app();
    let progress = collect(&app, "finetune-progress");
    let dir = tmp("cancel_ft");
    let src = dir.join("base.gguf").to_string_lossy().into_owned();
    std::fs::write(&src, crate::test_support::tiny_llama_gguf(8, 2, 2, 16, 32)).unwrap();
    let corpus = dir.join("c.txt");
    std::fs::write(
        &corpus,
        "hi abc hi abc hi abc hi abc hi abc hi abc ".repeat(4),
    )
    .unwrap();
    let ckpt = dir.join("tuned.flck").to_string_lossy().into_owned();
    let params = json!({"params": {"modelPath": src, "corpusPath": corpus.to_string_lossy(), "outPath": ckpt,
        "epochs": 1_000_000, "lr": 0.001, "batchSize": 2, "seqLen": 8, "warmup": 0, "clip": 1.0, "weightDecay": 0,
        "dropout": 0, "qat": false, "seed": 1, "threads": 1, "resume": null, "force": false, "verbose": false}});
    let (w2, p2) = (w.clone(), params.clone());
    let job = std::thread::spawn(move || call(&w2, "finetune_gguf", p2));
    wait_for(&progress, 1);
    let e = call(&w, "finetune_gguf", params.clone()).unwrap_err();
    assert_eq!(msg(&e), "a fine-tuning job is already running");
    assert_eq!(
        call(&w, "cancel_job", json!({"kind": "finetune"})).unwrap(),
        true
    );
    let r = job.join().unwrap().unwrap();
    assert_eq!(r["stopped"], true, "{r}");
    assert!(r["epochs"].as_u64().unwrap() >= 1 && r["steps"].as_u64().unwrap() >= 1);
    assert!(std::fs::metadata(&ckpt).unwrap().len() > 0);
    no_temp_files(&dir);
    // The stopped checkpoint overlays on the base model (a resume) and runs.
    let run = call(
        &w,
        "run_gguf",
        json!({"params": {"modelPath": src, "prompt": "hi", "quant": "f32", "maxNew": 3, "temp": 0, "genSeed": 1,
            "ids": null, "force": false, "resume": ckpt, "raw": true}}),
    )
    .unwrap();
    assert!(run["text"].as_str().unwrap().starts_with("hi"));
    assert_eq!(run["stopped"], false);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn command_errors_reach_the_ui_as_code_message_and_hint() {
    let (_app, w) = app();
    let dir = tmp("trainerr");
    let corpus = dir.join("c.txt");
    std::fs::write(&corpus, "abc ".repeat(50)).unwrap();
    let base = |heads: u64| {
        json!({"params": {
        "method": "transformer", "corpusPath": corpus.to_string_lossy(), "modelPath": dir.join("m.gguf").to_string_lossy(),
        "contextLen": 8, "embedDim": 16, "numHeads": heads, "numBlocks": 1, "hiddenDim": 32, "epochs": 1,
        "lr": 0.01, "batchSize": 4, "vocabSize": 0, "seed": 1, "threads": 1, "verbose": false}})
    };
    // The whole object: exactly these three keys, strings, and a hint that says what to do.
    let e = call(&w, "train_slm", base(3)).unwrap_err();
    assert!(msg(&e).contains("divisible"), "{e}");
    assert_eq!(e["code"], "invalid");
    assert_eq!(e.as_object().unwrap().len(), 3, "{e}");
    assert!(e["hint"].as_str().unwrap().len() > 20, "{e}");
    let mut missing = base(2);
    missing["params"]["corpusPath"] = json!(dir.join("nope.txt").to_string_lossy());
    let e = call(&w, "train_slm", missing).unwrap_err();
    assert!(msg(&e).contains("does not exist"));
    assert_eq!(e["code"], "not_found");
    // A device path is a path error; a command that never produced a message of its own still
    // gets a code from what it says.
    let e = call(&w, "read_text_file", json!({"path": "/dev/zero"})).unwrap_err();
    assert_eq!(e["code"], "path", "{e}");
    let e = call(
        &w,
        "model_info",
        json!({"path": dir.join("c.txt").to_string_lossy()}),
    )
    .unwrap_err();
    assert_eq!(e["code"], "model", "{e}");
    let e = call(&w, "download_text", json!({"url": "ftp://x/y"})).unwrap_err();
    assert_eq!(e["code"], "invalid", "{e}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn imported_gguf_inspect_run_finetune_export_through_ipc() {
    let (app, w) = app();
    let ft_progress = collect(&app, "finetune-progress");
    let export_progress = collect(&app, "export-progress");
    let dir = tmp("llama");
    let src = dir.join("base.gguf").to_string_lossy().into_owned();
    std::fs::write(&src, crate::test_support::tiny_llama_gguf(8, 2, 2, 16, 32)).unwrap();

    let info = call(&w, "gguf_info", json!({"path": src})).unwrap();
    assert_eq!(
        (info["architecture"].as_str(), info["runnable"].as_bool()),
        (Some("llama"), Some(true))
    );
    assert!(info["paramCount"].as_u64().unwrap() > 0);

    let run = call(
        &w,
        "run_gguf",
        json!({"params": {"modelPath": src, "prompt": "hi", "quant": "int8", "maxNew": 4, "temp": 0.8,
            "genSeed": 3, "ids": null, "force": false, "resume": null, "topK": 0, "topP": 1.0, "repPenalty": 1.0, "raw": true}}),
    )
    .unwrap();
    assert!(run["text"].as_str().unwrap().starts_with("hi"));
    assert!(run["generated"].as_u64().unwrap() <= 4);
    // Greedy (temp 0) is accepted and ignores the seed.
    let greedy = |seed: u64| {
        call(
            &w,
            "run_gguf",
            json!({"params": {"modelPath": src, "prompt": "hi", "quant": "int8", "maxNew": 6, "temp": 0,
                "genSeed": seed, "ids": null, "force": false, "resume": null, "topK": 0, "topP": 1.0, "repPenalty": 1.0, "raw": true}}),
        )
        .unwrap()["text"]
            .clone()
    };
    assert_eq!(greedy(1), greedy(99));

    let ckpt = dir.join("tuned.flck").to_string_lossy().into_owned();
    let corpus = dir.join("c.txt");
    std::fs::write(
        &corpus,
        "hi abc hi abc hi abc hi abc hi abc hi abc hi abc hi abc ",
    )
    .unwrap();
    let ft = call(
        &w,
        "finetune_gguf",
        json!({"params": {"modelPath": src, "corpusPath": corpus.to_string_lossy(), "outPath": ckpt, "epochs": 2,
            "lr": 0.001, "batchSize": 2, "seqLen": 8, "warmup": 0, "clip": 1.0, "weightDecay": 0, "dropout": 0,
            "qat": false, "seed": 1, "threads": 1, "resume": null, "force": false, "verbose": false}}),
    )
    .unwrap();
    assert_eq!(ft["epochs"], 2);
    assert!(std::fs::metadata(&ckpt).unwrap().len() > 0);
    assert_eq!(ft_progress.lock().unwrap().len(), 2);

    // The fine-tune estimate is computed from this real header and checkpoint.
    let est = call(&w, "estimate_job", json!({"kind": "gguf_finetune", "params": {
        "modelPath": src, "corpusPath": corpus.to_string_lossy(), "epochs": 2, "batchSize": 2, "seqLen": 8, "threads": 1}})).unwrap();
    assert!(est["seconds"].as_f64().is_some());

    let out = dir.join("out.gguf").to_string_lossy().into_owned();
    let ex = call(
        &w,
        "export_gguf",
        json!({"params": {"modelPath": src, "outPath": out, "quant": "f32", "resume": ckpt, "force": false, "verbose": false}}),
    )
    .unwrap();
    assert!(
        ex["bytes"].as_u64().unwrap() > 0 && ex["tensorSummary"].as_str().unwrap().contains("F32")
    );
    // G6: the Evaluate tab scores the imported model, with and without the fine-tune checkpoint.
    let held = dir.join("held.txt");
    std::fs::write(&held, "hi abc hi abc hi abc hi abc hi abc hi abc").unwrap();
    let eval_gguf = |resume: Value| {
        call(
            &w,
            "evaluate_slm",
            json!({"params": {"modelPath": src, "text": null, "textPath": held.to_string_lossy(),
                "quant": "f32", "ctx": 0, "resume": resume, "force": false}}),
        )
        .unwrap()[0]
            .clone()
    };
    let base = eval_gguf(json!(null));
    let tuned = eval_gguf(json!(ckpt));
    assert_eq!(base["uniformBaseline"], 32.0);
    assert!(base["perplexity"].as_f64().unwrap().is_finite());
    assert_ne!(
        base["crossEntropy"], tuned["crossEntropy"],
        "the checkpoint changes the weights"
    );
    let phases = export_progress.lock().unwrap().join("|");
    assert!(phases.contains("applying"), "{phases}");
    let _ = std::fs::remove_dir_all(dir);
}

/// The arguments app.js really sends (captured by ui/tests/app_flows.test.js) must
/// deserialize into the Rust parameter types; a renamed or mistyped required
/// field fails here rather than in a window.
#[test]
fn arguments_captured_from_app_js_deserialize_into_the_command_types() {
    use crate::commands::{
        CleanOpts, ExportParams, FinetuneParams, GenParams, GgufRunParams, TrainParams,
    };
    use crate::data::prepare::PrepareParams;
    use crate::data::tokenize::TokenizeParams;
    use crate::evaluate::EvalParams;
    use crate::serving::runtime_export::{RuntimeParams, Target};
    let fixture: Value =
        serde_json::from_str(include_str!("../ui/tests/command_params.fixture.json")).unwrap();
    macro_rules! parses {
        ($key:literal, $ty:ty) => {
            serde_json::from_value::<$ty>(fixture[$key].clone()).unwrap()
        };
    }
    let t = parses!("train_slm", TrainParams);
    assert_eq!(
        (t.context_len, t.embed_dim, t.max_tokens, t.cosine_lr),
        (16, 32, 50_000_000, true)
    );
    assert!(!t.force, "the RAM override is off unless ticked");
    assert_eq!(parses!("generate_slm", GenParams).gen_seed, Some(7));
    let e = parses!("evaluate_slm", EvalParams);
    assert_eq!(e.text.as_deref(), Some("held out"));
    assert_eq!(
        (e.quant.as_str(), e.ctx, e.lossy, e.force),
        ("f32", 0, false, false)
    );
    let g = parses!("run_gguf", GgufRunParams);
    assert_eq!((g.quant.as_str(), g.top_k, g.raw), ("int4", 40, true));
    assert_eq!(parses!("finetune_gguf", FinetuneParams).epochs, 3);
    assert_eq!(parses!("export_gguf", ExportParams).quant, "q8_0");
    let p = parses!("prepare_dataset", PrepareParams);
    assert_eq!(
        (
            p.inputs.len(),
            p.seed,
            p.min_chars,
            p.max_chars,
            p.dedup,
            p.lossy
        ),
        (2, 7, 50, 20_000, true, true)
    );
    assert_eq!(p.doc_sep, ferrum_core::DOC_SEP);
    let sv = parses!("serve_start", crate::serving::serve::ServeParams);
    assert_eq!(
        (sv.host.as_str(), sv.port, sv.max_batch, sv.max_tokens),
        ("127.0.0.1", 0, 64, 4096)
    );
    assert_eq!(sv.token.as_deref().map(str::len), Some(48));
    let k = parses!("tokenize_corpus", TokenizeParams);
    assert_eq!(
        (k.input.as_str(), k.doc_sep.as_str()),
        ("/d/corpus", ferrum_core::DOC_SEP)
    );
    let r = parses!("export_to_runtime", RuntimeParams);
    assert_eq!(
        (r.target, r.name.as_str()),
        (Target::Ollama, "ferrum-model")
    );
    assert!(r.modelfile_out.is_empty() && r.repo.is_empty());
    assert_eq!(
        parses!("preview_runtime_export", RuntimeParams).target,
        Target::Ollama
    );
    assert!(
        serde_json::from_value::<CleanOpts>(fixture["clean_text"]["opts"].clone())
            .unwrap()
            .lowercase
    );

    // clean_text runs end to end with exactly those arguments.
    let (_app, w) = app();
    let r = call(&w, "clean_text", fixture["clean_text"].clone()).unwrap();
    assert_eq!(r["cleaned"].as_str().unwrap().trim_end(), "hello  there"); // lowercased; inner spaces are kept
}

#[test]
fn prepare_dataset_runs_through_ipc_streams_progress_and_shares_one_job_slot() {
    use crate::infra::jobs::Job;
    use tauri::Manager;
    let (app, w) = app();
    let progress = collect(&app, "prepare-progress");
    let dir = tmp("prepare");
    let docs: Vec<String> = (0..40)
        .map(|i| format!("Story number {i}: once upon a time there was a small fox."))
        .collect();
    let a = dir.join("a.txt");
    std::fs::write(&a, docs.join(ferrum_core::DOC_SEP)).unwrap();
    let out = dir.join("out");
    let body = |extra: Value| {
        let mut p = json!({"inputs": [a.to_string_lossy()], "outDir": out.to_string_lossy(),
            "valFrac": 0.1, "testFrac": 0.1, "seed": 1, "minChars": 5, "maxChars": 0,
            "docSep": ferrum_core::DOC_SEP, "dedup": true, "nfkc": false, "lowercase": false,
            "stripGutenberg": false, "lossy": false});
        for (k, v) in extra.as_object().unwrap() {
            p[k] = v.clone();
        }
        json!({"params": p})
    };

    // A job in the same kind's slot refuses a second start; cancel reaches it.
    {
        let state = app.state::<crate::AppState>();
        let held = state.jobs.start(Job::Prepare).unwrap();
        let e = call(&w, "prepare_dataset", body(json!({}))).unwrap_err();
        assert_eq!(msg(&e), "a dataset preparation is already running");
        assert_eq!(
            call(&w, "cancel_job", json!({"kind": "prepare"})).unwrap(),
            true
        );
        assert!(held.cancelled());
    }
    assert_eq!(
        call(&w, "cancel_job", json!({"kind": "prepare"})).unwrap(),
        false
    );
    assert!(!out.exists(), "a refused start writes nothing");

    // The run: camelCase result, files on disk, progress events with the documented keys.
    let r = call(&w, "prepare_dataset", body(json!({}))).unwrap();
    let docs_in = |name: &str| r[name]["documents"].as_u64().unwrap();
    assert_eq!(docs_in("train") + docs_in("val") + docs_in("test"), 40);
    assert_eq!(r["files"], 1);
    assert_eq!(r["rejected"]["duplicate"], 0);
    assert_eq!(r["lossyRepaired"], 0);
    assert_eq!(
        r["manifestPath"],
        out.join("manifest.json").to_string_lossy().as_ref()
    );
    assert!(r["summary"].as_str().unwrap().starts_with("prepared "));
    let train = std::fs::read_to_string(out.join("train.txt")).unwrap();
    assert_eq!(r["train"]["bytes"].as_u64().unwrap(), train.len() as u64);
    // The manifest records the equivalent command line.
    let m: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["args"][0], "--input");
    assert!(m["args"].as_array().unwrap().iter().any(|x| x == "--dedup"));
    wait_for(&progress, 2); // the start of the file and the closing report
    let first: Value = serde_json::from_str(&progress.lock().unwrap()[0]).unwrap();
    for k in [
        "file",
        "files",
        "path",
        "bytesDone",
        "bytesTotal",
        "documents",
    ] {
        assert!(first.get(k).is_some(), "progress lacks {k}: {first}");
    }

    // Errors reach the UI as plain strings in the GUI's words; a bad path never runs.
    let bad = dir.join("bad.txt");
    std::fs::write(&bad, b"text \xff more").unwrap();
    let e = call(
        &w,
        "prepare_dataset",
        body(json!({"inputs": [bad.to_string_lossy()]})),
    )
    .unwrap_err();
    assert!(msg(&e).contains("invalid UTF-8 at byte 5"), "{e}");
    assert!(msg(&e).contains(&*bad.to_string_lossy()), "{e}");
    let e = call(
        &w,
        "prepare_dataset",
        body(json!({"inputs": ["/proc/cpuinfo"]})),
    )
    .unwrap_err();
    assert!(msg(&e).contains("not allowed"), "{e}");
    let e = call(
        &w,
        "prepare_dataset",
        body(json!({"outDir": "/proc/prepared"})),
    )
    .unwrap_err();
    assert!(msg(&e).contains("not allowed"), "{e}");
    assert!(
        call(&w, "prepare_dataset", json!({"params": {"outDir": "x"}})).is_err(),
        "inputs are required"
    );
    // The estimate takes the same list and reads its size from disk.
    let est = call(
        &w,
        "estimate_job",
        json!({"kind": "prepare", "params": {
        "inputPaths": [a.to_string_lossy()], "outDir": out.to_string_lossy()}}),
    )
    .unwrap();
    assert!(est["seconds"].as_f64().unwrap() > 0.0 && est["peakRamMb"].as_f64().is_some());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn runtime_export_runs_through_ipc_with_fake_tools_and_shares_one_job_slot() {
    use crate::infra::jobs::Job;
    use std::os::unix::fs::PermissionsExt;
    use tauri::Manager;
    let (app, w) = app();
    let dir = tmp("runtime");
    let mut b = ferrum_core::GgufBuilder::new();
    b.meta(
        "general.architecture",
        ferrum_core::MetaValue::String("gpt2".into()),
    );
    b.meta("gpt2.context_length", ferrum_core::MetaValue::U32(32));
    let model = dir.join("m.gguf");
    std::fs::write(&model, b.into_bytes()).unwrap();
    let script = |name: &str, body: &str| {
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p.to_string_lossy().into_owned()
    };
    let ran = dir.join("ran");
    let ollama = script(
        "ollama",
        &format!("echo \"$@\" > {}; echo made", ran.display()),
    );
    let form = |target: &str, extra: Value| {
        let mut p = json!({"modelPath": model.to_string_lossy(), "target": target, "name": "ferrum-x",
            "repo": "", "bin": ollama, "modelfileOut": ""});
        for (k, v) in extra.as_object().unwrap() {
            p[k] = v.clone();
        }
        json!({"params": p})
    };

    // Preview: the exact Modelfile and command, nothing started.
    let pv = call(&w, "preview_runtime_export", form("ollama", json!({}))).unwrap();
    assert_eq!(pv["ran"], false);
    assert!(pv["modelfile"]
        .as_str()
        .unwrap()
        .contains("PARAMETER num_ctx 32"));
    assert!(pv["commands"][0]
        .as_str()
        .unwrap()
        .contains("create ferrum-x -f"));
    assert_eq!(pv["tool"], json!(ollama));
    assert!(!ran.exists());

    // A held slot refuses a second start; cancel_job knows the kind.
    {
        let state = app.state::<crate::AppState>();
        let held = state.jobs.start(Job::RuntimeExport).unwrap();
        let e = call(&w, "export_to_runtime", form("ollama", json!({}))).unwrap_err();
        assert_eq!(msg(&e), "an Ollama / LM Studio export is already running");
        assert_eq!(
            call(&w, "cancel_job", json!({"kind": "runtime_export"})).unwrap(),
            true
        );
        assert!(held.cancelled());
    }
    assert!(!ran.exists());

    // The run: camelCase result, the fake tool was started with the shown command.
    let r = call(&w, "export_to_runtime", form("ollama", json!({}))).unwrap();
    assert_eq!(
        (r["ran"].clone(), r["target"].clone()),
        (json!(true), json!("ollama"))
    );
    assert_eq!(r["output"].as_str().unwrap().trim(), "made");
    assert!(std::fs::read_to_string(&ran)
        .unwrap()
        .starts_with("create ferrum-x -f "));
    assert_eq!(r["modelfile"], pv["modelfile"]);
    // Write the Modelfile only.
    let out = dir.join("o.Modelfile");
    std::fs::remove_file(&ran).unwrap();
    let r = call(
        &w,
        "export_to_runtime",
        form("ollama", json!({"modelfileOut": out.to_string_lossy()})),
    )
    .unwrap();
    assert_eq!(r["ran"], false);
    assert_eq!(
        std::fs::read_to_string(&out).unwrap(),
        r["modelfile"].as_str().unwrap()
    );
    assert!(!ran.exists());
    // LM Studio, and the errors a user would meet.
    let lms = script("lms", "case \"$1\" in ls) echo '[{\"modelKey\":\"ferrum/m\",\"path\":\"ferrum/m/m.gguf\"}]';; esac");
    let r = call(
        &w,
        "export_to_runtime",
        form("lmstudio", json!({"bin": lms})),
    )
    .unwrap();
    assert_eq!(r["key"], "ferrum/m");
    let e = call(
        &w,
        "export_to_runtime",
        form("ollama", json!({"name": "-rf"})),
    )
    .unwrap_err();
    assert!(msg(&e).contains("not a usable model name"), "{e}");
    let e = call(&w, "preview_runtime_export", form("other", json!({}))).unwrap_err();
    assert!(
        e.to_string().contains("other") || e.to_string().contains("variant"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The same tiny model from text and from a token file made by the GUI's Tokenize action:
/// byte-identical model files (so the same sha256), the Train tab accepting the token file
/// as its corpus, and the pre-flight and estimates describing it.
#[test]
fn training_from_a_token_file_equals_training_from_text_and_tokenizing_runs_through_ipc() {
    use crate::infra::jobs::Job;
    use tauri::Manager;
    let (app, w) = app();
    let progress = collect(&app, "tokenize-progress");
    let dir = tmp("tokens");
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
            let body: Vec<&str> = (0..8 + next(7)).map(|_| words[next(words.len())]).collect();
            format!("{}.", body.join(" "))
        })
        .collect();
    let corpus = dir.join("c.txt");
    std::fs::write(&corpus, docs.join(ferrum_core::DOC_SEP)).unwrap();
    let s = |n: &str| dir.join(n).to_string_lossy().into_owned();
    let train = |input: &str, model: &str, extra: Value| {
        let mut p = json!({"method": "transformer", "corpusPath": input, "modelPath": model,
            "contextLen": 16, "embedDim": 16, "numHeads": 2, "numBlocks": 2, "hiddenDim": 32,
            "epochs": 2, "lr": 0.01, "batchSize": 8, "vocabSize": 300, "seed": 1337, "threads": 1,
            "verbose": false, "tokenizerPath": s("tok.gguf")});
        for (k, v) in extra.as_object().unwrap() {
            p[k] = v.clone();
        }
        call(&w, "train_slm", json!({"params": p}))
    };

    // A first run creates the tokenizer; then text and tokens train with it reused.
    train(&s("c.txt"), &s("seed.gguf"), json!({"epochs": 1})).unwrap();
    let from_text = train(&s("c.txt"), &s("text.gguf"), json!({})).unwrap();
    let tk = call(
        &w,
        "tokenize_corpus",
        json!({"params": {"input": s("c.txt"),
        "tokenizerPath": s("tok.gguf"), "outPath": s("c.tok.gguf")}}),
    )
    .unwrap();
    assert!(tk["tokens"].as_u64().unwrap() > 100 && tk["documents"] == 60 && tk["files"] == 1);
    assert_eq!(tk["vocabSize"], from_text["outputDim"]);
    assert!(tk["summary"].as_str().unwrap().starts_with("tokenized "));
    wait_for(&progress, 1);
    let first: Value = serde_json::from_str(&progress.lock().unwrap()[0]).unwrap();
    for k in ["file", "files", "path", "bytesDone", "bytesTotal", "tokens"] {
        assert!(first.get(k).is_some(), "progress lacks {k}: {first}");
    }
    let from_tokens = train(&s("c.tok.gguf"), &s("tokens.gguf"), json!({})).unwrap();
    let (a, b) = (
        std::fs::read(dir.join("text.gguf")).unwrap(),
        std::fs::read(dir.join("tokens.gguf")).unwrap(),
    );
    assert_eq!(
        slm_cli::hash::sha256_hex(&a),
        slm_cli::hash::sha256_hex(&b),
        "a token file trains the same model as the text it came from"
    );
    assert_eq!(a, b);
    assert_eq!(from_tokens["tokenizer"], "byte-level BPE");
    assert_eq!(from_tokens["finalLoss"], from_text["finalLoss"]);
    // Validation works from a token file too (the tail of the tokens is held out).
    let v = train(
        &s("c.tok.gguf"),
        &s("val.gguf"),
        json!({"valFraction": 0.2, "patience": 1}),
    )
    .unwrap();
    assert!(v["bytes"].as_u64().unwrap() > 0);

    // The errors a user meets, in the tab's words.
    let e = train(&s("c.tok.gguf"), &s("x.gguf"), json!({"tokenizerPath": ""})).unwrap_err();
    assert!(msg(&e).contains("needs the Tokenizer file field"), "{e}");
    let e = train(&s("c.tok.gguf"), &s("x.gguf"), json!({"vocabSize": 0})).unwrap_err();
    assert!(msg(&e).contains("Vocab 0 (character-level)"), "{e}");
    // Another tokenizer than the one the file was made with.
    let other =
        ferrum_core::ByteBpeTokenizer::train("something else entirely, with other words", 280)
            .unwrap();
    std::fs::write(dir.join("other.gguf"), other.to_gguf_bytes()).unwrap();
    let e = train(
        &s("c.tok.gguf"),
        &s("x.gguf"),
        json!({"tokenizerPath": s("other.gguf")}),
    )
    .unwrap_err();
    assert!(msg(&e).contains("is not the tokenizer"), "{e}");
    assert!(!dir.join("x.gguf").exists());

    // The pre-flight: the token file's own vocabulary, no tokenizing time, RAM that does not scale.
    let form = |corpus: &str, tokenizer: &str, vocab: u64| {
        json!({"params": {
        "contextLen": 16, "embedDim": 16, "numHeads": 2, "numBlocks": 2, "hiddenDim": 32,
        "epochs": 2, "batchSize": 8, "vocabSize": vocab, "threads": 1, "corpusPath": corpus,
        "tokenizerPath": tokenizer}})
    };
    let pf = call(
        &w,
        "preflight_train",
        form(&s("c.tok.gguf"), &s("tok.gguf"), 512),
    )
    .unwrap();
    assert_eq!(pf["corpus"]["tokens"], tk["tokens"]);
    assert_eq!(pf["corpus"]["documents"], 60);
    assert_eq!(pf["corpus"]["ramScalesWithCorpus"], false);
    assert_eq!(pf["problems"], json!([]));
    let net = ferrum_core::TransformerNet::check_size(
        tk["vocabSize"].as_u64().unwrap() as usize,
        16,
        16,
        32,
        2,
    )
    .unwrap();
    assert_eq!(
        pf["params"], net,
        "the net uses the token file's vocabulary, not the Vocab field"
    );
    assert_eq!(pf["llamaCpp"]["runs"], true);
    let bad = call(&w, "preflight_train", form(&s("c.tok.gguf"), "", 0)).unwrap();
    assert_eq!(bad["problems"].as_array().unwrap().len(), 2, "{bad}");
    assert!(call(&w, "preflight_train", form(&s("c.txt"), "", 512)).unwrap()["corpus"].is_null());
    // Estimates: Train from tokens has no tokenizing time; Tokenize reads the sizes from disk.
    let est = call(
        &w,
        "estimate_job",
        json!({"kind": "train", "params": form(&s("c.tok.gguf"), &s("tok.gguf"), 512)["params"]}),
    )
    .unwrap();
    assert!(est["notes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|n| n.as_str().unwrap().contains("RAM does not grow")));
    let est = call(
        &w,
        "estimate_job",
        json!({"kind": "tokenize", "params": {
        "inputPaths": [s("c.txt")], "outDir": s("c.tok.gguf")}}),
    )
    .unwrap();
    assert!(est["seconds"].as_f64().unwrap() > 0.0 && est["peakRamMb"].as_f64().is_some());

    // One tokenize job at a time; cancel_job knows the kind.
    let st = app.state::<crate::AppState>();
    let held = st.jobs.start(Job::Tokenize).unwrap();
    let e = call(
        &w,
        "tokenize_corpus",
        json!({"params": {"input": s("c.txt"),
        "tokenizerPath": s("tok.gguf"), "outPath": s("again.tok.gguf")}}),
    )
    .unwrap_err();
    assert_eq!(msg(&e), "a tokenizing job is already running");
    assert_eq!(
        call(&w, "cancel_job", json!({"kind": "tokenize"})).unwrap(),
        true
    );
    assert!(held.cancelled() && !dir.join("again.tok.gguf").exists());
    let _ = std::fs::remove_dir_all(dir);
}

/// The command names registered in `generate_handler![...]` (the last path segment of each).
pub(crate) fn handler_commands(lib_rs: &str) -> Vec<String> {
    let list = lib_rs
        .split("generate_handler![")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .expect("a generate_handler![...] list");
    list.split(',')
        .filter_map(|entry| entry.trim().rsplit("::").next())
        .filter(|name| !name.is_empty())
        .map(String::from)
        .collect()
}

/// The commands whose name never appears as a quoted string in `tests` (how every IPC call
/// names its command).
fn commands_without_a_test<'a>(commands: &'a [String], tests: &str) -> Vec<&'a str> {
    commands
        .iter()
        .map(String::as_str)
        .filter(|c| !tests.contains(&format!("\"{c}\"")))
        .collect()
}

/// The command matrix: a command added to the registry without an IPC test fails here.
#[test]
fn every_command_has_an_ipc_test() {
    let commands = handler_commands(include_str!("lib.rs"));
    assert!(commands.len() > 20, "parsed {commands:?}");
    let tests = [include_str!("ipc_tests.rs"), include_str!("ipc_errors.rs")].concat();
    assert_eq!(
        commands_without_a_test(&commands, &tests),
        Vec::<&str>::new()
    );
}

/// The guard itself: it reads the real list shape and names exactly the commands left out.
#[test]
fn the_command_matrix_guard_finds_a_missing_test() {
    let lib = "builder.invoke_handler(tauri::generate_handler![\n  commands::alpha_cmd,\n  \
               beta::beta_cmd,\n  gamma_cmd,\n ])\n .other([x])";
    let commands = handler_commands(lib);
    assert_eq!(commands, ["alpha_cmd", "beta_cmd", "gamma_cmd"]);
    let tests = r#"call(&w, "alpha_cmd", json!({})); call(&w, "gamma_cmd_two", json!({}));"#;
    assert_eq!(
        commands_without_a_test(&commands, tests),
        ["beta_cmd", "gamma_cmd"]
    );
    assert!(commands_without_a_test(&commands, r#""alpha_cmd" "beta_cmd" "gamma_cmd""#).is_empty());
}

#[test]
fn download_text_returns_the_text_and_sends_progress_with_the_url_as_id() {
    let (app, w) = app();
    let got = collect(&app, "download-progress");
    let (url, _rx) = crate::test_support::http_server("200 OK", b"text".to_vec(), 1);
    // No maxBytes: the 8 MiB default applies.
    let text = call(&w, "download_text", json!({"url": url})).unwrap();
    assert_eq!(text, json!("text"));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !got
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.contains("\"total\":4"))
        && std::time::Instant::now() < deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let last: Value = serde_json::from_str(got.lock().unwrap().last().unwrap()).unwrap();
    assert_eq!(last, json!({"id": url, "bytes": 4, "total": 4}));
}

#[test]
fn download_progress_events_carry_the_id_bytes_and_total() {
    let (app, _w) = app();
    let got = collect(&app, "download-progress");
    let mut p = crate::infra::progress::Progress::events(app.handle().clone(), "o/n/f.txt".into());
    p.update(5, None);
    p.finish(9);
    // Mock-runtime events are delivered on the listener thread; wait for both.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while got.lock().unwrap().len() < 2 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let ev: Vec<Value> = got
        .lock()
        .unwrap()
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(
        ev,
        [
            json!({"id": "o/n/f.txt", "bytes": 5, "total": null}),
            json!({"id": "o/n/f.txt", "bytes": 9, "total": 9}),
        ]
    );
}

/// The helpers above are what every IPC test stands on; these pin their behaviour.
#[cfg(test)]
mod tests {
    use super::*;
    use tauri::Emitter;

    #[test]
    fn msg_reads_the_message_of_an_error_object() {
        let e = json!({"code": "invalid", "message": "bad value", "hint": "fix it"});
        assert_eq!(msg(&e), "bad value");
        assert!(std::panic::catch_unwind(|| msg(&json!("plain")).len()).is_err());
    }

    #[test]
    fn tmp_gives_a_fresh_empty_directory() {
        let d = tmp("ipc_selftest");
        std::fs::write(d.join("left-over"), "x").unwrap();
        let again = tmp("ipc_selftest");
        assert_eq!(d, again);
        assert_eq!(std::fs::read_dir(&again).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(again);
    }

    #[test]
    fn collect_and_wait_for_see_emitted_events() {
        let (app, _w) = app();
        let got = collect(&app, "ipc-selftest");
        app.emit("ipc-selftest", "one").unwrap();
        app.emit("ipc-selftest", "two").unwrap();
        wait_for(&got, 2);
        assert_eq!(*got.lock().unwrap(), ["\"one\"", "\"two\""]);
    }

    #[test]
    fn handler_commands_reads_the_real_registry() {
        let cmds = handler_commands(include_str!("lib.rs"));
        for c in ["train_slm", "serve_status", "download_kaggle_file"] {
            assert!(cmds.iter().any(|x| x == c), "{c} missing from {cmds:?}");
        }
        assert!(cmds
            .iter()
            .all(|c| !c.contains("::") && !c.contains(char::is_whitespace)));
    }
}
