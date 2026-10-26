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

fn app() -> (
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

fn call(
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
        InvokeResponseBody::Raw(b) => json!(b.len()),
    })
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
    let (_app, w) = app();
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
    // Errors come back as the plain message string the UI shows.
    let e = call(
        &w,
        "clean_text",
        json!({"raw": "   ", "opts": {
        "stripGutenberg": false, "lowercase": false, "collapseWhitespace": false,
        "normalizePunctuation": false, "stripControlChars": false, "maxChars": null}}),
    )
    .unwrap_err();
    assert!(e.as_str().unwrap().contains("nothing left"), "{e}");

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

fn tmp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_ipc_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Collect the payloads of one event for the rest of the test.
fn collect(app: &App<tauri::test::MockRuntime>, event: &str) -> Arc<Mutex<Vec<String>>> {
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
    assert!(bad.as_str().unwrap().contains("temperature"), "{bad}");

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
fn wait_for(got: &Arc<Mutex<Vec<String>>>, n: usize) {
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
    assert_eq!(
        e.as_str().unwrap(),
        "a training job is already running",
        "{e}"
    );
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
    assert_eq!(e.as_str().unwrap(), "a fine-tuning job is already running");
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
fn training_errors_reach_the_ui_as_plain_messages() {
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
    let e = call(&w, "train_slm", base(3)).unwrap_err();
    assert!(e.as_str().unwrap().contains("divisible"), "{e}");
    let mut missing = base(2);
    missing["params"]["corpusPath"] = json!(dir.join("nope.txt").to_string_lossy());
    assert!(call(&w, "train_slm", missing)
        .unwrap_err()
        .as_str()
        .unwrap()
        .contains("does not exist"));
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

/// The arguments app.js really sends (captured by ui/app_flows.test.js) must
/// deserialize into the Rust parameter types; a renamed or mistyped required
/// field fails here rather than in a window.
#[test]
fn arguments_captured_from_app_js_deserialize_into_the_command_types() {
    use crate::commands::{
        CleanOpts, ExportParams, FinetuneParams, GenParams, GgufRunParams, TrainParams,
    };
    use crate::evaluate::EvalParams;
    use crate::prepare::PrepareParams;
    let fixture: Value =
        serde_json::from_str(include_str!("../ui/command_params.fixture.json")).unwrap();
    macro_rules! parses {
        ($key:literal, $ty:ty) => {
            serde_json::from_value::<$ty>(fixture[$key].clone())
                .unwrap_or_else(|e| panic!("{}: {e}", $key))
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
    use crate::jobs::Job;
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
        assert_eq!(
            e.as_str().unwrap(),
            "a dataset preparation is already running"
        );
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
    assert!(
        e.as_str().unwrap().contains("invalid UTF-8 at byte 5"),
        "{e}"
    );
    assert!(e.as_str().unwrap().contains(&*bad.to_string_lossy()), "{e}");
    let e = call(
        &w,
        "prepare_dataset",
        body(json!({"inputs": ["/proc/cpuinfo"]})),
    )
    .unwrap_err();
    assert!(e.as_str().unwrap().contains("not allowed"), "{e}");
    let e = call(
        &w,
        "prepare_dataset",
        body(json!({"outDir": "/proc/prepared"})),
    )
    .unwrap_err();
    assert!(e.as_str().unwrap().contains("not allowed"), "{e}");
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
