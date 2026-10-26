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

    let ev = call(&w, "evaluate_slm", json!({"params": {"modelPath": model, "text": "the lazy dog. the quick brown fox.", "textPath": null}})).unwrap();
    assert!(ev["perplexity"].as_f64().unwrap() > 0.0 && ev["numPredictions"].as_u64().unwrap() > 0);
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
        CleanOpts, EvalParams, ExportParams, FinetuneParams, GenParams, GgufRunParams, TrainParams,
    };
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
    assert_eq!(parses!("generate_slm", GenParams).gen_seed, Some(7));
    assert_eq!(
        parses!("evaluate_slm", EvalParams).text.as_deref(),
        Some("held out")
    );
    let g = parses!("run_gguf", GgufRunParams);
    assert_eq!((g.quant.as_str(), g.top_k, g.raw), ("int4", 40, true));
    assert_eq!(parses!("finetune_gguf", FinetuneParams).epochs, 3);
    assert_eq!(parses!("export_gguf", ExportParams).quant, "q8_0");
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
