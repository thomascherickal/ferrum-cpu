//! No test here starts Ollama or LM Studio: every tool is a shell script in a temp dir, passed as
//! `bin` (the first place the lookup checks), or nothing is run at all.

use super::*;
use ferrum_core::{GenerativeSLM, GgufBuilder, MetaValue, Rng, TransformerConfig};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_gui_rx_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A GGUF with just the header fields the export checks read.
fn header_gguf(dir: &Path, file: &str, arch: &str, ctx: u32) -> PathBuf {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String(arch.into()));
    b.meta("gpt2.context_length", MetaValue::U32(ctx));
    let p = dir.join(file);
    std::fs::write(&p, b.into_bytes()).unwrap();
    p
}

fn fake(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn ollama(model: &Path, bin: &Path) -> RuntimeParams {
    RuntimeParams {
        model_path: s(model),
        target: Target::Ollama,
        name: "ferrum-x".into(),
        repo: String::new(),
        bin: s(bin),
        modelfile_out: String::new(),
    }
}

fn lmstudio(model: &Path, bin: &Path) -> RuntimeParams {
    RuntimeParams {
        target: Target::Lmstudio,
        ..ollama(model, bin)
    }
}

const LS: &str = r#"[{"modelKey":"ferrum/m","path":"ferrum/m/m.gguf"}]"#;

fn lms(dir: &Path) -> PathBuf {
    std::fs::write(dir.join("ls.json"), LS).unwrap();
    fake(
        dir,
        "lms",
        &format!(
            "case \"$1\" in import) echo \"$@\" > {0}/args;; ls) cat {0}/ls.json;; esac",
            s(dir)
        ),
    )
}

#[test]
fn only_ferrum_gpt2_models_are_accepted_and_the_refusal_comes_before_any_tool() {
    let d = tmp("refuse");
    let marker = d.join("ran");
    let tool = fake(&d, "tool", &format!("touch {}", s(&marker)));
    let llama = header_gguf(&d, "llama.gguf", "llama", 64);
    let plain = header_gguf(&d, "plain.gguf", "ferrum", 64);
    for (model, want) in [
        (&llama, "Modelfile by hand (Manual 9.5)"),
        (&plain, "retrain with a BPE vocabulary"),
    ] {
        for make in [ollama, lmstudio] {
            let p = make(model, &tool);
            let e = do_preview(&p).unwrap_err();
            assert!(e.contains(want), "{e}");
            let e = do_export(&p, &|| false).unwrap_err();
            assert!(e.contains(want), "{e}");
        }
    }
    assert!(!marker.exists(), "a refused model never starts a tool");

    // Real models: a byte-level BPE model is a gpt2 GGUF and passes, a character-level one does not.
    let cfg = |vocab_size| TransformerConfig {
        context_len: 16,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        vocab_size,
        epochs: 1,
        batch_size: 8,
        lr: 0.01,
        ..TransformerConfig::default()
    };
    let text = "the quick brown fox jumps over the lazy dog. ".repeat(40);
    for (vocab, ok) in [(300, true), (0, false)] {
        let slm = GenerativeSLM::train_transformer_config_threaded(
            &text,
            &cfg(vocab),
            1,
            &mut Rng::new(1),
            |_, _| {},
        )
        .unwrap();
        let path = d.join(format!("real{vocab}.gguf"));
        slm.save(&s(&path)).unwrap();
        let r = do_preview(&ollama(&path, &tool));
        if ok {
            let r = r.unwrap();
            assert!(r.modelfile.unwrap().contains("PARAMETER num_ctx 16"));
        } else {
            assert!(r.unwrap_err().contains("not runnable in llama.cpp"));
        }
    }
    assert!(!marker.exists());
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn a_preview_shows_the_exact_modelfile_and_command_and_starts_nothing() {
    let d = tmp("preview");
    let marker = d.join("ran");
    let tool = fake(&d, "ollama", &format!("touch {}", s(&marker)));
    let model = header_gguf(&d, "m.gguf", "gpt2", 64);
    let r = do_preview(&ollama(&model, &tool)).unwrap();
    assert!(!r.ran && r.written.is_none() && r.output.is_empty());
    assert_eq!(
        r.modelfile.as_deref(),
        Some(
            slm_cli::runtime_export::modelfile_for(&model)
                .unwrap()
                .as_str()
        )
    );
    let m = r.modelfile.unwrap();
    assert!(
        m.contains(&format!("FROM \"{}\"", s(&model)))
            && m.contains("PARAMETER stop \"<|endoftext|>\"")
    );
    assert_eq!(r.tool.as_deref(), Some(s(&tool).as_str()));
    assert_eq!(
        r.commands,
        vec![format!(
            "{} create ferrum-x -f \"<temporary Modelfile>\"",
            s(&tool)
        )]
    );
    let lm = lms(&d);
    let r = do_preview(&lmstudio(&model, &lm)).unwrap();
    assert!(r.modelfile.is_none() && r.key.is_none());
    assert_eq!(r.commands.len(), 2);
    assert!(
        r.commands[0].contains("--user-repo ferrum/m "),
        "the default repo is ferrum/<stem>: {:?}",
        r.commands
    );
    assert!(
        !marker.exists() && !d.join("args").exists(),
        "nothing was run"
    );
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn write_modelfile_only_writes_the_file_and_runs_nothing() {
    let d = tmp("mfonly");
    let marker = d.join("ran");
    let tool = fake(&d, "ollama", &format!("touch {}", s(&marker)));
    let model = header_gguf(&d, "m.gguf", "gpt2", 48);
    let out = d.join("sub-not-yet/x.Modelfile");
    let p = RuntimeParams {
        modelfile_out: s(&d.join("x.Modelfile")),
        ..ollama(&model, &tool)
    };
    let r = do_export(&p, &|| false).unwrap();
    assert!(!r.ran && r.output.is_empty());
    assert_eq!(
        std::fs::read_to_string(d.join("x.Modelfile")).unwrap(),
        r.modelfile.clone().unwrap()
    );
    assert_eq!(
        r.written.as_deref(),
        Some(s(&d.join("x.Modelfile")).as_str())
    );
    assert!(r.modelfile.unwrap().contains("PARAMETER num_ctx 48"));
    assert!(r.summary.starts_with("wrote ") && r.summary.ends_with("nothing was run"));
    assert!(!marker.exists());
    // A directory that does not exist is an error naming the path; the guard refuses /proc.
    let bad = RuntimeParams {
        modelfile_out: s(&out),
        ..ollama(&model, &tool)
    };
    assert!(do_export(&bad, &|| false)
        .unwrap_err()
        .contains("x.Modelfile"));
    let dev = RuntimeParams {
        modelfile_out: "/proc/x".into(),
        ..ollama(&model, &tool)
    };
    assert!(do_export(&dev, &|| false)
        .unwrap_err()
        .contains("not allowed"));
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn the_ollama_run_gives_the_tool_the_same_modelfile_it_reports() {
    let d = tmp("ollama");
    let model = header_gguf(&d, "my model.gguf", "gpt2", 64);
    let tool = fake(
        &d,
        "ollama",
        &format!(
            "echo \"$@\" > {0}/args; cp \"$4\" {0}/seen; echo success",
            s(&d)
        ),
    );
    let r = do_export(&ollama(&model, &tool), &|| false).unwrap();
    assert!(r.ran && r.output.trim() == "success");
    assert_eq!(
        r.summary,
        "exported to Ollama as 'ferrum-x'  (try: ollama run ferrum-x)"
    );
    let seen = std::fs::read_to_string(d.join("seen")).unwrap();
    assert_eq!(
        Some(seen.as_str()),
        r.modelfile.as_deref(),
        "the shown Modelfile is the one used"
    );
    let args = std::fs::read_to_string(d.join("args")).unwrap();
    assert!(
        args.starts_with("create ferrum-x -f ") && r.commands[0].contains("create ferrum-x -f ")
    );
    // The tool failing (the server is down) gives the CLI's hint, and an error is not a run.
    let down = fake(&d, "down", "echo 'Error: connection refused' >&2; exit 1");
    let e = do_export(&ollama(&model, &down), &|| false).unwrap_err();
    assert!(
        e.contains("ollama serve") && e.contains("connection refused"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn the_lmstudio_run_imports_and_returns_the_model_key() {
    let d = tmp("lmstudio");
    let model = header_gguf(&d, "m.gguf", "gpt2", 8);
    let tool = lms(&d);
    let r = do_export(&lmstudio(&model, &tool), &|| false).unwrap();
    assert!(r.ran && r.key.as_deref() == Some("ferrum/m"));
    assert_eq!(
        r.summary,
        "imported into LM Studio as modelKey 'ferrum/m'  (try: lms load ferrum/m)"
    );
    assert_eq!(r.commands.len(), 2);
    let args = std::fs::read_to_string(d.join("args")).unwrap();
    assert!(
        args.starts_with("import -y -l --user-repo ferrum/m "),
        "{args}"
    );
    // A repo that is not publisher/name is refused before the tool starts.
    let bad = RuntimeParams {
        repo: "just-a-name".into(),
        ..lmstudio(&model, &tool)
    };
    assert!(do_export(&bad, &|| false)
        .unwrap_err()
        .contains("<publisher>/<name>"));
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn cancel_before_the_tool_starts_and_form_validation() {
    let d = tmp("validate");
    let marker = d.join("ran");
    let tool = fake(&d, "tool", &format!("touch {}", s(&marker)));
    let model = header_gguf(&d, "m.gguf", "gpt2", 8);
    for p in [ollama(&model, &tool), lmstudio(&model, &tool)] {
        assert_eq!(
            do_export(&p, &|| true).unwrap_err(),
            slm_cli::runtime_export::CANCELLED
        );
    }
    assert!(!marker.exists());
    for name in ["", "  ", "-rf", "a b", "../x", "x;y", ".hidden"] {
        let p = RuntimeParams {
            name: name.into(),
            ..ollama(&model, &tool)
        };
        let err = do_preview(&p).unwrap_err();
        let ok = err.contains("not a usable model name") || name.trim().is_empty();
        assert!(ok, "{name:?}: {err}");
        assert!(do_export(&p, &|| false).is_err(), "{name:?}");
    }
    for ok in ["ferrum", "me/ferrum:latest", "a.b_c-d"] {
        let p = RuntimeParams {
            name: ok.into(),
            ..ollama(&model, &tool)
        };
        assert!(do_preview(&p).is_ok(), "{ok}");
    }
    let missing_tool = RuntimeParams {
        bin: s(&d.join("nope")),
        ..ollama(&model, &tool)
    };
    assert!(do_export(&missing_tool, &|| false)
        .unwrap_err()
        .contains("does not exist"));
    let missing_model = ollama(&d.join("nope.gguf"), &tool);
    assert!(do_preview(&missing_model)
        .unwrap_err()
        .contains("does not exist"));
    let dev = ollama(Path::new("/dev/null"), &tool);
    assert!(do_preview(&dev).unwrap_err().contains("not allowed"));
    let dash = RuntimeParams {
        repo: "-x/y".into(),
        ..lmstudio(&model, &tool)
    };
    assert!(do_preview(&dash).unwrap_err().contains("not a usable repo"));
    assert!(!marker.exists());
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn a_preview_without_a_tool_path_looks_the_tool_up_and_reports_what_it_found() {
    let d = tmp("lookup");
    let model = header_gguf(&d, "m.gguf", "gpt2", 8);
    for target in [Target::Ollama, Target::Lmstudio] {
        let p = RuntimeParams {
            target,
            bin: String::new(),
            ..ollama(&model, &d)
        };
        // Whether the tool is installed here is not the test's business: exactly one of the
        // two answers is given.
        let r = do_preview(&p).unwrap();
        assert!(r.tool.is_some() != r.tool_error.is_some(), "{r:?}");
    }
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn the_tool_found_or_the_reason_for_none_is_recorded() {
    let mut r = base(&ollama(Path::new("m.gguf"), Path::new("t")));
    record_tool(&mut r, Ok(PathBuf::from("/usr/bin/ollama")));
    assert_eq!(
        (r.tool.as_deref(), r.tool_error.as_deref()),
        (Some("/usr/bin/ollama"), None)
    );
    let mut r = base(&ollama(Path::new("m.gguf"), Path::new("t")));
    record_tool(&mut r, Err("cannot find 'ollama'".into()));
    assert_eq!(
        (r.tool.as_deref(), r.tool_error.as_deref()),
        (None, Some("cannot find 'ollama'"))
    );
}
