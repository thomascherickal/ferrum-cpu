//! `run` and `generate`: sample text from a Ferrum model.

use crate::args::*;
use crate::train::*;
use crate::util::*;
use crate::validate::*;
use ferrum_core::{GenerativeSLM, Rng};

pub(crate) fn cmd_run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 3 {
        return Err(
            "usage: train_transformer run <corpus.txt> <model.gguf> <seed text> [options]".into(),
        );
    }
    sampling_params(args, 0.8, "repetition-penalty")?; // fail before training starts
    require_resume_file(args.value("resume"), ".fckp")?; // also when the model exists and training is skipped
    let corpus_path = &args.positional[0];
    let model_path = &args.positional[1];
    let seed_text = args.positional[2..].join(" ");
    if ferrum_core::is_gguf_file(corpus_path) {
        return Err(format!(
            "{corpus_path} is a token file; `run` needs a text corpus (use `train` on the token file, then `generate`)"
        )
        .into());
    }
    let corpus = read_corpus(corpus_path)?;

    let slm = if std::path::Path::new(model_path).exists() && !args.has("force") {
        let slm = GenerativeSLM::load(model_path)?;
        println!("Loaded saved weights from {model_path} — no retraining needed.\n");
        slm
    } else {
        train_and_save(&corpus, corpus_path, model_path, args)?
    };

    let out = generate_text(&slm, &seed_text, args)?;
    if !args.has("stream") {
        println!("{out}");
    }
    Ok(())
}

pub(crate) fn cmd_generate(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 2 {
        return Err("usage: train_transformer generate <model.gguf> <seed text> [options]".into());
    }
    sampling_params(args, 0.8, "repetition-penalty")?; // fail before the model loads
    let model_path = &args.positional[0];
    // Join remaining positionals so multi-word seeds work without quoting.
    let seed_text = args.positional[1..].join(" ");

    let slm = GenerativeSLM::load(model_path)
        .map_err(|e| format!("cannot load model {model_path}: {e}"))?;
    let out = generate_text(&slm, &seed_text, args)?;
    // Streaming already printed the text live; avoid printing it twice.
    if !args.has("stream") {
        println!("{out}");
    }
    Ok(())
}

pub(crate) fn generate_text(
    slm: &GenerativeSLM,
    seed_text: &str,
    args: &Args,
) -> Result<String, Box<dyn std::error::Error>> {
    let num_chars: usize = args.get("chars", 200);
    let params = sampling_params(args, 0.8, "repetition-penalty")?;
    let seed: u64 = args.get("gen-seed", time_seed());

    let stops = args.get_all("stop");
    if args.value("max-tokens").is_some() || !stops.is_empty() {
        // Token-budgeted path: any prompt length, stop strings excluded.
        let max_tokens: usize = args.get("max-tokens", num_chars);
        let mut rng = Rng::new(seed);
        if args.has("stream") {
            // Live: only a possible stop-string prefix is held back, never stop text.
            use std::io::Write;
            print!("{seed_text}");
            let _ = std::io::stdout().flush();
            let g = slm.generate_tokens_with(
                seed_text,
                max_tokens,
                &stops,
                &params,
                &mut rng,
                &mut |t| {
                    print!("{t}");
                    let _ = std::io::stdout().flush();
                },
            )?;
            println!(); // callers skip their own print when streaming
            return Ok(format!("{seed_text}{}", g.text));
        }
        let g = slm.generate_tokens(seed_text, max_tokens, &stops, &params, &mut rng)?;
        return Ok(format!("{seed_text}{}", g.text));
    }

    let mut rng = Rng::new(seed);
    if args.has("stream") {
        // Print the seed, then each generated fragment as it lands, flushing so
        // the terminal shows the completion appearing live.
        use std::io::Write;
        print!("{seed_text}");
        let _ = std::io::stdout().flush();
        let out = slm.generate_stream_with(seed_text, num_chars, &params, &mut rng, |frag| {
            print!("{frag}");
            let _ = std::io::stdout().flush();
        })?;
        println!();
        Ok(out)
    } else {
        Ok(slm.generate_with(seed_text, num_chars, &params, &mut rng)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{args, err, micro_model, Tmp};

    fn text(slm: &GenerativeSLM, raw: &[&str]) -> String {
        generate_text(slm, "ab", &args("generate", raw)).unwrap()
    }

    #[test]
    fn commands_check_their_arguments_first() {
        assert!(err(cmd_run(&args("run", &["c.txt", "m.gguf"])))
            .starts_with("usage: train_transformer run "));
        assert!(err(cmd_generate(&args("generate", &["m.gguf"])))
            .starts_with("usage: train_transformer generate "));
        let e = err(cmd_generate(&args(
            "generate",
            &["/nonexistent/m.gguf", "hi", "--temp", "-1"],
        )));
        assert!(
            e.contains("--temp"),
            "sampling flags fail before the load: {e}"
        );
        let e = err(cmd_generate(&args(
            "generate",
            &["/nonexistent/m.gguf", "hi"],
        )));
        assert!(
            e.starts_with("cannot load model /nonexistent/m.gguf"),
            "{e}"
        );
        let e = err(cmd_run(&args(
            "run",
            &["c.txt", "m.gguf", "hi", "--top-p", "2"],
        )));
        assert!(e.contains("--top-p"), "{e}");
        let e = err(cmd_run(&args(
            "run",
            &["c.txt", "m.gguf", "hi", "--resume", "/nonexistent/x.fckp"],
        )));
        assert!(e.contains("does not exist"), "{e}");
    }

    #[test]
    fn run_refuses_a_token_file_and_reuses_an_existing_model() {
        let t = Tmp::new("generate_run");
        let model = micro_model(&t, "m.gguf");
        let before = std::fs::read(&model).unwrap();
        // A GGUF as the corpus is a token file: `run` needs text.
        let e = err(cmd_run(&args("run", &[&model, &t.path("x.gguf"), "ab"])));
        assert!(e.contains("is a token file"), "{e}");
        let corpus = t.write("c.txt", "abc abd abe ".repeat(8));
        cmd_run(&args("run", &[&corpus, &model, "ab", "--chars", "5"])).unwrap();
        assert_eq!(
            std::fs::read(&model).unwrap(),
            before,
            "an existing model is loaded, not retrained"
        );
        cmd_generate(&args(
            "generate",
            &[&model, "a", "b", "--chars", "3", "--stream"],
        ))
        .unwrap();
    }

    #[test]
    fn generate_text_is_seeded_and_streaming_returns_the_same_text() {
        let t = Tmp::new("generate_text");
        let slm = GenerativeSLM::load(&micro_model(&t, "m.gguf")).unwrap();
        for path in [
            vec!["--chars", "12", "--gen-seed", "5"],
            vec!["--max-tokens", "12", "--gen-seed", "5"],
        ] {
            let a = text(&slm, &path);
            assert!(a.starts_with("ab") && a.len() > 2, "{a:?}");
            assert_eq!(a, text(&slm, &path), "same --gen-seed, same text");
            let mut streamed = path.clone();
            streamed.push("--stream");
            assert_eq!(text(&slm, &streamed), a, "{path:?}");
        }
    }

    #[test]
    fn stop_strings_end_the_text_and_are_excluded() {
        let t = Tmp::new("generate_stop");
        let slm = GenerativeSLM::load(&micro_model(&t, "m.gguf")).unwrap();
        let free = text(
            &slm,
            &["--max-tokens", "30", "--gen-seed", "3", "--temp", "0"],
        );
        // Stop on a character the free run produced: the stopped text is a strict prefix.
        let stop = free[2..].chars().next().unwrap().to_string();
        let cut = text(
            &slm,
            &[
                "--max-tokens",
                "30",
                "--gen-seed",
                "3",
                "--temp",
                "0",
                "--stop",
                &stop,
            ],
        );
        assert!(!cut[2..].contains(&stop), "{cut:?} still holds {stop:?}");
        assert!(
            free.starts_with(&cut) && cut.len() < free.len(),
            "{cut:?} vs {free:?}"
        );
        let mut streamed = vec![
            "--max-tokens",
            "30",
            "--gen-seed",
            "3",
            "--temp",
            "0",
            "--stop",
            &stop,
        ];
        streamed.push("--stream");
        assert_eq!(text(&slm, &streamed), cut);
    }
}
