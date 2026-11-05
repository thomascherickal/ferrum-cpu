//! `eval-gguf`: perplexity and bits per byte of an imported (Llama/Qwen-style) GGUF on a text file.

use crate::args::*;
use crate::eval::{report, Stats};
use crate::memory::*;
use crate::util::*;
use crate::validate::*;
use std::time::Instant;

/// Default scoring window: long enough for useful context, short enough that a 32k-context
/// model is not scored with one enormous (quadratic) forward pass.
const DEFAULT_WINDOW: usize = 512;

pub(crate) fn cmd_eval_gguf(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use ferrum_core::{Gguf, GgufTokenizer};

    if args.positional.len() < 2 {
        return Err(
            "usage: train_transformer eval-gguf <model.gguf> <text.txt> \
                    [--ctx N] [--quant f32|int8|int4] [--resume ckpt.flck] [--json out.json]"
                .into(),
        );
    }
    require_resume_file(args.value("resume"), ".flck")?;
    if let Some(json) = args.value("json") {
        check_writable(json)?;
    }
    let (path, text_path) = (&args.positional[0], &args.positional[1]);
    let resume = args.value("resume");
    let quant = args.value("quant").unwrap_or("f32");
    let prec = ferrum_core::memory::import_precision(quant)
        .ok_or_else(|| format!("--quant must be f32|int8|int4 (got '{quant}')"))?;
    // A checkpoint holds f32 weights, so applying one needs an f32 load.
    let prec = if resume.is_some() {
        if prec.is_some() {
            eprintln!("note: --resume applies f32 fine-tuned weights; ignoring --quant {quant}");
        }
        None
    } else {
        prec
    };
    let text = read_corpus(text_path)?;

    eprintln!("Opening {path} (streamed)…");
    let g = Gguf::open(path).map_err(|e| format!("cannot open GGUF {path}: {e}"))?;
    let tok = GgufTokenizer::from_gguf(&g)
        .map_err(|_| "this GGUF has no tokenizer; perplexity needs one")?;
    let mut est = estimate_resident_bytes(&g, prec);
    if let Some(ckpt) = resume {
        est = est.max(resume_peak_bytes(est, file_len(ckpt)));
    }
    if let Some(avail) = available_memory_bytes() {
        if let Some(e) = memory_error(
            "resident memory",
            est,
            avail,
            args.has("force"),
            "try --quant int8 or int4, or pass --force to attempt it anyway.",
        ) {
            return Err(e.into());
        }
    }
    eprintln!("Loading weights…");
    let mut model = g
        .load_llama_prec(prec)
        .map_err(|e| format!("cannot load model: {e}"))?;
    if let Some(ckpt) = resume {
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        ferrum_core::apply_checkpoint_weights(&mut model, bytes)
            .map_err(|e| format!("cannot apply checkpoint {ckpt}: {e}"))?;
        eprintln!("  applied fine-tuned checkpoint {ckpt}");
    }
    let max_window = model.cfg.context_len;
    let window = args.get("ctx", DEFAULT_WINDOW.min(max_window));
    if window < 2 || window > max_window {
        return Err(format!("--ctx must be between 2 and {max_window} (got {window})").into());
    }

    // The BOS token (when the tokenizer adds one) lets the first text token be scored too.
    let mut ids: Vec<usize> = tok.prefix_bos().into_iter().collect();
    ids.extend(tok.encode(&text));
    if ids.len() < 2 {
        return Err("the text is too short: scoring needs at least two tokens".into());
    }
    let t0 = Instant::now();
    let score = model.score_rolling(&ids, window)?;
    let seconds = t0.elapsed().as_secs_f64();
    let stats = Stats::from_nll(-score.logprob, score.n_tokens);
    report(path, text_path, &text, &stats, seconds, args.value("json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{letter_vocab, tiny_llama_gguf};
    use crate::test_util::{args, err, Tmp};
    use ferrum_core::MetaValue;

    fn eval(raw: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        cmd_eval_gguf(&args("eval-gguf", raw))
    }

    /// The scored-token count and perplexity a successful run writes to `--json`.
    fn scored(t: &Tmp, model: &str, text: &str, extra: &[&str]) -> (u64, f64) {
        let json = t.path("out.json");
        let mut raw = vec![model, text, "--json", &json];
        raw.extend_from_slice(extra);
        eval(&raw).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
        (
            v["tokens"].as_u64().unwrap(),
            v["perplexity"].as_f64().unwrap(),
        )
    }

    #[test]
    fn arguments_are_checked_before_the_model_is_opened() {
        assert!(err(eval(&["m.gguf"])).starts_with("usage: train_transformer eval-gguf"));
        let e = err(eval(&["m.gguf", "t.txt", "--quant", "q9"]));
        assert_eq!(e, "--quant must be f32|int8|int4 (got 'q9')");
        let e = err(eval(&[
            "m.gguf",
            "t.txt",
            "--resume",
            "/nonexistent/x.flck",
        ]));
        assert!(e.contains("does not exist"), "{e}");
        // A path below a regular file can be created on no OS (a root-relative
        // "/nonexistent/dir" can on a Windows admin runner).
        let t = Tmp::new("evalgguf_badjson");
        let blocker = t.write("blocker", "x");
        let json = format!("{blocker}/dir/o.json");
        let e = err(eval(&["m.gguf", "t.txt", "--json", &json]));
        assert!(e.contains("blocker"), "{e}");
    }

    #[test]
    fn the_model_must_open_and_carry_a_tokenizer() {
        let t = Tmp::new("evalgguf_tok");
        let text = t.write("t.txt", "abc def");
        let e = err(eval(&[&t.path("none.gguf"), &text]));
        assert!(e.starts_with("cannot open GGUF "), "{e}");
        let bare = t.write("bare.gguf", tiny_llama_gguf(None, &[]));
        assert_eq!(
            err(eval(&[&bare, &text])),
            "this GGUF has no tokenizer; perplexity needs one"
        );
    }

    #[test]
    fn window_and_text_length_are_bounded_by_the_model() {
        let t = Tmp::new("evalgguf_ctx");
        let model = t.write("m.gguf", tiny_llama_gguf(Some(&letter_vocab()), &[]));
        let text = t.write("t.txt", "abc def");
        for bad in ["1", "17"] {
            let e = err(eval(&[&model, &text, "--ctx", bad]));
            assert_eq!(e, format!("--ctx must be between 2 and 16 (got {bad})"));
        }
        let one = t.write("one.txt", "a");
        assert!(err(eval(&[&model, &one])).starts_with("the text is too short"));
    }

    /// Every token after the first is scored; a BOS-adding tokenizer makes the first one
    /// scorable too, and `--ctx` really changes the scoring window.
    #[test]
    fn scores_every_token_and_honours_bos_and_the_window() {
        let t = Tmp::new("evalgguf_ok");
        let model = t.write("m.gguf", tiny_llama_gguf(Some(&letter_vocab()), &[]));
        let text = t.write("t.txt", "abcdefgh ijklmnop qrst"); // 22 letter/space tokens
        let (n, full) = scored(&t, &model, &text, &[]);
        assert_eq!(n, 21);
        assert!(full.is_finite() && full > 1.0);
        let (n4, short) = scored(&t, &model, &text, &["--ctx", "4"]);
        assert_eq!(n4, 21);
        assert_ne!(
            short, full,
            "a 4-token window sees less context than the default 16"
        );
        assert_eq!(scored(&t, &model, &text, &["--quant", "int8"]).0, 21);
        let bos = [
            ("tokenizer.ggml.bos_token_id", MetaValue::U32(27)),
            ("tokenizer.ggml.add_bos_token", MetaValue::Bool(true)),
        ];
        let with_bos = t.write("bos.gguf", tiny_llama_gguf(Some(&letter_vocab()), &bos));
        assert_eq!(scored(&t, &with_bos, &text, &[]).0, 22);
    }
}
