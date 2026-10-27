//! The Generate tab: sampling controls, stop strings, streaming and cancel.

use super::time_seed;
use crate::error::CmdError;
use crate::jobs::{Job, JobGuard};
use crate::paths::checked_path;
use crate::AppState;
use ferrum_core::{GenerativeSLM, Rng, SamplingParams};
use serde::Deserialize;
use tauri::{AppHandle, Emitter, Runtime, State};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenParams {
    pub model_path: String,
    pub seed_text: String,
    pub num_chars: usize,
    /// Softmax temperature; `0` is exact argmax (greedy), like `--temp 0`.
    pub temp: f32,
    pub gen_seed: Option<u64>,
    pub stream: bool,
    pub verbose: bool,
    /// Keep only the `k` highest-probability tokens (`0` disables).
    #[serde(default)]
    pub top_k: usize,
    /// Nucleus sampling threshold (`1.0` disables).
    #[serde(default = "one_f32")]
    pub top_p: f32,
    /// Down-weight already-seen tokens (`1.0` disables).
    #[serde(default = "one_f32")]
    pub rep_penalty: f32,
    /// Token budget (`0` = none: `num_chars` counts characters, as before).
    #[serde(default)]
    pub max_tokens: usize,
    /// Stop strings, one per entry; generation ends before the first one that
    /// appears. `\n`, `\t` and `\\` are unescaped (a line cannot hold a real newline).
    #[serde(default)]
    pub stop: Vec<String>,
}

/// The sampler settings shared by the Generate and GGUF tabs, validated like
/// the CLI's `sampling_params` (`--temp >= 0`, `--top-p` in [0, 1], a positive
/// repetition penalty).
pub(super) fn sampling_from(
    temp: f32,
    top_k: usize,
    top_p: f32,
    rep_penalty: f32,
) -> Result<SamplingParams, String> {
    if !(temp.is_finite() && temp >= 0.0) {
        return Err(format!(
            "temperature must be a number >= 0 (0 = greedy), got {temp}"
        ));
    }
    if !(top_p.is_finite() && (0.0..=1.0).contains(&top_p)) {
        return Err(format!("top-p must be a number in [0, 1], got {top_p}"));
    }
    if !(rep_penalty.is_finite() && rep_penalty > 0.0) {
        return Err(format!(
            "repetition penalty must be a positive number, got {rep_penalty}"
        ));
    }
    Ok(SamplingParams {
        temperature: temp,
        top_k,
        top_p,
        repetition_penalty: rep_penalty,
    })
}

/// Stop strings as typed (one per line): empty lines dropped, `\n` `\t` `\\` unescaped.
fn parse_stops(raw: &[String]) -> Vec<String> {
    raw.iter()
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut out = String::new();
            let mut it = s.chars();
            while let Some(c) = it.next() {
                if c != '\\' {
                    out.push(c);
                    continue;
                }
                match it.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('\\') => out.push('\\'),
                    Some(o) => {
                        out.push('\\');
                        out.push(o);
                    }
                    None => out.push('\\'),
                }
            }
            out
        })
        .collect()
}

/// Carried by `resume_unwind` out of the engine's text callback to end a cancelled generation.
struct Cancelled;

/// Generate exactly as `train_transformer generate` does for the same flags:
/// a token budget or stop strings select the token path (`generate_tokens_with`,
/// stop text excluded), otherwise `num_chars` characters. Returns seed +
/// continuation; `on_text` receives only the continuation, as it is produced.
fn generate_once(
    slm: &GenerativeSLM,
    p: &GenParams,
    on_text: &mut dyn FnMut(&str),
) -> Result<String, String> {
    let params = sampling_from(p.temp, p.top_k, p.top_p, p.rep_penalty)?;
    let stops = parse_stops(&p.stop);
    let mut rng = Rng::new(p.gen_seed.unwrap_or_else(time_seed));
    let out = if p.max_tokens > 0 || !stops.is_empty() {
        let max_tokens = if p.max_tokens > 0 {
            p.max_tokens
        } else {
            p.num_chars
        };
        slm.generate_tokens_with(&p.seed_text, max_tokens, &stops, &params, &mut rng, on_text)
            .map(|g| format!("{}{}", p.seed_text, g.text))
    } else {
        slm.generate_stream_with(&p.seed_text, p.num_chars, &params, &mut rng, |f| on_text(f))
    };
    out.map_err(|e| format!("generation failed: {e}"))
}

/// [`generate_once`] that ends as soon as `cancelled()` answers true after a fragment, returning
/// (seed + the text so far, true).
///
/// ponytail: the three decode loops in `ferrum_core` have no stop hook, so the callback unwinds
/// (`resume_unwind` runs no panic hook and prints nothing) and `catch_unwind` stops here; the
/// loops only borrow the model, so nothing is left half-built. Upgrade path: a cancel hook in
/// `generate_stream_core` and the stream types.
pub(crate) fn run_generation(
    slm: &GenerativeSLM,
    p: &GenParams,
    cancelled: &dyn Fn() -> bool,
    on_text: &mut dyn FnMut(&str),
) -> Result<(String, bool), String> {
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    let mut so_far = String::new();
    let run = catch_unwind(AssertUnwindSafe(|| {
        generate_once(slm, p, &mut |frag| {
            so_far.push_str(frag);
            on_text(frag);
            if cancelled() {
                resume_unwind(Box::new(Cancelled));
            }
        })
    }));
    match run {
        Ok(text) => text.map(|t| (t, false)),
        Err(payload) if payload.is::<Cancelled>() => Ok((format!("{}{so_far}", p.seed_text), true)),
        Err(payload) => resume_unwind(payload),
    }
}

/// Load a model and generate text. With `stream = true`, each fragment is
/// emitted as a `gen-fragment` event as it is produced; the full string is also
/// returned. `cancel_job("generate")` ends it after the current fragment and
/// returns the text produced so far.
#[tauri::command]
pub async fn generate_slm<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    params: GenParams,
) -> Result<String, CmdError> {
    let job = state.jobs.start(Job::Generate)?;
    crate::error::blocking(move || gen_inner(app, params, job)).await
}

fn gen_inner<R: Runtime>(app: AppHandle<R>, p: GenParams, job: JobGuard) -> Result<String, String> {
    sampling_from(p.temp, p.top_k, p.top_p, p.rep_penalty)?; // fail before the model loads
    checked_path("model path", &p.model_path, true)?;
    let slm = GenerativeSLM::load(&p.model_path)
        .map_err(|e| format!("cannot load model {}: {e}", p.model_path))?;

    ferrum_core::set_verbose(p.verbose);
    let cancelled = || job.cancelled();
    let result = if p.stream {
        run_generation(&slm, &p, &cancelled, &mut |frag| {
            let _ = app.emit("gen-fragment", frag.to_string());
        })
    } else {
        run_generation(&slm, &p, &cancelled, &mut |_| {})
    };
    ferrum_core::set_verbose(false);
    result.map(|(text, _stopped)| text)
}

pub(super) fn one_f32() -> f32 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::TransformerConfig;

    fn tiny_bpe() -> GenerativeSLM {
        let corpus = "the quick brown fox jumps over the lazy dog. ".repeat(40);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 16,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 32,
            epochs: 2,
            lr: 0.01,
            batch_size: 8,
            vocab_size: 300,
            ..TransformerConfig::default()
        };
        GenerativeSLM::train_transformer_config_threaded(
            &corpus,
            &cfg,
            1,
            &mut Rng::new(3),
            |_, _| {},
        )
        .unwrap()
    }

    fn gen_params(seed_text: &str) -> GenParams {
        serde_json::from_value(serde_json::json!({
            "modelPath": "m", "seedText": seed_text, "numChars": 60, "temp": 0.8,
            "genSeed": 5, "stream": false, "verbose": false
        }))
        .unwrap()
    }

    fn generate_with_gui(slm: &GenerativeSLM, p: &GenParams) -> (String, String) {
        let mut streamed = String::new();
        let (out, stopped) =
            run_generation(slm, p, &|| false, &mut |f| streamed.push_str(f)).unwrap();
        assert!(!stopped);
        (out, streamed)
    }

    #[test]
    fn gen_params_new_fields_default_to_the_old_behaviour() {
        let p = gen_params("x");
        assert_eq!((p.top_k, p.top_p, p.rep_penalty), (0, 1.0, 1.0));
        assert_eq!((p.max_tokens, p.stop.len()), (0, 0));
    }

    #[test]
    fn character_path_equals_the_library_and_streams_the_continuation() {
        let slm = tiny_bpe();
        let p = gen_params("the quick");
        let (out, streamed) = generate_with_gui(&slm, &p);
        let want = slm
            .generate_with(
                "the quick",
                60,
                &SamplingParams::with_temperature(0.8),
                &mut Rng::new(5),
            )
            .unwrap();
        assert_eq!(out, want);
        assert_eq!(format!("the quick{streamed}"), out);
    }

    #[test]
    fn token_path_equals_generate_tokens_for_every_sampling_control() {
        let slm = tiny_bpe();
        let mut p = gen_params("the ");
        p.temp = 0.9;
        p.top_k = 5;
        p.top_p = 0.9;
        p.rep_penalty = 1.2;
        p.max_tokens = 25;
        let (out, streamed) = generate_with_gui(&slm, &p);
        let params = SamplingParams {
            temperature: 0.9,
            top_k: 5,
            top_p: 0.9,
            repetition_penalty: 1.2,
        };
        let g = slm
            .generate_tokens("the ", 25, &[], &params, &mut Rng::new(5))
            .unwrap();
        assert_eq!(out, format!("the {}", g.text));
        assert_eq!(streamed, g.text);
        assert!(g.n_tokens <= 25);
        // Without max_tokens the character budget doubles as the token budget (as in the CLI).
        let mut q = gen_params("the ");
        q.stop = vec!["zzzz-never".into()];
        let g = slm
            .generate_tokens(
                "the ",
                60,
                &["zzzz-never".to_string()],
                &SamplingParams::with_temperature(0.8),
                &mut Rng::new(5),
            )
            .unwrap();
        assert_eq!(generate_with_gui(&slm, &q).0, format!("the {}", g.text));
    }

    #[test]
    fn temperature_zero_is_exact_argmax_whatever_the_seed() {
        let slm = tiny_bpe();
        let mut p = gen_params("the ");
        p.temp = 0.0;
        p.gen_seed = Some(1);
        let a = generate_with_gui(&slm, &p).0;
        p.gen_seed = Some(999);
        let b = generate_with_gui(&slm, &p).0;
        p.gen_seed = None; // time-seeded: still identical
        let c = generate_with_gui(&slm, &p).0;
        assert_eq!(a, b);
        assert_eq!(a, c);
        let want = slm
            .generate_with(
                "the ",
                60,
                &SamplingParams::with_temperature(0.0),
                &mut Rng::new(0),
            )
            .unwrap();
        assert_eq!(a, want);
    }

    #[test]
    fn stop_strings_end_generation_and_are_not_in_the_text() {
        let slm = tiny_bpe();
        let mut p = gen_params("the ");
        p.temp = 0.0;
        let full = generate_with_gui(&slm, &p).0;
        let cont: Vec<char> = full.chars().skip(4).collect();
        assert!(cont.len() > 12, "{full:?}");
        let stop: String = cont[8..10].iter().collect();
        p.stop = vec!["".into(), stop.clone()]; // the empty line is ignored
        let (out, streamed) = generate_with_gui(&slm, &p);
        let at = full[4..].find(&stop).unwrap();
        assert_eq!(out, format!("the {}", &full[4..][..at]));
        assert!(!streamed.contains(&stop));
        assert_eq!(format!("the {streamed}"), out);
    }

    #[test]
    fn stop_escapes_and_validation() {
        let raw = |s: &str| vec![s.to_string()];
        assert_eq!(parse_stops(&raw("a\\nb")), vec!["a\nb"]);
        assert_eq!(parse_stops(&raw("\\t.")), vec!["\t."]);
        assert_eq!(parse_stops(&raw("c:\\\\x")), vec!["c:\\x"]);
        assert_eq!(parse_stops(&raw("\\q")), vec!["\\q"]);
        assert_eq!(parse_stops(&raw("end\\")), vec!["end\\"]);
        assert!(parse_stops(&raw("")).is_empty());

        assert!(sampling_from(0.0, 0, 1.0, 1.0).is_ok());
        for (t, p, r, word) in [
            (-0.1, 1.0, 1.0, "temperature"),
            (f32::NAN, 1.0, 1.0, "temperature"),
            (0.5, 1.5, 1.0, "top-p"),
            (0.5, -0.1, 1.0, "top-p"),
            (0.5, 1.0, 0.0, "repetition"),
        ] {
            let e = sampling_from(t, 0, p, r).unwrap_err();
            assert!(e.contains(word), "{e}");
        }
    }

    #[test]
    fn a_cancelled_generation_returns_the_text_so_far_for_both_paths() {
        let slm = tiny_bpe();
        for max_tokens in [0usize, 30] {
            let mut p = gen_params("the ");
            p.temp = 0.0;
            p.max_tokens = max_tokens;
            let (full, _) = generate_with_gui(&slm, &p);
            let polls = std::cell::Cell::new(0);
            let mut streamed = String::new();
            let (part, stopped) = run_generation(
                &slm,
                &p,
                &|| {
                    polls.set(polls.get() + 1);
                    polls.get() >= 3
                },
                &mut |f| streamed.push_str(f),
            )
            .unwrap();
            assert!(stopped, "max_tokens {max_tokens}");
            assert_eq!(part, format!("the {streamed}"));
            assert!(
                full.starts_with(&part) && part.len() < full.len(),
                "{part:?} vs {full:?}"
            );
        }
        // A cancel that never comes changes nothing; other panics still propagate.
        let r = std::panic::catch_unwind(|| {
            let mut p = gen_params("the ");
            p.temp = 0.0;
            run_generation(&tiny_bpe(), &p, &|| false, &mut |_| panic!("boom"))
        });
        assert!(r.is_err());
    }
}
