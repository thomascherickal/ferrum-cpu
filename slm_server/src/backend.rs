//! The model a server answers with: Ferrum's own [`GenerativeSLM`] or an imported Llama/Qwen
//! GGUF ([`Imported`]), behind the three primitives the routes need. Every answer is the JSON
//! object a route returns per request, the same shape for both kinds.

use crate::imported::Imported;
use ferrum_core::{GenerativeSLM, InferError, Result, Rng, SamplingParams};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// One value per server, built once on the model thread and never moved after that.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Backend {
    Ferrum(GenerativeSLM),
    Imported(Imported),
    /// Test stand-in for a slow model: counts generate starts and ticks.
    #[cfg(test)]
    Stub(Arc<StubStats>),
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct StubStats {
    pub(crate) started: std::sync::atomic::AtomicUsize,
    pub(crate) ticks: std::sync::atomic::AtomicUsize,
}

/// `max_tokens` when a generate request omits it (capped by the server's `--max-tokens`).
pub(crate) const DEFAULT_MAX_TOKENS: usize = 64;
fn default_top_p() -> f32 {
    1.0
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GenRequest {
    pub(crate) prompt: String,
    /// `None` until the HTTP layer fills in the default, capped by the server limit.
    #[serde(default)]
    pub(crate) max_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) stop: Vec<String>,
    #[serde(default)]
    pub(crate) temperature: f32,
    #[serde(default)]
    pub(crate) top_k: usize,
    #[serde(default = "default_top_p")]
    pub(crate) top_p: f32,
    #[serde(default)]
    pub(crate) seed: u64,
    /// `None` until the HTTP layer fills in the server default.
    #[serde(default)]
    pub(crate) repetition_penalty: Option<f32>,
    /// Wrap the prompt in the imported model's chat template (default: a raw completion).
    /// A Ferrum model has no chat template, so `true` is an error there.
    #[serde(default)]
    pub(crate) chat: bool,
    /// Set by the HTTP layer when the client disconnects; never read from the body.
    #[serde(skip)]
    pub(crate) cancel: Option<Arc<AtomicBool>>,
}

impl GenRequest {
    fn cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed))
    }
}

impl Backend {
    /// `/v1/loglikelihood`: `{logprob, n_tokens, greedy}`.
    pub(crate) fn score(&self, context: &str, continuation: &str) -> Result<Value> {
        let (logprob, n_tokens, greedy) = match self {
            #[cfg(test)]
            Backend::Stub(_) => unreachable!("the stub only generates"),
            Backend::Ferrum(m) => {
                let s = m.score(context, continuation)?;
                (s.logprob, s.n_tokens, s.greedy)
            }
            Backend::Imported(m) => {
                let (s, greedy) = m.score(context, continuation)?;
                (s.logprob, s.n_tokens, greedy)
            }
        };
        Ok(json!({"logprob": logprob, "n_tokens": n_tokens, "greedy": greedy}))
    }

    /// `/v1/loglikelihood_rolling`: `{logprob, n_tokens}`.
    pub(crate) fn score_rolling(&self, text: &str) -> Result<Value> {
        let (logprob, n_tokens) = match self {
            #[cfg(test)]
            Backend::Stub(_) => unreachable!("the stub only generates"),
            Backend::Ferrum(m) => {
                let s = m.score_rolling(text)?;
                (s.logprob, s.n_tokens)
            }
            Backend::Imported(m) => {
                let s = m.score_rolling(text)?;
                (s.logprob, s.n_tokens)
            }
        };
        Ok(json!({"logprob": logprob, "n_tokens": n_tokens}))
    }

    /// `/v1/generate`: `{text, n_tokens}`.
    /// A request whose client already left is skipped (the reply is discarded); one that
    /// disconnects mid-run ends at its next token. Scoring has no such check.
    pub(crate) fn generate(&self, g: &GenRequest) -> Result<Value> {
        if g.cancelled() {
            return Ok(Value::Null);
        }
        let (text, n_tokens) =
            match self {
                #[cfg(test)]
                Backend::Stub(st) => {
                    st.started.fetch_add(1, Ordering::SeqCst);
                    for _ in 0..g.max_tokens.unwrap_or(1) {
                        if g.cancelled() {
                            break;
                        }
                        st.ticks.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    (String::new(), 0)
                }
                Backend::Ferrum(_) if g.chat => return Err(InferError::Format(
                    "chat: a Ferrum model has no chat template; send the prompt without \"chat\""
                        .into(),
                )),
                Backend::Ferrum(m) => {
                    let params = SamplingParams {
                        temperature: g.temperature,
                        top_k: g.top_k,
                        top_p: g.top_p,
                        repetition_penalty: g.repetition_penalty.unwrap_or(1.0),
                        cancel: g.cancel.clone(),
                    };
                    let o = m.generate_tokens(
                        &g.prompt,
                        g.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
                        &g.stop,
                        &params,
                        &mut Rng::new(g.seed),
                    )?;
                    (o.text, o.n_tokens)
                }
                Backend::Imported(m) => {
                    let o = m.generate(g)?;
                    (o.text, o.n_tokens)
                }
            };
        Ok(json!({"text": text, "n_tokens": n_tokens}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{imported, llama, model};

    #[test]
    fn default_helpers() {
        assert_eq!((DEFAULT_MAX_TOKENS, default_top_p()), (64, 1.0));
    }

    #[test]
    fn both_kinds_answer_with_the_same_shapes() {
        let req: GenRequest =
            serde_json::from_value(json!({"prompt": "ab", "max_tokens": 2})).unwrap();
        for b in [
            Backend::Ferrum(model(0)),
            Backend::Imported(imported(&llama::tiny_llama_bos_eos())),
        ] {
            let s = b.score("ab", "c").unwrap();
            let keys: Vec<_> = s.as_object().unwrap().keys().cloned().collect();
            assert_eq!(keys, ["greedy", "logprob", "n_tokens"]);
            let r = b.score_rolling("abc").unwrap();
            assert!(r["logprob"].as_f64().unwrap() < 0.0 && r["n_tokens"].as_u64() > Some(0));
            let g = b.generate(&req).unwrap();
            assert!(g["text"].is_string() && g["n_tokens"].as_u64() <= Some(2));
        }
    }

    #[test]
    fn chat_is_refused_on_a_ferrum_model_and_parsed_off_by_default() {
        let off: GenRequest = serde_json::from_value(json!({"prompt": "ab"})).unwrap();
        assert!(!off.chat);
        let on: GenRequest =
            serde_json::from_value(json!({"prompt": "ab", "max_tokens": 2, "chat": true})).unwrap();
        let e = Backend::Ferrum(model(0)).generate(&on).unwrap_err();
        assert!(e.to_string().contains("no chat template"), "{e}");
        assert!(
            serde_json::from_value::<GenRequest>(json!({"prompt": "a", "chat": "yes"})).is_err()
        );
    }

    #[test]
    fn a_request_cancelled_before_it_starts_is_skipped_on_both_kinds() {
        let mut req: GenRequest =
            serde_json::from_value(json!({"prompt": "ab", "max_tokens": 4})).unwrap();
        req.cancel = Some(Arc::new(AtomicBool::new(true)));
        for b in [
            Backend::Ferrum(model(0)),
            Backend::Imported(imported(&llama::tiny_llama_bos_eos())),
        ] {
            assert_eq!(b.generate(&req).unwrap(), Value::Null);
        }
    }
}
