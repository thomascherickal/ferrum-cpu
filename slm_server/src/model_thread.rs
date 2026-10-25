//! The thread that owns the model (`GenerativeSLM` is not `Send`/`Sync`): jobs
//! come in over a channel, replies go out through oneshot channels, and a panic
//! while answering one job fails that job only.

use crate::info::{info_for, Info};
use ferrum_core::{GenerativeSLM, Rng, SamplingParams};
use serde::Deserialize;
use serde_json::{json, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc, Weak};
use tokio::sync::oneshot;

pub(crate) type Reply = oneshot::Sender<Result<Vec<Value>, String>>;

pub(crate) enum Job {
    Score(Vec<(String, String)>, Reply),
    Rolling(Vec<String>, Reply),
    Generate(Vec<GenRequest>, Reply),
    /// Test hook: panics on the model thread.
    #[cfg(test)]
    Panic(Reply),
    /// Test hook: a parallel-pool kernel (panics on the pool workers if `true`).
    #[cfg(test)]
    Pool(bool, Reply),
}

/// Apply `f` to every item, stopping at the first model error.
fn collect<T>(
    items: &[T],
    f: impl Fn(&T) -> ferrum_core::Result<Value>,
) -> Result<Vec<Value>, String> {
    items
        .iter()
        .map(f)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

/// Turn a panic payload into a message (`&str`/`String` payloads only).
fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

/// Run `f`, converting a panic into an error so the model thread survives.
fn guarded(f: impl FnOnce() -> Result<Vec<Value>, String>) -> Result<Vec<Value>, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        Err(format!(
            "internal error: model panicked: {}",
            panic_message(&*p)
        ))
    })
}

/// Answer one job on the model thread.
fn handle(model: &GenerativeSLM, job: Job) {
    let (reply, result) = match job {
        Job::Score(reqs, reply) => (
            reply,
            guarded(|| {
                collect(&reqs, |(c, k)| {
                    model.score(c, k).map(
                        |s| json!({"logprob": s.logprob, "n_tokens": s.n_tokens, "greedy": s.greedy}),
                    )
                })
            }),
        ),
        Job::Rolling(texts, reply) => (
            reply,
            guarded(|| {
                collect(&texts, |t| {
                    model
                        .score_rolling(t)
                        .map(|s| json!({"logprob": s.logprob, "n_tokens": s.n_tokens}))
                })
            }),
        ),
        Job::Generate(reqs, reply) => (
            reply,
            guarded(|| {
                collect(&reqs, |g| {
                    let params = SamplingParams {
                        temperature: g.temperature,
                        top_k: g.top_k,
                        top_p: g.top_p,
                        repetition_penalty: 1.0,
                    };
                    model
                        .generate_tokens(
                            &g.prompt,
                            g.max_tokens,
                            &g.stop,
                            &params,
                            &mut Rng::new(g.seed),
                        )
                        .map(|o| json!({"text": o.text, "n_tokens": o.n_tokens}))
                })
            }),
        ),
        #[cfg(test)]
        Job::Panic(reply) => (reply, guarded(|| panic!("boom"))),
        #[cfg(test)]
        Job::Pool(fail, reply) => (
            reply,
            guarded(|| {
                let out = ferrum_core::parallel::run(512, 4, move |r0, r1, b| {
                    assert!(!fail, "kernel boom");
                    b[..(r1 - r0) * 4].fill(1.0);
                });
                Ok(vec![json!(out.iter().sum::<f32>())])
            }),
        ),
    };
    let _ = reply.send(result);
}

/// Load the model from `bytes` on a dedicated thread and answer jobs there
/// until every sender is dropped. The thread must build the model itself:
/// `GenerativeSLM` is not `Send`. Returns the job channel and the model's info.
pub(crate) type Spawned = (mpsc::Sender<Job>, Info, Weak<()>);

pub(crate) fn spawn_model_thread(bytes: Vec<u8>) -> Result<Spawned, String> {
    // ponytail: unbounded job queue; fine for a loopback server with one serial
    // client (LightEval). Use a bounded channel + 503 if exposed to many clients.
    let (tx, rx) = mpsc::channel::<Job>();
    let alive = Arc::new(());
    let weak = Arc::downgrade(&alive);
    let (ready_tx, ready_rx) = mpsc::channel::<Result<Info, String>>();
    std::thread::Builder::new()
        .name("ferrum-model".into())
        .spawn(move || {
            let _alive = alive;
            let model = match GenerativeSLM::from_bytes(&bytes) {
                Ok(m) => m,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(info_for(&model, &bytes)));
            drop(bytes);
            for job in rx {
                handle(&model, job);
            }
        })
        .map_err(|e| format!("cannot spawn model thread: {e}"))?;
    let info = ready_rx
        .recv()
        .map_err(|_| "model thread died while loading".to_string())??;
    Ok((tx, info, weak))
}

fn default_max_tokens() -> usize {
    64
}
fn default_top_p() -> f32 {
    1.0
}

#[derive(Deserialize)]
pub(crate) struct GenRequest {
    pub(crate) prompt: String,
    #[serde(default = "default_max_tokens")]
    pub(crate) max_tokens: usize,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::model;

    #[test]
    fn spawn_model_thread_rejects_bad_bytes_and_serves_good() {
        assert!(spawn_model_thread(b"nope".to_vec()).is_err());
        let m = model(0);
        let (tx, info, _alive) = spawn_model_thread(m.to_bytes_quantized().unwrap()).unwrap();
        assert_eq!(info.tokenizer, "char");
        let (rtx, rrx) = oneshot::channel();
        tx.send(Job::Score(vec![("abc".into(), " abc".into())], rtx))
            .unwrap();
        let r = rrx.blocking_recv().unwrap().unwrap();
        assert!(r[0]["logprob"].as_f64().unwrap() < 0.0);
    }

    #[test]
    fn collect_stops_at_first_error() {
        let ok = collect(&[1, 2], |n| Ok(json!(n))).unwrap();
        assert_eq!(ok, vec![json!(1), json!(2)]);
        let err = collect(&[1, 2], |n| {
            if *n == 2 {
                Err(ferrum_core::InferError::Format("bad".into()))
            } else {
                Ok(json!(n))
            }
        })
        .unwrap_err();
        assert!(!err.is_empty());
        assert!(collect::<u8>(&[], |_| unreachable!()).unwrap().is_empty());
    }

    #[test]
    fn default_helpers() {
        assert_eq!((default_max_tokens(), default_top_p()), (64, 1.0));
    }

    #[test]
    fn panic_message_variants() {
        assert_eq!(panic_message(&"lit"), "lit");
        assert_eq!(panic_message(&String::from("owned")), "owned");
        assert_eq!(panic_message(&42u8), "unknown panic");
    }

    #[test]
    fn guarded_passes_through_and_catches() {
        assert_eq!(guarded(|| Ok(vec![json!(1)])).unwrap(), vec![json!(1)]);
        assert_eq!(guarded(|| Err("e".into())).unwrap_err(), "e");
        let e = guarded(|| panic!("kaboom")).unwrap_err();
        assert_eq!(e, "internal error: model panicked: kaboom");
        let e = guarded(|| std::panic::panic_any(7u8)).unwrap_err();
        assert!(e.ends_with("unknown panic"));
    }

    #[test]
    fn health_alive_flag_drops_with_thread() {
        let m = model(0);
        let (tx, _info, alive) = spawn_model_thread(m.to_bytes_quantized().unwrap()).unwrap();
        assert!(alive.strong_count() > 0);
        drop(tx);
        while alive.strong_count() > 0 {
            std::thread::yield_now();
        }
    }
}
