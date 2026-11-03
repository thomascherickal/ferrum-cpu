//! HTTP inference server for Ferrum SLMs and imported Llama/Qwen GGUFs: loglikelihood,
//! rolling loglikelihood, and generation — the three primitives eval harnesses such as
//! LightEval need. One thread owns the model (it is not `Sync`); requests are
//! queued to it over a channel and answered through oneshot replies.
#![forbid(unsafe_code)]

mod auth;
mod backend;
pub mod cli;
mod http;
mod imported;
mod info;
mod limits;
mod load;
mod model_thread;
mod serve;
#[cfg(test)]
mod test_util;

pub use auth::with_bearer_token;
pub use http::{app, app_from_file};
pub use info::{info_for, Info};
pub use limits::Limits;
pub use serve::{bind, Bound, RequestLog};
