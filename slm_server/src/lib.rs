//! HTTP inference server for Ferrum SLMs: loglikelihood, rolling
//! loglikelihood, and generation — the three primitives eval harnesses such as
//! LightEval need. One thread owns the model (it is not `Sync`); requests are
//! queued to it over a channel and answered through oneshot replies.
#![forbid(unsafe_code)]

pub mod cli;
mod http;
mod info;
mod limits;
mod model_thread;
#[cfg(test)]
mod test_util;

pub use http::app;
pub use info::{info_for, Info};
pub use limits::Limits;
