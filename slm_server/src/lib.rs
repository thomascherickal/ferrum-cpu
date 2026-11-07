//! HTTP inference server for Ferrum SLMs and imported Llama/Qwen GGUFs: loglikelihood,
//! rolling loglikelihood, and generation — the three primitives eval harnesses such as
//! LightEval need. One thread owns the model (it is not `Sync`); requests are
//! queued to it over a channel and answered through oneshot replies.
#![forbid(unsafe_code)]

mod auth;
mod backend;
mod cancel;
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

pub(crate) fn str_err<T, E: std::fmt::Display>(res: Result<T, E>) -> Result<T, String> {
    match res {
        Ok(v) => Ok(v),
        Err(e) => Err(e.to_string()),
    }
}

pub(crate) fn fmt_err<T, E: std::fmt::Display>(
    res: Result<T, E>,
    prefix: &str,
) -> Result<T, String> {
    match res {
        Ok(v) => Ok(v),
        Err(e) => Err(format!("{prefix}: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_err_and_fmt_err_work() {
        assert_eq!(str_err::<i32, &str>(Ok(1)), Ok(1));
        assert_eq!(str_err::<i32, &str>(Err("boom")), Err("boom".into()));
        assert_eq!(fmt_err::<i32, &str>(Ok(1), "p"), Ok(1));
        assert_eq!(
            fmt_err::<i32, &str>(Err("boom"), "p"),
            Err("p: boom".into())
        );
    }
}
