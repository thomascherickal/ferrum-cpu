//! `ferrum-serve --model <path> [--host 127.0.0.1] [--port 8080]
//!   [--max-batch 64] [--max-chars 65536] [--max-body 8388608] [--max-tokens 4096]
//!   [--token <t>] [--allow-unauthenticated]` (token also from `$FERRUM_TOKEN`)
#![forbid(unsafe_code)]

use slm_server::cli::start;

// Thin process wrapper: all logic lives in (tested) `slm_server::cli`.
#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = start(&args, async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await;
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `main` maps `start`'s result to the exit status: these pin the three outcomes it relies on.
#[cfg(test)]
mod tests {
    use super::start;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[tokio::test]
    async fn help_and_version_succeed_and_bad_arguments_fail() {
        assert_eq!(start(&args(&["--version"]), async {}).await, Ok(()));
        assert_eq!(start(&args(&["--help"]), async {}).await, Ok(()));
        let e = start(&args(&["--bogus"]), async {}).await.unwrap_err();
        assert!(e.contains("--bogus"), "{e}");
        let e = start(&args(&["--model", "/nonexistent/ferrum/m.gguf"]), async {})
            .await
            .unwrap_err();
        assert!(e.contains("/nonexistent/ferrum/m.gguf"), "{e}");
    }
}
