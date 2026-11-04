//! The one HTTP client every download uses.

/// HTTP client with bounded connect/read timeouts so a hung or slow host can
/// never block a download task indefinitely (G2).
pub(crate) fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout_read(std::time::Duration::from_secs(30))
        .timeout_write(std::time::Duration::from_secs(30))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_agent_builds() {
        // Building the timeout-bounded agent must not panic.
        let _ = http_agent();
    }
}
