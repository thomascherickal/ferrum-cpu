//! Request limits and the fixed server-side caps.

/// Most POST requests the server holds at once (running, queued, or still being
/// read); more get an immediate 503 before their body is read, so a flood
/// cannot pile up parsed bodies or queued jobs in memory.
// ponytail: fixed; make it a `Limits` field if a multi-client deployment needs more.
pub(crate) const MAX_INFLIGHT: usize = 32;
/// Most stop strings per generate request, and the longest one (bytes). Every
/// generated token rescans the tail for each stop, so unbounded stops are a
/// CPU-exhaustion lever.
pub(crate) const MAX_STOPS: usize = 16;
pub(crate) const MAX_STOP_BYTES: usize = 1024;

/// Request limits, enforced before any model compute.
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    pub max_batch: usize,
    pub max_chars: usize,
    pub max_body: usize,
    /// Cap on `max_tokens` per generate request.
    pub max_tokens: usize,
    /// Repetition penalty for a generate request that omits the field (`--repetition-penalty`).
    pub repetition_penalty: f32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_batch: 64,
            max_chars: 65_536,
            max_body: 8 << 20,
            max_tokens: 4096,
            repetition_penalty: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_default_values() {
        let d = Limits::default();
        assert_eq!(
            (d.max_batch, d.max_chars, d.max_body, d.max_tokens),
            (64, 65_536, 8 << 20, 4096)
        );
        assert_eq!(d.repetition_penalty, 1.0);
    }
}
