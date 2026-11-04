//! Shared plumbing every command uses: structured errors, job slots and cancel, progress
//! events, path helpers, the HTTP client.

pub(crate) mod error;
pub(crate) mod http;
pub(crate) mod jobs;
pub(crate) mod paths;
pub(crate) mod progress;
