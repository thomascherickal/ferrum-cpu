//! What the machine can do and what a job will cost: hardware probe, time/RAM estimates,
//! the pre-flight check and the Capable tab's report. Pure computation plus read-only probes.

pub(crate) mod capable;
pub mod estimate;
pub mod estimate_cmd;
pub mod hardware;
pub(crate) mod preflight;
