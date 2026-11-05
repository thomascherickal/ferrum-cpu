//! What the machine can do and what a job will cost: hardware probe, time/RAM estimates,
//! the pre-flight check and the Capable tab's report. Pure computation plus read-only probes.

pub(crate) mod capable;
pub mod estimate;
pub mod estimate_cmd;
pub mod hardware;
pub(crate) mod preflight;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_group_exposes_probe_estimates_preflight_and_report() {
        assert_eq!(hardware::parse_hex_id("0x10de"), Some(0x10de));
        let _: fn(estimate::JobKind, &mut estimate::JobParams) = estimate_cmd::resolve;
        for name in [
            std::any::type_name::<preflight::Preflight>(),
            std::any::type_name::<hardware::HardwareInfo>(),
        ] {
            assert!(name.starts_with("ferrum_gui_lib::sizing::"), "{name}");
        }
        const { assert!(capable::BPP_INT4 < capable::BPP_INT8 && capable::BPP_INT8 < capable::BPP_F32) };
    }

    /// Dependencies point `commands` -> `sizing` -> `infra` (CLAUDE.md, Layout).
    #[test]
    fn the_group_depends_only_on_infra() {
        let hits = crate::test_support::sources_using("sizing", &["commands", "data", "serving"]);
        assert!(hits.is_empty(), "{hits:?}");
    }
}
