//! The JavaScript side builds `estimate_job` arguments (ui/core/live.js); this test
//! reads the file the Node test pins them to (ui/tests/live_params.fixture.json) and
//! checks them against the Rust types. serde silently ignores unknown fields,
//! so a misspelt key in the UI would otherwise just read as 0.

use ferrum_gui_lib::estimate::{JobKind, JobParams};
use ferrum_gui_lib::estimate_cmd::estimate_now;
use serde_json::Value;

const FIXTURE: &str = include_str!("../ui/tests/live_params.fixture.json");

#[test]
fn every_key_the_ui_sends_is_a_real_job_param() {
    let known: Vec<String> = serde_json::to_value(JobParams::default())
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let buttons = fixture.as_object().unwrap();
    assert!(buttons.len() >= 10, "fixture lost entries");
    for (id, entry) in buttons {
        let kind: JobKind = serde_json::from_value(entry["kind"].clone())
            .unwrap_or_else(|e| panic!("#{id}: kind {} is not a JobKind: {e}", entry["kind"]));
        for key in entry["params"].as_object().unwrap().keys() {
            assert!(
                known.contains(key),
                "#{id} sends '{key}', which JobParams does not have"
            );
        }
        // And the whole request goes through: typed parameters, a real estimate.
        let params: JobParams = serde_json::from_value(entry["params"].clone())
            .unwrap_or_else(|e| panic!("#{id}: params do not deserialize: {e}"));
        let est = estimate_now(kind, params, None);
        assert_eq!(est.kind, kind, "#{id}");
        assert!(!est.uses_gpu);
    }
}

#[test]
fn the_default_train_form_gets_a_real_time_estimate() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let params: JobParams = serde_json::from_value(fixture["trStart"]["params"].clone()).unwrap();
    // The page's default values: context 16, embed 32, 4 heads, 2 blocks, 100 epochs.
    assert_eq!(
        (params.context_len, params.embed_dim, params.epochs),
        (16, 32, 100)
    );
    let est = estimate_now(JobKind::Train, params, None);
    assert!(est.seconds.is_some_and(|s| s > 0.0), "{est:?}");
    assert!(est.peak_ram_mb.is_some());
}
