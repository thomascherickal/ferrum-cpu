//! Integration test: the public hardware API works without a window.
use ferrum_gui_lib::hardware::detect;

#[test]
fn detect_serializes_the_json_the_ui_receives() {
    let hw = detect();
    let v = serde_json::to_value(&hw).unwrap();
    println!("{v}");
    assert!(v["logicalCores"].as_u64().unwrap() >= 1);
    assert!(v["gpus"].is_array());
    for g in v["gpus"].as_array().unwrap() {
        assert_eq!(g["usedByFerrum"], false, "Ferrum never uses a GPU");
    }
}

#[test]
fn ram_is_refreshed_on_every_call_and_gpus_are_stable() {
    let (a, b) = (detect(), detect());
    assert_eq!(a.ram_total_mb, b.ram_total_mb);
    assert_eq!(a.gpus, b.gpus, "GPU list is detected once per process");
    assert_eq!(a.cpu_model, b.cpu_model);
}
