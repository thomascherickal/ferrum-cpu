//! The DVC `train` stage must pin the data-parallel shard count, or `dvc.lock`
//! hashes only reproduce on a machine with the same core count.

use std::path::Path;

fn read(name: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    std::fs::read_to_string(root.join(name)).unwrap()
}

#[test]
fn train_stage_pins_threads_from_params() {
    assert!(read("dvc.yaml").contains("--threads ${train.threads}"));
    let params = read("params.yaml");
    let line = params
        .lines()
        .find(|l| l.trim_start().starts_with("threads:"))
        .expect("params.yaml defines train.threads");
    let n: usize = line
        .split(':')
        .nth(1)
        .unwrap()
        .split('#')
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(n >= 1, "0 would mean auto, which is not reproducible");
}
