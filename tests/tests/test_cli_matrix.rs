//! The functional-matrix guard: every `train_transformer` subcommand is driven through the real
//! binary by some `slm_cli/tests/*.rs`, and every `ferrum-serve` route by the HTTP end-to-end
//! test. A new subcommand or route cannot land without one.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// The text of every `slm_cli/tests/*.rs`, joined.
fn cli_tests() -> String {
    let dir = root().join("slm_cli/tests");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    files.sort();
    files
        .iter()
        .map(|p| std::fs::read_to_string(p).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The first double-quoted string of `line`, if any.
fn first_quoted(line: &str) -> Option<&str> {
    let start = line.find('"')? + 1;
    let len = line[start..].find('"')?;
    Some(&line[start..start + len])
}

/// The canonical subcommand names: the first quoted string of each match arm of `spec_for`.
fn subcommands() -> Vec<String> {
    let src = read("slm_cli/src/args.rs");
    let body = src
        .split("pub(crate) fn spec_for")
        .nth(1)
        .and_then(|rest| rest.split("_ => return None").next())
        .expect("spec_for in slm_cli/src/args.rs");
    body.lines()
        .filter(|l| l.contains("=>"))
        .filter_map(first_quoted)
        .map(String::from)
        .collect()
}

/// Whether `tests` passes `name` to the binary as a subcommand: the first element of an argument
/// array (`["name"`, `&["name"`, `vec!["name"`) or `.arg("name")`. A quoted mention elsewhere (an
/// assertion, a later argument) does not count.
fn invokes(tests: &str, name: &str) -> bool {
    tests.contains(&format!("[\"{name}\"")) || tests.contains(&format!(".arg(\"{name}\")"))
}

#[test]
fn the_invocation_matcher_wants_a_subcommand_position_and_whole_names() {
    for hit in [
        "tt(&[\"eval\", m])",
        "vec![\"eval\"]",
        "Command::new(B).arg(\"eval\")",
    ] {
        assert!(invokes(hit, "eval"), "{hit}");
    }
    for miss in [
        "assert!(out.contains(\"eval\"))",
        "tt(&[\"run\", \"eval\"])",
        "tt(&[\"eval-gguf\", m])",
        "tt(&[\"evaluate\"])",
        "// eval",
    ] {
        assert!(!invokes(miss, "eval"), "{miss}");
    }
}

#[test]
fn every_subcommand_has_a_binary_test() {
    let names = subcommands();
    assert_eq!(names.len(), 13, "spec_for lists 13 subcommands: {names:?}");
    let tests = cli_tests();
    for name in &names {
        assert!(
            invokes(&tests, name),
            "no slm_cli/tests/*.rs passes \"{name}\" to the binary; add a test that runs it"
        );
    }
}

/// The paths of the `.route("…"` calls in `slm_server/src/http.rs` outside its test module.
fn routes() -> Vec<String> {
    let src = read("slm_server/src/http.rs");
    let code = src.split("#[cfg(test)]").next().unwrap();
    code.split(".route(")
        .skip(1)
        .map(str::trim_start)
        .filter(|after| after.starts_with('"'))
        .filter_map(first_quoted)
        .map(String::from)
        .collect()
}

#[test]
fn every_route_has_an_http_e2e_test() {
    let paths = routes();
    assert_eq!(paths.len(), 5, "http.rs routes: {paths:?}");
    let e2e = read("slm_server/tests/tiny_http_e2e.rs");
    for path in &paths {
        assert!(
            e2e.contains(&format!("\"{path}\"")),
            "slm_server/tests/tiny_http_e2e.rs never requests {path}; add a test that does"
        );
    }
}
