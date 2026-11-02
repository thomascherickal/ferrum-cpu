#!/usr/bin/env bash
# Line coverage for Rust (workspace + ferrum_gui), JS (Node) and Python (coverage.py).
# usage: scripts/coverage.sh [--rust|--gui|--js|--py|--all] [--fail-under N]   (default: --rust, N=100)
# Reports go to target/coverage/ (override: FERRUM_COVERAGE_OUT). Needs cargo-llvm-cov, node >= 22;
# --py needs `python3 -m venv .venv-cov && .venv-cov/bin/pip install -r requirements-coverage.txt`.
set -euo pipefail

# 15 GB machine: low-memory build, always --release for the workspace, never beside another cargo.
export CARGO_BUILD_JOBS=3 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
# Named in the plan, "What 100 % coverage means here": Tauri entry point (the CI boot smoke test runs it).
EXCLUDE='ferrum_gui/src/main\.rs'
# ferrum_gui pulls the workspace crates in by path; they are measured by --rust, not here.
GUI_EXCLUDE="$EXCLUDE|/(ferrum_core|slm_cli|slm_server)/src/"

usage() { echo "usage: scripts/coverage.sh [--rust|--gui|--js|--py|--all] [--fail-under N]" >&2; exit 2; }

modes=() N=100
while [ $# -gt 0 ]; do
  case "$1" in
    --rust|--gui|--js|--py|--all) modes+=("$1") ;;
    --fail-under) [ $# -ge 2 ] && [[ "$2" =~ ^[0-9]+$ ]] || usage; N=$2; shift ;;
    *) usage ;;
  esac
  shift
done
[ ${#modes[@]} -gt 0 ] || modes=(--rust)
[[ " ${modes[*]} " == *" --all "* ]] && modes=(--rust --gui --js --py)

cd "$(dirname "$0")/.."
if [ -n "$(pgrep -x cargo || true)" ]; then
  echo "another cargo process is running; llvm-cov must not share the machine with a build or test" >&2
  exit 1
fi
OUT=${FERRUM_COVERAGE_OUT:-$PWD/target/coverage}
mkdir -p "$OUT"

rust() {
  cargo llvm-cov --workspace --release --ignore-filename-regex "$EXCLUDE" \
    --fail-under-lines "$N" --summary-only | tee "$OUT/summary.txt"
  cargo llvm-cov report --release --ignore-filename-regex "$EXCLUDE" --show-missing-lines > "$OUT/missing.txt"
}

gui() {
  # No --release: the crate's dev profile already optimizes the engine.
  (cd ferrum_gui
   cargo llvm-cov --ignore-filename-regex "$GUI_EXCLUDE" --fail-under-lines "$N" --summary-only | tee "$OUT/gui-summary.txt")
}

# Node lists only the files a test loaded, so a source file nobody loads would show nothing, not 0 %:
# fail when any non-test .js in the directory is missing from the report.
js_dir() { # dir
  local dir=$1 rc=0 rep="$OUT/js.$(echo "$1" | tr / _).txt"
  (cd "$dir" && node --test --experimental-test-coverage --test-coverage-lines="$N" \
    --test-coverage-exclude='**/*.test.js' ./*.test.js) > "$rep" 2>&1 || rc=$?
  cat "$rep" >> "$OUT/js.txt"
  local f missing=0
  for f in "$dir"/*.js; do
    case "$f" in *.test.js) continue ;; esac
    grep -q "$(basename "$f")" "$rep" || { echo "not in the coverage report: $f" >&2; missing=1; }
  done
  [ "$rc" -eq 0 ] && [ "$missing" -eq 0 ]
}
js() {
  : > "$OUT/js.txt"
  local rc=0
  # ponytail: demo.js/worker.js are read as text by their tests, never run, so the demo gate cannot pass
  # until Task 1.6 loads them; CI sets FERRUM_JS_DIRS=ferrum_gui/ui until then.
  local d
  for d in ${FERRUM_JS_DIRS:-ferrum_gui/ui slm_wasm/demo}; do js_dir "$d" || rc=1; done
  tail -n 40 "$OUT/js.txt" >&2 || true
  [ "$rc" -eq 0 ]
}

py() {
  local cov=coverage
  [ -x .venv-cov/bin/coverage ] && cov=.venv-cov/bin/coverage
  export COVERAGE_FILE="$OUT/.coverage"
  # Code that needs LightEval is skipped without it: reported, not gated by a different rule.
  "$cov" run --source=eval/lighteval --omit='*/test_*.py' -m unittest discover -s eval/lighteval -p 'test_*.py'
  "$cov" report --show-missing --fail-under="$N" | tee "$OUT/py.txt"
}

for m in "${modes[@]}"; do
  case "$m" in --rust) rust ;; --gui) gui ;; --js) js ;; --py) py ;; esac
done
