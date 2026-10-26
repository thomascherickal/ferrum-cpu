#!/usr/bin/env bash
# scripts/build_wasm.sh
# Compile slm_wasm (browser inference for transformer SLMs) to WebAssembly and
# generate JS bindings in slm_wasm/pkg/, and copy them next to the browser demo
# (slm_wasm/demo/pkg/), so the demo folder can be served as it is.
# Usage: bash scripts/build_wasm.sh
#        python3 -m http.server -d slm_wasm/demo 8000   # then open http://localhost:8000/
set -euo pipefail
cd "$(dirname "$0")/.."

need() { command -v "$1" &>/dev/null || { echo "ERROR: $1 not found. Install with: cargo install $1 --version 0.2.122"; exit 1; }; }
need wasm-bindgen
command -v rustup &>/dev/null && rustup target add wasm32-unknown-unknown &>/dev/null

echo "=== Compiling slm_wasm to WASM ==="
cargo build -p slm_wasm --target wasm32-unknown-unknown --release

echo "=== Generating JS bindings ==="
mkdir -p slm_wasm/pkg
wasm-bindgen \
  target/wasm32-unknown-unknown/release/slm_wasm.wasm \
  --out-dir slm_wasm/pkg \
  --target web \
  --no-typescript

echo "=== Copying pkg/ next to the demo ==="
rm -rf slm_wasm/demo/pkg
cp -r slm_wasm/pkg slm_wasm/demo/pkg

echo ""
echo "WASM binary : $(du -sh slm_wasm/pkg/slm_wasm_bg.wasm | cut -f1)"
echo "JS glue     : $(du -sh slm_wasm/pkg/slm_wasm.js      | cut -f1)"
echo "Demo        : slm_wasm/demo/ (serve it over http; a module worker does not load from file://)"
