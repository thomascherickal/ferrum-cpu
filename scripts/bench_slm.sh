#!/usr/bin/env bash
# Benchmark the 507,552-parameter reference SLM: training throughput, eval
# time, and generation speed. Usage: scripts/bench_slm.sh <corpus.txt> [tokens]
set -euo pipefail
corpus=${1:?usage: scripts/bench_slm.sh <corpus.txt> [tokens]}
# Resolve before the cd below, so a path relative to the caller's directory works.
[[ $corpus = /* ]] || corpus=$PWD/$corpus
[[ -r $corpus ]] || { echo "bench_slm.sh: cannot read corpus $corpus" >&2; exit 1; }
cd "$(dirname "$0")/.."
tokens=${2:-2e5}
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
cargo build --release -q -p slm_cli
bin=./target/release/train_transformer
cfg=(--context 128 --embed 96 --heads 4 --blocks 3 --hidden 320 --vocab 1024)
# Document separators are stripped so generation never stops early on EOS.
strip() { sed 's/<|endoftext|>//g'; }
head -c 300000 "$corpus" | strip > "$out/train.txt"
tail -c 30000 "$corpus" | strip > "$out/heldout.txt"
now() { date +%s.%N; }
echo "== threads: ${FERRUM_NUM_THREADS:-auto}"
s=$(now)
$bin train "$out/train.txt" "$out/m.gguf" "${cfg[@]}" --tokens "$tokens" --lr 5e-3 \
  --cosine --warmup 30 --tokenizer "$out/tok.gguf" --force >/dev/null
e=$(now)
awk -v t="$tokens" -v s="$s" -v e="$e" 'BEGIN{printf "train: %.0f tokens/s (%.1f s)\n", t/(e-s), e-s}'
s=$(now)
$bin eval "$out/m.gguf" "$out/heldout.txt" | grep -E "Predictions|Perplexity"
e=$(now)
awk -v s="$s" -v e="$e" 'BEGIN{printf "eval: %.2f s\n", e-s}'
s=$(now)
bytes=$($bin generate "$out/m.gguf" "Once upon a time" --max-tokens 4096 --gen-seed 1 --temp 0.8 | wc -c)
e=$(now)
# Includes process start + model load; `bytes` shows whether EOS cut the run short.
awk -v s="$s" -v e="$e" -v b="$bytes" 'BEGIN{printf "generate: %.3f ms/token (4096 max tokens, %d bytes out)\n", (e-s)*1000/4096, b}'
