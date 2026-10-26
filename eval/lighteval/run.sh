#!/usr/bin/env bash
# Evaluate a Ferrum SLM with LightEval 0.13.0 through ferrum-serve.
# Usage: eval/lighteval/run.sh --model M --test T --out O [--max-samples N] [--wikitext] [--port P]
# Env: FERRUM_LIGHTEVAL_VENV  venv holding python + lighteval (default: eval/lighteval/.venv)
#      FERRUM_SERVE_WAIT      seconds to wait for the server to load the model and answer /health (default 60)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
model= test= out= max= port=8765 wikitext=0
isint() { case $1 in ''|*[!0-9]*) return 1;; esac; }
need() { [ "$#" -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case $1 in
    --model) need "$@"; model=$2; shift 2;;
    --test) need "$@"; test=$2; shift 2;;
    --out) need "$@"; out=$2; shift 2;;
    --max-samples) need "$@"; max=$2; shift 2;;
    --port) need "$@"; port=$2; shift 2;;
    --wikitext) wikitext=1; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
: "${model:?--model required}" "${test:?--test required}" "${out:?--out required}"
# Fail before the long run, not after it (the summary step used to int() a bad --max-samples at the very end).
{ [ -z "$max" ] || { isint "$max" && [ "$max" -ge 1 ]; }; } || { echo "--max-samples must be a positive integer" >&2; exit 2; }
isint "$port" || { echo "--port must be an integer" >&2; exit 2; }
wait_secs=${FERRUM_SERVE_WAIT:-60}
isint "$wait_secs" || { echo "FERRUM_SERVE_WAIT must be an integer number of seconds" >&2; exit 2; }
venv="${FERRUM_LIGHTEVAL_VENV:-$here/.venv}"
py="$venv/bin/python"; le="$venv/bin/lighteval"
[ -x "$le" ] || { echo "run: python3 -m venv $venv && $venv/bin/pip install -r $here/requirements.txt" >&2; exit 1; }

work="$here/out"; rm -rf "$work"; mkdir -p "$work"
"$py" "$here/make_task_data.py" "$test" "$work/data"

(cd "$root" && cargo build --release -q -p slm_server)

# The server and lighteval must never outlive this script, whatever the exit path
# (success, set -e failure, health timeout, INT/TERM): kill both and reap them.
srv= lpid=
cleanup() {
  [ -z "$lpid" ] || kill "$lpid" 2>/dev/null || true
  [ -z "$srv" ] || kill "$srv" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${CARGO_TARGET_DIR:-$root/target}/release/ferrum-serve" --model "$model" --port "$port" 2>"$work/server.log" &
srv=$!
# The server prints "... on http://HOST:PORT" only after it bound its port, so a busy port
# (some other server answering /health) can never be mistaken for ours.
url=
for _ in $(seq 1 $((wait_secs * 10))); do
  kill -0 "$srv" 2>/dev/null || break
  url=$(sed -n 's/.* on \(http:\/\/[^ ]*\)$/\1/p' "$work/server.log" | head -n 1)
  [ -n "$url" ] && curl -sf "$url/health" >/dev/null && break
  url=
  sleep 0.1
done
[ -n "$url" ] || { echo "ferrum-serve did not become healthy within ${wait_secs}s (or exited); server log:" >&2; cat "$work/server.log" >&2; exit 1; }

tasks="ferrum_heldout|0,tinystories_cloze|0,blimp:anaphor_gender_agreement|0,blimp:anaphor_number_agreement|0,blimp:regular_plural_subject_verb_agreement_1|0,arc:easy|0"
[ "$wikitext" = 1 ] && tasks="$tasks,wikitext:103:document_level|0"

# ${max:+...} is unquoted on purpose: max is a validated integer, and (unlike an empty
# "${extra[@]}") it is safe under `set -u` on bash < 4.4.
# Hugging Face downloads (datasets, hub) stay inside the project, never in ~/.cache. The
# directory survives between runs (unlike $work) so the datasets are fetched once.
hf="${FERRUM_HF_CACHE:-$here/.hf-cache}"; mkdir -p "$hf"
FERRUM_URL="$url" FERRUM_TASK_DATA="$work/data" FERRUM_CACHE_DIR="$work/cache" \
  HF_HOME="$hf" HF_DATASETS_CACHE="$hf/datasets" HF_HUB_CACHE="$hf/hub" \
  "$le" custom ferrum "$here/ferrum_model.py" "$tasks" \
  --custom-tasks "$here/ferrum_tasks.py" --output-dir "$work/results" ${max:+--max-samples $max} &
lpid=$!
wait "$lpid"
lpid=

"$py" "$here/summarize.py" "$work/results" "$out" "$model" ${max:+$max}
