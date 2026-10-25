#!/usr/bin/env bash
# Check that a Ferrum GGUF runs in llama.cpp, Ollama and LM Studio and that all three
# produce the same greedy continuation.
#
#   scripts/verify_runtimes.sh <model.gguf>
#
# Tool discovery (each: env var, then PATH, then the default location):
#   llama.cpp  FERRUM_LLAMA_BIN (dir holding llama-completion), ~/.local/opt/llama.cpp-b11521
#   Ollama     FERRUM_OLLAMA (the binary),                       ~/.local/bin/ollama
#   LM Studio  FERRUM_LMS (the `lms` binary),                    ~/.lmstudio/bin/lms
#
# A tool that is not found is skipped with a message, unless FERRUM_REQUIRE_RUNTIMES=1, which
# makes it a failure. FERRUM_SKIP_LMSTUDIO=1 skips LM Studio even then (CI has no desktop
# app; local pre-release runs must not set it). llama.cpp is the reference, so it is always
# needed. Exit status: 0 = every checked runtime matched llama.cpp, 1 = mismatch/missing
# required tool, 2 = usage.
#
# Env knobs: FERRUM_VERIFY_PROMPT (default "The cat sat"), FERRUM_VERIFY_N (default 16),
# FERRUM_VERIFY_OLLAMA_PORT (port for an Ollama we start, default 11545).
set -u
LC_ALL=C

model=${1:-}
[ -f "$model" ] || { echo "usage: $0 <model.gguf>" >&2; exit 2; }
model=$(realpath "$model")
prompt=${FERRUM_VERIFY_PROMPT:-The cat sat}
n=${FERRUM_VERIFY_N:-16}
require=${FERRUM_REQUIRE_RUNTIMES:-0}
failed=0
tmp=$(mktemp -d)
cleanups=()
cleanup() { local i; for ((i=${#cleanups[@]}-1; i>=0; i--)); do eval "${cleanups[i]}" >/dev/null 2>&1; done; rm -rf "$tmp"; }
trap cleanup EXIT

say()  { echo "verify_runtimes: $*"; }
fail() { say "FAIL $*"; failed=1; }
# $1 = what is missing. Skip, or fail when runtimes are required.
missing() { if [ "$require" = 1 ]; then fail "$1 not found (FERRUM_REQUIRE_RUNTIMES=1)"; else say "SKIP $1 not found"; fi; }

command -v python3 >/dev/null || { echo "python3 is required (JSON parsing)" >&2; exit 2; }
# json_field <key>: first string value of <key> in the JSON on stdin ("" if absent).
json_field() { python3 -c 'import json,sys
d=json.load(sys.stdin)
def find(o,k):
    if isinstance(o,dict):
        if k in o: return o[k]
        for v in o.values():
            r=find(v,k)
            if r is not None: return r
    elif isinstance(o,list):
        for v in o:
            r=find(v,k)
            if r is not None: return r
print(find(d,sys.argv[1]) or "",end="")' "$1"; }
answers() { (exec 3<>"/dev/tcp/${1%:*}/${1##*:}") 2>/dev/null; }
# trimmed: strip trailing whitespace/newlines (llama-completion appends some).
trimmed() { local s=$1; printf '%s' "${s%"${s##*[![:space:]]}"}"; }

# ---- llama.cpp (the reference) ----
llama_dir=""
for d in "${FERRUM_LLAMA_BIN:-}" "$(dirname "$(command -v llama-completion 2>/dev/null || echo /nonexistent/x)")" "$HOME/.local/opt/llama.cpp-b11521"; do
  [ -n "$d" ] && [ -x "$d/llama-completion" ] && { llama_dir=$d; break; }
done
[ -n "$llama_dir" ] || { missing "llama.cpp (FERRUM_LLAMA_BIN / PATH / ~/.local/opt/llama.cpp-b11521)"; exit "$failed"; }

# reference <tokens>: greedy continuation of the prompt (prompt echo removed), run a few times
# because llama.cpp's async logger can drop the tail of a tiny model's output.
reference() {
  local best="" out i
  for i in 1 2 3; do
    out=$("$llama_dir/llama-completion" -m "$model" -p "$prompt" -n "$1" --temp 0 -no-cnv --no-warmup </dev/null 2>/dev/null)
    out=${out#"$prompt"}
    [ "${#out}" -gt "${#best}" ] && best=$out
  done
  trimmed "${best//\[end of text\]/}"
}
want=$(reference "$n")
if [ -z "$(trimmed "$want")" ]; then fail "llama.cpp generated nothing for $model"; exit 1; fi
say "llama.cpp: $(printf '%q' "$want")"

# same <got> <want...>: got equals one of the candidates (after trimming).
same() { local g w; g=$(trimmed "$1"); shift; for w in "$@"; do [ "$g" = "$(trimmed "$w")" ] && return 0; done; return 1; }

# ---- Ollama ----
ollama=${FERRUM_OLLAMA:-$(command -v ollama 2>/dev/null || true)}
[ -x "${ollama:-/nonexistent}" ] || ollama=$HOME/.local/bin/ollama
if [ -x "$ollama" ]; then
  host=${OLLAMA_HOST:-127.0.0.1:11434}
  if ! answers "$host"; then
    host=127.0.0.1:${FERRUM_VERIFY_OLLAMA_PORT:-11545}
    answers "$host" && { fail "port $host is already in use by something that is not Ollama"; host=""; }
    if [ -n "$host" ]; then
      OLLAMA_HOST=$host OLLAMA_MODELS="$tmp/ollama-models" "$ollama" serve >"$tmp/ollama.log" 2>&1 &
      cleanups+=("kill $! ; wait $!")
      for _ in $(seq 100); do answers "$host" && break; sleep 0.1; done
      answers "$host" || { fail "ollama serve did not come up on $host (see below)"; cat "$tmp/ollama.log" >&2; host=""; }
    fi
  fi
  if [ -n "$host" ]; then
    name=ferrum-verify-$$
    printf 'FROM %s\n' "$model" > "$tmp/Modelfile"
    cleanups+=("OLLAMA_HOST=$host '$ollama' rm $name")
    if OLLAMA_HOST=$host "$ollama" create "$name" -f "$tmp/Modelfile" >"$tmp/create.log" 2>&1; then
      body=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":sys.argv[2],"raw":True,"stream":False,"options":{"temperature":0,"num_predict":int(sys.argv[3])}}))' "$name" "$prompt" "$n")
      got=$(curl -s -H 'Content-Type: application/json' "http://$host/api/generate" -d "$body" | json_field response)
      say "ollama:    $(printf '%q' "$got")"
      if [ -n "$(trimmed "$got")" ] && same "$got" "$want"; then say "PASS ollama"; else fail "ollama output differs from llama.cpp"; fi
    else
      fail "ollama create failed: $(cat "$tmp/create.log")"
    fi
  fi
else
  missing "ollama (FERRUM_OLLAMA / PATH / ~/.local/bin/ollama)"
fi

# ---- LM Studio ----
if [ "${FERRUM_SKIP_LMSTUDIO:-0}" = 1 ]; then
  say "SKIP lm-studio (FERRUM_SKIP_LMSTUDIO=1)"
else
  lms=${FERRUM_LMS:-$(command -v lms 2>/dev/null || true)}
  [ -x "${lms:-/nonexistent}" ] || lms=$HOME/.lmstudio/bin/lms
  if [ -x "$lms" ]; then
    repo=ferrum-test/verify$$
    file=$(basename "$model")
    if answers 127.0.0.1:1234; then
      fail "port 1234 is already in use by a server this script did not start; stop it and rerun"
    else
      cleanups+=("rm -rf '$HOME/.lmstudio/models/$repo'; rmdir '$HOME/.lmstudio/models/ferrum-test'")
      if "$lms" import -y -l --user-repo "$repo" "$model" >"$tmp/import.log" 2>&1; then
        # The desktop app indexes new models asynchronously: poll the listing.
        key=""
        for _ in $(seq 30); do
          key=$("$lms" ls --json 2>/dev/null | python3 -c 'import json,sys
for m in json.load(sys.stdin):
    if m.get("path","").endswith(sys.argv[1]): print(m["modelKey"]); break' "$repo/$file")
          [ -n "$key" ] && break
          sleep 0.5
        done
        if [ -z "$key" ]; then
          fail "lms ls --json never listed $repo/$file"
        else
          id=ferrum-verify-$$
          cleanups+=("'$lms' unload '$id'")
          if "$lms" load "$key" --gpu off --identifier "$id" -y >"$tmp/load.log" 2>&1 \
             && { cleanups+=("'$lms' server stop"); "$lms" server start -p 1234 >"$tmp/server.log" 2>&1; }; then
            want1=$(reference $((n + 1)))
            body=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":sys.argv[2],"max_tokens":int(sys.argv[3]),"temperature":0,"repeat_penalty":1.0}))' "$id" "$prompt" $((n + 1)))
            got=$(curl -s -H 'Content-Type: application/json' http://127.0.0.1:1234/v1/completions -d "$body" | json_field text)
            say "lm-studio: $(printf '%q' "$got")"
            # LM Studio engines disagree on whether max_tokens counts the first token (n or n+1 visible).
            if [ -n "$(trimmed "$got")" ] && same "$got" "$want" "$want1"; then say "PASS lm-studio"; else fail "lm-studio output differs from llama.cpp"; fi
          else
            fail "lms load/server start failed: $(cat "$tmp/load.log" "$tmp/server.log" 2>/dev/null)"
          fi
        fi
      else
        fail "lms import failed: $(cat "$tmp/import.log")"
      fi
    fi
  else
    missing "lms (FERRUM_LMS / PATH / ~/.lmstudio/bin/lms)"
  fi
fi

[ "$failed" = 0 ] && say "OK" || say "FAILED"
exit "$failed"
