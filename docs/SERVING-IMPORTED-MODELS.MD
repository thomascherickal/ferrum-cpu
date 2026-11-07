<p align="center"><img src="assets/logo.png" width="96" alt="Ferrum logo"></p>

# Serving an imported Llama/Qwen GGUF with `ferrum-serve`

`ferrum-serve` serves imported `llama` and `qwen2` GGUFs (SmolLM2, Qwen2.5, ...) on the
same five routes it uses for Ferrum's own models. The reference is
[Manual §9.7](../MANUAL.MD#97-serving-ferrum-serve). Every command and output below
comes from a real run on this machine on 2026-11-03, with SmolLM2-135M (Q8_0, 145 MB).
The model path is shortened to `smollm2-135m-q8_0.gguf`, and the JSON is printed the
way the server sent it.

## 1. Start the server

```bash
ferrum-serve --model smollm2-135m-q8_0.gguf --port 8093
```

```text
ferrum-serve: smollm2-135m-q8_0.gguf (134515008 params, context 8192, vocab 49152) on http://127.0.0.1:8093
ferrum-serve: auth OFF (loopback only)
```

The server finds the model kind in `general.architecture`. `--quant` sets the
precision the weights are held at in memory: `int8` (the default), `int4` or `f32`.
The file is already 8-bit, so `int8` costs it little, while `int4` costs it a lot of
quality (see Status §5). Before it reads any weights the
server estimates the resident size. If that is more than 90 % of the available memory
it refuses to start. `--force` starts it anyway. Ferrum's own models ignore both flags.

```bash
curl -s http://127.0.0.1:8093/v1/info
```

```json
{"params":134515008,"context_len":8192,"vocab":49152,"quant":"int8","tokenizer":"bpe","model_sha256":"5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"}
```

`quant` is the precision the model computes at: here the default, `int8`.

## 2. Generate

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8093/v1/generate \
  -d '{"requests":[{"prompt":"The capital of France is","max_tokens":12,"stop":["."]}]}'
```

```json
{"results":[{"n_tokens":2,"text":" Paris"}]}
```

This is a raw completion, with no chat template. Add `"chat": true` to a request to wrap
the prompt in the file's chat template the way `run-gguf` does (Manual §9.7); that was
tested on a synthetic tiny model only and was not run on SmolLM2 for this walk-through. The stop string `.` is cut from the
text, but the token that completed it is counted, which makes `n_tokens` 2. EOS also
stops generation, and it is neither in the text nor counted. `temperature`, `top_k`,
`top_p`, `seed` and `repetition_penalty` work as they do for a Ferrum model. A prompt
too long for the 8,192-token context is cut from the left (its most recent tokens are
kept, leaving room for `max_tokens`), so it never fails the batch.

## 3. Score

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8093/v1/loglikelihood \
  -d '{"requests":[{"context":"The capital of France is","continuation":" Paris"},
                   {"context":"The capital of France is","continuation":" Berlin"}]}'
```

```json
{"results":[{"greedy":true,"logprob":-0.5385536647130459,"n_tokens":1},{"greedy":false,"logprob":-9.166077086954257,"n_tokens":1}]}
```

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8093/v1/loglikelihood_rolling \
  -d '{"texts":["The capital of France is Paris."]}'
```

```json
{"results":[{"logprob":-18.072027314095283,"n_tokens":6}]}
```

Scoring follows `add_bos_token` exactly as generation does: the file's BOS token goes
first only when the file asks for it and defines one. SmolLM2 defines a BOS id but sets
`add_bos_token = false`, so no BOS is added. The first token of the sentence therefore has
nothing to condition on and is not scored, which is why 6 of the 7 tokens count. A
loglikelihood request with an empty `context` has the same problem, and the server
answers it with a `422` instead of guessing:

```bash
curl -s -w ' [%{http_code}]' -H 'Content-Type: application/json' http://127.0.0.1:8093/v1/loglikelihood \
  -d '{"requests":[{"context":"","continuation":"Paris"}]}'
```

```text
{"error":"dimension mismatch: context is empty and the model prepends no BOS token"} [422]
```

Texts longer than the scoring window are scored in overlapping windows. The window is the
model's context length, capped at 512 tokens like `eval-gguf`. Each window starts half
a window before its first unscored token, so the continuation always sees the most
recent tokens.

Stop the server with Ctrl-C. The GUI's Serve tab runs the same server. It has no
`--quant` control, so it loads imported files at the default, int8. That was read from
the code and not tried in the app. The browser bindings
(`slm_wasm`) load Ferrum's own models only.
