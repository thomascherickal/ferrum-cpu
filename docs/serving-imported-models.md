<p align="center"><img src="assets/logo.png" width="96" alt="Ferrum logo"></p>

# Serving an imported Llama/Qwen GGUF with `ferrum-serve`

`ferrum-serve` serves imported `llama` and `qwen2` GGUFs (SmolLM2, Qwen2.5, ...) on the
same five routes it uses for Ferrum's own models. The reference is
[Manual §9.7](../Manual.md#97-serving-ferrum-serve). Every command and output below
comes from a real run on this machine on 2026-11-03, with SmolLM2-135M (Q8_0, 145 MB).
The model path is shortened to `smollm2-135m-q8_0.gguf`, and the JSON is printed the
way the server sent it.

## 1. Start the server

```bash
ferrum-serve --model smollm2-135m-q8_0.gguf --quant int8 --port 8093
```

```text
ferrum-serve: smollm2-135m-q8_0.gguf (134515008 params, context 8192, vocab 49152) on http://127.0.0.1:8093
ferrum-serve: auth OFF (loopback only)
```

The server finds the model kind in `general.architecture`. `--quant` sets the
precision the weights are held at in memory: `int4` (the default, the same as
`run-gguf`), `int8` or `f32`. The file is already 8-bit, and int4 costs it a lot of
quality (see Status §5), so this run uses `int8`. Before it reads any weights the
server estimates the resident size. If that is more than 90 % of the available memory
it refuses to start. `--force` starts it anyway. Ferrum's own models ignore both flags.

```bash
curl -s http://127.0.0.1:8093/v1/info
```

```json
{"params":134515008,"context_len":8192,"vocab":49152,"quant":"int8","tokenizer":"bpe","model_sha256":"5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"}
```

`quant` is the precision the model computes at, the `--quant` it was loaded with.

## 2. Generate

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8093/v1/generate \
  -d '{"requests":[{"prompt":"The capital of France is","max_tokens":12,"stop":["."]}]}'
```

```json
{"results":[{"n_tokens":2,"text":" Paris"}]}
```

This is a raw completion, with no chat template. The stop string `.` is cut from the
text, but the token that completed it is counted, which makes `n_tokens` 2. EOS also
stops generation, and it is neither in the text nor counted. `temperature`, `top_k`,
`top_p`, `seed` and `repetition_penalty` work as they do for a Ferrum model. A prompt
that fills the 8,192-token context gets a `422`.

## 3. Score

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8093/v1/loglikelihood \
  -d '{"requests":[{"context":"The capital of France is","continuation":" Paris"},
                   {"context":"The capital of France is","continuation":" Berlin"}]}'
```

```json
{"results":[{"greedy":false,"logprob":-1.6240879556881536,"n_tokens":1},{"greedy":false,"logprob":-9.642947818969404,"n_tokens":1}]}
```

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8093/v1/loglikelihood_rolling \
  -d '{"texts":["The capital of France is Paris."]}'
```

```json
{"results":[{"logprob":-39.40575684989872,"n_tokens":7}]}
```

Scoring puts the file's BOS token first whenever the file defines one. SmolLM2 defines
one but sets `add_bos_token = false`. Because of that, all 7 tokens of the sentence are
scored, the first one given BOS. Generation follows `add_bos_token`, so the prompt
above got no BOS. The two routes therefore see different inputs: ` Paris` was
generated greedily from the prompt without BOS, and it scores `greedy: false` with BOS
in front. Texts longer than the context are scored in windows of the context length.
Each window starts half a window before its first unscored token.

Stop the server with Ctrl-C. The GUI's Serve tab runs the same server. It has no
`--quant` control yet, so according to the code it loads imported files at int4. That
was read from the code and not tried in this walk-through. The browser bindings
(`slm_wasm`) load Ferrum's own models only.
