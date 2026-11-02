//! Streaming decode loops: the shared driver plus the char-level and BPE (KV-cached / full-forward) paths.
use crate::error::Result;
use crate::rng::Rng;
use crate::slm::cached::CachedTransformer;
use crate::slm::corpus::{char_to_hex, hex_to_char};
use crate::slm::generate::drain_utf8;
use crate::slm::sampling::{sample_with_params, SamplingParams};
use crate::slm::GenerativeSLM;
use crate::tensor::Tensor;
use crate::tokenizer::ByteBpeTokenizer;
use crate::verbose;

impl GenerativeSLM {
    /// Core streaming generation shared by the plain and stop-criterion APIs.
    /// When `stop` is `Some`, generation halts once the stop string appears in
    /// the continuation (I3).
    pub(super) fn generate_stream_core(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        stop: Option<&str>,
        rng: &mut Rng,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<String> {
        vprintln!("[slm::GenerativeSLM::generate_stream] seed=\"{}\", num_chars={}, temp={:.2}, top_k={}, top_p={:.2}, rep={:.2}, stop={:?}",
            seed.chars().take(50).collect::<String>(), num_chars,
            params.temperature, params.top_k, params.top_p, params.repetition_penalty, stop);

        if let Some(tok) = self.bpe()? {
            return self.generate_bpe_stream(&tok, seed, num_chars, params, stop, rng, on_text);
        }

        self.require_token_id_model()?;
        // A seed char outside the vocabulary used to map silently to token 0;
        // name it instead (scoring has always errored the same way). CR stays
        // tolerated, as `encode_text` drops it.
        self.encode_text(seed)?;
        let context_len = self.meta.input_dim;
        vprintln!(
            "[slm::generate] vocab_size={}, context_len={}",
            self.meta.output_dim,
            context_len
        );

        // Fast path: genuine transformer models generate token-at-a-time with a
        // per-block KV cache (O(context) per token). Token-ID models without
        // transformer blocks fall through to the full-forward loop.
        if let Some(cached) = CachedTransformer::try_new(&self.model) {
            return self
                .generate_char_cached_stream(seed, num_chars, params, stop, rng, on_text, cached);
        }

        let mut generated = seed.to_string();
        let seed_bytes = seed.len();

        // Map each class name (a char's hex) → its index once, so per-step token
        // lookup is O(1) instead of an O(vocab) linear scan per character.
        let hex_to_idx: std::collections::HashMap<&str, usize> = self
            .meta
            .class_names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let char_idx = |ch: char| -> usize {
            hex_to_idx
                .get(char_to_hex(ch).as_str())
                .copied()
                .unwrap_or(0)
        };

        for step in 0..num_chars {
            let current_len = generated.chars().count();
            if current_len == 0 {
                break; // nothing to condition on
            }
            // Short seeds are left-padded with token 0 (as the BPE path does).
            let pad = context_len.saturating_sub(current_len);
            let context_chars: Vec<char> = generated
                .chars()
                .skip(current_len.saturating_sub(context_len))
                .collect();

            vprintln!(
                "[slm::generate] Step {}: context=\"{}\"",
                step,
                context_chars.iter().collect::<String>()
            );

            // Token-ID input → [T, vocab] probabilities; keep the last row.
            let next_dist: Vec<f32> = {
                let ids: Vec<f32> = std::iter::repeat_n(0.0, pad)
                    .chain(context_chars.iter().map(|&ch| char_idx(ch) as f32))
                    .collect();
                let input = Tensor::matrix(1, context_len, ids)?;
                let out = self.model.forward(&input)?;
                let (rows, cols) = out.matrix_dims()?;
                // The model ends in Softmax; convert probabilities back to
                // log-space so temperature scaling behaves correctly.
                out.data[(rows - 1) * cols..]
                    .iter()
                    .map(|&p| p.max(1e-12).ln())
                    .collect()
            };

            if verbose::is_verbose() {
                let (lmin, lmax, lmean) = verbose::stats(&next_dist);
                vprintln!(
                    "[slm::generate] Step {}: logits stats: min={:.4}, max={:.4}, mean={:.4}",
                    step,
                    lmin,
                    lmax,
                    lmean
                );
            }

            // Sample with the full decoding controls; the repetition penalty
            // sees the tokens currently in the context window.
            let recent: Vec<usize> = context_chars.iter().map(|&ch| char_idx(ch)).collect();
            let next_idx = sample_with_params(&next_dist, params, &recent, rng);

            // Decode prediction
            let predicted_hex = &self.meta.class_names[next_idx];
            let next_char = hex_to_char(predicted_hex);

            vprintln!(
                "[slm::generate] Step {}: sampled idx={}, hex=\"{}\", char='{}'",
                step,
                next_idx,
                predicted_hex,
                next_char
            );

            generated.push(next_char);
            // Stop early at a natural boundary (I3): the newest char may have
            // completed the stop string in the continuation.
            let hit_stop = stop.is_some_and(|s| generated[seed_bytes..].ends_with(s));
            // Stream exactly the new character (never the seed).
            let mut buf = [0u8; 4];
            on_text(next_char.encode_utf8(&mut buf));
            if hit_stop {
                break;
            }
        }

        vprintln!("[slm::generate] Generated {} total chars", generated.len());
        Ok(generated)
    }

    /// KV-cached char-level generation for genuine transformer models (I1).
    ///
    /// Maintains a rolling window of the most recent ≤ `context_len` token IDs
    /// in a per-block [`KvCache`] and samples one character per step. While the
    /// window is below capacity each step is a single O(context) cached `feed`.
    /// When it overflows, the cache is re-primed from the newest half window
    /// (learned absolute positions rule out sliding the cache by one), so the
    /// cost is one re-prime per `context_len / 2` characters: linear in the
    /// number of generated characters. Output within the first `context_len`
    /// characters matches the old full-forward path bit for bit; past that the
    /// model sees a half-window of history right after each re-prime.
    #[allow(clippy::too_many_arguments)]
    fn generate_char_cached_stream(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        stop: Option<&str>,
        rng: &mut Rng,
        on_text: &mut dyn FnMut(&str),
        mut cached: CachedTransformer,
    ) -> Result<String> {
        let mut generated = seed.to_string();
        let seed_bytes = seed.len();
        let cap = cached.capacity().max(1);

        // O(1) class lookup (see `generate_stream_core`): built once, reused for
        // every seed and generated character.
        let hex_to_idx: std::collections::HashMap<&str, usize> = self
            .meta
            .class_names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let char_idx = |ch: char| -> usize {
            hex_to_idx
                .get(char_to_hex(ch).as_str())
                .copied()
                .unwrap_or(0)
        };

        // The window holds the token IDs currently represented in the cache.
        let mut window: Vec<usize> = seed.chars().map(char_idx).collect();
        // A seed shorter than the context window primes a partial window
        // (positions 0..len), exactly like the BPE cached path. An empty seed
        // is conditioned on token 0 (`bos_id`), as in `generate_tokens`.
        if num_chars == 0 {
            return Ok(generated);
        }
        if window.is_empty() {
            window.push(0);
        }
        // Prime with the most recent `cap` tokens (positions 0..cap-1).
        if window.len() > cap {
            window.drain(..window.len() - cap);
        }
        let mut dist = cached.prime(&window)?;

        for step in 0..num_chars {
            // The model ends in Softmax; convert probabilities back to log-space
            // so temperature scaling behaves exactly as on the full-forward path.
            let logits: Vec<f32> = dist.iter().map(|&p| p.max(1e-12).ln()).collect();
            if verbose::is_verbose() {
                let (lmin, lmax, lmean) = verbose::stats(&logits);
                vprintln!("[slm::generate_cached] Step {}: logits stats: min={:.4}, max={:.4}, mean={:.4}",
                    step, lmin, lmax, lmean);
            }
            // Repetition penalty sees the tokens currently in the cache window.
            let next_idx = sample_with_params(&logits, params, &window, rng);
            let next_char = hex_to_char(&self.meta.class_names[next_idx]);
            generated.push(next_char);
            let hit_stop = stop.is_some_and(|s| generated[seed_bytes..].ends_with(s));
            let mut buf = [0u8; 4];
            on_text(next_char.encode_utf8(&mut buf));
            if hit_stop {
                break; // I3: halt at the natural boundary
            }

            // Advance the window by the freshly sampled token. If it would
            // overflow the context, slide and re-prime; otherwise extend the
            // cache incrementally with a single cheap feed.
            window.push(next_idx);
            if window.len() > cap {
                // Re-prime from the newest half window: one re-prime per cap/2
                // tokens instead of one per token.
                window.drain(..window.len() - (cap / 2).max(1));
                dist = cached.prime(&window)?;
            } else {
                dist = cached.feed(next_idx)?;
            }
        }

        vprintln!(
            "[slm::generate_cached] Generated {} total chars",
            generated.chars().count()
        );
        Ok(generated)
    }

    /// Streaming BPE generation path. Rebuilds the stored [`ByteBpeTokenizer`],
    /// encodes the seed (document separators become EOS), samples one token at a
    /// time and stops at EOS (the end of the document), at `num_chars`
    /// characters past the seed, or at the `stop` string, whichever comes first.
    /// The result is `seed` verbatim followed by the continuation, so the
    /// character contract matches the char-level path.
    ///
    /// Genuine transformer models drive a per-block [`KvCache`] (the same fast
    /// path as the char-level and WASM builds): one cached `feed` per token and
    /// one re-prime from the newest half window per `context_len / 2` tokens,
    /// with the continuation decoded incrementally, so the cost is linear in
    /// the generated length. Models without transformer blocks (the
    /// embedded-MLP BPE family) fall back to a full forward over the last
    /// `context_len` tokens, left-padded with token 0 when the prompt is shorter
    /// than the window.
    ///
    /// Newly decoded continuation characters are streamed to `on_text` as tokens
    /// land. A trailing partial multi-byte character is held back until it
    /// completes, so a `U+FFFD` placeholder is never emitted and then revised.
    #[allow(clippy::too_many_arguments)]
    fn generate_bpe_stream(
        &self,
        tok: &ByteBpeTokenizer,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        stop: Option<&str>,
        rng: &mut Rng,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let context_len = self.meta.input_dim;
        let eos = tok.eos_id();
        let ids = tok.encode_docs(seed);
        vprintln!(
            "[slm::generate_bpe] seed encoded to {} tokens, ctx={}, num_chars={}",
            ids.len(),
            context_len,
            num_chars
        );

        // Bound the loop: every step adds ≥1 byte, but guard against a model
        // that keeps emitting zero-width or repeated control tokens.
        let max_steps = num_chars.saturating_mul(8).saturating_add(64);

        if let Some(mut cached) = CachedTransformer::try_new(&self.model) {
            // Fast path: KV-cached incremental decoding over a rolling window.
            let cap = cached.capacity().max(1);
            let mut window: Vec<usize> = if ids.is_empty() {
                vec![eos.unwrap_or(0)]
            } else {
                ids[ids.len().saturating_sub(cap)..].to_vec()
            };
            let mut dist = cached.prime(&window)?;
            let stop_len = stop.map_or(0, str::len);
            let mut cont = String::new(); // decoded continuation (complete chars)
            let mut cont_chars = 0usize;
            let mut pending: Vec<u8> = Vec::new(); // incomplete trailing UTF-8
                                                   // Continuation chars (and their bytes) already streamed.
            let (mut emitted, mut emitted_bytes) = (0usize, 0usize);
            // The continuation-char budget: `num_chars`, tightened to the stop
            // position when one is hit.
            let mut budget = num_chars;
            let mut steps = 0;
            while cont_chars < num_chars && steps < max_steps {
                let logits: Vec<f32> = dist.iter().map(|&p| p.max(1e-12).ln()).collect();
                let next = sample_with_params(&logits, params, &window, rng);
                if Some(next) == eos {
                    break; // end of document
                }
                steps += 1;
                let before = cont.len();
                pending.extend_from_slice(tok.token_bytes(next).unwrap_or(&[]));
                drain_utf8(&mut pending, &mut cont);
                cont_chars += cont[before..].chars().count();
                if let Some(s) = stop {
                    // Only the tail can contain a newly completed stop string.
                    let from = (0..=before.saturating_sub(stop_len))
                        .rev()
                        .find(|&i| cont.is_char_boundary(i))
                        .unwrap_or(0);
                    if let Some(i) = cont[from..].find(s) {
                        budget = cont[..from + i + s.len()].chars().count().min(num_chars);
                        break;
                    }
                }
                let avail = cont_chars.min(budget);
                if avail > emitted {
                    let delta: String = cont[emitted_bytes..]
                        .chars()
                        .take(avail - emitted)
                        .collect();
                    on_text(&delta);
                    emitted = avail;
                    emitted_bytes += delta.len();
                }
                window.push(next);
                if window.len() > cap {
                    // One re-prime per cap/2 tokens, not one per token.
                    window.drain(..window.len() - (cap / 2).max(1));
                    dist = cached.prime(&window)?;
                } else {
                    dist = cached.feed(next)?;
                }
            }
            if !pending.is_empty() {
                cont.push_str(&String::from_utf8_lossy(&pending));
            }
            let tail: String = cont[emitted_bytes..]
                .chars()
                .take(budget.saturating_sub(emitted))
                .collect();
            if !tail.is_empty() {
                on_text(&tail);
            }
            return Ok(seed.chars().chain(cont.chars().take(budget)).collect());
        }

        // Fallback (no transformer blocks): full forward over the last
        // `context_len` tokens, left-padded with token 0 when too short.
        // ponytail: re-decodes the whole stream every step (O(n²)); this family
        // is tiny and legacy. Port the incremental decode above if it matters.
        let mut ids = ids;
        let seed_chars = seed.chars().count();
        let target_chars = seed_chars.saturating_add(num_chars);
        let mut emitted_cont = 0usize;
        let mut budget = num_chars;
        let mut flush = |full: &str, budget: usize, on_text: &mut dyn FnMut(&str)| {
            let avail = full.chars().count().saturating_sub(seed_chars).min(budget);
            if avail > emitted_cont {
                let delta: String = full
                    .chars()
                    .skip(seed_chars + emitted_cont)
                    .take(avail - emitted_cont)
                    .collect();
                on_text(&delta);
                emitted_cont = avail;
            }
        };
        let mut steps = 0;
        while tok.decode(&ids).chars().count() < target_chars && steps < max_steps {
            let mut ctx: Vec<f32> = Vec::with_capacity(context_len);
            if ids.len() < context_len {
                ctx.resize(context_len - ids.len(), 0.0);
                ctx.extend(ids.iter().map(|&t| t as f32));
            } else {
                ctx.extend(ids[ids.len() - context_len..].iter().map(|&t| t as f32));
            }
            let input = Tensor::matrix(1, context_len, ctx)?;
            let out = self.model.forward(&input)?;
            let (rows, cols) = out.matrix_dims()?;
            let logits: Vec<f32> = out.data[(rows - 1) * cols..]
                .iter()
                .map(|&p| p.max(1e-12).ln())
                .collect();
            let recent = &ids[ids.len().saturating_sub(context_len)..];
            let next = sample_with_params(&logits, params, recent, rng);
            if Some(next) == eos {
                break; // end of document
            }
            ids.push(next);
            steps += 1;
            let full = tok.decode(&ids);
            if let Some(s) = stop {
                let cont: String = full.chars().skip(seed_chars).collect();
                if let Some(b) = cont.find(s) {
                    budget = cont[..b + s.len()].chars().count().min(num_chars);
                    break;
                }
            }
            flush(&full, budget, on_text);
        }
        let full = tok.decode(&ids);
        flush(&full, budget, on_text);
        Ok(full.chars().take(seed_chars + budget).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::rng::Rng;
    use crate::slm::cached::FEEDS;
    use crate::slm::config::TransformerConfig;
    use crate::slm::generate::StopReason;
    use crate::slm::sampling::SamplingParams;
    use crate::slm::test_support::bias_only_slm;
    use crate::slm::test_support::{
        constant_bpe_slm, feeds, greedy, tiny_bpe_slm, tiny_char_slm, tiny_eos_slm, tiny_utf8_slm,
    };
    use crate::slm::GenerativeSLM;
    use crate::tokenizer::ByteBpeTokenizer;

    #[test]
    fn bpe_generation_reprimes_once_per_half_window() {
        let slm = tiny_bpe_slm(); // context 8
        FEEDS.with(|f| f.set(0));
        let n = 400; // chars >> context
        let out = slm
            .generate_with(
                "the cat",
                n,
                &SamplingParams::with_temperature(0.8),
                &mut Rng::new(5),
            )
            .unwrap();
        assert!(out.chars().count() >= "the cat".len());
        // Each token costs one feed plus, every cap/2 tokens, a re-prime of cap/2
        // feeds: <= 2 feeds per token. The old path re-primed the whole window
        // for every token (about cap feeds per token).
        let tokens = ByteBpeTokenizer::from_state(&slm.meta.tokenizer_state)
            .unwrap()
            .encode(&out["the cat".len()..])
            .len();
        assert!(
            feeds() <= 2 * tokens + 16,
            "too many feeds for {tokens} tokens"
        );
    }

    #[test]
    fn seed_with_doc_separator_is_preserved() {
        let corpus = "the cat sat.<|endoftext|>the dog ran.<|endoftext|>".repeat(30);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs: 1,
            vocab_size: 280,
            ..TransformerConfig::default()
        };
        let slm =
            GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(2), |_, _| {})
                .unwrap();
        let seed = "the cat sat.<|endoftext|>the";
        let out = slm.generate(seed, 20, 0.8, &mut Rng::new(1)).unwrap();
        assert!(out.starts_with(seed), "{out:?}");
        // The model may end the document early; it never exceeds the budget.
        assert!(out.chars().count() <= seed.chars().count() + 20);
    }

    #[test]
    fn char_generation_reprimes_once_per_half_window() {
        let slm = tiny_char_slm(); // context 8
        FEEDS.with(|f| f.set(0));
        let n = 300;
        let out = slm.generate("abcabcab", n, 0.8, &mut Rng::new(1)).unwrap();
        assert_eq!(out.chars().count(), 8 + n);
        assert!(feeds() <= 2 * n + 16, "feeds={}", feeds());
    }

    #[test]
    fn char_generation_survives_a_one_token_window() {
        // Window of 1: the half-window keeps at least one token.
        let corpus = "abab".repeat(30);
        let cfg = TransformerConfig {
            context_len: 1,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs: 1,
            vocab_size: 0,
            ..TransformerConfig::default()
        };
        let slm =
            GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(1), |_, _| {})
                .unwrap();
        let out = slm.generate("a", 12, 0.8, &mut Rng::new(1)).unwrap();
        assert_eq!(out.chars().count(), 13);
    }

    #[test]
    fn bpe_generation_survives_a_one_token_window() {
        // Context 1: the window may hold one token, never the two `cap.max(2)`
        // would allow (the cache would overflow).
        let corpus = "the cat sat on the mat. ".repeat(30);
        let cfg = TransformerConfig {
            context_len: 1,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs: 1,
            vocab_size: 270,
            ..TransformerConfig::default()
        };
        let slm =
            GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(1), |_, _| {})
                .unwrap();
        let out = slm.generate("the", 40, 0.8, &mut Rng::new(1)).unwrap();
        assert!(
            out.starts_with("the") && out.chars().count() == 43,
            "{out:?}"
        );
        let g = slm
            .generate_tokens(
                "the",
                20,
                &[],
                &SamplingParams::with_temperature(0.8),
                &mut Rng::new(2),
            )
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::MaxTokens, "{g:?}");
    }

    #[test]
    fn bpe_generation_streams_multibyte_text_without_replacement_chars() {
        let slm = tiny_utf8_slm();
        let p = greedy();
        let mut streamed = String::new();
        let out = slm
            .generate_stream_with("\u{e9}", 5, &p, &mut Rng::new(1), |f| streamed.push_str(f))
            .unwrap();
        assert_eq!(out, "\u{e9}".repeat(6));
        assert_eq!(streamed, "\u{e9}".repeat(5));
        // A multi-byte stop string sits on char boundaries.
        let until = slm
            .generate_until("\u{e9}", 20, "\u{e9}\u{e9}", &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(until, "\u{e9}".repeat(3));
    }

    #[test]
    fn bpe_generation_from_an_empty_seed_starts_at_bos() {
        let slm = tiny_eos_slm();
        let out = slm
            .generate_with(
                "",
                6,
                &SamplingParams::with_temperature(0.8),
                &mut Rng::new(2),
            )
            .unwrap();
        assert!(out.chars().count() <= 6);
        let p = greedy();
        let g = slm
            .generate_tokens("", 20, &[], &p, &mut Rng::new(2))
            .unwrap();
        let want: String = g.text.chars().take(6).collect();
        assert_eq!(
            slm.generate_with("", 6, &p, &mut Rng::new(2)).unwrap(),
            want
        );
    }

    #[test]
    fn bpe_fallback_path_stops_at_eos_stop_and_budget() {
        // Always EOS: nothing is generated.
        let eos_slm = constant_bpe_slm(|t| {
            let mut b = vec![0.0; t.vocab_size()];
            b[t.eos_id().unwrap()] = 30.0;
            b
        });
        let p = greedy();
        assert_eq!(
            eos_slm
                .generate_with("ab", 10, &p, &mut Rng::new(1))
                .unwrap(),
            "ab"
        );
        // Always the byte 'z': the budget bounds the run, stops cut it.
        let z_slm = constant_bpe_slm(|t| {
            let mut b = vec![0.0; t.vocab_size()];
            b[b'z' as usize] = 30.0;
            b
        });
        let mut streamed = String::new();
        let out = z_slm
            .generate_stream_with("ab", 5, &p, &mut Rng::new(1), |f| streamed.push_str(f))
            .unwrap();
        assert_eq!(out, "abzzzzz");
        assert_eq!(streamed, "zzzzz");
        assert_eq!(
            z_slm
                .generate_until("ab", 9, "zz", &p, &mut Rng::new(1))
                .unwrap(),
            "abzz"
        );
        assert_eq!(
            z_slm.generate_with("ab", 0, &p, &mut Rng::new(1)).unwrap(),
            "ab"
        );
    }

    #[test]
    fn eos_after_an_incomplete_character_flushes_a_replacement() {
        // Near-uniform sampling over bytes: sooner or later EOS follows a lone
        // UTF-8 lead byte, which is flushed lossily (never dropped silently).
        let slm = tiny_utf8_slm();
        let p = SamplingParams::with_temperature(50.0);
        let hit = (0..400u64).any(|seed| {
            let mut streamed = String::new();
            let out = slm
                .generate_stream_with("", 5000, &p, &mut Rng::new(seed), |f| streamed.push_str(f))
                .unwrap();
            assert_eq!(out, streamed, "fragments must add up to the output");
            out.chars().count() < 5000 && out.ends_with('\u{FFFD}')
        });
        assert!(hit, "no run ended with an incomplete character");
    }

    #[test]
    fn char_level_models_without_blocks_use_the_full_forward_loop() {
        // bias_only_slm always predicts 'e' (the largest bias), so greedy
        // decoding is a closed form we can check, streamed and plain.
        let slm = bias_only_slm();
        let mut streamed = String::new();
        let out = slm
            .generate_stream_with("ab", 3, &greedy(), &mut Rng::new(1), |t| {
                streamed.push_str(t)
            })
            .unwrap();
        assert_eq!(out, "abeee");
        assert_eq!(streamed, "eee"); // the seed is never streamed

        // The seed is shorter than the window (left-padded) and longer than it.
        assert_eq!(
            slm.generate_with("a", 2, &greedy(), &mut Rng::new(1))
                .unwrap(),
            "aee"
        );
        assert_eq!(
            slm.generate_with("abcdabcd", 1, &greedy(), &mut Rng::new(1))
                .unwrap(),
            "abcdabcde"
        );

        // A stop string cuts the continuation as soon as it completes.
        let mut pieces = Vec::new();
        let cut = slm
            .generate_stream_until("ab", 10, "ee", &greedy(), &mut Rng::new(1), |t| {
                pieces.push(t.to_string())
            })
            .unwrap();
        assert_eq!(cut, "abee");
        assert_eq!(pieces, ["e", "e"]);

        // With temperature, every sampled char is still one of the vocabulary.
        let hot = SamplingParams::with_temperature(1.0);
        let out = slm.generate_with("ab", 30, &hot, &mut Rng::new(9)).unwrap();
        assert!(out.chars().all(|c| "abcde".contains(c)), "{out}");

        // An empty seed has nothing to condition on.
        assert_eq!(
            slm.generate_with("", 5, &greedy(), &mut Rng::new(1))
                .unwrap(),
            ""
        );
    }
}
