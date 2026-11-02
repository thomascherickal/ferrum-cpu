//! GGUF vocabulary arrays and standalone tokenizer files.
use crate::error::{InferError, Result};
use crate::gguf::{Gguf, MetaValue, NumArray};
use crate::gguf_write::GgufBuilder;
use crate::tokenizer::pretokenize::byte_to_unicode;
use crate::tokenizer::{ByteBpeTokenizer, BASE_VOCAB};
use std::collections::HashMap;

impl ByteBpeTokenizer {
    /// `tokenizer.ggml.tokens`: each token's bytes mapped through
    /// [`byte_to_unicode`]; special tokens by name.
    pub fn gguf_tokens(&self) -> Vec<String> {
        let n_text = BASE_VOCAB + self.merges.len();
        (0..self.vocab.len())
            .map(|i| match i.checked_sub(n_text) {
                Some(s) => self.special[s].clone(),
                None => self.vocab[i].iter().map(|&b| byte_to_unicode(b)).collect(),
            })
            .collect()
    }

    /// `tokenizer.ggml.merges` in rank order: `"<left> <right>"`.
    pub fn gguf_merges(&self) -> Vec<String> {
        let toks = self.gguf_tokens();
        self.merges
            .iter()
            .map(|&(a, b)| format!("{} {}", toks[a as usize], toks[b as usize]))
            .collect()
    }

    /// `tokenizer.ggml.token_type`: 1 = normal, 3 = control (specials).
    pub fn gguf_token_types(&self) -> Vec<i32> {
        let n_text = BASE_VOCAB + self.merges.len();
        (0..self.vocab.len())
            .map(|i| if i < n_text { 1 } else { 3 })
            .collect()
    }

    /// Rebuild from the three GGUF arrays (inverse of the exporters). Expects
    /// Ferrum's layout: 256 byte tokens, then one token per merge in rank
    /// order, then control tokens. Errors on any deviation, except that the
    /// `token_type` of the text tokens (the first `256 + merges`) is not
    /// inspected: foreign GPT-2 GGUFs label them inconsistently (normal, byte,
    /// unused) and the `tokens`/`merges` arrays alone define them. Only the
    /// tokens after the merges must be control (3).
    pub fn from_gguf_vocab(
        tokens: &[String],
        merges: &[String],
        token_types: &[i32],
    ) -> Result<Self> {
        let bad = |m: String| InferError::Format(format!("gguf vocab: {m}"));
        let n_text = BASE_VOCAB + merges.len();
        if tokens.len() < n_text || token_types.len() != tokens.len() {
            return Err(bad(format!(
                "{} tokens / {} types for {} merges",
                tokens.len(),
                token_types.len(),
                merges.len()
            )));
        }
        let ids: HashMap<&str, u32> = tokens[..n_text]
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i as u32))
            .collect();
        let mut tok = Self::byte_level();
        for (b, t) in tokens[..BASE_VOCAB].iter().enumerate() {
            if *t != byte_to_unicode(b as u8).to_string() {
                return Err(bad(format!("token {b} is not byte {b}")));
            }
        }
        for (i, m) in merges.iter().enumerate() {
            // llama.cpp: split at the first space searching from index 1.
            let from = m.chars().next().map_or(0, char::len_utf8);
            let (l, r) = m[from..]
                .find(' ')
                .map(|j| (&m[..from + j], &m[from + j + 1..]))
                .ok_or_else(|| bad(format!("bad merge {m:?}")))?;
            let (Some(&a), Some(&b)) = (ids.get(l), ids.get(r)) else {
                return Err(bad(format!("merge {m:?} has a part that is not a token")));
            };
            if a as usize >= BASE_VOCAB + i || b as usize >= BASE_VOCAB + i {
                return Err(bad(format!("merge {m:?} uses a later token")));
            }
            let merged = [tok.vocab[a as usize].clone(), tok.vocab[b as usize].clone()].concat();
            let want: String = merged.iter().map(|&x| byte_to_unicode(x)).collect();
            if tokens[BASE_VOCAB + i] != want {
                return Err(bad(format!("token {} is not merge {m:?}", BASE_VOCAB + i)));
            }
            tok.vocab.push(merged);
            tok.merges.push((a, b));
        }
        for (k, (name, ty)) in tokens[n_text..]
            .iter()
            .zip(&token_types[n_text..])
            .enumerate()
        {
            if *ty != 3 {
                return Err(bad(format!("token {name:?} after merges is not control")));
            }
            if tok.add_special_token(name)? != n_text + k {
                return Err(bad(format!("duplicate special token {name:?}")));
            }
        }
        tok.rebuild_ranks();
        Ok(tok)
    }

    /// The reusable tokenizer file: a vocab-only GGUF (`ferrum-tokenizer`) whose
    /// `tokenizer.ggml.{model,tokens,merges,token_type}` arrays are also a valid
    /// llama.cpp vocab, plus `tokenizer.ferrum.state` = [`encode_state`](Self::encode_state)
    /// as the lossless source of truth.
    pub fn to_gguf_bytes(&self) -> Vec<u8> {
        let strings =
            |v: Vec<String>| MetaValue::Array(v.into_iter().map(MetaValue::String).collect());
        let types = NumArray::I32(self.gguf_token_types()).into_meta();
        let mut b = GgufBuilder::new();
        b.meta(
            "general.architecture",
            MetaValue::String("ferrum-tokenizer".into()),
        )
        .meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()))
        .meta("tokenizer.ggml.tokens", strings(self.gguf_tokens()))
        .meta("tokenizer.ggml.merges", strings(self.gguf_merges()))
        .meta("tokenizer.ggml.token_type", types)
        .meta(
            "tokenizer.ferrum.state",
            MetaValue::String(self.encode_state()),
        );
        b.into_bytes()
    }

    /// Read a [`to_gguf_bytes`](Self::to_gguf_bytes) file. The state string is
    /// authoritative and is cross-checked against the vocab arrays; a file
    /// without it is rebuilt from the arrays alone. The pre-GGUF text form is
    /// rejected with a pointed error.
    pub fn from_gguf_bytes(bytes: &[u8]) -> Result<Self> {
        let bad = |m: String| InferError::Format(m);
        if bytes.len() < 4 {
            return Err(bad("empty or truncated tokenizer file".into()));
        }
        if !bytes.starts_with(b"GGUF") {
            return Err(bad(
                "old tokenizer file (pre-GGUF); delete it to retrain the tokenizer".into(),
            ));
        }
        let g = Gguf::parse(bytes.to_vec())?;
        if g.architecture() != Some("ferrum-tokenizer") {
            return Err(bad(format!(
                "not a ferrum-tokenizer file (general.architecture = {:?})",
                g.architecture()
            )));
        }
        let array = |key: &str| match g.meta(key) {
            Some(MetaValue::Array(a)) => Ok(a),
            _ => Err(bad(format!(
                "tokenizer file key '{key}' missing or not an array"
            ))),
        };
        let strings = |key: &str| -> Result<Vec<String>> {
            array(key)?
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or_else(|| bad(format!("tokenizer file key '{key}' is not a string array")))
        };
        let tokens = strings("tokenizer.ggml.tokens")?;
        let merges = strings("tokenizer.ggml.merges")?;
        let types: Vec<i32> = match g.meta("tokenizer.ggml.token_type") {
            Some(MetaValue::Nums(NumArray::I32(v))) => v.clone(),
            Some(MetaValue::Array(a)) if a.is_empty() => Vec::new(),
            Some(MetaValue::Array(_) | MetaValue::Nums(_)) => {
                return Err(bad("tokenizer file token_type is not an i32 array".into()))
            }
            _ => {
                return Err(bad(
                    "tokenizer file key 'tokenizer.ggml.token_type' missing or not an array".into(),
                ))
            }
        };
        let Some(state) = g.meta("tokenizer.ferrum.state").and_then(MetaValue::as_str) else {
            return Self::from_gguf_vocab(&tokens, &merges, &types);
        };
        let tok = Self::from_state(state)?;
        if tok.gguf_tokens() != tokens
            || tok.gguf_merges() != merges
            || tok.gguf_token_types() != types
        {
            return Err(bad(
                "tokenizer file is corrupt: tokenizer.ferrum.state and the tokenizer.ggml.* arrays disagree"
                    .into(),
            ));
        }
        Ok(tok)
    }
}

#[cfg(test)]
mod tests {

    use crate::gguf_write::GgufBuilder;
    use crate::tokenizer::pretokenize::byte_to_unicode;
    use crate::tokenizer::test_support::{gguf_parts, small_tok, trained_tok};
    use crate::tokenizer::ByteBpeTokenizer;

    #[test]
    fn training_reserves_eos_and_gguf_vocab_roundtrips() {
        // Needs enough distinct words that 43 merges exist (re-derived: the
        // brief's 4-word corpus runs out of repeated pairs at 272 tokens).
        let corpus: String = (0..200)
            .map(|i| format!("the cat{i} sat{} on mat{}. ", i % 13, i % 7))
            .collect::<String>()
            .repeat(2);
        let tok = ByteBpeTokenizer::train(&corpus, 300).unwrap();
        assert_eq!(tok.vocab_size(), 300);
        assert_eq!(tok.eos_id(), Some(299));
        let (t, m, ty) = (tok.gguf_tokens(), tok.gguf_merges(), tok.gguf_token_types());
        assert_eq!(t.len(), 300);
        assert_eq!(t[b' ' as usize], "Ġ");
        assert_eq!(t[299], "<|endoftext|>");
        assert_eq!(ty[299], 3);
        assert!(m.iter().all(|x| x.matches(' ').count() == 1));
        let back = ByteBpeTokenizer::from_gguf_vocab(&t, &m, &ty).unwrap();
        assert_eq!(back.encode_state(), tok.encode_state());
        let text = "the cat sat. ünïcode 123";
        assert_eq!(back.encode(text), tok.encode(text));
    }

    #[test]
    fn from_gguf_vocab_rejects_bad_merges() {
        let tok = ByteBpeTokenizer::train(&"abab cdcd ".repeat(10), 280).unwrap();
        let (t, mut m, ty) = (tok.gguf_tokens(), tok.gguf_merges(), tok.gguf_token_types());
        m[0] = "zz\u{1F338} yy".into();
        assert!(ByteBpeTokenizer::from_gguf_vocab(&t, &m, &ty).is_err());
    }

    #[test]
    fn gguf_tokens_map_every_byte_through_the_table() {
        let t = small_tok();
        let toks = t.gguf_tokens();
        for b in 0..=255u8 {
            assert_eq!(toks[b as usize], byte_to_unicode(b).to_string(), "byte {b}");
        }
        // a merged token is the concatenation of its parts' mapped chars
        let m = t.gguf_merges();
        for (i, line) in m.iter().enumerate() {
            let (l, r) = line.split_once(' ').unwrap();
            assert_eq!(format!("{l}{r}"), toks[256 + i]);
        }
        // multi-byte UTF-8 (ü = C3 BC) is two mapped chars (Ã¼), not one raw ü
        assert_eq!((toks[0xC3].as_str(), toks[0xBC].as_str()), ("Ã", "¼"));
    }

    #[test]
    fn from_gguf_vocab_error_paths() {
        let t = small_tok();
        let (tk, mg, ty) = gguf_parts(&t);
        assert!(mg.len() >= 3);
        let err = |tk: &[String], mg: &[String], ty: &[i32]| {
            ByteBpeTokenizer::from_gguf_vocab(tk, mg, ty)
                .err()
                .map(|e| e.to_string())
        };
        // lengths
        assert!(err(&tk[..tk.len() - 3], &mg, &ty[..ty.len() - 3])
            .unwrap()
            .contains("tokens"));
        assert!(err(&tk, &mg, &ty[1..]).unwrap().contains("types"));
        // byte table
        let mut bad = tk.clone();
        bad.swap(65, 66);
        assert!(err(&bad, &mg, &ty).unwrap().contains("is not byte"));
        // merge line without a separator / with unknown parts / forward reference
        let mut m = mg.clone();
        m[1] = "nospace".into();
        assert!(err(&tk, &m, &ty).unwrap().contains("bad merge"));
        let mut m = mg.clone();
        m[1] = "ZZ YY".into();
        assert!(err(&tk, &m, &ty).unwrap().contains("not a token"));
        let mut m = mg.clone();
        let j = t
            .merges
            .iter()
            .position(|&(a, b)| a >= 256 || b >= 256)
            .unwrap();
        m[0] = mg[j].clone(); // refers to a merged token that does not exist yet
        assert!(err(&tk, &m, &ty).unwrap().contains("later token"));
        // token that is not the concatenation of its merge
        let mut bad = tk.clone();
        bad.swap(256, 257);
        assert!(err(&bad, &mg, &ty).unwrap().contains("is not merge"));
        // non-control token after the merges
        let mut ty2 = ty.clone();
        *ty2.last_mut().unwrap() = 1;
        assert!(err(&tk, &mg, &ty2).unwrap().contains("not control"));
        // invalid special name (contains ',')
        let mut bad = tk.clone();
        *bad.last_mut().unwrap() = "a,b".into();
        assert!(err(&bad, &mg, &ty).is_some());
        // empty merge list / no specials is fine
        let byte_only = ByteBpeTokenizer::from_gguf_vocab(&tk[..256], &[], &ty[..256]).unwrap();
        assert_eq!((byte_only.vocab_size(), byte_only.num_special()), (256, 0));
    }

    /// Pins the documented leniency: the type of a text token is not inspected
    /// (foreign GPT-2 GGUFs label them inconsistently), only specials must be control.
    #[test]
    fn from_gguf_vocab_ignores_token_types_of_text_tokens() {
        let t = small_tok();
        let (tk, mg, mut ty) = gguf_parts(&t);
        let n_text = 256 + mg.len();
        for (i, ty) in ty.iter_mut().enumerate().take(n_text) {
            *ty = [1, 4, 5, 6][i % 4]; // normal / user-defined / unused / byte
        }
        let back = ByteBpeTokenizer::from_gguf_vocab(&tk, &mg, &ty).unwrap();
        assert_eq!(back.encode_state(), t.encode_state());
    }

    #[test]
    fn from_gguf_vocab_rejects_duplicate_special_names() {
        let mut t = small_tok();
        t.add_special_token("<pad>").unwrap();
        let (mut tk, mg, mut ty) = gguf_parts(&t);
        tk.push("<pad>".into()); // same name twice: ids after it would silently shift
        ty.push(3);
        assert!(ByteBpeTokenizer::from_gguf_vocab(&tk, &mg, &ty).is_err());
    }

    #[test]
    fn from_gguf_vocab_multibyte_first_char_split_follows_llama_cpp() {
        // Merge lines split at the first space after index 1, so a left token whose
        // first char is multi-byte (e.g. 'Ġ') still parses.
        let t = ByteBpeTokenizer::train(&"the the the the ".repeat(10), 300).unwrap();
        let (tk, mg, ty) = gguf_parts(&t);
        assert!(mg.iter().any(|m| m.starts_with('Ġ')), "{mg:?}");
        let back = ByteBpeTokenizer::from_gguf_vocab(&tk, &mg, &ty).unwrap();
        assert_eq!(back.encode(" the the"), t.encode(" the the"));
    }

    #[test]
    fn tokenizer_file_is_gguf_and_roundtrips() {
        let tok = trained_tok();
        let b = tok.to_gguf_bytes();
        assert_eq!(&b[..4], b"GGUF");
        let back = ByteBpeTokenizer::from_gguf_bytes(&b).unwrap();
        assert_eq!(back.encode_state(), tok.encode_state());
        assert_eq!(back.gguf_tokens(), tok.gguf_tokens());
        assert!(ByteBpeTokenizer::from_gguf_bytes(b"256;257").is_err());
        let g = crate::gguf::Gguf::parse(b).unwrap();
        assert_eq!(g.architecture(), Some("ferrum-tokenizer"));
        assert_eq!(
            g.meta("tokenizer.ggml.model").and_then(|v| v.as_str()),
            Some("gpt2")
        );
        for k in ["tokens", "merges", "token_type"] {
            assert!(g.meta(&format!("tokenizer.ggml.{k}")).is_some(), "{k}");
        }
        assert_eq!(
            g.meta("tokenizer.ferrum.state").and_then(|v| v.as_str()),
            Some(tok.encode_state().as_str())
        );
    }

    #[test]
    fn tokenizer_file_roundtrips_byte_level_and_no_specials() {
        for tok in [
            ByteBpeTokenizer::byte_level(),
            ByteBpeTokenizer::train("low lower lowest low low", 300).unwrap(),
        ] {
            let back = ByteBpeTokenizer::from_gguf_bytes(&tok.to_gguf_bytes()).unwrap();
            assert_eq!(back.encode_state(), tok.encode_state());
            assert_eq!(back.encode("low lowest"), tok.encode("low lowest"));
        }
    }

    #[test]
    fn tokenizer_file_old_text_form_is_a_clear_error() {
        let old = trained_tok().encode_state();
        for bytes in [old.as_bytes(), b"256;257"] {
            let e = ByteBpeTokenizer::from_gguf_bytes(bytes)
                .err()
                .unwrap()
                .to_string();
            assert!(e.contains("old tokenizer file (pre-GGUF)"), "{e}");
            assert!(e.contains("delete it to retrain the tokenizer"), "{e}");
        }
    }

    #[test]
    fn tokenizer_file_empty_or_tiny_is_truncated_not_old() {
        for bytes in [&b""[..], b"G", b"256"] {
            let e = ByteBpeTokenizer::from_gguf_bytes(bytes)
                .err()
                .unwrap()
                .to_string();
            assert!(e.contains("empty or truncated tokenizer file"), "{e}");
            assert!(!e.contains("pre-GGUF"), "{e}");
        }
    }

    /// An array count far beyond the file size is rejected by the container
    /// parser before anything is reserved or built from it.
    #[test]
    fn tokenizer_file_array_count_is_bounded_by_file_size() {
        let mut f = Vec::new();
        f.extend_from_slice(b"GGUF");
        f.extend_from_slice(&3u32.to_le_bytes());
        f.extend_from_slice(&0u64.to_le_bytes()); // tensors
        f.extend_from_slice(&2u64.to_le_bytes()); // kv
        let key = |f: &mut Vec<u8>, k: &str| {
            f.extend_from_slice(&(k.len() as u64).to_le_bytes());
            f.extend_from_slice(k.as_bytes());
        };
        key(&mut f, "general.architecture");
        f.extend_from_slice(&8u32.to_le_bytes());
        f.extend_from_slice(&16u64.to_le_bytes());
        f.extend_from_slice(b"ferrum-tokenizer");
        key(&mut f, "tokenizer.ggml.tokens");
        f.extend_from_slice(&9u32.to_le_bytes()); // array
        f.extend_from_slice(&8u32.to_le_bytes()); // of strings
        f.extend_from_slice(&(u64::MAX / 2).to_le_bytes());
        let e = ByteBpeTokenizer::from_gguf_bytes(&f).err().unwrap();
        assert!(e.to_string().contains("array count exceeds"), "{e}");
    }

    #[test]
    fn tokenizer_file_rejects_truncation_wrong_architecture_and_bad_state() {
        use crate::gguf::MetaValue;
        let b = trained_tok().to_gguf_bytes();
        assert!(ByteBpeTokenizer::from_gguf_bytes(&b[..b.len() / 2]).is_err());
        assert!(ByteBpeTokenizer::from_gguf_bytes(b"GGUF").is_err());
        let set = |k: &str, v: MetaValue| {
            let k = k.to_string();
            crate::ckpt::rewrite(
                &b,
                move |m| {
                    m.insert(k, v);
                },
                None,
            )
        };
        let e = ByteBpeTokenizer::from_gguf_bytes(&set(
            "general.architecture",
            MetaValue::String("gpt2".into()),
        ))
        .err()
        .unwrap();
        assert!(e.to_string().contains("not a ferrum-tokenizer"), "{e}");
        let mut no_arch = b.clone();
        no_arch = crate::ckpt::rewrite(&no_arch, |m| drop(m.remove("general.architecture")), None);
        assert!(ByteBpeTokenizer::from_gguf_bytes(&no_arch).is_err());
        // A state string that does not parse.
        let e = ByteBpeTokenizer::from_gguf_bytes(&set(
            "tokenizer.ferrum.state",
            MetaValue::String("not-a-pair".into()),
        ));
        assert!(e.is_err());
        // A state that disagrees with the arrays (different merge count).
        let other = ByteBpeTokenizer::train("abab abab abab", 270)
            .unwrap()
            .encode_state();
        let e = ByteBpeTokenizer::from_gguf_bytes(&set(
            "tokenizer.ferrum.state",
            MetaValue::String(other),
        ))
        .err()
        .unwrap();
        assert!(e.to_string().contains("disagree"), "{e}");
        // Arrays edited while the state is intact.
        for key in ["tokens", "merges", "token_type"] {
            let key = format!("tokenizer.ggml.{key}");
            let bad = crate::ckpt::rewrite(
                &b,
                |m| {
                    if let Some(MetaValue::Array(a)) = m.get_mut(&key) {
                        a.pop();
                    } else if let Some(MetaValue::Nums(crate::gguf::NumArray::I32(a))) =
                        m.get_mut(&key)
                    {
                        a.pop();
                    }
                },
                None,
            );
            let e = ByteBpeTokenizer::from_gguf_bytes(&bad).err().unwrap();
            assert!(e.to_string().contains("disagree"), "{key}: {e}");
        }
        // An empty token_type array is accepted by the reader, then the state disagrees.
        let empty = crate::ckpt::rewrite(
            &b,
            |m| {
                m.insert("tokenizer.ggml.token_type".into(), MetaValue::Array(vec![]));
            },
            None,
        );
        let e = ByteBpeTokenizer::from_gguf_bytes(&empty).err().unwrap();
        assert!(e.to_string().contains("disagree"), "{e}");
        // Arrays removed or mistyped.
        for key in ["tokens", "merges", "token_type"] {
            let key = format!("tokenizer.ggml.{key}");
            let gone = crate::ckpt::rewrite(&b, |m| drop(m.remove(&key)), None);
            assert!(
                ByteBpeTokenizer::from_gguf_bytes(&gone).is_err(),
                "{key} gone"
            );
            let wrong = crate::ckpt::rewrite(
                &b,
                |m| {
                    m.insert(key.clone(), MetaValue::Array(vec![MetaValue::U32(1)]));
                },
                None,
            );
            assert!(
                ByteBpeTokenizer::from_gguf_bytes(&wrong).is_err(),
                "{key} type"
            );
        }
    }

    #[test]
    fn tokenizer_file_without_state_falls_back_to_the_arrays() {
        let tok = trained_tok();
        let b = crate::ckpt::rewrite(
            &tok.to_gguf_bytes(),
            |m| drop(m.remove("tokenizer.ferrum.state")),
            None,
        );
        let back = ByteBpeTokenizer::from_gguf_bytes(&b).unwrap();
        assert_eq!(back.encode_state(), tok.encode_state());
        // Without state, broken arrays are rejected by the vocab parser.
        let broken = crate::ckpt::rewrite(
            &b,
            |m| {
                if let Some(crate::gguf::MetaValue::Array(a)) = m.get_mut("tokenizer.ggml.merges") {
                    a[0] = crate::gguf::MetaValue::String("zz zz".into());
                }
            },
            None,
        );
        assert!(ByteBpeTokenizer::from_gguf_bytes(&broken).is_err());
    }

    /// A tokenizer file normally has no tensors; one that carries two
    /// overlapping ones is rejected by the shared GGUF extent check.
    #[test]
    fn tokenizer_file_rejects_overlapping_tensors() {
        let good = trained_tok().to_gguf_bytes();
        let mut b = GgufBuilder::new();
        for (k, v) in &crate::gguf::Gguf::parse(good).unwrap().metadata {
            b.meta(k, v.clone());
        }
        b.tensor("a", &[8], crate::gguf::GGML_F32, vec![0; 32]);
        b.tensor("b", &[8], crate::gguf::GGML_F32, vec![0; 32]);
        let mut bytes = b.into_bytes();
        ByteBpeTokenizer::from_gguf_bytes(&bytes).unwrap(); // control
        crate::gguf::patch_tensor_offset(&mut bytes, 1, 16);
        let e = ByteBpeTokenizer::from_gguf_bytes(&bytes).err().unwrap();
        assert!(e.to_string().contains("overlap"), "{e}");
    }
}
