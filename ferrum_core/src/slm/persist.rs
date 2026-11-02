//! Serialization: GGUF bytes, save / load, and load-or-train.
use crate::error::Result;
use crate::loader::{from_vec, to_bytes};
use crate::rng::Rng;
use crate::slm::config::TransformerConfig;
use crate::slm::{GenerativeSLM, TokenizerCache};

impl GenerativeSLM {
    /// Serialize the trained Generative SLM model to self-contained GGUF bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        vprintln!("[slm::GenerativeSLM::to_bytes] Serializing model...");
        let bytes = to_bytes(&self.model, &self.norm, &self.meta)?;
        vprintln!(
            "[slm::GenerativeSLM::to_bytes] Serialized to {} bytes",
            bytes.len()
        );
        Ok(bytes)
    }

    /// Serialize to Ferrum GGUF with int8-quantised weights (≈4× smaller files).
    pub fn to_bytes_quantized(&self) -> Result<Vec<u8>> {
        vprintln!("[slm::GenerativeSLM::to_bytes_quantized] Serializing quantized model...");
        let bytes = crate::loader::to_bytes_quantized(&self.model, &self.norm, &self.meta)?;
        vprintln!(
            "[slm::GenerativeSLM::to_bytes_quantized] Serialized to {} bytes",
            bytes.len()
        );
        Ok(bytes)
    }

    /// Serialize to Ferrum GGUF with **int4** weights (~8× smaller than f32). The
    /// recommended on-disk format for large (≥1B) models: files are ⅛ the
    /// size on disk (weights are dequantized to f32 on load).
    pub fn to_bytes_quantized_int4(&self) -> Result<Vec<u8>> {
        vprintln!("[slm::GenerativeSLM::to_bytes_quantized_int4] Serializing int4 model...");
        let bytes = crate::loader::to_bytes_quantized_int4(&self.model, &self.norm, &self.meta)?;
        vprintln!(
            "[slm::GenerativeSLM::to_bytes_quantized_int4] Serialized to {} bytes",
            bytes.len()
        );
        Ok(bytes)
    }

    /// Save the trained model to `model_path` as int8-quantized Ferrum GGUF
    /// (the default on-disk format — ≈4× smaller than f32). Parent
    /// directories are created as needed; the write is atomic (temp file + rename).
    pub fn save(&self, model_path: &str) -> Result<()> {
        let bytes = self.to_bytes_quantized()?;
        write_model(model_path, &bytes)?;
        vprintln!(
            "[slm::GenerativeSLM::save] Wrote {} bytes → {}",
            bytes.len(),
            model_path
        );
        Ok(())
    }

    /// Save as **int4** Ferrum GGUF (~8× smaller than f32; the matrices load back
    /// as f32). Note:
    /// for the small QAT-trained SLMs produced here, weights are int8-snapped
    /// during training, so [`GenerativeSLM::save`] (int8) is bit-faithful to the
    /// in-memory model whereas int4 adds extra quantization drift — int4 pays off
    /// where file size matters more than the last bit of small-model accuracy.
    pub fn save_int4(&self, model_path: &str) -> Result<()> {
        let bytes = self.to_bytes_quantized_int4()?;
        write_model(model_path, &bytes)?;
        vprintln!(
            "[slm::GenerativeSLM::save_int4] Wrote {} bytes → {}",
            bytes.len(),
            model_path
        );
        Ok(())
    }

    /// Save as full-precision (f32) Ferrum GGUF: exact weights, ≈4× the int8
    /// size. Parent directories are created as needed.
    pub fn save_f32(&self, model_path: &str) -> Result<()> {
        let bytes = self.to_bytes()?;
        write_model(model_path, &bytes)?;
        vprintln!(
            "[slm::GenerativeSLM::save_f32] Wrote {} bytes → {}",
            bytes.len(),
            model_path
        );
        Ok(())
    }

    /// Load a previously saved model (Ferrum GGUF) from `model_path`.
    pub fn load(model_path: &str) -> Result<Self> {
        Self::from_vec(std::fs::read(model_path)?)
    }

    /// Load the model from `model_path` if it exists; otherwise train a
    /// transformer SLM on `corpus` with `cfg` (QAT, int8), save it to
    /// `model_path`, and return it.
    ///
    /// The returned `bool` is `true` when the model was loaded from disk
    /// (no training happened) and `false` when it was freshly trained.
    pub fn load_or_train<F>(
        model_path: &str,
        corpus: &str,
        cfg: &TransformerConfig,
        rng: &mut Rng,
        progress_callback: F,
    ) -> Result<(Self, bool)>
    where
        F: FnMut(usize, f32),
    {
        if std::path::Path::new(model_path).exists() {
            vprintln!(
                "[slm::load_or_train] Found {} — loading instead of retraining",
                model_path
            );
            return Ok((Self::load(model_path)?, true));
        }
        let slm = Self::train_transformer_config(corpus, cfg, rng, progress_callback)?;
        slm.save(model_path)?;
        Ok((slm, false))
    }

    /// Load a trained Generative SLM model from self-contained GGUF bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_vec(bytes.to_vec())
    }

    /// [`GenerativeSLM::from_bytes`] taking ownership of the bytes: the file is
    /// parsed in place instead of copied ([`GenerativeSLM::load`] uses this).
    pub fn from_vec(bytes: Vec<u8>) -> Result<Self> {
        vprintln!(
            "[slm::GenerativeSLM::from_bytes] Deserializing from {} bytes...",
            bytes.len()
        );
        let (model, norm, meta) = from_vec(bytes)?;
        crate::loader::check_slm_consistency(&model, &meta)?;
        vprintln!(
            "[slm::GenerativeSLM::from_bytes] Loaded: {} layers, input_dim={}, output_dim={}",
            model.len(),
            meta.input_dim,
            meta.output_dim
        );
        Ok(Self {
            model,
            norm,
            meta,
            tokenizer: TokenizerCache::default(),
        })
    }
}

/// Create the parent directories of `model_path`, then write `bytes` atomically.
fn write_model(model_path: &str, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = std::path::Path::new(model_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    crate::fs_atomic::atomic_write(model_path, bytes)
}

#[cfg(test)]
mod tests {
    use crate::meta::ModelMetadata;
    use crate::rng::Rng;
    use crate::slm::config::TransformerConfig;
    use crate::slm::test_support::{greedy, tiny_bpe_slm};
    use crate::slm::GenerativeSLM;

    /// End-to-end int4: a trained transformer serialized to int4 and reloaded
    /// must generate deterministically.
    #[test]
    fn int4_serialize_load_generates() {
        let corpus: String = "the quick brown fox jumps over the lazy dog. ".repeat(8);
        let mut rng = Rng::new(7);
        // dim 32 / 1 block keeps the test cheap while the [32,32] projections are
        // still well above QUANT_MIN_LEN, so int4 packing genuinely engages.
        let slm =
            GenerativeSLM::train_transformer(&corpus, 8, 32, 4, 1, 64, 8, 0.01, 8, 0, &mut rng)
                .unwrap();

        let bytes = slm.to_bytes_quantized_int4().unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        let loaded = GenerativeSLM::from_bytes(&bytes).unwrap();

        // Generation runs and is deterministic for a fixed RNG.
        let a = loaded
            .generate("the quick", 20, 0.7, &mut Rng::new(99))
            .unwrap();
        let b = loaded
            .generate("the quick", 20, 0.7, &mut Rng::new(99))
            .unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with("the quick"));
        assert_eq!(a.chars().count(), "the quick".chars().count() + 20);
    }

    #[test]
    fn from_vec_loads_like_from_bytes() {
        let slm = tiny_bpe_slm();
        let bytes = slm.to_bytes().unwrap();
        let a = GenerativeSLM::from_bytes(&bytes).unwrap();
        let b = GenerativeSLM::from_vec(bytes).unwrap();
        assert_eq!(a.meta.class_names, b.meta.class_names);
        assert_eq!(a.meta.tokenizer_state, b.meta.tokenizer_state);
        let p = greedy();
        assert_eq!(
            a.generate_tokens("the ", 12, &[], &p, &mut Rng::new(1))
                .unwrap()
                .text,
            b.generate_tokens("the ", 12, &[], &p, &mut Rng::new(1))
                .unwrap()
                .text
        );
        assert!(GenerativeSLM::from_vec(Vec::new()).is_err());
    }

    #[test]
    fn save_f32_roundtrips_exactly_and_creates_parent_dirs() {
        let slm = tiny_bpe_slm();
        let dir = std::env::temp_dir().join(format!("ferrum_save_f32_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested/m.gguf");
        let path = path.to_str().unwrap();
        slm.save_f32(path).unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(crate::loader::gguf_quant(&bytes).unwrap(), "f32");
        let back = GenerativeSLM::load(path).unwrap();
        let p = greedy();
        assert_eq!(
            back.generate_with("the", 20, &p, &mut Rng::new(1)).unwrap(),
            slm.generate_with("the", 20, &p, &mut Rng::new(1)).unwrap()
        );
        // A path whose parent is a regular file cannot be created.
        let blocked = format!("{path}/x.gguf");
        assert!(slm.save_f32(&blocked).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn persist_save_to_unwritable_dir_errors() {
        let slm = tiny_bpe_slm();
        let file =
            std::env::temp_dir().join(format!("ferrum_persist_blocked_{}", std::process::id()));
        std::fs::write(&file, b"not a directory").unwrap();
        let blocked = format!("{}/x.gguf", file.display());
        assert!(slm.save(&blocked).is_err());
        assert!(slm.save_int4(&blocked).is_err());
        assert!(slm.save_f32(&blocked).is_err());
        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn from_bytes_rejects_metadata_that_disagrees_with_the_layers() {
        // Regression (fuzz): a char-level file whose metadata said input_dim 0 loaded fine
        // and then asked score() for a 1.6 GB buffer.
        let corpus = "abcabc abc\n".repeat(30);
        let cfg = TransformerConfig {
            context_len: 8,
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
        assert!(GenerativeSLM::from_bytes(&slm.to_bytes().unwrap()).is_ok());
        let GenerativeSLM {
            model, norm, meta, ..
        } = slm;
        for edit in [
            |m: &mut ModelMetadata| m.input_dim = 0,
            |m: &mut ModelMetadata| m.input_dim = 1 << 30,
            |m: &mut ModelMetadata| m.output_dim += 1,
            |m: &mut ModelMetadata| {
                m.class_names.pop();
            },
        ] {
            let mut bad = meta.clone();
            edit(&mut bad);
            let bytes = crate::loader::to_bytes(&model, &norm, &bad).unwrap();
            let e = GenerativeSLM::from_bytes(&bytes)
                .err()
                .expect("must be rejected");
            assert!(e.to_string().contains("disagrees"), "{e}");
        }
    }

    #[test]
    fn saving_to_a_bare_filename_needs_no_parent_directory() {
        let slm = tiny_bpe_slm();
        let name = format!("ferrum_persist_bare_{}", std::process::id());
        for (suffix, save) in [
            (
                "q8.gguf",
                GenerativeSLM::save as fn(&GenerativeSLM, &str) -> crate::error::Result<()>,
            ),
            ("q4.gguf", GenerativeSLM::save_int4),
            ("f32.gguf", GenerativeSLM::save_f32),
        ] {
            let path = format!("{name}_{suffix}");
            save(&slm, &path).unwrap();
            let loaded = GenerativeSLM::load(&path);
            let _ = std::fs::remove_file(&path);
            assert!(loaded.is_ok(), "{suffix}");
        }
    }

    #[test]
    fn saves_are_atomic_byte_identical_and_a_failed_save_keeps_the_old_file() {
        let slm = tiny_bpe_slm();
        let dir =
            std::env::temp_dir().join(format!("ferrum_persist_atomic_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        type Save = fn(&GenerativeSLM, &str) -> crate::error::Result<()>;
        type Bytes = fn(&GenerativeSLM) -> crate::error::Result<Vec<u8>>;
        let cases: [(&str, Save, Bytes); 3] = [
            ("q8", GenerativeSLM::save, GenerativeSLM::to_bytes_quantized),
            (
                "q4",
                GenerativeSLM::save_int4,
                GenerativeSLM::to_bytes_quantized_int4,
            ),
            ("f32", GenerativeSLM::save_f32, GenerativeSLM::to_bytes),
        ];
        for (name, save, bytes) in cases {
            let p = dir.join(format!("{name}.gguf"));
            let p = p.to_str().unwrap();
            std::fs::write(p, b"previous model").unwrap();
            // A directory squatting on the temp name makes the save fail midway.
            std::fs::create_dir(format!("{p}.tmp")).unwrap();
            assert!(save(&slm, p).is_err(), "{name}");
            assert_eq!(
                std::fs::read(p).unwrap(),
                b"previous model",
                "{name}: old file damaged"
            );
            std::fs::remove_dir(format!("{p}.tmp")).unwrap();
            // A good save writes exactly the serialized bytes and leaves no temp file.
            save(&slm, p).unwrap();
            assert_eq!(std::fs::read(p).unwrap(), bytes(&slm).unwrap(), "{name}");
            assert!(
                !std::path::Path::new(&format!("{p}.tmp")).exists(),
                "{name}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
