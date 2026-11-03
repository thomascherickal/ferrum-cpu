//! Opening a model file: tell Ferrum's own models from imported Llama/Qwen GGUFs by
//! `general.architecture`, refuse what cannot be served, and run the RAM guard before an
//! imported model's weights are read.

use crate::backend::Backend;
use crate::imported::Imported;
use crate::info::{info_for, info_for_imported, Info};
use ferrum_core::memory::{memory_error, resident_bytes};
use ferrum_core::{GenerativeSLM, Gguf, GgufTokenizer, QKind};
use sha2::{Digest, Sha256};
use std::io::Read;

/// Architectures served through `LlamaModel` (what `Gguf::load_llama_prec` accepts).
pub(crate) const IMPORTED_ARCHS: &[&str] = &["llama", "qwen2"];

/// A model file, opened but not yet loaded (the model thread loads it).
pub(crate) enum ModelFile {
    /// Ferrum's own model (`gpt2`/`ferrum` layout): the whole file, loaded as before.
    Ferrum(Vec<u8>),
    /// An imported GGUF, header parsed (tensor data streamed on load), with the precision to
    /// load at and the file's SHA-256.
    Imported {
        gguf: Gguf,
        prec: Option<QKind>,
        sha256: String,
    },
}

/// SHA-256 of the file at `path`, read in 1 MiB chunks so a large file is never resident.
fn sha256_file(path: &str) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let (mut h, mut buf) = (Sha256::new(), vec![0u8; 1 << 20]);
    loop {
        match f.read(&mut buf)? {
            0 => break,
            n => h.update(&buf[..n]),
        }
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Open `path`. Ferrum's own layouts (and anything that is not a readable GGUF, which then
/// fails with the old error) read the whole file; `llama`/`qwen2` are checked for a tokenizer
/// and against the RAM guard (`avail` bytes free, `None` = unknown, never refuses; `force`
/// skips it) at precision `prec`; any other architecture is refused by name.
pub(crate) fn open(
    path: &str,
    prec: Option<QKind>,
    force: bool,
    avail: Option<usize>,
) -> Result<ModelFile, String> {
    let arch = Gguf::open(path)
        .ok()
        .and_then(|g| Some((g.architecture()?.to_string(), g)));
    let gguf = match arch {
        Some((a, g)) if IMPORTED_ARCHS.contains(&a.as_str()) => g,
        Some((a, _)) if a != "gpt2" && a != "ferrum" => {
            return Err(format!(
                "cannot serve {path}: architecture '{a}' is not supported (Ferrum's own gpt2 / \
                 ferrum models, or imported {} GGUFs)",
                IMPORTED_ARCHS.join(" / ")
            ))
        }
        _ => {
            return std::fs::read(path)
                .map(ModelFile::Ferrum)
                .map_err(|e| format!("cannot read {path}: {e}"))
        }
    };
    if GgufTokenizer::from_gguf(&gguf).is_err() {
        return Err(format!(
            "cannot serve {path}: the GGUF has no tokenizer, and the server reads text"
        ));
    }
    if let Some(e) = avail.and_then(|avail| {
        memory_error(
            "resident memory",
            resident_bytes(&gguf, prec),
            avail,
            force,
            "try a smaller --quant, or pass --force to attempt it anyway.",
        )
    }) {
        return Err(e);
    }
    let sha256 = sha256_file(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    Ok(ModelFile::Imported { gguf, prec, sha256 })
}

/// Load `file` into a backend (on the model thread: `GenerativeSLM` is not `Send`).
pub(crate) fn load(file: ModelFile) -> Result<(Backend, Info), String> {
    match file {
        ModelFile::Ferrum(bytes) => {
            let m = GenerativeSLM::from_bytes(&bytes).map_err(|e| e.to_string())?;
            let info = info_for(&m, &bytes);
            Ok((Backend::Ferrum(m), info))
        }
        ModelFile::Imported { gguf, prec, sha256 } => {
            let tok = GgufTokenizer::from_gguf(&gguf).map_err(|e| e.to_string())?;
            let model = gguf.load_llama_prec(prec).map_err(|e| e.to_string())?;
            let info = info_for_imported(&gguf, &model, &tok, prec, sha256);
            Ok((Backend::Imported(Imported { model, tok }), info))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{llama, model, TempFile};
    use ferrum_core::{GgufBuilder, MetaValue};

    fn kind(f: &ModelFile) -> &'static str {
        match f {
            ModelFile::Ferrum(_) => "ferrum",
            ModelFile::Imported { .. } => "imported",
        }
    }

    #[test]
    fn ferrum_models_take_the_old_path_and_imported_ones_the_new() {
        let bpe = TempFile::new("bpe.gguf", &model(270).to_bytes_quantized().unwrap());
        let chr = TempFile::new("chr.gguf", &model(0).to_bytes_quantized().unwrap());
        let lla = TempFile::new("llama.gguf", &llama::tiny_llama(&[]));
        let qwen = TempFile::new(
            "qwen.gguf",
            &llama::tiny_llama(&[("general.architecture", MetaValue::String("qwen2".into()))]),
        );
        for (f, want) in [
            (&bpe, "ferrum"),
            (&chr, "ferrum"),
            (&lla, "imported"),
            (&qwen, "imported"),
        ] {
            let opened = open(f.path(), None, false, None).unwrap();
            assert_eq!(kind(&opened), want, "{}", f.path());
            let (_, info) = load(opened).unwrap();
            assert!(info.params > 0);
        }
        // The Ferrum path holds the very bytes of the file.
        match open(bpe.path(), None, false, None).unwrap() {
            ModelFile::Ferrum(b) => assert_eq!(b, std::fs::read(bpe.path()).unwrap()),
            ModelFile::Imported { .. } => unreachable!(),
        }
    }

    #[test]
    fn unsupported_architectures_and_missing_tokenizers_are_refused_by_name() {
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("mamba".into()));
        let f = TempFile::new("mamba.gguf", &b.into_bytes());
        let e = open(f.path(), None, false, None).err().unwrap();
        assert!(
            e.contains("architecture 'mamba' is not supported") && e.contains("llama / qwen2"),
            "{e}"
        );
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        let f = TempFile::new("notok.gguf", &b.into_bytes());
        let e = open(f.path(), None, false, None).err().unwrap();
        assert!(e.contains("has no tokenizer"), "{e}");
    }

    #[test]
    fn missing_and_non_gguf_files_keep_the_old_errors() {
        let e = open("/nonexistent/m.gguf", None, false, None)
            .err()
            .unwrap();
        assert!(e.starts_with("cannot read /nonexistent/m.gguf"), "{e}");
        let f = TempFile::new("junk.gguf", b"not a model file");
        let e = load(open(f.path(), None, false, None).unwrap())
            .err()
            .unwrap();
        assert!(!e.is_empty());
    }

    #[test]
    fn the_ram_guard_refuses_over_90_percent_unless_forced_or_unknown() {
        let f = TempFile::new("guard.gguf", &llama::tiny_llama(&[]));
        let g = Gguf::open(f.path()).unwrap();
        let (f32_est, int4_est) = (
            resident_bytes(&g, None),
            resident_bytes(&g, Some(QKind::Int4)),
        );
        assert!(int4_est < f32_est);
        // Just enough RAM for int4 but not for f32.
        let avail = Some(int4_est * 10 / 9 + 1);
        let e = open(f.path(), None, false, avail).err().unwrap();
        assert!(
            e.starts_with("estimated resident memory (")
                && e.ends_with("try a smaller --quant, or pass --force to attempt it anyway."),
            "{e}"
        );
        assert!(open(f.path(), Some(QKind::Int4), false, avail).is_ok());
        assert!(open(f.path(), None, true, avail).is_ok(), "--force");
        assert!(open(f.path(), None, false, None).is_ok(), "unknown RAM");
        // Ferrum's own models are never guarded (and ignore the precision).
        let m = TempFile::new("own.gguf", &model(0).to_bytes_quantized().unwrap());
        assert!(open(m.path(), Some(QKind::Int4), false, Some(1)).is_ok());
    }

    #[test]
    fn imported_files_are_hashed_whole() {
        let bytes = llama::tiny_llama(&[]);
        let f = TempFile::new("hash.gguf", &bytes);
        let want: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(sha256_file(f.path()).unwrap(), want);
        match open(f.path(), None, false, None).unwrap() {
            ModelFile::Imported { sha256, prec, .. } => assert_eq!((sha256, prec), (want, None)),
            ModelFile::Ferrum(_) => unreachable!(),
        }
    }
}
