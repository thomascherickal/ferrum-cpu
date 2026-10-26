//! Pure-`std` GGUF reader — import externally pretrained weights into Ferrum.
//!
//! GGUF (the llama.cpp model container) is the lingua franca for distributing
//! quantized LLM/SLM weights. This module parses it with **zero dependencies**
//! and **no `unsafe`**: header, the full typed metadata key/value table, the
//! tensor directory, and dequantization of the common block formats (F32, F16,
//! Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_1) into f32 — from which [`crate::quant::QWeight`] packs
//! them to Ferrum's in-memory int4/int8 (Opt#1).
//!
//! There is no `mmap` (it would need `unsafe`), so [`Gguf::from_path`] reads the
//! file into memory; that is fine for the import step, where the int4 result is
//! what stays resident.
//!
//! ## Architecture gap — read this before expecting a Llama to *run*
//!
//! Importing weights ≠ running the model. Ferrum's [`crate::layer::TransformerBlock`]
//! is a **learned-positional-embedding + LayerNorm + ReLU-FFN** decoder. Modern
//! GGUF models (Llama/Qwen/Mistral/Phi) use **RoPE** positions, **RMSNorm**,
//! **SwiGLU** gated FFNs, and often **grouped-query attention** — none of which
//! this block implements. So this reader will faithfully import the *tensors* of
//! such a model, but running it end-to-end additionally requires a matching
//! block (RoPE + RMSNorm + SwiGLU + GQA). `Gguf::into_ferrum_slm` therefore
//! converts only models whose GGUF `*.architecture` Ferrum can actually execute,
//! and returns a clear error otherwise rather than producing wrong output.

mod cursor;
mod dequant;
mod format;
mod header;
mod llama;
mod tensors;
#[cfg(test)]
mod test_support;

pub use self::dequant::f16_to_f32;
#[cfg(test)]
pub(crate) use self::dequant::get_scale_min_k4;
pub(crate) use self::format::{
    DEFAULT_ALIGNMENT, GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
    GGML_Q8_0, GGML_Q8_1, GGUF_MAGIC, Q4_K_BLOCK, Q5_K_BLOCK, Q6_K_BLOCK, QK, QK_K, VT_ARRAY,
    VT_BOOL, VT_F32, VT_F64, VT_I16, VT_I32, VT_I64, VT_I8, VT_STRING, VT_U16, VT_U32, VT_U64,
    VT_U8,
};
pub(crate) use self::header::bounded;
#[cfg(test)]
pub(crate) use self::header::patch_tensor_offset;
pub(crate) use self::llama::transpose_2d;

use crate::error::Result;
use crate::gguf::cursor::{fmt, to_usize};
use crate::gguf::header::{check_extents, is_truncation, parse_header, read_fully};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::sync::Mutex;

/// A typed GGUF metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// Best-effort `usize` view of any integer scalar.
    pub fn as_usize(&self) -> Option<usize> {
        match *self {
            MetaValue::U8(v) => Some(v as usize),
            MetaValue::I8(v) if v >= 0 => Some(v as usize),
            MetaValue::U16(v) => Some(v as usize),
            MetaValue::I16(v) if v >= 0 => Some(v as usize),
            MetaValue::U32(v) => Some(v as usize),
            MetaValue::I32(v) if v >= 0 => Some(v as usize),
            MetaValue::U64(v) => usize::try_from(v).ok(),
            MetaValue::I64(v) if v >= 0 => usize::try_from(v).ok(),
            _ => None,
        }
    }
    /// `f32` view of any float/integer scalar.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            MetaValue::F32(v) => Some(v),
            MetaValue::F64(v) => Some(v as f32),
            _ => self.as_usize().map(|u| u as f32),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// One tensor's directory entry (name, shape, element type, data offset).
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGML order (`dims[0]` is the fastest-varying axis).
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Byte offset of this tensor within the tensor-data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total element count (product of dimensions). Saturates at `usize::MAX`
    /// instead of overflowing, so a malicious dims table cannot panic a debug
    /// build (or truncate on 32-bit); the impossible size then fails the
    /// checked sizing in `type_nbytes` with a clean error rather than wrapping.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().fold(1usize, |acc, &d| {
            acc.saturating_mul(usize::try_from(d).unwrap_or(usize::MAX))
        })
    }
}

/// Where a [`Gguf`]'s tensor data lives. `Memory` holds the whole file (the
/// `from_path`/`parse` path); `Streamed` keeps only an open file handle and
/// reads each tensor's bytes on demand (the `open` path — see [`Gguf::open`]),
/// so peak RAM during import is one tensor at a time, not the whole multi-GB
/// file. No `mmap` (that needs `unsafe`); a plain `File` + `seek` + `read`,
/// guarded by a `Mutex` so reads work through a shared `&self`.
enum Source {
    Memory(Vec<u8>),
    Streamed { file: Mutex<File>, len: usize },
}

/// A parsed GGUF file: metadata, tensor directory, and a handle to the tensor
/// data (in memory or streamed from disk — see `Source`).
pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute byte offset where the (aligned) tensor-data section begins.
    data_offset: usize,
    source: Source,
}

impl Gguf {
    /// Read and parse a GGUF file from disk **fully into memory**. Convenient for
    /// small files and tests; for large checkpoints prefer [`Gguf::open`], which
    /// streams tensor data from disk instead of holding the whole file resident.
    pub fn from_path(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::parse(bytes)
    }

    /// Open a GGUF file in **streamed** mode (G-mmap): parse only the header
    /// prefix into memory and keep the file open, reading each tensor's bytes on
    /// demand during [`Self::dequantize`] / [`Self::load_llama`]. Peak resident
    /// memory for the parse step is the header (typically a few MB), not the
    /// whole multi-GB file. The prefix is grown geometrically until the header
    /// fits, so tensor *data* is never read here.
    pub fn open(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let len = to_usize(file.metadata()?.len());
        let mut cap = (1usize << 20).min(len.max(1)); // start at 1 MiB
        loop {
            file.seek(SeekFrom::Start(0))?;
            let mut prefix = vec![0u8; cap];
            let got = read_fully(&mut file, &mut prefix)?;
            prefix.truncate(got);
            match parse_header(&prefix, len) {
                Ok((version, metadata, tensors, data_offset)) => {
                    if data_offset > len {
                        return Err(fmt("GGUF tensor-data section begins past EOF"));
                    }
                    check_extents(&tensors, len - data_offset)?;
                    return Ok(Self {
                        version,
                        metadata,
                        tensors,
                        data_offset,
                        source: Source::Streamed {
                            file: Mutex::new(file),
                            len,
                        },
                    });
                }
                // The header didn't fit in the prefix yet: read a bigger one.
                Err(ref e) if cap < len && is_truncation(e) => {
                    cap = cap.saturating_mul(2).min(len);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Parse GGUF from an owned byte buffer (held resident; see [`Self::open`]
    /// for the streamed alternative).
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let (version, metadata, tensors, data_offset) = parse_header(&bytes, bytes.len())?;
        if data_offset > bytes.len() {
            return Err(fmt("GGUF tensor-data section begins past EOF"));
        }
        check_extents(&tensors, bytes.len() - data_offset)?;
        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            source: Source::Memory(bytes),
        })
    }

    /// The model architecture string (`general.architecture`), e.g. `"llama"`.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    /// Look up a metadata value by exact key.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// Size of the whole file in bytes: the bound for metadata-sized
    /// allocations that no single tensor accounts for.
    pub(crate) fn file_len(&self) -> usize {
        match &self.source {
            Source::Memory(b) => b.len(),
            Source::Streamed { len, .. } => *len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Gguf, MetaValue};
    use crate::error::InferError;
    use crate::gguf::format::{GGML_F32, GGML_Q8_0, GGUF_MAGIC, QK};
    use crate::gguf::test_support::{parse_and_open, put_u32, put_u64, synth_gguf, synth_llama};
    use crate::gguf_write::GgufBuilder;
    use crate::quant::QKind;
    use std::fs::File;

    #[test]
    fn parses_header_metadata_and_directory() {
        let (bytes, _, _, _) = synth_gguf();
        let g = Gguf::parse(bytes).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("ferrum-test"));
        assert_eq!(g.tensors.len(), 3);
        assert!(g.tensor("t_f32").is_some());
        assert_eq!(g.tensor("t_q8").unwrap().ggml_type, GGML_Q8_0);
        assert_eq!(g.tensor("t_q4").unwrap().num_elements(), QK);
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        assert!(Gguf::parse(vec![0, 1, 2, 3, 4, 5, 6, 7]).is_err());
        let (mut bytes, _, _, _) = synth_gguf();
        bytes[4] = 99; // version byte
        assert!(Gguf::parse(bytes).is_err());
    }

    #[test]
    fn unsupported_architecture_is_rejected_clearly() {
        let (bytes, _, _, _) = synth_gguf();
        let g = Gguf::parse(bytes).unwrap();
        // "ferrum-test" is neither llama nor qwen2.
        match g.load_llama(QKind::Int8) {
            Err(InferError::Format(_)) => {}
            _ => panic!("expected a Format error for an unsupported architecture"),
        }
    }

    #[test]
    fn streamed_open_matches_in_memory_parse() {
        let bytes = synth_llama();
        let path =
            std::env::temp_dir().join(format!("ferrum_gguf_stream_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let mem = Gguf::parse(bytes).unwrap();
        let streamed = Gguf::open(path.to_str().unwrap()).unwrap();
        assert_eq!(streamed.tensors.len(), mem.tensors.len());
        assert_eq!(streamed.architecture(), Some("llama"));

        // A tensor dequantizes identically whether held in memory or streamed.
        let tm = mem.tensor("token_embd.weight").unwrap().clone();
        let ts = streamed.tensor("token_embd.weight").unwrap().clone();
        assert_eq!(
            mem.dequantize(&tm).unwrap(),
            streamed.dequantize(&ts).unwrap()
        );

        // And a full model loads through the streamed path.
        let model = streamed.load_llama(QKind::Int8).unwrap();
        assert_eq!(model.cfg.vocab_size, 10);
        assert!(model
            .forward_tokens(&[1, 2, 3])
            .unwrap()
            .data
            .iter()
            .all(|v| v.is_finite()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn metavalue_accessors() {
        assert_eq!(MetaValue::U8(5).as_usize(), Some(5));
        assert_eq!(MetaValue::U64(7).as_usize(), Some(7));
        assert_eq!(MetaValue::I32(9).as_usize(), Some(9));
        assert_eq!(MetaValue::I32(-1).as_usize(), None);
        assert_eq!(MetaValue::I8(-3).as_usize(), None);
        assert_eq!(MetaValue::F32(2.5).as_f32(), Some(2.5));
        assert_eq!(MetaValue::U16(4).as_f32(), Some(4.0));
        assert_eq!(MetaValue::String("hi".into()).as_str(), Some("hi"));
        assert_eq!(MetaValue::Bool(true).as_usize(), None);
        assert_eq!(MetaValue::Array(vec![MetaValue::U8(1)]).as_f32(), None);
    }

    #[test]
    fn open_grows_its_prefix_past_an_array_longer_than_the_first_read() {
        // 1.1 Mi one-byte elements: more than `open`'s first 1 MiB read, so
        // the count check must ask for more bytes, not reject the file.
        let n = 1_100_000;
        let mut b = crate::gguf_write::GgufBuilder::new();
        b.meta("big", MetaValue::Array(vec![MetaValue::U8(7); n]));
        b.tensor("t", &[2], GGML_F32, vec![0; 8]);
        let bytes = b.into_bytes();
        let (p, o) = parse_and_open(&bytes);
        for g in [p.unwrap(), o.unwrap()] {
            assert!(matches!(g.meta("big"), Some(MetaValue::Array(a)) if a.len() == n));
            assert_eq!(g.file_len(), bytes.len());
        }
    }

    #[test]
    fn open_rejects_an_absurd_length_claim_without_reading_the_whole_file() {
        // A sparse 64 MiB file whose header claims a 2^40-byte key: rejected from the
        // first 1 MiB read (before this fix `open` doubled its prefix to the file size).
        let path =
            std::env::temp_dir().join(format!("ferrum_gguf_claim_{}.gguf", std::process::id()));
        let mut hdr = Vec::new();
        put_u32(&mut hdr, GGUF_MAGIC);
        put_u32(&mut hdr, 3);
        put_u64(&mut hdr, 0);
        put_u64(&mut hdr, 1);
        put_u64(&mut hdr, 1 << 40);
        {
            use std::io::Write;
            let mut f = File::create(&path).unwrap();
            f.write_all(&hdr).unwrap();
            f.set_len(64 << 20).unwrap();
        }
        let e = Gguf::open(path.to_str().unwrap())
            .err()
            .unwrap()
            .to_string();
        let _ = std::fs::remove_file(&path);
        assert!(e.contains("truncated"), "{e}");
    }

    #[test]
    fn wide_scalar_metadata_roundtrips_and_converts() {
        let mut b = GgufBuilder::new();
        b.meta("t.i16", MetaValue::I16(7))
            .meta("t.i16neg", MetaValue::I16(-1))
            .meta("t.i64", MetaValue::I64(-5))
            .meta("t.f64", MetaValue::F64(1.5));
        let g = Gguf::parse(b.into_bytes()).unwrap();
        assert_eq!(g.meta("t.i16"), Some(&MetaValue::I16(7)));
        assert_eq!(g.meta("t.i64"), Some(&MetaValue::I64(-5)));
        assert_eq!(g.meta("t.f64"), Some(&MetaValue::F64(1.5)));
        assert_eq!(g.meta("t.i16").unwrap().as_usize(), Some(7));
        assert_eq!(g.meta("t.i16neg").unwrap().as_usize(), None);
        assert_eq!(g.meta("t.f64").unwrap().as_f32(), Some(1.5));
    }

    #[test]
    fn from_path_reads_a_file_and_open_rejects_a_data_section_past_eof() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();

        // from_path == parse of the same bytes.
        let (bytes, _, _, _) = synth_gguf();
        let ok = dir.join(format!("ferrum_gguf_from_path_{pid}.gguf"));
        std::fs::write(&ok, &bytes).unwrap();
        let g = Gguf::from_path(ok.to_str().unwrap()).unwrap();
        assert_eq!(g.architecture(), Some("ferrum-test"));
        assert!(Gguf::from_path("/nonexistent/ferrum.gguf").is_err());
        let _ = std::fs::remove_file(&ok);

        // A header-only file whose aligned data section starts past EOF: both
        // the in-memory and the streamed reader refuse it.
        let mut short = Vec::new();
        short.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        short.extend_from_slice(&3u32.to_le_bytes());
        short.extend_from_slice(&0u64.to_le_bytes());
        short.extend_from_slice(&0u64.to_le_bytes());
        assert!(Gguf::parse(short.clone()).is_err());
        let p = dir.join(format!("ferrum_gguf_short_{pid}.gguf"));
        std::fs::write(&p, &short).unwrap();
        let err = Gguf::open(p.to_str().unwrap()).err().unwrap();
        let _ = std::fs::remove_file(&p);
        assert!(err.to_string().contains("past EOF"), "{err}");
    }
}
