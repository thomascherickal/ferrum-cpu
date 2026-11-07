//! GgufBuilder: metadata + tensor directory + aligned data to GGUF v3 bytes.
use crate::fs_atomic::atomic_write;
use crate::gguf::{
    MetaValue, DEFAULT_ALIGNMENT, GGUF_MAGIC, VT_ARRAY, VT_BOOL, VT_F32, VT_F64, VT_I16, VT_I32,
    VT_I64, VT_I8, VT_STRING, VT_U16, VT_U32, VT_U64, VT_U8,
};

fn align_up(x: usize, a: usize) -> usize {
    if a == 0 {
        x
    } else {
        x.div_ceil(a) * a
    }
}

/// A tensor queued for emission: name, dims (GGML order), type id, encoded bytes.
pub(super) struct TensorOut {
    pub(super) name: String,
    pub(super) dims: Vec<u64>,
    pub(super) ggml_type: u32,
    pub(super) data: Vec<u8>,
}

/// Accumulates metadata + tensors, then emits a byte-exact GGUF v3 file.
/// The exact inverse of [`crate::gguf`]'s `parse_header`.
pub struct GgufBuilder {
    pub(super) metadata: Vec<(String, MetaValue)>,
    pub(super) tensors: Vec<TensorOut>,
    pub(super) alignment: usize,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    /// An empty builder with no metadata or tensors and the default GGUF alignment.
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT as usize,
        }
    }

    /// Append a metadata key/value pair; written in call order.
    pub fn meta(&mut self, key: &str, val: MetaValue) -> &mut Self {
        self.metadata.push((key.to_string(), val));
        self
    }

    /// Append a tensor of the given dims, GGML type id and raw encoded bytes.
    /// Panics if a tensor with the same name was already added.
    pub fn tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, data: Vec<u8>) -> &mut Self {
        // Duplicate names would parse (readers resolve first-match) but the
        // file would be malformed; this is a caller bug, so refuse it (a real assert: it
        // must hold in release builds too, and the test below pins it).
        assert!(
            self.tensors.iter().all(|t| t.name != name),
            "GgufBuilder: duplicate tensor name '{name}'"
        );
        self.tensors.push(TensorOut {
            name: name.to_string(),
            dims: dims.to_vec(),
            ggml_type,
            data,
        });
        self
    }

    /// Serialize the header, metadata, tensor infos and aligned data into a GGUF v3 file.
    pub fn into_bytes(self) -> Vec<u8> {
        let mut o = Vec::new();
        // Header.
        put_u32(&mut o, GGUF_MAGIC);
        put_u32(&mut o, 3); // version
        put_u64(&mut o, self.tensors.len() as u64);
        put_u64(&mut o, self.metadata.len() as u64);

        // Metadata KV table, in insertion order.
        for (k, v) in &self.metadata {
            put_str(&mut o, k);
            put_value(&mut o, v);
        }

        // Tensor directory. Offsets are relative to the (aligned) data section
        // and each is aligned to `self.alignment`.
        let mut running = 0usize;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let off = align_up(running, self.alignment);
            offsets.push(off);
            running = off + t.data.len();
        }
        for (t, &off) in self.tensors.iter().zip(&offsets) {
            put_str(&mut o, &t.name);
            put_u32(&mut o, t.dims.len() as u32);
            for &d in &t.dims {
                put_u64(&mut o, d);
            }
            put_u32(&mut o, t.ggml_type);
            put_u64(&mut o, off as u64);
        }

        // Pad to the data-section start, then write each tensor at its offset.
        let data_start = align_up(o.len(), self.alignment);
        o.resize(data_start, 0);
        for (t, &off) in self.tensors.iter().zip(&offsets) {
            let abs = data_start + off;
            if o.len() < abs {
                o.resize(abs, 0); // inter-tensor alignment padding
            }
            o.extend_from_slice(&t.data);
        }
        o
    }

    /// Emit to a file (atomically — see `atomic_write`).
    pub fn write(self, path: &str) -> crate::error::Result<()> {
        let bytes = self.into_bytes();
        atomic_write(path, &bytes)
    }
}

fn put_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn put_str(o: &mut Vec<u8>, s: &str) {
    put_u64(o, s.len() as u64);
    o.extend_from_slice(s.as_bytes());
}

/// Serialize one metadata value (type tag + payload). Mirrors `read_value`.
fn put_value(o: &mut Vec<u8>, v: &MetaValue) {
    match v {
        MetaValue::U8(x) => {
            put_u32(o, VT_U8);
            o.push(*x);
        }
        MetaValue::I8(x) => {
            put_u32(o, VT_I8);
            o.push(*x as u8);
        }
        MetaValue::U16(x) => {
            put_u32(o, VT_U16);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::I16(x) => {
            put_u32(o, VT_I16);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::U32(x) => {
            put_u32(o, VT_U32);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::I32(x) => {
            put_u32(o, VT_I32);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::F32(x) => {
            put_u32(o, VT_F32);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::Bool(x) => {
            put_u32(o, VT_BOOL);
            o.push(*x as u8);
        }
        MetaValue::String(s) => {
            put_u32(o, VT_STRING);
            put_str(o, s);
        }
        MetaValue::U64(x) => {
            put_u32(o, VT_U64);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::I64(x) => {
            put_u32(o, VT_I64);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::F64(x) => {
            put_u32(o, VT_F64);
            o.extend_from_slice(&x.to_le_bytes());
        }
        MetaValue::Array(items) => {
            put_u32(o, VT_ARRAY);
            // Element type tag from the first item (empty arrays default to STRING,
            // matching the common tokenizer-metadata case).
            let elem_tag = items.first().map(value_type_tag).unwrap_or(VT_STRING);
            put_u32(o, elem_tag);
            put_u64(o, items.len() as u64);
            for it in items {
                put_value_payload(o, it);
            }
        }
        MetaValue::Nums(n) => {
            put_u32(o, VT_ARRAY);
            put_u32(o, n.elem_tag());
            put_u64(o, n.len() as u64);
            n.write_payload(o);
        }
    }
}

/// The GGUF type tag for a value (without writing it).
fn value_type_tag(v: &MetaValue) -> u32 {
    match v {
        MetaValue::U8(_) => VT_U8,
        MetaValue::I8(_) => VT_I8,
        MetaValue::U16(_) => VT_U16,
        MetaValue::I16(_) => VT_I16,
        MetaValue::U32(_) => VT_U32,
        MetaValue::I32(_) => VT_I32,
        MetaValue::F32(_) => VT_F32,
        MetaValue::Bool(_) => VT_BOOL,
        MetaValue::String(_) => VT_STRING,
        MetaValue::U64(_) => VT_U64,
        MetaValue::I64(_) => VT_I64,
        MetaValue::F64(_) => VT_F64,
        MetaValue::Array(_) | MetaValue::Nums(_) => VT_ARRAY,
    }
}

/// Write only a value's payload (no type tag) — used for array elements, whose
/// type is written once for the whole array.
fn put_value_payload(o: &mut Vec<u8>, v: &MetaValue) {
    match v {
        MetaValue::U8(x) => o.push(*x),
        MetaValue::I8(x) => o.push(*x as u8),
        MetaValue::U16(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I16(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::U32(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I32(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F32(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Bool(x) => o.push(*x as u8),
        MetaValue::String(s) => put_str(o, s),
        MetaValue::U64(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I64(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F64(x) => o.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Array(_) | MetaValue::Nums(_) => {
            // The reader rejects nested arrays; the writer never produces them.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GgufBuilder;

    use crate::gguf_write::test_support::f32s_to_le_bytes;

    #[test]
    fn builder_roundtrips_through_reader() {
        use crate::gguf::{Gguf, MetaValue, GGML_F32};

        // Two f32 tensors + scalar/string/array metadata.
        let t0: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let t1: Vec<f32> = (0..4).map(|i| -(i as f32)).collect();

        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        b.meta("llama.block_count", MetaValue::U32(2));
        b.meta(
            "tokenizer.ggml.tokens",
            MetaValue::Array(vec![
                MetaValue::String("a".into()),
                MetaValue::String("b".into()),
            ]),
        );
        b.tensor("t0", &[8], GGML_F32, f32s_to_le_bytes(&t0));
        b.tensor("t1", &[4], GGML_F32, f32s_to_le_bytes(&t1));
        let bytes = b.into_bytes();

        let g = Gguf::parse(bytes).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.meta("llama.block_count").unwrap().as_usize(), Some(2));
        // Tensors present with correct dims and decoded values.
        assert_eq!(g.dequantize(g.tensor("t0").unwrap()).unwrap(), t0);
        assert_eq!(g.dequantize(g.tensor("t1").unwrap()).unwrap(), t1);
        // Array metadata preserved.
        let tokens = g.meta("tokenizer.ggml.tokens").unwrap();
        assert_eq!(tokens.array_len(), Some(2));
        assert_eq!(tokens.array_get(0).unwrap().as_str(), Some("a"));
    }

    #[test]
    #[should_panic(expected = "duplicate tensor name 't0'")]
    fn duplicate_tensor_names_are_refused() {
        let mut b = GgufBuilder::new();
        b.tensor("t0", &[1], crate::gguf::GGML_F32, vec![0; 4]);
        b.tensor("t0", &[1], crate::gguf::GGML_F32, vec![0; 4]);
    }

    #[test]
    fn every_metadata_kind_and_array_roundtrips_through_the_reader() {
        use crate::gguf::{Gguf, MetaValue};
        let kinds = [
            ("k.u8", MetaValue::U8(200)),
            ("k.i8", MetaValue::I8(-7)),
            ("k.u16", MetaValue::U16(60000)),
            ("k.i16", MetaValue::I16(-300)),
            ("k.u32", MetaValue::U32(4_000_000_000)),
            ("k.i32", MetaValue::I32(-5)),
            ("k.f32", MetaValue::F32(2.5)),
            ("k.bool", MetaValue::Bool(true)),
            ("k.str", MetaValue::String("hi".into())),
            ("k.u64", MetaValue::U64(1 << 40)),
            ("k.i64", MetaValue::I64(-(1 << 40))),
            ("k.f64", MetaValue::F64(-0.125)),
        ];
        let mut b = GgufBuilder::default();
        for (k, v) in &kinds {
            b.meta(k, v.clone());
            // The same scalar as a one-element array exercises the element writers.
            b.meta(&format!("{k}.arr"), MetaValue::Array(vec![v.clone()]));
        }
        let g = Gguf::parse(b.into_bytes()).unwrap();
        for (k, v) in &kinds {
            assert_eq!(g.meta(k), Some(v), "{k}");
            // Strings stay `Array`; numeric/bool arrays read back as compact `Nums`.
            let a = g.meta(&format!("{k}.arr")).unwrap();
            assert_eq!(a.array_len(), Some(1), "{k}.arr");
            assert_eq!(a.array_get(0).as_ref(), Some(v), "{k}.arr");
            assert_eq!(
                matches!(a, MetaValue::Array(_)),
                matches!(v, MetaValue::String(_)),
                "{k}.arr"
            );
        }
    }

    #[test]
    fn nested_arrays_are_written_but_refused_by_the_reader() {
        use crate::gguf::{Gguf, MetaValue};
        let mut b = GgufBuilder::new();
        b.meta(
            "k.nested",
            MetaValue::Array(vec![MetaValue::Array(vec![MetaValue::U8(1)])]),
        );
        assert!(Gguf::parse(b.into_bytes()).is_err());
    }

    #[test]
    fn write_creates_a_readable_file_atomically() {
        use crate::gguf::{Gguf, MetaValue};
        let path =
            std::env::temp_dir().join(format!("ferrum_builder_write_{}.gguf", std::process::id()));
        let p = path.to_str().unwrap();
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("x".into()));
        b.write(p).unwrap();
        let g = Gguf::from_path(p).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(g.architecture(), Some("x"));
        assert!(!std::path::Path::new(&format!("{p}.tmp")).exists());
    }

    #[test]
    fn align_up_rounds_to_the_alignment_and_zero_means_unaligned() {
        use super::align_up;
        assert_eq!(align_up(7, 0), 7);
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(33, 32), 64);
    }
}
