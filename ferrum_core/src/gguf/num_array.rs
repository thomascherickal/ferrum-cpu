//! Compact storage for GGUF numeric metadata arrays.
//!
//! A `Vec<MetaValue>` costs 32 bytes per element, so a hostile 400 MB file of
//! one-byte elements reached 12.8 GB of RSS. [`NumArray`] stores the elements
//! at their on-disk width (1-8 bytes) and decodes them in one bulk read whose
//! size is checked against the file first.
use crate::error::Result;
use crate::gguf::cursor::{fmt, Cur};
use crate::gguf::format::{
    VT_BOOL, VT_F32, VT_F64, VT_I16, VT_I32, VT_I64, VT_I8, VT_U16, VT_U32, VT_U64, VT_U8,
};
use crate::gguf::MetaValue;

fn le<const N: usize>(b: &[u8]) -> [u8; N] {
    let mut a = [0u8; N];
    a.copy_from_slice(b);
    a
}

macro_rules! num_array {
    ($($V:ident($T:ty) $tag:ident $size:literal, $dec:expr, $enc:expr;)*) => {
        /// A homogeneous numeric (or bool) metadata array at its on-disk width.
        /// Non-empty arrays of these element types are read as `NumArray`;
        /// string arrays stay [`MetaValue::Array`].
        #[derive(Clone, Debug, PartialEq)]
        pub enum NumArray {
            $($V(Vec<$T>),)*
        }

        impl NumArray {
            /// Number of elements.
            pub fn len(&self) -> usize {
                match self { $(NumArray::$V(v) => v.len(),)* }
            }
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
            /// Element `i` as a scalar [`MetaValue`].
            pub fn get(&self, i: usize) -> Option<MetaValue> {
                match self { $(NumArray::$V(v) => v.get(i).map(|&x| MetaValue::$V(x)),)* }
            }
            /// The elements as scalar [`MetaValue`]s.
            pub fn iter(&self) -> impl Iterator<Item = MetaValue> + '_ {
                (0..self.len()).filter_map(|i| self.get(i))
            }
            /// GGUF element type tag.
            pub(crate) fn elem_tag(&self) -> u32 {
                match self { $(NumArray::$V(_) => $tag,)* }
            }
            /// Append the little-endian element payload (no header).
            pub(crate) fn write_payload(&self, o: &mut Vec<u8>) {
                match self {
                    $(NumArray::$V(v) => v.iter().for_each(|x| ($enc)(x, o)),)*
                }
            }
            /// On-disk bytes per element of GGUF type `tag`, `None` if not numeric.
            pub(super) fn elem_size(tag: u32) -> Option<usize> {
                match tag { $($tag => Some($size),)* _ => None }
            }
            /// Decode `count` elements of type `tag`. The caller has checked
            /// `count * elem_size` against the file, so the allocation is bounded.
            pub(super) fn read(c: &mut Cur, tag: u32, count: usize) -> Result<NumArray> {
                match tag {
                    $($tag => {
                        let raw = c.take(count * $size)?;
                        Ok(NumArray::$V(raw.chunks_exact($size).map($dec).collect()))
                    })*
                    other => Err(fmt(&format!("GGUF unknown metadata value type {other}"))),
                }
            }
        }
    };
}

num_array! {
    U8(u8) VT_U8 1, |b: &[u8]| b[0], |x: &u8, o: &mut Vec<u8>| o.push(*x);
    I8(i8) VT_I8 1, |b: &[u8]| b[0] as i8, |x: &i8, o: &mut Vec<u8>| o.push(*x as u8);
    U16(u16) VT_U16 2, |b: &[u8]| u16::from_le_bytes(le(b)),
        |x: &u16, o: &mut Vec<u8>| o.extend_from_slice(&x.to_le_bytes());
    I16(i16) VT_I16 2, |b: &[u8]| i16::from_le_bytes(le(b)),
        |x: &i16, o: &mut Vec<u8>| o.extend_from_slice(&x.to_le_bytes());
    U32(u32) VT_U32 4, |b: &[u8]| u32::from_le_bytes(le(b)),
        |x: &u32, o: &mut Vec<u8>| o.extend_from_slice(&x.to_le_bytes());
    I32(i32) VT_I32 4, |b: &[u8]| i32::from_le_bytes(le(b)),
        |x: &i32, o: &mut Vec<u8>| o.extend_from_slice(&x.to_le_bytes());
    F32(f32) VT_F32 4, |b: &[u8]| f32::from_le_bytes(le(b)),
        |x: &f32, o: &mut Vec<u8>| o.extend_from_slice(&x.to_le_bytes());
    Bool(bool) VT_BOOL 1, |b: &[u8]| b[0] != 0, |x: &bool, o: &mut Vec<u8>| o.push(*x as u8);
    U64(u64) VT_U64 8, |b: &[u8]| u64::from_le_bytes(le(b)),
        |x: &u64, o: &mut Vec<u8>| o.extend_from_slice(&x.to_le_bytes());
    I64(i64) VT_I64 8, |b: &[u8]| i64::from_le_bytes(le(b)),
        |x: &i64, o: &mut Vec<u8>| o.extend_from_slice(&x.to_le_bytes());
    F64(f64) VT_F64 8, |b: &[u8]| f64::from_le_bytes(le(b)),
        |x: &f64, o: &mut Vec<u8>| o.extend_from_slice(&x.to_le_bytes());
}

impl NumArray {
    /// Wrap as a [`MetaValue`]. An empty array becomes `MetaValue::Array(vec![])`,
    /// which is what the reader yields and how the writer has always encoded it.
    pub fn into_meta(self) -> MetaValue {
        if self.is_empty() {
            MetaValue::Array(Vec::new())
        } else {
            MetaValue::Nums(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NumArray;
    use crate::gguf::format::{GGUF_MAGIC, VT_ARRAY, VT_U8};
    use crate::gguf::test_support::{put_str, put_u32, put_u64};
    use crate::gguf::{Gguf, MetaValue};
    use crate::gguf_write::GgufBuilder;

    fn one(key: &str, v: MetaValue) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.meta(key, v);
        b.into_bytes()
    }

    #[test]
    fn typed_arrays_write_the_same_bytes_as_scalar_arrays_and_roundtrip() {
        let cases: Vec<(NumArray, Vec<MetaValue>)> = vec![
            (
                NumArray::U8(vec![0, 255]),
                vec![MetaValue::U8(0), MetaValue::U8(255)],
            ),
            (
                NumArray::I8(vec![-1, 7]),
                vec![MetaValue::I8(-1), MetaValue::I8(7)],
            ),
            (NumArray::U16(vec![65535]), vec![MetaValue::U16(65535)]),
            (NumArray::I16(vec![-300]), vec![MetaValue::I16(-300)]),
            (
                NumArray::U32(vec![4_000_000_000]),
                vec![MetaValue::U32(4_000_000_000)],
            ),
            (
                NumArray::I32(vec![-5, 3, 1]),
                [-5, 3, 1].map(MetaValue::I32).to_vec(),
            ),
            (
                NumArray::F32(vec![-1.5, 0.25]),
                vec![MetaValue::F32(-1.5), MetaValue::F32(0.25)],
            ),
            (
                NumArray::Bool(vec![true, false]),
                vec![MetaValue::Bool(true), MetaValue::Bool(false)],
            ),
            (NumArray::U64(vec![1 << 40]), vec![MetaValue::U64(1 << 40)]),
            (
                NumArray::I64(vec![-(1 << 40)]),
                vec![MetaValue::I64(-(1 << 40))],
            ),
            (NumArray::F64(vec![0.125]), vec![MetaValue::F64(0.125)]),
        ];
        for (n, items) in cases {
            let typed = one("k", MetaValue::Nums(n.clone()));
            assert_eq!(typed, one("k", MetaValue::Array(items.clone())), "{n:?}");
            let g = Gguf::parse(typed).unwrap();
            assert_eq!(g.meta("k"), Some(&MetaValue::Nums(n.clone())));
            assert_eq!(n.iter().collect::<Vec<_>>(), items);
        }
    }

    #[test]
    fn empty_and_string_arrays_keep_the_array_representation() {
        let g = Gguf::parse(one("e", NumArray::I32(vec![]).into_meta())).unwrap();
        assert_eq!(g.meta("e"), Some(&MetaValue::Array(vec![])));
        let strs = MetaValue::Array(vec![
            MetaValue::String("a".into()),
            MetaValue::String("bc".into()),
        ]);
        let g = Gguf::parse(one("s", strs.clone())).unwrap();
        assert_eq!(g.meta("s"), Some(&strs));
    }

    #[test]
    fn tokenizer_scores_and_types_roundtrip_with_slice_accessors() {
        let mut b = GgufBuilder::new();
        b.meta(
            "tokenizer.ggml.scores",
            MetaValue::Nums(NumArray::F32(vec![0.0, -1.5, -2.25])),
        );
        b.meta(
            "tokenizer.ggml.token_type",
            MetaValue::Nums(NumArray::I32(vec![3, 1, 1])),
        );
        let g = Gguf::parse(b.into_bytes()).unwrap();
        assert_eq!(
            g.meta("tokenizer.ggml.scores").unwrap().as_f32_slice(),
            Some(&[0.0, -1.5, -2.25][..])
        );
        let t = g.meta("tokenizer.ggml.token_type").unwrap();
        assert_eq!(t.as_i32_slice(), Some(&[3, 1, 1][..]));
        assert_eq!(
            (t.array_len(), t.array_get(1)),
            (Some(3), Some(MetaValue::I32(1)))
        );
    }

    /// A header that claims `count` one-byte elements and then ends.
    fn hostile(count: u64) -> Vec<u8> {
        let mut o = Vec::new();
        put_u32(&mut o, GGUF_MAGIC);
        put_u32(&mut o, 3);
        put_u64(&mut o, 0);
        put_u64(&mut o, 1);
        put_str(&mut o, "k");
        put_u32(&mut o, VT_ARRAY);
        put_u32(&mut o, VT_U8);
        put_u64(&mut o, count);
        o.extend_from_slice(&[1, 2, 3]);
        o
    }

    #[test]
    fn a_huge_array_count_with_a_short_body_is_rejected_before_allocating() {
        let t = std::time::Instant::now();
        // 100M u8 (and 100M f32: 400 MB) claimed, 3 bytes present.
        let e = Gguf::parse(hostile(100_000_000)).err().unwrap();
        assert!(e.to_string().contains("array count exceeds"), "{e}");
        let e = Gguf::parse(hostile(u64::MAX)).err().unwrap();
        assert!(e.to_string().contains("array count exceeds"), "{e}");
        let mut f32s = hostile(100_000_000);
        let at = f32s.len() - 3 - 12;
        f32s[at..at + 4].copy_from_slice(&6u32.to_le_bytes()); // elem type F32
        assert!(Gguf::parse(f32s).is_err());
        assert!(t.elapsed().as_secs() < 2);
    }

    #[test]
    fn read_rejects_a_tag_that_is_not_numeric() {
        let mut c = crate::gguf::cursor::Cur::new(&[0; 8]);
        let e = NumArray::read(&mut c, crate::gguf::format::VT_STRING, 1)
            .err()
            .unwrap();
        assert!(e.to_string().contains("unknown metadata value type"), "{e}");
    }
}
