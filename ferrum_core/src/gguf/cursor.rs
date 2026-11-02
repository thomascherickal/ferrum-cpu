//! Bounds-checked little-endian cursor and metadata value decoding.
use crate::error::{InferError, Result};
use crate::gguf::format::{
    VT_ARRAY, VT_BOOL, VT_F32, VT_F64, VT_I16, VT_I32, VT_I64, VT_I8, VT_STRING, VT_U16, VT_U32,
    VT_U64, VT_U8,
};
use crate::gguf::{MetaValue, NumArray};

/// Little-endian cursor over the GGUF byte stream.
pub(super) struct Cur<'a> {
    pub(super) b: &'a [u8],
    pub(super) pos: usize,
    /// Size of the whole file `b` is (a prefix of). A read past it is final
    /// ("truncated"); a read past `b` but within it means "need a longer prefix".
    pub(super) total: usize,
}

impl<'a> Cur<'a> {
    pub(super) fn new(b: &'a [u8]) -> Self {
        Self::with_total(b, b.len())
    }
    pub(super) fn with_total(b: &'a [u8], total: usize) -> Self {
        Self { b, pos: 0, total }
    }
    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| fmt("offset overflow"))?;
        if end > self.total {
            return Err(fmt(&format!(
                "GGUF truncated at +{}: need {n} bytes, file has {}",
                self.pos, self.total
            )));
        }
        if end > self.b.len() {
            return Err(fmt(&format!("GGUF EOF at +{}: need {n}", self.pos)));
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    pub(super) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub(super) fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    pub(super) fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    pub(super) fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    pub(super) fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }
    pub(super) fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    pub(super) fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    pub(super) fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    pub(super) fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    pub(super) fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    /// A GGUF string: `u64` byte length followed by UTF-8 bytes.
    pub(super) fn string(&mut self) -> Result<String> {
        let n = to_usize(self.u64()?);
        let s = self.take(n)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }
}

/// A file-supplied `u64` as a `usize`, saturating: on a 32-bit target (wasm32) `as usize` would
/// wrap a hostile 2^32 + 5 into 5 and parse garbage instead of failing as "too large".
pub(super) fn to_usize(v: u64) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

pub(super) fn fmt(msg: &str) -> InferError {
    InferError::Format(msg.to_string())
}

pub(super) fn read_value(c: &mut Cur, vtype: u32) -> Result<MetaValue> {
    Ok(match vtype {
        VT_U8 => MetaValue::U8(c.u8()?),
        VT_I8 => MetaValue::I8(c.i8()?),
        VT_U16 => MetaValue::U16(c.u16()?),
        VT_I16 => MetaValue::I16(c.i16()?),
        VT_U32 => MetaValue::U32(c.u32()?),
        VT_I32 => MetaValue::I32(c.i32()?),
        VT_F32 => MetaValue::F32(c.f32()?),
        VT_BOOL => MetaValue::Bool(c.u8()? != 0),
        VT_STRING => MetaValue::String(c.string()?),
        VT_U64 => MetaValue::U64(c.u64()?),
        VT_I64 => MetaValue::I64(c.i64()?),
        VT_F64 => MetaValue::F64(c.f64()?),
        VT_ARRAY => {
            let elem_type = c.u32()?;
            if elem_type == VT_ARRAY {
                return Err(fmt("GGUF nested arrays are not supported"));
            }
            let count = to_usize(c.u64()?);
            if count == 0 {
                return Ok(MetaValue::Array(Vec::new()));
            }
            // Every element takes at least `min` bytes (its width, or the 8-byte
            // length prefix of a string), so the count is checked against the
            // bytes left in the file before anything is allocated: beyond the
            // file is final, beyond the bytes read so far means "read more"
            // (the "GGUF EOF" prefix lets `Gguf::open` grow its prefix).
            let num_size = NumArray::elem_size(elem_type);
            let min = match elem_type {
                VT_STRING => 8,
                _ => num_size
                    .ok_or_else(|| fmt(&format!("GGUF unknown metadata value type {elem_type}")))?,
            };
            let need = count
                .checked_mul(min)
                .ok_or_else(|| fmt("GGUF array count exceeds the file size"))?;
            if need > c.total - c.pos {
                return Err(fmt("GGUF array count exceeds the file size"));
            }
            if need > c.b.len() - c.pos {
                return Err(fmt("GGUF EOF: array count exceeds the bytes read"));
            }
            if num_size.is_some() {
                return Ok(MetaValue::Nums(NumArray::read(c, elem_type, count)?));
            }
            let mut items = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                items.push(read_value(c, elem_type)?);
            }
            MetaValue::Array(items)
        }
        other => return Err(fmt(&format!("GGUF unknown metadata value type {other}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::to_usize;
    use crate::gguf::format::{GGUF_MAGIC, VT_ARRAY};
    use crate::gguf::test_support::{put_str, put_u32, put_u64};
    use crate::gguf::{Gguf, MetaValue};

    #[test]
    fn parse_rejects_nested_arrays_and_unknown_types() {
        // metadata kv: key, value_type=ARRAY, elem_type=ARRAY → rejected.
        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 1);
        put_str(&mut bytes, "k");
        put_u32(&mut bytes, VT_ARRAY);
        put_u32(&mut bytes, VT_ARRAY); // nested
        put_u64(&mut bytes, 1);
        assert!(Gguf::parse(bytes).is_err());
    }

    #[test]
    fn parse_rejects_a_scalar_of_an_unknown_value_type() {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 1);
        put_str(&mut bytes, "k");
        put_u32(&mut bytes, 99);
        let e = Gguf::parse(bytes).err().unwrap();
        assert!(
            e.to_string().contains("unknown metadata value type 99"),
            "{e}"
        );
    }

    #[test]
    fn parse_rejects_an_array_of_an_unknown_element_type() {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 1);
        put_str(&mut bytes, "k");
        put_u32(&mut bytes, VT_ARRAY);
        put_u32(&mut bytes, 99); // no such element type
        put_u64(&mut bytes, 1);
        bytes.push(0);
        let e = Gguf::parse(bytes).err().unwrap();
        assert!(
            e.to_string().contains("unknown metadata value type 99"),
            "{e}"
        );
    }

    #[test]
    fn to_usize_passes_values_through_and_saturates() {
        assert_eq!(to_usize(0), 0);
        assert_eq!(to_usize(12345), 12345);
        assert_eq!(
            to_usize(u64::MAX),
            usize::try_from(u64::MAX).unwrap_or(usize::MAX)
        );
        // as_usize on 64-bit integers uses the same checked conversion.
        assert_eq!(MetaValue::U64(7).as_usize(), Some(7));
        assert_eq!(MetaValue::I64(7).as_usize(), Some(7));
        assert_eq!(MetaValue::I64(-7).as_usize(), None);
    }
}
