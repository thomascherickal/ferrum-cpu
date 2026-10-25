//! Header and tensor-directory parsing, extent validation and offset patching.
use crate::error::{InferError, Result};
use crate::gguf::cursor::{fmt, read_value, to_usize, Cur};
use crate::gguf::format::{align_up, type_nbytes, DEFAULT_ALIGNMENT, GGUF_MAGIC};
use crate::gguf::{MetaValue, TensorInfo};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;

/// `(version, metadata, tensor directory, absolute tensor-data offset)`.
type ParsedHeader = (u32, BTreeMap<String, MetaValue>, Vec<TensorInfo>, usize);

/// Parse the header, metadata table, and tensor directory from a buffer that
/// contains at least the whole header region (it may be a prefix of a larger
/// file of `total` bytes — tensor data after `data_offset` is not required).
pub(super) fn parse_header(b: &[u8], total: usize) -> Result<ParsedHeader> {
    let mut c = Cur::with_total(b, total);
    if c.u32()? != GGUF_MAGIC {
        return Err(fmt("not a GGUF file (bad magic)"));
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        return Err(fmt(&format!(
            "unsupported GGUF version {version} (need 2 or 3)"
        )));
    }
    let tensor_count = to_usize(c.u64()?);
    let kv_count = to_usize(c.u64()?);

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = c.string()?;
        let vtype = c.u32()?;
        let value = read_value(&mut c, vtype)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(4096));
    for _ in 0..tensor_count {
        let name = c.string()?;
        let n_dims = c.u32()? as usize;
        if n_dims > 8 {
            return Err(fmt(&format!(
                "tensor '{name}' has implausible {n_dims} dims"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.u64()?);
        }
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_usize())
        .unwrap_or(to_usize(DEFAULT_ALIGNMENT));
    let data_offset = align_up(c.pos, alignment);
    Ok((version, metadata, tensors, data_offset))
}

/// The one bound every count read from a GGUF (a metadata int, a tensor dim,
/// an array length) must pass before it sizes an allocation or a loop: `n` may
/// not exceed `max`, what the file actually holds (its bytes, tensors or
/// metadata entries), so a few hostile bytes can never request more memory or
/// work than the file itself justifies.
pub(crate) fn bounded(n: usize, max: usize, what: &str) -> Result<usize> {
    if n > max {
        return Err(fmt(&format!(
            "GGUF {what} = {n} exceeds what the file can hold ({max})"
        )));
    }
    Ok(n)
}

/// Every GGML type spends at least one bit per element (the smallest, the
/// IQ1 formats, ~1.56), so `avail` bytes can hold at most `8 · avail` elements.
const MAX_ELEMS_PER_BYTE: usize = 8;

/// Validate the tensor directory against the data section (`avail` bytes),
/// so every tensor dim is bounded by the file:
/// - an empty tensor must be exactly 1-D `[0]` (how Ferrum writes an empty
///   parameter); a zero in a multi-dim shape would let the other dims claim
///   any size while costing no bytes;
/// - no tensor, even one of a type we cannot size, may claim more elements
///   than [`MAX_ELEMS_PER_BYTE`] · `avail`;
/// - sized tensors must lie inside the section and must not overlap. Valid
///   GGUF files never overlap (only alignment padding separates tensors), and
///   a hostile one that points many tensors at the same bytes would otherwise
///   make summed element counts quadratic in the file size.
///
/// Tensors whose byte size cannot be computed (unsupported type, ragged block
/// count) skip the extent check; their element count is still bounded above
/// and they fail cleanly when read.
pub(super) fn check_extents(tensors: &[TensorInfo], avail: usize) -> Result<()> {
    let max_elems = avail.saturating_mul(MAX_ELEMS_PER_BYTE);
    let avail = avail as u64;
    let mut spans: Vec<(u64, u64, &str)> = Vec::new();
    for t in tensors {
        let n = t.num_elements();
        if n == 0 && t.dims != [0] {
            return Err(fmt(&format!(
                "tensor '{}' has empty shape {:?}; only a 1-D [0] tensor may be empty",
                t.name, t.dims
            )));
        }
        bounded(n, max_elems, &format!("tensor '{}' element count", t.name))?;
        let Ok(len) = type_nbytes(t.ggml_type, n) else {
            continue;
        };
        if len == 0 {
            continue;
        }
        let end = t
            .offset
            .checked_add(len as u64)
            .filter(|&e| e <= avail)
            .ok_or_else(|| fmt(&format!("tensor '{}' data runs past EOF", t.name)))?;
        spans.push((t.offset, end, &t.name));
    }
    spans.sort_unstable_by_key(|s| s.0);
    for w in spans.windows(2) {
        if w[1].0 < w[0].1 {
            return Err(fmt(&format!(
                "tensors '{}' and '{}' overlap",
                w[0].2, w[1].2
            )));
        }
    }
    Ok(())
}

/// Whether an error is the "ran off the end of the buffer" kind (vs. a real
/// format error). Used by [`Gguf::open`] to know when to read a larger prefix.
pub(super) fn is_truncation(e: &InferError) -> bool {
    matches!(e, InferError::Format(m) if m.starts_with("GGUF EOF"))
}

/// Read into `buf` until it is full or EOF (handling short/interrupted reads).
/// Returns the number of bytes actually read.
pub(super) fn read_fully(f: &mut File, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match f.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(filled)
}

/// Test helper: overwrite tensor `idx`'s directory offset inside a GGUF byte
/// buffer, to craft overlapping / reordered / out-of-range extents.
#[cfg(test)]
pub(crate) fn patch_tensor_offset(bytes: &mut [u8], idx: usize, off: u64) {
    let pos = {
        let mut c = Cur::new(bytes);
        c.u32().unwrap();
        c.u32().unwrap();
        let (n_tensors, n_kv) = (c.u64().unwrap(), c.u64().unwrap());
        for _ in 0..n_kv {
            c.string().unwrap();
            let vtype = c.u32().unwrap();
            read_value(&mut c, vtype).unwrap();
        }
        assert!((idx as u64) < n_tensors, "tensor index out of range");
        let mut pos = 0;
        for _ in 0..=idx {
            c.string().unwrap();
            for _ in 0..c.u32().unwrap() {
                c.u64().unwrap();
            }
            c.u32().unwrap();
            pos = c.pos;
            c.u64().unwrap();
        }
        pos
    };
    bytes[pos..pos + 8].copy_from_slice(&off.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::{bounded, check_extents, is_truncation, parse_header, patch_tensor_offset};
    use crate::gguf::format::{
        type_nbytes, GGML_F16, GGML_F32, GGML_Q8_0, GGUF_MAGIC, VT_ARRAY, VT_U8,
    };
    use crate::gguf::test_support::{
        extents_err, n_tensor_file, parse_and_open, put_str, put_u32, put_u64, synth_llama, ti,
    };
    use crate::gguf::{Gguf, TensorInfo};

    // A hostile dims table must degrade to a clean error, never an overflow
    // panic (debug builds) or a wrapped size (release builds).
    #[test]
    fn hostile_dims_error_instead_of_panicking() {
        let t = TensorInfo {
            name: "x".into(),
            dims: vec![u64::MAX, 8],
            ggml_type: GGML_F32,
            offset: 0,
        };
        assert_eq!(t.num_elements(), usize::MAX); // saturated, not wrapped
        assert!(type_nbytes(GGML_F32, t.num_elements()).is_err());
        assert!(type_nbytes(GGML_F16, t.num_elements()).is_err());
    }

    #[test]
    fn parse_rejects_truncation() {
        let mut bytes = synth_llama();
        bytes.truncate(bytes.len() - 20); // chop tensor data
                                          // Parse succeeds (directory intact) but dequantizing the last tensor
                                          // must fail because its data ran off the end.
        let g = Gguf::parse(bytes);
        if let Ok(g) = g {
            let last = g.tensors.last().unwrap().clone();
            assert!(g.dequantize(&last).is_err());
        }
    }

    #[test]
    fn check_extents_accepts_valid_layouts_in_any_order() {
        // Three 16-byte F32 tensors, a gap between the 2nd and 3rd, listed out
        // of offset order; the last ends exactly at `avail`.
        let ts = [
            ti("c", &[4], GGML_F32, 48),
            ti("a", &[4], GGML_F32, 0),
            ti("b", &[4], GGML_F32, 16),
        ];
        check_extents(&ts, 64).unwrap();
        check_extents(&[], 0).unwrap();
    }

    #[test]
    fn check_extents_rejects_overlap_and_out_of_range() {
        let a = ti("a", &[4], GGML_F32, 0);
        // One byte of overlap, an identical offset, and containment.
        assert!(extents_err(&[a.clone(), ti("b", &[4], GGML_F32, 15)], 64).contains("overlap"));
        assert!(extents_err(&[a.clone(), ti("b", &[4], GGML_F32, 0)], 64).contains("overlap"));
        assert!(extents_err(
            &[ti("big", &[8], GGML_F32, 0), ti("in", &[1], GGML_F32, 8)],
            64
        )
        .contains("overlap"));
        // Past the data section, and an offset that overflows u64 when the
        // length is added.
        assert!(extents_err(&[ti("a", &[4], GGML_F32, 52)], 64).contains("past EOF"));
        assert!(extents_err(&[ti("a", &[4], GGML_F32, u64::MAX)], 64).contains("past EOF"));
        // Many tensors sharing one offset: the sum exceeds `avail` by far but
        // each is individually in range - still rejected, without allocating.
        let many: Vec<_> = (0..5000)
            .map(|i| ti(&format!("t{i}"), &[16], GGML_F32, 0))
            .collect();
        assert!(extents_err(&many, 64).contains("overlap"));
    }

    #[test]
    fn check_extents_skips_tensors_it_cannot_size() {
        // Unsupported type, ragged quantized count, and the legal 1-D empty
        // tensor: none has a byte extent, so none is extent-checked here,
        // even where they collide with a real tensor or lie past `avail`.
        let ts = [
            ti("real", &[4], GGML_F32, 0),
            ti("unsupported", &[4], 99, 0),
            ti("ragged", &[33], GGML_Q8_0, 0),
            ti("empty", &[0], GGML_F32, 4),
        ];
        check_extents(&ts, 16).unwrap();
        // ...and each still errors when actually read.
        let mut b = crate::gguf_write::GgufBuilder::new();
        b.tensor("x", &[4], 99, vec![0; 16]);
        let g = Gguf::parse(b.into_bytes()).unwrap();
        assert!(g.dequantize(&g.tensors[0]).is_err());
    }

    #[test]
    fn alignment_padding_and_reordered_offsets_still_load() {
        // 5 floats = 20 bytes, so the writer pads 12 bytes between tensors.
        let (bytes, _) = n_tensor_file(3, 5);
        let (p, o) = parse_and_open(&bytes);
        let (p, o) = (p.unwrap(), o.unwrap());
        assert_eq!(p.tensors[1].offset, 32);
        assert_eq!(o.dequantize(&o.tensors[2]).unwrap()[0], 10.0);
        // Reverse the on-disk order (directory order unchanged): t0 <- slot 2.
        let mut re = bytes.clone();
        for (idx, off) in [(0, 64), (1, 32), (2, 0)] {
            patch_tensor_offset(&mut re, idx, off);
        }
        let (p, o) = parse_and_open(&re);
        let (p, o) = (p.unwrap(), o.unwrap());
        assert_eq!(p.dequantize(&p.tensors[0]).unwrap()[0], 10.0);
        assert_eq!(o.dequantize(&o.tensors[2]).unwrap()[0], 0.0);
    }

    #[test]
    fn overlapping_and_oversized_tensors_are_rejected_by_parse_and_open() {
        let (bytes, _) = n_tensor_file(3, 8);
        // Two tensors overlapping (t1 moved onto t0).
        let mut ov = bytes.clone();
        patch_tensor_offset(&mut ov, 1, 0);
        // A tensor partly overlapping its neighbour (t2 starts 4 bytes early).
        let mut part = bytes.clone();
        patch_tensor_offset(&mut part, 2, 60);
        for bad in [ov, part] {
            let (p, o) = parse_and_open(&bad);
            for e in [p.err().unwrap(), o.err().unwrap()] {
                assert!(e.to_string().contains("overlap"), "{e}");
            }
        }
        // Extent past the data section: offset moved out, and tail truncated.
        let mut far = bytes.clone();
        patch_tensor_offset(&mut far, 0, 1 << 20);
        for bad in [far, bytes[..bytes.len() - 1].to_vec()] {
            let (p, o) = parse_and_open(&bad);
            for e in [p.err().unwrap(), o.err().unwrap()] {
                assert!(e.to_string().contains("past EOF"), "{e}");
            }
        }
        // Control: the unmodified file loads.
        let (p, o) = parse_and_open(&bytes);
        p.unwrap();
        o.unwrap();
    }

    #[test]
    fn many_tensors_aliasing_one_buffer_fail_fast_without_allocating() {
        // The quadratic-allocation attack: N tensors each claim the same
        // 4 KiB at offset 0 (N x 4 KiB = 12 MiB claimed from a ~100 KiB file).
        let (mut bytes, start) = n_tensor_file(3000, 1024);
        for i in 0..3000 {
            patch_tensor_offset(&mut bytes, i, 0);
        }
        bytes.truncate(start + 4096);
        assert!(bytes.len() < 200_000, "{}", bytes.len());
        let t = std::time::Instant::now();
        let (p, o) = parse_and_open(&bytes);
        for e in [p.err().unwrap(), o.err().unwrap()] {
            assert!(e.to_string().contains("overlap"), "{e}");
        }
        assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
    }

    #[test]
    fn bounded_accepts_up_to_max_and_rejects_beyond() {
        assert_eq!(bounded(0, 0, "x").unwrap(), 0);
        assert_eq!(bounded(7, 7, "x").unwrap(), 7);
        let e = bounded(8, 7, "block_count").unwrap_err().to_string();
        assert!(e.contains("block_count = 8") && e.contains("(7)"), "{e}");
    }

    #[test]
    fn check_extents_allows_empty_only_as_1d_zero() {
        check_extents(&[ti("e", &[0], GGML_F32, 0)], 0).unwrap();
        // A zero anywhere in a multi-dim (or 1-D-of-0-in-2-D) shape lets the
        // other dims claim anything for free: rejected, whatever the type.
        for dims in [&[0u64, 1 << 40][..], &[1 << 40, 0], &[0, 0], &[4, 0, 4]] {
            for ty in [GGML_F32, GGML_Q8_0, 99] {
                let e = extents_err(&[ti("z", dims, ty, 0)], 64);
                assert!(e.contains("only a 1-D [0]"), "{dims:?}: {e}");
            }
        }
    }

    #[test]
    fn check_extents_bounds_element_count_even_for_unsizable_tensors() {
        // 16 bytes hold at most 128 elements of any GGML type.
        check_extents(&[ti("ok", &[128], 99, 0)], 16).unwrap();
        for t in [
            ti("unsupported", &[129], 99, 0),
            ti("ragged", &[1 << 40], GGML_Q8_0, 0),
            ti("hostile", &[u64::MAX, 8], GGML_F32, 1000),
        ] {
            assert!(extents_err(&[t], 16).contains("exceeds what the file"));
        }
    }

    #[test]
    fn hostile_header_counts_fail_fast() {
        let header = |n_tensors: u64, n_kv: u64| {
            let mut o = Vec::new();
            put_u32(&mut o, GGUF_MAGIC);
            put_u32(&mut o, 3);
            put_u64(&mut o, n_tensors);
            put_u64(&mut o, n_kv);
            o
        };
        // tensor / kv counts far beyond the bytes present.
        assert!(Gguf::parse(header(u64::MAX, 0)).is_err());
        assert!(Gguf::parse(header(0, u64::MAX)).is_err());
        // An array count larger than the file, and a string length past EOF.
        let mut a = header(0, 1);
        put_str(&mut a, "k");
        put_u32(&mut a, VT_ARRAY);
        put_u32(&mut a, VT_U8);
        put_u64(&mut a, 1 << 40);
        let e = Gguf::parse(a).err().unwrap().to_string();
        assert!(e.contains("array count exceeds"), "{e}");
        let mut s = header(0, 1);
        put_u64(&mut s, u64::MAX);
        assert!(Gguf::parse(s).is_err());
        // More than 8 dims.
        let mut d = header(1, 0);
        put_str(&mut d, "t");
        put_u32(&mut d, 9);
        assert!(Gguf::parse(d).err().unwrap().to_string().contains("9 dims"));
    }

    #[test]
    fn patch_tensor_offset_rewrites_only_the_target_entry() {
        // Two 16-byte tensors at offsets 0 and 32; move the second to 16.
        let (mut bytes, _) = n_tensor_file(2, 4);
        patch_tensor_offset(&mut bytes, 1, 16);
        let g = Gguf::parse(bytes).unwrap();
        assert_eq!(
            g.tensors.iter().map(|t| t.offset).collect::<Vec<_>>(),
            [0, 16]
        );
    }

    #[test]
    fn claims_beyond_the_file_are_final_but_within_it_ask_for_more_bytes() {
        // Header prefix whose first key claims `len` bytes (a 1-byte "k" follows).
        let with_key_len = |len: u64| {
            let mut o = Vec::new();
            put_u32(&mut o, GGUF_MAGIC);
            put_u32(&mut o, 3);
            put_u64(&mut o, 0);
            put_u64(&mut o, 1);
            put_u64(&mut o, len);
            o.push(b'k');
            o
        };
        let prefix = with_key_len(1 << 40);
        // The file is only 1 MiB: the claim can never be satisfied, so `open` must not
        // grow its prefix (it would read the whole file) -- the error is not a "need more".
        let e = parse_header(&prefix, 1 << 20).err().unwrap();
        assert!(!is_truncation(&e), "{e}");
        assert!(e.to_string().contains("truncated"), "{e}");
        // The same claim in a file big enough to hold it is a genuine "read more".
        let e = parse_header(&prefix, 1 << 41).err().unwrap();
        assert!(is_truncation(&e), "{e}");
        // Same split for array counts.
        let arr = |count: u64| {
            let mut o = with_key_len(1);
            put_u32(&mut o, VT_ARRAY);
            put_u32(&mut o, VT_U8);
            put_u64(&mut o, count);
            o
        };
        let e = parse_header(&arr(1 << 40), 1 << 20).err().unwrap();
        assert!(
            !is_truncation(&e) && e.to_string().contains("array count exceeds"),
            "{e}"
        );
        let e = parse_header(&arr(5000), 1 << 20).err().unwrap();
        assert!(is_truncation(&e), "{e}");
    }
}
