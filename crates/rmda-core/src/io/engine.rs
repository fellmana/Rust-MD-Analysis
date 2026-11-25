//! Fast-parsing engine for line-oriented MD formats (design §8.1, §8.2).

use crate::error::{Error, Result};
use std::cell::Cell;
use std::fs;
use std::path::Path;

use super::format::is_gzip;

/// Backing storage for a [`Cursor`].
///
/// An owned buffer is used for small inputs, for files whose metadata reports
/// 0 bytes (`/proc`-style files report `len == 0` yet still have content), and
/// for gzip input, which has no file pages to map — see `Cursor::load`.
/// Non-empty regular files are memory-mapped instead: the bytes are
/// file-backed pages the OS can evict and re-read under memory pressure, rather
/// than unreclaimable heap. That is what keeps peak RSS near the parsed frame
/// instead of *file + frame* when analysing a single very large trajectory.
enum Buf {
    Owned(Box<[u8]>),
    Mapped(memmap2::Mmap),
}

impl Buf {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            Buf::Owned(b) => b,
            Buf::Mapped(m) => m,
        }
    }
}

/// A zero-copy line iterator over an internal byte buffer.
///
/// The whole file is loaded once; `next_line()` returns the next `\n`-terminated
/// slice, `peek_line()` returns it without consuming. Lines are `&[u8]` — the
/// per-format readers convert them to `&str` on demand (ASCII formats).
///
/// The cursor position lives in a `Cell`, so all reading methods take `&self`
/// and the returned line slices do not conflict with building error values
/// (source path / line number) that also borrow the cursor.
pub struct Cursor {
    buf: Buf,
    pos: Cell<usize>,
    line_no: Cell<usize>,
    pub path: String,
}

impl Cursor {
    pub fn from_file(path: &Path) -> Result<Cursor> {
        let buf = Self::load(path)?;
        Ok(Cursor {
            buf,
            pos: Cell::new(0),
            line_no: Cell::new(0),
            path: path.display().to_string(),
        })
    }

    /// Load `path` for reading, mapping it when that saves holding the whole
    /// file on the heap. Falls back to `fs::read` if the map fails (non-regular
    /// file, filesystem that does not support mapping, …). A `.gz` path is
    /// decompressed into an owned buffer instead — there are no file pages to
    /// map, only compressed soup.
    fn load(path: &Path) -> Result<Buf> {
        if is_gzip(path) {
            return Self::decompress(path);
        }
        let meta = fs::metadata(path)
            .map_err(|e| Error::Io(format!("cannot stat {}: {e}", path.display())))?;
        if meta.len() > 0 && meta.is_file() {
            let mapped = (|| -> std::io::Result<memmap2::Mmap> {
                let f = fs::File::open(path)?;
                // SAFETY: the mapping is read-only and we never write through
                // it; the file is an input trajectory that is not truncated
                // while an analysis is reading it.
                unsafe { memmap2::MmapOptions::new().map(&f) }
            })();
            if let Ok(m) = mapped {
                return Ok(Buf::Mapped(m));
            }
        }
        let buf = fs::read(path)
            .map_err(|e| Error::Io(format!("cannot read {}: {e}", path.display())))?;
        Ok(Buf::Owned(buf.into_boxed_slice()))
    }

    /// Decompress a `.gz` file into an owned buffer.
    ///
    /// The compressed stream is decoded through a small IO buffer straight into
    /// the output `Vec` — never `read` whole-then-inflate — so peak RSS is the
    /// decompressed size plus kilobytes, not compressed + decompressed. This is
    /// the one input that must hold the whole file on the heap, the same
    /// fallback the mapper above already takes for non-regular files.
    ///
    /// Every member is read: a multi-member file, as `GzWriter` writes, must
    /// not be cut after its first member.
    fn decompress(path: &Path) -> Result<Buf> {
        use std::io::Read;
        let mut gz = crate::io::gzpar::open(path)?;
        let mut out = Vec::new();
        gz.read_to_end(&mut out)
            .map_err(|e| Error::Io(format!("cannot decompress {}: {e}", path.display())))?;
        Ok(Buf::Owned(out.into_boxed_slice()))
    }

    pub fn from_bytes(buf: &[u8], name: &str) -> Cursor {
        Cursor {
            buf: Buf::Owned(buf.to_vec().into_boxed_slice()),
            pos: Cell::new(0),
            line_no: Cell::new(0),
            path: name.to_string(),
        }
    }

    /// A cursor over an owned buffer whose first line is line `first_line + 1`
    /// of `name` (a frame cut out of a stream keeps its file line numbers).
    pub fn from_vec(buf: Vec<u8>, name: &str, first_line: usize) -> Cursor {
        Cursor {
            buf: Buf::Owned(buf.into_boxed_slice()),
            pos: Cell::new(0),
            line_no: Cell::new(first_line),
            path: name.to_string(),
        }
    }

    pub fn line_no(&self) -> usize {
        self.line_no.get()
    }

    /// The whole buffer.
    pub fn bytes(&self) -> &[u8] {
        self.buf.as_bytes()
    }

    /// Byte offset of the next unread line.
    pub fn pos(&self) -> usize {
        self.pos.get()
    }

    /// The unread bytes.
    pub fn rest(&self) -> &[u8] {
        &self.buf.as_bytes()[self.pos.get()..]
    }

    /// Consume `nbytes` holding `nlines` lines (a block parsed elsewhere).
    pub fn advance(&self, nbytes: usize, nlines: usize) {
        let end = (self.pos.get() + nbytes).min(self.buf.as_bytes().len());
        self.pos.set(end);
        self.line_no.set(self.line_no.get() + nlines);
    }

    /// Jump to byte `pos`, which must be a line start, numbered `line_no`.
    pub fn seek(&self, pos: usize, line_no: usize) {
        self.pos.set(pos.min(self.buf.as_bytes().len()));
        self.line_no.set(line_no);
    }

    /// Consume the next line as a `&[u8]` without the trailing newline.
    pub fn next_line(&self) -> Option<&[u8]> {
        let bytes = self.buf.as_bytes();
        let start = self.pos.get();
        if start >= bytes.len() {
            return None;
        }
        let rel = find_newline(&bytes[start..]);
        let end = match rel {
            Some(r) => start + r,
            None => bytes.len(),
        };
        self.pos.set(if end < bytes.len() { end + 1 } else { end });
        self.line_no.set(self.line_no.get() + 1);
        Some(&bytes[start..end])
    }

    /// Peek at the next line without consuming it.
    pub fn peek_line(&self) -> Option<&[u8]> {
        let bytes = self.buf.as_bytes();
        let start = self.pos.get();
        if start >= bytes.len() {
            return None;
        }
        let rel = find_newline(&bytes[start..]);
        let end = match rel {
            Some(r) => start + r,
            None => bytes.len(),
        };
        Some(&bytes[start..end])
    }

    pub fn eof(&self) -> bool {
        self.pos.get() >= self.buf.as_bytes().len()
    }

    /// Whether the buffer is a memory map rather than an owned read. Test-only,
    /// so the large-file path is actually exercised instead of silently falling
    /// back to `fs::read`.
    #[cfg(test)]
    fn is_mapped(&self) -> bool {
        matches!(self.buf, Buf::Mapped(_))
    }

    /// Consume one line and require it to start with `prefix`.
    pub fn expect_line(&self, prefix: &[u8]) -> Result<&[u8]> {
        let before = self.line_no.get() + 1;
        let line = match self.next_line() {
            Some(l) => l,
            None => {
                return Err(Error::at_line(
                    &self.path,
                    self.line_no(),
                    format!("unexpected EOF, expected {prefix:?}"),
                ))
            }
        };
        if !line.starts_with(prefix) {
            return Err(Error::at_line(
                &self.path,
                before,
                format!(
                    "expected line starting with {:?}, got {:?}",
                    String::from_utf8_lossy(prefix),
                    String::from_utf8_lossy(line)
                ),
            ));
        }
        Ok(line)
    }

    /// Read a single expected token `key: value` lines like `ITEM: TIMESTEP\n3`.
    pub fn read_after(&self, prefix: &[u8]) -> Result<i64> {
        self.expect_line(prefix)?;
        let at = self.line_no.get() + 1;
        let v = self
            .next_line()
            .ok_or_else(|| Error::at_line(&self.path, at, "unexpected EOF while reading value"))?;
        fast_i64(v).ok_or_else(|| {
            Error::at_line(
                &self.path,
                at,
                format!("expected integer, got {:?}", String::from_utf8_lossy(v)),
            )
        })
    }
}

/// Index of the first `\n` in `hay`, or `None`.
///
/// Owned SIMD scan (AVX2 with an SSE2 fallback, no `memchr` dependency).
/// `next_line`/`peek_line` call this once per line,
/// so it is the inner loop of every line-oriented reader: one comparison per
/// 32 bytes instead of one per byte, with no dependency to pull in.
#[inline]
fn find_newline(hay: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the runtime check gates the AVX2 instructions; the loop
            // only loads full 32-byte chunks that are in bounds (`i + 32 <=
            // len`), and the sub-32-byte tail is scanned scalar.
            return unsafe { find_newline_avx2(hay) };
        }
        // SSE2 is part of the x86_64 baseline, so it needs no detection.
        find_newline_sse2(hay)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        find_newline_scalar(hay)
    }
}

/// Byte-at-a-time reference scan: the tail of the SIMD paths and the entire
/// path on non-x86 targets.
fn find_newline_scalar(hay: &[u8]) -> Option<usize> {
    hay.iter().position(|&b| b == b'\n')
}

/// 16-byte SSE2 chunk scan (baseline on x86_64, so no feature check).
#[cfg(target_arch = "x86_64")]
fn find_newline_sse2(hay: &[u8]) -> Option<usize> {
    use std::arch::x86_64::{
        __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
    };
    let mut i = 0;
    // SAFETY: SSE2 is part of the x86_64 baseline, so the target_feature
    // requirement of the intrinsics always holds here; each 16-byte load is
    // guarded by `i + 16 <= hay.len()`, and the rest are register ops.
    unsafe {
        let needle = _mm_set1_epi8(b'\n' as i8);
        while i + 16 <= hay.len() {
            let v = _mm_loadu_si128(hay.as_ptr().add(i) as *const __m128i);
            let mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v, needle));
            if mask != 0 {
                // movemask bit k corresponds to byte k of the block, so the
                // lowest set bit is the first newline in the chunk.
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 16;
        }
    }
    find_newline_scalar(&hay[i..]).map(|p| i + p)
}

/// 32-byte AVX2 chunk scan; only called after a runtime feature check.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_newline_avx2(hay: &[u8]) -> Option<usize> {
    use std::arch::x86_64::*;
    let needle = _mm256_set1_epi8(b'\n' as i8);
    let mut i = 0;
    while i + 32 <= hay.len() {
        let v = _mm256_loadu_si256(hay.as_ptr().add(i) as *const __m256i);
        let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, needle));
        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 32;
    }
    find_newline_scalar(&hay[i..]).map(|p| i + p)
}

/// A whitespace-separated field cursor over a single line's bytes.
#[derive(Clone, Copy)]
pub struct Fields<'a> {
    rest: &'a [u8],
}

impl<'a> Fields<'a> {
    pub fn new(line: &'a [u8]) -> Self {
        Fields { rest: line }
    }

    /// Split a line into fields (single allocation, amortised).
    pub fn all(line: &'a [u8]) -> Vec<&'a [u8]> {
        line.split(|b: &u8| matches!(b, b' ' | b'\t' | b'\r'))
            .filter(|t| !t.is_empty())
            .collect()
    }

    pub fn next_field(&mut self) -> Option<&'a [u8]> {
        let s = self.rest;
        let mut i = 0;
        // skip leading whitespace
        while i < s.len() && (s[i] == b' ' || s[i] == b'\t' || s[i] == b'\r') {
            i += 1;
        }
        let start = i;
        while i < s.len() && !(s[i] == b' ' || s[i] == b'\t' || s[i] == b'\r') {
            i += 1;
        }
        if start == i {
            return None;
        }
        self.rest = &s[i..];
        Some(&s[start..i])
    }

    pub fn next_f64(&mut self) -> Option<f64> {
        self.next_field().and_then(fast_f64)
    }
    pub fn next_i64(&mut self) -> Option<i64> {
        self.next_field().and_then(fast_i64)
    }
    pub fn next_str(&mut self) -> Option<&'a [u8]> {
        self.next_field()
    }
}

/// Trim ASCII whitespace from a byte slice.
#[inline]
fn trim(b: &[u8]) -> &[u8] {
    let mut s = 0;
    let mut e = b.len();
    while s < e && b[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && b[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    &b[s..e]
}

/// Parse of an i64 from bytes, surrounding whitespace allowed (no heap).
pub fn fast_i64(b: &[u8]) -> Option<i64> {
    parse_i64(trim(b))
}

/// Parse of a u32 from bytes, surrounding whitespace allowed.
pub fn fast_u32(b: &[u8]) -> Option<u32> {
    let t = trim(b);
    // `str::parse::<u32>` refuses any `-`, including `-0`.
    if t.first() == Some(&b'-') {
        return None;
    }
    parse_i64(t).and_then(|v| u32::try_from(v).ok())
}

/// Parse of an f64 from bytes, surrounding whitespace allowed. Bit-identical
/// to `str::parse::<f64>` (see [`parse_f64`]).
pub fn fast_f64(b: &[u8]) -> Option<f64> {
    parse_f64(trim(b))
}

/// Exact integer parse of a whole token: what `str::parse::<i64>` accepts
/// (optional `+`/`-`, ASCII digits, no whitespace), with overflow checks.
#[inline]
pub fn parse_i64(b: &[u8]) -> Option<i64> {
    let (neg, digits) = match b.first()? {
        b'-' => (true, &b[1..]),
        b'+' => (false, &b[1..]),
        _ => (false, b),
    };
    if digits.is_empty() {
        return None;
    }
    // Accumulate negatively so `i64::MIN` parses without overflow.
    let mut v: i64 = 0;
    for &c in digits {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v.checked_mul(10)?.checked_sub(d as i64)?;
    }
    if neg {
        Some(v)
    } else {
        v.checked_neg()
    }
}

/// Whether the 8 bytes of `v` are all ASCII digits.
#[inline]
fn is_eight_digits(v: u64) -> bool {
    (v & 0xF0F0_F0F0_F0F0_F0F0)
        | (((v.wrapping_add(0x0606_0606_0606_0606)) & 0xF0F0_F0F0_F0F0_F0F0) >> 4)
        == 0x3333_3333_3333_3333
}

/// The value of 8 ASCII digits loaded little-endian (fast_float's
/// `parse_eight_digits_unrolled`).
#[inline]
fn eight_digits(v: u64) -> u64 {
    eight_digit_values(v - ZEROS)
}

/// The value of 8 digit values (`0..=9` per byte, the first digit in the
/// low byte); fast_float's `parse_eight_digits_unrolled` after its `- '0'`.
#[inline]
fn eight_digit_values(v: u64) -> u64 {
    const MASK: u64 = 0x0000_00FF_0000_00FF;
    const MUL1: u64 = 0x000F_4240_0000_0064; // 100 + (1000000 << 32)
    const MUL2: u64 = 0x0000_2710_0000_0001; // 1 + (10000 << 32)
    let v = v * 10 + (v >> 8);
    ((v & MASK).wrapping_mul(MUL1)).wrapping_add(((v >> 16) & MASK).wrapping_mul(MUL2)) >> 32
}

/// [`take_digits`] without a loop per digit: each 8 bytes are loaded once,
/// and the digits that lead them are counted from a bit mask and valued
/// together, so a run of any length costs `len / 8 + 1` steps and its end
/// steers no branch. Bytes past the end of `b` read as non-digits.
#[inline]
fn take_digits_swar(b: &[u8], mut i: usize, mant: &mut u64) -> usize {
    loop {
        let v = if let Some(c) = b.get(i..i + 8) {
            u64::from_le_bytes(c.try_into().unwrap())
        } else if b.len() >= 8 {
            // The last `r` bytes, shifted down; the zero bytes above them
            // are not digits.
            let r = b.len() - i;
            if r == 0 {
                return i;
            }
            u64::from_le_bytes(b[b.len() - 8..].try_into().unwrap()) >> (8 * (8 - r))
        } else {
            return take_digits(b, i, mant);
        };
        // Digit bytes of `x` hold their value, 0..=9; adding 0x76 to the low
        // 7 bits sets bit 7 for 10..=127, and `| x` for 128..=255.
        let x = v ^ ZEROS;
        let nondigit =
            (((x & 0x7F7F_7F7F_7F7F_7F7F) + 0x7676_7676_7676_7676) | x) & 0x8080_8080_8080_8080;
        if nondigit == 0 {
            *mant = mant
                .wrapping_mul(100_000_000)
                .wrapping_add(eight_digit_values(x));
            i += 8;
            continue;
        }
        let k = (nondigit.trailing_zeros() / 8) as usize;
        if k > 0 {
            // The k digits moved to the top bytes, zeros (leading zeros)
            // below them.
            let top = eight_digit_values(x << (8 * (8 - k)));
            *mant = mant.wrapping_mul(POW10_U64[k]).wrapping_add(top);
        }
        return i + k;
    }
}

/// `b'0'` in every byte of a u64.
const ZEROS: u64 = 0x3030_3030_3030_3030;

/// `10^n` for `n < 8`.
const POW10_U64: [u64; 8] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];

/// Accumulate the digits of `b` from `i` into `mant` (wrapping past 19
/// digits, which the caller detects by count); returns the end index.
#[inline]
fn take_digits(b: &[u8], mut i: usize, mant: &mut u64) -> usize {
    while i + 8 <= b.len() {
        let v = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        if !is_eight_digits(v) {
            break;
        }
        *mant = mant.wrapping_mul(100_000_000).wrapping_add(eight_digits(v));
        i += 8;
    }
    while i < b.len() {
        let d = b[i].wrapping_sub(b'0');
        if d > 9 {
            break;
        }
        *mant = mant.wrapping_mul(10).wrapping_add(d as u64);
        i += 1;
    }
    i
}

/// Exactly rounded float parse of a whole token, bit-identical to
/// `str::parse::<f64>` (design §8.2).
///
/// Fast path: a decimal `m × 10^e` with at most 19 digits in `m` and
/// `|e| ≤ 64` is rounded by [`eisel_lemire`], exactly, whatever the length of
/// `m` (LAMMPS `write_data` prints 17 significant digits). Every other
/// spelling — more than 19 digits, a huge exponent, `inf`/`nan`, or anything
/// malformed — goes to `str::parse`, which is exact too and decides validity.
#[inline]
pub fn parse_f64(b: &[u8]) -> Option<f64> {
    match scan_f64(b, 0) {
        (Some(v), end) if end == b.len() => Some(v),
        _ => parse_f64_slow(b),
    }
}

/// The fast paths of [`parse_f64`] on the number starting at `b[start]`:
/// `(value, end)`, where `end` is one past the last byte of numeric syntax.
/// `value` is `None` when the spelling needs `str::parse`; it is only the
/// token's value when `end` is also the end of the token.
#[inline(always)]
fn scan_f64(b: &[u8], start: usize) -> (Option<f64>, usize) {
    let n = b.len();
    // Common spelling first: `[+-]I[.F]`, at most 19 digits counting
    // leading zeros, no exponent. On it the full scan below reads the same
    // exact mantissa (the leading zeros it skips add nothing to the value)
    // and the same exponent, and rounds them alike, at the cost of that
    // bookkeeping; any other spelling is handed to it.
    let neg = b.get(start) == Some(&b'-');
    let mut i = start + usize::from(neg || b.get(start) == Some(&b'+'));
    let int_start = i;
    let mut mant: u64 = 0;
    while i < n {
        let d = b[i].wrapping_sub(b'0');
        if d > 9 {
            break;
        }
        mant = mant.wrapping_mul(10).wrapping_add(d as u64);
        i += 1;
    }
    let mut digits = i - int_start;
    let mut exp = 0i64;
    if i < n && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        i = take_digits_swar(b, i, &mut mant);
        exp = -((i - frac_start) as i64);
        digits += i - frac_start;
    }
    if (1..=19).contains(&digits) && (i == n || (b[i] | 0x20) != b'e') {
        if let Some(v) = round_f64(mant, exp, neg) {
            return (Some(v), i);
        }
    }
    scan_f64_general(b, start)
}

/// The full scan behind [`scan_f64`]: leading zeros, up to 19 significant
/// digits, an exponent.
#[inline(never)]
fn scan_f64_general(b: &[u8], start: usize) -> (Option<f64>, usize) {
    let n = b.len();
    let mut i = start;
    let neg = match b.get(i) {
        Some(b'-') => {
            i += 1;
            true
        }
        Some(b'+') => {
            i += 1;
            false
        }
        Some(_) => false,
        None => return (None, i),
    };
    let mut mant: u64 = 0;
    // Leading zeros are not significant digits; skip them up front so every
    // digit after them counts towards the 19-digit exactness limit.
    let digits0 = i;
    while i < n && b[i] == b'0' {
        i += 1;
    }
    let mut any = i > digits0;
    let int_start = i;
    i = take_digits(b, i, &mut mant);
    let mut sig = i - int_start;
    let mut exp: i64 = 0;
    if i < n && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        if sig == 0 {
            while i < n && b[i] == b'0' {
                i += 1;
            }
        }
        let sig_start = i;
        i = take_digits(b, i, &mut mant);
        sig += i - sig_start;
        any |= i > frac_start;
        exp = -((i - frac_start) as i64);
    }
    any |= sig > 0;
    if i < n && (b[i] | 0x20) == b'e' {
        i += 1;
        let eneg = match b.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let estart = i;
        let mut e: i64 = 0;
        while i < n {
            let d = b[i].wrapping_sub(b'0');
            if d > 9 {
                break;
            }
            if e < 100_000 {
                e = e * 10 + d as i64;
            }
            i += 1;
        }
        if i == estart {
            return (None, i);
        }
        exp += if eneg { -e } else { e };
    }
    if !any || sig > 19 {
        return (None, i);
    }
    (round_f64(mant, exp, neg), i)
}

/// The f64 nearest `±mant × 10^exp` for an exact `mant` of at most 19
/// digits, or `None` when that needs `str::parse`: the last step of every
/// fast float path, so they all round alike.
///
/// Eisel–Lemire for every mantissa: it is exact on all of them, and taking
/// it alone keeps out a branch on the mantissa's size (Clinger's `m ≤ 2^53`
/// test), which real coordinates, 15 to 17 digits long, miss about half the
/// time.
#[inline(always)]
fn round_f64(mant: u64, exp: i64, neg: bool) -> Option<f64> {
    let v = if mant == 0 {
        0.0
    } else {
        eisel_lemire(exp, mant)?
    };
    Some(if neg { -v } else { v })
}

/// The next whitespace-separated field of `line` parsed as an f64, advancing
/// `*at` past it, in one scan of its bytes. `Err(None)` when the line has no
/// more fields, `Err(Some(field))` when the field is not a number.
#[inline(always)]
pub fn next_f64<'a>(line: &'a [u8], at: &mut usize) -> std::result::Result<f64, Option<&'a [u8]>> {
    let n = line.len();
    let mut p = *at;
    while p < n && is_blank(line[p]) {
        p += 1;
    }
    if p == n {
        *at = p;
        return Err(None);
    }
    if let (Some(v), end) = scan_f64(line, p) {
        if end == n || is_blank(line[end]) {
            *at = end;
            return Ok(v);
        }
    }
    let mut q = p;
    while q < n && !is_blank(line[q]) {
        q += 1;
    }
    *at = q;
    let tok = &line[p..q];
    parse_f64(tok).ok_or(Some(tok))
}

/// [`next_f64`] for an i64 field.
#[inline(always)]
pub fn next_i64<'a>(line: &'a [u8], at: &mut usize) -> std::result::Result<i64, Option<&'a [u8]>> {
    let n = line.len();
    let mut p = *at;
    while p < n && is_blank(line[p]) {
        p += 1;
    }
    if p == n {
        *at = p;
        return Err(None);
    }
    // One pass over `[+-]D` with at most 18 digits, which cannot overflow:
    // the value `parse_i64` gives that token. Anything else is scanned to
    // its end and parsed in full.
    let neg = line[p] == b'-';
    let d0 = p + usize::from(neg || line[p] == b'+');
    let mut q = d0;
    let mut v: i64 = 0;
    while q < n {
        let d = line[q].wrapping_sub(b'0');
        if d > 9 {
            break;
        }
        v = v.wrapping_mul(10).wrapping_add(d as i64);
        q += 1;
    }
    if (1..=18).contains(&(q - d0)) && (q == n || is_blank(line[q])) {
        *at = q;
        return Ok(if neg { -v } else { v });
    }
    let mut q = p;
    while q < n && !is_blank(line[q]) {
        q += 1;
    }
    *at = q;
    let tok = &line[p..q];
    parse_i64(tok).ok_or(Some(tok))
}

/// 128-bit approximations of 5^q for q in [-64, 64], generated by the
/// fast_float `table_generation.py` recipe (truncated for q >= 0, rounded
/// up for q < 0); `[high, low]`.
const POW5_128: [[u64; 2]; 129] = [
    [0xa87fea27a539e9a5, 0x3f2398d747b36224], // 5^-64
    [0xd29fe4b18e88640e, 0x8eec7f0d19a03aad], // 5^-63
    [0x83a3eeeef9153e89, 0x1953cf68300424ac], // 5^-62
    [0xa48ceaaab75a8e2b, 0x5fa8c3423c052dd7], // 5^-61
    [0xcdb02555653131b6, 0x3792f412cb06794d], // 5^-60
    [0x808e17555f3ebf11, 0xe2bbd88bbee40bd0], // 5^-59
    [0xa0b19d2ab70e6ed6, 0x5b6aceaeae9d0ec4], // 5^-58
    [0xc8de047564d20a8b, 0xf245825a5a445275], // 5^-57
    [0xfb158592be068d2e, 0xeed6e2f0f0d56712], // 5^-56
    [0x9ced737bb6c4183d, 0x55464dd69685606b], // 5^-55
    [0xc428d05aa4751e4c, 0xaa97e14c3c26b886], // 5^-54
    [0xf53304714d9265df, 0xd53dd99f4b3066a8], // 5^-53
    [0x993fe2c6d07b7fab, 0xe546a8038efe4029], // 5^-52
    [0xbf8fdb78849a5f96, 0xde98520472bdd033], // 5^-51
    [0xef73d256a5c0f77c, 0x963e66858f6d4440], // 5^-50
    [0x95a8637627989aad, 0xdde7001379a44aa8], // 5^-49
    [0xbb127c53b17ec159, 0x5560c018580d5d52], // 5^-48
    [0xe9d71b689dde71af, 0xaab8f01e6e10b4a6], // 5^-47
    [0x9226712162ab070d, 0xcab3961304ca70e8], // 5^-46
    [0xb6b00d69bb55c8d1, 0x3d607b97c5fd0d22], // 5^-45
    [0xe45c10c42a2b3b05, 0x8cb89a7db77c506a], // 5^-44
    [0x8eb98a7a9a5b04e3, 0x77f3608e92adb242], // 5^-43
    [0xb267ed1940f1c61c, 0x55f038b237591ed3], // 5^-42
    [0xdf01e85f912e37a3, 0x6b6c46dec52f6688], // 5^-41
    [0x8b61313bbabce2c6, 0x2323ac4b3b3da015], // 5^-40
    [0xae397d8aa96c1b77, 0xabec975e0a0d081a], // 5^-39
    [0xd9c7dced53c72255, 0x96e7bd358c904a21], // 5^-38
    [0x881cea14545c7575, 0x7e50d64177da2e54], // 5^-37
    [0xaa242499697392d2, 0xdde50bd1d5d0b9e9], // 5^-36
    [0xd4ad2dbfc3d07787, 0x955e4ec64b44e864], // 5^-35
    [0x84ec3c97da624ab4, 0xbd5af13bef0b113e], // 5^-34
    [0xa6274bbdd0fadd61, 0xecb1ad8aeacdd58e], // 5^-33
    [0xcfb11ead453994ba, 0x67de18eda5814af2], // 5^-32
    [0x81ceb32c4b43fcf4, 0x80eacf948770ced7], // 5^-31
    [0xa2425ff75e14fc31, 0xa1258379a94d028d], // 5^-30
    [0xcad2f7f5359a3b3e, 0x096ee45813a04330], // 5^-29
    [0xfd87b5f28300ca0d, 0x8bca9d6e188853fc], // 5^-28
    [0x9e74d1b791e07e48, 0x775ea264cf55347e], // 5^-27
    [0xc612062576589dda, 0x95364afe032a819e], // 5^-26
    [0xf79687aed3eec551, 0x3a83ddbd83f52205], // 5^-25
    [0x9abe14cd44753b52, 0xc4926a9672793543], // 5^-24
    [0xc16d9a0095928a27, 0x75b7053c0f178294], // 5^-23
    [0xf1c90080baf72cb1, 0x5324c68b12dd6339], // 5^-22
    [0x971da05074da7bee, 0xd3f6fc16ebca5e04], // 5^-21
    [0xbce5086492111aea, 0x88f4bb1ca6bcf585], // 5^-20
    [0xec1e4a7db69561a5, 0x2b31e9e3d06c32e6], // 5^-19
    [0x9392ee8e921d5d07, 0x3aff322e62439fd0], // 5^-18
    [0xb877aa3236a4b449, 0x09befeb9fad487c3], // 5^-17
    [0xe69594bec44de15b, 0x4c2ebe687989a9b4], // 5^-16
    [0x901d7cf73ab0acd9, 0x0f9d37014bf60a11], // 5^-15
    [0xb424dc35095cd80f, 0x538484c19ef38c95], // 5^-14
    [0xe12e13424bb40e13, 0x2865a5f206b06fba], // 5^-13
    [0x8cbccc096f5088cb, 0xf93f87b7442e45d4], // 5^-12
    [0xafebff0bcb24aafe, 0xf78f69a51539d749], // 5^-11
    [0xdbe6fecebdedd5be, 0xb573440e5a884d1c], // 5^-10
    [0x89705f4136b4a597, 0x31680a88f8953031], // 5^-9
    [0xabcc77118461cefc, 0xfdc20d2b36ba7c3e], // 5^-8
    [0xd6bf94d5e57a42bc, 0x3d32907604691b4d], // 5^-7
    [0x8637bd05af6c69b5, 0xa63f9a49c2c1b110], // 5^-6
    [0xa7c5ac471b478423, 0x0fcf80dc33721d54], // 5^-5
    [0xd1b71758e219652b, 0xd3c36113404ea4a9], // 5^-4
    [0x83126e978d4fdf3b, 0x645a1cac083126ea], // 5^-3
    [0xa3d70a3d70a3d70a, 0x3d70a3d70a3d70a4], // 5^-2
    [0xcccccccccccccccc, 0xcccccccccccccccd], // 5^-1
    [0x8000000000000000, 0x0000000000000000], // 5^0
    [0xa000000000000000, 0x0000000000000000], // 5^1
    [0xc800000000000000, 0x0000000000000000], // 5^2
    [0xfa00000000000000, 0x0000000000000000], // 5^3
    [0x9c40000000000000, 0x0000000000000000], // 5^4
    [0xc350000000000000, 0x0000000000000000], // 5^5
    [0xf424000000000000, 0x0000000000000000], // 5^6
    [0x9896800000000000, 0x0000000000000000], // 5^7
    [0xbebc200000000000, 0x0000000000000000], // 5^8
    [0xee6b280000000000, 0x0000000000000000], // 5^9
    [0x9502f90000000000, 0x0000000000000000], // 5^10
    [0xba43b74000000000, 0x0000000000000000], // 5^11
    [0xe8d4a51000000000, 0x0000000000000000], // 5^12
    [0x9184e72a00000000, 0x0000000000000000], // 5^13
    [0xb5e620f480000000, 0x0000000000000000], // 5^14
    [0xe35fa931a0000000, 0x0000000000000000], // 5^15
    [0x8e1bc9bf04000000, 0x0000000000000000], // 5^16
    [0xb1a2bc2ec5000000, 0x0000000000000000], // 5^17
    [0xde0b6b3a76400000, 0x0000000000000000], // 5^18
    [0x8ac7230489e80000, 0x0000000000000000], // 5^19
    [0xad78ebc5ac620000, 0x0000000000000000], // 5^20
    [0xd8d726b7177a8000, 0x0000000000000000], // 5^21
    [0x878678326eac9000, 0x0000000000000000], // 5^22
    [0xa968163f0a57b400, 0x0000000000000000], // 5^23
    [0xd3c21bcecceda100, 0x0000000000000000], // 5^24
    [0x84595161401484a0, 0x0000000000000000], // 5^25
    [0xa56fa5b99019a5c8, 0x0000000000000000], // 5^26
    [0xcecb8f27f4200f3a, 0x0000000000000000], // 5^27
    [0x813f3978f8940984, 0x4000000000000000], // 5^28
    [0xa18f07d736b90be5, 0x5000000000000000], // 5^29
    [0xc9f2c9cd04674ede, 0xa400000000000000], // 5^30
    [0xfc6f7c4045812296, 0x4d00000000000000], // 5^31
    [0x9dc5ada82b70b59d, 0xf020000000000000], // 5^32
    [0xc5371912364ce305, 0x6c28000000000000], // 5^33
    [0xf684df56c3e01bc6, 0xc732000000000000], // 5^34
    [0x9a130b963a6c115c, 0x3c7f400000000000], // 5^35
    [0xc097ce7bc90715b3, 0x4b9f100000000000], // 5^36
    [0xf0bdc21abb48db20, 0x1e86d40000000000], // 5^37
    [0x96769950b50d88f4, 0x1314448000000000], // 5^38
    [0xbc143fa4e250eb31, 0x17d955a000000000], // 5^39
    [0xeb194f8e1ae525fd, 0x5dcfab0800000000], // 5^40
    [0x92efd1b8d0cf37be, 0x5aa1cae500000000], // 5^41
    [0xb7abc627050305ad, 0xf14a3d9e40000000], // 5^42
    [0xe596b7b0c643c719, 0x6d9ccd05d0000000], // 5^43
    [0x8f7e32ce7bea5c6f, 0xe4820023a2000000], // 5^44
    [0xb35dbf821ae4f38b, 0xdda2802c8a800000], // 5^45
    [0xe0352f62a19e306e, 0xd50b2037ad200000], // 5^46
    [0x8c213d9da502de45, 0x4526f422cc340000], // 5^47
    [0xaf298d050e4395d6, 0x9670b12b7f410000], // 5^48
    [0xdaf3f04651d47b4c, 0x3c0cdd765f114000], // 5^49
    [0x88d8762bf324cd0f, 0xa5880a69fb6ac800], // 5^50
    [0xab0e93b6efee0053, 0x8eea0d047a457a00], // 5^51
    [0xd5d238a4abe98068, 0x72a4904598d6d880], // 5^52
    [0x85a36366eb71f041, 0x47a6da2b7f864750], // 5^53
    [0xa70c3c40a64e6c51, 0x999090b65f67d924], // 5^54
    [0xd0cf4b50cfe20765, 0xfff4b4e3f741cf6d], // 5^55
    [0x82818f1281ed449f, 0xbff8f10e7a8921a4], // 5^56
    [0xa321f2d7226895c7, 0xaff72d52192b6a0d], // 5^57
    [0xcbea6f8ceb02bb39, 0x9bf4f8a69f764490], // 5^58
    [0xfee50b7025c36a08, 0x02f236d04753d5b4], // 5^59
    [0x9f4f2726179a2245, 0x01d762422c946590], // 5^60
    [0xc722f0ef9d80aad6, 0x424d3ad2b7b97ef5], // 5^61
    [0xf8ebad2b84e0d58b, 0xd2e0898765a7deb2], // 5^62
    [0x9b934c3b330c8577, 0x63cc55f49f88eb2f], // 5^63
    [0xc2781f49ffcfa6d5, 0x3cbf6b71c76b25fb], // 5^64
];
const POW5_QMIN: i64 = -64;
const POW5_QMAX: i64 = 64;

/// Eisel–Lemire (Lemire 2021, as in fast_float `compute_float`): the exactly
/// rounded f64 of `w × 10^q` for an exact 19-digit-or-shorter `w`, or `None`
/// when `q` is outside the table or the result would be subnormal or
/// infinite. With 55 bits of product precision the result needs no fallback
/// (Mushtak & Lemire, "Fast number parsing without fallback").
#[inline]
fn eisel_lemire(q: i64, w: u64) -> Option<f64> {
    if w == 0 || !(POW5_QMIN..=POW5_QMAX).contains(&q) {
        return None;
    }
    let lz = w.leading_zeros() as i64;
    let w = w << lz;
    let t = POW5_128[(q - POW5_QMIN) as usize];
    let first = (w as u128) * (t[0] as u128);
    let (mut hi, mut lo) = ((first >> 64) as u64, first as u64);
    if hi & 0x1FF == 0x1FF {
        let second_hi = (((w as u128) * (t[1] as u128)) >> 64) as u64;
        let (sum, carry) = lo.overflowing_add(second_hi);
        lo = sum;
        if carry {
            hi = hi.wrapping_add(1);
        }
    }
    let upperbit = (hi >> 63) as i64;
    let shift = upperbit + 9;
    let mut mant = hi >> shift;
    let mut power2 = (((152_170 + 65_536) * q) >> 16) + 63 + upperbit - lz + 1023;
    if power2 <= 0 {
        return None;
    }
    // Exact halfway case: round to even.
    if lo <= 1 && (-4..=23).contains(&q) && mant & 3 == 1 && (mant << shift) == hi {
        mant &= !1;
    }
    mant += mant & 1;
    mant >>= 1;
    if mant >= (2u64 << 52) {
        mant = 1u64 << 52;
        power2 += 1;
    }
    mant &= !(1u64 << 52);
    if power2 >= 0x7FF {
        return None;
    }
    Some(f64::from_bits(mant | ((power2 as u64) << 52)))
}

#[cold]
#[inline(never)]
fn parse_f64_slow(b: &[u8]) -> Option<f64> {
    std::str::from_utf8(b).ok()?.parse::<f64>().ok()
}

#[inline]
fn is_blank(c: u8) -> bool {
    c == b' ' || c == b'\t' || c == b'\r'
}

/// Next whitespace-separated token of `line` at or after `*i`, advancing `*i`
/// past it. The allocation-free field scan of every row parser.
#[inline]
pub fn next_token<'a>(line: &'a [u8], i: &mut usize) -> Option<&'a [u8]> {
    let n = line.len();
    let mut p = *i;
    while p < n && is_blank(line[p]) {
        p += 1;
    }
    let start = p;
    while p < n && !is_blank(line[p]) {
        p += 1;
    }
    *i = p;
    if start == p {
        None
    } else {
        Some(&line[start..p])
    }
}

/// Number of `\n` bytes in `hay`.
pub fn count_newlines(hay: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
        };
        let mut i = 0;
        let mut total = 0usize;
        // SAFETY: SSE2 is baseline on x86_64; every load is bounded by
        // `i + 16 <= hay.len()`.
        unsafe {
            let needle = _mm_set1_epi8(b'\n' as i8);
            while i + 16 <= hay.len() {
                let v = _mm_loadu_si128(hay.as_ptr().add(i) as *const __m128i);
                total +=
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(v, needle)) as u32).count_ones() as usize;
                i += 16;
            }
        }
        total + hay[i..].iter().filter(|&&b| b == b'\n').count()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        hay.iter().filter(|&&b| b == b'\n').count()
    }
}

/// Which lines of a body are rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowRule {
    /// Every line is a row (dump and xyz atom blocks).
    Every,
    /// Blank lines and `#` comment lines are not rows (data-file sections,
    /// which writers such as OVITO pad with blank lines).
    Content,
}

impl RowRule {
    #[inline]
    pub fn is_row(self, line: &[u8]) -> bool {
        match self {
            RowRule::Every => true,
            RowRule::Content => match line.iter().position(|&c| !is_blank(c)) {
                Some(p) => line[p] != b'#',
                None => false,
            },
        }
    }
}

/// `(rows, lines)` in a block of whole lines.
fn count_rows(block: &[u8], rule: RowRule) -> (usize, usize) {
    match rule {
        RowRule::Every => {
            let nl = count_newlines(block);
            let lines = nl + usize::from(!block.is_empty() && *block.last().unwrap() != b'\n');
            (lines, lines)
        }
        RowRule::Content => {
            let (mut rows, mut lines) = (0, 0);
            for line in Lines::new(block) {
                lines += 1;
                rows += usize::from(rule.is_row(line));
            }
            (rows, lines)
        }
    }
}

/// Lines of a byte block, without their `\n`; a final unterminated line
/// counts, an empty block has none.
pub struct Lines<'a> {
    rest: &'a [u8],
}

impl<'a> Lines<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Lines { rest: b }
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = &'a [u8];
    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.rest.is_empty() {
            return None;
        }
        let s = self.rest;
        match find_newline(s) {
            Some(p) => {
                self.rest = &s[p + 1..];
                Some(&s[..p])
            }
            None => {
                self.rest = &[];
                Some(s)
            }
        }
    }
}

/// End of the line containing `at`, i.e. one past its `\n` (or `b.len()`).
fn line_end_from(b: &[u8], at: usize) -> usize {
    if at == 0 || at >= b.len() {
        return at.min(b.len());
    }
    // `at` is a line start already when the byte before it is a newline.
    if b[at - 1] == b'\n' {
        return at;
    }
    match find_newline(&b[at..]) {
        Some(p) => at + p + 1,
        None => b.len(),
    }
}

/// Cut `b` into about `parts` blocks of whole lines.
fn line_blocks(b: &[u8], parts: usize) -> Vec<(usize, usize)> {
    let parts = parts.max(1);
    let step = (b.len() / parts).max(1);
    let mut out = Vec::with_capacity(parts);
    let mut start = 0;
    while start < b.len() {
        let end = line_end_from(b, (start + step).min(b.len()));
        out.push((start, end));
        start = end;
    }
    out
}

/// Below this many bytes a body is parsed on one thread.
const PAR_MIN_BYTES: usize = 1 << 20;

fn par_parts(len: usize) -> usize {
    if len < PAR_MIN_BYTES {
        1
    } else {
        (rayon::current_num_threads() * 4)
            .min(len / (PAR_MIN_BYTES / 4))
            .max(1)
    }
}

/// Byte length and line count of the first `nrows` rows of `body`, through
/// the newline that ends the last row. `None` when `body` holds fewer rows.
pub fn rows_extent(body: &[u8], nrows: usize, rule: RowRule) -> Option<(usize, usize)> {
    row_chunks(body, nrows, rule).map(|(len, lines, _)| (len, lines))
}

/// [`rows_extent`] plus the body already cut into parse chunks.
///
/// Counted in parallel over blocks of whole lines, one bounded window at a
/// time, so a multi-frame file is never scanned far past the frame asked for;
/// the counted blocks are the chunks, so the body is counted once.
pub fn row_chunks(
    body: &[u8],
    nrows: usize,
    rule: RowRule,
) -> Option<(usize, usize, Vec<RowChunk>)> {
    use rayon::prelude::*;
    let mut chunks = Vec::new();
    if nrows == 0 {
        return Some((0, 0, chunks));
    }
    let threads = rayon::current_num_threads().max(1);
    // Window: four blocks per thread, each at least 1 MB.
    let window = threads * 4 * PAR_MIN_BYTES;
    let mut pos = 0usize;
    let mut row = 0usize;
    let mut lines = 0usize;
    while pos < body.len() {
        let wend = line_end_from(body, (pos + window).min(body.len()));
        let win = &body[pos..wend];
        // Small bodies (a few-atom frame) are one block on this thread.
        let parts = if row == 0 && nrows < 4096 {
            1
        } else {
            threads * 4
        };
        let blocks = line_blocks(win, if win.len() < PAR_MIN_BYTES { 1 } else { parts });
        let counts: Vec<(usize, usize)> = if blocks.len() == 1 {
            vec![count_rows(win, rule)]
        } else {
            blocks
                .par_iter()
                .map(|&(a, b)| count_rows(&win[a..b], rule))
                .collect()
        };
        for (&(a, b), &(rows, nl)) in blocks.iter().zip(&counts) {
            if row + rows < nrows {
                chunks.push(RowChunk {
                    start: pos + a,
                    end: pos + b,
                    first_row: row,
                    nrows: rows,
                    first_line: lines,
                });
                row += rows;
                lines += nl;
                continue;
            }
            // The last row ends inside this block: walk it line by line.
            let blk = &win[a..b];
            let (mut off, mut k, mut r) = (0usize, 0usize, 0usize);
            for line in Lines::new(blk) {
                k += 1;
                off += line.len();
                if off < blk.len() {
                    off += 1; // its newline
                }
                if rule.is_row(line) {
                    r += 1;
                    if row + r == nrows {
                        break;
                    }
                }
            }
            chunks.push(RowChunk {
                start: pos + a,
                end: pos + a + off,
                first_row: row,
                nrows: r,
                first_line: lines,
            });
            return Some((pos + a + off, lines + k, chunks));
        }
        pos = wend;
    }
    None
}

/// One thread's share of a body: whole lines `start..end`, holding rows
/// `first_row..first_row + nrows`, starting at line `first_line` of the body.
#[derive(Clone, Copy, Debug)]
pub struct RowChunk {
    pub start: usize,
    pub end: usize,
    pub first_row: usize,
    pub nrows: usize,
    pub first_line: usize,
}

/// Split a body of exactly known rows into chunks for parallel parsing: cut at
/// newlines, count rows per chunk in parallel, prefix-sum the first rows.
pub fn split_rows(body: &[u8], rule: RowRule) -> Vec<RowChunk> {
    use rayon::prelude::*;
    let blocks = line_blocks(body, par_parts(body.len()));
    let counts: Vec<(usize, usize)> = if blocks.len() <= 1 {
        blocks
            .iter()
            .map(|&(a, b)| count_rows(&body[a..b], rule))
            .collect()
    } else {
        blocks
            .par_iter()
            .map(|&(a, b)| count_rows(&body[a..b], rule))
            .collect()
    };
    let mut out = Vec::with_capacity(blocks.len());
    let (mut row, mut line) = (0, 0);
    for (&(start, end), &(nrows, nlines)) in blocks.iter().zip(&counts) {
        out.push(RowChunk {
            start,
            end,
            first_row: row,
            nrows,
            first_line: line,
        });
        row += nrows;
        line += nlines;
    }
    out
}

/// Split `s` (`components` values per row) into one sub-slice per chunk.
pub fn split_by_chunks<'a, T>(
    mut s: &'a mut [T],
    chunks: &[RowChunk],
    components: usize,
) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(chunks.len());
    for c in chunks {
        let (head, tail) = std::mem::take(&mut s).split_at_mut(c.nrows * components);
        out.push(head);
        s = tail;
    }
    out
}

/// A row-parse failure: the body line it happened on and the message.
#[derive(Debug)]
pub struct RowError {
    pub line: usize,
    pub msg: String,
}

/// Parse every row of `body` in parallel, one sink per chunk (the sinks stay
/// with the caller, so per-chunk side results can be merged afterwards).
///
/// `parse(sink, r, line)` receives the chunk-local row index and the row's
/// bytes. Rows map to slots deterministically, and the error reported is that
/// of the lowest failing row, as a serial parse would report it.
pub fn parse_rows<'b, S, F>(
    body: &'b [u8],
    chunks: &[RowChunk],
    sinks: &mut [S],
    rule: RowRule,
    parse: F,
) -> std::result::Result<(), RowError>
where
    S: Send,
    F: Fn(&mut S, usize, &'b [u8]) -> std::result::Result<(), String> + Sync,
{
    use rayon::prelude::*;
    let run = |c: &RowChunk, sink: &mut S| -> std::result::Result<(), RowError> {
        let mut r = 0usize;
        for (k, line) in Lines::new(&body[c.start..c.end]).enumerate() {
            if !rule.is_row(line) {
                continue;
            }
            parse(sink, r, line).map_err(|msg| RowError {
                line: c.first_line + k,
                msg,
            })?;
            r += 1;
        }
        Ok(())
    };
    let results: Vec<std::result::Result<(), RowError>> = if chunks.len() <= 1 {
        chunks
            .iter()
            .zip(sinks.iter_mut())
            .map(|(c, s)| run(c, s))
            .collect()
    } else {
        chunks
            .par_iter()
            .zip(sinks.par_iter_mut())
            .map(|(c, s)| run(c, s))
            .collect()
    };
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_lines() {
        let c = Cursor::from_bytes(b"a\nb\n\nc\n", "t");
        assert_eq!(c.next_line(), Some(&b"a"[..]));
        assert_eq!(c.next_line(), Some(&b"b"[..]));
        assert_eq!(c.next_line(), Some(&b""[..]));
        assert_eq!(c.next_line(), Some(&b"c"[..]));
        assert_eq!(c.next_line(), None);
    }

    #[test]
    fn fields_split() {
        let f: Vec<_> = Fields::all(b"3 1.5 -2.0E+01");
        assert_eq!(f.len(), 3);
        assert_eq!(fast_i64(f[0]), Some(3));
        assert_eq!(fast_f64(f[1]), Some(1.5));
        assert_eq!(fast_f64(f[2]), Some(-2.0E+01));
    }

    #[test]
    fn fast_parsers() {
        assert_eq!(fast_i64(b"  42"), Some(42));
        assert_eq!(fast_i64(b"-7"), Some(-7));
        assert_eq!(fast_i64(b"x"), None);
        assert_eq!(fast_u32(b"65535"), Some(65535));
        assert!((fast_f64(b"1.25e-3").unwrap() - 0.00125).abs() < 1e-12);
    }

    /// A non-empty file is memory-mapped; the mapped path must yield the same
    /// lines as the in-memory path, including a final line with no trailing
    /// newline.
    #[test]
    fn cursor_from_file_reads_mapped_content() {
        let dir = std::env::temp_dir().join("rmda_cursor_mmap");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("lines.txt");
        std::fs::write(&p, b"alpha\nbeta\n\ngamma").unwrap();
        let c = Cursor::from_file(&p).unwrap();
        assert!(c.is_mapped(), "non-empty regular file should be mapped");
        assert_eq!(c.next_line(), Some(&b"alpha"[..]));
        assert_eq!(c.next_line(), Some(&b"beta"[..]));
        assert_eq!(c.next_line(), Some(&b""[..]));
        assert_eq!(c.next_line(), Some(&b"gamma"[..]), "no trailing newline");
        assert_eq!(c.next_line(), None);
        assert_eq!(c.line_no(), 4);
        assert!(c.eof());
    }

    /// A file whose metadata length is 0 is read (not mapped) and is at EOF.
    #[test]
    fn cursor_from_file_empty_file_is_eof() {
        let dir = std::env::temp_dir().join("rmda_cursor_mmap");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("empty.txt");
        std::fs::write(&p, b"").unwrap();
        let c = Cursor::from_file(&p).unwrap();
        assert!(!c.is_mapped());
        assert!(c.eof());
        assert_eq!(c.next_line(), None);
    }

    /// Every newline position — through several 16- and 32-byte chunk
    /// boundaries — must give the same answer from the scalar, SSE2, and AVX2
    /// scanners, both from an allocation start and from an unaligned offset
    /// (a `Cursor` hands them `bytes[start..]` at arbitrary offsets).
    #[test]
    fn find_newline_matches_naive_at_every_position() {
        const LEN: usize = 160;
        let check_all = |hay: &[u8], expect: Option<usize>, ctx: &str| {
            assert_eq!(find_newline_scalar(hay), expect, "scalar {ctx}");
            #[cfg(target_arch = "x86_64")]
            {
                assert_eq!(find_newline_sse2(hay), expect, "sse2 {ctx}");
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: runtime feature check above.
                    let got = unsafe { find_newline_avx2(hay) };
                    assert_eq!(got, expect, "avx2 {ctx}");
                }
            }
        };
        for pos in 0..=LEN {
            let mut buf = vec![b'a'; LEN];
            let expect = if pos < LEN {
                buf[pos] = b'\n';
                Some(pos)
            } else {
                None // no newline at all
            };
            let ctx = format!("pos={pos}");
            check_all(&buf, expect, &ctx);
            // unaligned start: shift the view3 bytes into a padded buffer
            let mut padded = vec![b'x'; 3];
            padded.extend_from_slice(&buf);
            check_all(&padded[3..], expect, &format!("pos={pos} +3 offset"));
        }
        // first byte is a newline, and an empty haystack
        check_all(b"\nabc", Some(0), "first-byte newline");
        check_all(b"", None, "empty");
    }

    /// Line lengths spanning the SIMD chunk sizes exercise the scanner and the
    /// cursor's position/line-number bookkeeping together.
    #[test]
    fn cursor_reads_lines_of_every_length() {
        for len in 0..=70usize {
            let mut data = vec![b'a'; len];
            data.push(b'\n');
            data.extend_from_slice(b"tail");
            let c = Cursor::from_bytes(&data, "t");
            assert_eq!(c.next_line(), Some(&data[..len]), "line len={len}");
            assert_eq!(c.peek_line(), Some(&b"tail"[..]), "peek len={len}");
            assert_eq!(c.next_line(), Some(&b"tail"[..]), "tail len={len}");
            assert_eq!(c.next_line(), None, "eof len={len}");
            assert_eq!(c.line_no(), 2, "line_no len={len}");
        }
    }

    /// Deterministic xorshift for the fuzz tests.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn std_f64(s: &str) -> Option<u64> {
        s.parse::<f64>().ok().map(f64::to_bits)
    }

    fn check_f64(s: &str) {
        let got = parse_f64(s.as_bytes()).map(f64::to_bits);
        assert_eq!(got, std_f64(s), "parse_f64({s:?})");
    }

    /// `parse_f64` must be bit-identical to `str::parse::<f64>`, acceptance
    /// included, on every path: the plain scan, the full scan with
    /// Eisel–Lemire, and the std fallback.
    #[test]
    fn parse_f64_small_mantissas_are_bit_identical_to_std() {
        // Every mantissa below 2^14 with every exponent of Clinger's exact
        // range (|e| <= 22), plus a sweep of 15- to 19-digit mantissas around
        // 2^53, all through Eisel–Lemire.
        for m in 1u64..(1 << 14) {
            for e in -22..=22 {
                check_f64(&format!("{m}e{e}"));
            }
        }
        let mut rng = Rng(0x5eed_1234_abcd);
        for _ in 0..200_000 {
            let m = (1u64 << 53) - (1 << 40) + rng.below(1 << 41);
            let m = m * 10u64.pow(rng.below(4) as u32);
            let e = rng.below(45) as i64 - 22;
            check_f64(&format!("{m}e{e}"));
            let digits = m.to_string();
            let k = 1 + rng.below(digits.len() as u64 - 1) as usize;
            check_f64(&format!("{}.{}", &digits[..k], &digits[k..]));
            check_f64(&format!("-{}.{}", &digits[..k], &digits[k..]));
        }
    }

    #[test]
    fn parse_f64_is_bit_identical_to_std() {
        for s in [
            "0",
            "-0",
            "+0",
            "0.0",
            "-0.0",
            "1",
            "-1",
            "1.",
            ".5",
            "+.5",
            "5e-1",
            "1E5",
            "1e+5",
            "9007199254740993",
            "9007199254740992.5",
            "2.2250738585072014e-308",
            "2.2250738585072011e-308",
            "4.9e-324",
            "1e-400",
            "1e400",
            "1.7976931348623157e308",
            "1.7976931348623159e308",
            "1e23",
            "8.98846567431158e307",
            "0.1",
            "0.3",
            "123456789012345678901234567890",
            "0.000000000000000000000000001",
            "1.5354482839240728",
            "0.019907988738591616",
            "2660.0221962614055",
            "inf",
            "-inf",
            "NaN",
            "infinity",
            "",
            "-",
            "+",
            ".",
            "e5",
            "1e",
            "1e+",
            "1.2.3",
            "1,5",
            " 1",
            "1 ",
            "0x10",
            "1_0",
            "--1",
            "1e5.5",
            "١",
            "00000000000000000000001.5",
            "0.00000000000000000000000000000000000000000000000000000000000000000000001",
        ] {
            check_f64(s);
        }
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let digits = |rng: &mut Rng, n: u64| -> String {
            (0..n)
                .map(|_| char::from(b'0' + rng.below(10) as u8))
                .collect()
        };
        for _ in 0..300_000 {
            // Decimal spelling: 1..21 digits, a point anywhere, an optional
            // exponent, an optional sign.
            let nd = 1 + rng.below(21);
            let mut d = digits(&mut rng, nd);
            if rng.below(3) > 0 {
                let at = rng.below(nd + 1) as usize;
                d.insert(at, '.');
            }
            if rng.below(2) == 0 {
                let e = rng.below(160) as i64 - 80;
                d.push_str(&format!("{}{e}", if rng.below(2) == 0 { 'e' } else { 'E' }));
            }
            match rng.below(3) {
                0 => d.insert(0, '-'),
                1 => d.insert(0, '+'),
                _ => {}
            }
            check_f64(&d);
            // Shortest round-trip and fixed/scientific spellings of random
            // finite doubles, as LAMMPS and Rust writers print them.
            let x = f64::from_bits(rng.next());
            if x.is_finite() {
                check_f64(&format!("{x}"));
                check_f64(&format!("{x:e}"));
                check_f64(&format!("{x:.16e}"));
            }
            let y = (rng.next() as f64 / u64::MAX as f64) * 10f64.powi(rng.below(12) as i32 - 4);
            check_f64(&format!("{y}"));
            check_f64(&format!("{y:.17}"));
            check_f64(&format!("{y:.10}"));
            // Garbage over the float alphabet.
            let n = rng.below(8);
            let g: String = (0..n)
                .map(|_| char::from(b"0123456789.eE+-x "[rng.below(17) as usize]))
                .collect();
            check_f64(&g);
        }
    }

    #[test]
    fn parse_i64_matches_std() {
        for s in [
            "0",
            "-0",
            "+0",
            "7",
            "-7",
            "+7",
            "9223372036854775807",
            "-9223372036854775808",
            "9223372036854775808",
            "-9223372036854775809",
            "",
            "-",
            "+",
            "1.0",
            "1e3",
            " 1",
            "1 ",
            "0x1",
            "١",
            "00012",
        ] {
            assert_eq!(parse_i64(s.as_bytes()), s.parse::<i64>().ok(), "{s:?}");
        }
        let mut rng = Rng(12345);
        for _ in 0..200_000 {
            let n = rng.below(22);
            let g: String = (0..n)
                .map(|_| char::from(b"0123456789+-"[rng.below(12) as usize]))
                .collect();
            assert_eq!(parse_i64(g.as_bytes()), g.parse::<i64>().ok(), "{g:?}");
        }
        assert_eq!(fast_u32(b"-0"), None);
        assert_eq!(fast_u32(b"4294967296"), None);
        assert_eq!(fast_u32(b" 4294967295 "), Some(u32::MAX));
    }

    /// `take_digits_swar` reads the same run, to the same end, as the byte
    /// loop of `take_digits`, wherever it ends relative to the slice end.
    #[test]
    fn take_digits_swar_matches_the_byte_loop() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for _ in 0..200_000 {
            let lead = rng.below(4) as usize;
            let run = rng.below(30) as usize;
            let tail = rng.below(12) as usize;
            let b: Vec<u8> = (0..lead + run + tail)
                .map(|k| {
                    if (lead..lead + run).contains(&k) {
                        b'0' + rng.below(10) as u8
                    } else {
                        b"0123456789 .eE-+\t\r#x\xff\x00/:"[rng.below(21) as usize]
                    }
                })
                .collect();
            let start = rng.next();
            let (mut a, mut w) = (start, start);
            let ea = take_digits(&b, lead, &mut a);
            let ew = take_digits_swar(&b, lead, &mut w);
            assert_eq!(
                (ea, a),
                (ew, w),
                "{:?} from {lead}",
                String::from_utf8_lossy(&b)
            );
        }
    }

    /// Field by field, `next_f64` and `next_i64` give what a whole-token
    /// parse of each field gives — value, rejection and the field reported —
    /// on rows of numbers, near-numbers and junk between any blanks.
    #[test]
    fn next_fields_match_whole_token_parses() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let alphabet = b"0123456789.eE+-#x";
        for _ in 0..100_000 {
            let nf = rng.below(6);
            let mut line = Vec::new();
            let mut toks = Vec::new();
            for _ in 0..nf {
                for _ in 0..rng.below(3) {
                    line.push(b" \t\r"[rng.below(3) as usize]);
                }
                if !line.is_empty() || rng.below(2) == 0 {
                    line.push(b' ');
                }
                let tok: Vec<u8> = match rng.below(4) {
                    0 => format!("{}", rng.next() as i64 >> rng.below(64)).into_bytes(),
                    1 => {
                        let x = f64::from_bits(rng.next());
                        let y = (rng.next() as f64 / u64::MAX as f64) * 1e3 - 500.0;
                        let v = if x.is_finite() && rng.below(4) == 0 {
                            x
                        } else {
                            y
                        };
                        match rng.below(3) {
                            0 => format!("{v}"),
                            1 => format!("{v:.16}"),
                            _ => format!("{v:e}"),
                        }
                        .into_bytes()
                    }
                    _ => (0..1 + rng.below(22))
                        .map(|_| alphabet[rng.below(alphabet.len() as u64) as usize])
                        .collect(),
                };
                line.extend_from_slice(&tok);
                toks.push(tok);
            }
            for _ in 0..rng.below(3) {
                line.push(b" \t\r"[rng.below(3) as usize]);
            }
            let (mut af, mut ai) = (0usize, 0usize);
            for tok in &toks {
                let t = tok.as_slice();
                match next_f64(&line, &mut af) {
                    Ok(v) => assert_eq!(Some(v.to_bits()), parse_f64(t).map(f64::to_bits)),
                    Err(e) => assert_eq!((e, parse_f64(t)), (Some(t), None)),
                }
                match next_i64(&line, &mut ai) {
                    Ok(v) => assert_eq!(Some(v), parse_i64(t), "{:?}", String::from_utf8_lossy(t)),
                    Err(e) => assert_eq!((e, parse_i64(t)), (Some(t), None)),
                }
            }
            assert_eq!(next_f64(&line, &mut af), Err(None));
            assert_eq!(next_i64(&line, &mut ai), Err(None));
        }
    }

    /// Row extents and chunking agree with a naive line walk for both rules,
    /// at every row count, including bodies with no final newline.
    #[test]
    fn rows_extent_and_split_match_a_naive_walk() {
        let body = b"1 a\n\n# c\n2 b\n   \n3 c\n4 d";
        for rule in [RowRule::Every, RowRule::Content] {
            let lines: Vec<&[u8]> = Lines::new(body).collect();
            let mut rows = 0;
            let mut off = 0;
            for (k, l) in lines.iter().enumerate() {
                off += l.len() + usize::from(off + l.len() < body.len());
                if rule.is_row(l) {
                    rows += 1;
                    assert_eq!(
                        rows_extent(body, rows, rule),
                        Some((off, k + 1)),
                        "{rule:?} {rows}"
                    );
                }
            }
            assert_eq!(rows_extent(body, rows + 1, rule), None);
            let chunks = split_rows(body, rule);
            assert_eq!(chunks.iter().map(|c| c.nrows).sum::<usize>(), rows);
        }
        // A large body takes the parallel path.
        let mut big = Vec::new();
        for i in 0..300_000 {
            big.extend_from_slice(format!("{i} 1 2.5 3.5 4.5\n").as_bytes());
            if i % 1000 == 0 {
                big.extend_from_slice(b"\n# note\n");
            }
        }
        let (len, nl) = rows_extent(&big, 300_000, RowRule::Content).unwrap();
        assert_eq!(len, big.len());
        assert_eq!(nl, count_newlines(&big[..len]));
        let chunks = split_rows(&big[..len], RowRule::Content);
        assert!(chunks.len() > 1, "expected a parallel split");
        let mut next = 0;
        for c in &chunks {
            assert_eq!(c.first_row, next);
            next += c.nrows;
        }
        assert_eq!(next, 300_000);
    }
}
