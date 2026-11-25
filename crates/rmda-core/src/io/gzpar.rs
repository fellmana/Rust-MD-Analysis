//! Parallel gzip input (design §8.1 `ByteSource::Gzip`).
//!
//! [`open`] returns a reader whose bytes are exactly those of flate2's
//! `MultiGzDecoder` over the same file — every member inflated by the same
//! `miniz_oxide` decoder, CRC-32 and length checked — but inflated on the rayon
//! pool when the file allows it. Parsing reads those bytes, so it is
//! bit-identical to the serial path by construction. Three ways in:
//!
//! - **Indexed** (`rmda index FILE.gz` wrote a current `FILE.gz.rmdaidx`): the
//!   index holds zran-style access points — a deflate block boundary, the
//!   unconsumed bits of its last input byte, and the 32 KiB of output before
//!   it — about every [`SPACING`] bytes of output. Each span between two points
//!   is inflated on its own worker; any gzip file, one member or many.
//! - **Members** (no index, a file of small members, as [`GzWriter`] and
//!   bgzip write): members are found by speculatively inflating from every
//!   gzip header signature ahead of the current member, in parallel. A
//!   speculative result is used only when its offset is exactly where the
//!   previous member ended, so a signature that merely occurs inside the
//!   compressed data costs work, never correctness.
//! - **Serial**: the first member, a member over [`MEMBER_CAP`] bytes and
//!   anything the parallel paths reject (bad header, CRC mismatch, trailing
//!   garbage) go through flate2 itself, so errors are flate2's.
//!
//! `--threads 1`, a reader inside a rayon worker, and a file that cannot be
//! mapped use `MultiGzDecoder` unchanged.
//!
//! [`GzWriter`]: crate::io::writer::GzWriter

use crate::error::{Error, Result};
use flate2::Crc;
use miniz_oxide::inflate::core::inflate_flags::{
    TINFL_FLAG_STOP_ON_BLOCK_BOUNDARY, TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF,
};
use miniz_oxide::inflate::core::{decompress, BlockBoundaryState, DecompressorOxide};
use miniz_oxide::inflate::TINFLStatus;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Read};
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

/// Deflate's history window.
const WINDOW: usize = 32 << 10;
/// Output bytes between index access points.
pub const SPACING: u64 = 8 << 20;
/// A member larger than this is inflated serially, not speculatively.
pub const MEMBER_CAP: usize = 64 << 20;
/// Bytes handed out per serial refill.
const SERIAL_PIECE: usize = 4 << 20;

type Map = Arc<memmap2::Mmap>;

fn io_err(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into())
}

// ---------------------------------------------------------------- headers

/// Offset of the deflate data of the gzip member whose header starts at
/// `at`, or `None` when no valid header starts there (RFC 1952 §2.3; reserved
/// flag bits and a wrong header CRC are rejected, as flate2 does).
pub(crate) fn parse_header(data: &[u8], at: usize) -> Option<usize> {
    let h = data.get(at..at.checked_add(10)?)?;
    if h[0] != 0x1f || h[1] != 0x8b || h[2] != 8 || h[3] & 0xe0 != 0 {
        return None;
    }
    let flg = h[3];
    let mut p = at + 10;
    if flg & 4 != 0 {
        let x = data.get(p..p + 2)?;
        p += 2 + usize::from(u16::from_le_bytes([x[0], x[1]]));
    }
    for bit in [8u8, 16] {
        if flg & bit != 0 {
            p += data.get(p..)?.iter().position(|&b| b == 0)? + 1;
        }
    }
    if flg & 2 != 0 {
        let c = data.get(p..p + 2)?;
        let mut crc = Crc::new();
        crc.update(data.get(at..p)?);
        if (crc.sum() & 0xffff) as u16 != u16::from_le_bytes([c[0], c[1]]) {
            return None;
        }
        p += 2;
    }
    (p <= data.len()).then_some(p)
}

/// The member trailer at `at`: (CRC-32, ISIZE).
fn trailer(data: &[u8], at: usize) -> Option<(u32, u32)> {
    let t = data.get(at..at + 8)?;
    Some((
        u32::from_le_bytes([t[0], t[1], t[2], t[3]]),
        u32::from_le_bytes([t[4], t[5], t[6], t[7]]),
    ))
}

// ---------------------------------------------------------------- index

/// One access point: inflation can restart here with only this state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Point {
    /// Input byte after the last (possibly partly) consumed one.
    pub in_byte: usize,
    /// Output offset (in the whole decompressed stream).
    pub out: u64,
    /// Unconsumed bits of the last consumed byte (`0..=7`) and their value.
    pub num_bits: u8,
    pub bit_buf: u8,
    /// Up to 32 KiB of the member's output before this point.
    pub window: Vec<u8>,
}

/// Access points of one gzip file (design §8.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GzIndex {
    pub total_out: u64,
    /// Sorted by `out`; `points[0]` is the start of the first member.
    pub points: Vec<Point>,
}

const FLAGS: u32 = TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF | TINFL_FLAG_STOP_ON_BLOCK_BOUNDARY;

/// Inflate all of `data` serially, checking every member, and record an
/// access point at the first block boundary every `spacing` output bytes.
/// Points sit only at boundaries inside a member (never after a final block),
/// so a span always ends inside a member and owns every trailer before it.
pub fn build_index(data: &[u8], spacing: u64) -> std::result::Result<GzIndex, String> {
    let mut buf = vec![0u8; WINDOW + (4 << 20)];
    let mut points: Vec<Point> = Vec::new();
    let mut at = 0usize;
    let mut total = 0u64;
    let mut last = 0u64;
    loop {
        let mut inp =
            parse_header(data, at).ok_or(format!("no gzip member header at byte {at}"))?;
        if points.is_empty() {
            points.push(Point {
                in_byte: inp,
                out: 0,
                num_bits: 0,
                bit_buf: 0,
                window: Vec::new(),
            });
        }
        let mut d = DecompressorOxide::new();
        let mut crc = Crc::new();
        let mut pos = 0usize;
        loop {
            if pos == buf.len() {
                buf.copy_within(pos - WINDOW..pos, 0);
                pos = WINDOW;
            }
            let (st, nin, nout) = decompress(&mut d, &data[inp..], &mut buf, pos, FLAGS);
            crc.update(&buf[pos..pos + nout]);
            inp += nin;
            pos += nout;
            total += nout as u64;
            match st {
                TINFLStatus::BlockBoundary => {
                    if total - last >= spacing {
                        let s = d.block_boundary_state().ok_or("no block boundary state")?;
                        points.push(Point {
                            in_byte: inp,
                            out: total,
                            num_bits: s.num_bits,
                            bit_buf: s.bit_buf,
                            window: buf[pos.saturating_sub(WINDOW)..pos].to_vec(),
                        });
                        last = total;
                    }
                }
                TINFLStatus::HasMoreOutput => {}
                TINFLStatus::Done => break,
                st => return Err(format!("corrupt deflate data near byte {inp} ({st:?})")),
            }
        }
        let (c, n) = trailer(data, inp).ok_or(format!("truncated member trailer at byte {inp}"))?;
        if c != crc.sum() || n != crc.amount() {
            return Err(format!(
                "CRC or length mismatch in the member ending at byte {inp}"
            ));
        }
        at = inp + 8;
        if at == data.len() {
            return Ok(GzIndex {
                total_out: total,
                points,
            });
        }
    }
}

const MAGIC: &[u8; 8] = b"MDAGZIX1";

impl GzIndex {
    /// Serialise with the file's `(bytes, mtime ns)` stamp. Windows are
    /// stored deflated.
    pub fn to_bytes(&self, stamp: (u64, u128)) -> Vec<u8> {
        use std::io::Write;
        let mut o = Vec::new();
        o.extend_from_slice(MAGIC);
        o.extend_from_slice(&stamp.0.to_le_bytes());
        o.extend_from_slice(&stamp.1.to_le_bytes());
        o.extend_from_slice(&self.total_out.to_le_bytes());
        o.extend_from_slice(&(self.points.len() as u64).to_le_bytes());
        for p in &self.points {
            o.extend_from_slice(&(p.in_byte as u64).to_le_bytes());
            o.extend_from_slice(&p.out.to_le_bytes());
            o.push(p.num_bits);
            o.push(p.bit_buf);
            let mut enc =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(&p.window).expect("in-memory write");
            let w = enc.finish().expect("in-memory write");
            o.extend_from_slice(&(p.window.len() as u32).to_le_bytes());
            o.extend_from_slice(&(w.len() as u32).to_le_bytes());
            o.extend_from_slice(&w);
        }
        o
    }

    /// Parse an index written by [`GzIndex::to_bytes`] for a file whose stamp
    /// is `stamp`; `None` when it is not one or is stale.
    pub fn from_bytes(b: &[u8], stamp: (u64, u128)) -> Option<GzIndex> {
        let mut p = 0usize;
        let mut take = |n: usize| -> Option<&[u8]> {
            let s = b.get(p..p + n)?;
            p += n;
            Some(s)
        };
        let u64_ = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
        let u32_ = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
        if take(8)? != MAGIC
            || u64_(take(8)?) != stamp.0
            || u128::from_le_bytes(take(16)?.try_into().ok()?) != stamp.1
        {
            return None;
        }
        let total_out = u64_(take(8)?);
        let n = u64_(take(8)?) as usize;
        let mut points = Vec::with_capacity(n.min(1 << 20));
        for _ in 0..n {
            let in_byte = u64_(take(8)?) as usize;
            let out = u64_(take(8)?);
            let nb = take(2)?;
            let wlen = u32_(take(4)?) as usize;
            let clen = u32_(take(4)?) as usize;
            let mut window = Vec::with_capacity(wlen);
            flate2::read::DeflateDecoder::new(take(clen)?)
                .read_to_end(&mut window)
                .ok()?;
            if window.len() != wlen || nb[0] > 7 || wlen > WINDOW {
                return None;
            }
            points.push(Point {
                in_byte,
                out,
                num_bits: nb[0],
                bit_buf: nb[1],
                window,
            });
        }
        let sorted = points.windows(2).all(|w| w[0].out <= w[1].out);
        (p == b.len() && sorted && !points.is_empty() && points.last()?.out <= total_out)
            .then_some(GzIndex { total_out, points })
    }
}

// ---------------------------------------------------------------- workers

/// A piece of one member's output: its CRC, and the member's trailer when
/// the member ends at the end of the piece.
struct Piece {
    crc: Crc,
    end: Option<(u32, u32)>,
}

/// Inflated bytes `buf[skip..]` (`buf[..skip]` is the window).
struct Chunk {
    buf: Vec<u8>,
    skip: usize,
    pieces: Vec<Piece>,
}

/// Inflate from access point `p` up to the point `stop` (input byte, output
/// offset), or to the end of the file when `stop` is `None` (then `end_out`
/// is the file's total output).
fn decode_span(
    data: &[u8],
    p: &Point,
    stop: Option<usize>,
    end_out: u64,
) -> std::io::Result<Chunk> {
    let bad = |what: &str| io_err(format!("gzip index does not match the file ({what})"));
    let len = usize::try_from(end_out - p.out).map_err(|_| bad("span"))?;
    let skip = p.window.len();
    let mut buf = vec![0u8; skip + len];
    buf[..skip].copy_from_slice(&p.window);
    let mut d = DecompressorOxide::from_block_boundary_state(&BlockBoundaryState {
        num_bits: p.num_bits,
        bit_buf: p.bit_buf,
        ..Default::default()
    });
    let mut inp = p.in_byte;
    let mut pos = skip;
    let mut crc = Crc::new();
    let mut pieces = Vec::new();
    loop {
        let src = data.get(inp..).ok_or_else(|| bad("offset"))?;
        let (st, nin, nout) = decompress(&mut d, src, &mut buf, pos, FLAGS);
        crc.update(&buf[pos..pos + nout]);
        inp += nin;
        pos += nout;
        match st {
            TINFLStatus::BlockBoundary => {
                if stop == Some(inp) && pos == buf.len() {
                    break;
                }
            }
            TINFLStatus::Done => {
                let t = trailer(data, inp).ok_or_else(|| bad("trailer"))?;
                pieces.push(Piece {
                    crc: std::mem::take(&mut crc),
                    end: Some(t),
                });
                inp += 8;
                if inp == data.len() {
                    if stop.is_none() && pos == buf.len() {
                        break;
                    }
                    return Err(bad("early end"));
                }
                inp = parse_header(data, inp).ok_or_else(|| bad("member header"))?;
                d = DecompressorOxide::new();
            }
            _ => return Err(bad("inflate")),
        }
    }
    pieces.push(Piece { crc, end: None });
    Ok(Chunk { buf, skip, pieces })
}

/// What speculative inflation from one offset found.
enum Member {
    /// A whole member, CRC and length checked, ending at byte `end`.
    Ok { bytes: Vec<u8>, end: usize },
    /// Larger than [`MEMBER_CAP`].
    TooBig,
    /// No valid member starts here.
    Bad,
}

fn decode_member(data: &[u8], at: usize, cap: usize) -> Member {
    let Some(mut inp) = parse_header(data, at) else {
        return Member::Bad;
    };
    let mut buf = vec![0u8; (1 << 20).min(cap)];
    let mut pos = 0usize;
    let mut d = DecompressorOxide::new();
    loop {
        let (st, nin, nout) = decompress(
            &mut d,
            &data[inp..],
            &mut buf,
            pos,
            TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF,
        );
        inp += nin;
        pos += nout;
        match st {
            TINFLStatus::Done => break,
            TINFLStatus::HasMoreOutput => {
                if buf.len() >= cap {
                    return Member::TooBig;
                }
                buf.resize((buf.len() * 2).min(cap), 0);
            }
            _ => return Member::Bad,
        }
    }
    buf.truncate(pos);
    let mut crc = Crc::new();
    crc.update(&buf);
    match trailer(data, inp) {
        Some((c, n)) if c == crc.sum() && n == crc.amount() => Member::Ok {
            bytes: buf,
            end: inp + 8,
        },
        _ => Member::Bad,
    }
}

/// Jobs on the rayon pool, results collected by key.
struct Pipe<T> {
    tx: Sender<(usize, T)>,
    rx: Receiver<(usize, T)>,
    ready: HashMap<usize, T>,
    busy: HashSet<usize>,
}

impl<T: Send + 'static> Pipe<T> {
    fn new() -> Self {
        let (tx, rx) = channel();
        Pipe {
            tx,
            rx,
            ready: HashMap::new(),
            busy: HashSet::new(),
        }
    }

    fn has(&self, k: usize) -> bool {
        self.busy.contains(&k) || self.ready.contains_key(&k)
    }

    fn submit(&mut self, k: usize, f: impl FnOnce() -> T + Send + 'static) {
        self.busy.insert(k);
        let tx = self.tx.clone();
        rayon::spawn(move || {
            let _ = tx.send((k, f()));
        });
    }

    fn wait(&mut self, k: usize) -> T {
        loop {
            if let Some(v) = self.ready.remove(&k) {
                return v;
            }
            let (got, v) = self.rx.recv().expect("gzip worker vanished");
            self.busy.remove(&got);
            self.ready.insert(got, v);
        }
    }

    /// Drop finished results with keys below `k`.
    fn prune(&mut self, k: usize) {
        while let Ok((got, v)) = self.rx.try_recv() {
            self.busy.remove(&got);
            self.ready.insert(got, v);
        }
        self.ready.retain(|&key, _| key >= k);
    }
}

// ---------------------------------------------------------------- reader

/// A `BufRead` over the mapped file from `pos`.
struct MapReader {
    map: Map,
    pos: usize,
}

impl Read for MapReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let src = &self.map[self.pos.min(self.map.len())..];
        let n = src.len().min(out.len());
        out[..n].copy_from_slice(&src[..n]);
        self.pos += n;
        Ok(n)
    }
}

impl BufRead for MapReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        Ok(&self.map[self.pos.min(self.map.len())..])
    }
    fn consume(&mut self, n: usize) {
        self.pos += n;
    }
}

enum Mode {
    /// One member through flate2; `out` bytes so far.
    Serial {
        dec: Box<flate2::bufread::GzDecoder<MapReader>>,
        out: usize,
    },
    /// Speculative members; the next member starts at `p`, signatures are
    /// searched from `scan`.
    Members {
        p: usize,
        scan: usize,
        pipe: Pipe<Member>,
    },
    Indexed {
        idx: Arc<GzIndex>,
        next: usize,
        pipe: Pipe<std::io::Result<Chunk>>,
        crc: Crc,
    },
    Done,
}

/// The parallel reader (see the module docs).
pub struct ParGz {
    map: Map,
    seg: Vec<u8>,
    at: usize,
    mode: Mode,
    /// Jobs in flight.
    depth: usize,
    /// Leftover pipe of a Members phase, reused when members resume.
    spare: Option<Pipe<Member>>,
    /// Largest member inflated speculatively ([`MEMBER_CAP`]).
    cap: usize,
    /// Compressed bytes below this have been handed back to the kernel.
    released: usize,
}

/// Compressed bytes consumed between `MADV_DONTNEED` calls.
const RELEASE: usize = 64 << 20;

/// A reader of the decompressed bytes of `path`, parallel when possible.
pub fn open(path: &Path) -> Result<Box<dyn Read + Send>> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Io(format!("cannot read {}: {e}", path.display())))?;
    let threads = rayon::current_num_threads();
    let serial = || -> Box<dyn Read + Send> {
        Box::new(flate2::read::MultiGzDecoder::new(
            std::io::BufReader::with_capacity(1 << 20, file.try_clone().expect("dup fd")),
        ))
    };
    if threads <= 1 || rayon::current_thread_index().is_some() {
        return Ok(serial());
    }
    // SAFETY: read-only mapping; a file truncated underneath us is the same
    // hazard every mapped reader in this crate accepts (design §8.1).
    let map = match unsafe { memmap2::Mmap::map(&file) } {
        Ok(m) if !m.is_empty() => Arc::new(m),
        _ => return Ok(serial()),
    };
    let idx = crate::io::index::load_gz(path).map(Arc::new);
    Ok(Box::new(ParGz::new(
        map,
        idx,
        (2 * threads).min(64),
        MEMBER_CAP,
    )))
}

impl ParGz {
    fn new(map: Map, idx: Option<Arc<GzIndex>>, depth: usize, cap: usize) -> ParGz {
        let mode = match idx {
            Some(idx) => Mode::Indexed {
                idx,
                next: 0,
                pipe: Pipe::new(),
                crc: Crc::new(),
            },
            None => serial_at(&map, 0),
        };
        ParGz {
            map,
            seg: Vec::new(),
            at: 0,
            mode,
            depth,
            spare: None,
            cap,
            released: 0,
        }
    }

    /// Drop the mapped pages of compressed input already consumed, so the
    /// file does not accumulate in RSS (a 7.9 GB `.gz` would otherwise be
    /// resident by the end). Pages a straggling worker still reads fault
    /// back in from the file; the bytes are the same.
    fn release(&mut self) {
        let low = match &self.mode {
            Mode::Serial { dec, .. } => dec.get_ref().pos,
            Mode::Members { p, .. } => *p,
            Mode::Indexed { idx, next, .. } => {
                idx.points.get(*next).map_or(self.map.len(), |q| q.in_byte)
            }
            Mode::Done => self.map.len(),
        }
        .min(self.map.len())
            & !0xfff;
        if low >= self.released + RELEASE {
            #[cfg(unix)]
            // SAFETY: a read-only file mapping; after DONTNEED the pages are
            // read from the file again on access, never zero-filled.
            unsafe {
                let _ = self.map.unchecked_advise_range(
                    memmap2::UncheckedAdvice::DontNeed,
                    self.released,
                    low - self.released,
                );
            }
            self.released = low;
        }
    }
}

fn serial_at(map: &Map, pos: usize) -> Mode {
    Mode::Serial {
        dec: Box::new(flate2::bufread::GzDecoder::new(MapReader {
            map: map.clone(),
            pos,
        })),
        out: 0,
    }
}

/// Next gzip header signature at or after `from`.
fn next_signature(data: &[u8], mut from: usize) -> Option<usize> {
    while from + 10 <= data.len() {
        let i = from + data[from..data.len() - 9].iter().position(|&b| b == 0x1f)?;
        if data[i + 1] == 0x8b && data[i + 2] == 8 && data[i + 3] & 0xe0 == 0 {
            return Some(i);
        }
        from = i + 1;
    }
    None
}

impl ParGz {
    /// The next decompressed segment, or `None` at the end.
    fn next_segment(&mut self) -> std::io::Result<Option<(Vec<u8>, usize)>> {
        loop {
            match &mut self.mode {
                Mode::Done => return Ok(None),
                Mode::Serial { dec, out } => {
                    let mut v = vec![0u8; SERIAL_PIECE];
                    let mut got = 0;
                    while got < v.len() {
                        match dec.read(&mut v[got..]) {
                            Ok(0) => break,
                            Ok(k) => got += k,
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                            Err(e) => return Err(e),
                        }
                    }
                    *out += got;
                    if got > 0 {
                        v.truncate(got);
                        return Ok(Some((v, 0)));
                    }
                    // Member finished: what follows decides the next mode.
                    let (p, size) = (dec.get_ref().pos, *out);
                    self.mode = if p >= self.map.len() {
                        Mode::Done
                    } else if size <= self.cap {
                        Mode::Members {
                            p,
                            scan: p,
                            pipe: self.spare.take().unwrap_or_else(Pipe::new),
                        }
                    } else {
                        serial_at(&self.map, p)
                    };
                }
                Mode::Members { p, scan, pipe } => {
                    let p0 = *p;
                    if p0 >= self.map.len() {
                        self.mode = Mode::Done;
                        continue;
                    }
                    pipe.prune(p0);
                    let data = &self.map[..];
                    let cap = self.cap;
                    let submit = |pipe: &mut Pipe<Member>, k: usize| {
                        if !pipe.has(k) {
                            let map = self.map.clone();
                            pipe.submit(k, move || decode_member(&map, k, cap));
                        }
                    };
                    submit(pipe, p0);
                    *scan = (*scan).max(p0 + 1);
                    while pipe.busy.len() + pipe.ready.len() < self.depth {
                        let Some(c) = next_signature(data, *scan) else {
                            *scan = data.len();
                            break;
                        };
                        submit(pipe, c);
                        *scan = c + 1;
                    }
                    match pipe.wait(p0) {
                        Member::Ok { bytes, end } => {
                            *p = end;
                            if bytes.is_empty() {
                                continue;
                            }
                            return Ok(Some((bytes, 0)));
                        }
                        // Not a small, valid member: flate2 decides.
                        Member::TooBig | Member::Bad => {
                            let Mode::Members { pipe, .. } =
                                std::mem::replace(&mut self.mode, serial_at(&self.map, p0))
                            else {
                                unreachable!()
                            };
                            self.spare = Some(pipe);
                        }
                    }
                }
                Mode::Indexed {
                    idx,
                    next,
                    pipe,
                    crc,
                } => {
                    let n = idx.points.len();
                    if *next == n {
                        self.mode = Mode::Done;
                        continue;
                    }
                    for k in *next..n.min(*next + self.depth) {
                        if !pipe.has(k) {
                            let (map, idx) = (self.map.clone(), idx.clone());
                            pipe.submit(k, move || {
                                let p = &idx.points[k];
                                let (stop, end) = match idx.points.get(k + 1) {
                                    Some(q) => (Some(q.in_byte), q.out),
                                    None => (None, idx.total_out),
                                };
                                decode_span(&map, p, stop, end)
                            });
                        }
                    }
                    let chunk = pipe.wait(*next)?;
                    *next += 1;
                    for piece in &chunk.pieces {
                        crc.combine(&piece.crc);
                        if let Some((c, len)) = piece.end {
                            if crc.sum() != c || crc.amount() != len {
                                return Err(io_err("corrupt gzip member (CRC or length mismatch)"));
                            }
                            crc.reset();
                        }
                    }
                    if *next == n && crc.amount() != 0 {
                        return Err(io_err("gzip index does not match the file (tail)"));
                    }
                    return Ok(Some((chunk.buf, chunk.skip)));
                }
            }
        }
    }
}

impl Read for ParGz {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        while self.at >= self.seg.len() {
            let next = self.next_segment()?;
            self.release();
            match next {
                Some((v, skip)) => {
                    self.seg = v;
                    self.at = skip;
                }
                None => return Ok(0),
            }
        }
        let n = (self.seg.len() - self.at).min(out.len());
        out[..n].copy_from_slice(&self.seg[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

/// Build the gzip index of `path` (one serial pass that checks every member).
pub fn index_file(path: &Path) -> Result<GzIndex> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Io(format!("cannot read {}: {e}", path.display())))?;
    // SAFETY: as in `open`.
    let map = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| Error::Io(format!("cannot map {}: {e}", path.display())))?;
    build_index(&map, SPACING)
        .map_err(|e| Error::InvalidData(format!("{}: cannot index: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn gz(data: &[u8], level: u32) -> Vec<u8> {
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    /// Text that compresses like a dump: varied rows, long back references.
    fn text(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut o = Vec::new();
        for i in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            writeln!(
                o,
                "{i} {} {:.10} {:.10} {:.10}",
                1 + (s >> 62),
                (s >> 11) as f64 / 9.0e15,
                (s >> 20) as f64 / 7.0e12,
                (s >> 33) as f64 / 3.0e9
            )
            .unwrap();
        }
        o
    }

    fn serial(z: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut o = Vec::new();
        flate2::read::MultiGzDecoder::new(z).read_to_end(&mut o)?;
        Ok(o)
    }

    fn write_tmp(name: &str, z: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("rmda_gzpar_{}_{name}", std::process::id()));
        std::fs::write(&p, z).unwrap();
        p
    }

    fn read_all(r: &mut dyn Read) -> std::io::Result<Vec<u8>> {
        let mut o = Vec::new();
        r.read_to_end(&mut o)?;
        Ok(o)
    }

    #[test]
    fn spans_between_access_points_equal_the_serial_stream() {
        let plain = text(200_000, 7);
        for level in [1, 6, 9] {
            let mut z = gz(&plain[..plain.len() / 3], level);
            z.extend(gz(&[], level));
            z.extend(gz(&plain[plain.len() / 3..], level));
            let idx = build_index(&z, 1 << 16).unwrap();
            assert!(idx.points.len() > 20, "{}", idx.points.len());
            assert_eq!(idx.total_out, plain.len() as u64);
            let mut out = Vec::new();
            for (k, p) in idx.points.iter().enumerate() {
                let (stop, end) = match idx.points.get(k + 1) {
                    Some(q) => (Some(q.in_byte), q.out),
                    None => (None, idx.total_out),
                };
                let c = decode_span(&z, p, stop, end).unwrap();
                out.extend_from_slice(&c.buf[c.skip..]);
            }
            assert!(out == plain, "level {level}");
            let stamp = (z.len() as u64, 12345u128);
            let back = GzIndex::from_bytes(&idx.to_bytes(stamp), stamp).unwrap();
            assert_eq!(back, idx);
            assert!(GzIndex::from_bytes(&idx.to_bytes(stamp), (1, 2)).is_none());
        }
    }

    #[test]
    fn index_refuses_corrupt_and_trailing_bytes() {
        let plain = text(20_000, 3);
        let mut z = gz(&plain, 6);
        z.extend_from_slice(b"junk");
        assert!(build_index(&z, 1 << 16).is_err());
        let mut z = gz(&plain, 6);
        let n = z.len();
        z[n - 6] ^= 1; // ISIZE
        assert!(build_index(&z, 1 << 16).is_err());
    }

    #[test]
    fn header_fields_are_parsed() {
        let mut z = Vec::new();
        let enc = flate2::GzBuilder::new()
            .filename("a.dump")
            .comment("c")
            .extra(vec![1, 2, 3])
            .write(&mut z, flate2::Compression::default());
        let mut enc = enc;
        enc.write_all(b"hello\n").unwrap();
        enc.finish().unwrap();
        let s = parse_header(&z, 0).unwrap();
        assert_eq!(s, 10 + 5 + 7 + 2);
        assert!(
            matches!(decode_member(&z, 0, 1 << 20), Member::Ok { ref bytes, end } if bytes == b"hello\n" && end == z.len())
        );
    }

    fn map_of(name: &str, z: &[u8]) -> Map {
        let p = write_tmp(name, z);
        let m = unsafe { memmap2::Mmap::map(&std::fs::File::open(&p).unwrap()).unwrap() };
        let _ = std::fs::remove_file(&p);
        Arc::new(m)
    }

    #[test]
    fn parallel_reader_equals_multigz_in_every_mode() {
        let plain = text(150_000, 11);
        // Many small members (GzWriter's layout), one big member, a mix with
        // members over the speculative cap, and empty members.
        let mut many = Vec::new();
        for c in plain.chunks(50_000) {
            many.extend(gz(c, 6));
        }
        many.extend(gz(&[], 6));
        let one = gz(&plain, 6);
        let mut mixed = gz(&plain[..1000], 6);
        mixed.extend(gz(&plain[1000..], 1));
        mixed.extend(gz(&[], 6));
        mixed.extend(gz(&plain[..70_000], 9));
        mixed.extend(gz(&plain[..300_000], 6));
        for (name, z) in [("many", &many), ("one", &one), ("mixed", &mixed)] {
            let want = serial(z).unwrap();
            let map = map_of(name, z);
            for cap in [MEMBER_CAP, 100_000] {
                for depth in [1, 3, 16] {
                    let got = read_all(&mut ParGz::new(map.clone(), None, depth, cap)).unwrap();
                    assert!(got == want, "{name} unindexed cap {cap} depth {depth}");
                }
            }
            for spacing in [1 << 15, 1 << 20, u64::MAX] {
                let idx = Arc::new(build_index(z, spacing).unwrap());
                let got = read_all(&mut ParGz::new(map.clone(), Some(idx), 5, MEMBER_CAP)).unwrap();
                assert!(got == want, "{name} indexed {spacing}");
            }
        }
    }

    #[test]
    fn open_reads_with_and_without_an_index() {
        let plain = text(80_000, 13);
        let mut z = Vec::new();
        for c in plain.chunks(1 << 20) {
            z.extend(gz(c, 6));
        }
        let p = write_tmp("open.dump.gz", &z);
        assert!(read_all(&mut *open(&p).unwrap()).unwrap() == plain);
        let stamp = crate::io::index::stamp(&p).unwrap();
        let idx = build_index(&z, 1 << 16).unwrap();
        std::fs::write(crate::io::index::index_path(&p), idx.to_bytes(stamp)).unwrap();
        assert_eq!(crate::io::index::load_gz(&p), Some(idx));
        assert!(read_all(&mut *open(&p).unwrap()).unwrap() == plain);
        let _ = std::fs::remove_file(crate::io::index::index_path(&p));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn corruption_is_an_error_in_every_mode() {
        let plain = text(60_000, 5);
        let mut z = Vec::new();
        for c in plain.chunks(20_000) {
            z.extend(gz(c, 6));
        }
        let idx = Arc::new(build_index(&z, 1 << 14).unwrap());
        let n = z.len();
        for at in [n - 6, n - 7, n / 2] {
            let mut bad = z.clone();
            bad[at] ^= 1;
            assert!(serial(&bad).is_err());
            let map = map_of("corrupt", &bad);
            assert!(read_all(&mut ParGz::new(map.clone(), None, 4, MEMBER_CAP)).is_err());
            assert!(read_all(&mut ParGz::new(map, Some(idx.clone()), 4, MEMBER_CAP)).is_err());
        }
        let mut junk = z.clone();
        junk.extend_from_slice(b"junk");
        assert!(serial(&junk).is_err());
        assert!(read_all(&mut ParGz::new(map_of("junk", &junk), None, 4, MEMBER_CAP)).is_err());
    }
}
