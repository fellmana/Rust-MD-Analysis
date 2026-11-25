//! Frame writers (design §8.6): dispatch by format, and the parallel
//! chunked row formatter they share (design §8.6).

use crate::data::property::{DType, Property, PropertyStore};
use crate::data::Frame;
use crate::error::{Error, Result};
use crate::io::format::Format;
use crate::io::xyz::{XyzOrigin, XyzWriter};
use std::io::Write;

pub struct Writer;

impl Writer {
    /// Write `frames` in `fmt` to any writer.
    pub fn write<W: Write>(w: &mut W, fmt: Format, frames: &[Frame]) -> Result<()> {
        match fmt {
            Format::Xyz => XyzWriter::write_all(w, frames),
            Format::LammpsDump => crate::io::lammps::write_dump(w, frames),
            Format::LammpsData => crate::io::lammps::write_data_file(w, frames),
            Format::LammpsDumpBinary => crate::io::binary::write_dump_binary(w, frames),
        }
    }

    /// Write `frames` in `fmt` with explicit [`WriteOptions`].
    ///
    /// `WriteOptions::default()` writes exactly the bytes of [`Writer::write`]
    /// (with the extended-XYZ origin taken from `opts.origin`).
    pub fn write_with<W: Write>(
        w: &mut W,
        fmt: Format,
        frames: &[Frame],
        opts: &WriteOptions,
    ) -> Result<()> {
        match fmt {
            Format::Xyz => {
                for f in frames {
                    XyzWriter::write_frame_opts(w, f, &PropertyStore::new(), opts)?;
                }
                Ok(())
            }
            Format::LammpsDump => crate::io::lammps::write_dump_opts(w, frames, opts),
            Format::LammpsData => crate::io::lammps::write_data_file_opts(w, frames, opts),
            Format::LammpsDumpBinary => crate::io::binary::write_dump_binary_opts(w, frames, opts),
        }
    }
}

/// How a float is spelled by a text writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatFmt {
    /// The shortest decimal that reads back to the same `f64` bit for bit
    /// (std's algorithm), in exponent form (`1.5e-7`) below 1e-5 and from 1e16
    /// up so a tiny or huge value stays short.
    Shortest,
    /// `N` digits after the decimal point (`{:.N}`), the form the dump and
    /// extended-XYZ writers use for positions by default (`N = 10`). A value
    /// reads back as exactly the decimal that was printed.
    Fixed(usize),
}

impl FloatFmt {
    /// Append `v` to `w`.
    #[inline]
    pub fn write<W: Write + ?Sized>(self, w: &mut W, v: f64) -> std::io::Result<()> {
        match self {
            FloatFmt::Fixed(p) => write!(w, "{v:.p$}"),
            FloatFmt::Shortest => write_shortest(w, v),
        }
    }

    /// `full` or a number of decimals, as `--precision` and
    /// `--prop-precision` take them.
    pub fn parse(s: &str) -> std::result::Result<FloatFmt, String> {
        if s.eq_ignore_ascii_case("full") {
            return Ok(FloatFmt::Shortest);
        }
        match s.parse::<usize>() {
            Ok(n) if n <= 30 => Ok(FloatFmt::Fixed(n)),
            _ => Err(format!(
                "expected `full` or a number of decimals 0..=30, got {s:?}"
            )),
        }
    }
}

/// Shortest round-trip spelling of `v` (see [`FloatFmt::Shortest`]).
#[inline]
pub fn write_shortest<W: Write + ?Sized>(w: &mut W, v: f64) -> std::io::Result<()> {
    let a = v.abs();
    if v != 0.0 && a.is_finite() && !(1e-5..1e16).contains(&a) {
        write!(w, "{v:e}")
    } else {
        write!(w, "{v}")
    }
}

/// Which per-atom properties a writer emits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum PropSelect {
    /// None (the default).
    #[default]
    None,
    /// Every per-atom property the format can carry; the rest are skipped.
    All,
    /// Exactly these, in this order; a name the frame lacks or the format
    /// cannot carry is an error.
    Names(Vec<String>),
}

impl PropSelect {
    /// `none`, `all` or a comma-separated list of names.
    pub fn parse(s: &str) -> PropSelect {
        match s.trim() {
            "" | "none" => PropSelect::None,
            "all" => PropSelect::All,
            list => PropSelect::Names(
                list.split(',')
                    .map(|n| n.trim().to_string())
                    .filter(|n| !n.is_empty())
                    .collect(),
            ),
        }
    }
}

/// Options of the frame writers (`rmda convert`).
///
/// The default is each writer's plain output: positions at the format's own
/// precision, no properties, and no type labels beyond the XYZ species.
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// Also carry what the format can hold besides properties: type labels
    /// (dump `element` column, data `Atom Type Labels`, extended-XYZ
    /// `species=` key), atom ids and the selection mask (extended-XYZ `id` and
    /// `Selection` columns) and image flags (data `Atoms` rows).
    pub extended: bool,
    /// Per-atom properties to write.
    pub props: PropSelect,
    /// Positions and box. `None` keeps the format's default: 10 decimals in
    /// dump and XYZ, the shortest round trip in data files. Binary dumps are
    /// always exact.
    pub precision: Option<FloatFmt>,
    /// Float property columns. `None` is the shortest round trip.
    pub prop_precision: Option<FloatFmt>,
    /// Extended-XYZ origin placement.
    pub origin: XyzOrigin,
    /// Data-file `Atoms` style (`convert --out-atom-style`). `None` picks it
    /// from the selected properties. Its `mol`/`q`/`diameter`/`density`
    /// columns are taken from the frame's properties of those names, which
    /// must exist; selected ones the style has no column for are refused
    /// (named) or skipped (`all`).
    pub atom_style: Option<String>,
}

/// A property's values, typed, for formatting.
#[derive(Clone, Copy)]
pub(crate) enum PropCol<'a> {
    U8(&'a [u8]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl<'a> PropCol<'a> {
    pub(crate) fn of(p: &'a Property) -> PropCol<'a> {
        match p.dtype {
            DType::U8 => PropCol::U8(p.as_u8().map_or(&[], |v| v.as_slice())),
            DType::I32 => PropCol::I32(p.as_i32().map_or(&[], |v| v.as_slice())),
            DType::I64 => PropCol::I64(p.as_i64().map_or(&[], |v| v.as_slice())),
            DType::F32 => PropCol::F32(p.as_f32().map_or(&[], |v| v.as_slice())),
            DType::F64 => PropCol::F64(p.as_f64().map_or(&[], |v| v.as_slice())),
        }
    }

    /// Append element `k`: integers as integers, floats in `fmt`. An `f32`
    /// is printed as the `f64` it widens to, so it reads back as that value.
    #[inline]
    pub(crate) fn write<W: Write + ?Sized>(
        self,
        w: &mut W,
        k: usize,
        fmt: FloatFmt,
    ) -> std::io::Result<()> {
        match self {
            PropCol::U8(v) => write!(w, "{}", v[k]),
            PropCol::I32(v) => write!(w, "{}", v[k]),
            PropCol::I64(v) => write!(w, "{}", v[k]),
            PropCol::F32(v) => fmt.write(w, v[k] as f64),
            PropCol::F64(v) => fmt.write(w, v[k]),
        }
    }

    /// Element `k` as `f64` (binary dumps store doubles).
    #[inline]
    pub(crate) fn get(self, k: usize) -> f64 {
        match self {
            PropCol::U8(v) => v[k] as f64,
            PropCol::I32(v) => v[k] as f64,
            PropCol::I64(v) => v[k] as f64,
            PropCol::F32(v) => v[k] as f64,
            PropCol::F64(v) => v[k],
        }
    }
}

/// Why a property name cannot be a whitespace-separated column header.
pub(crate) fn plain_name(name: &str) -> std::result::Result<(), String> {
    if name.is_empty() || name.bytes().any(|b| b.is_ascii_whitespace() || b == b'"') {
        Err(format!(
            "property name {name:?} cannot be a column header (empty, or holds whitespace or quotes)"
        ))
    } else {
        Ok(())
    }
}

/// The properties of `f` that `sel` asks for and a format can carry.
///
/// `ok(p)` says whether the format can carry `p`, and why not otherwise. With
/// [`PropSelect::Names`] a missing, per-frame or refused property is an error;
/// with [`PropSelect::All`] it is skipped and named in the second list, so the
/// caller can say what was not written.
pub(crate) fn select_props<'a>(
    f: &'a Frame,
    sel: &PropSelect,
    ok: impl Fn(&Property) -> std::result::Result<(), String>,
) -> Result<(Vec<&'a Property>, Vec<String>)> {
    let n = f.natoms();
    let usable = |p: &Property| -> std::result::Result<(), String> {
        if !p.per_particle || p.components == 0 || p.len() != n {
            return Err(format!(
                "property {} is not one value per atom ({} values for {n} atoms)",
                p.name,
                p.len()
            ));
        }
        ok(p)
    };
    let props = &f.particles.props;
    match sel {
        PropSelect::None => Ok((Vec::new(), Vec::new())),
        PropSelect::All => {
            let (mut keep, mut skipped) = (Vec::new(), Vec::new());
            for p in props.iter() {
                match usable(p) {
                    Ok(()) => keep.push(p),
                    Err(_) => skipped.push(p.name.to_string()),
                }
            }
            Ok((keep, skipped))
        }
        PropSelect::Names(names) => {
            let mut keep: Vec<&Property> = Vec::new();
            for name in names {
                let p = props.get(name).ok_or_else(|| {
                    let have: Vec<&str> = props.iter().map(|p| p.name.as_ref()).collect();
                    Error::invalid(format!(
                        "no per-atom property {name:?} in the frame (it has: {})",
                        if have.is_empty() {
                            "none".to_string()
                        } else {
                            have.join(", ")
                        }
                    ))
                })?;
                usable(p).map_err(Error::invalid)?;
                if !keep.iter().any(|q| q.name == p.name) {
                    keep.push(p);
                }
            }
            Ok((keep, Vec::new()))
        }
    }
}

/// The names of the properties `opts` writes to `fmt` for frame `f`, and of
/// those `PropSelect::All` skips because the format cannot carry them.
pub fn planned_props(
    fmt: Format,
    f: &Frame,
    opts: &WriteOptions,
) -> Result<(Vec<String>, Vec<String>)> {
    let (keep, skipped) = match fmt {
        Format::Xyz => select_props(f, &opts.props, crate::io::xyz::extxyz_prop_ok)?,
        Format::LammpsDump | Format::LammpsDumpBinary => {
            select_props(f, &opts.props, crate::io::lammps::dump_prop_ok)?
        }
        Format::LammpsData => {
            let plan = crate::io::lammps::data_plan(f, opts)?;
            return Ok((plan.written, plan.skipped));
        }
    };
    Ok((keep.iter().map(|p| p.name.to_string()).collect(), skipped))
}

/// A gzip writer that compresses in parallel: the stream is cut into
/// [`GZ_BLOCK`]-byte blocks, each compressed on a worker thread as its own
/// gzip member, and the members are written in order. Concatenated members
/// are one valid gzip file (RFC 1952 §2.2): `gzip -d`, Python's `gzip` and
/// this crate's reader (`MultiGzDecoder`) read it as the plain bytes. The
/// output does not depend on the thread count.
///
/// [`GzWriter::finish`] must be called: dropping the writer without it loses
/// the buffered tail.
pub struct GzWriter<W: Write> {
    inner: W,
    buf: Vec<u8>,
    /// A member has been written.
    wrote: bool,
}

/// Uncompressed bytes per gzip member.
pub const GZ_BLOCK: usize = 4 << 20;

impl<W: Write> GzWriter<W> {
    pub fn new(inner: W) -> Self {
        GzWriter {
            inner,
            buf: Vec::new(),
            wrote: false,
        }
    }

    /// Compress and write every whole block held, or everything when `all`.
    fn drain(&mut self, all: bool) -> std::io::Result<()> {
        use rayon::prelude::*;
        let whole = if all {
            self.buf.len()
        } else {
            self.buf.len() / GZ_BLOCK * GZ_BLOCK
        };
        if whole == 0 {
            return Ok(());
        }
        let members: Vec<std::io::Result<Vec<u8>>> = self.buf[..whole]
            .par_chunks(GZ_BLOCK)
            .map(|block| {
                let mut enc = flate2::write::GzEncoder::new(
                    Vec::with_capacity(block.len() / 3),
                    flate2::Compression::default(),
                );
                enc.write_all(block)?;
                enc.finish()
            })
            .collect();
        for m in members {
            self.inner.write_all(&m?)?;
        }
        self.wrote = true;
        self.buf.drain(..whole);
        Ok(())
    }

    /// Compress the tail, flush, and return the inner writer.
    pub fn finish(mut self) -> std::io::Result<W> {
        if self.buf.is_empty() && !self.wrote {
            // Zero bytes is not a gzip file: write one empty member.
            let enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            self.inner.write_all(&enc.finish()?)?;
        }
        self.drain(true)?;
        self.inner.flush()?;
        Ok(self.inner)
    }
}

impl<W: Write> Write for GzWriter<W> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        if self.buf.len() >= GZ_BLOCK * rayon::current_num_threads().max(1) {
            self.drain(false)?;
        }
        Ok(data.len())
    }

    /// Flushes the inner writer only: compressing a partial block here would
    /// make the member layout depend on when callers flush.
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Rows formatted per task.
const CHUNK_ROWS: usize = 1 << 16;

/// Format rows `0..n` with `row` and write them to `w` in order.
///
/// Worker threads each format a chunk of [`CHUNK_ROWS`] rows into its own
/// buffer; the buffers are written in row order, one batch of
/// `threads × CHUNK_ROWS` rows at a time, so memory stays bounded however
/// large the frame. The bytes are exactly those of a serial loop.
pub fn write_rows<W, F>(w: &mut W, n: usize, row: F) -> Result<()>
where
    W: Write + ?Sized,
    F: Fn(&mut Vec<u8>, usize) -> std::io::Result<()> + Sync,
{
    use rayon::prelude::*;
    let chunk = |a: usize, b: usize| -> std::io::Result<Vec<u8>> {
        let mut buf = Vec::with_capacity((b - a) * 64);
        for i in a..b {
            row(&mut buf, i)?;
        }
        Ok(buf)
    };
    if n <= CHUNK_ROWS {
        w.write_all(&chunk(0, n)?)?;
        return Ok(());
    }
    let batch = CHUNK_ROWS * rayon::current_num_threads().max(1);
    let mut b0 = 0;
    while b0 < n {
        let b1 = (b0 + batch).min(n);
        let starts: Vec<usize> = (b0..b1).step_by(CHUNK_ROWS).collect();
        let bufs: Vec<std::io::Result<Vec<u8>>> = starts
            .par_iter()
            .map(|&a| chunk(a, (a + CHUNK_ROWS).min(b1)))
            .collect();
        for b in bufs {
            w.write_all(&b?)?;
        }
        b0 = b1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_rows_equal_a_serial_loop() {
        for n in [0usize, 1, CHUNK_ROWS, CHUNK_ROWS + 1, 5 * CHUNK_ROWS + 17] {
            let mut par = Vec::new();
            write_rows(&mut par, n, |b, i| {
                writeln!(b, "{i} {:.10}", i as f64 / 7.0)
            })
            .unwrap();
            let mut ser = Vec::new();
            for i in 0..n {
                writeln!(ser, "{i} {:.10}", i as f64 / 7.0).unwrap();
            }
            assert!(par == ser, "n={n}");
        }
    }
}
