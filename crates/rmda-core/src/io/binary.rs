//! LAMMPS binary dump reader and writer (`dump atom/custom FILE.bin`, design
//! §8.3).
//!
//! Layout, from `src/dump_atom.cpp` / `src/dump_custom.cpp`
//! (`header_binary*`, `header_format_binary`, `write_binary`), all native
//! endian, which this reader requires to be little-endian:
//!
//! ```text
//! [new format only]  bigint -len, char magic[len] ("DUMPATOM"/"DUMPCUSTOM"),
//!                    int endian (0x0001), int revision (0x0002)
//! bigint ntimestep, bigint natoms, int triclinic (0, 1, or 2 = general),
//! int boundary[3][2] (0 p, 1 f, 2 s, 3 m),
//! triclinic 0: double xlo xhi ylo yhi zlo zhi
//! triclinic 1: double xlo xhi ylo yhi zlo zhi xy xz yz   (bounding box, as text)
//! triclinic 2: double avec[3] bvec[3] cvec[3] boxlo[3]
//! int size_one
//! [revision > 1]     int len, char units[len]; char time_flag [, double time];
//!                    int len, char columns[len]
//! int nchunk, then nchunk x (int n, double buf[n])   n = rows * size_one
//! ```
//!
//! Files without the magic (pre-2020 LAMMPS) carry no column names; they are
//! read as `dump atom` output, `id type xs ys zs [ix iy iz]` by `size_one`.

use crate::cell::{pack_image, UnitCell};
use crate::data::frame::Frame;
use crate::error::{Error, Result};
use crate::io::engine::{Cursor, RowChunk};
use crate::io::lammps::{DumpHeader, DumpPlan, Sink};
use crate::io::writer::{select_props, PropCol, WriteOptions};
use std::io::Write;

/// A little-endian field reader over the buffer.
struct Rd<'a> {
    b: &'a [u8],
    pos: usize,
    path: &'a str,
}

impl<'a> Rd<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.b.len() - self.pos {
            return Err(Error::Parse {
                source: Some(self.path.to_string()),
                offset: Some(self.pos as u64),
                line: None,
                msg: "binary dump ends inside a frame".into(),
            });
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn err(&self, msg: impl Into<String>) -> Error {
        Error::Parse {
            source: Some(self.path.to_string()),
            offset: Some(self.pos as u64),
            line: None,
            msg: msg.into(),
        }
    }
}

/// Whether `head` starts like a new-format binary dump (`-len` + `DUMP...`).
pub fn sniff(head: &[u8]) -> bool {
    if head.len() < 12 {
        return false;
    }
    let len = i64::from_le_bytes(head[..8].try_into().unwrap());
    (-32..0).contains(&len) && head[8..12] == *b"DUMP"
}

/// One frame's header and where its chunks are.
struct BinFrame {
    header: DumpHeader,
    size_one: usize,
    /// `(byte offset of the doubles, rows)` per chunk.
    chunks: Vec<(usize, usize)>,
    end: usize,
}

fn parse_header(rd: &mut Rd) -> Result<BinFrame> {
    let mut ntimestep = rd.i64()?;
    let mut revision = 0;
    if ntimestep < 0 {
        let len = (-ntimestep) as usize;
        if len > 64 {
            return Err(rd.err("binary dump: implausible magic-string length"));
        }
        let magic = rd.bytes(len)?;
        if !magic.starts_with(b"DUMP") {
            return Err(rd.err(format!(
                "binary dump: unknown magic {:?}",
                String::from_utf8_lossy(magic)
            )));
        }
        let endian = rd.i32()?;
        if endian != 1 {
            return Err(Error::Unsupported(format!(
                "binary dump written with foreign byte order (endian flag {endian:#x})"
            )));
        }
        revision = rd.i32()?;
        ntimestep = rd.i64()?;
    }
    let natoms = rd.i64()?;
    if natoms < 0 {
        return Err(rd.err("binary dump: negative atom count"));
    }
    let triclinic = rd.i32()?;
    let mut boundary = [0i32; 6];
    for b in boundary.iter_mut() {
        *b = rd.i32()?;
    }
    let mut pbc = [true; 3];
    for d in 0..3 {
        pbc[d] = boundary[2 * d] == 0 && boundary[2 * d + 1] == 0;
    }
    let nbox = match triclinic {
        0 => 6,
        1 => 9,
        2 => 12,
        t => return Err(rd.err(format!("binary dump: triclinic flag {t}"))),
    };
    let mut bx = [0.0f64; 12];
    for v in bx.iter_mut().take(nbox) {
        *v = rd.f64()?;
    }
    let size_one = rd.i32()?;
    if size_one <= 0 {
        return Err(rd.err("binary dump: non-positive size_one"));
    }
    let size_one = size_one as usize;
    let mut units = None;
    let mut time = None;
    let mut columns: Option<Vec<String>> = None;
    if revision > 1 {
        let len = rd.i32()?;
        if len > 0 {
            units = Some(String::from_utf8_lossy(rd.bytes(len as usize)?).into_owned());
        }
        if rd.bytes(1)?[0] != 0 {
            time = Some(rd.f64()?);
        }
        let len = rd.i32()?.max(0) as usize;
        let text = String::from_utf8_lossy(rd.bytes(len)?).into_owned();
        columns = Some(text.split_whitespace().map(str::to_string).collect());
    }
    let columns = match columns {
        Some(c) => c,
        // Old format: `dump atom` defaults (`DumpAtom::init_style`).
        None => match size_one {
            5 => "id type xs ys zs",
            8 => "id type xs ys zs ix iy iz",
            n => {
                return Err(Error::Unsupported(format!(
                    "binary dump without column names has {n} columns per atom; \
                     only `dump atom` layouts (5 or 8) can be inferred"
                )))
            }
        }
        .split(' ')
        .map(str::to_string)
        .collect(),
    };
    if columns.len() != size_one {
        return Err(rd.err(format!(
            "binary dump: {} column names for {size_one} values per atom",
            columns.len()
        )));
    }
    let nchunk = rd.i32()?;
    let mut chunks = Vec::with_capacity(nchunk.max(0) as usize);
    let mut rows = 0usize;
    for _ in 0..nchunk.max(0) {
        let n = rd.i32()?;
        if n < 0 || n as usize % size_one != 0 {
            return Err(rd.err(format!(
                "binary dump: chunk of {n} values is not a multiple of {size_one}"
            )));
        }
        let at = rd.pos;
        rd.bytes(n as usize * 8)?;
        chunks.push((at, n as usize / size_one));
        rows += n as usize / size_one;
    }
    if rows != natoms as usize {
        return Err(rd.err(format!(
            "binary dump: chunks hold {rows} atoms, header says {natoms}"
        )));
    }
    let (cell, g2r, general) = match triclinic {
        0 => {
            let mut c = UnitCell::orthorhombic(bx[1] - bx[0], bx[3] - bx[2], bx[5] - bx[4]);
            c.origin = [bx[0], bx[2], bx[4]];
            (c, None, None)
        }
        1 => {
            let (xy, xz, yz) = (bx[6], bx[7], bx[8]);
            let (xlo, xhi, ylo, yhi) =
                crate::io::lammps::bounds_to_lohi(bx[0], bx[1], bx[2], bx[3], xy, xz, yz);
            (
                UnitCell::from_box_bounds(xlo, xhi, xy, ylo, yhi, xz, bx[4], bx[5], yz),
                None,
                None,
            )
        }
        _ => {
            let (a, b, c, o) = (
                [bx[0], bx[1], bx[2]],
                [bx[3], bx[4], bx[5]],
                [bx[6], bx[7], bx[8]],
                [bx[9], bx[10], bx[11]],
            );
            let (cell, r) = UnitCell::from_general_triclinic_rot(a, b, c, o)?;
            let given = UnitCell::general_as_given(a, b, c, o)?.with_pbc(pbc);
            (cell, Some(r), Some(given))
        }
    };
    Ok(BinFrame {
        header: DumpHeader {
            timestep: ntimestep,
            time,
            units,
            natoms: natoms as usize,
            cell: cell.with_pbc(pbc),
            g2r,
            general,
            columns,
        },
        size_one,
        chunks,
        end: rd.pos,
    })
}

/// Rows per parallel piece of a chunk.
const PIECE_ROWS: usize = 1 << 16;

/// Read the frame at the cursor, or `None` at the end of the file.
pub fn read_next(cur: &Cursor) -> Result<Option<Frame>> {
    read_next_with(cur, &crate::io::ReadOptions::all())
}

/// [`read_next`] keeping only the columns `opts` allows.
pub fn read_next_with(cur: &Cursor, opts: &crate::io::ReadOptions) -> Result<Option<Frame>> {
    use rayon::prelude::*;
    let b = cur.bytes();
    if cur.pos() >= b.len() {
        return Ok(None);
    }
    let mut rd = Rd {
        b,
        pos: cur.pos(),
        path: &cur.path,
    };
    let mut f = parse_header(&mut rd)?;
    f.header.apply_keep_general(opts);
    let plan = DumpPlan::resolve(&f.header.columns, opts)?;
    if plan.sinks.contains(&Sink::Element) {
        return Err(Error::Unsupported(
            "binary dump with an `element` column (LAMMPS cannot write one)".into(),
        ));
    }
    // Pieces of at most PIECE_ROWS rows, in file order.
    let mut pieces: Vec<(usize, RowChunk)> = Vec::new();
    let mut row = 0usize;
    for &(at, rows) in &f.chunks {
        let mut done = 0;
        while done < rows {
            let k = (rows - done).min(PIECE_ROWS);
            pieces.push((
                at + done * f.size_one * 8,
                RowChunk {
                    start: 0,
                    end: 0,
                    first_row: row,
                    nrows: k,
                    first_line: 0,
                },
            ));
            row += k;
            done += k;
        }
    }
    let chunks: Vec<RowChunk> = pieces.iter().map(|p| p.1).collect();
    let mut out = plan.alloc(f.header.natoms);
    let mut sinks = out.split(&chunks, plan.images);
    let (cell, g2r) = (&f.header.cell, f.header.g2r.as_ref());
    let size_one = f.size_one;
    let res: Vec<std::result::Result<(), String>> = pieces
        .par_iter()
        .zip(sinks.par_iter_mut())
        .map(|((at, c), s)| {
            let vals = &b[*at..*at + c.nrows * size_one * 8];
            for r in 0..c.nrows {
                let row = &vals[r * size_one * 8..(r + 1) * size_one * 8];
                let mut v = [0.0f64; 3];
                let mut img = [0i64; 3];
                for (j, sink) in plan.sinks.iter().enumerate() {
                    let x = f64::from_le_bytes(row[j * 8..j * 8 + 8].try_into().unwrap());
                    let int = || -> std::result::Result<i64, String> {
                        if x.fract() == 0.0 && x.abs() < 9.0e15 {
                            Ok(x as i64)
                        } else {
                            Err(format!(
                                "atom {}: column {} holds {x}, not an integer",
                                c.first_row + r,
                                f.header.columns[j]
                            ))
                        }
                    };
                    match *sink {
                        Sink::Id => s.ids[r] = int()?,
                        Sink::Type => {
                            let t = int()?;
                            if t < 1 {
                                return Err(format!("atom type {t} is not a positive integer"));
                            }
                            s.types[r] = t as u32;
                        }
                        Sink::Pos(d) => v[d] = x,
                        Sink::Image(d) => img[d] = int()?,
                        Sink::PropF(k) => s.pf[k][r] = x,
                        Sink::PropI(k) => s.pi[k][r] = int()?,
                        Sink::Skip => {}
                        Sink::Element => unreachable!(),
                    }
                }
                s.pos[r] = plan.pos.place(v, cell, g2r);
                if !s.images.is_empty() {
                    s.images[r] = pack_image(img[0], img[1], img[2]);
                }
            }
            Ok(())
        })
        .collect();
    drop(sinks);
    for r in res {
        r.map_err(|m| rd.err(m))?;
    }
    let frame = plan.to_frame(out, &f.header);
    cur.seek(f.end, 0);
    Ok(Some(frame))
}

/// Skip the frame at the cursor without decoding its atoms; `false` at EOF.
pub fn skip_next(cur: &Cursor) -> Result<bool> {
    let b = cur.bytes();
    if cur.pos() >= b.len() {
        return Ok(false);
    }
    let mut rd = Rd {
        b,
        pos: cur.pos(),
        path: &cur.path,
    };
    let f = parse_header(&mut rd)?;
    cur.seek(f.end, 0);
    Ok(true)
}

/// Timestep and atom count of the frame at the cursor, which is left at the
/// next frame (for the frame index).
pub(crate) fn scan_next(cur: &Cursor) -> Result<Option<(i64, usize)>> {
    let b = cur.bytes();
    if cur.pos() >= b.len() {
        return Ok(None);
    }
    let mut rd = Rd {
        b,
        pos: cur.pos(),
        path: &cur.path,
    };
    let f = parse_header(&mut rd)?;
    cur.seek(f.end, 0);
    Ok(Some((f.header.timestep, f.header.natoms)))
}

/// Write frames as a new-format (revision 2) `dump custom` binary file with
/// columns `id type x y z [ix iy iz]` and one chunk per frame, readable by
/// LAMMPS `rerun`/`read_dump` and `tools/binary2txt`.
pub fn write_dump_binary<W: Write>(w: &mut W, frames: &[Frame]) -> Result<()> {
    write_dump_binary_opts(w, frames, &WriteOptions::default())
}

/// [`write_dump_binary`] with the selected properties as extra double
/// columns (`name` or `name[1] .. name[k]`, as in a text dump). Values are
/// stored exactly; integer properties are exact up to 2^53. Type labels have
/// no place in a binary dump (LAMMPS cannot write an `element` column there).
/// A frame larger than one chunk's `int` count is split over several chunks.
pub fn write_dump_binary_opts<W: Write>(
    w: &mut W,
    frames: &[Frame],
    opts: &WriteOptions,
) -> Result<()> {
    write_dump_binary_chunked(w, frames, opts, i32::MAX as usize)
}

/// [`write_dump_binary_opts`] with at most `max_values` doubles per chunk.
pub(crate) fn write_dump_binary_chunked<W: Write>(
    w: &mut W,
    frames: &[Frame],
    opts: &WriteOptions,
    max_values: usize,
) -> Result<()> {
    use crate::cell::{image_ipart, image_jpart, image_kpart};
    const MAGIC: &[u8] = b"DUMPCUSTOM";
    for f in frames {
        let n = f.natoms();
        let images = f.particles.images.as_ref();
        let (props, _) = select_props(f, &opts.props, crate::io::lammps::dump_prop_ok)?;
        let cols: Vec<(usize, PropCol)> = props
            .iter()
            .map(|p| (p.components, PropCol::of(p)))
            .collect();
        let mut columns = String::from(if images.is_some() {
            "id type x y z ix iy iz"
        } else {
            "id type x y z"
        });
        for p in &props {
            for name in crate::io::lammps::dump_column_names(p) {
                columns.push(' ');
                columns.push_str(&name);
            }
        }
        let size_one: usize =
            if images.is_some() { 8 } else { 5 } + cols.iter().map(|(k, _)| k).sum::<usize>();
        // Rows per chunk: the chunk's value count is an `int`.
        let chunk_rows = (max_values / size_one).max(1);
        let nchunk = if n == 0 {
            1
        } else {
            (n + chunk_rows - 1) / chunk_rows
        };
        w.write_all(&(-(MAGIC.len() as i64)).to_le_bytes())?;
        w.write_all(MAGIC)?;
        w.write_all(&1i32.to_le_bytes())?;
        w.write_all(&2i32.to_le_bytes())?;
        w.write_all(&f.meta.timestep.to_le_bytes())?;
        w.write_all(&(n as i64).to_le_bytes())?;
        let (xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz, tric) = f.cell.to_box_bounds();
        let general = !f.cell.is_restricted();
        let flag: i32 = if general { 2 } else { i32::from(tric) };
        w.write_all(&flag.to_le_bytes())?;
        for d in 0..3 {
            let b: i32 = if f.cell.pbc[d] { 0 } else { 1 };
            w.write_all(&b.to_le_bytes())?;
            w.write_all(&b.to_le_bytes())?;
        }
        let bx: Vec<f64> = if general {
            // `header_binary_triclinic_general`: avec, bvec, cvec, boxlo.
            let (a, b, c, o) = f.cell.lammps_general()?;
            [a, b, c, o].concat()
        } else if tric {
            let (bxlo, bxhi, bylo, byhi) =
                crate::io::lammps::lohi_to_bounds(xlo, xhi, ylo, yhi, xy, xz, yz);
            vec![bxlo, bxhi, bylo, byhi, zlo, zhi, xy, xz, yz]
        } else {
            vec![xlo, xhi, ylo, yhi, zlo, zhi]
        };
        for v in bx {
            w.write_all(&v.to_le_bytes())?;
        }
        w.write_all(&(size_one as i32).to_le_bytes())?;
        match &f.meta.units {
            Some(u) => {
                w.write_all(&(u.len() as i32).to_le_bytes())?;
                w.write_all(u.as_bytes())?;
            }
            None => w.write_all(&0i32.to_le_bytes())?,
        }
        match f.meta.time {
            Some(t) => {
                w.write_all(&[1u8])?;
                w.write_all(&t.to_le_bytes())?;
            }
            None => w.write_all(&[0u8])?,
        }
        w.write_all(&(columns.len() as i32).to_le_bytes())?;
        w.write_all(columns.as_bytes())?;
        w.write_all(&(nchunk as i32).to_le_bytes())?;
        let mut buf = Vec::with_capacity(size_one * 8 * PIECE_ROWS.min(n.max(1)));
        for i in 0..n {
            if i % chunk_rows == 0 {
                w.write_all(&buf)?;
                buf.clear();
                let rows = chunk_rows.min(n - i);
                w.write_all(&((rows * size_one) as i32).to_le_bytes())?;
            }
            let p = f.particles.positions[i];
            buf.extend_from_slice(&(f.particles.ids[i] as f64).to_le_bytes());
            buf.extend_from_slice(&(f.particles.types[i] as f64).to_le_bytes());
            for c in p {
                buf.extend_from_slice(&c.to_le_bytes());
            }
            if let Some(img) = images {
                for v in [
                    image_ipart(img[i]),
                    image_jpart(img[i]),
                    image_kpart(img[i]),
                ] {
                    buf.extend_from_slice(&(v as f64).to_le_bytes());
                }
            }
            for &(k, col) in &cols {
                for c in 0..k {
                    buf.extend_from_slice(&col.get(i * k + c).to_le_bytes());
                }
            }
            if buf.len() >= (1 << 20) {
                w.write_all(&buf)?;
                buf.clear();
            }
        }
        if n == 0 {
            w.write_all(&0i32.to_le_bytes())?;
        }
        w.write_all(&buf)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::particle::ParticleData;

    fn tilted() -> Frame {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/data/triclinic_tilted.dump");
        crate::io::read_frames(&p).unwrap().remove(0)
    }

    fn read_bytes(b: &[u8]) -> Vec<Frame> {
        let cur = Cursor::from_bytes(b, "t.bin");
        let mut out = Vec::new();
        while let Some(f) = read_next(&cur).unwrap() {
            out.push(f);
        }
        out
    }

    /// Regenerates `tests/data/binary2txt_oracle.bin`: set `RMDA_BIN_OUT` to a
    /// path, run `--ignored`, then convert it with LAMMPS's own
    /// `tools/binary2txt.cpp` to get the `.dump` twin.
    #[test]
    #[ignore]
    fn write_oracle_file() {
        let Ok(p) = std::env::var("RMDA_BIN_OUT") else {
            return;
        };
        let mut f = tilted();
        f.meta.timestep = 1234;
        f.meta.time = Some(2.5);
        f.meta.units = Some("metal".into());
        let n = f.natoms();
        f.particles.images = Some(
            (0..n)
                .map(|i| pack_image(i as i64 % 3 - 1, 2, -3))
                .collect(),
        );
        let mut g = f.clone();
        g.meta.timestep = 1300;
        g.particles.images = None;
        let mut out = Vec::new();
        write_dump_binary(&mut out, &[f, g]).unwrap();
        std::fs::write(p, out).unwrap();
    }

    /// `tests/data/binary2txt_oracle.dump` is LAMMPS `tools/binary2txt`'s
    /// decoding of `binary2txt_oracle.bin` (written by [`write_dump_binary`]):
    /// both must read to the same frames. The text side opens each frame with
    /// `ITEM: UNITS` and `ITEM: TIME` ahead of `ITEM: TIMESTEP`.
    #[test]
    fn binary_matches_lammps_binary2txt() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data");
        let bin = crate::io::read_frames(&dir.join("binary2txt_oracle.bin")).unwrap();
        let txt = crate::io::read_frames(&dir.join("binary2txt_oracle.dump")).unwrap();
        assert_eq!(bin.len(), 2);
        assert_eq!(txt.len(), 2);
        for (b, t) in bin.iter().zip(&txt) {
            assert_eq!(b.meta.timestep, t.meta.timestep);
            assert_eq!(b.particles.ids, t.particles.ids);
            assert_eq!(b.particles.types, t.particles.types);
            assert_eq!(b.particles.images, t.particles.images);
            assert_eq!(b.cell.pbc, t.cell.pbc);
            // binary2txt prints atom values with `%g` (6 significant digits).
            for (p, q) in b.particles.positions.iter().zip(&t.particles.positions) {
                for d in 0..3 {
                    assert!((p[d] - q[d]).abs() < 1e-4, "{p:?} vs {q:?}");
                }
            }
        }
        assert_eq!(txt[0].meta.units.as_deref(), Some("metal"));
        assert_eq!(txt[0].meta.time, Some(2.5));
        assert_eq!(bin[0].meta.time, Some(2.5));
        assert!(bin[0].particles.images.is_some() && bin[1].particles.images.is_none());
    }

    #[test]
    fn binary_round_trip_is_exact() {
        let mut f = tilted();
        f.meta.timestep = 77;
        f.meta.time = Some(0.125);
        f.meta.units = Some("real".into());
        let n = f.natoms();
        f.particles.images = Some((0..n).map(|i| pack_image(i as i64 % 3 - 1, 0, 4)).collect());
        let mut out = Vec::new();
        write_dump_binary(&mut out, &[f.clone(), f.clone()]).unwrap();
        let back = read_bytes(&out);
        assert_eq!(back.len(), 2);
        for b in &back {
            assert_eq!(b.particles.positions, f.particles.positions);
            assert_eq!(b.particles.ids, f.particles.ids);
            assert_eq!(b.particles.types, f.particles.types);
            assert_eq!(b.particles.images, f.particles.images);
            assert_eq!(b.cell.h_matrix().m, f.cell.h_matrix().m);
            assert_eq!(b.cell.origin, f.cell.origin);
            assert_eq!(b.cell.pbc, f.cell.pbc);
            assert_eq!(b.meta.timestep, 77);
            assert_eq!(b.meta.time, Some(0.125));
            assert_eq!(b.meta.units.as_deref(), Some("real"));
        }
    }

    /// Bytes laid out field by field in the order of LAMMPS
    /// `DumpCustom::header_binary` + `header_format_binary` +
    /// `header_unit_style_binary` + `header_time_binary` +
    /// `header_columns_binary` + `write_binary`, spelled out here
    /// independently of the writer: two chunks (two "procs"), an extra
    /// property column, and fixed boundaries on z.
    #[test]
    fn reads_the_dump_custom_layout_with_two_chunks() {
        let mut b = Vec::new();
        let i32b = |b: &mut Vec<u8>, v: i32| b.extend_from_slice(&v.to_le_bytes());
        let i64b = |b: &mut Vec<u8>, v: i64| b.extend_from_slice(&v.to_le_bytes());
        let f64b = |b: &mut Vec<u8>, v: f64| b.extend_from_slice(&v.to_le_bytes());
        i64b(&mut b, -10);
        b.extend_from_slice(b"DUMPCUSTOM");
        i32b(&mut b, 1); // ENDIAN
        i32b(&mut b, 2); // FORMAT_REVISION
        i64b(&mut b, 500); // ntimestep
        i64b(&mut b, 3); // natoms
        i32b(&mut b, 0); // triclinic
        for v in [0, 0, 0, 0, 1, 2] {
            i32b(&mut b, v); // boundary[3][2]: pp pp fs
        }
        for v in [0.0, 10.0, -1.0, 9.0, 0.0, 30.0] {
            f64b(&mut b, v);
        }
        i32b(&mut b, 6); // nfield
        i32b(&mut b, 0); // no units
        b.push(0); // no time
        let cols = "id type x y z c_pe";
        i32b(&mut b, cols.len() as i32);
        b.extend_from_slice(cols.as_bytes());
        i32b(&mut b, 2); // nchunk
        i32b(&mut b, 12); // 2 atoms * 6
        for v in [5.0, 1.0, 1.0, 2.0, 3.0, -0.5, 9.0, 2.0, 4.0, 5.0, 6.0, -1.5] {
            f64b(&mut b, v);
        }
        i32b(&mut b, 6);
        for v in [2.0, 1.0, 7.0, 8.0, 9.0, -2.5] {
            f64b(&mut b, v);
        }
        assert!(sniff(&b));
        let f = read_bytes(&b).remove(0);
        assert_eq!(f.meta.timestep, 500);
        assert_eq!(f.particles.ids, vec![5, 9, 2]);
        assert_eq!(f.particles.types, vec![1, 2, 1]);
        assert_eq!(f.particles.positions[2], [7.0, 8.0, 9.0]);
        assert_eq!(f.cell.pbc, [true, true, false]);
        assert_eq!(f.cell.origin, [0.0, -1.0, 0.0]);
        let pe = f
            .particles
            .props
            .get("c_pe")
            .unwrap()
            .as_f64()
            .unwrap()
            .clone();
        assert_eq!(pe, vec![-0.5, -1.5, -2.5]);
        // A truncated file is an error, not a short frame.
        let cur = Cursor::from_bytes(&b[..b.len() - 4], "t.bin");
        assert!(read_next(&cur).is_err());
    }

    /// Pre-magic files: `dump atom` scaled columns, inferred from size_one.
    #[test]
    fn reads_an_old_format_dump_atom_file() {
        let mut b = Vec::new();
        b.extend_from_slice(&7i64.to_le_bytes());
        b.extend_from_slice(&1i64.to_le_bytes());
        b.extend_from_slice(&0i32.to_le_bytes());
        for _ in 0..6 {
            b.extend_from_slice(&0i32.to_le_bytes());
        }
        for v in [0.0, 4.0, 0.0, 4.0, 0.0, 4.0] {
            b.extend_from_slice(&f64::to_le_bytes(v));
        }
        b.extend_from_slice(&5i32.to_le_bytes());
        b.extend_from_slice(&1i32.to_le_bytes());
        b.extend_from_slice(&5i32.to_le_bytes());
        for v in [1.0, 1.0, 0.5, 0.25, 0.0] {
            b.extend_from_slice(&f64::to_le_bytes(v));
        }
        let f = read_bytes(&b).remove(0);
        assert_eq!(f.meta.timestep, 7);
        assert_eq!(f.particles.positions[0], [2.0, 1.0, 0.0]);
    }

    #[test]
    fn a_non_integer_id_is_rejected() {
        let mut parts = ParticleData::with_capacity(1);
        parts.add_atom([1.0, 1.0, 1.0], 3, 1);
        let f = Frame::new(UnitCell::orthorhombic(5.0, 5.0, 5.0), parts);
        let mut out = Vec::new();
        write_dump_binary(&mut out, &[f]).unwrap();
        // patch the id (first double of the only row) to 3.5
        let at = out.len() - 5 * 8;
        out[at..at + 8].copy_from_slice(&3.5f64.to_le_bytes());
        let cur = Cursor::from_bytes(&out, "t.bin");
        let e = read_next(&cur).unwrap_err().to_string();
        assert!(e.contains("not an integer"), "{e}");
    }
}
