//! XYZ / extended-XYZ reader and writer (design §8.5).
//!
//! - Plain XYZ: `N`, blank comment, `elem x y z` rows.
//! - extended XYZ: plain XYZ plus a comment line of `key="value"` pairs where
//!   `Lattice="ax ay az bx by bz cx cy cz"` uses the **ASE/OVITO vector-major
//!   row order** (rows a,b,c) and `pbc="T T T"`. `Properties=species:S:1:pos:R:3`
//!   selects which columns are which.

use crate::cell::{Mat3, UnitCell};
use crate::data::frame::Frame;
use crate::data::particle::ParticleData;
use crate::data::property::{DType, Property, PropertyStore, TypedVec};
use crate::error::{Error, Result};
use crate::io::engine::Cursor;
use crate::io::writer::{plain_name, select_props, FloatFmt, PropCol, WriteOptions};
use std::io::Write;
use std::path::Path;

pub struct XyzReader;

impl XyzReader {
    pub fn read_all(path: &Path) -> Result<Vec<Frame>> {
        let cur = Cursor::from_file(path)?;
        let mut frames = Vec::new();
        while !cur.eof() {
            let peek = cur.peek_line().unwrap_or(&b""[..]);
            if peek.is_empty() {
                cur.next_line();
                continue;
            }
            frames.push(parse_frame(&cur)?);
        }
        Ok(frames)
    }

    pub fn read_first(path: &Path) -> Result<Frame> {
        Self::read_all(path)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidData("empty xyz file".into()))
    }
}

fn parse_frame(cur: &Cursor) -> Result<Frame> {
    parse_single_frame(cur)
}

fn err(cur: &Cursor, msg: &str) -> Error {
    Error::at_line(&cur.path, cur.line_no(), msg)
}

/// Comment line metadata.
pub(crate) struct CommentMeta {
    pub(crate) cell: Option<UnitCell>,
    /// Explicit `species=` list, when the writer provided one.
    pub(crate) species: Vec<String>,
    /// Column layout from `Properties=`, as `(name, kind, components)` pairs.
    pub(crate) columns: Vec<(String, String, usize)>,
    /// `Timestep=` (the key RMDA, OVITO and ASE's LAMMPS tools write).
    pub(crate) timestep: Option<i64>,
    /// `Time=`.
    pub(crate) time: Option<f64>,
}

/// The stand-in box for input with no cell of its own: a 1e30 cube.
///
/// Kept for the callers that need to *name* the "no cell" state. The reader
/// itself does not hand this box to an analysis: see [`infer_cell`].
pub const NO_CELL_DEFAULT: f64 = 1e30;

/// Shortest edge an inferred box may have, in ångström.
///
/// A cell whose edge is zero is not a box: the cell list divides the span by
/// the bin size, and a span of zero has no bins in it. One ångström is short
/// enough not to change what a real cell measures and long enough to be
/// divided.
const MIN_INFERRED_EDGE: f64 = 1.0;

/// The box a cell-less file is measured in: the atoms' own bounding box.
///
/// A file with no `Lattice` line carries neither a box nor a periodicity, and
/// a stand-in such as a 1e30 cube is not a box at
/// all but a way of saying "none". It is a bad one, because every radius in
/// the toolkit is a density estimate and the density comes from the cell's
/// volume: a 1e30 cube has a volume of 1e90, so the density of a 67157-atom
/// crystal read out of it is 6.7e-86 atoms/å^3, the radius holding eighteen
/// neighbours comes out at 7e8 å, and the neighbour search that follows is
/// quadratic in the atoms and never finishes.
///
/// The bounding box is the same "no box" answer with numbers that mean
/// something: the atoms are where they are, the box is as long as they are,
/// and the density that comes out of it is the density they actually have.
/// The *semantics* do not change -- a file that did not say it was periodic is
/// still not treated as periodic -- so this fixes the numbers without
/// changing what the analysis believes.
///
/// **Not periodic.** A box cut to the atoms and treated as periodic would put
/// each atom's own image next to it: the image of an atom at `x = 0.1` sits at
/// `x = L + 0.1`, and the atoms at the far face are then each other's
/// neighbours. A cell-less file is read as the isolated system it says it is;
/// `rmda --cell A B C` is how a periodic box is stated.
pub fn infer_cell(positions: &[crate::cell::Coord3]) -> UnitCell {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in positions {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    if !lo[0].is_finite() {
        // No atoms at all: a well-formed box, so nothing downstream has to
        // special-case an empty frame's cell.
        return UnitCell::orthorhombic(MIN_INFERRED_EDGE, MIN_INFERRED_EDGE, MIN_INFERRED_EDGE)
            .with_pbc([false; 3]);
    }
    let mut h = [[0.0f64; 3]; 3];
    for d in 0..3 {
        h[d][d] = (hi[d] - lo[d]).max(MIN_INFERRED_EDGE);
    }
    UnitCell::from_matrix(crate::cell::Mat3 { m: h }, [false; 3], lo)
}

/// Where one extended-XYZ field goes, resolved once per frame from
/// `Properties=` (design §8.5).
#[derive(Clone, Copy, PartialEq, Debug)]
enum XCol {
    Species,
    Pos(usize),
    Type,
    Selection,
    /// `id`: the atom id.
    Id,
    /// Component `comp` of extra property `k`.
    Extra(usize, usize),
}

/// How an extra column's text becomes a value.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ExtraKind {
    Real,
    Int,
    Logical,
    /// Any other kind (`S`): kept as NaN.
    Other,
}

/// One chunk's output windows.
struct XyzSink<'a, 'b> {
    pos: &'a mut [crate::cell::Coord3],
    types: &'a mut [u32],
    selection: &'a mut [u8],
    ids: &'a mut [i64],
    extra: Vec<ExtraSlice<'a>>,
    /// Species seen in this chunk, in first-seen order; `types` holds
    /// `index + 1` into it until the merge when the file has no type column.
    species: Vec<&'b [u8]>,
    /// With a `type` column and no `species=` key: the symbol each type's
    /// atoms carry in this chunk (index type-1).
    type_sym: Vec<TypeSymbol<'b>>,
}

/// The symbol the atoms of one type carry, as far as the rows seen so far say.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum TypeSymbol<'b> {
    Unseen,
    One(&'b [u8]),
    /// Atoms of this type carry different symbols.
    Mixed,
}

impl<'b> TypeSymbol<'b> {
    fn merge(self, o: TypeSymbol<'b>) -> TypeSymbol<'b> {
        match (self, o) {
            (TypeSymbol::Unseen, x) | (x, TypeSymbol::Unseen) => x,
            (TypeSymbol::One(a), TypeSymbol::One(b)) if a == b => self,
            _ => TypeSymbol::Mixed,
        }
    }
}

/// Note that an atom of type `t` (1-based) carries `sym`.
#[inline]
pub(crate) fn note_type_symbol<'b>(tab: &mut Vec<TypeSymbol<'b>>, t: u32, sym: &'b [u8]) {
    let k = t as usize - 1;
    if tab.len() <= k {
        tab.resize(k + 1, TypeSymbol::Unseen);
    }
    tab[k] = tab[k].merge(TypeSymbol::One(sym));
}

/// The labels [`labels_from_type_symbols`] gives a frame whose atom `i` has
/// type `types[i]` and symbol `symbols[i]`: the extxyz reader's rule for a
/// `type` column without `species=`, for callers that build a frame from
/// another source (an ASE `Atoms` with a `type` array).
pub fn labels_for_typed_symbols(types: &[u32], symbols: &[&str]) -> Vec<String> {
    let mut tab = Vec::new();
    for (&t, s) in types.iter().zip(symbols) {
        if t >= 1 {
            note_type_symbol(&mut tab, t, s.as_bytes());
        }
    }
    labels_from_type_symbols(&tab)
}

/// Merge per-chunk tables (chunk order does not matter).
pub(crate) fn merge_type_symbols<'b>(tabs: &[&[TypeSymbol<'b>]]) -> Vec<TypeSymbol<'b>> {
    let n = tabs.iter().map(|t| t.len()).max().unwrap_or(0);
    (0..n)
        .map(|k| {
            tabs.iter().fold(TypeSymbol::Unseen, |acc, t| {
                acc.merge(t.get(k).copied().unwrap_or(TypeSymbol::Unseen))
            })
        })
        .collect()
}

/// Type labels of an extxyz frame with a `type` column and no `species=`
/// key (design §8.5): type `t` is labelled with the symbol its atoms carry
/// when every atom of type `t` carries the same symbol and no other type
/// carries it -- a label names one type, as LAMMPS type labels and OVITO
/// type names do. A type without such a symbol (absent, mixed symbols, or a
/// symbol shared with another type, e.g. the `X` placeholder RMDA writes for
/// unlabelled types) gets an empty label; no labels at all when no type has
/// one.
pub(crate) fn labels_from_type_symbols(tab: &[TypeSymbol<'_>]) -> Vec<String> {
    let owners = |s: &[u8]| {
        tab.iter()
            .filter(|t| matches!(t, TypeSymbol::One(x) if *x == s))
            .count()
    };
    let labels: Vec<String> = tab
        .iter()
        .map(|t| match t {
            TypeSymbol::One(s) if owners(s) == 1 => String::from_utf8_lossy(s).into_owned(),
            _ => String::new(),
        })
        .collect();
    if labels.iter().all(String::is_empty) {
        Vec::new()
    } else {
        labels
    }
}

enum ExtraSlice<'a> {
    F64(&'a mut [f64]),
    I64(&'a mut [i64]),
    U8(&'a mut [u8]),
}

fn parse_single_frame(cur: &Cursor) -> Result<Frame> {
    parse_single_frame_with(cur, &crate::io::ReadOptions::all())
}

fn parse_single_frame_with(cur: &Cursor, opts: &crate::io::ReadOptions) -> Result<Frame> {
    use crate::io::engine::{
        next_f64, next_i64, next_token, parse_f64, parse_rows, row_chunks, split_by_chunks, RowRule,
    };
    use rayon::prelude::*;

    let nline = cur.next_line().ok_or_else(|| err(cur, "unexpected EOF"))?;
    let n: usize = std::str::from_utf8(nline)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .ok_or_else(|| err(cur, "bad atom count"))?;

    let comment = cur
        .next_line()
        .ok_or_else(|| err(cur, "unexpected EOF after count"))?
        .to_vec();
    let meta = parse_comment(&comment)?;

    let columns: Vec<(String, String, usize)> = if meta.columns.is_empty() {
        vec![
            ("species".into(), "S".into(), 1),
            ("pos".into(), "R".into(), 3),
        ]
    } else {
        meta.columns.clone()
    };
    // Field plan; the first `species`, `pos`, `type` and `Selection` win.
    let mut plan: Vec<XCol> = Vec::new();
    let mut extra: Vec<(String, ExtraKind, usize)> = Vec::new();
    let (mut has_species, mut has_pos, mut has_type, mut has_sel) = (false, false, false, false);
    let mut has_id = false;
    for (name, kind, count) in &columns {
        let nm = name.as_str();
        if nm == "species" && !has_species {
            has_species = true;
            plan.push(XCol::Species);
            plan.extend((1..*count).map(|_| XCol::Extra(usize::MAX, 0)));
        } else if nm == "pos" && !has_pos {
            has_pos = true;
            plan.extend((0..*count).map(|d| {
                if d < 3 {
                    XCol::Pos(d)
                } else {
                    XCol::Extra(usize::MAX, 0)
                }
            }));
        } else if nm == "type" && !has_type {
            has_type = true;
            plan.push(XCol::Type);
            plan.extend((1..*count).map(|_| XCol::Extra(usize::MAX, 0)));
        } else if (nm == "Selection" || nm == "selection") && !has_sel {
            has_sel = true;
            plan.push(XCol::Selection);
            plan.extend((1..*count).map(|_| XCol::Extra(usize::MAX, 0)));
        } else if nm == "id" && !has_id && !opts.skips(nm) {
            // The atom id, as the dump and data readers route it; a skipped
            // `id` leaves the ids 1..N.
            has_id = true;
            plan.push(XCol::Id);
            plan.extend((1..*count).map(|_| XCol::Extra(usize::MAX, 0)));
        } else if matches!(
            nm,
            "species" | "pos" | "type" | "id" | "Selection" | "selection"
        ) || opts.skips(nm)
        {
            // A repeated special column is skipped, and so is a column the
            // read options drop.
            plan.extend((0..*count).map(|_| XCol::Extra(usize::MAX, 0)));
        } else {
            let k = extra.len();
            let ek = match kind.as_str() {
                "R" => ExtraKind::Real,
                "I" => ExtraKind::Int,
                "L" => ExtraKind::Logical,
                _ => ExtraKind::Other,
            };
            extra.push((name.clone(), ek, *count));
            plan.extend((0..*count).map(|c| XCol::Extra(k, c)));
        }
    }
    if !(has_species && has_pos) {
        return Err(err(cur, "Properties= must name both `species` and `pos`"));
    }
    if plan.iter().filter(|c| matches!(c, XCol::Pos(_))).count() < 3 {
        return Err(err(cur, "`pos` needs 3 components"));
    }
    let expected = plan.len();

    let body = cur.rest();
    let (len, nlines, chunks) = row_chunks(body, n, RowRule::Every).ok_or_else(|| {
        Error::at_line(
            &cur.path,
            cur.line_no() + crate::io::engine::count_newlines(body),
            "truncated atom block",
        )
    })?;
    let body = &body[..len];

    let mut pos = vec![[0.0f64; 3]; n];
    let mut types = vec![0u32; n];
    let mut selection = if has_sel { vec![0u8; n] } else { Vec::new() };
    let mut ids = if has_id { vec![0i64; n] } else { Vec::new() };
    enum ExtraVec {
        F64(Vec<f64>),
        I64(Vec<i64>),
        U8(Vec<u8>),
    }
    let mut extra_data: Vec<ExtraVec> = extra
        .iter()
        .map(|(_, k, c)| match k {
            ExtraKind::Int => ExtraVec::I64(vec![0; n * c]),
            ExtraKind::Logical => ExtraVec::U8(vec![0; n * c]),
            _ => ExtraVec::F64(vec![0.0; n * c]),
        })
        .collect();
    let mut sinks: Vec<XyzSink> = {
        let mut p = split_by_chunks(&mut pos, &chunks, 1).into_iter();
        let mut t = split_by_chunks(&mut types, &chunks, 1).into_iter();
        let mut sel = split_by_chunks(&mut selection, &chunks, usize::from(has_sel)).into_iter();
        let mut idv = split_by_chunks(&mut ids, &chunks, usize::from(has_id)).into_iter();
        let mut ex: Vec<std::vec::IntoIter<ExtraSlice>> = extra_data
            .iter_mut()
            .zip(&extra)
            .map(|(v, (_, _, c))| match v {
                ExtraVec::F64(v) => split_by_chunks(v, &chunks, *c)
                    .into_iter()
                    .map(ExtraSlice::F64)
                    .collect::<Vec<_>>()
                    .into_iter(),
                ExtraVec::I64(v) => split_by_chunks(v, &chunks, *c)
                    .into_iter()
                    .map(ExtraSlice::I64)
                    .collect::<Vec<_>>()
                    .into_iter(),
                ExtraVec::U8(v) => split_by_chunks(v, &chunks, *c)
                    .into_iter()
                    .map(ExtraSlice::U8)
                    .collect::<Vec<_>>()
                    .into_iter(),
            })
            .collect();
        (0..chunks.len())
            .map(|_| XyzSink {
                pos: p.next().unwrap(),
                types: t.next().unwrap(),
                selection: sel.next().unwrap(),
                ids: idv.next().unwrap(),
                extra: ex.iter_mut().map(|it| it.next().unwrap()).collect(),
                species: Vec::new(),
                type_sym: Vec::new(),
            })
            .collect()
    };
    // Pair each type with its symbol only when the labels must come from
    // the rows: a `type` column and no `species=` key.
    let pair_types = has_type && meta.species.is_empty();

    let res = parse_rows(body, &chunks, &mut sinks, RowRule::Every, |s, r, line| {
        let mut at = 0usize;
        let short = |line: &[u8]| {
            let mut i = 0;
            let mut nf = 0;
            while next_token(line, &mut i).is_some() {
                nf += 1;
            }
            format!("atom row has {nf} fields, Properties= needs {expected}")
        };
        let mut v = [0.0f64; 3];
        let mut sym: &[u8] = b"";
        for col in &plan {
            match *col {
                XCol::Species => {
                    let tok = next_token(line, &mut at).ok_or_else(|| short(line))?;
                    if !has_type {
                        let k = match s.species.iter().position(|e| *e == tok) {
                            Some(k) => k,
                            None => {
                                s.species.push(tok);
                                s.species.len() - 1
                            }
                        };
                        s.types[r] = k as u32 + 1;
                    } else {
                        sym = tok;
                    }
                }
                XCol::Pos(d) => {
                    v[d] = next_f64(line, &mut at).map_err(|e| match e {
                        None => short(line),
                        Some(_) => "bad coordinates".to_string(),
                    })?
                }
                XCol::Type => {
                    s.types[r] = match next_f64(line, &mut at) {
                        Ok(v) if v >= 1.0 && v.fract() == 0.0 => v as u32,
                        Ok(v) => {
                            return Err(format!("type column holds {v}, not a positive integer"))
                        }
                        Err(None) => return Err(short(line)),
                        Err(Some(_)) => return Err("bad type column".into()),
                    }
                }
                XCol::Selection => {
                    s.selection[r] = match next_f64(line, &mut at) {
                        Ok(v) if v == 0.0 || v == 1.0 => v as u8,
                        Ok(v) => return Err(format!("Selection column holds {v}, not 0 or 1")),
                        Err(None) => return Err(short(line)),
                        Err(Some(_)) => return Err("bad Selection column".into()),
                    }
                }
                XCol::Id => {
                    let tok = next_token(line, &mut at).ok_or_else(|| short(line))?;
                    let mut i = 0;
                    s.ids[r] = match next_i64(tok, &mut i) {
                        Ok(v) if i == tok.len() => v,
                        _ => match parse_f64(tok) {
                            Some(v) if v.fract() == 0.0 && v.abs() < 9.0e15 => v as i64,
                            _ => {
                                return Err(format!(
                                    "id column holds {:?}, not an integer",
                                    String::from_utf8_lossy(tok)
                                ))
                            }
                        },
                    };
                }
                XCol::Extra(k, c) => {
                    let tok = next_token(line, &mut at).ok_or_else(|| short(line))?;
                    if k == usize::MAX {
                        continue;
                    }
                    let comps = extra[k].2;
                    match &mut s.extra[k] {
                        ExtraSlice::F64(out) => {
                            out[r * comps + c] = if extra[k].1 == ExtraKind::Real {
                                parse_f64(tok).unwrap_or(f64::NAN)
                            } else {
                                f64::NAN
                            }
                        }
                        ExtraSlice::I64(out) => {
                            let mut i = 0;
                            out[r * comps + c] = next_i64(tok, &mut i)
                                .ok()
                                .or_else(|| parse_f64(tok).map(|v| v as i64))
                                .unwrap_or(0)
                        }
                        ExtraSlice::U8(out) => {
                            out[r * comps + c] =
                                u8::from(matches!(tok, b"T" | b"t" | b"True" | b"true" | b"1"))
                        }
                    }
                }
            }
        }
        // Fields beyond `Properties=` are tolerated.
        s.pos[r] = v;
        if pair_types {
            note_type_symbol(&mut s.type_sym, s.types[r], sym);
        }
        Ok(())
    });
    res.map_err(|e| Error::at_line(&cur.path, cur.line_no() + e.line + 1, e.msg))?;
    cur.advance(len, nlines);

    // Labels: the file's `species=` list; else, with a `type` column, each
    // type's own symbol (`labels_from_type_symbols`); else (the symbols are
    // the types) the sorted set of the symbols present.
    let labels: Vec<String> = if !meta.species.is_empty() {
        meta.species.clone()
    } else if has_type {
        let tabs: Vec<&[TypeSymbol]> = sinks.iter().map(|s| s.type_sym.as_slice()).collect();
        labels_from_type_symbols(&merge_type_symbols(&tabs))
    } else {
        let mut syms: Vec<String> = sinks
            .iter()
            .flat_map(|s| s.species.iter())
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect();
        syms.sort();
        syms.dedup();
        syms
    };
    if !has_type {
        // Remap each chunk's local species indices to global types.
        let maps: Vec<Vec<u32>> = sinks
            .iter()
            .map(|s| {
                s.species
                    .iter()
                    .map(|b| {
                        let name = String::from_utf8_lossy(b);
                        labels
                            .iter()
                            .position(|l| *l == name)
                            .map(|i| i as u32 + 1)
                            .ok_or_else(|| {
                                err(cur, &format!("element {name} is not a known species"))
                            })
                    })
                    .collect::<Result<Vec<u32>>>()
            })
            .collect::<Result<_>>()?;
        sinks
            .par_iter_mut()
            .zip(maps.par_iter())
            .for_each(|(s, m)| s.types.iter_mut().for_each(|t| *t = m[*t as usize - 1]));
    }
    drop(sinks);

    let mut parts = ParticleData::new(n);
    parts.positions = pos;
    parts.ids = if has_id {
        ids
    } else {
        (1..=n as i64).collect()
    };
    parts.types = types;
    parts.selection = has_sel.then_some(selection);

    let (cell, cell_inferred) = match meta.cell {
        Some(c) => (c, false),
        None => (infer_cell(&parts.positions), true),
    };

    let mut frame = Frame::new(cell, parts);
    frame.meta.cell_inferred = cell_inferred;
    frame.meta.timestep = meta.timestep.unwrap_or(0);
    frame.meta.time = meta.time;
    frame.type_labels = labels;
    for ((name, _, comps), data) in extra.into_iter().zip(extra_data) {
        let data = match data {
            ExtraVec::F64(v) => TypedVec::F64(v),
            ExtraVec::I64(v) => TypedVec::I64(v),
            ExtraVec::U8(v) => TypedVec::U8(v),
        };
        frame
            .particles
            .props
            .register(Property::from_data(name, comps, data));
    }
    Ok(frame)
}

pub(crate) fn parse_comment(comment: &[u8]) -> Result<CommentMeta> {
    if comment.is_empty() {
        return Ok(CommentMeta {
            cell: None,
            species: Vec::new(),
            columns: Vec::new(),
            timestep: None,
            time: None,
        });
    }
    let text =
        std::str::from_utf8(comment).map_err(|_| Error::InvalidData("comment not utf8".into()))?;

    let mut cell: Option<UnitCell> = None;
    let mut pbc = [true; 3];
    let mut origin_key: Option<[f64; 3]> = None;
    let mut species: Vec<String> = Vec::new();
    let mut columns: Vec<(String, String, usize)> = Vec::new();
    let mut timestep: Option<i64> = None;
    let mut time: Option<f64> = None;

    for (key, value) in kv_pairs(text) {
        match key.as_str() {
            // An integer, or a real holding one (`Timestep=1000.0`); anything
            // else is not a timestep and is dropped like an unknown key.
            "Timestep" | "timestep" => {
                let v = value.trim();
                timestep = v.parse::<i64>().ok().or_else(|| {
                    v.parse::<f64>()
                        .ok()
                        .filter(|x| x.fract() == 0.0 && x.abs() < 9.0e15)
                        .map(|x| x as i64)
                });
            }
            "Time" | "time" => time = value.trim().parse::<f64>().ok().filter(|x| x.is_finite()),
            "Lattice" | "lattice" => {
                // ASE/OVITO order is vector-major: ax ay az bx by bz cx cy cz,
                // i.e. the three cell vectors in sequence, optionally followed by
                // the three origin components. Atomsk's component-major spelling
                // is deliberately NOT auto-detected.
                //
                // Nine numbers is the standard form; the origin then comes from
                // `Origin=` (OVITO) or is zero. Twelve numbers (nine plus the
                // origin) are also accepted on read.
                let v: Vec<f64> = value
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                let (m, origin) = match v.len() {
                    9 => (
                        Mat3::from_rows([v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]),
                        [0.0, 0.0, 0.0],
                    ),
                    12 => (
                        Mat3::from_rows([v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]),
                        [v[9], v[10], v[11]],
                    ),
                    n => {
                        return Err(Error::InvalidData(format!(
                            "Lattice= needs 9 or 12 numbers, got {n}"
                        )))
                    }
                };
                cell = Some(UnitCell::from_matrix(m, pbc, origin));
            }
            // OVITO's spelling of the cell origin. It wins over a 12-number
            // `Lattice=` if a file somehow carries both.
            "Origin" | "origin" => {
                let v: Vec<f64> = value
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if v.len() != 3 {
                    return Err(Error::InvalidData(format!(
                        "Origin= needs 3 numbers, got {}",
                        v.len()
                    )));
                }
                origin_key = Some([v[0], v[1], v[2]]);
            }
            "pbc" | "PBC" => {
                let toks: Vec<&str> = value.split_whitespace().collect();
                if toks.len() == 3 {
                    for (d, t) in toks.iter().enumerate() {
                        pbc[d] = matches!(*t, "T" | "t" | "True" | "1");
                    }
                }
            }
            "species" | "Species" => {
                species = value.split_whitespace().map(|s| s.to_string()).collect();
            }
            "Properties" | "properties" => {
                // `name:kind:count` triples, e.g. `species:S:1:pos:R:3:type:I:1`.
                let toks: Vec<&str> = value.split(':').collect();
                let mut cols = Vec::new();
                for t in toks.chunks_exact(3) {
                    let count: usize = t[2].trim().parse().map_err(|_| {
                        Error::InvalidData(format!("Properties= count {:?} is not a number", t[2]))
                    })?;
                    cols.push((t[0].trim().to_string(), t[1].trim().to_string(), count));
                }
                columns = cols;
            }
            _ => {}
        }
    }

    if let Some(mut c) = cell {
        c.pbc = pbc;
        if let Some(o) = origin_key {
            c.origin = o;
        }
        cell = Some(c);
    }

    Ok(CommentMeta {
        cell,
        species,
        columns,
        timestep,
        time,
    })
}

/// Split an extended-XYZ comment line into `(key, value)` pairs, following
/// ASE's `key_val_str_to_dict` grammar (design §8.5):
///
/// - `key=value`, with optional blanks around `=`;
/// - a value (or key) in double quotes may hold blanks, with `\"` and `\\`
///   escapes;
/// - a `{...}` or `[...]` value is an array: its text with commas turned to
///   blanks (`pbc=[T, T, F]` reads like `pbc="T T F"`);
/// - a bare token with no `=` is a boolean flag (true in ASE); it is stepped
///   over and parsing carries on after it.
fn kv_pairs(text: &str) -> Vec<(String, String)> {
    let bytes = text.as_bytes();
    let blank = |b: u8| b == b' ' || b == b'\t' || b == b'\r' || b == b'\n';
    let skip_blank = |i: &mut usize| {
        while *i < bytes.len() && blank(bytes[*i]) {
            *i += 1;
        }
    };
    // A quoted string starting at `*i` (on the opening quote).
    let quoted = |i: &mut usize| -> Vec<u8> {
        let mut s = Vec::new();
        *i += 1;
        while *i < bytes.len() && bytes[*i] != b'"' {
            if bytes[*i] == b'\\' && *i + 1 < bytes.len() {
                *i += 1;
            }
            s.push(bytes[*i]);
            *i += 1;
        }
        *i = (*i + 1).min(bytes.len());
        s
    };
    let lossy = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    let mut out = Vec::new();
    let mut i = 0;
    loop {
        skip_blank(&mut i);
        if i >= bytes.len() {
            break;
        }
        let key = if bytes[i] == b'"' {
            quoted(&mut i)
        } else {
            let k0 = i;
            while i < bytes.len() && bytes[i] != b'=' && !blank(bytes[i]) {
                i += 1;
            }
            bytes[k0..i].to_vec()
        };
        let mut j = i;
        skip_blank(&mut j);
        if j >= bytes.len() || bytes[j] != b'=' {
            // A bare flag: true in ASE. Every key the reader interprets needs
            // a value, and there is no frame attribute store, so the flag is
            // stepped over -- it does not end the line -- and not returned:
            // a plain-XYZ title such as `Lattice constant 3.6` must not become
            // `Lattice=T`.
            continue;
        }
        i = j + 1;
        skip_blank(&mut i);
        let val = if i >= bytes.len() {
            Vec::new()
        } else if bytes[i] == b'"' {
            quoted(&mut i)
        } else if bytes[i] == b'{' || bytes[i] == b'[' {
            let close = if bytes[i] == b'{' { b'}' } else { b']' };
            let v0 = i + 1;
            while i < bytes.len() && bytes[i] != close {
                i += 1;
            }
            let v: Vec<u8> = bytes[v0..i.min(bytes.len())]
                .iter()
                .map(|&b| if b == b',' { b' ' } else { b })
                .collect();
            i = (i + 1).min(bytes.len());
            v
        } else {
            let v0 = i;
            while i < bytes.len() && !blank(bytes[i]) {
                i += 1;
            }
            bytes[v0..i].to_vec()
        };
        if !key.is_empty() {
            out.push((lossy(&key), lossy(&val)));
        }
    }
    out
}

/// Where the written cell's origin is placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum XyzOrigin {
    /// Write the input cell's own origin (the default).
    ///
    /// The written box is then the same box the input had, which is what makes
    /// an overlay line up: atom coordinates and cell borders are in one
    /// coordinate system.
    #[default]
    Native,
    /// Shift the box so its origin is `(0, 0, 0)`, translating atom coordinates
    /// by the same amount.
    ///
    /// For readers that ignore the `Lattice=` origin entirely -- and for tools
    /// that assume a box starts at zero -- this is the form that lines up.
    /// The translation is not optional: zeroing the origin while leaving
    /// absolute coordinates in place would put every atom that far from where
    /// its own cell says it is.
    Zero,
}

/// Writer to an extended-XYZ stream (ASE vector-major convention).
pub struct XyzWriter;

impl XyzWriter {
    pub fn write_all<W: Write>(w: &mut W, frames: &[Frame]) -> Result<()> {
        for f in frames {
            Self::write_frame(w, f)?;
        }
        Ok(())
    }

    /// [`XyzWriter::write_all`] with an explicit origin placement.
    pub fn write_all_with_origin<W: Write>(
        w: &mut W,
        frames: &[Frame],
        origin: XyzOrigin,
    ) -> Result<()> {
        for f in frames {
            Self::write_frame_with_origin(w, f, &PropertyStore::new(), origin)?;
        }
        Ok(())
    }

    pub fn write_frame<W: Write>(w: &mut W, f: &Frame) -> Result<()> {
        Self::write_frame_with(w, f, &PropertyStore::new())
    }

    /// Write one frame, emitting every per-particle property as an extra
    /// extended-XYZ column.
    ///
    /// A `components > 1` property is stored **atom-major**
    /// (`data[atom*components + component]`), which is also how extended XYZ
    /// lays out an atom's values, so no transposition is needed. That layout is
    /// what [`crate::data::property::Property::push_value`] produces, what the
    /// PTM engine emits (`orientations.extend_from_slice(&r.orientation)` per
    /// atom), and what the reference test reads (`chunks_exact(4)`).
    ///
    /// Columns follow `PropertyStore` registration order, which is deterministic
    /// for a given analysis, so the header is stable between runs.
    pub fn write_frame_with<W: Write>(w: &mut W, f: &Frame, props: &PropertyStore) -> Result<()> {
        Self::write_frame_with_origin(w, f, props, XyzOrigin::Native)
    }

    /// [`XyzWriter::write_frame_with`] with an explicit origin placement.
    ///
    /// `XyzOrigin::Zero` translates the atom coordinates and zeroes the lattice
    /// origin together, so the geometry inside the box is unchanged and only
    /// where the box is drawn moves.
    pub fn write_frame_with_origin<W: Write>(
        w: &mut W,
        f: &Frame,
        props: &PropertyStore,
        origin: XyzOrigin,
    ) -> Result<()> {
        let opts = WriteOptions {
            origin,
            ..WriteOptions::default()
        };
        Self::write_frame_opts(w, f, props, &opts)
    }

    /// Write one frame with explicit [`WriteOptions`].
    ///
    /// `extra` holds an analysis' properties, written after the frame's own
    /// (`opts.props`) with the default precision of 10 decimals. With
    /// `WriteOptions::default()` (plus an origin) the bytes are those of
    /// [`XyzWriter::write_frame_with_origin`].
    ///
    /// `opts.extended` adds an `id:I:1` column and, when the frame has one,
    /// `Selection:I:1`, both read back into the atoms, and a `species="..."`
    /// key listing the type labels in type order, so labels survive a round
    /// trip even when the alphabetical order of the symbols differs.
    pub fn write_frame_opts<W: Write>(
        w: &mut W,
        f: &Frame,
        props: &PropertyStore,
        opts: &WriteOptions,
    ) -> Result<()> {
        let n = f.natoms();
        let origin = opts.origin;
        let extra: Vec<&Property> = props.iter().filter(|p| p.per_particle).collect();
        let (own, _) = select_props(f, &opts.props, extxyz_prop_ok)?;
        let own: Vec<(&Property, PropCol)> = own.into_iter().map(|p| (p, PropCol::of(p))).collect();
        let pf = opts.precision.unwrap_or(FloatFmt::Fixed(10));
        let own_fmt = opts.prop_precision.unwrap_or(FloatFmt::Shortest);
        let extra_fmt = opts.prop_precision.unwrap_or(FloatFmt::Fixed(10));
        let ids = opts.extended && f.particles.ids.len() == n;
        let selection = f
            .particles
            .selection
            .as_deref()
            .filter(|s| opts.extended && s.len() == n);

        let mut spec = String::from("species:S:1:pos:R:3:type:I:1");
        if ids {
            spec.push_str(":id:I:1");
        }
        if selection.is_some() {
            spec.push_str(":Selection:I:1");
        }
        for p in own.iter().map(|(p, _)| *p).chain(extra.iter().copied()) {
            spec.push_str(&format!(
                ":{}:{}:{}",
                p.name,
                extxyz_kind(p.dtype),
                p.components
            ));
        }
        // Labels in type order; omitted when one cannot be a bare token.
        let species_key = (opts.extended
            && !f.type_labels.is_empty()
            && f.type_labels.iter().all(|l| plain_name(l).is_ok()))
        .then(|| format!(" species=\"{}\"", f.type_labels.join(" ")))
        .unwrap_or_default();
        // `Time=` only when the frame has a time (a dump's `ITEM: TIME`); a
        // frame without one gets no `Time=` key.
        let species_key = match f.meta.time {
            Some(t) => format!(" Time={t:?}{species_key}"),
            None => species_key,
        };

        writeln!(w, "{}", n)?;
        let h = f.cell.h_matrix();
        let (r0, r1, r2) = (h.row(0), h.row(1), h.row(2));
        let pbc = f.cell.pbc;
        let pbc_str: Vec<&str> = pbc.iter().map(|&b| if b { "T" } else { "F" }).collect();
        // Nine `Lattice=` numbers plus OVITO's `Origin=` key. The origin is what
        // makes the written box the *same* box as the input's: a LAMMPS `data`
        // file places `xlo` wherever the simulation put it. ASE rejects a
        // 12-number `Lattice=`, so the origin cannot ride along there; ASE keeps
        // `Origin` as an unknown key, and OVITO reads it as the cell origin.
        let (o, shift) = match origin {
            XyzOrigin::Native => (f.cell.origin, [0.0, 0.0, 0.0]),
            XyzOrigin::Zero => {
                let o = f.cell.origin;
                ([0.0, 0.0, 0.0], [-o[0], -o[1], -o[2]])
            }
        };
        if pf == FloatFmt::Fixed(10) {
            writeln!(
                w,
                "Lattice=\"{:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10}\" Origin=\"{:.10} {:.10} {:.10}\" pbc=\"{}\" Properties={} Timestep={}{}",
                r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], o[0], o[1], o[2],
                pbc_str.join(" "),
                spec,
                f.meta.timestep,
                species_key,
            )?;
        } else {
            let mut head: Vec<u8> = Vec::new();
            let nums = |head: &mut Vec<u8>, vals: &[f64]| -> std::io::Result<()> {
                for (k, v) in vals.iter().enumerate() {
                    if k > 0 {
                        head.push(b' ');
                    }
                    pf.write(head, *v)?;
                }
                Ok(())
            };
            head.extend_from_slice(b"Lattice=\"");
            nums(
                &mut head,
                &[
                    r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2],
                ],
            )?;
            head.extend_from_slice(b"\" Origin=\"");
            nums(&mut head, &o)?;
            writeln!(
                head,
                "\" pbc=\"{}\" Properties={} Timestep={}{}",
                pbc_str.join(" "),
                spec,
                f.meta.timestep,
                species_key
            )?;
            w.write_all(&head)?;
        }
        crate::io::writer::write_rows(w, n, |w, i| {
            let p = f.particles.positions[i];
            let t = f.particles.types[i];
            let elem = f
                .type_labels
                .get((t as usize).saturating_sub(1))
                .map_or("X", String::as_str);
            // `type` carries the numeric identity, which `species` cannot when
            // the input had no element labels -- a five-component LAMMPS alloy
            // arrives as "X" for every atom, and writing only that back loses
            // which atoms are which. The number is the identity; the symbol is
            // a label for readers that have nothing better.
            if pf == FloatFmt::Fixed(10) {
                write!(
                    w,
                    "{} {:.10} {:.10} {:.10} {}",
                    elem,
                    p[0] + shift[0],
                    p[1] + shift[1],
                    p[2] + shift[2],
                    t
                )?;
            } else {
                w.extend_from_slice(elem.as_bytes());
                for d in 0..3 {
                    w.push(b' ');
                    // No `+ 0.0` in native mode: it would turn -0.0 into 0.0.
                    let x = match origin {
                        XyzOrigin::Native => p[d],
                        XyzOrigin::Zero => p[d] + shift[d],
                    };
                    pf.write(w, x)?;
                }
                write!(w, " {t}")?;
            }
            if ids {
                write!(w, " {}", f.particles.ids[i])?;
            }
            if let Some(sel) = selection {
                write!(w, " {}", sel[i])?;
            }
            for (prop, col) in &own {
                let k = prop.components;
                for c in 0..k {
                    w.push(b' ');
                    col.write(w, i * k + c, own_fmt)?;
                }
            }
            for prop in &extra {
                for c in 0..prop.components {
                    let v = Self::component_at(prop, c, i, prop.components).unwrap_or(f64::NAN);
                    if extxyz_kind(prop.dtype) == "I" {
                        write!(w, " {}", v as i64)?;
                    } else if extra_fmt == FloatFmt::Fixed(10) {
                        write!(w, " {:.10}", v)?;
                    } else {
                        w.push(b' ');
                        extra_fmt.write(w, v)?;
                    }
                }
            }
            writeln!(w)
        })?;
        Ok(())
    }

    /// Component `c` of atom `i` in an atom-major property.
    fn component_at(p: &Property, c: usize, i: usize, components: usize) -> Option<f64> {
        let idx = i * components + c;
        match p.dtype {
            DType::U8 => p.as_u8().and_then(|v| v.get(idx)).map(|&x| x as f64),
            DType::I32 => p.as_i32().and_then(|v| v.get(idx)).map(|&x| x as f64),
            DType::I64 => p.as_i64().and_then(|v| v.get(idx)).map(|&x| x as f64),
            DType::F32 => p.as_f32().and_then(|v| v.get(idx)).map(|&x| x as f64),
            DType::F64 => p.as_f64().and_then(|v| v.get(idx)).copied(),
        }
    }
}

/// Whether a frame property can be an extended-XYZ column: not one of the
/// columns the reader routes into the atoms, and a name `Properties=` can
/// hold.
pub(crate) fn extxyz_prop_ok(p: &Property) -> std::result::Result<(), String> {
    plain_name(&p.name)?;
    if p.name.contains(':') {
        return Err(format!(
            "property name {:?} holds ':', the Properties= separator",
            p.name
        ));
    }
    if matches!(
        p.name.as_ref(),
        "species" | "pos" | "type" | "id" | "Selection" | "selection"
    ) {
        return Err(format!(
            "property {:?} would be read back as an atom field",
            p.name
        ));
    }
    Ok(())
}

/// The extended-XYZ column kind for a dtype: `I` for integers, `R` for reals.
fn extxyz_kind(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 | DType::I32 | DType::I64 => "I",
        DType::F32 | DType::F64 => "R",
    }
}

pub fn read_next_from_cursor(cur: &Cursor) -> Result<Option<Frame>> {
    read_next_with(cur, &crate::io::ReadOptions::all())
}

/// [`read_next_from_cursor`] keeping only the columns `opts` allows.
pub fn read_next_with(cur: &Cursor, opts: &crate::io::ReadOptions) -> Result<Option<Frame>> {
    while !cur.eof() {
        let peek = cur.peek_line().unwrap_or(&b""[..]);
        if peek.is_empty() {
            cur.next_line();
            continue;
        }
        let first = peek.first().copied().unwrap_or(b' ');
        if first == b'#' {
            cur.next_line();
            continue;
        }
        let f = parse_single_frame_with(cur, opts)?;
        return Ok(Some(f));
    }
    Ok(None)
}

/// Step over the frame at the cursor without parsing its atoms;
/// `Ok(false)` at end of input.
pub fn skip_next(cur: &Cursor) -> Result<bool> {
    use crate::io::engine::{rows_extent, RowRule};
    while let Some(peek) = cur.peek_line() {
        if peek.is_empty() || peek[0] == b'#' {
            cur.next_line();
            continue;
        }
        let nline = cur.next_line().unwrap_or_default();
        let n: usize = std::str::from_utf8(nline)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| err(cur, "bad atom count"))?;
        cur.next_line()
            .ok_or_else(|| err(cur, "unexpected EOF after count"))?;
        let (len, lines) = rows_extent(cur.rest(), n, RowRule::Every)
            .ok_or_else(|| err(cur, "truncated atom block"))?;
        cur.advance(len, lines);
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_parser_basics() {
        let pairs = kv_pairs(r#"Lattice="1 2" pbc="T T F" Timestep=3"#);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, "Lattice");
        assert_eq!(pairs[0].1, "1 2");
        assert_eq!(pairs[2].1, "3");
    }

    /// ASE grammar: a bare flag does not end the line, quotes may hold
    /// escapes, and `[...]`/`{...}` values are arrays.
    #[test]
    fn kv_parser_steps_over_bare_flags_and_reads_arrays() {
        let pairs = kv_pairs(
            r#"is_relaxed Lattice="1 0 0 0 2 0 0 0 3" frozen pbc=[T, T, F] name="a \"b\"" key = v"#,
        );
        let get = |k: &str| pairs.iter().find(|p| p.0 == k).map(|p| p.1.as_str());
        assert_eq!(get("Lattice"), Some("1 0 0 0 2 0 0 0 3"));
        assert_eq!(
            get("pbc").map(|s| s.split_whitespace().collect::<Vec<_>>()),
            Some(vec!["T", "T", "F"])
        );
        assert_eq!(get("name"), Some(r#"a "b""#));
        assert_eq!(get("key"), Some("v"));
        assert_eq!(get("is_relaxed"), None);
        assert_eq!(get("frozen"), None);
        // a plain-XYZ title is not mistaken for keys
        let meta = parse_comment(b"Lattice constant 3.6 from a relaxation").unwrap();
        assert!(meta.cell.is_none());
        // and the keys after a flag count
        let meta = parse_comment(br#"relaxed Lattice="1 0 0 0 2 0 0 0 3" pbc="T F T""#).unwrap();
        let c = meta.cell.expect("cell after a bare flag");
        assert_eq!(c.pbc, [true, false, true]);
    }

    /// An `id` column is the atom id (not a property), in any position.
    #[test]
    fn an_id_column_becomes_the_atom_ids() {
        let src =
            "3\nProperties=species:S:1:id:I:1:pos:R:3\nCu 17 0 0 0\nNi 5 1 0 0\nCu 99 2 0 0\n";
        let f = read_next_from_cursor(&Cursor::from_bytes(src.as_bytes(), "id.xyz"))
            .unwrap()
            .unwrap();
        assert_eq!(f.particles.ids, vec![17, 5, 99]);
        assert!(f.particles.props.get("id").is_none());
        assert_eq!(f.particles.positions[2], [2.0, 0.0, 0.0]);
        let bad = "1\nProperties=species:S:1:id:R:1:pos:R:3\nCu 1.5 0 0 0\n";
        let e = read_next_from_cursor(&Cursor::from_bytes(bad.as_bytes(), "id.xyz")).unwrap_err();
        assert!(e.to_string().contains("not an integer"), "{e}");
    }

    #[test]
    fn parse_then_rewrite_roundtrip() {
        let dir = std::env::temp_dir().join("rmda_test_xyz");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.xyz");
        {
            let content = br#"2
Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 5.0" pbc="T T T" Properties=species:S:1:pos:R:3
Cu 0.0 0.0 0.0
Cu 2.0 2.0 2.5
"#;
            std::fs::write(&p, content).unwrap();
        }
        let frames = XyzReader::read_all(&p).unwrap();
        assert_eq!(frames.len(), 1);
        let f = &frames[0];
        assert_eq!(f.natoms(), 2);
        assert_eq!(f.particles.positions[1][2], 2.5);
        assert_eq!(f.cell.h_matrix().row(0)[0], 4.0);
        assert_eq!(f.cell.pbc, [true, true, true]);
        assert_eq!(f.type_labels[0], "Cu");
    }

    /// A multi-component property is stored atom-major and must reach the file
    /// with each atom's components consecutive. Because both sides use that
    /// layout the writer is a straight copy, which means a transposition bug
    /// here would be invisible in the column count and would only show up as
    /// scrambled values -- and a unit-norm check cannot catch it, since the sum
    /// of squares is order-invariant. So pin it with distinct numbers.
    #[test]
    fn multi_component_property_stays_with_its_atom() {
        let frames = XyzReader::read_all(&{
            let p = std::env::temp_dir().join("rmda_test_xyz_interleave.xyz");
            std::fs::write(
                &p,
                b"3\nLattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\nCu 0.0 0.0 0.0\nCu 1.0 0.0 0.0\nCu 2.0 0.0 0.0\n",
            )
            .unwrap();
            p
        })
        .unwrap();

        // Atom i carries quaternion (w=i+1, x=10+i, y=20+i, z=30+i), stored
        // atom-major exactly as the PTM engine does.
        let mut store = PropertyStore::new();
        let flat: Vec<f64> = (0..3)
            .flat_map(|i| {
                [
                    (i + 1) as f64,
                    (10 + i) as f64,
                    (20 + i) as f64,
                    (30 + i) as f64,
                ]
            })
            .collect();
        store.register_flat(DType::F64, "StructureOrientation", 4, flat);

        let mut buf: Vec<u8> = Vec::new();
        XyzWriter::write_frame_with(&mut buf, &frames[0], &store).unwrap();
        let text = String::from_utf8(buf).unwrap();

        assert!(
            text.contains("Properties=species:S:1:pos:R:3:type:I:1:StructureOrientation:R:4"),
            "header missing the property column: {text}"
        );
        let rows: Vec<&str> = text.lines().filter(|l| l.starts_with("Cu")).collect();
        assert_eq!(rows.len(), 3);
        for (i, row) in rows.iter().enumerate() {
            // species + pos(3) + type, so the property block starts at field 5.
            let got: Vec<f64> = row
                .split_whitespace()
                .skip(5)
                .map(|v| v.parse::<f64>().unwrap())
                .collect();
            assert_eq!(
                got,
                vec![
                    (i + 1) as f64,
                    (10 + i) as f64,
                    (20 + i) as f64,
                    (30 + i) as f64
                ],
                "atom {i} quaternion components were not kept together"
            );
        }
    }

    /// Integers must be written without a decimal point, or a reader that honours
    /// the `I` kind will reject or coerce them.
    #[test]
    fn integer_property_is_written_as_integer() {
        let dir = std::env::temp_dir().join("rmda_test_xyz_int");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.xyz");
        std::fs::write(
            &p,
            b"2\nLattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\nCu 0.0 0.0 0.0\nCu 1.0 0.0 0.0\n",
        )
        .unwrap();
        let frames = XyzReader::read_all(&p).unwrap();

        let mut store = PropertyStore::new();
        store.register_scalar(DType::U8, "StructureType", vec![1u8, 0u8]);

        let mut buf: Vec<u8> = Vec::new();
        XyzWriter::write_frame_with(&mut buf, &frames[0], &store).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains(":StructureType:I:1"), "{text}");
        let rows: Vec<&str> = text.lines().filter(|l| l.starts_with("Cu")).collect();
        assert!(
            rows[0].ends_with(" 1"),
            "StructureType not integral: {}",
            rows[0]
        );
        assert!(
            rows[1].ends_with(" 0"),
            "StructureType not integral: {}",
            rows[1]
        );
    }

    /// A box whose origin is not at zero must survive a write/read round trip.
    ///
    /// Dropping the origin slides the written cell relative to the input, so
    /// overlaying output on the original frame lines the atoms up but not the
    /// cell borders. LAMMPS `data` boxes are routinely offset like this.
    #[test]
    fn a_nonzero_cell_origin_survives_a_round_trip() {
        let dir = std::env::temp_dir().join("rmda_test_xyz_origin");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.xyz");

        let mut cell = UnitCell::orthorhombic(4.0, 5.0, 6.0);
        cell.origin = [-0.25, 1.5, -3.0];
        let mut parts = ParticleData::with_capacity(1);
        parts.add_atom([1.0, 2.0, 3.0], 1, 1);
        let frame = Frame::new(cell, parts);

        let mut buf: Vec<u8> = Vec::new();
        XyzWriter::write_frame(&mut buf, &frame).unwrap();
        std::fs::write(&p, &buf).unwrap();

        let back = XyzReader::read_all(&p).unwrap();
        assert_eq!(back[0].cell.origin, [-0.25, 1.5, -3.0]);
        assert_eq!(back[0].particles.positions[0], [1.0, 2.0, 3.0]);
    }

    /// Atom types must survive even when no element labels distinguish them.
    ///
    /// A multi-component input read without labels gives every atom the species
    /// `X`, so the symbol alone cannot say which component an atom is. The
    /// explicit `type` column is what carries that identity, and the reader has
    /// to prefer it over inferring types from the species symbol.
    #[test]
    fn atom_types_survive_when_every_species_is_the_same_label() {
        let dir = std::env::temp_dir().join("rmda_test_xyz_types");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.xyz");

        let mut parts = ParticleData::with_capacity(3);
        parts.add_atom([0.0, 0.0, 0.0], 1, 1);
        parts.add_atom([1.0, 0.0, 0.0], 2, 3);
        parts.add_atom([2.0, 0.0, 0.0], 3, 2);
        let frame = Frame::new(UnitCell::orthorhombic(4.0, 4.0, 4.0), parts);

        let mut buf: Vec<u8> = Vec::new();
        XyzWriter::write_frame(&mut buf, &frame).unwrap();
        std::fs::write(&p, &buf).unwrap();

        let back = XyzReader::read_all(&p).unwrap();
        assert_eq!(back[0].particles.types, vec![1, 3, 2]);
    }

    /// A file may put `type` before `pos`, so the reader has to use the declared
    /// column order rather than assume positions come first.
    #[test]
    fn a_reordered_type_column_is_read_by_name() {
        let dir = std::env::temp_dir().join("rmda_test_xyz_reorder");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.xyz");
        std::fs::write(
            &p,
            b"3\nLattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\" pbc=\"T T T\" Properties=type:I:1:pos:R:3:species:S:1\n2 0.0 0.0 0.0 X\n1 1.0 0.0 0.0 X\n3 2.0 0.0 0.0 X\n",
        )
        .unwrap();

        let frames = XyzReader::read_all(&p).unwrap();
        assert_eq!(frames[0].particles.types, vec![2, 1, 3]);
        assert_eq!(frames[0].particles.positions[1], [1.0, 0.0, 0.0]);
    }

    fn read_str(name: &str, text: &str) -> Frame {
        read_next_from_cursor(&Cursor::from_bytes(text.as_bytes(), name))
            .unwrap()
            .unwrap()
    }

    /// With a `type` column and no `species=` key, each type is labelled with
    /// its own atoms' symbol, not with the alphabetically sorted symbols
    /// (which would name type 1 `Cu` here). OVITO 3.15 reads this file as Ni = 1,
    /// Cu = 2.
    #[test]
    fn a_type_column_without_a_species_key_labels_each_type_with_its_symbol() {
        let f = read_str(
            "t.xyz",
            "3\nLattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3:type:I:1\n\
             Ni 0 0 0 1\nCu 1 0 0 2\nNi 2 0 0 1\n",
        );
        assert_eq!(f.particles.types, vec![1, 2, 1]);
        assert_eq!(f.type_labels, vec!["Ni".to_string(), "Cu".to_string()]);
        // Types out of first-seen order, and an absent type 2.
        let f = read_str(
            "t.xyz",
            "3\nProperties=species:S:1:pos:R:3:type:I:1\nW 0 0 0 3\nFe 1 0 0 1\nW 2 0 0 3\n",
        );
        assert_eq!(
            f.type_labels,
            vec!["Fe".to_string(), String::new(), "W".to_string()]
        );
    }

    /// A symbol shared by several types, or a type whose atoms carry different
    /// symbols, names no type: RMDA's own `X` placeholder for unlabelled types
    /// reads back as no labels, so an unlabelled frame stays unlabelled.
    #[test]
    fn ambiguous_type_symbols_give_no_label() {
        let f = read_str(
            "t.xyz",
            "3\nProperties=species:S:1:pos:R:3:type:I:1\nX 0 0 0 1\nX 1 0 0 3\nX 2 0 0 2\n",
        );
        assert_eq!(f.particles.types, vec![1, 3, 2]);
        assert!(f.type_labels.is_empty());
        let f = read_str(
            "t.xyz",
            "5\nProperties=species:S:1:pos:R:3:type:I:1\n\
             Ni 0 0 0 1\nFe 1 0 0 1\nCu 2 0 0 2\nW 3 0 0 3\nW 3 1 0 4\n",
        );
        assert_eq!(
            f.type_labels,
            vec![
                String::new(),
                "Cu".to_string(),
                String::new(),
                String::new()
            ]
        );
        // A partly labelled frame through the plain writer: the one type
        // written as `X` keeps it, as a name it uniquely owns.
        let mut parts = ParticleData::with_capacity(3);
        parts.add_atom([0.0, 0.0, 0.0], 1, 2);
        parts.add_atom([1.0, 0.0, 0.0], 2, 1);
        parts.add_atom([2.0, 0.0, 0.0], 3, 3);
        let mut frame = Frame::new(UnitCell::orthorhombic(4.0, 4.0, 4.0), parts);
        frame.type_labels = vec!["W".into(), "Mo".into()];
        let mut buf = Vec::new();
        XyzWriter::write_frame(&mut buf, &frame).unwrap();
        let back = read_str("w.xyz", std::str::from_utf8(&buf).unwrap());
        assert_eq!(back.particles.types, vec![2, 1, 3]);
        assert_eq!(
            back.type_labels,
            vec!["W".to_string(), "Mo".to_string(), "X".to_string()]
        );
    }

    /// A `species=` key still wins over the rows' symbols.
    #[test]
    fn a_species_key_wins_over_the_type_symbols() {
        let f = read_str(
            "t.xyz",
            "2\nProperties=species:S:1:pos:R:3:type:I:1 species=\"Cu Ni\"\nNi 0 0 0 1\nCu 1 0 0 2\n",
        );
        assert_eq!(f.type_labels, vec!["Cu".to_string(), "Ni".to_string()]);
    }

    /// `Timestep=` and `Time=` are read into the frame, so a written frame
    /// keeps both; a file without them reads as timestep 0 and no time.
    #[test]
    fn timestep_and_time_are_read_back() {
        let f = read_str(
            "t.xyz",
            "1\nLattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3 Timestep=42 Time=1.5\nCu 0 0 0\n",
        );
        assert_eq!((f.meta.timestep, f.meta.time), (42, Some(1.5)));
        let f = read_str("t.xyz", "1\nTimestep=1000.0 Time=x\nCu 0 0 0\n");
        assert_eq!((f.meta.timestep, f.meta.time), (1000, None));
        let f = read_str("t.xyz", "1\n\nCu 0 0 0\n");
        assert_eq!((f.meta.timestep, f.meta.time), (0, None));

        let mut parts = ParticleData::with_capacity(1);
        parts.add_atom([0.5, 0.0, 0.0], 1, 1);
        let mut frame = Frame::new(UnitCell::orthorhombic(4.0, 4.0, 4.0), parts);
        frame.meta.timestep = 987_654_321_012;
        let mut buf = Vec::new();
        XyzWriter::write_frame(&mut buf, &frame).unwrap();
        assert!(!String::from_utf8_lossy(&buf).contains("Time="));
        let back = read_str("w.xyz", std::str::from_utf8(&buf).unwrap());
        assert_eq!(
            (back.meta.timestep, back.meta.time),
            (987_654_321_012, None)
        );
        frame.meta.time = Some(0.1 + 0.2);
        let mut buf = Vec::new();
        XyzWriter::write_frame(&mut buf, &frame).unwrap();
        let back = read_str("w.xyz", std::str::from_utf8(&buf).unwrap());
        assert_eq!(back.meta.time, Some(0.1 + 0.2));
    }

    #[test]
    fn a_selection_column_becomes_a_mask_and_its_absence_is_not_an_all_zero_one() {
        // `selection` is `None` when the file has no `Selection` column, which is
        // what lets an analysis distinguish "no selection was given" from "every
        // atom was deselected". An all-zero mask would make those two cases
        // identical and turn `--only-selected` into a silent no-op.
        let dir = std::env::temp_dir().join("rmda_test_xyz_selection");
        std::fs::create_dir_all(&dir).unwrap();

        let plain = dir.join("plain.xyz");
        std::fs::write(
            &plain,
            b"2\nLattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:type:I:1\n1 0.0 0.0 0.0 1\n2 1.0 0.0 0.0 2\n",
        )
        .unwrap();
        let frames = XyzReader::read_all(&plain).unwrap();
        assert_eq!(frames[0].particles.selection, None);

        // Written by `rmda wc --only-selected -o out.xyz`, then read back.
        let masked = dir.join("masked.xyz");
        std::fs::write(
            &masked,
            b"3\nLattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:type:I:1:Selection:I:1\n1 0.0 0.0 0.0 1 1\n2 1.0 0.0 0.0 2 0\n1 2.0 0.0 0.0 1 1\n",
        )
        .unwrap();
        let frames = XyzReader::read_all(&masked).unwrap();
        assert_eq!(frames[0].particles.selection, Some(vec![1, 0, 1]));

        // Lowercase, because OVITO's export uses whichever case the property was
        // given and readers that demand one spelling fail on the other.
        let lower = dir.join("lower.xyz");
        std::fs::write(
            &lower,
            b"2\nLattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:type:I:1:selection:I:1\n1 0.0 0.0 0.0 1 1\n2 1.0 0.0 0.0 2 1\n",
        )
        .unwrap();
        let frames = XyzReader::read_all(&lower).unwrap();
        assert_eq!(frames[0].particles.selection, Some(vec![1, 1]));
    }

    #[test]
    fn a_selection_column_that_is_not_zero_or_one_is_refused() {
        // A float weight, or a per-atom property someone has renamed to
        // `Selection`, is a different quantity wearing the same name. Coercing it
        // would either select every atom (`0.5` is truthy) or none.
        let dir = std::env::temp_dir().join("rmda_test_xyz_badselection");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, value) in [("frac.xyz", "0.5"), ("big.xyz", "2"), ("word.xyz", "yes")] {
            let p = dir.join(name);
            let content = format!(
                "1\nLattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\" pbc=\"T T T\" \
                 Properties=species:S:1:pos:R:3:type:I:1:Selection:I:1\n1 0.0 0.0 0.0 1 {value}\n"
            );
            std::fs::write(&p, content).unwrap();
            let err = XyzReader::read_all(&p).unwrap_err().to_string();
            assert!(
                err.contains("Selection"),
                "{name} should be refused by name, said {err}"
            );
        }
    }

    /// `XyzOrigin::Zero` must move the box, not distort it.
    ///
    /// The atom's position *relative to its own cell* has to be identical in both
    /// modes; only the box's placement changes. Zeroing the lattice origin
    /// without translating the coordinates would keep every atom that far from
    /// where its cell says it is, which is the same overlay bug in the opposite
    /// direction.
    #[test]
    fn the_zero_origin_mode_translates_atoms_with_the_box() {
        let dir = std::env::temp_dir().join("rmda_test_xyz_zeroorigin");
        std::fs::create_dir_all(&dir).unwrap();

        let mut cell = UnitCell::orthorhombic(4.0, 5.0, 6.0);
        cell.origin = [-0.25, 1.5, -3.0];
        let mut parts = ParticleData::with_capacity(1);
        parts.add_atom([1.0, 2.0, 3.0], 1, 1);
        let frame = Frame::new(cell, parts);

        let mut native = Vec::new();
        XyzWriter::write_frame_with_origin(
            &mut native,
            &frame,
            &PropertyStore::new(),
            XyzOrigin::Native,
        )
        .unwrap();
        let mut zero = Vec::new();
        XyzWriter::write_frame_with_origin(
            &mut zero,
            &frame,
            &PropertyStore::new(),
            XyzOrigin::Zero,
        )
        .unwrap();

        let native_path = dir.join("native.xyz");
        let zero_path = dir.join("zero.xyz");
        std::fs::write(&native_path, &native).unwrap();
        std::fs::write(&zero_path, &zero).unwrap();

        let n = XyzReader::read_all(&native_path).unwrap().remove(0);
        let z = XyzReader::read_all(&zero_path).unwrap().remove(0);

        assert_eq!(n.cell.origin, [-0.25, 1.5, -3.0]);
        assert_eq!(z.cell.origin, [0.0, 0.0, 0.0]);

        // Same cell vectors, and the atom sits at the same offset inside it.
        assert_eq!(n.cell.h_matrix(), z.cell.h_matrix());
        let dn = [
            n.particles.positions[0][0] - n.cell.origin[0],
            n.particles.positions[0][1] - n.cell.origin[1],
            n.particles.positions[0][2] - n.cell.origin[2],
        ];
        let dz = [
            z.particles.positions[0][0] - z.cell.origin[0],
            z.particles.positions[0][1] - z.cell.origin[1],
            z.particles.positions[0][2] - z.cell.origin[2],
        ];
        assert!(
            (0..3).all(|d| (dn[d] - dz[d]).abs() < 1e-9),
            "atom moved inside the box: {dn:?} vs {dz:?}"
        );
        // And the coordinates actually shifted by the origin.
        assert!((z.particles.positions[0][0] - (n.particles.positions[0][0] + 0.25)).abs() < 1e-9);
    }

    /// The nine-number form with no `Origin=` key means origin zero.
    #[test]
    fn a_nine_number_lattice_still_reads() {
        let dir = std::env::temp_dir().join("rmda_test_xyz_nine");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.xyz");
        std::fs::write(
            &p,
            b"1\nLattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\nCu 0.0 0.0 0.0\n",
        )
        .unwrap();

        let frames = XyzReader::read_all(&p).unwrap();
        assert_eq!(frames[0].cell.origin, [0.0, 0.0, 0.0]);
    }
}
