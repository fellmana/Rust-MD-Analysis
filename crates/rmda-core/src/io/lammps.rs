//! LAMMPS dump file reader (design §8.3) and data-file reader (§8.4).
//!
//! Handles both restricted-triclinic `BOX BOUNDS xy xz yz ...` and plain
//! `BOX BOUNDS pp pp pp` headers, and arbitrary `ITEM: ATOMS` column sets.
//! Positions are stored **wrapped** (design §8.3): `x,y,z` are kept as read;
//! `xu` only frames are wrapped with `UnitCell::wrap`. Optional `ix iy iz`
//! image columns are retained for unwrap-on-demand.

use crate::cell::{pack_image, Coord3, MdaImage, UnitCell};
use crate::data::frame::Frame;
use crate::data::particle::{MdaTag, MdaType, ParticleData};
use crate::data::property::{DType, Property, TypedVec};
use crate::error::{Error, Result};
use crate::io::engine::{
    count_newlines, fast_f64, fast_i64, next_f64, next_i64, next_token, parse_rows, row_chunks,
    split_by_chunks, Cursor, Fields, Lines, RowChunk, RowError, RowRule,
};
use crate::io::writer::{select_props, FloatFmt, PropCol, PropSelect, WriteOptions};
use crate::io::ReadOptions;
use std::io::Write;
use std::path::Path;

pub struct DumpReader;

impl DumpReader {
    pub fn read_all(path: &Path) -> Result<Vec<Frame>> {
        let cur = Cursor::from_file(path)?;
        let mut frames = Vec::new();
        while !cur.eof() {
            let Some(line) = cur.peek_line() else { break };
            if line.is_empty() {
                cur.next_line();
                continue;
            }
            if line.starts_with(b"ITEM:") {
                frames.push(parse_frame(&cur)?);
            } else {
                // stray content outside a frame: LAMMPS files have a single
                // trailing newline; anything else is an error.
                return Err(parse_error(&cur, "unexpected content outside frame"));
            }
        }
        Ok(frames)
    }

    pub fn read_first(path: &Path) -> Result<Frame> {
        let frames = Self::read_all(path)?;
        frames
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidData("empty dump file".into()))
    }
}

/// Read a LAMMPS data file (single frame).
pub fn read_data_file(path: &Path) -> Result<Vec<Frame>> {
    let cur = Cursor::from_file(path)?;
    let frame = parse_data_frame(&cur)?;
    Ok(vec![frame])
}

/// Parse a LAMMPS data frame from a cursor positioned at the start of the file.
pub(crate) fn parse_data_frame(cur: &Cursor) -> Result<Frame> {
    parse_data_frame_with(cur, &ReadOptions::all())
}

/// [`parse_data_frame`] keeping only the columns `opts` allows.
pub(crate) fn parse_data_frame_with(cur: &Cursor, opts: &ReadOptions) -> Result<Frame> {
    let mut natoms: Option<i64> = None;
    let mut ntypes: Option<usize> = None;
    // restricted form: (lo, hi, tilt) per row
    let mut bx: [[f64; 3]; 3] = [[0.0; 3]; 3];
    // general triclinic form: `avec`/`bvec`/`cvec` + `abc origin`
    let mut avec: Option<Coord3> = None;
    let mut bvec: Option<Coord3> = None;
    let mut cvec: Option<Coord3> = None;
    let mut origin: Coord3 = [0.0, 0.0, 0.0];

    macro_rules! perr {
        ($msg:expr) => {
            Error::at_line(&cur.path, cur.line_no(), $msg)
        };
    }

    // --- header section
    //
    // Stop *before* the first section keyword rather than after it: the section
    // loop below has to see that keyword itself to dispatch on it. A data file
    // whose first section is `Atoms` (i.e. no `Masses` block, which is how ASE
    // and several other writers emit them) would otherwise be handed straight to
    // the section loop at its first atom row, never match the `Atoms` branch,
    // and silently produce a frame with the header's atom count but no
    // positions.
    loop {
        let Some(peeked) = cur.peek_line() else {
            return Err(perr!("data file ended in header"));
        };
        if section_of(peeked).is_some() {
            break; // sections begin; leave the keyword for the loop below
        }
        let line = cur.next_line().expect("peeked line disappeared");
        // Header lines may carry a trailing `# comment` (`5759994 atoms # film`).
        let text = String::from_utf8_lossy(line);
        let t = strip_comment(&text).trim();
        if t.is_empty() {
            continue;
        }
        let lower = t.to_ascii_lowercase();
        if lower.ends_with("atoms") && natoms.is_none() {
            natoms = t.split_whitespace().next().and_then(|s| s.parse().ok());
            continue;
        }
        if lower.ends_with("atom types") && ntypes.is_none() {
            ntypes = t.split_whitespace().next().and_then(|s| s.parse().ok());
            continue;
        }
        if lower.ends_with("avec") {
            let v: Vec<f64> = t
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() >= 3 {
                avec = Some([v[0], v[1], v[2]]);
            }
            continue;
        }
        if lower.ends_with("bvec") {
            let v: Vec<f64> = t
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() >= 3 {
                bvec = Some([v[0], v[1], v[2]]);
            }
            continue;
        }
        if lower.ends_with("cvec") {
            let v: Vec<f64> = t
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() >= 3 {
                cvec = Some([v[0], v[1], v[2]]);
            }
            continue;
        }
        if lower.contains("abc origin") {
            let v: Vec<f64> = t
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() >= 3 {
                origin = [v[0], v[1], v[2]];
            }
            continue;
        }
        // Restricted-triclinic tilt line: `<xy> <xz> <yz> xy xz yz`, exactly how
        // LAMMPS `write_data` emits it and `read_data` matches it. Skipping this
        // line would silently drop the shear of every triclinic data file on read.
        {
            let words: Vec<&str> = lower.split_whitespace().collect();
            if words.len() >= 4 && words[words.len() - 3..] == ["xy", "xz", "yz"] {
                let v: Vec<f64> = words[..3]
                    .iter()
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                if v.len() >= 3 {
                    bx[0][2] = v[0];
                    bx[1][2] = v[1];
                    bx[2][2] = v[2];
                }
                continue;
            }
        }
        if lower.contains("xlo") || lower.contains("xhi") {
            let v: Vec<f64> = t
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() >= 2 {
                bx[0][0] = v[0];
                bx[0][1] = v[1];
                if let Some(&xy) = v.get(2) {
                    bx[0][2] = xy;
                }
            }
            continue;
        }
        if lower.contains("ylo") || lower.contains("yhi") {
            let v: Vec<f64> = t
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() >= 2 {
                bx[1][0] = v[0];
                bx[1][1] = v[1];
                if let Some(&xz) = v.get(2) {
                    bx[1][2] = xz;
                }
            }
            continue;
        }
        if lower.contains("zlo") || lower.contains("zhi") {
            let v: Vec<f64> = t
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() >= 2 {
                bx[2][0] = v[0];
                bx[2][1] = v[1];
                if let Some(&yz) = v.get(2) {
                    bx[2][2] = yz;
                }
            }
            continue;
        }
        // tolerate unknown header keywords (`N bonds`, `extra bond per atom`, ...)
    }

    let natoms_i64 = natoms.ok_or_else(|| perr!("missing atoms count"))?;
    if natoms_i64 < 0 {
        return Err(parse_error(cur, "negative atom count"));
    }
    let natoms = natoms_i64 as usize;
    // Whether the `N atom types` header was present; only then can a row's type
    // be range-checked against it (a file that omits the header is common, and
    // defaulting it to 1 would reject every valid row above type 1).
    let ntypes_declared = ntypes;
    let ntypes_val = ntypes_declared.unwrap_or(1);
    if ntypes_val == 0 {
        return Err(parse_error(cur, "invalid number of atom types"));
    }
    let ntypes = ntypes_val;

    // General (avec/bvec/cvec) and restricted (xlo xhi / xy xz yz) triclinic
    // boxes are both accepted; general is rotated into the restricted frame
    // exactly as `Domain::define_general_triclinic` does.
    let mut g2r: Option<crate::cell::Mat3> = None;
    let cell = match (avec, bvec, cvec) {
        (Some(a), Some(b), Some(c)) if opts.keep_general => {
            UnitCell::general_as_given(a, b, c, origin)?
        }
        (Some(a), Some(b), Some(c)) => {
            let (cell, rot) = UnitCell::from_general_triclinic_rot(a, b, c, origin)?;
            g2r = Some(rot);
            cell
        }
        _ => {
            // detectable shear: any tilt non-zero tells us triclinic conventions
            let triclinic = bx[0][2] != 0.0 || bx[1][2] != 0.0 || bx[2][2] != 0.0;
            if triclinic {
                UnitCell::from_box_bounds(
                    bx[0][0], bx[0][1], bx[0][2], bx[1][0], bx[1][1], bx[1][2], bx[2][0], bx[2][1],
                    bx[2][2],
                )
            } else {
                let mut c = UnitCell::orthorhombic(
                    bx[0][1] - bx[0][0],
                    bx[1][1] - bx[1][0],
                    bx[2][1] - bx[2][0],
                );
                c.origin = [bx[0][0], bx[1][0], bx[2][0]];
                c
            }
        }
    };

    let mut parts: Option<ParticleData> = None;
    let mut masses: Vec<f64> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    // `Velocities` rows, matched to the atoms by id once every section is read.
    let mut velocities: Option<(Vec<MdaTag>, VelColumns)> = None;

    // LAMMPS separates sections with blank lines, and writers (OVITO) also
    // leave blank lines *inside* them, so rows are read with blanks/comments
    // skipped rather than assuming a strict layout.
    fn next_row(cur: &Cursor) -> Option<Vec<u8>> {
        while let Some(l) = cur.next_line() {
            let t = String::from_utf8_lossy(l);
            let t = t.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            return Some(l.to_vec());
        }
        None
    }
    // A type field: an integer, or a label from `Atom Type Labels`.
    let type_of = |tok: &str, labels: &[String]| -> Option<i64> {
        tok.parse::<i64>().ok().or_else(|| {
            labels
                .iter()
                .position(|l| !l.is_empty() && l == tok)
                .map(|i| i as i64 + 1)
        })
    };

    while let Some(line) = cur.next_line() {
        let Some(section) = section_of(line) else {
            // Rows of a section that is not read (Bonds, Pair Coeffs, ...),
            // blank lines and comments.
            continue;
        };
        match section {
            "Atom Type Labels" => {
                labels = vec![String::new(); ntypes];
                for _ in 0..ntypes {
                    let row =
                        next_row(cur).ok_or_else(|| perr!("truncated Atom Type Labels section"))?;
                    let text = String::from_utf8_lossy(&row);
                    let mut w = strip_comment(&text).split_whitespace();
                    let (Some(Ok(t)), Some(label)) = (w.next().map(str::parse::<usize>), w.next())
                    else {
                        return Err(perr!(format!("bad Atom Type Labels row '{}'", text.trim())));
                    };
                    if t == 0 || t > ntypes {
                        return Err(perr!(format!("type label for type {t} of {ntypes}")));
                    }
                    labels[t - 1] = label.to_string();
                }
            }
            "Masses" => {
                // `type mass [# comment]`, one row per type.
                masses = vec![f64::NAN; ntypes];
                for _ in 0..ntypes {
                    let row = next_row(cur).ok_or_else(|| perr!("truncated Masses section"))?;
                    let text = String::from_utf8_lossy(&row);
                    let mut w = text.split_whitespace();
                    let (Some(t), Some(m)) = (w.next(), w.next()) else {
                        return Err(perr!("Masses row needs `type mass`"));
                    };
                    let (Some(t), Ok(m)) = (type_of(t, &labels), m.parse::<f64>()) else {
                        return Err(perr!(&format!("bad Masses row '{}'", text.trim())));
                    };
                    if t < 1 || t as usize > ntypes {
                        return Err(perr!(&format!("Masses row for type {t} of {ntypes}")));
                    }
                    masses[t as usize - 1] = m;
                }
            }
            "Atoms" => {
                let text = String::from_utf8_lossy(line);
                // `Atoms # style`: the style is the first word of the comment.
                let comment = text
                    .split_once('#')
                    .and_then(|(_, c)| c.split_whitespace().next());
                let style = match (comment, opts.atom_style.as_deref()) {
                    (Some(c), Some(s)) if c.split('/').next() != s.split('/').next() => {
                        return Err(Error::at_line(
                            &cur.path,
                            cur.line_no(),
                            format!("the Atoms section says `# {c}` but --atom-style is {s}"),
                        ))
                    }
                    (Some(c), _) => Some(c),
                    (None, s) => s,
                };
                let first = Lines::new(cur.rest())
                    .find(|l| RowRule::Content.is_row(l))
                    .map(data_fields)
                    .unwrap_or(0);
                let (cols, images) = atom_layout(style, first)
                    .map_err(|m| Error::at_line(&cur.path, cur.line_no() + 1, m))?;
                let p = parse_atoms(
                    cur,
                    natoms,
                    cols,
                    images,
                    &labels,
                    ntypes_declared,
                    &cell,
                    g2r.as_ref(),
                    opts,
                )?;
                parts = Some(p);
            }
            "Velocities" => {
                if ["vx", "vy", "vz"].iter().all(|n| opts.skips(n)) {
                    // Stepped over: counted, never parsed or stored.
                    let (len, nlines) =
                        crate::io::engine::rows_extent(cur.rest(), natoms, RowRule::Content)
                            .ok_or_else(|| perr!("truncated Velocities section"))?;
                    cur.advance(len, nlines);
                } else {
                    velocities = Some(parse_velocities(cur, natoms)?);
                }
            }
            _ => {}
        }
    }

    let mut parts = parts.unwrap_or_else(|| ParticleData::new(natoms));
    if let Some((vids, vel)) = velocities {
        attach_velocities(&mut parts, vids, vel, opts).map_err(|m| perr!(m))?;
    }
    if let Some(rot) = &g2r {
        rotate_vector_props(&mut parts.props, rot);
    }
    let mut frame = Frame::new(cell, parts);
    frame.type_masses = masses;
    if labels.iter().any(|l| !l.is_empty()) {
        frame.type_labels = labels;
    }
    Ok(frame)
}

/// Per-atom vectors LAMMPS stores in the general frame of a general-triclinic
/// file and rotates with the atoms (`read_data` velocities, `dump_modify
/// triclinic/general yes` columns): velocities, forces, dipoles, angular
/// velocities and momenta.
const VECTOR_COLUMNS: &[[&str; 3]] = &[
    ["vx", "vy", "vz"],
    ["fx", "fy", "fz"],
    ["mux", "muy", "muz"],
    ["omegax", "omegay", "omegaz"],
    ["angmomx", "angmomy", "angmomz"],
];

/// Rotate the [`VECTOR_COLUMNS`] triples of a general-triclinic input into the
/// restricted frame with the rotation that took the cell and the positions
/// there, so that velocities stay attached to the atoms they belong to. A
/// triple counts only when all three are one-component f64 properties.
pub(crate) fn rotate_vector_props(
    props: &mut crate::data::property::PropertyStore,
    rot: &crate::cell::Mat3,
) {
    for names in VECTOR_COLUMNS {
        let cols: Option<Vec<Vec<f64>>> = names
            .iter()
            .map(|n| {
                props
                    .get(n)
                    .filter(|p| p.components == 1)
                    .and_then(|p| p.as_f64())
                    .cloned()
            })
            .collect();
        let Some(cols) = cols else { continue };
        let n = cols[0].len();
        if cols.iter().any(|c| c.len() != n) {
            continue;
        }
        let mut out = cols.clone();
        for i in 0..n {
            let r = rot.mul_v([cols[0][i], cols[1][i], cols[2][i]]);
            for d in 0..3 {
                out[d][i] = r[d];
            }
        }
        for (name, v) in names.iter().zip(out) {
            if let Some(dst) = props.get_mut(name).and_then(|p| p.as_mut_f64()) {
                *dst = v;
            }
        }
    }
}

/// Data-file section headers (LAMMPS `read_data`), matched against the whole
/// line once its `# comment` is removed.
const DATA_SECTIONS: &[&str] = &[
    "Atoms",
    "Velocities",
    "Masses",
    "Ellipsoids",
    "Lines",
    "Triangles",
    "Bodies",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Pair Coeffs",
    "PairIJ Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "BondBond Coeffs",
    "BondAngle Coeffs",
    "MiddleBondTorsion Coeffs",
    "EndBondTorsion Coeffs",
    "AngleTorsion Coeffs",
    "AngleAngleTorsion Coeffs",
    "BondBond13 Coeffs",
    "AngleAngle Coeffs",
    "Atom Type Labels",
    "Bond Type Labels",
    "Angle Type Labels",
    "Dihedral Type Labels",
    "Improper Type Labels",
];

/// The text before a `#`.
fn strip_comment(t: &str) -> &str {
    t.split('#').next().unwrap_or("")
}

/// The section a line opens, if it is a section header.
fn section_of(line: &[u8]) -> Option<&'static str> {
    // Section headers are short; anything long is a row.
    if line.len() > 128 {
        return None;
    }
    let text = String::from_utf8_lossy(line);
    let t = strip_comment(&text).trim();
    DATA_SECTIONS
        .iter()
        .copied()
        .find(|s| s.eq_ignore_ascii_case(t))
}

/// A column of an `Atoms` row.
#[derive(Clone, Copy, PartialEq, Debug)]
enum DCol {
    Id,
    Mol,
    Type,
    Q,
    X,
    Y,
    Z,
    Diameter,
    Density,
}

/// Column layouts of the supported atom styles (LAMMPS `atom_vec_*.cpp`
/// `fields_data_atom`).
fn style_columns(base: &str) -> Option<&'static [DCol]> {
    use DCol::*;
    Some(match base {
        "atomic" => &[Id, Type, X, Y, Z],
        "charge" => &[Id, Type, Q, X, Y, Z],
        "full" => &[Id, Mol, Type, Q, X, Y, Z],
        "molecular" | "bond" | "angle" => &[Id, Mol, Type, X, Y, Z],
        "sphere" => &[Id, Type, Diameter, Density, X, Y, Z],
        _ => return None,
    })
}

/// Fields of a data row, up to a `#` comment.
fn data_fields(line: &[u8]) -> usize {
    let mut at = 0;
    let mut n = 0;
    while let Some(t) = next_token(line, &mut at) {
        if t[0] == b'#' {
            break;
        }
        n += 1;
    }
    n
}

/// The `Atoms` column layout and whether rows end in `ix iy iz`, from the
/// style comment, or — when the line has none — from the first row's column
/// count, as ASE and atomsk infer it (design §8.4).
fn atom_layout(
    style: Option<&str>,
    ncols: usize,
) -> std::result::Result<(&'static [DCol], bool), String> {
    match style {
        Some(style) => {
            // LAMMPS package suffixes decorate the style name (`atomic/kk`,
            // `full/gpu`, `molecular/omp`); the column layout is that of the
            // base style, so strip the suffix before matching.
            let base = style.split('/').next().unwrap_or(style);
            // An *unknown* style is rejected rather than decoded as another:
            // reading `id mol type x y z` as `atomic` would take `mol` for the
            // type and `type` for x -- plausible, wrong, and silent.
            let cols = style_columns(base).ok_or_else(|| {
                format!(
                    "unsupported atom style '{style}' \
                     (supported: atomic, charge, full, molecular, bond, angle, sphere)"
                )
            })?;
            if ncols == 0 || ncols == cols.len() {
                Ok((cols, false))
            } else if ncols == cols.len() + 3 {
                Ok((cols, true))
            } else {
                Err(format!(
                    "atom row has {ncols} cols, expected {} for style {base} \
                     (or {} with image flags)",
                    cols.len(),
                    cols.len() + 3
                ))
            }
        }
        None => match ncols {
            0 | 5 => Ok((style_columns("atomic").unwrap(), false)),
            8 => Ok((style_columns("atomic").unwrap(), true)),
            7 => Ok((style_columns("full").unwrap(), false)),
            10 => Ok((style_columns("full").unwrap(), true)),
            6 | 9 => Err(format!(
                "cannot infer the atom style of {ncols}-column Atoms rows \
                 (charge or molecular); name it on the section line, e.g. `Atoms # charge`, \
                 or give --atom-style"
            )),
            n => Err(format!(
                "cannot infer the atom style of {n}-column Atoms rows; \
                 name it on the section line, e.g. `Atoms # atomic`, or give --atom-style"
            )),
        },
    }
}

/// Parse the `natoms` rows of an `Atoms` section in parallel.
#[allow(clippy::too_many_arguments)]
fn parse_atoms(
    cur: &Cursor,
    natoms: usize,
    cols: &[DCol],
    images: bool,
    labels: &[String],
    ntypes_declared: Option<usize>,
    cell: &UnitCell,
    g2r: Option<&crate::cell::Mat3>,
    opts: &ReadOptions,
) -> Result<ParticleData> {
    let body = cur.rest();
    let (len, nlines, chunks) = row_chunks(body, natoms, RowRule::Content).ok_or_else(|| {
        Error::at_line(
            &cur.path,
            cur.line_no() + count_newlines(body),
            "truncated Atoms section",
        )
    })?;
    let body = &body[..len];
    // Columns other than id/type/x/y/z become properties.
    let mut propf = Vec::new();
    let mut propi = Vec::new();
    let sinks: Vec<Sink> = cols
        .iter()
        .map(|c| match c {
            DCol::Id => Sink::Id,
            DCol::Type => Sink::Type,
            DCol::X => Sink::Pos(0),
            DCol::Y => Sink::Pos(1),
            DCol::Z => Sink::Pos(2),
            DCol::Mol if opts.skips("mol") => Sink::Skip,
            DCol::Q if opts.skips("q") => Sink::Skip,
            DCol::Diameter if opts.skips("diameter") => Sink::Skip,
            DCol::Density if opts.skips("density") => Sink::Skip,
            DCol::Mol => {
                propi.push("mol".to_string());
                Sink::PropI(propi.len() - 1)
            }
            DCol::Q | DCol::Diameter | DCol::Density => {
                propf.push(
                    match c {
                        DCol::Q => "q",
                        DCol::Diameter => "diameter",
                        _ => "density",
                    }
                    .to_string(),
                );
                Sink::PropF(propf.len() - 1)
            }
        })
        .collect();
    // Image flags in the rows are still stepped over when skipped.
    let skip_images = images && opts.skips_images();
    let plan = DumpPlan {
        sinks,
        pos: PosKind::Wrapped,
        images: images && !skip_images,
        propf,
        propi,
    };
    let mut out = plan.alloc(natoms);
    let mut parts_sinks = out.split(&chunks, plan.images);
    let ncol = plan.sinks.len() + if images { 3 } else { 0 };
    let short = |line: &[u8]| format!("atom row has {} cols, expected {ncol}", data_fields(line));
    // A failed field: missing, a `#` comment where a field was due, or junk.
    let fail = |e: Option<&[u8]>, line: &[u8], what: &str| match e {
        Some(tok) if tok[0] != b'#' => {
            format!("atom row: bad {what} '{}'", String::from_utf8_lossy(tok))
        }
        _ => short(line),
    };
    // `atomic` rows, `id type x y z [ix iy iz]`, the layout of nearly every
    // large data file, take a straight-line parse; a row it does not accept
    // (type labels, a bad field, ...) goes through the general loop, which
    // also words the error.
    let atomic = plan.sinks
        == [
            Sink::Id,
            Sink::Type,
            Sink::Pos(0),
            Sink::Pos(1),
            Sink::Pos(2),
        ];
    let plain_row = |s: &mut DumpSink, r: usize, line: &[u8]| -> Option<()> {
        let mut at = 0usize;
        let id = next_i64(line, &mut at).ok()?;
        let t = next_i64(line, &mut at).ok()?;
        if t < 1 || ntypes_declared.map_or(false, |decl| t as usize > decl) {
            return None;
        }
        let v = [
            next_f64(line, &mut at).ok()?,
            next_f64(line, &mut at).ok()?,
            next_f64(line, &mut at).ok()?,
        ];
        let mut img = [0i64; 3];
        if images {
            for slot in &mut img {
                *slot = next_i64(line, &mut at).ok()?;
            }
        }
        if next_token(line, &mut at).map_or(false, |t| t[0] != b'#') {
            return None;
        }
        s.ids[r] = id;
        s.types[r] = t as u32;
        if images && !skip_images {
            s.images[r] = pack_image(img[0], img[1], img[2]);
        }
        s.pos[r] = match g2r {
            Some(rot) => crate::cell::general_to_restricted_point(rot, cell.origin, v),
            None => v,
        };
        Some(())
    };
    let res = parse_rows(
        body,
        &chunks,
        &mut parts_sinks,
        RowRule::Content,
        |s, r, line| {
            if atomic && plain_row(s, r, line).is_some() {
                return Ok(());
            }
            let mut row = || -> std::result::Result<(), String> {
                let mut at = 0usize;
                let mut v = [0.0f64; 3];
                for sink in &plan.sinks {
                    match *sink {
                        Sink::Id => {
                            s.ids[r] = next_i64(line, &mut at).map_err(|e| fail(e, line, "id"))?
                        }
                        Sink::Type => {
                            let t = match next_i64(line, &mut at) {
                                Ok(t) => t,
                                Err(Some(tok)) if tok[0] != b'#' => std::str::from_utf8(tok)
                                    .ok()
                                    .and_then(|tok| {
                                        labels.iter().position(|l| !l.is_empty() && l == tok)
                                    })
                                    .map(|i| i as i64 + 1)
                                    .ok_or_else(|| fail(Some(tok), line, "type"))?,
                                Err(e) => return Err(fail(e, line, "type")),
                            };
                            if t < 1 {
                                return Err(format!(
                                    "atom row: type {t} is not a positive integer"
                                ));
                            }
                            // Range-check against the header's `atom types` when it
                            // was present; without it there is no bound.
                            if let Some(decl) = ntypes_declared {
                                if t as usize > decl {
                                    return Err(format!(
                                        "atom row: type {t} exceeds the declared {decl} atom types"
                                    ));
                                }
                            }
                            s.types[r] = t as u32;
                        }
                        Sink::Pos(d) => {
                            v[d] = next_f64(line, &mut at)
                                .map_err(|e| fail(e, line, ["x", "y", "z"][d]))?
                        }
                        Sink::PropF(k) => {
                            s.pf[k][r] = next_f64(line, &mut at)
                                .map_err(|e| fail(e, line, &plan.propf[k]))?
                        }
                        Sink::PropI(k) => {
                            s.pi[k][r] = next_i64(line, &mut at)
                                .map_err(|e| fail(e, line, &plan.propi[k]))?
                        }
                        Sink::Skip => {
                            if next_token(line, &mut at).map_or(true, |t| t[0] == b'#') {
                                return Err(short(line));
                            }
                        }
                        Sink::Image(_) | Sink::Element => unreachable!("not a data column"),
                    }
                }
                if images {
                    let mut img = [0i64; 3];
                    for (d, slot) in img.iter_mut().enumerate() {
                        *slot = next_i64(line, &mut at).map_err(|e| {
                            fail(
                                e,
                                line,
                                ["image flag ix", "image flag iy", "image flag iz"][d],
                            )
                        })?;
                    }
                    if !skip_images {
                        s.images[r] = pack_image(img[0], img[1], img[2]);
                    }
                }
                if next_token(line, &mut at).map_or(false, |t| t[0] != b'#') {
                    return Err(short(line));
                }
                // General-triclinic files store coordinates in the unrotated
                // frame: map them with the rotation that produced the cell.
                s.pos[r] = match g2r {
                    Some(rot) => crate::cell::general_to_restricted_point(rot, cell.origin, v),
                    None => v,
                };
                Ok(())
            };
            row().map_err(|m| {
                // A section header here means the block ended before the header
                // count of rows did.
                if section_of(line).is_some() {
                    "Atoms section has fewer rows than the header count".to_string()
                } else {
                    m
                }
            })
        },
    );
    drop(parts_sinks);
    res.map_err(|e| Error::at_line(&cur.path, cur.line_no() + e.line + 1, e.msg))?;
    cur.advance(len, nlines);
    let mut p = ParticleData::new(natoms);
    p.positions = out.pos;
    p.ids = out.ids;
    p.types = out.types;
    p.images = plan.images.then_some(out.images);
    for (name, v) in plan.propf.iter().zip(out.pf) {
        p.props
            .register(Property::from_data(name.clone(), 1, TypedVec::F64(v)));
    }
    for (name, v) in plan.propi.iter().zip(out.pi) {
        p.props
            .register(Property::from_data(name.clone(), 1, TypedVec::I64(v)));
    }
    Ok(p)
}

/// Velocity columns `vx vy vz` in row order.
type VelColumns = [Vec<f64>; 3];

/// Parse the `natoms` rows `id vx vy vz [...]` of a `Velocities` section.
fn parse_velocities(cur: &Cursor, natoms: usize) -> Result<(Vec<MdaTag>, VelColumns)> {
    let body = cur.rest();
    let (len, nlines, chunks) = row_chunks(body, natoms, RowRule::Content).ok_or_else(|| {
        Error::at_line(
            &cur.path,
            cur.line_no() + count_newlines(body),
            "truncated Velocities section",
        )
    })?;
    let body = &body[..len];
    let mut ids = vec![0 as MdaTag; natoms];
    let mut vel: VelColumns = [vec![0.0; natoms], vec![0.0; natoms], vec![0.0; natoms]];
    {
        let [vx, vy, vz] = &mut vel;
        let mut sinks: Vec<_> = split_by_chunks(&mut ids, &chunks, 1)
            .into_iter()
            .zip(split_by_chunks(vx, &chunks, 1))
            .zip(split_by_chunks(vy, &chunks, 1))
            .zip(split_by_chunks(vz, &chunks, 1))
            .collect();
        parse_rows(
            body,
            &chunks,
            &mut sinks,
            RowRule::Content,
            |(((ids, vx), vy), vz), r, line| {
                let mut at = 0usize;
                let bad = |_| {
                    if section_of(line).is_some() {
                        "Velocities section has fewer rows than the header count".to_string()
                    } else {
                        format!(
                            "bad Velocities row '{}'",
                            String::from_utf8_lossy(line).trim()
                        )
                    }
                };
                ids[r] = next_i64(line, &mut at).map_err(bad)?;
                vx[r] = next_f64(line, &mut at).map_err(bad)?;
                vy[r] = next_f64(line, &mut at).map_err(bad)?;
                vz[r] = next_f64(line, &mut at).map_err(bad)?;
                Ok(())
            },
        )
        .map_err(|e| Error::at_line(&cur.path, cur.line_no() + e.line + 1, e.msg))?;
    }
    cur.advance(len, nlines);
    Ok((ids, vel))
}

/// Store velocities as the `vx`, `vy`, `vz` properties (the dump column
/// names), in atom order. Rows are matched by id; LAMMPS writes both sections
/// in the same order, which is checked first and needs no lookup.
fn attach_velocities(
    p: &mut ParticleData,
    vids: Vec<MdaTag>,
    vel: VelColumns,
    opts: &ReadOptions,
) -> std::result::Result<(), String> {
    use rayon::prelude::*;
    let n = p.ids.len();
    if vids.len() != n {
        return Err(format!("Velocities has {} rows for {n} atoms", vids.len()));
    }
    let same_order = p.ids.par_iter().zip(vids.par_iter()).all(|(a, b)| a == b);
    let vel = if same_order {
        vel
    } else {
        // Sort both id lists and walk them together: row k of the sorted
        // velocities belongs to atom `atoms[k]`.
        let mut atoms: Vec<usize> = (0..n).collect();
        atoms.par_sort_unstable_by_key(|&i| p.ids[i]);
        let mut rows: Vec<usize> = (0..n).collect();
        rows.par_sort_unstable_by_key(|&i| vids[i]);
        let mut src = vec![0usize; n];
        for (&a, &v) in atoms.iter().zip(&rows) {
            if p.ids[a] != vids[v] {
                return Err(format!(
                    "Velocities row for atom id {} has no matching Atoms row",
                    vids[v]
                ));
            }
            src[a] = v;
        }
        vel.map(|col| src.par_iter().map(|&k| col[k]).collect())
    };
    for (name, col) in ["vx", "vy", "vz"].into_iter().zip(vel) {
        if opts.skips(name) {
            continue;
        }
        p.props
            .register(Property::from_data(name, 1, TypedVec::F64(col)));
    }
    Ok(())
}

/// Whether a frame property can be a dump column (text or binary).
///
/// Not a column the reader routes into the atoms; a bare token; and not a
/// float under a name the reader takes as integer (`mol`, `i_*`, ...), which
/// it would refuse to read back.
pub(crate) fn dump_prop_ok(p: &Property) -> std::result::Result<(), String> {
    crate::io::writer::plain_name(&p.name)?;
    if matches!(
        p.name.as_ref(),
        "id" | "type" | "x" | "y" | "z" | "ix" | "iy" | "iz" | "element"
    ) {
        return Err(format!(
            "property {:?} would be read back as an atom field",
            p.name
        ));
    }
    if is_int_column(&p.name) && matches!(p.dtype, DType::F32 | DType::F64) {
        return Err(format!(
            "float property {:?} has a name dump readers take as an integer column",
            p.name
        ));
    }
    Ok(())
}

/// Dump column names of a property: `name`, or `name[1] .. name[k]` for a
/// `k`-component one (the LAMMPS spelling of a per-atom vector).
pub(crate) fn dump_column_names(p: &Property) -> Vec<String> {
    if p.components == 1 {
        vec![p.name.to_string()]
    } else {
        (1..=p.components)
            .map(|c| format!("{}[{c}]", p.name))
            .collect()
    }
}

/// Type labels usable as a dump `element` column: every type in the frame
/// has a non-empty label that is a bare token.
pub(crate) fn element_labels(f: &Frame) -> Option<&[String]> {
    let labels = &f.type_labels;
    let max_type = f.particles.types.iter().copied().max().unwrap_or(0) as usize;
    (!labels.is_empty()
        && max_type <= labels.len()
        && f.particles.types.iter().all(|&t| t >= 1)
        && labels
            .iter()
            .all(|l| crate::io::writer::plain_name(l).is_ok()))
    .then_some(labels.as_slice())
}

/// Write frames as a LAMMPS dump stream (id type x y z, unwrapped xu/yu/zu
/// omitted; the `ix iy iz` image columns are emitted when present).
pub fn write_dump<T: std::io::Write>(w: &mut T, frames: &[Frame]) -> Result<()> {
    write_dump_opts(w, frames, &WriteOptions::default())
}

/// [`write_dump`] with [`WriteOptions`]: positions and box in
/// `opts.precision` (default 10 decimals), then `element` when
/// `opts.extended` and the frame has usable type labels, then the selected
/// properties (`name` or `name[1] .. name[k]`), integers as integers and
/// floats in `opts.prop_precision` (default: shortest round trip).
pub fn write_dump_opts<T: std::io::Write>(
    w: &mut T,
    frames: &[Frame],
    opts: &WriteOptions,
) -> Result<()> {
    let pf = opts.precision.unwrap_or(FloatFmt::Fixed(10));
    let prop_fmt = opts.prop_precision.unwrap_or(FloatFmt::Shortest);
    for f in frames {
        let (props, _) = select_props(f, &opts.props, dump_prop_ok)?;
        let cols: Vec<(usize, PropCol)> = props
            .iter()
            .map(|p| (p.components, PropCol::of(p)))
            .collect();
        let labels = element_labels(f).filter(|_| opts.extended);
        // A cell that is not LAMMPS-restricted is written in the general
        // `abc origin` form, by the general path.
        if pf != FloatFmt::Fixed(10)
            || labels.is_some()
            || !props.is_empty()
            || !f.cell.is_restricted()
        {
            write_dump_frame_general(w, f, pf, labels, &props, &cols, prop_fmt)?;
            continue;
        }
        writeln!(w, "ITEM: TIMESTEP")?;
        writeln!(w, "{}", f.meta.timestep)?;
        writeln!(w, "ITEM: NUMBER OF ATOMS")?;
        writeln!(w, "{}", f.natoms())?;
        let (xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz, tric) = f.cell.to_box_bounds();
        let pbc = f.cell.pbc;
        let flag = |b| if b { "pp" } else { "ff" };
        let (fx, fy, fz) = (flag(pbc[0]), flag(pbc[1]), flag(pbc[2]));
        if tric {
            let (xlo, xhi, ylo, yhi) = lohi_to_bounds(xlo, xhi, ylo, yhi, xy, xz, yz);
            writeln!(w, "ITEM: BOX BOUNDS xy xz yz {fx} {fy} {fz}")?;
            writeln!(w, "{xlo:.10} {xhi:.10} {xy:.10}")?;
            writeln!(w, "{ylo:.10} {yhi:.10} {xz:.10}")?;
            writeln!(w, "{zlo:.10} {zhi:.10} {yz:.10}")?;
        } else {
            writeln!(w, "ITEM: BOX BOUNDS {fx} {fy} {fz}")?;
            writeln!(w, "{xlo:.10} {xhi:.10}")?;
            writeln!(w, "{ylo:.10} {yhi:.10}")?;
            writeln!(w, "{zlo:.10} {zhi:.10}")?;
        }
        let with_images = f.particles.images.is_some();
        if with_images {
            writeln!(w, "ITEM: ATOMS id type x y z ix iy iz")?;
        } else {
            writeln!(w, "ITEM: ATOMS id type x y z")?;
        }
        let images = f.particles.images.as_deref().filter(|_| with_images);
        crate::io::writer::write_rows(w, f.natoms(), |w, i| {
            let p = f.particles.positions[i];
            write!(
                w,
                "{} {} {:.10} {:.10} {:.10}",
                f.particles.ids[i], f.particles.types[i], p[0], p[1], p[2]
            )?;
            if let Some(images) = images {
                let img = images[i];
                use crate::cell::{image_ipart, image_jpart, image_kpart};
                writeln!(
                    w,
                    " {} {} {}",
                    image_ipart(img),
                    image_jpart(img),
                    image_kpart(img)
                )
            } else {
                writeln!(w)
            }
        })?;
    }
    Ok(())
}

/// One dump frame through the general path of [`write_dump_opts`].
fn write_dump_frame_general<T: std::io::Write>(
    w: &mut T,
    f: &Frame,
    pf: FloatFmt,
    labels: Option<&[String]>,
    props: &[&Property],
    cols: &[(usize, PropCol)],
    prop_fmt: FloatFmt,
) -> Result<()> {
    use crate::cell::{image_ipart, image_jpart, image_kpart};
    let mut head: Vec<u8> = Vec::new();
    writeln!(head, "ITEM: TIMESTEP")?;
    writeln!(head, "{}", f.meta.timestep)?;
    writeln!(head, "ITEM: NUMBER OF ATOMS")?;
    writeln!(head, "{}", f.natoms())?;
    let (xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz, tric) = f.cell.to_box_bounds();
    let pbc = f.cell.pbc;
    let flag = |b| if b { "pp" } else { "ff" };
    let (fx, fy, fz) = (flag(pbc[0]), flag(pbc[1]), flag(pbc[2]));
    let line = |head: &mut Vec<u8>, vals: &[f64]| -> std::io::Result<()> {
        for (k, v) in vals.iter().enumerate() {
            if k > 0 {
                head.push(b' ');
            }
            pf.write(head, *v)?;
        }
        head.push(b'\n');
        Ok(())
    };
    if !f.cell.is_restricted() {
        // General triclinic (`dump_custom.cpp` `header_item_triclinic_general`):
        // the edge vectors and the origin; atoms stay in the same frame, which
        // the reader rotates into the restricted one as LAMMPS does.
        let (a, b, c, o) = f.cell.lammps_general()?;
        writeln!(head, "ITEM: BOX BOUNDS abc origin {fx} {fy} {fz}")?;
        line(&mut head, &[a[0], a[1], a[2], o[0]])?;
        line(&mut head, &[b[0], b[1], b[2], o[1]])?;
        line(&mut head, &[c[0], c[1], c[2], o[2]])?;
    } else if tric {
        let (xlo, xhi, ylo, yhi) = lohi_to_bounds(xlo, xhi, ylo, yhi, xy, xz, yz);
        writeln!(head, "ITEM: BOX BOUNDS xy xz yz {fx} {fy} {fz}")?;
        line(&mut head, &[xlo, xhi, xy])?;
        line(&mut head, &[ylo, yhi, xz])?;
        line(&mut head, &[zlo, zhi, yz])?;
    } else {
        writeln!(head, "ITEM: BOX BOUNDS {fx} {fy} {fz}")?;
        line(&mut head, &[xlo, xhi])?;
        line(&mut head, &[ylo, yhi])?;
        line(&mut head, &[zlo, zhi])?;
    }
    let images = f.particles.images.as_deref();
    head.extend_from_slice(b"ITEM: ATOMS id type x y z");
    if images.is_some() {
        head.extend_from_slice(b" ix iy iz");
    }
    if labels.is_some() {
        head.extend_from_slice(b" element");
    }
    for p in props {
        for name in dump_column_names(p) {
            head.push(b' ');
            head.extend_from_slice(name.as_bytes());
        }
    }
    head.push(b'\n');
    w.write_all(&head)?;
    crate::io::writer::write_rows(w, f.natoms(), |w, i| {
        let p = f.particles.positions[i];
        let t = f.particles.types[i];
        write!(w, "{} {}", f.particles.ids[i], t)?;
        for c in p {
            w.push(b' ');
            pf.write(w, c)?;
        }
        if let Some(images) = images {
            let img = images[i];
            write!(
                w,
                " {} {} {}",
                image_ipart(img),
                image_jpart(img),
                image_kpart(img)
            )?;
        }
        if let Some(labels) = labels {
            w.push(b' ');
            w.extend_from_slice(labels[t as usize - 1].as_bytes());
        }
        for &(k, col) in cols {
            for c in 0..k {
                w.push(b' ');
                col.write(w, i * k + c, prop_fmt)?;
            }
        }
        writeln!(w)
    })
}

fn parse_error(cur: &Cursor, msg: &str) -> Error {
    Error::at_line(&cur.path, cur.line_no(), msg)
}

/// The `ITEM:` header of one dump frame.
///
/// LAMMPS writes the optional `UNITS` and `TIME` items (`dump_modify units
/// yes`, `time yes`) ahead of `TIMESTEP`, so the items are accepted in any
/// order up to `ITEM: ATOMS`, which ends the header.
pub(crate) struct DumpHeader {
    pub timestep: i64,
    pub time: Option<f64>,
    pub units: Option<String>,
    pub natoms: usize,
    pub cell: UnitCell,
    pub g2r: Option<crate::cell::Mat3>,
    /// A general-triclinic box as given (rows `a b c`, unrotated).
    pub general: Option<UnitCell>,
    pub columns: Vec<String>,
}

/// A box read from a file: the cell, for a general-triclinic box the
/// rotation into the restricted frame, and the box as given.
pub(crate) type BoxRead = (UnitCell, Option<crate::cell::Mat3>, Option<UnitCell>);

impl DumpHeader {
    /// Under `ReadOptions::keep_general`, take a general-triclinic box as
    /// given: no rotation of the cell, positions or vector columns.
    pub(crate) fn apply_keep_general(&mut self, opts: &ReadOptions) {
        if opts.keep_general {
            if let Some(g) = self.general.take() {
                self.cell = g.with_pbc(self.cell.pbc);
                self.g2r = None;
            }
        }
    }
}

pub(crate) fn parse_dump_header(cur: &Cursor) -> Result<DumpHeader> {
    let mut timestep: Option<i64> = None;
    let mut time: Option<f64> = None;
    let mut units: Option<String> = None;
    let mut natoms: Option<usize> = None;
    let mut bx: Option<BoxRead> = None;
    let value = |what: &str| -> Result<&[u8]> {
        cur.next_line()
            .ok_or_else(|| parse_error(cur, &format!("unexpected EOF after ITEM: {what}")))
    };
    let columns = loop {
        let line = cur
            .next_line()
            .ok_or_else(|| parse_error(cur, "unexpected EOF in dump frame header"))?;
        let text = String::from_utf8_lossy(line);
        let Some(item) = text.trim_end().strip_prefix("ITEM:") else {
            return Err(parse_error(
                cur,
                &format!("expected an `ITEM:` line, got {:?}", text.trim_end()),
            ));
        };
        let item = item.trim();
        let dup = |seen: bool| -> Result<()> {
            if seen {
                Err(parse_error(
                    cur,
                    &format!("duplicate ITEM: {item} in one frame"),
                ))
            } else {
                Ok(())
            }
        };
        match item {
            "TIMESTEP" => {
                dup(timestep.is_some())?;
                timestep = Some(
                    fast_i64(value(item)?).ok_or_else(|| parse_error(cur, "bad TIMESTEP value"))?,
                );
            }
            "TIME" => {
                dup(time.is_some())?;
                time =
                    Some(fast_f64(value(item)?).ok_or_else(|| parse_error(cur, "bad TIME value"))?);
            }
            "UNITS" => {
                dup(units.is_some())?;
                units = Some(String::from_utf8_lossy(value(item)?).trim().to_string());
            }
            "NUMBER OF ATOMS" => {
                dup(natoms.is_some())?;
                let n = fast_i64(value(item)?)
                    .ok_or_else(|| parse_error(cur, "bad NUMBER OF ATOMS"))?;
                if n < 0 {
                    return Err(parse_error(cur, "negative NUMBER OF ATOMS"));
                }
                natoms = Some(n as usize);
            }
            _ if item.starts_with("BOX BOUNDS") => {
                dup(bx.is_some())?;
                bx = Some(parse_box(cur, line)?);
            }
            _ if item == "ATOMS" || item.starts_with("ATOMS ") => {
                break item[5..]
                    .split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>();
            }
            _ => {
                return Err(parse_error(
                    cur,
                    &format!("unsupported dump item `ITEM: {item}`"),
                ))
            }
        }
    };
    let missing = |what: &str| parse_error(cur, &format!("dump frame has no ITEM: {what}"));
    let timestep = timestep.ok_or_else(|| missing("TIMESTEP"))?;
    let natoms = natoms.ok_or_else(|| missing("NUMBER OF ATOMS"))?;
    let (cell, g2r, general) = bx.ok_or_else(|| missing("BOX BOUNDS"))?;
    Ok(DumpHeader {
        timestep,
        time,
        units,
        natoms,
        cell,
        g2r,
        general,
        columns,
    })
}

pub(crate) fn parse_frame(cur: &Cursor) -> Result<Frame> {
    parse_frame_with(cur, &ReadOptions::all())
}

fn parse_frame_with(cur: &Cursor, opts: &ReadOptions) -> Result<Frame> {
    let mut h = parse_dump_header(cur)?;
    h.apply_keep_general(opts);
    let plan = DumpPlan::resolve(&h.columns, opts)?;
    let body = cur.rest();
    let (len, nlines, chunks) = row_chunks(body, h.natoms, RowRule::Every).ok_or_else(|| {
        let have = count_newlines(body);
        Error::at_line(&cur.path, cur.line_no() + have, "truncated atom block")
    })?;
    let frame = plan
        .parse_text(&body[..len], &chunks, &h)
        .map_err(|e| Error::at_line(&cur.path, cur.line_no() + e.line + 1, e.msg))?;
    cur.advance(len, nlines);
    Ok(frame)
}

/// Which position columns a dump frame carries, resolved once per header.
///
/// LAMMPS writes one of `x y z` (wrapped), `xu yu zu` (unwrapped), `xs ys zs`
/// (scaled, the default of `dump atom`) or `xsu ysu zsu` (scaled unwrapped);
/// the first present in that order wins, as in LAMMPS `reader_native.cpp`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum PosKind {
    Wrapped,
    Unwrapped,
    Scaled,
    ScaledUnwrapped,
}

impl PosKind {
    /// The atom's position in the (restricted) cell frame from its three
    /// column values.
    #[inline]
    pub(crate) fn place(
        self,
        v: Coord3,
        cell: &UnitCell,
        g2r: Option<&crate::cell::Mat3>,
    ) -> Coord3 {
        match self {
            // Cartesian columns of a general-triclinic dump are in the
            // unrotated frame. LAMMPS wraps `x y z` itself.
            PosKind::Wrapped | PosKind::Unwrapped => {
                let p = match g2r {
                    Some(rot) => crate::cell::general_to_restricted_point(rot, cell.origin, v),
                    None => v,
                };
                if self == PosKind::Unwrapped {
                    cell.wrap(p)
                } else {
                    p
                }
            }
            // Fractional coordinates do not depend on the frame's rotation.
            PosKind::Scaled => cell.lamda2x(v),
            PosKind::ScaledUnwrapped => cell.wrap(cell.lamda2x(v)),
        }
    }
}

/// Where one dump column goes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Sink {
    Id,
    Type,
    Pos(usize),
    Image(usize),
    /// `element`: a per-type label, collected into `Frame::type_labels`.
    Element,
    /// The k-th f64 property column.
    PropF(usize),
    /// The k-th i64 property column.
    PropI(usize),
    /// A column the read options drop: stepped over, never parsed.
    Skip,
}

/// Integer-valued dump columns, kept as i64 properties.
fn is_int_column(name: &str) -> bool {
    matches!(name, "mol" | "proc" | "procp1") || name.starts_with("i_") || name.starts_with("i2_")
}

/// The column plan of a dump frame, resolved once per header (design §8.1):
/// every column has a sink, and every column that is not id, type, the chosen
/// position triple or an image flag is kept as a property.
#[derive(Debug)]
pub(crate) struct DumpPlan {
    pub sinks: Vec<Sink>,
    pub pos: PosKind,
    pub images: bool,
    pub propf: Vec<String>,
    pub propi: Vec<String>,
}

impl DumpPlan {
    pub(crate) fn resolve(cols: &[String], opts: &ReadOptions) -> Result<Self> {
        let find = |n: &str| cols.iter().position(|c| c == n);
        let (Some(id_col), Some(type_col)) = (find("id"), find("type")) else {
            return Err(Error::Unsupported(
                "dump atom block must contain id and type columns".into(),
            ));
        };
        let mut pos: Option<(PosKind, [usize; 3])> = None;
        for (kind, names) in [
            (PosKind::Wrapped, ["x", "y", "z"]),
            (PosKind::Unwrapped, ["xu", "yu", "zu"]),
            (PosKind::Scaled, ["xs", "ys", "zs"]),
            (PosKind::ScaledUnwrapped, ["xsu", "ysu", "zsu"]),
        ] {
            if let (Some(a), Some(b), Some(c)) = (find(names[0]), find(names[1]), find(names[2])) {
                pos = Some((kind, [a, b, c]));
                break;
            }
        }
        let Some((kind, pcols)) = pos else {
            return Err(Error::Unsupported(
                "dump atom block has no complete x/y/z, xu/yu/zu, xs/ys/zs or xsu/ysu/zsu columns"
                    .into(),
            ));
        };
        let img = [find("ix"), find("iy"), find("iz")];
        let nimg = img.iter().filter(|c| c.is_some()).count();
        // Skipped image flags are dropped whole, partial or not.
        let (img, nimg) = if opts.skips_images() {
            ([None; 3], 0)
        } else {
            (img, nimg)
        };
        if nimg != 0 && nimg != 3 {
            let have: Vec<&str> = ["ix", "iy", "iz"]
                .iter()
                .zip(&img)
                .filter(|(_, c)| c.is_some())
                .map(|(n, _)| *n)
                .collect();
            return Err(Error::parse(format!(
                "dump image flags need all of ix, iy, iz; the frame has only {}",
                have.join(", ")
            )));
        }
        let mut sinks = Vec::with_capacity(cols.len());
        let (mut propf, mut propi) = (Vec::new(), Vec::new());
        for (j, name) in cols.iter().enumerate() {
            let sink = if j == id_col {
                Sink::Id
            } else if j == type_col {
                Sink::Type
            } else if let Some(d) = pcols.iter().position(|&c| c == j) {
                Sink::Pos(d)
            } else if let Some(d) = img.iter().position(|&c| c == Some(j)) {
                Sink::Image(d)
            } else if opts.skips(name)
                || (opts.skips_images() && matches!(name.as_str(), "ix" | "iy" | "iz"))
            {
                Sink::Skip
            } else if name == "element" {
                Sink::Element
            } else if is_int_column(name) {
                propi.push(name.clone());
                Sink::PropI(propi.len() - 1)
            } else {
                propf.push(name.clone());
                Sink::PropF(propf.len() - 1)
            };
            sinks.push(sink);
        }
        Ok(DumpPlan {
            sinks,
            pos: kind,
            images: nimg == 3,
            propf,
            propi,
        })
    }

    /// Output arrays for `n` atoms, to be split per chunk.
    pub(crate) fn alloc(&self, n: usize) -> DumpArrays {
        DumpArrays {
            pos: vec![[0.0; 3]; n],
            ids: vec![0; n],
            types: vec![0; n],
            images: if self.images { vec![0; n] } else { Vec::new() },
            pf: self.propf.iter().map(|_| vec![0.0; n]).collect(),
            pi: self.propi.iter().map(|_| vec![0; n]).collect(),
            labels: Vec::new(),
        }
    }

    /// Parse a text body of exactly `h.natoms` rows in parallel.
    fn parse_text(
        &self,
        body: &[u8],
        chunks: &[RowChunk],
        h: &DumpHeader,
    ) -> std::result::Result<Frame, RowError> {
        let mut out = self.alloc(h.natoms);
        let mut sinks = out.split(chunks, self.images);
        let (cell, g2r) = (&h.cell, h.g2r.as_ref());
        // `id type x y z [ix iy iz]` (any position kind) takes a
        // straight-line parse; a row it does not accept goes through the
        // general loop, which also words the error.
        use Sink::{Id, Image, Pos, Type};
        let n = self.sinks.len();
        let plain = self.sinks.starts_with(&[Id, Type, Pos(0), Pos(1), Pos(2)])
            && (n == 5 || self.sinks[5..] == [Image(0), Image(1), Image(2)]);
        let plain_row = |s: &mut DumpSink, r: usize, line: &[u8]| -> Option<()> {
            let mut at = 0usize;
            let id = next_i64(line, &mut at).ok()?;
            let t = next_i64(line, &mut at).ok()?;
            if t < 1 {
                return None;
            }
            let v = [
                next_f64(line, &mut at).ok()?,
                next_f64(line, &mut at).ok()?,
                next_f64(line, &mut at).ok()?,
            ];
            let mut img = [0i64; 3];
            if n == 8 {
                for slot in &mut img {
                    *slot = next_i64(line, &mut at).ok()?;
                }
            }
            if next_token(line, &mut at).is_some() {
                return None;
            }
            s.ids[r] = id;
            s.types[r] = t as u32;
            s.pos[r] = self.pos.place(v, cell, g2r);
            if !s.images.is_empty() {
                s.images[r] = pack_image(img[0], img[1], img[2]);
            }
            Some(())
        };
        parse_rows(body, chunks, &mut sinks, RowRule::Every, |s, r, line| {
            if plain && plain_row(s, r, line).is_some() {
                return Ok(());
            }
            let mut at = 0usize;
            let mut v = [0.0f64; 3];
            let mut img = [0i64; 3];
            let mut elem: Option<&[u8]> = None;
            const SHORT: &str = "atom row column count mismatch";
            let bad = |e: Option<&[u8]>, what: &dyn Fn(&[u8]) -> String| match e {
                None => SHORT.to_string(),
                Some(tok) => what(tok),
            };
            for sink in &self.sinks {
                match *sink {
                    Sink::Id => {
                        s.ids[r] = next_i64(line, &mut at).map_err(|e| {
                            bad(e, &|t| {
                                format!("bad atom id '{}'", String::from_utf8_lossy(t))
                            })
                        })?
                    }
                    Sink::Type => {
                        let t = next_i64(line, &mut at).map_err(|e| {
                            bad(e, &|t| {
                                format!("bad atom type '{}'", String::from_utf8_lossy(t))
                            })
                        })?;
                        if t < 1 {
                            return Err(format!("atom type {t} is not a positive integer"));
                        }
                        s.types[r] = t as u32;
                    }
                    Sink::Pos(d) => {
                        v[d] = next_f64(line, &mut at).map_err(|e| {
                            bad(e, &|t| {
                                format!("bad coordinate '{}'", String::from_utf8_lossy(t))
                            })
                        })?
                    }
                    Sink::Image(d) => {
                        img[d] = next_i64(line, &mut at).map_err(|e| {
                            bad(e, &|t| {
                                format!(
                                    "bad image flag '{}' '{}'",
                                    ["ix", "iy", "iz"][d],
                                    String::from_utf8_lossy(t)
                                )
                            })
                        })?
                    }
                    Sink::Element => {
                        elem = Some(next_token(line, &mut at).ok_or_else(|| SHORT.to_string())?)
                    }
                    Sink::PropF(k) => {
                        s.pf[k][r] = next_f64(line, &mut at)
                            .map_err(|e| bad(e, &|t| bad_value(t, &self.propf[k])))?
                    }
                    Sink::PropI(k) => {
                        s.pi[k][r] = next_i64(line, &mut at)
                            .map_err(|e| bad(e, &|t| bad_value(t, &self.propi[k])))?
                    }
                    Sink::Skip => {
                        next_token(line, &mut at).ok_or_else(|| SHORT.to_string())?;
                    }
                }
            }
            if next_token(line, &mut at).is_some() {
                return Err("atom row column count mismatch".into());
            }
            if let Some(e) = elem {
                s.label(r, e);
            }
            s.pos[r] = self.pos.place(v, cell, g2r);
            if !s.images.is_empty() {
                s.images[r] = pack_image(img[0], img[1], img[2]);
            }
            Ok(())
        })?;
        let labels = merge_labels(&sinks);
        drop(sinks);
        out.labels = labels;
        Ok(self.to_frame(out, h))
    }

    /// Assemble the frame from filled arrays.
    pub(crate) fn to_frame(&self, out: DumpArrays, h: &DumpHeader) -> Frame {
        let mut parts = ParticleData::new(h.natoms);
        parts.positions = out.pos;
        parts.ids = out.ids;
        parts.types = out.types;
        parts.natoms = parts.ids.len();
        parts.images = self.images.then_some(out.images);
        for (name, v) in self.propf.iter().zip(out.pf) {
            parts
                .props
                .register(Property::from_data(name.clone(), 1, TypedVec::F64(v)));
        }
        for (name, v) in self.propi.iter().zip(out.pi) {
            parts
                .props
                .register(Property::from_data(name.clone(), 1, TypedVec::I64(v)));
        }
        if let Some(rot) = &h.g2r {
            rotate_vector_props(&mut parts.props, rot);
        }
        let mut frame = Frame::new(h.cell, parts);
        frame.meta.timestep = h.timestep;
        frame.meta.time = h.time;
        frame.meta.units = h.units.clone();
        frame.type_labels = out.labels;
        frame
    }
}

fn bad_value(tok: &[u8], column: &str) -> String {
    format!(
        "bad value '{}' in column {column}",
        String::from_utf8_lossy(tok)
    )
}

/// Whole-frame output arrays of a dump read.
pub(crate) struct DumpArrays {
    pub pos: Vec<Coord3>,
    pub ids: Vec<MdaTag>,
    pub types: Vec<MdaType>,
    pub images: Vec<MdaImage>,
    pub pf: Vec<Vec<f64>>,
    pub pi: Vec<Vec<i64>>,
    pub labels: Vec<String>,
}

/// One chunk's window into [`DumpArrays`].
pub(crate) struct DumpSink<'a, 'b> {
    pub pos: &'a mut [Coord3],
    pub ids: &'a mut [MdaTag],
    pub types: &'a mut [MdaType],
    pub images: &'a mut [MdaImage],
    pub pf: Vec<&'a mut [f64]>,
    pub pi: Vec<&'a mut [i64]>,
    /// First `element` label seen per type in this chunk (index type-1).
    pub labels: Vec<Option<&'b [u8]>>,
}

impl<'a, 'b> DumpSink<'a, 'b> {
    #[inline]
    fn label(&mut self, r: usize, tok: &'b [u8]) {
        let t = self.types[r] as usize;
        if self.labels.len() < t {
            self.labels.resize(t, None);
        }
        if self.labels[t - 1].is_none() {
            self.labels[t - 1] = Some(tok);
        }
    }
}

impl DumpArrays {
    pub(crate) fn split<'b>(&mut self, chunks: &[RowChunk], images: bool) -> Vec<DumpSink<'_, 'b>> {
        let mut pos = split_by_chunks(&mut self.pos, chunks, 1).into_iter();
        let mut ids = split_by_chunks(&mut self.ids, chunks, 1).into_iter();
        let mut types = split_by_chunks(&mut self.types, chunks, 1).into_iter();
        let mut img = split_by_chunks(&mut self.images, chunks, usize::from(images)).into_iter();
        let mut pf: Vec<_> = self
            .pf
            .iter_mut()
            .map(|v| split_by_chunks(v, chunks, 1).into_iter())
            .collect();
        let mut pi: Vec<_> = self
            .pi
            .iter_mut()
            .map(|v| split_by_chunks(v, chunks, 1).into_iter())
            .collect();
        (0..chunks.len())
            .map(|_| DumpSink {
                pos: pos.next().unwrap(),
                ids: ids.next().unwrap(),
                types: types.next().unwrap(),
                images: img.next().unwrap(),
                pf: pf.iter_mut().map(|it| it.next().unwrap()).collect(),
                pi: pi.iter_mut().map(|it| it.next().unwrap()).collect(),
                labels: Vec::new(),
            })
            .collect()
    }
}

/// Type labels from the chunks' `element` columns, first occurrence wins.
fn merge_labels(sinks: &[DumpSink<'_, '_>]) -> Vec<String> {
    let n = sinks.iter().map(|s| s.labels.len()).max().unwrap_or(0);
    (0..n)
        .map(|t| {
            sinks
                .iter()
                .find_map(|s| s.labels.get(t).copied().flatten())
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_default()
        })
        .collect()
}

/// Parse `ITEM: BOX BOUNDS`, returning the cell and — for a general triclinic
/// box — the rotation that maps atom coordinates into the restricted frame.
fn parse_box(cur: &Cursor, line: &[u8]) -> Result<BoxRead> {
    let words: Vec<&[u8]> = Fields::all(line);
    // words[0]=ITEM:, [1]=BOX, [2]=BOUNDS, then one of:
    //   "pp pp pp"                      -> 2 numbers/line (orthogonal)
    //   "xy xz yz pp pp pp"             -> 3 numbers/line (restricted triclinic)
    //   "abc origin pp pp pp"           -> 4 numbers/line (general triclinic)
    let general = words.iter().any(|w| *w == b"abc");
    let triclinic = general
        || words
            .iter()
            .any(|w| *w == b"xy" || *w == b"xz" || *w == b"yz");
    let mut nums = [[0.0f64; 4]; 3];
    for slot in nums.iter_mut() {
        let row = cur
            .next_line()
            .ok_or_else(|| parse_error(cur, "truncated box bounds"))?;
        let vals: Vec<&[u8]> = Fields::all(row);
        if vals.len() < 2 {
            return Err(parse_error(cur, "box bound line needs >= 2 numbers"));
        }
        for (k, v) in vals.iter().take(4).enumerate() {
            slot[k] = super::engine::fast_f64(v)
                .ok_or_else(|| parse_error(cur, "bad box bound number"))?;
        }
    }
    let pbc = parse_pbc(line)?;
    let mut rot: Option<crate::cell::Mat3> = None;
    let mut given: Option<UnitCell> = None;
    let cell = if general {
        // `avec[0..2] boxlo[0]`, `bvec[0..2] boxlo[1]`, `cvec[0..2] boxlo[2]`
        // (LAMMPS dump_custom.cpp `header_item_triclinic_general`).
        let abco = (
            [nums[0][0], nums[0][1], nums[0][2]],
            [nums[1][0], nums[1][1], nums[1][2]],
            [nums[2][0], nums[2][1], nums[2][2]],
            [nums[0][3], nums[1][3], nums[2][3]],
        );
        let (cell, r) = UnitCell::from_general_triclinic_rot(abco.0, abco.1, abco.2, abco.3)?;
        rot = Some(r);
        given = Some(UnitCell::general_as_given(abco.0, abco.1, abco.2, abco.3)?.with_pbc(pbc));
        cell
    } else if triclinic {
        // The rows are `xlo_bound xhi_bound xy`, `ylo_bound yhi_bound xz`,
        // `zlo zhi yz`: the bounding box of the tilted cell, not its lo/hi.
        let (xy, xz, yz) = (nums[0][2], nums[1][2], nums[2][2]);
        let (xlo, xhi, ylo, yhi) =
            bounds_to_lohi(nums[0][0], nums[0][1], nums[1][0], nums[1][1], xy, xz, yz);
        UnitCell::from_box_bounds(xlo, xhi, xy, ylo, yhi, xz, nums[2][0], nums[2][1], yz)
    } else {
        let mut c = UnitCell::orthorhombic(
            nums[0][1] - nums[0][0],
            nums[1][1] - nums[1][0],
            nums[2][1] - nums[2][0],
        );
        // orthogonal cells still carry their boxlo origin
        c.origin = [nums[0][0], nums[1][0], nums[2][0]];
        c
    };
    Ok((cell.with_pbc(pbc), rot, given))
}

/// Triclinic dump bounding box to the cell's lo/hi, inverting LAMMPS
/// `Domain::set_global_box` (`boxlo_bound = lo + min(0,xy) + min(0,xz)`, …).
pub(crate) fn bounds_to_lohi(
    xlo_b: f64,
    xhi_b: f64,
    ylo_b: f64,
    yhi_b: f64,
    xy: f64,
    xz: f64,
    yz: f64,
) -> (f64, f64, f64, f64) {
    (
        xlo_b - xy.min(0.0) - xz.min(0.0),
        xhi_b - xy.max(0.0) - xz.max(0.0),
        ylo_b - yz.min(0.0),
        yhi_b - yz.max(0.0),
    )
}

/// The cell's lo/hi to the triclinic dump bounding box (inverse of
/// [`bounds_to_lohi`]).
pub(crate) fn lohi_to_bounds(
    xlo: f64,
    xhi: f64,
    ylo: f64,
    yhi: f64,
    xy: f64,
    xz: f64,
    yz: f64,
) -> (f64, f64, f64, f64) {
    (
        xlo + xy.min(0.0) + xz.min(0.0),
        xhi + xy.max(0.0) + xz.max(0.0),
        ylo + yz.min(0.0),
        yhi + yz.max(0.0),
    )
}

/// Per-dimension periodicity from the `ITEM: BOX BOUNDS` flags.
///
/// Each dimension has a two-letter token from `p f s m` (periodic, fixed,
/// shrink-wrapped, shrink-wrapped with a minimum). A dimension is periodic only
/// when its token is `pp`; LAMMPS cannot mix `p` with another kind. A header
/// with no flags (very old dumps) reads as fully periodic.
fn parse_pbc(header: &[u8]) -> Result<[bool; 3]> {
    let is_flag = |w: &[u8]| w.len() == 2 && w.iter().all(|c| b"pfsm".contains(c));
    let flags: Vec<&[u8]> = Fields::all(header)
        .into_iter()
        .skip(3)
        .filter(|w| is_flag(w))
        .collect();
    match flags.len() {
        0 => Ok([true; 3]),
        3 => {
            let mut p = [true; 3];
            for (d, w) in flags.iter().enumerate() {
                p[d] = match *w {
                    b"pp" => true,
                    w if w.contains(&b'p') => {
                        return Err(Error::parse(format!(
                            "boundary flag `{}` mixes periodic with non-periodic",
                            String::from_utf8_lossy(w)
                        )))
                    }
                    _ => false,
                };
            }
            Ok(p)
        }
        n => Err(Error::parse(format!(
            "ITEM: BOX BOUNDS has {n} boundary flags, expected 3"
        ))),
    }
}

/// Write a **single** [`Frame`] as a LAMMPS data file (`read_data` format).
///
/// The box comes from [`UnitCell::to_box_bounds`], so the origin and the
/// restricted-triclinic tilts (`xy`, `xz`, `yz`, i.e. the *lower* triangle of
/// the h-matrix) are preserved. Writing them from the upper triangle would emit
/// `0 0 0` for every restricted cell, because those entries are zero by
/// construction.
///
/// Atom ids are taken from the frame when present and synthesised `1..=natoms`
/// otherwise. A LAMMPS data file describes one configuration, so more than one
/// frame is rejected rather than silently dropping all but the first.
pub fn write_data_file<W: Write>(w: &mut W, frames: &[Frame]) -> Result<()> {
    write_data_file_opts(w, frames, &WriteOptions::default())
}

/// Whether a frame property has a place in a data file: `q`, `mol`,
/// `diameter` and `density` as `Atoms` style columns, `vx vy vz` as the
/// `Velocities` section. Which combinations fit one style is decided by
/// [`data_plan`].
pub(crate) fn data_prop_ok(p: &Property) -> std::result::Result<(), String> {
    let name = p.name.as_ref();
    if !matches!(
        name,
        "q" | "mol" | "diameter" | "density" | "vx" | "vy" | "vz"
    ) {
        return Err(format!(
            "a data file has no place for property {name:?} \
             (only q, mol, diameter, density and vx vy vz)"
        ));
    }
    if p.components != 1 {
        return Err(format!("data property {name} must have one component"));
    }
    if name == "mol" && matches!(p.dtype, DType::F32 | DType::F64) {
        return Err("data property mol must be an integer property".into());
    }
    Ok(())
}

/// The `Atoms` layout and `Velocities` of a data file written with `opts`.
pub(crate) struct DataPlan<'a> {
    pub style: &'static str,
    /// Values of the `q`, `mol`, `diameter`, `density` columns the style has.
    pub q: Option<PropCol<'a>>,
    pub mol: Option<PropCol<'a>>,
    pub diameter: Option<PropCol<'a>>,
    pub density: Option<PropCol<'a>>,
    pub vel: Option<[PropCol<'a>; 3]>,
    pub written: Vec<String>,
    pub skipped: Vec<String>,
}

/// Fit the selected properties to one atom style: `mol` and `q` give
/// `full`, `charge` or `molecular`; `diameter` with `density` gives `sphere`
/// (and cannot share a style with `mol` or `q`); `vx vy vz` go together.
/// A property that does not fit is an error when named and skipped under
/// `PropSelect::All`.
pub(crate) fn data_plan<'a>(f: &'a Frame, opts: &WriteOptions) -> Result<DataPlan<'a>> {
    let (props, mut skipped) = select_props(f, &opts.props, data_prop_ok)?;
    let named = matches!(opts.props, PropSelect::Names(_));
    let get = |n: &str| props.iter().find(|p| p.name == n).copied();
    let mut plan = DataPlan {
        style: "atomic",
        q: get("q").map(PropCol::of),
        mol: get("mol").map(PropCol::of),
        diameter: None,
        density: None,
        vel: None,
        written: Vec::new(),
        skipped: Vec::new(),
    };
    if let Some(style) = opts.atom_style.as_deref() {
        return forced_style_plan(f, plan, style, &props, skipped, named);
    }
    let mut refuse = |names: &[&str], why: &str| -> Result<()> {
        let present: Vec<&str> = names.iter().copied().filter(|n| get(n).is_some()).collect();
        if present.is_empty() {
            return Ok(());
        }
        if named {
            return Err(Error::invalid(format!(
                "cannot write {} to a data file: {why}",
                present.join(", ")
            )));
        }
        skipped.extend(present.iter().map(|s| s.to_string()));
        Ok(())
    };
    let sphere = get("diameter").is_some() && get("density").is_some();
    if plan.q.is_some() || plan.mol.is_some() {
        refuse(
            &["diameter", "density"],
            "no supported atom style has both mol/q and diameter/density",
        )?;
    } else if sphere {
        plan.diameter = get("diameter").map(PropCol::of);
        plan.density = get("density").map(PropCol::of);
    } else {
        refuse(
            &["diameter", "density"],
            "atom style sphere needs both diameter and density",
        )?;
    }
    plan.style = match (
        plan.mol.is_some(),
        plan.q.is_some(),
        plan.diameter.is_some(),
    ) {
        (true, true, _) => "full",
        (false, true, _) => "charge",
        (true, false, _) => "molecular",
        (false, false, true) => "sphere",
        (false, false, false) => "atomic",
    };
    match (get("vx"), get("vy"), get("vz")) {
        (Some(x), Some(y), Some(z)) => {
            plan.vel = Some([PropCol::of(x), PropCol::of(y), PropCol::of(z)])
        }
        _ => refuse(
            &["vx", "vy", "vz"],
            "the Velocities section needs all of vx, vy, vz",
        )?,
    }
    for p in &props {
        if !skipped.iter().any(|s| s == p.name.as_ref()) {
            plan.written.push(p.name.to_string());
        }
    }
    plan.skipped = skipped;
    Ok(plan)
}

/// Atom styles `--out-atom-style` accepts (those [`style_columns`] knows).
pub const WRITABLE_STYLES: [&str; 7] = [
    "atomic",
    "charge",
    "full",
    "molecular",
    "bond",
    "angle",
    "sphere",
];

/// [`data_plan`] for an explicit atom style: the style's per-atom columns
/// come from the frame's properties of those names (selected or not, and
/// required), the other style properties among the selected ones are refused
/// when named and skipped under `all`, velocities as usual.
fn forced_style_plan<'a>(
    f: &'a Frame,
    mut plan: DataPlan<'a>,
    style: &str,
    props: &[&'a Property],
    mut skipped: Vec<String>,
    named: bool,
) -> Result<DataPlan<'a>> {
    let canon = WRITABLE_STYLES
        .iter()
        .copied()
        .find(|s| *s == style)
        .ok_or_else(|| {
            Error::invalid(format!(
                "unknown atom style {style:?} for data output (one of: {})",
                WRITABLE_STYLES.join(", ")
            ))
        })?;
    let cols = style_columns(canon).expect("writable styles have columns");
    let has = |c: DCol| cols.contains(&c);
    let n = f.natoms();
    let mut written: Vec<String> = Vec::new();
    let mut need = |name: &str| -> Result<PropCol<'a>> {
        let p = props
            .iter()
            .copied()
            .find(|p| p.name == name)
            .or_else(|| f.particles.props.get(name))
            .ok_or_else(|| {
                Error::invalid(format!(
                    "atom style {canon} needs a per-atom property {name:?}, \
                     which the frame does not have"
                ))
            })?;
        if !p.per_particle || p.components == 0 || p.len() != n {
            return Err(Error::invalid(format!(
                "property {name} is not one value per atom ({} values for {n} atoms)",
                p.len()
            )));
        }
        data_prop_ok(p).map_err(Error::invalid)?;
        written.push(name.to_string());
        Ok(PropCol::of(p))
    };
    plan.mol = if has(DCol::Mol) {
        Some(need("mol")?)
    } else {
        None
    };
    plan.q = if has(DCol::Q) { Some(need("q")?) } else { None };
    plan.diameter = if has(DCol::Diameter) {
        Some(need("diameter")?)
    } else {
        None
    };
    plan.density = if has(DCol::Density) {
        Some(need("density")?)
    } else {
        None
    };
    plan.style = canon;
    let in_style = |name: &str| written.iter().any(|w| w == name);
    let mut extra: Vec<&str> = Vec::new();
    for name in ["mol", "q", "diameter", "density"] {
        if !in_style(name) && props.iter().any(|p| p.name == name) {
            extra.push(name);
        }
    }
    if !extra.is_empty() {
        if named {
            return Err(Error::invalid(format!(
                "cannot write {} to a data file: atom style {canon} has no column for {}",
                extra.join(", "),
                if extra.len() == 1 { "it" } else { "them" }
            )));
        }
        skipped.extend(extra.iter().map(|s| s.to_string()));
    }
    let get = |n: &str| props.iter().copied().find(|p| p.name == n);
    match (get("vx"), get("vy"), get("vz")) {
        (Some(x), Some(y), Some(z)) => {
            plan.vel = Some([PropCol::of(x), PropCol::of(y), PropCol::of(z)]);
            written.extend(["vx", "vy", "vz"].map(String::from));
        }
        (None, None, None) => {}
        _ => {
            let present: Vec<String> = ["vx", "vy", "vz"]
                .into_iter()
                .filter(|n| get(n).is_some())
                .map(String::from)
                .collect();
            if named {
                return Err(Error::invalid(format!(
                    "cannot write {} to a data file: the Velocities section needs all of vx, vy, vz",
                    present.join(", ")
                )));
            }
            skipped.extend(present);
        }
    }
    plan.written = written;
    plan.skipped = skipped;
    Ok(plan)
}

/// Type labels a data file's `Atom Type Labels` section can hold: one per
/// type, each a bare token not starting with a digit (LAMMPS' rule, which
/// keeps a label from reading as a type number).
fn data_type_labels(f: &Frame, ntypes: usize) -> Option<&[String]> {
    let l = &f.type_labels;
    (l.len() == ntypes
        && l.iter().all(|s| {
            crate::io::writer::plain_name(s).is_ok()
                && !s.starts_with(|c: char| c.is_ascii_digit() || c == '#')
        }))
    .then_some(l.as_slice())
}

/// [`write_data_file`] with [`WriteOptions`]: the selected properties as
/// `Atoms` style columns and a `Velocities` section ([`data_plan`]), and with
/// `opts.extended` the `Atom Type Labels` section and image flags.
/// `opts.precision` sets box and positions (default: shortest round trip).
pub fn write_data_file_opts<W: Write>(
    w: &mut W,
    frames: &[Frame],
    opts: &WriteOptions,
) -> Result<()> {
    if frames.is_empty() {
        return Ok(());
    }
    if frames.len() > 1 {
        return Err(Error::InvalidData(format!(
            "LAMMPS data format holds a single frame, got {}",
            frames.len()
        )));
    }
    let f = &frames[0];
    let nat = f.natoms();
    let max_type = f.particles.types.iter().max().copied().unwrap_or(0).max(1) as usize;
    // Never fewer types than the atoms use: a file declaring fewer is
    // refused by every reader.
    let ntypes = f.type_labels.len().max(max_type).max(1);
    let plan = data_plan(f, opts)?;
    let prop_fmt = opts.prop_precision.unwrap_or(FloatFmt::Shortest);
    let general =
        opts.extended || opts.precision.is_some() || plan.style != "atomic" || plan.vel.is_some();

    writeln!(w, "LAMMPS data file created by rmda")?;
    writeln!(w)?;
    writeln!(w, "{} atoms", nat)?;
    writeln!(w, "{} atom types", ntypes)?;
    writeln!(w)?;

    let (xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz, tric) = f.cell.to_box_bounds();
    if !f.cell.is_restricted() {
        // General triclinic (`read_data` `avec bvec cvec abc origin`, LAMMPS
        // 7Feb2024 and later); the atoms stay in the same frame.
        let (a, b, c, o) = f.cell.lammps_general()?;
        let mut head: Vec<u8> = Vec::new();
        for (vals, tail) in [(a, "avec"), (b, "bvec"), (c, "cvec"), (o, "abc origin")] {
            for v in vals {
                match opts.precision {
                    None => write!(head, "{v}")?,
                    Some(pf) => pf.write(&mut head, v)?,
                }
                head.push(b' ');
            }
            writeln!(head, "{tail}")?;
        }
        w.write_all(&head)?;
    } else {
        match opts.precision {
            None => {
                writeln!(w, "{xlo} {xhi} xlo xhi")?;
                writeln!(w, "{ylo} {yhi} ylo yhi")?;
                writeln!(w, "{zlo} {zhi} zlo zhi")?;
                if tric {
                    writeln!(w, "{xy} {xz} {yz} xy xz yz")?;
                }
            }
            Some(pf) => {
                let mut head: Vec<u8> = Vec::new();
                for (vals, tail) in [
                    (&[xlo, xhi][..], "xlo xhi"),
                    (&[ylo, yhi][..], "ylo yhi"),
                    (&[zlo, zhi][..], "zlo zhi"),
                    (&[xy, xz, yz][..], "xy xz yz"),
                ] {
                    if tail == "xy xz yz" && !tric {
                        continue;
                    }
                    for v in vals {
                        pf.write(&mut head, *v)?;
                        head.push(b' ');
                    }
                    writeln!(head, "{tail}")?;
                }
                w.write_all(&head)?;
            }
        }
    }
    writeln!(w)?;

    if opts.extended {
        if let Some(labels) = data_type_labels(f, ntypes) {
            writeln!(w, "Atom Type Labels")?;
            writeln!(w)?;
            for (t, l) in labels.iter().enumerate() {
                writeln!(w, "{} {l}", t + 1)?;
            }
            writeln!(w)?;
        }
    }

    // Masses only when every type has one. A made-up value would run in
    // LAMMPS with the wrong dynamics; a missing section makes LAMMPS ask for
    // `mass` commands instead.
    let masses: Option<Vec<f64>> = (1..=ntypes)
        .map(|t| f.type_masses.get(t - 1).copied().filter(|m| m.is_finite()))
        .collect();
    if let Some(masses) = masses {
        writeln!(w, "Masses")?;
        writeln!(w)?;
        for (t, m) in masses.iter().enumerate() {
            writeln!(w, "{} {m}", t + 1)?;
        }
        writeln!(w)?;
    }

    writeln!(w, "Atoms # {}", plan.style)?;
    writeln!(w)?;
    let have_ids = f.particles.ids.len() == nat;
    let id_of = |i: usize| {
        if have_ids {
            f.particles.ids[i]
        } else {
            (i as i64) + 1
        }
    };
    if !general {
        return crate::io::writer::write_rows(w, nat, |w, i| {
            let id = id_of(i);
            let t = f.particles.types.get(i).copied().unwrap_or(1).max(1);
            let p = f.particles.positions[i];
            writeln!(w, "{} {} {:e} {:e} {:e}", id, t, p[0], p[1], p[2])
        });
    }
    let images = f
        .particles
        .images
        .as_deref()
        .filter(|im| opts.extended && im.len() == nat);
    let cols = style_columns(plan.style).expect("data_plan picks a known style");
    let pos = |w: &mut Vec<u8>, v: f64| match opts.precision {
        None => write!(w, "{v:e}"),
        Some(pf) => pf.write(w, v),
    };
    crate::io::writer::write_rows(w, nat, |w, i| {
        let t = f.particles.types.get(i).copied().unwrap_or(1).max(1);
        let p = f.particles.positions[i];
        for (k, c) in cols.iter().enumerate() {
            if k > 0 {
                w.push(b' ');
            }
            match c {
                DCol::Id => write!(w, "{}", id_of(i))?,
                DCol::Type => write!(w, "{t}")?,
                DCol::X => pos(w, p[0])?,
                DCol::Y => pos(w, p[1])?,
                DCol::Z => pos(w, p[2])?,
                DCol::Mol => plan.mol.expect("style has mol").write(w, i, prop_fmt)?,
                DCol::Q => plan.q.expect("style has q").write(w, i, prop_fmt)?,
                DCol::Diameter => plan.diameter.expect("sphere").write(w, i, prop_fmt)?,
                DCol::Density => plan.density.expect("sphere").write(w, i, prop_fmt)?,
            }
        }
        if let Some(images) = images {
            use crate::cell::{image_ipart, image_jpart, image_kpart};
            let img = images[i];
            write!(
                w,
                " {} {} {}",
                image_ipart(img),
                image_jpart(img),
                image_kpart(img)
            )?;
        }
        writeln!(w)
    })?;
    if let Some(vel) = plan.vel {
        writeln!(w)?;
        writeln!(w, "Velocities")?;
        writeln!(w)?;
        crate::io::writer::write_rows(w, nat, |w, i| {
            write!(w, "{}", id_of(i))?;
            for c in vel {
                w.push(b' ');
                c.write(w, i, prop_fmt)?;
            }
            writeln!(w)
        })?;
    }
    Ok(())
}

pub fn read_next_dump_from_cursor(cur: &Cursor) -> Result<Option<Frame>> {
    read_next_dump_with(cur, &ReadOptions::all())
}

/// [`read_next_dump_from_cursor`] keeping only the columns `opts` allows.
pub fn read_next_dump_with(cur: &Cursor, opts: &ReadOptions) -> Result<Option<Frame>> {
    while !cur.eof() {
        let Some(line) = cur.peek_line() else { break };
        if line.is_empty() {
            cur.next_line();
            continue;
        }
        if line.starts_with(b"ITEM:") {
            let f = parse_frame_with(cur, opts)?;
            return Ok(Some(f));
        }
        cur.next_line();
    }
    Ok(None)
}

/// Step over the dump frame at the cursor: its header is read, its atom rows
/// are only counted. Returns `(timestep, natoms)`, `None` at EOF.
pub fn skip_next_dump(cur: &Cursor) -> Result<Option<(i64, usize)>> {
    while let Some(line) = cur.peek_line() {
        if line.is_empty() {
            cur.next_line();
            continue;
        }
        if !line.starts_with(b"ITEM:") {
            cur.next_line();
            continue;
        }
        let h = parse_dump_header(cur)?;
        let (len, lines) = crate::io::engine::rows_extent(cur.rest(), h.natoms, RowRule::Every)
            .ok_or_else(|| parse_error(cur, "truncated atom block"))?;
        cur.advance(len, lines);
        return Ok(Some((h.timestep, h.natoms)));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn read_data_bytes_for_test(name: &str, src: &[u8]) -> Frame {
        let cur = Cursor::from_bytes(src, name);
        parse_data_frame(&cur).expect("data file should parse")
    }

    /// Parse a data file that is expected to fail; return the error text.
    fn parse_data_err(name: &str, src: &[u8]) -> String {
        let cur = Cursor::from_bytes(src, name);
        parse_data_frame(&cur).unwrap_err().to_string()
    }

    fn write_dump(tmp: &Path, content: &str) -> PathBuf {
        let p = tmp.join("f.dump");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        p
    }

    const D: &str = "ITEM: TIMESTEP\n3\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id type x y z\n1 1 1 2 3\n2 1 8 9 10\n";

    /// General-triclinic dump: the cell is rotated into the restricted frame
    /// *and* the atoms are rotated with it. Rotating only the cell is the
    /// classic silent bug (atoms land in the wrong place while the cell looks
    /// plausible), so this pins both.
    const GEN: &str = concat!(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n",
        "ITEM: BOX BOUNDS abc origin pp pp pp\n",
        "4 1.2 0.3 0\n0 4.2 0.5 0\n0 0 5 0\n",
        "ITEM: ATOMS id type x y z\n",
        "1 1 0 0 0\n",
        "2 1 1 1 1\n",
    );

    fn tilted_fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/triclinic_tilted.dump")
    }

    /// `tools/gen_tilted_dump.py`: the cell ASE reads from the same file.
    const TILTED_H: [[f64; 3]; 3] = [[10.0, 0.0, 0.0], [-2.0, 9.0, 0.0], [1.5, -1.0, 8.0]];
    const TILTED_ORIGIN: [f64; 3] = [1.0, -2.0, 0.5];

    #[test]
    fn restricted_triclinic_dump_reads_bounds_not_lohi() {
        let f = crate::io::read_frames(&tilted_fixture()).unwrap().remove(0);
        let m = f.cell.h_matrix();
        for (r, row) in TILTED_H.iter().enumerate() {
            for (c, want) in row.iter().enumerate() {
                assert!(
                    (m.m[r][c] - want).abs() < 1e-12,
                    "h[{r}][{c}] = {}",
                    m.m[r][c]
                );
            }
        }
        for (d, want) in TILTED_ORIGIN.iter().enumerate() {
            assert!((f.cell.origin[d] - want).abs() < 1e-12, "origin[{d}]");
        }
        // `pp fm ss`: only x is periodic.
        assert_eq!(f.cell.pbc, [true, false, false]);
    }

    #[test]
    fn restricted_triclinic_dump_round_trips_through_the_bounds() {
        let f = crate::io::read_frames(&tilted_fixture()).unwrap().remove(0);
        let mut out = Vec::new();
        super::write_dump(&mut out, std::slice::from_ref(&f)).unwrap();
        let text = String::from_utf8(out).unwrap();
        // The written rows are the LAMMPS bounding box, as the fixture has them.
        assert!(
            text.contains("ITEM: BOX BOUNDS xy xz yz pp ff ff"),
            "{text}"
        );
        let rows: Vec<Vec<f64>> = text
            .lines()
            .skip_while(|l| !l.starts_with("ITEM: BOX"))
            .skip(1)
            .take(3)
            .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
            .collect();
        let want = [[-1.0, 12.5, -2.0], [-3.0, 7.0, 1.5], [0.5, 8.5, -1.0]];
        for (r, row) in want.iter().enumerate() {
            for (c, w) in row.iter().enumerate() {
                assert!(
                    (rows[r][c] - w).abs() < 1e-9,
                    "row {r} col {c}: {}",
                    rows[r][c]
                );
            }
        }
        let dir = std::env::temp_dir().join("rmda_test_tilted_roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let back = crate::io::read_frames(&write_dump(&dir, &text))
            .unwrap()
            .remove(0);
        assert_eq!(back.cell.h_matrix().m, f.cell.h_matrix().m);
        assert_eq!(back.cell.origin, f.cell.origin);
    }

    #[test]
    fn scaled_and_scaled_unwrapped_columns_read_as_cartesian() {
        let f = crate::io::read_frames(&tilted_fixture()).unwrap().remove(0);
        let text = std::fs::read_to_string(tilted_fixture()).unwrap();
        let header: String = text
            .lines()
            .take_while(|l| !l.starts_with("ITEM: ATOMS"))
            .map(|l| format!("{l}\n"))
            .collect();
        let dir = std::env::temp_dir().join("rmda_test_scaled_dump");
        std::fs::create_dir_all(&dir).unwrap();
        // `xsu` carries an image shift of +1/-1 that must wrap back into the box;
        // the open axes (y, z here) keep theirs, so only x is shifted.
        for (cols, shift) in [("xs ys zs", 0.0), ("xsu ysu zsu", 1.0)] {
            let mut body = format!("{header}ITEM: ATOMS id type {cols}\n");
            for (i, p) in f.particles.positions.iter().enumerate() {
                let s = f.cell.x2lamda(*p).unwrap();
                let sx = if i % 2 == 0 {
                    s[0] + shift
                } else {
                    s[0] - shift
                };
                body.push_str(&format!("{} 1 {sx:?} {:?} {:?}\n", i + 1, s[1], s[2]));
            }
            let back = crate::io::read_frames(&write_dump(&dir, &body))
                .unwrap()
                .remove(0);
            for (a, b) in back.particles.positions.iter().zip(&f.particles.positions) {
                for d in 0..3 {
                    assert!((a[d] - b[d]).abs() < 1e-9, "{cols}: {a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn boundary_flags_are_periodic_only_for_pp() {
        let h = |s: &str| format!("ITEM: BOX BOUNDS {s}");
        assert_eq!(parse_pbc(h("pp pp pp").as_bytes()).unwrap(), [true; 3]);
        assert_eq!(
            parse_pbc(h("pp pp ss").as_bytes()).unwrap(),
            [true, true, false]
        );
        assert_eq!(
            parse_pbc(h("mm fs pp").as_bytes()).unwrap(),
            [false, false, true]
        );
        assert_eq!(
            parse_pbc(h("xy xz yz pp ff mm").as_bytes()).unwrap(),
            [true, false, false]
        );
        assert_eq!(parse_pbc(b"ITEM: BOX BOUNDS").unwrap(), [true; 3]);
        assert!(parse_pbc(h("pp pf pp").as_bytes()).is_err());
        assert!(parse_pbc(h("pp pp").as_bytes()).is_err());
    }

    #[test]
    fn general_triclinic_dump_restricted_and_positions_rotated() {
        let dir = std::env::temp_dir().join("rmda_test_gen_triclinic");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_dump(&dir, GEN);
        let f = crate::io::read_frames(&p).unwrap().remove(0);

        // restricted form: upper triangle must vanish
        let m = f.cell.h_matrix();
        assert!(m.m[0][1].abs() < 1e-12 && m.m[0][2].abs() < 1e-12);
        assert!(m.m[1][2].abs() < 1e-12);
        // rotation preserves the cell metric
        let expect: [[f64; 3]; 3] = [[4.0, 1.2, 0.3], [0.0, 4.2, 0.5], [0.0, 0.0, 5.0]];
        for (k, _) in expect.iter().enumerate() {
            let got = (m.m[k][0].powi(2) + m.m[k][1].powi(2) + m.m[k][2].powi(2)).sqrt();
            let want = (expect[k][0].powi(2) + expect[k][1].powi(2) + expect[k][2].powi(2)).sqrt();
            assert!((got - want).abs() < 1e-9, "length {k}: {got} vs {want}");
        }
        // volume is invariant under rotation
        let v = f.cell.volume().abs();
        assert!(
            (v - 4.0 * (4.2 * 5.0 - 0.5 * 0.0)).abs() < 1e-6,
            "volume {v}"
        );

        // atom 0 is the origin corner and must stay there
        assert!(
            f.particles.positions[0].iter().all(|c| c.abs() < 1e-9),
            "origin atom moved: {:?}",
            f.particles.positions[0]
        );
        // atom 1 at general (1,1,1) must have been rotated, i.e. differ from
        // the raw value, while keeping its distance to atom 0
        let q = f.particles.positions[1];
        let moved =
            (q[0] - 1.0).abs() > 1e-6 || (q[1] - 1.0).abs() > 1e-6 || (q[2] - 1.0).abs() > 1e-6;
        assert!(moved, "atom 1 was not rotated: {q:?}");
        let d = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        assert!((d - 3f64.sqrt()).abs() < 1e-9, "distance changed: {d}");
    }

    #[test]
    fn general_triclinic_data_positions_rotated() {
        let dir = std::env::temp_dir().join("rmda_test_gen_data");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("gen.data");
        std::fs::write(
            &p,
            concat!(
                "LAMMPS general triclinic\n",
                "\n2 atoms\n1 atom types\n",
                "0.0 4.0 xlo xhi\n0.0 4.2 ylo yhi\n0.0 5.0 zlo zhi\n",
                "avec 4.0 1.2 0.3\nbvec 0.0 4.2 0.5\ncvec 0.0 0.0 5.0\n",
                "Masses\n\n1 1.0\n\n",
                "Atoms # atomic\n\n1 1 0.0 0.0 0.0\n2 1 1.0 1.0 1.0\n\n",
            ),
        )
        .unwrap();
        let f = read_data_file(&p).unwrap().remove(0);
        assert!(f.cell.h_matrix().m[1][2].abs() < 1e-12, "not restricted");
        assert!(f.particles.positions[0].iter().all(|c| c.abs() < 1e-9));
        let q = f.particles.positions[1];
        let d = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        assert!((d - 3f64.sqrt()).abs() < 1e-9, "distance changed: {d}");
    }

    #[test]
    fn parses_ortho_dump() {
        let dir = std::env::temp_dir().join("rmda_test_dump");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_dump(&dir, D);
        let frames = DumpReader::read_all(&p).unwrap();
        assert_eq!(frames.len(), 1);
        let f = &frames[0];
        assert_eq!(f.meta.timestep, 3);
        assert_eq!(f.natoms(), 2);
        assert_eq!(f.particles.positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(f.particles.ids[0], 1);
    }

    /// A data file whose first section is `Atoms` (no `Masses` block, which is
    /// how ASE and several other writers emit them) must still be parsed.
    ///
    /// A header pre-pass that consumed the `Atoms` keyword itself would start
    /// the section loop on the first atom row, never match its `Atoms` branch,
    /// and return a frame with the header's atom count but *no* positions —
    /// silent data corruption that only surfaces as an out-of-bounds panic
    /// once an analysis indexes the positions.
    #[test]
    fn data_file_without_masses_section_parses_atoms() {
        let src = b"(written by ASE)\n\
                    \n\
                    2 atoms\n\
                    1 atom types\n\
                    \n\
                    0.0 4.0 xlo xhi\n\
                    0.0 4.0 ylo yhi\n\
                    0.0 4.0 zlo xhi\n\
                    \n\
                    Atoms # atomic\n\
                    \n\
                    1 1 0.0 0.0 0.0\n\
                    2 1 2.0 2.0 2.0\n";
        let f = read_data_bytes_for_test("nomass.data", src);
        assert_eq!(f.natoms(), 2);
        assert_eq!(
            f.particles.positions.len(),
            2,
            "positions must be populated"
        );
        assert_eq!(f.particles.positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(f.particles.positions[1], [2.0, 2.0, 2.0]);
        assert_eq!(f.particles.ids, vec![1, 2]);
    }

    /// Same file with a `Masses` section ahead of `Atoms`: both shapes must
    /// parse.
    #[test]
    fn data_file_with_masses_section_still_parses_atoms() {
        let src = b"2 atoms\n\
                    1 atom types\n\
                    \n\
                    0.0 4.0 xlo xhi\n\
                    0.0 4.0 ylo yhi\n\
                    0.0 4.0 zlo zhi\n\
                    \n\
                    Masses\n\
                    \n\
                    1 26.98\n\
                    \n\
                    Atoms # atomic\n\
                    \n\
                    1 1 0.0 0.0 0.0\n\
                    2 1 2.0 2.0 2.0\n";
        let f = read_data_bytes_for_test("withmass.data", src);
        assert_eq!(f.natoms(), 2);
        assert_eq!(f.particles.positions.len(), 2);
        assert_eq!(f.particles.positions[1], [2.0, 2.0, 2.0]);
        assert_eq!(f.type_masses, vec![26.98]);
    }

    /// Masses survive read -> write -> read; a frame without them writes no
    /// `Masses` section rather than inventing unit masses.
    #[test]
    fn data_file_masses_round_trip_and_are_never_invented() {
        let src = b"2 atoms\n2 atom types\n\n0 4 xlo xhi\n0 4 ylo yhi\n0 4 zlo zhi\n\n\
                    Masses\n\n2 183.84 # W\n1 95.95\n\n\
                    Atoms # atomic\n\n1 1 0 0 0\n2 2 2 2 2\n";
        let f = read_data_bytes_for_test("masses.data", src);
        assert_eq!(f.type_masses, vec![95.95, 183.84]);
        let mut out = Vec::new();
        super::write_data_file(&mut out, std::slice::from_ref(&f)).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Masses\n\n1 95.95\n2 183.84\n"), "{text}");
        let back = read_data_bytes_for_test("masses2.data", text.as_bytes());
        assert_eq!(back.type_masses, f.type_masses);

        let mut bare = f.clone();
        bare.type_masses.clear();
        let mut out = Vec::new();
        super::write_data_file(&mut out, std::slice::from_ref(&bare)).unwrap();
        assert!(!String::from_utf8(out).unwrap().contains("Masses"));
    }

    /// The dump writer must emit the cell's `pbc` in the `BOX BOUNDS` header,
    /// not a hard-coded `pp pp pp`. A non-periodic dimension written as `pp`
    /// makes a round trip wrap atoms the frame never intended to wrap.
    #[test]
    fn dump_writer_emits_pbc_flags_and_round_trips() {
        let mut cell = UnitCell::orthorhombic(10.0, 20.0, 30.0);
        cell.set_pbc([true, false, true]);
        let mut parts = ParticleData::with_capacity(1);
        parts.add_atom([1.0, 2.0, 3.0], 7, 1);
        let f = Frame::new(cell, parts);

        let mut buf = Vec::new();
        super::write_dump(&mut buf, &[f]).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(
            s.contains("ITEM: BOX BOUNDS pp ff pp"),
            "expected 'pp ff pp', got:\n{s}"
        );

        let dir = std::env::temp_dir().join("rmda_test_dump_pbc");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.dump");
        std::fs::write(&p, &s).unwrap();
        let back = crate::io::read_frames(&p).unwrap();
        let fb = &back[0];
        assert_eq!(
            fb.cell.pbc,
            [true, false, true],
            "pbc flags lost in round trip"
        );
        assert_eq!(fb.particles.positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(fb.particles.ids[0], 7);
    }

    /// The data writer must carry the cell's origin and tilt through
    /// `to_box_bounds` and the frame's real atom ids. Writing the upper
    /// triangle of the h-matrix (which is zero for a restricted cell) would emit
    /// a box with no shear, and synthesising ids would drop the frame's ids.
    #[test]
    fn data_writer_round_trips_origin_tilt_and_ids() {
        // restricted-triclinic: lower-triangular h with a non-zero tilt
        let mut cell = UnitCell::from_matrix(
            crate::cell::Mat3::from_rows([10.0, 0.0, 0.0], [3.0, 20.0, 0.0], [-1.0, 4.0, 30.0]),
            [true; 3],
            [5.0, -7.0, 2.0],
        );
        cell.set_pbc([true, true, false]);
        let mut parts = ParticleData::with_capacity(2);
        parts.add_atom([6.0, -6.0, 4.0], 42, 2);
        parts.add_atom([7.0, -5.0, 5.0], 99, 1);
        let f = Frame::new(cell, parts);

        let mut buf = Vec::new();
        write_data_file(&mut buf, &[f]).unwrap();
        let s = String::from_utf8(buf).unwrap();

        // xlo = origin.x (5), not 0
        assert!(
            s.contains("5 15 xlo xhi"),
            "origin/extent lost in data header:\n{s}"
        );
        // tilt row: xy = h[1][0] = 3, xz = h[2][0] = -1, yz = h[2][1] = 4
        assert!(
            s.contains("3 -1 4 xy xz yz"),
            "tilt written from the wrong triangle:\n{s}"
        );
        // ids preserved
        assert!(s.contains("42 2 "), "atom id 42 dropped:\n{s}");

        let dir = std::env::temp_dir().join("rmda_test_data_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("rt.data");
        std::fs::write(&p, &s).unwrap();
        let back = read_data_file(&p).unwrap();
        let fb = &back[0];
        assert!((fb.cell.origin[0] - 5.0).abs() < 1e-9, "origin lost");
        let h = fb.cell.h_matrix().m;
        assert!((h[1][0] - 3.0).abs() < 1e-9, "xy tilt lost: {}", h[1][0]);
        assert!((h[2][0] + 1.0).abs() < 1e-9, "xz tilt lost: {}", h[2][0]);
        assert!((h[2][1] - 4.0).abs() < 1e-9, "yz tilt lost: {}", h[2][1]);
        assert_eq!(fb.particles.ids, vec![42, 99], "ids not round-tripped");
        assert_eq!(fb.particles.positions[0], [6.0, -6.0, 4.0]);
    }

    /// A multi-frame write to a single LAMMPS data file must error rather than
    /// silently drop every frame but the first.
    #[test]
    fn data_writer_rejects_multi_frame() {
        let mut c1 = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        c1.set_pbc([true; 3]);
        let mut p1 = ParticleData::with_capacity(1);
        p1.add_atom([0.0, 0.0, 0.0], 1, 1);
        let mut p2 = ParticleData::with_capacity(1);
        p2.add_atom([1.0, 1.0, 1.0], 1, 1);
        let frames = vec![Frame::new(c1, p1), Frame::new(c1, p2)];
        let mut buf = Vec::new();
        let err = write_data_file(&mut buf, &frames).unwrap_err();
        assert!(
            err.to_string().contains("single frame"),
            "unexpected error: {err}"
        );
    }

    /// An unknown atom style must be rejected rather than decoded as `atomic`:
    /// an `ellipsoid` row is `id type ellipsoidflag density x y z`, so the
    /// atomic fallback would read the flag as x — plausible but wrong, silently.
    #[test]
    fn unknown_atom_style_is_rejected() {
        let src = "2 atoms\n\
                   1 atom types\n\
                   \n\
                   0.0 4.0 xlo xhi\n\
                   0.0 4.0 ylo yhi\n\
                   0.0 4.0 zlo zhi\n\
                   \n\
                   Atoms # ellipsoid\n\
                   \n\
                   1 1 1 1.0 0.0 0.0 0.0\n\
                   2 1 1 1.0 2.0 2.0 2.0\n";
        let e = parse_data_err("ellipsoid.data", src.as_bytes());
        assert!(e.contains("unsupported atom style"), "got: {e}");
    }

    /// LAMMPS package suffixes (`atomic/kk`, `full/gpu`, `molecular/omp`)
    /// decorate the style name but keep the base style's column layout, so
    /// `atomic/kk` must parse as `atomic`, not be rejected as unknown.
    #[test]
    fn package_suffixed_style_parses_as_its_base_style() {
        let src = "2 atoms\n\
                   1 atom types\n\
                   \n\
                   0.0 4.0 xlo xhi\n\
                   0.0 4.0 ylo yhi\n\
                   0.0 4.0 zlo zhi\n\
                   \n\
                   Atoms # atomic/kk\n\
                   \n\
                   1 1 0.0 0.0 0.0\n\
                   2 1 2.0 2.0 2.0\n";
        let f = read_data_bytes_for_test("atomic_kk.data", src.as_bytes());
        assert_eq!(f.natoms(), 2);
        assert_eq!(f.particles.positions[1], [2.0, 2.0, 2.0]);
        assert_eq!(f.particles.ids, vec![1, 2]);
    }

    /// Malformed id/type/coordinate fields must be parse errors, not silent
    /// `0`/`1`/`0.0` substitutes that yield valid-looking, wrong results.
    #[test]
    fn malformed_atom_fields_are_rejected() {
        let hdr = "1 atoms\n\
                   1 atom types\n\
                   \n\
                   0.0 4.0 xlo xhi\n\
                   0.0 4.0 ylo yhi\n\
                   0.0 4.0 zlo zhi\n\
                   \n\
                   Atoms # atomic\n\
                   \n";
        let cases = [
            ("abc 1 0.0 0.0 0.0", "bad id"),
            ("1 zz 0.0 0.0 0.0", "bad type"),
            ("1 0 0.0 0.0 0.0", "not a positive integer"),
            ("1 1 0.0 0.0 qq", "bad z"),
        ];
        for (row, want) in cases {
            let src = format!("{hdr}{row}\n");
            let e = parse_data_err("malformed.data", src.as_bytes());
            assert!(e.contains(want), "row {row:?}: want {want:?}, got {e}");
        }
    }

    /// A row type above the header's `atom types` count is rejected (the header
    /// was present, so a bound exists to check against).
    #[test]
    fn atom_type_exceeding_declared_types_is_rejected() {
        let src = "2 atoms\n\
                   1 atom types\n\
                   \n\
                   0.0 4.0 xlo xhi\n\
                   0.0 4.0 ylo yhi\n\
                   0.0 4.0 zlo zhi\n\
                   \n\
                   Atoms # atomic\n\
                   \n\
                   1 1 0.0 0.0 0.0\n\
                   2 5 1.0 1.0 1.0\n";
        let e = parse_data_err("range.data", src.as_bytes());
        assert!(e.contains("exceeds"), "got: {e}");
    }

    fn dump_frame(src: &str) -> Result<Frame> {
        let cur = Cursor::from_bytes(src.as_bytes(), "t.dump");
        read_next_dump_from_cursor(&cur).map(|f| f.expect("one frame"))
    }

    /// Every column that is not id/type/position/image survives as a
    /// property under its own name; integer-valued names get an i64 column,
    /// `element` becomes the type labels.
    #[test]
    fn every_dump_column_is_kept() {
        let f = dump_frame(
            "ITEM: TIMESTEP\n5\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n\
             ITEM: ATOMS element vx id c_pe[1] mol type x y z xu v_t i_flag q\n\
             Cu 0.5 1 -3.25 7 1 1 2 3 11 1e-3 4 0.1\n\
             W -0.5 2 -4.5 8 2 4 5 6 14 2e-3 -5 -0.1\n",
        )
        .unwrap();
        let get = |n: &str| {
            f.particles
                .props
                .get(n)
                .unwrap_or_else(|| panic!("{n} dropped"))
        };
        assert_eq!(get("vx").as_f64().unwrap(), &vec![0.5, -0.5]);
        assert_eq!(get("c_pe[1]").as_f64().unwrap(), &vec![-3.25, -4.5]);
        assert_eq!(get("mol").as_i64().unwrap(), &vec![7, 8]);
        assert_eq!(get("i_flag").as_i64().unwrap(), &vec![4, -5]);
        assert_eq!(get("q").as_f64().unwrap(), &vec![0.1, -0.1]);
        // `x` wins over `xu`, which is kept as data.
        assert_eq!(f.particles.positions[1], [4.0, 5.0, 6.0]);
        assert_eq!(get("xu").as_f64().unwrap(), &vec![11.0, 14.0]);
        assert_eq!(f.type_labels, vec!["Cu".to_string(), "W".to_string()]);
        assert_eq!(f.particles.ids, vec![1, 2]);
        // A bad value in a property column is an error naming the column.
        let e = dump_frame(
            "ITEM: TIMESTEP\n5\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n\
             ITEM: ATOMS id type x y z vx\n1 1 1 2 3 fast\n",
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("column vx"), "{e}");
    }

    /// `UNITS` and `TIME` (dump_modify units/time yes) come before
    /// `TIMESTEP`; any item order up to `ATOMS` is accepted, a repeated or
    /// unknown item is an error, and a missing required one is named.
    #[test]
    fn dump_items_in_any_order() {
        let f = dump_frame(
            "ITEM: UNITS\nmetal\nITEM: TIME\n0.025\nITEM: BOX BOUNDS pp pp ff\n0 10\n0 10\n0 10\n\
             ITEM: NUMBER OF ATOMS\n1\nITEM: TIMESTEP\n25\nITEM: ATOMS id type x y z\n1 1 1 2 3\n",
        )
        .unwrap();
        assert_eq!(f.meta.timestep, 25);
        assert_eq!(f.meta.time, Some(0.025));
        assert_eq!(f.meta.units.as_deref(), Some("metal"));
        assert_eq!(f.cell.pbc, [true, true, false]);
        let bad = [
            ("ITEM: TIMESTEP\n1\nITEM: TIMESTEP\n2\n", "duplicate"),
            ("ITEM: TIMESTEP\n1\nITEM: ENTRIES c_1\n", "unsupported dump item"),
            (
                "ITEM: TIMESTEP\n1\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\nITEM: ATOMS id type x y z\n",
                "NUMBER OF ATOMS",
            ),
        ];
        for (src, want) in bad {
            let e = dump_frame(src).unwrap_err().to_string();
            assert!(e.contains(want), "{want}: {e}");
        }
    }

    /// `ix` without `iy`/`iz` must not index a missing column and panic.
    #[test]
    fn partial_image_flags_are_an_error_not_a_panic() {
        let e = dump_frame(
            "ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n\
             ITEM: ATOMS id type x y z ix iz\n1 1 0 0 0 1 1\n",
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("ix, iy, iz") && e.contains("ix, iz"), "{e}");
    }

    const DATA_HEAD: &str =
        "LAMMPS data file\n\n3 atoms # three of them\n2 atom types # W and Mo\n\n\
                             0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n";

    /// `N atoms # comment`, Atom Type Labels (also used in Masses and
    /// Atoms rows), trailing image flags, and Velocities matched by id even
    /// when their order differs from the Atoms section.
    #[test]
    fn data_labels_images_and_velocities() {
        let src = format!(
            "{DATA_HEAD}Atom Type Labels\n\n1 W\n2 Mo\n\nMasses\n\nW 183.84\nMo 95.95\n\n\
             Atoms # atomic\n\n10 W 1 1 1 0 0 1\n20 Mo 2 2 2 -1 0 0\n30 2 3 3 3 0 2 0 # end\n\n\
             Velocities\n\n30 0.3 0 0\n10 0.1 0 0\n20 0.2 0 -1\n"
        );
        let f = read_data_bytes_for_test("labels.data", src.as_bytes());
        assert_eq!(f.natoms(), 3);
        assert_eq!(f.particles.types, vec![1, 2, 2]);
        assert_eq!(f.type_labels, vec!["W".to_string(), "Mo".to_string()]);
        assert_eq!(f.type_masses, vec![183.84, 95.95]);
        let img = f.particles.images.as_ref().expect("image flags read");
        use crate::cell::{image_ipart, image_jpart, image_kpart};
        assert_eq!(
            (
                image_ipart(img[1]),
                image_jpart(img[1]),
                image_kpart(img[1])
            ),
            (-1, 0, 0)
        );
        assert_eq!(image_jpart(img[2]), 2);
        let vx = f.particles.props.get("vx").unwrap().as_f64().unwrap();
        assert_eq!(vx, &vec![0.1, 0.2, 0.3]);
        let vz = f.particles.props.get("vz").unwrap().as_f64().unwrap();
        assert_eq!(vz, &vec![0.0, -1.0, 0.0]);
        // A velocity for an unknown atom is an error.
        let bad = src.replace("30 0.3 0 0", "31 0.3 0 0");
        let e = parse_data_err("vel.data", bad.as_bytes());
        assert!(e.contains("no matching Atoms row"), "{e}");
    }

    /// Without a style comment the column count decides; an ambiguous
    /// count asks for the style instead of guessing.
    #[test]
    fn data_atom_style_is_inferred_from_the_column_count() {
        let atoms = |rows: &str| format!("{DATA_HEAD}Atoms\n\n{rows}");
        let f = read_data_bytes_for_test(
            "full.data",
            atoms("1 4 1 -0.5 1 1 1\n2 4 2 0.5 2 2 2\n3 5 1 0 3 3 3\n").as_bytes(),
        );
        assert_eq!(f.particles.types, vec![1, 2, 1]);
        assert_eq!(f.particles.positions[1], [2.0, 2.0, 2.0]);
        assert_eq!(
            f.particles.props.get("mol").unwrap().as_i64().unwrap(),
            &vec![4, 4, 5]
        );
        assert_eq!(
            f.particles.props.get("q").unwrap().as_f64().unwrap(),
            &vec![-0.5, 0.5, 0.0]
        );
        let f = read_data_bytes_for_test(
            "img.data",
            atoms("1 1 1 1 1 0 0 0\n2 2 2 2 2 1 1 1\n3 1 3 3 3 0 0 0\n").as_bytes(),
        );
        assert!(f.particles.images.is_some());
        let e = parse_data_err(
            "six.data",
            atoms("1 1 1 1 1 1\n2 1 2 2 2 2\n3 1 3 3 3 3\n").as_bytes(),
        );
        assert!(
            e.contains("cannot infer") && e.contains("Atoms # charge"),
            "{e}"
        );
        // An explicit style with a wrong column count is refused.
        let e = parse_data_err(
            "short.data",
            format!("{DATA_HEAD}Atoms # charge\n\n1 1 1 1 1\n").as_bytes(),
        );
        assert!(e.contains("expected 6"), "{e}");
    }

    /// `--atom-style` (`ReadOptions::atom_style`) names the style of an
    /// ambiguous section; a comment naming another style is refused.
    #[test]
    fn an_atom_style_option_resolves_an_ambiguous_section() {
        let six = format!("{DATA_HEAD}Atoms\n\n1 1 1 1 1 1\n2 2 2 2 2 2\n3 1 2 3 3 3\n");
        let read = |text: &str, style: &str| {
            let opts = ReadOptions::all().with_atom_style(Some(style.to_string()));
            parse_data_frame_with(&Cursor::from_bytes(text.as_bytes(), "s.data"), &opts)
        };
        // charge: id type q x y z
        let f = read(&six, "charge").unwrap();
        assert_eq!(f.particles.types, vec![1, 2, 1]);
        assert_eq!(f.particles.positions[2], [3.0, 3.0, 3.0]);
        assert!(f.particles.props.get("q").is_some());
        // molecular: id mol type x y z
        let f = read(&six, "molecular").unwrap();
        assert_eq!(f.particles.types, vec![1, 2, 2]);
        assert_eq!(f.particles.positions[2], [3.0, 3.0, 3.0]);
        assert_eq!(
            f.particles.props.get("mol").unwrap().as_i64().unwrap(),
            &vec![1, 2, 1]
        );
        // A matching comment is fine (package suffixes aside); another is not.
        let tagged = six.replace("Atoms\n", "Atoms # charge/kk\n");
        assert!(read(&tagged, "charge").is_ok());
        let e = read(&tagged, "molecular").unwrap_err().to_string();
        assert!(e.contains("--atom-style"), "{e}");
        assert!(read(&six, "nonsense").is_err());
    }

    /// A `molecular` row is `id mol type x y z`: the type is the third field.
    #[test]
    fn molecular_style_reads_type_after_mol() {
        let src =
            format!("{DATA_HEAD}Atoms # molecular\n\n1 9 1 0 0 0\n2 9 2 1 1 1\n3 8 1 2 2 2\n");
        let f = read_data_bytes_for_test("mol.data", src.as_bytes());
        assert_eq!(f.particles.types, vec![1, 2, 1]);
        assert_eq!(f.particles.positions[2], [2.0, 2.0, 2.0]);
    }

    /// Dump rows must reject a bad id and a non-positive type instead of
    /// defaulting them to `0`/`1`.
    #[test]
    fn dump_bad_id_and_type_are_rejected() {
        let hdr = "ITEM: TIMESTEP\n\
                   1\n\
                   ITEM: NUMBER OF ATOMS\n\
                   1\n\
                   ITEM: BOX BOUNDS pp pp pp\n\
                   0 10\n\
                   0 10\n\
                   0 10\n\
                   ITEM: ATOMS id type x y z\n";
        let cases = [
            ("ab 1 1 2 3", "bad atom id"),
            ("1 0 1 2 3", "not a positive integer"),
            ("1 zz 1 2 3", "bad atom type"),
        ];
        for (row, want) in cases {
            let dump = format!("{hdr}{row}\n");
            let cur = Cursor::from_bytes(dump.as_bytes(), "bad.dump");
            let e = parse_frame(&cur).unwrap_err().to_string();
            assert!(e.contains(want), "row {row:?}: want {want:?}, got {e}");
        }
    }
}
