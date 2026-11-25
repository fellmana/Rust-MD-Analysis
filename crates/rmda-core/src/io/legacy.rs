//! A plain serial reader (std `parse`, no parallelism), test-only: the
//! reference the production readers must match bit for bit.
#![allow(clippy::all, dead_code)]

use crate::cell::{pack_image, Coord3, UnitCell};
use crate::data::frame::Frame;
use crate::data::particle::ParticleData;
use crate::data::property::{DType, Property};
use crate::error::{Error, Result};
use crate::io::engine::Cursor;
use std::collections::HashMap;

/// The reference field splitter and number parsers (std `parse`).
struct Fields;
impl Fields {
    fn all(line: &[u8]) -> Vec<&[u8]> {
        line.split(|b: &u8| matches!(b, b' ' | b'\t' | b'\r'))
            .filter(|t| !t.is_empty())
            .collect()
    }
}
fn trim(b: &[u8]) -> &[u8] {
    let mut s = 0;
    let mut e = b.len();
    while s < e && (b[s] as char).is_ascii_whitespace() {
        s += 1;
    }
    while e > s && (b[e - 1] as char).is_ascii_whitespace() {
        e -= 1;
    }
    &b[s..e]
}
pub(super) fn fast_i64(b: &[u8]) -> Option<i64> {
    std::str::from_utf8(trim(b)).ok()?.parse::<i64>().ok()
}
pub(super) fn fast_f64(b: &[u8]) -> Option<f64> {
    std::str::from_utf8(trim(b)).ok()?.parse::<f64>().ok()
}
mod engine {
    pub(super) use super::{fast_f64, fast_i64};
}

pub fn parse_data_frame(cur: &Cursor) -> Result<Frame> {
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
        let peek_text = String::from_utf8_lossy(peeked);
        let peek_t = peek_text.trim();
        if peek_t.starts_with("Masses")
            || peek_t.starts_with("MASSES")
            || peek_t.starts_with("Atoms")
            || peek_t.starts_with("ATOMS")
            || peek_t.starts_with("Velocities")
            || peek_t.starts_with("Bonds")
            || peek_t.starts_with("Angles")
        {
            break; // sections begin; leave the keyword for the loop below
        }
        let line = cur.next_line().expect("peeked line disappeared");
        if line.is_empty() {
            continue;
        }
        let text = String::from_utf8_lossy(line);
        let t = text.trim();
        if t.starts_with('#') {
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
        if t.starts_with("Velocities") || t.starts_with("Bonds") || t.starts_with("Angles") {
            // handled above, before this line was consumed
            continue;
        }
        // tolerate unknown header keywords (Pair Coeffs etc.)
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

    // sections: skip until Atoms; default atom style columns
    let mut atom_columns: Vec<&str>;
    let mut parts: Option<ParticleData> = None;
    let mut masses: Vec<f64> = Vec::new();

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

    while let Some(line) = cur.next_line() {
        if line.is_empty() {
            continue;
        }
        let text = String::from_utf8_lossy(line);
        let t = text.trim();
        if t.starts_with('#') {
            continue;
        }
        let kw = t.split_whitespace().next().unwrap_or("");
        match kw {
            "Masses" | "masses" => {
                // `type mass [# comment]`, one row per type.
                masses = vec![f64::NAN; ntypes];
                for _ in 0..ntypes {
                    let row = next_row(cur).ok_or_else(|| perr!("truncated Masses section"))?;
                    let text = String::from_utf8_lossy(&row);
                    let mut w = text.split_whitespace();
                    let (Some(t), Some(m)) = (w.next(), w.next()) else {
                        return Err(perr!("Masses row needs `type mass`"));
                    };
                    let (Ok(t), Ok(m)) = (t.parse::<usize>(), m.parse::<f64>()) else {
                        return Err(perr!(&format!("bad Masses row '{}'", text.trim())));
                    };
                    if t == 0 || t > ntypes {
                        return Err(perr!(&format!("Masses row for type {t} of {ntypes}")));
                    }
                    masses[t - 1] = m;
                }
            }
            "Atoms" | "atoms" => {
                let words: Vec<&str> = t.split_whitespace().collect();
                // words[1] may be "#" then the style keyword
                let style = words
                    .iter()
                    .position(|&w| w == "#")
                    .and_then(|p| words.get(p + 1))
                    .copied()
                    .unwrap_or("atomic");
                // LAMMPS package suffixes decorate the style name (`atomic/kk`,
                // `full/gpu`, `molecular/omp`); the column layout is that of the
                // base style, so strip the suffix before matching.
                let base = style.split('/').next().unwrap_or(style);
                // An *unknown* style is rejected rather than decoded as `atomic`:
                // `molecular` is `id mol type x y z`, so the fallback would read
                // `mol` as the type and `type` as x — plausible, wrong, and
                // silent. Absent style still defaults to `atomic` (the common
                // bare `Atoms` line).
                atom_columns = match base {
                    "atomic" => vec!["id", "type", "x", "y", "z"],
                    "charge" => vec!["id", "type", "q", "x", "y", "z"],
                    "full" => vec!["id", "mol", "type", "q", "x", "y", "z"],
                    _ => {
                        return Err(perr!(format!(
                            "unsupported atom style '{style}' (supported: atomic, charge, full)"
                        )));
                    }
                };
                let ncol = atom_columns.len();
                let mut p = ParticleData::with_capacity(natoms);
                for _ in 0..natoms {
                    let row = next_row(cur).ok_or_else(|| perr!("truncated Atoms section"))?;
                    // a section keyword here means the Atoms block ended early
                    if is_section_keyword(&row) {
                        return Err(perr!("Atoms section has fewer rows than the header count"));
                    }
                    let f = Fields::all(&row);
                    if f.len() < ncol {
                        return Err(perr!(format!(
                            "atom row has {} cols, expected {ncol}",
                            f.len()
                        )));
                    }
                    // Strictly decode the required fields: a malformed id/type/
                    // coordinate is a parse error, not a silent `0`/`1`/`0.0`.
                    let mut x = 0.0;
                    let mut y = 0.0;
                    let mut z = 0.0;
                    let mut id: Option<i64> = None;
                    let mut ty: Option<u32> = None;
                    for (ci, &name) in atom_columns.iter().enumerate() {
                        match name {
                            "id" => {
                                id = Some(fast_i64(f[ci]).ok_or_else(|| {
                                    perr!(format!(
                                        "atom row: bad id '{}'",
                                        String::from_utf8_lossy(f[ci])
                                    ))
                                })?);
                            }
                            "type" => {
                                let t = fast_i64(f[ci]).ok_or_else(|| {
                                    perr!(format!(
                                        "atom row: bad type '{}'",
                                        String::from_utf8_lossy(f[ci])
                                    ))
                                })?;
                                if t < 1 {
                                    return Err(perr!(format!(
                                        "atom row: type {t} is not a positive integer"
                                    )));
                                }
                                ty = Some(t as u32);
                            }
                            "x" => {
                                x = fast_f64(f[ci]).ok_or_else(|| {
                                    perr!(format!(
                                        "atom row: bad x '{}'",
                                        String::from_utf8_lossy(f[ci])
                                    ))
                                })?;
                            }
                            "y" => {
                                y = fast_f64(f[ci]).ok_or_else(|| {
                                    perr!(format!(
                                        "atom row: bad y '{}'",
                                        String::from_utf8_lossy(f[ci])
                                    ))
                                })?;
                            }
                            "z" => {
                                z = fast_f64(f[ci]).ok_or_else(|| {
                                    perr!(format!(
                                        "atom row: bad z '{}'",
                                        String::from_utf8_lossy(f[ci])
                                    ))
                                })?;
                            }
                            _ => {}
                        }
                    }
                    let id = id.ok_or_else(|| perr!("atom row missing id column"))?;
                    let ty = ty.ok_or_else(|| perr!("atom row missing type column"))?;
                    // Range-check against the header's `atom types` when it was
                    // present; without it there is no bound to check against.
                    if let Some(decl) = ntypes_declared {
                        if (ty as usize) > decl {
                            return Err(perr!(format!(
                                "atom row: type {ty} exceeds the declared {decl} atom types"
                            )));
                        }
                    }
                    // general-triclinic files store coordinates in the
                    // unrotated frame: map them with the same rotation that
                    // produced the restricted cell (general_to_restricted_coords)
                    let pos = match &g2r {
                        Some(rot) => {
                            crate::cell::general_to_restricted_point(rot, cell.origin, [x, y, z])
                        }
                        None => [x, y, z],
                    };
                    p.add_atom(pos, id, ty);
                }
                parts = Some(p);
                // done; remaining sections (Velocities, charge, etc.) skipped on
                // the next loop iterations harmlessly.
            }
            "Velocities" | "velocities" => {
                // skip (we don't store velocities in this data model)
                for _ in 0..natoms {
                    cur.next_line();
                }
            }
            _ => {
                // unknown section keyword: consume until a blank line or section header
                // to be safe with Bonds/Angles etc. we just continue; content lines
                // usually have numbers and are tolerated (ignored).
            }
        }
    }

    let mut frame = Frame::new(cell, parts.unwrap_or_else(|| ParticleData::new(natoms)));
    frame.type_masses = masses;
    Ok(frame)
}

/// Is this line the start of another data-file section rather than a row?
fn is_section_keyword(line: &[u8]) -> bool {
    let text = String::from_utf8_lossy(line);
    let t = text.trim();
    const SECTIONS: [&str; 10] = [
        "Atoms",
        "Velocities",
        "Masses",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Impropers",
        "Pair Coeffs",
        "Bond Coeffs",
        "Ellipsoids",
    ];
    match t.split_whitespace().next() {
        Some(kw) => SECTIONS.iter().any(|s| kw.eq_ignore_ascii_case(s)),
        None => false,
    }
}

fn parse_error(cur: &Cursor, msg: &str) -> Error {
    Error::at_line(&cur.path, cur.line_no(), msg)
}

pub fn parse_frame(cur: &Cursor) -> Result<Frame> {
    cur.expect_line(b"ITEM: TIMESTEP")?;
    let timestep = cur
        .next_line()
        .and_then(engine::fast_i64)
        .ok_or_else(|| parse_error(cur, "bad TIMESTEP value"))?;

    cur.expect_line(b"ITEM: NUMBER OF ATOMS")?;
    let natoms_i64 = cur
        .next_line()
        .and_then(engine::fast_i64)
        .ok_or_else(|| parse_error(cur, "bad NUMBER OF ATOMS"))?;
    if natoms_i64 < 0 {
        return Err(parse_error(cur, "negative NUMBER OF ATOMS"));
    }
    let natoms = natoms_i64 as usize;

    let (cell, g2r) = parse_box(cur)?;

    let header = cur.expect_line(b"ITEM: ATOMS")?;
    // split "ITEM: ATOMS" + column names
    let cols = split_columns(header)?;
    let ncols = cols.len();

    // Resolved once per frame rather than looked up per row.
    let pos_cols = PosColumns::resolve(&cols)?;
    let (id_col, type_col) = (cols["id"], cols["type"]);

    let mut parts = ParticleData::with_capacity(natoms);
    let mut positions = Vec::with_capacity(natoms);
    let mut image_flags: Option<Vec<i64>> = if cols.contains_key("ix") {
        Some(Vec::with_capacity(natoms))
    } else {
        None
    };

    for _ in 0..natoms {
        let line = cur
            .next_line()
            .ok_or_else(|| parse_error(cur, "truncated atom block"))?;
        let f = Fields::all(line);
        if f.len() != ncols {
            return Err(parse_error(cur, "atom row column count mismatch"));
        }
        let id_raw = f[id_col];
        let id = engine::fast_i64(id_raw).ok_or_else(|| {
            parse_error(
                cur,
                &format!("bad atom id '{}'", String::from_utf8_lossy(id_raw)),
            )
        })? as crate::data::particle::MdaTag;
        let ty_raw = f[type_col];
        let ty_i = engine::fast_i64(ty_raw).ok_or_else(|| {
            parse_error(
                cur,
                &format!("bad atom type '{}'", String::from_utf8_lossy(ty_raw)),
            )
        })?;
        if ty_i < 1 {
            return Err(parse_error(
                cur,
                &format!("atom type {ty_i} is not a positive integer"),
            ));
        }
        let ty = ty_i as u32;
        let pos = pos_cols.read(&f, &cell, g2r.as_ref(), cur)?;
        positions.push(pos);
        parts.ids.push(id);
        parts.types.push(ty);
        if let Some(img) = image_flags.as_mut() {
            let img_raw = |name: &str| -> Result<i64> {
                let bytes = f[cols[name]];
                engine::fast_i64(bytes).ok_or_else(|| {
                    parse_error(
                        cur,
                        &format!(
                            "bad image flag '{name}' '{}'",
                            String::from_utf8_lossy(bytes)
                        ),
                    )
                })
            };
            let ix = img_raw("ix")?;
            let iy = img_raw("iy")?;
            let iz = img_raw("iz")?;
            img.push(pack_image(ix, iy, iz));
        }
    }

    // store wrapped positions
    parts.positions = positions;
    parts.natoms = parts.ids.len();
    parts.images = image_flags;

    let mut frame = Frame::new(cell, parts);
    frame.meta.timestep = timestep;
    Ok(frame)
}

/// Which position columns a dump frame carries, resolved once per header.
///
/// LAMMPS writes one of `x y z` (wrapped), `xu yu zu` (unwrapped), `xs ys zs`
/// (scaled, the default of `dump atom`) or `xsu ysu zsu` (scaled unwrapped);
/// the first present in that order wins, as in LAMMPS `reader_native.cpp`.
struct PosColumns {
    kind: PosKind,
    col: [usize; 3],
}

#[derive(Clone, Copy, PartialEq)]
enum PosKind {
    Wrapped,
    Unwrapped,
    Scaled,
    ScaledUnwrapped,
}

impl PosColumns {
    fn resolve(cols: &HashMap<&str, usize>) -> Result<Self> {
        for (kind, names) in [
            (PosKind::Wrapped, ["x", "y", "z"]),
            (PosKind::Unwrapped, ["xu", "yu", "zu"]),
            (PosKind::Scaled, ["xs", "ys", "zs"]),
            (PosKind::ScaledUnwrapped, ["xsu", "ysu", "zsu"]),
        ] {
            if let (Some(&a), Some(&b), Some(&c)) =
                (cols.get(names[0]), cols.get(names[1]), cols.get(names[2]))
            {
                return Ok(Self {
                    kind,
                    col: [a, b, c],
                });
            }
        }
        Err(Error::Unsupported(
            "dump atom block has no complete x/y/z, xu/yu/zu, xs/ys/zs or xsu/ysu/zsu columns"
                .into(),
        ))
    }

    /// The atom's position in the (restricted) cell frame.
    fn read(
        &self,
        f: &[&[u8]],
        cell: &UnitCell,
        g2r: Option<&crate::cell::Mat3>,
        cur: &Cursor,
    ) -> Result<Coord3> {
        let mut v = [0.0f64; 3];
        for (d, &c) in self.col.iter().enumerate() {
            v[d] = engine::fast_f64(f[c]).ok_or_else(|| {
                parse_error(
                    cur,
                    &format!("bad coordinate '{}'", String::from_utf8_lossy(f[c])),
                )
            })?;
        }
        Ok(match self.kind {
            // Cartesian columns of a general-triclinic dump are in the
            // unrotated frame. LAMMPS wraps `x y z` itself.
            PosKind::Wrapped | PosKind::Unwrapped => {
                let p = match g2r {
                    Some(rot) => crate::cell::general_to_restricted_point(rot, cell.origin, v),
                    None => v,
                };
                if self.kind == PosKind::Unwrapped {
                    cell.wrap(p)
                } else {
                    p
                }
            }
            // Fractional coordinates do not depend on the frame's rotation.
            PosKind::Scaled => cell.lamda2x(v),
            PosKind::ScaledUnwrapped => cell.wrap(cell.lamda2x(v)),
        })
    }
}

fn split_columns(header: &[u8]) -> Result<HashMap<&str, usize>> {
    let words: Vec<&[u8]> = Fields::all(header);
    let mut m = HashMap::new();
    for (i, w) in words.iter().enumerate() {
        if i == 0 {
            continue; // "ITEM:"
        }
        if i == 1 {
            continue; // "ATOMS"
        }
        let name = std::str::from_utf8(w).map_err(|_| Error::parse("non-utf8 column name"))?;
        m.insert(name, i - 2);
    }
    if !m.contains_key("id") || !m.contains_key("type") {
        return Err(Error::Unsupported(
            "dump atom block must contain id and type columns".into(),
        ));
    }
    Ok(m)
}

/// Parse `ITEM: BOX BOUNDS`, returning the cell and — for a general triclinic
/// box — the rotation that maps atom coordinates into the restricted frame.
fn parse_box(cur: &Cursor) -> Result<(UnitCell, Option<crate::cell::Mat3>)> {
    let line = cur.expect_line(b"ITEM: BOX BOUNDS")?;
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
            slot[k] =
                engine::fast_f64(v).ok_or_else(|| parse_error(cur, "bad box bound number"))?;
        }
    }
    let pbc = parse_pbc(line)?;
    let mut rot: Option<crate::cell::Mat3> = None;
    let cell = if general {
        // `avec[0..2] boxlo[0]`, `bvec[0..2] boxlo[1]`, `cvec[0..2] boxlo[2]`
        // (LAMMPS dump_custom.cpp `header_item_triclinic_general`).
        let (cell, r) = UnitCell::from_general_triclinic_rot(
            [nums[0][0], nums[0][1], nums[0][2]],
            [nums[1][0], nums[1][1], nums[1][2]],
            [nums[2][0], nums[2][1], nums[2][2]],
            [nums[0][3], nums[1][3], nums[2][3]],
        )?;
        rot = Some(r);
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
    Ok((cell.with_pbc(pbc), rot))
}

/// Triclinic dump bounding box to the cell's lo/hi, inverting LAMMPS
/// `Domain::set_global_box` (`boxlo_bound = lo + min(0,xy) + min(0,xz)`, …).
fn bounds_to_lohi(
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
fn lohi_to_bounds(
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

fn xerr(cur: &Cursor, msg: &str) -> Error {
    Error::at_line(&cur.path, cur.line_no(), msg)
}

pub fn parse_xyz_frame(cur: &Cursor) -> Result<Frame> {
    let nline = cur.next_line().ok_or_else(|| xerr(cur, "unexpected EOF"))?;
    let n: usize = std::str::from_utf8(nline)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .ok_or_else(|| xerr(cur, "bad atom count"))?;

    let comment = cur
        .next_line()
        .ok_or_else(|| xerr(cur, "unexpected EOF after count"))?
        .to_vec();
    let meta = crate::io::xyz::parse_comment(&comment)?;

    let layout_simple: Vec<(String, usize)> = if meta.columns.is_empty() {
        vec![("species".into(), 1), ("pos".into(), 3)]
    } else {
        meta.columns
            .iter()
            .map(|(n, _k, c)| (n.clone(), *c))
            .collect()
    };
    let offset = |name: &str| -> Option<usize> {
        let mut at = 0;
        for (n, c) in &layout_simple {
            if n == name {
                return Some(at);
            }
            at += c;
        }
        None
    };
    let species_at = offset("species");
    let pos_at = offset("pos");
    let type_at = offset("type");
    let selection_at = offset("Selection").or_else(|| offset("selection"));
    let expected: usize = layout_simple.iter().map(|(_, c)| c).sum();
    let (Some(pos_at), Some(species_at)) = (pos_at, species_at) else {
        return Err(xerr(cur, "Properties= must name both `species` and `pos`"));
    };

    // Compute extra columns
    let mut extra_cols: Vec<(String, String, usize, usize)> = Vec::new();
    if !meta.columns.is_empty() {
        let mut at = 0;
        for (n, k, c) in &meta.columns {
            let nm = n.as_str();
            if nm == "species" || nm == "pos" || nm == "type" {
                at += *c;
                continue;
            }
            if nm == "Selection" || nm == "selection" {
                at += *c;
                continue;
            }
            extra_cols.push((n.clone(), k.clone(), *c, at));
            at += *c;
        }
    }

    type Row = (
        String,
        f64,
        f64,
        f64,
        Option<u32>,
        Option<u8>,
        Vec<(usize, String, String, usize, Vec<f64>)>, // (index, name, kind, comps, vals)
    );

    let mut rows: Vec<Row> = Vec::with_capacity(n);
    for _ in 0..n {
        let line = cur
            .next_line()
            .ok_or_else(|| xerr(cur, "truncated atom block"))?;
        let f = Fields::all(line);
        if f.len() < expected {
            return Err(xerr(
                cur,
                &format!(
                    "atom row has {} fields, Properties= needs {expected}",
                    f.len()
                ),
            ));
        }
        let element = std::str::from_utf8(f[species_at]).unwrap_or("").to_string();
        let (Some(x), Some(y), Some(z)) = (
            fast_f64(f[pos_at]),
            fast_f64(f[pos_at + 1]),
            fast_f64(f[pos_at + 2]),
        ) else {
            return Err(xerr(cur, "bad coordinates"));
        };
        let ty = match type_at {
            Some(at) => match fast_f64(f[at]) {
                Some(v) if v >= 1.0 && v.fract() == 0.0 => Some(v as u32),
                Some(v) => {
                    return Err(xerr(
                        cur,
                        &format!("type column holds {v}, not a positive integer"),
                    ))
                }
                None => return Err(xerr(cur, "bad type column")),
            },
            None => None,
        };
        let selected = match selection_at {
            Some(at) => match fast_f64(f[at]) {
                Some(v) if v == 0.0 || v == 1.0 => Some(v as u8),
                Some(v) => {
                    return Err(xerr(
                        cur,
                        &format!("Selection column holds {v}, not 0 or 1"),
                    ))
                }
                None => return Err(xerr(cur, "bad Selection column")),
            },
            None => None,
        };
        let mut extra_vals: Vec<(usize, String, String, usize, Vec<f64>)> = Vec::new();
        for (ei, (ecname, eckind, eccount, ecoff)) in extra_cols.iter().enumerate() {
            let mut vals = vec![0f64; *eccount];
            for (c, val) in vals.iter_mut().enumerate() {
                let idx = *ecoff + c;
                if let Some(v) = fast_f64(f[idx]) {
                    *val = v;
                } else {
                    let s = std::str::from_utf8(f[idx]).unwrap_or("");
                    if let Ok(iv) = s.parse::<i64>() {
                        *val = iv as f64;
                    } else if let Ok(fv) = s.parse::<f64>() {
                        *val = fv;
                    } else {
                        *val = f64::NAN;
                    }
                }
            }
            extra_vals.push((ei, ecname.clone(), eckind.clone(), *eccount, vals));
        }
        rows.push((element, x, y, z, ty, selected, extra_vals));
    }

    let labels: Vec<String> = if !meta.species.is_empty() {
        meta.species.clone()
    } else if type_at.is_some() {
        // Each type's own symbol (the rule of the parallel reader).
        let mut tab = Vec::new();
        for (e, _, _, _, ty, ..) in &rows {
            crate::io::xyz::note_type_symbol(&mut tab, ty.unwrap_or(1), e.as_bytes());
        }
        crate::io::xyz::labels_from_type_symbols(&tab)
    } else {
        let mut syms: Vec<String> = rows.iter().map(|(e, ..)| e.clone()).collect();
        syms.sort();
        syms.dedup();
        syms
    };
    let type_map: HashMap<String, u32> = labels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u32 + 1))
        .collect();

    let mut parts = ParticleData::with_capacity(n);
    let mut selection: Option<Vec<u8>> = selection_at.map(|_| Vec::with_capacity(n));
    for (idx, (element, x, y, z, declared, selected, _extra_vals)) in rows.iter().enumerate() {
        let ty = match *declared {
            Some(t) => t,
            None => *type_map
                .get(element)
                .ok_or_else(|| xerr(cur, &format!("element {element} is not a known species")))?,
        };
        parts.add_atom([*x, *y, *z], idx as i64 + 1, ty);
        if let Some(mask) = selection.as_mut() {
            mask.push(selected.unwrap_or(0));
        }
    }

    let (cell, cell_inferred) = match meta.cell {
        Some(c) => (c, false),
        None => (crate::io::xyz::infer_cell(&parts.positions), true),
    };

    let mut frame = Frame::new(cell, parts);
    frame.meta.cell_inferred = cell_inferred;
    frame.meta.timestep = meta.timestep.unwrap_or(0);
    frame.meta.time = meta.time;
    frame.type_labels = labels;
    frame.particles.selection = selection;
    // Add extra properties
    for (ei, (ecname, eckind, eccount, _ecoff)) in extra_cols.iter().enumerate() {
        let dtype = if eckind == "I" {
            DType::I64
        } else {
            DType::F64
        };
        let mut prop = Property::with_capacity(ecname.clone(), dtype, true, *eccount, n);
        for row in rows.iter() {
            // find this extra col
            if let Some(ev) = row.6.iter().find(|e| e.0 == ei) {
                for v in &ev.4 {
                    prop.push(*v);
                }
            } else {
                // fill zeros
                for _ in 0..*eccount {
                    prop.push(0.0);
                }
            }
        }
        frame.particles.props.register(prop);
    }
    Ok(frame)
}

#[cfg(test)]
mod tests {
    //! The parallel readers against the serial reference: positions, ids,
    //! types and cell must be bit-identical.
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// A float in one of the spellings LAMMPS, OVITO and ASE write.
    fn spell(rng: &mut Rng, x: f64) -> String {
        match rng.below(5) {
            0 => format!("{x}"),
            1 => format!("{x:.16e}"),
            2 => format!("{x:.6}"),
            3 => format!("{x:e}"),
            _ => format!("{x:.17}"),
        }
    }

    /// `spell` of a uniform draw from `[off, off + scale)`.
    fn rspell(rng: &mut Rng, scale: f64, off: f64) -> String {
        let x = rng.unit() * scale + off;
        spell(rng, x)
    }

    fn assert_same(new: &Frame, old: &Frame, ctx: &str) {
        let bits = |f: &Frame| -> Vec<u64> {
            f.particles
                .positions
                .iter()
                .flat_map(|p| p.iter().map(|c| c.to_bits()))
                .collect()
        };
        assert_eq!(new.natoms(), old.natoms(), "{ctx}: natoms");
        assert!(bits(new) == bits(old), "{ctx}: positions differ");
        assert_eq!(new.particles.ids, old.particles.ids, "{ctx}: ids");
        assert_eq!(new.particles.types, old.particles.types, "{ctx}: types");
        let h = |f: &Frame| -> Vec<u64> {
            f.cell
                .h_matrix()
                .m
                .iter()
                .flatten()
                .chain(f.cell.origin.iter())
                .map(|c| c.to_bits())
                .collect()
        };
        assert_eq!(h(new), h(old), "{ctx}: cell");
        assert_eq!(new.cell.pbc, old.cell.pbc, "{ctx}: pbc");
        assert_eq!(new.meta.timestep, old.meta.timestep, "{ctx}: timestep");
    }

    fn random_dump(rng: &mut Rng, natoms: usize, frames: usize) -> String {
        let mut s = String::new();
        let pos_sets = [
            ["x", "y", "z"],
            ["xu", "yu", "zu"],
            ["xs", "ys", "zs"],
            ["xsu", "ysu", "zsu"],
        ];
        for f in 0..frames {
            let pos = pos_sets[rng.below(4) as usize];
            let images = rng.below(2) == 0;
            let mut cols: Vec<&str> = vec!["id", "type", pos[0], pos[1], pos[2]];
            if images {
                cols.extend(["ix", "iy", "iz"]);
            }
            // shuffle the column order
            for i in (1..cols.len()).rev() {
                cols.swap(i, rng.below(i as u64 + 1) as usize);
            }
            s.push_str(&format!(
                "ITEM: TIMESTEP\n{}\nITEM: NUMBER OF ATOMS\n{natoms}\n",
                f * 100
            ));
            let flags = ["pp", "ff", "ss", "mm"];
            let fl: Vec<&str> = (0..3).map(|_| flags[rng.below(4) as usize]).collect();
            match rng.below(3) {
                0 => {
                    s.push_str(&format!("ITEM: BOX BOUNDS {} {} {}\n", fl[0], fl[1], fl[2]));
                    for _ in 0..3 {
                        let lo = rng.unit() * 10.0 - 5.0;
                        s.push_str(&format!(
                            "{} {}\n",
                            spell(rng, lo),
                            rspell(rng, 1.0, lo + 20.0)
                        ));
                    }
                }
                1 => {
                    s.push_str(&format!(
                        "ITEM: BOX BOUNDS xy xz yz {} {} {}\n",
                        fl[0], fl[1], fl[2]
                    ));
                    for _ in 0..3 {
                        let lo = rng.unit() * 2.0;
                        let t = rng.unit() * 4.0 - 2.0;
                        s.push_str(&format!(
                            "{} {} {}\n",
                            spell(rng, lo - 2.0),
                            spell(rng, lo + 25.0),
                            spell(rng, t)
                        ));
                    }
                }
                _ => {
                    s.push_str(&format!(
                        "ITEM: BOX BOUNDS abc origin {} {} {}\n",
                        fl[0], fl[1], fl[2]
                    ));
                    s.push_str("20 1.2 0.3 0.5\n0.4 21 0.5 -1\n0.2 0.1 22 0.25\n");
                }
            }
            s.push_str(&format!("ITEM: ATOMS {}\n", cols.join(" ")));
            for i in 0..natoms {
                let row: Vec<String> = cols
                    .iter()
                    .map(|c| match *c {
                        "id" => format!("{}", i * 7 + 1),
                        "type" => format!("{}", 1 + rng.below(5)),
                        "ix" | "iy" | "iz" => format!("{}", rng.below(5) as i64 - 2),
                        "xs" | "ys" | "zs" | "xsu" | "ysu" | "zsu" => rspell(rng, 1.4, -0.2),
                        _ => rspell(rng, 30.0, -5.0),
                    })
                    .collect();
                s.push_str(&row.join(if rng.below(10) == 0 { "  " } else { " " }));
                s.push('\n');
            }
        }
        s
    }

    #[test]
    fn parallel_dump_reader_is_bit_identical_to_the_serial_one() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for (natoms, frames) in [(1, 1), (7, 3), (500, 2), (60_000, 2)] {
            for rep in 0..3 {
                let text = random_dump(&mut rng, natoms, frames);
                let a = Cursor::from_bytes(text.as_bytes(), "new.dump");
                let b = Cursor::from_bytes(text.as_bytes(), "old.dump");
                for k in 0..frames {
                    let new = crate::io::lammps::read_next_dump_from_cursor(&a)
                        .unwrap()
                        .unwrap();
                    let old = parse_frame(&b).unwrap();
                    assert_same(&new, &old, &format!("natoms={natoms} rep={rep} frame={k}"));
                    assert_eq!(new.particles.images, old.particles.images);
                }
            }
        }
    }

    fn random_data(rng: &mut Rng, natoms: usize) -> String {
        let style = ["atomic", "charge", "full", ""][rng.below(4) as usize];
        let mut s = String::from("LAMMPS data file via write_data\n\n");
        s.push_str(&format!("{natoms} atoms\n5 atom types\n\n"));
        for d in ["x", "y", "z"] {
            let lo = rng.unit() * 3.0;
            s.push_str(&format!(
                "{} {} {d}lo {d}hi\n",
                spell(rng, lo),
                spell(rng, lo + 30.0)
            ));
        }
        if rng.below(2) == 0 {
            s.push_str("1.5 -0.5 0.25 xy xz yz\n");
        }
        s.push_str("\nMasses\n\n1 95.95\n2 92.9\n3 180.9\n4 50.9\n5 183.84\n\n");
        let images = style == "atomic" && rng.below(2) == 0 || style.is_empty();
        if style.is_empty() {
            s.push_str("Atoms\n\n");
        } else {
            s.push_str(&format!("Atoms # {style}\n\n"));
        }
        for i in 0..natoms {
            let id = natoms * 3 - i;
            let t = 1 + rng.below(5);
            let xyz = format!(
                "{} {} {}",
                rspell(rng, 30.0, 0.0),
                rspell(rng, 30.0, 0.0),
                rspell(rng, 30.0, 0.0)
            );
            let row = match style {
                "charge" => format!("{id} {t} {} {xyz}", rspell(rng, 1.0, -0.5)),
                "full" => format!("{id} 1 {t} {} {xyz}", rspell(rng, 1.0, -0.5)),
                _ => format!("{id} {t} {xyz}"),
            };
            s.push_str(&row);
            if images {
                s.push_str(" 0 1 -1");
            }
            s.push('\n');
            if i % 997 == 5 {
                s.push_str("\n# a comment inside the section\n");
            }
        }
        s.push_str("\nVelocities\n\n");
        for i in 0..natoms {
            s.push_str(&format!("{} 0.1 -0.2 0.3\n", natoms * 3 - i));
        }
        s
    }

    #[test]
    fn parallel_data_reader_is_bit_identical_to_the_serial_one() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for natoms in [1usize, 9, 2000, 70_000] {
            for rep in 0..4 {
                let text = random_data(&mut rng, natoms);
                let new = crate::io::lammps::parse_data_frame(&Cursor::from_bytes(
                    text.as_bytes(),
                    "n.data",
                ))
                .unwrap();
                let old = parse_data_frame(&Cursor::from_bytes(text.as_bytes(), "o.data")).unwrap();
                assert_same(&new, &old, &format!("natoms={natoms} rep={rep}"));
                assert_eq!(new.type_masses, old.type_masses);
                let vx = new.particles.props.get("vx").expect("velocities read");
                assert_eq!(vx.as_f64().unwrap()[0], 0.1);
            }
        }
    }

    #[test]
    fn parallel_xyz_reader_is_bit_identical_to_the_serial_one() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for n in [1usize, 50, 80_000] {
            for header in [
                "Lattice=\"20 0 0 0 21 0 1 0 22\" pbc=\"T T F\" Properties=species:S:1:pos:R:3",
                "Lattice=\"20 0 0 0 21 0 0 0 22\" Properties=id:I:1:species:S:1:pos:R:3:c:R:2:type:I:1",
                "Properties=species:S:1:pos:R:3:Selection:I:1",
                "",
            ] {
                let mut s = format!("{n}\n{header}\n");
                let el = ["Cu", "Al", "W", "Zr"];
                for i in 0..n {
                    let sp = el[rng.below(4) as usize];
                    let xyz = format!("{} {} {}", rspell(&mut rng, 20.0, 0.0), rspell(&mut rng, 20.0, 0.0), rspell(&mut rng, 20.0, 0.0));
                    let row = if header.contains("id:I") {
                        format!("{} {sp} {xyz} {} 2.5 {}", i + 1, rspell(&mut rng, 1.0, 0.0), 1 + rng.below(3))
                    } else if header.contains("Selection") {
                        format!("{sp} {xyz} {}", rng.below(2))
                    } else {
                        format!("{sp} {xyz}")
                    };
                    s.push_str(&row);
                    s.push('\n');
                }
                let new = crate::io::xyz::read_next_from_cursor(&Cursor::from_bytes(s.as_bytes(), "n.xyz")).unwrap().unwrap();
                let old = parse_xyz_frame(&Cursor::from_bytes(s.as_bytes(), "o.xyz")).unwrap();
                assert_same(&new, &old, &format!("n={n} {header}"));
                assert_eq!(new.type_labels, old.type_labels);
                assert_eq!(new.particles.selection, old.particles.selection);
                for p in old.particles.props.iter() {
                    if p.name == "id" {
                        // The serial reference keeps an `id` column as a property;
                        // the production reader makes it the atom id (design §8.5).
                        // Here the ids are 1..n either way, so `assert_same`
                        // compared them; the property must be gone.
                        assert!(new.particles.props.get("id").is_none());
                        continue;
                    }
                    let q = new.particles.props.get(&p.name).expect("property kept");
                    assert_eq!(q.dtype, p.dtype, "{}", p.name);
                    for k in 0..p.len() * p.components {
                        let (a, b) = (q.get_flat(k).unwrap(), p.get_flat(k).unwrap());
                        assert!(a == b || (a.is_nan() && b.is_nan()), "{} [{k}]", p.name);
                    }
                }
            }
        }
    }

    /// The acceptance check on a real file: set `RMDA_EQUIV_FILE` to a LAMMPS
    /// data or dump file (e.g. the 5.76M-atom thin film) and run with
    /// `--ignored`. Every frame of a dump is compared.
    #[test]
    #[ignore]
    fn real_file_is_bit_identical_to_the_serial_reader() {
        let Ok(path) = std::env::var("RMDA_EQUIV_FILE") else {
            eprintln!("RMDA_EQUIV_FILE not set; nothing to compare");
            return;
        };
        let path = std::path::Path::new(&path);
        let fmt = crate::io::guess_format(path).unwrap();
        let a = Cursor::from_file(path).unwrap();
        let b = Cursor::from_file(path).unwrap();
        match fmt {
            crate::io::Format::LammpsData => {
                let t = std::time::Instant::now();
                let new = crate::io::lammps::parse_data_frame(&a).unwrap();
                let tn = t.elapsed();
                let t = std::time::Instant::now();
                let old = parse_data_frame(&b).unwrap();
                eprintln!("new {tn:?}, old {:?}, {} atoms", t.elapsed(), new.natoms());
                assert_same(&new, &old, "data");
            }
            crate::io::Format::LammpsDump => {
                let mut k = 0;
                while let Some(new) = crate::io::lammps::read_next_dump_from_cursor(&a).unwrap() {
                    let old = parse_frame(&b).unwrap();
                    assert_same(&new, &old, &format!("frame {k}"));
                    while b.peek_line().map_or(false, |l| l.is_empty()) {
                        b.next_line();
                    }
                    k += 1;
                }
                eprintln!("{k} frames identical");
            }
            f => panic!("no serial reference for {f:?}"),
        }
    }
}
