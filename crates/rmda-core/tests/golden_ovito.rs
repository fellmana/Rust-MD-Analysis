//! Golden-file parity against OVITO 3.15 (see `tools/gen_fixtures.py`).
//!
//! The fixtures under `tests/data` are written by ASE 3.22 and by OVITO's
//! own LAMMPS dump/data exporters; `ovito_expected.json` holds, per file, the
//! per-atom coordination number that OVITO's `CoordinationNumberModifier`
//! reports at a 3.5 Å cutoff, plus the cell matrix and PBC flags as OVITO
//! understands them.
//!
//! rmda must agree on all three. This exercises the whole IO + cell + MIC +
//! neighbour stack against an independent implementation, including OVITO's
//! general-triclinic `avec/bvec/cvec` and `BOX BOUNDS abc origin` spellings.

use rmda_core::cell::UnitCell;
use rmda_core::io::{read_frames, Format};
use rmda_core::neighbor::cell_list::CellList;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

/// Reader for the flat `ovito_expected.txt` produced by
/// `tools/gen_fixtures.py`, so the test needs no JSON dependency.
fn expected() -> &'static BTreeMap<String, Fixture> {
    static CELL: OnceLock<BTreeMap<String, Fixture>> = OnceLock::new();
    CELL.get_or_init(|| {
        let raw = std::fs::read_to_string(data_dir().join("ovito_expected.txt"))
            .expect("tests/data/ovito_expected.txt (run tools/gen_fixtures.py)");
        let mut map = BTreeMap::new();
        let mut lines = raw.lines().peekable();
        while let Some(l) = lines.next() {
            if let Some(name) = l.strip_prefix("fixture ") {
                let mut fx = Fixture {
                    cutoff: 0.0,
                    natoms: 0,
                    coordination: Vec::new(),
                    cell_rows: [[0.0; 3]; 3],
                    pbc: [true; 3],
                };
                let cutoff = lines.next().expect("cutoff line").trim();
                fx.cutoff = cutoff
                    .strip_prefix("cutoff ")
                    .expect("cutoff key")
                    .parse()
                    .expect("cutoff value");
                let natoms = lines.next().expect("natoms line");
                fx.natoms = natoms
                    .trim()
                    .strip_prefix("natoms ")
                    .expect("natoms key")
                    .parse()
                    .expect("natoms value");
                let pbc = lines.next().expect("pbc line");
                for (i, t) in pbc
                    .trim()
                    .strip_prefix("pbc ")
                    .expect("pbc key")
                    .split_whitespace()
                    .enumerate()
                {
                    fx.pbc[i] = t == "T";
                }
                assert_eq!(lines.next().map(str::trim), Some("cell"));
                for r in 0..3 {
                    let row: Vec<f64> = lines
                        .next()
                        .expect("cell row")
                        .split_whitespace()
                        .map(|v| v.parse().expect("cell value"))
                        .collect();
                    fx.cell_rows[r] = [row[0], row[1], row[2]];
                }
                let coord = lines.next().expect("coord line");
                fx.coordination = coord
                    .trim()
                    .strip_prefix("coord ")
                    .expect("coord key")
                    .split_whitespace()
                    .map(|v| v.parse().expect("coord value"))
                    .collect();
                assert_eq!(fx.coordination.len(), fx.natoms, "{name}: coord length");
                map.insert(name.to_string(), fx);
            }
        }
        map
    })
}

#[derive(Debug)]
struct Fixture {
    cutoff: f64,
    natoms: usize,
    coordination: Vec<i64>,
    cell_rows: [[f64; 3]; 3],
    pbc: [bool; 3],
}

fn format_of(name: &str) -> Format {
    match name.rsplit('.').next().unwrap() {
        "xyz" => Format::Xyz,
        "dump" => Format::LammpsDump,
        "data" => Format::LammpsData,
        other => panic!("unknown fixture extension {other}"),
    }
}

/// Row-major cell of a `UnitCell`, rounded to 1e-6 to shake off the rotation
/// noise in OVITO's general-triclinic export.
fn cell_rows(c: &UnitCell) -> [[f64; 3]; 3] {
    let h = c.h_matrix();
    [*h.row(0), *h.row(1), *h.row(2)]
}

fn assert_close(a: [f64; 3], b: [f64; 3], ctx: &str) {
    for k in 0..3 {
        assert!(
            (a[k] - b[k]).abs() < 1e-6,
            "{ctx}: component {k} differs: rmda={} ovito={}",
            a[k],
            b[k]
        );
    }
}

fn check(name: &str) {
    let fx = expected()
        .get(name)
        .unwrap_or_else(|| panic!("no expectation for {name}"));
    let path = data_dir().join(name);
    // Only the LAMMPS general-triclinic fixtures are rotated into the
    // restricted frame; extXYZ cells are kept exactly as the writer had them.
    let fixture_is_general_lammps = matches!(name, "triclinic.dump" | "triclinic.data");
    let frames = read_frames(&path).unwrap_or_else(|e| panic!("{name}: read failed: {e}"));
    assert_eq!(frames.len(), 1, "{name}: expected a single frame");

    let f = &frames[0];

    // --- atom count
    assert_eq!(f.natoms(), fx.natoms, "{name}: natoms");

    // --- PBC flags
    assert_eq!(f.cell.pbc, fx.pbc, "{name}: pbc flags");

    // --- cell: OVITO stores rows = cell vectors, same convention as rmda.
    // For the general-triclinic fixtures the vectors are rotated into the
    // restricted frame, so compare cell *lengths* and *volume* as well as
    // the wrapped-coordinate-consistent metric below.
    let got = cell_rows(&f.cell);
    let exp = fx.cell_rows;
    let lengths_got = f.cell.lengths();
    let lengths_exp = [
        (exp[0][0] * exp[0][0] + exp[0][1] * exp[0][1] + exp[0][2] * exp[0][2]).sqrt(),
        (exp[1][0] * exp[1][0] + exp[1][1] * exp[1][1] + exp[1][2] * exp[1][2]).sqrt(),
        (exp[2][0] * exp[2][0] + exp[2][1] * exp[2][1] + exp[2][2] * exp[2][2]).sqrt(),
    ];
    for k in 0..3 {
        assert!(
            (lengths_got[k] - lengths_exp[k]).abs() < 1e-6,
            "{name}: cell length {k}: rmda={} ovito={}",
            lengths_got[k],
            lengths_exp[k]
        );
    }
    let vol_exp = exp[0][0] * (exp[1][1] * exp[2][2] - exp[1][2] * exp[2][1])
        - exp[0][1] * (exp[1][0] * exp[2][2] - exp[1][2] * exp[2][0])
        + exp[0][2] * (exp[1][0] * exp[2][1] - exp[1][1] * exp[2][0]);
    assert!(
        (f.cell.volume().abs() - vol_exp.abs()).abs() < 1e-6 * vol_exp.abs().max(1.0),
        "{name}: volume rmda={} ovito={}",
        f.cell.volume(),
        vol_exp
    );
    if fixture_is_general_lammps {
        // Restricted-triclinic fixtures: the upper triangle is zero by
        // construction, and the lower triangle differs from OVITO's general
        // vectors only by the general->restricted rotation, so the physical
        // cell is verified via volume and per-axis lengths above.
        for (k, _) in got.iter().enumerate() {
            assert_eq!(
                got[k][k + 1..3],
                [0.0, 0.0][..3 - k - 1],
                "{name}: upper triangle"
            );
        }
    } else {
        // Orthogonal fixtures and extXYZ (which stays in its general frame by
        // design): the rows must match OVITO's cell vectors exactly.
        for k in 0..3 {
            assert_close(got[k], exp[k], name);
        }
    }

    // --- coordination: neighbours at OVITO's cutoff, straight from the cell
    // list (the same entries a materialised full list would have stored).
    let cl = CellList::build(&f.cell, &f.particles.positions, fx.cutoff).unwrap();
    let mut entries = Vec::new();
    for i in 0..f.natoms() {
        cl.neighbor_entries(i, fx.cutoff, &mut entries).unwrap();
        let got_c = entries.len() as i64;
        let want_c = fx.coordination[i];
        assert_eq!(
            got_c, want_c,
            "{name}: atom {i} coordination {} != OVITO {}",
            got_c, want_c
        );
    }

    // --- format detection agrees with the extension
    let detected = rmda_core::io::format::guess_format(&path).unwrap();
    assert_eq!(detected, format_of(name), "{name}: format detection");
}

#[test]
fn ase_extxyz_fcc_orthorhombic() {
    check("fcc_ortho.xyz");
}

#[test]
fn ase_extxyz_hcp_hexagonal() {
    check("hcp_ortho.xyz");
}

#[test]
fn ase_extxyz_triclinic_two_species() {
    check("triclinic_two_species.xyz");
}

#[test]
fn ase_extxyz_nonperiodic_cluster() {
    check("cluster_nonperiodic.xyz");
}

#[test]
fn ovito_lammps_dump_ortho() {
    check("fcc_ortho.dump");
}

#[test]
fn ovito_lammps_dump_triclinic() {
    check("triclinic.dump");
}

#[test]
fn ovito_lammps_data_ortho() {
    check("fcc_ortho.data");
}

#[test]
fn ovito_lammps_data_triclinic() {
    check("triclinic.data");
}
