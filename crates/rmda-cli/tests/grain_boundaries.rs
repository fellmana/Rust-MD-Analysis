//! `rmda grain-boundaries` end to end.
//!
//! The rule itself is pinned in `rmda-core` (unit tests on chains, integration tests
//! on a real bicrystal). What is checked here is the part that only exists at the
//! command line: that the seam is found on a frame that went through the real PTM
//! and segmentation path rather than a hand-assigned `Grain` column, that the flags
//! survive a round trip to disk, that the per-grain counts reach the CSV table, and
//! that the refusals are refusals rather than warnings.

use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::io::XyzWriter;
use rmda_core::lattice::fcc::Fcc;
use rmda_core::lattice::sites::SiteLattice;
use rmda_core::lattice::Lattice;
use std::path::PathBuf;
use std::process::Command;

const A: f64 = 3.615;

fn tempdir() -> PathBuf {
    let dir = std::env::temp_dir().join("rmda-cli-grain-boundaries-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn rmda(args: &[&str]) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(args)
        .output()
        .expect("failed to run rmda");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

fn scalar(stdout: &str, key: &str) -> f64 {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}: ")))
        .unwrap_or_else(|| panic!("no {key} in:\n{stdout}"))
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{key} is not a number: {e}"))
}

fn about_z(deg: f64) -> [[f64; 3]; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn write_fixture(name: &str, cell: UnitCell, pos: &[[f64; 3]]) -> PathBuf {
    let n = pos.len();
    assert!(n > 100, "fixture has too few atoms to segment: {n}");
    let mut pd = ParticleData::with_capacity(n);
    for (i, p) in pos.iter().enumerate() {
        pd.add_atom(*p, i as i64 + 1, 1);
    }
    let frame = Frame::new(cell, pd);
    let path = tempdir().join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
}

/// One undeformed FCC block filling the box.
fn single_crystal(name: &str) -> PathBuf {
    let len = 5.0 * A;
    let cell = UnitCell::orthorhombic(len, len, len);
    let pos: Vec<[f64; 3]> = SiteLattice::build(&cell, &Fcc { a: A }.primitive(), [0.0; 3])
        .unwrap()
        .iter()
        .collect();
    write_fixture(name, cell, &pos)
}

/// Two FCC blocks meeting across the box's mid-plane, the right one rotated
/// `tilt` degrees about z.
///
/// The blocks touch, so the *only* thing keeping them in separate grains is the
/// orientation test — which is what makes this a test of the seam on a real
/// segmented frame rather than of the fixture geometry.
fn bicrystal(name: &str, tilt: f64) -> PathBuf {
    let len = 5.0 * A;
    let seam = len;
    let cell = UnitCell::orthorhombic(2.0 * len, len, len);
    let sites: Vec<[f64; 3]> = SiteLattice::build(&cell, &Fcc { a: A }.primitive(), [0.0; 3])
        .unwrap()
        .iter()
        .collect();
    let r = about_z(tilt);
    let mut pos: Vec<[f64; 3]> = Vec::new();
    for s in sites {
        if s[0] < seam {
            pos.push(s);
        } else {
            let b = [s[0] - seam, s[1], s[2]];
            let mut p = [0.0f64; 3];
            for d in 0..3 {
                p[d] = r[d][0] * b[0] + r[d][1] * b[1] + r[d][2] * b[2];
            }
            p[0] += seam;
            pos.push(p);
        }
    }
    write_fixture(name, cell, &pos)
}

#[test]
fn a_bicrystal_reports_a_seam_and_a_single_crystal_does_not() {
    let bi = bicrystal("bi30.xyz", 30.0);
    let (out, err, ok) = rmda(&["grain-boundaries", bi.to_str().unwrap()]);
    assert!(ok, "run failed:\n{err}");
    assert!(
        err.contains("w_db=4"),
        "the width in force should be on stderr, got:\n{err}"
    );

    let n_boundary = scalar(&out, "gb_atoms_boundary");
    let assigned = scalar(&out, "gb_atoms_assigned");
    assert_eq!(assigned, scalar(&out, "gb_atoms_interior") + n_boundary);
    // Not 2. An ideal sharp bicrystal is disordered exactly where it is interesting, so
    // PTM cannot classify the seam and segmentation returns the two blocks *plus* the
    // fragments between them. Asserting 2 here would be asserting that PTM never
    // misclassifies, which is a different claim and not true.
    let n_grains = scalar(&out, "gb_grains");
    assert!(
        n_grains >= 2.0,
        "two blocks cannot segment into fewer than two grains, got {n_grains}"
    );
    assert!(
        n_boundary > 0.0 && n_boundary < assigned,
        "a bicrystal has a seam, and it is not the whole sample: \
         {n_boundary} of {assigned}"
    );

    // At least the two blocks own seam; a lone interior fragment is allowed to own
    // none, so the histogram may have a zero row but must not be empty of bulk.
    let owners = out
        .lines()
        .filter_map(|l| l.trim().strip_prefix("boundary_atoms="))
        .filter_map(|l| l.split_whitespace().next())
        .filter_map(|v| v.parse::<u64>().ok())
        .filter(|&c| c > 20)
        .count();
    assert!(
        owners >= 2,
        "both blocks should own a substantial part of the seam, got {owners} \
         grains with >20 boundary atoms:\n{out}"
    );

    let sc = single_crystal("single.xyz");
    let (out2, _, ok2) = rmda(&["grain-boundaries", sc.to_str().unwrap()]);
    assert!(ok2);
    assert_eq!(scalar(&out2, "gb_atoms_boundary"), 0.0);
    assert_eq!(scalar(&out2, "gb_fraction_boundary"), 0.0);
    assert_eq!(scalar(&out2, "gb_slab_thickness"), 0.0);
}

#[test]
fn the_flags_survive_a_write_and_a_reread() {
    let bi = bicrystal("bi30-roundtrip.xyz", 30.0);
    let written = tempdir().join("seam.xyz");
    let (out, err, ok) = rmda(&[
        "grain-boundaries",
        "-o",
        written.to_str().unwrap(),
        bi.to_str().unwrap(),
    ]);
    assert!(ok, "run failed:\n{err}");
    let expected = scalar(&out, "gb_atoms_boundary");
    assert!(expected > 0.0);

    let text = std::fs::read_to_string(&written).unwrap();
    assert!(
        text.contains("GrainBoundary:I:1"),
        "the per-atom flag should be in the written header:\n{}",
        text.lines().next().unwrap_or("")
    );
    let flagged_in_file = text
        .lines()
        .skip(2)
        .filter(|l| l.split_whitespace().last() == Some("1"))
        .count();
    assert_eq!(
        flagged_in_file as f64, expected,
        "the file holds {flagged_in_file} flagged atoms, the summary said {expected}"
    );
}

#[test]
fn the_per_grain_counts_reach_the_csv_table() {
    let bi = bicrystal("bi30-table.xyz", 30.0);
    let table = tempdir().join("grains.csv");
    let (out, err, ok) = rmda(&[
        "grain-boundaries",
        "--grain-table",
        table.to_str().unwrap(),
        bi.to_str().unwrap(),
    ]);
    assert!(ok, "run failed:\n{err}");
    assert!(
        !out.contains("grain_boundary_histogram"),
        "with a table asked for, the histogram would be a second way to read the \
         same per-grain numbers:\n{out}"
    );

    let text = std::fs::read_to_string(&table).unwrap();
    // The table opens with a run-dependent number of `#` comment lines, so the header
    // is found by content rather than by position -- a positional read breaks every
    // time a scalar is added to the preamble.
    let header = text
        .lines()
        .find(|l| l.starts_with("grain,"))
        .unwrap_or_else(|| panic!("no header row in:\n{text}"));
    assert!(
        header.contains("boundary_atoms"),
        "the seam count should be a column in the grain table, got: {header}"
    );
    // One row per grain. The count is not 2: PTM fragments the disordered seam, and
    // the table faithfully reports whatever segmentation produced.
    // Strip the `#` preamble (run parameters) and then the header, so what is left is
    // one line per grain. Filtering only the blanks, or only skipping one line, both
    // silently miscount -- the preamble length varies with the run.
    let mut body: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .collect();
    assert_eq!(body.remove(0), header, "header should follow the preamble");
    let rows = body;
    let n_grains = text
        .lines()
        .find_map(|l| l.strip_prefix("# grains: "))
        .unwrap()
        .trim()
        .parse::<usize>()
        .unwrap();
    assert_eq!(rows.len(), n_grains, "one row per grain:\n{text}");
    assert!(n_grains >= 2, "a bicrystal is at least two grains:\n{text}");
    // The header names are lower-case (`boundary_atoms`), unlike the per-grain
    // property they come from (`GrainBoundaryCount`) -- the table is a CSV meant to be
    // read by a person or a plotting script, not a property dump.
    let idx = header
        .split(',')
        .position(|h| h == "boundary_atoms")
        .unwrap();
    let total: f64 = rows
        .iter()
        .map(|r| r.split(',').nth(idx).unwrap().parse::<f64>().unwrap())
        .sum();
    assert_eq!(total, scalar(&out, "gb_atoms_boundary"));
    assert!(
        rows.iter()
            .filter(|r| r.split(',').nth(idx).unwrap().parse::<f64>().unwrap() > 20.0)
            .count()
            >= 2,
        "both blocks touch the seam:\n{text}"
    );
}

#[test]
fn a_narrower_w_db_reports_a_smaller_seam() {
    let bi = bicrystal("bi30-widths.xyz", 30.0);
    let count = |extra: &[&str]| -> f64 {
        let mut args = vec!["grain-boundaries"];
        args.extend_from_slice(extra);
        args.push(bi.to_str().unwrap());
        let (out, err, ok) = rmda(&args);
        assert!(ok, "{extra:?} failed:\n{err}");
        scalar(&out, "gb_atoms_boundary")
    };
    // 0.2 A is below the closest pair in the frame. That pair is not a bond: the
    // two blocks are rotated against each other and do not register at the seam,
    // so the nearest inter-grain spacing is 0.25 A. With orphan adoption on (the
    // default) those interface atoms carry grain ids and a width that reaches
    // them does report a seam, which is why the below-anything width is 0.2 and
    // not 0.5.
    let below_nn = count(&["--w-db", "0.2"]);
    let mid = count(&["--w-db", "3.0"]);
    let wide = count(&["--w-db", "5.0"]);
    assert_eq!(
        below_nn, 0.0,
        "0.2 A cannot reach across the 0.25 A seam pair"
    );
    assert!(
        mid > below_nn && wide > mid,
        "the seam should grow with the width: {below_nn} / {mid} / {wide}"
    );
    assert!(
        scalar(
            &rmda(&["grain-boundaries", "--w-db", "5.0", bi.to_str().unwrap()]).0,
            "gb_w_db"
        ) == 5.0
    );
}

#[test]
fn the_reported_width_is_echoed_so_a_default_run_is_readable() {
    let bi = bicrystal("bi30-echo.xyz", 30.0);
    let (out, err, ok) = rmda(&["grain-boundaries", bi.to_str().unwrap()]);
    assert!(ok);
    assert_eq!(
        scalar(&out, "gb_w_db"),
        4.0,
        "the default is a number, not a blank"
    );
    assert!(err.contains("w_db=4"));
}

#[test]
fn a_bad_width_is_refused_before_anything_is_computed() {
    let bi = bicrystal("bi30-badwidth.xyz", 30.0);
    // `--w-db=<v>` rather than `--w-db <v>`: clap reads a space-separated `-1` as
    // another flag ("unexpected argument") and never reaches the check, so the test
    // would be asserting about clap rather than about the validation. The `=` form is
    // what a user types for a negative number anyway.
    for bad in ["0", "-1", "nan", "inf"] {
        let flag = format!("--w-db={bad}");
        let (_, err, ok) = rmda(&["grain-boundaries", &flag, bi.to_str().unwrap()]);
        assert!(!ok, "--w-db={bad} should have been refused");
        assert!(
            err.contains("--w-db"),
            "the refusal should name the flag, got:\n{err}"
        );
        assert!(
            !err.contains("gb_atoms_boundary"),
            "nothing should have been computed:\n{err}"
        );
    }
}

#[test]
fn a_single_crystal_writes_no_flags() {
    let sc = single_crystal("single-write.xyz");
    let written = tempdir().join("single-seam.xyz");
    let (out, err, ok) = rmda(&[
        "grain-boundaries",
        "-o",
        written.to_str().unwrap(),
        sc.to_str().unwrap(),
    ]);
    assert!(ok, "run failed:\n{err}");
    assert_eq!(scalar(&out, "gb_atoms_boundary"), 0.0);
    let text = std::fs::read_to_string(&written).unwrap();
    let flagged = text
        .lines()
        .skip(2)
        .filter(|l| l.split_whitespace().last() == Some("1"))
        .count();
    assert_eq!(flagged, 0, "a single crystal has no seam to write");
}

#[test]
fn a_missing_input_is_reported_rather_than_panicking() {
    let (_, err, ok) = rmda(&["grain-boundaries"]);
    assert!(!ok);
    assert!(err.contains("no input given"), "got:\n{err}");
}

#[test]
fn unassigned_is_boundary_is_off_by_default_and_reported_when_on() {
    let bi = bicrystal("bi30-unassigned.xyz", 30.0);
    let (out, _, ok) = rmda(&["grain-boundaries", bi.to_str().unwrap()]);
    assert!(ok);
    assert!(
        !out.contains("gb_unassigned_is_boundary"),
        "the default run should not claim to have set it:\n{out}"
    );
    let (out2, err2, ok2) = rmda(&[
        "grain-boundaries",
        "--unassigned-is-boundary",
        bi.to_str().unwrap(),
    ]);
    assert!(ok2);
    assert!(err2.contains("unassigned_is_boundary=true"));
    assert_eq!(scalar(&out2, "gb_unassigned_is_boundary"), 1.0);
}
