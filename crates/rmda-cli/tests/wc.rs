//! End-to-end tests for the `rmda wc` subcommand.
//!
//! The library tests in `analysis::warren_cowley` pin the parameters against
//! hand-counted lattices. What only this level can check is the wiring: that
//! `--wc-shells` reaches the analysis as typed, that a `Selection` column written
//! by one run is read back as a mask by the next, that `--only-same-type` and
//! `--per-particle` remove and add work rather than printing zeros, and that the
//! two places `-o` cannot be honoured say so instead of writing nothing.
//!
//! Fixtures are built here rather than stored, so every expectation is a known
//! answer and not a recording of this implementation's own output.

use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::data::property::{DType, PropertyStore};
use rmda_core::io::XyzWriter;
use std::path::PathBuf;
use std::process::Command;

fn tempdir() -> PathBuf {
    let dir = std::env::temp_dir().join("rmda-cli-wc-tests");
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
        .unwrap_or_else(|e| panic!("{key} is not a number: {e:?}"))
}

fn keys(stdout: &str, prefix: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|l| l.split_once(": ").map(|(k, _)| k.to_string()))
        .filter(|k| k.starts_with(prefix))
        .collect()
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() < 1e-9,
        "{what}: got {got}, expected {want}"
    );
}

/// Two well-separated simple-cubic blocks of distinct types, `ncell` cells a side
/// at spacing `a`. Every atom's `k` nearest are its own species, so `alpha_jj = 1
/// - 1/c_j` and `alpha_jk = 1` with no counting left to do.
fn two_blocks(name: &str, ncell: usize, a: f64) -> PathBuf {
    let span = ncell as f64 * a;
    let mut pd = ParticleData::with_capacity(2 * ncell.pow(3));
    let mut id = 0i64;
    for (base, ty) in [(0.0, 1u32), (10.0 * span, 2u32)] {
        for i in 0..ncell {
            for j in 0..ncell {
                for k in 0..ncell {
                    id += 1;
                    pd.add_atom([base + i as f64 * a, j as f64 * a, k as f64 * a], id, ty);
                }
            }
        }
    }
    let frame = Frame::new(UnitCell::orthorhombic(12.0 * span, span, span), pd);
    let path = tempdir().join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
}

/// The same two blocks with a `Selection` column flagging the first `nsel` atoms,
/// which are all type 1 because the blocks are written one after the other.
fn selected_blocks(name: &str, ncell: usize, a: f64, nsel: usize) -> PathBuf {
    let path = two_blocks(name, ncell, a);
    let frames = rmda_core::io::XyzReader::read_all(&path).unwrap();
    let natoms = frames[0].natoms();
    let mut store = PropertyStore::new();
    store.register_flat(
        DType::U8,
        "Selection",
        1,
        (0..natoms).map(|i| f64::from(u8::from(i < nsel))).collect(),
    );
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame_with(&mut f, &frames[0], &store).unwrap();
    path
}

#[test]
fn two_pure_blocks_read_minus_one_on_the_diagonal_and_plus_one_across() {
    let path = two_blocks("pure.extxyz", 8, 3.0);
    let (out, err, ok) = rmda(&["wc", "--wc-shells", "6", path.to_str().unwrap()]);
    assert!(ok, "{out}{err}");
    assert_close(scalar(&out, "wc_conc_1"), 0.5, "c_1");
    assert_close(scalar(&out, "wc_conc_2"), 0.5, "c_2");
    assert_close(scalar(&out, "wc_m1_1_1"), -1.0, "alpha_11");
    assert_close(scalar(&out, "wc_m1_2_2"), -1.0, "alpha_22");
    assert_close(scalar(&out, "wc_m1_1_2"), 1.0, "alpha_12");
    assert_close(scalar(&out, "wc_m1_2_1"), 1.0, "alpha_21");
    assert_close(scalar(&out, "wc_asymmetry_max"), 0.0, "asymmetry");
    assert_close(scalar(&out, "wc_atoms_central"), 1024.0, "central");
    // The type ids are on stderr, because the scalar names carry them and nothing
    // else would say which is which.
    assert!(err.contains("types=1,2"), "{err}");
}

#[test]
fn a_leading_zero_in_the_rank_list_is_optional_and_changes_nothing() {
    let path = two_blocks("zero.extxyz", 6, 3.0);
    let (bare, _, ok1) = rmda(&["wc", "--wc-shells", "6", path.to_str().unwrap()]);
    let (zeroed, _, ok2) = rmda(&["wc", "--wc-shells", "0,6", path.to_str().unwrap()]);
    assert!(ok1 && ok2);
    assert_eq!(
        keys(&bare, "wc_m"),
        keys(&zeroed, "wc_m"),
        "the matrix must not depend on the spelling"
    );
    assert_close(
        scalar(&bare, "wc_m1_nneigh"),
        scalar(&zeroed, "wc_m1_nneigh"),
        "shell width",
    );
}

#[test]
fn a_shell_width_is_the_difference_of_ranks_not_the_rank() {
    // Two shells of six rather than one shell of twelve. Reading `nneigh` as
    // widths would put the same six neighbours in both and report 12 for each.
    let path = two_blocks("shells.extxyz", 6, 3.0);
    let (out, err, ok) = rmda(&["wc", "--wc-shells", "6,12", path.to_str().unwrap()]);
    assert!(ok, "{out}{err}");
    assert_close(scalar(&out, "wc_shells"), 2.0, "shell count");
    assert_close(scalar(&out, "wc_m1_nneigh"), 6.0, "shell 1");
    assert_close(scalar(&out, "wc_m2_nneigh"), 6.0, "shell 2");
}

#[test]
fn a_bad_rank_list_is_refused_with_the_reason() {
    let path = two_blocks("bad.extxyz", 4, 3.0);
    for (list, needle) in [
        ("18,12", "strictly increasing"),
        ("6,0,12", "positive"),
        ("6,6", "strictly increasing"),
        ("abc", "whole number"),
        ("", "at least one shell end"),
    ] {
        let (_, err, ok) = rmda(&["wc", "--wc-shells", list, path.to_str().unwrap()]);
        assert!(!ok, "--wc-shells {list:?} should have been refused");
        assert!(
            err.contains(needle),
            "--wc-shells {list:?} should mention {needle:?}, said:\n{err}"
        );
    }
}

#[test]
fn only_selected_reads_the_mask_from_the_file() {
    // 256 of the 1024 atoms flagged. The selection's halo contains no type 2 at
    // all, so `c_2 = 0` and type 2 has neither a concentration to divide by nor a
    // central atom to average: every one of its entries is `NaN`, and the run says
    // so on stderr rather than leaving the reader to guess.
    let path = selected_blocks("selected.extxyz", 8, 3.0, 256);
    let (out, err, ok) = rmda(&[
        "wc",
        "--wc-shells",
        "6",
        "--only-selected",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    assert_close(scalar(&out, "wc_atoms_central"), 256.0, "central");
    assert_close(scalar(&out, "wc_conc_1"), 1.0, "c_1");
    assert_close(scalar(&out, "wc_conc_2"), 0.0, "c_2");
    assert_close(scalar(&out, "wc_m1_1_1"), 0.0, "alpha_11");
    assert!(scalar(&out, "wc_m1_2_2").is_nan(), "alpha_22");
    assert!(
        err.contains("wc_m1_2_2") && err.contains("no central atom"),
        "an unmeasured type must be announced:\n{err}"
    );
    // Without the flag the same file measures the whole box.
    let (all, _, ok) = rmda(&["wc", "--wc-shells", "6", path.to_str().unwrap()]);
    assert!(ok);
    assert_close(
        scalar(&all, "wc_atoms_central"),
        1024.0,
        "central unselected",
    );
    assert_close(scalar(&all, "wc_conc_2"), 0.5, "c_2 unselected");
}

#[test]
fn only_selected_without_a_selection_column_is_refused() {
    // Analysing everything would report a whole-box average under a flag that says
    // otherwise, and the numbers would look entirely normal.
    let path = two_blocks("nomask.extxyz", 6, 3.0);
    let (_, err, ok) = rmda(&["wc", "--only-selected", path.to_str().unwrap()]);
    assert!(!ok);
    assert!(err.contains("Selection"), "{err}");
}

#[test]
fn only_same_type_removes_the_off_diagonal_and_with_it_the_asymmetry_check() {
    let path = two_blocks("same.extxyz", 6, 3.0);
    let (out, err, ok) = rmda(&[
        "wc",
        "--wc-shells",
        "6",
        "--only-same-type",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    assert_eq!(
        keys(&out, "wc_m"),
        vec!["wc_m1_nneigh", "wc_m1_1", "wc_m1_2"]
    );
    // Nothing is left to compare, so reporting a zero would read like a symmetry
    // check that passed.
    assert!(
        !out.contains("wc_asymmetry_max"),
        "no asymmetry scalar should be printed:\n{out}"
    );
}

#[test]
fn the_per_particle_matrix_is_one_column_that_stays_with_its_atom() {
    let path = two_blocks("pp.extxyz", 4, 3.0);
    let out_path = tempdir().join("pp-out.extxyz");
    let (out, err, ok) = rmda(&[
        "wc",
        "--wc-shells",
        "6",
        "--per-particle",
        "-o",
        out_path.to_str().unwrap(),
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");

    let text = std::fs::read_to_string(&out_path).unwrap();
    assert!(
        text.contains(":WarrenCowleyShell1:R:4"),
        "the per-particle column is missing:\n{text}"
    );
    let rows: Vec<&str> = text
        .lines()
        .skip(2)
        .filter(|l| l.starts_with('X'))
        .collect();
    assert_eq!(rows.len(), 128);
    // Atom 0 is type 1: its own row is (-1, +1) and the other row is NaN, because a
    // per-atom heat map is this atom's parameters, not a copy of the whole matrix.
    // Reading atom 1 instead would put the type-2 row first and the check below
    // would still pass, so the row count and the header order are both asserted.
    // `X x y z type v1..v4`: five fields precede the matrix.
    let vals: Vec<f64> = rows[0]
        .split_whitespace()
        .skip(5)
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(vals.len(), 4, "four components for two types");
    assert_close(vals[0], -1.0, "alpha_11");
    assert_close(vals[1], 1.0, "alpha_12");
    assert!(
        vals[2].is_nan() && vals[3].is_nan(),
        "row 2 is not atom 0's"
    );

    // Round trip: reading the written file back gives the same global numbers, so
    // the property did not disturb the analysis it came from.
    let (again, _, ok) = rmda(&["wc", "--wc-shells", "6", out_path.to_str().unwrap()]);
    assert!(ok, "{again}");
    assert_close(
        scalar(&again, "wc_m1_1_1"),
        scalar(&out, "wc_m1_1_1"),
        "global matrix after a round trip",
    );
}

#[test]
fn an_output_path_without_per_particle_is_refused_instead_of_writing_nothing() {
    // `-o` is global, so passing it means a file was expected. Writing no file and
    // exiting 0 is how someone concludes the alloy had no order.
    let path = two_blocks("noop.extxyz", 4, 3.0);
    let out_path = tempdir().join("never.extxyz");
    let (out, err, ok) = rmda(&[
        "wc",
        "--wc-shells",
        "6",
        "-o",
        out_path.to_str().unwrap(),
        path.to_str().unwrap(),
    ]);
    assert!(!ok);
    assert!(err.contains("--per-particle"), "{err}");
    assert!(
        scalar(&out, "wc_m1_1_1").is_finite(),
        "the numbers still belong on stdout:\n{out}"
    );
    assert!(!out_path.exists(), "nothing should have been written");
}

#[test]
fn two_runs_of_the_same_frame_agree_to_the_last_digit() {
    let path = two_blocks("repeat.extxyz", 6, 3.0);
    let (a, _, ok1) = rmda(&["wc", "--wc-shells", "6,12", path.to_str().unwrap()]);
    let (b, _, ok2) = rmda(&["wc", "--wc-shells", "6,12", path.to_str().unwrap()]);
    assert!(ok1 && ok2);
    assert_eq!(a, b);
}

#[test]
fn the_thread_count_does_not_move_a_digit() {
    // 1458 atoms: two parallel tasks of the neighbour pass.
    let path = two_blocks("threads.extxyz", 9, 3.0);
    let (a, _, ok1) = rmda(&[
        "-t",
        "1",
        "wc",
        "--wc-shells",
        "6,18",
        path.to_str().unwrap(),
    ]);
    let (b, _, ok2) = rmda(&[
        "-t",
        "4",
        "wc",
        "--wc-shells",
        "6,18",
        path.to_str().unwrap(),
    ]);
    assert!(ok1 && ok2);
    assert_eq!(a, b);
}

#[test]
fn edge_correct_excludes_the_atoms_whose_shells_reach_a_surface() {
    // The blocks sit in vacuum, so their faces are free surfaces. The outer
    // atoms are excluded and reported; the rest still give the exact
    // pure-block answer.
    let path = two_blocks("edge.extxyz", 8, 3.0);
    let (out, err, ok) = rmda(&[
        "wc",
        "--wc-shells",
        "6",
        "--edge-correct",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");
    let surface = scalar(&out, "wc_atoms_surface");
    let measured = scalar(&out, "wc_atoms_measured");
    assert!(surface > 0.0 && measured > 0.0, "{out}");
    assert_eq!(surface + measured, 2.0 * 512.0);
    assert_close(scalar(&out, "wc_m1_1_2"), 1.0, "alpha_12");
    // Without the flag nothing is excluded and neither scalar is printed.
    let (plain, _, ok) = rmda(&["wc", "--wc-shells", "6", path.to_str().unwrap()]);
    assert!(ok);
    assert!(keys(&plain, "wc_atoms_surface").is_empty());
}

#[test]
fn a_missing_file_is_reported_rather_than_producing_an_empty_matrix() {
    let (_, err, ok) = rmda(&["wc", "--wc-shells", "6", "/nonexistent/nope.extxyz"]);
    assert!(!ok);
    assert!(err.contains("nope.extxyz"), "{err}");
}
