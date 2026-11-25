//! End-to-end tests for the `rmda analyze` subcommand.
//!
//! Each analysis is pinned by its own test file; what only this level can
//! check is the one-pass wiring. That includes: analyses running into one
//! store in the right order (`Grain` built on top of PTM's output in the
//! *same* run, not a re-read of a sidecar), `--grains` implying `--ptm` rather
//! than erroring or guessing an orientation, the frame window being honoured,
//! the refusals being refused with a message that says why, and `-o` acting as
//! a per-frame prefix (property `.xyz` sidecars and `g(r)` tables).

use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::io::XyzWriter;
use std::path::{Path, PathBuf};
use std::process::Command;

fn tempdir() -> PathBuf {
    let dir = std::env::temp_dir().join("rmda-cli-analyze-tests");
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
        .find_map(|l| l.trim_start().strip_prefix(&format!("{key}: ")))
        .unwrap_or_else(|| panic!("no {key} in:\n{stdout}"))
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{key} is not a number: {e}"))
}

fn series(stdout: &str, key: &str) -> Vec<f64> {
    stdout
        .lines()
        .find_map(|l| l.trim_start().strip_prefix(&format!("{key}: ")))
        .unwrap_or_else(|| panic!("no {key} in:\n{stdout}"))
        .split_whitespace()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|e| panic!("{key} has a non-number {v:?}: {e}"))
        })
        .collect()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// An axis-aligned FCC lattice of `ncell` conventional cells per side.
///
/// Periodic in all directions, so every atom sees the full 12-neighbour shell
/// and PTM classifies the whole crystal fcc -- which is exactly what lets the
/// analyze tests assert that segmentation found one grain.
fn fcc(name: &str, ncell: usize, a: f64) -> PathBuf {
    let mut pd = ParticleData::with_capacity(4 * ncell.pow(3));
    let mut id = 0i64;
    for i in 0..ncell {
        for j in 0..ncell {
            for k in 0..ncell {
                for (fx, fy, fz) in [
                    (0.0f64, 0.0f64, 0.0f64),
                    (0.5, 0.5, 0.0),
                    (0.5, 0.0, 0.5),
                    (0.0, 0.5, 0.5),
                ] {
                    pd.add_atom(
                        [
                            (i as f64 + fx) * a,
                            (j as f64 + fy) * a,
                            (k as f64 + fz) * a,
                        ],
                        {
                            id += 1;
                            id
                        },
                        1,
                    );
                }
            }
        }
    }
    let frame = Frame::new(
        UnitCell::orthorhombic(ncell as f64 * a, ncell as f64 * a, ncell as f64 * a),
        pd,
    );
    let path = tempdir().join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
}

/// A two-frame window: the same FCC crystal twice.
///
/// The point is the window, not the contents -- analysis must attach each
/// frame's own scalars to that frame's section, without confusing the two.
fn fcc_two_frames(name: &str, ncell: usize, a: f64) -> PathBuf {
    let one = |shift: f64| -> Frame {
        let mut pd = ParticleData::with_capacity(4 * ncell.pow(3));
        let mut id = 0i64;
        for i in 0..ncell {
            for j in 0..ncell {
                for k in 0..ncell {
                    for (fx, fy, fz) in [
                        (0.0f64, 0.0f64, 0.0f64),
                        (0.5, 0.5, 0.0),
                        (0.5, 0.0, 0.5),
                        (0.0, 0.5, 0.5),
                    ] {
                        pd.add_atom(
                            [
                                shift + (i as f64 + fx) * a,
                                (j as f64 + fy) * a,
                                (k as f64 + fz) * a,
                            ],
                            {
                                id += 1;
                                id
                            },
                            1,
                        );
                    }
                }
            }
        }
        Frame::new(
            UnitCell::orthorhombic(ncell as f64 * a, ncell as f64 * a, ncell as f64 * a),
            pd,
        )
    };
    let path = tempdir().join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &one(0.0)).unwrap();
    XyzWriter::write_frame(&mut f, &one(0.25)).unwrap();
    path
}

/// A checkerboard two-species SC lattice: type alternates on parity, so both
/// species have central atoms and the 1-2 partial cross term exists.
fn sc_checkerboard(name: &str, ncell: usize, a: f64) -> PathBuf {
    let mut pd = ParticleData::with_capacity(ncell.pow(3));
    let mut id = 0i64;
    for i in 0..ncell {
        for j in 0..ncell {
            for k in 0..ncell {
                pd.add_atom(
                    [i as f64 * a, j as f64 * a, k as f64 * a],
                    {
                        id += 1;
                        id
                    },
                    if (i + j + k) % 2 == 0 { 1 } else { 2 },
                );
            }
        }
    }
    let frame = Frame::new(
        UnitCell::orthorhombic(ncell as f64 * a, ncell as f64 * a, ncell as f64 * a),
        pd,
    );
    let path = tempdir().join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
}

/// Three-cell, 36-atom FCC crystals used throughout.
const A: f64 = 3.5;

#[test]
fn runs_ptm_rdf_cluster_in_one_pass() {
    let p = fcc("one_pass.xyz", 3, A);
    let (out, err, ok) = rmda(&[
        "analyze",
        p.to_str().unwrap(),
        "--ptm",
        "--rdf",
        "--rdf-rmax",
        "5",
        "--cluster",
        "--cluster-cutoff",
        "2.7",
    ]);
    assert!(ok, "stderr:\n{err}");
    assert!(out.contains("frame 0:"), "stdout:\n{out}");
    // PTM classified every atom fcc...
    assert_eq!(scalar(&out, "ptm_fcc"), 108.0, "stdout:\n{out}");
    // ...and one connected component under the first-shell cutoff.
    assert_eq!(scalar(&out, "cluster_count"), 1.0, "stdout:\n{out}");
    // g(r) is a well-formed series, and the edges come first.
    assert!(series(&out, "rdf_g_r").iter().all(|v| v.is_finite()));
    assert!(out.contains("# r="), "stdout:\n{out}");
}

#[test]
fn grains_is_retired_in_favour_of_segment() {
    // `analyze` has no segmentation; `rmda segment` does it.
    let p = fcc("grains_retired.xyz", 3, A);
    let (_, err, ok) = rmda(&["analyze", p.to_str().unwrap(), "--grains"]);
    assert!(!ok);
    assert!(err.contains("--grains"), "stderr:\n{err}");
}

#[test]
fn the_frame_window_is_honoured() {
    let p = fcc_two_frames("window.xyz", 2, A);
    let (out, _err, ok) = rmda(&["analyze", p.to_str().unwrap(), "--ptm"]);
    assert!(ok);
    assert!(out.contains("frame 0:"), "stdout:\n{out}");
    assert!(out.contains("frame 1:"), "stdout:\n{out}");
    assert_eq!(scalar(&out, "ptm_fcc"), 32.0, "stdout:\n{out}");
}

#[test]
fn wc_runs_alongside_the_rest() {
    let path = sc_checkerboard("wc.xyz", 3, 4.0);

    let (out, err, ok) = rmda(&[
        "analyze",
        path.to_str().unwrap(),
        "--wc",
        "--wc-shells",
        "12",
    ]);
    assert!(ok, "stderr:\n{err}");
    assert!(out.contains("wc_atoms_central"), "stdout:\n{out}");
    let central = scalar(&out, "wc_atoms_central");
    assert_eq!(central, 27.0, "stdout:\n{out}");
    // A cross-species alpha must exist once both species have central atoms.
    assert!(out.contains("wc_m1_1_2"), "stdout:\n{out}");
}

#[test]
fn output_prefix_writes_per_frame_sidecars() {
    let p = fcc("sidecar.xyz", 3, A);
    let prefix = tempdir().join("run");
    let (out, err, ok) = rmda(&[
        "analyze",
        p.to_str().unwrap(),
        "--ptm",
        "--rdf",
        "--rdf-rmax",
        "5",
        "-o",
        prefix.to_str().unwrap(),
    ]);
    assert!(ok, "stderr:\n{err}\nstdout:\n{out}");

    // Per-atom properties for frame 0, as extended XYZ.
    let props = read(&PathBuf::from(format!("{}.0.xyz", prefix.display())));
    assert!(props.contains("StructureType"), "props:\n{props}");
    assert!(props.contains("StructureOrientation"), "props:\n{props}");

    // And the g(r) table for frame 0, edges first.
    let dat = read(&PathBuf::from(format!("{}.rdf.0.dat", prefix.display())));
    assert!(dat.starts_with("# r="), "dat:\n{dat}");
}

#[test]
fn no_analysis_requested_is_refused() {
    let p = fcc("nothing.xyz", 3, A);
    let (out, err, ok) = rmda(&["analyze", p.to_str().unwrap()]);
    assert!(!ok, "stdout:\n{out}");
    assert!(err.contains("at least one analysis"), "stderr:\n{err}");
}

#[test]
fn rdf_without_rmax_is_refused() {
    let p = fcc("norange.xyz", 3, A);
    let (_out, err, ok) = rmda(&["analyze", p.to_str().unwrap(), "--rdf"]);
    assert!(!ok);
    assert!(err.contains("--rdf-rmax"), "stderr:\n{err}");
}

#[test]
fn cluster_without_a_radius_is_refused() {
    let p = fcc("noradius.xyz", 3, A);
    let (_out, err, ok) = rmda(&["analyze", p.to_str().unwrap(), "--cluster"]);
    assert!(!ok);
    assert!(
        err.contains("--cluster-cutoff or --cluster-knear"),
        "stderr:\n{err}"
    );
}
