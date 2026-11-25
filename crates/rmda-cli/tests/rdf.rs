//! End-to-end tests for the `rmda rdf` subcommand.
//!
//! The library tests in `analysis::rdf` pin the normalisation against exact
//! hand-counted simple-cubic shells. What only this level can check is the wiring:
//! that `--rmax`/`--dr` reach the analysis as typed, that the refused range is
//! refused with a message that says why, that the bin edges printed above the
//! histograms match the requested range, and that `--no-partials` /
//! `--no-coordination` really remove work rather than printing zeros.
//!
//! Fixtures are built here rather than stored: the assertion is about a *known*
//! answer, and a stored file could only be checked against this implementation's
//! own output, which proves nothing.

use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::io::XyzWriter;
use std::path::PathBuf;
use std::process::Command;

fn tempdir() -> PathBuf {
    let dir = std::env::temp_dir().join("rmda-cli-rdf-tests");
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

fn series(stdout: &str, key: &str) -> Vec<f64> {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}: ")))
        .unwrap_or_else(|| panic!("no {key} in:\n{stdout}"))
        .split_whitespace()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|e| panic!("{key} has a non-number {v:?}: {e}"))
        })
        .collect()
}

fn histogram_names(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|l| l.split_once(": ").map(|(k, _)| k.to_string()))
        .filter(|k| k.starts_with("rdf_") && k != "rdf_rmax" && k != "rdf_dr" && k != "rdf_bins")
        .collect()
}

/// Bin edges printed on the `# r=` line.
fn edges(stdout: &str) -> Vec<f64> {
    let line = stdout
        .lines()
        .find(|l| l.starts_with("# r="))
        .unwrap_or_else(|| panic!("no '# r=' edge line in:\n{stdout}"));
    line.trim_start_matches("# r=")
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

/// Simple cubic lattice of spacing `a`, `ncell` cells per side, one type.
fn sc(name: &str, ncell: usize, a: f64) -> PathBuf {
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
                    1,
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

/// Two well-separated simple-cubic blocks of distinct types.
fn two_types(name: &str, ncell: usize, a: f64) -> PathBuf {
    let span = ncell as f64 * a;
    let mut pd = ParticleData::with_capacity(2 * ncell.pow(3));
    let mut id = 0i64;
    for (base, ty) in [(0.0, 1u32), (10.0 * span, 2u32)] {
        for i in 0..ncell {
            for j in 0..ncell {
                for k in 0..ncell {
                    pd.add_atom(
                        [base + i as f64 * a, j as f64 * a, k as f64 * a],
                        {
                            id += 1;
                            id
                        },
                        ty,
                    );
                }
            }
        }
    }
    let frame = Frame::new(
        UnitCell::orthorhombic(12.0 * span, 12.0 * span, 12.0 * span),
        pd,
    );
    let path = tempdir().join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
}

#[test]
fn the_coordination_number_of_a_uniform_lattice_is_the_exact_shell_sum() {
    // 6 neighbours past the first shell at `a = 3`, and 6 + 12 = 18 past the second
    // at `3*sqrt(2)`. These are exact lattice identities, so this pins the whole
    // chain (half-list counting, the x2 diagonal factor, the shell volumes and the
    // normalisation) at the level of "the answer is a small integer".
    let path = sc("sc15.extxyz", 15, 3.0);
    let (out, err, ok) = rmda(&[
        "rdf",
        "--rmax",
        "6.0",
        "--dr",
        "0.05",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    let cn = series(&out, "rdf_coordination_number");
    let at = |r: f64| ((r / 0.05).round() as usize).min(cn.len() - 1);
    assert!((cn[at(3.5)] - 6.0).abs() < 0.05, "n(3.5)={}", cn[at(3.5)]);
    assert!(
        (cn[at(4.6)] - 18.0).abs() < 0.05,
        "n(4.6)={} (shell 2 adds 12)",
        cn[at(4.6)]
    );
    // The reported scalar agrees with the last bin of the histogram, so a consumer
    // reading either gets the same number.
    let last = *cn.last().unwrap();
    assert!(
        (scalar(&out, "rdf_coordination_at_rmax") - last).abs() < 1e-9,
        "scalar {} vs histogram {last}",
        scalar(&out, "rdf_coordination_at_rmax")
    );
}

#[test]
fn the_pair_counts_are_the_lattice_vector_count() {
    // 15³ atoms, `a = 3`: the lattice vectors strictly inside 9 Å number 92 per
    // atom, so the unordered pair count is `N * 92 / 2`. Hand-countable, and it
    // fails loudly if the pair list is doubled or truncated.
    let path = sc("sc15-r9.extxyz", 15, 3.0);
    let (out, err, ok) = rmda(&[
        "rdf",
        "--rmax",
        "9.0",
        "--dr",
        "0.1",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    let n = scalar(&out, "rdf_atoms");
    let expected = n * 92.0 / 2.0;
    assert_eq!(scalar(&out, "rdf_pairs"), expected);
    let total: f64 = series(&out, "rdf_pair_counts").iter().sum();
    assert_eq!(total, expected, "the histogram must sum to rdf_pairs");
    assert_eq!(scalar(&out, "rdf_pairs_possible"), n * (n - 1.0) / 2.0);
}

#[test]
fn the_g_histogram_sums_to_the_coordination_number() {
    // `n(r)` is defined as the integral of `g(r)`, so it is not an independent
    // quantity: `sum(g_k * vol_k * rho)` must equal the last coordination value.
    // Checking it here means a future change to either formula cannot drift without
    // the two disagreeing in the output a user actually reads.
    let path = sc("sc12-r7.extxyz", 12, 3.0);
    let rmax = 7.0;
    let dr = 0.05;
    let (out, err, ok) = rmda(&[
        "rdf",
        "--rmax",
        &rmax.to_string(),
        "--dr",
        &dr.to_string(),
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    let g = series(&out, "rdf_g_r");
    let cn = series(&out, "rdf_coordination_number");
    let rho = scalar(&out, "rdf_number_density");
    let e = edges(&out);
    let mut acc = 0.0;
    for (k, &v) in g.iter().enumerate() {
        let vol = 4.0 / 3.0 * std::f64::consts::PI * (e[k + 1].powi(3) - e[k].powi(3));
        acc += v * vol * rho;
        assert!(
            (acc - cn[k]).abs() < 1e-6,
            "bin {k}: integral {acc} vs reported n(r) {}",
            cn[k]
        );
    }
}

#[test]
fn the_printed_edges_cover_exactly_the_requested_range() {
    // The edges are printed because a histogram without its abscissa is a list of
    // numbers. They must be `nbins + 1` values from 0 to `rmax` -- and the last
    // edge must be `rmax` even when `rmax` is not a multiple of `dr`, or a consumer
    // would integrate over a range no pair was counted in.
    let path = sc("sc10.extxyz", 10, 3.0);
    for (rmax, dr) in [(6.0, 0.5), (5.3, 0.5), (4.0, 0.25)] {
        let (out, err, ok) = rmda(&[
            "rdf",
            "--rmax",
            &rmax.to_string(),
            "--dr",
            &dr.to_string(),
            path.to_str().unwrap(),
        ]);
        assert!(ok, "{out}{err}");
        let e = edges(&out);
        assert_eq!(e.len(), scalar(&out, "rdf_bins") as usize + 1, "{out}");
        assert_eq!(e[0], 0.0);
        assert!(
            (e[e.len() - 1] - rmax).abs() < 1e-9,
            "last edge {} vs rmax {rmax}",
            e[e.len() - 1]
        );
        assert!(
            e.windows(2).all(|w| w[1] > w[0]),
            "edges must increase: {e:?}"
        );
    }
}

#[test]
fn partials_are_printed_per_type_pair_and_can_be_turned_off() {
    let path = two_types("two-types.extxyz", 8, 3.0);
    let (out, err, ok) = rmda(&[
        "rdf",
        "--rmax",
        "4.0",
        "--dr",
        "0.1",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    let names = histogram_names(&out);
    assert!(names.contains(&"rdf_g_1_1".to_string()), "{names:?}");
    assert!(names.contains(&"rdf_g_2_2".to_string()), "{names:?}");
    // No cross pair is inside 4 A, so no (1,2) partial: an absent partial and a zero
    // one are different claims.
    assert!(!names.contains(&"rdf_g_1_2".to_string()), "{names:?}");

    let (out2, err2, ok2) = rmda(&[
        "rdf",
        "--rmax",
        "4.0",
        "--dr",
        "0.1",
        "--no-partials",
        path.to_str().unwrap(),
    ]);
    assert!(ok2, "{out2}{err2}");
    let names2 = histogram_names(&out2);
    assert!(
        !names2
            .iter()
            .any(|n| n.starts_with("rdf_g_") && n != "rdf_g_r"),
        "{names2:?}"
    );
    // Turning partials off must not change the total histogram.
    assert_eq!(series(&out2, "rdf_g_r"), series(&out, "rdf_g_r"));
}

#[test]
fn coordination_can_be_turned_off_without_changing_g() {
    let path = sc("sc10b.extxyz", 10, 3.0);
    let (out, _, ok) = rmda(&[
        "rdf",
        "--rmax",
        "5.0",
        "--dr",
        "0.2",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}");
    let (out2, _, ok2) = rmda(&[
        "rdf",
        "--rmax",
        "5.0",
        "--dr",
        "0.2",
        "--no-coordination",
        "--no-partials",
        path.to_str().unwrap(),
    ]);
    assert!(ok2, "{out2}");
    assert!(!out2.contains("rdf_coordination_number"), "{out2}");
    assert!(!out2.contains("rdf_coordination_at_rmax"), "{out2}");
    assert_eq!(series(&out2, "rdf_g_r"), series(&out, "rdf_g_r"));
    assert_eq!(scalar(&out2, "rdf_pairs"), scalar(&out, "rdf_pairs"));
}

#[test]
fn an_rmax_beyond_half_the_cell_is_refused_with_a_reason() {
    // The frame is 30 Å per side, so half is 15 A. Asking for more cannot be
    // answered by the minimum-image convention, and the error has to say so rather
    // than printing a histogram that looks fine.
    let path = sc("sc10c.extxyz", 10, 3.0);
    let (out, err, ok) = rmda(&[
        "rdf",
        "--rmax",
        "16.0",
        "--dr",
        "0.1",
        path.to_str().unwrap(),
    ]);
    assert!(!ok, "should have failed: {out}");
    assert!(
        err.contains("half the smallest cell height"),
        "error must name the limit: {err}"
    );
    assert!(
        out.is_empty() || !out.contains("rdf_g_r"),
        "no histogram on refusal"
    );
    // Exactly half the cell is allowed: the boundary is inclusive on the usable side.
    let (out2, err2, ok2) = rmda(&[
        "rdf",
        "--rmax",
        "15.0",
        "--dr",
        "0.1",
        path.to_str().unwrap(),
    ]);
    assert!(ok2, "rmax at the limit should work: {out2}{err2}");
}

#[test]
fn a_nonsense_range_is_refused_rather_than_producing_an_empty_histogram() {
    let path = sc("sc10d.extxyz", 10, 3.0);
    for (rmax, dr, needle) in [
        ("6.0", "0.0", "bin width"),
        ("6.0", "-0.1", "bin width"),
        ("6.0", "nan", "bin width"),
        ("0.0", "0.1", "rmax"),
        ("-2.0", "0.1", "rmax"),
    ] {
        // `--dr=-0.1` rather than `--dr -0.1`: clap reads a leading '-' as the start
        // of another flag, and an "unexpected argument" failure would look like the
        // range had been rejected for the wrong reason.
        let (out, err, ok) = rmda(&[
            "rdf",
            &format!("--rmax={rmax}"),
            &format!("--dr={dr}"),
            path.to_str().unwrap(),
        ]);
        assert!(!ok, "rmax={rmax} dr={dr} should fail: {out}");
        assert!(err.contains(needle), "rmax={rmax} dr={dr}: {err}");
    }
}

#[test]
fn rmax_is_required_rather_than_defaulted() {
    // A default `rmax` would put a number in the output that nobody chose, and
    // every peak position would depend on it. Refusing costs a flag.
    let path = sc("sc10e.extxyz", 10, 3.0);
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(["rdf", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!out.status.success(), "rmda rdf without --rmax must fail");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--rmax") || err.contains("required"),
        "the error must say what is missing: {err}"
    );
}

#[test]
fn the_run_is_reproducible() {
    // Same input, same bytes out. The histogram is accumulated in atom order with
    // no parallelism and no atomics, so any difference here would mean something
    // reordered underneath.
    let path = two_types("repro.extxyz", 10, 3.0);
    let args = [
        "rdf",
        "--rmax",
        "4.0",
        "--dr",
        "0.05",
        path.to_str().unwrap(),
    ];
    let (a, _, ok1) = rmda(&args);
    let (b, _, ok2) = rmda(&args);
    assert!(ok1 && ok2);
    assert_eq!(a, b);
}

#[test]
fn output_is_refused_rather_than_silently_ignored() {
    // `-o` is a global flag, so passing it means a file was expected. Writing no file
    // and saying nothing would let someone read "no pairs in range" into a run that
    // measured nothing per atom.
    let path = sc("sc10f.extxyz", 10, 3.0);
    let target = tempdir().join("rdf-output.xyz");
    let _ = std::fs::remove_file(&target);
    let (out, err, ok) = rmda(&[
        "rdf",
        "--rmax",
        "5.0",
        "-o",
        target.to_str().unwrap(),
        path.to_str().unwrap(),
    ]);
    assert!(!ok, "should refuse: {out}");
    assert!(err.contains("cannot write"), "{err}");
    assert!(
        err.contains("frame-level"),
        "the reason should say why: {err}"
    );
    assert!(
        !target.exists(),
        "refusing must not leave a file behind: {}",
        target.display()
    );
    // The histogram is still on stdout: the refusal is about the atom file, not the
    // analysis.
    assert!(out.contains("rdf_g_r"), "{out}");
}

#[test]
fn a_missing_file_is_reported_not_panicked() {
    let (out, err, ok) = rmda(&["rdf", "--rmax", "3.0", "/nonexistent/nope.extxyz"]);
    assert!(!ok);
    assert!(
        err.contains("nope.extxyz") || err.contains("No such file"),
        "{err}"
    );
    assert!(!out.contains("rdf_g_r"));
}
