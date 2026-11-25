//! Malformed input must end in an error that says what is wrong, never in a
//! panic (exit 101), a hang or a result computed from garbage. Each case is a
//! valid fixture with one defect, as a truncated MD output or a corrupted
//! file presents it. Found by a mutation sweep over the dump, data and xyz
//! readers and the info/ptm/defects/rdf commands.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn data(name: &str) -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/data")
            .join(name),
    )
    .unwrap()
}

fn tempdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("rmda_malformed_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run `rmda args` with a time limit; (exit code, stderr).
fn rmda(args: &[&str]) -> (i32, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let t0 = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            let mut err = String::new();
            use std::io::Read;
            child
                .stderr
                .take()
                .unwrap()
                .read_to_string(&mut err)
                .unwrap();
            return (status.code().unwrap_or(-1), err);
        }
        if t0.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            panic!("rmda {args:?} did not finish in 60 s");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// `text` written as `name` must be refused by every analysis command with
/// a message containing `want`, and exit code 3 (unusable input, design
/// §21).
fn refused(dir: &Path, name: &str, text: &str, want: &str) {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    let f = p.to_str().unwrap();
    let out = dir.join("out.xyz");
    for args in [
        vec!["ptm", f, "-o", out.to_str().unwrap()],
        vec!["defects", f],
        vec!["rdf", f, "--rmax", "3", "--dr", "0.1"],
    ] {
        let (code, err) = rmda(&args);
        assert_eq!(
            code, 3,
            "{name}: rmda {} exited {code} (3: unusable input)\n{err}",
            args[0]
        );
        assert!(!err.contains("panicked"), "{name}: {err}");
        assert!(
            err.contains(want),
            "{name}: rmda {}: want {want:?} in\n{err}",
            args[0]
        );
    }
}

#[test]
fn a_data_file_cut_before_its_atoms_section_is_refused() {
    let dir = tempdir();
    let head: String = data("fcc_ortho.data")
        .lines()
        .take(13)
        .map(|l| format!("{l}\n"))
        .collect();
    refused(
        &dir,
        "cut.data",
        &head,
        "declares 32 atoms but holds 0 positions",
    );
}

#[test]
fn non_finite_coordinates_are_refused_with_the_atom() {
    let dir = tempdir();
    // xyz: a nan; dump: an overflowing literal (1e400 parses as inf).
    let xyz = data("fcc_ortho.xyz");
    let mut lines: Vec<String> = xyz.lines().map(str::to_owned).collect();
    lines[4] = lines[4].replacen("1.80750000", "nan", 1);
    refused(
        &dir,
        "nan.xyz",
        &(lines.join("\n") + "\n"),
        "non-finite coordinate",
    );
    let dump = data("fcc_ortho.dump");
    let bad = dump.replacen("2 1 0.0 1.8075 1.8075", "2 1 0.0 1e400 1.8075", 1);
    assert_ne!(bad, dump);
    refused(
        &dir,
        "inf.dump",
        &bad,
        "atom id 2 (row 2 of the frame) has a non-finite coordinate",
    );
}

#[test]
fn a_non_finite_or_degenerate_box_is_refused() {
    let dir = tempdir();
    let dump = data("fcc_ortho.dump");
    let lines: Vec<&str> = dump.lines().collect();
    // The c vector of the `abc origin` box: a nan, then a tilt of 1e20 Å on a
    // 7.23 Å box (which hung the neighbour search).
    for (row, want) in [
        ("nan 0.0 7.23 0.0", "the box is not finite"),
        ("1e20 0.0 7.23 0.0", "the box is degenerate"),
        // A zero vector: the dump reader's own check.
        ("0.0 0.0 0.0 0.0", "edge vectors are co-planar"),
    ] {
        let mut l: Vec<&str> = lines.clone();
        l[7] = row;
        refused(&dir, "box.dump", &(l.join("\n") + "\n"), want);
    }
}

#[test]
fn an_rdf_histogram_too_large_to_allocate_is_refused() {
    let dir = tempdir();
    let p = dir.join("plain.xyz");
    std::fs::write(&p, "2\nplain\nW 0 0 0\nW 1 0 0\n").unwrap();
    for args in [
        ["--rmax", "1e18", "--dr", "0.1"],
        ["--rmax", "5", "--dr", "1e-18"],
    ] {
        let mut a = vec!["rdf", p.to_str().unwrap()];
        a.extend(args);
        let (code, err) = rmda(&a);
        assert_eq!(code, 1, "{err}");
        assert!(err.contains("bins; at most"), "{err}");
    }
}

#[test]
fn a_flat_atom_set_in_an_open_box_is_quick() {
    // The occupancy grid once sized its cells from a near-zero volume: two
    // atoms on a line took 22 s and 7.6 GB.
    let dir = tempdir();
    let p = dir.join("two.xyz");
    std::fs::write(
        &p,
        "2\nLattice=\"20 0 0 0 20 0 0 0 20\" pbc=\"F F F\" Properties=species:S:1:pos:R:3\n\
         W 0 0 0\nW 20 0 0\n",
    )
    .unwrap();
    let t0 = Instant::now();
    let (code, err) = rmda(&["rdf", p.to_str().unwrap(), "--rmax", "25", "--dr", "0.5"]);
    assert_eq!(code, 0, "{err}");
    assert!(
        t0.elapsed() < Duration::from_secs(10),
        "took {:?}",
        t0.elapsed()
    );
}

#[test]
fn exit_codes_tell_input_usage_and_unmeasured_apart() {
    // design §21: 3 unusable input, 2 invalid request, 5 no grain measured
    // (results written), 0 otherwise.
    let dir = tempdir();
    let missing = dir.join("no-such-file.dump");
    let (code, err) = rmda(&["defects", missing.to_str().unwrap()]);
    assert_eq!(code, 3, "missing input: {err}");
    let (code, err) = rmda(&["info", missing.to_str().unwrap()]);
    assert_eq!(code, 3, "missing input: {err}");

    let cfg = dir.join("bad.toml");
    std::fs::write(&cfg, "[defects]\nrmsd_cutof = 0.1\n").unwrap();
    let (code, err) = rmda(&["run", cfg.to_str().unwrap()]);
    assert_eq!(code, 2, "unknown config key: {err}");
    let (code, _) = rmda(&["defects", "--no-such-flag"]);
    assert_eq!(code, 2);

    // Two atoms: a grain too small to measure; the table is still written.
    let p = dir.join("tiny.xyz");
    std::fs::write(
        &p,
        "2\nLattice=\"20 0 0 0 20 0 0 0 20\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n\
         W 0 0 0\nW 1.4 1.4 1.4\n",
    )
    .unwrap();
    let table = dir.join("tiny.csv");
    let (code, err) = rmda(&[
        "defects",
        p.to_str().unwrap(),
        "--table",
        table.to_str().unwrap(),
    ]);
    assert_eq!(code, 5, "{err}");
    assert!(err.contains("no grain was measured"), "{err}");
    assert!(table.exists(), "results are still written");

    let good = dir.join("fcc.dump");
    std::fs::write(&good, data("fcc_ortho.dump")).unwrap();
    let (code, err) = rmda(&["info", good.to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
}
