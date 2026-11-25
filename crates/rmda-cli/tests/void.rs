//! End-to-end tests for the `rmda void` subcommand.
//!
//! Two things are being checked here that library tests cannot reach. The first is
//! the gate: method B is refused without `--experimental`, and that refusal is the
//! design's §17.2 position rather than a formality, so it is worth pinning at the
//! boundary a user actually meets it. The second is the wiring of the sample, the
//! reference and the ideal configuration, which is where a CLI can quietly
//! differ from the library by measuring the reference against itself.
//!
//! The physics of the field is covered exhaustively in `void_b_benchmark.rs` and
//! `roto_strain_hea.rs`; these tests only check that the right atoms went into the
//! right field and that the numbers land near the analytic answer.

use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::io::XyzWriter;
use rmda_core::lattice::fcc::Fcc;
use rmda_core::lattice::sites::SiteLattice;
use rmda_core::lattice::Lattice;
use std::path::PathBuf;
use std::process::Command;

const A: f64 = 3.5;

/// A perfect FCC block with its `victims` central sites removed, as extended XYZ.
///
/// The victims are the sites nearest the block's centre, not the first `n` by index.
/// That is not tidiness: the command integrates the box interior inset by one
/// nearest-neighbour distance, so a vacancy near a corner is outside the region the
/// measurement is taken over and reads as no void at all -- which is a correct answer
/// to the question that was asked and a wrong answer to the one meant.
fn fixture(name: &str, cells: usize, victims: usize) -> PathBuf {
    let len = cells as f64 * A;
    let cell = UnitCell::orthorhombic(len, len, len);
    let sites: Vec<[f64; 3]> = SiteLattice::build(&cell, &Fcc { a: A }.primitive(), [0.0; 3])
        .unwrap()
        .iter()
        .collect();

    let mid = [len / 2.0; 3];
    let mut order: Vec<usize> = (0..sites.len()).collect();
    order.sort_by(|&i, &j| {
        let d = |p: [f64; 3]| {
            (p[0] - mid[0]).powi(2) + (p[1] - mid[1]).powi(2) + (p[2] - mid[2]).powi(2)
        };
        d(sites[i])
            .partial_cmp(&d(sites[j]))
            .unwrap()
            .then(i.cmp(&j))
    });
    let drop = &order[..victims.min(order.len())];

    let mut pd = ParticleData::new(sites.len() - drop.len());
    for (i, p) in sites.iter().enumerate() {
        if drop.contains(&i) {
            continue;
        }
        pd.positions.push(*p);
        pd.ids.push(i as i64 + 1);
        pd.types.push(1);
    }
    let frame = Frame::new(cell, pd);

    let dir = std::env::temp_dir().join("rmda-cli-void-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
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

/// The volume line: `<volume> A^3 of <region> A^3 = <share> % of the region`.
fn detail_line(stdout: &str) -> &str {
    stdout
        .lines()
        .find(|l| l.contains("% of the region"))
        .unwrap_or_else(|| panic!("no volume line in:\n{stdout}"))
}

/// The summary line: `void volume: <v> +/- <se> A^3 (<share>% of region, <hits> of <n> samples)`.
fn summary_line(stdout: &str) -> &str {
    stdout
        .lines()
        .find(|l| l.contains("% of region"))
        .unwrap_or_else(|| panic!("no summary line in:\n{stdout}"))
}

fn volume(stdout: &str) -> f64 {
    detail_line(stdout)
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap_or_else(|e| panic!("volume unparseable: {e}"))
}

/// One vacancy is one empty W-S cell, so the exact answer is `a^3/4`.
const PER_SITE: f64 = A * A * A / 4.0;

/// The block's nearest-neighbour distance. Passed explicitly on every run rather than left
/// at the CLI's default: the default is Fe's 2.86 A, and a declared lattice that
/// disagrees with the sample returns a field of the wrong size without erroring -- the
/// exact mistake the command warns about, which these tests should not be tripping.
const NN: f64 = 2.4748737341529163; // A / sqrt(2)

/// Sample count for the volume assertions.
///
/// High, because a single site is 10.7 A^3 out of a 27,000 A^3 region: at 40000 samples that
/// is a handful of hits and the estimate swings by a factor of two between runs. The relative
/// standard error is ~12% at 200000, which is what the bands below are written against. This
/// is not a property of the command being slow -- 200000 samples of a 4000-atom block take
/// milliseconds -- it is a property of counting ~30 hits.
const SAMPLES: &str = "200000";

fn nn_args() -> [String; 2] {
    ["--nn-distance".to_owned(), format!("{NN:.6}")]
}

/// The hit count from the line the command prints its share on.
fn hits(stdout: &str) -> usize {
    // `(share% of region, <hits> of <n> samples)`
    let line = summary_line(stdout);
    let tail = line
        .rsplit(", ")
        .next()
        .unwrap_or_else(|| panic!("no hit count in {line:?}"));
    tail.split(" of ")
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("hit count unparseable in {line:?}: {e}"))
}

/// The volume from the per-threshold line the `--excess` path prints.
fn excess_volume(stdout: &str) -> f64 {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with("epsilon "))
        .unwrap_or_else(|| panic!("no per-threshold line in:\n{stdout}"))
        .split_whitespace()
        .nth(2)
        .unwrap()
        .parse()
        .unwrap_or_else(|e| panic!("excess volume unparseable: {e}"))
}

#[test]
fn method_b_is_refused_without_the_experimental_flag() {
    // Without --experimental, `rmda void` is method A (the census of design 17,
    // `void_a.rs`): no phi field is built, and method B's options are refused.
    let path = fixture("gate.extxyz", 6, 0);
    let (out, err, ok) = rmda(&["void", path.to_str().unwrap()]);
    assert!(ok, "{out}\n{err}");
    assert!(!out.contains("phi field"), "{out}");
    assert!(out.contains("voids: 0"), "{out}");
    let (out, err, ok) = rmda(&["void", path.to_str().unwrap(), "--epsilon", "0.02"]);
    assert!(!ok, "method B ran without --experimental:\n{out}");
    assert!(err.contains("--experimental"), "{err}");
}

#[test]
fn argument_mistakes_are_refused_before_the_field_is_built() {
    let path = fixture("args.extxyz", 6, 1);
    let p = path.to_str().unwrap();
    for (args, why) in [
        (
            vec!["void", "--experimental", "--epsilons", "0.01,0.02", p],
            "two epsilons",
        ),
        (
            vec!["void", "--experimental", "--strain-spacing", "1.5", p],
            "strain spacing without strain",
        ),
        (
            vec!["void", "--experimental", "--excess", p],
            "excess without a reference",
        ),
        (
            vec!["void", "--experimental", "--map", "to-reference", p],
            "map without a reference",
        ),
        (
            vec!["void", "--experimental", "--region", "1,2,3", p],
            "three region bounds",
        ),
        (
            vec!["void", "--experimental", "--region", "5,1,5,1,5,1", p],
            "inverted region",
        ),
        (
            vec!["void", "--experimental", "--samples", "0", p],
            "zero samples",
        ),
    ] {
        let (out, err, ok) = rmda(&args);
        assert!(!ok, "{why} was accepted:\n{out}");
        assert!(err.contains("error"), "{why}: {err}");
    }
}

#[test]
fn a_single_vacancy_measures_close_to_one_site_volume() {
    let path = fixture("one_vacancy.extxyz", 10, 1);
    let p = path.to_str().unwrap();
    let nn = nn_args();
    let (out, err, ok) = rmda(&[
        "void",
        "--experimental",
        "--nn-distance",
        &nn[1],
        "--samples",
        SAMPLES,
        p,
    ]);
    assert!(ok, "rmda void failed: {err}");
    let v = volume(&out);
    // The library benchmarks put this at 0.95-0.97 of the exact value on carved
    // clusters, and the sampling error here is ~12%, so the band is what those two
    // allow rather than something tighter the estimator cannot hold.
    assert!(
        (0.60..=1.25).contains(&(v / PER_SITE)),
        "{v} A^3 against a {PER_SITE:.3} A^3 site ({:.3} sites)\n{out}",
        v / PER_SITE
    );
}

#[test]
fn the_field_is_built_on_the_sample_and_not_on_the_reference() {
    // A reference with no vacancies and a sample with three is the only way to
    // catch the CLI measuring the reference against itself: then both fields are
    // void-free and the excess comes out at zero.
    let ref_path = fixture("clean_reference.extxyz", 10, 0);
    let sample = fixture("three_vacancies.extxyz", 10, 3);
    let (out, err, ok) = rmda(&[
        "void",
        "--experimental",
        "--samples",
        SAMPLES,
        "--reference",
        ref_path.to_str().unwrap(),
        "--excess",
        sample.to_str().unwrap(),
    ]);
    assert!(ok, "rmda void --excess failed: {err}");
    // The sample's atom count, not the reference's: four sites per cell, ten cells per
    // edge, minus the three the sample lost.
    assert!(
        out.contains("atoms: 3997 in the sample"),
        "the field was not built on the sample:\n{out}"
    );
    let v = excess_volume(&out);
    assert!(
        v > 2.0 * PER_SITE,
        "excess {v} A^3 does not see the three removed sites ({:.2} sites)\n{out}",
        v / PER_SITE
    );
}

#[test]
fn a_perfect_lattice_measures_no_void() {
    let path = fixture("perfect.extxyz", 10, 0);
    let (out, err, ok) = rmda(&[
        "void",
        "--experimental",
        "--samples",
        "40000",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "rmda void failed: {err}");
    // FCC's W-S cells tile exactly, so `phi <= 1` everywhere and a finite sample
    // can only find points within the Monte Carlo noise of the boundary. Anything
    // above a handful of hits means the lattice is not commensurate with the cell.
    let h = hits(&out);
    assert!(
        h < 40,
        "{h} of 40000 samples hit in a perfect lattice:\n{out}"
    );
}

#[test]
fn the_strain_field_runs_and_reports_its_size() {
    let path = fixture("strain.extxyz", 10, 2);
    let (out, err, ok) = rmda(&[
        "void",
        "--experimental",
        "--samples",
        "20000",
        "--strain",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "rmda void --strain failed: {err}");
    assert!(out.contains("roto-strain:"), "{out}");
    // The node count is the mesh, so a mesh that ignored the cell would show up
    // here as a mesh sized by something else entirely.
    assert!(err.contains("nodes,"), "{err}");
}

#[test]
fn the_extrapolation_reports_its_own_diagnostics() {
    let path = fixture("extrapolate.extxyz", 10, 1);
    let nn = nn_args();
    let (out, err, ok) = rmda(&[
        "void",
        "--experimental",
        "--nn-distance",
        &nn[1],
        "--samples",
        SAMPLES,
        "--epsilons",
        "0.01,0.02,0.05",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "rmda void --epsilons failed: {err}");
    assert!(out.contains("extrapolated to phi = 1"), "{out}");
    assert!(out.contains("falling in epsilon"), "{out}");
    assert!(out.contains("fit trustworthy"), "{out}");
}

#[test]
fn a_declared_lattice_that_disagrees_with_the_sample_is_called_out() {
    // The failure this guards against is silent and has a plausible-looking number:
    // too large a declared nn makes the ideal cells swallow the sample, so a crater
    // reads as zero void. Nothing errors, the run succeeds, and the answer is wrong.
    let path = fixture("mismatch.extxyz", 10, 1);
    let (out, err, ok) = rmda(&[
        "void",
        "--experimental",
        "--samples",
        "10000",
        "--nn-distance",
        "2.86",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "rmda void failed: {err}");
    assert!(err.contains("warning"), "{err}");
    // The measurement goes to stdout with the rest of the report; only the warning is
    // advice, so only it is on stderr.
    assert!(out.contains("observed nn:"), "{out}");
    assert!(out.contains("-13.5 %"), "{out}");
    // And the number it goes on to print is the misleading one, which is the point.
    assert!(volume(&out) < 0.5 * PER_SITE, "{out}");
}

#[test]
fn a_non_finite_epsilon_is_a_usage_error_not_a_panic() {
    // `--epsilons nan,1,2` must not panic (exit 101) inside a
    // `sort_by(|x, y| x.partial_cmp(y).unwrap())`. A threshold list that
    // cannot be ordered is an invalid request: exit 2 (design §21).
    let path = fixture("nan_eps.extxyz", 6, 0);
    let p = path.to_str().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(["void", "--experimental", "--epsilons", "nan,1,2", p])
        .output()
        .expect("failed to run rmda");
    let status = out.status.code().unwrap_or(-1);
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(
        status, 2,
        "expected exit code 2 (usage), got {status}: {err}"
    );
    assert!(err.contains("--epsilons must be finite"), "{err}");
}

/// `src` (an extended XYZ of W) with Gaussian displacements of width `noise` on
/// every coordinate, written beside the other test files; `src` itself at 0.
fn noisy(src: &std::path::Path, noise: f64) -> PathBuf {
    if noise == 0.0 {
        return src.to_path_buf();
    }
    let text = std::fs::read_to_string(src).unwrap();
    let mut seed: u64 = 17;
    let mut uniform = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = String::new();
    for (i, l) in text.lines().enumerate() {
        if i < 2 {
            out.push_str(l);
        } else {
            out.push('W');
            for (k, x) in l.split_whitespace().skip(1).enumerate() {
                if k >= 3 {
                    out.push_str(&format!(" {x}"));
                    continue;
                }
                let x: f64 = x.parse().unwrap();
                let (u, w) = (uniform(), uniform());
                let g = noise * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * w).cos();
                out.push_str(&format!(" {}", x + g));
            }
        }
        out.push('\n');
    }
    let dir = std::env::temp_dir().join("rmda-cli-void-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!(
        "noisy-{noise}-{}",
        src.file_name().unwrap().to_str().unwrap()
    ));
    std::fs::write(&path, out).unwrap();
    path
}

/// The W lattice constant of the gate files.
const W_A: f64 = 3.1652;

/// A perfect 10-cell BCC W box, the size of the gate files.
fn w_perfect() -> PathBuf {
    let len = 10.0 * W_A;
    let cell = UnitCell::orthorhombic(len, len, len);
    let sites: Vec<[f64; 3]> = SiteLattice::build(
        &cell,
        &rmda_core::lattice::bcc::Bcc { a: W_A }.primitive(),
        [0.0; 3],
    )
    .unwrap()
    .iter()
    .collect();
    let mut pd = ParticleData::new(sites.len());
    for (i, p) in sites.iter().enumerate() {
        pd.positions.push(*p);
        pd.ids.push(i as i64 + 1);
        pd.types.push(1);
    }
    let dir = std::env::temp_dir().join("rmda-cli-void-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("w_perfect_10.xyz");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut w, &Frame::new(cell, pd)).unwrap();
    path
}

/// Method B's extrapolated volume of `path` (W, BCC), with `mode` the arguments
/// choosing the extrapolation.
fn w_volume(path: &std::path::Path, mode: &[&str]) -> f64 {
    let nn = format!("{}", W_A * 3f64.sqrt() / 2.0);
    let mut args = vec![
        "void",
        "--experimental",
        "--lattice",
        "bcc",
        "--nn-distance",
        &nn,
        path.to_str().unwrap(),
    ];
    args.extend_from_slice(mode);
    let (out, err, ok) = rmda(&args);
    assert!(ok, "{out}\n{err}");
    out.lines()
        .find_map(|l| l.strip_prefix("extrapolated to phi = 1: "))
        .unwrap_or_else(|| panic!("no extrapolation in\n{out}"))
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

/// Design 17's method-B gate on pure W: vacancy clusters of known size
/// (15 and 89 sites) in BCC W relaxed at 0 K with the Finnis-Sinclair
/// potential (`tools/fs_w.py`, `tools/void_b_gate.py`), measured by the
/// quadratic extrapolation to phi = 1 against the analytic `n a^3 / 2`;
/// `noise` adds Gaussian displacements (a thermal crystal). Returns the
/// relative error.
fn w_gate(file: &str, sites: f64, noise: f64) -> f64 {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file);
    let v = w_volume(
        &noisy(&src, noise),
        &["--samples", "200000", "--epsilons", "0.01,0.02,0.04"],
    );
    v / (sites * W_A * W_A * W_A / 2.0) - 1.0
}

#[test]
fn method_b_passes_the_relaxed_w_gate_at_zero_kelvin() {
    // Within 10 %. Before BCC's cell had its six second-shell faces
    // (`face_normals`) the same runs read 5-20 % small.
    for (file, n) in [
        ("w_fs_relaxed_void15.xyz", 15.0),
        ("w_fs_relaxed_void89.xyz", 89.0),
    ] {
        let rel = w_gate(file, n, 0.0);
        assert!(rel.abs() < 0.10, "{file}: {:+.1} %", 100.0 * rel);
    }
}

#[test]
fn method_b_fails_the_gate_with_thermal_displacements() {
    // Why method B stays --experimental: 0.03 A of Gaussian noise (below W's
    // room-temperature amplitude) puts both voids outside 10 % (high).
    for (file, n) in [
        ("w_fs_relaxed_void15.xyz", 15.0),
        ("w_fs_relaxed_void89.xyz", 89.0),
    ] {
        let rel = w_gate(file, n, 0.03);
        assert!(rel > 0.10, "{file}: {:+.1} %", 100.0 * rel);
    }
}

#[test]
fn a_single_threshold_excess_headline_is_its_own_row() {
    // Design §17.2: with one epsilon, a quadratic fitted through one point would make the
    // `--excess` headline read 3603.5 A^3 against its row's 1393.8 A^3 (an
    // unrelaxed 89-site W void, analytic 1411 A^3). One threshold is a measurement: the
    // headline must be the row, and on the relaxed 89-site file it is the void.
    let a = W_A;
    let ref_path = w_perfect();
    let sample =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/w_fs_relaxed_void89.xyz");
    let nn = format!("{}", a * 3f64.sqrt() / 2.0);
    let (out, err, ok) = rmda(&[
        "void",
        "--experimental",
        "--lattice",
        "bcc",
        "--nn-distance",
        &nn,
        "--samples",
        "200000",
        "--reference",
        ref_path.to_str().unwrap(),
        "--excess",
        sample.to_str().unwrap(),
    ]);
    assert!(ok, "{out}\n{err}");
    let head: f64 = out
        .lines()
        .find_map(|l| l.strip_prefix("excess over reference: "))
        .unwrap_or_else(|| panic!("no headline in\n{out}"))
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let row = excess_volume(&out);
    assert!(
        (head - row).abs() < 0.01,
        "headline {head} vs row {row}\n{out}"
    );
    assert!(out.contains("not extrapolated"), "{out}");
    let want = 89.0 * a * a * a / 2.0;
    assert!(
        (row / want - 1.0).abs() < 0.10,
        "{row} against {want}\n{out}"
    );
}

#[test]
fn the_thermal_window_passes_the_w_gate_with_thermal_displacements() {
    // Design 17.2: above the frame's largest face gap the thermal shell is
    // closed, and V^(1/3) extrapolates linearly to phi = 1. The relaxed 15- and
    // 89-site voids at 0, 0.03 and 0.06 A of noise are within 10 % (measured
    // within 4 % by tools/void_b_gate.py), and the defect-free box with noise
    // reads no void, where the quadratic reads 115 and 405 A^3 of shell.
    let centre = 5.0 * W_A + 0.01;
    let region = format!(
        "{lo},{hi},{lo},{hi},{lo},{hi}",
        lo = centre - 3.0 * W_A,
        hi = centre + 3.0 * W_A
    );
    let mode = [
        "--samples",
        "400000",
        "--region",
        region.as_str(),
        "--thermal-window",
    ];
    for (file, n) in [
        ("w_fs_relaxed_void15.xyz", 15.0),
        ("w_fs_relaxed_void89.xyz", 89.0),
    ] {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(file);
        for noise in [0.0, 0.03, 0.06] {
            let v = w_volume(&noisy(&src, noise), &mode);
            let rel = v / (n * W_A * W_A * W_A / 2.0) - 1.0;
            assert!(
                rel.abs() < 0.10,
                "{file}, noise {noise}: {:+.1} %",
                100.0 * rel
            );
        }
    }
    let perfect = w_perfect();
    for noise in [0.03, 0.06] {
        let v = w_volume(&noisy(&perfect, noise), &mode);
        assert!(v.abs() < 1.0, "perfect, noise {noise}: {v} A^3");
    }
}
