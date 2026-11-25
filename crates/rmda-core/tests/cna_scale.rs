//! Scale and differential coverage for CNA (design §25).
//!
//! Two things live here:
//!
//! * **Differential at scale.** A 108k-atom thermally-perturbed FCC block is
//!   generated deterministically, written to a temp file, and classified. The
//!   per-atom codes committed in `tests/data/m2_100k_expected.txt` are OVITO's
//!   (design §25 differential tests); the fixture itself is *not* committed, so
//!   the test regenerates the exact same atom block at run time. Regenerate the
//!   ground truth with:
//!
//!   ```text
//!   cargo test -p rmda-core --test cna_scale -- --ignored emit_100k_xyz
//!   python3 tools/capture_100k.py
//!   ```
//!
//! * **Correctness-at-scale smoke.** Ideal FCC blocks of 2M and 16M atoms,
//!   fully periodic, so every atom is bulk FCC and any deviation is a real bug.
//!   These are the design §25.1 smoke checks and are `#[ignore]`d: they do not
//!   belong in the per-PR suite. Run them in release:
//!
//!   ```text
//!   cargo test --release -p rmda-core --test cna_scale -- --ignored --nocapture
//!   ```

use rmda_core::analysis::cna::{Cna, CnaMode, BCC, FCC, HCP, ICO, OTHER};
use rmda_core::analysis::{Analysis, AnalysisContext, NeighborhoodContext, Summary};
use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::data::property::PropertyStore;
use rmda_core::exec::Par;
use rmda_core::io::{read_frames, XyzWriter};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const A: f64 = 3.615;
/// 4·30³ = 108000 atoms: a full block, no surface, all PBC.
const REPEAT: usize = 30;
const SEED: u64 = 0x5EED_1000;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// FCC conventional basis.
const BASIS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
];

/// SplitMix64: integer-only, so the sequence is bit-reproducible.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform on `[-s, s)` from the top 53 bits, so the mapping is exact.
fn symmetric(state: &mut u64, s: f64) -> f64 {
    let bits = splitmix64(state) >> 11;
    let u = (bits as f64) / ((1u64 << 53) as f64);
    (u * 2.0 - 1.0) * s
}

/// Ideal FCC block, `n` conventional cells per side, fully periodic.
fn fcc_ideal(n: usize) -> (UnitCell, Vec<[f64; 3]>) {
    let mut pos = Vec::with_capacity(4 * n * n * n);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                for b in BASIS {
                    pos.push([
                        (ix as f64 + b[0]) * A,
                        (iy as f64 + b[1]) * A,
                        (iz as f64 + b[2]) * A,
                    ]);
                }
            }
        }
    }
    let l = n as f64 * A;
    (UnitCell::orthorhombic(l, l, l), pos)
}

/// The same block with jitter that grows along z, so the block spans a spectrum
/// of classes: the first layers stay ideal FCC, later ones are progressively
/// displaced until neighbours break symmetry. A single fixed sigma is either
/// too gentle (every atom FCC — a trivial differential) or too harsh (every atom
/// OTHER). The gradient exercises the adaptive cutoff and the OTHER boundary at
/// scale, and makes the OVITO comparison sensitive to every class.
fn fcc_thermal() -> (UnitCell, Vec<[f64; 3]>) {
    const SIGMAS: [f64; 5] = [0.05, 0.15, 0.25, 0.35, 0.45];
    let mut state = SEED;
    let mut pos = Vec::with_capacity(4 * REPEAT * REPEAT * REPEAT);
    for ix in 0..REPEAT {
        for iy in 0..REPEAT {
            for iz in 0..REPEAT {
                let sigma = SIGMAS[iz * SIGMAS.len() / REPEAT];
                for b in BASIS {
                    pos.push([
                        (ix as f64 + b[0]) * A + symmetric(&mut state, sigma),
                        (iy as f64 + b[1]) * A + symmetric(&mut state, sigma),
                        (iz as f64 + b[2]) * A + symmetric(&mut state, sigma),
                    ]);
                }
            }
        }
    }
    let l = REPEAT as f64 * A;
    (UnitCell::orthorhombic(l, l, l), pos)
}

fn to_frame(cell: UnitCell, pos: &[[f64; 3]]) -> Frame {
    let mut p = ParticleData::with_capacity(pos.len());
    for (i, q) in pos.iter().enumerate() {
        p.add_atom(*q, i as i64 + 1, 1);
    }
    let mut frame = Frame::new(cell, p);
    frame.type_labels = vec!["Cu".to_string()];
    frame
}

fn write_xyz(path: &Path, frame: &Frame) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    XyzWriter::write_all(&mut w, std::slice::from_ref(frame)).unwrap();
    w.flush().unwrap();
}

fn run_cna(frame: &Frame, mode: CnaMode, threads: usize) -> Vec<u8> {
    let cna = Cna::new(mode);
    // CNA builds its own k-NN index.
    let nkeys = NeighborhoodContext { cell: None };
    let par = Par::new(threads);
    let ctx = AnalysisContext {
        frame,
        nkeys: &nkeys,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    cna.run(&ctx, &mut out, &mut summary).unwrap();
    out.get("StructureType").unwrap().as_u8().unwrap().clone()
}

/// Per-atom OVITO codes captured by `tools/capture_100k.py`.
fn read_expected(path: &Path) -> Vec<u8> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\nregenerate with tools/capture_100k.py",
            path.display()
        )
    });
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("cna_adaptive ") {
            return rest
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
        }
    }
    panic!("no cna_adaptive line in {}", path.display());
}

/// OVITO differential on a 108k-atom block, rebuilt from scratch at test time.
#[test]
fn cna_100k_matches_ovito() {
    let expected = read_expected(&data_dir().join("m2_100k_expected.txt"));

    let path = std::env::temp_dir().join("rmda_cna_100k/m2_100k.xyz");
    let (cell, pos) = fcc_thermal();
    assert_eq!(pos.len(), expected.len(), "regenerate the expectation");
    let frame = to_frame(cell, &pos);
    write_xyz(&path, &frame);

    // Round-trip through the reader so the geometry is bit-identical to the
    // file OVITO was given, rather than to this test's in-memory doubles.
    let frames = read_frames(&path).unwrap();
    let got = run_cna(&frames[0], CnaMode::Adaptive, 1);

    let mismatches: Vec<usize> = got
        .iter()
        .zip(&expected)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} / {} atoms differ from OVITO; first few: {:?}",
        mismatches.len(),
        expected.len(),
        &mismatches[..mismatches.len().min(10)]
    );
}

/// The block must not be a trivial test: if the jitter never moved an atom out
/// of its class, the differential above would pass for the wrong reason.
#[test]
fn cna_100k_exercises_several_classes() {
    let (cell, pos) = fcc_thermal();
    let frame = to_frame(cell, &pos);
    let got = run_cna(&frame, CnaMode::Adaptive, 1);
    let n = |code: u8| got.iter().filter(|&&t| t == code).count();
    let other = n(OTHER);
    let fcc = n(FCC);
    assert!(fcc > 0, "expected a bulk FCC majority, got none");
    assert!(
        other > 0,
        "no OTHER: the jitter never broke the ideal shell"
    );
    // The graded block should span more than the two bulk classes; BCC and HCP
    // are rare here (OVITO finds a couple of BCC), so require the *set* of
    // non-FCC classes to be non-empty rather than pinning an exact histogram.
    let distinct: usize = [OTHER, FCC, HCP, BCC, ICO]
        .iter()
        .filter(|&&c| n(c) > 0)
        .count();
    assert!(
        distinct >= 2,
        "the block is a single class; fixture is trivial"
    );
}

/// Thread-count independence at scale (design §25.1 determinism gate).
#[test]
#[ignore = "2M-atom determinism check; run in release"]
fn cna_2m_is_thread_count_independent() {
    let (cell, pos) = fcc_ideal(80); // 2.048M atoms
    let frame = to_frame(cell, &pos);
    let one = run_cna(&frame, CnaMode::Adaptive, 1);
    for t in [2, 4] {
        let many = run_cna(&frame, CnaMode::Adaptive, t);
        assert_eq!(many, one, "threads={t} differ");
    }
    assert!(one.iter().all(|&t| t == FCC));
}

/// Design §25.1 correctness-at-scale smoke. An ideal, fully periodic FCC
/// block has no surface and no defects, so every atom must be FCC: any OTHER is
/// a genuine neighbour-search or classification bug, not a physical result.
fn scale_smoke(cells_per_side: usize) {
    let n = 4 * cells_per_side * cells_per_side * cells_per_side;
    let t0 = Instant::now();
    let (cell, pos) = fcc_ideal(cells_per_side);
    let frame = to_frame(cell, &pos);
    let build = t0.elapsed();
    let t1 = Instant::now();
    let got = run_cna(&frame, CnaMode::Adaptive, 0);
    let classify = t1.elapsed();

    let bad = got.iter().filter(|&&t| t != FCC).count();
    let rate = n as f64 / classify.as_secs_f64();
    eprintln!(
        "{n} atoms: build {:?}, cna {:?} ({rate:.1} atoms/s), non-FCC {bad}",
        build, classify
    );
    assert_eq!(bad, 0, "{bad} atoms in an ideal FCC block were not FCC");
}

#[test]
#[ignore = "2M-atom scale smoke; run in release"]
fn cna_scale_2m() {
    scale_smoke(80);
}

#[test]
#[ignore = "16M-atom scale smoke (design §25.1); run in release on a large machine"]
fn cna_scale_16m() {
    scale_smoke(160);
}

/// Write the block OVITO should be run on. The result is deliberately *not*
/// committed; only the codes it yields are.
#[test]
#[ignore = "one-shot fixture emitter for the OVITO capture"]
fn emit_100k_xyz() {
    let path = repo_root().join("target/m2_100k.xyz");
    let (cell, pos) = fcc_thermal();
    let frame = to_frame(cell, &pos);
    write_xyz(&path, &frame);
    eprintln!(
        "wrote {} ({} atoms)\nnow run: python3 tools/capture_100k.py",
        path.display(),
        pos.len()
    );
}

/// One-shot emitter for the ~10M-atom `rmda analyze` benchmark: a periodic ideal
/// FCC block of 136 conventional cells per side (10,059,776 atoms). Fire with
/// `cargo test --release -p rmda-core --test cna_scale -- --ignored emit_10m_xyz`.
#[test]
#[ignore = "one-shot fixture emitter for the 10M benchmark"]
fn emit_10m_xyz() {
    let path = repo_root().join("target/m10m.xyz");
    let (cell, pos) = fcc_ideal(136); // 4 * 136^3 = 10,059,776
    let frame = to_frame(cell, &pos);
    write_xyz(&path, &frame);
    eprintln!(
        "wrote {} ({} atoms)\nnow run: target/release/rmda analyze --ptm --grains --rdf --wc --cluster on it",
        path.display(),
        pos.len()
    );
}
