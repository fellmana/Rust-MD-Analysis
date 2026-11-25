//! CNA validation against OVITO on large, physically realistic MD snapshots.
//!
//! The `reference_structures/` directory holds real 2000 K MD frames that are
//! genuinely hard for common-neighbour analysis:
//!
//! | file | structure | atoms | why it is a good test |
//! |---|---|---|---|
//! | `data.final-lastframe-Al-2000` | FCC | 500 000 | few HCP stacking faults among 92 % FCC |
//! | `data.final-lastframe-W-2000` | BCC | 500 094 | BCC at high T, huge OTHER fraction at the surface of the melt-free box |
//! | `data.final-lastframe-Zr-2000` | HCP | 492 800 | HCP is the hardest case for the 4/5 common-neighbour tie-break |
//! | `lastframe-HEA-poly.data` | FCC HEA polycrystal | 42 432 | 5 species, grain boundaries, shifted box origin |
//!
//! These are not committed (~240 MB); the test skips itself when they are
//! absent. The OVITO per-atom oracle is captured once with
//!
//! ```text
//! python3 tools/capture_cna.py reference_structures/<file> <out>.cna
//! ```
//!
//! and cached under `target/refcheck/`, which is likewise not committed. Both
//! OVITO and `rmda` read the atoms in file order, so the codes can be compared
//! index-wise.

use rmda_core::analysis::cna::{Cna, CnaMode};
use rmda_core::analysis::{Analysis, AnalysisContext, NeighborhoodContext, Summary};
use rmda_core::data::frame::Frame;
use rmda_core::data::property::PropertyStore;
use rmda_core::exec::Par;
use rmda_core::io::read_frames;
use std::path::{Path, PathBuf};

mod common;

const OTHER: u8 = 0;
const FCC: u8 = 1;
const HCP: u8 = 2;
const BCC: u8 = 3;
const ICO: u8 = 4;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reference_dir() -> PathBuf {
    common::reference_dir()
}

fn oracle_dir() -> PathBuf {
    repo_root().join("target/refcheck")
}

fn classify(frame: &Frame) -> Vec<u8> {
    let cna = Cna::new(CnaMode::Adaptive);
    // CNA builds its own k-NN index.
    let nkeys = NeighborhoodContext { cell: None };
    let par = Par::default();
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

/// The oracle is two lines: a `cna_adaptive <histogram>` summary, then one
/// whitespace-separated code per atom in file order.
fn read_oracle(path: &Path) -> Vec<u8> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\nregenerate with tools/capture_cna.py",
            path.display()
        )
    });
    let mut lines = raw.lines();
    let header = lines
        .next()
        .and_then(|l| l.strip_prefix("cna_adaptive "))
        .unwrap_or_else(|| panic!("missing cna_adaptive header in {}", path.display()));
    assert!(
        header.contains('x'),
        "header should be a `<code>x<count>` histogram, got: {header}"
    );
    lines
        .next()
        .expect("oracle has no code line")
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

fn histogram(codes: &[u8]) -> [usize; 5] {
    let mut h = [0usize; 5];
    for &c in codes {
        h[c as usize] += 1;
    }
    h
}

/// The name OVITO and we agree should dominate, plus the minimum share we
/// expect. These are physical assertions, not OVITO restatements: 2000 K bulk
/// Al/W/Zr are overwhelmingly their own lattice, and a defect population in
/// the few-percent range is what thermal disorder produces.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn cna_matches_ovito_on_reference_structures() {
    let ref_dir = reference_dir();
    if !ref_dir.is_dir() {
        eprintln!("skipping: {} not present", ref_dir.display());
        return;
    }
    let oracle_dir = oracle_dir();
    if !oracle_dir.is_dir() {
        eprintln!(
            "skipping: no OVITO oracle in {}; run tools/capture_cna.py first",
            oracle_dir.display()
        );
        return;
    }

    // (file, expected dominant class, minimum share of it)
    let cases: [(&str, u8, f64); 4] = [
        ("data.final-lastframe-Al-2000", FCC, 0.85),
        ("data.final-lastframe-W-2000", BCC, 0.85),
        ("data.final-lastframe-Zr-2000", HCP, 0.85),
        ("lastframe-HEA-poly.data", FCC, 0.40),
    ];
    let names = [OTHER, FCC, HCP, BCC, ICO];

    let mut checked = 0;
    for (file, dominant, min_share) in cases {
        let path = ref_dir.join(file);
        let oracle = oracle_dir.join(format!("{file}.cna"));
        if !path.is_file() || !oracle.is_file() {
            eprintln!("skipping {file}: missing structure or oracle");
            continue;
        }

        let frames = read_frames(&path).unwrap();
        let got = classify(&frames[0]);
        let expected = read_oracle(&oracle);
        assert_eq!(
            got.len(),
            expected.len(),
            "{file}: atom count differs from the oracle"
        );

        // Exact per-atom agreement is the goal: CNA is a discrete test on
        // exact floating-point shells, and matching OVITO atom-for-atom on
        // 1.5M thermally disordered atoms is the strongest statement we can
        // make. A regression here is a real divergence, not rounding noise.
        let mismatches: Vec<usize> = got
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        let agree = 1.0 - mismatches.len() as f64 / got.len() as f64;

        let hist = histogram(&got);
        let oracle_hist = histogram(&expected);
        let share = hist[dominant as usize] as f64 / got.len() as f64;
        println!(
            "{file}: n={} agree={:.6} ({}/{} differ)\n  rmda   {}",
            got.len(),
            agree,
            mismatches.len(),
            got.len(),
            hist.iter()
                .enumerate()
                .map(|(c, &n)| format!("{}x{}", names[c], n))
                .collect::<Vec<_>>()
                .join(" ")
        );
        println!(
            "  ovito {}",
            oracle_hist
                .iter()
                .enumerate()
                .map(|(c, &n)| format!("{}x{}", names[c], n))
                .collect::<Vec<_>>()
                .join(" ")
        );

        assert!(
            share >= min_share,
            "{file}: expected >= {min_share:.0}% {} at 2000 K, got {:.2}%",
            names[dominant as usize],
            share * 100.0
        );
        assert_eq!(
            hist, oracle_hist,
            "{file}: class histogram differs from OVITO"
        );
        assert!(
            mismatches.is_empty(),
            "{file}: {} of {} atoms differ from OVITO (agree {:.6}); first mismatches {:?} \
             -- a small count is usually the adaptive search radius being too \
             small, which under-coordinates atoms and turns them into OTHER",
            mismatches.len(),
            got.len(),
            agree,
            &mismatches[..mismatches.len().min(10)]
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no reference structures were actually compared"
    );
}
