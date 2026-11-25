//! OVITO parity of `SegMode::Auto` on the HEA polycrystal (design §15.5).
//!
//! `tests/data/hea_poly_ovito_grains.txt` holds OVITO 3.15's
//! GrainSegmentationModifier labels (all defaults) for
//! `reference_structures/lastframe-HEA-poly.data` (42 432 atoms, 39 %
//! non-crystalline). With PTM in OVITO's Voronoi neighbour order and the auto
//! mode, the labels agree with OVITO's at an adjusted Rand index of 0.9999 and
//! the automatic threshold to 1e-4. The test skips when the snapshot is absent.

mod common;

use rmda_core::analysis::grain_seg::{segment, SegConfig, SegMode};
use rmda_core::analysis::ptm::{check, NeighbourOrder, Ptm};
use rmda_core::exec::Par;
use rmda_core::io::read_frames;
use std::collections::HashMap;
use std::path::Path;

/// Adjusted Rand index of two labellings (any integer labels; `-1` is a label).
fn ari(a: &[i64], b: &[i64]) -> f64 {
    let mut m: HashMap<(i64, i64), f64> = HashMap::new();
    let mut ra: HashMap<i64, f64> = HashMap::new();
    let mut rb: HashMap<i64, f64> = HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *m.entry((x, y)).or_default() += 1.0;
        *ra.entry(x).or_default() += 1.0;
        *rb.entry(y).or_default() += 1.0;
    }
    let c2 = |v: f64| v * (v - 1.0) / 2.0;
    let s: f64 = m.values().map(|&v| c2(v)).sum();
    let sa: f64 = ra.values().map(|&v| c2(v)).sum();
    let sb: f64 = rb.values().map(|&v| c2(v)).sum();
    let e = sa * sb / c2(a.len() as f64);
    (s - e) / (0.5 * (sa + sb) - e)
}

#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn auto_mode_matches_ovito_on_the_hea_polycrystal() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let frame = common::reference_dir().join("lastframe-HEA-poly.data");
    if !frame.exists() {
        eprintln!("skipping: {} absent", frame.display());
        return;
    }
    let text = std::fs::read_to_string(root.join("tests/data/hea_poly_ovito_grains.txt")).unwrap();
    let threshold: f64 = text
        .lines()
        .find_map(|l| l.split("auto_merge_threshold = ").nth(1))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let ovito: Vec<i64> = text
        .lines()
        .find(|l| !l.starts_with('#'))
        .unwrap()
        .bytes()
        .map(|c| (c - b'0') as i64)
        .collect();

    let f = read_frames(&frame).unwrap().into_iter().next().unwrap();
    assert_eq!(ovito.len(), f.natoms());
    let par = Par::new(4);
    let ptm = Ptm {
        rmsd_cutoff: 0.1,
        flags: check::OVITO_DEFAULT,
    }
    .compute_ordered(
        &f.cell,
        &f.particles.positions,
        &par,
        NeighbourOrder::Voronoi,
    )
    .unwrap();
    let cfg = SegConfig {
        mode: SegMode::Auto,
        ..Default::default()
    };
    let s = segment(&f.cell, &f.particles.positions, &ptm, &cfg, &par).unwrap();
    let mine: Vec<i64> = s.grain.iter().map(|&g| g as i64).collect();
    let r = ari(&mine, &ovito);
    let t = s.merge_threshold.unwrap();
    println!(
        "HEA auto: {} grains, ARI {r:.5}, threshold {t:.6} (OVITO {threshold:.6})",
        s.grains.len()
    );
    assert_eq!(s.grains.len(), 4);
    assert!(r >= 0.999, "ARI {r} against OVITO");
    assert!(
        (t - threshold).abs() < 1e-3,
        "threshold {t} vs OVITO {threshold}"
    );
}
