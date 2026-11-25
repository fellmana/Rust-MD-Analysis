//! The §17.2 gate on real relaxed data, not on a synthetic crystal.
//!
//! Everything else that exercises `RotoStrainField` builds its own fixture: a perfect lattice, a
//! dilation, a rotation, jitter drawn from a seeded generator. Those answer "does the arithmetic
//! work". They cannot answer the
//! question the design turns on, which is whether fitting a deformation field to a 256k-atom
//! relaxed alloy removes void that is really there.
//!
//! Measured on these fixtures (256000 atoms, `a = 3.5259 A`, 40 cells per edge):
//!
//! ```text
//!                   rigid          strained        reduction   max |T - I|
//!   clean           2.861 % of box  2.720 %         4.9 %       0.036
//!   + defects       2.909 % of box  2.722 %         6.4 %       0.089
//! ```
//!
//! Two things follow, and both are asserted below rather than left in a comment.
//!
//! The first is the negative result that matters. A relaxed solid has no porosity, yet the rigid
//! field reports ~2.9 % of the box as `{φ > 1}` — 0.31 A^3 per atom against an atomic volume of
//! 10.95 A^3, which is a shell of thermal thickness, not porosity. Fitting strain removes about 5 %
//! of it and leaves 2.72 %. §17.2 assumes a single affine `T(x)` per node can absorb the
//! displacement field; on a real alloy most of that field is non-affine thermal noise, which a
//! strain cannot represent. The unit test `strain_cannot_remove_a_non_affine_shell` is the
//! controlled version of this and agrees. So the gate does **not** pass: the method does not
//! recover the void fraction of a relaxed solid, and Method B stays experimental.
//!
//! The second is that strain does no harm. The deformed field never inflates the void and never
//! swallows a void that was there, which is the failure mode worth guarding: a neighbourhood
//! containing a vacancy sees a coherent pull inward and will encode it as contraction.
//!
//! A third measurement is here because it is what stopped this from being a defect detector. A
//! sampled volume integral of 40000 points resolves ~2.1e3 A^3 on this fixture; the two files
//! differ by 60 A^3. Planted point defects are a per-site signal and a volume integral cannot see
//! them — that is what Method A's Wigner-Seitz self-reference counting is for, and it is why this
//! file asserts the resolution limit instead of pretending to resolve the defects.

use rmda_core::analysis::roto_strain::{RotoStrainField, RotoStrainOptions};
use rmda_core::analysis::void_b::{estimate_void_volume, PhiField};
use rmda_core::cell::Coord3;
use rmda_core::data::frame::Frame;
use rmda_core::io::read_frames;
use rmda_core::lattice::fcc::Fcc;
use rmda_core::lattice::{Lattice, PrimitiveCell};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

mod common;

/// The fixtures' lattice constant, fixed rather than re-derived per test: it is a property of the
/// files, and a test that recomputes it from the box cannot notice if the box changed.
const A: f64 = 3.5259;

/// Smallest strictly-positive threshold. `estimate_void_volume` rejects `epsilon = 0` because the
/// void set `{φ > 1}` would be empty by construction, not by measurement.
const TINY: f64 = 1e-9;

/// Samples for the volume integral. The same count and seed everywhere, so the rigid and strained
/// numbers differ only by the field they were computed from.
const SAMPLES: usize = 40_000;
const SEED: u64 = 11;

/// §17.2's schedule on real data: mesh at `a/2`, Gaussian width at `a` — one nearest-neighbour
/// distance, coarse enough that a node's neighbourhood averages over several shells rather than
/// fitting whatever sits next to it.
fn opts() -> RotoStrainOptions {
    RotoStrainOptions {
        node_spacing: A / 2.0,
        sigma: A,
        refine_passes: 1,
        max_deviation: None,
        fill_unsolved: true,
    }
}

fn reference_dir() -> PathBuf {
    common::reference_dir()
}

fn frame(name: &str) -> Option<Frame> {
    let path = reference_dir().join(name);
    if !path.exists() {
        eprintln!("skipping {}: {} not present", name, path.display());
        return None;
    }
    Some(read_frames(&path).unwrap().into_iter().next().unwrap())
}

/// Sites, independent of any other particle property the fixture may carry.
fn sites(f: &Frame) -> &[Coord3] {
    &f.particles.positions[..f.natoms()]
}

/// The interior of the box, inset by one lattice constant so the integral never straddles a
/// boundary where the periodic images are.
fn interior(f: &Frame) -> [(f64, f64); 3] {
    let (lo, hi) = f.cell.bounds();
    (0..3)
        .map(|d| (lo[d] + A, hi[d] - A))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn box_volume(f: &Frame) -> f64 {
    let (lo, hi) = f.cell.bounds();
    (0..3).map(|d| hi[d] - lo[d]).product()
}

/// Both fields on one frame, with the report the gate is read from.
#[derive(Clone)]
struct Measurement {
    rigid: f64,
    strained: f64,
    rigid_se: f64,
    strained_se: f64,
    max_t: f64,
    unsolved: usize,
}

/// Both fixtures are fitted once per process and reused, because each fit is a full 256k-atom
/// deformation-field solve and three tests re-deriving it is three times the wall clock for no
/// extra information. `None` records "fixture absent, tests skip" so a missing file is not
/// silently recomputed.
fn measurement(name: &'static str, label: &str) -> Option<Measurement> {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, Option<Measurement>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().unwrap().get(name) {
        return hit.as_ref().cloned();
    }
    let pc = Fcc { a: A }.primitive();
    let fresh = frame(name).map(|f| measure(&f, &pc, label));
    cache.lock().unwrap().insert(name, fresh.clone());
    fresh
}

fn measure(f: &Frame, pc: &PrimitiveCell, label: &str) -> Measurement {
    let region = interior(f);
    let sites = sites(f);
    let bv = box_volume(f);

    let rigid_field = PhiField::for_threshold(&f.cell, sites, pc, TINY).unwrap();
    let rigid = estimate_void_volume(&rigid_field, &region, TINY, SAMPLES, SEED).unwrap();

    let strain = RotoStrainField::fit(&f.cell, sites, pc, &opts()).unwrap();
    let deformed =
        PhiField::for_threshold_strained(&f.cell, sites, pc, TINY, Some(strain.clone())).unwrap();
    let deformed_est = estimate_void_volume(&deformed, &region, TINY, SAMPLES, SEED).unwrap();

    println!(
        "{label}: {} atoms, box {bv:.0} A^3, region {:.0} A^3\n  \
         rigid    {:>10.2} +- {:.0} A^3 ({:>6.3} % of box)\n  \
         strained {:>10.2} +- {:.0} A^3 ({:>6.3} % of box)\n  \
         reduction{:>10.2} A^3 ({:>5.1} %)\n  \
         max |T - I| {:.4}, unsolved nodes {}",
        f.natoms(),
        rigid.region_volume,
        rigid.volume,
        rigid.standard_error,
        100.0 * rigid.volume / bv,
        deformed_est.volume,
        deformed_est.standard_error,
        100.0 * deformed_est.volume / bv,
        rigid.volume - deformed_est.volume,
        100.0 * (rigid.volume - deformed_est.volume) / rigid.volume.max(f64::MIN_POSITIVE),
        strain.max_deviation_from_identity(),
        strain.unsolved(),
    );

    Measurement {
        rigid: rigid.volume,
        strained: deformed_est.volume,
        rigid_se: rigid.standard_error,
        strained_se: deformed_est.standard_error,
        max_t: strain.max_deviation_from_identity(),
        unsolved: strain.unsolved(),
    }
}

/// The clean relaxed alloy: strain must reduce the reported void and must not inflate it.
///
/// The reduction bound is the honest part. 10 % is generous against the measured 4.9 %, and it is
/// asserted *above* what the method achieves on purpose: if a change ever makes strain remove
/// substantially more shell, this fails and forces `design.md` §17.2 and the Method B status to be
/// revisited rather than letting the claim drift away from the measurement.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn a_relaxed_crystal_measures_a_shell_that_strain_only_creases() {
    let (Some(m), Some(f)) = (
        measurement("data.relaxed-40-HEA-reference", "relaxed clean"),
        frame("data.relaxed-40-HEA-reference"),
    ) else {
        return;
    };

    assert_eq!(
        m.unsolved, 0,
        "every node should solve on a 256k-atom solid"
    );
    // The fixture is relaxed, so the fit must find real deformation — a field that came back as
    // exactly identity would mean the neighbourhood solve silently no-ops — and it must stay small
    // enough to be thermal strain rather than a fit chasing local disorder.
    assert!(
        m.max_t > 1e-3,
        "the fit returned identity (max |T - I| = {}); the solve is not seeing the relaxed field",
        m.max_t
    );
    assert!(
        m.max_t < 0.15,
        "max |T - I| = {:.3} is not thermal strain on a solid at this lattice constant",
        m.max_t
    );
    assert!(
        m.strained <= m.rigid,
        "fitting strain inflated the void on a solid: {} -> {} A^3",
        m.rigid,
        m.strained
    );
    assert!(
        (m.rigid - m.strained) < 0.10 * m.rigid,
        "strain removed {:.1} % of the reported void, which is more than the roto-strain field (design §17.2) has been shown to do; \
         update design.md and the Method B status before raising this bound",
        100.0 * (m.rigid - m.strained) / m.rigid
    );
    // The residual is a thermal shell, not porosity: per atom it must stay far below the atomic
    // volume of an FCC site, `a^3/4`.
    let atomic = A * A * A / 4.0;
    let per_atom = m.rigid / f.natoms() as f64;
    assert!(
        per_atom < 0.10 * atomic,
        "rigid void is {per_atom:.3} A^3 per atom against an atomic volume of {atomic:.2} A^3; \
         that is porosity, not a shell"
    );
}

/// The defect fixture: strain must not selectively erase the defects.
///
/// A deformation field fitted to a neighbourhood containing a vacancy sees a coherent pull inward
/// and encodes it as contraction — indistinguishable from real lattice strain, and it closes the
/// hole. That is the failure mode that matters for defect counting.
///
/// It cannot be tested by a floor on the void volume, because strain removes ~5 % of the thermal
/// shell from the clean fixture too, and that confound is exactly as large as the effect being
/// hunted. What is testable is the difference: strain may remove the common shell from both
/// fixtures, but it must not remove more from the one carrying the defects. Anything more would be
/// strain preferentially closing holes, which is the failure.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn strain_does_not_selectively_erase_planted_defects() {
    let (Some(clean), Some(defective)) = (
        measurement("data.relaxed-40-HEA-reference", "clean"),
        measurement("data.relaxed-40-HEA-lastframe", "defective"),
    ) else {
        return;
    };

    assert!(
        defective.strained <= defective.rigid,
        "fitting strain inflated the void on the defect fixture: {} -> {} A^3",
        defective.rigid,
        defective.strained
    );

    let removed_clean = clean.rigid - clean.strained;
    let removed_defective = defective.rigid - defective.strained;
    let se = (clean.rigid_se.powi(2)
        + clean.strained_se.powi(2)
        + defective.rigid_se.powi(2)
        + defective.strained_se.powi(2))
    .sqrt();
    let excess = removed_defective - removed_clean;
    println!(
        "strain removed {removed_clean:.0} A^3 from clean and {removed_defective:.0} A^3 from \
         defective; the excess is {excess:.0} A^3 against a combined se of {se:.0} A^3"
    );
    assert!(
        excess < 3.0 * se,
        "strain removed {:.0} A^3 more from the defect fixture than from the clean one ({excess:.0} \
         vs {se:.0} A^3 of noise); a neighbourhood containing a vacancy is being fitted as \
         contraction",
        excess
    );
}

/// A sampled volume integral cannot resolve planted point defects, and that is the reason
/// Method A exists.
///
/// The two fixtures are the same alloy and the same relaxation, differing only by their point
/// defects. The volume difference is far inside the sampling error of the integral, so any test
/// claiming this measurement separates them is reading noise. Asserting the resolution limit keeps
/// that honest: it will fail if the sampling improves enough to matter, which is the point at which
/// this file should be replaced by a real per-grain comparison.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn a_volume_integral_cannot_resolve_planted_point_defects() {
    let (Some(c), Some(d)) = (
        measurement("data.relaxed-40-HEA-reference", "clean"),
        measurement("data.relaxed-40-HEA-lastframe", "defective"),
    ) else {
        return;
    };

    let se = (c.rigid_se.powi(2) + d.rigid_se.powi(2)).sqrt();
    let diff = (d.rigid - c.rigid).abs();
    println!(
        "rigid: |defective - clean| = {diff:.2} A^3, combined se = {se:.0} A^3 ({:.2} sigma)",
        diff / se
    );
    assert!(
        diff < 3.0 * se,
        "the volume integral now resolves the two fixtures ({diff:.0} A^3 vs se {se:.0} A^3); \
         this file is out of date and needs a per-grain or per-site comparison instead"
    );
    assert!(
        (d.strained - c.strained).abs()
            < 3.0 * (c.strained_se.powi(2) + d.strained_se.powi(2)).sqrt(),
        "the strained volume integral now separates the fixtures; revisit what this file claims"
    );
}
