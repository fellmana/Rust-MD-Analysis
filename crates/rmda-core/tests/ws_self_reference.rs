//! Self-WS regressions against the reference structures.
//!
//! These are the tests that decide whether the fitter works: the two HEA fixtures are
//! 256k atoms of a relaxed FCC solid with a known number of point defects, and a defect
//! count that comes out right on 256k atoms says the region, the assignment and the
//! defect semantics all agree. Every synthetic fixture in `lattice::fit` is small enough
//! to be consistent by luck; these are not.
//!
//! Both fixtures are 40 lattice constants per edge, so a crystal that fills its box has
//! 100 cells of four sites and 256000 atoms exactly. The counts are asserted as literals
//! rather than recomputed from the lattice parameter, because a region that quietly
//! changes size with the gauge still balances atoms against sites -- it just reports
//! hundreds of vacancies that are not there.

use rmda_core::analysis::defects::explicit_ws::ExplicitWs;
use rmda_core::analysis::ws::WsSelf;
use rmda_core::data::frame::Frame;
use rmda_core::exec::Par;
use rmda_core::io::read_frames;
use rmda_core::lattice::fcc::Fcc;
use std::path::PathBuf;

mod common;

/// Sites in a perfect FCC block that fills its box: four basis atoms per cell. The
/// fixtures are 141.037 A on a side at a = 3.5259 A, which is 40 cells.
const CELLS_PER_EDGE: usize = 40;
const SITES: usize = 4 * CELLS_PER_EDGE * CELLS_PER_EDGE * CELLS_PER_EDGE;

/// Half the nearest-neighbour distance of the reference lattice, so it is the largest
/// tolerance that cannot misassign an atom.
const TOLERANCE: f64 = 1.19;

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

/// The nearest-neighbour distance implied by the box, which is the answer the estimator
/// has to recover: the fixture is a dense random alloy, so a lattice constant from the
/// volume is exact to six figures.
fn box_nn(f: &Frame) -> f64 {
    let (clo, chi) = f.cell.bounds();
    let l: Vec<f64> = (0..3).map(|d| chi[d] - clo[d]).collect();
    let a = (l[0] * l[1] * l[2] / (f.natoms() as f64 / 4.0)).cbrt();
    a / 2f64.sqrt()
}

#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn a_relaxed_crystal_registers_no_defects() {
    let Some(f) = frame("data.relaxed-40-HEA-reference") else {
        return;
    };
    let n = f.natoms();
    assert_eq!(n, SITES, "fixture is not the expected size");

    // The estimator is what every fit's starting scale comes from, and it has to be
    // unbiased or nothing downstream can recover: 4% low is enough to leave this same
    // crystal reporting half a million vacancies.
    let nn = rmda_core::lattice::fit::estimate_nn(&f.cell, &f.particles.positions[..n], box_nn(&f))
        .unwrap();
    let ratio = nn / box_nn(&f);
    assert!(
        (ratio - 1.0).abs() < 0.01,
        "nearest neighbour {} A against {} A from the box, ratio {ratio}",
        nn,
        box_nn(&f)
    );

    let r = WsSelf::new(Box::new(Fcc { a: 3.5 }))
        .run_region(&f.cell, &f.particles.positions[..n], &Par::new(8))
        .unwrap();
    assert_eq!(r.n_sites(), SITES, "{r:?}");
    assert_eq!(r.vacancy_count(), 0, "{r:?}");
    assert_eq!(r.interstitial_count(), 0, "{r:?}");
    assert!(r.mean_displacement() < 0.2, "{r:?}");
    assert!(r.max_displacement < TOLERANCE, "{r:?}");
}

#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn a_thermal_snapshot_recovers_its_planted_point_defects() {
    let Some(f) = frame("data.relaxed-40-HEA-lastframe") else {
        return;
    };
    let n = f.natoms();
    assert_eq!(n, SITES, "fixture is not the expected size");

    let r = WsSelf::new(Box::new(Fcc { a: 3.5 }))
        .run_region(&f.cell, &f.particles.positions[..n], &Par::new(8))
        .unwrap();
    assert_eq!(r.n_sites(), SITES, "{r:?}");
    assert_eq!(r.vacancy_count(), 7, "{r:?}");
    assert_eq!(r.interstitial_count(), 7, "{r:?}");
    assert!(r.mean_displacement() < 0.25, "{r:?}");
    // The worst-displaced atom sits just inside the tolerance, which is the point: it
    // is a real interstitial, and the fit has to stay tight enough to call it one.
    assert!(r.max_displacement < TOLERANCE, "{r:?}");
}

/// What the fitted lattice was, on the real data.
///
/// The two defect counts above are the acceptance criterion, but they are two integers
/// that happen to come out right; this pins *why*, so a run that reaches 0/0 by
/// compensating errors is distinguishable from one that reaches it because it found the
/// lattice. It is also the only place the fitted parameter is checked against an
/// independent measurement of the same quantity -- the box-implied spacing, which needs
/// no fitting at all and so cannot share a bug with the estimator.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn the_fitted_lattice_matches_the_box_implied_one() {
    let Some(f) = frame("data.relaxed-40-HEA-reference") else {
        return;
    };
    let n = f.natoms();
    let r = WsSelf::new(Box::new(Fcc { a: 3.5 }))
        .run_region(&f.cell, &f.particles.positions[..n], &Par::new(8))
        .unwrap();

    // `Fcc { a }` is a prototype whose scale is ignored, so the fitted `a` is the only
    // scale in play. 40 cells of it span the box.
    let a_fit = r.fit.cell.h.row(0)[0];
    let (clo, chi) = f.cell.bounds();
    let l = chi[0] - clo[0];
    let a_box = l / CELLS_PER_EDGE as f64;
    assert!(
        (a_fit / a_box - 1.0).abs() < 0.002,
        "fitted a = {a_fit} A against {a_box} A from the box"
    );

    // The region is 40 cells per axis, which is the same arithmetic as the site count
    // but caught here because it is the region's *shape*: a region that is 40x40x39
    // gives the same 256000 sites' worth of atoms only by accident.
    assert_eq!(r.fit.dims, [CELLS_PER_EDGE; 3], "{:?}", r.fit.dims);

    // And the fit converged rather than being cut off mid-search.
    assert!(
        r.mean_displacement() < 0.2 && r.fit.tolerance > 0.0,
        "{r:?}"
    );
}

/// Void clustering on the real thermal snapshot.
///
/// The counts are asserted as literals because the clustering is a *result*, not a
/// consequence of the defect counts: 7 vacancies that happen to be adjacent are a
/// different physical statement from 7 vacancies scattered through 256000 sites, and
/// only one of them is a void count. They were checked by hand against the fitted
/// lattice before being written down.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn voids_cluster_on_the_real_thermal_snapshot() {
    let Some(f) = frame("data.relaxed-40-HEA-lastframe") else {
        return;
    };
    let n = f.natoms();
    let ws = WsSelf::new(Box::new(Fcc { a: 3.5 }))
        .run_region(&f.cell, &f.particles.positions[..n], &Par::new(8))
        .unwrap();

    // At the default threshold of 1 no cluster is *demoted*, but adjacency still merges:
    // the 7 vacancies fall into 4 runs (3, 2, 1, 1). This is the check that the threshold
    // and the adjacency criterion are different things -- pinning 7 here would assert
    // that empty sites never join, which the distance criterion exists to prevent.
    let all = ws.count_voids(1).unwrap();
    assert_eq!(all.n_voids(), 4, "{all:?}");
    assert_eq!(all.n_void_sites(), 7, "{all:?}");
    assert_eq!(all.n_empty_sites(), 7, "{all:?}");
    assert_eq!(all.n_isolated_sites, 0, "{all:?}");

    // Two sites is where this data actually has something to say: 5 of the 7 vacancies
    // sit in clusters, 2 do not. A cell-offset adjacency criterion instead reports all 7
    // as one cluster, so this is the assertion that distinguishes the two.
    let paired = ws.count_voids(2).unwrap();
    assert_eq!(paired.n_voids(), 2, "{paired:?}");
    assert_eq!(paired.n_void_sites(), 5, "{paired:?}");
    assert_eq!(paired.n_isolated_sites, 2, "{paired:?}");

    // Nothing is lost either way: the site total is the vacancy count whatever the
    // threshold, and the cutoff is the fitted bond length.
    assert_eq!(paired.n_empty_sites(), 7, "{paired:?}");
    let a_fit = ws.fit.cell.h.row(0)[0];
    // The cutoff is the bond length plus `default_cutoff`'s slack. Checked against
    // `a/sqrt(2)` rather than against the value the function returns, because a check
    // that recomputes the code's own expression proves nothing. The prototype's nearest
    // distance and `a/sqrt(2)` differ in the last couple of bits, hence the tolerance;
    // a cell edge instead of a bond is out by 41%, so this still has teeth.
    assert!(
        (paired.cutoff / (a_fit / 2f64.sqrt()) - 1.05).abs() < 1e-7,
        "cutoff {} against {} x bond {}",
        paired.cutoff,
        1.05,
        a_fit / 2f64.sqrt()
    );

    // The largest void is three sites, so its radius is that volume as a sphere: three
    // Wigner-Seitz cells of the fitted lattice.
    let site_volume = ws.fit.cell.h.det().abs() / ws.assign.n_basis as f64;
    let expected = (3.0 * 3.0 * site_volume / (4.0 * std::f64::consts::PI)).cbrt();
    assert!(
        (paired.max_radius() - expected).abs() < 1e-9,
        "radius {} against {expected}",
        paired.max_radius()
    );

    // The relaxed reference is defect-free, so it has no voids at any threshold.
    let Some(ref_f) = frame("data.relaxed-40-HEA-reference") else {
        return;
    };
    let rn = ref_f.natoms();
    let ws = WsSelf::new(Box::new(Fcc { a: 3.5 }))
        .run_region(&ref_f.cell, &ref_f.particles.positions[..rn], &Par::new(8))
        .unwrap();
    for min in [1usize, 2, 64] {
        let c = ws.count_voids(min).unwrap();
        assert_eq!(c.n_voids(), 0, "threshold {min}: {c:?}");
        assert_eq!(c.n_empty_sites(), 0, "threshold {min}: {c:?}");
        assert_eq!(c.max_radius(), 0.0, "threshold {min}: {c:?}");
    }
}

/// Reference-configuration WS against the same fixtures.
///
/// The count asserted here is the one the self-fit void analysis reports for the same
/// frame, which is the point: a defect that survives both a fitted lattice and a
/// remembered configuration is a defect in the crystal, and one that appears in only
/// one is an artefact of that engine's reference.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn a_relaxed_crystal_is_clean_against_itself() {
    let Some(ref_f) = frame("data.relaxed-40-HEA-reference") else {
        return;
    };
    let n = ref_f.natoms();
    assert_eq!(n, SITES, "fixture is not the expected size");
    let r = ExplicitWs::new(ref_f.cell, ref_f.particles.positions[..n].to_vec())
        .run(&ref_f.cell, &ref_f.particles.positions[..n], &Par::new(4))
        .unwrap();
    assert_eq!(r.vacancy_count, 0, "{r:?}");
    assert_eq!(r.interstitial_count, 0, "{r:?}");
    assert_eq!(r.unmatched_count, 0, "{r:?}");
    // Zero displacement, not merely zero defects: the assignment has to find each
    // atom's own site. On 256000 atoms a search that slips by one periodic image
    // would still balance atoms against sites, so only this asserts it.
    assert_eq!(r.max_displacement, 0.0, "{r:?}");
    assert_eq!(r.occupancy.len(), n);
    assert!(
        r.occupancy.iter().all(|&o| o == 1),
        "every site claimed once"
    );
    // The measured bond, and the tolerance derived from it, are the explanation for
    // every other number this module reports.
    assert!(
        (r.tolerance / box_nn(&ref_f) - 1.05).abs() < 0.05,
        "tolerance {} against {} x the box bond",
        r.tolerance,
        1.05
    );
}

#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn the_planted_defects_survive_a_configuration_reference() {
    let (Some(ref_f), Some(cur_f)) = (
        frame("data.relaxed-40-HEA-reference"),
        frame("data.relaxed-40-HEA-lastframe"),
    ) else {
        return;
    };
    let n = ref_f.natoms();
    let r = ExplicitWs::new(ref_f.cell, ref_f.particles.positions[..n].to_vec())
        .run(
            &cur_f.cell,
            &cur_f.particles.positions[..cur_f.natoms()],
            &Par::new(4),
        )
        .unwrap();
    // The two fixtures are the same crystal at the same cell, so the defects are the
    // ones planted in it. Asserted as literals, like every other count here.
    assert_eq!(r.vacancy_count, 7, "{r:?}");
    assert_eq!(r.interstitial_count, 7, "{r:?}");
    assert_eq!(r.interstitial_site_count, 7, "{r:?}");
    assert_eq!(r.unmatched_count, 0, "{r:?}");
    assert_eq!(r.occupancy.len(), n);
    // A relaxed crystal has thermal displacement, and the reference is the same
    // crystal relaxed the other way, so a mean displacement of a few tenths of an
    // Angstrom is the expected answer rather than a defect count.
    assert!(r.mean_displacement > 0.1, "{r:?}");
    assert!(r.mean_displacement < box_nn(&ref_f), "{r:?}");
    assert_eq!(r.site_of_atom.len(), cur_f.natoms());
}
