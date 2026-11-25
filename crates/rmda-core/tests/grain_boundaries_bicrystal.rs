//! The grain-boundary pass on a real three-dimensional bicrystal.
//!
//! The unit tests in `analysis::grain_boundaries` pin the *rule* on one-dimensional
//! chains, where every neighbour relation is unambiguous. This file checks it on the
//! geometry it is actually for: two FCC blocks meeting across a plane, where the
//! seam is planar and its position is known exactly, so "which atoms are on the
//! boundary" has an analytic answer to be compared against rather than a
//! hand-counted one.
//!
//! Grain ids are assigned from the split plane directly instead of through PTM.
//! That is deliberate — the pass consumes a `Grain` column and does not care how it
//! was produced, so running the full structure-classifier path here would only add a
//! second thing that can be wrong.

use rmda_core::analysis::grain_boundaries::{GrainBoundaries, GB_COUNT, GB_FLAG};
use rmda_core::analysis::{Analysis, AnalysisContext, NeighborhoodContext, Summary};
use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::data::property::{DType, PropertyStore};
use rmda_core::lattice::fcc::Fcc;
use rmda_core::lattice::sites::SiteLattice;
use rmda_core::lattice::Lattice;

const A: f64 = 3.615;
/// FCC nearest-neighbour distance: `a/sqrt(2)`.
fn d_nn() -> f64 {
    A / 2.0f64.sqrt()
}

/// Rotation by `deg` about z.
fn about_z(deg: f64) -> [[f64; 3]; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// A two-block sample meeting across the plane `x = seam_x`, plus the grain ids.
///
/// Returns `(frame, grain_ids, seam_x)`. The left block is an FCC lattice; the right
/// block is a copy rotated `tilt` about z, built in its own frame and then
/// translated across — never rotated in place, because wrapping folds distant lattice
/// lines on top of each other and the result is not a crystal.
fn bicrystal(tilt: f64, cells: usize) -> (Frame, Vec<i64>, f64) {
    let len = cells as f64 * A;
    let seam_x = len;
    let cell = UnitCell::orthorhombic(2.0 * len, len, len);
    let sites = SiteLattice::build(&cell, &Fcc { a: A }.primitive(), [0.0; 3])
        .expect("lattice")
        .iter()
        .collect::<Vec<[f64; 3]>>();
    let r = about_z(tilt);
    let mut pos: Vec<[f64; 3]> = Vec::new();
    for s in sites {
        if s[0] < seam_x {
            pos.push(s);
        } else {
            let b = [s[0] - seam_x, s[1], s[2]];
            let mut p = [0.0f64; 3];
            for d in 0..3 {
                p[d] = r[d][0] * b[0] + r[d][1] * b[1] + r[d][2] * b[2];
            }
            p[0] += seam_x;
            pos.push(p);
        }
    }
    let n = pos.len();
    let ids: Vec<i64> = pos.iter().map(|p| i64::from(p[0] < seam_x)).collect();
    let mut pd = ParticleData::with_capacity(n);
    for (i, p) in pos.iter().enumerate() {
        pd.add_atom(*p, i as i64 + 1, 1);
    }
    (Frame::new(cell, pd), ids, seam_x)
}

fn run(a: &GrainBoundaries, frame: &Frame, ids: &[i64]) -> (Summary, PropertyStore) {
    let mut out = PropertyStore::new();
    out.register_flat(
        DType::I32,
        "Grain",
        1,
        ids.iter().map(|&g| g as f64).collect(),
    );
    let par = rmda_core::exec::Par::new(1);
    let nkeys = NeighborhoodContext { cell: None };
    let ctx = AnalysisContext {
        frame,
        nkeys: &nkeys,
        par: &par,
    };
    let mut s = Summary::default();
    a.run(&ctx, &mut out, &mut s).expect("boundary pass");
    (s, out)
}

fn scalar(s: &Summary, k: &str) -> f64 {
    s.scalars
        .iter()
        .find(|(n, _)| n == k)
        .map(|(_, v)| *v)
        .unwrap_or_else(|| panic!("no scalar {k}"))
}

fn flags(out: &PropertyStore) -> Vec<u8> {
    out.get(GB_FLAG).expect("flag").as_u8().unwrap().clone()
}

fn per_grain(out: &PropertyStore) -> Vec<i64> {
    out.get(GB_COUNT)
        .expect("count")
        .as_i32()
        .unwrap()
        .iter()
        .map(|&v| i64::from(v))
        .collect()
}

/// The seam is a plane, so no flagged atom may sit further from it than `w_db`.
/// A straggler here would mean the rule is picking up something other than the
/// boundary — a distorted lattice, an image artifact — and reporting it as seam.
#[test]
fn the_seam_is_planar_within_w_db() {
    let (frame, ids, seam_x) = bicrystal(30.0, 5);
    let w_db = 4.0;
    let a = GrainBoundaries {
        w_db: Some(w_db),
        ..Default::default()
    };
    let (s, out) = run(&a, &frame, &ids);
    let f = flags(&out);
    let pos = &frame.particles.positions;

    let flagged: Vec<usize> = (0..f.len()).filter(|&i| f[i] == 1).collect();
    assert!(!flagged.is_empty(), "a bicrystal has a seam to find");
    for &i in &flagged {
        let dx = (pos[i][0] - seam_x).abs();
        assert!(
            dx <= w_db,
            "atom {i} flagged at |x-seam| = {dx:.3} A, beyond w_db = {w_db}: \
             the seam is not planar, so the rule is flagging something else"
        );
    }
    // And the containment is not vacuous: the far halves are untouched.
    let deep_left = (0..f.len()).filter(|&i| pos[i][0] < seam_x - w_db).count();
    assert!(
        deep_left > 100,
        "fixture too small to be a test: {deep_left}"
    );
    for i in 0..f.len() {
        if (pos[i][0] - seam_x).abs() > w_db {
            assert_eq!(f[i], 0, "atom {i} is {w_db}+ A from the seam but flagged");
        }
    }
    assert!(scalar(&s, "gb_atoms_boundary") > 0.0);
    assert_eq!(scalar(&s, "gb_grains"), 2.0);
    assert_eq!(scalar(&s, "gb_atoms_unassigned"), 0.0);
}

/// The seam must have no holes. Every atom sitting within one nearest-neighbour
/// distance of the plane has an across-seam neighbour by construction, so a miss
/// there is a hole in the boundary, not a thin spot.
#[test]
fn the_seam_has_no_holes_near_the_plane() {
    let (frame, ids, seam_x) = bicrystal(30.0, 5);
    let a = GrainBoundaries {
        w_db: Some(4.0),
        ..Default::default()
    };
    let (_, out) = run(&a, &frame, &ids);
    let f = flags(&out);
    let pos = &frame.particles.positions;

    let mut checked = 0;
    for i in 0..f.len() {
        if (pos[i][0] - seam_x).abs() < d_nn() {
            checked += 1;
            assert_eq!(
                f[i],
                1,
                "atom {i} at |x-seam| = {:.3} A < d_nn = {:.3} A was not flagged, \
                 so the seam has a hole in it",
                (pos[i][0] - seam_x).abs(),
                d_nn()
            );
        }
    }
    assert!(
        checked > 100,
        "only {checked} atoms near the plane; fixture too small"
    );
}

/// The order dependence the rule guards against, on real geometry rather than a chain: the
/// same crystal numbered in the opposite order must give the same seam. The shipped
/// rule does; a half-list rule charges one grain's boundary to whichever grain holds
/// the lower atom ids, so this is the assertion that would catch it coming back.
#[test]
fn the_seam_does_not_depend_on_how_the_atoms_were_numbered() {
    let (frame, ids, _) = bicrystal(30.0, 5);
    let a = GrainBoundaries {
        w_db: Some(4.0),
        ..Default::default()
    };

    // A genuine renumbering: the atoms are physically reordered, so atom `i` of the
    // new frame is the same physical atom as atom `n-1-i` of the old, and it keeps its
    // grain id. Reversing only the ids would relabel the halves instead, which tests
    // nothing about ordering.
    let n = ids.len();
    let rev_ids: Vec<i64> = (0..n).rev().map(|i| ids[i]).collect();
    let mut pd = ParticleData::with_capacity(n);
    for i in (0..n).rev() {
        pd.add_atom(frame.particles.positions[i], i as i64 + 1, 1);
    }
    let reversed = Frame::new(frame.cell, pd);
    let (_, forward) = run(&a, &frame, &ids);
    let (_, backward) = run(&a, &reversed, &rev_ids);

    let ff = flags(&forward);
    let fb = flags(&backward);
    let mapped: Vec<u8> = (0..n).map(|i| fb[n - 1 - i]).collect();

    assert_eq!(
        ff, mapped,
        "reversing the atom order changed which atoms are on the seam"
    );
    // The per-grain numbers are the part an order-dependent rule moves, so
    // they are asserted separately rather than as a corollary of the above.
    assert_eq!(
        per_grain(&forward),
        per_grain(&backward),
        "per-grain boundary counts must not follow atom numbering"
    );
}

/// Each grain owns the half of the seam nearest to it. A tilt boundary is not a
/// mirror, so the counts need not be exactly equal — but they must be close, and
/// wildly lopsided counts are what an asymmetric rule produces.
#[test]
fn both_grains_own_roughly_half_the_seam() {
    let (frame, ids, _) = bicrystal(30.0, 6);
    let a = GrainBoundaries {
        w_db: Some(4.0),
        ..Default::default()
    };
    let (s, out) = run(&a, &frame, &ids);
    let counts = per_grain(&out);
    assert_eq!(counts.len(), 2, "two grains, two rows");
    let total = counts.iter().sum::<i64>() as f64;
    assert_eq!(total, scalar(&s, "gb_atoms_boundary"));
    let imbalance = (counts[0] - counts[1]).abs() as f64 / total;
    assert!(
        imbalance < 0.25,
        "seam ownership is lopsided: {counts:?} of {total}, relative imbalance {imbalance:.3}"
    );
}

/// A single crystal has no seam, and the pass must say so rather than finding one.
#[test]
fn a_single_crystal_has_no_seam() {
    let cells = 5;
    let len = cells as f64 * A;
    let cell = UnitCell::orthorhombic(len, len, len);
    let pos: Vec<[f64; 3]> = SiteLattice::build(&cell, &Fcc { a: A }.primitive(), [0.0; 3])
        .expect("lattice")
        .iter()
        .collect();
    let mut pd = ParticleData::with_capacity(pos.len());
    for (i, p) in pos.iter().enumerate() {
        pd.add_atom(*p, i as i64 + 1, 1);
    }
    let frame = Frame::new(cell, pd);
    let ids = vec![0i64; pos.len()];
    let (s, out) = run(&GrainBoundaries::new(), &frame, &ids);
    assert!(flags(&out).iter().all(|&b| b == 0));
    assert_eq!(scalar(&s, "gb_atoms_boundary"), 0.0);
    assert_eq!(scalar(&s, "gb_fraction_boundary"), 0.0);
    assert_eq!(per_grain(&out), vec![0]);
}

/// The reported slab thickness has to be consistent with `w_db`, because it is what
/// tells a user whether the exclusion width was the binding constraint.
#[test]
fn the_reported_slab_thickness_is_within_w_db() {
    for tilt in [15.0, 30.0, 45.0] {
        let (frame, ids, _) = bicrystal(tilt, 5);
        let w_db = 4.0;
        let a = GrainBoundaries {
            w_db: Some(w_db),
            ..Default::default()
        };
        let (s, _) = run(&a, &frame, &ids);
        let t = scalar(&s, "gb_slab_thickness");
        assert!(
            t > 0.0 && t <= w_db,
            "tilt {tilt}: slab thickness {t} should be in (0, w_db={w_db}]"
        );
    }
}

/// A narrow `w_db` finds a narrower seam. Without this the pass could be returning a
/// constant and the other tests would still pass.
#[test]
fn a_narrower_w_db_finds_a_thinner_seam() {
    let (frame, ids, _) = bicrystal(30.0, 6);
    let count = |w: f64| -> f64 {
        let a = GrainBoundaries {
            w_db: Some(w),
            ..Default::default()
        };
        scalar(&run(&a, &frame, &ids).0, "gb_atoms_boundary")
    };
    // Below the nearest-neighbour distance no atom has an across-seam neighbour at
    // all, so the seam vanishes; above it, more of the seam comes into range.
    assert_eq!(
        count(0.5),
        0.0,
        "w_db below d_nn cannot see across the seam"
    );
    let thin = count(d_nn() + 1e-6);
    let wide = count(4.0);
    assert!(
        thin > 0.0 && wide > thin,
        "seam should grow with w_db: {thin} at d_nn, {wide} at 4 A"
    );
}
