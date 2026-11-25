//! Synthetic validation of the defect analysis (design §16.8).
//!
//! Every case plants a known defect population into a crystal built here, runs
//! PTM and then `analysis::defects`, and asserts the exact vacancy and
//! interstitial counts plus conservation. The cases are chosen to break a
//! rigid per-grain reference: thermal noise, a sheared slab with vacuum, a
//! rotated free cluster, a bicrystal and HCP.

use rmda_core::analysis::defects::{self, DefectsConfig, ReferenceMode, Status};
use rmda_core::analysis::ptm::{check, structure, Ptm};
use rmda_core::cell::{Coord3, Mat3, UnitCell};
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::exec::Par;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Normal deviate (Box–Muller).
    fn g(&mut self) -> f64 {
        let (u, v) = (self.f().max(1e-300), self.f());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Sites of `n` conventional cells of a motif with lattice constant `a`
/// (and `c` for HCP), as positions relative to the block's corner.
fn block(st: i32, a: f64, n: [usize; 3]) -> (Vec<Coord3>, [f64; 3]) {
    let s3 = 3.0f64.sqrt();
    let (cell, basis): ([f64; 3], Vec<Coord3>) = match st {
        structure::FCC => (
            [a, a, a],
            vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
        ),
        structure::BCC => ([a, a, a], vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        structure::HCP => (
            [a, a * s3, a * (8.0f64 / 3.0).sqrt()],
            vec![
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 1.0 / 3.0, 0.5],
                [0.5, 5.0 / 6.0, 0.5],
            ],
        ),
        _ => unreachable!(),
    };
    let mut out = Vec::new();
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                for b in &basis {
                    out.push([
                        (i as f64 + b[0]) * cell[0],
                        (j as f64 + b[1]) * cell[1],
                        (k as f64 + b[2]) * cell[2],
                    ]);
                }
            }
        }
    }
    (
        out,
        [
            n[0] as f64 * cell[0],
            n[1] as f64 * cell[1],
            n[2] as f64 * cell[2],
        ],
    )
}

fn rotation(axis: Coord3, deg: f64) -> Mat3 {
    let l = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (x, y, z) = (axis[0] / l, axis[1] / l, axis[2] / l);
    let (s, c) = deg.to_radians().sin_cos();
    let t = 1.0 - c;
    Mat3::from_rows(
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    )
}

struct Run {
    result: defects::DefectsResult,
}

/// PTM, then the defect analysis, with every atom in grain `grain_of`.
fn analyse(cell: UnitCell, pos: &[Coord3], grain_of: &[i32], threads: usize) -> Run {
    analyse_with(cell, pos, grain_of, threads, &DefectsConfig::default())
}

fn analyse_with(
    cell: UnitCell,
    pos: &[Coord3],
    grain_of: &[i32],
    threads: usize,
    cfg: &DefectsConfig,
) -> Run {
    let mut p = ParticleData::with_capacity(pos.len());
    for (i, q) in pos.iter().enumerate() {
        p.add_atom(*q, i as i64 + 1, 1);
    }
    let frame = Frame::new(cell, p);
    let par = Par::new(threads);
    let view = Ptm {
        rmsd_cutoff: 0.1,
        flags: check::OVITO_DEFAULT,
    }
    .compute(&frame.cell, &frame.particles.positions, &par)
    .unwrap();
    let result = defects::run(
        &frame.cell,
        &frame.particles.positions,
        grain_of,
        &view,
        cfg,
        &par,
    )
    .unwrap();
    Run { result }
}

fn report(r: &Run) -> String {
    r.result
        .grains
        .iter()
        .map(|g| {
            format!(
                "grain {} {:?} ref={} patches={}/{} phase={:.3} st={} pairs={:.3} rms={:.3} off={:.4} snapped={:?} int={:?} gb={:?} t={:?} o={:?}\n",
                g.grain,
                g.status,
                g.reference,
                g.n_patches_fitted,
                g.n_patches,
                g.phase_agreement,
                g.structure,
                g.pair_fraction,
                g.rms_displacement,
                g.offlattice_fraction,
                g.snapped,
                g.interior,
                g.boundary,
                g.t,
                g.origin
            )
        })
        .collect()
}

fn jitter(pos: &mut [Coord3], sigma: f64, seed: u64) {
    let mut r = Rng(seed);
    for p in pos.iter_mut() {
        for v in p.iter_mut() {
            *v += sigma * r.g();
        }
    }
}

/// Remove `k` atoms spread through the block, away from its edges.
fn remove_spread(pos: &mut Vec<Coord3>, k: usize, keep_inside: impl Fn(&Coord3) -> bool) -> usize {
    let cand: Vec<usize> = (0..pos.len()).filter(|&i| keep_inside(&pos[i])).collect();
    let mut idx: Vec<usize> = (0..k).map(|t| cand[(t * 7919 + 13) % cand.len()]).collect();
    idx.sort_unstable();
    idx.dedup();
    for &i in idx.iter().rev() {
        pos.swap_remove(i);
    }
    idx.len()
}

#[test]
fn thermal_bcc_with_vacancies_and_dumbbells() {
    let a = 3.165;
    let (mut pos, l) = block(structure::BCC, a, [12, 12, 12]);
    let cell = UnitCell::orthorhombic(l[0], l[1], l[2]);
    // Two <110> dumbbells: an extra atom on two sites, split along [110].
    for &s in &[1000usize, 2000] {
        let p = pos[s];
        let d = 0.35 * a / 2f64.sqrt();
        pos[s] = [p[0] - d, p[1] - d, p[2]];
        pos.push([p[0] + d, p[1] + d, p[2]]);
    }
    let removed = remove_spread(&mut pos, 5, |p| p[0] > 3.0 && p[0] < l[0] - 3.0);
    assert_eq!(removed, 5);
    jitter(&mut pos, 0.06, 1);
    let r = analyse(cell, &pos, &vec![0; pos.len()], 4);
    let g = &r.result.grains[0];
    assert_eq!(g.status, Status::Measured, "{}", report(&r));
    assert_eq!(g.interior.vacancies, 5, "{}", report(&r));
    assert_eq!(g.interior.interstitials, 2, "{}", report(&r));
    assert_eq!(g.snapped, [true, true, true]);
    assert_eq!(r.result.interstitials.len(), 2);
}

/// The fitted lattice is unbiased under thermal noise. Fitting `T` to each
/// atom's `z` nearest neighbours keeps the shortest bonds of a noisy crystal
/// (long first-shell bonds lose their rank to short second-shell ones): on
/// this free block at σ = 0.12 Å the bond comes out 1 % short, the block
/// fails `gate_rms_displacement` (0.42 Å), and at 0.15 Å the withheld count
/// is 49 V / 6 I for 10 planted. Nothing snaps an open axis,
/// so the bias shows in full.
#[test]
fn a_noisy_free_block_fits_an_unbiased_lattice() {
    let a = 3.165;
    let bond = a * 3f64.sqrt() / 2.0;
    for (sigma, seed) in [(0.12, 7u64), (0.15, 8)] {
        let (mut pos, l) = block(structure::BCC, a, [16, 16, 16]);
        let cell = UnitCell::orthorhombic(l[0], l[1], l[2]).with_pbc([false; 3]);
        let removed = remove_spread(&mut pos, 10, |p| {
            (0..3).all(|d| p[d] > 2.0 * a && p[d] < l[d] - 2.0 * a)
        });
        jitter(&mut pos, sigma, seed);
        let cfg = DefectsConfig {
            reference: ReferenceMode::Global,
            ..DefectsConfig::default()
        };
        let r = analyse_with(cell, &pos, &vec![0; pos.len()], 4, &cfg);
        let g = &r.result.grains[0];
        assert!(
            (g.bond / bond - 1.0).abs() < 1e-3,
            "σ {sigma}: bond {} for {bond}\n{}",
            g.bond,
            report(&r)
        );
        assert_eq!(g.status, Status::Measured, "σ {sigma}\n{}", report(&r));
        assert_eq!(
            (g.interior.vacancies, g.interior.interstitials),
            (removed as u64, 0),
            "σ {sigma}\n{}",
            report(&r)
        );
    }
}

#[test]
fn sheared_and_tetragonal_slab_in_vacuum_is_clean_and_finds_its_vacancies() {
    // A rigid reference cannot fit this and reports a moire of Frenkel pairs.
    let a = 3.2;
    let (mut pos, l) = block(structure::BCC, a, [10, 10, 14]);
    for p in pos.iter_mut() {
        p[0] += 0.026 * p[2]; // simple shear u_x ∝ z
        p[2] *= 1.008; // tetragonal
        p[2] += 20.0;
    }
    // x/y periodic, the slab sits in vacuum along a periodic z.
    let cell = UnitCell::orthorhombic(l[0], l[1], l[2] * 3.0);
    let removed = remove_spread(&mut pos, 3, |p| p[2] > 35.0 && p[2] < 50.0);
    jitter(&mut pos, 0.05, 2);
    let r = analyse(cell, &pos, &vec![0; pos.len()], 2);
    let g = &r.result.grains[0];
    assert_eq!(g.status, Status::Measured, "{}", report(&r));
    assert_eq!(g.snapped, [true, true, false], "{}", report(&r));
    assert_eq!(g.interior.vacancies, removed as u64, "{}", report(&r));
    assert_eq!(g.interior.interstitials, 0, "{}", report(&r));
}

#[test]
fn rotated_free_fcc_cluster_counts_only_its_interior_vacancies() {
    let a = 3.615;
    let (block_pos, _) = block(structure::FCC, a, [9, 9, 9]);
    let r0 = rotation([0.3, -0.5, 0.8], 37.0);
    let mut pos: Vec<Coord3> = block_pos
        .iter()
        .map(|p| {
            let c = [p[0] - 16.0, p[1] - 16.0, p[2] - 16.0];
            let v = r0.mul_v(c);
            [v[0] + 40.0, v[1] + 40.0, v[2] + 40.0]
        })
        .collect();
    let removed = remove_spread(&mut pos, 4, |p| (0..3).all(|d| (p[d] - 40.0).abs() < 7.0));
    jitter(&mut pos, 0.05, 3);
    let cell = UnitCell::orthorhombic(80.0, 80.0, 80.0).with_pbc([false; 3]);
    let r = analyse(cell, &pos, &vec![0; pos.len()], 3);
    let g = &r.result.grains[0];
    assert_eq!(g.status, Status::Measured, "{}", report(&r));
    assert_eq!(
        g.interior.vacancies + g.boundary.vacancies,
        removed as u64,
        "{}",
        report(&r)
    );
    assert_eq!(
        g.interior.interstitials + g.boundary.interstitials,
        0,
        "{}",
        report(&r)
    );
}

#[test]
fn bicrystal_interior_counts_are_exact_and_boundaries_conserve() {
    let a = 3.165;
    let (base, l) = block(structure::BCC, a, [14, 14, 16]);
    let cell = UnitCell::orthorhombic(l[0], l[1], l[2]);
    let rot = rotation([0.0, 0.0, 1.0], 21.0);
    let half = l[2] / 2.0;
    // Lower half: the base lattice. Upper half: the base rotated about z by
    // 21 degrees, clipped to the box footprint.
    let mut pos: Vec<Coord3> = base.iter().copied().filter(|p| p[2] < half).collect();
    let cx = l[0] / 2.0;
    let cy = l[1] / 2.0;
    let (big, _) = block(structure::BCC, a, [28, 28, 8]);
    for p in big {
        let c = [p[0] - 2.0 * cx, p[1] - 2.0 * cy, p[2]];
        let v = rot.mul_v(c);
        let q = [v[0] + cx, v[1] + cy, v[2] + half + 0.5 * a];
        if q[0] >= 0.0 && q[0] < l[0] && q[1] >= 0.0 && q[1] < l[1] && q[2] < l[2] {
            pos.push(q);
        }
    }
    let mut grain: Vec<i32> = pos.iter().map(|p| i32::from(p[2] >= half)).collect();
    // One vacancy deep in each grain.
    let mut drop = Vec::new();
    for (target, gid) in [(half * 0.5, 0), (half * 1.5, 1)] {
        let i = (0..pos.len())
            .filter(|&i| grain[i] == gid)
            .min_by(|&i, &j| {
                let d = |k: usize| {
                    (pos[k][0] - cx).powi(2)
                        + (pos[k][1] - cy).powi(2)
                        + (pos[k][2] - target).powi(2)
                };
                d(i).total_cmp(&d(j))
            })
            .unwrap();
        drop.push(i);
    }
    drop.sort_unstable();
    for &i in drop.iter().rev() {
        pos.swap_remove(i);
        grain.swap_remove(i);
    }
    jitter(&mut pos, 0.04, 4);
    let r = analyse(cell, &pos, &grain, 4);
    for g in &r.result.grains {
        // The rotated grain does not tile the box in x/y, so it is not
        // commensurate there and its edges against the box are boundary.
        assert!(
            matches!(g.status, Status::Measured | Status::GateFailed(_)),
            "{}",
            report(&r)
        );
        assert_eq!(g.interior.residual(), 0);
        assert_eq!(g.boundary.residual(), 0);
    }
    assert_eq!(
        r.result.grains[0].status,
        Status::Measured,
        "{}",
        report(&r)
    );
    assert_eq!(r.result.grains[0].interior.vacancies, 1, "{}", report(&r));
    assert_eq!(
        r.result.grains[0].interior.interstitials,
        0,
        "{}",
        report(&r)
    );
}

#[test]
fn rotated_hcp_cluster_is_clean() {
    let a = 3.23;
    let (block_pos, _) = block(structure::HCP, a, [10, 6, 6]);
    let r0 = rotation([0.7, 0.2, -0.4], 52.0);
    let mut pos: Vec<Coord3> = block_pos
        .iter()
        .map(|p| {
            let v = r0.mul_v([p[0] - 16.0, p[1] - 16.0, p[2] - 16.0]);
            [v[0] + 40.0, v[1] + 40.0, v[2] + 40.0]
        })
        .collect();
    let removed = remove_spread(&mut pos, 2, |p| (0..3).all(|d| (p[d] - 40.0).abs() < 6.0));
    jitter(&mut pos, 0.04, 5);
    let cell = UnitCell::orthorhombic(80.0, 80.0, 80.0).with_pbc([false; 3]);
    let r = analyse(cell, &pos, &vec![0; pos.len()], 2);
    let g = &r.result.grains[0];
    assert_eq!(g.structure, structure::HCP, "{}", report(&r));
    assert_eq!(g.status, Status::Measured, "{}", report(&r));
    assert_eq!(
        g.interior.vacancies + g.boundary.vacancies,
        removed as u64,
        "{}",
        report(&r)
    );
    assert_eq!(
        g.interior.interstitials + g.boundary.interstitials,
        0,
        "{}",
        report(&r)
    );
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let a = 3.165;
    let (mut pos, l) = block(structure::BCC, a, [8, 8, 8]);
    let cell = UnitCell::orthorhombic(l[0], l[1], l[2]);
    remove_spread(&mut pos, 3, |_| true);
    jitter(&mut pos, 0.08, 6);
    let g = vec![0; pos.len()];
    let one = analyse(cell, &pos, &g, 1);
    let many = analyse(cell, &pos, &g, 6);
    let (a1, a2) = (&one.result.grains[0], &many.result.grains[0]);
    assert_eq!(a1.interior, a2.interior);
    assert_eq!(a1.t, a2.t);
    assert_eq!(a1.origin, a2.origin);
}

/// Pure bending of a free BCC block: x along the arc, z radial, total
/// rotation `deg` over the block's length. One affine lattice misses the ends
/// by many Å; patches follow it.
#[test]
fn a_bent_crystal_is_measured_with_patches_and_finds_its_vacancies() {
    let a = 3.165;
    let (mut pos, l) = block(structure::BCC, a, [40, 8, 12]);
    // Away from every free surface: a surface-layer vacancy is not a bulk one.
    let removed = remove_spread(&mut pos, 6, |p| {
        p[0] > 10.0
            && p[0] < l[0] - 10.0
            && p[1] > 5.0
            && p[1] < l[1] - 5.0
            && p[2] > 8.0
            && p[2] < l[2] - 8.0
    });
    let theta = 6.0f64.to_radians();
    let r = l[0] / theta;
    let zc = l[2] / 2.0;
    for p in pos.iter_mut() {
        let phi = (p[0] - l[0] / 2.0) / r;
        let rr = r + (p[2] - zc);
        *p = [
            rr * phi.sin() + 80.0,
            p[1] + 20.0,
            rr * phi.cos() - r + 40.0,
        ];
    }
    jitter(&mut pos, 0.05, 21);
    let cell = UnitCell::orthorhombic(160.0, 70.0, 80.0).with_pbc([false; 3]);

    let global = analyse_with(
        cell,
        &pos,
        &vec![0; pos.len()],
        4,
        &DefectsConfig {
            reference: ReferenceMode::Global,
            ..Default::default()
        },
    );
    assert!(
        matches!(global.result.grains[0].status, Status::GateFailed(_)),
        "one lattice should not fit a 6-degree bend: {}",
        report(&global)
    );

    let r = analyse(cell, &pos, &vec![0; pos.len()], 4);
    let g = &r.result.grains[0];
    assert_eq!(g.reference, "patch", "{}", report(&r));
    assert_eq!(g.status, Status::Measured, "{}", report(&r));
    assert_eq!(
        g.interior.vacancies + g.boundary.vacancies,
        removed as u64,
        "{}",
        report(&r)
    );
    assert_eq!(
        g.interior.interstitials + g.boundary.interstitials,
        0,
        "{}",
        report(&r)
    );
    assert!(g.phase_agreement > 0.99, "{}", report(&r));
}

/// A 2 % lattice-parameter gradient along x.
#[test]
fn a_graded_lattice_parameter_is_measured_with_patches() {
    let a = 3.165;
    let (mut pos, l) = block(structure::BCC, a, [40, 8, 8]);
    let removed = remove_spread(&mut pos, 4, |p| p[0] > 10.0 && p[0] < l[0] - 10.0);
    for p in pos.iter_mut() {
        let x = p[0];
        p[0] = x * (1.0 + 0.02 * x / (2.0 * l[0])) + 10.0;
        p[1] += 10.0;
        p[2] += 10.0;
    }
    jitter(&mut pos, 0.05, 22);
    let cell = UnitCell::orthorhombic(170.0, 50.0, 50.0).with_pbc([false; 3]);
    let r = analyse(cell, &pos, &vec![0; pos.len()], 4);
    let g = &r.result.grains[0];
    assert_eq!(g.status, Status::Measured, "{}", report(&r));
    assert_eq!(
        g.interior.vacancies + g.boundary.vacancies,
        removed as u64,
        "{}",
        report(&r)
    );
    assert_eq!(
        g.interior.interstitials + g.boundary.interstitials,
        0,
        "{}",
        report(&r)
    );
}

/// Patch seams must not make defects: a thermal periodic crystal forced into
/// patch mode counts exactly what the global reference counts.
#[test]
fn patch_seams_create_no_defects() {
    let a = 3.165;
    let (mut pos, l) = block(structure::BCC, a, [24, 24, 24]);
    let removed = remove_spread(&mut pos, 5, |_| true);
    jitter(&mut pos, 0.08, 23);
    let cell = UnitCell::orthorhombic(l[0], l[1], l[2]);
    let cfg = DefectsConfig {
        reference: ReferenceMode::Patch,
        patch_size: Some(20.0),
        ..Default::default()
    };
    let r = analyse_with(cell, &pos, &vec![0; pos.len()], 4, &cfg);
    let g = &r.result.grains[0];
    assert_eq!(g.reference, "patch");
    assert!(g.n_patches_fitted > 20, "{}", report(&r));
    assert_eq!(g.status, Status::Measured, "{}", report(&r));
    assert_eq!(g.interior.vacancies, removed as u64, "{}", report(&r));
    assert_eq!(g.interior.interstitials, 0, "{}", report(&r));

    let one = analyse_with(cell, &pos, &vec![0; pos.len()], 1, &cfg);
    assert_eq!(one.result.grains[0].interior, g.interior);
}
