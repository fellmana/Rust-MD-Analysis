//! Validation of grain segmentation (design §15.5).

use rmda_core::analysis::grain_seg::{segment, SegConfig, SegMode, Segmentation};
use rmda_core::analysis::ptm::{check, structure, Ptm};
use rmda_core::cell::{Coord3, Mat3, UnitCell};
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::exec::Par;
use rmda_core::neighbor::{KnnIndex, KnnScratch};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn g(&mut self) -> f64 {
        let (u, v) = (self.f().max(1e-300), self.f());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
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

/// Lattice points of a structure filling a cube of half-width `r` around the
/// origin, rotated by `rot`.
fn lattice_ball(st: i32, a: f64, r: f64, rot: &Mat3) -> Vec<Coord3> {
    let s3 = 3.0f64.sqrt();
    let (cell, basis): ([f64; 3], Vec<Coord3>) = match st {
        structure::FCC => (
            [a; 3],
            vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
        ),
        structure::BCC => ([a; 3], vec![[0.0; 3], [0.5; 3]]),
        structure::HCP => (
            [a, a * s3, a * (8.0f64 / 3.0).sqrt()],
            vec![
                [0.0; 3],
                [0.5, 0.5, 0.0],
                [0.0, 1.0 / 3.0, 0.5],
                [0.5, 5.0 / 6.0, 0.5],
            ],
        ),
        _ => unreachable!(),
    };
    let n: Vec<i32> = cell.iter().map(|c| (r / c).ceil() as i32 + 1).collect();
    let mut out = Vec::new();
    for i in -n[0]..=n[0] {
        for j in -n[1]..=n[1] {
            for k in -n[2]..=n[2] {
                for b in &basis {
                    let p = [
                        (i as f64 + b[0]) * cell[0],
                        (j as f64 + b[1]) * cell[1],
                        (k as f64 + b[2]) * cell[2],
                    ];
                    let q = rot.mul_v(p);
                    if q.iter().all(|v| v.abs() <= r) {
                        out.push(q);
                    }
                }
            }
        }
    }
    out
}

fn ptm_then_segment(
    cell: UnitCell,
    pos: &[Coord3],
    cfg: &SegConfig,
    threads: usize,
) -> Segmentation {
    let mut p = ParticleData::with_capacity(pos.len());
    for (i, q) in pos.iter().enumerate() {
        p.add_atom(*q, i as i64 + 1, 1);
    }
    let frame = Frame::new(cell, p);
    let par = Par::new(threads);
    let ptm = Ptm {
        rmsd_cutoff: 0.1,
        flags: check::OVITO_DEFAULT,
    }
    .compute(&frame.cell, &frame.particles.positions, &par)
    .unwrap();
    segment(&frame.cell, &frame.particles.positions, &ptm, cfg, &par).unwrap()
}

fn jitter(pos: &mut [Coord3], sigma: f64, seed: u64) {
    let mut r = Rng(seed);
    for p in pos.iter_mut() {
        for v in p.iter_mut() {
            *v += sigma * r.g();
        }
    }
}

fn summary(s: &Segmentation) -> String {
    format!(
        "{} grains, {} crystallites: sizes {:?}, pairs {:?}",
        s.grains.len(),
        s.n_crystallites,
        s.grains.iter().map(|g| g.size).collect::<Vec<_>>(),
        s.pairs
            .iter()
            .map(|p| (p.a, p.b, p.bonds, p.disorientation))
            .collect::<Vec<_>>()
    )
}

#[test]
fn single_crystals_near_the_fundamental_zone_boundary_are_one_grain() {
    // 44.6 degrees about z puts thermally jittered FCC/BCC atoms on both sides
    // of the cubic zone boundary (45 degrees): a naive disorientation splits such grains.
    let cases = [
        (structure::FCC, 3.615, rotation([0.02, -0.01, 1.0], 44.6)),
        (structure::BCC, 3.165, rotation([0.01, 0.03, 1.0], 44.6)),
        (structure::HCP, 3.2, rotation([0.4, -0.7, 0.2], 71.0)),
    ];
    for (st, a, rot) in cases {
        let mut pos: Vec<Coord3> = lattice_ball(st, a, 16.0, &rot)
            .into_iter()
            .map(|p| [p[0] + 30.0, p[1] + 30.0, p[2] + 30.0])
            .collect();
        jitter(&mut pos, 0.07, st as u64);
        let cell = UnitCell::orthorhombic(60.0, 60.0, 60.0).with_pbc([false; 3]);
        let s = ptm_then_segment(cell, &pos, &SegConfig::default(), 4);
        assert_eq!(s.grains.len(), 1, "structure {st}: {}", summary(&s));
        assert_eq!(s.grains[0].structure, st);
        let unassigned = s.grain.iter().filter(|&&g| g < 0).count();
        assert_eq!(unassigned, 0, "structure {st}: {}", summary(&s));
    }
}

#[test]
fn bicrystals_split_into_two_with_the_right_misorientation() {
    for deg in [8.0, 15.0, 40.0] {
        let base = rotation([0.3, 0.1, 0.9], 12.0);
        let twist = rotation([1.0, 1.0, 1.0], deg).mul(&base);
        let mut pos: Vec<Coord3> = lattice_ball(structure::BCC, 3.165, 15.0, &base)
            .into_iter()
            .filter(|p| p[2] < 0.0)
            .collect();
        pos.extend(
            lattice_ball(structure::BCC, 3.165, 15.0, &twist)
                .into_iter()
                .filter(|p| p[2] >= 1.0),
        );
        for p in pos.iter_mut() {
            for v in p.iter_mut() {
                *v += 30.0;
            }
        }
        jitter(&mut pos, 0.05, 11);
        let cell = UnitCell::orthorhombic(60.0, 60.0, 60.0).with_pbc([false; 3]);
        let s = ptm_then_segment(cell, &pos, &SegConfig::default(), 4);
        assert_eq!(s.grains.len(), 2, "{deg}: {}", summary(&s));
        let p = s
            .pairs
            .iter()
            .find(|p| p.a == 0 && p.b == 1)
            .expect("the grains touch");
        // Cubic symmetry can map a rotation about <111> onto a smaller one;
        // below 30 degrees it is the angle itself.
        if deg < 30.0 {
            assert!(
                (p.disorientation - deg).abs() < 1.0,
                "{deg}: {}",
                summary(&s)
            );
        }
    }
}

/// Periodic BCC Voronoi polycrystal with known grain labels.
/// Also returns, per atom, `d2 - d1`: how much nearer its own seed is than the
/// next, a proxy for depth inside the grain.
fn voronoi(
    seeds: &[(Coord3, Mat3)],
    l: f64,
    a: f64,
) -> (UnitCell, Vec<Coord3>, Vec<usize>, Vec<f64>) {
    let cell = UnitCell::orthorhombic(l, l, l);
    let mic = |d: f64| d - l * (d / l).round();
    let nearest = |p: &Coord3| -> (usize, f64, f64) {
        let mut d: Vec<(f64, usize)> = seeds
            .iter()
            .enumerate()
            .map(|(k, (s, _))| {
                let v = [mic(p[0] - s[0]), mic(p[1] - s[1]), mic(p[2] - s[2])];
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2], k)
            })
            .collect();
        d.sort_by(|x, y| x.0.total_cmp(&y.0));
        (d[0].1, d[0].0.sqrt(), d[1].0.sqrt())
    };
    let mut pos = Vec::new();
    let mut lab = Vec::new();
    let mut depth = Vec::new();
    for (k, (s, rot)) in seeds.iter().enumerate() {
        // One periodic copy around the seed: [-l/2, l/2) per axis covers the
        // box exactly once, so wrapping creates no duplicates.
        for p in lattice_ball(structure::BCC, a, 0.5 * l, rot)
            .into_iter()
            .filter(|p| p.iter().all(|v| *v < 0.5 * l))
        {
            let q = [
                (s[0] + p[0]).rem_euclid(l),
                (s[1] + p[1]).rem_euclid(l),
                (s[2] + p[2]).rem_euclid(l),
            ];
            let (owner, d1, d2) = nearest(&q);
            if owner == k {
                pos.push(q);
                lab.push(k);
                depth.push(d2 - d1);
            }
        }
    }
    // Drop atoms that collide with another grain's atom at the boundary.
    let idx = KnnIndex::build(&cell, &pos, &Par::new(4)).unwrap();
    let mut s = KnnScratch::default();
    let mut nb = Vec::new();
    let keep: Vec<bool> = (0..pos.len())
        .map(|i| {
            idx.knn(i, pos[i], 1, &mut s, &mut nb);
            !nb.iter()
                .any(|&(j, _, d2)| d2 < (0.7 * a).powi(2) && (j as usize) < i)
        })
        .collect();
    let pos2: Vec<Coord3> = pos
        .iter()
        .zip(&keep)
        .filter(|x| *x.1)
        .map(|x| *x.0)
        .collect();
    let lab2: Vec<usize> = lab
        .iter()
        .zip(&keep)
        .filter(|x| *x.1)
        .map(|x| *x.0)
        .collect();
    let depth2: Vec<f64> = depth
        .iter()
        .zip(&keep)
        .filter(|x| *x.1)
        .map(|x| *x.0)
        .collect();
    (cell, pos2, lab2, depth2)
}

/// `k` seeds with random orientations, every pair more than `min_deg` apart,
/// so the ground truth is `k` distinguishable grains.
fn separated_seeds(k: usize, l: f64, min_deg: f64, seed: u64) -> Vec<(Coord3, Mat3)> {
    use rmda_core::analysis::orient;
    use rmda_core::lattice::fit::mat_to_quat;
    let g = orient::group(structure::BCC).unwrap();
    let mut r = Rng(seed);
    let mut out: Vec<(Coord3, Mat3)> = Vec::new();
    while out.len() < k {
        let rot = rotation([r.g(), r.g(), r.g()], 180.0 * r.f());
        let q = mat_to_quat(&rot);
        if out
            .iter()
            .all(|(_, o)| orient::disorientation(&mat_to_quat(o), &q, &g) > min_deg)
        {
            out.push(([r.f() * l, r.f() * l, r.f() * l], rot));
        }
    }
    out
}

#[test]
fn a_voronoi_polycrystal_is_recovered() {
    let l = 64.0;
    let seeds = separated_seeds(5, l, 15.0, 42);
    let (cell, mut pos, truth, depth) = voronoi(&seeds, l, 3.165);
    jitter(&mut pos, 0.06, 7);
    let s = ptm_then_segment(cell, &pos, &SegConfig::default(), 4);
    assert_eq!(s.grains.len(), 5, "{}", summary(&s));
    // Each found grain maps to one true grain; count interior atoms (more than
    // ~2 Å from the boundary) that disagree. At the boundary the Voronoi
    // "truth" is arbitrary.
    let interior = |i: usize| depth[i] > 4.0;
    let mut wrong = 0;
    let mut total = 0;
    for g in 0..s.grains.len() as i32 {
        let mut votes = [0usize; 5];
        for (i, &x) in s.grain.iter().enumerate() {
            if x == g && interior(i) {
                votes[truth[i]] += 1;
            }
        }
        let best = *votes.iter().max().unwrap();
        wrong += votes.iter().sum::<usize>() - best;
        total += votes.iter().sum::<usize>();
    }
    let frac = wrong as f64 / total as f64;
    assert!(
        frac < 0.01,
        "{wrong} of {total} interior atoms mislabelled: {}",
        summary(&s)
    );
}

/// `SegMode::Auto` (OVITO's graph clustering with the automatic threshold).
fn auto() -> SegConfig {
    SegConfig {
        mode: SegMode::Auto,
        ..Default::default()
    }
}

#[test]
fn auto_mode_keeps_thermal_single_crystals_whole() {
    let cases = [
        (structure::FCC, 3.615, rotation([0.02, -0.01, 1.0], 44.6)),
        (structure::BCC, 3.165, rotation([0.01, 0.03, 1.0], 44.6)),
    ];
    for (st, a, rot) in cases {
        let mut pos: Vec<Coord3> = lattice_ball(st, a, 16.0, &rot)
            .into_iter()
            .map(|p| [p[0] + 30.0, p[1] + 30.0, p[2] + 30.0])
            .collect();
        jitter(&mut pos, 0.07, st as u64);
        let cell = UnitCell::orthorhombic(60.0, 60.0, 60.0).with_pbc([false; 3]);
        let s = ptm_then_segment(cell, &pos, &auto(), 4);
        assert_eq!(s.grains.len(), 1, "structure {st}: {}", summary(&s));
        assert_eq!(s.grain.iter().filter(|&&g| g < 0).count(), 0);
        assert!(s.merge_threshold.is_some_and(f64::is_finite));
    }
}

#[test]
fn auto_mode_recovers_a_voronoi_polycrystal() {
    let l = 64.0;
    let seeds = separated_seeds(5, l, 15.0, 42);
    let (cell, mut pos, truth, depth) = voronoi(&seeds, l, 3.165);
    jitter(&mut pos, 0.06, 7);
    let s = ptm_then_segment(cell, &pos, &auto(), 4);
    assert_eq!(s.grains.len(), 5, "{}", summary(&s));
    let mut wrong = 0;
    let mut total = 0;
    for g in 0..5 {
        let mut votes = [0usize; 5];
        for (i, &x) in s.grain.iter().enumerate() {
            if x == g && depth[i] > 4.0 {
                votes[truth[i]] += 1;
            }
        }
        wrong += votes.iter().sum::<usize>() - votes.iter().max().unwrap();
        total += votes.iter().sum::<usize>();
    }
    assert!(wrong * 100 < total, "{wrong} of {total}: {}", summary(&s));
    assert_eq!(s.grain.iter().filter(|&&g| g < 0).count(), 0);
}

#[test]
fn auto_mode_does_not_depend_on_thread_count() {
    let l = 48.0;
    let seeds = separated_seeds(3, l, 15.0, 9);
    let (cell, mut pos, _, _) = voronoi(&seeds, l, 3.165);
    jitter(&mut pos, 0.08, 3);
    let a = ptm_then_segment(cell, &pos, &auto(), 1);
    let b = ptm_then_segment(cell, &pos, &auto(), 6);
    assert_eq!(a.grain, b.grain);
    assert_eq!(a.merge_threshold, b.merge_threshold);
}

#[test]
fn segmentation_does_not_depend_on_thread_count() {
    let l = 48.0;
    let seeds = separated_seeds(3, l, 15.0, 9);
    let (cell, mut pos, _, _) = voronoi(&seeds, l, 3.165);
    jitter(&mut pos, 0.08, 3);
    let a = ptm_then_segment(cell, &pos, &SegConfig::default(), 1);
    let b = ptm_then_segment(cell, &pos, &SegConfig::default(), 6);
    assert_eq!(a.grain, b.grain);
}

/// FCC as close-packed layers along z with stacking `seq` (letters A, B, C),
/// periodic in x and y, free in z; nearest-neighbour distance `d`.
fn stacked(seq: &str, nx: usize, ny: usize, d: f64) -> (UnitCell, Vec<Coord3>) {
    let (lx, ly) = (nx as f64 * d, ny as f64 * d * 3f64.sqrt());
    let dz = d * (2.0f64 / 3.0).sqrt();
    let mut pos = Vec::new();
    for (k, c) in seq.chars().enumerate() {
        // Shift of layer B and C along the in-plane [1/3, 2/3] of the
        // primitive cell, in the rectangular cell's Cartesian terms.
        let s = match c {
            'A' => 0.0,
            'B' => 1.0,
            _ => 2.0,
        };
        let (ox, oy) = (s * 0.5 * d, s * d / (2.0 * 3f64.sqrt()));
        for i in 0..nx {
            for j in 0..ny {
                for (fx, fy) in [(0.0, 0.0), (0.5, 0.5)] {
                    let x = (i as f64 + fx) * d + ox;
                    let y = (j as f64 + fy) * d * 3f64.sqrt() + oy;
                    pos.push([x.rem_euclid(lx), y.rem_euclid(ly), 10.0 + k as f64 * dz]);
                }
            }
        }
    }
    let lz = 20.0 + seq.len() as f64 * dz;
    (
        UnitCell::orthorhombic(lx, ly, lz).with_pbc([true, true, false]),
        pos,
    )
}

#[test]
fn an_intrinsic_stacking_fault_stays_inside_one_fcc_grain() {
    // ABCABC|BCABC...: the two layers next to the missing A are HCP.
    let seq = format!("{}{}", "ABC".repeat(4), "BCA".repeat(4));
    let (cell, mut pos) = stacked(&seq, 12, 7, 2.556);
    jitter(&mut pos, 0.05, 5);
    let s = ptm_then_segment(cell, &pos, &SegConfig::default(), 4);
    assert_eq!(s.grains.len(), 1, "{}", summary(&s));
    assert_eq!(s.grains[0].structure, structure::FCC);
    assert_eq!(s.grain.iter().filter(|&&g| g < 0).count(), 0);

    // Without the close-packed map the fault planes split the crystal.
    let cfg = SegConfig {
        stacking_faults: false,
        ..Default::default()
    };
    let s = ptm_then_segment(cell, &pos, &cfg, 4);
    assert!(s.grains.len() >= 2, "{}", summary(&s));
}

#[test]
fn a_sigma3_twin_is_two_grains_sixty_degrees_apart() {
    // ...ABC|BAC...: the C layer at the mirror is HCP and joins one twin.
    let seq = format!("{}{}", "ABC".repeat(5), "BAC".repeat(5));
    let (cell, mut pos) = stacked(&seq, 12, 7, 2.556);
    jitter(&mut pos, 0.05, 6);
    let s = ptm_then_segment(cell, &pos, &SegConfig::default(), 4);
    assert_eq!(s.grains.len(), 2, "{}", summary(&s));
    assert!(s.grains.iter().all(|g| g.structure == structure::FCC));
    let p = s.pairs.iter().find(|p| p.a == 0 && p.b == 1).unwrap();
    assert!((p.disorientation - 60.0).abs() < 1.0, "{}", summary(&s));
    assert_eq!(s.grain.iter().filter(|&&g| g < 0).count(), 0);
    // The twins are the halves: every atom more than a layer from the mirror
    // is in its own half's grain.
    let dz = 2.556 * (2.0f64 / 3.0).sqrt();
    let mirror = 10.0 + 14.0 * dz;
    let lower = s.grain[pos.iter().position(|p| p[2] < mirror - 3.0 * dz).unwrap()];
    for (p, &g) in pos.iter().zip(&s.grain) {
        if (p[2] - mirror).abs() > 1.5 * dz {
            assert_eq!(g == lower, p[2] < mirror, "atom at z = {:.2}", p[2]);
        }
    }
}
