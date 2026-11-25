//! Fitting an affine site lattice to a crystal (design §16.3b).
//!
//! 1. **Seed.** A rotation from the grain's mean PTM orientation and a scale
//!    from its mean PTM first-neighbour distance give `T₀ = R·a·C`. Template-frame
//!    conventions differ between structures, so the few plausible frames are
//!    scored on neighbour vectors and the best is kept; a seed is never
//!    trusted unscored.
//! 2. **Homogeneous fit.** Every first-shell neighbour vector `Δx` of a
//!    crystalline atom is matched to its ideal fractional shell vector
//!    `v = round_S(T⁻¹Δx)`, and `T = (Σ Δx vᵀ)(Σ v vᵀ)⁻¹` is solved by least
//!    squares. All nine components are free, which is what a rigid
//!    rotation-plus-scale fit lacks: shear and tetragonal strain. Pairs whose
//!    residual exceeds a quarter bond are dropped and the fit repeated.
//!    Difference vectors carry no origin, so this needs none.
//! 3. **Origin.** Seeded from atoms assumed to sit on each basis site in
//!    turn, scored by mean squared displacement over a sample, then refined
//!    by the mean displacement.
//! 4. **Commensurability.** On a periodic axis the crystal spans, the box
//!    vector must be a lattice translation, or the lattice does not close on
//!    itself and the seam reads as a plane of defects. `T` is snapped so that
//!    it is (Mason *et al.*, PRM 5, 095403, `T = A·n⁻¹`), keeping the fitted
//!    action on any open direction.

use super::lattice::{AffineLattice, Motif};
use crate::analysis::ptm::structure;
use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::{Error, Result};
use crate::exec::Par;

/// A neighbour vector's match to the ideal shell is accepted below this
/// residual, as a fraction of the bond.
const PAIR_TOLERANCE: f64 = 0.25;

/// Atoms the seed and origin scores look at.
const SCORE_SAMPLE: usize = 20_000;

/// Snap only when the box is this close to a whole number of translations.
const SNAP_TOLERANCE: f64 = 0.02;

/// And, for a grain's global lattice, at most this many first-neighbour
/// distances in absolute terms: in a large box 2 % is several bonds, enough
/// to snap a lattice that is not periodic at all (a grain meeting its own
/// image across a translation fault) and strain it into a plane of false
/// defects.
const SNAP_TOLERANCE_BONDS: f64 = 0.25;

/// Neighbours queried per atom for a fit of `T`: twice the coordination,
/// so the first shell is never truncated. Taking only the `z` nearest keeps
/// the shortest bonds of a noisy crystal (its long first-shell bonds lose
/// their rank to short second-shell ones), which biases `T` short: −1 % at
/// σ = 0.12 Å in BCC W. Second-shell vectors never match: two neighbour
/// vectors of one atom differ by an interatomic distance, ≥ 1 bond, far
/// beyond [`PAIR_TOLERANCE`].
pub(crate) fn fit_neighbours(z: usize) -> usize {
    2 * z
}

/// Inputs of a fit: the grain's (unwrapped) positions and, per crystalline
/// atom, its neighbour vectors (up to [`fit_neighbours`] each).
pub struct FitInput<'a> {
    pub motif: &'a Motif,
    /// Grain atoms' positions, unwrapped so the grain is contiguous.
    pub positions: &'a [Coord3],
    /// Neighbour displacement vectors of the crystalline atoms, flattened.
    pub pairs: &'a [Coord3],
    /// First-shell pairs those atoms should have: per atom, its same-grain
    /// atoms among its `z` nearest. The denominator of the match fractions,
    /// since `pairs` also holds second-shell vectors.
    pub expected_pairs: usize,
    /// Mean PTM orientation `(w, x, y, z)` of the grain.
    pub orientation: [f64; 4],
    /// Mean PTM interatomic (first-neighbour) distance of the grain. PTM's
    /// lattice constant is not used: for HCP it is the FCC-equivalent value.
    pub bond: f64,
    /// Per periodic axis, the box vector's lattice translation (fractional),
    /// when the caller knows it from a walk around the box. The snap then
    /// uses it instead of the nearest translation of the averaged `T`, and
    /// only the relative tolerance applies: a grain strained unevenly around
    /// the box has an average `T` that misses its own closure by up to half a
    /// bond, where rounding can pick the wrong translation.
    pub translations: [Option<Coord3>; 3],
}

/// A fitted lattice and how well it fits.
#[derive(Debug, Clone)]
pub struct Fit {
    pub lattice: AffineLattice,
    /// Fraction of the expected first-shell pairs matched within
    /// [`PAIR_TOLERANCE`].
    pub pair_fraction: f64,
    /// Neighbour pairs used in the final least-squares step.
    pub pairs_used: usize,
    /// RMS residual of matched pairs, Å.
    pub pair_rms: f64,
    /// Per periodic axis: whether `T` was snapped to the box.
    pub snapped: [bool; 3],
    /// Per snapped axis, the lattice translation (fractional) of its box
    /// vector, for the patches to snap to the same one.
    pub snap_translation: [Option<Coord3>; 3],
    /// Largest relative change the snap made to a box-vector image.
    pub snap_mismatch: f64,
    /// Per requested periodic axis: distance, Å, from the box vector to the
    /// nearest lattice translation before snapping (`NaN` elsewhere).
    pub snap_residual: [f64; 3],
    /// The same as a vector: box vector minus that translation, Å.
    pub snap_offset: [Coord3; 3],
}

/// Fit `T` and the origin. `snap[d]` asks for commensurability on axis `d`.
pub fn fit(input: &FitInput, cell: &UnitCell, snap: [bool; 3], par: &Par) -> Result<Fit> {
    let m = input.motif;
    if input.pairs.is_empty() || input.positions.is_empty() {
        return Err(Error::invalid("no crystalline neighbour pairs to fit"));
    }
    let a0 = input.bond / m.nn;
    if !(a0.is_finite() && a0 > 0.0) {
        return Err(Error::invalid("no usable interatomic distance"));
    }

    // 1. Seed: score each plausible frame on a sample of pairs.
    let rot = crate::lattice::fit::quat_to_mat(input.orientation);
    let scaled = |r: &Mat3| -> Mat3 {
        let c = m.cell;
        let mut s = r.mul(&c);
        for row in s.m.iter_mut() {
            for v in row.iter_mut() {
                *v *= a0;
            }
        }
        s
    };
    let mut frames = vec![rot, rot.transposed()];
    if m.structure == structure::HCP || m.structure == structure::DHCP {
        // Hexagonal templates may put the in-plane neighbour along x or y.
        for r in [rot, rot.transposed()] {
            for deg in [30.0f64, 90.0] {
                frames.push(r.mul(&rot_z(deg.to_radians())));
            }
        }
    }
    let stride = (input.pairs.len() / SCORE_SAMPLE).max(1);
    let sample: Vec<Coord3> = input.pairs.iter().step_by(stride).copied().collect();
    // First-shell pairs per listed pair, to turn a sample's matches into a
    // fraction of the pairs expected.
    let per_pair = input.expected_pairs.max(1) as f64 / input.pairs.len() as f64;
    let mut best: Option<(f64, Mat3)> = None;
    for f in &frames {
        let t = scaled(f);
        let score = (match_fraction(m, &t, &sample, a0 * m.nn) / per_pair).min(1.0);
        if best.as_ref().map_or(true, |b| score > b.0) {
            best = Some((score, t));
        }
    }
    let (seed_score, mut t) = best.unwrap();
    if seed_score < 0.5 {
        return Err(Error::invalid(format!(
            "orientation seed matches only {:.0}% of neighbour vectors",
            100.0 * seed_score
        )));
    }

    // 2. Homogeneous least squares, re-matching against the improved T.
    for _ in 0..4 {
        let stats = accumulate(m, &t, input.pairs, par);
        if stats.n < 3 {
            return Err(Error::invalid("too few matched neighbour pairs"));
        }
        let b_inv = stats
            .vv
            .inverse()
            .ok_or_else(|| Error::invalid("degenerate neighbour-vector set"))?;
        t = stats.xv.mul(&b_inv);
    }

    // 4 (before the origin, which depends on it). Commensurability.
    let (t, snapped, snap_mismatch, snap_offset, snap_translation) =
        snap_to_box(m, t, cell, snap, SNAP_TOLERANCE_BONDS, input.translations);

    // 3. Origin.
    let lattice = AffineLattice::new(m.clone(), t, [0.0; 3])
        .ok_or_else(|| Error::invalid("singular fitted cell"))?;
    let lattice = fit_origin(lattice, input.positions, cell);

    let final_stats = accumulate(m, &lattice.t, input.pairs, par);
    Ok(Fit {
        pair_fraction: (final_stats.n as f64 / input.expected_pairs.max(1) as f64).min(1.0),
        pairs_used: final_stats.n,
        pair_rms: (final_stats.r2 / final_stats.n.max(1) as f64).sqrt(),
        snapped,
        snap_translation,
        snap_mismatch,
        snap_residual: snap_offset.map(|o| (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt()),
        snap_offset,
        lattice,
    })
}

/// Least-squares `T` from neighbour pairs, starting from `t`; `None` when too
/// few pairs match.
pub(crate) fn refine(
    m: &Motif,
    mut t: Mat3,
    pairs: &[Coord3],
    iters: usize,
    par: &Par,
) -> Option<(Mat3, PairStats)> {
    let mut last = None;
    for _ in 0..iters {
        let s = accumulate(m, &t, pairs, par);
        if s.n < 12 {
            return None;
        }
        t = s.xv.mul(&s.vv.inverse()?);
        last = Some(s);
    }
    last.map(|s| (t, s))
}

fn rot_z(t: f64) -> Mat3 {
    let (s, c) = t.sin_cos();
    Mat3::from_rows([c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0])
}

/// The nearest ideal shell vector to fractional `u`.
#[inline]
fn nearest_shell(m: &Motif, u: Coord3) -> Coord3 {
    let mut best = m.shell_vectors[0];
    let mut bd = f64::INFINITY;
    for v in &m.shell_vectors {
        let d = (u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2) + (u[2] - v[2]).powi(2);
        if d < bd {
            bd = d;
            best = *v;
        }
    }
    best
}

fn match_fraction(m: &Motif, t: &Mat3, pairs: &[Coord3], bond: f64) -> f64 {
    let Some(ti) = t.inverse() else {
        return 0.0;
    };
    let tol2 = (PAIR_TOLERANCE * bond).powi(2);
    let ok = pairs
        .iter()
        .filter(|dx| {
            let v = nearest_shell(m, ti.mul_v(**dx));
            let p = t.mul_v(v);
            (p[0] - dx[0]).powi(2) + (p[1] - dx[1]).powi(2) + (p[2] - dx[2]).powi(2) < tol2
        })
        .count();
    ok as f64 / pairs.len().max(1) as f64
}

#[derive(Clone, Copy)]
pub(crate) struct PairStats {
    /// `Σ Δx vᵀ`
    pub xv: Mat3,
    /// `Σ v vᵀ`
    pub vv: Mat3,
    pub n: usize,
    pub r2: f64,
}

impl Default for PairStats {
    fn default() -> Self {
        Self {
            xv: Mat3 { m: [[0.0; 3]; 3] },
            vv: Mat3 { m: [[0.0; 3]; 3] },
            n: 0,
            r2: 0.0,
        }
    }
}

/// Sums over matched pairs, per fixed chunk and merged in chunk order, so the
/// fit is bit-identical for any thread count.
pub(crate) fn accumulate(m: &Motif, t: &Mat3, pairs: &[Coord3], par: &Par) -> PairStats {
    let Some(ti) = t.inverse() else {
        return PairStats::default();
    };
    let bond = {
        let l = m
            .shell_vectors
            .iter()
            .map(|v| {
                let c = t.mul_v(*v);
                (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
            })
            .sum::<f64>();
        l / m.shell_vectors.len() as f64
    };
    let tol2 = (PAIR_TOLERANCE * bond).powi(2);
    let parts = par.map_chunks(pairs.len(), |a, b| {
        let mut s = PairStats::default();
        for dx in &pairs[a..b] {
            let v = nearest_shell(m, ti.mul_v(*dx));
            let p = t.mul_v(v);
            let r2 = (p[0] - dx[0]).powi(2) + (p[1] - dx[1]).powi(2) + (p[2] - dx[2]).powi(2);
            if r2 >= tol2 {
                continue;
            }
            for r in 0..3 {
                for c in 0..3 {
                    s.xv.m[r][c] += dx[r] * v[c];
                    s.vv.m[r][c] += v[r] * v[c];
                }
            }
            s.n += 1;
            s.r2 += r2;
        }
        s
    });
    let mut out = PairStats::default();
    for s in parts {
        for r in 0..3 {
            for c in 0..3 {
                out.xv.m[r][c] += s.xv.m[r][c];
                out.vv.m[r][c] += s.vv.m[r][c];
            }
        }
        out.n += s.n;
        out.r2 += s.r2;
    }
    out
}

/// Snap `t` so that each requested periodic box vector is a lattice
/// translation, when it is within 2 % and `max_bonds` first-neighbour
/// distances of one. Returns the new `t`, which axes were snapped, the largest
/// relative correction, the per-axis residual (box vector minus nearest
/// translation, Å; `NaN` where not requested) and the snapped translations.
///
/// An axis with a `known` translation snaps to it, within the relative
/// tolerance alone (see [`FitInput::translations`]).
pub(crate) fn snap_to_box(
    m: &Motif,
    t: Mat3,
    cell: &UnitCell,
    want: [bool; 3],
    max_bonds: f64,
    known: [Option<Coord3>; 3],
) -> (Mat3, [bool; 3], f64, [Coord3; 3], [Option<Coord3>; 3]) {
    let mut residual = [[f64::NAN; 3]; 3];
    let mut chosen = [None; 3];
    let Some(ti) = t.inverse() else {
        return (t, [false; 3], 0.0, residual, chosen);
    };
    let h = cell.h_matrix();
    let nn = m.shell_vectors.first().map_or(f64::INFINITY, |&f| {
        let v = t.mul_v(f);
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    });
    let mut snapped = [false; 3];
    let mut worst = 0.0f64;
    // (box vector A_d, its lattice translation N_d) for each snapped axis.
    let mut pairs: Vec<(Coord3, Coord3)> = Vec::new();
    for d in 0..3 {
        if !(want[d] && cell.pbc[d]) {
            continue;
        }
        let a = *h.row(d);
        let f = ti.mul_v(a);
        // Nearest translation: integer cell plus one of the motif's pure
        // translations.
        let mut best: Option<(f64, Coord3)> = None;
        for tr in &m.translations {
            let n = [
                (f[0] - tr[0]).round() + tr[0],
                (f[1] - tr[1]).round() + tr[1],
                (f[2] - tr[2]).round() + tr[2],
            ];
            let p = t.mul_v(n);
            let e = ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2) + (p[2] - a[2]).powi(2)).sqrt();
            if best.map_or(true, |b| e < b.0) {
                best = Some((e, n));
            }
        }
        let (mut e, mut n) = best.unwrap();
        let mut limit = max_bonds * nn;
        if let Some(k) = known[d] {
            let p = t.mul_v(k);
            e = ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2) + (p[2] - a[2]).powi(2)).sqrt();
            n = k;
            limit = f64::INFINITY;
        }
        let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let rel = e / len.max(1e-300);
        let p = t.mul_v(n);
        residual[d] = [a[0] - p[0], a[1] - p[1], a[2] - p[2]];
        if rel <= SNAP_TOLERANCE && e <= limit {
            snapped[d] = true;
            chosen[d] = Some(n);
            worst = worst.max(rel);
            pairs.push((a, n));
        }
    }
    if pairs.is_empty() {
        return (t, snapped, 0.0, residual, chosen);
    }
    // Complete the snapped directions with fitted ones: a fractional
    // direction orthogonal (in Cartesian terms) to the snapped box vectors
    // keeps its fitted image.
    let mut cols_x: Vec<Coord3> = pairs.iter().map(|p| p.0).collect();
    let mut cols_n: Vec<Coord3> = pairs.iter().map(|p| p.1).collect();
    let mut extra = Vec::new();
    match cols_x.len() {
        1 => {
            let a = cols_x[0];
            let u = any_perpendicular(a);
            extra.push(u);
            extra.push(cross(a, u));
        }
        2 => extra.push(cross(cols_x[0], cols_x[1])),
        _ => {}
    }
    for e in extra {
        let f = ti.mul_v(e);
        cols_x.push(t.mul_v(f));
        cols_n.push(f);
    }
    let x = Mat3::from_rows(
        [cols_x[0][0], cols_x[1][0], cols_x[2][0]],
        [cols_x[0][1], cols_x[1][1], cols_x[2][1]],
        [cols_x[0][2], cols_x[1][2], cols_x[2][2]],
    );
    let n = Mat3::from_rows(
        [cols_n[0][0], cols_n[1][0], cols_n[2][0]],
        [cols_n[0][1], cols_n[1][1], cols_n[2][1]],
        [cols_n[0][2], cols_n[1][2], cols_n[2][2]],
    );
    match n.inverse() {
        Some(ni) => (x.mul(&ni), snapped, worst, residual, chosen),
        None => (t, [false; 3], 0.0, residual, [None; 3]),
    }
}

fn cross(a: Coord3, b: Coord3) -> Coord3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn any_perpendicular(a: Coord3) -> Coord3 {
    let e = if a[0].abs() < a[1].abs().max(a[2].abs()) {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let c = cross(a, e);
    let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let s = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() / l;
    [c[0] * s, c[1] * s, c[2] * s]
}

/// Origin: try each of a few atoms on each basis site, keep the lowest mean
/// squared displacement over a sample, then shift by the mean displacement
/// twice.
fn fit_origin(mut lat: AffineLattice, positions: &[Coord3], cell: &UnitCell) -> AffineLattice {
    let sample: Vec<Coord3> = super::place_sample(cell, positions, |x| x, SCORE_SAMPLE);
    let msd = |l: &AffineLattice| -> f64 {
        sample.iter().map(|x| l.nearest(*x).2).sum::<f64>() / sample.len() as f64
    };
    let seeds = 8.min(sample.len());
    let mut best: Option<(f64, Coord3)> = None;
    for s in 0..seeds {
        let x = sample[s * sample.len() / seeds];
        for b in lat.motif.basis.clone() {
            let tb = lat.t.mul_v(b);
            lat.origin = [x[0] - tb[0], x[1] - tb[1], x[2] - tb[2]];
            let e = msd(&lat);
            if best.map_or(true, |bb| e < bb.0) {
                best = Some((e, lat.origin));
            }
        }
    }
    lat.origin = best.map_or([0.0; 3], |b| b.1);
    for _ in 0..2 {
        let mut mean = [0.0; 3];
        for x in &sample {
            let (_, dr, _) = lat.nearest(*x);
            for a in 0..3 {
                mean[a] += dr[a];
            }
        }
        for (o, m) in lat.origin.iter_mut().zip(mean) {
            *o += m / sample.len() as f64;
        }
    }
    lat
}
