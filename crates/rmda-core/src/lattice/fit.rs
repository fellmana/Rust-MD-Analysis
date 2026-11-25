//! Reference-lattice fitting from the data itself (design §16.3b).
//!
//! [`WsRef`](crate::analysis::ws::WsRef) needs a reference that is already known
//! exactly: a lattice type, a scale, an origin, and a commensurability
//! guarantee. That is right for a single crystal whose reference you already
//! hold, and unusable for the two cases that matter most here — a polycrystal,
//! and an interface between two materials — because one global lattice cannot
//! describe either. So this module fits the reference *per region* from the
//! atoms of that region.
//!
//! # Why not axis sign flips
//!
//! A box-aligned reference (Wigner–Seitz against a lattice fitted to the frame,
//! Nordlund & Averback, PRB 57, 7556 (1998)) can be rescued from a poor fit by
//! running all eight axis sign flips and keeping whichever reports the fewest
//! vacancies. Two assumptions behind that fail on real data, and the flips
//! cannot repair either:
//!
//! - A lattice origin found by binning the x, y and z *projections*
//!   independently is only meaningful for a crystal aligned with the box. A
//!   rotated crystal smears the projections, the "minimum lattice position" it
//!   returns is noise, and every count that follows is off.
//! - A hand-written parity table over cell indices describes an axis-aligned
//!   lattice and nothing else. An HCP table additionally requires the box to be
//!   commensurate with `a·√3` along y, which a cubic box generally is not.
//!
//! A sign change is not a rotation, so the eight flips contain no orientation
//! that is not already axis-aligned, and HCP has no cubic symmetry for them to
//! exploit. An HCP crystal in any other orientation therefore fails all eight
//! times and reports roughly an order of magnitude too many defects.
//!
//! # Orientation is searched from a seed, never globally
//!
//! What replaces the flips is minimisation over two continuous quantities:
//! orientation and origin, refined by coordinate descent.
//!
//! Orientation starts from an externally supplied seed — a PTM orientation
//! quaternion, the natural per-grain source (§11.3, §14.2). Seeds come first
//! because they are free and already accurate to well under a degree.
//!
//! There is deliberately **no** global orientation search here, and that is an
//! evidence-based decision rather than an omission. A coarse Euler-angle scan
//! plus descent was implemented and measured on a crystal rotated by 34°: the
//! scan is 1728 candidates at 30° resolution, and every one of the eight
//! best-scoring candidates still showed ~104 collisions, i.e. the ranking was
//! choosing arbitrarily among rotations that were all wrong. The objective is
//! flat there — at a gross orientation error, which atoms collide is
//! essentially random, so there is no gradient to descend and no basin to fall
//! into. Increasing the resolution does not rescue it either; 4096 candidates
//! that are all wrong is still 4096 wrong answers, and an affordable scan is
//! necessarily far too coarse to be conclusive.
//!
//! So a fit started from the identity can only improve *locally*. That is fine
//! for an axis-aligned region and useless for a rotated one. [`fit_region`]
//! reports `n_sites` against `n_atoms` and a mean displacement precisely so a
//! caller can notice this rather than believe a confident wrong answer; see
//! `unseeded_fit_of_a_rotated_grain_is_not_silently_wrong`.
//!
//! # Scale comes from the measured spacing
//!
//! The cell is sized from the nearest-neighbour distance measured on the atoms
//! ([`estimate_nn`]). Two alternatives are wrong, and both fail on the same
//! case — a region that does not fill its box:
//!
//! - Searching integer cell counts and scoring each by defect count is circular:
//!   the score is minimised by a lattice too small to hold the atoms, since every
//!   missing atom becomes a "vacancy" the count is trying to minimise. On a
//!   subsample it is catastrophic: it picked a 6×6×6 cell for a region of 2048
//!   atoms where 8×8×8 was correct.
//! - Density over the region is shape dependent. The region around a 14 A grain
//!   tilted by 0.6 rad held three times the sites the grain had atoms, so the
//!   estimate came out 1.55× too coarse, and no amount of local refinement recovers
//!   that: the fit reported thousands of phantom vacancies.
//!
//! A bond length is shape independent, which is the whole point — it says nothing
//! about how much space the atoms are spread over. The `nn` argument is a hint
//! for the neighbour search and the matching tolerance, never a fitted value.
//!
//! Scale and origin are coupled: a wrong scale makes the best origin look bad,
//! and a wrong origin makes the best scale look bad. They are alternated, and
//! the order matters. The spacing estimate already pins the scale, so the origin
//! is searched first and the scale is only ever polished afterwards. Sweeping
//! scale against a lattice that is not yet aligned inverts the logic, because
//! shrinking a lattice drags its sites towards the atoms and scores better; that
//! inverted order settled on the edge of its bracket, 1.25× too coarse, on both
//! an aligned block and a tilted grain. A coarse log scan feeds golden section
//! rather than the reverse, because a fine local search cannot step out of the
//! basin belonging to a wrong alignment.
//!
//! # The search runs on a sample, the score does not
//!
//! The searches see at most [`FitOptions::search_sample`] atoms, taken as evenly
//! spaced indices rather than a random draw: deterministic, so a failure can be
//! reproduced, and unbiased in position, which matters because LAMMPS writes
//! atoms in domain order.
//!
//! That is only sound because the quantity being searched is mean displacement.
//! Defect counts are *not* subsampleable — a subsample cannot see a missing atom
//! — so the winning reference is always re-scored over every atom, and every
//! number `fit_region` reports comes from that full pass.
//!
//! # Sites: the region is the cells the atoms occupy
//!
//! Vacancies need a finite region. For a region that fills its box that is the box,
//! wrapped periodically. For a finite grain it is the *integer cells* the atoms'
//! nearest lattice points fall in -- not the bounding box of the atoms, which for a
//! tilted grain is not a whole number of cells in any direction.
//!
//! That distinction is not cosmetic. The box around a 14 A grain tilted by 0.6 rad
//! holds 546 lattice sites where the grain has 256 atoms, and all 290 of the excess
//! are reported as vacancies against an exact reference. Bounding the region by cell
//! indices instead gives 256 sites and none, and the atom count against the site
//! count stays exact for a rotated grain rather than only for an aligned one.
//!
//! The region is gauge-covariant: a lattice translation sends every atom's cell index
//! to a different index and leaves the span alone. It is not, however, invariant, and
//! FCC's lattice has more translations than the cell vectors -- (a/2, a/2, 0) is one.
//! The origin search can stop on any of them, and on an 8-cell block the difference
//! between gauges is 8 cells against 9 in each direction, so 544 sites with no atom
//! behind them. The gauge is therefore chosen rather than inherited: every origin that
//! maps the lattice onto itself is tried and the one whose region is tightest wins.
//!
//! Two things bound the enumeration. Sites are indexed per cell, so a candidate
//! whose scale has diverged is refused rather than allowed to allocate: the cell
//! count is a function of the fitted scale, so a wrong scale becomes an enormous
//! but perfectly representable allocation. And the search enumerates only the
//! bounded sample's region, so its memory is a function of the sample rather than
//! of the run.
//!
//! The final pass is the one that scales with the region, and it holds two counters:
//! one `u32` per *enumerated* site and one `u32` per *in-region* site. With the
//! region pinned to whole cells the two are close, which is the point -- a counter
//! that tracks the box instead would scale with the box.
//!
//! If either bound stops being acceptable, the escape is to stop materialising
//! the site set: stream the region in chunks so the full enumeration never
//! exists, and score each chunk against the atoms inside it. Recorded as a
//! future option, not yet built.
//!
use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::{Error, Result};
use crate::lattice::{Lattice, PrimitiveCell};
use std::sync::atomic::{AtomicU32, Ordering};

/// Nearest-neighbour distance of a primitive cell, by checking its basis atoms
/// against the 26 neighbouring images.
///
/// Exact for a primitive cell, unlike `h.row(1).len()` and friends, and it is
/// what turns a measured bond length into a cell scale.
pub fn nn_distance(pc: &PrimitiveCell) -> Result<f64> {
    if pc.basis.is_empty() {
        return Err(Error::invalid("primitive cell has no basis atoms"));
    }
    // `PrimitiveCell::basis` is FRACTIONAL (`PrimitiveCell::basis_cart` exists for
    // exactly this reason). Adding images to it directly mixes fractional basis
    // with Cartesian vectors, and silently returns a distance off by a factor of
    // the cell constant.
    let cart = pc.basis_cart();
    let mut best = f64::INFINITY;
    for a in &cart {
        for b in &cart {
            let delta = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            // Wrap the difference into the primitive cell's own fractional frame.
            let f = pc.h.solve(delta)?;
            let n = [f[0].round(), f[1].round(), f[2].round()];
            let img = pc.h.mul_v(n);
            let d = [delta[0] - img[0], delta[1] - img[1], delta[2] - img[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r > 0.0 && r < best {
                best = r;
            }
        }
    }

    // Pure lattice translations are neighbours as well, and the pair loop above
    // cannot see them: for a == b the difference is zero and every image of it
    // is skipped, so a one-atom basis (SC) reports "degenerate" and `fit_region`
    // cannot build a self-fit for it at all. Neighbour coefficients are ±1 for
    // every cell shipped here — edges plus pairwise sums and differences, since
    // in a skewed cell the shortest translation is `v1 - v2`, not an edge.
    let v = [
        pc.h.mul_v([1.0, 0.0, 0.0]),
        pc.h.mul_v([0.0, 1.0, 0.0]),
        pc.h.mul_v([0.0, 0.0, 1.0]),
    ];
    for vi in &v {
        for vj in &v {
            for t in [
                // The edge itself: `vi - vi` is zero and `vi + vi` is the second
                // image, so without this row the shortest translation is missed.
                [vi[0], vi[1], vi[2]],
                [vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]],
                [vi[0] + vj[0], vi[1] + vj[1], vi[2] + vj[2]],
            ] {
                let r = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                if r > 0.0 && r < best {
                    best = r;
                }
            }
        }
    }

    if !(best.is_finite() && best > 0.0) {
        return Err(Error::invalid("degenerate nearest-neighbour distance"));
    }
    Ok(best)
}

/// Rotation of `angle` radians about `axis`, which need not be normalised.
pub fn axis_angle(axis: Coord3, angle: f64) -> Mat3 {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if n.is_nan() || n <= 0.0 {
        return Mat3::identity();
    }
    let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    Mat3::from_rows(
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    )
}

fn mul_mat(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
        }
    }
    Mat3::from_rows(m[0], m[1], m[2])
}

/// Quaternion to a rotation matrix, as a plain array.
///
/// Exists so a caller can hand PTM's per-atom orientation to [`fit_region`] as a
/// seed. A degenerate quaternion is returned unchanged, matching [`quat_to_mat`].
pub fn to_orientation(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 0.0 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        q
    }
}

/// Quaternion `(w, x, y, z)` to a rotation matrix.
pub fn quat_to_mat(q: [f64; 4]) -> Mat3 {
    let q = to_orientation(q);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    Mat3::from_rows(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    )
}

/// Rotation matrix to quaternion `(w, x, y, z)`.
///
/// Shepperd's method: branch on the largest candidate denominator so the square
/// root never divides by a quantity near zero, which is what makes this stable
/// for the 180° rotations a descent passes through.
pub fn mat_to_quat(r: &Mat3) -> [f64; 4] {
    let m = &r.m;
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    };
    to_orientation(q)
}

/// Defect counts for one candidate reference, in the §16 sense.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitScore {
    /// Sites in the box carrying no atom.
    pub vacancies: usize,
    /// Extra atoms beyond one per site, summed over over-occupied sites.
    pub interstitials: usize,
    /// Sites holding more than one atom, as distinct from `interstitials`.
    pub interstitial_sites: usize,
    /// Sites enumerated over the box.
    pub n_sites: usize,
    /// Atoms scored.
    pub n_atoms: usize,
    /// Mean distance from atom to its assigned site.
    pub mean_displacement: f64,
}

impl FitScore {
    /// Vacancies plus interstitial extra-atoms: the §16 headline number.
    pub fn defects(&self) -> usize {
        self.vacancies + self.interstitials
    }

    /// Total order for the search: smallest displacement, then fewest defects.
    ///
    /// Displacement comes first because the obvious order -- defects first -- lets
    /// a badly wrong reference win. A lattice compressed by four percent still
    /// seats every atom, so it reports zero vacancies while holding the true sites
    /// 0.46 A away, and "fewer defects" is no reason to prefer it. The
    /// planted-vacancy fixture landed on exactly that: zero vacancies at 0.46 A
    /// beat four vacancies at 1e-7 A.
    ///
    /// The defect counts are a *measurement* of the fitted reference, not the thing
    /// being fitted, so they only break ties between references sitting equally
    /// close to the atoms. Displacement is quantised to 1e-6 A so the search can
    /// see past float noise: without that, differences of 1e-17 count as
    /// improvements and the search crawls across a plateau of equally good
    /// references instead of settling on one.
    pub fn key(&self) -> (u64, usize) {
        (
            if self.mean_displacement.is_finite() {
                (self.mean_displacement * 1e6).max(0.0) as u64
            } else {
                u64::MAX
            },
            self.defects(),
        )
    }
}

/// A candidate reference: a cell plus the origin it is anchored at.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Primitive cell, already scaled and rotated.
    pub pc: PrimitiveCell,
    /// Absolute Cartesian origin, in the same frame as the input positions.
    pub origin: Coord3,
}

/// Occupancy of the fitted lattice over a region, for one candidate.
///
/// # The region, and why it is not the box
///
/// A grain's lattice does not in general tile the simulation box: tilt it by 34
/// degrees and the box is incommensurate with it, so the lattice points inside
/// the box leave boundary layers with no atom and the atoms near the faces have
/// no lattice point inside the box at all. Counting the former and the latter
/// gave 411 vacancies and 383 interstitial sites on a defect-free 2048-atom
/// tilted crystal — an artefact of the box, not a property of the material.
///
/// So the two questions are separated:
///
/// - An atom belongs to the **infinite** lattice or it does not. Its nearest
///   lattice point is found without any clipping, so a rotated grain resolves
///   exactly like an aligned one.
/// - A vacancy is a lattice point **inside the region the atoms define**, with
///   nothing on it. The region is the whole lattice cells those atoms sit in, so
///   it is finite, and for a crystal filling its box it is the box.
///
/// The remaining subtlety is the outermost lattice point of the region: an atom
/// inside the region can have its true nearest lattice point just outside it.
/// Those atoms are found by searching the neighbouring cell indices, so a boundary
/// atom is still assigned to its own site instead of being written off as an
/// interstitial.
/// Enumerated sites of a region, with the slot each one owns.
///
/// The grid carries no box and no basis-to-Cartesian transform: it is a
/// row-major index space over `[i0, i0 + n)`, and whether a cell index is inside
/// the region is a question for [`Region`], not for the grid.
struct SiteGrid {
    /// Lowest enumerated integer cell index per axis.
    i0: [i64; 3],
    /// Enumerated counts per axis, including the slack either side.
    n: [usize; 3],
    /// Element stride per axis of the row-major enumeration.
    stride: [usize; 3],
    /// In-region counts per axis.
    dims: [usize; 3],
    /// Occupancy index per enumerated site; `OUTSIDE` marks one outside the region.
    slot: Vec<u32>,
    /// Sites inside the region.
    n_sites: usize,
}

/// Occupancy sentinel for a lattice point outside the region.
const OUTSIDE: u32 = u32::MAX;

impl SiteGrid {
    #[inline]
    fn index(&self, i: [i64; 3], bi: usize) -> Option<usize> {
        let (d0, d1, d2) = (i[0] - self.i0[0], i[1] - self.i0[1], i[2] - self.i0[2]);
        if d0 < 0 || d1 < 0 || d2 < 0 {
            return None;
        }
        let (d0, d1, d2) = (d0 as usize, d1 as usize, d2 as usize);
        if d0 >= self.n[0] || d1 >= self.n[1] || d2 >= self.n[2] {
            return None;
        }
        Some(d0 * self.stride[0] + d1 * self.stride[1] + d2 * self.stride[2] + bi)
    }

    fn dims(&self) -> [usize; 3] {
        self.dims
    }
}

/// Bounding box of the atoms, clamped to the simulation box.
///
/// Positions are wrapped, so this is inside the cell already; the clamp is a
/// guard against a caller handing in something unwrapped.
///
/// Not the region, and not one term of it either. This is where the atoms
/// physically are, which is what the nearest-neighbour histogram and the rotation
/// pivot want. The region needs a reference to say which cells they occupy, and
/// for a tilted grain that is not this box.
fn atom_span(cell: &UnitCell, positions: &[Coord3]) -> Result<(Coord3, Coord3)> {
    let (clo, chi) = cell.bounds();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in positions {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    for d in 0..3 {
        lo[d] = lo[d].max(clo[d]);
        hi[d] = hi[d].min(chi[d]);
        if hi[d].is_nan() || lo[d].is_nan() || hi[d] <= lo[d] {
            return Err(Error::invalid("atoms of the region do not span the box"));
        }
    }
    Ok((lo, hi))
}

/// Which lattice cells a candidate's site count covers, and whether the region
/// wraps.
struct Region {
    /// Integer cell bounds, half-open: `[ilo, ihi)`.
    ilo: [i64; 3],
    ihi: [i64; 3],
    /// Whether an atom's site is taken modulo these bounds.
    ///
    /// True exactly when the crystal fills the box. This is not a cosmetic
    /// distinction: it decides whether a cell index outside the region is an error
    /// or a periodic image.
    periodic: bool,
}

impl Region {
    fn dims(&self) -> [usize; 3] {
        [
            (self.ihi[0] - self.ilo[0]).max(1) as usize,
            (self.ihi[1] - self.ilo[1]).max(1) as usize,
            (self.ihi[2] - self.ilo[2]).max(1) as usize,
        ]
    }

    /// Fold a cell index into the region.
    #[inline]
    fn wrap(&self, mut i: [i64; 3]) -> [i64; 3] {
        for (d, &ih) in self.ihi.iter().enumerate() {
            let n = (ih - self.ilo[d]).max(1);
            i[d] = self.ilo[d] + (i[d] - self.ilo[d]).rem_euclid(n);
        }
        i
    }
}

fn region_of(cell: &UnitCell, r: &Ref3, positions: &[Coord3]) -> Result<Region> {
    let (clo, chi) = cell.bounds();
    let (lo, hi) = atom_span(cell, positions)?;

    // The region is the set of lattice cells that hold a seated atom.
    //
    // Taken from each atom's own nearest cell rather than from its bounding box,
    // for the reasons below.
    //
    // Robustness: a bounding box's cell bounds are decided by `ceil` of a number that
    // is exactly on a plane whenever the atoms fill their box, so a scale error of
    // 0.007% turned an 8-cell block into 9 cells and invented 868 vacancies in a
    // crystal with none. An atom belongs to whichever cell its site is in, and no
    // plausible scale error moves it across a plane.
    //
    // Gauge: the bounds move with the origin, but their *size* does not, because a
    // lattice translation sends every atom's cell index to a different cell index
    // and leaves the span alone. That is why the origin is left wherever the search
    // put it -- it is defined only modulo the lattice, and "canonicalising" it by
    // shifting the atoms' lower corner onto a lattice point moves the lattice by a
    // *fractional* cell, which is not a gauge freedom at all.
    //
    // Accuracy: for a tilted grain the fitted cell is aligned with the grain, so the
    // atoms occupy a tidy cube of cells while their *bounding box* in the frame is a
    // fat tilted one. Measuring the region from the box counts 392 sites for a
    // 4x4x4 block that has 256; measuring it from the atoms counts the 256.
    let mut ilo = [i64::MAX; 3];
    let mut ihi = [i64::MIN; 3];
    for &p in positions {
        let (i, _, _) = nearest_lattice(r, p);
        for d in 0..3 {
            ilo[d] = ilo[d].min(i[d]);
            ihi[d] = ihi[d].max(i[d] + 1);
        }
    }
    if ilo[0] == i64::MAX {
        return Err(Error::invalid("no atoms in the fitting region"));
    }

    // Does the crystal fill the box?
    //
    // If it does, the region is the box and the site count is the box's own cell
    // count -- not the atom bounding box's, which lands wherever the thermal
    // displacements happen to put the atoms and moves by a few tenths of an
    // angstrom from frame to frame. A site count that depends on where the atoms
    // happen to sit cannot report zero defects on a perfect crystal.
    //
    // The test is commensurability of the fitted cell with the box. A 300 K FCC
    // crystal is not exactly commensurate with its box -- the mean first-shell
    // distance exceeds `a/sqrt(2)` by about a tenth of a percent, because
    // `E|r| > |E r|` -- so the tolerance here is a percent, wide enough to accept
    // that and narrow enough to reject a genuinely different lattice.
    //
    // Measured against the *box*, not against the atom bbox: the origin is a fitted
    // quantity and lands a hair outside the box corner often enough, which would
    // shift `ilo` by one cell and turn a 40-cell region into 41 cells with 19,691
    // phantom vacancies in it.
    let mut periodic = true;
    let mut nbox = [0i64; 3];
    for d in 0..3 {
        let span = chi[d] - clo[d];
        let frac = span / r.row_len[d];
        let n = frac.round();
        let covers = (hi[d] - lo[d]) / span;
        if n.is_nan() || n < 1.0 || (frac - n).abs() > 0.01 * n || covers < 0.98 {
            periodic = false;
            break;
        }
        nbox[d] = n as i64;
    }
    if periodic {
        ilo = [0; 3];
        ihi = nbox;
    }

    for d in 0..3 {
        if ihi[d] <= ilo[d] {
            return Err(Error::invalid("fitted region is empty"));
        }
    }
    Ok(Region { ilo, ihi, periodic })
}

/// Build the in-region occupancy grid for a candidate reference.
fn site_grid(r: &Ref3, region: &Region) -> Result<SiteGrid> {
    let dims = region.dims();
    let nb = r.basis.len();

    // Enumerate the whole region and nothing else.
    //
    // The range *is* the region, so no site's Cartesian coordinates are tested
    // against a region box: such a test would be the one place where a boundary
    // could be decided by a floating-point comparison, and getting it wrong costs
    // either 9,721 phantom vacancies or 9,481 atoms with no site at all.
    let mut i0 = region.ilo;
    let mut n = dims;

    // The atom loop can land a site one cell outside the region before wrapping,
    // and a grain's boundary atoms can be outside it legitimately, so keep a cell
    // of slack on each side of the *enumerated* range.
    let mut stride = [0usize; 3];
    let mut total = nb;
    for d in 0..3 {
        i0[d] -= 1;
        n[d] += 2;
        stride[d] = total;
        total = total
            .checked_mul(n[d])
            .ok_or_else(|| Error::invalid("lattice enumeration too large"))?;
    }

    // The cell count is a function of the fitted scale, so a badly wrong scale turns
    // into an enormous but perfectly representable allocation. Refuse rather than
    // thrash: it means the scale search diverged, not that the region is large.
    const MAX_CELLS: usize = 512 * 1024 * 1024;
    if total > MAX_CELLS {
        return Err(Error::invalid(format!(
            "lattice enumeration of {}x{}x{} cells is implausible for this region; \
             the fitted scale is wrong",
            n[0], n[1], n[2]
        )));
    }

    let mut grid = SiteGrid {
        i0,
        n,
        stride,
        dims,
        slot: vec![OUTSIDE; total],
        n_sites: 0,
    };

    for d0 in 0..dims[0] {
        for d1 in 0..dims[1] {
            for d2 in 0..dims[2] {
                let i = [
                    region.ilo[0] + d0 as i64,
                    region.ilo[1] + d1 as i64,
                    region.ilo[2] + d2 as i64,
                ];
                for bi in 0..nb {
                    let idx = grid.index(i, bi).expect("index from the enumerated range");
                    grid.slot[idx] = grid.n_sites as u32;
                    grid.n_sites += 1;
                }
            }
        }
    }

    Ok(grid)
}

/// Reusable per-atom output of an assignment.
///
/// Buffers rather than fresh `Vec`s, because the final pass runs once per region
/// and a per-grain loop would otherwise reallocate per grain.
#[derive(Debug, Clone, Default)]
pub struct PerAtom {
    /// Site index per atom, `-1` when the atom has no lattice point within
    /// tolerance, and `-2` when it has one that the region does not cover.
    pub site_of_atom: Vec<i64>,
    /// Atom-minus-site vector, three components per atom. For an off-lattice atom
    /// this points at the *nearest* in-region site, so it stays finite and
    /// comparable with the others instead of being a hole in the data.
    pub displacement: Vec<f64>,
}

/// Assignment of one region's atoms to a candidate reference.
///
/// Split out of [`score_candidate`] because the site tally is not all a caller
/// wants: [`WsSelf`](crate::analysis::ws::WsSelf) must publish which site each
/// atom claimed and how far off it sits, and that has to be *the same*
/// assignment the counts came from rather than a second pass that could
/// disagree with them.
#[derive(Debug, Clone)]
pub struct SiteAssignment {
    /// Occupancy per in-region site.
    pub occupancy: Vec<u32>,
    /// Empty in-region sites, in enumeration order, for clustering.
    ///
    /// Vacancy *clusters* are what a void detector needs and a vacancy *count* is
    /// not, so the list is built here where the grid geometry is to hand: recovering
    /// it later means rebuilding the same index arithmetic and hoping it matches.
    pub empty_sites: Vec<usize>,
    /// Cell counts spanning the region.
    pub dims: [usize; 3],
    /// Lowest integer cell index of the *region* per axis, i.e. the cell that site
    /// index 0's basis atom 0 sits in.
    ///
    /// This is deliberately not the enumeration origin. The enumerator keeps a cell of
    /// slack either side so an atom can claim a site one cell outside the region before
    /// wrapping, and publishing that shifted origin instead would make every site
    /// position computed from it come out a whole lattice constant low.
    pub cell_origin: [i64; 3],
    /// Basis atoms per cell.
    pub n_basis: usize,
    /// Defect counts, identical to what [`score_candidate`] reports.
    pub score: FitScore,
    /// Positions of the in-region sites, resolved from the same grid that
    /// produced these occupancy indices.
    pub sites: InRegionSites,
}

/// Cartesian positions of a self-built run's in-region sites.
///
/// The indices in [`SiteAssignment`] are region-local: they are laid out from the
/// region's *lowest* integer cell, and the enumerator deliberately keeps a cell of
/// slack either side so an atom can claim a site just outside the region before
/// wrapping. Recovering a position from an index therefore needs that shifted
/// origin (`cell_origin`) and not the enumeration origin -- publishing the latter
/// would put every vacancy site a whole lattice constant low, which is the kind of error
/// that looks like a physical result rather than an error.
///
/// Held by value rather than as a `Vec<Coord3>`: the fit needs one position per
/// site only when someone asks to write defects out, and materialising
/// 3 floats per site at 256k sites is 6 MB that the common case would pay for
/// nothing.
#[derive(Debug, Clone)]
pub struct InRegionSites {
    /// Fitted reference cell, rows are lattice vectors.
    h: Mat3,
    /// Absolute origin of fractional (0,0,0).
    origin: Coord3,
    /// Fractional basis atoms per conventional cell.
    basis: Vec<Coord3>,
    /// Lowest in-region integer cell index per axis, i.e. where site index 0's
    /// basis atom 0 sits.
    cell_origin: [i64; 3],
    /// In-region cell counts per axis.
    dims: [usize; 3],
    /// Basis atoms per cell.
    n_basis: usize,
    /// Sites inside the region.
    n_sites: usize,
}

impl InRegionSites {
    fn new(r: &Ref3, region: &Region, dims: [usize; 3], n_basis: usize, n_sites: usize) -> Self {
        Self {
            h: r.h,
            origin: r.origin,
            basis: r.basis.clone(),
            cell_origin: region.ilo,
            dims,
            n_basis,
            n_sites,
        }
    }

    /// Sites inside the region, matching the occupancy array's length.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_sites
    }

    /// Always false: a fitted region holds at least one site.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_sites == 0
    }

    /// Cartesian position of in-region site `idx`.
    ///
    /// The exact inverse of the enumeration in `site_grid`: cell-major over
    /// `dims`, basis-minor within a cell, and **z is the fastest axis**. That
    /// last part is the whole difficulty — `SiteLattice::get`, the other
    /// implementation of this same idea in this repository, enumerates
    /// x-fastest, because its stride runs the other way. Guessing the wrong one
    /// yields positions that are perfectly plausible lattice points of the same
    /// cell and simply belong to different sites, so nothing downstream notices
    /// until a vacancy file disagrees with a displacement.
    pub fn position(&self, idx: usize) -> Coord3 {
        debug_assert!(idx < self.n_sites, "site index out of range");
        let rem = idx / self.n_basis;
        let b = self.basis[idx % self.n_basis];
        let dz = rem % self.dims[2];
        let rem = rem / self.dims[2];
        let dy = rem % self.dims[1];
        let dx = rem / self.dims[1];
        let x = self.h.mul_v([
            (self.cell_origin[0] + dx as i64) as f64 + b[0],
            (self.cell_origin[1] + dy as i64) as f64 + b[1],
            (self.cell_origin[2] + dz as i64) as f64 + b[2],
        ]);
        [
            x[0] + self.origin[0],
            x[1] + self.origin[1],
            x[2] + self.origin[2],
        ]
    }

    /// Every in-region site position, in index order.
    pub fn iter(&self) -> impl Iterator<Item = Coord3> + '_ {
        (0..self.n_sites).map(move |i| self.position(i))
    }
}

/// The parts of a candidate a scoring pass needs, resolved once.
///
/// Inverting the cell and cloning the basis are both O(1), and would otherwise be
/// redone inside the atom loop's caller on every single trial.
struct Ref3 {
    h: Mat3,
    inv_h: Mat3,
    origin: Coord3,
    basis: Vec<Coord3>,
    /// Cartesian length of each fractional axis, for the commensurability test.
    row_len: [f64; 3],
}

impl Ref3 {
    fn of(cand: &Candidate) -> Result<Self> {
        let inv_h = Mat3::from_rows(
            cand.pc.h.solve([1.0, 0.0, 0.0])?,
            cand.pc.h.solve([0.0, 1.0, 0.0])?,
            cand.pc.h.solve([0.0, 0.0, 1.0])?,
        )
        .transposed();
        debug_assert!(!cand.pc.basis.is_empty());
        let mut row_len = [0.0f64; 3];
        for (d, slot) in row_len.iter_mut().enumerate() {
            let v = [cand.pc.h.m[0][d], cand.pc.h.m[1][d], cand.pc.h.m[2][d]];
            *slot = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if !slot.is_finite() || *slot <= 0.0 {
                return Err(Error::invalid("degenerate cell row"));
            }
        }
        Ok(Ref3 {
            h: cand.pc.h,
            inv_h,
            origin: cand.origin,
            basis: cand.pc.basis.clone(),
            row_len,
        })
    }
}

/// Nearest point of the **infinite** lattice to `p`, exactly.
///
/// Returns the winning cell, the basis atom, and the vector from the lattice
/// point to `p`.
///
/// # Why this needs no region and no search
///
/// For a fixed basis atom `b` the nearest cell is `round(f - basis[b])`, so the
/// globally nearest lattice point is always one of the `basis.len()` candidates.
/// There is no neighbourhood to sweep and nothing to enumerate.
///
/// That exactness is what lets the entire search run on this function. Finding
/// the nearest *in-region* site instead would mean building the region's whole
/// occupancy grid on every trial: 2.5 ms per call in release, independent of how
/// many atoms are being scored, times ~4000 trials per fit.
/// Region membership is a property of the *tally*, not of the geometry, and
/// moving it there is what makes the search cost O(atoms) and nothing else.
#[inline]
fn nearest_lattice(r: &Ref3, p: Coord3) -> ([i64; 3], usize, Coord3) {
    let f = r
        .inv_h
        .mul_v([p[0] - r.origin[0], p[1] - r.origin[1], p[2] - r.origin[2]]);
    let mut best_r2 = f64::INFINITY;
    let mut best_i = [0i64; 3];
    let mut best_b = 0usize;
    let mut best_d = [0.0f64; 3];
    for (bi, b) in r.basis.iter().enumerate() {
        let g = [f[0] - b[0], f[1] - b[1], f[2] - b[2]];
        let i = [g[0].round(), g[1].round(), g[2].round()];
        // The residual is the fractional part, so the Cartesian difference is a
        // single `h` multiply -- no absolute position, no origin add.
        let d = r.h.mul_v([g[0] - i[0], g[1] - i[1], g[2] - i[2]]);
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if r2 < best_r2 {
            best_r2 = r2;
            best_i = [i[0] as i64, i[1] as i64, i[2] as i64];
            best_b = bi;
            best_d = d;
        }
    }
    (best_i, best_b, best_d)
}

/// Mean distance from each atom to the nearest point of the infinite lattice.
///
/// This is the search's objective, and deliberately the cheapest quantity that
/// measures how well a reference explains the atoms. It is also the only part of
/// the score a subsample can measure honestly: a *missing* atom is invisible to a
/// sample, so vacancy and interstitial counts cannot be optimised against and are
/// measured once on the winner instead.
///
/// Takes no box and builds nothing, so it is safe to call in the innermost loop
/// of the search. [`Candidate`] by value would be wasteful; [`Ref3`] is private,
/// so this is the public face of it.
fn mean_displacement(r: &Ref3, positions: &[Coord3]) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &p in positions {
        let (_, _, d) = nearest_lattice(r, p);
        sum += norm3(d);
    }
    sum / positions.len() as f64
}

fn norm3(d: Coord3) -> f64 {
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Vacancies, over-occupied sites, and the extra atoms on them.
fn tally_counts(occupancy: &[u32]) -> (usize, usize, usize) {
    let mut vacancies = 0usize;
    let mut interstitial_sites = 0usize;
    let mut interstitials = 0usize;
    for &o in occupancy {
        if o == 0 {
            vacancies += 1;
        } else if o > 1 {
            interstitial_sites += 1;
            interstitials += (o - 1) as usize;
        }
    }
    (vacancies, interstitial_sites, interstitials)
}

/// Assignment of a region's atoms to a fitted reference, counts included.
///
/// `per_atom` and `par` are both optional and both exist for the same reason:
/// the search wants neither and pays for neither, and the final pass at 128M atoms
/// wants both.
///
/// # Semantics of a boundary atom
///
/// Every atom is assigned to its nearest point of the infinite lattice, and the
/// displacement reported is that distance -- so an atom sitting exactly on a
/// lattice point just outside the region has displacement ~0. Whether it
/// consumes a site depends on where that point is:
///
/// - in the region: it fills that slot, and an over-full slot is an interstitial.
/// - outside the region, within tolerance: it is on the lattice and is neither.
///   The region is finite and the lattice is not, so "outside" is a statement about
///   the region, not about the crystal.
/// - outside the region, beyond tolerance, or further from every in-region point
///   than tolerance: off-lattice, and counted as an interstitial.
fn assign_region_inner(
    cell: &UnitCell,
    r: &Ref3,
    positions: &[Coord3],
    tolerance: f64,
    par: Option<&crate::exec::Par>,
    per_atom: Option<&mut PerAtom>,
) -> Result<SiteAssignment> {
    let region = region_of(cell, r, positions)?;
    let grid = site_grid(r, &region)?;
    let tol2 = tolerance.max(0.0).powi(2);
    let occupancy: Vec<AtomicU32> = (0..grid.n_sites).map(|_| AtomicU32::new(0)).collect();
    let mut sum = 0.0f64;
    let mut off_lattice = 0usize;

    // Single-threaded and buffer-free: the search's shape, and `score_candidate`.
    if par.is_none() && per_atom.is_none() {
        let mut plain = vec![0u32; grid.n_sites];
        for &p in positions {
            let (i, bi, d) = nearest_lattice(r, p);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let slot = grid
                .index(if region.periodic { region.wrap(i) } else { i }, bi)
                .map(|idx| grid.slot[idx])
                .filter(|&s| s != OUTSIDE);
            match slot {
                Some(s) if r2 <= tol2 => plain[s as usize] += 1,
                // On the lattice but outside a grain's region is neither a vacancy
                // nor an interstitial: the region is finite and the lattice is not, so
                // "outside" describes the region rather than the crystal. Only a
                // distance past tolerance is a defect. Charging every one of them gives
                // a tilted grain with a perfect seed lattice 128 interstitials at a
                // mean displacement of 0.0005 A.
                _ if r2 <= tol2 => {}
                _ => off_lattice += 1,
            }
            sum += norm3(d);
        }
        let (vacancies, interstitial_sites, interstitials) = tally_counts(&plain);
        return Ok(finish(
            plain,
            &grid,
            &region,
            InRegionSites::new(r, &region, grid.dims(), r.basis.len(), grid.n_sites),
            (vacancies, interstitial_sites, interstitials + off_lattice),
            positions.len(),
            sum,
        ));
    }

    // Chunked, gathering per-atom results in index order. `map_collect` rather
    // than disjoint in-place writes: the closure has to be `Sync`, and gathering
    // in index order is what makes the mean below independent of thread count.
    // Float addition is not associative, so summing per thread and combining
    // would make `mean_displacement` vary with `--threads`.
    let par = par.unwrap();
    let per_atom = per_atom.expect("parallel pass needs the per-atom buffers");
    let results: Vec<(i64, Coord3)> = par.map_collect(positions.len(), |i| {
        let (cell_i, bi, d) = nearest_lattice(r, positions[i]);
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let slot = grid
            .index(
                if region.periodic {
                    region.wrap(cell_i)
                } else {
                    cell_i
                },
                bi,
            )
            .map(|idx| grid.slot[idx])
            .filter(|&s| s != OUTSIDE);
        let slot = match slot {
            Some(s) if r2 <= tol2 => {
                occupancy[s as usize].fetch_add(1, Ordering::Relaxed);
                s as i64
            }
            // `-2`: seated on a lattice point that the region does not cover. Not a
            // defect, and it consumes no site, so it is reported distinctly from the
            // off-lattice `-1`.
            _ if r2 <= tol2 => -2,
            _ => -1,
        };
        Ok::<_, Error>((slot, d))
    })?;
    let occ: Vec<u32> = occupancy
        .iter()
        .map(|c| c.load(Ordering::Relaxed))
        .collect();

    per_atom.site_of_atom.clear();
    per_atom.displacement.clear();
    per_atom.site_of_atom.reserve(positions.len());
    per_atom.displacement.reserve(3 * positions.len());
    for (slot, d) in results {
        if slot == -1 {
            off_lattice += 1;
        }
        sum += norm3(d);
        per_atom.site_of_atom.push(slot);
        per_atom.displacement.extend_from_slice(&d);
    }

    let (vacancies, interstitial_sites, interstitials) = tally_counts(&occ);
    Ok(finish(
        occ,
        &grid,
        &region,
        InRegionSites::new(r, &region, grid.dims(), r.basis.len(), grid.n_sites),
        (vacancies, interstitial_sites, interstitials + off_lattice),
        positions.len(),
        sum,
    ))
}

fn finish(
    occupancy: Vec<u32>,
    grid: &SiteGrid,
    region: &Region,
    sites: InRegionSites,
    counts: (usize, usize, usize),
    n_atoms: usize,
    sum: f64,
) -> SiteAssignment {
    let (vacancies, interstitial_sites, interstitials) = counts;
    // Enumeration order, so the index is usable directly: a site index is
    // `((i - i0[0]) * stride[0] + (j - i0[1])) * stride[1] + (k - i0[2])) * n_basis + bi`
    // and the strides are exactly `dims[1] * dims[2] * n_basis` and so on. Deriving the
    // empty list here is the last moment the grid is in hand.
    let empty_sites = occupancy
        .iter()
        .enumerate()
        .filter(|(_, &o)| o == 0)
        .map(|(i, _)| i)
        .collect();
    SiteAssignment {
        occupancy,
        empty_sites,
        dims: grid.dims(),
        cell_origin: region.ilo,
        n_basis: sites.n_basis,
        sites,
        score: FitScore {
            vacancies,
            interstitials,
            interstitial_sites,
            n_sites: grid.n_sites,
            n_atoms,
            mean_displacement: if n_atoms == 0 {
                0.0
            } else {
                sum / n_atoms as f64
            },
        },
    }
}

/// Assign a region's atoms to a fitted reference and count defects.
///
/// This is the whole engine behind [`score_candidate`]: give it a `par` and a
/// [`PerAtom`] and it additionally publishes the per-atom detail and spreads the
/// atom loop over threads. Both are optional and both exist for the same reason --
/// the search wants neither.
pub fn assign_region(
    cell: &UnitCell,
    cand: &Candidate,
    positions: &[Coord3],
    tolerance: f64,
    par: Option<&crate::exec::Par>,
    per_atom: Option<&mut PerAtom>,
) -> Result<SiteAssignment> {
    let r = Ref3::of(cand)?;
    assign_region_inner(cell, &r, positions, tolerance, par, per_atom)
}

/// Score a candidate against every atom in the region, counts only.
pub fn score_candidate(
    cell: &UnitCell,
    cand: &Candidate,
    positions: &[Coord3],
    tolerance: f64,
) -> Result<FitScore> {
    let r = Ref3::of(cand)?;
    Ok(assign_region_inner(cell, &r, positions, tolerance, None, None)?.score)
}

/// Mean distance from each atom to its nearest lattice point, for a candidate.
///
/// The search's inner objective. No region, no site enumeration, no allocation.
pub fn mean_displacement_of(cand: &Candidate, positions: &[Coord3]) -> Result<f64> {
    Ok(mean_displacement(&Ref3::of(cand)?, positions))
}

/// Knobs for [`fit_region`].
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Orientation seed `(w, x, y, z)`, normally a per-grain PTM quaternion.
    ///
    /// `None` starts from the identity, which only suits an axis-aligned region.
    pub seed_orientation: Option<[f64; 4]>,
    /// Divisions per axis in the coarse origin grid.
    pub origin_grid: usize,
    /// Refinement rounds for the origin.
    pub rounds: usize,
    /// Angular step of the orientation descent, radians.
    pub angle_step: f64,
    /// Maximum outer rounds of the orientation descent.
    pub max_rounds: usize,
    /// Smallest descent step; the step halves on every unproductive round and the
    /// descent stops once it falls below this.
    pub angle_tol: f64,
    /// How far an atom may sit from a lattice point and still count as on it.
    ///
    /// `None` uses half the nearest-neighbour distance. A 300 K crystal
    /// displaces atoms by a few tenths of an angstrom and a fitted reference
    /// never matches exactly, so an exact-match test would report every atom in a
    /// real structure as an interstitial.
    pub tolerance: Option<f64>,
    /// Atoms used for the *scale* and orientation search.
    ///
    /// Vacancy and interstitial counts cannot be estimated from a subsample --
    /// a subsample cannot see a missing atom -- so the winning reference is always
    /// re-scored on the whole region. Mean displacement can, which is what makes
    /// this safe, and it is what the search minimises.
    pub search_sample: usize,
    /// Multiplicative range for the scale sweep about the nearest-neighbour
    /// estimate, spanning `[1/window, window]`.
    ///
    /// Narrow on purpose. The sweep is a polish on an already-good estimate, and a
    /// wide bracket is actively harmful: mean displacement falls monotonically as
    /// the lattice is made finer, so a grid too coarse to contain a point near the
    /// truth never sees the dip and reports the smallest lattice it tried. At a
    /// factor of 69 the grid's nearest point to the answer was 0.74, and the fit
    /// settled 0.12 of the true scale.
    pub scale_window: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            seed_orientation: None,
            origin_grid: 3,
            rounds: 14,
            angle_step: 0.05,
            max_rounds: 24,
            angle_tol: 1e-5,
            tolerance: None,
            search_sample: 100_000,
            scale_window: 1.25,
        }
    }
}

/// A fitted reference for one region.
#[derive(Debug, Clone)]
pub struct RegionFit {
    /// Fitted primitive cell, scale and orientation included.
    pub cell: PrimitiveCell,
    /// Absolute origin.
    pub origin: Coord3,
    /// Integer cell counts spanning the fitted region.
    pub dims: [usize; 3],
    /// Orientation as a quaternion `(w, x, y, z)`.
    pub orientation: [f64; 4],
    /// Defect counts for the winning candidate, measured over every atom.
    pub score: FitScore,
    /// Matching tolerance the score used.
    ///
    /// Part of the reference's definition, not an incidental detail: it is what
    /// decides whether an atom counts as sitting on a site, so anyone re-scoring
    /// this reference -- [`assign_region`](crate::lattice::fit::assign_region),
    /// which is how [`WsSelf`](crate::analysis::ws::WsSelf) builds its output --
    /// has to use the same value or get different counts from the same lattice.
    pub tolerance: f64,
    /// Candidates scored, for diagnostics.
    pub trials: usize,
}

/// Fit a reference lattice to the atoms of one region.
///
/// `lattice` supplies the prototype and `nn` is a starting guess at the
/// nearest-neighbour distance; both are only hints, and the returned reference is
/// sized from the atoms. `opts.seed_orientation` is the orientation to start
/// from, normally a PTM quaternion — see the module docs for why there is no
/// global search to fall back on.
///
/// Everything in the returned [`RegionFit`] is measured over *all* `positions`,
/// not over the bounded sample the search uses. A caller can sanity-check the
/// result with `score.mean_displacement` against `score.n_sites` and
/// `score.n_atoms`: a seed that was too coarse leaves a large displacement, which
/// is the failure this API is built to make visible rather than hide behind a
/// confident zero-defect count.
pub fn fit_region(
    cell: &UnitCell,
    positions: &[Coord3],
    lattice: &dyn Lattice,
    nn: f64,
    opts: &FitOptions,
) -> Result<RegionFit> {
    if positions.is_empty() {
        return Err(Error::invalid("cannot fit a reference to an empty region"));
    }
    if !(nn.is_finite() && nn > 0.0) {
        return Err(Error::invalid(format!(
            "nearest-neighbour distance must be positive and finite, got {nn}"
        )));
    }

    let nominal = lattice.primitive();
    let (rlo, rhi) = atom_span(cell, positions)?;
    let rc = [
        0.5 * (rlo[0] + rhi[0]),
        0.5 * (rlo[1] + rhi[1]),
        0.5 * (rlo[2] + rhi[2]),
    ];

    // Seed rotation, applied once here so the origin search below is scored
    // against a cell that already points the right way.
    let rot0 = opts
        .seed_orientation
        .map(to_orientation)
        .map(quat_to_mat)
        .unwrap_or_else(Mat3::identity);

    // The search runs on a bounded, evenly strided sample; the winner is always
    // re-scored on every atom. Mean displacement is the only quantity a sample
    // measures honestly -- a *missing* atom is invisible to a sample -- so the
    // sample is used for exactly the search, and never for a defect count.
    let sample = search_indices(positions.len(), opts.search_sample);
    let search_pos: Vec<Coord3> = sample.iter().map(|&i| positions[i]).collect();

    // Scale: from the nearest-neighbour spacing measured on the atoms, so the
    // starting cell is right before anything else is searched. `nn` is only a hint
    // for the search and for the matching tolerance, never the fitted value.
    let nn_fit = estimate_nn(cell, &search_pos, nn)?;
    if nn_fit.is_nan() || nn_fit <= 0.0 {
        return Err(Error::invalid("degenerate nearest-neighbour estimate"));
    }
    let scale0 = nn_fit / nn_distance(&nominal)?;
    let fitted = scale_rows(&nominal, scale0);

    let tolerance = opts.tolerance.unwrap_or(0.5 * nn_fit);
    if tolerance.is_nan() || tolerance <= 0.0 {
        return Err(Error::invalid(format!(
            "tolerance must be positive, got {tolerance}"
        )));
    }

    let mut trials = 0usize;

    // --- Scale and origin: alternated, each fitted to mean displacement -------
    // The density estimate alone is biased: it counts a vacancy as a missing atom
    // and shrinks the reference to match, which on a block with four planted
    // vacancies gave a mean displacement of 0.46 A. Defect counts cannot detect
    // that error either, because a compressed lattice still seats every atom, just
    // badly, and the matching tolerance hides the difference. Mean displacement
    // is the one part of the score a subsample can measure honestly, so it is what
    // both parameters are fitted to.
    //
    // The two alternate instead of being refined jointly. Sweeping scale with a
    // fixed origin costs one scoring pass per trial, whereas refining origin
    // inside every trial would cost a full origin search per trial; and the
    // translation that aligns a lattice is very nearly independent of its scale,
    // so alternating converges in a few passes where nesting does not.
    //
    // Every sweep keeps its own refined origin. Rebuilding the candidate from the
    // box corner instead would fit the scale against a knowingly bad origin and
    // lands several percent away.
    let rows_at = |mult: f64| scale_rows(&fitted, mult).h;
    let candidate_at = |rot: &Mat3, mult: f64, origin: Coord3| -> Candidate {
        Candidate {
            pc: PrimitiveCell::new(mul_mat(rot, &rows_at(mult)), fitted.basis.clone()),
            origin,
        }
    };
    let displacement_at = |rot: &Mat3, rows: &Mat3, origin: Coord3, trials: &mut usize| -> f64 {
        *trials += 1;
        let cand = Candidate {
            pc: PrimitiveCell::new(mul_mat(rot, rows), fitted.basis.clone()),
            origin,
        };
        mean_displacement_of(&cand, &search_pos).unwrap_or(f64::INFINITY)
    };

    // A log-spaced scan finds the basin, then golden section pins it down. The
    // nearest-neighbour estimate puts the sweep close to the answer already, so this
    // is a polish; the coarse scan stays because a wrong lattice alignment can
    // present a lower mean displacement than the right scale at a poor origin, and a
    // fine local search would never step out of it.
    let span = opts.scale_window;
    let coarse = 7usize;
    let log_span = span.max(1e-6).ln();
    // Spans [1/span, span]: k = 0 gives 1/span, k = coarse gives span. Getting this
    // backwards yields multipliers around span^2 instead, which shrinks the lattice
    // until the enumeration range explodes.
    let coarse_mult = |k: usize| (log_span * (2.0 * (k as f64) / coarse as f64 - 1.0)).exp();

    // --- Origin and scale, alternated ------------------------------------------
    // Two coupled parameters: a wrong scale makes the best origin look bad, and a
    // wrong origin makes the best scale look bad. Alternating resolves that, but the
    // order decides whether it works at all.
    //
    // The nearest-neighbour estimate already pins the scale, so the *origin* is
    // searched first and the scale is only ever polished afterwards. Sweeping scale
    // against a lattice that is not yet aligned inverts the logic, because shrinking
    // the lattice drags sites towards the atoms and scores better: on both an aligned
    // block and a tilted grain it settled on the edge of its bracket, 1.25x too
    // coarse, and reported thousands of phantom vacancies.
    let mut centroid = [0.0f64; 3];
    for p in positions {
        for d in 0..3 {
            centroid[d] += p[d];
        }
    }
    for c in centroid.iter_mut() {
        *c /= positions.len() as f64;
    }
    // Anchored with the first basis site on the centroid. Starting from the region's
    // corner instead leaves a tilted grain's anchor 9 A out, which origin_grid's single
    // cell cannot reach.
    let mut origin = sub(centroid, mul_mat(&rot0, &fitted.h).mul_v(fitted.basis[0]));
    let mut scale = 1.0f64;
    let mut best = (f64::INFINITY, scale, origin);
    for _ in 0..3 {
        let (sc, refined) = origin_search(
            &candidate_at(&rot0, scale, origin),
            &search_pos,
            opts.origin_grid,
            opts.rounds,
            &mut trials,
        )?;
        origin = refined.origin;
        if sc < best.0 {
            best = (sc, scale, origin);
        }

        let mut means = Vec::with_capacity(coarse + 1);
        for k in 0..=coarse {
            means.push(displacement_at(
                &rot0,
                &rows_at(coarse_mult(k)),
                origin,
                &mut trials,
            ));
        }
        let bi = means
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(k, _)| k)
            .unwrap_or(0);
        let bracket = (
            coarse_mult(bi.saturating_sub(1)),
            coarse_mult((bi + 1).min(coarse)),
        );
        let (mult, _) = golden_min(bracket.0, bracket.1, 40, &mut |m: f64| {
            displacement_at(&rot0, &rows_at(m), origin, &mut trials)
        });
        scale = mult;
    }
    let seed_cand = candidate_at(&rot0, best.1, best.2);
    let (mut mean, mut cur) = refine_origin(&seed_cand, &search_pos, opts.rounds, &mut trials)?;

    // --- Orientation descent and scale, alternated -----------------------------
    //
    // Every scale sweep before this point ran against a seed orientation that was
    // still a few degrees out, so it had nothing to work with: it returned 0.9933 and
    // the 0.67% it left is exactly the 0.046 A of displacement the fit went on to
    // report. Alternating until both stop moving closes it, and the second pass is
    // what actually reaches it.
    let mut rot = rot0;
    for _ in 0..3 {
        let (s2, c2, r2) = descend(cur, mean, rot, rc, &search_pos, opts, &mut trials)?;
        (mean, cur, rot) = (s2, c2, r2);

        // Scale polish at the orientation just found. Scaling the rows leaves `rot`
        // untouched, so the decomposition pc.h = rot * rows(scale) stays exact.
        let mut means = Vec::with_capacity(coarse + 1);
        for k in 0..=coarse {
            means.push(displacement_at(
                &rot,
                &rows_at(coarse_mult(k)),
                cur.origin,
                &mut trials,
            ));
        }
        let bi = means
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(k, _)| k)
            .unwrap_or(0);
        let bracket = (
            coarse_mult(bi.saturating_sub(1)),
            coarse_mult((bi + 1).min(coarse)),
        );
        let (mult, _) = golden_min(bracket.0, bracket.1, 40, &mut |m: f64| {
            displacement_at(&rot, &rows_at(m), cur.origin, &mut trials)
        });
        let polished = candidate_at(&rot, mult, cur.origin);
        let (sc, refined) = refine_origin(&polished, &search_pos, opts.rounds, &mut trials)?;
        if sc < mean {
            (mean, cur) = (sc, refined);
        }
    }

    // --- Choose the gauge, then report -----------------------------------------
    //
    // The origin is defined only modulo the lattice, and FCC's lattice has more
    // translations than the cell vectors: (a/2, a/2, 0) is one. The origin search is
    // free to stop on any of them, and the gauge it picks changes how the atoms fall
    // across cell boundaries even though the set of lattice points is identical.
    //
    // On an 8-cell FCC block that is the difference between 8 and 9 cells per axis,
    // i.e. 544 sites with no atom behind them on a crystal with four planted
    // vacancies. So the gauge is chosen, not inherited: every origin that maps the
    // lattice onto itself is tried and the one whose region is tightest wins.
    //
    // Translating by a basis offset is enough to reach every gauge that matters, and
    // `basis[0]` is the origin itself for every prototype here, so the unshifted
    // candidate is always among them.
    let mut dims = region_of(cell, &Ref3::of(&cur)?, positions)?.dims();
    let mut best_vol: usize = dims.iter().product();
    for b in 0..cur.pc.basis.len() {
        let base = Ref3::of(&cur)?;
        let cand = Candidate {
            pc: cur.pc.clone(),
            origin: sub(cur.origin, base.h.mul_v(cur.pc.basis[b])),
        };
        let Ok(cand_dims) = region_of(cell, &Ref3::of(&cand)?, positions).map(|r| r.dims()) else {
            continue;
        };
        let vol: usize = cand_dims.iter().product();
        if vol < best_vol {
            best_vol = vol;
            dims = cand_dims;
            cur = cand;
        }
    }

    // --- Report the winner, re-scored on the whole region ---------------------
    let score = score_candidate(cell, &cur, positions, tolerance)?;
    debug_assert_eq!(dims, region_of(cell, &Ref3::of(&cur)?, positions)?.dims());

    Ok(RegionFit {
        tolerance,
        cell: cur.pc,
        origin: cur.origin,
        dims,
        orientation: mat_to_quat(&rot),
        score,
        trials,
    })
}

/// Evenly spaced indices into `n`, capped at `want`.
///
/// Strided rather than random: it is deterministic, so a fit is reproducible and a
/// failure can be re-run, and it needs no RNG state. It is also unbiased in
/// position, which matters because LAMMPS writes atoms in domain order -- a
/// prefix or a random draw from a short prefix would sample one corner of the
/// box.
fn search_indices(n: usize, want: usize) -> Vec<usize> {
    let want = want.max(1);
    if n <= want {
        return (0..n).collect();
    }
    let stride = n as f64 / want as f64;
    (0..want)
        .map(|i| ((i as f64 * stride) as usize).min(n - 1))
        .collect()
}

/// Derivative-free minimisation of `f` on `[lo, hi]` by golden-section search.
///
/// Chosen over a grid because the scale has to be pinned far more tightly than a
/// grid can afford: on a 26 A block a 1e-5 relative scale error already shows up
/// as ~1e-4 A of displacement, and the defect tests need the scale right to well
/// inside the matching tolerance.
fn golden_min<F: FnMut(f64) -> f64>(lo: f64, hi: f64, iters: usize, f: &mut F) -> (f64, f64) {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - INV_PHI * (b - a);
    let mut d = a + INV_PHI * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = f(d);
        }
    }
    let x = if fc < fd { c } else { d };
    (x, f(x))
}

/// Nearest-neighbour spacing among `positions`, the fit's initial length scale.
///
/// Estimating the scale from the region's volume instead fails: density over a
/// region is shape dependent -- the box around a 14 A grain tilted by 0.6 rad holds
/// 546 lattice sites where the grain has 256 atoms, so a density estimate lands
/// 1.55x too coarse and no amount of local refinement recovers it. A bond length
/// says nothing about how much space the atoms are spread over.
///
/// The mean distance over the *first coordination shell*, pooled over every pair in
/// it, rather than any per-atom statistic. Per-atom minima read 4% low because the
/// nearest neighbour of an atom is the smallest of twelve draws from the spacing
/// distribution, and a per-atom median reads a second-shell distance on a corner atom
/// of a small block, which has one first-shell neighbour and three second-shell ones.
/// Pooled, the first shell is identifiable by rank: it is the shortest run of the
/// distance histogram holding a real share of the pairs.
pub fn estimate_nn(cell: &UnitCell, positions: &[Coord3], hint: f64) -> Result<f64> {
    if positions.len() < 2 {
        return Err(Error::invalid("nearest-neighbour estimate needs two atoms"));
    }
    let (lo, hi) = atom_span(cell, positions)?;
    let span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let longest = span[0].max(span[1]).max(span[2]);
    if longest.is_nan() || longest <= 0.0 {
        return Err(Error::invalid("degenerate fitting region"));
    }

    // `hint` only sets the bin size. It comes from the caller, who knows the
    // prototype, so a poor hint costs a wider scan and nothing else.
    let bin = if hint > 0.0 { hint } else { longest / 32.0 };
    let cap = 256usize;
    let mut dims = [1usize; 3];
    for d in 0..3 {
        let want = (span[d] / bin).ceil().max(1.0) as usize;
        dims[d] = want.min(cap);
    }
    // Scan radius, per axis.
    //
    // Not one cell. The bins are sized from the hint with a `ceil`, which makes the
    // cells *narrower* than the first shell whenever `span/bin` is not near an integer
    // -- 2.625 A against a 3.031 A shell on a 3-cell BCC block -- and then a neighbour
    // can sit two cells away, where a 27-cell scan never looks. That would silently
    // drop first-shell pairs and leave the BCC estimate reading 3.5 A, a second-shell
    // distance.
    let mut rad = [1i64; 3];
    for d in 0..3 {
        let w = span[d] / dims[d] as f64;
        rad[d] = (bin / w).ceil() as i64;
    }

    let ncells = dims[0] * dims[1] * dims[2];

    let cell_of = |p: Coord3| -> Option<usize> {
        let mut c = [0usize; 3];
        for d in 0..3 {
            let f = (p[d] - lo[d]) / span[d].max(f64::MIN_POSITIVE);
            if !(0.0..1.0).contains(&f) {
                return None;
            }
            c[d] = ((f * dims[d] as f64) as usize).min(dims[d] - 1);
        }
        Some((c[0] * dims[1] + c[1]) * dims[2] + c[2])
    };

    // Head/next linked list: two flat arrays, unlike a Vec per bin, which would
    // cost one allocation per bin.
    let mut head = vec![-1i32; ncells];
    let mut next = vec![-1i32; positions.len()];
    for (a, &p) in positions.iter().enumerate() {
        if let Some(c) = cell_of(p) {
            next[a] = head[c];
            head[c] = a as i32;
        }
    }

    // Everything the scan reaches, out to a quarter of the hint beyond it.
    //
    // Three things had to be true of this estimator at once, and each one broke it
    // differently:
    //
    // - Not a minimum. The nearest neighbour of each first-shell atom is the
    //   *smallest* of 12 draws from the spacing distribution, so it reads low by
    //   construction, and taking the median over atoms does not recover it: 2.3836 A
    //   against a true 2.4932 A on the HEA reference, an error of the estimator and
    //   not of the data, and it started every fit 4% short.
    // - Not a per-atom statistic. A corner atom of a 3x3x3 BCC block has one
    //   first-shell neighbour and three second-shell ones, so its own median *is* a
    //   second-shell distance -- and half the atoms in a block that small are corners.
    //   Pooled, the first shell is the majority: 125 pairs at 0.866a against 108 at a.
    // - Not a histogram alone. Binned over the whole scanned range, the bins are
    //   ~1e-3 A wide, which is 200x coarser than the exactness a perfect lattice
    //   can be checked to. The bin only *locates* the answer; the distances themselves
    //   are kept and the estimate is taken from them exactly.
    //
    // The window starts at zero, not at three quarters of the hint. The hint is
    // only a bin size -- that is the contract this function documents -- so the
    // window must not assume the first shell is *near* the hint. It need not be:
    // a grain's lattice constant seeded from its bounding box can come out 65%
    // high, the hint with it, and a window of [0.75h, 1.25h] then brackets the
    // *third* coordination shell of a BCC lattice rather than the first. The
    // estimate reads 4.38 A against a true 2.68 A, the fitted reference is
    // 63% too coarse, and a defect-free grain reports 39582 vacancies.
    //
    // Starting the window at zero costs a wider histogram and nothing else: the
    // shell that is chosen is still the shortest populated run, which is the
    // first shell whatever the hint was.
    let lo2 = 0.0f64;
    let hi2 = (1.25 * bin) * (1.25 * bin);
    let span2 = hi2 - lo2;

    // Bounded, because this runs on every atom of the search sample and a sample of
    // 100k atoms has 1.2M first-shell pairs in it. A million is far more than the
    // shell mean needs to be stable.
    const CAP: usize = 1 << 20;
    let mut shell: Vec<f64> = Vec::with_capacity(CAP.min(positions.len() * 8));
    let mut fallback: Vec<f64> = Vec::with_capacity(positions.len().min(8192));

    for (a, &p) in positions.iter().enumerate() {
        let c = match cell_of(p) {
            Some(c) => c,
            None => continue,
        };
        let ci = [
            c / (dims[1] * dims[2]),
            (c / dims[2]) % dims[1],
            c % dims[2],
        ];
        let mut nearest = f64::INFINITY;
        for a0 in -rad[0]..=rad[0] {
            for a1 in -rad[1]..=rad[1] {
                for a2 in -rad[2]..=rad[2] {
                    let mut q = [0i64; 3];
                    let mut ok = true;
                    for d in 0..3 {
                        let v = ci[d] as i64 + [a0, a1, a2][d];
                        if v < 0 || v >= dims[d] as i64 {
                            ok = false;
                            break;
                        }
                        q[d] = v;
                    }
                    if !ok {
                        continue;
                    }
                    let nc = (q[0] as usize * dims[1] + q[1] as usize) * dims[2] + q[2] as usize;
                    let mut b = head[nc];
                    while b >= 0 {
                        let bi = b as usize;
                        if bi != a {
                            let d = [
                                p[0] - positions[bi][0],
                                p[1] - positions[bi][1],
                                p[2] - positions[bi][2],
                            ];
                            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                            if r2 > 0.0 {
                                if r2 < nearest {
                                    nearest = r2;
                                }
                                if r2 >= lo2 && r2 <= hi2 && shell.len() < CAP {
                                    shell.push(r2);
                                }
                            }
                        }
                        b = next[bi];
                    }
                }
            }
        }
        if nearest.is_finite() {
            fallback.push(nearest.sqrt());
        }
    }

    if shell.is_empty() {
        // A hint far enough off to put the real neighbours outside the window is
        // still better served by a biased low estimate than by dropping every atom.
        if fallback.is_empty() {
            return Err(Error::invalid(
                "no neighbour found; check the position hint",
            ));
        }
        fallback.sort_by(f64::total_cmp);
        return Ok(fallback[fallback.len() / 2]);
    }

    // The first shell is the shortest run of populated bins carrying a real share of
    // the pairs, and the estimator is the mean distance over that run.
    //
    // Two failures had to be told apart to get here.
    //
    // A *median* over the window is the wrong statistic. On a 3-cell BCC block the
    // shells are 64 pairs against 66, a coin flip a small block genuinely loses, and
    // the median reported the second shell's 3.5 A for a lattice whose spacing is
    // 3.031 A. Selecting by shell *rank* does not depend on the shells being uneven:
    // the first shell is the shortest one, on any lattice and at any size.
    //
    // The shortest *run* alone is not enough either, and the alloy is why. A relaxed
    // solid has a few anomalously short pairs -- thermal fluctuation tails, an
    // occasional bad contact -- and those land in the histogram as a small cluster of
    // bins isolated from everything else. The HEA reference has such a cluster at
    // 2.167 A, and taking the shortest run selects it and nothing else, moving the
    // estimate 13% low and every fit with it. A coordination shell always carries a
    // large share of the pairs; a handful of outliers never do, and the run has to
    // hold at least 1% of them to count as a shell. For reference: a perfect lattice
    // puts about half its pairs in the first shell, and a dense alloy a bit less than
    // half, so 1% is a floor only an outlier cluster falls through.
    //
    // Runs, not individual bins, because a first shell is not a delta: the HEA first
    // shell spreads over 2.28-2.71 A, so the lowest populated bin is the tail of that
    // spread and taking it alone is the 4%-low per-atom minimum again. Bins are
    // also allowed to be empty inside a run, and three in a row end it -- a perfect
    // lattice leaves 1364 of them between its shells, and a sparse one is not cut in
    // half by a single dropout.
    const HIST: usize = 4096;
    let mut hist = [0u64; HIST];
    for &r2 in &shell {
        hist[(((r2 - lo2) / span2 * (HIST - 1) as f64) as usize).min(HIST - 1)] += 1;
    }
    let width = span2 / (HIST - 1) as f64;
    let min_pairs = (((shell.len() as f64) * 0.01).ceil() as u64).max(1);

    let mut chosen = None;
    let mut i = 0usize;
    while i < HIST {
        if hist[i] == 0 {
            i += 1;
            continue;
        }
        let start = i;
        let mut last = i;
        let mut gap = 0usize;
        while i < HIST {
            if hist[i] > 0 {
                last = i;
                gap = 0;
                i += 1;
            } else {
                gap += 1;
                if gap > 2 {
                    break;
                }
                i += 1;
            }
        }
        let count: u64 = hist[start..=last].iter().sum();
        if count >= min_pairs {
            chosen = Some((start, last));
            break;
        }
    }

    let (mut sum, mut cnt) = (0.0f64, 0usize);
    if let Some((start, last)) = chosen {
        let lo_run = lo2 + start as f64 * width;
        let hi_run = lo2 + (last + 1) as f64 * width;
        for &r2 in &shell {
            if r2 >= lo_run && r2 < hi_run {
                sum += r2.sqrt();
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        shell.sort_by(f64::total_cmp);
        let m = shell.len() / 2;
        return Ok(shell[m].sqrt());
    }
    Ok(sum / cnt as f64)
}

/// Multiply every lattice vector by `s`, which scales volume by `s³`.
fn scale_rows(pc: &PrimitiveCell, s: f64) -> PrimitiveCell {
    PrimitiveCell::new(
        Mat3::from_rows(
            [pc.h.m[0][0] * s, pc.h.m[0][1] * s, pc.h.m[0][2] * s],
            [pc.h.m[1][0] * s, pc.h.m[1][1] * s, pc.h.m[1][2] * s],
            [pc.h.m[2][0] * s, pc.h.m[2][1] * s, pc.h.m[2][2] * s],
        ),
        pc.basis.clone(),
    )
}

/// Rotate a candidate about the box's lower corner.
///
/// About `lo` rather than the global origin, because that is the point the region
/// is actually attached to; rotating about `(0,0,0)` would swing a lattice in a
/// box that does not touch the origin.
fn rotate_about(c: &Candidate, lo: Coord3, r: &Mat3) -> Candidate {
    Candidate {
        pc: PrimitiveCell::new(mul_mat(r, &c.pc.h), c.pc.basis.clone()),
        origin: add(lo, (*r).mul_v(sub(c.origin, lo))),
    }
}

/// Origins spanning the fractional cell, anchored at the box corner.
///
/// The offsets are in lattice coordinates, so `origin_grid: 1` reproduces the
/// input's origin exactly and larger values explore the cell.
fn origin_grid(base: &Candidate, n: usize) -> Vec<Candidate> {
    let n = n.max(1);
    let step = 1.0 / n as f64;
    let mut out = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let f = [i as f64 * step, j as f64 * step, k as f64 * step];
                // Offsets from the candidate's own anchor. Spreading them from the
                // box corner instead would put the whole scan 60 A from a grain
                // sitting away from the corner, so none of it would touch the atoms.
                out.push(Candidate {
                    pc: base.pc.clone(),
                    origin: add(base.origin, base.pc.h.mul_v(f)),
                });
            }
        }
    }
    out
}

/// Diminishing-step pattern search over the three rotation axes.
///
/// The step halves every round rather than only after a round that finds nothing,
/// for the same reason as `refine_origin`: a round that improved still leaves the
/// step far coarser than the residual, so without this the descent re-walks the same
/// 2.9 degree step forever and a tilted grain keeps 0.06 A it could have closed.
///
/// Returns the score, the candidate and the rotation it settled on.
#[allow(clippy::too_many_arguments)]
fn descend(
    mut cur: Candidate,
    mut mean: f64,
    mut rot: Mat3,
    pivot: Coord3,
    sample: &[Coord3],
    opts: &FitOptions,
    trials: &mut usize,
) -> Result<(f64, Candidate, Mat3)> {
    let halvings = (opts.angle_step / opts.angle_tol).log2().ceil().max(1.0) as usize;
    let max_passes = halvings + opts.max_rounds.max(1);
    let mut step_size = opts.angle_step;
    let mut passes = 0usize;
    while step_size > opts.angle_tol && passes < max_passes {
        for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            for sign in [1.0f64, -1.0] {
                let step = axis_angle(axis, sign * step_size);
                let cand = rotate_about(&cur, pivot, &step);
                *trials += 1;
                // Two origin rounds are enough to judge a rotation step; the full
                // refinement is only worth paying for once the step is accepted.
                let (sc, refined) = refine_origin(&cand, sample, 2, trials)?;
                if sc < mean {
                    rot = mul_mat(&step, &rot);
                    cur = refined;
                    mean = sc;
                }
            }
        }
        step_size *= 0.5;
        passes += 1;
    }
    Ok((mean, cur, rot))
}

/// Coarse grid over a full fractional cell, then local refinement.
///
/// A fractional offset is only ever meaningful modulo one cell, so scanning a whole
/// cell covers every anchor the lattice can take; the local pass then polishes it.
fn origin_search(
    start: &Candidate,
    sample: &[Coord3],
    grid: usize,
    rounds: usize,
    trials: &mut usize,
) -> Result<(f64, Candidate)> {
    let mut best = (mean_displacement_of(start, sample)?, start.clone());
    for cand in origin_grid(start, grid) {
        *trials += 1;
        let sc = mean_displacement_of(&cand, sample)?;
        if sc < best.0 {
            best = (sc, cand);
        }
    }
    refine_origin(&best.1, sample, rounds, trials)
}

/// Nudge the origin to the sample's best fit.
///
/// Returns the sample's mean displacement, not a [`FitScore`]: on a sample the
/// defect counts are meaningless -- an unsampled vacancy is simply an absent atom
/// -- so this stage optimises the one part of the score a sample can measure.
fn refine_origin(
    start: &Candidate,
    sample: &[Coord3],
    rounds: usize,
    trials: &mut usize,
) -> Result<(f64, Candidate)> {
    let mut best_score = mean_displacement_of(start, sample)?;
    let mut best = start.clone();
    // Half a cell is the most a fractional offset can mean; start just under it
    // so the first step is meaningful.
    let mut delta = 0.5;
    for _ in 0..rounds.max(1) {
        for d in 0..3 {
            for sign in [1.0f64, -1.0] {
                let mut f = [0.0f64; 3];
                f[d] = sign * delta;
                let cand = Candidate {
                    pc: best.pc.clone(),
                    origin: add(best.origin, best.pc.h.mul_v(f)),
                };
                *trials += 1;
                let sc = mean_displacement_of(&cand, sample)?;
                if sc < best_score {
                    best_score = sc;
                    best = cand;
                }
            }
        }
        // Halve on every pass, improving or not. Halving only after a failure sounds
        // like it saves work, but a pass that improved still leaves the step far
        // larger than the residual, so the next pass re-walks the same ground at the
        // same coarse size and stops 0.05 A short. A fixed geometric schedule
        // converges instead: 0.5 halving away over `rounds`.
        delta *= 0.5;
    }
    Ok((best_score, best))
}

fn sub(a: Coord3, b: Coord3) -> Coord3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Coord3, b: Coord3) -> Coord3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::bcc::Bcc;
    use crate::lattice::fcc::Fcc;
    use crate::lattice::sites::SiteLattice;

    /// Wrap into `[lo, hi)` per axis.
    ///
    /// Rotating a lattice that fills the box sends some sites outside, and the fit
    /// only assigns atoms to sites inside the box, so unwrapped positions read as
    /// vacancies. Real input is always wrapped (`ParticleData`: "Positions are
    /// stored wrapped (in-cell)"), so the test has to wrap too — otherwise it
    /// measures the harness rather than the fit (three spurious defects out of
    /// 2048).
    fn wrap(p: Coord3, lo: f64, len: f64) -> Coord3 {
        [
            lo + (p[0] - lo) - ((p[0] - lo) / len).floor() * len,
            lo + (p[1] - lo) - ((p[1] - lo) / len).floor() * len,
            lo + (p[2] - lo) - ((p[2] - lo) / len).floor() * len,
        ]
    }

    fn fcc_block(a: f64, n: usize) -> (UnitCell, Vec<Coord3>) {
        let cell = UnitCell::orthorhombic(n as f64 * a, n as f64 * a, n as f64 * a);
        let sites = SiteLattice::build(&cell, &Fcc { a }.primitive(), [0.0; 3]).unwrap();
        (cell, sites.iter().collect())
    }

    fn bcc_block(a: f64, n: usize) -> (UnitCell, Vec<Coord3>) {
        let cell = UnitCell::orthorhombic(n as f64 * a, n as f64 * a, n as f64 * a);
        let sites = SiteLattice::build(&cell, &Bcc { a }.primitive(), [0.0; 3]).unwrap();
        (cell, sites.iter().collect())
    }

    fn fit(cell: &UnitCell, pos: &[Coord3], a: f64, opts: &FitOptions) -> RegionFit {
        fit_region(cell, pos, &Fcc { a }, a / 2f64.sqrt(), opts).unwrap()
    }

    /// `InRegionSites::position` must land on the site each atom actually claimed.
    ///
    /// The displacement is computed by a completely different route --
    /// `nearest_lattice`, inside the assignment loop -- so `atom - displacement`
    /// is an *independent* position for every occupied site. Agreement is what
    /// makes the vacancy sites trustworthy: those have no atom and so no such
    /// cross-check, and they come out of the same index arithmetic. If that
    /// arithmetic were wrong, an occupied site would miss too.
    #[test]
    fn in_region_site_positions_agree_with_the_assigned_displacements() {
        let (cell, positions) = fcc_block(3.5, 4);
        let cand = Candidate {
            pc: Fcc { a: 3.5 }.primitive(),
            origin: [0.0; 3],
        };
        let mut per_atom = PerAtom::default();
        let par = crate::exec::Par::new(1);
        let assign = assign_region(
            &cell,
            &cand,
            &positions,
            1.0,
            Some(&par),
            Some(&mut per_atom),
        )
        .unwrap();

        assert_eq!(assign.sites.len(), assign.occupancy.len());
        let mut checked = 0usize;
        for (i, &slot) in per_atom.site_of_atom.iter().enumerate() {
            if slot < 0 {
                continue;
            }
            let site = assign.sites.position(slot as usize);
            let d = &per_atom.displacement[i * 3..i * 3 + 3];
            for k in 0..3 {
                let derived = positions[i][k] - d[k];
                assert!(
                    (derived - site[k]).abs() < 1e-9,
                    "atom {i} slot {slot} axis {k}: derived {derived} vs site {}",
                    site[k]
                );
            }
            checked += 1;
        }
        assert!(
            checked > positions.len() / 2,
            "too few occupied sites cross-checked: {checked} of {}",
            positions.len()
        );
    }

    /// The nearest-neighbour estimate is what sets the fit's initial scale, so it has
    /// to be right for every prototype, not just FCC, and on a block small enough
    /// that most atoms are on the surface.
    #[test]
    fn nearest_neighbour_estimate_matches_the_prototype_lattice() {
        let a = 3.5;
        for n in [3usize, 8] {
            let (cell, sites) = fcc_block(a, n);
            let want = a / 2f64.sqrt();
            let got = estimate_nn(&cell, &sites, want).unwrap();
            assert!(
                (got - want).abs() < 1e-9,
                "FCC n={n}: got {got}, want {want}"
            );

            let (bcell, bsites) = bcc_block(a, n);
            let bwant = a * 3f64.sqrt() / 2.0;
            let bgot = estimate_nn(&bcell, &bsites, bwant).unwrap();
            assert!(
                (bgot - bwant).abs() < 1e-9,
                "BCC n={n}: got {bgot}, want {bwant}"
            );
        }
    }

    #[test]
    fn nn_distance_reproduces_analytic_values() {
        let a = 3.5;
        assert!((nn_distance(&Fcc { a }.primitive()).unwrap() - a / 2f64.sqrt()).abs() < 1e-12);
        assert!(
            (nn_distance(&Bcc { a }.primitive()).unwrap() - a * 3f64.sqrt() / 2.0).abs() < 1e-12
        );
    }

    /// Matrix -> quaternion -> matrix for rotations that hit each of
    /// Shepperd's four branches (trace > 0, then x, y or z largest). A branch
    /// with the sign of `w` flipped returns the *inverse* rotation; a
    /// quaternion-to-quaternion round trip cannot see that.
    #[test]
    fn matrix_round_trips_through_a_quaternion_on_every_branch() {
        for (axis, angle) in [
            ([0.3, -0.2, 0.9], 0.7),
            ([1.0, 0.1, 0.05], 2.9),
            ([0.1, 1.0, -0.05], 2.9),
            ([0.05, -0.1, 1.0], 2.9),
            ([-0.06, -0.34, 0.83], 2.23),
        ] {
            let r = axis_angle(axis, angle);
            let back = quat_to_mat(mat_to_quat(&r));
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (back.m[i][j] - r.m[i][j]).abs() < 1e-9,
                        "axis {axis:?} angle {angle}: {back:?} vs {r:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn quaternion_round_trips_through_a_matrix() {
        for q in [
            [1.0, 0.0, 0.0, 0.0],
            mat_to_quat(&axis_angle([0.0, 0.0, 1.0], 0.6)),
            mat_to_quat(&axis_angle([1.0, 1.0, 1.0], 2.9)),
        ] {
            let back = mat_to_quat(&quat_to_mat(q));
            assert!((back[0] - q[0]).abs() < 1e-9, "{q:?} -> {back:?}");
            assert!((back[1] - q[1]).abs() < 1e-9, "{q:?} -> {back:?}");
            assert!((back[2] - q[2]).abs() < 1e-9, "{q:?} -> {back:?}");
            assert!((back[3] - q[3]).abs() < 1e-9, "{q:?} -> {back:?}");
        }
    }

    #[test]
    fn perfect_crystal_fits_with_zero_defects() {
        let a = 3.5;
        let (cell, pos) = fcc_block(a, 8);
        assert_eq!(pos.len(), 2048);
        let f = fit(&cell, &pos, a, &FitOptions::default());
        // Exactly, not approximately. The region is the cells the atoms occupy and
        // the gauge is the one whose region is tightest, so an 8-cell block is 8 cells
        // along every axis however the origin search happened to stop.
        assert_eq!(f.score.n_sites, pos.len(), "{:?}", f.score);
        assert_eq!(f.score.vacancies, 0, "perfect FCC reported vacancies");
        assert_eq!(
            f.score.interstitials, 0,
            "perfect FCC reported interstitials"
        );
        assert!(
            f.score.mean_displacement < 1e-3,
            "mean {}",
            f.score.mean_displacement
        );
    }

    #[test]
    fn planted_vacancies_are_recovered() {
        let a = 3.5;
        let (cell, pos) = fcc_block(a, 8);
        // Drop atoms at distinct, well-separated indices, front to back so the
        // later indices stay valid.
        let mut atoms = pos.clone();
        for i in 0..4 {
            atoms.remove(i * 401);
        }
        let f = fit(&cell, &atoms, a, &FitOptions::default());
        assert_eq!(f.score.n_atoms, 2044);
        assert_eq!(f.score.vacancies, 4, "vacancies: {:?}", f.score);
        assert_eq!(f.score.interstitials, 0, "interstitials: {:?}", f.score);
    }

    #[test]
    fn planted_interstitials_are_recovered() {
        let a = 3.5;
        let (cell, pos) = fcc_block(a, 8);
        // Sit each extra atom on the midpoint between two lattice sites a/2 apart.
        // Its nearest site is one of those two, so it registers as a second
        // occupant. Three well-separated sites keep the three from colliding.
        let mut atoms = pos.clone();
        for k in [10usize, 1000, 2000] {
            let p = atoms[k];
            atoms.push([p[0] + a / 2.0, p[1], p[2]]);
        }
        let f = fit(&cell, &atoms, a, &FitOptions::default());
        assert_eq!(f.score.n_atoms, 2051);
        assert_eq!(f.score.vacancies, 0, "vacancies: {:?}", f.score);
        assert_eq!(f.score.interstitials, 3, "interstitials: {:?}", f.score);
    }

    /// The score an exact reference produces, which the fit has to reach.
    ///
    /// This is the ground truth the rotated-grain test is measured against: a 14 A
    /// FCC block rotated by 0.6 rad, and it has to come out of the region unscathed.
    ///
    /// The bounding box of the atoms of a rotated grain is not a whole number of cells
    /// in any direction: it holds 546 sites where the grain has 256 atoms, so an exact
    /// reference would score 290 vacancies as a region of that box -- an artefact of
    /// the region, not defects in the reference. Bounding the region by the cells the
    /// atoms actually occupy gives 256 sites and none.
    #[test]
    fn exact_candidate_is_the_reference_a_fit_has_to_reach() {
        let a = 3.5;
        let n = 4usize;
        let len = n as f64 * a;
        let (_, sites) = fcc_block(a, n);
        let half = [len / 2.0; 3];
        let r = mul_mat(
            &axis_angle([0.0, 0.0, 1.0], 0.6),
            &axis_angle([1.0, 1.0, 1.0], 0.4),
        );
        let mid = 30.0;
        let cell = UnitCell::orthorhombic(60.0, 60.0, 60.0);
        let pos: Vec<Coord3> = sites
            .iter()
            .map(|p| add(r.mul_v(sub(*p, half)), [mid; 3]))
            .collect();
        let exact = Candidate {
            pc: PrimitiveCell::new(
                mul_mat(
                    &r,
                    &Mat3::from_rows([a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]),
                ),
                Fcc { a }.primitive().basis.clone(),
            ),
            origin: sub([mid; 3], r.mul_v(half)),
        };
        let tol = 0.5 * (a / 2f64.sqrt());
        let sc = score_candidate(&cell, &exact, &pos, tol).unwrap();
        assert!(sc.mean_displacement < 1e-12, "{sc:?}");
        assert_eq!(sc.interstitials, 0, "{sc:?}");
        assert_eq!(sc.n_atoms, 256, "{sc:?}");
        assert_eq!(sc.n_sites, 256, "{sc:?}");
        assert_eq!(sc.vacancies, 0, "{sc:?}");
    }

    /// The threaded final pass must agree with the sequential one *exactly*,
    /// bit for bit.
    ///
    /// Two ways this could have gone wrong, both invisible in the counts alone:
    /// the occupancy counters are atomic and unordered, and `mean_displacement`
    /// sums floats. Float addition is not associative, so a pass that summed per
    /// thread and combined would report a slightly different mean for every
    /// `--threads` value -- and §10.2 promises the output does not depend on
    /// thread count. Gathering the per-atom results in index order before
    /// reducing is what makes that hold, so the test compares the raw buffers
    /// rather than just the totals.
    #[test]
    fn threaded_assignment_matches_the_sequential_one() {
        let a = 3.5;
        let (cell, pos) = fcc_block(a, 6);
        // Perturb so displacements are non-zero and therefore order sensitive.
        let atoms: Vec<Coord3> = pos
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d = 0.013 * ((i % 7) as f64 - 3.0);
                [p[0] + d, p[1] - d, p[2] + 0.5 * d]
            })
            .collect();
        let cand = Candidate {
            pc: Fcc { a }.primitive(),
            origin: [0.0; 3],
        };
        let tol = 0.5 * (a / 2f64.sqrt());

        let mut one = PerAtom::default();
        let seq = assign_region(
            &cell,
            &cand,
            &atoms,
            tol,
            Some(&crate::exec::Par::new(1)),
            Some(&mut one),
        )
        .unwrap();

        let mut many = PerAtom::default();
        let par = crate::exec::Par::new(8);
        let threaded =
            assign_region(&cell, &cand, &atoms, tol, Some(&par), Some(&mut many)).unwrap();

        assert_eq!(seq.score, threaded.score);
        assert_eq!(seq.occupancy, threaded.occupancy);
        assert_eq!(one.site_of_atom, many.site_of_atom);
        assert_eq!(
            one.displacement, many.displacement,
            "per-atom displacement vectors must not depend on thread count"
        );
        assert_eq!(one.site_of_atom.len(), atoms.len());
        assert!(
            seq.score.mean_displacement > 0.0,
            "test needs non-zero work"
        );
    }

    /// A free-standing rotated grain, seeded from a quaternion, is recovered
    /// exactly.
    ///
    /// This is the case the whole module exists for: PTM hands over a quaternion
    /// for a grain, and the fitter must land on that grain's lattice.
    ///
    /// The grain is a *finite block*, not a rotated copy of a periodic block, and
    /// that distinction is the whole test. Wrapping a rotated block into a
    /// periodic box shifts each atom by a different lattice vector, so unless the
    /// rotation is commensurate with the box the atom set is not a Bravais lattice
    /// at all and no fit can be exact -- it floors out around 0.7 A of mean
    /// displacement. Real MD does contain commensurate orientations, but a grain
    /// inside a polycrystal is a finite piece of one, so this is the fixture worth
    /// pinning.
    #[test]
    fn finite_rotated_grain_is_recovered_from_a_ptm_style_seed() {
        let a = 3.5;
        let n = 4usize;
        let len = n as f64 * a;
        let (_, sites) = fcc_block(a, n);
        let half = [len / 2.0; 3];
        let r = mul_mat(
            &axis_angle([0.0, 0.0, 1.0], 0.6),
            &axis_angle([1.0, 1.0, 1.0], 0.4),
        );
        // Room for the block's corners once rotated; the grain stays well inside
        // the box, so nothing wraps.
        let mid = 30.0;
        let cell = UnitCell::orthorhombic(60.0, 60.0, 60.0);
        let pos: Vec<Coord3> = sites
            .iter()
            .map(|p| add(r.mul_v(sub(*p, half)), [mid; 3]))
            .collect();

        // Seed perturbed by about a degree per axis, so this exercises the descent
        // rather than asserting the seed straight through.
        let mut q = mat_to_quat(&r);
        for d in q.iter_mut().skip(1) {
            *d += 0.017;
        }
        let opts = FitOptions {
            seed_orientation: Some(to_orientation(q)),
            ..FitOptions::default()
        };

        let f = fit(&cell, &pos, a, &opts);
        // Every atom gets a site of its own, so nothing is an interstitial.
        assert_eq!(
            f.score.interstitials, 0,
            "seeded rotated grain: {:?}",
            f.score
        );
        assert!(
            // Looser than the aligned case because the search starts several degrees
            // out and spends its rounds closing that before it can polish. Still two
            // orders below a thermal amplitude.
            f.score.mean_displacement < 1e-2,
            "seeded rotated grain: {:?}",
            f.score
        );
        // Zero vacancies too. With the region taken as the atoms' bounding box this
        // could not hold: a 14 A block tilted by 0.6 rad has a box holding 546
        // lattice sites where the grain has 256 atoms, so an exact reference would
        // score 290 vacancies. Bounding the region by the cells the atoms
        // occupy removes the excess, and a rotated grain has to come out as clean as an
        // aligned one -- that is the whole claim of the region rule.
        assert_eq!(
            f.score.vacancies, 0,
            "a rotated grain should be as clean as an aligned one: {:?}",
            f.score
        );
        assert_eq!(f.score.n_sites, pos.len(), "{:?}", f.score);
        assert_eq!(f.score.n_atoms, pos.len());
        assert!(
            (quat_to_mat(f.orientation).det() - 1.0).abs() < 1e-9,
            "orientation not a rotation"
        );
    }

    /// Guards the honest-failure requirement from the module docs.
    ///
    /// Unseeded, the search can only refine locally, so a grain rotated well away
    /// from the seed cell is *not* recovered. This test pins that down
    /// deliberately: if a future global search makes it converge, this test fails
    /// and the docs need revisiting, rather than the improvement passing unnoticed.
    /// The important part is that the fit reports a visibly wrong answer -- a large
    /// displacement -- instead of quietly reporting zero defects.
    #[test]
    fn unseeded_fit_of_a_rotated_grain_is_not_silently_wrong() {
        let a = 3.5;
        let n = 4usize;
        let len = n as f64 * a;
        let (_, sites) = fcc_block(a, n);
        let half = [len / 2.0; 3];
        let r = mul_mat(
            &axis_angle([0.0, 0.0, 1.0], 0.6),
            &axis_angle([1.0, 1.0, 1.0], 0.4),
        );
        let mid = 30.0;
        let cell = UnitCell::orthorhombic(60.0, 60.0, 60.0);
        let pos: Vec<Coord3> = sites
            .iter()
            .map(|p| add(r.mul_v(sub(*p, half)), [mid; 3]))
            .collect();

        let f = fit(&cell, &pos, a, &FitOptions::default());
        assert!(
            f.score.mean_displacement > 0.1,
            "unseeded fit of a rotated grain must not look good: {:?}",
            f.score
        );
    }

    /// An incommensurate rotation in a *periodic* box is not a Bravais lattice, and
    /// the fit must say so rather than claim a perfect crystal.
    ///
    /// Wrapping a rotated block into a periodic box displaces each atom by a
    /// different lattice vector, so the wrapped positions are not lattice points of
    /// any single cell. This is a property of the fixture, not a fitter limitation,
    /// and it is worth pinning: it is the reason the rotated-grain tests use a
    /// finite block, and it looks exactly like a fitter bug if met later without
    /// this context.
    #[test]
    fn incommensurate_wrap_is_reported_as_a_bad_fit_not_a_perfect_crystal() {
        let a = 3.5;
        let n = 8usize;
        let len = n as f64 * a;
        let (cell, sites) = fcc_block(a, n);
        let r = mul_mat(
            &axis_angle([0.0, 0.0, 1.0], 0.6),
            &axis_angle([1.0, 1.0, 1.0], 0.4),
        );
        let pos: Vec<Coord3> = sites.iter().map(|p| wrap(r.mul_v(*p), 0.0, len)).collect();
        let opts = FitOptions {
            seed_orientation: Some(to_orientation(mat_to_quat(&r))),
            ..FitOptions::default()
        };

        let f = fit(&cell, &pos, a, &opts);
        assert!(
            f.score.mean_displacement > 0.1,
            "an incommensurate wrap cannot fit perfectly, so reporting it as such \
             is required, not a failure of the fitter: {:?}",
            f.score
        );
    }
}
