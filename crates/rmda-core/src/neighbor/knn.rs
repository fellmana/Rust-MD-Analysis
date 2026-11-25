//! Exact, unbounded k-nearest-neighbour index (design §9.1–§9.2).
//!
//! Atoms are counting-sorted into bins of fractional space (CSR: `start` per
//! bin, positions stored contiguously in bin order). A query visits bins in
//! Chebyshev shells around the query's home bin and keeps the best `k` in a
//! small sorted array. It stops once the k-th distance cannot be beaten by any
//! atom outside the shells visited so far.
//!
//! The termination bound is exact: an atom outside the `(2s+1)³` block of bins
//! around the home bin lies beyond one of the block's six faces, and the
//! Cartesian distance to a face is its fractional distance times the cell's
//! perpendicular height on that axis. So a query never depends on a search
//! radius, a density or the build cutoff. That is what makes it correct for
//! hot, expanded lattices and free surfaces, and cheap with vacuum: a
//! non-periodic axis is binned over the atoms' own extent rather than over the
//! box, and the bin size comes from the occupied volume rather than the box
//! volume.
//!
//! Periodic images are explicit: a bin offset that runs past the end of a
//! periodic axis is the next image of that bin. Every image of every atom is a
//! candidate exactly once, so a cell smaller than the k-th neighbour distance
//! returns the right images, and the query atom's own images are excluded the
//! same way the cell-list path (`nearest_k_excluding_self`) excludes them.

use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::{Error, Result};
use crate::exec::Par;

/// Atoms per bin the grid aims for. Smaller bins mean more empty bins to visit;
/// larger bins mean more candidates per bin. Two to four is the usual optimum
/// for k ≈ 12–18.
const TARGET_PER_BIN: f64 = 3.0;

/// Coarse grid resolution used to measure the occupied volume.
const COARSE_PER_AXIS: usize = 48;

/// Largest supported `k`.
pub const MAX_K: usize = 64;

/// One neighbour: atom index, displacement from the query (image-true) and
/// squared distance.
pub type Nbr = (u32, Coord3, f64);

/// A built index over one frame's positions.
pub struct KnnIndex {
    /// Lattice vectors (rows), for image shifts.
    h: Mat3,
    /// `(H⁻¹)ᵀ`, for fractional coordinates.
    inv_t: Mat3,
    origin: Coord3,
    pbc: [bool; 3],
    /// Bins per axis.
    n: [usize; 3],
    /// Fractional range binned per axis: `[lo, lo + span)`. `[0, 1)` on
    /// periodic axes, the atoms' own extent on open ones.
    lo: [f64; 3],
    span: [f64; 3],
    /// Cartesian perpendicular thickness of one bin per axis.
    gap: [f64; 3],
    /// Whether the lattice vectors are the Cartesian axes, so the per-axis
    /// slab distances to a bin are orthogonal and combine as a Euclidean sum.
    ortho: bool,
    /// CSR offsets, `nbins + 1` entries.
    start: Vec<u32>,
    /// Positions in bin order, wrapped into the cell on periodic axes.
    pos: Vec<Coord3>,
    /// Original atom index of each slot.
    idx: Vec<u32>,
}

/// Per-thread query scratch. Create one per worker and reuse it.
#[derive(Default)]
pub struct KnnScratch {
    /// Best candidates so far, sorted by (d², atom, image).
    best: Vec<(f64, u32, [i32; 3], Coord3)>,
    /// The current shell's bins with their distance lower bound, so they are
    /// scanned nearest first.
    bins: Vec<(f64, usize, [i32; 3])>,
}

impl KnnIndex {
    /// Build the index over `positions`.
    pub fn build(cell: &UnitCell, positions: &[Coord3], par: &Par) -> Result<Self> {
        let n_atoms = positions.len();
        if n_atoms >= u32::MAX as usize {
            return Err(Error::invalid("k-NN index supports fewer than 2^32 atoms"));
        }
        let h = *cell.h_matrix();
        let inv_t = cell.inverse_transposed()?;
        let origin = cell.origin;
        let pbc = cell.pbc;
        let heights = cell.heights();

        // Fractional coordinates, wrapped on periodic axes. Recomputed per pass
        // rather than stored: a stored copy is 24 B per atom.
        let frac = |i: usize| -> Coord3 {
            let p = positions[i];
            let d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
            let mut l = inv_t.mul_v(d);
            wrap_periodic(&mut l, pbc);
            l
        };

        // Binned range per axis.
        let mut lo = [0.0f64; 3];
        let mut span = [1.0f64; 3];
        let extents = par.map_chunks(n_atoms, |a, b| {
            let mut mn = [f64::INFINITY; 3];
            let mut mx = [f64::NEG_INFINITY; 3];
            for i in a..b {
                let l = frac(i);
                for d in 0..3 {
                    mn[d] = mn[d].min(l[d]);
                    mx[d] = mx[d].max(l[d]);
                }
            }
            (mn, mx)
        });
        for a in 0..3 {
            if !pbc[a] && n_atoms > 0 {
                let (mn, mx) = extents
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), e| {
                        (mn.min(e.0[a]), mx.max(e.1[a]))
                    });
                let width = (mx - mn).max(1e-12);
                lo[a] = mn;
                // A hair wider than the extent so the top atom is inside.
                span[a] = width * (1.0 + 1e-9) + 1e-12;
            }
        }
        let extent = [
            heights[0] * span[0],
            heights[1] * span[1],
            heights[2] * span[2],
        ];

        // Bin width from the *occupied* volume, so vacuum does not dilute it.
        let occupied = occupied_volume(n_atoms, &frac, &lo, &span, extent, par);
        let w = (occupied * TARGET_PER_BIN / (n_atoms.max(1) as f64)).cbrt();
        let mut n = [1usize; 3];
        for a in 0..3 {
            if extent[a].is_finite() && extent[a] > 0.0 && w.is_finite() && w > 0.0 {
                n[a] = ((extent[a] / w).floor() as usize).max(1);
            }
        }
        // Never more bins than atoms (times two): `start` is dense.
        let cap = (2 * n_atoms).max(1) as f64;
        while (n[0] as f64) * (n[1] as f64) * (n[2] as f64) > cap {
            let a = (0..3).max_by_key(|&a| n[a]).unwrap();
            if n[a] == 1 {
                break;
            }
            n[a] = (n[a] / 2).max(1);
        }
        let nbins = n[0] * n[1] * n[2];
        let gap = [
            extent[0] / n[0] as f64,
            extent[1] / n[1] as f64,
            extent[2] / n[2] as f64,
        ];

        // Counting sort into bins.
        let bin_of = |l: &Coord3| -> usize {
            let mut b = [0usize; 3];
            for a in 0..3 {
                let f = ((l[a] - lo[a]) / span[a] * n[a] as f64).floor();
                b[a] = (f.max(0.0) as usize).min(n[a] - 1);
            }
            (b[0] * n[1] + b[1]) * n[2] + b[2]
        };
        let mut bin: Vec<u32> = vec![0; n_atoms];
        par.fill(&mut bin, |i, out| *out = bin_of(&frac(i)) as u32);
        let mut start = vec![0u32; nbins + 1];
        for &b in &bin {
            start[b as usize + 1] += 1;
        }
        for k in 0..nbins {
            start[k + 1] += start[k];
        }
        let mut cursor: Vec<u32> = start[..nbins].to_vec();
        let mut idx = vec![0u32; n_atoms];
        for (i, &b) in bin.iter().enumerate() {
            let b = b as usize;
            let s = cursor[b] as usize;
            cursor[b] += 1;
            idx[s] = i as u32;
        }
        drop(bin);
        // Store the wrapped Cartesian position, so displacements only need the
        // explicit image shift of the bin they were found in. Filled per slot
        // in parallel, after the serial scatter has fixed the slot order.
        let mut pos = vec![[0.0f64; 3]; n_atoms];
        par.fill(&mut pos, |s, out| {
            *out = lamda_to_x(&h, &origin, &frac(idx[s] as usize));
        });

        let ortho = (0..3).all(|r| (0..3).all(|c| r == c || h.m[r][c] == 0.0));
        Ok(Self {
            h,
            inv_t,
            origin,
            pbc,
            n,
            lo,
            span,
            gap,
            ortho,
            start,
            pos,
            idx,
        })
    }

    /// Number of atoms indexed.
    pub fn len(&self) -> usize {
        self.idx.len()
    }

    /// Whether the index holds no atoms.
    pub fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }

    /// The original atom index of each slot, in bin order. Running queries in
    /// this order keeps consecutive queries on the same or adjacent bins, so
    /// the bins they read are already in cache; file order rarely does.
    pub fn order(&self) -> &[u32] {
        &self.idx
    }

    /// Bins per axis (diagnostics and tests).
    pub fn dims(&self) -> [usize; 3] {
        self.n
    }

    /// The `k` nearest neighbours of atom `i` at position `p`, excluding every
    /// image of `i` itself, closest first, ties broken by (atom, image).
    ///
    /// Fewer than `k` are returned only when the frame has fewer candidates
    /// (open boundaries with fewer than `k + 1` atoms).
    pub fn knn(&self, i: usize, p: Coord3, k: usize, scratch: &mut KnnScratch, out: &mut Vec<Nbr>) {
        self.query(p, Some(i as u32), k, scratch, out);
    }

    /// The `k` nearest atoms to an arbitrary point.
    pub fn knn_point(&self, p: Coord3, k: usize, scratch: &mut KnnScratch, out: &mut Vec<Nbr>) {
        self.query(p, None, k, scratch, out);
    }

    /// Every atom image within `r` of an arbitrary point, unordered. Unlike
    /// [`Self::knn_point`] there is no cap on the count; `r` must be below
    /// half the shortest periodic length for each atom to appear once.
    pub fn within_point(&self, p: Coord3, r: f64, out: &mut Vec<Nbr>) {
        out.clear();
        if self.idx.is_empty() {
            return;
        }
        let r2 = r * r;
        let d = [
            p[0] - self.origin[0],
            p[1] - self.origin[1],
            p[2] - self.origin[2],
        ];
        let mut l = self.inv_t.mul_v(d);
        wrap_periodic(&mut l, self.pbc);
        let q = lamda_to_x(&self.h, &self.origin, &l);
        let mut home = [0i64; 3];
        let mut frac = [0.0f64; 3];
        for a in 0..3 {
            let f = (l[a] - self.lo[a]) / self.span[a] * self.n[a] as f64;
            let b = f.floor().clamp(0.0, (self.n[a] - 1) as f64);
            home[a] = b as i64;
            frac[a] = f - b;
        }
        let mut shell: i64 = 0;
        loop {
            self.visit_shell(shell, &home, |b, shift, _| {
                let t = self.shift_vector(shift);
                for slot in self.start[b] as usize..self.start[b + 1] as usize {
                    let pj = self.pos[slot];
                    let dr = [
                        pj[0] + t[0] - q[0],
                        pj[1] + t[1] - q[1],
                        pj[2] + t[2] - q[2],
                    ];
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 < r2 {
                        out.push((self.idx[slot], dr, d2));
                    }
                }
            });
            // Nearest distance to anything outside the visited block.
            let mut bound = f64::INFINITY;
            let mut more = false;
            for a in 0..3 {
                let s = shell as f64;
                if home[a] - shell > 0 || self.pbc[a] {
                    bound = bound.min((frac[a] + s) * self.gap[a]);
                    more = true;
                }
                if home[a] + shell < self.n[a] as i64 - 1 || self.pbc[a] {
                    bound = bound.min((1.0 - frac[a] + s) * self.gap[a]);
                    more = true;
                }
            }
            if !more || bound >= r {
                break;
            }
            shell += 1;
        }
    }

    fn query(
        &self,
        p: Coord3,
        skip: Option<u32>,
        k: usize,
        s: &mut KnnScratch,
        out: &mut Vec<Nbr>,
    ) {
        out.clear();
        let k = k.min(MAX_K);
        if k == 0 || self.idx.is_empty() {
            return;
        }
        let KnnScratch { best, bins } = s;
        best.clear();

        // Query point: fractional coordinates, wrapped on periodic axes, and its
        // Cartesian position in that same wrapped frame.
        let d = [
            p[0] - self.origin[0],
            p[1] - self.origin[1],
            p[2] - self.origin[2],
        ];
        let mut l = self.inv_t.mul_v(d);
        wrap_periodic(&mut l, self.pbc);
        let q = lamda_to_x(&self.h, &self.origin, &l);

        // Home bin and the query's offset inside it, in bin units.
        let mut home = [0i64; 3];
        let mut frac = [0.0f64; 3];
        for a in 0..3 {
            let f = (l[a] - self.lo[a]) / self.span[a] * self.n[a] as f64;
            let b = f.floor().clamp(0.0, (self.n[a] - 1) as f64);
            home[a] = b as i64;
            // May fall outside [0, 1) for a point query outside an open axis's
            // extent; the bound below then stays conservative.
            frac[a] = f - b;
        }

        let mut shell: i64 = 0;
        loop {
            // Scan the shell's bins nearest first, and stop at the first whose
            // lower bound cannot beat the k-th candidate: the rest are farther
            // still. The result is a top-k under a total order, so the visiting
            // order cannot change it; this only skips bins that could not
            // contribute (in a 3x3x3 block, most corner and many edge bins).
            bins.clear();
            self.visit_shell(shell, &home, |b, shift, off| {
                bins.push((self.bin_bound2(off, &frac), b, shift));
            });
            bins.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            for &(lb2, b, shift) in bins.iter() {
                // The bound is computed in floating point, like the shell
                // bound below; the relative margin keeps a rounding error from
                // ever skipping a bin holding a true candidate.
                if best.len() == k && lb2 * (1.0 - 1e-9) > best[k - 1].0 {
                    break;
                }
                let t = self.shift_vector(shift);
                let lo_s = self.start[b] as usize;
                let hi_s = self.start[b + 1] as usize;
                for slot in lo_s..hi_s {
                    let j = self.idx[slot];
                    if Some(j) == skip {
                        continue;
                    }
                    let pj = self.pos[slot];
                    let dr = [
                        pj[0] + t[0] - q[0],
                        pj[1] + t[1] - q[1],
                        pj[2] + t[2] - q[2],
                    ];
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    insert(best, k, (d2, j, shift, dr));
                }
            }

            // Lower bound on the distance to anything outside this block.
            let mut bound = f64::INFINITY;
            let mut more = false;
            for a in 0..3 {
                let s = shell as f64;
                let below = home[a] - shell > 0 || self.pbc[a];
                let above = home[a] + shell < self.n[a] as i64 - 1 || self.pbc[a];
                if below {
                    bound = bound.min((frac[a] + s) * self.gap[a]);
                    more = true;
                }
                if above {
                    bound = bound.min((1.0 - frac[a] + s) * self.gap[a]);
                    more = true;
                }
            }
            let done = best.len() == k && best[k - 1].0 <= bound.max(0.0).powi(2);
            if done || !more {
                break;
            }
            // Every bin has been visited once by now; if that found nothing, the
            // query atom is alone (all its images are excluded) and no shell
            // ever will. Otherwise periodic images always fill `k` eventually;
            // stop at an absurd radius anyway.
            let all_bins = shell as usize >= *self.n.iter().max().unwrap();
            if (all_bins && best.is_empty()) || bound > 1e7 {
                break;
            }
            shell += 1;
        }

        out.extend(best.iter().map(|&(d2, j, _, dr)| (j, dr, d2)));
    }

    /// Squared lower bound on the distance from a query at `frac` (its offset
    /// inside the home bin, in bin units) to any point of the bin at offset
    /// `off`. Per axis it is the distance to the near face of that bin's slab;
    /// the slabs' normals are orthogonal only for an orthogonal cell, so a
    /// triclinic cell takes the largest single axis instead of the sum.
    #[inline]
    fn bin_bound2(&self, off: [i64; 3], frac: &[f64; 3]) -> f64 {
        let mut per = [0.0f64; 3];
        for a in 0..3 {
            let o = off[a] as f64;
            let d = match off[a].signum() {
                1 => o - frac[a],
                -1 => frac[a] - o - 1.0,
                _ => 0.0,
            };
            per[a] = d.max(0.0) * self.gap[a];
        }
        if self.ortho {
            per[0] * per[0] + per[1] * per[1] + per[2] * per[2]
        } else {
            let m = per[0].max(per[1]).max(per[2]);
            m * m
        }
    }

    /// Visit every bin at Chebyshev distance exactly `s` from `home`, as
    /// `(linear bin, image shift, offset from home)`. Open axes skip
    /// out-of-range offsets; periodic axes turn them into image shifts.
    fn visit_shell<F: FnMut(usize, [i32; 3], [i64; 3])>(&self, s: i64, home: &[i64; 3], mut f: F) {
        let rng = |a: usize| -> (i64, i64) {
            if self.pbc[a] {
                (-s, s)
            } else {
                ((-s).max(-home[a]), s.min(self.n[a] as i64 - 1 - home[a]))
            }
        };
        let (x0, x1) = rng(0);
        let (y0, y1) = rng(1);
        let (z0, z1) = rng(2);
        for ox in x0..=x1 {
            for oy in y0..=y1 {
                let on_face = ox.abs() == s || oy.abs() == s;
                let mut oz = z0;
                while oz <= z1 {
                    if on_face || oz.abs() == s {
                        let raw = [home[0] + ox, home[1] + oy, home[2] + oz];
                        let mut w = [0usize; 3];
                        let mut shift = [0i32; 3];
                        for a in 0..3 {
                            let na = self.n[a] as i64;
                            // In range is the common case; skip the divisions.
                            if (0..na).contains(&raw[a]) {
                                w[a] = raw[a] as usize;
                                continue;
                            }
                            let r = raw[a].rem_euclid(na);
                            w[a] = r as usize;
                            shift[a] = ((raw[a] - r) / na) as i32;
                        }
                        f(
                            (w[0] * self.n[1] + w[1]) * self.n[2] + w[2],
                            shift,
                            [ox, oy, oz],
                        );
                        oz += 1;
                    } else {
                        // Interior of the face-less column: jump to the far side.
                        oz = s.min(z1).max(oz + 1);
                    }
                }
            }
        }
    }

    #[inline]
    fn shift_vector(&self, s: [i32; 3]) -> Coord3 {
        let m = self.h.m;
        let (x, y, z) = (s[0] as f64, s[1] as f64, s[2] as f64);
        [
            m[0][0] * x + m[1][0] * y + m[2][0] * z,
            m[0][1] * x + m[1][1] * y + m[2][1] * z,
            m[0][2] * x + m[1][2] * y + m[2][2] * z,
        ]
    }
}

/// Wrap fractional coordinates into `[0, 1)` on the periodic axes.
#[inline]
fn wrap_periodic(l: &mut Coord3, pbc: [bool; 3]) {
    for (v, &p) in l.iter_mut().zip(&pbc) {
        if p {
            *v -= v.floor();
            // `x - floor(x)` can round to exactly 1.0 for tiny negatives.
            if *v >= 1.0 {
                *v = 0.0;
            }
        }
    }
}

#[inline]
fn lamda_to_x(h: &Mat3, origin: &Coord3, l: &Coord3) -> Coord3 {
    let m = h.m;
    [
        m[0][0] * l[0] + m[1][0] * l[1] + m[2][0] * l[2] + origin[0],
        m[0][1] * l[0] + m[1][1] * l[1] + m[2][1] * l[2] + origin[1],
        m[0][2] * l[0] + m[1][2] * l[1] + m[2][2] * l[2] + origin[2],
    ]
}

/// Insert into the sorted best-`k` list, keyed by (d², atom, image).
#[inline]
fn insert(best: &mut Vec<(f64, u32, [i32; 3], Coord3)>, k: usize, c: (f64, u32, [i32; 3], Coord3)) {
    let less = |a: &(f64, u32, [i32; 3], Coord3), b: &(f64, u32, [i32; 3], Coord3)| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
            .is_lt()
    };
    if best.len() == k {
        // Most candidates are strictly farther than the k-th: one float
        // compare rejects them (`>` implies `total_cmp` Greater for any
        // non-NaN pair; a NaN falls through to the full order).
        if c.0 > best[k - 1].0 || !less(&c, &best[k - 1]) {
            return;
        }
    }
    let mut at = best.len();
    while at > 0 && less(&c, &best[at - 1]) {
        at -= 1;
    }
    if best.len() == k {
        best.pop();
    }
    best.insert(at, c);
}

/// Volume of the coarse-grid cells that hold at least one atom.
///
/// Each chunk of atoms marks its own bitset of coarse cells; the union is
/// order-independent, so the parallel count is exact.
fn occupied_volume(
    n: usize,
    frac: &(dyn Fn(usize) -> Coord3 + Sync),
    lo: &[f64; 3],
    span: &[f64; 3],
    extent: [f64; 3],
    par: &Par,
) -> f64 {
    let total = extent[0] * extent[1] * extent[2];
    if n == 0 || !(total.is_finite() && total > 0.0) {
        return total.max(1.0);
    }
    let c = COARSE_PER_AXIS;
    let words = (c * c * c + 63) / 64;
    let marks = par.map_chunks(n, |a, b| {
        let mut seen = vec![0u64; words];
        for i in a..b {
            let l = frac(i);
            let mut b = [0usize; 3];
            for a in 0..3 {
                let f = ((l[a] - lo[a]) / span[a] * c as f64).floor();
                b[a] = (f.max(0.0) as usize).min(c - 1);
            }
            let cell = (b[0] * c + b[1]) * c + b[2];
            seen[cell / 64] |= 1 << (cell % 64);
        }
        seen
    });
    let filled: u32 = (0..words)
        .map(|w| marks.iter().fold(0u64, |acc, m| acc | m[w]).count_ones())
        .sum();
    total * filled as f64 / (c * c * c) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*, so the tests need no `rand`.
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Brute force over explicit images within `reach` cells per periodic axis.
    fn brute(cell: &UnitCell, pos: &[Coord3], i: usize, k: usize, reach: i32) -> Vec<f64> {
        let p = pos[i];
        let mut d = Vec::new();
        let r = |a: usize| if cell.pbc[a] { reach } else { 0 };
        for (j, &pj) in pos.iter().enumerate() {
            if j == i {
                continue;
            }
            for sx in -r(0)..=r(0) {
                for sy in -r(1)..=r(1) {
                    for sz in -r(2)..=r(2) {
                        let t = cell.shift_vector([sx as i64, sy as i64, sz as i64]);
                        let dr = [
                            pj[0] + t[0] - p[0],
                            pj[1] + t[1] - p[1],
                            pj[2] + t[2] - p[2],
                        ];
                        d.push(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
                    }
                }
            }
        }
        d.sort_by(f64::total_cmp);
        d.truncate(k);
        d
    }

    fn check(cell: &UnitCell, pos: &[Coord3], k: usize, reach: i32) {
        let idx = KnnIndex::build(cell, pos, &Par::new(1)).unwrap();
        let mut s = KnnScratch::default();
        let mut out = Vec::new();
        for i in 0..pos.len() {
            idx.knn(i, pos[i], k, &mut s, &mut out);
            let want = brute(cell, pos, i, k, reach);
            assert_eq!(out.len(), want.len(), "atom {i}: count");
            for (t, (&(j, dr, d2), w)) in out.iter().zip(&want).enumerate() {
                assert!(
                    (d2 - w).abs() < 1e-9 * w.max(1.0),
                    "atom {i} rank {t}: {d2} vs {w}"
                );
                assert_ne!(j as usize, i, "self returned");
                let dd = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                assert!((dd - d2).abs() < 1e-9 * d2.max(1.0));
            }
            assert!(out.windows(2).all(|w| w[0].2 <= w[1].2));
        }
    }

    fn random(n: usize, cell: &UnitCell, seed: u64) -> Vec<Coord3> {
        let mut r = Rng(seed);
        (0..n)
            .map(|_| cell.lamda2x([r.f(), r.f(), r.f()]))
            .collect()
    }

    #[test]
    fn orthogonal_periodic_matches_brute_force() {
        let cell = UnitCell::orthorhombic(12.0, 10.0, 14.0);
        let pos = random(300, &cell, 1);
        for k in [1, 6, 12, 18] {
            check(&cell, &pos, k, 1);
        }
    }

    #[test]
    fn triclinic_with_origin_matches_brute_force() {
        let mut cell = UnitCell::from_box_bounds(1.0, 13.0, -3.0, -2.0, 9.0, 2.5, 0.5, 10.5, -1.5);
        cell.origin = [1.0, -2.0, 0.5];
        let pos = random(250, &cell, 2);
        for k in [1, 14, 18] {
            check(&cell, &pos, k, 2);
        }
    }

    #[test]
    fn slab_with_vacuum_matches_brute_force() {
        // Atoms fill a quarter of z; the rest of the periodic box is vacuum.
        let cell = UnitCell::orthorhombic(10.0, 10.0, 80.0);
        let mut r = Rng(3);
        let pos: Vec<Coord3> = (0..300)
            .map(|_| [10.0 * r.f(), 10.0 * r.f(), 20.0 * r.f() + 30.0])
            .collect();
        check(&cell, &pos, 18, 1);
        let open = cell.with_pbc([true, true, false]);
        check(&open, &pos, 18, 1);
    }

    #[test]
    fn a_cell_smaller_than_the_neighbour_shell_returns_images() {
        // Two atoms in a 3 Å periodic cube: 18 neighbours must come from images.
        let cell = UnitCell::orthorhombic(3.0, 3.0, 3.0);
        let pos = vec![[0.1, 0.2, 0.3], [1.6, 1.4, 1.5]];
        check(&cell, &pos, 18, 3);
    }

    #[test]
    fn open_boundaries_with_too_few_atoms_return_what_exists() {
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0).with_pbc([false; 3]);
        let pos = random(5, &cell, 4);
        let idx = KnnIndex::build(&cell, &pos, &Par::new(1)).unwrap();
        let mut out = Vec::new();
        idx.knn(0, pos[0], 12, &mut KnnScratch::default(), &mut out);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn a_lone_atom_in_a_periodic_box_returns_nothing_and_stops() {
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = vec![[1.0, 2.0, 3.0]];
        let idx = KnnIndex::build(&cell, &pos, &Par::new(1)).unwrap();
        let mut out = Vec::new();
        idx.knn(0, pos[0], 12, &mut KnnScratch::default(), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn parallel_build_is_identical() {
        let cell = UnitCell::orthorhombic(20.0, 20.0, 20.0);
        let pos = random(5000, &cell, 5);
        let a = KnnIndex::build(&cell, &pos, &Par::new(1)).unwrap();
        let b = KnnIndex::build(&cell, &pos, &Par::new(4)).unwrap();
        assert_eq!(a.idx, b.idx);
        assert_eq!(a.start, b.start);
    }
}
