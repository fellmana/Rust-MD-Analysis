//! Building the point set a PTM match consumes (upstream's `get_neighbours`).
//!
//! # Why the library delegates this
//!
//! `ptm_index` obtains neighbours through a `get_neighbours` *function pointer*
//! supplied by the host program, and the same is true of
//! `calculate_two_shell_neighbour_ordering`. So the ordering that decides a
//! match is defined by the host. This file builds the distance-ordered list;
//! OVITO reorders it by Voronoi solid angle first, which
//! `NeighbourOrder::Voronoi` reproduces (see [`super::voronoi`]).
//!
//! # The convention
//!
//! Index 0 is the central atom and indices `1..=num_nbrs` are its neighbours in
//! ascending distance. This is forced by the rest of the port:
//!
//! * [`normalize_vertices_n`](super::normalize::normalize_vertices_n) centres
//!   *all* `n` points but measures the mean neighbour length over `1..n`, so
//!   index 0 is the central atom and the rest are its neighbours.
//! * Each reference reads a fixed prefix: SC `0..7`, FCC/HCP/ICO `0..13`, BCC
//!   `0..15` ([`num_nbrs`](super::refdata::RefData::num_nbrs) plus one).
//!
//! The prefix, not the whole set, is what matters. A neighbour further out than
//! the reference reads is dropped by that truncation, so the match is
//! insensitive to how much of a far shell the list happens to contain - and
//! that in turn is why no element cutoff is needed here. [`gather`] only has to
//! find at least `num_nbrs` neighbours; there is no radius beyond which a
//! neighbour would wrongly displace a nearer one.
//!
//! Ties in distance are broken by atom index, then by periodic shift, by
//! [`nearest_k_excluding_self`]. That is a deterministic but *arbitrary* choice within a
//! distance group, and it is not the choice OVITO or LAMMPS makes. It is
//! nevertheless safe, because the reference graphs are compared up to their
//! automorphism group: reordering a tied shell cannot change the verdict.
//!
//! This is not a safe assumption to leave implicit, because it is load-bearing
//! for *every* FCC atom rather than some corner case - an ideal FCC/HCP atom
//! has 12 neighbours at exactly the same distance, so the order within that
//! shell is a free choice on every single atom. It is pinned by
//! `matcher::tests::tied_neighbour_order_does_not_change_the_verdict`, which
//! feeds each ideal template back through the matcher under hundreds of
//! random shell permutations and requires the same structure and a sub-1e-6
//! fit every time.
//!
//! What is *not* safe is feeding a genuinely different environment, which is
//! why the ordering is pinned to distance rather than left to whatever order
//! the cell traversal happens to produce.

use crate::error::Result;
use crate::neighbor::cell_list::CellList;
use crate::neighbor::nnearest::nearest_k_excluding_self;

use super::MAX_POINTS;

/// Points gathered for one central atom (upstream `PTM_MAX_INPUT_POINTS`).
///
/// Wider than [`MAX_POINTS`] because upstream fetches 19 and then reads at most
/// [`MAX_POINTS`]; only the widest reference's prefix is ever used.
pub const MAX_INPUT_POINTS: usize = 19;

/// One central atom and its neighbours, ready for a match.
#[derive(Debug, Clone)]
pub struct Neighbourhood {
    /// `points[0]` is the central atom at the origin; `points[1..len]` are
    /// neighbours in ascending distance.
    pub points: [[f64; 3]; MAX_INPUT_POINTS],
    /// Number of populated entries, at least 1 when the gather succeeded.
    pub len: usize,
}

impl Default for Neighbourhood {
    fn default() -> Self {
        Self {
            points: [[0.0; 3]; MAX_INPUT_POINTS],
            len: 0,
        }
    }
}

impl Neighbourhood {
    /// Collect the central atom and its [`MAX_INPUT_POINTS`] - 1 nearest
    /// neighbours.
    ///
    /// Positions are stored *relative* to the central atom, which is
    /// translation-invariant downstream: `normalize_vertices_n` subtracts the
    /// mean of all gathered points, so feeding absolute coordinates would give
    /// the same match while rounding the geometry against the magnitude of the
    /// frame's origin.
    pub fn gather(cell: &CellList, i: usize) -> Result<Self> {
        let mut entries: Vec<(u32, [f64; 3], f64)> = Vec::with_capacity(MAX_INPUT_POINTS);
        nearest_k_excluding_self(cell, i, MAX_INPUT_POINTS - 1, &mut entries)?;

        let mut points = [[0.0f64; 3]; MAX_INPUT_POINTS];
        // Built from a zeroed array so `points[0]` is structurally the origin -
        // the central atom, by construction. The neighbours go in slots 1.., so
        // the two can never collide.
        for (slot, (_, dr, _)) in points[1..].iter_mut().zip(entries.iter()) {
            *slot = *dr;
        }
        Ok(Self {
            points,
            len: 1 + entries.len(),
        })
    }

    /// Fill from neighbours already sorted by distance (closest first), e.g.
    /// from [`crate::neighbor::KnnIndex::knn`]. At most
    /// [`MAX_INPUT_POINTS`] - 1 are used.
    pub fn set_from(&mut self, nbrs: &[(u32, [f64; 3], f64)]) {
        let m = nbrs.len().min(MAX_INPUT_POINTS - 1);
        self.points[0] = [0.0; 3];
        for (slot, (_, dr, _)) in self.points[1..=m].iter_mut().zip(nbrs) {
            *slot = *dr;
        }
        self.len = 1 + m;
    }

    /// The gathered points in the fixed-width form the matcher expects.
    ///
    /// The matcher's arrays are [`MAX_POINTS`] wide, so a neighbourhood holding
    /// more points than that is truncated here rather than at each call site.
    pub fn as_match_points(&self) -> [[f64; 3]; MAX_POINTS] {
        let mut out = [[0.0f64; 3]; MAX_POINTS];
        let n = self.len.min(MAX_POINTS);
        out[..n].copy_from_slice(&self.points[..n]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::UnitCell;

    fn cell_of(a: f64) -> UnitCell {
        UnitCell::orthorhombic(a, a, a)
    }

    fn dist(p: &[f64; 3]) -> f64 {
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
    }

    /// A `n^3` supercell of a triclinic primitive cell with the given basis.
    fn supercell(
        avec: [f64; 3],
        bvec: [f64; 3],
        cvec: [f64; 3],
        basis: &[[f64; 3]],
        n: usize,
    ) -> (UnitCell, Vec<[f64; 3]>) {
        let rep = |v: [f64; 3], k: usize| [v[0] * k as f64, v[1] * k as f64, v[2] * k as f64];
        let (cell, _) = UnitCell::from_general_triclinic_rot(
            rep(avec, n),
            rep(bvec, n),
            rep(cvec, n),
            [0.0; 3],
        )
        .unwrap();
        let h = cell.h_matrix();
        // `Mat3` stores cell vectors as *rows*, so this is `f0*row0 + f1*row1 +
        // f2*row2` = `f * H`. Reading columns instead gives `H * f`, which
        // shears the lattice and silently yields a plausible-looking but wrong
        // neighbour shell structure - see the same note in `cna::tests`.
        let to_cart = |f: [f64; 3]| {
            [
                f[0] * h.m[0][0] + f[1] * h.m[1][0] + f[2] * h.m[2][0],
                f[0] * h.m[0][1] + f[1] * h.m[1][1] + f[2] * h.m[2][1],
                f[0] * h.m[0][2] + f[1] * h.m[1][2] + f[2] * h.m[2][2],
            ]
        };
        let mut pos = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for b in basis {
                        pos.push(to_cart([
                            (i as f64 + b[0]) / n as f64,
                            (j as f64 + b[1]) / n as f64,
                            (k as f64 + b[2]) / n as f64,
                        ]));
                    }
                }
            }
        }
        (cell, pos)
    }

    /// Simple cubic: the 6 nearest sit at `a` and the 7th is a face diagonal.
    #[test]
    fn gathers_a_simple_cubic_environment() {
        // The cell must be the supercell, not one lattice constant: a cell of
        // size `a` around a 3x3x3 grid makes positions 0, 1 and 2 the *same*
        // point modulo the cell, so every neighbour is really a self-image.
        let a = 1.0;
        let (cell, pos) = supercell(
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a],
            &[[0.0, 0.0, 0.0]],
            3,
        );
        let cl = CellList::build(&cell, &pos, 2.0).unwrap();
        let nb = Neighbourhood::gather(&cl, 0).unwrap();
        assert!(nb.len >= 7, "got {} points", nb.len);
        assert_eq!(nb.points[0], [0.0, 0.0, 0.0], "central atom at the origin");
        for p in &nb.points[1..7] {
            assert!((dist(p) - a).abs() < 1e-12, "6th shell at {}", dist(p));
        }
        assert!(
            (dist(&nb.points[7]) - a * 2f64.sqrt()).abs() < 1e-12,
            "7th at {}",
            dist(&nb.points[7])
        );
    }

    /// Ideal HCP: 12 nearest neighbours at `a`, then a second shell at `c`.
    ///
    /// This is the close-packed environment PTM has to get right, and the one
    /// where a stray zero-distance entry would be most damaging: the references
    /// read exactly 13 points, so a single ghost displaces a real neighbour out
    /// of the prefix.
    #[test]
    fn an_hcp_environment_has_twelve_nearest_at_the_same_distance() {
        let a = 1.0;
        // Ideal ratio: c/a = sqrt(8/3). With a non-ideal c the 6 in-plane
        // neighbours still sit at `a` but the shell is no longer exactly
        // degenerate, so pin the exact value rather than a rounded one.
        let c = a * (8.0f64 / 3.0).sqrt();
        let (cell, pos) = supercell(
            [a, 0.0, 0.0],
            [-a / 2.0, a * 3f64.sqrt() / 2.0, 0.0],
            [0.0, 0.0, c],
            &[[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.5]],
            3,
        );
        let cl = CellList::build(&cell, &pos, 2.0).unwrap();
        let nb = Neighbourhood::gather(&cl, 0).unwrap();
        assert!(nb.len >= 13, "got {} points", nb.len);
        for (i, p) in nb.points[1..13].iter().enumerate() {
            assert!((dist(p) - a).abs() < 1e-9, "neighbour {i} at {}", dist(p));
        }
        assert!(
            dist(&nb.points[13]) > a + 1e-9,
            "12th shell at {}",
            dist(&nb.points[13])
        );
    }

    /// The atom's own periodic images must never be reported as neighbours.
    ///
    /// With a single atom in a small periodic cell every candidate *is* a
    /// self-image. The cell list already skips the query atom, and the correct
    /// answer is therefore "no neighbours" rather than a prefix of ghosts at
    /// distance 0.
    #[test]
    fn self_images_are_never_reported_as_neighbours() {
        let pos = vec![[0.0, 0.0, 0.0]];
        let cl = CellList::build(&cell_of(1.0), &pos, 2.0).unwrap();
        let nb = Neighbourhood::gather(&cl, 0).unwrap();
        assert_eq!(nb.len, 1, "only the central atom");
    }

    /// Neighbours must come back in ascending distance order. This is the
    /// property the whole design rests on: every reference reads a fixed-length
    /// prefix of this list, so an out-of-order entry means a structurally valid
    /// environment gets matched against the wrong reference.
    #[test]
    fn neighbours_are_in_ascending_distance_order() {
        // Deliberately not a symmetric lattice: a distorted supercell is what
        // actually exercises the sort rather than an accidental tie pattern.
        let (cell, pos) = supercell(
            [1.07, 0.0, 0.0],
            [0.0, 1.03, 0.0],
            [0.0, 0.0, 1.11],
            &[
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
            ],
            3,
        );
        let cl = CellList::build(&cell, &pos, 3.0).unwrap();
        // Check every atom, not just the first: the sort is per query.
        for i in 0..pos.len() {
            let nb = Neighbourhood::gather(&cl, i).unwrap();
            assert!(nb.len > 1, "atom {i} has no neighbours");
            let ds: Vec<f64> = nb.points[1..nb.len].iter().map(dist).collect();
            for w in ds.windows(2) {
                assert!(w[0] <= w[1] + 1e-12, "atom {i}: {} then {}", w[0], w[1]);
            }
        }
    }

    /// The fixed-width view must never be wider than the matcher's array, and
    /// must preserve the prefix exactly.
    #[test]
    fn match_points_are_the_prefix() {
        let mut nb = Neighbourhood::default();
        for i in 0..nb.points.len() {
            nb.points[i] = [i as f64, 0.0, 0.0];
        }
        nb.len = nb.points.len();
        let wide = nb.as_match_points();
        assert_eq!(wide.len(), MAX_POINTS);
        for (i, p) in wide.iter().enumerate() {
            assert_eq!(*p, [i as f64, 0.0, 0.0]);
        }
    }
}
