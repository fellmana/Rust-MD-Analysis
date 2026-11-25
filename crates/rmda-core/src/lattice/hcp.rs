//! HCP lattice (hexagonal).

use crate::cell::Mat3;
use crate::lattice::{Lattice, PrimitiveCell, Shell, HCP};

/// HCP with lattice parameter `a` and ideal c/a ratio preserved via `c`.
#[derive(Debug, Clone, Copy)]
pub struct Hcp {
    pub a: f64,
    pub c: f64,
}

impl Default for Hcp {
    fn default() -> Self {
        Hcp {
            a: 1.0,
            c: 1.632_993_161_855_2, // ideal c/a * a
        }
    }
}

impl Lattice for Hcp {
    fn name(&self) -> &str {
        "hcp"
    }

    fn primitive(&self) -> PrimitiveCell {
        let a = self.a;
        let c = self.c;
        // Orthohexagonal CONVENTIONAL cell, vectors as columns (diagonal here, so
        // rows and columns are the same matrix): v1 = (a,0,0), v2 = (0,a√3,0),
        // v3 = (0,0,c), four basis atoms. This is what `Fcc::primitive` also
        // does — "primitive" in this codebase means "the cell the site lattice
        // enumerates", and for FCC that is the conventional cube.
        //
        // Two things follow from the cell being axis aligned, and a 60-degree
        // two-atom cell gets both wrong:
        //
        // - A box of shape (n·a, n·a√3, n·c) is an *integer block* of cells, so
        //   `SiteLattice` generates exactly 4·n0·n1·n2 sites. Under a skewed
        //   cell the box maps to a sheared parallelepiped whose per-axis bounding
        //   box holds 1.67x the cells: on a perfect 6x4x4 block that is 640 sites
        //   for 384 atoms, i.e. 256 phantom vacancies with mean displacement
        //   0.0001 A — a perfect crystal reported as 256 defective.
        // - The stacking phase can be written down explicitly instead of being
        //   pinned by the choice of in-plane vectors. The basis below is the
        //   ABAB stack with B over the (0, a√3/3) triangle centre, whose parity
        //   classes are (0,0,0), (1,3,0), (0,2,1), (1,5,1) in half-cells. B at
        //   (a/2, a√3/6), the 60-degree-rotated hollow, is the other stacking
        //   phase: half the atoms of an HCP crystal stacked as below are
        //   off-lattice by a/√3 against it.
        let h = Mat3::from_rows([a, 0.0, 0.0], [0.0, a * 3.0_f64.sqrt(), 0.0], [0.0, 0.0, c]);
        let basis = vec![
            [0.0, 0.0, 0.0],       // A: (0, 0, 0)
            [0.5, 0.5, 0.0],       // A: (a/2, a√3/2, 0)
            [0.0, 1.0 / 3.0, 0.5], // B: (0, a√3/3, c/2)
            [0.5, 5.0 / 6.0, 0.5], // B: (a/2, 5a√3/6, c/2)
        ];
        PrimitiveCell::new(h, basis)
    }

    // The shell radii are the upstream C++ literals verbatim. They are decimal
    // renderings of surds, but `std::f64::consts` holds different doubles, and
    // swapping them in would shift the shell radii by an ULP.
    #[allow(clippy::approx_constant)]
    fn neighbour_shells(&self) -> &[Shell] {
        const SHELLS: [Shell; 4] = [
            Shell::new(12, 1.0),
            Shell::new(6, 1.414_213_56),
            Shell::new(2, 1.632_993_16),
            Shell::new(18, 1.732_050_80),
        ];
        &SHELLS
    }
}

pub const CODE: u8 = HCP;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::fit::nn_distance;
    use crate::lattice::Lattice;

    #[test]
    fn four_atom_basis_is_the_close_packed_stack() {
        // The stored cell has to be a real close-packed frame, because a sheared
        // one still yields plausible site *counts* while every site sits in the
        // wrong place. `nn_distance` is the same oracle the fit pipeline uses.
        let lat = Hcp::default();
        let (a, c) = (lat.a, lat.c);
        let la = a * 3.0_f64.sqrt();
        let pc = lat.primitive();
        assert_eq!(pc.basis.len(), 4, "hcp needs a 4-atom basis");

        let cart = pc.basis_cart();
        let want = [
            [0.0, 0.0, 0.0],
            [a * 0.5, la * 0.5, 0.0],
            [0.0, la / 3.0, c * 0.5],
            [a * 0.5, la * 5.0 / 6.0, c * 0.5],
        ];
        for (i, w) in want.iter().enumerate() {
            for (k, (got, exp)) in cart[i].iter().zip(w.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-12,
                    "basis[{i}] cart[{k}] = {got}, expected {exp}"
                );
            }
        }

        // The B layer sits over the (0, a√3/3) triangle centre of the A layer:
        // horizontal distance a/√3 to the three A atoms under it (the centroid
        // of (0,0), (±a/2, a√3/2)), out-of-plane distance c/2, so the whole
        // first shell is at `a` at the ideal ratio.
        let b = cart[2];
        for (k, a_atom) in [
            [0.0, 0.0, 0.0],
            [a * 0.5, la * 0.5, 0.0],
            [-a * 0.5, la * 0.5, 0.0],
        ]
        .iter()
        .enumerate()
        {
            let d = ((b[0] - a_atom[0]).powi(2) + (b[1] - a_atom[1]).powi(2)).sqrt();
            assert!(
                (d - a / 3.0_f64.sqrt()).abs() < 1e-12,
                "B-to-A[{k}] in plane = {d}, expected a/√3"
            );
        }

        let nn = nn_distance(&pc).expect("primitive must be invertible");
        assert!(
            (nn - lat.a).abs() < 1e-12,
            "hcp nearest-neighbour distance {nn}, expected a = {}",
            lat.a
        );

        // The declared shells, which are the upstream literals, have to describe
        // the cell that is actually stored: 12 at a, 6 at √2·a, 2 at c, 18 at √3·a.
        // The radius literals are truncated decimal renderings (1.632_993_16), so
        // they agree with `c` to 1e-8, not to the last bit.
        let shells = lat.neighbour_shells();
        assert_eq!((shells[0].count, shells[1].count), (12, 6));
        assert!((shells[0].radius - 1.0).abs() < 1e-9);
        assert!((shells[2].radius * nn - c).abs() < 1e-8, "third shell is c");
    }
}
