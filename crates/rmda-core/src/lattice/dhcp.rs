//! Double-hexagonal-close-packed (DHCP) lattice (design §11.1 registry
//! extension).
//!
//! DHCP is the ABAC stacking: four close-packed layers per cell, the fourth at
//! the *other* hollow site of the A layer. This is what distinguishes it from
//! HCP (ABAB) and FCC (ABC everywhere): the first shell is 12 at the
//! close-packing distance for all three, and they diverge only in the deeper
//! shells.
//!
//! Geometry: the same hexagonal cell as HCP, with the cell height spanning the
//! four layers, and the standard hollow offsets
//!
//! ```text
//! A (0,0)     B (1/3,1/3)     A (0,0)     C (2/3,2/3)
//! z = 0       z = c/4         z = c/2     z = 3c/4
//! ```
//!
//! With the ideal axial ratio (`c = 2·√(8/3)·a`, two stacked HCP cells) every
//! atom has 12 nearest neighbours at distance exactly `a`, so `a` is a genuine
//! nearest-neighbour distance and the first shell matches HCP's.
//!
//! # Row layout of the stored cell matrix
//!
//! `Mat3`/`SiteLattice` treat the cell matrix as an arbitrary invertible frame
//! (`mul_v` maps fractional to Cartesian, `solve` maps back); nothing in the
//! lattice, fit or WS machinery assumes rows are lattice vectors, so a lattice
//! is "physically faithful" exactly when its stored frame maps the fractional
//! grid to real crystal positions. For DHCP we store the three lattice vectors
//! as *columns*:
//!
//! ```text
//! row0 = (a, a/2, 0)      //  a1 = (a, 0, 0)        (column 0)
//! row1 = (0, a√3/2, 0)    //  a2 = (a/2, a√3/2, 0)  (column 1)
//! row2 = (0, 0, c)        //  a3 = (0, 0, c)        (column 2)
//! ```
//!
//! since `mul_v(f) = f_a·a1 + f_b·a2 + f_c·a3` then reproduces the true
//! close-packed positions — the B layer lands at `(a/2, a√3/6, c/4)`. Any
//! other hexagonal lattice must pass the same nearest-neighbour test: a
//! rows-as-vectors layout maps the grid through an affine shear (`hcp.rs`
//! avoids this with an axis-aligned orthohexagonal cell).

use crate::cell::Mat3;
use crate::lattice::{Lattice, PrimitiveCell, Shell};

/// DHCP with lattice parameter `a` and cell height `c` spanning four layers.
#[derive(Debug, Clone, Copy)]
pub struct Dhcp {
    pub a: f64,
    pub c: f64,
}

impl Default for Dhcp {
    fn default() -> Self {
        // Ideal axial ratio: four layers, each layer at the close-packed
        // spacing (`c/4 = √(2/3)·a`).
        Dhcp {
            a: 1.0,
            c: 2.0 * (8.0_f64 / 3.0_f64).sqrt(),
        }
    }
}

impl Lattice for Dhcp {
    fn name(&self) -> &str {
        "dhcp"
    }

    fn primitive(&self) -> PrimitiveCell {
        let a = self.a;
        let c = self.c;
        // Lattice vectors as columns of the stored matrix (see module docs).
        let h = Mat3::from_rows(
            [a, a * 0.5, 0.0],
            [0.0, a * 3.0_f64.sqrt() * 0.5, 0.0],
            [0.0, 0.0, c],
        );
        let basis = vec![
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.25],
            [0.0, 0.0, 0.5],
            [2.0 / 3.0, 2.0 / 3.0, 0.75],
        ];
        PrimitiveCell::new(h, basis)
    }

    fn neighbour_shells(&self) -> &[Shell] {
        // First shell only: 12 at the close-packing distance, as for HCP. The
        // deeper shells of the four-layer stack distinguish DHCP from FCC/HCP,
        // but no upstream oracle pins them, so none are asserted (the field is
        // unused by the analyses today).
        const SHELLS: [Shell; 1] = [Shell::new(12, 1.0)];
        &SHELLS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::fit::nn_distance;
    use crate::lattice::{Lattice, LatticeRegistry};

    #[test]
    fn four_layer_basis_and_close_packed_nearest_neighbour() {
        // The registry extension check: a 4-atom ABAC basis, and a
        // nearest-neighbour distance of `a` at the ideal axial ratio. With the
        // physical frame above the B layer lands at `(a/2, a√3/6, c/4)` and the
        // nearest-neighbour distance is
        // exactly `a` — checked with the same oracle the fit pipeline uses,
        // so it pins the stored cell, not a hand-rolled distance.
        let lat = Dhcp::default();
        let (a, c) = (lat.a, lat.c);
        let pc = lat.primitive();
        assert_eq!(pc.basis.len(), 4, "dhcp needs a 4-atom basis");

        let b = pc.basis_cart()[1];
        for (k, want) in [a * 0.5, a * 3.0_f64.sqrt() / 6.0, c / 4.0]
            .iter()
            .enumerate()
        {
            assert!(
                (b[k] - want).abs() < 1e-9,
                "B layer cart[{k}] = {}, expected {want}",
                b[k]
            );
        }

        let nn = nn_distance(&pc).expect("primitive must be invertible");
        assert!(
            (nn - lat.a).abs() < 1e-9,
            "dhcp nearest-neighbour distance {nn}, expected a = {}",
            lat.a
        );
    }

    #[test]
    fn dhcp_is_resolvable_from_the_registry() {
        let mut r = LatticeRegistry::new();
        r.register(Box::<Dhcp>::default());
        let got = r.get("dhcp").expect("dhcp must be registered");
        assert_eq!(got.name(), "dhcp");
    }
}
