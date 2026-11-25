//! FCC lattice.

use crate::cell::Mat3;
use crate::lattice::{Lattice, PrimitiveCell, Shell, FCC};

/// FCC with conventional cubic cell of side `a`.
#[derive(Debug, Clone, Copy)]
pub struct Fcc {
    pub a: f64,
}

impl Default for Fcc {
    fn default() -> Self {
        Fcc { a: 1.0 }
    }
}

impl Lattice for Fcc {
    fn name(&self) -> &str {
        "fcc"
    }

    fn primitive(&self) -> PrimitiveCell {
        let a = self.a;
        // Conventional cubic cell as rows; basis = 4 atoms.
        let h = Mat3::from_rows([a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]);
        let basis = vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];
        PrimitiveCell::new(h, basis)
    }

    // The shell radii are the upstream C++ literals verbatim. They are decimal
    // renderings of surds, but `std::f64::consts` holds different doubles, and
    // swapping them in would shift the shell radii by an ULP.
    #[allow(clippy::approx_constant)]
    fn neighbour_shells(&self) -> &[Shell] {
        // Radii as multiples of the nearest-neighbour distance a/√2: the
        // shells at a/√2, a and a·√(3/2) are 1, √2 and √3 of it.
        const SQRT2: f64 = 1.414_213_562_373_095_1;
        const SQRT3: f64 = 1.732_050_807_568_877_2;
        const SHELLS: [Shell; 3] = [
            Shell::new(12, 1.0),
            Shell::new(6, SQRT2),
            Shell::new(24, SQRT3),
        ];
        &SHELLS
    }
}

/// The code associated with this lattice in grain/PTM contexts.
pub const CODE: u8 = FCC;
