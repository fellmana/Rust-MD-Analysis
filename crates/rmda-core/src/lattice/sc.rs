//! Simple cubic lattice.

use crate::cell::Mat3;
use crate::lattice::{Lattice, PrimitiveCell, Shell, SC};

#[derive(Debug, Clone, Copy)]
pub struct Sc {
    pub a: f64,
}

impl Default for Sc {
    fn default() -> Self {
        Sc { a: 1.0 }
    }
}

impl Lattice for Sc {
    fn name(&self) -> &str {
        "sc"
    }

    fn primitive(&self) -> PrimitiveCell {
        let a = self.a;
        let h = Mat3::from_rows([a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]);
        PrimitiveCell::new(h, vec![[0.0, 0.0, 0.0]])
    }

    // The shell radii are the upstream C++ literals verbatim. They are decimal
    // renderings of surds, but `std::f64::consts` holds different doubles, and
    // swapping them in would shift the shell radii by an ULP.
    #[allow(clippy::approx_constant)]
    fn neighbour_shells(&self) -> &[Shell] {
        const SQRT2: f64 = 1.414_213_562_373_095_1;
        const SQRT3: f64 = 1.732_050_807_568_877_2;
        const SHELLS: [Shell; 3] = [
            Shell::new(6, 1.0),
            Shell::new(12, SQRT2),
            Shell::new(8, SQRT3),
        ];
        &SHELLS
    }
}

pub const CODE: u8 = SC;
