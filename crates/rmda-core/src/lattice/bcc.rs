//! BCC lattice.

use crate::cell::Mat3;
use crate::lattice::{Lattice, PrimitiveCell, Shell, BCC};

/// BCC with conventional cubic cell of side `a`.
#[derive(Debug, Clone, Copy)]
pub struct Bcc {
    pub a: f64,
}

impl Default for Bcc {
    fn default() -> Self {
        Bcc { a: 1.0 }
    }
}

impl Lattice for Bcc {
    fn name(&self) -> &str {
        "bcc"
    }

    fn primitive(&self) -> PrimitiveCell {
        let a = self.a;
        let h = Mat3::from_rows([a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]);
        let basis = vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
        PrimitiveCell::new(h, basis)
    }

    fn neighbour_shells(&self) -> &[Shell] {
        // nearest-neighbour distance a√3/2; shells as multiples of it:
        // 1, 2/√3, 2√(2/3)
        const TWO_OVER_SQRT3: f64 = 1.154_700_538_379_251_5;
        const TWO_SQRT_TWO_OVER_SQRT3: f64 = 1.632_993_161_855_452;
        const SHELLS: [Shell; 3] = [
            Shell::new(8, 1.0),
            Shell::new(6, TWO_OVER_SQRT3),
            Shell::new(12, TWO_SQRT_TWO_OVER_SQRT3),
        ];
        &SHELLS
    }
}

pub const CODE: u8 = BCC;
