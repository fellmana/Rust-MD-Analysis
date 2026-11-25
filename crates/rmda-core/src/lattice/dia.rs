//! Diamond cubic lattice.

use crate::cell::Mat3;
use crate::lattice::{Lattice, PrimitiveCell, Shell, CUBIC_DIAMOND};

#[derive(Debug, Clone, Copy)]
pub struct Dia {
    pub a: f64,
}

impl Default for Dia {
    fn default() -> Self {
        Dia { a: 1.0 }
    }
}

impl Lattice for Dia {
    fn name(&self) -> &str {
        "dia"
    }

    fn primitive(&self) -> PrimitiveCell {
        let a = self.a;
        let h = Mat3::from_rows([a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]);
        let basis = vec![
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.0],
            [0.75, 0.75, 0.25],
            [0.5, 0.0, 0.5],
            [0.75, 0.25, 0.75],
            [0.0, 0.5, 0.5],
            [0.25, 0.75, 0.75],
        ];
        PrimitiveCell::new(h, basis)
    }

    fn neighbour_shells(&self) -> &[Shell] {
        const SHELLS: [Shell; 3] = [
            Shell::new(4, 1.0),
            Shell::new(12, 1.632_993_161_855_2),
            Shell::new(12, 1.914_854_215_513_7),
        ];
        &SHELLS
    }
}

pub const CODE: u8 = CUBIC_DIAMOND;
