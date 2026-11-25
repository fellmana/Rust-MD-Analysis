//! Unit tags and conversion factors.
//!
//! Design §5: default I/O unit set is LAMMPS `metal` (Å, eV, ps, g/mol, K, bar).
//! Unit conversion is explicit CLI state, never silently assumed. All internal
//! analysis is length-only (`f64` coordinates, Å); velocities/forces pass through
//! as opaque properties.

use core::fmt;

/// Physical dimensions we track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    Length,
    Time,
    Energy,
    Mass,
    Temperature,
    Pressure,
}

/// A named unit set mirroring LAMMPS `units`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum System {
    /// Å, eV, ps, g/mol, K, bar — default.
    #[default]
    Metal,
    /// Å, kcal/mol, fs, g/mol, K, atm (LAMMPS `real`).
    Real,
    /// Å, kcal/mol, ps, g/mol, K, atm (LAMMPS `lj`-like; rarely used in data files).
    Lammps,
    /// kJ/mol, nm, ps, g/mol, K, bar (Gromacs). Not a LAMMPS native set.
    SiPico,
}

impl System {
    /// Convert `val` from this system's unit to the internal one, per
    /// dimension. Internal lengths are Å; everything else is carried opaquely.
    pub fn to_internal(&self, val: f64, dim: Dim) -> f64 {
        match dim {
            Dim::Length => match self {
                System::Metal | System::Real | System::Lammps => val, // Å in all three
                System::SiPico => val * 10.0,                         // nm -> Å
            },
            _ => val, // opaque; analysis is length-only
        }
    }

    /// Inverse of [`System::to_internal`].
    pub fn from_internal(&self, val: f64, dim: Dim) -> f64 {
        match dim {
            Dim::Length => match self {
                System::Metal | System::Real | System::Lammps => val,
                System::SiPico => val / 10.0,
            },
            _ => val,
        }
    }
}

/// A quantity expressed in an explicit unit system plus a physical dimension.
///
/// The design keeps unit state explicit at the CLI boundary; internal code
/// works in raw angstroms. This type is used by the I/O layer to tag values.
#[derive(Debug, Clone, Copy)]
pub struct Quantity {
    pub value: f64,
    pub system: System,
}

impl fmt::Display for System {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            System::Metal => write!(f, "metal"),
            System::Real => write!(f, "real"),
            System::Lammps => write!(f, "lammps"),
            System::SiPico => write!(f, "si-pico"),
        }
    }
}

/// The unit state carried by a read/write session. Explicit, never assumed.
#[derive(Debug, Clone, Copy, Default)]
pub struct Units {
    pub system: System,
    /// Length scale tag; for human-facing output.
    pub length_unit: LengthUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LengthUnit {
    #[default]
    Angstrom,
    Nanometer,
    Bohr,
}

impl LengthUnit {
    pub const fn symbol(&self) -> &'static str {
        match self {
            LengthUnit::Angstrom => "Å",
            LengthUnit::Nanometer => "nm",
            LengthUnit::Bohr => "bohr",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_is_length_default() {
        let u = Units::default();
        assert_eq!(u.system, System::Metal);
        assert_eq!(u.length_unit, LengthUnit::Angstrom);
    }

    #[test]
    fn nm_to_angstrom() {
        assert_eq!(System::SiPico.to_internal(1.0, Dim::Length), 10.0);
        assert_eq!(System::SiPico.from_internal(10.0, Dim::Length), 1.0);
        // metal is already Å
        assert_eq!(System::Metal.to_internal(2.5, Dim::Length), 2.5);
    }
}
