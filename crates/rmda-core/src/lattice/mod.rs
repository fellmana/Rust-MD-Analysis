//! Lattice registry (design §11.1).
//!
//! Provides polymorphic lattice definitions (FCC/BCC/HCP/DHCP/SC/DIA)
//! so a polycrystalline pipeline can mix structures. Also hosts the shared
//! structure-type codes used by CNA/PTM/grains.

use crate::cell::{Coord3, Mat3, UnitCell};

pub mod bcc;
pub mod dhcp;
pub mod dia;
pub mod fcc;
pub mod fit;
pub mod hcp;
pub mod sc;
pub mod sites;

/// Structure type codes shared by all analyses (design §11.1).
pub const OTHER: u8 = 0;
pub const FCC: u8 = 1;
pub const HCP: u8 = 2;
pub const BCC: u8 = 3;
pub const ICO: u8 = 4;
pub const SC: u8 = 5;
pub const CUBIC_DIAMOND: u8 = 6;
pub const HEX_DIAMOND: u8 = 7;
pub const GRAPHENE: u8 = 8;

pub fn structure_name(code: u8) -> &'static str {
    match code {
        OTHER => "OTHER",
        FCC => "FCC",
        HCP => "HCP",
        BCC => "BCC",
        ICO => "ICO",
        SC => "SC",
        CUBIC_DIAMOND => "CUBIC_DIAMOND",
        HEX_DIAMOND => "HEX_DIAMOND",
        GRAPHENE => "GRAPHENE",
        _ => "UNKNOWN",
    }
}

/// Conventional cell edge vectors (columns) and a fractional-coordinate basis.
#[derive(Debug, Clone)]
pub struct PrimitiveCell {
    /// Lattice vectors as COLUMNS of `h`: fractional `f` maps to Cartesian as
    /// `h · f` (`basis_cart`, `sites`, `fit::nn_distance` and every rotation in
    /// `fit_region` all apply `h` that way), and so do `hcp.rs` and `dhcp.rs`:
    /// storing rows instead shears the whole site lattice by the transpose.
    /// [`Mat3::col_norm`] is the row-length counterpart for this
    /// layout; [`Mat3::row_norm`] is the one for the rows layout of `UnitCell`.
    pub h: Mat3,
    /// Basis atoms in FRACTIONAL coordinates.
    pub basis: Vec<Coord3>,
}

impl PrimitiveCell {
    pub fn new(h: Mat3, basis: Vec<Coord3>) -> Self {
        PrimitiveCell { h, basis }
    }

    /// Cartesian coordinates of the basis atoms (no origin shift).
    pub fn basis_cart(&self) -> Vec<Coord3> {
        self.basis.iter().map(|&b| self.h.mul_v(b)).collect()
    }
}

/// A neighbour shell: how many atoms at what distance (as a factor of the
/// nearest-neighbour distance).
#[derive(Debug, Clone, Copy)]
pub struct Shell {
    pub count: usize,
    /// Radius factor relative to the nearest-neighbour distance.
    pub radius: f64,
}

impl Shell {
    pub const fn new(count: usize, radius: f64) -> Self {
        Shell { count, radius }
    }
}

/// A lattice definition.
pub trait Lattice: Send + Sync + 'static {
    fn name(&self) -> &str;

    fn primitive(&self) -> PrimitiveCell;

    fn neighbour_shells(&self) -> &[Shell];

    /// Reference lattice-site positions covering a box (for WS, §16).
    ///
    /// Provided by the generic generator in [`sites`] so a new lattice only has
    /// to implement `primitive()`. Prefer [`sites::SiteLattice`] directly: this
    /// materialises every site, which §16.1 rules out at scale and which is
    /// retained only for small-box callers and tests.
    fn ws_sites(&self, cell: &UnitCell, origin: Coord3) -> Vec<Coord3> {
        match crate::lattice::sites::site_lattice_for(self, cell, origin) {
            Ok(sl) => sl.iter().collect(),
            Err(_) => Vec::new(),
        }
    }
}

/// Registry of named lattices.
#[derive(Default)]
pub struct LatticeRegistry {
    lat: Vec<Box<dyn Lattice>>,
}

impl LatticeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, l: Box<dyn Lattice>) {
        self.lat.push(l);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Lattice> {
        self.lat
            .iter()
            .find(|l| l.name() == name)
            .map(|b| b.as_ref())
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Lattice> {
        self.lat.iter().map(|b| b.as_ref())
    }

    pub fn names(&self) -> Vec<&str> {
        self.lat.iter().map(|l| l.name()).collect()
    }

    pub fn builtins() -> Self {
        let mut r = LatticeRegistry::new();
        r.register(Box::<fcc::Fcc>::default());
        r.register(Box::<bcc::Bcc>::default());
        r.register(Box::<hcp::Hcp>::default());
        r.register(Box::<sc::Sc>::default());
        r.register(Box::<dia::Dia>::default());
        r.register(Box::<dhcp::Dhcp>::default());
        r
    }
}

/// Build the lattice `name` from a measured nearest-neighbour distance `nn`.
///
/// `rmda ws --lattice` and the Python `ws(lattice=..., nn_distance=...)` both
/// take a nearest-neighbour distance and have to turn it into a conventional
/// parameter. One table here means the two cannot drift apart, and one place
/// means one round trip can check it: only SC has
/// nearest-neighbour distance equal to its parameter, FCC's conventional cell is
/// `nn·√2` across, HCP's `a` *is* the nearest-neighbour distance, DIA's is
/// `a√3/4`.
///
/// [`fit::nn_distance`] inverts this. The round trip below is what catches a
/// wrong factor: it gives every site the wrong spacing while the defect *counts*
/// stay plausible.
pub fn from_nn(name: &str, nn: f64) -> Option<Box<dyn Lattice>> {
    let lat: Box<dyn Lattice> = match name {
        "fcc" => Box::new(fcc::Fcc {
            a: nn * 2.0_f64.sqrt(),
        }),
        "bcc" => Box::new(bcc::Bcc {
            a: nn * 2.0 / 3.0_f64.sqrt(),
        }),
        // Twelve neighbours at `a`: six in plane, six out of plane at
        // sqrt(a²/3 + c²/4), which is `a` at the ideal ratio built here.
        "hcp" => Box::new(hcp::Hcp {
            a: nn,
            c: nn * (8.0_f64 / 3.0).sqrt(),
        }),
        "sc" => Box::new(sc::Sc { a: nn }),
        // Four neighbours at a√3/4.
        "dia" => Box::new(dia::Dia {
            a: nn * 4.0 / 3.0_f64.sqrt(),
        }),
        "dhcp" => Box::new(dhcp::Dhcp {
            a: nn,
            c: nn * 2.0 * (8.0_f64 / 3.0).sqrt(),
        }),
        _ => return None,
    };
    Some(lat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::fit::nn_distance;

    /// The declared neighbour shells describe the lattice actually built:
    /// counted from every basis atom of a block of primitive cells, the first
    /// shells have the declared counts at the declared radii (to 1e-7, the
    /// precision of the HCP literals).
    #[test]
    fn declared_neighbour_shells_match_the_built_lattice() {
        for name in ["fcc", "bcc", "sc", "dia", "hcp"] {
            let lat = from_nn(name, 1.0).expect("known lattice");
            let pc = lat.primitive();
            let basis = pc.basis_cart();
            let mut points = Vec::new();
            for i in -5i32..=5 {
                for j in -5i32..=5 {
                    for k in -5i32..=5 {
                        let t = pc.h.mul_v([i as f64, j as f64, k as f64]);
                        for b in &basis {
                            points.push([b[0] + t[0], b[1] + t[1], b[2] + t[2]]);
                        }
                    }
                }
            }
            for centre in &basis {
                let mut d: Vec<f64> = points
                    .iter()
                    .map(|p| {
                        ((p[0] - centre[0]).powi(2)
                            + (p[1] - centre[1]).powi(2)
                            + (p[2] - centre[2]).powi(2))
                        .sqrt()
                    })
                    .filter(|&r| r > 1e-9)
                    .collect();
                d.sort_by(f64::total_cmp);
                let mut shells: Vec<(usize, f64)> = Vec::new();
                for r in d {
                    match shells.last_mut() {
                        Some((n, r0)) if (r - *r0).abs() < 1e-6 => *n += 1,
                        _ => shells.push((1, r)),
                    }
                }
                for (s, want) in lat.neighbour_shells().iter().enumerate() {
                    let (count, radius) = shells[s];
                    assert_eq!(count, want.count, "{name} shell {s}: count");
                    assert!(
                        (radius - want.radius).abs() < 1e-7,
                        "{name} shell {s}: radius {radius}, declared {}",
                        want.radius
                    );
                }
            }
        }
    }

    #[test]
    fn a_nearest_neighbour_distance_round_trips_through_every_lattice() {
        for name in ["fcc", "bcc", "hcp", "sc", "dia", "dhcp"] {
            let nn = 2.5;
            let lat = from_nn(name, nn).expect("known lattice");
            let got = nn_distance(&lat.primitive()).expect("primitive must be invertible");
            assert!(
                (got - nn).abs() < 1e-9,
                "{name}: nn_distance {got}, wanted {nn}"
            );
        }
        assert!(from_nn("graphite", 1.0).is_none(), "unknown lattice");
    }
}
