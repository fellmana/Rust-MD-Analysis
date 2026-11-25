//! Affine site lattices (design §16.3b).
//!
//! A reference is `x = o + T·(n + b_k)`: `T` holds the conventional cell's
//! three vectors as columns in Cartesian space (nine free components, so
//! rotation, strain and shear are all representable), `n ∈ ℤ³` indexes the
//! cell and `b_k` is the motif's fractional basis. Sites are addressed, never
//! stored: the nearest site of any point is closed-form.

use crate::analysis::ptm::structure;
use crate::cell::{Coord3, Mat3};

/// The motif of a structure in its conventional cell, for `a = 1`.
#[derive(Debug, Clone)]
pub struct Motif {
    pub name: &'static str,
    /// PTM structure code this motif describes.
    pub structure: i32,
    /// Conventional cell vectors as columns, `a = 1`.
    pub cell: Mat3,
    /// Basis in fractional coordinates of `cell`.
    pub basis: Vec<Coord3>,
    /// The basis offsets that are pure lattice translations (always includes 0).
    /// For a Bravais lattice that is the whole basis; for HCP only the
    /// centring vector.
    pub translations: Vec<Coord3>,
    /// First-shell distance for `a = 1`.
    pub nn: f64,
    /// First-shell neighbours of basis site `k`, as `(dn, l)`: the site
    /// `(n + dn, l)` is a first neighbour of `(n, k)`.
    pub shell: Vec<Vec<([i32; 3], u8)>>,
    /// Every first-shell vector in fractional coordinates, over all basis sites.
    pub shell_vectors: Vec<Coord3>,
}

impl Motif {
    /// The motif for a PTM structure code, if it has a lattice.
    pub fn for_structure(code: i32) -> Option<Motif> {
        let s3 = 3.0_f64.sqrt();
        let (name, cell, basis, translations, nn): (_, Mat3, Vec<Coord3>, Vec<Coord3>, f64) =
            match code {
                structure::FCC => {
                    let b = vec![
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ];
                    ("fcc", Mat3::identity(), b.clone(), b, 0.5_f64.sqrt())
                }
                structure::BCC => {
                    let b = vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
                    ("bcc", Mat3::identity(), b.clone(), b, s3 / 2.0)
                }
                structure::SC => {
                    let b = vec![[0.0, 0.0, 0.0]];
                    ("sc", Mat3::identity(), b.clone(), b, 1.0)
                }
                structure::HCP => {
                    // Orthohexagonal cell, ideal c/a; the fit frees c/a. Same
                    // basis as `lattice::hcp`.
                    let c = (8.0_f64 / 3.0).sqrt();
                    let cell = Mat3::from_rows([1.0, 0.0, 0.0], [0.0, s3, 0.0], [0.0, 0.0, c]);
                    let b = vec![
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.0],
                        [0.0, 1.0 / 3.0, 0.5],
                        [0.5, 5.0 / 6.0, 0.5],
                    ];
                    ("hcp", cell, b, vec![[0.0; 3], [0.5, 0.5, 0.0]], 1.0)
                }
                structure::DHCP => {
                    // Orthohexagonal cell over four layers, ideal c/a (twice
                    // HCP's); the fit frees c/a. ABAC: A at (0,0), B at
                    // (0,1/3), C at (0,2/3) in the (a, √3a) rectangle, each
                    // with its (1/2,1/2) partner. Same stacking as
                    // `lattice::dhcp`.
                    let c = 2.0 * (8.0_f64 / 3.0).sqrt();
                    let cell = Mat3::from_rows([1.0, 0.0, 0.0], [0.0, s3, 0.0], [0.0, 0.0, c]);
                    let b = vec![
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.0],
                        [0.0, 1.0 / 3.0, 0.25],
                        [0.5, 5.0 / 6.0, 0.25],
                        [0.0, 0.0, 0.5],
                        [0.5, 0.5, 0.5],
                        [0.0, 2.0 / 3.0, 0.75],
                        [0.5, 1.0 / 6.0, 0.75],
                    ];
                    ("dhcp", cell, b, vec![[0.0; 3], [0.5, 0.5, 0.0]], 1.0)
                }
                _ => return None,
            };
        let mut m = Motif {
            name,
            structure: code,
            cell,
            basis,
            translations,
            nn,
            shell: Vec::new(),
            shell_vectors: Vec::new(),
        };
        m.build_shell();
        Some(m)
    }

    fn build_shell(&mut self) {
        let tol = 1e-6 * self.nn.max(1.0);
        let nb = self.basis.len();
        self.shell = vec![Vec::new(); nb];
        for k in 0..nb {
            for l in 0..nb {
                for dx in -2..=2 {
                    for dy in -2..=2 {
                        for dz in -2..=2 {
                            let v = [
                                dx as f64 + self.basis[l][0] - self.basis[k][0],
                                dy as f64 + self.basis[l][1] - self.basis[k][1],
                                dz as f64 + self.basis[l][2] - self.basis[k][2],
                            ];
                            let c = self.cell.mul_v(v);
                            let d = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                            if (d - self.nn).abs() < tol {
                                self.shell[k].push(([dx, dy, dz], l as u8));
                                if !self
                                    .shell_vectors
                                    .iter()
                                    .any(|w| (0..3).all(|a| (w[a] - v[a]).abs() < 1e-9))
                                {
                                    self.shell_vectors.push(v);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// First-shell coordination number.
    pub fn coordination(&self) -> usize {
        self.shell.first().map_or(0, Vec::len)
    }
}

/// A site: cell index and basis index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Site {
    pub n: [i32; 3],
    pub k: u8,
}

/// A fitted affine lattice.
#[derive(Debug, Clone)]
pub struct AffineLattice {
    pub motif: Motif,
    /// Conventional cell vectors as columns, Cartesian.
    pub t: Mat3,
    pub t_inv: Mat3,
    pub origin: Coord3,
}

impl AffineLattice {
    pub fn new(motif: Motif, t: Mat3, origin: Coord3) -> Option<Self> {
        let t_inv = t.inverse()?;
        Some(Self {
            motif,
            t,
            t_inv,
            origin,
        })
    }

    /// Fractional (cell) coordinates of a Cartesian point.
    #[inline]
    pub fn frac(&self, x: Coord3) -> Coord3 {
        self.t_inv.mul_v([
            x[0] - self.origin[0],
            x[1] - self.origin[1],
            x[2] - self.origin[2],
        ])
    }

    /// Cartesian position of a site.
    #[inline]
    pub fn position(&self, s: Site) -> Coord3 {
        let b = self.motif.basis[s.k as usize];
        let f = [
            s.n[0] as f64 + b[0],
            s.n[1] as f64 + b[1],
            s.n[2] as f64 + b[2],
        ];
        let c = self.t.mul_v(f);
        [
            c[0] + self.origin[0],
            c[1] + self.origin[1],
            c[2] + self.origin[2],
        ]
    }

    /// Nearest site to `x`, with the displacement `x − site` and its square.
    ///
    /// Per sublattice the nearest cell is the rounding of the fractional
    /// offset, exact for the near-orthogonal conventional cells used here;
    /// the sublattices are then compared in Cartesian space, ties to the lower
    /// basis index.
    #[inline]
    pub fn nearest(&self, x: Coord3) -> (Site, Coord3, f64) {
        let u = self.frac(x);
        let mut best = (Site { n: [0; 3], k: 0 }, [0.0; 3], f64::INFINITY);
        for (k, b) in self.motif.basis.iter().enumerate() {
            let n = [
                (u[0] - b[0]).round() as i32,
                (u[1] - b[1]).round() as i32,
                (u[2] - b[2]).round() as i32,
            ];
            let s = Site { n, k: k as u8 };
            let p = self.position(s);
            let dr = [x[0] - p[0], x[1] - p[1], x[2] - p[2]];
            let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            if d2 < best.2 {
                best = (s, dr, d2);
            }
        }
        best
    }

    /// Fitted first-neighbour distance (mean over the shell).
    pub fn nn(&self) -> f64 {
        let v = &self.motif.shell_vectors;
        if v.is_empty() {
            return f64::NAN;
        }
        v.iter()
            .map(|f| {
                let c = self.t.mul_v(*f);
                (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
            })
            .sum::<f64>()
            / v.len() as f64
    }
}

/// A site reference: maps positions to site keys and back. Keys are global
/// lattice indices; `global` is the single lattice whose index space every
/// implementation shares (and which carries the box commensurability, so
/// periodic wrapping is index arithmetic on it).
pub trait Reference: Sync {
    fn global(&self) -> &AffineLattice;
    /// Nearest site to `x`, displacement `x − site`, and its square.
    fn nearest(&self, x: Coord3) -> (Site, Coord3, f64);
    /// Cartesian position of a site.
    fn position(&self, s: Site) -> Coord3;
}

impl Reference for AffineLattice {
    fn global(&self) -> &AffineLattice {
        self
    }
    fn nearest(&self, x: Coord3) -> (Site, Coord3, f64) {
        AffineLattice::nearest(self, x)
    }
    fn position(&self, s: Site) -> Coord3 {
        AffineLattice::position(self, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dhcp_motif_is_abac_with_twelve_neighbours_everywhere() {
        let m = Motif::for_structure(structure::DHCP).unwrap();
        assert_eq!(m.basis.len(), 8);
        for k in 0..8 {
            assert_eq!(m.shell[k].len(), 12, "basis site {k}");
        }
        // Four layers, one close-packed spacing apart: c = 4·√(2/3).
        assert!((m.cell.m[2][2] - 4.0 * (2.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn coordination_numbers() {
        assert_eq!(
            Motif::for_structure(structure::FCC).unwrap().coordination(),
            12
        );
        assert_eq!(
            Motif::for_structure(structure::BCC).unwrap().coordination(),
            8
        );
        assert_eq!(
            Motif::for_structure(structure::SC).unwrap().coordination(),
            6
        );
        let h = Motif::for_structure(structure::HCP).unwrap();
        assert!(h.shell.iter().all(|s| s.len() == 12));
    }

    #[test]
    fn nearest_recovers_every_site_of_a_sheared_lattice() {
        let m = Motif::for_structure(structure::BCC).unwrap();
        let t = Mat3::from_rows([3.2, 0.08, 0.0], [0.0, 3.1, -0.05], [0.04, 0.0, 3.3]);
        let lat = AffineLattice::new(m, t, [0.3, -0.2, 1.0]).unwrap();
        for n0 in -2..3 {
            for n1 in -2..3 {
                for k in 0..2u8 {
                    let s = Site { n: [n0, n1, 1], k };
                    let p = lat.position(s);
                    let q = [p[0] + 0.3, p[1] - 0.2, p[2] + 0.1];
                    assert_eq!(lat.nearest(q).0, s);
                }
            }
        }
    }

    #[test]
    fn inverse_is_an_inverse() {
        let t = Mat3::from_rows([3.2, 0.08, 0.5], [0.1, 3.1, -0.05], [0.04, 0.7, 3.3]);
        let i = t.mul(&t.inverse().unwrap());
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((i.m[r][c] - want).abs() < 1e-12);
            }
        }
    }
}
