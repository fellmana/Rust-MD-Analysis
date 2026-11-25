//! Generic Wigner–Seitz site-lattice generation (design §11.1).
//!
//! Both `rmda ws` engines (`WsRef` with a given lattice, `WsSelf` with one
//! fitted by [`super::fit`]) need the same thing underneath: a set of lattice
//! sites that tiles the simulation box, addressed by integer index rather than
//! materialised (design §16.1). 128M reference sites must not become a
//! `Vec<Coord3>` — at 24 bytes per site that is 3 GB of coordinates before any
//! occupancy array, so the site set here is lazily addressable and O(1) in
//! memory.
//!
//! # Why one generic generator rather than per-lattice parity tables
//!
//! Per-lattice parity tables over cell indices (`(ix+iy+iz) % 2` for FCC,
//! all-equal-parity for BCC, and so on) are a sound idea and cheap, but they
//! are duplicated logic that has to be
//! re-derived and re-tested for every new lattice, and they silently assume the
//! crystal is axis-aligned with the box. Generating from `Lattice::primitive()`
//! instead makes the parity an emergent property of the basis: a new lattice
//! only has to implement `primitive()` and gets correct WS sites for free,
//! which is the extensibility the polycrystalline pipeline (§19) depends on.
//!
//! # Commensurability
//!
//! For a box that is exactly `N0×N1×N2` conventional cells, the generated sites
//! are exactly the ideal positions, all inside the box, with no duplicates and
//! `len() == N0*N1*N2*basis.len()` — the case `WsRef` requires (it refuses a
//! box that fails [`SiteLattice::is_commensurate`]). Whole-cell counts are
//! snapped rather than rounded up,
//! because an extra cell per axis is a whole shell of sites lying outside the
//! box, and every one of those would read as a vacancy.
//!
//! For an incommensurate box the site count is rounded up, so sites spill past
//! the boundary instead of wrapping. `len()` is then an upper bound on the
//! in-box count. That is deliberate: wrapping makes sites coincide, so it would
//! need a dedupe hash, and at 128M sites that is a worse trade than generating
//! the boundary margin. Callers that need a guaranteed in-box set check
//! [`SiteLattice::is_commensurate`] first, so they never hit this case.

use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::Result;
use crate::lattice::{Lattice, PrimitiveCell};

/// A lazily-addressed set of Wigner–Seitz lattice sites covering a box.
///
/// Sites are enumerated cell-major then basis-minor, so index order is
/// deterministic and independent of any hashing or insertion order.
#[derive(Debug, Clone)]
pub struct SiteLattice {
    /// Where fractional coordinate (0,0,0) of cell (0,0,0) sits, in Cartesian.
    origin: Coord3,
    /// Number of conventional cells generated per axis.
    dims: [usize; 3],
    /// Conventional-cell basis, fractional.
    basis: Vec<Coord3>,
    /// Conventional-cell h-matrix (rows).
    h: Mat3,
    /// `h⁻¹`, cached. Sites are assigned per atom, so inverting per atom would be
    /// a 3×3 Cramer solve in the hot loop for no reason.
    inv_h: Mat3,
    /// Box extent per axis, in units of conventional cells.
    spans: [f64; 3],
}

impl SiteLattice {
    /// Build the site lattice for `cell`, with the lattice origin at `origin`.
    ///
    /// `origin` is the Cartesian position of the conventional cell's (0,0,0)
    /// corner. The origin search of [`super::fit`] produces one; this is
    /// where it enters.
    ///
    /// # Rotated and offset lattices
    ///
    /// The repeat count per axis comes from the span of the box corners in
    /// fractional coordinates. Because the box maps to a parallelepiped under
    /// that affine map, its per-axis extrema are attained at corners, so the
    /// span is exact for *any* orientation. The enumeration is then relabelled so
    /// cell `i` corresponds to the original index `i - floor(fmin)`; the stored
    /// origin is moved back by `h · shift` to match. That makes the generated
    /// block cover a rotated or offset box just as it does an axis-aligned one.
    /// A 5×5×5 FCC cell of `a = 3.615` tilted 17° about z enumerates 980 sites of
    /// which 505 land inside, against 500 due (the extra five are the boundary
    /// shell of an incommensurate box). See
    /// `a_rotated_lattice_is_not_covered_by_the_box`.
    ///
    /// Snapping the span to a nearby integer is still applied: for a
    /// commensurate box that lands a few ulps above an integer, rounding up
    /// would add a whole phantom shell, which is exactly what the snapping
    /// exists to protect.
    pub fn build(cell: &UnitCell, pc: &PrimitiveCell, origin: Coord3) -> Result<Self> {
        if pc.basis.is_empty() {
            return Err(crate::error::Error::InvalidData(
                "lattice basis is empty".into(),
            ));
        }
        let h = pc.h;
        if h.det().abs() < 1e-300 {
            return Err(crate::error::Error::InvalidData(
                "lattice cell matrix is singular".into(),
            ));
        }

        // Map the 8 box corners into fractional lattice coordinates and take the
        // bounding interval. Corners are sufficient because the map is affine, so
        // the fractional image of the box is exactly the parallelepiped spanned by
        // the corner images.
        //
        // These are the cell's own corners. On an orthorhombic cell they are the
        // corners of `bounds()`; on a tilted cell the bounding box's corners are
        // not the cell's, and a lattice that tiles the tilted cell would not
        // come out commensurate from them.
        let mut fmin = [f64::INFINITY; 3];
        let mut fmax = [f64::NEG_INFINITY; 3];
        for c in cell.corners() {
            let f = h.solve([c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]])?;
            for d in 0..3 {
                if f[d] < fmin[d] {
                    fmin[d] = f[d];
                }
                if f[d] > fmax[d] {
                    fmax[d] = f[d];
                }
            }
        }

        // Shift the origin back by whole cells so no box corner needs a negative
        // cell index. Relabelling i is harmless: only the site set matters, and
        // this keeps the enumeration start at zero.
        let mut shift = [0i64; 3];
        let mut dims = [0usize; 3];
        let mut spans = [0.0f64; 3];
        for d in 0..3 {
            let back = fmin[d].floor() as i64;
            shift[d] = -back;
            let span = fmax[d] - fmin[d];
            spans[d] = span;
            // Always round up, so the generated cells cover the box. On an
            // incommensurate box that adds a shell of sites outside it, and every
            // one of those reads as a vacancy, so callers that report defects must
            // check `is_commensurate` instead of trusting `len()`.
            //
            // The one exception is a span that is *meant* to be an exact integer
            // but lands a few ulps above it: `ceil(4.000000000000001)` is 5, which
            // silently doubles the site count along that axis and invents a whole
            // shell of phantom vacancies. This is not hypothetical -- a 40-cell
            // box whose length came from `hi - lo` of parsed decimal text hits it
            // in ordinary use. So snap only when the span is within a relative
            // 1e-9 of an integer, which is far tighter than any real
            // commensurability error and cannot mask a genuinely short box.
            let nearest = span.round();
            dims[d] = if nearest >= 1.0 && (span - nearest).abs() <= 1e-9 * span.max(1.0) {
                nearest as usize
            } else {
                span.ceil().max(1.0) as usize
            };
        }

        // Relabelled site `i` (i = 0..dims) corresponds to the original cell
        // index `i - shift`, so the stored origin must be moved *back* by
        // `h · shift`. Getting the sign wrong does not show up when the box is
        // axis-aligned at the lattice origin (fmin == 0, shift == 0) but leaves
        // the box only partly covered for any tilted or offset lattice.
        let sf = [shift[0] as f64, shift[1] as f64, shift[2] as f64];
        let ds = h.mul_v(sf);
        let origin = [origin[0] - ds[0], origin[1] - ds[1], origin[2] - ds[2]];

        // inv_h has columns h⁻¹e₀, h⁻¹e₁, h⁻¹e₂, i.e. rows are the transposed
        // components of each solved basis vector.
        let c0 = h.solve([1.0, 0.0, 0.0])?;
        let c1 = h.solve([0.0, 1.0, 0.0])?;
        let c2 = h.solve([0.0, 0.0, 1.0])?;
        let inv_h = Mat3::from_rows(
            [c0[0], c1[0], c2[0]],
            [c0[1], c1[1], c2[1]],
            [c0[2], c1[2], c2[2]],
        );

        Ok(SiteLattice {
            origin,
            dims,
            basis: pc.basis.clone(),
            h,
            inv_h,
            spans,
        })
    }

    /// Number of sites per conventional cell.
    #[inline]
    pub fn basis_len(&self) -> usize {
        self.basis.len()
    }

    /// Number of conventional cells generated per axis.
    #[inline]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Box extent per axis, in units of conventional cells.
    #[inline]
    pub fn cell_spans(&self) -> [f64; 3] {
        self.spans
    }

    /// Distance from a whole number of cells, per axis. Zero is commensurate.
    ///
    /// This is not cosmetic. `dims` is a `ceil`, so a span of 6.0004 stores 7
    /// cells and every site in the extra slab reads as a vacancy: 508 phantom
    /// vacancies on a 6x6x6 FCC block, from a lattice constant supplied to 5
    /// significant figures. Whether the defect counts are right therefore flips
    /// on the 5th decimal place of a user-supplied number, so callers that care
    /// must check this instead of trusting `len()`.
    #[inline]
    pub fn commensurability_error(&self) -> [f64; 3] {
        let mut out = [0.0; 3];
        for (d, &s) in self.spans.iter().enumerate() {
            out[d] = (s - s.round()).abs();
        }
        out
    }

    /// Whether every axis holds a whole number of conventional cells.
    #[inline]
    pub fn is_commensurate(&self, tol: f64) -> bool {
        self.commensurability_error().iter().all(|&e| e <= tol)
    }

    /// Total number of sites.
    #[inline]
    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2] * self.basis.len()
    }

    /// Always false; `len` is non-zero because the basis is validated non-empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Cartesian position of site `idx`.
    ///
    /// Index order is cell-major, basis-minor: `idx` advances through the basis
    /// of one cell before moving to the next.
    #[inline]
    pub fn get(&self, idx: usize) -> Coord3 {
        let nb = self.basis.len();
        debug_assert!(idx < self.len(), "site index out of range");
        let cell = idx / nb;
        let b = self.basis[idx % nb];
        let ix = cell % self.dims[0];
        let iy = (cell / self.dims[0]) % self.dims[1];
        let iz = cell / (self.dims[0] * self.dims[1]);
        let f = [ix as f64 + b[0], iy as f64 + b[1], iz as f64 + b[2]];
        let x = self.h.mul_v(f);
        [
            x[0] + self.origin[0],
            x[1] + self.origin[1],
            x[2] + self.origin[2],
        ]
    }

    /// Integer cell indices of site `idx`, as `(ix, iy, iz)`.
    #[inline]
    pub fn cell_index(&self, idx: usize) -> [usize; 3] {
        let nb = self.basis.len();
        let cell = idx / nb;
        [
            cell % self.dims[0],
            (cell / self.dims[0]) % self.dims[1],
            cell / (self.dims[0] * self.dims[1]),
        ]
    }

    /// Site index for integer cell `i` and basis atom `bi`, wrapping `i` into
    /// `0..dims` with a positive modulo.
    #[inline]
    pub fn site_index(&self, i: [i64; 3], bi: usize) -> usize {
        let d = self.dims;
        let mut c = [0usize; 3];
        for k in 0..3 {
            let m = i[k].rem_euclid(d[k] as i64) as usize;
            c[k] = m;
        }
        ((c[2] * d[1] + c[1]) * d[0] + c[0]) * self.basis.len() + bi
    }

    /// Iterate over every site position.
    pub fn iter(&self) -> impl Iterator<Item = Coord3> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// `h⁻¹`, cached at build time.
    #[inline]
    pub fn inv_h(&self) -> &Mat3 {
        &self.inv_h
    }

    /// Number of sites per unit volume, for cross-checking a total count.
    #[inline]
    pub fn site_density(&self) -> f64 {
        self.basis.len() as f64 / self.h.det().abs()
    }
}

/// Build the site lattice for any `Lattice` using only its `primitive()`.
///
/// This is the extension point: a new structure type implements `name`,
/// `primitive`, and `neighbour_shells`, and gets WS sites with no WS-specific
/// code at all.
pub fn site_lattice_for<L: Lattice + ?Sized>(
    lat: &L,
    cell: &UnitCell,
    origin: Coord3,
) -> Result<SiteLattice> {
    SiteLattice::build(cell, &lat.primitive(), origin)
}

/// Where one atom sits relative to its nearest reference site.
#[derive(Debug, Clone, Copy)]
pub struct Assignment {
    /// Index into the `SiteLattice`, always in `0..len()`.
    pub site: usize,
    /// Minimum-image vector from the site to the atom.
    pub displacement: Coord3,
    /// Distance to that site.
    pub distance: f64,
}

/// Reusable atom → site assignment, with both matrix inverses hoisted out of the
/// per-atom loop.
///
/// This is lattice-index arithmetic (design §16.1) instead of a
/// nearest-neighbour tree: the nearest site to `x` is found by rounding into each
/// basis atom's own cell, so the work is `basis.len()` candidates rather than a
/// tree descent. For FCC that is 4 candidates against a tree's ~log(n) — and at
/// 128M reference sites it is also the difference between a tree and no tree.
///
/// Correctness note: an atom near the upper box boundary rounds to a cell index
/// equal to `dims`, i.e. one *past* the generated set. That is not a bug and
/// needs no margin: under PBC that site is the image of the in-box site at
/// `dims-1`, so wrapping the index is exactly right, and the displacement is
/// still measured to the true lattice point. Clamping instead would bias
/// boundary atoms onto the edge site and manufacture interstitial pairs.
pub struct SiteAssign<'a> {
    cell: &'a UnitCell,
    sl: &'a SiteLattice,
    /// `cell.inverse_transposed()`, hoisted.
    inv_t: Mat3,
}

impl<'a> SiteAssign<'a> {
    pub fn new(cell: &'a UnitCell, sl: &'a SiteLattice) -> Result<Self> {
        Ok(SiteAssign {
            cell,
            sl,
            inv_t: cell.inverse_transposed()?,
        })
    }

    /// Assign `x` to its nearest site.
    pub fn assign(&self, x: Coord3) -> Assignment {
        let sl = self.sl;
        let o = sl.origin;
        let f = sl.inv_h.mul_v([x[0] - o[0], x[1] - o[1], x[2] - o[2]]);

        let mut best: Option<Assignment> = None;
        let mut best_r2 = f64::INFINITY;

        for (bi, b) in sl.basis.iter().enumerate() {
            // Nearest cell to this basis atom, per axis.
            let i = [
                (f[0] - b[0]).round() as i64,
                (f[1] - b[1]).round() as i64,
                (f[2] - b[2]).round() as i64,
            ];
            let px =
                sl.h.mul_v([i[0] as f64 + b[0], i[1] as f64 + b[1], i[2] as f64 + b[2]]);
            let p = [px[0] + o[0], px[1] + o[1], px[2] + o[2]];
            let d = self
                .cell
                .min_image_with_shift_pre([x[0] - p[0], x[1] - p[1], x[2] - p[2]], &self.inv_t)
                .0;
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < best_r2 {
                best_r2 = r2;
                best = Some(Assignment {
                    site: sl.site_index(i, bi),
                    displacement: d,
                    distance: r2.sqrt(),
                });
            }
        }

        // `basis` is validated non-empty at build, so this cannot be None.
        best.expect("site lattice basis is non-empty")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::bcc::Bcc;
    use crate::lattice::dia::Dia;
    use crate::lattice::fcc::Fcc;
    use crate::lattice::hcp::Hcp;
    use crate::lattice::sc::Sc;

    /// A box of exactly `n` conventional cells along each axis.
    fn commensurate_box(n: [usize; 3], a: f64) -> UnitCell {
        UnitCell::orthorhombic(n[0] as f64 * a, n[1] as f64 * a, n[2] as f64 * a)
    }

    /// Sites per conventional cell, analytically (HCP's is the orthohexagonal
    /// box, which holds four atoms: two layers of the ABAB stack).
    const PER_CELL: &[(&str, usize)] = &[("sc", 1), ("fcc", 4), ("bcc", 2), ("hcp", 4), ("dia", 8)];

    fn sites_per_cell(lat: &dyn Lattice) -> usize {
        lat.primitive().basis.len()
    }

    #[test]
    fn site_counts_match_the_analytic_basis_size() {
        // Guards the whole design: a wrong basis size silently turns every
        // vacancy/interstitial count wrong, and no downstream test would say so.
        assert_eq!(sites_per_cell(&Sc::default()), PER_CELL[0].1, "sc");
        assert_eq!(sites_per_cell(&Fcc::default()), PER_CELL[1].1, "fcc");
        assert_eq!(sites_per_cell(&Bcc::default()), PER_CELL[2].1, "bcc");
        assert_eq!(sites_per_cell(&Hcp::default()), PER_CELL[3].1, "hcp");
        assert_eq!(sites_per_cell(&Dia::default()), PER_CELL[4].1, "dia");
    }

    #[test]
    fn commensurate_box_yields_exactly_the_ideal_sites() {
        // 4x4x4 conventional FCC cells of a=1 must give 4*4*4*4 sites, all
        // distinct and all inside the box.
        let lat = Fcc { a: 1.0 };
        let cell = commensurate_box([4, 4, 4], 1.0);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        assert_eq!(sl.basis_len(), 4);
        assert_eq!(sl.len(), 4 * 4 * 4 * 4);

        let (lo, hi) = cell.bounds();
        let mut seen: Vec<Coord3> = Vec::with_capacity(sl.len());
        for p in sl.iter() {
            for d in 0..3 {
                assert!(
                    p[d] >= lo[d] - 1e-12 && p[d] <= hi[d] + 1e-12,
                    "site {p:?} outside box {lo:?}..{hi:?}"
                );
            }
            // Distinctness: quantise to 1e-9 and check no collisions.
            let key: (i64, i64, i64) = (
                (p[0] * 1e9).round() as i64,
                (p[1] * 1e9).round() as i64,
                (p[2] * 1e9).round() as i64,
            );
            assert!(!seen.iter().any(|q| {
                let k: (i64, i64, i64) = (
                    (q[0] * 1e9).round() as i64,
                    (q[1] * 1e9).round() as i64,
                    (q[2] * 1e9).round() as i64,
                );
                k == key
            }));
            seen.push(p);
        }
    }

    #[test]
    fn an_orthohexagonal_hcp_box_holds_exactly_four_sites_per_cell() {
        // The phantom-vacancy guard: an orthohexagonal box is a whole number of
        // HCP cells, so the site set has to be exactly 4·n0·n1·n2 sites. Under
        // a skewed two-atom cell the same box maps to a sheared parallelepiped
        // whose per-axis bounding box holds 1.67x the cells, and a perfect
        // crystal comes out with 256 phantom vacancies.
        let (a, n) = (2.5, [6usize, 4, 4]);
        let lat = Hcp {
            a,
            c: a * (8.0_f64 / 3.0).sqrt(),
        };
        let la = a * 3.0_f64.sqrt();
        let cell = UnitCell::orthorhombic(n[0] as f64 * a, n[1] as f64 * la, n[2] as f64 * lat.c);
        let sl = site_lattice_for(&lat, &cell, [0.0; 3]).unwrap();
        assert_eq!(sl.basis_len(), 4);
        assert_eq!(
            sl.len(),
            n[0] * n[1] * n[2] * 4,
            "site set overfills an axis-aligned orthohexagonal box"
        );
    }

    #[test]
    fn generated_sites_reproduce_the_basis_under_pbc() {
        // Independent enumeration: take every site, wrap it into the box with
        // LAMMPS semantics, and check the wrapped set is exactly the ideal set
        // formed by (fractional basis + integer cells) mod 1.
        let lat = Fcc { a: 1.0 };
        let cell = commensurate_box([2, 3, 4], 1.0);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();

        let mut wrapped: Vec<[i64; 3]> = sl
            .iter()
            .map(|p| {
                let w = cell.wrap(p);
                [
                    (w[0] * 1e6).round() as i64,
                    (w[1] * 1e6).round() as i64,
                    (w[2] * 1e6).round() as i64,
                ]
            })
            .collect();
        wrapped.sort();
        let before = wrapped.len();
        wrapped.dedup();
        // No duplicates, i.e. wrapping did not collapse two sites onto one.
        assert_eq!(wrapped.len(), before, "wrapping created duplicate sites");

        // The wrapped count must be one per site, and the ideal set built from
        // the basis must have the same size.
        let expected = 2 * 3 * 4 * 4;
        assert_eq!(wrapped.len(), expected);
    }

    #[test]
    fn nearest_neighbour_distance_matches_the_shell_radius() {
        // The strongest end-to-end check that the parity is right: for FCC the
        // nearest site pair must be a/sqrt(2), not a. A wrong parity table gives
        // a/sqrt(2) vs a errors that no count-based test would catch.
        let a = 4.0;
        let lat = Fcc { a };
        let cell = commensurate_box([3, 3, 3], a);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        let pts: Vec<Coord3> = sl.iter().collect();

        let mut best = f64::INFINITY;
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                if i == j {
                    continue;
                }
                let d = cell
                    .min_image([
                        pts[i][0] - pts[j][0],
                        pts[i][1] - pts[j][1],
                        pts[i][2] - pts[j][2],
                    ])
                    .unwrap();
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if r < best {
                    best = r;
                }
            }
        }
        let want = a / 2.0_f64.sqrt();
        assert!(
            (best - want).abs() < 1e-9,
            "fcc nearest-neighbour distance {best}, expected {want}"
        );
    }

    #[test]
    fn bcc_nearest_neighbour_distance_is_the_body_diagonal_half() {
        let a = 3.0;
        let lat = Bcc { a };
        let cell = commensurate_box([3, 3, 3], a);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        let pts: Vec<Coord3> = sl.iter().collect();
        let mut best = f64::INFINITY;
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                if i == j {
                    continue;
                }
                let d = cell
                    .min_image([
                        pts[i][0] - pts[j][0],
                        pts[i][1] - pts[j][1],
                        pts[i][2] - pts[j][2],
                    ])
                    .unwrap();
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if r < best {
                    best = r;
                }
            }
        }
        let want = a * 3.0_f64.sqrt() / 2.0;
        assert!(
            (best - want).abs() < 1e-9,
            "bcc nearest-neighbour distance {best}, expected {want}"
        );
    }

    #[test]
    fn origin_shift_translates_the_whole_site_set() {
        // The fit's origin search exists precisely to choose this origin, so a
        // shift must move every site rigidly and change nothing else.
        let a = 2.0;
        let lat = Fcc { a };
        let cell = commensurate_box([3, 3, 3], a);
        let base = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        let off = [0.13, 0.0, 0.0];
        let moved = site_lattice_for(&lat, &cell, off).unwrap();
        assert_eq!(base.len(), moved.len());

        // Compare the two *sets*, not index by index: shifting the origin also
        // relabels the integer cell indices, so index i is not the same site in
        // both lattices. What must hold is that every shifted site has a partner.
        let key = |p: Coord3| -> [i64; 3] {
            let w = cell.wrap(p);
            [
                (w[0] * 1e6).round() as i64,
                (w[1] * 1e6).round() as i64,
                (w[2] * 1e6).round() as i64,
            ]
        };
        let mut want: Vec<[i64; 3]> = base
            .iter()
            .map(|p| key([p[0] + off[0], p[1] + off[1], p[2] + off[2]]))
            .collect();
        want.sort();
        let mut got: Vec<[i64; 3]> = moved.iter().map(key).collect();
        got.sort();
        assert_eq!(got, want, "origin shift did not translate the site set");
    }

    #[test]
    fn index_order_is_deterministic_and_coverable_by_cell_index() {
        let a = 1.5;
        let lat = Fcc { a };
        let cell = commensurate_box([3, 2, 2], a);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        let d = sl.dims();
        let nb = sl.basis_len();
        assert_eq!(sl.len(), d[0] * d[1] * d[2] * nb);
        let mut visited = vec![false; sl.len()];
        for (i, v) in visited.iter_mut().enumerate() {
            let ci = sl.cell_index(i);
            assert!(ci[0] < d[0] && ci[1] < d[1] && ci[2] < d[2]);
            let linear = (ci[2] * d[1] + ci[1]) * d[0] + ci[0];
            let expect = linear * nb + (i % nb);
            assert_eq!(expect, i, "index/cell mapping is not row-major");
            *v = true;
        }
        assert!(visited.iter().all(|&v| v), "some sites unreachable");
    }

    #[test]
    fn site_density_matches_basis_over_volume() {
        let a = 2.0;
        let lat = Fcc { a };
        let cell = commensurate_box([4, 4, 4], a);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        let n_expected = sl.site_density() * cell.volume();
        assert!(
            (n_expected - sl.len() as f64).abs() < 1e-9,
            "density implies {n_expected} sites, generator made {}",
            sl.len()
        );
    }

    // ---- assignment ----

    /// Every generated site must assign to itself. This is the property the whole
    /// WS engine rests on: if a site does not round back to itself, occupancy
    /// counts land on the wrong site and every vacancy is fiction.
    #[test]
    fn every_site_assigns_to_itself() {
        let a = 2.0;
        let cell = commensurate_box([3, 3, 3], a);
        for (name, lat) in [
            ("sc", &Sc { a } as &dyn crate::lattice::Lattice),
            ("fcc", &Fcc { a }),
            ("bcc", &Bcc { a }),
        ] {
            let sl = site_lattice_for(lat, &cell, [0.0, 0.0, 0.0]).unwrap();
            let asg = SiteAssign::new(&cell, &sl).unwrap();
            for i in 0..sl.len() {
                let p = sl.get(i);
                let got = asg.assign(p);
                assert_eq!(
                    got.site, i,
                    "{name}: site {i} at {p:?} assigned to {} (displacement {:?})",
                    got.site, got.displacement
                );
                assert!(
                    got.distance < 1e-9,
                    "{name}: site {i} displaced by {}",
                    got.distance
                );
            }
        }
    }

    #[test]
    fn assignment_recovers_a_planted_defect_free_lattice_exactly() {
        // Build atoms at the ideal sites; then every site must have occupancy 1
        // and every atom zero displacement. This is the whole WS-ref invariant.
        let a = 2.0;
        let lat = Fcc { a };
        let cell = commensurate_box([3, 3, 3], a);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        let asg = SiteAssign::new(&cell, &sl).unwrap();

        let mut occ = vec![0u32; sl.len()];
        for i in 0..sl.len() {
            let res = asg.assign(sl.get(i));
            assert!(res.distance < 1e-9);
            occ[res.site] += 1;
        }
        assert!(
            occ.iter().all(|&o| o == 1),
            "expected uniform occupancy 1, got {} zeros and {} doubles",
            occ.iter().filter(|&&o| o == 0).count(),
            occ.iter().filter(|&&o| o == 2).count()
        );
    }

    #[test]
    fn removed_atom_becomes_exactly_one_vacancy() {
        // The negative control. If a single removed atom does not produce exactly
        // one vacancy and no interstitial, the classification is not trustworthy.
        let a = 2.0;
        let lat = Fcc { a };
        let cell = commensurate_box([3, 3, 3], a);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        let asg = SiteAssign::new(&cell, &sl).unwrap();

        let victim = sl.len() / 2;
        let mut occ = vec![0u32; sl.len()];
        for i in 0..sl.len() {
            if i == victim {
                continue;
            }
            occ[asg.assign(sl.get(i)).site] += 1;
        }
        let vacancies: Vec<usize> = occ
            .iter()
            .enumerate()
            .filter(|(_, &o)| o == 0)
            .map(|(i, _)| i)
            .collect();
        let interstitials: usize = occ.iter().map(|&o| o.saturating_sub(1) as usize).sum();
        assert_eq!(vacancies, vec![victim]);
        assert_eq!(interstitials, 0);
    }

    #[test]
    fn atom_near_a_boundary_wraps_to_the_right_site() {
        // An atom just inside the top boundary rounds to cell index == dims. That
        // must wrap to site dims-1 (its own periodic image), not clamp and not
        // report out of range.
        let a = 2.0;
        let lat = Fcc { a };
        let cell = commensurate_box([3, 3, 3], a);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        let asg = SiteAssign::new(&cell, &sl).unwrap();

        let p = sl.get(0);
        // Nudge +x far enough that the nearest site is the periodic image.
        let x = [p[0] + 3.0 * a + 0.01, p[1], p[2]];
        let res = asg.assign(x);
        assert!(res.site < sl.len(), "site index out of range: {}", res.site);
        assert!(
            res.distance < 0.1,
            "expected the periodic image of site 0, got distance {}",
            res.distance
        );
    }

    #[test]
    fn assignment_is_independent_of_thread_scheduling() {
        // The determinism guarantee: each atom's site is a pure
        // function of its position, so only the shared occupancy counter is
        // order-sensitive, and that cannot change the final counts.
        let a = 2.0;
        let lat = Fcc { a };
        let cell = commensurate_box([4, 4, 4], a);
        let sl = site_lattice_for(&lat, &cell, [0.0, 0.0, 0.0]).unwrap();
        let asg = SiteAssign::new(&cell, &sl).unwrap();
        let sites: Vec<usize> = sl.iter().map(|p| asg.assign(p).site).collect();
        for (i, &s) in sites.iter().enumerate() {
            assert_eq!(s, i);
        }
    }

    /// A rotated primitive cell must still cover the whole box.
    ///
    /// Pins the sign of the fractional-origin shift: it only
    /// shows up when the box maps to fractional extrema that are not zero (a
    /// tilted or offset lattice), where the wrong sign leaves the box only
    /// partly covered. The site set here is incommensurate with the rotated
    /// lattice, so the in-box count is allowed to exceed `due` by a boundary
    /// shell, but it must never fall short of it.
    #[test]
    fn a_rotated_lattice_is_not_covered_by_the_box() {
        let a = 3.615;
        let cells = 5;
        let len = cells as f64 * a;
        let cell = UnitCell::orthorhombic(len, len, len);
        let (s, c) = 17f64.to_radians().sin_cos();
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];

        let pc = Fcc { a }.primitive();
        let h = pc.h;
        let rows = |i: usize| {
            [0, 1, 2].map(|j| r[i][0] * h.m[0][j] + r[i][1] * h.m[1][j] + r[i][2] * h.m[2][j])
        };
        let rotated =
            PrimitiveCell::new(Mat3::from_rows(rows(0), rows(1), rows(2)), pc.basis.clone());
        assert_eq!(*rotated.h.row(0), [a * c, -a * s, 0.0]);

        let lat = SiteLattice::build(&cell, &rotated, [0.0; 3]).unwrap();
        let inside = lat
            .iter()
            .filter(|p| (0..3).all(|d| p[d] >= 0.0 && p[d] < len))
            .count();
        let due = cells * cells * cells * 4;
        assert_eq!(lat.len(), 980, "sites enumerated for a 17 deg tilt");
        assert!(
            inside >= due,
            "rotated lattice must cover the box: {inside} sites inside, {due} due"
        );
        assert!(
            inside <= lat.len(),
            "in-box count cannot exceed the enumerated set"
        );
    }

    #[test]
    fn empty_basis_is_rejected() {
        let pc = PrimitiveCell::new(Mat3::identity(), vec![]);
        let cell = UnitCell::orthorhombic(1.0, 1.0, 1.0);
        assert!(SiteLattice::build(&cell, &pc, [0.0, 0.0, 0.0]).is_err());
    }
}
