//! `UnitCell`, PBC, minimum image, triclinic math, and image flags.
//!
//! This is the single most error-prone structure in the codebase (design §6.1).
//! All conventions are ported from LAMMPS `Domain`/`lmptype.h` and cross-checked
//! against ASE's `Cell`.
//!
//! Convention (design §6.1): **restricted triclinic**, rows are the three
//! lattice vectors, LOWER-triangular:
//! ```text
//!   a = (lx, 0, 0)
//!   b = (xy, ly, 0)
//!   c = (xz, yz, lz)
//! ```
//! positions `p = λx·a + λy·b + λz·c` with fractional λ ∈ [0,1):
//! ```text
//!   px = λx·lx + λy·xy + λz·xz
//!   py = λy·ly + λz·yz
//!   pz = λz·lz
//! ```
//! LAMMPS packs the same matrix with `h[6] = (lx, ly, lz, yz, xz, xy)`;
//! `lamda2x`/`x2lamda` below port `domain.cpp` verbatim.

use crate::error::{Error, Result};
use std::fmt;

mod minkowski;

/// Crate-wide 3-vector.
pub type Coord3 = [f64; 3];

/// IMGMASK / IMGMAX / IMGBITS exactly as LAMMPS `lmptype.h`.
pub const IMGMASK: i64 = 1023;
pub const IMGMAX: i64 = 512;
pub const IMGBITS: i64 = 10;
pub const IMG2BITS: i64 = 20;

/// Packed image flags: three 10-bit biased fields `(ix, iy, iz)` each in
/// `[-512, 511]`. `image = (ix) | (iy << 10) | (iz << 20)`.
pub type MdaImage = i64;

#[inline]
pub fn image_ipart(image: MdaImage) -> i64 {
    (image & IMGMASK) - IMGMAX
}
#[inline]
pub fn image_jpart(image: MdaImage) -> i64 {
    ((image >> IMGBITS) & IMGMASK) - IMGMAX
}
#[inline]
pub fn image_kpart(image: MdaImage) -> i64 {
    (image >> IMG2BITS) - IMGMAX
}

#[inline]
pub fn pack_image(ix: i64, iy: i64, iz: i64) -> MdaImage {
    ((ix + IMGMAX) & IMGMASK)
        | (((iy + IMGMAX) & IMGMASK) << IMGBITS)
        | (((iz + IMGMAX) & IMGMASK) << IMG2BITS)
}

/// 3×3 matrix stored row-major; rows are the lattice vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Mat3::identity()
    }
}

impl Mat3 {
    pub const fn identity() -> Self {
        Mat3 {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub const fn from_rows(r0: [f64; 3], r1: [f64; 3], r2: [f64; 3]) -> Self {
        Mat3 { m: [r0, r1, r2] }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f64; 3] {
        &self.m[i]
    }

    /// Length of row `i` — the lattice-vector length for a cell stored with its
    /// vectors as rows ([`UnitCell`], and the cubic `PrimitiveCell`s, where
    /// `h = a·I` makes rows and columns the same vectors).
    ///
    /// Invariant under rotation of the cell, unlike `row(i)[0]`: a fitted cell
    /// carries its orientation, so reading a single component of a rotated row
    /// reports `a·R₁₁` (→ 0.71 a at 45°, → 0 at 90°) while the label says
    /// lattice constant. Row norms are what the side length actually is.
    ///
    /// For cells stored with their vectors as columns
    /// ([`crate::lattice::PrimitiveCell`], where `mul_v`/`solve` apply `h` to a
    /// fractional column) the same quantity is [`Mat3::col_norm`]: the hexagonal
    /// cell's row 0 is `(a, a/2, 0)`, so `row_norm(0)` there reads 1.12·a.
    #[inline]
    pub fn row_norm(&self, i: usize) -> f64 {
        let r = self.row(i);
        (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt()
    }

    /// Length of column `i` — the lattice-vector length for a cell stored with
    /// its vectors as columns ([`crate::lattice::PrimitiveCell`]).
    ///
    /// Invariant under rotation for the same reason [`Mat3::row_norm`] is: the
    /// fitted cell is built as `R·h`, and left-multiplication rotates every
    /// column without changing its length. The two norms agree exactly when `h`
    /// is symmetric, which is why this only shows up on a hexagonal cell.
    #[inline]
    pub fn col_norm(&self, i: usize) -> f64 {
        let c0 = self.m[0][i];
        let c1 = self.m[1][i];
        let c2 = self.m[2][i];
        (c0 * c0 + c1 * c1 + c2 * c2).sqrt()
    }

    /// `out = M · v` (column vector).
    #[inline]
    pub fn mul_v(&self, v: Coord3) -> Coord3 {
        [
            self.m[0][0] * v[0] + self.m[0][1] * v[1] + self.m[0][2] * v[2],
            self.m[1][0] * v[0] + self.m[1][1] * v[1] + self.m[1][2] * v[2],
            self.m[2][0] * v[0] + self.m[2][1] * v[1] + self.m[2][2] * v[2],
        ]
    }

    /// Solve `M·x = b` by Cramer's rule (3×3, fine for our small sizes).
    pub fn solve(&self, b: Coord3) -> Result<Coord3> {
        let [m0, m1, m2] = self.m;
        let d = self.det();
        if d.abs() < 1e-300 {
            return Err(Error::InvalidData("singular cell matrix".into()));
        }
        let dx = [
            [b[0], m0[1], m0[2]],
            [b[1], m1[1], m1[2]],
            [b[2], m2[1], m2[2]],
        ];
        let dy = [
            [m0[0], b[0], m0[2]],
            [m1[0], b[1], m1[2]],
            [m2[0], b[2], m2[2]],
        ];
        let dz = [
            [m0[0], m0[1], b[0]],
            [m1[0], m1[1], b[1]],
            [m2[0], m2[1], b[2]],
        ];
        Ok([det3(dx) / d, det3(dy) / d, det3(dz) / d])
    }

    pub fn det(&self) -> f64 {
        det3(self.m)
    }

    /// Transpose (rows ↔ columns).
    pub fn transposed(&self) -> Mat3 {
        let m = self.m;
        Mat3 {
            m: [
                [m[0][0], m[1][0], m[2][0]],
                [m[0][1], m[1][1], m[2][1]],
                [m[0][2], m[1][2], m[2][2]],
            ],
        }
    }

    /// Matrix product `self · o`.
    pub fn mul(&self, o: &Mat3) -> Mat3 {
        let (a, b) = (&self.m, &o.m);
        let mut m = [[0.0; 3]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
            }
        }
        Mat3 { m }
    }

    /// Inverse by cofactors, or `None` when (numerically) singular.
    pub fn inverse(&self) -> Option<Mat3> {
        let a = &self.m;
        let det = self.det();
        if !det.is_finite() || det.abs() < 1e-300 {
            return None;
        }
        let c = |r0: usize, r1: usize, c0: usize, c1: usize| {
            a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]
        };
        let inv = 1.0 / det;
        Some(Mat3 {
            m: [
                [
                    c(1, 2, 1, 2) * inv,
                    -c(0, 2, 1, 2) * inv,
                    c(0, 1, 1, 2) * inv,
                ],
                [
                    -c(1, 2, 0, 2) * inv,
                    c(0, 2, 0, 2) * inv,
                    -c(0, 1, 0, 2) * inv,
                ],
                [
                    c(1, 2, 0, 1) * inv,
                    -c(0, 2, 0, 1) * inv,
                    c(0, 1, 0, 1) * inv,
                ],
            ],
        })
    }

    /// Follows ASE's sign convention: volume = det(H) > 0 for right-handed.
    pub fn volume(&self) -> f64 {
        self.det()
    }
}

#[inline]
fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[inline]
fn cross3(a: Coord3, b: Coord3) -> Coord3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: Coord3, b: Coord3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn len3(a: Coord3) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn norm3(a: &mut Coord3) {
    let l = len3(*a);
    if l > 0.0 {
        a[0] /= l;
        a[1] /= l;
        a[2] /= l;
    }
}

/// Quaternion product `c = a * b` (LAMMPS `MathExtra::quatquat`).
#[inline]
fn quatquat(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1],
    ]
}

/// Unit axis + angle to quaternion (LAMMPS `MathExtra::axisangle_to_quat`).
#[inline]
fn axisangle_to_quat(v: Coord3, angle: f64) -> [f64; 4] {
    let halfa = 0.5 * angle;
    let sina = halfa.sin();
    [halfa.cos(), v[0] * sina, v[1] * sina, v[2] * sina]
}

/// Quaternion to rotation matrix (LAMMPS `MathExtra::quat_to_mat`,
/// `math_extra.cpp:424`), quat = `[w, i, j, k]`.
#[inline]
fn quat_to_mat(q: [f64; 4]) -> Mat3 {
    let w2 = q[0] * q[0];
    let i2 = q[1] * q[1];
    let j2 = q[2] * q[2];
    let k2 = q[3] * q[3];
    let twoij = 2.0 * q[1] * q[2];
    let twoik = 2.0 * q[1] * q[3];
    let twojk = 2.0 * q[2] * q[3];
    let twoiw = 2.0 * q[1] * q[0];
    let twojw = 2.0 * q[2] * q[0];
    let twokw = 2.0 * q[3] * q[0];
    Mat3::from_rows(
        [w2 + i2 - j2 - k2, twoij - twokw, twojw + twoik],
        [twoij + twokw, w2 - i2 + j2 - k2, twojk - twoiw],
        [twoik - twojw, twojk + twoiw, w2 - i2 - j2 + k2],
    )
}

/// Exact port of LAMMPS `Domain::general_to_restricted_rotation`
/// (`domain.cpp:600`).
///
/// Also used to rotate *atom coordinates* from the general frame into the
/// restricted frame, mirroring `Domain::general_to_restricted_coords`.
///
/// Returns `(rotmat_g2r, a', b', c')` where `rotmat_g2r` maps general
/// coordinates onto the restricted frame and `a'`, `b'`, `c'` are the
/// rotated edge vectors (lower-triangular).
pub fn general_to_restricted_rotation(
    a: Coord3,
    b: Coord3,
    c: Coord3,
) -> Result<(Mat3, Coord3, Coord3, Coord3)> {
    // A x B must point along C (right-handed) and the vectors must not be
    // co-planar - LAMMPS errors out otherwise.
    let abcross = cross3(a, b);
    let dot = dot3(abcross, c);
    if dot == 0.0 {
        return Err(Error::invalid(
            "general triclinic edge vectors are co-planar",
        ));
    }
    if dot < 0.0 {
        return Err(Error::invalid(
            "general triclinic edge vectors must be right-handed",
        ));
    }

    // quat1 rotates A onto the +x axis
    let xaxis: Coord3 = [1.0, 0.0, 0.0];
    let alen = len3(a);
    let mut rot1 = cross3(a, xaxis);
    norm3(&mut rot1);
    let theta1 = (a[0] / alen).acos();
    let quat1 = axisangle_to_quat(rot1, theta1);
    let rotmat1 = quat_to_mat(quat1);
    let b1 = rotmat1.mul_v(b);

    // quat2 rotates B1 into the xy plane
    let mut byzvec1 = b1;
    byzvec1[0] = 0.0;
    let byzvec1_len = len3(byzvec1);
    let mut theta2 = (b1[1] / byzvec1_len).acos();
    if b1[2] > 0.0 {
        theta2 = -theta2;
    }
    let quat2 = axisangle_to_quat(xaxis, theta2);

    let quat_single = quatquat(quat2, quat1);
    let rot = quat_to_mat(quat_single);
    Ok((rot, rot.mul_v(a), rot.mul_v(b), rot.mul_v(c)))
}

/// Map a point from the general frame into the restricted frame, rotating
/// about `origin` exactly like LAMMPS `Domain::general_to_restricted_coords`.
///
/// `x_new = rot·(x - origin) + origin`
pub fn general_to_restricted_point(rot: &Mat3, origin: Coord3, x: Coord3) -> Coord3 {
    let s = [x[0] - origin[0], x[1] - origin[1], x[2] - origin[2]];
    let r = rot.mul_v(s);
    [r[0] + origin[0], r[1] + origin[1], r[2] + origin[2]]
}

/// The unit cell: lattice-vector matrix (rows), origin, and periodicity.
#[derive(Debug, Clone, Copy)]
pub struct UnitCell {
    /// Rows are lattice vectors a,b,c (restricted triclinic, LAMMPS convention).
    h: Mat3,
    /// Periodicity flags per dimension.
    pub pbc: [bool; 3],
    /// Origin (boxlo) — `lo` corner, the design's `boxlo[3]`.
    pub origin: Coord3,
}

impl fmt::Display for UnitCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a={:?} b={:?} c={:?} det={}",
            self.h.row(0),
            self.h.row(1),
            self.h.row(2),
            self.h.det()
        )
    }
}

impl UnitCell {
    /// Orthorhombic cell from per-dimension lengths, origin at zero, all-periodic.
    pub fn orthorhombic(lx: f64, ly: f64, lz: f64) -> Self {
        UnitCell {
            h: Mat3::from_rows([lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]),
            pbc: [true, true, true],
            origin: [0.0, 0.0, 0.0],
        }
    }

    /// Restricted triclinic cell from LAMMPS `h[6] = (lx, ly, lz, yz, xz, xy)`.
    pub fn from_h(h: [f64; 6]) -> Self {
        UnitCell {
            h: Mat3::from_rows([h[0], 0.0, 0.0], [h[5], h[1], 0.0], [h[4], h[3], h[2]]),
            pbc: [true; 3],
            origin: [0.0, 0.0, 0.0],
        }
    }

    /// General (non-restricted) triclinic cell, as LAMMPS writes in
    /// `avec`/`bvec`/`cvec` data files and `BOX BOUNDS abc origin` dumps.
    ///
    /// LAMMPS internally rotates the three arbitrary edge vectors onto the
    /// lower-triangular restricted form; this is an exact port of
    /// `Domain::define_general_triclinic` + `general_to_restricted_rotation`
    /// (`domain.cpp`) so that wrapped coordinates and MIC agree with LAMMPS.
    /// The returned cell's `origin` is the `abc origin` triple; coordinates in
    /// such files are absolute Cartesian, which `UnitCell` already assumes.
    pub fn from_general_triclinic(
        avec: Coord3,
        bvec: Coord3,
        cvec: Coord3,
        origin: Coord3,
    ) -> Result<Self> {
        Self::from_general_triclinic_rot(avec, bvec, cvec, origin).map(|(cell, _)| cell)
    }

    /// As [`UnitCell::from_general_triclinic`], but also returns the
    /// general→restricted rotation so that atom coordinates read from the same
    /// file can be mapped with [`general_to_restricted_point`].
    ///
    /// Pairing the two is mandatory: rotating the cell but not the atoms would
    /// silently misplace every particle in a general-triclinic file.
    pub fn from_general_triclinic_rot(
        avec: Coord3,
        bvec: Coord3,
        cvec: Coord3,
        origin: Coord3,
    ) -> Result<(Self, Mat3)> {
        let (rot, aprime, bprime, cprime) = general_to_restricted_rotation(avec, bvec, cvec)?;
        // The rotation lands A on +x and B in the xy plane, so a' = (lx,0,0)
        // and b' = (xy,ly,0) up to FP dust, while c' keeps its two tilt
        // components: c' = (xz,yz,lz) is exactly LAMMPS' third restricted row.
        let h = Mat3::from_rows([aprime[0], 0.0, 0.0], [bprime[0], bprime[1], 0.0], cprime);
        Ok((
            UnitCell {
                h,
                pbc: [true; 3],
                origin,
            },
            rot,
        ))
    }

    /// A general-triclinic cell exactly as given (`ReadOptions::keep_general`):
    /// rows `a b c`, no rotation. Validated like
    /// [`UnitCell::from_general_triclinic`] (a right-handed, non-degenerate
    /// cell), so it is the rotated cell's twin.
    pub fn general_as_given(
        avec: Coord3,
        bvec: Coord3,
        cvec: Coord3,
        origin: Coord3,
    ) -> Result<Self> {
        general_to_restricted_rotation(avec, bvec, cvec)?;
        Ok(UnitCell {
            h: Mat3::from_rows(avec, bvec, cvec),
            pbc: [true; 3],
            origin,
        })
    }

    /// Rotation matrix taking general-triclinic coordinates into the
    /// restricted frame (the inverse of `Domain::restricted_to_general_coords`).
    ///
    /// Needed when a *general* triclinic file's coordinates must be mapped into
    /// this restricted cell.
    pub fn general_to_restricted_rotation(&self, a: Coord3, b: Coord3, c: Coord3) -> Result<Mat3> {
        let (rot, _, _, _) = general_to_restricted_rotation(a, b, c)?;
        Ok(rot)
    }

    /// From box bounds rows as LAMMPS dump writes them:
    /// `(xlo, xhi, xy)`, `(ylo, yhi, xz)`, `(zlo, zhi, yz)`.
    // The nine scalars mirror the wire order of the `ITEM: BOX` line; folding
    // them into a struct would hide the correspondence they encode.
    #[allow(clippy::too_many_arguments)]
    pub fn from_box_bounds(
        xlo: f64,
        xhi: f64,
        xy: f64,
        ylo: f64,
        yhi: f64,
        xz: f64,
        zlo: f64,
        zhi: f64,
        yz: f64,
    ) -> Self {
        let mut c = UnitCell {
            h: Mat3::from_rows(
                [xhi - xlo, 0.0, 0.0],
                [xy, yhi - ylo, 0.0],
                [xz, yz, zhi - zlo],
            ),
            pbc: [true, true, true],
            origin: [xlo, ylo, zlo],
        };
        c.orthonormalize_triclinic();
        c
    }

    /// General triclinic from a full 3×3 matrix (rows = vectors), e.g. from
    /// `ITEM: BOX BOUNDS abc origin`; origin shift by `origin`.
    pub fn from_matrix(m: Mat3, pbc: [bool; 3], origin: Coord3) -> Self {
        UnitCell { h: m, pbc, origin }
    }

    /// The h-matrix (rows = a,b,c).
    pub fn h_matrix(&self) -> &Mat3 {
        &self.h
    }

    pub fn set_pbc(&mut self, pbc: [bool; 3]) {
        self.pbc = pbc;
    }

    /// Builder-style PBC override.
    pub fn with_pbc(mut self, pbc: [bool; 3]) -> Self {
        self.pbc = pbc;
        self
    }

    pub fn lengths(&self) -> Coord3 {
        let m = self.h.m;
        [
            (m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]).sqrt(),
            (m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2]).sqrt(),
            (m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]).sqrt(),
        ]
    }

    /// Cartesian displacement of the periodic image `shift = (nx, ny, nz)`,
    /// i.e. `nx·a + ny·b + nz·c`.
    ///
    /// Used by the neighbour search to enumerate every image of a candidate
    /// atom that can fall within the cutoff (which matters when the cutoff
    /// exceeds half the cell width, LAMMPS "cutghost" territory).
    #[inline]
    pub fn shift_vector(&self, shift: [i64; 3]) -> Coord3 {
        let (nx, ny, nz) = (shift[0] as f64, shift[1] as f64, shift[2] as f64);
        let m = self.h.m;
        [
            m[0][0] * nx + m[1][0] * ny + m[2][0] * nz,
            m[0][1] * nx + m[1][1] * ny + m[2][1] * nz,
            m[0][2] * nx + m[1][2] * ny + m[2][2] * nz,
        ]
    }

    /// Perpendicular widths of the cell along each axis:
    /// `[V/|b×c|, V/|a×c|, V/|a×b|]` (LAMMPS `h_inv` scaling).
    pub fn heights(&self) -> Coord3 {
        let m = self.h.m;
        let a = [m[0][0], m[0][1], m[0][2]];
        let b = [m[1][0], m[1][1], m[1][2]];
        let c = [m[2][0], m[2][1], m[2][2]];
        let area = |u: Coord3, v: Coord3| -> f64 {
            let cr = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt()
        };
        let vol = self.volume();
        [vol / area(b, c), vol / area(a, c), vol / area(a, b)]
    }

    pub fn periods(&self) -> Coord3 {
        let l = self.lengths();
        [l[0], l[1], l[2]]
    }

    /// The cell's eight corners `origin + i·a + j·b + k·c`, `i, j, k ∈ {0, 1}`,
    /// in the order `(i, j, k)` = (0,0,0), (0,0,1), (0,1,0), …, (1,1,1).
    ///
    /// The (0,0,0) corner is `origin` itself.
    pub fn corners(&self) -> [Coord3; 8] {
        let m = self.h.m;
        let mut out = [self.origin; 8];
        for (n, c) in out.iter_mut().enumerate().skip(1) {
            let (i, j, k) = ((n >> 2) & 1, (n >> 1) & 1, n & 1);
            let (fi, fj, fk) = (i as f64, j as f64, k as f64);
            for (d, cd) in c.iter_mut().enumerate() {
                *cd = self.origin[d] + (fi * m[0][d] + fj * m[1][d] + fk * m[2][d]);
            }
        }
        out
    }

    /// Cartesian bounding box `(lo, hi)` of the cell: the per-axis extremes of
    /// its eight [corners](Self::corners).
    ///
    /// For a restricted triclinic cell this is LAMMPS' `boxlo_bound` /
    /// `boxhi_bound` (`domain.cpp`, `dump_custom.cpp`):
    /// ```text
    ///   x: xlo + min(0, xy, xz, xy+xz)  ..  xhi + max(0, xy, xz, xy+xz)
    ///   y: ylo + min(0, yz)             ..  yhi + max(0, yz)
    ///   z: zlo                          ..  zhi
    /// ```
    /// and for any other cell the box around its corners. On an orthorhombic
    /// cell it is `(origin, origin + (lx, ly, lz))`. (`(origin, origin + a + b + c)`
    /// would be neither the bounding box nor the cell on a tilted one.)
    pub fn bounds(&self) -> (Coord3, Coord3) {
        let m = self.h.m;
        let mut lo_off = [0.0f64; 3];
        let mut hi_off = [0.0f64; 3];
        for n in 1..8usize {
            let (fi, fj, fk) = (((n >> 2) & 1) as f64, ((n >> 1) & 1) as f64, (n & 1) as f64);
            for d in 0..3 {
                let off = fi * m[0][d] + fj * m[1][d] + fk * m[2][d];
                lo_off[d] = lo_off[d].min(off);
                hi_off[d] = hi_off[d].max(off);
            }
        }
        let o = self.origin;
        let mut lo = o;
        let mut hi = o;
        for d in 0..3 {
            if lo_off[d] < 0.0 {
                lo[d] = o[d] + lo_off[d];
            }
            hi[d] = o[d] + hi_off[d];
        }
        (lo, hi)
    }

    pub fn volume(&self) -> f64 {
        self.h.det().abs()
    }

    /// Minimum-image displacement plus the integer image index that produced it.
    ///
    /// `base == raw + shift_vector(index)`, i.e. `base` is the displacement from
    /// `p_i` to the image `p_j + index·H` that is closest. Callers that need the
    /// other images (cutoff larger than half a cell width) start from `base` and
    /// add further multiples of the cell vectors.
    pub fn min_image_with_shift(&self, raw: Coord3) -> Result<(Coord3, [i64; 3])> {
        Ok(self.min_image_with_shift_pre(raw, &self.inverse_transposed()?))
    }

    /// [`Self::min_image_with_shift`] with the inverse supplied by the caller.
    ///
    /// The cell matrix is constant for a whole run, so a caller sweeping
    /// candidates must not re-invert it per candidate; this is infallible
    /// because the caller already proved the matrix invertible — and where the
    /// general path (which can fail on a degenerate lattice) declines, it
    /// falls back to the naive image rather than an error.
    ///
    /// Dispatch follows design §6.3 and ASE's `find_mic`:
    ///
    /// 1. no periodic axis → the displacement untouched;
    /// 2. orthogonal cell → per-axis rounding (exact for a diagonal metric,
    ///    so it needs neither a guard nor the general path);
    /// 3. otherwise round in fractional λ-space **on the periodic axes only**
    ///    (an image must never shift a free surface), and trust that naive
    ///    image while it stays below half the shortest periodic-lattice
    ///    vector — Smith's criterion with the sound lower bound of
    ///    [`Self::min_periodic_vector_lower_bound`] (ASE checks
    ///    `0.5*min(lengths)`, which a short row combination can defeat);
    /// 4. anything longer → the general path: Minkowski-reduce the periodic
    ///    lattice, wrap its dual coordinates, and test the Voronoi-relevant
    ///    images.
    ///
    /// Step 4 is what makes severely skewed boxes correct. Fractional rounding
    /// alone is *not* the nearest lattice vector there: on the restricted cell
    /// `a=(10,0,0), b=(5,10,0)` the displacement `(7.35,4.9,0)` rounds to
    /// itself (8.83 Å) while the image across the `a` boundary is 5.57 Å —
    /// see `min_image_general_path_beats_naive_rounding`. On a *partially*
    /// periodic cell the wrap in step 4 uses the periodic sublattice's dual
    /// coordinates; plain λ wrapping leaves the free coordinate unbounded and
    /// can miss the true image (ASE's `general_find_mic` does — see
    /// [`Self::general_min_image`]).
    ///
    /// `raw` is a displacement, so the origin must not come into it: this
    /// multiplies by `H⁻ᵀ` directly rather than calling `x2lamda_pre`, which
    /// takes a point and subtracts the origin. That would choose the image
    /// index from `H⁻ᵀ(raw − origin)` — invisible at origin zero, and still a
    /// valid image of the pair, but frequently a distant one.
    #[inline]
    pub fn min_image_with_shift_pre(&self, raw: Coord3, inv_t: &Mat3) -> (Coord3, [i64; 3]) {
        if self.pbc == [false, false, false] {
            return (raw, [0, 0, 0]);
        }
        if self.is_orthogonal() {
            return self.naive_orthogonal(raw);
        }
        let lam = inv_t.mul_v(raw);
        let mut index = [0i64; 3];
        for d in 0..3 {
            if self.pbc[d] {
                index[d] = -(lam[d].round() as i64);
            }
        }
        let d = self.shift_vector(index);
        let naive = [raw[0] + d[0], raw[1] + d[1], raw[2] + d[2]];
        if self.naive_mic_is_safe(&naive) {
            return (naive, index);
        }
        self.general_min_image(raw).unwrap_or((naive, index))
    }

    /// Per-axis rounding for an orthogonal cell — the cheap path of design
    /// §6.3. Rounds half away from zero (`f64::round`).
    #[inline]
    fn naive_orthogonal(&self, raw: Coord3) -> (Coord3, [i64; 3]) {
        let mut out = raw;
        let mut index = [0i64; 3];
        for d in 0..3 {
            let period = self.h.m[d][d];
            if self.pbc[d] && period != 0.0 {
                let n = -(out[d] / period).round() as i64;
                index[d] = n;
                out[d] += n as f64 * period;
            }
        }
        (out, index)
    }

    /// Smith's criterion: the naive image is the true minimum image while it
    /// stays shorter than half the shortest nonzero vector of the periodic
    /// sublattice. The bound must be a *lower* bound on that shortest vector
    /// or the criterion silently over-accepts: ASE checks
    /// `0.5 * min(cell.lengths())`, which is only the shortest vector when no
    /// row combination (say `b − c`) is shorter than every row — false on
    /// strongly sheared cells. [`Self::min_periodic_vector_lower_bound`] keeps
    /// the same value on orthogonal cells (where ASE is exact) but stays sound
    /// everywhere, so a failed bound always falls through to the general path
    /// rather than returning a long "image".
    #[inline]
    fn naive_mic_is_safe(&self, naive: &Coord3) -> bool {
        let min_l1 = match self.min_periodic_vector_lower_bound() {
            Some(l) => l,
            None => return false,
        };
        let n2 = naive[0] * naive[0] + naive[1] * naive[1] + naive[2] * naive[2];
        n2 < 0.25 * min_l1 * min_l1
    }

    /// A sound lower bound `L ≤ λ₁`, where `λ₁` is the shortest nonzero
    /// vector of the lattice spanned by the periodic rows. Pack the periodic
    /// rows as a rank-`d` basis; successive minima satisfy `λ_d ≤ M` with `M`
    /// the longest basis row, so with `V` the covolume in the span
    /// (`λ₁···λ_d = V`) this gives `λ₁ ≥ V / M^(d−1)`. Rank 1 is exact. On an
    /// orthogonal cell `V = ∏|rows|` and `M = max|rows|`, so the bound is the
    /// shortest row — identical to ASE's threshold there, stricter only where
    /// ASE would be wrong.
    fn min_periodic_vector_lower_bound(&self) -> Option<f64> {
        let mut per = [[0.0f64; 3]; 3];
        let mut d = 0usize;
        for i in 0..3 {
            if self.pbc[i] {
                per[d] = self.h.m[i];
                d += 1;
            }
        }
        let mut m2 = 0.0f64;
        for &r in per[..d].iter() {
            let l2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            if l2 > m2 {
                m2 = l2;
            }
        }
        let m = m2.sqrt();
        match d {
            0 => None,
            1 => {
                let r = per[0];
                Some((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt())
            }
            2 => {
                let a = per[0];
                let b = per[1];
                let c = [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ];
                let area = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if m > 0.0 {
                    Some(area / m)
                } else {
                    Some(0.0)
                }
            }
            3 => {
                if m > 0.0 {
                    Some(self.h.det().abs() / (m * m))
                } else {
                    Some(0.0)
                }
            }
            _ => None,
        }
    }

    /// General minimum image (design §6.3): Minkowski-reduce the periodic
    /// lattice, wrap the *dual* coordinates of the periodic sublattice into
    /// `[0,1)`, then take the shortest of the Voronoi-relevant images
    /// `{-1,0,1}^d` (the zero image is tested first, like ASE's prepended
    /// `(0,0,0)`, so ties keep the wrapped candidate). Returns `None` when
    /// the reduction or the dual solve fails, i.e. on a degenerate lattice.
    ///
    /// The dual wrap matters only when some axis is non-periodic. Plain λ
    /// wrapping (fractional coordinates of the full cell, the way ASE's
    /// `general_find_mic` wraps via `wrap_positions`) leaves the free
    /// coordinate unbounded, and that free coordinate can tilt the answer
    /// far enough that the nearest periodic image sits two cells away from
    /// the wrapped point — outside the `{-1,0,1}` walk. ASE returns a
    /// suboptimal image in exactly this way: on the cell
    /// `a=(-2.64,1.55,2.06), b=(-1.01,2.97,-0.34), c=(-0.55,-0.99,2.81)`
    /// with `pbc=[f,t,t]` and `dr=(1.22,4.88,-0.92)` it reports 11.72 Å
    /// where the image `dr − b + c` has length 8.60 Å. Wrapping the dual
    /// coordinates `β = G⁻¹·(raw·r_i)` of the periodic rows puts the target
    /// inside the sublattice's own fundamental region, where reduction
    /// guarantees the `±1` walk is enough. For a fully periodic cell `β` is
    /// exactly `H⁻ᵀ·raw`, so that case is unchanged.
    fn general_min_image(&self, raw: Coord3) -> Option<(Coord3, [i64; 3])> {
        let (rcell, op) = minkowski::reduce(&self.h.m, &self.pbc).ok()?;
        let r = Mat3::from_rows(rcell[0], rcell[1], rcell[2]);
        // Dual coordinates of `raw` along the periodic rows — no origin
        // (see the doc on `min_image_with_shift_pre`).
        let beta = if self.pbc == [true, true, true] {
            r.transposed().solve(raw).ok()?
        } else {
            // Embed the periodic Gram system in a full 3×3 (free axes get
            // an identity row, right-hand side 0) so `Mat3::solve` handles
            // every rank; the free components of `beta` stay 0 and unused.
            let mut g = [[0.0f64; 3]; 3];
            let mut u = [0.0f64; 3];
            for i in 0..3 {
                if !self.pbc[i] {
                    continue;
                }
                let ri = r.row(i);
                u[i] = ri[0] * raw[0] + ri[1] * raw[1] + ri[2] * raw[2];
                for j in 0..3 {
                    if self.pbc[j] {
                        let rj = r.row(j);
                        g[i][j] = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2];
                    }
                }
            }
            for (k, &p) in self.pbc.iter().enumerate() {
                if !p {
                    g[k][k] = 1.0;
                }
            }
            Mat3::from_rows(g[0], g[1], g[2]).solve(u).ok()?
        };
        let mut base = raw;
        let mut s = [0i64; 3];
        for d in 0..3 {
            if self.pbc[d] {
                let fl = beta[d].floor();
                s[d] = -(fl as i64);
                base[0] -= fl * rcell[d][0];
                base[1] -= fl * rcell[d][1];
                base[2] -= fl * rcell[d][2];
            }
        }
        let ranges: [[i64; 3]; 3] = [
            Self::image_range(self.pbc[0]),
            Self::image_range(self.pbc[1]),
            Self::image_range(self.pbc[2]),
        ];
        let mut best = base;
        let mut best_s = s;
        let mut best_d2 = base[0] * base[0] + base[1] * base[1] + base[2] * base[2];
        for &k0 in &ranges[0] {
            for &k1 in &ranges[1] {
                for &k2 in &ranges[2] {
                    let cand = [
                        base[0]
                            + k0 as f64 * rcell[0][0]
                            + k1 as f64 * rcell[1][0]
                            + k2 as f64 * rcell[2][0],
                        base[1]
                            + k0 as f64 * rcell[0][1]
                            + k1 as f64 * rcell[1][1]
                            + k2 as f64 * rcell[2][1],
                        base[2]
                            + k0 as f64 * rcell[0][2]
                            + k1 as f64 * rcell[1][2]
                            + k2 as f64 * rcell[2][2],
                    ];
                    let d2 = cand[0] * cand[0] + cand[1] * cand[1] + cand[2] * cand[2];
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = cand;
                        best_s = [s[0] + k0, s[1] + k1, s[2] + k2];
                    }
                }
            }
        }
        // The reduced shift is an integer combination of the reduced rows;
        // map it back through `op` so the caller's `shift_vector(index)` —
        // which works in the *original* basis — reproduces `best` exactly.
        let index = [
            best_s[0] * op[0][0] + best_s[1] * op[1][0] + best_s[2] * op[2][0],
            best_s[0] * op[0][1] + best_s[1] * op[1][1] + best_s[2] * op[2][1],
            best_s[0] * op[0][2] + best_s[1] * op[1][2] + best_s[2] * op[2][2],
        ];
        Some((best, index))
    }

    /// `{-1,0,1}` on a periodic axis, `[0]` on a free one, so the candidate
    /// walk can never step through a non-periodic boundary.
    #[inline]
    fn image_range(periodic: bool) -> [i64; 3] {
        if periodic {
            [-1, 0, 1]
        } else {
            [0, 0, 0]
        }
    }

    /// LAMMPS' `lamda2x` — fractional → Cartesian (includes origin).
    ///
    /// `p = λx·a + λy·b + λz·c` with `a, b, c` the *rows* of `h`, so component `d` is
    /// `Σᵢ h[i][d]·λ[i]`. All nine terms, the same sum [`Self::shift_vector`] forms for an
    /// integer shift — the two must agree, since one is the other's generalisation.
    ///
    /// LAMMPS assembles only the terms a restricted triclinic cell needs, zeroing `h[0][1]`,
    /// `h[0][2]` and `h[1][2]`. On a rotated cell that projects onto a lower-triangular
    /// approximation of a cell that is not one, which moves every wrapped position, every bin
    /// assignment and every image shift — far enough that the cell list reports a perfect FCC
    /// crystal's twelve first-shell neighbours as five.
    ///
    /// [`Self::x2lamda`] is the exact inverse of the *full* matrix, so the pair must agree:
    /// `x2lamda(lamda2x(λ)) = λ` on any cell, including one LAMMPS would call general
    /// triclinic. `rotated_cell_survives_a_lamda_round_trip` and `lamda2x_maps_each_axis_onto_its
    /// _lattice_vector` are the assertions that keep them agreeing.
    #[inline]
    pub fn lamda2x(&self, lamda: Coord3) -> Coord3 {
        let h = self.h.m;
        [
            h[0][0] * lamda[0] + h[1][0] * lamda[1] + h[2][0] * lamda[2] + self.origin[0],
            h[0][1] * lamda[0] + h[1][1] * lamda[1] + h[2][1] * lamda[2] + self.origin[1],
            h[0][2] * lamda[0] + h[1][2] * lamda[1] + h[2][2] * lamda[2] + self.origin[2],
        ]
    }

    /// LAMMPS' `x2lamda` — Cartesian → fractional.
    ///
    /// `lamda2x` assembles `p = λx·a + λy·b + λz·c`, i.e. in matrix terms
    /// `p = Hᵀ·λ` (the lattice vectors are the ROWS of H). The inverse is
    /// therefore `λ = (Hᵀ)⁻¹·(p − origin) = (H⁻¹)ᵀ·(p − origin)`; using
    /// `H⁻¹` directly would be the transpose and silently wrong for sheared
    /// cells.
    pub fn x2lamda(&self, x: Coord3) -> Result<Coord3> {
        Ok(self.x2lamda_pre(x, &self.inverse_transposed()?))
    }

    /// [`Self::x2lamda`] with `(H⁻¹)ᵀ` supplied by the caller.
    ///
    /// Hoisting the inverse out of a candidate loop keeps a neighbour search from
    /// redoing the cofactor expansion per neighbour.
    #[inline]
    pub fn x2lamda_pre(&self, x: Coord3, inv_t: &Mat3) -> Coord3 {
        let delta = [
            x[0] - self.origin[0],
            x[1] - self.origin[1],
            x[2] - self.origin[2],
        ];
        inv_t.mul_v(delta)
    }

    /// `(H⁻¹)ᵀ` — the transform [`Self::x2lamda_pre`] consumes.
    pub fn inverse_transposed(&self) -> Result<Mat3> {
        Ok(self.inverse_matrix()?.transposed())
    }

    /// Inverse of the h-matrix (general 3×3).
    pub fn inverse_matrix(&self) -> Result<Mat3> {
        let m = self.h.m;
        let d = self.h.det();
        if d.abs() < 1e-300 {
            return Err(Error::InvalidData("singular cell matrix".into()));
        }
        let inv_d = 1.0 / d;
        let co = |i: usize, j: usize| -> f64 {
            // cofactor of m[i][j]
            let (r0, r1) = match i {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let (c0, c1) = match j {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let s = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            s * (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0])
        };
        // inv[j][i] = cofactor_{i,j} / d  (adjugate transpose)
        let mut r = [[0.0; 3]; 3];
        for (j, row) in r.iter_mut().enumerate() {
            for (i, slot) in row.iter_mut().enumerate() {
                *slot = co(i, j) * inv_d;
            }
        }
        Ok(Mat3 { m: r })
    }

    /// Wrap a Cartesian coordinate into the periodic box: periodic dimensions land in
    /// `[lo, lo + period)`, non-periodic ones are returned untouched.
    ///
    /// Both paths fold by *any* number of box lengths, not one.
    ///
    /// A single correction — `x < lo → += period`, `x >= hi → −= period` on the Cartesian axes,
    /// or one `±1` step on the fractional ones — is enough only for a coordinate already within
    /// one box. Callers hand it more: `read_pos` wraps LAMMPS' *unwrapped* `xu,yu,zu` columns, and
    /// a trajectory that has diffused across the box puts those coordinates many box lengths
    /// outside; `LatticeSites` wraps the positions of a supercell that is several unit cells
    /// across. A single fold would leave them outside the box — sometimes by a whole box — and
    /// every later stage would measure positions that are not the ones in the file.
    ///
    /// Example: rotating a 16 Å FCC block by 0.35 rad throws sites up to 20 Å out, and a single
    /// fold leaves them at 48 Å where they belong at 0. The block stops being commensurate with
    /// itself — distinct sites land on top of each other, a perfect crystal's 2.828 Å first shell
    /// comes out as 0.0 Å — and the cell list, correctly for the positions it is given, reports
    /// five of twelve neighbours.
    pub fn wrap(&self, x: Coord3) -> Coord3 {
        let mut out = x;
        let lo = self.origin;
        // For triclinic, LAMMPS works in lamda space; we do the same.
        if self.is_triclinic() {
            // x2lamda, apply wrap on lamda, back
            if let Ok(lam) = self.x2lamda(x) {
                let mut l = lam;
                let pbc = self.pbc;
                let mut changed = false;
                for d in 0..3 {
                    if pbc[d] {
                        let f = l[d].rem_euclid(1.0);
                        // rem_euclid can round a hair-negative value up to exactly 1.0, which is
                        // the one value outside the half-open box.
                        let f = if f >= 1.0 { 0.0 } else { f };
                        if f != l[d] {
                            l[d] = f;
                            changed = true;
                        }
                    }
                }
                if changed {
                    out = self.lamda2x(l);
                }
            }
        } else {
            let period = self.periods();
            for d in 0..3 {
                if self.pbc[d] && period[d].is_finite() && period[d] > 0.0 {
                    let t = (out[d] - lo[d]).rem_euclid(period[d]);
                    out[d] = lo[d] + if t >= period[d] { 0.0 } else { t };
                }
            }
        }

        // guard against non-finite results silently: keep the source if broken
        if !out.iter().all(|v| v.is_finite()) {
            out = x;
        }
        out
    }

    /// Minimum-image vector from `dr` (u − v); picks the closest image.
    ///
    /// See [`Self::min_image_with_shift_pre`] for the dispatch — this is the
    /// same algorithm, minus the returned image index. Cheap path: per-axis
    /// rounding for orthogonal cells. General path (design §6.3): fractional
    /// λ-space rounding guarded by Smith's criterion, falling back to
    /// Minkowski reduction + nearest-lattice-vector search so that severely
    /// skewed boxes get the true minimum image (ASE `find_mic`).
    pub fn min_image(&self, dr: Coord3) -> Result<Coord3> {
        if self.pbc == [false, false, false] {
            return Ok(dr);
        }
        if self.is_orthogonal() {
            return Ok(self.naive_orthogonal(dr).0);
        }
        let inv_t = self.inverse_transposed()?;
        Ok(self.min_image_with_shift_pre(dr, &inv_t).0)
    }

    /// Unwrap a wrapped coordinate with its image flags (LAMMPS
    /// `compute property/atom` `xu` triclinic formula). Triclinic unwrap uses
    /// `dx = ix·xprd + iy·xy + iz·xz` etc.
    pub fn unwrap(&self, x: Coord3, image: MdaImage) -> Coord3 {
        let xb = image_ipart(image) as f64;
        let yb = image_jpart(image) as f64;
        let zb = image_kpart(image) as f64;
        let h = self.h.m;
        [
            x[0] + h[0][0] * xb + h[1][0] * yb + h[2][0] * zb,
            x[1] + h[1][1] * yb + h[2][1] * zb,
            x[2] + h[2][2] * zb,
        ]
    }

    /// True when any off-diagonal h entry is non-zero (sheared cell).
    pub fn is_triclinic(&self) -> bool {
        let m = self.h.m;
        (m[0][1] != 0.0)
            || (m[0][2] != 0.0)
            || (m[1][0] != 0.0)
            || (m[1][2] != 0.0)
            || (m[2][0] != 0.0)
            || (m[2][1] != 0.0)
    }

    /// No off-diagonal h entry: the cell takes the cheap per-axis MIC path (ASE's
    /// orthogonal case).
    pub fn is_orthogonal(&self) -> bool {
        !self.is_triclinic()
    }

    /// Whether this is a LAMMPS *restricted* cell: `a` along +x, `b` in the
    /// xy plane (the upper triangle of `h` zero) and a positive diagonal.
    /// Only such a cell has `xlo xhi / xy xz yz` box bounds; any other cell
    /// (a rotated extxyz `Lattice`, say) is general triclinic for LAMMPS.
    pub fn is_restricted(&self) -> bool {
        let m = self.h.m;
        m[0][1] == 0.0
            && m[0][2] == 0.0
            && m[1][2] == 0.0
            && m[0][0] > 0.0
            && m[1][1] > 0.0
            && m[2][2] > 0.0
    }

    /// The LAMMPS general-triclinic form `(avec, bvec, cvec, origin)` of a
    /// cell that is not [restricted](Self::is_restricted), as written in a
    /// data file's `avec bvec cvec abc origin` header and a dump's
    /// `BOX BOUNDS abc origin` rows. LAMMPS requires right-handed edge
    /// vectors, so a left-handed (or flat) cell is an error.
    pub fn lammps_general(&self) -> Result<(Coord3, Coord3, Coord3, Coord3)> {
        let m = self.h.m;
        if self.h.det() <= 0.0 {
            return Err(Error::invalid(format!(
                "the cell {self} is left-handed or flat, and LAMMPS has no form for it \
                 (general triclinic edge vectors must be right-handed)"
            )));
        }
        Ok((m[0], m[1], m[2], self.origin))
    }

    /// LAMMPS-style tilt reduction of a restricted cell.
    ///
    /// LAMMPS keeps `|xy| ≤ lx/2`, `|xz| ≤ lx/2`, `|yz| ≤ ly/2` by *flipping*
    /// the box to an equivalent lattice (`fix deform ... flip yes`,
    /// `fix_deform.cpp`): `c ± b` when `yz` is out of range (which moves `xz`
    /// by `∓xy`), then `c ± a` for `xz` and `b ± a` for `xy`. This applies
    /// that rule, with LAMMPS' comparisons (`tilt·(1/prd)` against ±0.5,
    /// strict), until every tilt is in range, so a tilt of any size is reduced.
    /// A flip along `a` needs x periodic and along `b` y periodic, as in
    /// LAMMPS; a tilt along a non-periodic axis is left as it is.
    ///
    /// Returns `None` when nothing changes (or the cell is not restricted);
    /// otherwise the reduced cell and the integer matrix `M` with
    /// `rows(h') = M · rows(h)` (unit lower-triangular, so `det M = 1`).
    pub fn reduce_tilts(&self) -> Option<(UnitCell, [[i64; 3]; 3])> {
        if !self.is_restricted() {
            return None;
        }
        let m = self.h.m;
        let (lx, ly) = (m[0][0], m[1][1]);
        let (xprdinv, yprdinv) = (1.0 / lx, 1.0 / ly);
        let (mut xy, mut xz, mut yz) = (m[1][0], m[2][0], m[2][1]);
        // b' = b + kb·a; c' = c + pc·b + qc·a.
        let (mut kb, mut pc, mut qc) = (0i64, 0i64, 0i64);
        // Each flip shrinks a tilt by a whole period, so this ends; the cap is
        // a guard against a non-finite cell.
        for _ in 0..1_000_000 {
            let mut changed = false;
            if self.pbc[1] {
                if yz * yprdinv < -0.5 {
                    yz += ly;
                    xz += xy;
                    pc += 1;
                    qc += kb;
                    changed = true;
                } else if yz * yprdinv > 0.5 {
                    yz -= ly;
                    xz -= xy;
                    pc -= 1;
                    qc -= kb;
                    changed = true;
                }
            }
            if self.pbc[0] {
                if xz * xprdinv < -0.5 {
                    xz += lx;
                    qc += 1;
                    changed = true;
                } else if xz * xprdinv > 0.5 {
                    xz -= lx;
                    qc -= 1;
                    changed = true;
                }
                if xy * xprdinv < -0.5 {
                    xy += lx;
                    kb += 1;
                    changed = true;
                } else if xy * xprdinv > 0.5 {
                    xy -= lx;
                    kb -= 1;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if (kb, pc, qc) == (0, 0, 0) {
            return None;
        }
        let mut cell = *self;
        cell.h = Mat3::from_rows([lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, m[2][2]]);
        Some((cell, [[1, 0, 0], [kb, 1, 0], [qc, pc, 1]]))
    }

    /// `BOX BOUNDS` rows the LAMMPS writer emits (with tilt where triclinic).
    /// Returns `(xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz, triclinic)`.
    pub fn to_box_bounds(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, bool) {
        let m = self.h.m;
        let triclinic = self.is_triclinic();
        (
            self.origin[0],
            self.origin[0] + m[0][0],
            m[1][0],
            self.origin[1],
            self.origin[1] + m[1][1],
            m[2][0],
            self.origin[2],
            self.origin[2] + m[2][2],
            m[2][1],
            triclinic,
        )
    }

    /// Normalise a triclinic cell built from box bounds: keep it restricted
    /// (zero out the two strictly-UPPER entries of the lamda matrix). LAMMPS
    /// guarantees invariants `|xy| ≤ xprd/2`, `|xz| ≤ xprd/2`, `|yz| ≤ yprd/2`
    /// after the usual conventions; we do not re-shear here, just zero the
    /// upper triangle so downstream code sees a clean lower-triangular matrix.
    fn orthonormalize_triclinic(&mut self) {
        self.h.m[0][1] = 0.0;
        self.h.m[0][2] = 0.0;
        self.h.m[1][2] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nearly(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// `bounds()` is LAMMPS' bounding box (`dump_custom.cpp`
    /// `header_item_triclinic`: `xlo + MIN(0,xy,xz,xy+xz)` …) for every sign
    /// of the tilts, and the plain box on an orthorhombic cell.
    #[test]
    fn bounds_are_the_lammps_bounding_box() {
        for &(xy, xz, yz) in &[
            (0.0, 0.0, 0.0),
            (1.5, -0.7, 0.4),
            (-2.0, 1.0, -1.25),
            (-1.0, -1.0, 2.0),
            (3.0, 2.0, 1.0),
        ] {
            let (xlo, ylo, zlo) = (-1.0, 2.0, 0.5);
            let (lx, ly, lz) = (10.0, 8.0, 6.0);
            let cell =
                UnitCell::from_box_bounds(xlo, xlo + lx, xy, ylo, ylo + ly, xz, zlo, zlo + lz, yz);
            let (lo, hi) = cell.bounds();
            let mn = |v: &[f64]| v.iter().copied().fold(0.0f64, f64::min);
            let mx = |v: &[f64]| v.iter().copied().fold(0.0f64, f64::max);
            let want_lo = [xlo + mn(&[xy, xz, xy + xz]), ylo + mn(&[yz]), zlo];
            let want_hi = [
                xlo + lx + mx(&[xy, xz, xy + xz]),
                ylo + ly + mx(&[yz]),
                zlo + lz,
            ];
            for d in 0..3 {
                assert!(nearly(lo[d], want_lo[d]), "{xy} {xz} {yz}: lo {lo:?}");
                assert!(nearly(hi[d], want_hi[d]), "{xy} {xz} {yz}: hi {hi:?}");
            }
            // Every corner lies in the box.
            for c in cell.corners() {
                for d in 0..3 {
                    assert!(c[d] >= lo[d] - 1e-12 && c[d] <= hi[d] + 1e-12);
                }
            }
        }
        // Orthorhombic: exactly origin .. origin + lengths.
        let mut c = UnitCell::orthorhombic(3.1, 4.7, 5.3);
        c.origin = [-0.3, 0.1, -7.9];
        let (lo, hi) = c.bounds();
        assert_eq!(lo, c.origin);
        assert_eq!(hi, [-0.3 + 3.1, 0.1 + 4.7, -7.9 + 5.3]);
    }

    /// A rotated cell is lower-triangular in nothing at all, so every off-diagonal term is live.
    ///
    /// Keeping only the terms LAMMPS needs for a *restricted* triclinic cell is invisible on an
    /// orthorhombic or restricted cell, where those terms are zero; on a general cell it drops a
    /// third of the matrix. `x2lamda` is the exact inverse of the full matrix, so a round trip is
    /// precisely the test that catches a truncated `lamda2x`.
    #[test]
    fn rotated_cell_survives_a_lamda_round_trip() {
        let ang: f64 = 0.35;
        let (c, sn) = (ang.cos(), ang.sin());
        let cell = UnitCell::from_matrix(
            Mat3::from_rows(
                [16.0 * c, 16.0 * sn, 0.0],
                [-16.0 * sn, 16.0 * c, 0.0],
                [0.0, 0.0, 16.0],
            ),
            [true; 3],
            [0.0; 3],
        );
        for lam in [
            [0.5, 0.5, 0.5],
            [0.123, -0.456, 0.789],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-0.25, 1.75, 0.5],
        ] {
            let x = cell.lamda2x(lam);
            let back = cell.x2lamda(x).unwrap();
            for d in 0..3 {
                assert!(
                    (back[d] - lam[d]).abs() < 1e-12,
                    "lamda {lam:?} -> x {x:?} -> {back:?}"
                );
            }
        }
    }

    /// `lamda2x([1,0,0])` is `a` by definition; a truncated assembly does not know that.
    #[test]
    fn lamda2x_maps_each_axis_onto_its_lattice_vector() {
        let cell = UnitCell::from_matrix(
            Mat3::from_rows([1.5, 0.25, -0.75], [-2.0, 3.5, 1.25], [0.5, -1.5, 4.5]),
            [true; 3],
            [0.0; 3],
        );
        for (d, lam) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            .iter()
            .enumerate()
        {
            let got = cell.lamda2x(*lam);
            // Rows are the lattice vectors, so the `d`th axis is the `d`th *row* of `h`.
            let want = cell.h.m[d];
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (*g - w).abs() < 1e-12,
                    "axis {d} component {i}: got {}, want {}",
                    g,
                    w
                );
            }
        }
    }

    /// `lamda2x` and `shift_vector` are the same sum with different arguments, so they must agree
    /// on the unit basis.
    #[test]
    /// `min_image` receives a *displacement*, so its fractional reduction must ignore the origin.
    ///
    /// The point-shaped pairing `x2lamda` → `lamda2x` happens to cancel an origin: subtracting it
    /// before the round and adding it back after leaves `dr` itself. A reduction built on it is
    /// right at origin zero and picks the wrong lattice image anywhere else, because it rounds
    /// `H⁻ᵀ(dr − origin)` to choose a shift that then has to undo a displacement it was never
    /// given. The cost is not a small error: the result stays a valid image of the pair, just a
    /// distant one.
    fn min_image_ignores_the_origin() {
        let theta: f64 = 0.35;
        let rot = Mat3::from_rows(
            [4.0, 0.0, 0.0],
            [4.0 * theta.cos(), 4.0 * theta.sin(), 0.0],
            [0.0, 0.0, 4.0],
        );
        let shift = UnitCell::from_matrix(rot, [true; 3], [17.0, -23.0, 41.0]);
        // A short displacement whose correct image is itself: no shift should be applied.
        let dr = [1.25, 0.5, -0.75];
        let at_origin = shift.min_image(dr).unwrap();
        let norm = at_origin.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            nearly(norm, dr.iter().map(|v| v * v).sum::<f64>().sqrt()),
            "minimum image of a short displacement must be itself, got {at_origin:?}"
        );
        // And the same displacement after a whole-box translation must reduce back to it.
        let whole = shift.shift_vector([1, -1, 2]);
        let shifted = [dr[0] + whole[0], dr[1] + whole[1], dr[2] + whole[2]];
        let back = shift.min_image(shifted).unwrap();
        assert!(
            back.iter().zip(dr.iter()).all(|(b, d)| nearly(*b, *d)),
            "one box away must reduce back, got {back:?} from {dr:?}"
        );
        // The same reduction on the path the cell list actually walks.
        let (reduced, image) = shift.min_image_with_shift(shifted).unwrap();
        assert!(
            reduced.iter().zip(dr.iter()).all(|(b, d)| nearly(*b, *d)),
            "min_image_with_shift must agree with min_image, got {reduced:?} for image {image:?}"
        );

        // Sanity: an origin at zero agrees, which is why only a shifted origin
        // can catch this.
        let zeroed = UnitCell::from_matrix(rot, [true; 3], [0.0, 0.0, 0.0]);
        let a = shift.min_image(shifted).unwrap();
        let b = zeroed.min_image(shifted).unwrap();
        assert!(a.iter().zip(b.iter()).all(|(x, y)| nearly(*x, *y)));
    }

    #[test]
    fn lamda2x_and_shift_vector_agree() {
        let cell = UnitCell::from_matrix(
            Mat3::from_rows([2.5, -1.0, 0.5], [0.75, 3.0, -2.0], [1.5, 0.25, 4.0]),
            [true; 3],
            [0.0; 3],
        );
        assert_eq!(cell.lamda2x([1.0, 0.0, 0.0]), cell.shift_vector([1, 0, 0]));
        assert_eq!(cell.lamda2x([0.0, 1.0, 0.0]), cell.shift_vector([0, 1, 0]));
        assert_eq!(cell.lamda2x([0.0, 0.0, 1.0]), cell.shift_vector([0, 0, 1]));
        // `shift_vector` only speaks integers, so compare it on an integer lambda and check the
        // general case against the definition instead.
        let lam = [2.0, -3.0, 0.5];
        let want = [
            lam[0] * cell.h.m[0][0] + lam[1] * cell.h.m[1][0] + lam[2] * cell.h.m[2][0],
            lam[0] * cell.h.m[0][1] + lam[1] * cell.h.m[1][1] + lam[2] * cell.h.m[2][1],
            lam[0] * cell.h.m[0][2] + lam[1] * cell.h.m[1][2] + lam[2] * cell.h.m[2][2],
        ];
        for (got, expect) in cell.lamda2x(lam).iter().zip(want.iter()) {
            assert!((*got - expect).abs() < 1e-12);
        }
    }

    #[test]
    fn ortho_lamda_roundtrip() {
        let c = UnitCell::orthorhombic(10.0, 20.0, 30.0);
        let x = [3.0, 4.0, 5.0];
        let l = c.x2lamda(x).unwrap();
        assert!(nearly(l[0], 0.3) && nearly(l[1], 0.2) && nearly(l[2], 1.0 / 6.0));
        let back = c.lamda2x(l);
        for d in 0..3 {
            assert!(nearly(back[d], x[d]));
        }
    }

    #[test]
    fn triclinic_lamda2x_port() {
        // LAMMPS h[6] = (lx, ly, lz, yz, xz, xy)
        let c = UnitCell::from_h([10.0, 20.0, 30.0, 1.5, 2.5, 3.5]);
        // LAMMPS lamda2x: x = h[0]*l0 + h[5]*l1 + h[4]*l2
        //              y = h[1]*l1 + h[3]*l2
        //              z = h[2]*l2
        let lam = [0.1, 0.2, 0.3];
        let x = c.lamda2x(lam);
        assert!(nearly(x[0], 10.0 * 0.1 + 3.5 * 0.2 + 2.5 * 0.3));
        assert!(nearly(x[1], 20.0 * 0.2 + 1.5 * 0.3));
        assert!(nearly(x[2], 30.0 * 0.3));
        // roundtrip
        let l2 = c.x2lamda(x).unwrap();
        assert!(nearly(l2[0], 0.1) && nearly(l2[1], 0.2) && nearly(l2[2], 0.3));
    }

    #[test]
    fn volume_of_shear_matches_det() {
        let c = UnitCell::from_h([10.0, 20.0, 30.0, 1.5, 2.5, 3.5]);
        // columns of lower-tri: rows a=(10,0,0) b=(3.5,20,0) c=(2.5,1.5,30)
        let vol = c.volume();
        let expected = 10.0 * 20.0 * 30.0;
        assert!(nearly(vol, expected), "got {vol} expected {expected}");
    }

    #[test]
    /// A single fold is only enough for a coordinate already within one box of it.
    ///
    /// These all land several box lengths out, which is the normal state of an unwrapped dump
    /// coordinate and of any supercell wider than the unit cell it was built from.
    fn wrap_folds_any_number_of_box_lengths() {
        let c = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        for k in -3..=3 {
            let raw = [0.5 + 10.0 * k as f64, 5.0, 0.0];
            let w = c.wrap(raw);
            assert!(nearly(w[0], 0.5), "{raw:?} wrapped to {w:?}, want x = 0.5");
            assert!(w[0] >= 0.0 && w[0] < 10.0, "{} not inside the box", w[0]);
        }
        // A shifted origin must be respected, not just the period.
        let shifted = UnitCell::from_matrix(*c.h_matrix(), [true; 3], [100.0, -7.0, 3.0]);
        let w = shifted.wrap([123.5, -30.0, 3.0]);
        // The box is [100, 110) x [-7, 3) x [3, 13); -30 is 2.3 boxes below -7.
        assert!(nearly(w[0], 103.5) && nearly(w[1], 0.0), "got {w:?}");

        // Same on the fractional path, the one a rotated cell takes.
        let theta: f64 = 0.35;
        let rot = Mat3::from_rows(
            [16.0 * theta.cos(), 16.0 * theta.sin(), 0.0],
            [-16.0 * theta.sin(), 16.0 * theta.cos(), 0.0],
            [0.0, 0.0, 16.0],
        );
        let t = UnitCell::from_matrix(rot, [true; 3], [0.0, 0.0, 0.0]);
        let inside = t.lamda2x([0.1, 0.2, 0.3]);
        let whole = t.shift_vector([3, -2, 1]);
        let far = [
            inside[0] + whole[0],
            inside[1] + whole[1],
            inside[2] + whole[2],
        ];
        let w = t.wrap(far);
        assert!(
            w.iter().zip(inside.iter()).all(|(a, b)| nearly(*a, *b)),
            "three box lengths out wrapped to {w:?}, want {inside:?}"
        );
        for l in t.x2lamda(w).unwrap() {
            assert!(
                (-1e-12..1.0).contains(&l),
                "fractional coordinate {l} escaped the box"
            );
        }
        // A point genuinely outside gets folded, not nudged.
        let outside = t.lamda2x([-0.25, 1.4, 0.5]);
        let w = t.wrap(outside);
        let want = t.lamda2x([0.75, 0.4, 0.5]);
        assert!(
            w.iter().zip(want.iter()).all(|(a, b)| nearly(*a, *b)),
            "{outside:?} wrapped to {w:?}, want {want:?}"
        );
    }

    #[test]
    fn wrap_ortho_lamps() {
        let c = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        // slightly negative -> wrap into box
        let w0 = c.wrap([-0.1, 5.0, 5.0]);
        assert!(nearly(w0[0], 9.9));
        // >= hi -> wrap down
        let w1 = c.wrap([10.0, 5.0, 5.0]);
        assert!(nearly(w1[0], 0.0));
        // 14.0 -> 4.0
        let w2 = c.wrap([14.0, 5.0, 5.0]);
        assert!(nearly(w2[0], 4.0));
        // stays put if inside
        let w3 = c.wrap([5.0, 5.0, 5.0]);
        assert!(nearly(w3[0], 5.0));
    }

    /// A free axis has no images, so a point outside the box along it stays
    /// outside.
    ///
    /// This is what the defect markers ride on. A cell that is periodic in two
    /// directions and free in the third is a slab, and a marker on the slab has
    /// no second image to be moved to: folding it into the box would put it
    /// somewhere the structure has no atom at all, and a viewer would be
    /// looking at a marker in the wrong place.
    #[test]
    fn a_free_axis_is_not_wrapped() {
        let mut c = UnitCell::from_box_bounds(0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0);
        c.pbc = [true, true, false];
        let w = c.wrap([3.0, 4.0, 25.0]);
        assert!(nearly(w[2], 25.0), "a free axis has no image: got {w:?}");
        // The periodic axes still wrap, so the point is not simply left alone.
        let w = c.wrap([13.0, 4.0, 25.0]);
        assert!(nearly(w[0], 3.0), "the periodic axes wrap: got {w:?}");
        assert!(nearly(w[2], 25.0), "the free axis does not: got {w:?}");
    }

    /// A triclinic cell wraps in fractional coordinates, and a box with a low
    /// corner of its own keeps it.
    ///
    /// The Cartesian arithmetic a marker would otherwise be put through --
    /// subtracting a whole number of axis lengths -- cannot express either of
    /// these: the axes of a triclinic cell are not perpendicular, and its low
    /// corner is not the origin.
    #[test]
    fn a_triclinic_cell_wraps_in_lambda_space_and_keeps_its_origin() {
        let mut c = UnitCell::from_h([10.0, 10.0, 10.0, 3.0, 1.0, 2.0]);
        c.origin = [1.0, -2.0, 3.0];
        let p = [14.5, 11.0, -6.0];
        let w = c.wrap(p);
        let lam = c.x2lamda(w).unwrap();
        for (d, &l) in lam.iter().enumerate() {
            assert!(
                (0.0..1.0).contains(&l),
                "lambda[{d}] = {l} is outside [0, 1) for {w:?}"
            );
        }
        // The wrapped point is the original plus a lattice translation, so the
        // distance between them is the length of that translation and nothing
        // else.
        let dr = c
            .min_image([w[0] - p[0], w[1] - p[1], w[2] - p[2]])
            .unwrap();
        let d: f64 = (0..3).map(|k| dr[k] * dr[k]).sum::<f64>().sqrt();
        assert!(
            d < 1e-9,
            "the wrap moved the point by {d} A, which is not a lattice vector"
        );
    }

    #[test]
    fn wrap_triclinic() {
        let c = UnitCell::from_h([10.0, 20.0, 30.0, 0.0, 0.0, 5.0]);
        // point beyond b's x-projection region but within lamda, wraps fine
        let _ = c.wrap([-1e-10, 0.0, 0.0]);
        // a reproducible sanity value
        let w = c.wrap([-0.005, 0.0, 0.0]);
        assert!(w[0] > 0.0 || w[0].abs() < 1e-6);
    }

    #[test]
    fn mic_ortho() {
        let c = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let dr = [9.0, 0.0, 0.0];
        let mic = c.min_image(dr).unwrap();
        assert!(nearly(mic[0], -1.0));
    }

    #[test]
    fn mic_nonperiodic_leaves() {
        let mut c = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        c.pbc = [false, true, true];
        let dr = [9.0, 0.5, 0.0];
        let mic = c.min_image(dr).unwrap();
        assert!(nearly(mic[0], 9.0));
    }

    #[test]
    fn image_pack_layout() {
        let img = pack_image(1, -2, 3);
        assert_eq!(image_ipart(img), 1);
        assert_eq!(image_jpart(img), -2);
        assert_eq!(image_kpart(img), 3);
    }

    #[test]
    fn unwrap_triclinic_formula() {
        let c = UnitCell::from_h([10.0, 20.0, 30.0, 1.5, 2.5, 3.5]);
        let x = [1.0, 2.0, 3.0];
        let img = pack_image(1, 2, 3);
        let u = c.unwrap(x, img);
        // ux = x + h0*1 + h5*2 + h4*3
        assert!(nearly(u[0], 1.0 + 10.0 * 1.0 + 3.5 * 2.0 + 2.5 * 3.0));
        assert!(nearly(u[1], 2.0 + 20.0 * 2.0 + 1.5 * 3.0));
        assert!(nearly(u[2], 3.0 + 30.0 * 3.0));
    }

    #[test]
    fn general_matrix_solve() {
        let m = Mat3::from_rows([1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]);
        let b = m.mul_v([1.0, 2.0, 3.0]);
        let x = m.solve(b).unwrap();
        assert!(nearly(x[0], 1.0) && nearly(x[1], 2.0) && nearly(x[2], 3.0));
    }

    /// Brute-force minimum-image length over `[-r, r]^3` image shifts — an
    /// oracle independent of the dispatch in `min_image_with_shift_pre`.
    /// Complete whenever the true image sits inside the box, which the
    /// property test below arranges via `|n_i| <= 2|dr|/height_i`.
    fn brute_min_len2(c: &UnitCell, dr: Coord3, r: i64) -> f64 {
        let mut best = f64::INFINITY;
        for ix in -r..=r {
            for iy in -r..=r {
                for iz in -r..=r {
                    let n = [ix, iy, iz];
                    if (0..3).any(|d| !c.pbc[d] && n[d] != 0) {
                        continue;
                    }
                    let s = c.shift_vector(n);
                    let v = [dr[0] + s[0], dr[1] + s[1], dr[2] + s[2]];
                    let l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    if l2 < best {
                        best = l2;
                    }
                }
            }
        }
        best
    }

    #[test]
    fn min_image_general_path_beats_naive_rounding() {
        // Restricted triclinic at LAMMPS' tilt limit (xy = xprd/2). The
        // displacement's fractional coordinates are (0.49, 0.49, 0), so naive
        // rounding keeps it — 8.83 Å, longer than the 5.57 Å image across the
        // a boundary. Smith's guard trips (> half the shortest vector, 5.0 Å)
        // and the Minkowski path must find the real minimum.
        let c = UnitCell::from_h([10.0, 10.0, 10.0, 0.0, 0.0, 5.0]);
        let dr = [7.35, 4.9, 0.0];
        let (mic, idx) = c.min_image_with_shift(dr).unwrap();
        let mic_len2 = mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2];
        let oracle = brute_min_len2(&c, dr, 3);
        assert!(
            (mic_len2 - oracle).abs() < 1e-9,
            "min_image {mic_len2} != brute force {oracle}"
        );
        assert!(oracle < 5.58 * 5.58 && oracle > 5.56 * 5.56, "{oracle}");
        // the returned pair keeps the documented invariant
        let s = c.shift_vector(idx);
        for d in 0..3 {
            assert!(
                nearly(mic[d], dr[d] + s[d]),
                "mic {mic:?} != raw {dr:?} + shift {s:?} (index {idx:?})"
            );
        }
        // and it really did beat what fractional rounding would have said
        let naive2 = 7.35 * 7.35 + 4.9 * 4.9;
        assert!(
            mic_len2 < naive2 * 0.5,
            "general path found {mic_len2} vs naive {naive2}"
        );
    }

    #[test]
    fn min_image_never_crosses_a_free_surface() {
        // In-plane skew forces the general path while pbc = [t, t, f]; the z
        // separation must come back untouched, and no image index may appear
        // on the free axis. Rounding every λ component would reach through the
        // surface.
        let c = UnitCell::from_matrix(
            Mat3::from_rows([10.0, 0.0, 0.0], [9.0, 10.0, 0.0], [0.0, 0.0, 50.0]),
            [true, true, false],
            [0.0; 3],
        );
        assert!(!c.is_orthogonal(), "the general path must be exercised");
        let dr = [0.5, 5.0, 47.0];
        let (mic, idx) = c.min_image_with_shift(dr).unwrap();
        assert_eq!(mic[2], dr[2], "free axis untouched: {mic:?}");
        assert_eq!(idx[2], 0, "no image along the free axis");
        let mic_len2 = mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2];
        let oracle = brute_min_len2(&c, dr, 3);
        assert!((mic_len2 - oracle).abs() < 1e-9, "{mic_len2} vs {oracle}");
    }

    /// Deterministic xorshift so the property test needs no dependency and
    /// fails the same way on every machine.
    struct Xor(u64);

    impl Xor {
        fn u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn f(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * (self.u64() as f64 / u64::MAX as f64)
        }
    }

    #[test]
    fn min_image_matches_brute_force_on_random_skewed_cells() {
        // The general path exists because fractional rounding lies on skewed
        // boxes; this pins both paths against an oracle that knows nothing
        // about Minkowski reduction: for random (often severely sheared)
        // cells and random displacements, `min_image` must return the
        // shortest image the lattice offers, and it must offer it as
        // `raw + shift_vector(index)`.
        let mut rng = Xor(0x2545_F491_4F6C_DD1D);
        let pbc_sets = [
            [true, true, true],
            [true, true, false],
            [true, true, true],
            [true, false, false],
            [false, true, true],
            [true, true, true],
        ];
        let mut checked = 0;
        let mut attempts = 0;
        while checked < 120 && attempts < 8000 {
            attempts += 1;
            let mut rows = [[0.0f64; 3]; 3];
            for row in rows.iter_mut() {
                for v in row.iter_mut() {
                    *v = rng.f(-3.0, 3.0);
                }
            }
            let pbc = pbc_sets[(attempts as usize) % pbc_sets.len()];
            let c =
                UnitCell::from_matrix(Mat3::from_rows(rows[0], rows[1], rows[2]), pbc, [0.0; 3]);
            let heights = c.heights();
            if c.volume() < 1.0 {
                continue;
            }
            if (0..3).any(|d| pbc[d] && (!heights[d].is_finite() || heights[d] < 1.0)) {
                continue;
            }
            let dr = [rng.f(-5.0, 5.0), rng.f(-5.0, 5.0), rng.f(-5.0, 5.0)];
            // Completeness bound: any image shorter than |dr| obeys
            // |n_i| <= 2|dr|/height_i; skip cases whose box gets silly.
            let dr_len = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt();
            let mut r = 1i64;
            let mut fits = true;
            for d in 0..3 {
                if !pbc[d] {
                    continue;
                }
                let nd = (2.0 * dr_len / heights[d]).ceil() as i64 + 1;
                if nd > 12 {
                    fits = false;
                    break;
                }
                r = r.max(nd);
            }
            if !fits {
                continue;
            }
            let (mic, idx) = c.min_image_with_shift(dr).unwrap();
            // the invariant that lets callers keep walking further images
            let s = c.shift_vector(idx);
            for d in 0..3 {
                assert!(
                    (mic[d] - (dr[d] + s[d])).abs() <= 1e-9 * (1.0 + dr[d].abs()),
                    "case {checked}: mic {mic:?} != raw {dr:?} + shift {s:?}"
                );
            }
            let mic_len2 = mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2];
            let oracle = brute_min_len2(&c, dr, r);
            assert!(
                mic_len2 <= oracle * (1.0 + 1e-9) + 1e-9,
                "case {checked}: min_image length² {mic_len2} exceeds oracle {oracle}\n\
                 cell pbc={pbc:?} h={:?} dr={dr:?} index={idx:?}",
                c.h_matrix().m
            );
            // the oracle must actually be a finite image (dr itself qualifies)
            assert!(oracle.is_finite());
            checked += 1;
        }
        assert!(checked >= 100, "only {checked} random cells were checked");
    }
}
