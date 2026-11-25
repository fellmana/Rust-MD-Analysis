//! Quaternion characteristic polynomial (QCP) fit: inner product, optimal
//! rotation and RMSD.
//!
//! Port of upstream `ptm_polar.cpp` (`InnerProduct`, `optimal_quaternion`,
//! `FastCalcRMSDAndRotation`) plus `quaternion_to_rotation_matrix` from
//! `ptm_quat.cpp`. Upstream credits the QCP method of Coutsias, Hartenstein and
//! Kinzel (the `qcprot.h` lineage recorded in the file header), adapted here for
//! a 3x3 matrix.
//!
//! The characteristic polynomial is solved by Newton-Raphson rather than a
//! closed-form quartic, and the optimal rotation comes from the adjugate of
//! `A - lambda*I` rather than an eigen-solve. Both are deliberate: the same
//! expressions, in the same order, give the same floating-point results as the
//! reference, which is what the OVITO differential test pins down.

/// Accumulate the covariance-like matrix `A = sum_i c1[i] (x) c2[perm[i]]`.
///
/// `coords2` is indexed through `permutation` rather than reordered in place,
/// because the caller wants to keep the template's natural vertex order.
pub fn inner_product<const N: usize>(
    coords1: &[[f64; 3]; N],
    coords2: &[[f64; 3]; N],
    permutation: &[usize; N],
    num_points: usize,
) -> [f64; 9] {
    let mut a = [0.0f64; 9];
    for i in 0..num_points {
        let x1 = coords1[i][0];
        let y1 = coords1[i][1];
        let z1 = coords1[i][2];

        let p = permutation[i];
        let x2 = coords2[p][0];
        let y2 = coords2[p][1];
        let z2 = coords2[p][2];

        a[0] += x1 * x2;
        a[1] += x1 * y2;
        a[2] += x1 * z2;

        a[3] += y1 * x2;
        a[4] += y1 * y2;
        a[5] += y1 * z2;

        a[6] += z1 * x2;
        a[7] += z1 * y2;
        a[8] += z1 * z2;
    }
    a
}

/// Solve for the largest eigenvalue and the corresponding unit quaternion.
///
/// `polar` selects the seed for the Newton iteration: `true` starts from the
/// Frobenius-norm bound (polar decomposition, where the eigenvalue is not known
/// in advance), `false` starts from `e0`, the already-known trace bound. Returns
/// `false` if the adjugate was numerically singular, in which case the caller
/// gets the identity rotation.
#[allow(clippy::needless_pass_by_value)]
fn optimal_quaternion(a_in: &[f64; 9], polar: bool, e0: f64) -> (bool, f64, [f64; 4]) {
    const EVEC_PREC: f64 = 1e-6;
    const EVAL_PREC: f64 = 1e-11;

    let sxx = a_in[0];
    let sxy = a_in[1];
    let sxz = a_in[2];
    let syx = a_in[3];
    let syy = a_in[4];
    let syz = a_in[5];
    let szx = a_in[6];
    let szy = a_in[7];
    let szz = a_in[8];

    let sxx2 = sxx * sxx;
    let syy2 = syy * syy;
    let szz2 = szz * szz;
    let sxy2 = sxy * sxy;
    let syz2 = syz * syz;
    let sxz2 = sxz * sxz;
    let syx2 = syx * syx;
    let szy2 = szy * szy;
    let szx2 = szx * szx;

    let fnorm_squared = sxx2 + syy2 + szz2 + sxy2 + syz2 + sxz2 + syx2 + szy2 + szx2;

    let syz_szy_minus_syy_szz2 = 2.0 * (syz * szy - syy * szz);
    let sxx2_syy2_szz2_syz2_szy2 = syy2 + szz2 - sxx2 + syz2 + szy2;
    let sxz_p_szx = sxz + szx;
    let syz_p_szy = syz + szy;
    let sxy_p_syx = sxy + syx;
    let syz_m_szy = syz - szy;
    let sxz_m_szx = sxz - szx;
    let sxy_m_syx = sxy - syx;
    let sxx_p_syy = sxx + syy;
    let sxx_m_syy = sxx - syy;
    let sxy2_sxz2_syx2_szx2 = sxy2 + sxz2 - syx2 - szx2;

    let mut c = [0.0f64; 3];
    c[0] = sxy2_sxz2_syx2_szx2 * sxy2_sxz2_syx2_szx2
        + (sxx2_syy2_szz2_syz2_szy2 + syz_szy_minus_syy_szz2)
            * (sxx2_syy2_szz2_syz2_szy2 - syz_szy_minus_syy_szz2)
        + (-(sxz_p_szx) * (syz_m_szy) + (sxy_m_syx) * (sxx_m_syy - szz))
            * (-(sxz_m_szx) * (syz_p_szy) + (sxy_m_syx) * (sxx_m_syy + szz))
        + (-(sxz_p_szx) * (syz_p_szy) - (sxy_p_syx) * (sxx_p_syy - szz))
            * (-(sxz_m_szx) * (syz_m_szy) - (sxy_p_syx) * (sxx_p_syy + szz))
        + ((sxy_p_syx) * (syz_p_szy) + (sxz_p_szx) * (sxx_m_syy + szz))
            * (-(sxy_m_syx) * (syz_m_szy) + (sxz_p_szx) * (sxx_p_syy + szz))
        + ((sxy_p_syx) * (syz_m_szy) + (sxz_m_szx) * (sxx_m_syy - szz))
            * (-(sxy_m_syx) * (syz_p_szy) + (sxz_m_szx) * (sxx_p_syy - szz));

    c[1] = 8.0
        * (sxx * syz * szy + syy * szx * sxz + szz * sxy * syx
            - sxx * syy * szz
            - syz * szx * sxy
            - szy * syx * sxz);
    c[2] = -2.0 * fnorm_squared;

    // Newton-Raphson on the cubic-ish largest root.
    let mut mx_eigen_v = if polar {
        (3.0 * fnorm_squared).sqrt()
    } else {
        e0
    };
    if mx_eigen_v > EVAL_PREC {
        for _ in 0..50 {
            let oldg = mx_eigen_v;
            let x2 = mx_eigen_v * mx_eigen_v;
            let b = (x2 + c[2]) * mx_eigen_v;
            let a = b + c[1];
            let delta = (a * mx_eigen_v + c[0]) / (2.0 * x2 * mx_eigen_v + b + a);
            mx_eigen_v -= delta;
            if (mx_eigen_v - oldg).abs() < (EVAL_PREC * mx_eigen_v).abs() {
                break;
            }
        }
    } else {
        mx_eigen_v = 0.0;
    }

    let nrmsdsq = (2.0 * (e0 - mx_eigen_v)).max(0.0);

    let a11 = sxx_p_syy + szz - mx_eigen_v;
    let a12 = syz_m_szy;
    let a13 = -sxz_m_szx;
    let a14 = sxy_m_syx;

    let a21 = syz_m_szy;
    let a22 = sxx_m_syy - szz - mx_eigen_v;
    let a23 = sxy_p_syx;
    let a24 = sxz_p_szx;

    let a31 = a13;
    let a32 = a23;
    let a33 = syy - sxx - szz - mx_eigen_v;
    let a34 = syz_p_szy;

    let a41 = a14;
    let a42 = a24;
    let a43 = a34;
    let a44 = szz - sxx_p_syy - mx_eigen_v;

    let a3344_4334 = a33 * a44 - a43 * a34;
    let a3244_4234 = a32 * a44 - a42 * a34;
    let a3243_4233 = a32 * a43 - a42 * a33;
    let a3143_4133 = a31 * a43 - a41 * a33;
    let a3144_4134 = a31 * a44 - a41 * a34;
    let a3142_4132 = a31 * a42 - a41 * a32;
    let a1324_1423 = a13 * a24 - a14 * a23;
    let a1224_1422 = a12 * a24 - a14 * a22;
    let a1223_1322 = a12 * a23 - a13 * a22;
    let a1124_1421 = a11 * a24 - a14 * a21;
    let a1123_1321 = a11 * a23 - a13 * a21;
    let a1122_1221 = a11 * a22 - a12 * a21;

    let mut q = [[0.0f64; 4]; 4];
    q[0][0] = a12 * a3344_4334 - a13 * a3244_4234 + a14 * a3243_4233;
    q[0][1] = -a11 * a3344_4334 + a13 * a3144_4134 - a14 * a3143_4133;
    q[0][2] = a11 * a3244_4234 - a12 * a3144_4134 + a14 * a3142_4132;
    q[0][3] = -a11 * a3243_4233 + a12 * a3143_4133 - a13 * a3142_4132;

    q[1][0] = a22 * a3344_4334 - a23 * a3244_4234 + a24 * a3243_4233;
    q[1][1] = -a21 * a3344_4334 + a23 * a3144_4134 - a24 * a3143_4133;
    q[1][2] = a21 * a3244_4234 - a22 * a3144_4134 + a24 * a3142_4132;
    q[1][3] = -a21 * a3243_4233 + a22 * a3143_4133 - a23 * a3142_4132;

    q[2][0] = a32 * a1324_1423 - a33 * a1224_1422 + a34 * a1223_1322;
    q[2][1] = -a31 * a1324_1423 + a33 * a1124_1421 - a34 * a1123_1321;
    q[2][2] = a31 * a1224_1422 - a32 * a1124_1421 + a34 * a1122_1221;
    q[2][3] = -a31 * a1223_1322 + a32 * a1123_1321 - a33 * a1122_1221;

    q[3][0] = a42 * a1324_1423 - a43 * a1224_1422 + a44 * a1223_1322;
    q[3][1] = -a41 * a1324_1423 + a43 * a1124_1421 - a44 * a1123_1321;
    q[3][2] = a41 * a1224_1422 - a42 * a1124_1421 + a44 * a1122_1221;
    q[3][3] = -a41 * a1223_1322 + a42 * a1123_1321 - a43 * a1122_1221;

    let mut qsqr = [0.0f64; 4];
    for i in 0..4 {
        qsqr[i] = q[i][0] * q[i][0] + q[i][1] * q[i][1] + q[i][2] * q[i][2] + q[i][3] * q[i][3];
    }

    let mut bi = 0;
    let mut max = 0.0f64;
    for (i, &qs) in qsqr.iter().enumerate() {
        if qs > max {
            bi = i;
            max = qs;
        }
    }

    let mut too_small = false;
    if qsqr[bi] < EVEC_PREC {
        // Degenerate fit: fall back to the identity rotation.
        q[bi] = [1.0, 0.0, 0.0, 0.0];
        too_small = true;
    } else {
        let normq = qsqr[bi].sqrt();
        for v in q[bi].iter_mut() {
            *v /= normq;
        }
    }

    (!too_small, nrmsdsq, q[bi])
}

/// Rotation matrix for a unit quaternion `(w, x, y, z)`.
pub fn quaternion_to_rotation_matrix(q: &[f64; 4]) -> [f64; 9] {
    let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
    [
        a * a + b * b - c * c - d * d,
        2.0 * b * c - 2.0 * a * d,
        2.0 * b * d + 2.0 * a * c,
        2.0 * b * c + 2.0 * a * d,
        a * a - b * b + c * c - d * d,
        2.0 * c * d - 2.0 * a * b,
        2.0 * b * d - 2.0 * a * c,
        2.0 * c * d + 2.0 * a * b,
        a * a - b * b - c * c + d * d,
    ]
}

/// Optimal rotation for `A` and the normalised RMSD of the fit.
///
/// `e0` is `(G1 + G2) / 2`, the midpoint of the two sets' total squared norms.
/// The returned `nrmsdsq` is an intermediate of the QCP solve, *not* the RMSD
/// the matcher reports: that comes from [`calc_rmsd`], which additionally fits
/// a scale factor.
pub fn fast_calc_rmsd_and_rotation(a: &[f64; 9], e0: f64) -> (f64, [f64; 4], [f64; 9]) {
    let (_ok, nrmsdsq, q) = optimal_quaternion(a, false, e0);
    let u = quaternion_to_rotation_matrix(&q);
    (nrmsdsq, q, u)
}

/// Upstream `polar_decomposition_3x3` (`ptm_polar.cpp`, as the PTM library
/// bundled with LAMMPS calls it from `output_data` with `right_sided =
/// false`): `F = P U` with `U` a proper rotation and `P = F Uᵀ` the symmetric
/// left stretch; with `right_sided`, `F = U P` and `P = Uᵀ F`. Row-major, the
/// same arithmetic in the same order as upstream.
pub fn polar_decomposition_3x3(f: &[f64; 9], right_sided: bool) -> ([f64; 9], [f64; 9]) {
    let det = f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6])
        + f[2] * (f[3] * f[7] - f[4] * f[6]);
    let mut a = *f;
    if det < 0.0 {
        a.iter_mut().for_each(|v| *v = -*v);
    }
    let (_ok, _nrmsdsq, mut q) = optimal_quaternion(&a, true, -1.0);
    q[0] = -q[0];
    let mut u = quaternion_to_rotation_matrix(&q);
    if det < 0.0 {
        u.iter_mut().for_each(|v| *v = -*v);
    }
    let ut = [u[0], u[3], u[6], u[1], u[4], u[7], u[2], u[5], u[8]];
    let p = if right_sided {
        matmul_3x3(&ut, f)
    } else {
        matmul_3x3(f, &ut)
    };
    (u, p)
}

/// Upstream `matmul_3x3`: `b = A x`, row-major, in upstream's order.
fn matmul_3x3(a: &[f64; 9], x: &[f64; 9]) -> [f64; 9] {
    let mut b = [0.0f64; 9];
    b[0] = a[0] * x[0] + a[1] * x[3] + a[2] * x[6];
    b[3] = a[3] * x[0] + a[4] * x[3] + a[5] * x[6];
    b[6] = a[6] * x[0] + a[7] * x[3] + a[8] * x[6];

    b[1] = a[0] * x[1] + a[1] * x[4] + a[2] * x[7];
    b[4] = a[3] * x[1] + a[4] * x[4] + a[5] * x[7];
    b[7] = a[6] * x[1] + a[7] * x[4] + a[8] * x[7];

    b[2] = a[0] * x[2] + a[1] * x[5] + a[2] * x[8];
    b[5] = a[3] * x[2] + a[4] * x[5] + a[5] * x[8];
    b[8] = a[6] * x[2] + a[7] * x[5] + a[8] * x[8];
    b
}

/// Rotate `ideal` by `rot`, measure it against `normalized` through
/// `mapping`, and fit the best isotropic scale.
///
/// This is upstream's `calc_rmsd`. The QCP rotation is computed on unit-ish
/// templates, but a thermally distorted neighbourhood also needs a scale, so the
/// reported RMSD is `sqrt(|G1 - scale * k0| / num_points)` with
/// `scale = k0 / G2`. `G1` and `G2` are the total squared norms of the template
/// and of the (already normalised) point set, and are passed in because the
/// caller computes them once per atom rather than once per automorphism.
///
/// `num_points` is the *reference's* point count (`num_nbrs + 1`), which is
/// 13 for FCC/HCP/ICO and 15 for BCC. It is emphatically not the width of the
/// caller's arrays: the gathered environment is 19 points wide, but the fit
/// only involves the reference's prefix. Dividing by 19 instead makes every
/// RMSD smaller by `sqrt(13/19)` and silently loosens the cutoff by the same
/// factor - a self-fit test cannot catch it, because the residual is ~1e-8
/// either way and the square root of it barely moves.
pub fn calc_rmsd<const N: usize>(
    ideal_points: &[[f64; 3]; N],
    normalized: &[[f64; 3]; N],
    mapping: &[usize; N],
    num_points: usize,
    g1: f64,
    g2: f64,
) -> (f64, f64, [f64; 4], [f64; 9]) {
    let a0 = inner_product(ideal_points, normalized, mapping, num_points);
    let e0 = (g1 + g2) / 2.0;
    let (_nrmsdsq, q, rot) = fast_calc_rmsd_and_rotation(&a0, e0);

    // k0 = sum_ij (rot * ideal_i)_j * normalized[mapping[i]]_j
    let mut k0 = 0.0;
    for i in 0..num_points {
        for j in 0..3 {
            let mut v = 0.0;
            for k in 0..3 {
                v += rot[j * 3 + k] * ideal_points[i][k];
            }
            k0 += v * normalized[mapping[i]][j];
        }
    }

    let scale = k0 / g2;
    let rmsd = ((g1 - scale * k0).abs() / num_points as f64).sqrt();
    (rmsd, scale, q, rot)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An exactly-known rotation: build a point set, rotate it, and check that
    /// the fit recovers the angle.
    fn rotation_z(theta: f64) -> [f64; 9] {
        let (s, c) = theta.sin_cos();
        [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
    }

    fn apply(u: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
        [
            u[0] * v[0] + u[1] * v[1] + u[2] * v[2],
            u[3] * v[0] + u[4] * v[1] + u[5] * v[2],
            u[6] * v[0] + u[7] * v[1] + u[8] * v[2],
        ]
    }

    fn norm2(p: [f64; 3]) -> f64 {
        p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
    }

    #[test]
    fn recovers_a_known_rotation() {
        // Four non-coplanar points, deliberately unsymmetrical so a wrong
        // rotation cannot accidentally score well.
        let base: [[f64; 3]; 4] = [
            [1.0, 0.0, 0.0],
            [0.3, 1.2, 0.4],
            [0.5, 0.1, 1.7],
            [1.1, 0.9, 0.2],
        ];
        let u = rotation_z(0.7);
        let rotated: [[f64; 3]; 4] = base.map(|p| apply(&u, p));

        let g1: f64 = base.iter().map(|p| norm2(*p)).sum();
        let g2: f64 = rotated.iter().map(|p| norm2(*p)).sum();
        let perm = [0usize, 1, 2, 3];
        let (rmsd, scale, _q, fit) = calc_rmsd(&base, &rotated, &perm, 4, g1, g2);

        assert!(rmsd < 1e-12, "exact fit should give ~0, got {rmsd}");
        // A pure rotation preserves norms, so the fitted scale is 1.
        assert!((scale - 1.0).abs() < 1e-12, "scale = {scale}");
        // `fit` rotates the template onto the data, which for an exact fit is
        // the rotation that was applied.
        for i in 0..3 {
            for j in 0..3 {
                let want = u[i * 3 + j];
                assert!(
                    (fit[i * 3 + j] - want).abs() < 1e-12,
                    "fit[{i}][{j}] = {} want {want}",
                    fit[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn a_scaled_neighbourhood_gives_the_reciprocal_scale() {
        // Doubling every point is a perfect fit up to scale, which is the case
        // that distinguishes the scale-fitted RMSD from the plain QCP `nrmsdsq`.
        //
        // Note the direction: `scale = k0 / G2` is the factor that takes the
        // *normalised data* onto the *template*, not the other way round. With
        // `data = 2 * template` the answer is therefore 0.5, and the lattice
        // scale the caller wants is its reciprocal. Getting this backwards would
        // still give rmsd ~ 0 here, so it is pinned explicitly.
        let base: [[f64; 3]; 4] = [
            [1.0, 0.0, 0.0],
            [0.3, 1.2, 0.4],
            [0.5, 0.1, 1.7],
            [1.1, 0.9, 0.2],
        ];
        let scaled: [[f64; 3]; 4] = base.map(|p| [2.0 * p[0], 2.0 * p[1], 2.0 * p[2]]);
        let g1: f64 = base.iter().map(|p| norm2(*p)).sum();
        let g2: f64 = scaled.iter().map(|p| norm2(*p)).sum();
        let perm = [0usize, 1, 2, 3];
        let (rmsd, scale, _q, _fit) = calc_rmsd(&base, &scaled, &perm, 4, g1, g2);
        assert!((scale - 0.5).abs() < 1e-12, "scale = {scale}");
        assert!((1.0 / scale - 2.0).abs() < 1e-12);
        // The residual is not zero: the QCP seeds its eigenvalue search at
        // `(G1 + G2) / 2` and Newton-Raphson only converges it to ~1e-11
        // relative, which leaves a small rotation error once the data is scaled
        // rather than normalised. This regime does not arise in real use, where
        // the data arrives normalised to unit mean squared neighbour distance,
        // so assert only that the residual is negligible rather than exact.
        assert!(rmsd < 1e-6, "rmsd = {rmsd}");
    }

    #[test]
    fn identity_rotation_is_recovered() {
        let base: [[f64; 3]; 4] = [
            [1.0, 0.0, 0.0],
            [0.3, 1.2, 0.4],
            [0.5, 0.1, 1.7],
            [1.1, 0.9, 0.2],
        ];
        let perm = [0usize; 4];
        let a = inner_product(&base, &base, &perm, 4);
        let e0 = 4.0; // four unit vectors
        let (nrmsdsq, _q, fit) = fast_calc_rmsd_and_rotation(&a, e0);
        assert!(nrmsdsq < 1e-20);
        for (i, want) in [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            .iter()
            .enumerate()
        {
            assert!((fit[i] - want).abs() < 1e-12, "fit[{i}] = {}", fit[i]);
        }
    }

    #[test]
    fn permutation_reorders_the_second_point_set() {
        // `permutation[i]` selects which point of `coords2` pairs with
        // `coords1[i]`, so swapping it must swap which outer products are
        // accumulated. Written out in full because the exact index layout
        // (A[3] is y-row/x-column, not a flat row) is easy to get wrong.
        let a: [[f64; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]];

        let identity = inner_product(&a, &b, &[0, 1], 2);
        // [1,2,3] x [0,0,0] is zero; [4,5,6] x [9,9,9] is 9*outer.
        for k in 0..3 {
            for l in 0..3 {
                let want = [4.0, 5.0, 6.0][k] * 9.0;
                assert_eq!(identity[k * 3 + l], want, "A[{k}][{l}]");
            }
        }

        let swapped = inner_product(&a, &b, &[1, 0], 2);
        // Now [1,2,3] pairs with [9,9,9] and [4,5,6] with the origin.
        for k in 0..3 {
            for l in 0..3 {
                let want = [1.0, 2.0, 3.0][k] * 9.0;
                assert_eq!(swapped[k * 3 + l], want, "A[{k}][{l}]");
            }
        }
    }

    #[test]
    fn quaternion_to_matrix_is_orthonormal() {
        let q = [0.5f64, 0.5, 0.5, 0.5];
        let u = quaternion_to_rotation_matrix(&q);
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| u[i * 3 + k] * u[j * 3 + k]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12, "row {i}.{j} dot = {dot}");
            }
        }
    }

    #[test]
    fn degenerate_fit_returns_identity() {
        // A zero matrix has no meaningful rotation; the fallback is identity
        // rather than a NaN quaternion.
        let a = [0.0f64; 9];
        let (_ok, nrmsdsq, q) = optimal_quaternion(&a, false, 0.0);
        assert!(nrmsdsq.is_finite());
        assert_eq!(q, [1.0, 0.0, 0.0, 0.0]);
    }
}
