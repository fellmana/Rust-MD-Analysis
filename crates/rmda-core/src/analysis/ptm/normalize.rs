//! Point-set preparation for matching (upstream `ptm_normalize_vertices.cpp`).

/// Translate a point set so its barycentre is at the origin.
///
/// The templates are defined with barycentre `{0, 0, 0}`, so the RMSD fit only
/// sees the residual translation; subtracting it here is what makes the fit
/// scale-and-translation tolerant in the right way.
pub fn subtract_barycentre<const N: usize>(points: &[[f64; 3]; N]) -> [[f64; 3]; N] {
    subtract_barycentre_n(points, N)
}

/// Translate the first `n` points so their barycentre is at the origin.
///
/// The count is separate from the array length because a caller can hand over a
/// full-width buffer holding only the `num_nbrs + 1` points a given reference
/// actually reads. Upstream passes that count explicitly, and letting the
/// padding leak into the mean would shift the centre - which is exactly how SC
/// and BCC, the two structures that read fewer points than the family code, are
/// affected. Rows past `n` are left zeroed; nothing reads them.
pub fn subtract_barycentre_n<const N: usize>(points: &[[f64; 3]; N], n: usize) -> [[f64; 3]; N] {
    let used = &points[..n];
    let mut sum = [0.0f64; 3];
    for p in used.iter() {
        for k in 0..3 {
            sum[k] += p[k];
        }
    }
    for s in sum.iter_mut() {
        *s /= n as f64;
    }

    let mut normalized = [[0.0f64; 3]; N];
    for (out, p) in normalized.iter_mut().zip(used.iter()) {
        for k in 0..3 {
            out[k] = p[k] - sum[k];
        }
    }
    normalized
}

/// Centre a point set and scale it so the mean neighbour length is 1.
///
/// The mean is over the *neighbours only* (`i` from 1), but divided by `num`,
/// which includes the central atom. That off-by-one-looking division is
/// upstream's and is reproduced exactly: it sets the absolute scale the hull
/// sees, and a different convention would change which tetrahedron
/// `initial_simplex` picks on near-degenerate inputs.
///
/// Returns the mean length, which is the distance the caller later converts
/// into a lattice constant.
pub fn normalize_vertices<const N: usize>(points: &[[f64; 3]; N]) -> ([[f64; 3]; N], f64) {
    normalize_vertices_n(points, N)
}

/// Centre the first `n` points and divide them by the summed neighbour length
/// over `n` (as [`normalize_vertices`]). See [`subtract_barycentre_n`] for why the count is
/// explicit.
pub fn normalize_vertices_n<const N: usize>(
    points: &[[f64; 3]; N],
    n: usize,
) -> ([[f64; 3]; N], f64) {
    let mut normalized = subtract_barycentre_n(points, n);

    let mut scale = 0.0f64;
    for p in normalized[1..n].iter() {
        scale += (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    }
    scale /= n as f64;

    // A degenerate input (no neighbours) divides by zero and would fill the
    // output with NaNs. Upstream has the same hazard and never hits it, because
    // every environment has at least one neighbour; returning the centred
    // points unchanged keeps a NaN from spreading silently.
    if scale == 0.0 {
        return (normalized, scale);
    }

    for p in normalized[..n].iter_mut() {
        for pk in p.iter_mut() {
            *pk /= scale;
        }
    }

    (normalized, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtract_barycentre_centres_the_set() {
        let points = [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ];
        let n = subtract_barycentre(&points);
        for axis in 0..3 {
            let mean: f64 = n.iter().map(|p| p[axis]).sum::<f64>() / 4.0;
            assert_eq!(mean, 0.0);
        }
        // Relative geometry must be untouched.
        for (a, b) in n.iter().zip(points.iter()) {
            for k in 0..3 {
                assert!((a[k] - (b[k] - 1.25)).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn normalize_vertices_makes_the_mean_neighbour_length_one() {
        let points = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let (n, scale) = normalize_vertices(&points);
        // The mean is taken *after* the barycentre shift, over the neighbours
        // only, and divided by num = 4 rather than 3. Both are upstream's
        // convention and both change the value, so pin them by recomputing the
        // mean from the centred points rather than from the raw ones.
        let centre = [0.5, 0.75, 1.0];
        let want: f64 = points[1..]
            .iter()
            .map(|p| {
                let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .sum::<f64>()
            / 4.0;
        assert!(
            (scale - want).abs() < 1e-12,
            "scale was {scale}, want {want}"
        );
        let mean: f64 = n[1..]
            .iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
            .sum::<f64>()
            / 4.0;
        assert!((mean - 1.0).abs() < 1e-12, "mean neighbour length {mean}");
    }

    #[test]
    fn normalize_vertices_is_scale_invariant() {
        // Scaling the input scales the output's points by the reciprocal, so the
        // hull topology is unaffected. The reported scale, however, is the
        // absolute input distance -- that is what becomes a lattice constant.
        let points = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ];
        let (a, sa) = normalize_vertices(&points);
        let scaled = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let (b, sb) = normalize_vertices(&scaled);
        assert!((sb / sa - 2.0).abs() < 1e-12);
        for (x, y) in a.iter().zip(b.iter()) {
            for k in 0..3 {
                assert!((x[k] - y[k]).abs() < 1e-12, "normalised points differ");
            }
        }
    }

    #[test]
    fn a_single_point_has_zero_scale_and_is_not_normalised() {
        // Guards the degenerate case: dividing by a zero mean length would
        // produce NaNs rather than an error.
        let points = [[0.0, 0.0, 0.0]];
        let (n, scale) = normalize_vertices(&points);
        assert_eq!(scale, 0.0);
        assert_eq!(n[0], [0.0, 0.0, 0.0]);
    }
}
