//! Incremental 3D convex hull (upstream `ptm_convex_hull_incremental.cpp`).
//!
//! The hull is grown one point at a time. For each new point, every existing
//! facet whose plane the point lies outside of is deleted, and the horizon
//! edges (those shared by one deleted and one surviving facet) are swept into
//! new facets. Edge visibility is tracked in a small bit matrix so that a
//! shared edge seen once as `VISIBLE` and once as `INVISIBLE` -- i.e. one of its
//! two facets is being removed -- is recognised as a horizon edge.
//!
//! Everything here is a faithful transcription; the facet *order* is
//! significant because [`super::matcher`] canonicalises it, and the exact
//! degenerate-point handling decides which inputs are rejected outright.

use crate::analysis::ptm::MAX_FACETS;
use crate::analysis::ptm::MAX_POINTS;

const VISIBLE: u8 = 1;
const INVISIBLE: u8 = 2;
const BOTH: u8 = VISIBLE | INVISIBLE;
const TOLERANCE: f64 = 1e-8;

/// Incremental hull state, carried across calls so a caller can feed points in
/// batches (upstream `convexhull_t`).
#[derive(Clone)]
pub struct ConvexHull {
    pub facets: [[i8; 3]; MAX_FACETS],
    pub plane_normal: [[f64; 3]; MAX_FACETS],
    pub processed: [bool; MAX_POINTS],
    pub initial_vertices: [usize; 4],
    pub barycentre: [f64; 3],
    pub num_facets: usize,
    pub num_prev: usize,
    pub ok: bool,
}

impl Default for ConvexHull {
    fn default() -> Self {
        Self {
            facets: [[0; 3]; MAX_FACETS],
            plane_normal: [[0.0; 3]; MAX_FACETS],
            processed: [false; MAX_POINTS],
            initial_vertices: [0; 4],
            barycentre: [0.0; 3],
            num_facets: 0,
            num_prev: 0,
            ok: false,
        }
    }
}

/// Why [`get_convex_hull`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HullError {
    /// All points are collinear, coplanar or coincident: no 3D hull exists.
    Degenerate,
    /// The hull would need more than [`MAX_FACETS`] facets.
    TooManyFacets,
}

fn norm_squared(p: [f64; 3]) -> f64 {
    let x = p[0];
    let y = p[1];
    let z = p[2];
    x * x + y * y + z * z
}

fn dot_product(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross_product(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Unit normal of the plane through `a`, `b`, `c`.
fn calculate_plane_normal(points: &[[f64; 3]], a: usize, b: usize, c: usize) -> [f64; 3] {
    let u = [
        points[b][0] - points[a][0],
        points[b][1] - points[a][1],
        points[b][2] - points[a][2],
    ];
    let v = [
        points[c][0] - points[a][0],
        points[c][1] - points[a][1],
        points[c][2] - points[a][2],
    ];
    let mut n = cross_product(u, v);
    let norm = norm_squared(n).sqrt();
    n[0] /= norm;
    n[1] /= norm;
    n[2] /= norm;
    n
}

/// Signed distance of `w` from the plane, positive on the outward side.
fn point_plane_distance(w: [f64; 3], plane_point: [f64; 3], plane_cross: [f64; 3]) -> f64 {
    plane_cross[0] * (plane_point[0] - w[0])
        + plane_cross[1] * (plane_point[1] - w[1])
        + plane_cross[2] * (plane_point[2] - w[2])
}

/// Add the facet `(a, b, c)`, flipping its winding if needed so the normal
/// points away from the barycentre of the initial simplex.
///
/// This is what makes every stored `plane_normal` an *outward* normal, which the
/// visibility test below relies on.
pub fn add_facet(
    points: &[[f64; 3]],
    a: usize,
    b: usize,
    c: usize,
    barycentre: [f64; 3],
) -> ([i8; 3], [f64; 3]) {
    let mut plane_normal = calculate_plane_normal(points, a, b, c);
    if point_plane_distance(barycentre, points[a], plane_normal) > 0.0 {
        plane_normal[0] = -plane_normal[0];
        plane_normal[1] = -plane_normal[1];
        plane_normal[2] = -plane_normal[2];
        ([b as i8, a as i8, c as i8], plane_normal)
    } else {
        ([a as i8, b as i8, c as i8], plane_normal)
    }
}

/// Per-axis extremes; fails if any axis is flat (a degenerate point set).
fn calc_max_extent(points: &[[f64; 3]]) -> Option<([usize; 3], [usize; 3])> {
    let mut min_index = [0usize; 3];
    let mut max_index = [0usize; 3];
    for j in 0..3 {
        let mut dmin = f64::MAX;
        let mut dmax = -f64::MAX;
        let mut imin = 0usize;
        let mut imax = 0usize;
        for (i, p) in points.iter().enumerate() {
            let d = p[j];
            if d < dmin {
                dmin = d;
                imin = i;
            }
            if d > dmax {
                dmax = d;
                imax = i;
            }
        }
        if imin == imax {
            return None; // degenerate point set
        }
        min_index[j] = imin;
        max_index[j] = imax;
    }
    Some((min_index, max_index))
}

/// Point furthest from the line `a`--`b`, as squared distance.
fn find_third_point(points: &[[f64; 3]], a: usize, b: usize) -> Option<usize> {
    let x1 = points[a];
    let x2 = points[b];
    let x2x1 = [x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]];
    let ns_x2x1 = norm_squared(x2x1);

    let mut bi = -1i32;
    let mut max_dist = 0.0f64;
    for (i, x0) in points.iter().enumerate() {
        if i == a || i == b {
            continue;
        }
        let x1x0 = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
        let dot = dot_product(x1x0, x2x1);
        let dist = (norm_squared(x1x0) * ns_x2x1 - dot * dot) / ns_x2x1;
        if dist > max_dist {
            max_dist = dist;
            bi = i as i32;
        }
    }
    (max_dist > TOLERANCE).then_some(bi as usize)
}

/// Point furthest from the plane through `a`, `b`, `c`.
fn find_fourth_point(points: &[[f64; 3]], a: usize, b: usize, c: usize) -> Option<usize> {
    let plane_normal = calculate_plane_normal(points, a, b, c);
    let mut bi = -1i32;
    let mut max_dist = 0.0f64;
    for (i, x0) in points.iter().enumerate() {
        if i == a || i == b || i == c {
            continue;
        }
        let dist = point_plane_distance(*x0, points[a], plane_normal).abs();
        if dist > max_dist {
            max_dist = dist;
            bi = i as i32;
        }
    }
    (max_dist > TOLERANCE).then_some(bi as usize)
}

/// Pick four points spanning a tetrahedron: the widest of the three principal
/// axis pairs, then the points furthest off that line and that plane.
fn initial_simplex(points: &[[f64; 3]]) -> Option<[usize; 4]> {
    let (min_index, max_index) = calc_max_extent(points)?;

    let mut bi = 0usize;
    let mut max_dist = 0.0f64;
    for i in 0..3 {
        let a = min_index[i];
        let b = max_index[i];
        let delta = [
            points[a][0] - points[b][0],
            points[a][1] - points[b][1],
            points[a][2] - points[b][2],
        ];
        let dist = norm_squared(delta);
        if dist > max_dist {
            bi = i;
            max_dist = dist;
        }
    }

    // The first two points are (a, b).
    let a = min_index[bi];
    let b = max_index[bi];
    let c = find_third_point(points, a, b)?;
    let d = find_fourth_point(points, a, b, c)?;
    Some([a, b, c, d])
}

fn initialize_convex_hull(points: &[[f64; 3]], ch: &mut ConvexHull) -> Result<(), HullError> {
    ch.processed = [false; MAX_POINTS];
    ch.barycentre = [0.0; 3];
    let iv = initial_simplex(points).ok_or(HullError::Degenerate)?;

    for v in iv {
        ch.processed[v] = true;
        ch.barycentre[0] += points[v][0];
        ch.barycentre[1] += points[v][1];
        ch.barycentre[2] += points[v][2];
    }
    ch.barycentre[0] /= 4.0;
    ch.barycentre[1] /= 4.0;
    ch.barycentre[2] /= 4.0;
    ch.initial_vertices = iv;

    // The four faces of the starting tetrahedron, in the (a,b,c,d) order.
    let quads = [
        (iv[0], iv[1], iv[2]),
        (iv[0], iv[1], iv[3]),
        (iv[0], iv[2], iv[3]),
        (iv[1], iv[2], iv[3]),
    ];
    for (i, &(a, b, c)) in quads.iter().enumerate() {
        let (facet, n) = add_facet(points, a, b, c, ch.barycentre);
        ch.facets[i] = facet;
        ch.plane_normal[i] = n;
    }
    ch.num_facets = 4;
    Ok(())
}

/// Convex hull of `points` (point 0 is the central atom).
///
/// Writes the hull's faces into `simplex` with indices **rebased to drop point
/// 0**, i.e. `simplex[f][k] == facet vertex - 1`, matching upstream. Returns
/// `Err(HullError::Degenerate)` if the points do not span 3D, or if the central
/// atom ends up on the hull rather than inside it (a facet containing index 0).
pub fn get_convex_hull(
    points: &[[f64; 3]],
    ch: &mut ConvexHull,
    simplex: &mut [[i8; 3]; MAX_FACETS],
) -> Result<(), HullError> {
    let num_points = points.len();
    assert!(num_points <= MAX_POINTS, "too many points for a PTM hull");

    let mut num_prev = ch.num_prev;
    ch.num_prev = num_points;
    if !ch.ok {
        initialize_convex_hull(points, ch)?;
        num_prev = 0;
    }

    let mut i = num_prev;
    while i < num_points {
        if ch.processed[i] {
            i += 1;
            continue;
        }
        ch.processed[i] = true;

        let mut num_to_add = 0usize;
        let mut to_add = [[0i8; 3]; MAX_FACETS];
        // Bit 0: this edge borders a facet that is going away.
        // Bit 1: this edge borders a facet that is staying.
        let mut edge_visible = [[0u8; MAX_POINTS]; MAX_POINTS];

        let mut j = 0usize;
        while j < ch.num_facets {
            let a = ch.facets[j][0] as usize;
            let b = ch.facets[j][1] as usize;
            let c = ch.facets[j][2] as usize;

            let (u, v, w);
            let removed =
                point_plane_distance(points[i], points[a], ch.plane_normal[j]) > TOLERANCE;
            if removed {
                edge_visible[a][b] |= VISIBLE;
                edge_visible[b][a] |= VISIBLE;
                u = edge_visible[a][b];

                edge_visible[b][c] |= VISIBLE;
                edge_visible[c][b] |= VISIBLE;
                v = edge_visible[b][c];

                edge_visible[c][a] |= VISIBLE;
                edge_visible[a][c] |= VISIBLE;
                w = edge_visible[c][a];
            } else {
                edge_visible[a][b] |= INVISIBLE;
                edge_visible[b][a] |= INVISIBLE;
                u = edge_visible[a][b];

                edge_visible[b][c] |= INVISIBLE;
                edge_visible[c][b] |= INVISIBLE;
                v = edge_visible[b][c];

                edge_visible[c][a] |= INVISIBLE;
                edge_visible[a][c] |= INVISIBLE;
                w = edge_visible[c][a];
            }

            // A horizon edge is one that borders both a doomed and a surviving
            // facet; sweeping it produces one new facet. This is evaluated for
            // doomed facets too -- an edge only becomes `BOTH` once its other
            // facet has also been visited.
            if u == BOTH {
                to_add[num_to_add] = [i as i8, a as i8, b as i8];
                num_to_add += 1;
            }
            if v == BOTH {
                to_add[num_to_add] = [i as i8, b as i8, c as i8];
                num_to_add += 1;
            }
            if w == BOTH {
                to_add[num_to_add] = [i as i8, c as i8, a as i8];
                num_to_add += 1;
            }

            if removed {
                // Drop this facet by moving the last one into its place, then
                // re-examine index j, which now holds the swapped-in facet.
                let last = ch.num_facets - 1;
                ch.facets[j] = ch.facets[last];
                ch.plane_normal[j] = ch.plane_normal[last];
                ch.num_facets -= 1;
            } else {
                j += 1;
            }
        }

        for t in to_add.iter().take(num_to_add) {
            if ch.num_facets >= MAX_FACETS {
                return Err(HullError::TooManyFacets);
            }
            let (facet, n) = add_facet(
                points,
                t[0] as usize,
                t[1] as usize,
                t[2] as usize,
                ch.barycentre,
            );
            ch.facets[ch.num_facets] = facet;
            ch.plane_normal[ch.num_facets] = n;
            ch.num_facets += 1;
        }

        i += 1;
    }

    // `f` drives two arrays (facets and simplex) by one count; an iterator
    // would only shuffle the same index around.
    #[allow(clippy::needless_range_loop)]
    for f in 0..ch.num_facets {
        let a = ch.facets[f][0];
        let b = ch.facets[f][1];
        let c = ch.facets[f][2];
        if a == 0 || b == 0 || c == 0 {
            // The central atom is a vertex of the hull, not enclosed by its
            // own neighbours, so this is not a valid local environment.
            return Err(HullError::Degenerate);
        }
        simplex[f] = [a - 1, b - 1, c - 1];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ptm::tables;

    /// Truncate a generated template to its `n` real points, dropping the
    /// zero padding the generator emits.
    fn template_points(t: &[[f64; 3]; MAX_POINTS], n: usize) -> Vec<[f64; 3]> {
        t[..n].to_vec()
    }

    /// Euler's formula for a triangulated closed surface: V - E + F = 2.
    fn euler_v_minus_e_plus_f(facets: &[[i8; 3]], num_facets: usize) -> i32 {
        let mut edges: Vec<(i8, i8)> = Vec::new();
        for f in facets.iter().take(num_facets) {
            for k in 0..3 {
                let (u, v) = (f[k], f[(k + 1) % 3]);
                let e = if u < v { (u, v) } else { (v, u) };
                if !edges.contains(&e) {
                    edges.push(e);
                }
            }
        }
        let vertices: std::collections::HashSet<i8> =
            facets.iter().take(num_facets).flatten().copied().collect();
        vertices.len() as i32 - edges.len() as i32 + num_facets as i32
    }

    fn hull_of(points: &[[f64; 3]]) -> (ConvexHull, Vec<[i8; 3]>) {
        let mut ch = ConvexHull::default();
        let mut simplex = [[0i8; 3]; MAX_FACETS];
        get_convex_hull(points, &mut ch, &mut simplex).expect("hull");
        let n = ch.num_facets;
        (ch, simplex[..n].to_vec())
    }

    // ---------------------------------------------------------------------
    // Reference facet lists captured from the upstream C++ implementation.
    //
    // Produced by compiling `ptm_convex_hull_incremental.cpp` from
    // pmla/polyhedral-template-matching @ master together with the upstream
    // templates and printing `simplex` after `get_convex_hull`. Facet *order*
    // is asserted too, because the structure matcher canonicalises it and a
    // divergence in ordering would change the matched type.
    // ---------------------------------------------------------------------

    const SC_REF: &[[i8; 3]] = &[
        [4, 1, 2],
        [5, 0, 2],
        [0, 4, 2],
        [2, 1, 5],
        [5, 3, 0],
        [0, 3, 4],
        [3, 1, 4],
        [1, 3, 5],
    ];

    const FCC_REF: &[[i8; 3]] = &[
        [4, 5, 3],
        [6, 0, 1],
        [7, 8, 1],
        [0, 2, 1],
        [3, 8, 7],
        [5, 6, 3],
        [2, 9, 7],
        [7, 9, 4],
        [7, 4, 3],
        [2, 7, 1],
        [8, 6, 1],
        [6, 8, 3],
        [9, 2, 0],
        [10, 5, 4],
        [10, 0, 6],
        [10, 6, 5],
        [11, 4, 9],
        [11, 9, 0],
        [4, 11, 10],
        [10, 11, 0],
    ];

    const HCP_REF: &[[i8; 3]] = &[
        [0, 9, 7],
        [3, 4, 7],
        [6, 3, 7],
        [0, 7, 4],
        [6, 5, 3],
        [3, 2, 4],
        [4, 2, 1],
        [5, 2, 3],
        [10, 5, 6],
        [2, 5, 1],
        [9, 8, 1],
        [1, 8, 4],
        [8, 0, 4],
        [8, 9, 0],
        [10, 6, 7],
        [10, 7, 9],
        [10, 11, 5],
        [5, 11, 1],
        [1, 11, 9],
        [9, 11, 10],
    ];

    const ICO_REF: &[[i8; 3]] = &[
        [10, 6, 9],
        [5, 9, 0],
        [1, 4, 6],
        [7, 8, 2],
        [3, 9, 6],
        [10, 9, 5],
        [1, 10, 2],
        [4, 3, 6],
        [0, 7, 5],
        [5, 7, 2],
        [8, 1, 2],
        [8, 4, 1],
        [9, 3, 0],
        [6, 10, 1],
        [2, 10, 5],
        [11, 8, 7],
        [4, 11, 3],
        [0, 11, 7],
        [8, 11, 4],
        [3, 11, 0],
    ];

    const BCC_REF: &[[i8; 3]] = &[
        [12, 1, 10],
        [12, 4, 3],
        [0, 12, 10],
        [0, 10, 8],
        [0, 8, 12],
        [1, 9, 10],
        [1, 12, 9],
        [10, 5, 2],
        [10, 2, 8],
        [9, 5, 10],
        [12, 3, 9],
        [4, 12, 8],
        [11, 7, 6],
        [11, 4, 7],
        [9, 6, 5],
        [3, 6, 9],
        [7, 8, 2],
        [7, 4, 8],
        [11, 3, 4],
        [3, 11, 6],
        [7, 13, 6],
        [13, 2, 5],
        [6, 13, 5],
        [2, 13, 7],
    ];

    fn assert_matches_reference(name: &str, points: &[[f64; 3]], expected: &[[i8; 3]]) {
        let (ch, got) = hull_of(points);
        assert_eq!(
            got.len(),
            expected.len(),
            "{name}: facet count {} != C++ {}",
            got.len(),
            expected.len()
        );
        for (f, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g, e, "{name}: facet {f} differs from C++");
        }
        // The result must be a closed, outward-oriented triangulated surface.
        assert_eq!(
            euler_v_minus_e_plus_f(&ch.facets, ch.num_facets),
            2,
            "{name}: hull is not a closed surface"
        );
    }

    #[test]
    fn sc_matches_cpp_reference() {
        let pts = template_points(&tables::SC_TEMPLATES[0], 7);
        assert_matches_reference("sc", &pts, SC_REF);
    }

    #[test]
    fn fcc_matches_cpp_reference() {
        let pts = template_points(&tables::FCC_TEMPLATES[0], 13);
        assert_matches_reference("fcc", &pts, FCC_REF);
    }

    #[test]
    fn hcp_matches_cpp_reference() {
        let pts = template_points(&tables::HCP_TEMPLATES[0], 13);
        assert_matches_reference("hcp", &pts, HCP_REF);
    }

    #[test]
    fn ico_matches_cpp_reference() {
        let pts = template_points(&tables::ICO_TEMPLATES[0], 13);
        assert_matches_reference("ico", &pts, ICO_REF);
    }

    #[test]
    fn bcc_matches_cpp_reference() {
        let pts = template_points(&tables::BCC_TEMPLATES[0], 15);
        assert_matches_reference("bcc", &pts, BCC_REF);
    }

    #[test]
    fn facet_counts_match_upstream_refdata() {
        // Upstream `refdata_t::num_facets` for the scoped structures. Note the
        // SC environment is an octahedron (6 neighbours, 8 faces) even though
        // the underlying lattice is simple cubic.
        for (name, pts, n) in [
            ("sc", template_points(&tables::SC_TEMPLATES[0], 7), 8),
            ("fcc", template_points(&tables::FCC_TEMPLATES[0], 13), 20),
            ("hcp", template_points(&tables::HCP_TEMPLATES[0], 13), 20),
            ("ico", template_points(&tables::ICO_TEMPLATES[0], 13), 20),
            ("bcc", template_points(&tables::BCC_TEMPLATES[0], 15), 24),
        ] {
            let (ch, _) = hull_of(&pts);
            assert_eq!(ch.num_facets, n, "{name}: unexpected facet count");
        }
    }

    #[test]
    fn every_facet_normal_points_outward() {
        // The neighbour centroid is strictly inside every hull, so its signed
        // distance to each facet plane must be negative for an outward normal.
        let pts = template_points(&tables::FCC_TEMPLATES[0], 13);
        let n = pts.len();
        let centre = [
            pts[1..n].iter().map(|p| p[0]).sum::<f64>() / (n - 1) as f64,
            pts[1..n].iter().map(|p| p[1]).sum::<f64>() / (n - 1) as f64,
            pts[1..n].iter().map(|p| p[2]).sum::<f64>() / (n - 1) as f64,
        ];
        let (ch, _) = hull_of(&pts);
        for f in 0..ch.num_facets {
            let a = ch.facets[f][0] as usize;
            let d = point_plane_distance(centre, pts[a], ch.plane_normal[f]);
            assert!(d < 0.0, "facet {f} normal is inward (signed distance {d})");
        }
    }

    #[test]
    fn indices_are_rebased_to_drop_the_central_atom() {
        let pts = template_points(&tables::BCC_TEMPLATES[0], 15);
        let (_, facets) = hull_of(&pts);
        for (f, facet) in facets.iter().enumerate() {
            for v in facet.iter() {
                assert!(
                    *v >= 0 && *v < 14,
                    "facet {f} vertex {v} outside 0..13 (points 1..=14 rebased)"
                );
            }
        }
    }

    #[test]
    fn collinear_points_are_rejected() {
        let points: Vec<[f64; 3]> = (0..13).map(|i| [i as f64 * 0.3, 0.0, 0.0]).collect();
        let mut ch = ConvexHull::default();
        let mut simplex = [[0i8; 3]; MAX_FACETS];
        assert_eq!(
            get_convex_hull(&points, &mut ch, &mut simplex),
            Err(HullError::Degenerate)
        );
    }

    #[test]
    fn coplanar_points_are_rejected() {
        // A ring in the z = 0 plane spans 2D, so no tetrahedron exists.
        let points: Vec<[f64; 3]> = (0..13)
            .map(|i| {
                let a = i as f64 * 0.5;
                [a.cos(), a.sin(), 0.0]
            })
            .collect();
        let mut ch = ConvexHull::default();
        let mut simplex = [[0i8; 3]; MAX_FACETS];
        assert_eq!(
            get_convex_hull(&points, &mut ch, &mut simplex),
            Err(HullError::Degenerate)
        );
    }

    #[test]
    fn coincident_points_are_rejected() {
        let points = vec![[0.0, 0.0, 0.0]; 13];
        let mut ch = ConvexHull::default();
        let mut simplex = [[0i8; 3]; MAX_FACETS];
        assert_eq!(
            get_convex_hull(&points, &mut ch, &mut simplex),
            Err(HullError::Degenerate)
        );
    }

    #[test]
    fn central_atom_inside_the_neighbour_hull_is_rejected() {
        // Neighbours on a ring around the origin, all in the z = 0 plane with
        // it: no 3D hull exists, so the environment is rejected as degenerate.
        let mut points = vec![[0.0, 0.0, 0.0]];
        for i in 0..12 {
            let t = i as f64 * 2.0 * std::f64::consts::PI / 12.0;
            points.push([t.cos(), t.sin(), 0.0]);
        }
        let mut ch = ConvexHull::default();
        let mut simplex = [[0i8; 3]; MAX_FACETS];
        let res = get_convex_hull(&points, &mut ch, &mut simplex);
        assert!(
            res.is_err(),
            "expected a degenerate/inside-centre rejection"
        );
    }

    #[test]
    fn a_second_point_batch_extends_the_same_hull() {
        // `get_convex_hull` is resumable: feeding a partial point set and then
        // extending it must agree with running it once over the full set.
        let full = template_points(&tables::FCC_TEMPLATES[0], 13);

        let mut ch = ConvexHull::default();
        let mut simplex = [[0i8; 3]; MAX_FACETS];
        get_convex_hull(&full[..5], &mut ch, &mut simplex).expect("partial hull");
        get_convex_hull(&full, &mut ch, &mut simplex).expect("full hull");

        let (_, got) = hull_of(&full);
        assert_eq!(
            &simplex[..ch.num_facets],
            &got[..],
            "resumed hull differs from the single-shot hull"
        );
    }

    #[test]
    fn an_unscaled_neighbourhood_gives_the_same_topology() {
        // The hull is scale invariant, so a uniformly scaled copy of the FCC
        // template must produce the identical facet list.
        let pts = template_points(&tables::FCC_TEMPLATES[0], 13);
        let scaled: Vec<[f64; 3]> = pts
            .iter()
            .map(|p| [p[0] * 3.7, p[1] * 3.7, p[2] * 3.7])
            .collect();
        let (_, a) = hull_of(&pts);
        let (_, b) = hull_of(&scaled);
        assert_eq!(a, b);
    }
}
