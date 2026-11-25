//! Graph utilities and canonical labelling for the convex-hull graph
//! (upstream `ptm_graph_tools.cpp` + `ptm_canonical_coloured.cpp`).
//!
//! The local environment is reduced to a graph whose nodes are the neighbours
//! and whose faces are the convex-hull facets. Two such graphs are the same
//! environment if they are related by a rotation *or* a reflection, so matching
//! is done on the graph's canonical form rather than on the geometry.
//!
//! [`canonical_form_coloured`] is Larsen's "Weinberg" traversal: a depth-first
//! walk that produces a lexicographically minimal vertex-code, with a
//! best-so-far bound that prunes the search. It returns both a canonical
//! labelling and a 64-bit hash of the code; the hash is what the matcher uses
//! to find candidate reference graphs cheaply, and the labelling then picks the
//! specific automorphism that fits the geometry best.

use crate::analysis::ptm::MAX_EDGES;
use crate::analysis::ptm::MAX_NBRS;
use crate::analysis::ptm::MAX_POINTS;

/// `common[a][b] == c` records that `c` is the third vertex of the facet
/// `(a, b, c)`, i.e. the neighbour reached by turning right from edge `a->b`.
///
/// Returns `false` if the facets are not a closed manifold, which shows up as
/// a directed edge appearing in two different facets.
pub fn build_facet_map(
    num_facets: usize,
    facets: &[[i8; 3]],
) -> Option<[[i8; MAX_NBRS]; MAX_NBRS]> {
    let mut common = [[-1i8; MAX_NBRS]; MAX_NBRS];

    for f in facets.iter().take(num_facets) {
        let a = f[0] as usize;
        let b = f[1] as usize;
        let c = f[2] as usize;

        if common[a][b] != -1 || common[b][c] != -1 || common[c][a] != -1 {
            return None;
        }

        common[a][b] = c as i8;
        common[b][c] = a as i8;
        common[c][a] = b as i8;
    }

    Some(common)
}

/// Degree of every node, where degree counts *facet incidences* rather than
/// distinct edges, plus the maximum degree.
///
/// Counting incidences means a square face split by a diagonal raises the
/// degree of two of its corners, which is what makes this sensitive to the
/// facet triangulation and not just the polyhedron.
pub fn graph_degree(
    num_facets: usize,
    facets: &[[i8; 3]],
    num_nodes: usize,
) -> ([i8; MAX_NBRS], i8) {
    let mut degree = [0i8; MAX_NBRS];

    for f in facets.iter().take(num_facets) {
        degree[f[0] as usize] += 1;
        degree[f[1] as usize] += 1;
        degree[f[2] as usize] += 1;
    }

    let mut max_degree = 0i8;
    for d in degree.iter().take(num_nodes) {
        if *d > max_degree {
            max_degree = *d;
        }
    }

    (degree, max_degree)
}

/// One Weinberg traversal starting from directed edge `a -> b`.
///
/// Writes a candidate labelling into `canonical_labelling` and a candidate code
/// into `best_code` only when this traversal wins. Returns `true` if it won.
#[allow(clippy::too_many_arguments)]
fn weinberg_coloured(
    num_nodes: usize,
    num_edges: usize,
    common: &[[i8; MAX_NBRS]; MAX_NBRS],
    colours: &[i8; MAX_POINTS],
    best_code: &mut [i8; 2 * MAX_EDGES],
    canonical_labelling: &mut [i8; MAX_POINTS],
    a: usize,
    b: usize,
) -> bool {
    let mut m = [[false; MAX_NBRS]; MAX_NBRS];
    let mut index = [-1i8; MAX_NBRS];

    let mut n = 0usize;
    let mut a = a;
    let mut b = b;
    index[a] = (colours[a] as i32 * num_nodes as i32 + n as i32) as i8;
    n += 1;
    if index[a] > best_code[0] {
        return false;
    }

    let mut winning = false;
    if index[a] < best_code[0] {
        best_code[0] = index[a];
        winning = true;
    }

    let mut c;
    // The counter is the walk's edge position, not a bare index; the iterator
    // form would need identical index arithmetic and obscure the algorithm.
    #[allow(clippy::needless_range_loop)]
    for it in 1..2 * num_edges {
        let newvertex = index[b] == -1;

        if newvertex {
            index[b] = (colours[b] as i32 * num_nodes as i32 + n as i32) as i8;
            n += 1;
        }

        if !winning && index[b] > best_code[it] {
            return false;
        }

        if winning || index[b] < best_code[it] {
            winning = true;
            best_code[it] = index[b];
        }

        if newvertex {
            // A new vertex is reached: take the right-most edge relative to
            // the edge it was reached on.
            c = common[a][b] as i32;
        } else if !m[b][a] {
            // An old vertex reached on a new path: back out along the reverse
            // direction.
            c = a as i32;
        } else {
            // An old vertex reached on an old path: leave on the right-most
            // edge not yet traversed in that direction.
            c = common[a][b] as i32;
            while m[b][c as usize] {
                c = common[c as usize][b] as i32;
            }
        }

        m[a][b] = true;
        a = b;
        b = c as usize;
    }

    if winning {
        canonical_labelling[..num_nodes].copy_from_slice(&index[..num_nodes]);
        true
    } else {
        false
    }
}

/// Canonical labelling and hash of a hull graph.
///
/// `colours` lets a caller mark structurally distinct node classes; PTM passes
/// all zeroes for the first-shell structures, which reduces this to the plain
/// (uncoloured) canonical form. DCUB/DHEX pass a real colouring.
///
/// Returns `Err(())` if the facets do not form a closed manifold.
#[allow(clippy::result_unit_err)]
pub fn canonical_form_coloured(
    num_facets: usize,
    facets: &[[i8; 3]],
    num_nodes: usize,
    degree: &[i8; MAX_NBRS],
    colours: &[i8; MAX_POINTS],
) -> Result<CanonicalForm, ()> {
    let common = build_facet_map(num_facets, facets).ok_or(())?;
    let num_edges = 3 * num_facets / 2;

    let mut best_code = [i8::MAX; 2 * MAX_EDGES];
    let mut canonical_labelling = [0i8; MAX_POINTS];

    let equal = (1..num_nodes).all(|i| degree[i] == degree[0] && colours[i] == colours[0]);

    if equal {
        // All nodes are interchangeable, so one traversal from the first
        // facet's first edge already yields the minimum.
        weinberg_coloured(
            num_nodes,
            num_edges,
            &common,
            colours,
            &mut best_code,
            &mut canonical_labelling,
            facets[0][0] as usize,
            facets[0][1] as usize,
        );
    } else {
        // Nodes differ, so only facets whose degree triple is maximal can start
        // a minimal traversal. There are three rotations of that triple, hence
        // up to three starting edges per qualifying facet.
        let mut best_degree = 0u32;
        for f in facets.iter().take(num_facets) {
            let (da, db, dc) = (
                degree[f[0] as usize],
                degree[f[1] as usize],
                degree[f[2] as usize],
            );
            let (d0, d1, d2) = (da as u32, db as u32, dc as u32);
            best_degree = best_degree.max((d0 << 16) | (d1 << 8) | d2);
            best_degree = best_degree.max(d0 | (d1 << 16) | (d2 << 8));
            best_degree = best_degree.max((d0 << 8) | d1 | (d2 << 16));
        }

        for f in facets.iter().take(num_facets) {
            let (da, db, dc) = (
                degree[f[0] as usize],
                degree[f[1] as usize],
                degree[f[2] as usize],
            );
            let (d0, d1, d2) = (da as u32, db as u32, dc as u32);
            let a = f[0] as usize;
            let b = f[1] as usize;
            let c = f[2] as usize;

            if best_degree == ((d0 << 16) | (d1 << 8) | d2) {
                weinberg_coloured(
                    num_nodes,
                    num_edges,
                    &common,
                    colours,
                    &mut best_code,
                    &mut canonical_labelling,
                    a,
                    b,
                );
            }
            if best_degree == (d0 | (d1 << 16) | (d2 << 8)) {
                weinberg_coloured(
                    num_nodes,
                    num_edges,
                    &common,
                    colours,
                    &mut best_code,
                    &mut canonical_labelling,
                    b,
                    c,
                );
            }
            if best_degree == ((d0 << 8) | d1 | (d2 << 16)) {
                weinberg_coloured(
                    num_nodes,
                    num_edges,
                    &common,
                    colours,
                    &mut best_code,
                    &mut canonical_labelling,
                    c,
                    a,
                );
            }
        }
    }

    // Convert the 0-based traversal order into a 1-based labelling, in reverse
    // so each read sees an already-converted value.
    for i in (0..num_nodes).rev() {
        canonical_labelling[i + 1] = (canonical_labelling[i] % num_nodes as i8) + 1;
    }
    canonical_labelling[0] = 0;

    let mut hash = 0u64;
    for (i, code) in best_code.iter().take(2 * num_edges).enumerate() {
        let mut e = *code as u64;
        e += (i % 8) as u64;
        e &= 0xF;
        e <<= (4 * i) % 64;
        hash ^= e;
    }

    Ok(CanonicalForm {
        hash,
        canonical_labelling,
    })
}

/// Result of [`canonical_form_coloured`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalForm {
    /// Hash of the minimal vertex code; used to look up candidate graphs.
    pub hash: u64,
    /// `canonical_labelling[i + 1]` is the 1-based rank of hull node `i` under
    /// the canonical labelling; entry 0 (the central atom) is always 0.
    pub canonical_labelling: [i8; MAX_POINTS],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ptm::hull;
    use crate::analysis::ptm::tables;
    use crate::analysis::ptm::MAX_FACETS;

    fn template_points(t: &[[f64; 3]; MAX_POINTS], n: usize) -> Vec<[f64; 3]> {
        t[..n].to_vec()
    }

    /// Rebuild the clockwise-wound hull of an ideal template, as
    /// `ptm_initialize_global` does before canonicalising the stored graphs.
    fn template_hull(
        t: &[[f64; 3]; MAX_POINTS],
        n: usize,
    ) -> (Vec<[f64; 3]>, [[i8; 3]; MAX_FACETS], usize) {
        let points = template_points(t, n);
        let mut ch = hull::ConvexHull::default();
        let mut simplex = [[0i8; 3]; MAX_FACETS];
        hull::get_convex_hull(&points, &mut ch, &mut simplex).expect("hull");
        (points, simplex, ch.num_facets)
    }

    #[test]
    fn facet_map_rejects_a_non_manifold() {
        // The same directed edge in two facets means the surface has a
        // boundary or a duplicated face.
        let facets = [[0i8, 1, 2], [0, 1, 2]];
        assert!(build_facet_map(2, &facets).is_none());
    }

    #[test]
    fn facet_map_records_the_third_vertex() {
        // A consistently wound tetrahedron: all 12 directed edges distinct.
        let facets = [[0i8, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let common = build_facet_map(4, &facets).expect("closed");

        // The map must be exactly the one implied by the facet list, and every
        // entry it does not account for must stay at the -1 sentinel.
        let mut expected = [[-1i8; MAX_NBRS]; MAX_NBRS];
        for f in &facets {
            let (a, b, c) = (f[0] as usize, f[1] as usize, f[2] as usize);
            expected[a][b] = c as i8;
            expected[b][c] = a as i8;
            expected[c][a] = b as i8;
        }
        assert_eq!(common, expected);
        // Self-loops never occur.
        for (i, row) in common.iter().enumerate() {
            assert_eq!(row[i], -1);
        }
    }

    #[test]
    fn degrees_match_the_cpp_reference() {
        // Degree vectors captured from upstream `graph_degree` run on each
        // ideal template's hull. `max_degree` is what `match_general` and
        // `match_fcc_hcp_ico` compare against the refdata bound, so these are
        // load-bearing numbers.
        for (name, t, n, num_nodes, expected, max_degree) in [
            (
                "sc",
                &tables::SC_TEMPLATES[0],
                7usize,
                6usize,
                &[4i8, 4, 4, 4, 4, 4][..],
                4,
            ),
            (
                "fcc",
                &tables::FCC_TEMPLATES[0],
                13,
                12,
                &[6i8, 5, 4, 5, 6, 4, 6, 6, 4, 5, 5, 4][..],
                6,
            ),
            (
                "hcp",
                &tables::HCP_TEMPLATES[0],
                13,
                12,
                &[4i8, 6, 4, 5, 6, 6, 4, 6, 4, 6, 5, 4][..],
                6,
            ),
            (
                "ico",
                &tables::ICO_TEMPLATES[0],
                13,
                12,
                &[5i8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5][..],
                5,
            ),
            (
                "bcc",
                &tables::BCC_TEMPLATES[0],
                15,
                14,
                &[3i8, 3, 5, 5, 5, 5, 6, 6, 6, 6, 7, 4, 7, 4][..],
                7,
            ),
        ] {
            let (_, facets, nf) = template_hull(t, n);
            let (degree, got_max) = graph_degree(nf, &facets, num_nodes);
            assert_eq!(got_max, max_degree, "{name}: wrong max degree");
            assert_eq!(&degree[..num_nodes], expected, "{name}: wrong degrees");
            // Incidences must account for every facet exactly three times.
            assert_eq!(
                degree[..num_nodes].iter().map(|d| *d as i32).sum::<i32>(),
                3 * nf as i32,
                "{name}: degrees do not account for all facets"
            );
        }
    }

    #[test]
    fn sc_hull_has_uniform_degree_four() {
        // `match_general` rejects SC unless every node has degree 4.
        let (_, facets, n) = template_hull(&tables::SC_TEMPLATES[0], 7);
        let (degree, max_degree) = graph_degree(n, &facets, 6);
        assert_eq!(max_degree, 4);
        assert!(degree[..6].iter().all(|d| *d == 4));
    }

    #[test]
    fn canonical_form_is_invariant_under_relabelling() {
        // Renaming the nodes must not change the canonical hash: the canonical
        // form is precisely what makes the graph match node-order independent.
        let (_, facets, n) = template_hull(&tables::FCC_TEMPLATES[0], 13);
        let (degree, _) = graph_degree(n, &facets, 12);
        let colours = [0i8; MAX_POINTS];

        let base = canonical_form_coloured(n, &facets, 12, &degree, &colours).expect("manifold");

        // Apply a permutation of the node indices to the facets.
        let perm = [7usize, 0, 5, 11, 2, 9, 1, 8, 3, 10, 4, 6];
        let mut inverse = [0i8; MAX_NBRS];
        for (old, new) in perm.iter().enumerate() {
            inverse[*new] = old as i8;
        }
        let mut relabelled = [[0i8; 3]; MAX_FACETS];
        for (i, f) in relabelled.iter_mut().enumerate().take(n) {
            *f = [
                inverse[facets[i][0] as usize],
                inverse[facets[i][1] as usize],
                inverse[facets[i][2] as usize],
            ];
        }
        let (degree2, _) = graph_degree(n, &relabelled, 12);
        let permuted =
            canonical_form_coloured(n, &relabelled, 12, &degree2, &colours).expect("manifold");
        assert_eq!(base.hash, permuted.hash);
    }

    #[test]
    fn canonical_form_is_invariant_under_facet_winding_and_order() {
        // The hull's own facet order is arbitrary, so the canonical form must
        // not depend on it.
        let (_, facets, n) = template_hull(&tables::FCC_TEMPLATES[0], 13);
        let (degree, _) = graph_degree(n, &facets, 12);
        let colours = [0i8; MAX_POINTS];
        let base = canonical_form_coloured(n, &facets, 12, &degree, &colours).expect("manifold");

        // Reverse the facet order and rotate each facet's winding.
        let mut reversed = [[0i8; 3]; MAX_FACETS];
        for i in 0..n {
            let src = facets[n - 1 - i];
            reversed[i] = [src[1], src[0], src[2]];
        }
        let (degree2, _) = graph_degree(n, &reversed, 12);
        let flipped =
            canonical_form_coloured(n, &reversed, 12, &degree2, &colours).expect("manifold");
        assert_eq!(base.hash, flipped.hash);
    }

    #[test]
    fn canonical_form_separates_distinct_structures() {
        // A canonical form that collided across structures would let a BCC
        // environment be matched against the FCC table.
        let colours = [0i8; MAX_POINTS];
        let mut hashes = Vec::new();
        for (name, t, n, num_nodes) in [
            ("sc", &tables::SC_TEMPLATES[0], 7usize, 6usize),
            ("fcc", &tables::FCC_TEMPLATES[0], 13, 12),
            ("hcp", &tables::HCP_TEMPLATES[0], 13, 12),
            ("ico", &tables::ICO_TEMPLATES[0], 13, 12),
            ("bcc", &tables::BCC_TEMPLATES[0], 15, 14),
        ] {
            let (_, facets, nf) = template_hull(t, n);
            let (degree, _) = graph_degree(nf, &facets, num_nodes);
            let form = canonical_form_coloured(nf, &facets, num_nodes, &degree, &colours)
                .unwrap_or_else(|_| panic!("{name} hull is not a closed manifold"));
            hashes.push((name, form.hash));
        }
        for (i, (ni, hi)) in hashes.iter().enumerate() {
            for (nj, hj) in hashes.iter().skip(i + 1) {
                assert_ne!(hi, hj, "{ni} and {nj} share a canonical hash");
            }
        }
    }

    #[test]
    fn canonical_labelling_is_a_permutation() {
        let (_, facets, n) = template_hull(&tables::BCC_TEMPLATES[0], 15);
        let (degree, _) = graph_degree(n, &facets, 14);
        let colours = [0i8; MAX_POINTS];
        let form = canonical_form_coloured(n, &facets, 14, &degree, &colours).expect("manifold");

        let mut seen: Vec<i8> = form.canonical_labelling[..=14].to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..=14).collect::<Vec<i8>>());
    }
}
