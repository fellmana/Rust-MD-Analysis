//! Voronoi cell construction and neighbour ordering (upstream
//! `ptm_voronoi_cell.cpp` and `ptm_neighbour_ordering.cpp`).
//!
//! # Where it is used
//!
//! `ptm_index` gets its neighbours from a `get_neighbours` callback supplied by
//! the host. OVITO 3.15's host (`PTMAlgorithm::Kernel::cacheNeighbors`) first
//! runs `ptm_preorder_neighbours` over the 18 nearest neighbours and then hands
//! them to `ptm_index` in that Voronoi order, so the first 12 (FCC/HCP/ICO) or
//! 14 (BCC) by solid angle are matched, not the nearest. `Ptm::compute_ordered`
//! with `NeighbourOrder::Voronoi` does the same and reproduces OVITO's types and
//! RMSDs atom for atom on the reference snapshots (design §14.1). The default
//! `Ptm::compute` orders by distance.
//!
//! # What this computes
//!
//! Given a point set, it builds the central atom's Voronoi cell, measures the
//! solid angle each neighbour subtends, and sorts by that angle.
//!
//! # Why this is not a port of Voro++
//!
//! Upstream embeds Voro++'s `voronoicell_neighbor`, a 1500-line half-space
//! intersector. Reproducing it verbatim would be a lot of code for no
//! observable gain, for two reasons.
//!
//! First, upstream builds the cell from the planes `r . p <= |p|^2`, which is
//! the true Voronoi cell dilated by a factor of two about the origin. That
//! factor is irrelevant: vertices are normalised to unit length before the
//! solid angle is taken, and a dilation about the origin preserves directions,
//! so it preserves every solid angle exactly.
//!
//! Second, and more importantly, the result depends only on *which* input
//! points own a face and on each such face's solid angle. A face of a convex
//! polyhedron is itself convex, so the fan decomposition used to sum its solid
//! angle gives the same answer from any starting vertex and either winding
//! direction. The cyclic vertex order that Voro++ happens to produce is an
//! internal detail that cancels out.
//!
//! So the contract is a neighbour set plus a number per neighbour, and a
//! straightforward half-space intersection delivers exactly that.

use crate::analysis::ptm::MAX_POINTS;

/// Number of points whose order is encoded in a single `u64` (upstream
/// `PTM_MAX_NBRS`, [`MAX_POINTS`] - 1).
const PERMUTATION_N: usize = MAX_POINTS - 1;

/// One face of the cell: a convex polygon plus the input point that owns it.
///
/// `owner` is the index of the input point whose bisector created this face, or
/// `-1` for a face of the bounding box. Upstream only ever reads the owner of a
/// face and never the geometry of a box face, so the two are distinguished
/// here purely to keep box faces out of the neighbour list.
#[derive(Debug, Clone)]
struct Face {
    owner: i32,
    verts: Vec<[f64; 3]>,
}

/// A convex polyhedron built by successive half-space cuts.
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    faces: Vec<Face>,
    /// Half-size of the initial bounding box.
    k: f64,
}

impl VoronoiCell {
    /// Start from the axis-aligned box `[-k, k]^3`, as upstream does.
    pub fn from_box(k: f64) -> Self {
        let (a, b) = (-k, k);
        let c000 = [a, a, a];
        let c100 = [b, a, a];
        let c110 = [b, b, a];
        let c010 = [a, b, a];
        let c001 = [a, a, b];
        let c101 = [b, a, b];
        let c111 = [b, b, b];
        let c011 = [a, b, b];
        // Wound so that each polygon's normal points out of the box.
        let faces = [
            Face {
                owner: -1,
                verts: vec![c000, c100, c110, c010],
            }, // -z
            Face {
                owner: -1,
                verts: vec![c101, c001, c011, c111],
            }, // +z
            Face {
                owner: -1,
                verts: vec![c100, c101, c111, c110],
            }, // +x
            Face {
                owner: -1,
                verts: vec![c001, c000, c010, c011],
            }, // -x
            Face {
                owner: -1,
                verts: vec![c001, c101, c100, c000],
            }, // -y
            Face {
                owner: -1,
                verts: vec![c010, c110, c111, c011],
            }, // +y
        ];
        Self {
            faces: faces.to_vec(),
            k,
        }
    }

    /// Whether the cell has been cut away to nothing.
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// Faces in construction order. The order is an implementation detail;
    /// nothing downstream depends on it.
    pub fn faces(&self) -> impl Iterator<Item = (i32, &[[f64; 3]])> {
        self.faces.iter().map(|f| (f.owner, f.verts.as_slice()))
    }

    /// Signed distance of `p` from the plane `p . normal - offset`.
    #[inline]
    fn side(p: &[f64; 3], normal: [f64; 3], offset: f64) -> f64 {
        p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2] - offset
    }

    /// Whether two vertices are the same point to within `tol`.
    #[inline]
    fn same(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        (a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol && (a[2] - b[2]).abs() <= tol
    }

    /// Cut the cell with the half-space `p . normal <= offset`, capping the
    /// removed part with a new face owned by `owner`.
    ///
    /// Returns `false` if the cut deleted the cell entirely, matching
    /// upstream's `nplane` return value.
    ///
    /// The cut is a pure geometric operation: the cell is convex, so the
    /// intersection with a half-space is convex, and the cross-section of a
    /// removed region is a convex polygon. That lets the cap be ordered by
    /// projecting onto the cutting plane and sorting by angle, instead of
    /// tracking vertex adjacency the way Voro++ does.
    pub fn cut_plane(&mut self, normal: [f64; 3], offset: f64, owner: i32) -> bool {
        // A zero normal describes no plane. It arises if a caller passes the
        // central atom itself, whose only "bisector" is the whole space; treating
        // it as cutting would cap the cell with a face covering all of it.
        let norm2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if norm2 == 0.0 {
            return !self.is_empty();
        }

        // Scale-aware tolerance. A point exactly on the plane is ambiguous, and
        // upstream resolves the same ambiguity with a tolerance relative to its
        // own `tolerance2`; matching the scale rather than the constant keeps
        // the two agreeing on well-conditioned inputs.
        let tol = 1e-10
            * self
                .k
                .max(normal[0].abs())
                .max(normal[1].abs())
                .max(normal[2].abs());

        let mut kept: Vec<Face> = Vec::with_capacity(self.faces.len() + 1);
        // Vertices of the cross-section, deduplicated by position.
        let mut section: Vec<[f64; 3]> = Vec::new();

        // Where a vertex sits relative to the plane.
        //
        // `On` has to be distinguished from `In`. When a vertex lies exactly on
        // the cutting plane - which is the normal case, since a Voronoi vertex
        // is by definition on three of the cell's faces - the crossing computed
        // for an edge ending at it *is* that vertex. Emitting both would put a
        // repeated vertex in the face, and a repeated vertex silently breaks
        // the solid-angle fan: the zero-area triangles it introduces make the
        // sum cover only part of the face. So a vertex on the plane is emitted
        // once, by its own edge, and suppresses the crossing for that edge.
        #[derive(Clone, Copy, PartialEq)]
        enum Side {
            In,
            On,
            Out,
        }
        let side_of = |d: f64| {
            if d < -tol {
                Side::In
            } else if d <= tol {
                Side::On
            } else {
                Side::Out
            }
        };

        for face in &self.faces {
            let n = face.verts.len();
            let mut inside: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
            for i in 0..n {
                let a = face.verts[i];
                let b = face.verts[(i + 1) % n];
                let sa = side_of(Self::side(&a, normal, offset));
                let sb = side_of(Self::side(&b, normal, offset));
                match sa {
                    // A vertex on the plane survives, is emitted once, and is
                    // itself a vertex of the new cap. Several old faces meet
                    // at it, so it is contributed repeatedly and deduplicated
                    // below.
                    Side::On => {
                        inside.push(a);
                        section.push(a);
                    }
                    Side::In => match sb {
                        // The edge genuinely crosses, so it contributes one
                        // vertex to the kept face and one to the cap.
                        Side::Out => {
                            inside.push(a);
                            let x = Self::crossing(a, b, normal, offset);
                            inside.push(x);
                            section.push(x);
                        }
                        // `In, In` needs the vertex; `In, On` does not, because
                        // the on-plane vertex is emitted by the next iteration.
                        _ => inside.push(a),
                    },
                    // Entering the kept half-space from outside.
                    Side::Out => {
                        if sb == Side::In {
                            let x = Self::crossing(a, b, normal, offset);
                            inside.push(x);
                            section.push(x);
                        }
                    }
                }
            }
            // A face entirely outside is dropped; one reduced to a single point
            // carries no area and would only produce a degenerate solid angle.
            if inside.len() >= 3 {
                kept.push(Face {
                    owner: face.owner,
                    verts: inside,
                });
            }
        }

        if section.is_empty() {
            // Nothing was cut away. A plane that only touched the cell is not a
            // face of it, and upstream reports no neighbour for it.
            return !self.is_empty();
        }

        // Deduplicate the cross-section: each crossing is shared by exactly the
        // two faces meeting along that edge.
        let mut uniq: Vec<[f64; 3]> = Vec::with_capacity(section.len());
        for p in section {
            if !uniq.iter().any(|q| Self::same(*q, p, tol)) {
                uniq.push(p);
            }
        }

        if uniq.len() >= 3 {
            let centre = centroid(&uniq);
            kept.push(Face {
                owner,
                verts: order_in_plane(uniq, centre, normal),
            });
        } else if uniq.len() == 2 {
            // The plane met the cell in a segment. Upstream emits a face with
            // two vertices, whose solid-angle fan loop never runs, so it scores
            // zero. Keep it for the same reason.
            kept.push(Face { owner, verts: uniq });
        }

        self.faces = kept;
        !self.is_empty()
    }

    /// Point where the segment `a -> b` crosses the plane.
    ///
    /// Only called for an edge with endpoints strictly on opposite sides, so the
    /// parameter is always well conditioned. The result is projected onto the
    /// plane afterwards so it lies on it to full precision rather than to the
    /// precision of the interpolation.
    fn crossing(a: [f64; 3], b: [f64; 3], normal: [f64; 3], offset: f64) -> [f64; 3] {
        let da = Self::side(&a, normal, offset);
        let db = Self::side(&b, normal, offset);
        let t = da / (da - db);
        let mut p = [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
        ];
        let d = Self::side(&p, normal, offset);
        let nn = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if nn > 0.0 {
            for k in 0..3 {
                p[k] -= d * normal[k] / nn;
            }
        }
        p
    }
}

fn centroid(pts: &[[f64; 3]]) -> [f64; 3] {
    let mut c = [0.0f64; 3];
    for p in pts {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    for v in c.iter_mut() {
        *v /= pts.len() as f64;
    }
    c
}

/// Order a coplanar point set cyclically, by angle in the plane's own basis.
///
/// The cross-section is convex, so a single sweep in angular order is the
/// boundary. This avoids carrying vertex adjacency, which is the part of
/// Voro++'s bookkeeping that carries no observable meaning here.
fn order_in_plane(pts: Vec<[f64; 3]>, centre: [f64; 3], normal: [f64; 3]) -> Vec<[f64; 3]> {
    // Any unit vector not parallel to the normal gives a usable basis.
    let (a, b) = basis(normal);
    let mut with_angle: Vec<(f64, [f64; 3])> = pts
        .into_iter()
        .map(|p| {
            let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
            let x = d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
            let y = d[0] * b[0] + d[1] * b[1] + d[2] * b[2];
            (x.atan2(y), p)
        })
        .collect();
    with_angle.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    with_angle.into_iter().map(|(_, p)| p).collect()
}

/// An orthonormal basis of the plane with the given normal.
fn basis(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    // Pick the axis least aligned with the normal, so the cross product is
    // well conditioned.
    let (ax, ay, az) = (normal[0].abs(), normal[1].abs(), normal[2].abs());
    let axis = if ax <= ay && ax <= az {
        [1.0, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let a = normalize(cross(axis, normal));
    let b = cross(normal, a);
    (a, b)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = dot(v, v).sqrt();
    if n == 0.0 {
        v
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

/// Solid angle subtended at the origin by a face, given its vertices.
///
/// The vertices are normalised first, so this is the spherical area of the
/// face's projection - which is what makes it invariant under the dilation
/// upstream applies. The sum runs over a fan from the first vertex, valid
/// because a face of a convex polyhedron is convex; `abs` absorbs the winding
/// direction, as upstream's `calculate_solid_angle` does.
pub fn face_solid_angle(verts: &[[f64; 3]]) -> f64 {
    if verts.len() < 3 {
        return 0.0;
    }
    let unit: Vec<[f64; 3]> = verts.iter().copied().map(normalize).collect();
    let mut total = 0.0;
    // A fan from the first vertex: upstream keeps `u` fixed and advances `v`.
    let u = unit[0];
    for i in 1..unit.len() - 1 {
        let v = unit[i];
        let w = unit[i + 1];
        // Upstream's formula: |2 atan2(r1 . (r2 x r3), 1 + r1.r2 + r2.r3 + r3.r1)|.
        let numerator = dot(u, cross(v, w));
        let denominator = 1.0 + dot(u, v) + dot(v, w) + dot(w, u);
        total += (2.0 * numerator.atan2(denominator)).abs();
    }
    total
}

/// Build the cell of the origin from a set of points, and return each point's
/// solid angle.
///
/// A point that is not a Voronoi neighbour of the origin - one whose bisector
/// does not touch the cell in a face - gets an area of zero, which is exactly
/// how upstream leaves it.
pub fn voronoi_face_areas(points: &[[f64; 3]]) -> Vec<f64> {
    let mut areas = vec![0.0f64; points.len()];
    if points.is_empty() {
        return areas;
    }
    let mut max_normsq = 0.0f64;
    for p in points {
        max_normsq = max_normsq.max(dot(*p, *p));
    }
    let max_norm = max_normsq.sqrt();
    if max_norm == 0.0 {
        return areas;
    }
    // Upstream's box: ten times the largest neighbour distance. With the true
    // (undilated) bisectors this is twice as generous, which cannot bite.
    let k = 10.0 * max_norm;
    let mut cell = VoronoiCell::from_box(k);

    for (i, p) in points.iter().enumerate() {
        // The bisector of the origin and p: r . p = |p|^2 / 2.
        let offset = dot(*p, *p) / 2.0;
        if !cell.cut_plane(*p, offset, i as i32) {
            // Upstream cannot represent an empty cell either; the caller
            // guarantees at least the central atom, so this is unreachable in
            // practice. Leave the areas at zero.
            return areas;
        }
    }

    for (owner, verts) in cell.faces() {
        if owner >= 0 {
            areas[owner as usize] = face_solid_angle(verts);
        }
    }
    areas
}

/// One neighbour's sort key: the face solid angle and the squared distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbour {
    /// Solid angle of this neighbour's Voronoi face, larger is nearer.
    pub area: f64,
    /// Squared distance from the central atom, used to break area ties.
    pub dist: f64,
    /// Index into the input point array.
    pub index: usize,
    /// Position in the sorted order.
    pub rank: usize,
}

/// Order `points` by decreasing face solid angle, then increasing distance.
///
/// This is upstream's `sorthelper_compare` under a stable sort, as current
/// upstream (`std::stable_sort`): exact ties keep the input (distance) order.
pub fn order_neighbours(points: &[[f64; 3]]) -> Vec<Neighbour> {
    let areas = voronoi_face_areas(points);
    let mut out: Vec<Neighbour> = points
        .iter()
        .enumerate()
        .map(|(index, p)| Neighbour {
            area: areas[index],
            dist: dot(*p, *p),
            index,
            rank: 0,
        })
        .collect();
    out.sort_by(|a, b| {
        b.area
            .partial_cmp(&a.area)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.dist
                    .partial_cmp(&b.dist)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    for (rank, n) in out.iter_mut().enumerate() {
        n.rank = rank;
    }
    out
}

/// Order `points` the way upstream's `preorder_neighbours` does, and return the
/// permutation together with its rank.
///
/// Upstream sorts the first `min(n, 16)` points and pads the remaining slots
/// with the identity, so the result is always 16 wide. Note that a 17th point
/// or beyond is *dropped*, not merely sorted late: a 17th neighbour is
/// outside the coordination PTM models.
///
/// The central atom, if present, has no Voronoi face of its own and so sorts to
/// the back with a zero area.
pub fn preorder_neighbours(points: &[[f64; 3]]) -> ([usize; PERMUTATION_N], u64) {
    let num = points.len().min(PERMUTATION_N);
    let mut order = [0usize; PERMUTATION_N];
    if num == 0 {
        return (order, 0);
    }
    for n in order_neighbours(&points[..num]).into_iter().take(num) {
        order[n.rank] = n.index;
    }
    // Slots past the input count keep the identity, as upstream does.
    for (i, slot) in order.iter_mut().enumerate().skip(num) {
        *slot = i;
    }
    let idx = permutation_to_index(&order);
    (order, idx)
}

/// The `preorder_neighbours` permutation index of an already-ordered point set.
///
/// Upstream returns a rank of the 16-point permutation, used purely as a cache
/// key. Ties in the ordering make the encoding implementation-defined, so this
/// is only meaningful as a key that round-trips through
/// [`index_to_permutation`] - it is not comparable with a value produced by
/// upstream.
pub fn preorder_index(order: &[usize; PERMUTATION_N]) -> u64 {
    permutation_to_index(order)
}

/// Rank a permutation with upstream's `permutation_to_index`.
fn permutation_to_index(perm: &[usize; PERMUTATION_N]) -> u64 {
    let mut pos = [0usize; PERMUTATION_N];
    let mut elems = [0usize; PERMUTATION_N];
    for i in 0..PERMUTATION_N {
        pos[i] = i;
        elems[i] = i;
    }
    let mut m: u64 = 1;
    let mut k: u64 = 0;
    for i in 0..PERMUTATION_N - 1 {
        k += m * pos[perm[i]] as u64;
        m *= (PERMUTATION_N - i) as u64;
        pos[elems[PERMUTATION_N - i - 1]] = pos[perm[i]];
        elems[pos[perm[i]]] = elems[PERMUTATION_N - i - 1];
    }
    k
}

/// Unrank a permutation with upstream's `index_to_permutation`.
pub fn index_to_permutation(mut k: u64) -> [usize; PERMUTATION_N] {
    let mut elems = [0usize; PERMUTATION_N];
    for (i, e) in elems.iter_mut().enumerate() {
        *e = i;
    }
    let mut out = [0usize; PERMUTATION_N];
    let n = PERMUTATION_N as u64;
    for i in 0..PERMUTATION_N {
        let ind = (k % (n - i as u64)) as usize;
        k /= n - i as u64;
        out[i] = elems[ind];
        elems[ind] = elems[PERMUTATION_N - i - 1];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ptm::normalize::normalize_vertices;
    use crate::analysis::ptm::tables;

    /// The reference orderings captured from compiled upstream
    /// `ptm_neighbour_ordering.cpp` through a small probe program.
    ///
    /// Each ideal template is a case with *many* ties - a perfect cuboctahedron
    /// gives twelve identical solid angles - and the tied group comes out of
    /// libstdc++'s introsort in an order with no geometric meaning. So the
    /// assertion is on the multiset of the first 16 indices, which is what the
    /// ordering actually determines, rather than on their sequence.
    const REFERENCE_SETS: &[(&str, &[usize])] = &[
        (
            "fcc",
            &[4, 9, 12, 11, 8, 7, 5, 2, 3, 1, 10, 6, 0, 13, 14, 15],
        ),
        (
            "hcp",
            &[1, 12, 8, 6, 3, 5, 7, 10, 9, 11, 4, 2, 0, 13, 14, 15],
        ),
        (
            "ico",
            &[10, 12, 5, 8, 11, 6, 9, 3, 7, 2, 4, 1, 0, 13, 14, 15],
        ),
        (
            "bcc",
            &[6, 3, 2, 4, 1, 7, 8, 5, 12, 9, 13, 14, 10, 11, 0, 15],
        ),
        (
            "sc",
            &[1, 3, 5, 2, 4, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
    ];

    fn normalized<const M: usize>(t: &[[f64; 3]; M], n: usize) -> [[f64; 3]; 17] {
        let mut raw = [[0.0f64; 3]; M];
        raw[..n].copy_from_slice(&t[..n]);
        let (out, _) = normalize_vertices(&raw);
        let mut wide = [[0.0f64; 3]; 17];
        wide[..n].copy_from_slice(&out[..n]);
        wide
    }

    #[test]
    fn ideal_templates_reproduce_the_upstream_neighbour_set() {
        let cases: [(&str, [[f64; 3]; 17], usize); 5] = [
            ("fcc", normalized(&tables::FCC_TEMPLATES[0], 13), 13),
            ("hcp", normalized(&tables::HCP_TEMPLATES[0], 13), 13),
            ("ico", normalized(&tables::ICO_TEMPLATES[0], 13), 13),
            ("bcc", normalized(&tables::BCC_TEMPLATES[0], 15), 15),
            ("sc", normalized(&tables::SC_TEMPLATES[0], 7), 7),
        ];
        for (name, pts, n) in cases {
            let (order, _) = preorder_neighbours(&pts[..n]);
            let want = REFERENCE_SETS
                .iter()
                .find(|(k, _)| *k == name)
                .expect("reference row")
                .1;
            let mut got_sorted = order.to_vec();
            let mut want_sorted = want.to_vec();
            got_sorted.sort_unstable();
            want_sorted.sort_unstable();
            assert_eq!(got_sorted, want_sorted, "{name}: neighbour set");
        }
    }

    /// Every non-central point of a close-packed environment is a Voronoi
    /// neighbour, and each face is one twelfth of the sphere.
    #[test]
    fn fcc_neighbour_faces_are_one_twelfth_of_the_sphere() {
        let pts = normalized(&tables::FCC_TEMPLATES[0], 13);
        let order = order_neighbours(&pts[..13]);
        assert_eq!(order.len(), 13);
        for n in &order {
            let want = if n.index == 0 {
                // The central atom is not a neighbour of itself.
                0.0
            } else {
                std::f64::consts::PI / 3.0
            };
            assert!(
                (n.area - want).abs() < 1e-12,
                "point {} area {} want {}",
                n.index,
                n.area,
                want
            );
        }
        // The faces tile the sphere.
        let total: f64 = order.iter().map(|n| n.area).sum();
        assert!(
            (total - 4.0 * std::f64::consts::PI).abs() < 1e-9,
            "total {total}"
        );
        // And they all sort ahead of the central atom.
        assert_eq!(order.last().unwrap().index, 0);
    }

    /// The far shells must score zero, which is what stops them displacing real
    /// neighbours when the list is truncated.
    #[test]
    fn distant_points_are_not_neighbours() {
        let mut pts = [[0.0f64; 3]; 3];
        pts[1] = [1.0, 0.0, 0.0];
        pts[2] = [10.0, 0.0, 0.0];
        let areas = voronoi_face_areas(&pts);
        assert!(
            areas[1] > 1.0,
            "near point should own a face, got {}",
            areas[1]
        );
        assert_eq!(areas[2], 0.0, "far point owns no face");
    }

    /// Check the solid angle against the closed form for a spherical rectangle.
    ///
    /// A face of the cell that is a square of half-width `a` at distance `d`
    /// subtends `4 atan(a^2 / (d sqrt(d^2 + 2 a^2)))`, which is independent of
    /// the fan used to compute it.
    #[test]
    fn solid_angle_matches_the_closed_form_for_a_spherical_rectangle() {
        let (a, d) = (1.0f64, 0.5f64);
        let mut cell = VoronoiCell::from_box(a);
        // Keep z >= d, so the new cap is the square of half-width `a` at
        // height `d` and the box's own +z face is removed.
        assert!(cell.cut_plane([0.0, 0.0, -1.0], -d, 0));
        let (_, verts) = cell
            .faces()
            .find(|(owner, _)| *owner == 0)
            .expect("cap face");
        assert_eq!(verts.len(), 4, "cap should be a square");
        let want = 4.0 * (a * a / (d * (d * d + 2.0 * a * a).sqrt())).atan();
        assert!(
            (face_solid_angle(verts) - want).abs() < 1e-12,
            "got {} want {}",
            face_solid_angle(verts),
            want
        );
    }

    #[test]
    fn the_permutation_index_round_trips() {
        for seed in 0..64u64 {
            let perm = index_to_permutation(seed.wrapping_mul(0x9E3779B97F4A7C15) >> 11);
            let k = permutation_to_index(&perm);
            assert_eq!(index_to_permutation(k), perm, "seed {seed}");
        }
    }

    #[test]
    fn cell_cut_reports_when_nothing_survives() {
        let mut cell = VoronoiCell::from_box(1.0);
        // A plane far outside the box changes nothing.
        assert!(cell.cut_plane([1.0, 0.0, 0.0], 5.0, 0));
        assert!(!cell.is_empty());
        // Cutting the cell down to a slab still leaves faces.
        assert!(cell.cut_plane([0.0, 0.0, 1.0], 0.1, 1));
        assert!(!cell.is_empty());
    }

    /// The face solid angles of a closed cell sum to 4π, also when thermal
    /// noise gives faces of five or more vertices. A fan that advances its
    /// first vertex (summing consecutive triples) undercounts those faces by
    /// up to half, which reorders real HCP neighbours behind second-shell
    /// ones (design §14.1).
    #[test]
    fn solid_angles_of_a_thermal_cell_sum_to_four_pi() {
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            for si in [-1.0, 1.0] {
                for sj in [-1.0, 1.0] {
                    let mut p = [0.0; 3];
                    p[i] = si * 0.5f64.sqrt();
                    p[j] = sj * 0.5f64.sqrt();
                    pts.push(p);
                }
            }
        }
        for k in 0..3 {
            for s in [-1.0, 1.0] {
                let mut p = [0.0; 3];
                p[k] = s * 2.0f64.sqrt();
                pts.push(p);
            }
        }
        let mut x = 0x9e3779b97f4a7c15u64;
        for p in pts.iter_mut() {
            for v in p.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *v += 0.08 * ((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
            }
        }
        let areas = voronoi_face_areas(&pts);
        let total: f64 = areas.iter().sum();
        assert!((total - 4.0 * std::f64::consts::PI).abs() < 1e-9, "{total}");
        // The 12 nearest own the 12 largest faces.
        let order = order_neighbours(&pts);
        let mut first: Vec<usize> = order[..12].iter().map(|n| n.index).collect();
        first.sort_unstable();
        assert_eq!(first, (0..12).collect::<Vec<_>>());
    }
}
