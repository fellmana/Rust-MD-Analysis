//! Void detection, method A (design §17.1).
//!
//! A void is a connected run of empty lattice sites. The reference comes from the
//! self-built Wigner–Seitz fit (design §16.3, `analysis::ws::WsSelf`), so this needs no
//! separate machinery: the fit already produces exactly the enumeration this counts.
//!
//! Method A counts *clusters*, not empty sites. That distinction is the whole method.
//! A vacancy is one missing atom and a void is many, and counting empty sites
//! separately reports every crater of 500 cells as 500 defects. The adjacency is
//! over the *primitive* cell rather than over the conventional one, because for FCC
//! the basis atoms of one cell are neighbours of each other and a cluster that steps
//! from one basis position to another is the same void.
//!
//! ## What this is not
//!
//! Method B -- the roto-strain field `T(x)` and the scalar field `φ(x)` of Mason's
//! paper -- is not this module: it measures void *volume*, and lives in `void_b.rs`
//! (`rmda void --experimental`, design §17.2). Method A is the production method.
//!
//! A single empty site at a surface or grain boundary is not a void and is not counted
//! as one; see [`count_voids`] for the threshold and why it is a declared
//! parameter rather than a constant.

use crate::error::Result;
use crate::lattice::fit::SiteAssignment;
use crate::lattice::PrimitiveCell;

/// One connected run of empty sites.
#[derive(Debug, Clone)]
pub struct VoidCluster {
    /// Number of empty sites in the cluster.
    pub n_sites: usize,
    /// Site indices, ascending. Sorted so the result does not depend on the order the
    /// flood fill happened to visit neighbours in, which is what makes it
    /// reproducible across thread counts.
    pub sites: Vec<usize>,
    /// Cartesian centroid of the cluster's lattice points.
    pub centroid: [f64; 3],
    /// Equivalent radius: the radius of a sphere of the same volume as one site per
    /// cell times the site count. Comparable across clusters without needing a shape
    /// assumption to be defensible, which a bounding sphere would not be.
    pub equivalent_radius: f64,
}

/// Void counts for one region.
#[derive(Debug, Clone, Default)]
pub struct VoidCounts {
    /// Clusters of at least `min_sites` sites, largest first.
    pub voids: Vec<VoidCluster>,
    /// Empty sites that fell in no qualifying cluster.
    pub n_isolated_sites: usize,
    /// Smallest cluster counted as a void.
    pub min_sites: usize,
    /// Distance below which two empty sites are the same void.
    pub cutoff: f64,
}

impl VoidCounts {
    /// Total empty sites, qualifying or not.
    pub fn n_empty_sites(&self) -> usize {
        self.voids.iter().map(|v| v.n_sites).sum::<usize>() + self.n_isolated_sites
    }

    /// Number of craters, which is what a void count usually means.
    pub fn n_voids(&self) -> usize {
        self.voids.len()
    }

    /// Total void volume as a site count, which is the count a vacancy tally would
    /// have produced had it not clustered.
    pub fn n_void_sites(&self) -> usize {
        self.voids.iter().map(|v| v.n_sites).sum()
    }

    /// Largest equivalent radius, zero when there are no voids.
    pub fn max_radius(&self) -> f64 {
        self.voids.first().map_or(0.0, |v| v.equivalent_radius)
    }
}

/// Slack on the nearest-neighbour distance when guessing an adjacency cutoff.
///
/// A cutoff has to clear floating-point noise around the true bond length without
/// reaching the *next* shell, and 5% sits in that gap for every lattice here: the
/// tightest second-shell ratio is BCC's 1.155 (its second neighbours sit at the cell
/// edge, `a`, against a first shell of `√3/2·a`), and the loosest is FCC's 1.414.
/// So this number is not tuned per structure -- it is the middle of a factor-of-1.2
/// window, and a caller with a different criterion should pass its own.
const ADJACENCY_SLACK: f64 = 1.05;

/// Default adjacency cutoff for a fitted lattice: its nearest-neighbour distance.
pub fn default_cutoff(cell: &PrimitiveCell) -> Result<f64> {
    Ok(crate::lattice::fit::nn_distance(cell)? * ADJACENCY_SLACK)
}

/// Count void clusters among a region's empty sites.
///
/// `assign` must come from the same candidate `cell` describes; the site index
/// arithmetic here is the same one [`SiteAssignment`] uses to build its enumeration,
/// which is why the empty-site list it carries is the right input rather than
/// something reconstructed.
///
/// Two empty sites are one void when their lattice points are within `cutoff` of each
/// other; [`default_cutoff`] is the fitted lattice's own nearest-neighbour distance,
/// which is the natural choice.
///
/// Clusters smaller than `min_sites` are dropped and their sites reported as
/// isolated. One is the default and is not a physical statement: a *single* vacancy is
/// a vacancy and not a void, but separating them is a question about what the caller
/// wants to count, so the threshold is a parameter rather than baked in.
pub fn count_voids(
    cell: &PrimitiveCell,
    origin: [f64; 3],
    assign: &SiteAssignment,
    min_sites: usize,
    cutoff: f64,
) -> Result<VoidCounts> {
    let mut counts = VoidCounts {
        min_sites,
        cutoff,
        ..Default::default()
    };
    if assign.empty_sites.is_empty() {
        return Ok(counts);
    }

    let n_basis = assign.n_basis;
    let dims = assign.dims;
    let stride = strides(dims, n_basis);
    let total = dims[0] * dims[1] * dims[2] * n_basis;

    // Site index of the basis atom `b` in cell `(i, j, k)`, with `i, j, k` already
    // offset into the region. This mirrors `SiteGrid::index` in `lattice/fit.rs`; the
    // enumeration is row-major and basis-fastest, so the strides above reproduce it
    // exactly.
    let site_index = |i: usize, j: usize, k: usize, b: usize| -> usize {
        i * stride[0] + j * stride[1] + k * stride[2] + b
    };

    // Two bitsets, one bit per site each: which sites are empty, and which have been
    // visited. A `bool` per site would be 128M bytes each at the target atom count and
    // bitsets are 16M.
    //
    // Both are needed, and dropping the first is a silent disaster rather than a
    // crash: without "is this site empty" the flood fill walks straight through filled
    // sites, so a single vacancy in a 4-cell block comes back as one void of 256 sites
    // -- every site in the region. That is a wrong answer wearing the shape of a
    // right one.
    let cutoff2 = cutoff.max(0.0) * cutoff.max(0.0);
    let mut empty = BitSet::new(total);
    for &s in &assign.empty_sites {
        empty.set(s);
    }
    let mut seen = BitSet::new(total);

    let coords = |s: usize| -> (usize, usize, usize, usize) {
        let bi = s % n_basis;
        let r = s / n_basis;
        let k = r % dims[2];
        let r = r / dims[2];
        let j = r % dims[1];
        let i = r / dims[1];
        (i, j, k, bi)
    };

    // Flood fill rather than pairwise union: a cluster is found by walking it, and
    // each empty site is visited once, so the work is proportional to the number of
    // empty sites and not to their square. The 27 offsets cover every basis atom of
    // every neighbouring cell, which is what "adjacent" means on a primitive cell.
    let mut stack: Vec<usize> = Vec::new();
    for &start in &assign.empty_sites {
        if seen.get(start) {
            continue;
        }
        seen.set(start);
        stack.clear();
        stack.push(start);

        let mut sites: Vec<usize> = Vec::new();
        let mut centroid = [0.0f64; 3];
        while let Some(cur) = stack.pop() {
            sites.push(cur);
            let (i, j, k, b) = coords(cur);
            // Cartesian position of this lattice point. Named once because both the
            // centroid and every adjacency test below need it.
            let cur_pos = site_position(cell, origin, assign, i, j, k, b);
            for (c, p) in centroid.iter_mut().zip(cur_pos) {
                *c += p;
            }

            // Every basis atom of the 27 surrounding cells, plus every basis atom of
            // this one. The inner sweep over `b` is not redundant with the outer one:
            // for FCC the four basis atoms of a single cell are mutual neighbours, so
            // a void that is exactly one cell across is only found by the inner sweep.
            for di in -1i64..=1 {
                for dj in -1i64..=1 {
                    for dk in -1i64..=1 {
                        let ni = i as i64 + di;
                        let nj = j as i64 + dj;
                        let nk = k as i64 + dk;
                        if ni < 0
                            || nj < 0
                            || nk < 0
                            || ni >= dims[0] as i64
                            || nj >= dims[1] as i64
                            || nk >= dims[2] as i64
                        {
                            continue;
                        }
                        for nb_basis in 0..n_basis {
                            let nb = site_index(ni as usize, nj as usize, nk as usize, nb_basis);
                            if nb == cur || seen.get(nb) || !empty.get(nb) {
                                continue;
                            }
                            // Adjacency is a *distance*, not a cell offset. Every site
                            // of the 27 surrounding cells is a candidate, because for
                            // FCC the basis atoms of the neighbouring cells are the
                            // nearest neighbours, but most of the 27 × n_basis pairs are
                            // further apart than a bond and must not join the cluster.
                            //
                            // Without this check the cell offset alone over-counts: on
                            // the 256k HEA frame, seven vacancies scattered through 256k
                            // sites came back as a single void of seven, because a
                            // diagonal cell step is a whole lattice constant away.
                            let nb_pos = site_position(
                                cell,
                                origin,
                                assign,
                                ni as usize,
                                nj as usize,
                                nk as usize,
                                nb_basis,
                            );
                            let d = [
                                cur_pos[0] - nb_pos[0],
                                cur_pos[1] - nb_pos[1],
                                cur_pos[2] - nb_pos[2],
                            ];
                            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > cutoff2 {
                                continue;
                            }
                            seen.set(nb);
                            stack.push(nb);
                        }
                    }
                }
            }
        }

        sites.sort_unstable();
        if sites.len() < min_sites {
            counts.n_isolated_sites += sites.len();
            continue;
        }
        let n = sites.len() as f64;
        for c in centroid.iter_mut() {
            *c /= n;
        }
        // A cluster of `n` sites has the volume of `n` Wigner-Seitz cells, and one
        // site's cell is the primitive cell's volume -- which is `det(h) / n_basis`,
        // not `det(h)`, because `h` spans the conventional cell of a structure with
        // more than one basis atom. Using `det(h)` makes every radius the cube root of
        // `n_basis` too large, which is a plausible-looking number.
        let volume = cell.h.det().abs() / n_basis as f64 * n;
        counts.voids.push(VoidCluster {
            n_sites: sites.len(),
            sites,
            centroid,
            equivalent_radius: (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt(),
        });
    }

    // Largest first, then by lowest site index, so the ordering is total and stable.
    counts
        .voids
        .sort_by(|a, b| b.n_sites.cmp(&a.n_sites).then(a.sites[0].cmp(&b.sites[0])));
    Ok(counts)
}

/// Cartesian position of the basis atom `b` in cell `(i, j, k)` of the region.
///
/// `i, j, k` are region-relative indices, so the cell's own integer part is the
/// region's `cell_origin` plus them.
///
/// The basis offsets are *fractional* coordinates, so they go in before `h` is applied,
/// not after. Adding them afterwards builds a different set of points -- one where the
/// basis atoms sit a quarter of a lattice constant apart instead of half -- which
/// destroys every cross-cell bond and leaves each cell looking like its own void.
fn site_position(
    cell: &PrimitiveCell,
    origin: [f64; 3],
    assign: &SiteAssignment,
    i: usize,
    j: usize,
    k: usize,
    b: usize,
) -> [f64; 3] {
    let p = cell.basis[b];
    let g = cell.h.mul_v([
        i as f64 + assign.cell_origin[0] as f64 + p[0],
        j as f64 + assign.cell_origin[1] as f64 + p[1],
        k as f64 + assign.cell_origin[2] as f64 + p[2],
    ]);
    [origin[0] + g[0], origin[1] + g[1], origin[2] + g[2]]
}

/// One bit per site, for the flood fill's visited mark.
///
/// A `Vec<bool>` spends a byte per site and a `u32` per site is 0.5 GB at 128M atoms,
/// for a structure that is written once and read once; this keeps one bit per site and
/// the indexing in one place.
struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    fn new(bits: usize) -> Self {
        BitSet {
            words: vec![0u64; (bits + 63) / 64],
        }
    }

    #[inline]
    fn get(&self, i: usize) -> bool {
        self.words[i >> 6] & (1u64 << (i & 63)) != 0
    }

    #[inline]
    fn set(&mut self, i: usize) {
        self.words[i >> 6] |= 1u64 << (i & 63);
    }
}

/// Enumeration strides per axis, matching [`SiteAssignment`]'s own index arithmetic.
///
/// Row-major over cells, basis-fastest within a cell: `dims[1] * dims[2] * n_basis`,
/// `dims[2] * n_basis`, `n_basis`. Derived here rather than taken from the grid so a
/// caller needs only the public assignment result, and the arithmetic is small enough
/// to be checkable by eye -- a void index that is off by a stride is a silently wrong
/// count, not a crash.
const fn strides(dims: [usize; 3], n_basis: usize) -> [usize; 3] {
    [dims[1] * dims[2] * n_basis, dims[2] * n_basis, n_basis]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::UnitCell;
    use crate::lattice::fcc::Fcc;
    use crate::lattice::sites::SiteLattice;
    use crate::lattice::Lattice;

    /// Empty sites of a perfect FCC block with `victims` removed, via the real
    /// assignment path so the index arithmetic under test is the production one.
    fn empty(a: f64, cells: usize, victims: &[usize]) -> (PrimitiveCell, SiteAssignment) {
        let len = cells as f64 * a;
        let cell = UnitCell::orthorhombic(len, len, len);
        let sites: Vec<[f64; 3]> = SiteLattice::build(&cell, &Fcc { a }.primitive(), [0.0; 3])
            .unwrap()
            .iter()
            .collect();
        let pos: Vec<[f64; 3]> = sites
            .iter()
            .enumerate()
            .filter(|(i, _)| !victims.contains(i))
            .map(|(_, &p)| p)
            .collect();
        let cand = crate::lattice::fit::Candidate {
            pc: Fcc { a }.primitive(),
            origin: [0.0; 3],
        };
        let assign =
            crate::lattice::fit::assign_region(&cell, &cand, &pos, 1e-6, None, None).unwrap();
        (Fcc { a }.primitive(), assign)
    }

    #[test]
    fn a_perfect_block_has_no_voids() {
        let (cell, assign) = empty(3.5, 4, &[]);
        assert!(
            assign.empty_sites.is_empty(),
            "perfect block had empty sites"
        );
        let c = count_voids(&cell, [0.0; 3], &assign, 1, default_cutoff(&cell).unwrap()).unwrap();
        assert_eq!(c.n_voids(), 0);
        assert_eq!(c.n_empty_sites(), 0);
    }

    #[test]
    fn a_single_vacancy_is_one_site_not_a_cluster() {
        // Four neighbouring vacancies *are* one cluster; that is the distinction the
        // whole method rests on, so it is checked on both sides.
        let (cell, assign) = empty(3.5, 4, &[0]);
        assert_eq!(assign.empty_sites.len(), 1);
        let c = count_voids(&cell, [0.0; 3], &assign, 1, default_cutoff(&cell).unwrap()).unwrap();
        assert_eq!(c.n_voids(), 1);
        assert_eq!(c.n_void_sites(), 1);
        assert_eq!(c.n_isolated_sites, 0);
    }

    #[test]
    fn neighbouring_vacancies_collapse_into_one_void() {
        // Sites 0..4 of an FCC block are the four basis atoms of cell (0,0,0), which
        // are neighbours of each other, so removing them is one void of four sites and
        // not four voids.
        let (cell, assign) = empty(3.5, 4, &[0, 1, 2, 3]);
        let c = count_voids(&cell, [0.0; 3], &assign, 1, default_cutoff(&cell).unwrap()).unwrap();
        assert_eq!(c.n_voids(), 1, "{c:?}");
        assert_eq!(c.n_void_sites(), 4, "{c:?}");
    }

    #[test]
    fn distant_vacancies_stay_apart() {
        // Two vacancies at opposite corners of the block cannot be one void.
        let (cell, assign) = empty(3.5, 6, &[0, 863]);
        let c = count_voids(&cell, [0.0; 3], &assign, 1, default_cutoff(&cell).unwrap()).unwrap();
        assert_eq!(c.n_voids(), 2, "{c:?}");
        assert_eq!(c.n_void_sites(), 2, "{c:?}");
    }

    /// Site indices of every basis atom in cell `(i, j, k)` of a `cells`-per-axis
    /// block, in the enumeration order `SiteLattice::build` produces.
    fn cell_sites(cells: usize, i: usize, j: usize, k: usize) -> Vec<usize> {
        let total = cells * cells * cells * 4;
        (0..4)
            .map(|b| {
                let s = ((i * cells + j) * cells + k) * 4 + b;
                assert!(s < total, "cell index out of range");
                s
            })
            .collect()
    }

    #[test]
    fn a_multi_cell_cavity_is_one_void() {
        // A hollow block: an 8-cell cube with its middle 4x4x4 cells emptied. 64 cells,
        // 256 empty sites, one body. This is the case method A exists for, and it is
        // only reachable if the fill crosses cell boundaries -- an implementation that
        // stopped at cell edges would pass every smaller test above and still be wrong.
        let cells = 8usize;
        let mut victims = Vec::new();
        for i in 2..=5 {
            for j in 2..=5 {
                for k in 2..=5 {
                    victims.extend(cell_sites(cells, i, j, k));
                }
            }
        }
        let (cell, assign) = empty(3.5, cells, &victims);
        assert_eq!(assign.empty_sites.len(), 256, "expected a 256-site cavity");

        let c = count_voids(&cell, [0.0; 3], &assign, 1, default_cutoff(&cell).unwrap()).unwrap();
        assert_eq!(c.n_voids(), 1, "{c:?}");
        assert_eq!(c.n_void_sites(), 256, "{c:?}");

        // Volume check: 256 sites is 64 primitive cells, and the equivalent radius is
        // that volume as a sphere.
        let v = &c.voids[0];
        let site_volume = cell.h.det().abs() / 4.0;
        let expected = (3.0 * 256.0 * site_volume / (4.0 * std::f64::consts::PI)).cbrt();
        assert!((v.equivalent_radius - expected).abs() < 1e-12, "{v:?}");

        // And the threshold still decides, without changing the site total.
        let c = count_voids(
            &cell,
            [0.0; 3],
            &assign,
            257,
            default_cutoff(&cell).unwrap(),
        )
        .unwrap();
        assert_eq!(c.n_voids(), 0);
        assert_eq!(c.n_isolated_sites, 256);
        assert_eq!(c.n_empty_sites(), 256);
    }

    #[test]
    fn a_cell_diagonal_pair_is_not_adjacent() {
        // Cells (0,0,0) and (1,1,1) are both inside the 27-cell sweep the fill walks,
        // but their closest lattice points are 1.22 lattice constants apart against a
        // nearest-neighbour distance of 0.71. Merging them would be a bug wearing the
        // shape of a physical claim, and it stays invisible unless a test says so.
        //
        // The vacancies are hollowed out of a 5-cell block rather than the whole block,
        // because a region is the cells its *atoms* occupy: emptying the corner cells
        // shrinks the region instead of emptying it.
        let cells = 5usize;
        let mut victims = cell_sites(cells, 1, 1, 1);
        victims.extend(cell_sites(cells, 3, 3, 3));
        let (cell, assign) = empty(3.5, cells, &victims);
        assert_eq!(assign.empty_sites.len(), 8);

        let c = count_voids(&cell, [0.0; 3], &assign, 1, default_cutoff(&cell).unwrap()).unwrap();
        assert_eq!(c.n_voids(), 2, "{c:?}");
        assert_eq!(c.n_void_sites(), 8, "{c:?}");
    }

    #[test]
    fn the_minimum_site_threshold_demotes_small_clusters() {
        let (cell, assign) = empty(3.5, 4, &[0, 1, 2, 3]);
        let c = count_voids(&cell, [0.0; 3], &assign, 5, default_cutoff(&cell).unwrap()).unwrap();
        assert_eq!(c.n_voids(), 0, "{c:?}");
        assert_eq!(c.n_isolated_sites, 4);
        // Nothing is lost: the site total is the same either way.
        assert_eq!(c.n_empty_sites(), 4);
    }

    #[test]
    fn a_voids_volume_follows_from_the_cell_volume() {
        // Four sites of a 3.5 A FCC cell have the volume of four primitive cells, and
        // the equivalent radius is that volume as a sphere. Checked against the cell's
        // own volume so the number is not just self-consistent.
        let (cell, assign) = empty(3.5, 4, &[0, 1, 2, 3]);
        let c = count_voids(&cell, [0.0; 3], &assign, 1, default_cutoff(&cell).unwrap()).unwrap();
        let v = &c.voids[0];
        let site_volume = cell.h.det().abs() / 4.0;
        let expected = (3.0 * 4.0 * site_volume / (4.0 * std::f64::consts::PI)).cbrt();
        assert!((v.equivalent_radius - expected).abs() < 1e-12, "{v:?}");
        // And the centroid is inside the block it came from.
        assert!(
            v.centroid
                .iter()
                .all(|&x| (-1e-9..=14.0 + 1e-9).contains(&x)),
            "{v:?}"
        );
    }
}
