//! OVITO's grain segmentation (`GrainSegmentationModifier`, algorithm
//! `GraphClusteringAuto`, OVITO 3.15), behind `SegMode::Auto` (design
//! §15.2a).
//!
//! The steps follow OVITO's `GrainSegmentationEngine` (Larsen's code, MIT
//! licensed) one for one:
//!
//! 1. **Bonds.** Each atom lists its PTM neighbours (the first 12 for FCC/HCP,
//!    14 for BCC, 6 for SC; the 8 nearest for a non-crystalline atom). A pair
//!    `i < j` is bonded when `j` is in `i`'s list: the list of the lower index
//!    decides, as in OVITO.
//! 2. **Coherent interfaces** (with `stacking_faults`). Atoms of the minority
//!    close-packed phase (HCP when FCC is the majority, and vice versa) that
//!    are bonded to a parent-phase atom within 4° are relabelled parent phase
//!    with the orientation mapped into the parent frame, smallest
//!    disorientation first, spreading through the minority phase.
//! 3. **Graph.** Bonds between atoms of the same (adjusted) structure with a
//!    disorientation θ below 4° are graph edges of weight `exp(-θ²/3)`.
//! 4. **Node pair sampling** (Bonald et al. 2018): agglomerative clustering
//!    by the nearest-neighbour-chain algorithm with the distance
//!    `d(a, b) = w(a) w(b) / w(a, b)`, `w(a)` the summed edge weight of `a`.
//!    This gives a dendrogram, sorted by distance.
//! 5. **Automatic threshold.** Per merge, `x = ln(harmonic mean of the two
//!    cluster sizes)`, `y = ln(d)`. A least-absolute-deviations line (IRLS,
//!    100 rounds, weights = merge size) is fitted to `(x, y)`; the threshold is
//!    the largest `y` whose residual is below 1.5 × the median absolute
//!    residual: the knee where merges stop following the within-grain trend.
//! 6. **Cut.** Merges with `ln(d)` up to the threshold are applied; clusters
//!    below `min_size` dissolve.
//! 7. **Orphan adoption** ([`adopt`]): every unassigned atom joins the grain
//!    nearest along bonds (step 1) through unassigned atoms, distance = summed
//!    bond lengths.
//!
//! Serial from step 2 on (as in OVITO), deterministic for any thread count.

use crate::analysis::orient;
use crate::analysis::ptm::{structure, PtmAtom};
use crate::cell::Coord3;
use crate::exec::Par;
use crate::neighbor::{KnnIndex, KnnScratch};
use std::collections::{BinaryHeap, HashMap};

/// OVITO's `_misorientationThreshold`: bonds above this never join, degrees.
const MAX_BOND_ANGLE: f64 = 4.0;
/// OVITO's `MAX_DISORDERED_NEIGHBORS`.
const DISORDERED_NEIGHBORS: usize = 8;
/// OVITO's outlier cutoff in multiples of the median absolute residual.
const KNEE_CUTOFF: f64 = 1.5;

/// A neighbour bond `a < b` of length `len`.
#[derive(Clone, Copy)]
pub(crate) struct Bond {
    pub a: u32,
    pub b: u32,
    pub len: f64,
}

/// Result of steps 1–6.
pub(crate) struct Clustering {
    /// Per atom, the cluster root, or `-1` (non-crystalline or dissolved).
    pub label: Vec<i64>,
    /// Structure after the coherent-interface relabelling.
    pub structure: Vec<i32>,
    /// Orientation after the coherent-interface relabelling.
    pub orientation: Vec<[f64; 4]>,
    /// The merge threshold, `ln(d)`.
    pub threshold: f64,
    /// Clusters left after the cut (before dissolving small ones).
    pub clusters: usize,
    pub bonds: Vec<Bond>,
}

/// PTM first-shell neighbour count, `0` for none.
fn shell(st: i32) -> usize {
    match st {
        structure::FCC | structure::HCP | structure::DHCP => 12,
        structure::BCC => 14,
        structure::SC => 6,
        _ => 0,
    }
}

/// Steps 1–6.
pub(crate) fn cluster(
    index: &KnnIndex,
    positions: &[Coord3],
    ptm: &[PtmAtom],
    min_size: usize,
    stacking_faults: bool,
    par: &Par,
) -> Clustering {
    let n = positions.len();
    let groups: HashMap<i32, Vec<[f64; 4]>> = [
        structure::FCC,
        structure::HCP,
        structure::BCC,
        structure::SC,
        structure::DHCP,
    ]
    .into_iter()
    .filter_map(|s| orient::group(s).map(|g| (s, g)))
    .collect();
    let crystalline = |i: usize| shell(ptm[i].structure()) > 0 && ptm[i].orientation[0].is_finite();
    let list_len = |i: usize| {
        if crystalline(i) {
            shell(ptm[i].structure())
        } else {
            DISORDERED_NEIGHBORS
        }
    };

    // 1. Bonds, from the lower index's neighbour list.
    let bonds: Vec<Bond> = {
        let _t = crate::prof::StageTimer::start("seg.bonds");
        par.map_chunks(n, |lo, hi| {
            let mut s = KnnScratch::default();
            let mut nb = Vec::new();
            let mut out = Vec::new();
            for (i, &p) in positions.iter().enumerate().take(hi).skip(lo) {
                index.knn(i, p, list_len(i), &mut s, &mut nb);
                for &(j, _, d2) in nb.iter() {
                    if (j as usize) > i {
                        out.push(Bond {
                            a: i as u32,
                            b: j,
                            len: d2.sqrt(),
                        });
                    }
                }
            }
            out
        })
        .concat()
    };

    let mut st: Vec<i32> = (0..n)
        .map(|i| {
            if crystalline(i) {
                ptm[i].structure()
            } else {
                structure::OTHER
            }
        })
        .collect();
    let mut q: Vec<[f64; 4]> = (0..n).map(|i| ptm[i].q()).collect();

    // 2. Coherent interfaces.
    if stacking_faults {
        let _t = crate::prof::StageTimer::start("seg.interfaces");
        rotate_interfaces(index, positions, ptm, &bonds, &mut st, &mut q, &groups);
    }

    // 3. Graph edges.
    let edges: Vec<(u32, u32, f64)> = {
        let _t = crate::prof::StageTimer::start("seg.edges");
        par.map_chunks(bonds.len(), |lo, hi| {
            bonds[lo..hi]
                .iter()
                .filter_map(|bd| {
                    let (a, b) = (bd.a as usize, bd.b as usize);
                    if st[a] != st[b] || shell(st[a]) == 0 {
                        return None;
                    }
                    let mut t = orient::disorientation(&q[a], &q[b], &groups[&st[a]]);
                    if t.is_nan() || t >= MAX_BOND_ANGLE {
                        return None;
                    }
                    if t < 1e-5 {
                        t = 0.0;
                    }
                    Some((bd.a, bd.b, (-t * t / 3.0).exp()))
                })
                .collect::<Vec<_>>()
        })
        .concat()
    };

    // 4. Node pair sampling.
    let dendrogram = {
        let _t = crate::prof::StageTimer::start("seg.nps");
        node_pair_sampling(n, &edges)
    };
    drop(edges);

    // 5. Threshold.
    let threshold = knee_threshold(n, &dendrogram);

    // 6. Cut and dissolve.
    let mut uf = Sizes::new(n);
    for &(d, a, b) in &dendrogram {
        if d.ln() > threshold {
            break;
        }
        uf.merge(a, b);
    }
    let mut clusters = 0;
    let mut label = vec![-1i64; n];
    for i in 0..n {
        let r = uf.find(i as u32);
        if r as usize == i && st[i] != structure::OTHER {
            clusters += 1;
        }
        if st[i] != structure::OTHER && uf.size[r as usize] as usize >= min_size {
            label[i] = r as i64;
        }
    }
    Clustering {
        label,
        structure: st,
        orientation: q,
        threshold,
        clusters,
        bonds,
    }
}

/// Min-heap entry of [`rotate_interfaces`]: disorientation, parent atom,
/// minority atom, the minority atom's orientation in the parent frame.
struct Flip(f64, u32, u32, [f64; 4]);

impl PartialEq for Flip {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == std::cmp::Ordering::Equal
    }
}
impl Eq for Flip {}
impl Ord for Flip {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.total_cmp(&self.0)
            .then(o.1.cmp(&self.1))
            .then(o.2.cmp(&self.2))
    }
}
impl PartialOrd for Flip {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

/// OVITO's `rotateInterfaceAtoms`: relabel minority-phase close-packed atoms
/// coherently bonded to the parent phase.
fn rotate_interfaces(
    index: &KnnIndex,
    positions: &[Coord3],
    ptm: &[PtmAtom],
    bonds: &[Bond],
    st: &mut [i32],
    q: &mut [[f64; 4]],
    groups: &HashMap<i32, Vec<[f64; 4]>>,
) {
    let fcc = st.iter().filter(|&&s| s == structure::FCC).count();
    let hcp = st.iter().filter(|&&s| s == structure::HCP).count();
    if fcc == 0 || hcp == 0 {
        return;
    }
    let (parent, minority) = if fcc >= hcp {
        (structure::FCC, structure::HCP)
    } else {
        (structure::HCP, structure::FCC)
    };
    // The bond as (parent atom, minority atom), if it is a coherent interface
    // bond below the angle cutoff, with the rotated minority orientation.
    let check = |st: &[i32], q: &[[f64; 4]], a: u32, b: u32| -> Option<Flip> {
        let (p, m) = match (st[a as usize], st[b as usize]) {
            (x, y) if x == parent && y == minority => (a, b),
            (x, y) if x == minority && y == parent => (b, a),
            _ => return None,
        };
        let (qp, qm) = (&q[p as usize], &q[m as usize]);
        let r = orient::convert(minority, qm, parent, qp);
        let t = orient::disorientation(qp, &r, &groups[&parent]);
        (t < MAX_BOND_ANGLE).then_some(Flip(t, p, m, r))
    };
    let mut heap: BinaryHeap<Flip> = bonds
        .iter()
        .filter_map(|bd| check(st, q, bd.a, bd.b))
        .collect();
    let mut s = KnnScratch::default();
    let mut nb = Vec::new();
    while let Some(f) = heap.pop() {
        // Re-check against the current labels.
        let Some(Flip(_, _, m, r)) = check(st, q, f.1, f.2) else {
            continue;
        };
        if m != f.2 {
            continue;
        }
        let m = m as usize;
        st[m] = parent;
        q[m] = r;
        // The flipped atom's own PTM neighbours (its original structure's shell).
        index.knn(m, positions[m], shell(ptm[m].structure()), &mut s, &mut nb);
        for &(j, _, _) in nb.iter() {
            if let Some(e) = check(st, q, m as u32, j) {
                heap.push(e);
            }
        }
    }
}

/// Union-find with component sizes; roots link by size.
struct Sizes {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl Sizes {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = p;
        }
        x
    }
    fn merge(&mut self, a: u32, b: u32) {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b {
            return;
        }
        if self.size[a as usize] < self.size[b as usize] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b as usize] = a;
        self.size[a as usize] += self.size[b as usize];
    }
}

/// Node pair sampling by the nearest-neighbour chain (OVITO's
/// `node_pair_sampling_clustering`). Returns `(distance, parent, child)` per
/// merge, sorted by distance (stable, so ties keep merge order).
fn node_pair_sampling(n: usize, edges: &[(u32, u32, f64)]) -> Vec<(f64, u32, u32)> {
    // Adjacency lists sorted by neighbour, and node weights.
    let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let mut wnode = vec![0.0f64; n];
    for &(a, b, w) in edges {
        adj[a as usize].push((b, w));
        adj[b as usize].push((a, w));
        wnode[a as usize] += w;
        wnode[b as usize] += w;
    }
    for l in adj.iter_mut() {
        l.sort_unstable_by_key(|e| e.0);
        l.shrink_to_fit();
    }
    let mut active: Vec<bool> = adj.iter().map(|l| !l.is_empty()).collect();
    let mut cursor = 0usize;

    let nearest = |adj: &[Vec<(u32, f64)>], wnode: &[f64], a: u32| -> Option<(f64, u32)> {
        let mut best: Option<(f64, u32)> = None;
        // Ascending neighbour ids: a strict `<` keeps the lowest id on ties.
        for &(v, w) in &adj[a as usize] {
            let d = wnode[v as usize] / w;
            if best.map_or(true, |b| d < b.0) {
                best = Some((d, v));
            }
        }
        best.map(|(d, v)| (d * wnode[a as usize], v))
    };

    let mut dendrogram = Vec::new();
    let mut chain: Vec<u32> = Vec::new();
    loop {
        while cursor < n && !active[cursor] {
            cursor += 1;
        }
        if cursor == n {
            break;
        }
        chain.push(cursor as u32);
        while let Some(a) = chain.pop() {
            match nearest(&adj, &wnode, a) {
                None => {
                    active[a as usize] = false;
                }
                Some((d, b)) => {
                    if let Some(c) = chain.pop() {
                        if b == c {
                            let parent = contract(&mut adj, &mut wnode, a, b);
                            let child = if parent == a { b } else { a };
                            active[child as usize] = false;
                            dendrogram.push((d, parent, child));
                        } else {
                            chain.push(c);
                            chain.push(a);
                            chain.push(b);
                        }
                    } else {
                        chain.push(a);
                        chain.push(b);
                    }
                }
            }
        }
    }
    dendrogram.sort_by(|x, y| x.0.total_cmp(&y.0));
    dendrogram
}

/// Contract edge `a`–`b` into the endpoint with more neighbours; returns it.
fn contract(adj: &mut [Vec<(u32, f64)>], wnode: &mut [f64], a: u32, b: u32) -> u32 {
    let (a, b) = if adj[b as usize].len() > adj[a as usize].len() {
        (b, a)
    } else {
        (a, b)
    };
    let remove = |l: &mut Vec<(u32, f64)>, v: u32| {
        if let Ok(k) = l.binary_search_by_key(&v, |e| e.0) {
            l.remove(k);
        }
    };
    remove(&mut adj[a as usize], b);
    let lb = std::mem::take(&mut adj[b as usize]);
    // Re-point b's neighbours at a.
    for &(v, w) in &lb {
        if v == a {
            continue;
        }
        let lv = &mut adj[v as usize];
        remove(lv, b);
        match lv.binary_search_by_key(&a, |e| e.0) {
            Ok(k) => lv[k].1 += w,
            Err(k) => lv.insert(k, (a, w)),
        }
    }
    // Merge b's list into a's (both sorted by neighbour).
    let la = std::mem::take(&mut adj[a as usize]);
    let mut out = Vec::with_capacity(la.len() + lb.len());
    let (mut i, mut j) = (0, 0);
    while i < la.len() || j < lb.len() {
        if j < lb.len() && lb[j].0 == a {
            j += 1;
            continue;
        }
        if j == lb.len() || (i < la.len() && la[i].0 < lb[j].0) {
            out.push(la[i]);
            i += 1;
        } else if i == la.len() || lb[j].0 < la[i].0 {
            out.push(lb[j]);
            j += 1;
        } else {
            out.push((la[i].0, la[i].1 + lb[j].1));
            i += 1;
            j += 1;
        }
    }
    adj[a as usize] = out;
    wnode[a as usize] += wnode[b as usize];
    a
}

/// OVITO's `ThresholdSelection::Regressor::calculate_threshold` over the
/// sorted dendrogram, in `ln(d)`.
///
/// When the fit is degenerate (fewer than two distinct merge sizes, so no
/// slope), OVITO's arithmetic yields NaN and no inliers, i.e. nothing merges;
/// here every merge is applied instead, so a perfect crystal stays one grain.
fn knee_threshold(n: usize, dendrogram: &[(f64, u32, u32)]) -> f64 {
    if dendrogram.is_empty() {
        return 0.0;
    }
    let mut uf = Sizes::new(n);
    let mut xs = Vec::with_capacity(dendrogram.len());
    let mut ys = Vec::with_capacity(dendrogram.len());
    let mut ws = Vec::with_capacity(dendrogram.len());
    for &(d, a, b) in dendrogram {
        let (ra, rb) = (uf.find(a), uf.find(b));
        let (sa, sb) = (uf.size[ra as usize] as f64, uf.size[rb as usize] as f64);
        let m = 2.0 / (1.0 / sa + 1.0 / sb);
        uf.merge(a, b);
        ws.push(m);
        xs.push(m.ln());
        ys.push(d.ln());
    }
    let (gradient, intercept, residuals) = least_absolute_deviations(&ws, &xs, &ys);
    let mad = median(residuals);
    let mut threshold = 0.0f64;
    let mut inlier = false;
    for (x, y) in xs.iter().zip(&ys) {
        let r = y - (x * gradient + intercept);
        if r < KNEE_CUTOFF * mad {
            threshold = threshold.max(*y);
            inlier = true;
        }
    }
    if !gradient.is_finite() || !intercept.is_finite() || !inlier {
        return f64::INFINITY;
    }
    threshold
}

fn weighted_linear_regression(w: &mut [f64], xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let s: f64 = w.iter().sum();
    for v in w.iter_mut() {
        *v /= s;
    }
    let (mut xm, mut ym) = (0.0, 0.0);
    for i in 0..w.len() {
        xm += w[i] * xs[i];
        ym += w[i] * ys[i];
    }
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for i in 0..w.len() {
        sxx += w[i] * (xs[i] - xm) * (xs[i] - xm);
        sxy += w[i] * (xs[i] - xm) * (ys[i] - ym);
    }
    let g = sxy / sxx;
    (g, ym - g * xm)
}

/// Iteratively reweighted least squares, 100 rounds, as OVITO.
fn least_absolute_deviations(weights: &[f64], xs: &[f64], ys: &[f64]) -> (f64, f64, Vec<f64>) {
    let mut w = weights.to_vec();
    let mut r = vec![0.0; xs.len()];
    let (mut g, mut c) = (0.0, 0.0);
    for _ in 0..100 {
        (g, c) = weighted_linear_regression(&mut w, xs, ys);
        for i in 0..xs.len() {
            r[i] = (ys[i] - (g * xs[i] + c)).abs();
            w[i] = weights[i] / r[i].max(1e-4);
        }
    }
    (g, c, r)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let mut m = v[n / 2];
    if n % 2 == 0 {
        m = (m + v[n / 2 - 1]) / 2.0;
    }
    m
}

/// Heap entry of [`adopt`]: path length, atom, grain; pops the shortest
/// path first, ties by atom then grain.
#[derive(PartialEq)]
struct Front(f64, u32, i64);

impl Eq for Front {}
impl Ord for Front {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.total_cmp(&self.0)
            .then(o.1.cmp(&self.1))
            .then(o.2.cmp(&self.2))
    }
}
impl PartialOrd for Front {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

/// OVITO's `mergeOrphanAtoms`: every unassigned atom (`label < 0`) takes the
/// grain nearest along bonds through unassigned atoms (path length = summed
/// bond lengths). Returns the number adopted.
pub(crate) fn adopt(bonds: &[Bond], label: &mut [i64]) -> usize {
    let n = label.len();
    // Bonds touching an orphan, both directions, as CSR by source atom.
    let mut count = vec![0u32; n + 1];
    for bd in bonds {
        if label[bd.a as usize] < 0 || label[bd.b as usize] < 0 {
            count[bd.a as usize + 1] += 1;
            count[bd.b as usize + 1] += 1;
        }
    }
    for i in 0..n {
        count[i + 1] += count[i];
    }
    let mut fill = count.clone();
    let mut nbr = vec![(0u32, 0.0f64); count[n] as usize];
    let mut heap = BinaryHeap::new();
    for bd in bonds {
        let (a, b) = (bd.a as usize, bd.b as usize);
        if label[a] < 0 || label[b] < 0 {
            nbr[fill[a] as usize] = (bd.b, bd.len);
            fill[a] += 1;
            nbr[fill[b] as usize] = (bd.a, bd.len);
            fill[b] += 1;
        }
        if label[a] >= 0 && label[b] < 0 {
            heap.push(Front(bd.len, bd.b, label[a]));
        } else if label[a] < 0 && label[b] >= 0 {
            heap.push(Front(bd.len, bd.a, label[b]));
        }
    }
    let mut adopted = 0;
    while let Some(Front(d, i, g)) = heap.pop() {
        let i = i as usize;
        if label[i] >= 0 {
            continue;
        }
        label[i] = g;
        adopted += 1;
        for &(j, len) in &nbr[count[i] as usize..count[i + 1] as usize] {
            if label[j as usize] < 0 {
                heap.push(Front(d + len, j, g));
            }
        }
    }
    adopted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nps_merges_the_strongest_pair_first() {
        // A path 0-1-2-3 with a weak middle bond: 0-1 and 2-3 merge before
        // the two pairs join.
        let e = [(0, 1, 1.0), (1, 2, 0.1), (2, 3, 1.0)];
        let d = node_pair_sampling(4, &e);
        assert_eq!(d.len(), 3);
        let first: Vec<(u32, u32)> = d[..2]
            .iter()
            .map(|&(_, p, c)| (p.min(c), p.max(c)))
            .collect();
        assert!(first.contains(&(0, 1)) && first.contains(&(2, 3)), "{d:?}");
        assert!(d[2].0 > d[1].0);
    }

    #[test]
    fn contract_sums_parallel_edges() {
        // Triangle 0-1-2: contracting 0-1 leaves one edge to 2 of weight 2.
        let mut adj = vec![
            vec![(1, 1.0), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0)],
        ];
        let mut w = vec![2.0, 2.0, 2.0];
        let p = contract(&mut adj, &mut w, 0, 1);
        assert_eq!(adj[p as usize], vec![(2, 2.0)]);
        assert_eq!(adj[2], vec![(p, 2.0)]);
        assert_eq!(w[p as usize], 4.0);
    }

    #[test]
    fn adoption_follows_the_shortest_bond_path() {
        // 0 (grain 7) - 1 - 2 - 3 (grain 9): orphan 1 is nearer 0, orphan 2
        // nearer 3.
        let b = [
            Bond {
                a: 0,
                b: 1,
                len: 1.0,
            },
            Bond {
                a: 1,
                b: 2,
                len: 1.0,
            },
            Bond {
                a: 2,
                b: 3,
                len: 0.5,
            },
        ];
        let mut l = vec![7, -1, -1, 9];
        assert_eq!(adopt(&b, &mut l), 2);
        assert_eq!(l, vec![7, 7, 9, 9]);
    }

    #[test]
    fn knee_cuts_between_the_grains_and_their_join() {
        // Two "grains" of 128 nodes, each built by balanced pairwise merges
        // whose distance follows a power law in the merge size (the
        // within-grain trend, with a little scatter), then one join of the
        // two grains far above the trend. The knee keeps every within-grain
        // merge and drops the join.
        let n = 256u32;
        let mut d = Vec::new();
        let mut jitter = 1u32;
        for level in 0..7 {
            let step = 1u32 << level;
            for half in [0, 128] {
                for b in (0..128).step_by(2 * step as usize) {
                    jitter = jitter.wrapping_mul(1103515245).wrapping_add(12345);
                    let f = 1.0 + 0.2 * ((jitter >> 16) % 100) as f64 / 100.0;
                    let dist = 10.0 * (step as f64).powf(1.3) * f;
                    d.push((dist, half + b, half + b + step));
                }
            }
        }
        let inner = d.iter().map(|e| e.0).fold(0.0f64, f64::max);
        d.push((1e4 * inner, 0, 128));
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        let t = knee_threshold(n as usize, &d);
        assert!(
            t >= inner.ln() - 1e-12 && t < (1e4 * inner).ln(),
            "{t} vs {}",
            inner.ln()
        );
    }
}
