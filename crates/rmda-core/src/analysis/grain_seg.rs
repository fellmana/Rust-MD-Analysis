//! Grain segmentation: crystallites, then a merge graph (design §15).
//!
//! 1. **Crystallites.** Each crystalline atom is bonded to its first-shell
//!    nearest neighbours of the same structure (12 for FCC/HCP, 14 for BCC, 6 for SC)
//!    when their disorientation is below `core_angle`. A concurrent
//!    union-find over those bonds gives orientation-coherent patches.
//! 2. **Crystallite graph.** Bonds between different crystallites are
//!    aggregated per pair, and each crystallite gets a symmetry-aligned mean
//!    orientation.
//! 3. **Merge.** Pairs are merged in order of the disorientation between their
//!    *current mean* orientations, while it is below `threshold` and the pair
//!    shares at least `min_bonds` bonds. Comparing means rather than following
//!    chains of atom bonds is what stops a gradual chain of sub-threshold bonds
//!    from bridging a real boundary. With `stacking_faults`, FCC
//!    and HCP crystallites are compared too, through the close-packed frame
//!    map (`orient::convert`); a merged cluster keeps the structure of its
//!    larger part. A stacking fault thus stays inside its FCC grain, and a
//!    twin plane joins one twin while the twins (60° apart) stay separate.
//! 4. **Cleanup.** Grains below `min_size` dissolve. Unassigned atoms
//!    (non-crystalline included) take the grain nearest to them along a path
//!    through unassigned atoms (multi-source Dijkstra over 12-nearest-neighbour
//!    bonds). This reproduces OVITO's orphan adoption on the HEA polycrystal
//!    fixture for 99.4 % of atoms, where a neighbour majority vote gets 97.9 %.
//!    Grains are numbered by decreasing size, ties by lowest atom index.
//!
//! Every pass is deterministic for any thread count: union-find roots are the
//! minimum atom index of a component, and per-chunk sums merge in chunk order.

use crate::analysis::grain_auto;
use crate::analysis::orient::{self, MeanAcc};
use crate::analysis::ptm::{structure, PtmAtom};
use crate::cell::{Coord3, UnitCell};
use crate::error::Result;
use crate::exec::Par;
use crate::neighbor::{KnnIndex, KnnScratch};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Merge algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SegMode {
    /// Crystallites merged on their mean orientations below `threshold`
    /// (steps 1–3 above), adoption over 12-nearest-neighbour bonds.
    #[default]
    Crystallite,
    /// OVITO 3.15's `GraphClusteringAuto` (`grain_auto`): node pair sampling
    /// over PTM-neighbour bonds, the merge threshold picked automatically from
    /// the dendrogram, OVITO's orphan adoption. `threshold`, `core_angle` and
    /// `min_bonds` are not used.
    Auto,
}

/// Settings.
#[derive(Debug, Clone)]
pub struct SegConfig {
    /// Merge algorithm.
    pub mode: SegMode,
    /// Bonds below this disorientation join a crystallite, degrees.
    pub core_angle: f64,
    /// Crystallites whose mean orientations differ by less than this merge,
    /// degrees.
    pub threshold: f64,
    /// A pair of crystallites needs at least this many bonds to merge.
    pub min_bonds: u32,
    /// Grains with fewer atoms dissolve.
    pub min_size: usize,
    /// Hand unassigned atoms to neighbouring grains.
    pub adopt: bool,
    /// Let FCC and HCP crystallites merge through the close-packed frame map
    /// (design §11.3, §15.2 step 4): HCP stacking-fault layers join the FCC
    /// grain around them, and FCC layers the HCP grain. Twin planes join one
    /// of the two twins; the twins stay apart (60° between their means).
    pub stacking_faults: bool,
    /// With `stacking_faults` and dHCP labels (`--dhcp`): let the stacking
    /// faults of a dHCP crystal (thin FCC/HCP runs between dHCP layers,
    /// `stacking::dhcp_faults`) join its grain, so a faulted dHCP crystal is
    /// one grain. Off by default: the defect census cannot count such a
    /// grain (the local reference reported 21–77 false interstitials beside
    /// a single fault where the split grains are exact), so `rmda defects`
    /// never sets it.
    pub dhcp_faults: bool,
}

impl Default for SegConfig {
    fn default() -> Self {
        Self {
            mode: SegMode::Crystallite,
            core_angle: 2.0,
            threshold: 5.0,
            min_bonds: 4,
            min_size: 100,
            adopt: true,
            stacking_faults: true,
            dhcp_faults: false,
        }
    }
}

/// One grain.
#[derive(Debug, Clone)]
pub struct GrainInfo {
    pub id: i32,
    pub size: usize,
    pub structure: i32,
    /// Symmetry-aligned mean orientation `(w, x, y, z)`.
    pub orientation: [f64; 4],
    pub crystallites: usize,
}

/// A pair of grains that share bonds.
#[derive(Debug, Clone)]
pub struct GrainPair {
    pub a: i32,
    pub b: i32,
    pub bonds: u32,
    /// Disorientation between the two grains' mean orientations, degrees
    /// (`NaN` for different structures).
    pub disorientation: f64,
}

/// Result of a segmentation.
#[derive(Debug, Clone, Default)]
pub struct Segmentation {
    /// Per atom, `-1` for none.
    pub grain: Vec<i32>,
    pub grains: Vec<GrainInfo>,
    pub pairs: Vec<GrainPair>,
    pub n_crystallites: usize,
    /// Atoms that took a grain in the adoption pass.
    pub adopted: usize,
    /// `SegMode::Auto`: the automatic merge threshold, `ln` of the node pair
    /// sampling distance (OVITO's `GrainSegmentation.auto_merge_threshold`).
    pub merge_threshold: Option<f64>,
}

/// PTM neighbour count of a structure's first shell, as grain bonds use it.
fn shell(st: i32) -> usize {
    match st {
        structure::FCC | structure::HCP | structure::DHCP => 12,
        structure::BCC => 14,
        structure::SC => 6,
        _ => 0,
    }
}

fn find(p: &[AtomicU32], mut x: u32) -> u32 {
    loop {
        let px = p[x as usize].load(Ordering::Relaxed);
        if px == x {
            return x;
        }
        let gp = p[px as usize].load(Ordering::Relaxed);
        let _ = p[x as usize].compare_exchange(px, gp, Ordering::Relaxed, Ordering::Relaxed);
        x = px;
    }
}

/// Link the larger root under the smaller, so every root is its component's
/// minimum index whatever the order of unions.
fn union(p: &[AtomicU32], a: u32, b: u32) {
    loop {
        let (ra, rb) = (find(p, a), find(p, b));
        if ra == rb {
            return;
        }
        let (hi, lo) = if ra > rb { (ra, rb) } else { (rb, ra) };
        if p[hi as usize]
            .compare_exchange(hi, lo, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
    }
}

/// Serial union-find over crystallites, with per-root aligned means.
struct Clusters {
    parent: Vec<u32>,
}

impl Clusters {
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = p;
        }
        x
    }
}

/// Heap entry of [`adopt`]: path length, atom, grain. Ordered so the
/// `BinaryHeap` pops the shortest path first, ties by atom then grain.
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

/// Give every unassigned atom (`label < 0`) the grain with the shortest path
/// to it, a path being a chain of 12-nearest-neighbour bonds through
/// unassigned atoms. Atoms with no such path stay unassigned. Returns the
/// number adopted.
///
/// Seeds (an orphan's nearest assigned neighbour) are found in parallel; the
/// growth is a serial Dijkstra that touches only orphans and stores 16 B per
/// orphan.
fn adopt(index: &KnnIndex, positions: &[Coord3], label: &mut [i64], par: &Par) -> usize {
    const K: usize = 12;
    let orphans: Vec<u32> = (0..label.len() as u32)
        .filter(|&i| label[i as usize] < 0)
        .collect();
    if orphans.is_empty() {
        return 0;
    }
    let lab: &[i64] = label;
    let seeds: Vec<(f64, i64)> = par
        .map_chunks(orphans.len(), |lo, hi| {
            let mut s = KnnScratch::default();
            let mut nb = Vec::new();
            orphans[lo..hi]
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    index.knn(i, positions[i], K, &mut s, &mut nb);
                    nb.iter()
                        .filter(|e| lab[e.0 as usize] >= 0)
                        .map(|e| (e.2.sqrt(), lab[e.0 as usize]))
                        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                        .unwrap_or((f64::INFINITY, -1))
                })
                .collect::<Vec<_>>()
        })
        .concat();
    let mut best: Vec<f64> = seeds.iter().map(|s| s.0).collect();
    let mut heap: std::collections::BinaryHeap<Front> = orphans
        .iter()
        .zip(&seeds)
        .filter(|(_, s)| s.1 >= 0)
        .map(|(&i, s)| Front(s.0, i, s.1))
        .collect();
    drop(seeds);
    let mut done = vec![false; orphans.len()];
    let slot = |i: u32| orphans.binary_search(&i).ok();
    let mut s = KnnScratch::default();
    let mut nb = Vec::new();
    let mut adopted = 0;
    while let Some(Front(d, i, g)) = heap.pop() {
        let k = slot(i).expect("heap holds orphans only");
        if done[k] || d > best[k] {
            continue;
        }
        done[k] = true;
        label[i as usize] = g;
        adopted += 1;
        index.knn(i as usize, positions[i as usize], K, &mut s, &mut nb);
        for e in nb.iter() {
            if let Some(m) = slot(e.0) {
                let dj = d + e.2.sqrt();
                if !done[m] && dj < best[m] {
                    best[m] = dj;
                    heap.push(Front(dj, e.0, g));
                }
            }
        }
    }
    adopted
}

/// Segment a frame from PTM structure and orientation.
pub fn segment(
    cell: &UnitCell,
    positions: &[Coord3],
    ptm: &[PtmAtom],
    cfg: &SegConfig,
    par: &Par,
) -> Result<Segmentation> {
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
    // With `dhcp_faults`, the stacking faults of dHCP crystals (thin FCC/HCP
    // runs between dHCP layers) are segmented as dHCP atoms, in the dHCP
    // frame, so they join the crystallite around them through the core
    // angle. Only segmentation sees this; the PTM labels are unchanged.
    let faulted: Option<Vec<PtmAtom>> =
        if cfg.stacking_faults && cfg.dhcp_faults && cfg.mode == SegMode::Crystallite {
            let f = crate::analysis::stacking::dhcp_faults(cell, positions, ptm, par)?;
            let mut v = ptm.to_vec();
            for (p, q) in v.iter_mut().zip(&f) {
                if let Some(q) = q {
                    p.structure = structure::DHCP as u8;
                    p.orientation = q.map(|x| x as f32);
                }
            }
            Some(v)
        } else {
            None
        };
    let ptm: &[PtmAtom] = faulted.as_deref().unwrap_or(ptm);
    let q = |i: usize| -> [f64; 4] { ptm[i].q() };
    let structure_of = |i: usize| -> i32 { ptm[i].structure() };
    let crystalline = |i: usize| shell(structure_of(i)) > 0 && ptm[i].orientation[0].is_finite();
    let index = {
        let _t = crate::prof::StageTimer::start("seg.knn_build");
        KnnIndex::build(cell, positions, par)?
    };
    if cfg.mode == SegMode::Auto {
        return segment_auto(&index, positions, ptm, cfg, par, &groups);
    }

    // 1. Crystallites.
    let parent: Vec<AtomicU32> = (0..n as u32).map(AtomicU32::new).collect();
    {
        let _t = crate::prof::StageTimer::start("seg.crystallites");
        par.for_each(n, |i| {
            if !crystalline(i) {
                return;
            }
            thread_local! {
                static BUF: std::cell::RefCell<(KnnScratch, Vec<crate::neighbor::knn::Nbr>)> =
                    std::cell::RefCell::new((KnnScratch::default(), Vec::new()));
            }
            let st = structure_of(i);
            let g = &groups[&st];
            let qi = q(i);
            BUF.with(|b| {
                let (s, nb) = &mut *b.borrow_mut();
                index.knn(i, positions[i], shell(st), s, nb);
                for &(j, _, _) in nb.iter() {
                    let j = j as usize;
                    if j > i && structure_of(j) == st && crystalline(j) {
                        let d = orient::disorientation(&qi, &q(j), g);
                        if d < cfg.core_angle {
                            union(&parent, i as u32, j as u32);
                        }
                    }
                }
            });
        });
    }
    let root: Vec<u32> = (0..n as u32).map(|i| find(&parent, i)).collect();
    drop(parent);
    // Dense crystallite ids in order of their root (= minimum atom index).
    let mut cid = vec![u32::MAX; n];
    let mut roots: Vec<u32> = Vec::new();
    for i in 0..n {
        if crystalline(i) && root[i] as usize == i {
            cid[i] = roots.len() as u32;
            roots.push(i as u32);
        }
    }
    for i in 0..n {
        if crystalline(i) {
            cid[i] = cid[root[i] as usize];
        }
    }
    drop(root);
    let nc = roots.len();

    // 2. Crystallite means, sizes and inter-crystallite bonds, per chunk.
    let _t = crate::prof::StageTimer::start("seg.graph");
    type Part = (HashMap<u32, MeanAcc>, HashMap<(u32, u32), u32>);
    let parts: Vec<Part> = par.map_chunks(n, |lo, hi| {
        let mut means: HashMap<u32, MeanAcc> = HashMap::new();
        let mut bonds: HashMap<(u32, u32), u32> = HashMap::new();
        let mut s = KnnScratch::default();
        let mut nb = Vec::new();
        for i in lo..hi {
            if !crystalline(i) {
                continue;
            }
            let c = cid[i];
            let st = structure_of(i);
            let g = &groups[&st];
            let reference = q(roots[c as usize] as usize);
            means
                .entry(c)
                .or_insert_with(|| MeanAcc::new(reference))
                .add(&q(i), 1.0, g);
            index.knn(i, positions[i], shell(st), &mut s, &mut nb);
            for &(j, _, _) in nb.iter() {
                let j = j as usize;
                if j > i
                    && crystalline(j)
                    && orient::compatible(st, structure_of(j), cfg.stacking_faults)
                    && cid[j] != c
                {
                    let key = (c.min(cid[j]), c.max(cid[j]));
                    *bonds.entry(key).or_insert(0) += 1;
                }
            }
        }
        (means, bonds)
    });
    let mut means: Vec<MeanAcc> = roots.iter().map(|&r| MeanAcc::new(q(r as usize))).collect();
    let mut bonds: HashMap<(u32, u32), u32> = HashMap::new();
    for (m, b) in parts {
        let mut keys: Vec<u32> = m.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            let a = m[&k];
            let acc = &mut means[k as usize];
            // Same reference on both sides, so the sums add directly.
            for (s, v) in acc.sum.iter_mut().zip(a.sum) {
                *s += v;
            }
            acc.count += a.count;
        }
        for (k, v) in b {
            *bonds.entry(k).or_insert(0) += v;
        }
    }
    drop(_t);

    // 3. Merge on mean orientations.
    let _t = crate::prof::StageTimer::start("seg.merge");
    let mut cl = Clusters {
        parent: (0..nc as u32).collect(),
    };
    let mut edges: Vec<((u32, u32), u32)> = bonds.into_iter().collect();
    edges.sort_unstable_by_key(|e| e.0);
    let st_of = |c: u32| structure_of(roots[c as usize] as usize);
    // Each cluster's structure: that of its larger part at every merge, the
    // frame its mean is kept in.
    let mut cst: Vec<i32> = (0..nc as u32).map(st_of).collect();
    // Each cluster's FCC parts on their own, in the FCC frame: where an FCC
    // host decides, a twin inside a cluster kept in the HCP frame must still
    // read as 60° away.
    let mut fcc: Vec<Option<MeanAcc>> = (0..nc)
        .map(|c| (cst[c] == structure::FCC).then(|| means[c]))
        .collect();
    let hexagonal = |s: i32| s == structure::HCP || s == structure::DHCP;
    // Close-packed pairs are compared in the frame of the larger cluster, so
    // the order of the arguments does not matter: a hexagonal host in the
    // HCP frame, where the 60° turn about c is a symmetry (an HCP crystal's
    // FCC-labelled faults join it whichever variant they are); an FCC host in
    // the FCC frame, where that turn is the twin operation (a twin never
    // joins its host, whatever hexagonal layers it came with).
    let dis = |cst: &[i32], means: &[MeanAcc], fcc: &[Option<MeanAcc>], a: u32, b: u32| {
        let (a, b) = (a as usize, b as usize);
        let close_packed = |s: i32| s == structure::FCC || hexagonal(s);
        if !(close_packed(cst[a]) && close_packed(cst[b])) || cst[a] == cst[b] {
            return orient::disorientation_between(
                cst[a],
                &means[a].mean(),
                cst[b],
                &means[b].mean(),
            );
        }
        let (big, small) = if means[a].count >= means[b].count {
            (a, b)
        } else {
            (b, a)
        };
        let host = means[big].mean();
        if hexagonal(cst[big]) {
            let q = orient::convert(cst[small], &means[small].mean(), cst[big], &host);
            return orient::disorientation(&host, &q, &groups[&cst[big]]);
        }
        let q = match &fcc[small] {
            Some(f) => f.mean(),
            None => orient::convert(cst[small], &means[small].mean(), structure::FCC, &host),
        };
        orient::disorientation(&host, &q, &groups[&structure::FCC])
    };
    for _pass in 0..4 {
        // Order by the current disorientation of the clusters' means.
        let mut keyed: Vec<(f64, (u32, u32), u32)> = edges
            .iter()
            .filter(|e| e.1 >= cfg.min_bonds)
            .map(|&(k, b)| {
                let (ra, rb) = (cl.find(k.0), cl.find(k.1));
                let d = if ra == rb {
                    f64::INFINITY
                } else {
                    dis(&cst, &means, &fcc, ra, rb)
                };
                (d, k, b)
            })
            .filter(|e| e.0 < cfg.threshold)
            .collect();
        if keyed.is_empty() {
            break;
        }
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut merged = 0;
        for (_, (a, b), _) in keyed {
            let (ra, rb) = (cl.find(a), cl.find(b));
            if ra == rb {
                continue;
            }
            // Re-check against the means as they stand now.
            let d = dis(&cst, &means, &fcc, ra, rb);
            if d.is_nan() || d >= cfg.threshold {
                continue;
            }
            let (keep, gone) = if ra < rb { (ra, rb) } else { (rb, ra) };
            let (k, o) = (keep as usize, gone as usize);
            // The merged cluster takes the structure (and frame) of its
            // larger part.
            if means[o].count > means[k].count {
                let q = orient::convert(cst[k], &means[k].mean(), cst[o], &means[o].mean());
                let count = means[k].count;
                means[k] = MeanAcc::new(q);
                means[k].add(&q, count, &groups[&cst[o]]);
                cst[k] = cst[o];
            }
            let q = orient::convert(cst[o], &means[o].mean(), cst[k], &means[k].mean());
            let count = means[o].count;
            means[k].add(&q, count, &groups[&cst[k]]);
            if let Some(fo) = fcc[o].take() {
                let cubic = &groups[&structure::FCC];
                match &mut fcc[k] {
                    Some(fk) => {
                        let q = orient::align_to(&fk.mean(), &fo.mean(), cubic);
                        fk.add(&q, fo.count, cubic);
                    }
                    none => *none = Some(fo),
                }
            }
            cl.parent[gone as usize] = keep;
            merged += 1;
        }
        if merged == 0 {
            break;
        }
    }
    drop(_t);

    // 4. Grains: cluster sizes, dissolve small ones, adopt, relabel.
    let mut grain_of_cluster: HashMap<u32, i64> = HashMap::new();
    let mut label = vec![-1i64; n];
    for i in 0..n {
        if crystalline(i) {
            let r = cl.find(cid[i]);
            label[i] = r as i64;
            *grain_of_cluster.entry(r).or_insert(0) += 1;
        }
    }
    for l in label.iter_mut() {
        if *l >= 0 && (grain_of_cluster[&(*l as u32)] as usize) < cfg.min_size {
            *l = -1;
        }
    }
    let mut adopted = 0;
    if cfg.adopt {
        let _t = crate::prof::StageTimer::start("seg.adopt");
        adopted = adopt(&index, positions, &mut label, par);
    }

    let info: HashMap<i64, (i32, [f64; 4])> = grain_of_cluster
        .keys()
        .map(|&r| (r as i64, (cst[r as usize], means[r as usize].mean())))
        .collect();
    let mut crystallites: HashMap<i64, usize> = HashMap::new();
    for c in 0..nc as u32 {
        *crystallites.entry(cl.find(c) as i64).or_insert(0) += 1;
    }
    Ok(finish(
        &index,
        positions,
        label,
        &info,
        &crystallites,
        nc,
        adopted,
        None,
        par,
    ))
}

/// `SegMode::Auto`: OVITO's graph clustering (see `grain_auto`).
fn segment_auto(
    index: &KnnIndex,
    positions: &[Coord3],
    ptm: &[PtmAtom],
    cfg: &SegConfig,
    par: &Par,
    groups: &HashMap<i32, Vec<[f64; 4]>>,
) -> Result<Segmentation> {
    let c = grain_auto::cluster(
        index,
        positions,
        ptm,
        cfg.min_size,
        cfg.stacking_faults,
        par,
    );
    let mut label = c.label;
    // Per grain: structure (shared by all its atoms, edges join one structure)
    // and the symmetry-aligned mean orientation of its crystalline atoms, in
    // two passes so the reference atom does not bias it.
    let mut info: HashMap<i64, (i32, [f64; 4])> = HashMap::new();
    for pass in 0..2 {
        let mut acc: HashMap<i64, orient::MeanAcc> = HashMap::new();
        for (i, &l) in label.iter().enumerate() {
            if l < 0 {
                continue;
            }
            let st = c.structure[i];
            let reference = if pass == 0 {
                c.orientation[l as usize]
            } else {
                info[&l].1
            };
            acc.entry(l).or_insert_with(|| MeanAcc::new(reference)).add(
                &c.orientation[i],
                1.0,
                &groups[&st],
            );
        }
        info = acc
            .into_iter()
            .map(|(l, a)| (l, (c.structure[l as usize], a.mean())))
            .collect();
    }
    let crystallites: HashMap<i64, usize> = info.keys().map(|&l| (l, 1)).collect();
    let mut adopted = 0;
    if cfg.adopt {
        let _t = crate::prof::StageTimer::start("seg.adopt");
        adopted = grain_auto::adopt(&c.bonds, &mut label);
    }
    drop(c.bonds);
    Ok(finish(
        index,
        positions,
        label,
        &info,
        &crystallites,
        c.clusters,
        adopted,
        Some(c.threshold),
        par,
    ))
}

/// Relabel by decreasing size (ties by the lowest atom index) and build the
/// grain and pair tables. `info` maps a label to its structure and mean
/// orientation, `crystallites` to its crystallite count.
#[allow(clippy::too_many_arguments)]
fn finish(
    index: &KnnIndex,
    positions: &[Coord3],
    label: Vec<i64>,
    info: &HashMap<i64, (i32, [f64; 4])>,
    crystallites: &HashMap<i64, usize>,
    n_crystallites: usize,
    adopted: usize,
    merge_threshold: Option<f64>,
    par: &Par,
) -> Segmentation {
    let n = positions.len();
    let mut stats: HashMap<i64, (usize, usize)> = HashMap::new(); // size, first atom
    for (i, &l) in label.iter().enumerate() {
        if l >= 0 {
            let e = stats.entry(l).or_insert((0, i));
            e.0 += 1;
        }
    }
    let mut order: Vec<(i64, usize, usize)> = stats.iter().map(|(&l, &(s, f))| (l, s, f)).collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));
    let new_id: HashMap<i64, i32> = order
        .iter()
        .enumerate()
        .map(|(k, o)| (o.0, k as i32))
        .collect();
    let grain: Vec<i32> = label
        .iter()
        .map(|l| if *l >= 0 { new_id[l] } else { -1 })
        .collect();

    // Grain table and grain pairs.
    let grains: Vec<GrainInfo> = order
        .iter()
        .enumerate()
        .map(|(k, &(l, size, _))| {
            let (structure, orientation) = info[&l];
            GrainInfo {
                id: k as i32,
                size,
                structure,
                orientation,
                crystallites: crystallites.get(&l).copied().unwrap_or(0),
            }
        })
        .collect();
    // Pairs from the final labels, so boundary atoms adopted from the
    // non-crystalline layer count: grains separated by that layer share no
    // crystalline bond at all.
    let parts: Vec<HashMap<(i32, i32), u32>> = par.map_chunks(n, |lo, hi| {
        let mut m: HashMap<(i32, i32), u32> = HashMap::new();
        let mut s = KnnScratch::default();
        let mut nb = Vec::new();
        for i in lo..hi {
            let gi = grain[i];
            if gi < 0 {
                continue;
            }
            index.knn(i, positions[i], 12, &mut s, &mut nb);
            for &(j, _, _) in nb.iter() {
                let gj = grain[j as usize];
                if gj >= 0 && gj != gi && (j as usize) > i {
                    *m.entry((gi.min(gj), gi.max(gj))).or_insert(0) += 1;
                }
            }
        }
        m
    });
    let mut pair_bonds: HashMap<(i32, i32), u32> = HashMap::new();
    for m in parts {
        for (k, v) in m {
            *pair_bonds.entry(k).or_insert(0) += v;
        }
    }
    let mut pairs: Vec<GrainPair> = pair_bonds
        .into_iter()
        .map(|((a, b), bonds)| {
            let (ga, gb) = (&grains[a as usize], &grains[b as usize]);
            let disorientation = orient::disorientation_between(
                ga.structure,
                &ga.orientation,
                gb.structure,
                &gb.orientation,
            );
            GrainPair {
                a,
                b,
                bonds,
                disorientation,
            }
        })
        .collect();
    pairs.sort_by_key(|p| (p.a, p.b));

    Segmentation {
        grain,
        grains,
        pairs,
        n_crystallites,
        adopted,
        merge_threshold,
    }
}
