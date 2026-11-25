//! Cluster analysis (design §18.3): connected components of atoms under a
//! distance criterion, with OVITO `ClusterAnalysisModifier` parity for the
//! per-cluster properties.
//!
//! Deliberately a smaller analysis than grain segmentation. A grain boundary is
//! a *physical* claim — orientation and structure type both have to agree — and
//! the interesting cases are the ones that make that claim hard. A cluster is
//! only "these atoms are within `cutoff` of each other, transitively", which has
//! no crystal-structure opinion in it at all. That is why clusters are cheap
//! enough to run over every atom in a frame and why they are the connective
//! tissue for the per-void statistics in `analysis::void_b`: a void region is a
//! connected set of empty cells, which is this problem with the roles of atoms
//! and cells swapped.
//!
//! The two differences from grain segmentation that matter:
//!
//! - **Every candidate atom is in exactly one cluster.** There is no
//!   orientation to go missing. Without a selection every atom is a candidate
//!   and the id space has no holes; with `select` / `only_selected` the
//!   unselected atoms get `Cluster == -1` and take no part in any bond.
//! - **Cluster ids are OVITO's** (3.17 and later): 0-based in first-atom order,
//!   -1 for an atom in no cluster; `sort_by_size` reorders them largest first,
//!   ties in first-atom order. Per-cluster columns have one row per cluster,
//!   row `k` for id `k` (`ClusterIdentifier`). OVITO before 3.17 numbered
//!   clusters from 1 with 0 for no cluster.
//!
//! ## Per-cluster properties (`properties`)
//!
//! Centre of mass, radius of gyration and gyration tensor, as OVITO computes
//! them: over **unwrapped** coordinates, mass-weighted when masses exist (a
//! per-atom `Mass`/`mass` column, else the data file's per-type masses), and
//! normalised by the cluster's total mass. The gyration tensor is
//! `G_ab = Σ m (u − com)_a (u − com)_b / M` in OVITO's component order
//! `xx yy zz xy xz yz`, and `Rg = sqrt(tr G)`. The centre of mass is not
//! wrapped back into the cell, as in OVITO.
//!
//! ## Unwrapping
//!
//! Each atom's unwrapped position is the first atom of its cluster (its
//! *seed*, OVITO's BFS root) plus the sum of minimum-image bond vectors along a
//! spanning tree. The tree comes from a union-find that carries each node's
//! offset to its parent (a weighted union-find), so no adjacency list and no
//! second neighbour pass are needed; in exact arithmetic every spanning tree
//! gives the same answer for a cluster that does not wrap onto itself, so this
//! agrees with OVITO's BFS to rounding. A cluster that percolates through a
//! periodic boundary (bonded to its own image) has no unwrapping, here or in
//! OVITO; its coordinates then depend on the tree and its Rg is not meaningful.
//! Unlike OVITO, such clusters are detected (a bond closing a loop whose
//! minimum-image vectors do not sum to zero) and flagged in
//! `ClusterPercolates`.
//!
//! ## Parallel bonds, serial unions
//!
//! Bonds are generated in parallel over fixed atom chunks, in batches, and
//! applied to the union-find serially **in atom order** (chunk, atom, then
//! neighbour order). That is exactly the order of the serial loop, so the
//! union-find, the labels and the floating-point offsets are identical at every
//! thread count; batching bounds the buffered bonds to a few MB.

use std::borrow::Cow;

use rayon::prelude::*;

use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::cell::Coord3;
use crate::data::frame::Frame;
use crate::data::property::{DType, Property, PropertyStore};
use crate::data::select::Selection;
use crate::error::{Error, Result};
use crate::exec::Par;
use crate::graph::{canonical_labels, DisjointSet};
use crate::neighbor::cell_list::CellList;
use crate::neighbor::nnearest::nearest_k_excluding_self;

/// A particle type in a pair-cutoff table: its number, or its name (the
/// frame's `type_labels`), as OVITO's `pair_cutoffs` keys.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeKey {
    Id(u32),
    Name(String),
}

impl TypeKey {
    /// A token that parses as an integer is a type number; anything else is a
    /// name.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Err(Error::invalid("empty particle type in a pair cutoff"));
        }
        match s.parse::<u32>() {
            Ok(0) => Err(Error::invalid(
                "particle type 0 does not exist; type numbers start at 1",
            )),
            Ok(t) => Ok(TypeKey::Id(t)),
            Err(_) => Ok(TypeKey::Name(s.to_owned())),
        }
    }

    /// The type number in `frame`. An unknown name or number is an error: OVITO
    /// skips such an entry, which silently leaves the pair unbonded.
    fn resolve(&self, frame: &Frame) -> Result<u32> {
        match self {
            TypeKey::Id(t) => {
                let declared = (*t as usize) <= frame.type_labels.len()
                    || (*t as usize) <= frame.type_masses.len();
                if declared || frame.particles.types.contains(t) {
                    Ok(*t)
                } else {
                    Err(Error::invalid(format!(
                        "cluster pair cutoff names particle type {t}, which the frame does not \
                         have"
                    )))
                }
            }
            TypeKey::Name(name) => frame
                .type_labels
                .iter()
                .position(|l| l == name)
                .map(|i| i as u32 + 1)
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "cluster pair cutoff names particle type '{name}', but the frame's \
                         type names are [{}]",
                        frame.type_labels.join(", ")
                    ))
                }),
        }
    }
}

impl std::fmt::Display for TypeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeKey::Id(t) => write!(f, "{t}"),
            TypeKey::Name(n) => f.write_str(n),
        }
    }
}

/// One entry of a pair-cutoff table: atoms of types `a` and `b` (either way
/// round) are bonded at or below `cutoff`.
#[derive(Clone, Debug, PartialEq)]
pub struct PairCutoff {
    pub a: TypeKey,
    pub b: TypeKey,
    pub cutoff: f64,
}

impl PairCutoff {
    /// `A-B=CUT` (the CLI's `--pair-cutoff`), types by number or name.
    pub fn parse(s: &str) -> Result<Self> {
        let bad = || {
            Error::invalid(format!(
                "pair cutoff '{s}' is not TYPE-TYPE=CUTOFF (types by number or name, e.g. \
                 1-2=2.9 or Cu-Zr=3.1)"
            ))
        };
        let (pair, cut) = s.rsplit_once('=').ok_or_else(bad)?;
        let (a, b) = pair.split_once('-').ok_or_else(bad)?;
        if b.contains('-') {
            return Err(bad());
        }
        let cutoff: f64 = cut.trim().parse().map_err(|_| bad())?;
        Ok(PairCutoff {
            a: TypeKey::parse(a)?,
            b: TypeKey::parse(b)?,
            cutoff,
        })
    }
}

/// Squared cutoffs indexed `[ti * stride + tj]` by type number; 0 for a pair
/// with no entry, which is then never bonded, as in OVITO.
struct PairTable {
    stride: usize,
    cut2: Vec<f64>,
}

impl PairTable {
    fn build(pairs: &[PairCutoff], frame: &Frame) -> Result<Self> {
        let mut resolved = Vec::with_capacity(pairs.len());
        let mut max_t = frame.ntypes();
        for p in pairs {
            let (a, b) = (p.a.resolve(frame)? as usize, p.b.resolve(frame)? as usize);
            max_t = max_t.max(a).max(b);
            resolved.push((a, b, p.cutoff));
        }
        let stride = max_t + 1;
        let mut cut2 = vec![0.0; stride * stride];
        let mut seen = vec![false; stride * stride];
        for (p, &(a, b, c)) in pairs.iter().zip(&resolved) {
            if seen[a * stride + b] {
                return Err(Error::invalid(format!(
                    "cluster pair cutoff {}-{} is given twice",
                    p.a, p.b
                )));
            }
            for k in [a * stride + b, b * stride + a] {
                seen[k] = true;
                cut2[k] = c * c;
            }
        }
        Ok(PairTable { stride, cut2 })
    }

    #[inline]
    fn cut2(&self, ti: u32, tj: u32) -> f64 {
        self.cut2
            .get(ti as usize * self.stride + tj as usize)
            .copied()
            .unwrap_or(0.0)
    }
}

/// How a bond between two atoms is decided.
#[derive(Clone, Debug, PartialEq)]
pub enum ClusterMode {
    /// Bonded when the minimum-image distance is at or below a cutoff.
    Cutoff { cutoff: f64 },
    /// Bonded to the `k` nearest atoms, whatever their distance.
    ///
    /// The bond count per atom is fixed at `k`, so this is the mode to use when
    /// the frame's density varies enough that no single cutoff is right
    /// everywhere. On a uniform frame it buys nothing over `Cutoff` and costs a
    /// per-atom heap.
    KNear { k: usize },
    /// Bonded when the minimum-image distance is at or below the cutoff of the
    /// two atoms' type pair (OVITO's `neighbor_mode = ToPairCutoffs`). A pair
    /// with no entry is never bonded.
    PairCutoffs { pairs: Vec<PairCutoff> },
}

impl ClusterMode {
    /// The cutoff this mode needs for the shared cell list, if any: the
    /// largest pair cutoff in pair mode.
    pub fn cutoff(&self) -> Option<f64> {
        match self {
            ClusterMode::Cutoff { cutoff } => Some(*cutoff),
            ClusterMode::KNear { .. } => None,
            ClusterMode::PairCutoffs { pairs } => pairs.iter().map(|p| p.cutoff).reduce(f64::max),
        }
    }

    /// A short label for logs: `cutoff(2.9)`, `knear(4)`, `pairs(1-1=2.5,1-2=3)`.
    pub fn describe(&self) -> String {
        match self {
            ClusterMode::Cutoff { cutoff } => format!("cutoff({cutoff})"),
            ClusterMode::KNear { k } => format!("knear({k})"),
            ClusterMode::PairCutoffs { pairs } => {
                let v: Vec<String> = pairs
                    .iter()
                    .map(|p| format!("{}-{}={}", p.a, p.b, p.cutoff))
                    .collect();
                format!("pairs({})", v.join(","))
            }
        }
    }
}

/// Starting search radius that should contain `k` neighbours per atom at number
/// density `rho`, with a safety factor.
///
/// `r = (3k / (4*pi*rho))^(1/3)` is the exact radius for a *uniform* fluid, so
/// it is a starting point and not an answer: a clustered frame has a local
/// density above the mean, and the true k-th neighbour can sit further out. The
/// factor covers the ordinary case and the k-nearest loop doubles until every
/// atom really has `k`, so an underestimate costs a rebuild rather than a wrong
/// partition.
fn knear_radius(k: usize, rho: f64) -> f64 {
    if rho <= 0.0 {
        return f64::INFINITY;
    }
    (3.0 * k as f64 / (4.0 * std::f64::consts::PI * rho)).cbrt() * 1.5
}

/// Most doublings of [`knear_radius`] before giving up.
///
/// Eight doublings reach 256x the density estimate, which covers a frame
/// whose densest region is orders of magnitude above its mean. Past that the frame is not one a
/// fixed-k cluster analysis describes well, and the error says so instead of
/// quietly bonding fewer than `k` neighbours.
const KNEAR_MAX_WIDENINGS: usize = 8;

/// Atoms per bond-generation task. Small so a 40k-atom frame still spreads over
/// every thread; the result does not depend on it (unions run in atom order).
const BOND_GRAIN: usize = 2048;

/// Tasks per thread in one batch: bonds of a batch are buffered before the
/// serial union, so this bounds that buffer at a few MB.
const BATCH_PER_THREAD: usize = 2;

/// A bond `a -> b` with the minimum-image vector from `a` to `b` (zero when the
/// union-find does not track offsets).
type Bond = (u32, u32, Coord3);

/// Union-find with path compression and union by size, optionally carrying each
/// node's offset to its parent so that a union also fixes the relative
/// (unwrapped) position of the two trees.
///
/// Without offsets this is [`DisjointSet`]; with them, `find(x)` also returns
/// `u_x − u_root`. A union `a -> b` with vector `d = u_b − u_a` attaches the
/// smaller root under the larger with the offset that makes `d` hold.
struct Forest {
    ds: Option<DisjointSet>,
    parent: Vec<u32>,
    size: Vec<u32>,
    off: Vec<Coord3>,
    /// Per root: some bond inside the tree closes a loop with a non-zero
    /// displacement sum, i.e. the cluster is bonded to its own periodic image.
    wraps: Vec<bool>,
}

/// A bond closing a loop whose displacements do not sum to zero (within this,
/// in Å) bonds a cluster to its own image. A genuine wrap misses by a lattice
/// vector, tens of Å; rounding misses by ~1e-12 Å.
const WRAP_TOLERANCE: f64 = 1e-3;

impl Forest {
    fn new(n: usize, weighted: bool) -> Self {
        if weighted {
            Forest {
                ds: None,
                parent: (0..n as u32).collect(),
                size: vec![1; n],
                off: vec![[0.0; 3]; n],
                wraps: vec![false; n],
            }
        } else {
            Forest {
                ds: Some(DisjointSet::new(n)),
                parent: Vec::new(),
                size: Vec::new(),
                off: Vec::new(),
                wraps: Vec::new(),
            }
        }
    }

    /// Root of `x` and `u_x − u_root` (zero when unweighted).
    fn find(&mut self, x: u32) -> (u32, Coord3) {
        if let Some(ds) = self.ds.as_mut() {
            return (ds.find(x), [0.0; 3]);
        }
        let mut r = x;
        let mut acc = [0.0f64; 3];
        while self.parent[r as usize] != r {
            let o = self.off[r as usize];
            acc = [acc[0] + o[0], acc[1] + o[1], acc[2] + o[2]];
            r = self.parent[r as usize];
        }
        // Full path compression: every node on the path now points at the root
        // with its own offset to it.
        let mut y = x;
        let mut rem = acc;
        while self.parent[y as usize] != y {
            let next = self.parent[y as usize];
            let o = self.off[y as usize];
            self.parent[y as usize] = r;
            self.off[y as usize] = rem;
            rem = [rem[0] - o[0], rem[1] - o[1], rem[2] - o[2]];
            y = next;
        }
        (r, acc)
    }

    /// Union `a` and `b`, given `d = u_b − u_a` (ignored when unweighted).
    fn union(&mut self, a: u32, b: u32, d: Coord3) {
        if let Some(ds) = self.ds.as_mut() {
            ds.union(a, b);
            return;
        }
        let (ra, oa) = self.find(a);
        let (rb, ob) = self.find(b);
        // u_rb − u_ra = (u_a − u_ra) + d − (u_b − u_rb)
        let rel = [
            oa[0] + d[0] - ob[0],
            oa[1] + d[1] - ob[1],
            oa[2] + d[2] - ob[2],
        ];
        if ra == rb {
            // A loop: consistent unless the cluster reaches its own image.
            if rel.iter().any(|v| v.abs() > WRAP_TOLERANCE) {
                self.wraps[ra as usize] = true;
            }
            return;
        }
        let w = self.wraps[ra as usize] || self.wraps[rb as usize];
        if self.size[ra as usize] >= self.size[rb as usize] {
            self.parent[rb as usize] = ra;
            self.off[rb as usize] = rel;
            self.size[ra as usize] += self.size[rb as usize];
            self.wraps[ra as usize] = w;
        } else {
            self.parent[ra as usize] = rb;
            self.off[ra as usize] = [-rel[0], -rel[1], -rel[2]];
            self.size[rb as usize] += self.size[ra as usize];
            self.wraps[rb as usize] = w;
        }
    }

    /// Contiguous labels in first-atom order, each atom's offset to its
    /// cluster's first atom (the seed), and per cluster whether it wraps onto
    /// its own image (the last two empty when unweighted).
    fn labels(&mut self, n: usize) -> (Vec<u32>, usize, Vec<Coord3>, Vec<bool>) {
        if let Some(ds) = self.ds.as_mut() {
            let sets: Vec<u32> = (0..n as u32).collect();
            let (l, k) = canonical_labels(ds, &sets);
            return (l, k, Vec::new(), Vec::new());
        }
        let mut label_of_root = vec![u32::MAX; n];
        let mut seed_off: Vec<Coord3> = Vec::new();
        let mut wraps: Vec<bool> = Vec::new();
        let mut labels = vec![0u32; n];
        let mut rel = vec![[0.0; 3]; n];
        for x in 0..n {
            let (r, o) = self.find(x as u32);
            let l = &mut label_of_root[r as usize];
            if *l == u32::MAX {
                *l = seed_off.len() as u32;
                seed_off.push(o);
                wraps.push(self.wraps[r as usize]);
            }
            labels[x] = *l;
            let s = seed_off[*l as usize];
            rel[x] = [o[0] - s[0], o[1] - s[1], o[2] - s[2]];
        }
        let k = seed_off.len();
        (labels, k, rel, wraps)
    }
}

/// Generate bonds over `0..n` in parallel batches and apply them to `forest`
/// serially in atom order. `gen(i, out)` appends atom `i`'s bonds and returns
/// whether its neighbour list was complete.
fn bond_in_batches<S, G>(par: &Par, n: usize, forest: &mut Forest, gen: G) -> Result<bool>
where
    S: Default,
    G: Fn(usize, &mut S, &mut Vec<Bond>) -> Result<bool> + Sync,
{
    let ntasks = (n + BOND_GRAIN - 1) / BOND_GRAIN;
    let per_batch = (par.threads * BATCH_PER_THREAD).max(1);
    let mut complete = true;
    let run = |t: usize| -> Result<(Vec<Bond>, bool)> {
        let (s, e) = (t * BOND_GRAIN, ((t + 1) * BOND_GRAIN).min(n));
        let mut scratch = S::default();
        let mut bonds = Vec::new();
        let mut ok = true;
        for i in s..e {
            ok &= gen(i, &mut scratch, &mut bonds)?;
        }
        Ok((bonds, ok))
    };
    let mut t0 = 0;
    while t0 < ntasks {
        let t1 = (t0 + per_batch).min(ntasks);
        let results: Vec<Result<(Vec<Bond>, bool)>> = if par.is_parallel() {
            let run = &run;
            par.install(move || (t0..t1).into_par_iter().map(run).collect())
        } else {
            (t0..t1).map(run).collect()
        };
        // Lowest task's error first, whatever the scheduling.
        for r in results {
            let (bonds, ok) = r?;
            complete &= ok;
            for (a, b, d) in bonds {
                forest.union(a, b, d);
            }
        }
        t0 = t1;
    }
    Ok(complete)
}

/// Connected-component labelling over a distance criterion.
#[derive(Clone, Debug)]
pub struct ClusterAnalysis {
    pub mode: ClusterMode,
    /// Clusters smaller than this are reported in `ClusterSize` but their atoms
    /// get `ParticleCluster == -1` (`Cluster` keeps their ids).
    pub min_size: usize,
    /// Cluster only the atoms matching this expression (`--select`).
    pub select: Option<Selection>,
    /// Cluster only the atoms flagged by the frame's `Selection` column
    /// (`--only-selected`, OVITO's `only_selected`). With `select` too, an atom
    /// must pass both.
    pub only_selected: bool,
    /// Renumber clusters largest first (ties in first-atom order).
    pub sort_by_size: bool,
    /// Write `UnwrappedPosition`, each atom's position unwrapped across the
    /// periodic boundaries along its cluster's bonds.
    pub unwrap: bool,
    /// Compute per-cluster mass, centre of mass, radius of gyration and
    /// gyration tensor.
    pub properties: bool,
}

impl Default for ClusterAnalysis {
    fn default() -> Self {
        ClusterAnalysis {
            mode: ClusterMode::Cutoff { cutoff: 3.5 },
            min_size: 1,
            select: None,
            only_selected: false,
            sort_by_size: false,
            unwrap: false,
            properties: false,
        }
    }
}

impl ClusterAnalysis {
    pub fn cutoff(cutoff: f64) -> Self {
        ClusterAnalysis {
            mode: ClusterMode::Cutoff { cutoff },
            ..Default::default()
        }
    }

    pub fn knear(k: usize) -> Self {
        ClusterAnalysis {
            mode: ClusterMode::KNear { k },
            ..Default::default()
        }
    }

    fn selecting(&self) -> bool {
        self.only_selected || self.select.is_some()
    }

    /// The candidate atoms, or `None` for all of them.
    fn mask(&self, frame: &Frame) -> Result<Option<Vec<bool>>> {
        if !self.selecting() {
            return Ok(None);
        }
        let n = frame.natoms();
        let mut m = vec![true; n];
        if self.only_selected {
            let Some(sel) = frame.particles.selection.as_ref() else {
                return Err(Error::invalid(
                    "cluster --only-selected needs a Selection property on the input, and the \
                     frame has none. Clustering every atom instead would silently ignore the \
                     request",
                ));
            };
            for (i, v) in m.iter_mut().enumerate() {
                *v = sel.get(i).is_some_and(|&s| s != 0);
            }
        }
        if let Some(expr) = &self.select {
            let inv_t = frame.cell.inverse_transposed()?;
            let p = &frame.particles;
            for (i, v) in m.iter_mut().enumerate() {
                if *v {
                    let x = p.positions[i];
                    let f = frame.cell.x2lamda_pre(x, &inv_t);
                    *v = expr.matches(x, f, p.types[i], p.ids[i]);
                }
            }
        }
        Ok(Some(m))
    }
}

/// Per-atom weights for the centre of mass: a `Mass`/`mass` column, else the
/// data file's per-type masses when every type present has a positive one, else
/// `None` (unit weights), as OVITO's `inputParticleMasses`.
fn atom_masses(frame: &Frame) -> Option<Vec<f64>> {
    let n = frame.natoms();
    for name in ["Mass", "mass"] {
        if let Some(p) = frame.particles.props.get(name) {
            if p.components == 1 && p.len() == n {
                return Some((0..n).map(|i| p.get_flat(i).unwrap_or(f64::NAN)).collect());
            }
        }
    }
    let tm = &frame.type_masses;
    if tm.is_empty() {
        return None;
    }
    let mass_of = |t: u32| -> Option<f64> {
        let m = *tm.get((t as usize).checked_sub(1)?)?;
        (m.is_finite() && m > 0.0).then_some(m)
    };
    frame
        .particles
        .types
        .iter()
        .map(|&t| mass_of(t))
        .collect::<Option<Vec<f64>>>()
}

impl Analysis for ClusterAnalysis {
    fn name(&self) -> &'static str {
        "cluster"
    }

    fn cutoffs(&self) -> Vec<f64> {
        self.mode.cutoff().into_iter().collect()
    }

    /// `KNear` has no configured radius, so it is sized from the frame.
    ///
    /// Reporting nothing here would leave the pipeline with no cutoff to build a
    /// cell list from, and the analysis would then have to guess one internally;
    /// publishing the same estimate where the pipeline can see it is what keeps
    /// the two agreeing.
    fn cutoffs_for(&self, frame: &crate::data::frame::Frame) -> Vec<f64> {
        match self.mode {
            ClusterMode::Cutoff { cutoff } => vec![cutoff],
            ClusterMode::PairCutoffs { .. } => self.mode.cutoff().into_iter().collect(),
            ClusterMode::KNear { k } if frame.natoms() >= 2 => {
                let rho = frame.natoms() as f64 / frame.cell.volume();
                let r = knear_radius(k, rho);
                // A non-finite radius means a degenerate cell; let the cell list
                // report that rather than sizing a list with infinity.
                if r.is_finite() {
                    vec![r]
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }

    fn run(
        &self,
        ctx: &AnalysisContext<'_>,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        let frame = ctx.frame;
        let n_all = frame.natoms();
        let par = ctx.par;

        // A cutoff of zero, a negative one, or NaN would make the comparison
        // answer "never bonded" or "always bonded" respectively while looking
        // like a configured radius. Both produce a plausible cluster map for the
        // wrong geometry, which is the failure mode worth refusing: same
        // reasoning as `Grains::resolve_cutoff`.
        if let ClusterMode::Cutoff { cutoff } = self.mode {
            if !cutoff.is_finite() || cutoff <= 0.0 {
                return Err(Error::invalid(format!(
                    "cluster cutoff must be positive and finite, got {cutoff}"
                )));
            }
        }
        // Same refusal per pair. OVITO ignores an entry <= 0 (the pair stays
        // unbonded); omitting the pair says that without a magic value.
        let pair_table = match &self.mode {
            ClusterMode::PairCutoffs { pairs } => {
                if pairs.is_empty() {
                    return Err(Error::invalid(
                        "cluster pair-cutoff mode needs at least one pair cutoff",
                    ));
                }
                for p in pairs {
                    if !p.cutoff.is_finite() || p.cutoff <= 0.0 {
                        return Err(Error::invalid(format!(
                            "cluster pair cutoff {}-{} must be positive and finite, got {}",
                            p.a, p.b, p.cutoff
                        )));
                    }
                }
                Some(PairTable::build(pairs, frame)?)
            }
            _ => None,
        };

        // The candidate atoms. With a selection the analysis runs on the
        // compacted subset (its own cell list, its own indices `0..n`) and maps
        // back at the end, so unselected atoms cost nothing and a k-nearest bond
        // can never land on one.
        let mask = self.mask(frame)?;
        let members: Option<Vec<usize>> = mask
            .as_ref()
            .map(|m| (0..n_all).filter(|&i| m[i]).collect());
        let all_pos = &frame.particles.positions[..n_all];
        let pos: Cow<[Coord3]> = match &members {
            Some(idx) => Cow::Owned(idx.iter().map(|&i| all_pos[i]).collect()),
            None => Cow::Borrowed(all_pos),
        };
        let n = pos.len();

        if let ClusterMode::KNear { k } = self.mode {
            if k == 0 {
                return Err(Error::invalid(
                    "cluster --knear must be at least 1; k=0 would bond no atoms and \
                     report every atom as its own cluster of one",
                ));
            }
            if k >= n {
                return Err(Error::invalid(format!(
                    "cluster --knear {k} needs at least {k} other atoms, but the {} has {}",
                    if members.is_some() {
                        "selection"
                    } else {
                        "frame"
                    },
                    n.saturating_sub(1)
                )));
            }
        }

        let weighted = self.unwrap || self.properties;
        let mut forest = Forest::new(n, weighted);

        match self.mode {
            ClusterMode::Cutoff { .. } | ClusterMode::PairCutoffs { .. } => {
                let cutoff = self.mode.cutoff().unwrap_or(0.0);
                // Per-candidate types, for the pair table.
                let types: Option<Vec<u32>> = pair_table.as_ref().map(|_| {
                    let t = &frame.particles.types;
                    match &members {
                        Some(idx) => idx.iter().map(|&i| t[i]).collect(),
                        None => t[..n_all].to_vec(),
                    }
                });
                // Reuse the pipeline's list only when it indexes these atoms and
                // is wide enough to answer this query. A `CellList` sizes its
                // candidate stencil once at build time, so a list built for a
                // narrower cutoff returns a *truncated* neighbour set rather than
                // an error, and a missing bond here silently splits a cluster in
                // two.
                let owned;
                let cell = match ctx.nkeys.cell {
                    Some(cl) if members.is_none() && cutoff <= cl.max_query_cutoff() => cl,
                    _ => {
                        let _t = crate::prof::StageTimer::start("cell_list.build");
                        owned = CellList::build(&frame.cell, &pos, cutoff)?;
                        &owned
                    }
                };
                let _t = crate::prof::StageTimer::start("cluster.bonds");
                bond_in_batches(par, n, &mut forest, |i, scratch: &mut Vec<_>, bonds| {
                    // The full list includes atoms with a lower index, so taking
                    // only `j > i` visits every pair once. Every image of a pair
                    // is kept: a cluster bonded to its own image percolates.
                    scratch.clear();
                    cell.neighbor_entries(i, cutoff, scratch)?;
                    match (&pair_table, &types) {
                        // Pair mode: the list is built at the largest pair
                        // cutoff; keep a bond only within its own pair's, with
                        // the same `d^2 <= c^2` test as the list's.
                        (Some(tab), Some(ty)) => {
                            for &(j, dr, _) in scratch.iter() {
                                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                                if j > i && d2 <= tab.cut2(ty[i], ty[j]) {
                                    bonds.push((i as u32, j as u32, dr));
                                }
                            }
                        }
                        _ => {
                            for &(j, dr, _) in scratch.iter() {
                                if j > i {
                                    bonds.push((i as u32, j as u32, dr));
                                }
                            }
                        }
                    }
                    Ok(true)
                })?;
            }
            ClusterMode::KNear { k } => {
                // Minimum-image bonds, so a cluster wraps the periodic boundary
                // with no seam: `nearest_k_excluding_self` already reports
                // image-shifted candidates.
                //
                // `nearest_k` reads neighbours out of a stencil fixed when the
                // list was built, so a list that is too narrow does not search
                // further -- it returns fewer than `k` and says nothing. Starting
                // from the density estimate and widening until every atom really
                // has `k` is what turns that silent shortfall into a loop
                // condition.
                let rho = n as f64 / frame.cell.volume();
                let mut radius = knear_radius(k, rho);
                let mut complete = false;
                for _ in 0..=KNEAR_MAX_WIDENINGS {
                    if !radius.is_finite() || radius <= 0.0 {
                        return Err(Error::invalid(
                            "cluster --knear needs a cell with positive volume",
                        ));
                    }
                    let list = CellList::build(&frame.cell, &pos, radius)?;
                    let _t = crate::prof::StageTimer::start("cluster.bonds");
                    complete =
                        bond_in_batches(par, n, &mut forest, |i, entries: &mut Vec<_>, bonds| {
                            nearest_k_excluding_self(&list, i, k, entries)?;
                            // Every returned pair is unioned, including `j < i`: the
                            // k nearest is *not* symmetric -- two atoms at very
                            // different densities can have each other outside their
                            // own k -- so a `j > i` test would silently drop bonds.
                            for &(j, dr, _) in entries.iter() {
                                bonds.push((i as u32, j, dr));
                            }
                            Ok(entries.len() >= k)
                        })?;
                    if complete {
                        break;
                    }
                    // Discard the partial bonds before retrying: they came from a
                    // list too narrow to answer, and a union-find cannot be
                    // un-unioned. Starting clean also keeps the retry's answer a
                    // function of the final list alone.
                    forest = Forest::new(n, weighted);
                    radius *= 2.0;
                }
                if !complete {
                    return Err(Error::invalid(format!(
                        "cluster --knear {k} found no radius within {}x the density \
                         estimate that gives every atom {k} neighbours; the frame's local \
                         density is too far above its mean for a fixed-k cluster \
                         analysis",
                        1u64 << KNEAR_MAX_WIDENINGS
                    )));
                }
            }
        }

        // Labels in first-atom order over the candidates, and each candidate's
        // offset from its cluster's seed when tracked.
        let (mut labels, n_clusters, rel, mut wraps) = forest.labels(n);
        drop(forest);

        let mut sizes = vec![0u32; n_clusters];
        for &l in &labels {
            sizes[l as usize] += 1;
        }

        // Largest first, ties in first-atom order: a stable sort of the ids.
        if self.sort_by_size && n_clusters > 1 {
            let mut order: Vec<u32> = (0..n_clusters as u32).collect();
            order.sort_by(|&a, &b| sizes[b as usize].cmp(&sizes[a as usize]));
            let mut new_id = vec![0u32; n_clusters];
            for (rank, &old) in order.iter().enumerate() {
                new_id[old as usize] = rank as u32;
            }
            for l in labels.iter_mut() {
                *l = new_id[*l as usize];
            }
            sizes = order.iter().map(|&o| sizes[o as usize]).collect();
            if !wraps.is_empty() {
                wraps = order.iter().map(|&o| wraps[o as usize]).collect();
            }
        }

        // Unwrapped positions over the candidates: seed position plus offset.
        let unwrapped: Vec<Coord3> = if weighted {
            let mut seed = vec![u32::MAX; n_clusters];
            for (x, &l) in labels.iter().enumerate() {
                if seed[l as usize] == u32::MAX {
                    seed[l as usize] = x as u32;
                }
            }
            (0..n)
                .map(|x| {
                    let s = pos[seed[labels[x] as usize] as usize];
                    let r = rel[x];
                    [s[0] + r[0], s[1] + r[1], s[2] + r[2]]
                })
                .collect()
        } else {
            Vec::new()
        };
        drop(rel);

        // Back to frame indices.
        let full = |v: &[u32], fill: i32| -> Vec<i32> {
            match &members {
                None => v.iter().map(|&l| l as i32).collect(),
                Some(idx) => {
                    let mut o = vec![fill; n_all];
                    for (k, &i) in idx.iter().enumerate() {
                        o[i] = v[k] as i32;
                    }
                    o
                }
            }
        };
        // OVITO's ids (3.17 and later): 0-based, -1 for an atom in no cluster.
        let cluster_col = full(&labels, -1);

        // `Cluster` keeps every cluster, including undersized ones, so the id
        // space is contiguous and `ClusterSize` (row `id`) covers it.
        // `ParticleCluster` is the filtered view: an atom in a cluster below
        // `min_size` (or not selected) reads as -1, like an atom in no cluster.
        // Both are written, because they answer different questions and a
        // downstream analysis should not have to re-derive one from the other.
        let particle_cluster: Vec<i32> = cluster_col
            .iter()
            .map(|&id| {
                if id >= 0 && sizes[id as usize] as usize >= self.min_size {
                    id
                } else {
                    -1
                }
            })
            .collect();

        let largest = sizes.iter().copied().max().unwrap_or(0) as usize;
        let mut count_retained = 0usize;
        let mut atoms_retained = 0usize;
        for &sz in &sizes {
            if sz as usize >= self.min_size {
                count_retained += 1;
                atoms_retained += sz as usize;
            }
        }
        let max_size = largest;

        out.register_scalar(DType::I32, "Cluster", cluster_col);
        // Per-cluster, not per-atom: one entry per cluster, indexed by cluster
        // id. Registered with `per_particle = false` explicitly, because
        // `register_scalar` defaults to per-atom and the atom writer filters on
        // that flag -- so the default would publish a per-cluster column as if it
        // were per-atom, where it is silently wrong for every frame with more
        // than one cluster and not detectable by reading a single row.
        let mut id_prop = Property::with_capacity(
            "ClusterIdentifier".to_owned(),
            DType::I32,
            false,
            1,
            n_clusters,
        );
        let mut size_prop =
            Property::with_capacity("ClusterSize".to_owned(), DType::I32, false, 1, n_clusters);
        for (k, &sz) in sizes.iter().enumerate() {
            id_prop.push_value(&[k as f64]);
            size_prop.push_value(&[sz as f64]);
        }
        out.register(id_prop);
        out.register(size_prop);
        out.register_scalar(DType::I32, "ParticleCluster", particle_cluster);

        if self.properties {
            let _t = crate::prof::StageTimer::start("cluster.properties");
            let masses = atom_masses(frame);
            let w = |x: usize| -> f64 {
                match (&masses, &members) {
                    (None, _) => 1.0,
                    (Some(m), None) => m[x],
                    (Some(m), Some(idx)) => m[idx[x]],
                }
            };
            // Two passes, serial in atom order (deterministic at any thread
            // count): centre of mass, then the second moments about it.
            let mut mass = vec![0f64; n_clusters];
            let mut com = vec![[0f64; 3]; n_clusters];
            for x in 0..n {
                let l = labels[x] as usize;
                let (m, u) = (w(x), unwrapped[x]);
                mass[l] += m;
                for (c, &ua) in com[l].iter_mut().zip(&u) {
                    *c += m * ua;
                }
            }
            for (c, &ml) in com.iter_mut().zip(&mass) {
                for v in c.iter_mut() {
                    *v /= ml;
                }
            }
            let mut gyr = vec![[0f64; 6]; n_clusters];
            for x in 0..n {
                let l = labels[x] as usize;
                let (m, u, c) = (w(x), unwrapped[x], com[labels[x] as usize]);
                let d = [u[0] - c[0], u[1] - c[1], u[2] - c[2]];
                let g = &mut gyr[l];
                g[0] += m * d[0] * d[0];
                g[1] += m * d[1] * d[1];
                g[2] += m * d[2] * d[2];
                g[3] += m * d[0] * d[1];
                g[4] += m * d[0] * d[2];
                g[5] += m * d[1] * d[2];
            }
            let mut p_mass =
                Property::with_capacity("ClusterMass".to_owned(), DType::F64, false, 1, n_clusters);
            let mut p_com = Property::with_capacity(
                "ClusterCenterOfMass".to_owned(),
                DType::F64,
                false,
                3,
                n_clusters,
            );
            let mut p_rg = Property::with_capacity(
                "ClusterRadiusOfGyration".to_owned(),
                DType::F64,
                false,
                1,
                n_clusters,
            );
            let mut p_g = Property::with_capacity(
                "ClusterGyrationTensor".to_owned(),
                DType::F64,
                false,
                6,
                n_clusters,
            );
            for l in 0..n_clusters {
                let g: Vec<f64> = gyr[l].iter().map(|v| v / mass[l]).collect();
                p_mass.push_value(&[mass[l]]);
                p_com.push_value(&com[l]);
                p_rg.push_value(&[(g[0] + g[1] + g[2]).sqrt()]);
                p_g.push_value(&g);
            }
            let mut p_wrap = Property::with_capacity(
                "ClusterPercolates".to_owned(),
                DType::I32,
                false,
                1,
                n_clusters,
            );
            for &w in &wraps {
                p_wrap.push_value(&[f64::from(u8::from(w))]);
            }
            out.register(p_wrap);
            out.register(p_mass);
            out.register(p_com);
            out.register(p_rg);
            out.register(p_g);
            summary.merge_from(Summary::scalar(
                "cluster_percolating",
                wraps.iter().filter(|&&w| w).count() as f64,
            ));
            summary.merge_from(Summary::scalar(
                "cluster_mass_weighted",
                if masses.is_some() { 1.0 } else { 0.0 },
            ));
        }

        if self.unwrap {
            let mut flat: Vec<f64> = Vec::with_capacity(3 * n_all);
            match &members {
                None => {
                    for u in &unwrapped {
                        flat.extend_from_slice(u);
                    }
                }
                Some(idx) => {
                    // Unselected atoms keep their position.
                    let mut k = 0usize;
                    for (i, p) in all_pos.iter().enumerate() {
                        if k < idx.len() && idx[k] == i {
                            flat.extend_from_slice(&unwrapped[k]);
                            k += 1;
                        } else {
                            flat.extend_from_slice(p);
                        }
                    }
                }
            }
            out.register_flat(DType::F64, "UnwrappedPosition", 3, flat);
        }

        summary.merge_from(Summary::scalar("cluster_count", n_clusters as f64));
        summary.merge_from(Summary::scalar("cluster_largest", largest as f64));
        // Fractions and means are over the clustered atoms: all of them, or the
        // selection.
        summary.merge_from(Summary::scalar(
            "cluster_largest_fraction",
            if n == 0 {
                0.0
            } else {
                largest as f64 / n as f64
            },
        ));
        summary.merge_from(Summary::scalar(
            "cluster_mean_size",
            if n_clusters == 0 {
                0.0
            } else {
                n as f64 / n_clusters as f64
            },
        ));
        // How many atoms survive `min_size`, and how many clusters that leaves.
        // The two are what a `min_size` sweep needs to plot, and deriving them
        // from `ClusterSize` after the fact means re-reading a per-cluster
        // property the atom output does not carry.
        summary.merge_from(Summary::scalar(
            "cluster_atoms_retained",
            atoms_retained as f64,
        ));
        summary.merge_from(Summary::scalar(
            "cluster_count_retained",
            count_retained as f64,
        ));
        summary.merge_from(Summary::scalar("cluster_min_size", self.min_size as f64));
        summary.merge_from(Summary::scalar(
            "cluster_mode_cutoff",
            self.mode.cutoff().unwrap_or(f64::NAN),
        ));
        if members.is_some() {
            summary.merge_from(Summary::scalar("cluster_atoms_selected", n as f64));
        }
        // Size distribution, so the percolation question ("is there one giant
        // cluster or many small ones") is answerable without a second pass.
        let mut hist = vec![0f64; max_size + 1];
        for &s in &sizes {
            hist[s as usize] += 1.0;
        }
        summary
            .histograms
            .push(("cluster_size_histogram".to_string(), hist));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::NeighborhoodContext;
    use crate::cell::UnitCell;
    use crate::data::frame::Frame;
    use crate::data::particle::ParticleData;
    use crate::exec::Par;

    fn frame(cell: UnitCell, pos: &[[f64; 3]]) -> Frame {
        let mut p = ParticleData::with_capacity(pos.len());
        for (i, q) in pos.iter().enumerate() {
            p.add_atom(*q, i as i64 + 1, 1);
        }
        Frame::new(cell, p)
    }

    fn run(f: &Frame, c: &ClusterAnalysis) -> (Vec<i32>, Vec<i32>, Vec<i32>, Summary) {
        let nk = NeighborhoodContext { cell: None };
        let ctx = AnalysisContext {
            frame: f,
            nkeys: &nk,
            par: &Par::new(1),
        };
        let mut store = PropertyStore::new();
        let mut sc = Summary::default();
        c.run(&ctx, &mut store, &mut sc).unwrap();
        let get = |n: &str| store.get(n).unwrap().as_i32().unwrap().clone();
        (
            get("Cluster"),
            get("ClusterSize"),
            get("ParticleCluster"),
            sc,
        )
    }

    fn scalar(sc: &Summary, k: &str) -> f64 {
        sc.scalars
            .iter()
            .find(|(n, _)| n == k)
            .unwrap_or_else(|| panic!("missing {k}"))
            .1
    }

    /// Three dimers along x with three *different* gaps: 9 A from the first to
    /// the second, 14 A to the third, and 24 A back round the periodic boundary
    /// to the first.
    ///
    /// The gaps have to differ. With two equal ones any cutoff that reaches one
    /// gap reaches both, so "joins the first pair only" is not expressible and a
    /// test claiming it is really asserting one cluster or three.
    fn dimers() -> Frame {
        frame(
            UnitCell::orthorhombic(50.0, 50.0, 50.0),
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [25.0, 0.0, 0.0],
                [26.0, 0.0, 0.0],
            ],
        )
    }

    #[test]
    fn separated_pairs_are_three_clusters() {
        let (labels, sizes, pc, sc) = run(&dimers(), &ClusterAnalysis::cutoff(2.0));
        assert_eq!(labels, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(sizes, vec![2, 2, 2]);
        assert_eq!(pc, labels, "min_size 1 keeps everything");
        assert_eq!(scalar(&sc, "cluster_count"), 3.0);
        assert_eq!(scalar(&sc, "cluster_largest"), 2.0);
        assert_eq!(scalar(&sc, "cluster_mean_size"), 2.0);
    }

    #[test]
    fn a_cutoff_that_reaches_neighbouring_dimers_joins_them() {
        // 9.5 Å bridges the 9 Å gap between the first two dimers but not the
        // 14 Å one after, so the answer is two clusters and not one and not three.
        let (labels, sizes, _, sc) = run(&dimers(), &ClusterAnalysis::cutoff(9.5));
        assert_eq!(labels, vec![0, 0, 0, 0, 1, 1]);
        assert_eq!(sizes, vec![4, 2]);
        assert_eq!(scalar(&sc, "cluster_count"), 2.0);
    }

    #[test]
    fn a_cutoff_too_small_to_reach_anything_splits_every_atom() {
        // The degenerate but legal answer: every atom its own cluster. Reported
        // rather than refused, because a caller asking for a 0.5 Å cutoff on a
        // metal wants to see this.
        let (labels, sizes, _, sc) = run(&dimers(), &ClusterAnalysis::cutoff(0.5));
        assert_eq!(labels, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(sizes, vec![1; 6]);
        assert_eq!(scalar(&sc, "cluster_count"), 6.0);
        assert_eq!(scalar(&sc, "cluster_largest"), 1.0);
    }

    #[test]
    fn min_size_zeroing_is_reported_separately_from_the_cluster_map() {
        // `Cluster` keeps all three, `ParticleCluster` zeroes the undersized
        // ones. Both are needed and they must not be the same property.
        let (labels, sizes, pc, sc) = run(
            &dimers(),
            &ClusterAnalysis {
                mode: ClusterMode::Cutoff { cutoff: 2.0 },
                min_size: 3,
                ..Default::default()
            },
        );
        assert_eq!(labels, vec![0, 0, 1, 1, 2, 2], "the map is unchanged");
        assert_eq!(sizes, vec![2, 2, 2], "sizes are still reported");
        assert_eq!(pc, vec![-1; 6], "everything is filtered out");
        assert_eq!(scalar(&sc, "cluster_count"), 3.0);
        assert_eq!(scalar(&sc, "cluster_count_retained"), 0.0);
        assert_eq!(scalar(&sc, "cluster_atoms_retained"), 0.0);
    }

    #[test]
    fn min_size_keeps_the_clusters_that_qualify() {
        // One large cluster plus two dimers, so the filter has to keep some.
        let f = frame(
            UnitCell::orthorhombic(40.0, 40.0, 40.0),
            &[
                // A 5-atom chain at spacing 1.0 along y, kept apart in x.
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 4.0, 0.0],
                [20.0, 0.0, 0.0],
                [21.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [31.0, 0.0, 0.0],
            ],
        );
        let (labels, sizes, pc, sc) = run(
            &f,
            &ClusterAnalysis {
                mode: ClusterMode::Cutoff { cutoff: 1.5 },
                min_size: 3,
                ..Default::default()
            },
        );
        assert_eq!(sizes, vec![5, 2, 2]);
        assert_eq!(labels, vec![0, 0, 0, 0, 0, 1, 1, 2, 2]);
        assert_eq!(
            pc,
            vec![0, 0, 0, 0, 0, -1, -1, -1, -1],
            "only cluster 0 qualifies"
        );
        assert_eq!(scalar(&sc, "cluster_count"), 3.0);
        assert_eq!(scalar(&sc, "cluster_count_retained"), 1.0);
        assert_eq!(scalar(&sc, "cluster_atoms_retained"), 5.0);
    }

    #[test]
    fn ids_follow_first_atom_order_not_size_order() {
        // Sizes 2, 2, 5 in atom order. If ids were handed out largest-first the
        // five-chain would be cluster 0.
        let f = frame(
            UnitCell::orthorhombic(40.0, 40.0, 40.0),
            &[
                [20.0, 0.0, 0.0],
                [21.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [31.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
        );
        let (labels, sizes, _, _) = run(&f, &ClusterAnalysis::cutoff(1.5));
        assert_eq!(sizes, vec![2, 2, 5]);
        assert_eq!(labels[0], 0, "atom 0's dimer is cluster 0");
        assert_eq!(labels[4], 2, "the 5-chain comes last despite being largest");
    }

    #[test]
    fn a_cluster_wraps_the_periodic_boundary_with_no_seam() {
        // Two atoms 1 Å apart through the boundary, 29 Å apart directly. Under
        // minimum image they are bonded, so this is one cluster of two. A
        // non-periodic or non-minimum-image bond test gives two.
        let f = frame(
            UnitCell::orthorhombic(30.0, 30.0, 30.0),
            &[[0.5, 15.0, 15.0], [29.5, 15.0, 15.0]],
        );
        let (labels, sizes, _, sc) = run(&f, &ClusterAnalysis::cutoff(2.0));
        assert_eq!(labels, vec![0, 0]);
        assert_eq!(sizes, vec![2]);
        assert_eq!(scalar(&sc, "cluster_count"), 1.0);
    }

    #[test]
    fn a_cluster_spanning_all_three_axes_wraps_on_each() {
        // Corner to corner through the boundary on x, y and z at once.
        let f = frame(
            UnitCell::orthorhombic(10.0, 10.0, 10.0),
            &[[0.5, 0.5, 0.5], [9.5, 9.5, 9.5]],
        );
        // Minimum image is 1 A on each axis, so the separation is sqrt(3) = 1.732
        // and the cutoff has to clear that, not 1 A.
        let (labels, _, _, sc) = run(&f, &ClusterAnalysis::cutoff(1.8));
        assert_eq!(labels, vec![0, 0]);
        assert_eq!(scalar(&sc, "cluster_count"), 1.0);
        // Just under that separation the same pair is two clusters, so the line
        // above is the minimum image being taken on all three axes rather than a
        // coincidence of the fixture.
        let (split, _, _, _) = run(&f, &ClusterAnalysis::cutoff(1.7));
        assert_eq!(split, vec![0, 1]);
    }

    #[test]
    fn knear_bonds_a_fixed_number_of_neighbours_whatever_the_distance() {
        // The same three dimers, but k=3 bonds each atom to its 3 nearest, so
        // every atom reaches across the 9 Å gap and the frame is one cluster.
        let (labels, sizes, _, sc) = run(&dimers(), &ClusterAnalysis::knear(3));
        assert_eq!(sizes, vec![6]);
        assert_eq!(labels, vec![0; 6]);
        assert_eq!(scalar(&sc, "cluster_count"), 1.0);
        assert!(scalar(&sc, "cluster_mode_cutoff").is_nan());
    }

    #[test]
    fn knear_and_cutoff_agree_when_the_frame_is_uniform() {
        // On a regular lattice the k nearest are exactly the ones inside a
        // cutoff just past the k-th shell, so the two modes must produce the
        // same partition. This is the check that `KNear` is not quietly a
        // different algorithm.
        let mut pos = Vec::new();
        for iz in 0..3 {
            for iy in 0..3 {
                for ix in 0..3 {
                    pos.push([ix as f64 * 2.0, iy as f64 * 2.0, iz as f64 * 2.0]);
                }
            }
        }
        let f = frame(UnitCell::orthorhombic(6.0, 6.0, 6.0), &pos);
        let (a, sa, _, _) = run(&f, &ClusterAnalysis::knear(6));
        let (b, sb, _, _) = run(&f, &ClusterAnalysis::cutoff(2.01));
        assert_eq!(sa, sb, "sizes must agree");
        assert_eq!(a, b, "and so must the labels");
        assert_eq!(sa.len(), 1, "a 3x3x3 simple-cubic block is one cluster");
        assert_eq!(sa[0], 27);
    }

    #[test]
    fn a_bond_is_kept_even_when_only_one_end_lists_the_other() {
        // The k-nearest relation is not symmetric, and a `j < i` filter would
        // drop such a bond with no error at all. This frame makes it decisive.
        //
        // Atoms 0..4 are a dense 5-atom cluster at spacing 0.5 A; atom 5 sits
        // alone, 20 A away. With k=4 each cluster atom's four nearest are all
        // inside the cluster, so none of them lists atom 5 -- but atom 5's four
        // nearest are four of them. Atom 5 also has the *highest* index, which is
        // the half a `j > i` filter throws away.
        let mut pos: Vec<[f64; 3]> = (0..5).map(|i| [20.0 + 0.5 * i as f64, 0.0, 0.0]).collect();
        pos.push([0.0, 0.0, 0.0]);
        let f = frame(UnitCell::orthorhombic(200.0, 200.0, 200.0), &pos);

        let (labels, sizes, _, sc) = run(&f, &ClusterAnalysis::knear(4));
        assert_eq!(sizes, vec![6], "the lone atom joins: {labels:?}");
        assert!(labels[..5].iter().all(|&l| l == labels[0]));
        assert_eq!(labels[5], labels[0]);
        assert_eq!(scalar(&sc, "cluster_count"), 1.0);
    }

    #[test]
    fn cluster_size_is_registered_as_per_cluster_not_per_atom() {
        // One entry per cluster, flagged `per_particle = false`. The atom writer
        // filters on that flag, so the default `register_scalar` would publish a
        // per-cluster column as if it were per-atom, and nothing else in the
        // library would catch it.
        let f = dimers();
        let nk = NeighborhoodContext { cell: None };
        let ctx = AnalysisContext {
            frame: &f,
            nkeys: &nk,
            par: &Par::new(1),
        };
        let mut store = PropertyStore::new();
        let mut sc = Summary::default();
        ClusterAnalysis::cutoff(2.0)
            .run(&ctx, &mut store, &mut sc)
            .unwrap();
        assert!(store.get("Cluster").unwrap().per_particle);
        assert!(store.get("ParticleCluster").unwrap().per_particle);
        assert!(
            !store.get("ClusterSize").unwrap().per_particle,
            "ClusterSize has one row per cluster"
        );
        assert_eq!(store.get("ClusterSize").unwrap().len(), 3);
    }

    #[test]
    fn the_size_histogram_covers_every_cluster_exactly_once() {
        let (_, sizes, _, sc) = run(&dimers(), &ClusterAnalysis::cutoff(2.0));
        let hist = sc
            .histograms
            .iter()
            .find(|(k, _)| k == "cluster_size_histogram")
            .unwrap();
        let total: f64 = hist.1.iter().sum();
        assert_eq!(total, sizes.len() as f64);
        assert_eq!(hist.1[2], 3.0, "three clusters of size two");
    }

    #[test]
    fn an_unusable_cutoff_is_refused() {
        let f = dimers();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut store = PropertyStore::new();
            let mut sc = Summary::default();
            let e = ClusterAnalysis::cutoff(bad)
                .run(&ctx, &mut store, &mut sc)
                .unwrap_err()
                .to_string();
            assert!(e.contains("cutoff"), "{bad}: {e}");
        }
    }

    #[test]
    fn an_unusable_knear_is_refused() {
        let f = dimers();
        let nk = NeighborhoodContext { cell: None };
        let ctx = AnalysisContext {
            frame: &f,
            nkeys: &nk,
            par: &Par::new(1),
        };
        for bad in [0usize, 6, 99] {
            let mut store = PropertyStore::new();
            let mut sc = Summary::default();
            assert!(
                ClusterAnalysis::knear(bad)
                    .run(&ctx, &mut store, &mut sc)
                    .is_err(),
                "knear {bad} was accepted"
            );
        }
    }

    #[test]
    fn an_empty_frame_is_not_an_error() {
        let f = frame(UnitCell::orthorhombic(10.0, 10.0, 10.0), &[]);
        let (labels, sizes, pc, sc) = run(&f, &ClusterAnalysis::cutoff(2.0));
        assert!(labels.is_empty() && sizes.is_empty() && pc.is_empty());
        assert_eq!(scalar(&sc, "cluster_count"), 0.0);
        assert_eq!(scalar(&sc, "cluster_largest_fraction"), 0.0);
        assert_eq!(scalar(&sc, "cluster_mean_size"), 0.0);
    }

    #[test]
    fn a_single_atom_is_its_own_cluster() {
        let f = frame(UnitCell::orthorhombic(10.0, 10.0, 10.0), &[[5.0, 5.0, 5.0]]);
        let (labels, sizes, _, sc) = run(&f, &ClusterAnalysis::cutoff(2.0));
        assert_eq!(labels, vec![0]);
        assert_eq!(sizes, vec![1]);
        assert_eq!(scalar(&sc, "cluster_largest_fraction"), 1.0);
    }

    #[test]
    fn the_result_does_not_depend_on_the_shared_cell_list_being_present() {
        // `cutoffs` reports nothing for KNear and the analysis must cope; the
        // converse (a list too narrow to answer) is the truncation hazard.
        assert!(ClusterAnalysis::knear(4).cutoffs().is_empty());
        assert_eq!(ClusterAnalysis::cutoff(2.5).cutoffs(), vec![2.5]);
    }

    // ---- selection, sorting, unwrapping and per-cluster properties ----

    fn run_store(f: &Frame, c: &ClusterAnalysis, threads: usize) -> (PropertyStore, Summary) {
        let nk = NeighborhoodContext { cell: None };
        let par = Par::new(threads);
        let ctx = AnalysisContext {
            frame: f,
            nkeys: &nk,
            par: &par,
        };
        let mut store = PropertyStore::new();
        let mut sc = Summary::default();
        c.run(&ctx, &mut store, &mut sc).unwrap();
        (store, sc)
    }

    fn f64s(store: &PropertyStore, name: &str) -> Vec<f64> {
        store
            .get(name)
            .unwrap_or_else(|| panic!("no {name}"))
            .as_f64()
            .unwrap()
            .clone()
    }

    fn i32s(store: &PropertyStore, name: &str) -> Vec<i32> {
        store.get(name).unwrap().as_i32().unwrap().clone()
    }

    /// The frame `tools/cluster_ovito_expected.py` builds, generated here with
    /// the same LCG and the same floating-point operations.
    fn synthetic(pbc: bool) -> Frame {
        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> f64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (self.0 >> 11) as f64 / (1u64 << 53) as f64
            }
        }
        let l = 30.0f64;
        let mut r = Lcg(7);
        let mut p = ParticleData::with_capacity(600);
        let mut sel = Vec::new();
        let mut id = 0i64;
        for _ in 0..40 {
            let c = [r.next() * l, r.next() * l, r.next() * l];
            let m = 3 + (r.next() * 20.0).floor() as usize;
            for _ in 0..m {
                let mut q = [0.0; 3];
                for d in 0..3 {
                    let x = c[d] + (r.next() * 4.0 - 2.0);
                    q[d] = x - l * (x / l).floor();
                }
                let ty = if r.next() < 0.7 { 1 } else { 2 };
                sel.push(u8::from(r.next() < 0.75));
                id += 1;
                p.add_atom(q, id, ty);
            }
        }
        p.selection = Some(sel);
        let mut f = Frame::new(UnitCell::orthorhombic(l, l, l).with_pbc([pbc, pbc, pbc]), p);
        f.type_masses = vec![26.9815385, 58.6934];
        f
    }

    #[test]
    fn matches_ovito_on_synthetic_frames() {
        // Expected values from OVITO 3.15 (tools/cluster_ovito_expected.py):
        // ids exactly (OVITO 3.17's: 0-based, -1 unselected), sizes exactly,
        // centre of mass, Rg and gyration tensor to 1e-9 relative.
        let text = include_str!("../../../../tests/data/cluster_ovito_expected.txt");
        let mut cases: Vec<(String, Vec<i32>, Vec<Vec<f64>>)> = Vec::new();
        for line in text.lines() {
            let mut w = line.split_whitespace();
            match w.next() {
                Some("case") => cases.push((w.next().unwrap().to_owned(), vec![], vec![])),
                Some("ids") => {
                    cases.last_mut().unwrap().1 = w.map(|v| v.parse().unwrap()).collect()
                }
                Some("row") => cases
                    .last_mut()
                    .unwrap()
                    .2
                    .push(w.map(|v| v.parse().unwrap()).collect()),
                _ => {}
            }
        }
        assert_eq!(cases.len(), 3);
        for (name, ids, rows) in &cases {
            let (pbc, only) = match name.as_str() {
                "periodic" => (true, false),
                "open" => (false, false),
                "selected" => (true, true),
                other => panic!("unknown case {other}"),
            };
            let f = synthetic(pbc);
            let a = ClusterAnalysis {
                only_selected: only,
                properties: true,
                unwrap: true,
                ..ClusterAnalysis::cutoff(1.9)
            };
            let (s, sc) = run_store(&f, &a, 2);
            assert_eq!(&i32s(&s, "Cluster"), ids, "{name}: ids");
            assert_eq!(scalar(&sc, "cluster_mass_weighted"), 1.0);
            let sizes = i32s(&s, "ClusterSize");
            let com = f64s(&s, "ClusterCenterOfMass");
            let rg = f64s(&s, "ClusterRadiusOfGyration");
            let g = f64s(&s, "ClusterGyrationTensor");
            assert_eq!(sizes.len(), rows.len(), "{name}: cluster count");
            let close = |a: f64, b: f64, what: &str| {
                assert!(
                    (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                    "{name} {what}: {a} vs OVITO {b}"
                );
            };
            for (k, row) in rows.iter().enumerate() {
                assert_eq!(row[0] as usize, k);
                assert_eq!(sizes[k] as f64, row[1], "{name}: size of {k}");
                for a in 0..3 {
                    close(com[3 * k + a], row[2 + a], "com");
                }
                close(rg[k], row[5], "rg");
                for c in 0..6 {
                    close(g[6 * k + c], row[6 + c], "gyration");
                }
            }
        }
    }

    #[test]
    fn pair_cutoffs_match_ovito() {
        // Expected from OVITO 3.15 CreateBondsModifier(Pairwise) +
        // ClusterAnalysisModifier(Bonding) (tools/cluster_pair_ovito_expected.py)
        // on the two-species synthetic frame: types by id and by name, a pair
        // left out (never bonded), open boundaries, only_selected.
        let text = include_str!("../../../../tests/data/cluster_pair_ovito_expected.txt");
        type Case = (Vec<String>, Vec<i32>, Vec<Vec<f64>>);
        let mut cases: Vec<Case> = Vec::new();
        for line in text.lines() {
            let mut w = line.split_whitespace();
            match w.next() {
                Some("case") => cases.push((w.map(str::to_owned).collect(), vec![], vec![])),
                Some("ids") => {
                    cases.last_mut().unwrap().1 = w.map(|v| v.parse().unwrap()).collect()
                }
                Some("row") => cases
                    .last_mut()
                    .unwrap()
                    .2
                    .push(w.map(|v| v.parse().unwrap()).collect()),
                _ => {}
            }
        }
        assert_eq!(cases.len(), 4);
        for (head, ids, rows) in &cases {
            let name = &head[0];
            let (pbc, only) = (head[1] == "1", head[2] == "1");
            let pairs: Vec<PairCutoff> = head[3..]
                .iter()
                .map(|s| PairCutoff::parse(s).unwrap())
                .collect();
            let mut f = synthetic(pbc);
            f.type_labels = vec!["Al".into(), "Ni".into()];
            let a = ClusterAnalysis {
                mode: ClusterMode::PairCutoffs { pairs },
                only_selected: only,
                properties: true,
                unwrap: true,
                ..Default::default()
            };
            let (s, _) = run_store(&f, &a, 2);
            assert_eq!(&i32s(&s, "Cluster"), ids, "{name}: ids");
            let sizes = i32s(&s, "ClusterSize");
            let com = f64s(&s, "ClusterCenterOfMass");
            let rg = f64s(&s, "ClusterRadiusOfGyration");
            let g = f64s(&s, "ClusterGyrationTensor");
            assert_eq!(sizes.len(), rows.len(), "{name}: cluster count");
            let close = |a: f64, b: f64, what: &str| {
                assert!(
                    (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                    "{name} {what}: {a} vs OVITO {b}"
                );
            };
            for (k, row) in rows.iter().enumerate() {
                assert_eq!(sizes[k] as f64, row[1], "{name}: size of {k}");
                for a in 0..3 {
                    close(com[3 * k + a], row[2 + a], "com");
                }
                close(rg[k], row[5], "rg");
                for c in 0..6 {
                    close(g[6 * k + c], row[6 + c], "gyration");
                }
            }
        }
    }

    #[test]
    fn pair_cutoffs_equal_a_uniform_cutoff_when_every_pair_has_it() {
        let f = synthetic(true);
        let uni = run_store(&f, &ClusterAnalysis::cutoff(1.9), 1).0;
        let pairs = ["1-1=1.9", "2-1=1.9", "2-2=1.9"]
            .iter()
            .map(|s| PairCutoff::parse(s).unwrap())
            .collect();
        let pair = ClusterAnalysis {
            mode: ClusterMode::PairCutoffs { pairs },
            ..Default::default()
        };
        assert_eq!(
            i32s(&run_store(&f, &pair, 1).0, "Cluster"),
            i32s(&uni, "Cluster")
        );
    }

    #[test]
    fn bad_pair_cutoffs_are_refused() {
        for s in ["1-2", "1=2.0", "1-2-3=2", "0-1=2", "1-2=x", "-1=2"] {
            assert!(PairCutoff::parse(s).is_err(), "{s}");
        }
        let p = PairCutoff::parse(" Cu - 2 = 3.5").unwrap();
        assert_eq!(p.a, TypeKey::Name("Cu".into()));
        assert_eq!(p.b, TypeKey::Id(2));
        assert_eq!(p.cutoff, 3.5);
        let f = synthetic(true);
        let err = |specs: &[&str]| {
            let a = ClusterAnalysis {
                mode: ClusterMode::PairCutoffs {
                    pairs: specs
                        .iter()
                        .map(|s| PairCutoff::parse(s).unwrap())
                        .collect(),
                },
                ..Default::default()
            };
            let nk = NeighborhoodContext { cell: None };
            let par = Par::new(1);
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &par,
            };
            a.run(&ctx, &mut PropertyStore::new(), &mut Summary::default())
                .unwrap_err()
                .to_string()
        };
        // The synthetic frame has no type names, and no type 3.
        assert!(err(&["Al-Al=2"]).contains("'Al'"));
        assert!(err(&["1-3=2"]).contains("type 3"));
        assert!(err(&["1-2=2", "2-1=3"]).contains("twice"));
        assert!(err(&["1-2=0"]).contains("positive"));
        assert!(err(&["1-2=NaN"]).contains("positive"));
    }

    #[test]
    fn bonds_are_identical_at_any_thread_count() {
        // A frame large enough for many bond tasks; every output, floats
        // included, bit for bit.
        let mut f = synthetic(true);
        // Tile it 3x3x3 so there are ~14k atoms across many tasks.
        let base = f.particles.clone();
        let mut p = ParticleData::with_capacity(base.positions.len() * 27);
        let mut id = 0;
        for t in 0..27 {
            let s = [
                (t % 3) as f64 * 30.0,
                ((t / 3) % 3) as f64 * 30.0,
                (t / 9) as f64 * 30.0,
            ];
            for (q, &ty) in base.positions.iter().zip(&base.types) {
                id += 1;
                p.add_atom([q[0] + s[0], q[1] + s[1], q[2] + s[2]], id, ty);
            }
        }
        f = Frame {
            particles: p,
            cell: UnitCell::orthorhombic(90.0, 90.0, 90.0),
            ..f
        };
        for mode in [
            ClusterMode::Cutoff { cutoff: 1.9 },
            ClusterMode::KNear { k: 3 },
            ClusterMode::PairCutoffs {
                pairs: ["1-1=1.9", "1-2=1.5", "2-2=2.3"]
                    .iter()
                    .map(|s| PairCutoff::parse(s).unwrap())
                    .collect(),
            },
        ] {
            let a = ClusterAnalysis {
                mode: mode.clone(),
                properties: true,
                unwrap: true,
                sort_by_size: true,
                ..Default::default()
            };
            let (s1, _) = run_store(&f, &a, 1);
            let (s4, _) = run_store(&f, &a, 4);
            for name in ["Cluster", "ClusterSize", "ParticleCluster"] {
                assert_eq!(i32s(&s1, name), i32s(&s4, name), "{mode:?} {name}");
            }
            for name in [
                "ClusterMass",
                "ClusterCenterOfMass",
                "ClusterRadiusOfGyration",
                "ClusterGyrationTensor",
                "UnwrappedPosition",
            ] {
                let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
                assert_eq!(
                    bits(f64s(&s1, name)),
                    bits(f64s(&s4, name)),
                    "{mode:?} {name}"
                );
            }
            // And the unweighted path (no offsets tracked) agrees on the labels.
            let plain = ClusterAnalysis {
                mode: mode.clone(),
                ..Default::default()
            };
            let (p1, _) = run_store(&f, &plain, 1);
            let (p4, _) = run_store(&f, &plain, 4);
            assert_eq!(i32s(&p1, "Cluster"), i32s(&p4, "Cluster"));
        }
    }

    #[test]
    fn the_periodic_probe_matches_ovito_by_hand() {
        // Three atoms bonded through the x boundary (masses 1, 3, 3) and a
        // dimer. OVITO 3.15 with per-type masses gives, for the first cluster:
        // COM (-4/7, 5.2571428571, 5), Rg 0.57711457, G_xx 0.24489796,
        // G_yy 0.08816327, G_xy -0.11020408; unwrapped x 0.5, -0.5, -1.0.
        let mut p = ParticleData::with_capacity(5);
        for (i, (q, t)) in [
            ([0.5, 5.0, 5.0], 1),
            ([9.5, 5.0, 5.0], 2),
            ([9.0, 5.6, 5.0], 2),
            ([3.0, 3.0, 3.0], 1),
            ([3.5, 3.0, 3.0], 1),
        ]
        .into_iter()
        .enumerate()
        {
            p.add_atom(q, i as i64 + 1, t);
        }
        let mut f = Frame::new(UnitCell::orthorhombic(10.0, 10.0, 10.0), p);
        f.type_masses = vec![1.0, 3.0];
        let a = ClusterAnalysis {
            properties: true,
            unwrap: true,
            sort_by_size: true,
            ..ClusterAnalysis::cutoff(1.5)
        };
        let (s, _) = run_store(&f, &a, 1);
        assert_eq!(i32s(&s, "Cluster"), vec![0, 0, 0, 1, 1]);
        let com = f64s(&s, "ClusterCenterOfMass");
        assert!((com[0] + 4.0 / 7.0).abs() < 1e-12);
        assert!((com[1] - 5.257142857142857).abs() < 1e-12);
        assert!((f64s(&s, "ClusterRadiusOfGyration")[0] - 0.577114569).abs() < 1e-8);
        let g = f64s(&s, "ClusterGyrationTensor");
        assert!((g[0] - 0.244897959).abs() < 1e-8);
        assert!((g[1] - 0.088163265).abs() < 1e-8);
        assert!((g[3] + 0.110204082).abs() < 1e-8);
        let u = f64s(&s, "UnwrappedPosition");
        assert_eq!([u[0], u[3], u[6]], [0.5, -0.5, -1.0]);
        assert_eq!(f64s(&s, "ClusterMass"), vec![7.0, 2.0]);

        // Without masses the weights are uniform.
        f.type_masses.clear();
        let (s, sc) = run_store(&f, &a, 1);
        assert_eq!(scalar(&sc, "cluster_mass_weighted"), 0.0);
        let com = f64s(&s, "ClusterCenterOfMass");
        assert!((com[0] - (0.5 - 0.5 - 1.0) / 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_cluster_bonded_to_its_own_image_is_flagged() {
        // A ring of ten through the x boundary percolates; the dimer does not.
        let mut pos: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 5.0, 5.0]).collect();
        pos.push([3.0, 2.0, 2.0]);
        pos.push([3.0, 2.0, 3.0]);
        let f = frame(UnitCell::orthorhombic(10.0, 10.0, 10.0), &pos);
        let a = ClusterAnalysis {
            properties: true,
            ..ClusterAnalysis::cutoff(1.5)
        };
        let (s, sc) = run_store(&f, &a, 1);
        assert_eq!(i32s(&s, "ClusterPercolates"), vec![1, 0]);
        assert_eq!(scalar(&sc, "cluster_percolating"), 1.0);
        // Open along x, the same chain is an ordinary cluster.
        let mut g = f.clone();
        g.cell = g.cell.with_pbc([false, true, true]);
        let (s, _) = run_store(&g, &a, 1);
        assert_eq!(i32s(&s, "ClusterPercolates"), vec![0, 0]);
    }

    #[test]
    fn sort_by_size_renumbers_largest_first_ties_in_atom_order() {
        // Sizes 2, 2, 5 in atom order -> 5 first, then the two dimers in order.
        let f = frame(
            UnitCell::orthorhombic(40.0, 40.0, 40.0),
            &[
                [20.0, 0.0, 0.0],
                [21.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [31.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
        );
        let a = ClusterAnalysis {
            sort_by_size: true,
            ..ClusterAnalysis::cutoff(1.5)
        };
        let (s, _) = run_store(&f, &a, 1);
        assert_eq!(i32s(&s, "ClusterSize"), vec![5, 2, 2]);
        assert_eq!(i32s(&s, "Cluster"), vec![1, 1, 2, 2, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn a_selection_clusters_only_the_selected_atoms() {
        // A chain of five at spacing 1; dropping the middle atom splits it.
        let mut f = frame(
            UnitCell::orthorhombic(40.0, 40.0, 40.0),
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
        );
        f.particles.types[2] = 2;
        let by_expr = ClusterAnalysis {
            select: Some(Selection::parse("type == 1").unwrap()),
            ..ClusterAnalysis::cutoff(1.5)
        };
        let (s, sc) = run_store(&f, &by_expr, 1);
        assert_eq!(i32s(&s, "Cluster"), vec![0, 0, -1, 1, 1]);
        assert_eq!(i32s(&s, "ParticleCluster"), vec![0, 0, -1, 1, 1]);
        assert_eq!(i32s(&s, "ClusterSize"), vec![2, 2]);
        assert_eq!(scalar(&sc, "cluster_atoms_selected"), 4.0);
        assert_eq!(scalar(&sc, "cluster_largest_fraction"), 0.5);

        // The same through a Selection column, and refused without one.
        let only = ClusterAnalysis {
            only_selected: true,
            ..ClusterAnalysis::cutoff(1.5)
        };
        let nk = NeighborhoodContext { cell: None };
        let par = Par::new(1);
        let ctx = AnalysisContext {
            frame: &f,
            nkeys: &nk,
            par: &par,
        };
        assert!(only
            .run(&ctx, &mut PropertyStore::new(), &mut Summary::default())
            .is_err());
        f.particles.selection = Some(vec![1, 1, 0, 1, 1]);
        let (s, _) = run_store(&f, &only, 1);
        assert_eq!(i32s(&s, "Cluster"), vec![0, 0, -1, 1, 1]);

        // k-nearest bonds only among the selected atoms: with k=1 the far
        // pair cannot reach across the unselected atom to the near pair.
        let kn = ClusterAnalysis {
            only_selected: true,
            ..ClusterAnalysis::knear(1)
        };
        let (s, _) = run_store(&f, &kn, 1);
        assert_eq!(i32s(&s, "Cluster"), vec![0, 0, -1, 1, 1]);

        // Unwrapped positions of unselected atoms are their own.
        let uw = ClusterAnalysis {
            unwrap: true,
            ..by_expr
        };
        let (s, _) = run_store(&f, &uw, 1);
        assert_eq!(&f64s(&s, "UnwrappedPosition")[6..9], &[2.0, 0.0, 0.0]);
    }
}
