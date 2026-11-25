//! Local Wigner–Seitz census (design §16.3d): a grain counted without one
//! lattice for all of it.
//!
//! A grain that bends, holds a dislocation network or meets its own image
//! across a relaxed seam has no lattice that puts all its atoms on sites, and
//! the global and patch references refuse it. Defects are local, though: a
//! few bonds from any defect the crystal is perfect, whatever it does further
//! away. So each atom off its grain's lattice (PTM structure not the grain's)
//! is put on a site of a lattice fitted to the lattice atoms around it, and
//! the census runs over those sites only; every other atom is on its own site
//! and adds nothing.
//!
//! **Local lattice.** `T` is the grain's homogeneous fit refined per chunk
//! (about ten bonds on a side) from its lattice atoms' first shells, which
//! follows rotation and strain across the grain. The origin is fitted per
//! defect atom to the lattice atoms among its nearest neighbours, seeded from
//! the nearest few and keeping the seed most of them agree with: next to a
//! dislocation that is the near side's lattice, not an average of both.
//!
//! **Sites.** A lattice atom's site is where it is. A defect atom's site is
//! the local lattice's site nearest to it; atoms whose sites coincide (within
//! 0.35 bonds) share one, so a dumbbell is two atoms on a site: one
//! interstitial. Empty sites come from each defect atom's first and second
//! shells, and join under the domain rule of [`super::census`]: at least half
//! the first shell (plus one) in the domain, repeated, which fills vacancy
//! clusters and stops at a free surface. A site another grain's atom holds is
//! never this grain's.
//!
//! **Conservation.** Every grain atom is on exactly one counted site, so
//! `vacancies − interstitials = sites − atoms` holds per class and in total.
//! A defect atom with too few lattice atoms around it for a fit (deep inside
//! an amorphous pocket) keeps a site of its own and is reported as
//! unresolved.
//!
//! **Not seen atom by atom.** A defect a lattice atom cannot tell from the
//! crystal: the interior of a large interstitial loop is a perfect plane of
//! the crystal locally, so only its rim counts. Clusters small enough are
//! therefore counted whole as regions (step 1b of `census`), against one
//! lattice fitted to the shell around them; a cluster too large for a region
//! (a dislocation network) is still counted atom by atom.
//!
//! **Antisites** (`types` given): read off this census's assignment by a
//! sublattice vote per region ([`antisites`]); the counts above are the same
//! with and without.

mod antisites;

use super::census::{Census, ClassCount};
use super::lattice::{AffineLattice, Motif};
use super::Env;
use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::{Error, Result};
use crate::exec::Par;
use crate::neighbor::knn::Nbr;
use crate::neighbor::{KnnIndex, KnnScratch};
use std::hash::BuildHasherDefault;

/// Maps and sets keyed by atom indices and lattice sites, hashed by a fast
/// multiplicative hash; nothing here depends on their order.
type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<Fx>>;
type HashSet<K> = std::collections::HashSet<K, BuildHasherDefault<Fx>>;

/// Bins per axis for a neighbour grid of Cartesian width `bin`: the cell's
/// heights (not its edge lengths) over `bin`, so a bin is at least `bin`
/// thick across in any cell shape and a point's neighbours within `bin` lie
/// in its own and the 26 adjacent bins.
fn frac_bins(cell: &UnitCell, bin: f64) -> [i64; 3] {
    let h = cell.heights();
    std::array::from_fn(|a| (h[a] / bin).floor().max(1.0) as i64)
}

/// The bin of `c` in a grid of `nb` bins per axis, in fractional
/// coordinates, wrapped on periodic axes. Fractional bins make a wrap by bin
/// index a lattice translation in a triclinic cell too; Cartesian bins do
/// not, and two images of one point would then fall in bins the ±1 lookup
/// never compares (a site on a face of a rhombohedral cell counted once per
/// image).
fn frac_key(cell: &UnitCell, inv_t: &Mat3, nb: [i64; 3], c: Coord3) -> [i64; 3] {
    let l = cell.x2lamda_pre(cell.wrap(c), inv_t);
    std::array::from_fn(|a| {
        let k = (l[a] * nb[a] as f64).floor() as i64;
        if cell.pbc[a] {
            k.rem_euclid(nb[a])
        } else {
            k
        }
    })
}

/// Below this many items a region's step runs on one thread.
const PAR_MIN: usize = 2048;

/// `f` over `v` in order, in `par`'s pool in pieces of at least `min`.
fn pmap<T: Sync, R: Send>(par: &Par, v: &[T], min: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    if par.is_parallel() && v.len() > min {
        use rayon::prelude::*;
        par.install(|| v.par_iter().with_min_len(min).map(&f).collect())
    } else {
        v.iter().map(&f).collect()
    }
}

/// `out[i] = f(i)` in `par`'s pool in pieces of at least `min`: the census
/// does much per item, on lists far shorter than [`Par::fill`]'s chunks.
fn pfill<T: Send>(par: &Par, out: &mut [T], min: usize, f: impl Fn(usize, &mut T) + Sync) {
    if par.is_parallel() && out.len() > min {
        use rayon::prelude::*;
        par.install(|| {
            out.par_iter_mut()
                .enumerate()
                .with_min_len(min)
                .for_each(|(i, o)| f(i, o))
        });
    } else {
        for (i, o) in out.iter_mut().enumerate() {
            f(i, o);
        }
    }
}

/// `f` over `0..n` in ranges of `size`, results in range order.
fn pchunks<T: Send>(
    par: &Par,
    n: usize,
    size: usize,
    f: impl Fn(usize, usize) -> T + Sync,
) -> Vec<T> {
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(size)
        .map(|lo| (lo, (lo + size).min(n)))
        .collect();
    pmap(par, &ranges, 1, |&(lo, hi)| f(lo, hi))
}

/// The rustc "Fx" hash: one rotate, xor and multiply per word.
#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl Fx {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl std::hash::Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(u64::from(b));
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }
    fn write_i32(&mut self, i: i32) {
        self.add(i as u32 as u64);
    }
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

/// Neighbours a defect atom's origin is fitted to (lattice atoms among them).
const K_FRAME: usize = 40;
/// Fewer lattice atoms than this among them, and the atom is unresolved.
const MIN_FRAME: usize = 4;
/// Seeds tried for the origin: the nearest lattice atoms.
const SEEDS: usize = 4;
/// A lattice atom is on the local lattice within this, in bonds.
const INLIER: f64 = 0.3;
/// Two sites (or an atom's site and a point) coincide within this, in bonds.
const SAME_SITE: f64 = 0.35;
/// Empty sites are looked for this far from a defect atom's site, in bonds:
/// the first and second shells of BCC, FCC and HCP.
const NEAR: f64 = 1.45;
/// Chunk edge for the per-chunk `T`, in bonds.
const CHUNK: f64 = 10.0;
/// Lattice atoms whose first shells refine a chunk's `T`.
const CHUNK_SAMPLE: usize = 48;
/// An atom's own `T` may differ from its chunk's by this much (relative,
/// Frobenius) before the chunk's is kept: more is a fit to a frame too
/// lopsided to fix all nine components.
const MAX_LOCAL_STRAIN: f64 = 0.15;
/// Atoms looked at for a point's occupancy.
const K_POINT: usize = 12;
/// An empty site has no atom nearer than this, in bonds. Next to a
/// dislocation, a lattice fitted on one side puts sites between the other
/// side's atoms; a vacant site has its neighbours a bond away.
const FREE: f64 = 0.7;
/// Empty sites closer than this, in bonds, are one site.
const EMPTY_MERGE: f64 = 0.6;
/// Defect atoms closer than this, in bonds, are one cluster.
const LINK: f64 = 1.2;
/// In a region, an atom farther than this from its nearest site of the
/// shell's lattice, in bonds, takes its site from its neighbours' instead.
/// Inside an interstitial loop the crystal is displaced by up to b/2 from
/// that lattice (0.5–0.58 bonds), where two sites are nearly equally near.
const AMBIGUOUS: f64 = 0.3;
/// Empty sites walked before a connected set of them counts as open space
/// rather than a cavity, atom by atom (each step is a neighbour query).
const LOCAL_CAVITY_MAX: usize = 4_000;
/// Neighbours an ambiguous atom's site is continued from: within this, in
/// bonds.
const CONTINUE: f64 = 1.25;
const K_LINK: usize = 14;
/// Only defect atoms with at least this many neighbours within the link
/// distance join clusters: a free surface's atoms have fewer, a loop's rim
/// and a vacancy's shell more, so a surface does not join every defect near
/// it into one cluster.
const COORD_MIN: usize = 12;
/// A cluster with no region alone, joined to nothing, is retried with its
/// sphere grown by one bond up to this many times.
const INFLATE: usize = 3;
/// A join's (or a fragment cluster's) region leaves at most this many empty
/// sites uncounted.
const OPEN_MAX: usize = 2;
/// A join's shell holds at least this fraction of the atoms its sites
/// predict (it does not reach into vacuum).
const SHELL_FULL: f64 = 0.9;
/// A cluster with no region alone is joined to the regions and failed
/// clusters whose spheres come within this of its own, in bonds.
const JOIN: f64 = 2.0;
/// Fragments of clusters smaller than a region are linked again among
/// themselves with this many neighbours (see the clustering).
const COORD_LOW: usize = 11;
/// Clusters of at least this many defect atoms are counted as a region.
const REGION_MIN: usize = 20;
/// A region's radius (centre to its farthest defect atom) is at most this,
/// in bonds; a larger cluster (a dislocation network) is counted atom by atom.
const REGION_MAX: f64 = 30.0;
/// The lattice atoms a region's lattice is fitted to lie between its radius
/// plus these, in bonds; sites within its radius plus the first are its own.
const SHELL_IN: f64 = 1.0;
const SHELL_OUT: f64 = 3.0;
/// A region is counted when at least this fraction of its shell is on the
/// lattice fitted to it: a shell a dislocation crosses is not.
const SHELL_COHERENT: f64 = 0.9;
/// A shell under [`SHELL_COHERENT`] but with at least this fraction on the
/// fitted lattice (strained by defects nearby, as loops a few bonds apart
/// strain each other's shells) is labelled by walking its bonds
/// ([`walk_labels`]) and fitted a smooth lattice ([`fit_curvature`]); the
/// region is counted on that when it puts the shell on its sites.
const SHELL_WALK_MIN: f64 = 0.5;
/// A walked bond is coherent when it is within this of a lattice vector, in
/// bonds.
const WALK_BOND: f64 = 0.35;
/// A walked shell is coherent when at most this fraction of its bonds
/// disagree with the labels (a dislocation crossing it breaks its circuits)
/// and the walk labels at least `WALK_COVER` of its lattice atoms.
const WALK_CONFLICT: f64 = 0.01;
const WALK_COVER: f64 = 0.95;
/// Merge rounds a region may take to absorb the regions it reaches.
const MERGE_ROUNDS: usize = 6;
/// A merged region (a union of its clusters' spheres) spans at most this
/// radius about its first sphere's centre, in bonds; each sphere is still at
/// most [`REGION_MAX`].
const SHAPE_MAX: f64 = 60.0;

/// What is known about a grain beyond its atoms.
pub(super) struct LocalInput<'a> {
    pub cell: &'a UnitCell,
    pub positions: &'a [Coord3],
    pub index: &'a KnnIndex,
    pub grain_of: &'a [i32],
    /// Per frame atom: within the boundary width of another grain.
    pub boundary: &'a [bool],
    /// Per frame atom: its PTM structure is the grain's.
    pub lattice_atom: &'a (dyn Fn(usize) -> bool + Sync),
    /// The grain's atoms (frame indices, ascending).
    pub atoms: &'a [u32],
    pub grain: i32,
    pub motif: &'a Motif,
    /// The grain's homogeneous fit, which seeds every chunk's.
    pub t: Mat3,
    pub env: &'a Env<'a>,
    pub max_domain_iterations: usize,
    /// Per grain atom: its type, to count antisites (`None`: not counted).
    pub types: Option<&'a [i32]>,
    /// How overlapping clusters are merged into regions.
    pub merge: super::LocalMerge,
}

/// What a local census reports beyond its counts.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalStats {
    /// Grain atoms off the grain's lattice: the ones placed by a local fit.
    pub defect_atoms: usize,
    /// Of those, the ones with too few lattice atoms around them for a fit.
    pub unresolved: usize,
    /// Chunks whose `T` was refined.
    pub chunks: usize,
    /// RMS residual of the lattice atoms the origins were fitted to, Å.
    pub frame_rms: f64,
    /// Clusters counted whole against their shell's lattice, and the atoms
    /// counted that way.
    pub regions: usize,
    pub region_atoms: usize,
    /// Defect atoms in clusters too large, or with a shell too incoherent,
    /// to be a region (dislocation lines and networks), counted atom by atom.
    pub extended: usize,
}

/// A defect atom's local lattice and its site in it.
#[derive(Clone, Copy)]
struct Placed {
    /// Site position, Cartesian (the atom's frame, not wrapped).
    site: Coord3,
    /// Basis index of the site.
    k: u8,
    /// The local lattice's `T`.
    t: Mat3,
    /// Lattice atoms the origin fit kept, and their squared residual sum.
    inliers: u16,
    r2: f64,
}

pub(super) fn census(inp: &LocalInput, par: &Par) -> Result<(Census, LocalStats)> {
    let m = inp.motif;
    let cell = inp.cell;
    let inv_t = cell.inverse_transposed()?;
    let nn = {
        let l = AffineLattice::new(m.clone(), inp.t, [0.0; 3])
            .ok_or_else(|| Error::invalid("singular grain lattice"))?;
        l.nn()
    };
    let in_grain = |j: usize| inp.grain_of[j] == inp.grain;
    let is_lat = |j: usize| in_grain(j) && (inp.lattice_atom)(j);

    let _st = crate::prof::StageTimer::start("defects.grain.local.chunks");
    // Chunks: a grid over the box, about CHUNK bonds on a side.
    let lengths = cell.lengths();
    let nc: [usize; 3] =
        std::array::from_fn(|a| ((lengths[a] / (CHUNK * nn)).floor() as usize).max(1));
    let chunk_of = |x: Coord3| -> u32 {
        let l = cell.x2lamda_pre(x, &inv_t);
        let c: [usize; 3] = std::array::from_fn(|a| {
            let f = l[a] - l[a].floor();
            ((f * nc[a] as f64) as usize).min(nc[a] - 1)
        });
        ((c[0] * nc[1] + c[1]) * nc[2] + c[2]) as u32
    };
    // A total order on positions: chunk, then the wrapped coordinates. Every
    // order below (which region is accepted first, which atom's chunk seeds
    // a fit, which lattice atoms sample a chunk) follows positions through
    // it, never the order of atoms in the file, so a permuted file counts
    // the same.
    let place_key = |x: Coord3| -> (u32, [u64; 3]) { (chunk_of(x), super::place_key(cell, x)) };

    // Defect atoms, by grain-local index, in position order.
    let mut defect: Vec<u32> = (0..inp.atoms.len() as u32)
        .filter(|&k| !(inp.lattice_atom)(inp.atoms[k as usize] as usize))
        .collect();
    defect.sort_by_cached_key(|&k| place_key(inp.positions[inp.atoms[k as usize] as usize]));
    let mut stats = LocalStats {
        defect_atoms: defect.len(),
        ..Default::default()
    };
    let defect_chunk: Vec<u32> = defect
        .iter()
        .map(|&k| chunk_of(inp.positions[inp.atoms[k as usize] as usize]))
        .collect();
    let mut wanted: Vec<u32> = defect_chunk.clone();
    wanted.sort_unstable();
    wanted.dedup();
    let chunk_slot: HashMap<u32, usize> = wanted.iter().enumerate().map(|(s, &c)| (c, s)).collect();
    // Each chunk is sampled by about CHUNK_SAMPLE of its lattice atoms,
    // spread over it and chosen by position (not the nearest to its centre:
    // a loop at the centre would bias the fit).
    let mut in_chunk: Vec<Vec<u32>> = vec![Vec::new(); wanted.len()];
    for &i in inp.atoms {
        let i = i as usize;
        if !(inp.lattice_atom)(i) {
            continue;
        }
        if let Some(&s) = chunk_slot.get(&chunk_of(inp.positions[i])) {
            in_chunk[s].push(i as u32);
        }
    }
    let samples: Vec<Vec<u32>> = in_chunk
        .iter()
        .map(|v| super::place_sample(cell, v, |i| inp.positions[i as usize], CHUNK_SAMPLE))
        .collect();
    let z = m.coordination();
    let serial = Par::new(1);
    let mut chunk_t = vec![inp.t; wanted.len()];
    pfill(par, &mut chunk_t, 1, |s, out| {
        let mut sc = KnnScratch::default();
        let mut nb = Vec::new();
        let mut pairs = Vec::new();
        for &i in &samples[s] {
            inp.index.knn(
                i as usize,
                inp.positions[i as usize],
                super::fit::fit_neighbours(z),
                &mut sc,
                &mut nb,
            );
            pairs.extend(nb.iter().filter(|e| is_lat(e.0 as usize)).map(|e| e.1));
        }
        if let Some((t, _)) = super::fit::refine(m, inp.t, &pairs, 3, &serial) {
            *out = t;
        }
    });
    stats.chunks = wanted.len();

    // Where each motif site's near sites and first shell are, fractional.
    let basis = &m.basis;
    let nb_basis = basis.len();
    let near: Vec<Vec<(Coord3, u8)>> = (0..nb_basis)
        .map(|k| {
            let mut v = Vec::new();
            for l in 0..nb_basis {
                for dx in -2..=2 {
                    for dy in -2..=2 {
                        for dz in -2..=2 {
                            let f = [
                                dx as f64 + basis[l][0] - basis[k][0],
                                dy as f64 + basis[l][1] - basis[k][1],
                                dz as f64 + basis[l][2] - basis[k][2],
                            ];
                            let c = m.cell.mul_v(f);
                            let d = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                            if d > 1e-9 && d <= NEAR * m.nn {
                                v.push((f, l as u8));
                            }
                        }
                    }
                }
            }
            v
        })
        .collect();
    let shell: Vec<Vec<(Coord3, u8)>> = (0..nb_basis)
        .map(|k| {
            m.shell[k]
                .iter()
                .map(|&(dn, l)| {
                    let b = basis[l as usize];
                    (
                        [
                            dn[0] as f64 + b[0] - basis[k][0],
                            dn[1] as f64 + b[1] - basis[k][1],
                            dn[2] as f64 + b[2] - basis[k][2],
                        ],
                        l,
                    )
                })
                .collect()
        })
        .collect();

    drop(_st);
    let _st = crate::prof::StageTimer::start("defects.grain.local.place");
    // 1. Each defect atom on its local lattice.
    let inlier2 = (INLIER * nn).powi(2);
    let mut placed: Vec<Option<Placed>> = vec![None; defect.len()];
    pfill(par, &mut placed, 64, |q, out| {
        thread_local! {
            static BUF: std::cell::RefCell<(KnnScratch, Vec<Nbr>)> =
                std::cell::RefCell::new((KnnScratch::default(), Vec::new()));
        }
        let i = inp.atoms[defect[q] as usize] as usize;
        let cs = chunk_slot[&defect_chunk[q]];
        let t = chunk_t[cs];
        BUF.with(|b| {
            let (sc, nb) = &mut *b.borrow_mut();
            inp.index.knn(i, inp.positions[i], K_FRAME, sc, nb);
            let frame: Vec<Coord3> = nb
                .iter()
                .filter(|e| is_lat(e.0 as usize))
                .map(|e| e.1)
                .collect();
            if frame.len() < MIN_FRAME {
                return;
            }
            let Some(Frame {
                t, ti, o, n_in, r2, ..
            }) = fit_frame(t, basis, &frame, inlier2, &serial)
            else {
                return;
            };
            let (k, d) = nearest_site(&t, &ti, basis, o, [0.0; 3]);
            let x = inp.positions[i];
            *out = Some(Placed {
                site: [x[0] + d[0], x[1] + d[1], x[2] + d[2]],
                k,
                t,
                inliers: n_in as u16,
                r2,
            });
        });
    });
    {
        let (n, r2) = placed.iter().flatten().fold((0usize, 0.0), |(n, s), p| {
            (n + p.inliers as usize, s + p.r2)
        });
        stats.frame_rms = (r2 / n.max(1) as f64).sqrt();
        stats.unresolved = placed.iter().filter(|p| p.is_none()).count();
    }

    drop(_st);
    let _st = crate::prof::StageTimer::start("defects.grain.local.links");
    // Grain-local index -> defect slot.
    let mut dslot: HashMap<u32, u32> =
        HashMap::with_capacity_and_hasher(defect.len(), Default::default());
    for (q, &k) in defect.iter().enumerate() {
        dslot.insert(inp.atoms[k as usize], q as u32);
    }
    // 1b. Regions. A cluster of defect atoms can hide content its own atoms
    // do not show: inside an interstitial loop the extra plane is perfect
    // crystal locally, and only the rim is off the lattice. A cluster small
    // enough is counted whole instead: a Wigner–Seitz census of every atom
    // within it against one lattice fitted to the coherent shell of lattice
    // atoms around it, which sees the plane as atoms over sites.
    let link2 = (LINK * nn).powi(2);
    let links: Vec<Vec<(u32, u32, bool)>> = pchunks(par, defect.len(), 1024, |lo, hi| {
        let mut sc = KnnScratch::default();
        let mut nb = Vec::new();
        let mut v = Vec::new();
        for (q, &d) in defect.iter().enumerate().take(hi).skip(lo) {
            let i = inp.atoms[d as usize] as usize;
            if inp.boundary[i] {
                continue;
            }
            inp.index.knn(i, inp.positions[i], K_LINK, &mut sc, &mut nb);
            let coord = nb.iter().filter(|e| e.2 < link2).count();
            if coord < COORD_LOW {
                continue;
            }
            let dense = coord >= COORD_MIN;
            for &(j, _, d2) in nb.iter() {
                if d2 < link2 {
                    if let Some(&r) = dslot.get(&j) {
                        v.push((q as u32, r, dense));
                    }
                }
            }
        }
        v
    });
    // A link counts when both ends are bulk-like (the list holds each
    // dense atom's links, so a pair seen from both ends is dense-dense);
    // `low` links have both ends at COORD_LOW or more, for the fragments.
    type Pairs = Vec<(u32, u32)>;
    let (links, low): (Vec<Pairs>, Pairs) = {
        let mut all: Vec<(u32, u32, bool)> = links
            .into_iter()
            .flatten()
            .map(|(a, b, d)| (a.min(b), a.max(b), d))
            .collect();
        all.sort_unstable();
        let mut both = Vec::new();
        let mut low = Vec::new();
        for w in all.windows(2) {
            if (w[0].0, w[0].1) == (w[1].0, w[1].1) {
                low.push((w[0].0, w[0].1));
                if w[0].2 && w[1].2 {
                    both.push((w[0].0, w[0].1));
                }
            }
        }
        both.dedup();
        low.dedup();
        (vec![both], low)
    };
    let mut cl: Vec<u32> = (0..defect.len() as u32).collect();
    for (a, b) in links.into_iter().flatten() {
        let (x, y) = (find(&mut cl, a), find(&mut cl, b));
        if x != y {
            cl[x.max(y) as usize] = x.min(y);
        }
    }
    let groups: Vec<(Vec<u32>, bool)> = {
        let mut by: HashMap<u32, Vec<u32>> = HashMap::default();
        for q in 0..defect.len() as u32 {
            let r = find(&mut cl, q);
            by.entry(r).or_default().push(q);
        }
        type Rooted = Vec<(u32, Vec<u32>)>;
        let (g, small): (Rooted, Rooted) = by.into_iter().partition(|(_, v)| v.len() >= REGION_MIN);
        // Fragments: a collapsed vacancy loop's rim has atoms with 7-13
        // neighbours, so at COORD_MIN its rim falls apart into clusters too
        // small to be regions and the loop's content is lost (an isolated
        // a/2<111> loop of 54 vacancies counts 14). The atoms of such
        // fragments are linked again among themselves at COORD_LOW (free
        // surfaces stay out: their outer layers have 7-10 neighbours); the
        // clusters already large enough are left as they are, as linking
        // them more would join a loop to its neighbours into one too large.
        let in_small: HashMap<u32, u32> = small
            .iter()
            .flat_map(|(r, v)| v.iter().map(move |&q| (q, *r)))
            .collect();
        let mut cl2: Vec<u32> = (0..defect.len() as u32).collect();
        for &(a, b) in &low {
            if in_small.contains_key(&a) && in_small.contains_key(&b) {
                let (x, y) = (find(&mut cl2, a), find(&mut cl2, b));
                if x != y {
                    cl2[x.max(y) as usize] = x.min(y);
                }
            }
        }
        let mut by2: HashMap<u32, Vec<u32>> = HashMap::default();
        for &q in in_small.keys() {
            let r = find(&mut cl2, q);
            by2.entry(r).or_default().push(q);
        }
        let mut g: Vec<(u32, Vec<u32>, bool)> = g.into_iter().map(|(r, v)| (r, v, false)).collect();
        for (r, mut v) in by2 {
            if v.len() >= REGION_MIN {
                v.sort_unstable();
                g.push((r, v, true));
            }
        }
        g.sort_unstable_by_key(|e| e.0);
        g.into_iter().map(|e| (e.1, e.2)).collect()
    };
    // Whether each group was linked from fragments (it must have a full
    // shell: a free face cut at a random angle has atoms with 11-13
    // neighbours, and a region round them reaches into the vacuum).
    let (groups, fragment): (Vec<Vec<u32>>, Vec<bool>) = groups.into_iter().unzip();
    let min_period = (0..3)
        .filter(|&a| cell.pbc[a])
        .map(|a| lengths[a])
        .fold(f64::INFINITY, f64::min);
    let max_r = (REGION_MAX * nn).min(0.5 * min_period - (SHELL_OUT + 1.0) * nn);
    let sphere_one = |qs: &[u32]| -> Sphere {
        let x0 = inp.positions[inp.atoms[defect[qs[0] as usize] as usize] as usize];
        let rel: Vec<Coord3> = qs
            .iter()
            .map(|&q| {
                let x = inp.positions[inp.atoms[defect[q as usize] as usize] as usize];
                cell.min_image_with_shift_pre([x[0] - x0[0], x[1] - x0[1], x[2] - x0[2]], &inv_t)
                    .0
            })
            .collect();
        let mut cen = [0.0; 3];
        for r in &rel {
            for a in 0..3 {
                cen[a] += r[a] / rel.len() as f64;
            }
        }
        let r = rel
            .iter()
            .map(|p| {
                ((p[0] - cen[0]).powi(2) + (p[1] - cen[1]).powi(2) + (p[2] - cen[2]).powi(2)).sqrt()
            })
            .fold(0.0, f64::max);
        Sphere::one(
            cell.wrap([x0[0] + cen[0], x0[1] + cen[1], x0[2] + cen[2]]),
            r,
        )
    };
    let dist = |a: Coord3, b: Coord3| -> f64 {
        let (d, _) = cell.min_image_with_shift_pre([a[0] - b[0], a[1] - b[1], a[2] - b[2]], &inv_t);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    // The shape of a merged region over `m`, from the shapes of its parts
    // (the cluster that merges, the regions and clusters it takes in): one
    // sphere round all of `m` while that is at most `max_r`; past it, the
    // union of the parts' shapes. A part keeps its own
    // shape, which holds its features whole (a loop whose rim is several
    // clusters is one sphere round them, its plane inside), so the union's
    // edge stays next to them.
    let sphere_of = |m: &[u32], parts: &[Sphere]| -> Sphere {
        let one = sphere_one(m);
        if one.r <= max_r || parts.len() < 2 {
            return one;
        }
        let all: Vec<(Coord3, f64)> = parts.iter().flat_map(|p| p.balls_abs()).collect();
        let c = all[0].0;
        // Each sphere at the image next to the placed sphere it comes
        // closest to: the parts touch in a chain, which may span past half
        // the box from `c`.
        let mut balls: Vec<(Coord3, f64)> = vec![([0.0; 3], all[0].1)];
        for &(b, rb) in &all[1..] {
            let mut best = ([0.0; 3], f64::INFINITY);
            for (k, &(pr, rp)) in balls.iter().enumerate() {
                let pa = all[k].0;
                let off = cell
                    .min_image_with_shift_pre([b[0] - pa[0], b[1] - pa[1], b[2] - pa[2]], &inv_t)
                    .0;
                let gap = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt() - rb - rp;
                if gap < best.1 {
                    best = ([pr[0] + off[0], pr[1] + off[1], pr[2] + off[2]], gap);
                }
            }
            balls.push((best.0, rb));
        }
        let r = balls
            .iter()
            .map(|(d, rb)| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() + rb)
            .fold(0.0, f64::max);
        Sphere {
            c: cell.wrap(c),
            r,
            balls: std::sync::Arc::new(balls),
        }
    };
    let shape_max = SHAPE_MAX * nn;
    // A merged region must not meet its own image: along each periodic axis
    // its spheres and their shells span less than the box (an orthogonal
    // box; a skewed one keeps it within half the shortest period of its
    // first sphere's centre).
    let orthogonal = cell.is_orthogonal();
    let wraps = |sp: &Sphere| -> bool {
        if !orthogonal {
            return sp.r > 0.5 * min_period - (SHELL_OUT + 1.0) * nn;
        }
        let pad = (SHELL_OUT + 1.0) * nn;
        (0..3).filter(|&a| cell.pbc[a]).any(|a| {
            let lo = sp
                .balls
                .iter()
                .map(|(b, r)| b[a] - r)
                .fold(f64::INFINITY, f64::min);
            let hi = sp
                .balls
                .iter()
                .map(|(b, r)| b[a] + r)
                .fold(f64::NEG_INFINITY, f64::max);
            hi - lo + 2.0 * pad >= lengths[a]
        })
    };
    // Too large to be a region: one cluster's sphere past `max_r`, or a
    // merged region with a sphere past it or spanning past `shape_max`.
    let too_big = |sp: &Sphere| -> bool {
        if sp.balls.is_empty() {
            sp.r > max_r
        } else {
            sp.r > shape_max || sp.balls.iter().any(|b| b.1 > max_r) || wraps(sp)
        }
    };
    // Whether two shapes come within `margin` of each other: some ball of
    // one within `margin` of some ball of the other.
    let close = |a: &Sphere, b: &Sphere, margin: f64| -> bool {
        a.balls_abs().iter().any(|&(ca, ra)| {
            b.balls_abs()
                .iter()
                .any(|&(cb, rb)| dist(ca, cb) < ra + rb + margin)
        })
    };
    // The spheres a partial region keeps out of, grown to the sites counted.
    let excl_balls = |sp: &Sphere| -> Vec<(Coord3, f64)> {
        if sp.balls.is_empty() {
            return vec![(sp.c, sp.r + SHELL_IN * nn)];
        }
        sp.balls_abs()
            .into_iter()
            .map(|(c, r)| (cell.wrap(c), r + SHELL_IN * nn))
            .collect()
    };
    let mut extended = 0usize;

    let spheres: Vec<Sphere> = groups.iter().map(|g| sphere_one(g)).collect();
    let mut todo: Vec<usize> = Vec::new();
    for (k, sp) in spheres.iter().enumerate() {
        if sp.r <= max_r {
            todo.push(k);
        } else {
            extended += groups[k].len();
        }
    }
    let ctx = RegionCtx {
        inp,
        basis,
        shell: &shell,
        nn,
        inlier2,
        need: z / 2 + 1,
        par,
    };
    drop(_st);
    let _st = crate::prof::StageTimer::start("defects.grain.local.regions");
    let shell_full = |sp: &Sphere, r: &RegionOut| -> bool {
        if !sp.balls.is_empty() {
            // A merged region counts its shell's sites and atoms itself.
            return r.band_sites > 0 && r.band_atoms as f64 >= SHELL_FULL * r.band_sites as f64;
        }
        let r_in = sp.r + SHELL_IN * nn;
        let r_out = sp.r + SHELL_OUT * nn;
        let sites = (r.interior.sites + r.boundary.sites) as f64;
        if sites == 0.0 {
            return false;
        }
        let per_site = 4.0 / 3.0 * std::f64::consts::PI * r_in.powi(3) / sites;
        let want = 4.0 / 3.0 * std::f64::consts::PI * (r_out.powi(3) - r_in.powi(3)) / per_site;
        let mut nb: Vec<Nbr> = Vec::new();
        inp.index.within_point(sp.c, r_out, &mut nb);
        let have = nb
            .iter()
            .filter(|e| e.2 >= r_in * r_in && inp.grain_of[e.0 as usize] == inp.grain)
            .count() as f64;
        have >= SHELL_FULL * want
    };
    // A join or a cluster linked from fragments may not cut a void either:
    // the void's sites inside it would reach its edge and go uncounted
    // (fragments of a void's surface, linked at COORD_LOW, would make
    // regions holding part of it: HCP voids -73 V). A single cluster's grown sphere
    // is not held to this (near a free surface it leaves a few vacuum sites
    // open and cuts nothing).
    let whole =
        |sp: &Sphere, r: &RegionOut| -> bool { shell_full(sp, r) && r.open_sites <= OPEN_MAX };
    let mut done: Vec<std::result::Result<RegionOut, &'static str>> =
        (0..todo.len()).map(|_| Err("")).collect();
    pfill(par, &mut done, 1, |w, o| {
        let k = todo[w];
        let q0 = groups[k][0] as usize;
        let t = chunk_t[chunk_slot[&defect_chunk[q0]]];
        *o = region(&ctx, &spheres[k], t, None).and_then(|r| {
            if fragment[k] && !whole(&spheres[k], &r) {
                Err("shell not full")
            } else {
                Ok(r)
            }
        });
    });
    drop(_st);
    let _st = crate::prof::StageTimer::start("defects.grain.local.merge");
    // Accept regions in order; one that would count an atom another already
    // counts merges with it (and any other it meets) when the union is still
    // a region, and is left to the atom-by-atom count otherwise.
    //
    // A union census, `region(sphere, T, None)`, reads nothing of the merge's
    // state: it is a function of its sphere and its chunk's `T`. So before a
    // batch of clusters is merged in order, each cluster's chain of union
    // censuses is worked out in parallel against the state at the batch's
    // start, and the in-order merge takes a census from that cache when it
    // asks for the same sphere and `T`, and counts it itself otherwise. The
    // result is the in-order one at any thread count; a guess the earlier
    // clusters of the batch make wrong costs only the wasted census. On the
    // 128M frame the in-order merge kept ~8 of 80 cores busy, and one census
    // barely runs faster on 12 threads than on 1 (design §16.3d).
    let frame_atom = |q: u32| inp.atoms[defect[q as usize] as usize];
    type Slot = (Sphere, RegionOut, Vec<u32>);
    type UKey = ([u64; 5], usize);
    type UOut = std::result::Result<RegionOut, &'static str>;
    // Appends slot `h`'s members and shape.
    type MembersInto<'m> = dyn Fn(usize, &mut Vec<u32>, &mut Vec<Sphere>) + Sync + 'm;
    let union_key = |sp: &Sphere, m: &[u32]| -> UKey {
        (
            [
                sp.c[0].to_bits(),
                sp.c[1].to_bits(),
                sp.c[2].to_bits(),
                sp.r.to_bits(),
                sp.balls.iter().fold(0u64, |h, (b, r)| {
                    [b[0], b[1], b[2], *r]
                        .iter()
                        .fold(h, |h, x| (h ^ x.to_bits()).wrapping_mul(0x100_0000_01b3))
                }),
            ],
            chunk_slot[&defect_chunk[m[0] as usize]],
        )
    };
    let union_members = |members0: &[u32], hit: &[usize], slots: &[Option<Slot>]| -> Vec<u32> {
        let mut m = members0.to_vec();
        for &h in hit {
            m.extend_from_slice(&slots[h].as_ref().unwrap().2);
        }
        m
    };
    let hits_of = |atoms: &[u32], owner: &HashMap<u32, usize>| -> Vec<usize> {
        let mut hit: Vec<usize> = atoms.iter().filter_map(|j| owner.get(j).copied()).collect();
        hit.sort_unstable();
        hit.dedup();
        hit
    };
    // The union censuses a cluster would ask for against a fixed state:
    // `owner_of` an atom, and `members_into` appending a slot's members.
    let speculate = |members0: &[u32],
                     sp0: &Sphere,
                     r: &RegionOut,
                     owner_of: &(dyn Fn(u32) -> Option<usize> + Sync),
                     members_into: &MembersInto|
     -> Vec<(UKey, UOut)> {
        let mut out = Vec::new();
        let mut hit: Vec<usize> = r.atoms.iter().filter_map(|&j| owner_of(j)).collect();
        hit.sort_unstable();
        hit.dedup();
        if hit.is_empty() {
            return out;
        }
        for _ in 0..MERGE_ROUNDS {
            let mut m = members0.to_vec();
            let mut parts = vec![sp0.clone()];
            for &h in &hit {
                members_into(h, &mut m, &mut parts);
            }
            let sp = sphere_of(&m, &parts);
            if too_big(&sp) {
                break;
            }
            let key = union_key(&sp, &m);
            let res = region(&ctx, &sp, chunk_t[key.1], None);
            let more: Vec<usize> = match &res {
                Ok(mr) => mr
                    .atoms
                    .iter()
                    .filter_map(|&j| owner_of(j))
                    .filter(|o| !hit.contains(o))
                    .collect(),
                Err(_) => Vec::new(),
            };
            out.push((key, res));
            if more.is_empty() {
                break;
            }
            hit.extend(more);
            hit.sort_unstable();
            hit.dedup();
        }
        out
    };
    let accept = |cands: Vec<Cand>| -> (Vec<Slot>, usize) {
        let mut extended = 0usize;
        let mut owner: HashMap<u32, usize> = HashMap::default();
        let mut slots: Vec<Option<Slot>> = Vec::new();
        let n = cands.len();
        let mut rs: Vec<Option<RegionOut>> = Vec::with_capacity(n);
        let mut mems: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut sps: Vec<Sphere> = Vec::with_capacity(n);
        for (sp, r, m) in cands {
            sps.push(sp);
            rs.push(r);
            mems.push(m);
        }
        // Clusters with no region alone, not yet joined to one.
        let mut pending: Vec<Option<(Sphere, Vec<u32>)>> = Vec::new();
        let mut cache: HashMap<UKey, UOut> = HashMap::default();
        let mut spec_end = 0usize;
        let batch = (2 * par.threads).clamp(2, 24);
        for i in 0..n {
            if par.is_parallel() && i >= spec_end {
                // Walk the next clusters in order against the state now: one
                // that meets no region is accepted as it is (so later ones
                // see it), one that meets some is speculated on, up to
                // `batch` of them.
                cache.clear();
                let mut extra_owner: HashMap<u32, usize> = HashMap::default();
                let mut extra: Vec<usize> = Vec::new();
                let mut pick = Vec::new();
                let mut k = i;
                while k < n && pick.len() < batch && k < i + 64 * batch {
                    if let Some(r) = &rs[k] {
                        let meets = r
                            .atoms
                            .iter()
                            .any(|j| owner.contains_key(j) || extra_owner.contains_key(j));
                        if meets {
                            pick.push(k);
                        } else {
                            let id = slots.len() + extra.len();
                            for &j in &r.atoms {
                                extra_owner.insert(j, id);
                            }
                            extra.push(k);
                        }
                    }
                    k += 1;
                }
                spec_end = k;
                if !pick.is_empty() {
                    let _st = crate::prof::StageTimer::start("defects.grain.local.merge.speculate");
                    let got: Vec<Vec<(UKey, UOut)>> = {
                        use rayon::prelude::*;
                        let (owner, slots, rs, mems) = (&owner, &slots, &rs, &mems);
                        let (extra_owner, extra) = (&extra_owner, &extra);
                        let base = slots.len();
                        let owner_of =
                            |j: u32| owner.get(&j).or_else(|| extra_owner.get(&j)).copied();
                        let members_into = |h: usize, m: &mut Vec<u32>, parts: &mut Vec<Sphere>| {
                            if h < base {
                                let s = slots[h].as_ref().unwrap();
                                m.extend_from_slice(&s.2);
                                parts.push(s.0.clone());
                            } else {
                                m.extend_from_slice(&mems[extra[h - base]]);
                                parts.push(sps[extra[h - base]].clone());
                            }
                        };
                        par.install(|| {
                            pick.par_iter()
                                .with_min_len(1)
                                .map(|&k| {
                                    speculate(
                                        &mems[k],
                                        &sps[k],
                                        rs[k].as_ref().unwrap(),
                                        &owner_of,
                                        &members_into,
                                    )
                                })
                                .collect()
                        })
                    };
                    cache.extend(got.into_iter().flatten());
                }
            }
            let sp0 = sps[i].clone();
            let members0 = std::mem::take(&mut mems[i]);
            let Some(r) = rs[i].take() else {
                // No region alone: its shell is crossed, as when it is an arc
                // of a loop (a loop's rim split into clusters). It is joined
                // to the regions and the failed clusters next to it, to a
                // fixed point as a union is; when no such union is a region
                // it waits for the failed clusters after it, and what is left
                // at the end is counted atom by atom. Counting it atom by atom
                // at once would lose a loop's content: a 216-atom loop whose
                // rim is three clusters counts none of it, and the 2M-atom
                // ground truth's interstitial loops lose 229 of 2010 (49 with
                // the join; design §16.8a).
                let near = |c: &Sphere| close(c, &sp0, JOIN * nn);
                let mut hit: Vec<usize> = (0..slots.len())
                    .filter(|&h| slots[h].as_ref().is_some_and(|x| near(&x.0)))
                    .collect();
                let mut pend: Vec<usize> = (0..pending.len())
                    .filter(|&k| pending[k].as_ref().is_some_and(|x| near(&x.0)))
                    .collect();
                let mut res = None;
                if !hit.is_empty() || !pend.is_empty() {
                    for _ in 0..MERGE_ROUNDS {
                        let mut m = union_members(&members0, &hit, &slots);
                        let mut parts = vec![sp0.clone()];
                        parts.extend(hit.iter().map(|&h| slots[h].as_ref().unwrap().0.clone()));
                        for &k in &pend {
                            m.extend_from_slice(&pending[k].as_ref().unwrap().1);
                            parts.push(pending[k].as_ref().unwrap().0.clone());
                        }
                        let sp = sphere_of(&m, &parts);
                        if too_big(&sp) {
                            break;
                        }
                        let key = union_key(&sp, &m);
                        let census = match cache.remove(&key) {
                            Some(c) => c,
                            None => {
                                let _ut = crate::prof::StageTimer::start(
                                    "defects.grain.local.merge.join",
                                );
                                region(&ctx, &sp, chunk_t[key.1], None)
                            }
                        };
                        let Ok(mr) = census else {
                            break;
                        };
                        // A join's sphere is larger than its parts' and may
                        // reach past a free surface: a shell half in vacuum
                        // is still coherent over the atoms it has, and the
                        // census then counts the vacuum (a 68 A join at the
                        // face of the 128M cut gave 1212 V and 709 I where
                        // the region it replaced had 185 and 76). Its shell
                        // must be full: as many grain atoms as the sites the
                        // census found inside it predict.
                        if !whole(&sp, &mr) {
                            break;
                        }
                        let counted: HashSet<u32> = mr.atoms.iter().copied().collect();
                        let mut more: Vec<usize> = mr
                            .atoms
                            .iter()
                            .filter_map(|j| owner.get(j).copied())
                            .filter(|o| !hit.contains(o))
                            .collect();
                        let more_p: Vec<usize> = (0..pending.len())
                            .filter(|&k| {
                                !pend.contains(&k)
                                    && pending[k].as_ref().is_some_and(|x| {
                                        x.1.iter().any(|&q| counted.contains(&frame_atom(q)))
                                    })
                            })
                            .collect();
                        if more.is_empty() && more_p.is_empty() {
                            res = Some((sp, mr, m));
                            break;
                        }
                        more.sort_unstable();
                        more.dedup();
                        hit.extend(more);
                        hit.sort_unstable();
                        pend.extend(more_p);
                        pend.sort_unstable();
                    }
                }
                // Alone and joined to nothing, a whole loop can still fail:
                // its shell, a bond outside the rim, is in the loop's
                // strongest strain. A larger sphere's shell is in weaker
                // strain (it falls off as 1/d^2): try a bond, two, three
                // more, while it meets no region.
                if res.is_none() {
                    for k in 1..=INFLATE {
                        let sp = Sphere::one(sp0.c, sp0.r + k as f64 * nn);
                        if sp.r > max_r {
                            break;
                        }
                        let t = chunk_t[chunk_slot[&defect_chunk[members0[0] as usize]]];
                        let _ut =
                            crate::prof::StageTimer::start("defects.grain.local.merge.inflate");
                        let Ok(mr) = region(&ctx, &sp, t, None) else {
                            continue;
                        };
                        if !shell_full(&sp, &mr) || mr.atoms.iter().any(|j| owner.contains_key(j)) {
                            break;
                        }
                        hit.clear();
                        pend.clear();
                        res = Some((sp, mr, members0.clone()));
                        break;
                    }
                }
                match res {
                    Some((sp, mr, m)) => {
                        for &h in &hit {
                            let (_, old, _) = slots[h].take().unwrap();
                            for j in &old.atoms {
                                owner.remove(j);
                            }
                        }
                        for &k in &pend {
                            pending[k] = None;
                        }
                        let id = slots.len();
                        for &j in &mr.atoms {
                            owner.insert(j, id);
                        }
                        slots.push(Some((sp, mr, m)));
                    }
                    None => pending.push(Some((sp0, members0))),
                }
                continue;
            };
            let mut hit = hits_of(&r.atoms, &owner);
            let (sp, r, members) = if hit.is_empty() {
                (sp0, r, members0)
            } else {
                // Absorb every region the union reaches, to a fixed point.
                let mut res = None;
                for _ in 0..MERGE_ROUNDS {
                    let m = union_members(&members0, &hit, &slots);
                    let mut parts = vec![sp0.clone()];
                    parts.extend(hit.iter().map(|&h| slots[h].as_ref().unwrap().0.clone()));
                    let sp = sphere_of(&m, &parts);
                    if too_big(&sp) {
                        break;
                    }
                    let key = union_key(&sp, &m);
                    let census = match cache.remove(&key) {
                        Some(c) => c,
                        None => {
                            let _ut =
                                crate::prof::StageTimer::start("defects.grain.local.merge.union");
                            region(&ctx, &sp, chunk_t[key.1], None)
                        }
                    };
                    let mr = match census {
                        Ok(mr) => mr,
                        Err(_) => break,
                    };
                    // A merged region, as a join: a full shell and no void it
                    // cuts (one sphere round a union is not held to this).
                    if !sp.balls.is_empty() && !whole(&sp, &mr) {
                        break;
                    }
                    let mut more: Vec<usize> = mr
                        .atoms
                        .iter()
                        .filter_map(|j| owner.get(j).copied())
                        .filter(|o| !hit.contains(o))
                        .collect();
                    if more.is_empty() {
                        res = Some((sp, mr, m));
                        break;
                    }
                    more.sort_unstable();
                    more.dedup();
                    hit.extend(more);
                    hit.sort_unstable();
                }
                let Some((sp, mr, m)) = res else {
                    // The union is no region (too large, or its shell is
                    // incoherent: a dislocation between them). Count the
                    // cluster as a partial region: what its own census
                    // finds outside the accepted regions. The atom-by-atom
                    // count would lose a loop's content: on thin film 999 it
                    // gives the substrate I 1038 for global's 2052 (2005
                    // with partial regions), and the film V 11 897 for
                    // patch's 12 963 (12 965).
                    let spheres: Vec<(Coord3, f64)> = slots
                        .iter()
                        .flatten()
                        .flat_map(|(s, _, _)| excl_balls(s))
                        .collect();
                    let excl = Exclusion {
                        owned: &owner,
                        spheres: &spheres,
                    };
                    let q0 = members0[0] as usize;
                    let t = chunk_t[chunk_slot[&defect_chunk[q0]]];
                    let _pt = crate::prof::StageTimer::start("defects.grain.local.merge.partial");
                    if let Ok(pr) = region(&ctx, &sp0, t, Some(&excl)) {
                        let id = slots.len();
                        for &j in &pr.atoms {
                            owner.insert(j, id);
                        }
                        slots.push(Some((sp0, pr, members0)));
                        continue;
                    }
                    extended += members0.len();
                    continue;
                };
                for &h in &hit {
                    let (_, old, _) = slots[h].take().unwrap();
                    for j in &old.atoms {
                        owner.remove(j);
                    }
                }
                (sp, mr, m)
            };
            let id = slots.len();
            for &j in &r.atoms {
                owner.insert(j, id);
            }
            slots.push(Some((sp, r, members)));
        }
        // The clusters still pending, joined among themselves. A failed
        // cluster's join takes in the regions next to it, and in a crowded
        // grain those reach others in turn until the union is too large
        // (128M frame: 2 regions, then 24, 148 A); joined only with the
        // other failed clusters near it, as the partial rule does, it can be
        // a region (five fragments of one feature, 44.7 A: 442 I where the
        // atom-by-atom count found 379). Such a group is counted as a
        // partial region round the regions already accepted.
        let left: Vec<usize> = (0..pending.len())
            .filter(|&k| pending[k].is_some())
            .collect();
        let mut root: Vec<usize> = (0..left.len()).collect();
        fn find(r: &mut [usize], mut x: usize) -> usize {
            while r[x] != x {
                r[x] = r[r[x]];
                x = r[x];
            }
            x
        }
        for a in 0..left.len() {
            for b in a + 1..left.len() {
                let (sa, sb) = (
                    &pending[left[a]].as_ref().unwrap().0,
                    &pending[left[b]].as_ref().unwrap().0,
                );
                if close(sa, sb, JOIN * nn) {
                    let (x, y) = (find(&mut root, a), find(&mut root, b));
                    if x != y {
                        root[x.max(y)] = x.min(y);
                    }
                }
            }
        }
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut at: HashMap<usize, usize> = HashMap::default();
        for (a, &k) in left.iter().enumerate() {
            let r = find(&mut root, a);
            let g = *at.entry(r).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(k);
        }
        for g in groups.into_iter().filter(|g| g.len() > 1) {
            let m: Vec<u32> = g
                .iter()
                .flat_map(|&k| pending[k].as_ref().unwrap().1.iter().copied())
                .collect();
            let parts: Vec<Sphere> = g
                .iter()
                .map(|&k| pending[k].as_ref().unwrap().0.clone())
                .collect();
            let sp = sphere_of(&m, &parts);
            if too_big(&sp) {
                continue;
            }
            let t = chunk_t[chunk_slot[&defect_chunk[m[0] as usize]]];
            let _jt = crate::prof::StageTimer::start("defects.grain.local.merge.join");
            let Ok(whole_r) = region(&ctx, &sp, t, None) else {
                continue;
            };
            if !whole(&sp, &whole_r) {
                continue;
            }
            let counted = if whole_r.atoms.iter().any(|j| owner.contains_key(j)) {
                let spheres: Vec<(Coord3, f64)> = slots
                    .iter()
                    .flatten()
                    .flat_map(|(s, _, _)| excl_balls(s))
                    .collect();
                let excl = Exclusion {
                    owned: &owner,
                    spheres: &spheres,
                };
                match region(&ctx, &sp, t, Some(&excl)) {
                    Ok(pr) => pr,
                    Err(_) => continue,
                }
            } else {
                whole_r
            };
            let id = slots.len();
            for &j in &counted.atoms {
                owner.insert(j, id);
            }
            slots.push(Some((sp, counted, m)));
            for &k in &g {
                pending[k] = None;
            }
        }
        extended += pending.iter().flatten().map(|p| p.1.len()).sum::<usize>();
        (slots.into_iter().flatten().collect(), extended)
    };
    // The chunk `T` a cluster's census starts from (its first atom's chunk).
    let chunk_of_members = |m: &[u32]| chunk_t[chunk_slot[&defect_chunk[m[0] as usize]]];
    let cands: Vec<Cand> = done
        .into_iter()
        .enumerate()
        .map(|(w, r)| {
            let k = todo[w];
            (spheres[k].clone(), r.ok(), groups[k].clone())
        })
        .collect();
    let (slots, ext) = match inp.merge {
        super::LocalMerge::Union => accept(cands),
        super::LocalMerge::Partial => accept_partial(
            &ctx,
            cands,
            &chunk_of_members,
            &PartialJoin {
                sphere_of: &sphere_of,
                shell_full: &shell_full,
                whole: &whole,
                too_big: &too_big,
                max_r,
            },
        ),
    };
    extended += ext;
    let accepted: Vec<(Sphere, RegionOut)> = slots.into_iter().map(|(s, r, _)| (s, r)).collect();
    let region_atom: HashSet<u32> = accepted
        .iter()
        .flat_map(|(_, r)| r.atoms.iter().copied())
        .collect();
    stats.regions = accepted.len();
    stats.region_atoms = region_atom.len();
    stats.extended = extended;
    drop(_st);
    let _st = crate::prof::StageTimer::start("defects.grain.local.sites");
    let sphere_cell = 2.0 * max_r.max(nn) + 2.0 * nn;
    let sng = frac_bins(cell, sphere_cell);
    // Keys wrap on a periodic axis as the lookup below wraps them: in a box
    // not a multiple of the cell (thin film 192 A, the 128M frame 1270 A,
    // cells ~150 A) an unwrapped key leaves a region whose centre lies past
    // the last whole cell unfound, and the atom-by-atom count counts its
    // sites again (an FCC Frank loop's 102 vacancies count 194).
    let skey = |c: Coord3| -> [i64; 3] { frac_key(cell, &inv_t, sng, c) };
    // Every accepted region's spheres (a merged region's each), wrapped.
    let counted_balls: Vec<(Coord3, f64)> = accepted
        .iter()
        .flat_map(|(sp, _)| {
            if sp.balls.is_empty() {
                vec![(sp.c, sp.r)]
            } else {
                sp.balls_abs()
                    .into_iter()
                    .map(|(c, r)| (cell.wrap(c), r))
                    .collect()
            }
        })
        .collect();
    let sgrid: HashMap<[i64; 3], Vec<u32>> = {
        let mut g: HashMap<[i64; 3], Vec<u32>> = HashMap::default();
        for (k, (c, _)) in counted_balls.iter().enumerate() {
            g.entry(skey(cell.wrap(*c))).or_default().push(k as u32);
        }
        g
    };
    // Whether `c` is a site some region counts.
    let in_region = |c: Coord3| -> bool {
        let w = cell.wrap(c);
        let k0 = skey(w);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let mut kk = [k0[0] + dx, k0[1] + dy, k0[2] + dz];
                    for a in 0..3 {
                        if cell.pbc[a] {
                            kk[a] = kk[a].rem_euclid(sng[a]);
                        }
                    }
                    for &o in sgrid.get(&kk).map_or(&[][..], |v| &v[..]) {
                        let (c, r) = counted_balls[o as usize];
                        if dist(w, c) < r + SHELL_IN * nn {
                            return true;
                        }
                    }
                }
            }
        }
        false
    };
    let in_reg = |j: u32| region_atom.contains(&j);

    let site_of = |j: usize| -> Option<Coord3> {
        if (inp.lattice_atom)(j) {
            Some(inp.positions[j])
        } else {
            dslot
                .get(&(j as u32))
                .and_then(|&q| placed[q as usize])
                .map(|p| p.site)
        }
    };
    let same2 = (SAME_SITE * nn).powi(2);
    // The grain atoms whose sites are at `c`.
    let occupants = |c: Coord3, sc: &mut KnnScratch, nb: &mut Vec<Nbr>, out: &mut Vec<u32>| {
        out.clear();
        inp.index.knn_point(c, K_POINT, sc, nb);
        for &(j, e, _) in nb.iter() {
            let j = j as usize;
            if !in_grain(j) {
                continue;
            }
            let Some(s) = site_of(j) else { continue };
            // Site relative to c: (x_j - c) + (s - x_j), in c's image.
            let x = inp.positions[j];
            let r = [e[0] + s[0] - x[0], e[1] + s[1] - x[1], e[2] + s[2] - x[2]];
            if r[0] * r[0] + r[1] * r[1] + r[2] * r[2] < same2 {
                out.push(j as u32);
            }
        }
    };

    // 2. Occupied sites: defect atoms grouped by shared site.
    let edges: Vec<Vec<(u32, u32)>> = pchunks(par, defect.len(), 1024, |lo, hi| {
        let mut sc = KnnScratch::default();
        let mut nb = Vec::new();
        let mut occ = Vec::new();
        let mut v = Vec::new();
        for q in lo..hi {
            let Some(p) = placed[q] else { continue };
            let i = inp.atoms[defect[q] as usize];
            if in_reg(i) {
                continue;
            }
            occupants(p.site, &mut sc, &mut nb, &mut occ);
            v.extend(
                occ.iter()
                    .filter(|&&j| j != i && !in_reg(j))
                    .map(|&j| (q as u32, j)),
            );
        }
        v
    });
    let mut uf: Vec<u32> = (0..defect.len() as u32).collect();
    fn find(uf: &mut [u32], mut a: u32) -> u32 {
        while uf[a as usize] != a {
            uf[a as usize] = uf[uf[a as usize] as usize];
            a = uf[a as usize];
        }
        a
    }
    let mut lattice_link: Vec<(u32, u32)> = Vec::new();
    for (q, j) in edges.into_iter().flatten() {
        match dslot.get(&j) {
            Some(&r) => {
                let (a, b) = (find(&mut uf, q), find(&mut uf, r));
                if a != b {
                    uf[a.max(b) as usize] = a.min(b);
                }
            }
            None => lattice_link.push((j, q)),
        }
    }
    // A lattice atom joins the sites of every defect atom on it.
    lattice_link.sort_unstable();
    for w in lattice_link.windows(2) {
        if w[0].0 == w[1].0 {
            let (a, b) = (find(&mut uf, w[0].1), find(&mut uf, w[1].1));
            if a != b {
                uf[a.max(b) as usize] = a.min(b);
            }
        }
    }
    let mut members: HashMap<u32, Vec<u32>> = HashMap::default();
    for q in 0..defect.len() as u32 {
        let i = inp.atoms[defect[q as usize] as usize];
        if in_reg(i) {
            continue;
        }
        let r = find(&mut uf, q);
        members.entry(r).or_default().push(i);
    }
    let mut lattice_in: HashMap<u32, Vec<u32>> = HashMap::default();
    let mut on_defect_site = HashSet::default();
    for &(j, q) in &lattice_link {
        let r = find(&mut uf, q);
        let v = lattice_in.entry(r).or_default();
        if !v.contains(&j) {
            v.push(j);
        }
        on_defect_site.insert(j);
    }

    let mut out = Census {
        displacement: vec![f32::NAN; inp.atoms.len()],
        domain_converged: true,
        ..Default::default()
    };
    let local_of = |j: u32| inp.atoms.binary_search(&j).expect("grain atom") as u32;
    let mut disp2 = 0.0;
    let mut n_disp = 0usize;
    let mut off = 0usize;
    for (q, p) in placed.iter().enumerate() {
        if let Some(p) = p {
            let x = inp.positions[inp.atoms[defect[q] as usize] as usize];
            let d2 = (0..3).map(|a| (p.site[a] - x[a]).powi(2)).sum::<f64>();
            out.displacement[defect[q] as usize] = d2.sqrt() as f32;
            disp2 += d2;
            n_disp += 1;
            if d2 > (0.35 * nn).powi(2) {
                off += 1;
            }
        }
    }
    out.rms_displacement = (disp2 / n_disp.max(1) as f64).sqrt();
    out.offlattice_fraction = off as f64 / n_disp.max(1) as f64;

    let count = |c: &mut Census, boundary: bool, sites: u64, atoms: u64, v: u64, i: u64| {
        let cc: &mut ClassCount = if boundary {
            &mut c.boundary
        } else {
            &mut c.interior
        };
        cc.sites += sites;
        cc.atoms += atoms;
        cc.vacancies += v;
        cc.interstitials += i;
    };
    // Lattice atoms on no defect atom's site: one atom, one site.
    for &j in inp.atoms {
        if (inp.lattice_atom)(j as usize) && !on_defect_site.contains(&j) && !in_reg(j) {
            count(&mut out, inp.boundary[j as usize], 1, 1, 0, 0);
        }
    }
    let mut roots: Vec<u32> = members.keys().copied().collect();
    roots.sort_unstable();
    for r in roots {
        let d = &members[&r];
        let l = lattice_in.get(&r).map_or(&[][..], |v| &v[..]);
        let unresolved = placed[r as usize].is_none();
        if unresolved {
            // An atom alone on a site of its own.
            debug_assert_eq!(d.len(), 1);
            count(&mut out, inp.boundary[d[0] as usize], 1, 1, 0, 0);
            continue;
        }
        let boundary = d.iter().chain(l).any(|&j| inp.boundary[j as usize]);
        let n = (d.len() + l.len()) as u64;
        count(&mut out, boundary, 1, n, 0, n - 1);
        if n > 1 {
            // A lattice atom keeps the site; then the defect atom nearest it.
            let mut by: Vec<(f32, u32)> = d
                .iter()
                .map(|&j| (out.displacement[local_of(j) as usize], j))
                .collect();
            by.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let skip = usize::from(l.is_empty());
            for &(_, j) in by.iter().skip(skip) {
                out.interstitials.push((local_of(j), boundary));
            }
        }
    }

    drop(_st);
    let _st = crate::prof::StageTimer::start("defects.grain.local.cand");
    // 3. Empty sites: candidates around defect atoms, then the domain rule.
    let need = z / 2 + 1;
    let free2 = (FREE * nn).powi(2);
    let claim = |c: Coord3| inp.env.zone(c, None);
    // Candidates: position, basis and the local `T` that put them there.
    let cand: Vec<Vec<(Coord3, u8, Mat3)>> = pchunks(par, defect.len(), 1024, |lo, hi| {
        let mut sc = KnnScratch::default();
        let mut nb = Vec::new();
        let mut occ = Vec::new();
        let mut v = Vec::new();
        for q in lo..hi {
            let Some(p) = &placed[q] else { continue };
            if in_reg(inp.atoms[defect[q] as usize]) {
                continue;
            }
            let t = p.t;
            for &(f, l) in &near[p.k as usize] {
                let w = t.mul_v(f);
                let c = [p.site[0] + w[0], p.site[1] + w[1], p.site[2] + w[2]];
                occupants(c, &mut sc, &mut nb, &mut occ);
                if occ.is_empty() && nb.first().map_or(true, |e| e.2 >= free2) && !in_region(c) {
                    v.push((c, l, p.t));
                }
            }
        }
        v
    });
    let merge2 = (EMPTY_MERGE * nn).powi(2);
    let bin = EMPTY_MERGE * nn;
    let nbins = frac_bins(cell, bin);
    let key = |c: Coord3| -> [i64; 3] { frac_key(cell, &inv_t, nbins, c) };
    let close = |a: Coord3, b: Coord3| -> bool {
        let (d, _) = cell.min_image_with_shift_pre([a[0] - b[0], a[1] - b[1], a[2] - b[2]], &inv_t);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < merge2
    };
    struct Empty {
        pos: Coord3,
        k: u8,
        t: Mat3,
        accepted: bool,
        dead: bool,
    }
    let mut empties: Vec<Empty> = Vec::new();
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::default();
    let lookup =
        |grid: &HashMap<[i64; 3], Vec<u32>>, empties: &[Empty], c: Coord3| -> Option<u32> {
            let k0 = key(c);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let mut k = [k0[0] + dx, k0[1] + dy, k0[2] + dz];
                        for a in 0..3 {
                            if cell.pbc[a] {
                                k[a] = k[a].rem_euclid(nbins[a]);
                            }
                        }
                        if let Some(v) = grid.get(&k) {
                            if let Some(&e) = v.iter().find(|&&e| close(empties[e as usize].pos, c))
                            {
                                return Some(e);
                            }
                        }
                    }
                }
            }
            None
        };
    let add = |grid: &mut HashMap<[i64; 3], Vec<u32>>,
               empties: &mut Vec<Empty>,
               c: Coord3,
               k: u8,
               t: Mat3|
     -> bool {
        if lookup(grid, empties, c).is_some() || in_region(c) {
            return false;
        }
        let (claimed, _) = claim(c);
        let mut kk = key(c);
        for a in 0..3 {
            if cell.pbc[a] {
                kk[a] = kk[a].rem_euclid(nbins[a]);
            }
        }
        grid.entry(kk).or_default().push(empties.len() as u32);
        empties.push(Empty {
            pos: c,
            k,
            t,
            accepted: false,
            dead: claimed,
        });
        true
    };
    for (c, k, ch) in cand.into_iter().flatten() {
        add(&mut grid, &mut empties, c, k, ch);
    }
    drop(_st);
    let _st = crate::prof::StageTimer::start("defects.grain.local.domain");
    // Rounds of the domain rule; each round may add the new sites' empty
    // first shells as candidates.
    let mut iterations = 0;
    loop {
        let todo: Vec<u32> = (0..empties.len() as u32)
            .filter(|&e| !empties[e as usize].accepted && !empties[e as usize].dead)
            .collect();
        let counts: Vec<(usize, Vec<(Coord3, u8)>)> = {
            let (g, em) = (&grid, &empties);
            let mut v = vec![(0usize, Vec::new()); todo.len()];
            pfill(par, &mut v, 16, |q, o| {
                let mut sc = KnnScratch::default();
                let mut nb = Vec::new();
                let mut occ = Vec::new();
                let e = &em[todo[q] as usize];
                let t = e.t;
                for &(f, l) in &shell[e.k as usize] {
                    let w = t.mul_v(f);
                    let c = [e.pos[0] + w[0], e.pos[1] + w[1], e.pos[2] + w[2]];
                    occupants(c, &mut sc, &mut nb, &mut occ);
                    if !occ.is_empty() {
                        o.0 += 1;
                    } else if let Some(x) = lookup(g, em, c) {
                        if em[x as usize].accepted {
                            o.0 += 1;
                        }
                    } else if nb.first().map_or(true, |e| e.2 >= free2) {
                        o.1.push((c, l));
                    }
                }
            });
            v
        };
        let mut grew = false;
        let mut fresh = Vec::new();
        for (q, (n, new)) in counts.into_iter().enumerate() {
            if n >= need {
                let e = todo[q] as usize;
                empties[e].accepted = true;
                grew = true;
                let ch = empties[e].t;
                fresh.extend(new.into_iter().map(|(c, l)| (c, l, ch)));
            }
        }
        iterations += 1;
        if !grew {
            break;
        }
        if iterations >= inp.max_domain_iterations {
            out.domain_converged = false;
            break;
        }
        for (c, l, ch) in fresh {
            add(&mut grid, &mut empties, c, l, ch);
        }
    }
    // Cavities, as in the census: no site of a compact void has the held
    // neighbours the rule needs. A connected set of empty sites, walked
    // through each site's local lattice, that closes within
    // LOCAL_CAVITY_MAX sites without wrapping round a periodic axis (a gap
    // through the whole box is two surfaces, not a void), touches no region,
    // and has no site claimed by or beside another grain, is a void, counted
    // whole.
    {
        // 0 unseen, 1 open, 2 being walked.
        let mut mark: Vec<u8> = vec![0; empties.len()];
        let mut sc = KnnScratch::default();
        let mut nb = Vec::new();
        let mut occ = Vec::new();
        let starts: Vec<u32> = (0..empties.len() as u32)
            .filter(|&e| !empties[e as usize].accepted && !empties[e as usize].dead)
            .collect();
        for e0 in starts {
            if mark[e0 as usize] != 0 {
                continue;
            }
            let mut comp = vec![e0];
            mark[e0 as usize] = 2;
            // Each walked site's position, unwrapped along the walk.
            let mut at: HashMap<u32, Coord3> = HashMap::default();
            at.insert(e0, empties[e0 as usize].pos);
            let mut open = false;
            let mut q = 0;
            while q < comp.len() && !open {
                let (pos, k, t) = {
                    let e = &empties[comp[q] as usize];
                    (at[&comp[q]], e.k, e.t)
                };
                q += 1;
                for &(f, l) in &shell[k as usize] {
                    let w = t.mul_v(f);
                    let c = [pos[0] + w[0], pos[1] + w[1], pos[2] + w[2]];
                    occupants(c, &mut sc, &mut nb, &mut occ);
                    if !occ.is_empty() {
                        continue;
                    }
                    let x = match lookup(&grid, &empties, c) {
                        Some(x) => x,
                        // An atom close to the site, though not on it: wall.
                        None if nb.first().is_some_and(|e| e.2 < free2) => continue,
                        None => {
                            if !add(&mut grid, &mut empties, c, l, t) {
                                // Inside a region, which counts its own sites.
                                open = true;
                                break;
                            }
                            mark.push(0);
                            (empties.len() - 1) as u32
                        }
                    };
                    let ex = &empties[x as usize];
                    if ex.accepted {
                        continue;
                    }
                    if ex.dead || mark[x as usize] == 1 {
                        open = true;
                        break;
                    }
                    if mark[x as usize] == 0 {
                        mark[x as usize] = 2;
                        at.insert(x, c);
                        comp.push(x);
                    } else {
                        // Met again: through another image if far apart.
                        let a = at[&x];
                        let d2 =
                            (a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2) + (a[2] - c[2]).powi(2);
                        if d2 >= merge2 {
                            open = true;
                            break;
                        }
                    }
                }
                open |= comp.len() > LOCAL_CAVITY_MAX;
            }
            if !open {
                open = comp.iter().any(|&x| {
                    let (claimed, boundary) = claim(empties[x as usize].pos);
                    claimed || boundary
                });
            }
            for &x in &comp {
                if open {
                    mark[x as usize] = 1;
                } else {
                    empties[x as usize].accepted = true;
                }
            }
        }
    }
    drop(_st);
    let _st = crate::prof::StageTimer::start("defects.grain.local.finish");
    out.domain_iterations = iterations;
    for e in empties.iter().filter(|e| e.accepted) {
        let (_, b) = claim(e.pos);
        count(&mut out, b, 1, 0, 1, 0);
        out.vacancies.push((cell.wrap(e.pos), b));
    }
    for (_, r) in &accepted {
        for (o, c) in [
            (&mut out.interior, &r.interior),
            (&mut out.boundary, &r.boundary),
        ] {
            o.sites += c.sites;
            o.atoms += c.atoms;
            o.vacancies += c.vacancies;
            o.interstitials += c.interstitials;
        }
        out.vacancies.extend_from_slice(&r.vacancies);
        out.interstitials
            .extend(r.interstitials.iter().map(|&(j, b)| (local_of(j), b)));
        for &(j, d) in &r.displacement {
            out.displacement[local_of(j) as usize] = d;
        }
    }
    // Antisites (§16.1a): over the singly occupied sites, read off the
    // assignment above, which they do not change.
    if let Some(types) = inp.types {
        let _st = crate::prof::StageTimer::start("defects.grain.local.antisites");
        let mut singles: Vec<antisites::Single> = Vec::new();
        for (k, &j) in inp.atoms.iter().enumerate() {
            if (inp.lattice_atom)(j as usize) && !on_defect_site.contains(&j) && !in_reg(j) {
                singles.push(antisites::Single {
                    atom: k as u32,
                    site: inp.positions[j as usize],
                    boundary: inp.boundary[j as usize],
                    lattice: true,
                });
            }
        }
        for (r, d) in &members {
            let resolved = placed[*r as usize];
            if let (Some(p), [j], None) = (resolved, &d[..], lattice_in.get(r)) {
                singles.push(antisites::Single {
                    atom: local_of(*j),
                    site: p.site,
                    boundary: inp.boundary[*j as usize],
                    lattice: false,
                });
            }
        }
        for (_, r) in &accepted {
            singles.extend(r.singles.iter().map(|&(j, site)| antisites::Single {
                atom: local_of(j),
                site,
                boundary: inp.boundary[j as usize],
                lattice: (inp.lattice_atom)(j as usize),
            }));
        }
        // Neither chemical order nor a host species: not defined.
        let (a, list) = antisites::count(inp, nn, types, &singles, par);
        out.antisite_order = Some((a.order, a.share));
        out.antisites = super::antisite::antisites_defined(a.order, a.share).then_some((a, list));
    }
    for c in [&out.interior, &out.boundary] {
        if c.residual() != 0 {
            return Err(Error::invalid(format!(
                "local census does not conserve atoms (residual {})",
                c.residual()
            )));
        }
    }
    Ok((out, stats))
}

/// Displacement of point `e` from its nearest site of the lattice with
/// origin `o` (both relative to the same point).
#[inline]
fn residual(t: &Mat3, ti: &Mat3, basis: &[Coord3], o: Coord3, e: Coord3) -> Coord3 {
    nearest_site(t, ti, basis, o, e).1.map(|x| -x)
}

/// The site nearest to `e` of the lattice `o + T(n + b_k)`: its basis index
/// and the vector from `e` to it.
#[inline]
fn nearest_site(t: &Mat3, ti: &Mat3, basis: &[Coord3], o: Coord3, e: Coord3) -> (u8, Coord3) {
    let u = ti.mul_v([e[0] - o[0], e[1] - o[1], e[2] - o[2]]);
    let mut best = (0u8, [0.0; 3], f64::INFINITY);
    for (k, b) in basis.iter().enumerate() {
        let n = [
            (u[0] - b[0]).round() + b[0],
            (u[1] - b[1]).round() + b[1],
            (u[2] - b[2]).round() + b[2],
        ];
        let p = t.mul_v(n);
        let d = [o[0] + p[0] - e[0], o[1] + p[1] - e[1], o[2] + p[2] - e[2]];
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if d2 < best.2 {
            best = (k as u8, d, d2);
        }
    }
    (best.0, best.1)
}

/// Lattice atoms of `frame` within the inlier distance of a site of the
/// lattice `o + T(n + b_k)`: their count, squared residual sum and residual
/// sum (atom minus site).
fn inliers(
    t: &Mat3,
    ti: &Mat3,
    basis: &[Coord3],
    o: Coord3,
    frame: &[Coord3],
    inlier2: f64,
) -> (usize, f64, Coord3) {
    let mut n_in = 0;
    let mut r2 = 0.0;
    let mut sum = [0.0; 3];
    for e in frame {
        let d = residual(t, ti, basis, o, *e);
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if d2 < inlier2 {
            n_in += 1;
            r2 += d2;
            for a in 0..3 {
                sum[a] += d[a];
            }
        }
    }
    (n_in, r2, sum)
}

/// Least-squares `T` and `o` from the inliers of `frame` on their current
/// sites: `e = o + T f` over each inlier `e` and its site's fractional
/// coordinate `f`. `None` for too few inliers or a degenerate spread.
fn affine(
    t: &Mat3,
    ti: &Mat3,
    basis: &[Coord3],
    o: Coord3,
    frame: &[Coord3],
    inlier2: f64,
) -> Option<(Mat3, Coord3)> {
    let mut pts: Vec<(Coord3, Coord3)> = Vec::with_capacity(frame.len());
    for e in frame {
        let (_, d) = nearest_site(t, ti, basis, o, *e);
        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] >= inlier2 {
            continue;
        }
        let s = [e[0] + d[0] - o[0], e[1] + d[1] - o[1], e[2] + d[2] - o[2]];
        let f = ti.mul_v(s).map(|x| (x * 2.0 * 3.0).round() / 6.0);
        pts.push((*e, f));
    }
    if pts.len() < 3 * MIN_FRAME {
        return None;
    }
    let n = pts.len() as f64;
    let mut eb = [0.0; 3];
    let mut fb = [0.0; 3];
    for (e, f) in &pts {
        for a in 0..3 {
            eb[a] += e[a] / n;
            fb[a] += f[a] / n;
        }
    }
    let mut xf = Mat3 { m: [[0.0; 3]; 3] };
    let mut ff = Mat3 { m: [[0.0; 3]; 3] };
    for (e, f) in &pts {
        for r in 0..3 {
            for c in 0..3 {
                xf.m[r][c] += (e[r] - eb[r]) * (f[c] - fb[c]);
                ff.m[r][c] += (f[r] - fb[r]) * (f[c] - fb[c]);
            }
        }
    }
    let t2 = xf.mul(&ff.inverse()?);
    let tf = t2.mul_v(fb);
    Some((t2, [eb[0] - tf[0], eb[1] - tf[1], eb[2] - tf[2]]))
}

/// A lattice fitted to a frame of lattice atoms (positions relative to a
/// common point): `sites = o + T(n + b_k)`.
struct Frame {
    t: Mat3,
    ti: Mat3,
    o: Coord3,
    /// Frame atoms within the inlier distance of a site, and their squared
    /// residual sum.
    n_in: usize,
    r2: f64,
    /// A smooth correction to the affine sites, for a strained shell:
    /// `curv[d] · m(u)`, with `m(u)` the [`monomials`] of the site's lattice
    /// coordinates `u = n + b_k`.
    curv: Option<[[f64; 10]; 3]>,
}

/// `1, u, u ⊗ u` (upper triangle): the terms of [`Frame::curv`].
fn monomials(u: Coord3) -> [f64; 10] {
    [
        1.0,
        u[0],
        u[1],
        u[2],
        u[0] * u[0],
        u[1] * u[1],
        u[2] * u[2],
        u[0] * u[1],
        u[0] * u[2],
        u[1] * u[2],
    ]
}

fn curv_at(c: &[[f64; 10]; 3], u: Coord3) -> Coord3 {
    let m = monomials(u);
    std::array::from_fn(|d| c[d].iter().zip(&m).map(|(a, b)| a * b).sum())
}

/// Least squares of the labelled atoms' residuals from their affine sites on
/// [`monomials`]: the correction [`Frame::curv`] holds. `None` when the
/// normal equations are singular (too few or degenerate atoms).
fn fit_curvature(
    f: &Frame,
    basis: &[Coord3],
    pts: &[(u32, Coord3)],
    labels: &[Option<Key>],
) -> Option<[[f64; 10]; 3]> {
    let affine = Frame { curv: None, ..*f };
    let mut a = [[0.0f64; 10]; 10];
    let mut rhs = [[0.0f64; 3]; 10];
    let mut n = 0usize;
    for (&(_, e), l) in pts.iter().zip(labels) {
        let Some((m, k)) = *l else { continue };
        let b = basis[k as usize];
        let u = [m[0] as f64 + b[0], m[1] as f64 + b[1], m[2] as f64 + b[2]];
        let p = site_pos(&affine, basis, (m, k));
        let r = [e[0] - p[0], e[1] - p[1], e[2] - p[2]];
        let x = monomials(u);
        for ((ai, ri), &xi) in a.iter_mut().zip(rhs.iter_mut()).zip(&x) {
            for (aij, &xj) in ai.iter_mut().zip(&x) {
                *aij += xi * xj;
            }
            for (rid, &rd) in ri.iter_mut().zip(&r) {
                *rid += xi * rd;
            }
        }
        n += 1;
    }
    if n < 30 {
        return None;
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..10 {
        let piv = (col..10).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-9 * a[0][0].abs().max(1.0) {
            return None;
        }
        a.swap(col, piv);
        rhs.swap(col, piv);
        for row in 0..10 {
            if row == col {
                continue;
            }
            let g = a[row][col] / a[col][col];
            if g == 0.0 {
                continue;
            }
            for j in col..10 {
                a[row][j] -= g * a[col][j];
            }
            for d in 0..3 {
                rhs[row][d] -= g * rhs[col][d];
            }
        }
    }
    Some(std::array::from_fn(|d| {
        std::array::from_fn(|i| rhs[i][d] / a[i][i])
    }))
}

/// Fit the origin to `frame` under `t`, seeded from its first few atoms
/// (nearest first) on each basis site and keeping the seed most atoms agree
/// with; then refine `T` and `o` together on the inliers. The refined `T`
/// is kept only within [`MAX_LOCAL_STRAIN`] of `t`.
fn fit_frame(
    t: Mat3,
    basis: &[Coord3],
    frame: &[Coord3],
    inlier2: f64,
    par: &Par,
) -> Option<Frame> {
    let ti = t.inverse()?;
    let fit = |o: Coord3| inliers(&t, &ti, basis, o, frame, inlier2);
    let seeds: Vec<Coord3> = frame
        .iter()
        .take(SEEDS)
        .flat_map(|e| {
            basis.iter().map(|b| {
                let tb = t.mul_v(*b);
                [e[0] - tb[0], e[1] - tb[1], e[2] - tb[2]]
            })
        })
        .collect();
    // A large frame tries its seeds in parallel.
    let seed_par = if frame.len() >= PAR_MIN {
        *par
    } else {
        Par::new(1)
    };
    let tried = pmap(&seed_par, &seeds, 1, |&o0| {
        let mut o = o0;
        for _ in 0..2 {
            let (n_in, _, s) = fit(o);
            if n_in == 0 {
                break;
            }
            for a in 0..3 {
                o[a] += s[a] / n_in as f64;
            }
        }
        let (n_in, r2, _) = fit(o);
        (n_in, r2, o)
    });
    let mut best: Option<(usize, f64, Coord3)> = None;
    for (n_in, r2, o) in tried {
        if best.map_or(true, |bb| n_in > bb.0 || (n_in == bb.0 && r2 < bb.1)) {
            best = Some((n_in, r2, o));
        }
    }
    let (n_in, r2, o) = best?;
    if n_in < MIN_FRAME {
        return None;
    }
    let norm = |m: &Mat3| m.m.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
    let mut f = Frame {
        t,
        ti,
        o,
        n_in,
        r2,
        curv: None,
    };
    for _ in 0..2 {
        let Some((t2, o2)) = affine(&f.t, &f.ti, basis, f.o, frame, inlier2) else {
            break;
        };
        let Some(ti2) = t2.inverse() else { break };
        let mut d = t2;
        for (r, row) in d.m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v -= t.m[r][c];
            }
        }
        if norm(&d) > MAX_LOCAL_STRAIN * norm(&t) {
            break;
        }
        let (n2, r22, _) = inliers(&t2, &ti2, basis, o2, frame, inlier2);
        if n2 < f.n_in {
            break;
        }
        f = Frame {
            t: t2,
            ti: ti2,
            o: o2,
            n_in: n2,
            r2: r22,
            curv: None,
        };
    }
    Some(f)
}

/// A region's shape. One cluster's is its sphere: centre (wrapped) and
/// radius, the farthest defect atom. A merged region's is one sphere round
/// all its atoms while that radius is within the region limit, else the
/// union of its clusters' spheres (`balls`, centres relative to `c`, the
/// first one's centre), and `r` then bounds it about `c`. A union of the
/// members' spheres, not one sphere round all of them, keeps the region's
/// edge next to its features: two loops 50 A across that touch need no 113 A sphere,
/// which would hold other features and cut them (design §16.8a).
#[derive(Clone)]
struct Sphere {
    c: Coord3,
    r: f64,
    balls: std::sync::Arc<Vec<(Coord3, f64)>>,
}

impl Sphere {
    fn one(c: Coord3, r: f64) -> Sphere {
        Sphere {
            c,
            r,
            balls: Default::default(),
        }
    }

    /// Whether `p` (relative to `c`; `d2` its squared length) lies within
    /// `extra` of the shape. One sphere tests `d2` directly.
    fn inside(&self, p: Coord3, d2: f64, extra: f64) -> bool {
        if self.balls.is_empty() {
            let r = self.r + extra;
            return d2 < r * r;
        }
        self.balls.iter().any(|&(b, rb)| {
            let r = rb + extra;
            (p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2) + (p[2] - b[2]).powi(2) < r * r
        })
    }

    /// Its spheres, centres absolute (not wrapped).
    fn balls_abs(&self) -> Vec<(Coord3, f64)> {
        if self.balls.is_empty() {
            return vec![(self.c, self.r)];
        }
        self.balls
            .iter()
            .map(|&(b, r)| ([self.c[0] + b[0], self.c[1] + b[1], self.c[2] + b[2]], r))
            .collect()
    }
}

struct RegionCtx<'a> {
    inp: &'a LocalInput<'a>,
    basis: &'a [Coord3],
    /// Per basis site, its first shell (fractional offset, basis).
    shell: &'a [Vec<(Coord3, u8)>],
    nn: f64,
    inlier2: f64,
    need: usize,
    par: &'a Par,
}

/// One region's census.
struct RegionOut {
    /// Frame atoms it counted.
    atoms: Vec<u32>,
    interior: ClassCount,
    boundary: ClassCount,
    vacancies: Vec<(Coord3, bool)>,
    /// Frame indices.
    interstitials: Vec<(u32, bool)>,
    displacement: Vec<(u32, f32)>,
    /// Singly occupied sites: frame atom and site position (absolute).
    singles: Vec<(u32, Coord3)>,
    /// Empty sites inside it left uncounted: neither crystal by the domain
    /// rule nor a cavity, because they reach its edge (a void it cuts, or
    /// vacuum) or another grain.
    open_sites: usize,
    /// A merged region's shell (between SHELL_IN and SHELL_OUT bonds of its
    /// spheres): its lattice sites and the grain atoms in it, for
    /// `shell_full` (0 for one cluster's sphere, whose shell is a spherical
    /// band and is reckoned from its volume).
    band_sites: usize,
    band_atoms: usize,
}

type Key = ([i32; 3], u8);

/// The site of `o + T(n + b_k)` nearest to `e`: its key and position.
fn site_key(f: &Frame, basis: &[Coord3], e: Coord3) -> (Key, Coord3) {
    let mut u = f.ti.mul_v([e[0] - f.o[0], e[1] - f.o[1], e[2] - f.o[2]]);
    if let Some(c) = &f.curv {
        // Invert the smooth map by fixed point (its correction varies
        // slowly with u).
        for _ in 0..3 {
            let d = curv_at(c, u);
            u = f.ti.mul_v([
                e[0] - f.o[0] - d[0],
                e[1] - f.o[1] - d[1],
                e[2] - f.o[2] - d[2],
            ]);
        }
    }
    let mut best = (([0; 3], 0u8), [0.0; 3], f64::INFINITY);
    for (k, b) in basis.iter().enumerate() {
        let n = [
            (u[0] - b[0]).round(),
            (u[1] - b[1]).round(),
            (u[2] - b[2]).round(),
        ];
        let p = site_pos(f, basis, ([n[0] as i32, n[1] as i32, n[2] as i32], k as u8));
        let d2 = (p[0] - e[0]).powi(2) + (p[1] - e[1]).powi(2) + (p[2] - e[2]).powi(2);
        if d2 < best.2 {
            best = (([n[0] as i32, n[1] as i32, n[2] as i32], k as u8), p, d2);
        }
    }
    (best.0, best.1)
}

/// The site an atom at `e` has when its neighbour at `e_n` is on site
/// `from`: the site whose lattice vector from `from` is nearest the bond
/// `e − e_n`, and how far it is from it.
fn continue_key(f: &Frame, basis: &[Coord3], from: Key, e_n: Coord3, e: Coord3) -> (Key, f64) {
    let bond = [e[0] - e_n[0], e[1] - e_n[1], e[2] - e_n[2]];
    let u = f.ti.mul_v(bond);
    let (n, k) = from;
    let bk = basis[k as usize];
    let mut best = (([0; 3], 0u8), f64::INFINITY);
    for (l, bl) in basis.iter().enumerate() {
        let m: [i32; 3] =
            std::array::from_fn(|a| (n[a] as f64 + bk[a] + u[a] - bl[a]).round() as i32);
        let df: Coord3 = std::array::from_fn(|a| m[a] as f64 + bl[a] - n[a] as f64 - bk[a]);
        let v = f.t.mul_v(df);
        let d2 = (v[0] - bond[0]).powi(2) + (v[1] - bond[1]).powi(2) + (v[2] - bond[2]).powi(2);
        if d2 < best.1 {
            best = ((m, l as u8), d2);
        }
    }
    (best.0, best.1.sqrt())
}

/// Labels of a strained region's shell, by walking its bonds: the shell's
/// lattice atoms on the fitted lattice (within `INLIER` of a site) seed a
/// breadth-first walk over the shell's lattice atoms, each bond taking the
/// lattice vector nearest it (within `WALK_BOND`), which follows elastic
/// strain an affine lattice cannot. The walk stays in the shell: through the
/// region it could cross a loop's plane and label the far side a Burgers
/// vector off. The labels must agree all round (a dislocation crossing the
/// shell breaks its circuits): `Err` when more than `WALK_CONFLICT` of the
/// shell's bonds disagree or the walk labels less than `WALK_COVER` of its
/// lattice atoms. `pts` are positions relative to the region's centre;
/// atoms off the shell, or not lattice atoms, get `None`.
fn walk_labels(
    cx: &RegionCtx,
    f: &Frame,
    pts: &[(u32, Coord3)],
    shell_of: &dyn Fn(Coord3) -> bool,
) -> std::result::Result<Vec<Option<Key>>, &'static str> {
    let inp = cx.inp;
    let nn = cx.nn;
    let q_of: HashMap<u32, usize> = pts.iter().enumerate().map(|(q, &(j, _))| (j, q)).collect();
    let lat: Vec<bool> = pts
        .iter()
        .map(|&(j, _)| (inp.lattice_atom)(j as usize))
        .collect();
    let r2 = |e: Coord3| e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    let in_shell = |q: usize| shell_of(pts[q].1);
    let mut lab: Vec<Option<Key>> = vec![None; pts.len()];
    let mut queue = std::collections::VecDeque::new();
    let tol = INLIER * nn;
    for (q, &(_, e)) in pts.iter().enumerate() {
        if !lat[q] || !in_shell(q) {
            continue;
        }
        let (k, p) = site_key(f, cx.basis, e);
        if r2([p[0] - e[0], p[1] - e[1], p[2] - e[2]]) <= tol * tol {
            lab[q] = Some(k);
            queue.push_back(q);
        }
    }
    let reach2 = (CONTINUE * nn).powi(2);
    let bond_tol = WALK_BOND * nn;
    let (mut edges, mut conflicts) = (0usize, 0usize);
    let mut sc = KnnScratch::default();
    let mut nb: Vec<Nbr> = Vec::new();
    while let Some(a) = queue.pop_front() {
        let (j, e) = pts[a];
        let Some(ka) = lab[a] else { continue };
        inp.index.knn(
            j as usize,
            inp.positions[j as usize],
            K_LINK,
            &mut sc,
            &mut nb,
        );
        for &(jn, rel, d2) in nb.iter() {
            if d2 >= reach2 || jn == j {
                continue;
            }
            let Some(&b) = q_of.get(&jn) else { continue };
            if !lat[b] || !in_shell(b) {
                continue;
            }
            let e_b = [e[0] + rel[0], e[1] + rel[1], e[2] + rel[2]];
            let (k, d) = continue_key(f, cx.basis, ka, e, e_b);
            if d > bond_tol {
                continue;
            }
            match lab[b] {
                None => {
                    lab[b] = Some(k);
                    queue.push_back(b);
                }
                Some(kb) => {
                    edges += 1;
                    conflicts += (kb != k) as usize;
                }
            }
        }
    }
    let shell_lat = (0..pts.len()).filter(|&q| lat[q] && in_shell(q)).count();
    let covered = (0..pts.len())
        .filter(|&q| lat[q] && in_shell(q) && lab[q].is_some())
        .count();
    if (covered as f64) < WALK_COVER * shell_lat as f64
        || conflicts as f64 > WALK_CONFLICT * edges.max(1) as f64
    {
        return Err("incoherent shell");
    }
    Ok(lab)
}

fn site_pos(f: &Frame, basis: &[Coord3], (n, k): Key) -> Coord3 {
    let b = basis[k as usize];
    let u = [n[0] as f64 + b[0], n[1] as f64 + b[1], n[2] as f64 + b[2]];
    let p = f.t.mul_v(u);
    let d = f.curv.as_ref().map_or([0.0; 3], |c| curv_at(c, u));
    [
        f.o[0] + p[0] + d[0],
        f.o[1] + p[1] + d[1],
        f.o[2] + p[2] + d[2],
    ]
}

/// A cluster to accept: its sphere, its census on its own (`None` when it
/// failed), its defect atoms.
type Cand = (Sphere, Option<RegionOut>, Vec<u32>);

/// What [`accept_partial`] needs to join and grow failed clusters.
struct PartialJoin<'a> {
    sphere_of: &'a (dyn Fn(&[u32], &[Sphere]) -> Sphere + Sync),
    shell_full: &'a (dyn Fn(&Sphere, &RegionOut) -> bool + Sync),
    /// A full shell and no void cut (joins).
    whole: &'a (dyn Fn(&Sphere, &RegionOut) -> bool + Sync),
    too_big: &'a (dyn Fn(&Sphere) -> bool + Sync),
    max_r: f64,
}

/// [`LocalMerge::Partial`](super::LocalMerge::Partial): clusters in order,
/// each counted on its own when it meets no accepted region and as a partial
/// region otherwise, never as a union. A cluster's outcome then depends only
/// on the earlier clusters near it, so the clusters are sorted into levels
/// (one more than the highest earlier cluster within reach) and each level is
/// counted in parallel against the levels before it. The result is the
/// sequential one at any thread count. Returns the accepted regions in
/// cluster order and the defect atoms left to the atom-by-atom count.
fn accept_partial(
    ctx: &RegionCtx,
    cands: Vec<Cand>,
    t_of: &(dyn Fn(&[u32]) -> Mat3 + Sync),
    pj: &PartialJoin,
) -> (Vec<(Sphere, RegionOut, Vec<u32>)>, usize) {
    let cell = ctx.inp.cell;
    let nn = ctx.nn;
    if cands.is_empty() {
        return (Vec::new(), 0);
    }
    let Ok(inv_t) = cell.inverse_transposed() else {
        return (Vec::new(), cands.iter().map(|c| c.2.len()).sum());
    };
    let dist = |a: Coord3, b: Coord3| -> f64 {
        let (d, _) = cell.min_image_with_shift_pre([a[0] - b[0], a[1] - b[1], a[2] - b[2]], &inv_t);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    // Failed clusters (no region alone), joined before the levels as the
    // union rule joins them as it goes: those whose spheres come within JOIN
    // bonds of each other form a group, counted as one region when its
    // sphere fits one and its shell is full; one still failing alone has its
    // sphere grown by up to INFLATE bonds. Each is a function of the input
    // only, so the result stays the sequential one at any thread count; what
    // the partial rule cannot do is join a failed cluster to a region.
    let cands: Vec<Cand> = {
        let failed: Vec<usize> = (0..cands.len()).filter(|&i| cands[i].1.is_none()).collect();
        let mut root: Vec<usize> = (0..failed.len()).collect();
        fn find(r: &mut [usize], mut x: usize) -> usize {
            while r[x] != x {
                r[x] = r[r[x]];
                x = r[x];
            }
            x
        }
        for a in 0..failed.len() {
            for b in a + 1..failed.len() {
                let (sa, sb) = (&cands[failed[a]].0, &cands[failed[b]].0);
                if dist(sa.c, sb.c) < sa.r + sb.r + JOIN * nn {
                    let (x, y) = (find(&mut root, a), find(&mut root, b));
                    if x != y {
                        root[x.max(y)] = x.min(y);
                    }
                }
            }
        }
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut at: HashMap<usize, usize> = HashMap::default();
        for (a, &i) in failed.iter().enumerate() {
            let r = find(&mut root, a);
            let g = *at.entry(r).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i);
        }
        // Each group's census: joined, else (one cluster) grown.
        let tried: Vec<Option<(Sphere, RegionOut, Vec<u32>)>> = pmap(ctx.par, &groups, 1, |g| {
            if g.len() > 1 {
                let m: Vec<u32> = g.iter().flat_map(|&i| cands[i].2.iter().copied()).collect();
                let parts: Vec<Sphere> = g.iter().map(|&i| cands[i].0.clone()).collect();
                let sp = (pj.sphere_of)(&m, &parts);
                if !(pj.too_big)(&sp) {
                    if let Ok(r) = region(ctx, &sp, t_of(&m), None) {
                        if (pj.whole)(&sp, &r) {
                            return Some((sp, r, m));
                        }
                    }
                }
                return None;
            }
            let (sp0, _, m) = &cands[g[0]];
            for k in 1..=INFLATE {
                let sp = Sphere::one(sp0.c, sp0.r + k as f64 * nn);
                if sp.r > pj.max_r {
                    break;
                }
                if let Ok(r) = region(ctx, &sp, t_of(m), None) {
                    if (pj.shell_full)(&sp, &r) {
                        return Some((sp, r, m.clone()));
                    }
                    break;
                }
            }
            None
        });
        let mut cands: Vec<Option<Cand>> = cands.into_iter().map(Some).collect();
        for (g, t) in groups.iter().zip(tried) {
            if let Some((sp, r, m)) = t {
                // In the place of the group's first cluster.
                for &i in &g[1..] {
                    cands[i] = None;
                }
                cands[g[0]] = Some((sp, Some(r), m));
            }
        }
        cands.into_iter().flatten().collect()
    };
    let n = cands.len();
    // Reach: every atom a cluster's census reads or counts lies within it
    // (its shell reaches SHELL_OUT bonds past its radius), with a margin for
    // atoms resolved onto sites away from them. Two clusters interact only
    // when their reaches overlap.
    let reach: Vec<f64> = cands
        .iter()
        .map(|c| c.0.r + (SHELL_OUT + 2.0) * nn)
        .collect();
    let bin = 2.0 * reach.iter().copied().fold(0.0, f64::max);
    let nbins = frac_bins(cell, bin);
    let key = |c: Coord3| -> [i64; 3] { frac_key(cell, &inv_t, nbins, c) };
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::default();
    let mut level = vec![0usize; n];
    for i in 0..n {
        let k0 = key(cands[i].0.c);
        let mut l = 0usize;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let mut k = [k0[0] + dx, k0[1] + dy, k0[2] + dz];
                    for a in 0..3 {
                        if cell.pbc[a] {
                            k[a] = k[a].rem_euclid(nbins[a]);
                        }
                    }
                    for &j in grid.get(&k).map_or(&[][..], |v| &v[..]) {
                        if dist(cands[i].0.c, cands[j].0.c) < reach[i] + reach[j] {
                            l = l.max(level[j] + 1);
                        }
                    }
                }
            }
        }
        level[i] = l;
        grid.entry(k0).or_default().push(i);
    }
    let levels = level.iter().copied().max().unwrap_or(0) + 1;
    let mut by_level: Vec<Vec<usize>> = vec![Vec::new(); levels];
    for (i, &l) in level.iter().enumerate() {
        by_level[l].push(i);
    }

    enum Outcome {
        Whole,
        Partial(Box<RegionOut>),
        Extended,
    }
    let mut cands: Vec<Option<Cand>> = cands.into_iter().map(Some).collect();
    let mut result: Vec<Option<(Sphere, RegionOut, Vec<u32>)>> = (0..n).map(|_| None).collect();
    let mut owner: HashMap<u32, usize> = HashMap::default();
    let mut spheres: Vec<(Coord3, f64)> = Vec::new();
    let mut extended = 0usize;
    for items in by_level {
        let outcomes: Vec<Outcome> = {
            let (owner, spheres, cands) = (&owner, &spheres, &cands);
            pmap(ctx.par, &items, 1, |&i| {
                let (sp, r, members) = cands[i].as_ref().unwrap();
                let Some(r) = r else {
                    return Outcome::Extended;
                };
                if !r.atoms.iter().any(|j| owner.contains_key(j)) {
                    return Outcome::Whole;
                }
                let excl = Exclusion {
                    owned: owner,
                    spheres,
                };
                match region(ctx, sp, t_of(members), Some(&excl)) {
                    Ok(pr) => Outcome::Partial(Box::new(pr)),
                    Err(_) => Outcome::Extended,
                }
            })
        };
        // Commit in cluster order; clusters of one level do not interact.
        for (&i, o) in items.iter().zip(outcomes) {
            let (sp, r, members) = cands[i].take().unwrap();
            let r = match o {
                Outcome::Whole => r.expect("a whole region has its census"),
                Outcome::Partial(pr) => *pr,
                Outcome::Extended => {
                    extended += members.len();
                    continue;
                }
            };
            for &j in &r.atoms {
                owner.insert(j, i);
            }
            if sp.balls.is_empty() {
                spheres.push((sp.c, sp.r + SHELL_IN * nn));
            } else {
                for (c, r) in sp.balls_abs() {
                    spheres.push((cell.wrap(c), r + SHELL_IN * nn));
                }
            }
            result[i] = Some((sp, r, members));
        }
    }
    (result.into_iter().flatten().collect(), extended)
}

/// What a partial region leaves to the regions already accepted: their
/// atoms, and the sites within their counted spheres (centre, radius).
struct Exclusion<'a> {
    owned: &'a HashMap<u32, usize>,
    spheres: &'a [(Coord3, f64)],
}

/// A region's Wigner–Seitz census, `Err` when its shell is too sparse or too
/// incoherent for one lattice. With `excl`, a **partial region**: it counts
/// only its sites outside every excluded sphere that hold no excluded atom,
/// and only atoms not excluded, so it never counts a site or an atom twice;
/// the excluded sites stand as crystal for its domain rule and cavity walk.
fn region(
    cx: &RegionCtx,
    sp: &Sphere,
    t: Mat3,
    excl: Option<&Exclusion>,
) -> std::result::Result<RegionOut, &'static str> {
    let inp = cx.inp;
    let nn = cx.nn;
    let r_in = sp.r + SHELL_IN * nn;
    let r_out = sp.r + SHELL_OUT * nn;
    let mut nb: Vec<Nbr> = Vec::new();
    if sp.balls.is_empty() {
        inp.index.within_point(sp.c, r_out, &mut nb);
    } else {
        // A merged region may reach past half the box from its centre (it
        // is checked not to meet its own image): its atoms are gathered
        // ball by ball, each relative to its ball and so to `c`. An atom
        // two balls place at different images means the region wraps.
        let mut at: HashMap<u32, Coord3> = HashMap::default();
        let mut part: Vec<Nbr> = Vec::new();
        for &(b, rb) in sp.balls.iter() {
            let cb = inp
                .cell
                .wrap([sp.c[0] + b[0], sp.c[1] + b[1], sp.c[2] + b[2]]);
            inp.index.within_point(cb, rb + SHELL_OUT * nn, &mut part);
            for &(j, e, _) in &part {
                let p = [b[0] + e[0], b[1] + e[1], b[2] + e[2]];
                match at.get(&j) {
                    None => {
                        at.insert(j, p);
                        nb.push((j, p, p[0] * p[0] + p[1] * p[1] + p[2] * p[2]));
                    }
                    Some(q) => {
                        if (0..3).any(|a| (q[a] - p[a]).abs() > 1e-6) {
                            return Err("wraps the box");
                        }
                    }
                }
            }
        }
        nb.sort_unstable_by_key(|e| e.0);
    }
    let in_grain = |j: usize| inp.grain_of[j] == inp.grain;
    let (si, so) = (SHELL_IN * nn, SHELL_OUT * nn);
    let pts: Vec<(u32, Coord3)> = nb
        .iter()
        .filter(|e| sp.inside(e.1, e.2, so) && in_grain(e.0 as usize))
        .map(|e| (e.0, e.1))
        .collect();
    let in_band = |p: Coord3, d2: f64| !sp.inside(p, d2, si) && sp.inside(p, d2, so);
    let frame: Vec<Coord3> = nb
        .iter()
        .filter(|e| in_band(e.1, e.2) && in_grain(e.0 as usize) && (inp.lattice_atom)(e.0 as usize))
        .map(|e| e.1)
        .collect();
    let band_atoms = if sp.balls.is_empty() {
        0
    } else {
        nb.iter()
            .filter(|e| in_band(e.1, e.2) && in_grain(e.0 as usize))
            .count()
    };
    if !sp.balls.is_empty() {
        // Before the census: a merged region whose shell is not full of
        // grain atoms (it reaches into vacuum past a free surface, or into
        // another grain) is no region. Its shell's volume, from a grid over
        // its bounding box, against the atoms in it.
        let step = 2.0 * nn;
        let lo: Coord3 = std::array::from_fn(|a| {
            sp.balls
                .iter()
                .map(|(b, r)| b[a] - r - so)
                .fold(f64::INFINITY, f64::min)
        });
        let hi: Coord3 = std::array::from_fn(|a| {
            sp.balls
                .iter()
                .map(|(b, r)| b[a] + r + so)
                .fold(f64::NEG_INFINITY, f64::max)
        });
        let n: [usize; 3] =
            std::array::from_fn(|a| ((hi[a] - lo[a]) / step).ceil().max(1.0) as usize);
        let mut in_b = 0usize;
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let p = [
                        lo[0] + (i as f64 + 0.5) * step,
                        lo[1] + (j as f64 + 0.5) * step,
                        lo[2] + (k as f64 + 0.5) * step,
                    ];
                    if in_band(p, p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) {
                        in_b += 1;
                    }
                }
            }
        }
        let omega = t.det().abs() / cx.basis.len() as f64;
        let want = in_b as f64 * step.powi(3) / omega;
        if (band_atoms as f64) < SHELL_FULL * want {
            return Err("shell not full");
        }
    }
    if frame.len() < 3 * REGION_MIN {
        return Err("sparse shell");
    }
    let mut f = fit_frame(t, cx.basis, &frame, cx.inlier2, cx.par).ok_or("no fit")?;
    // Under SHELL_COHERENT the shell is strained past one affine lattice (by
    // defects nearby): its atoms are labelled by walking its bonds, and a
    // smooth (quadratic) lattice is fitted to the labels; the region is then
    // counted on that, exactly as on an affine one, if it puts the shell on
    // its sites.
    if (f.n_in as f64) < SHELL_COHERENT * frame.len() as f64 {
        if (f.n_in as f64) < SHELL_WALK_MIN * frame.len() as f64 {
            return Err("incoherent shell");
        }
        let shell_of = |e: Coord3| !sp.inside(e, e[0] * e[0] + e[1] * e[1] + e[2] * e[2], si);
        let labels = walk_labels(cx, &f, &pts, &shell_of)?;
        f.curv = Some(fit_curvature(&f, cx.basis, &pts, &labels).ok_or("no fit")?);
        let tol2 = cx.inlier2;
        let on = pts
            .iter()
            .zip(&labels)
            .filter(|(&(_, e), l)| {
                l.is_some_and(|l| {
                    let (k, p) = site_key(&f, cx.basis, e);
                    let d2 = (p[0] - e[0]).powi(2) + (p[1] - e[1]).powi(2) + (p[2] - e[2]).powi(2);
                    k == l && d2 <= tol2
                })
            })
            .count();
        if (on as f64) < SHELL_COHERENT * frame.len() as f64 {
            return Err("incoherent shell");
        }
    }
    // The region's sites: the lattice's sites within r_in, in a box of
    // lattice cells with a margin for their first shells.
    // A merged region's box is its spheres' bounding box, shell included
    // (it counts its shell's sites): an elongated region would fill little
    // of a cube round `c`.
    let (blo, bhi): (Coord3, Coord3) = if sp.balls.is_empty() {
        ([-r_in; 3], [r_in; 3])
    } else {
        let so = SHELL_OUT * nn;
        (
            std::array::from_fn(|a| {
                sp.balls
                    .iter()
                    .map(|(b, r)| b[a] - r - so)
                    .fold(f64::INFINITY, f64::min)
            }),
            std::array::from_fn(|a| {
                sp.balls
                    .iter()
                    .map(|(b, r)| b[a] + r + so)
                    .fold(f64::NEG_INFINITY, f64::max)
            }),
        )
    };
    let mut lo = [i32::MAX; 3];
    let mut hi = [i32::MIN; 3];
    for corner in 0..8 {
        let c = [
            if corner & 1 == 0 { blo[0] } else { bhi[0] },
            if corner & 2 == 0 { blo[1] } else { bhi[1] },
            if corner & 4 == 0 { blo[2] } else { bhi[2] },
        ];
        let u = f.ti.mul_v([c[0] - f.o[0], c[1] - f.o[1], c[2] - f.o[2]]);
        for a in 0..3 {
            lo[a] = lo[a].min(u[a].floor() as i32 - 1);
            hi[a] = hi[a].max(u[a].ceil() as i32 + 1);
        }
    }
    let nbasis = cx.basis.len();
    let dim: [usize; 3] = std::array::from_fn(|a| (hi[a] - lo[a] + 3) as usize);
    let cells = dim[0] * dim[1] * dim[2] * nbasis;
    // Site key -> slot in the box, if inside it.
    let slot = |(n, k): Key| -> Option<usize> {
        let mut i = 0usize;
        for a in 0..3 {
            let v = n[a] - lo[a] + 1;
            if v < 0 || v as usize >= dim[a] {
                return None;
            }
            i = i * dim[a] + v as usize;
        }
        Some(i * nbasis + k as usize)
    };
    let dist = |p: Coord3, e: Coord3| {
        ((p[0] - e[0]).powi(2) + (p[1] - e[1]).powi(2) + (p[2] - e[2]).powi(2)).sqrt()
    };
    // Every atom onto its nearest site, unless that is ambiguous: then the
    // site its neighbours' sites and the bonds to them agree on, resolved
    // outward from the unambiguous atoms, so the sites follow the region's
    // displacement field. Where the fronts from either side of a loop meet,
    // its extra atoms share sites.
    let nearest: Vec<(Key, f64)> = pmap(cx.par, &pts, PAR_MIN, |&(_, e)| {
        let (key, p) = site_key(&f, cx.basis, e);
        (key, dist(p, e))
    });
    let amb_lim = AMBIGUOUS * nn;
    let mut keys: Vec<Key> = nearest.iter().map(|x| x.0).collect();
    let amb: Vec<usize> = (0..pts.len()).filter(|&q| nearest[q].1 > amb_lim).collect();
    if !amb.is_empty() {
        let reach2 = (CONTINUE * nn).powi(2);
        let bonds: Vec<Vec<(u32, Coord3)>> = pmap(cx.par, &amb, 64, |&q| {
            let (j, e) = pts[q];
            let mut sc = KnnScratch::default();
            let mut nb = Vec::new();
            inp.index.knn(
                j as usize,
                inp.positions[j as usize],
                K_LINK,
                &mut sc,
                &mut nb,
            );
            nb.iter()
                .filter(|x| x.2 < reach2 && in_grain(x.0 as usize))
                .map(|x| (x.0, [e[0] + x.1[0], e[1] + x.1[1], e[2] + x.1[2]]))
                .collect()
        });
        // Ambiguous atoms resolved in earlier rounds.
        let mut resolved: HashMap<u32, Key> = HashMap::default();
        let mut pending: Vec<usize> = (0..amb.len()).collect();
        loop {
            let votes = pmap(cx.par, &pending, 16, |&i| -> Option<Key> {
                let e = pts[amb[i]].1;
                let mut tally: Vec<(Key, u32, f64)> = Vec::new();
                for &(j, e_n) in &bonds[i] {
                    let (k_n, p_n) = site_key(&f, cx.basis, e_n);
                    let from = if dist(p_n, e_n) <= amb_lim {
                        k_n
                    } else if let Some(&k) = resolved.get(&j) {
                        k
                    } else {
                        continue;
                    };
                    let (k, d) = continue_key(&f, cx.basis, from, e_n, e);
                    match tally.iter_mut().find(|t| t.0 == k) {
                        Some(t) => {
                            t.1 += 1;
                            t.2 += d;
                        }
                        None => tally.push((k, 1, d)),
                    }
                }
                // Most votes; then the closer bonds; then the lower key.
                tally
                    .into_iter()
                    .min_by(|a, b| b.1.cmp(&a.1).then(a.2.total_cmp(&b.2)).then(a.0.cmp(&b.0)))
                    .map(|t| t.0)
            });
            let mut left = Vec::new();
            let mut decided = Vec::new();
            for (&i, v) in pending.iter().zip(votes) {
                match v {
                    Some(k) => decided.push((i, k)),
                    None => left.push(i),
                }
            }
            if decided.is_empty() {
                break;
            }
            for (i, k) in decided {
                resolved.insert(pts[amb[i]].0, k);
                keys[amb[i]] = k;
            }
            if left.is_empty() {
                break;
            }
            pending = left;
        }
    }
    // Atoms outside the box neighbour no site of the region. An excluded
    // atom is another region's; its site here is not this region's either.
    // `wall`: not this region's site (another region's atom holds it, or it
    // lies in another region's sphere). `held_by_other`: held by another
    // region's atom, so crystal for the domain rule. A site that is merely in
    // another sphere may be vacuum past a free surface, and must not prop up
    // the fill.
    let mut wall = vec![false; cells];
    let mut held_by_other = vec![false; cells];
    let keyed: Vec<Option<(u32, u32, f32)>> = (0..pts.len())
        .map(|q| {
            let (j, e) = pts[q];
            let d = dist(site_pos(&f, cx.basis, keys[q]), e);
            let s = slot(keys[q])?;
            if excl.is_some_and(|x| x.owned.contains_key(&j)) {
                wall[s] = true;
                held_by_other[s] = true;
                return None;
            }
            Some((s as u32, j, d as f32))
        })
        .collect();
    // Occupants by slot: `occ[start[s]..start[s + 1]]`.
    let mut start = vec![0u32; cells + 1];
    for &(s, _, _) in keyed.iter().flatten() {
        start[s as usize + 1] += 1;
    }
    for s in 0..cells {
        start[s + 1] += start[s];
    }
    let mut occ = vec![(0u32, 0f32); start[cells] as usize];
    {
        let mut fill = start.clone();
        for &(s, j, d) in keyed.iter().flatten() {
            occ[fill[s as usize] as usize] = (j, d);
            fill[s as usize] += 1;
        }
    }
    let held = |s: usize| start[s + 1] > start[s];
    let xs: Vec<i32> = (lo[0]..=hi[0]).collect();
    let merged = !sp.balls.is_empty();
    let per_x: Vec<(Vec<(usize, Coord3)>, usize)> = pmap(cx.par, &xs, 1, |&x| {
        let mut v = Vec::new();
        let mut band = 0usize;
        for y in lo[1]..=hi[1] {
            for z in lo[2]..=hi[2] {
                for k in 0..nbasis as u8 {
                    let key = ([x, y, z], k);
                    let p = site_pos(&f, cx.basis, key);
                    let d2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
                    let inside = if merged {
                        sp.inside(p, d2, si)
                    } else {
                        d2 < r_in * r_in
                    };
                    if inside {
                        v.push((slot(key).expect("site in its box"), p));
                    } else if merged && sp.inside(p, d2, so) {
                        band += 1;
                    }
                }
            }
        }
        (v, band)
    });
    let band_sites: usize = per_x.iter().map(|x| x.1).sum();
    let sites: Vec<(usize, Coord3)> = per_x.into_iter().flat_map(|x| x.0).collect();
    let abs = |p: Coord3| [sp.c[0] + p[0], sp.c[1] + p[1], sp.c[2] + p[2]];
    // A partial region's sites inside an excluded sphere are another
    // region's: walls, like the sites excluded atoms hold.
    let mut sites = sites;
    if let Some(x) = excl {
        let inv_t = inp.cell.inverse_transposed().map_err(|_| "singular cell")?;
        let near: Vec<(Coord3, f64)> = x
            .spheres
            .iter()
            .copied()
            .filter(|&(c, r)| {
                let (d, _) = inp.cell.min_image_with_shift_pre(
                    [c[0] - sp.c[0], c[1] - sp.c[1], c[2] - sp.c[2]],
                    &inv_t,
                );
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < r + r_in
            })
            .collect();
        for &(s, p) in &sites {
            let a = abs(p);
            let inside = near.iter().any(|&(c, r)| {
                let (d, _) = inp
                    .cell
                    .min_image_with_shift_pre([a[0] - c[0], a[1] - c[1], a[2] - c[2]], &inv_t);
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < r * r
            });
            if inside {
                wall[s] = true;
            }
        }
        sites.retain(|&(s, _)| !wall[s]);
        if sites.is_empty() {
            return Err("nothing left");
        }
    }
    let solid = |s: usize| held(s) || held_by_other[s];
    // Empty sites join by the domain rule, inside the region. A site's first
    // shell is a fixed slot offset per basis site, inside the box's margin.
    let stride = [dim[1] * dim[2] * nbasis, dim[2] * nbasis, nbasis];
    let shell_off: Vec<Vec<isize>> = (0..nbasis)
        .map(|k| {
            cx.shell[k]
                .iter()
                .map(|&(fr, l)| {
                    let b0 = cx.basis[k];
                    let b1 = cx.basis[l as usize];
                    let mut o = l as isize - k as isize;
                    for a in 0..3 {
                        let dn = (fr[a] - b1[a] + b0[a]).round() as isize;
                        o += dn * stride[a] as isize;
                    }
                    o
                })
                .collect()
        })
        .collect();
    let neighbours = |s: usize| {
        shell_off[s % nbasis]
            .iter()
            .map(move |&o| (s as isize + o) as usize)
    };
    // Per slot: 0 not an empty site of the region, 1 empty, 2 accepted.
    let mut state = vec![0u8; cells];
    for &(s, _) in &sites {
        if !held(s) {
            state[s] = 1;
        }
    }
    // Each empty site's count of held neighbours (occupied, or accepted in
    // an earlier round); a round rechecks only the sites next to the last
    // round's.
    let mut count = vec![0u8; cells];
    let counted = pmap(cx.par, &sites, PAR_MIN, |&(s, _)| {
        (state[s] == 1).then(|| (s, neighbours(s).filter(|&n| solid(n)).count() as u8))
    });
    for &(s, n) in counted.iter().flatten() {
        count[s] = n;
    }
    let mut pos = vec![[0.0; 3]; 0];
    let mut pos_of: HashMap<usize, usize> = HashMap::default();
    for &(s, p) in &sites {
        if state[s] == 1 {
            pos_of.insert(s, pos.len());
            pos.push(p);
        }
    }
    let mut check: Vec<usize> = counted.iter().flatten().map(|e| e.0).collect();
    // Whether another grain claims a site, asked only of sites the domain
    // rule would take: a query in the vacuum past a free surface is slow.
    // 0 unknown, 1 free, 2 claimed.
    let mut claimed = vec![0u8; cells];
    for _ in 0..inp.max_domain_iterations.max(1) {
        let mut new: Vec<usize> = Vec::new();
        for &s in &check {
            if state[s] != 1 || (count[s] as usize) < cx.need {
                continue;
            }
            if claimed[s] == 0 {
                let p = pos[pos_of[&s]];
                claimed[s] = if inp.env.zone(abs(p), None).0 { 2 } else { 1 };
            }
            if claimed[s] == 1 {
                new.push(s);
            }
        }
        if new.is_empty() {
            break;
        }
        for &s in &new {
            state[s] = 2;
        }
        check.clear();
        for s in new {
            for n in neighbours(s) {
                if state[n] == 1 {
                    count[n] += 1;
                    check.push(n);
                }
            }
        }
        check.sort_unstable();
        check.dedup();
    }
    // Cavities inside the region, as in the census: a connected set of empty
    // region sites that does not leave the region and has no site claimed by
    // or beside another grain is a void.
    {
        // 0 unseen, 1 open, 2 being walked.
        let mut mark = vec![0u8; cells];
        for &(s0, _) in &sites {
            if state[s0] != 1 || mark[s0] != 0 {
                continue;
            }
            let mut comp = vec![s0];
            mark[s0] = 2;
            let mut open = false;
            let mut q = 0;
            while q < comp.len() && !open {
                let si = comp[q];
                q += 1;
                for n in neighbours(si) {
                    if solid(n) || state[n] == 2 {
                        continue;
                    }
                    if state[n] == 0 || mark[n] == 1 {
                        open = true;
                        break;
                    }
                    if mark[n] == 0 {
                        mark[n] = 2;
                        comp.push(n);
                    }
                }
            }
            if !open {
                open = comp.iter().any(|&si| {
                    let z = inp.env.zone(abs(pos[pos_of[&si]]), None);
                    z.0 || z.1
                });
            }
            for &si in &comp {
                if open {
                    mark[si] = 1;
                } else {
                    state[si] = 2;
                }
            }
        }
    }
    let open_sites = sites.iter().filter(|&&(s, _)| state[s] == 1).count();
    let mut out = RegionOut {
        atoms: Vec::new(),
        interior: ClassCount::default(),
        boundary: ClassCount::default(),
        vacancies: Vec::new(),
        interstitials: Vec::new(),
        displacement: Vec::new(),
        singles: Vec::new(),
        open_sites,
        band_sites,
        band_atoms,
    };
    let mut by = Vec::new();
    for &(s, p) in &sites {
        if held(s) {
            let v = &occ[start[s] as usize..start[s + 1] as usize];
            let b = v.iter().any(|&(j, _)| inp.boundary[j as usize]);
            let c = if b {
                &mut out.boundary
            } else {
                &mut out.interior
            };
            c.sites += 1;
            c.atoms += v.len() as u64;
            c.interstitials += v.len() as u64 - 1;
            if let [(j, _)] = v {
                out.singles.push((*j, abs(p)));
            }
            by.clear();
            by.extend_from_slice(v);
            by.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            for (n, &(j, d)) in by.iter().enumerate() {
                out.atoms.push(j);
                if !(inp.lattice_atom)(j as usize) {
                    out.displacement.push((j, d));
                }
                if n > 0 {
                    out.interstitials.push((j, b));
                }
            }
        } else if state[s] == 2 {
            let a = abs(p);
            let b = inp.env.zone(a, None).1;
            let c = if b {
                &mut out.boundary
            } else {
                &mut out.interior
            };
            c.sites += 1;
            c.vacancies += 1;
            out.vacancies.push((inp.cell.wrap(a), b));
        }
    }
    out.atoms.sort_unstable();
    Ok(out)
}
