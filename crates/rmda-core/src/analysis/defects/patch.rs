//! Patch-local reference (design §16.3c).
//!
//! One affine lattice cannot follow a grain that bends, twists or is strained
//! inhomogeneously: a 1° rotation gradient over 100 nm already moves atoms
//! many Å from any single lattice. A patch reference tiles the grain into
//! cubic voxels and fits each its own `T_p`, `o_p`.
//!
//! **One index space.** All patches label sites with the *global* lattice's
//! integer indices; a patch is only a local map from index to position. So a
//! physical site has one key whichever patch looks at it, and patch seams can
//! neither create nor lose sites, which a "each patch owns the sites in its
//! voxel" rule cannot guarantee (any phase mismatch at a face puts sites in
//! both voxels or in neither). Periodic wrapping stays index arithmetic on the
//! global lattice, which every patch shares the box commensurability of.
//!
//! **Phase continuity.** Patches are visited breadth-first from the best-fitted
//! one. Each new patch fits `T_p` on its own neighbour vectors, then takes the
//! origin that makes its labels agree with its already-placed neighbour on the
//! atoms nearest to that neighbour. The fraction of those atoms whose label
//! the two patches agree on is reported as the phase agreement.
//!
//! Patches with too few crystalline atoms (vacuum, cavities, a boundary)
//! inherit their neighbour's lattice.

use super::fit;
use super::lattice::{AffineLattice, Reference, Site};
use crate::cell::{Coord3, UnitCell};
use crate::exec::Par;

/// Patches with fewer fitted atoms inherit their parent's lattice.
const MIN_PATCH_ATOMS: usize = 60;

/// A patch's `T` is fitted on at most this many of its atoms' first shells.
const MAX_PATCH_FIT_ATOMS: usize = 400;

/// First-shell neighbour vectors of a grain atom (by local index), computed
/// on demand.
pub trait NeighbourSource: Sync {
    /// Appends up to [`super::fit::fit_neighbours`] neighbour vectors of
    /// grain atom `local` and returns how many first-shell pairs it should
    /// have (its same-grain atoms among its `z` nearest).
    fn vectors(&self, local: usize, out: &mut Vec<Coord3>) -> usize;
}

/// A patch-local reference.
pub struct PatchLattice {
    global: AffineLattice,
    lo: Coord3,
    size: f64,
    dims: [usize; 3],
    patches: Vec<AffineLattice>,
    /// Per patch: whether it was fitted (rather than inherited).
    pub fitted: Vec<bool>,
    /// Lowest per-patch phase agreement over fitted, non-seed patches.
    pub min_phase_agreement: f64,
    /// Lowest agreement over every pair of face-adjacent fitted patches, not
    /// only the parent-child pairs placement checks: round a dislocation the
    /// tree's two sides meet with labels a Burgers vector apart.
    pub min_face_agreement: f64,
}

impl PatchLattice {
    #[inline]
    fn voxel(&self, x: Coord3) -> usize {
        let mut v = [0usize; 3];
        for a in 0..3 {
            let f = ((x[a] - self.lo[a]) / self.size).floor();
            v[a] = (f.max(0.0) as usize).min(self.dims[a] - 1);
        }
        (v[0] * self.dims[1] + v[1]) * self.dims[2] + v[2]
    }

    pub fn n_patches(&self) -> usize {
        self.patches.len()
    }

    pub fn n_fitted(&self) -> usize {
        self.fitted.iter().filter(|&&f| f).count()
    }
}

impl Reference for PatchLattice {
    fn global(&self) -> &AffineLattice {
        &self.global
    }

    fn nearest(&self, x: Coord3) -> (Site, Coord3, f64) {
        self.patches[self.voxel(x)].nearest(x)
    }

    fn position(&self, s: Site) -> Coord3 {
        // The global position may be off by a voxel in a bent grain; two
        // steps through the local lattices settle on the right patch.
        let mut p = self.global.position(s);
        for _ in 0..2 {
            p = self.patches[self.voxel(p)].position(s);
        }
        p
    }
}

/// Build a patch reference.
///
/// `positions` are the grain's (unwrapped) positions; `fit_atoms` index the
/// crystalline ones, whose neighbour vectors `source` provides. `global` is
/// the grain's fitted, snapped lattice and `translations` the lattice
/// translations of the box vectors it was snapped to; every patch snaps to
/// the same ones, so their labels share one index space.
#[allow(clippy::too_many_arguments)]
pub fn build(
    global: &AffineLattice,
    translations: [Option<Coord3>; 3],
    cell: &UnitCell,
    positions: &[Coord3],
    fit_atoms: &[u32],
    source: &dyn NeighbourSource,
    size: f64,
    par: &Par,
) -> PatchLattice {
    let m = &global.motif;
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in positions {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let mut dims = [1usize; 3];
    for a in 0..3 {
        dims[a] = (((hi[a] - lo[a]) / size).floor() as usize + 1).max(1);
    }
    let nvox = dims[0] * dims[1] * dims[2];
    let mut lat = PatchLattice {
        global: global.clone(),
        lo,
        size,
        dims,
        patches: vec![global.clone(); nvox],
        fitted: vec![false; nvox],
        min_phase_agreement: 1.0,
        min_face_agreement: 1.0,
    };

    // Atoms per voxel (all grain atoms, for phases) and fit atoms per voxel.
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); nvox];
    for (i, p) in positions.iter().enumerate() {
        members[lat.voxel(*p)].push(i as u32);
    }
    let mut fit_of: Vec<Vec<u32>> = vec![Vec::new(); nvox];
    for &i in fit_atoms {
        fit_of[lat.voxel(positions[i as usize])].push(i);
    }

    // Local T per voxel, from its own neighbour vectors. One voxel per task:
    // `Par::fill` hands out 65 536-item chunks, which would put every patch of a
    // grain (the thin film's 2268) on one thread.
    let serial = Par::new(1);
    let mut local_t: Vec<Option<crate::cell::Mat3>> = vec![None; nvox];
    let fit_voxel = |v: usize, o: &mut Option<crate::cell::Mat3>| {
        if fit_of[v].len() < MIN_PATCH_ATOMS {
            return;
        }
        let sample = super::place_sample(
            cell,
            &fit_of[v],
            |i| positions[i as usize],
            MAX_PATCH_FIT_ATOMS,
        );
        let mut pv: Vec<Coord3> = Vec::new();
        for &i in &sample {
            let _ = source.vectors(i as usize, &mut pv);
        }
        if let Some((t, _)) = fit::refine(m, global.t, &pv, 3, &serial) {
            let snapped = translations.map(|n| n.is_some());
            *o = Some(fit::snap_to_box(m, t, cell, snapped, f64::INFINITY, translations).0);
        }
    };
    if par.is_parallel() {
        use rayon::prelude::*;
        par.install(|| {
            local_t
                .par_iter_mut()
                .enumerate()
                .with_min_len(1)
                .for_each(|(v, o)| fit_voxel(v, o))
        });
    } else {
        for (v, o) in local_t.iter_mut().enumerate() {
            fit_voxel(v, o);
        }
    }

    // Breadth-first placement from the patch with the most fitted atoms.
    let Some(seed) = (0..nvox)
        .filter(|&v| local_t[v].is_some())
        .max_by(|&a, &b| fit_of[a].len().cmp(&fit_of[b].len()).then(b.cmp(&a)))
    else {
        return lat;
    };
    let mut done = vec![false; nvox];
    let mut queue = std::collections::VecDeque::new();
    // The seed keeps the global lattice's labels: same T as its local fit,
    // origin chosen to agree with the global lattice on its own atoms.
    place(
        &mut lat,
        seed,
        None,
        local_t[seed].unwrap(),
        &members[seed],
        positions,
    );
    lat.fitted[seed] = true;
    done[seed] = true;
    queue.push_back(seed);
    let mut min_agree = 1.0f64;
    while let Some(p) = queue.pop_front() {
        for q in neighbours(p, dims) {
            if done[q] {
                continue;
            }
            done[q] = true;
            match local_t[q] {
                Some(t) => {
                    let agree = place(&mut lat, q, Some(p), t, &members[q], positions);
                    min_agree = min_agree.min(agree);
                    lat.fitted[q] = true;
                }
                None => lat.patches[q] = lat.patches[p].clone(),
            }
            queue.push_back(q);
        }
    }
    lat.min_phase_agreement = min_agree;
    let wrap = translations.map(|n| n.is_some());
    lat.min_face_agreement = face_agreement(&lat, &members, positions).min(
        periodic_face_agreement(&lat, &members, positions, cell, wrap),
    );
    lat
}

/// The lowest agreement, over every pair of face-adjacent fitted patches, of
/// the two patches' labels on the atoms of both nearest their shared face.
fn face_agreement(lat: &PatchLattice, members: &[Vec<u32>], positions: &[Coord3]) -> f64 {
    let mut lowest = 1.0f64;
    for v in 0..lat.patches.len() {
        if !lat.fitted[v] {
            continue;
        }
        for w in neighbours(v, lat.dims) {
            if w < v || !lat.fitted[w] {
                continue;
            }
            let (cv, cw) = (voxel_centre(lat, v), voxel_centre(lat, w));
            let mid = [
                (cv[0] + cw[0]) / 2.0,
                (cv[1] + cw[1]) / 2.0,
                (cv[2] + cw[2]) / 2.0,
            ];
            // Atoms within a quarter patch of the face, from either side.
            let near = |i: &&u32| {
                let p = positions[**i as usize];
                (0..3)
                    .map(|a| (p[a] - mid[a]) * (cw[a] - cv[a]) / lat.size)
                    .sum::<f64>()
                    .abs()
                    < 0.25 * lat.size
            };
            let atoms: Vec<u32> = members[v]
                .iter()
                .chain(&members[w])
                .filter(near)
                .copied()
                .collect();
            if atoms.len() < MIN_PATCH_ATOMS / 2 {
                continue;
            }
            let same = atoms
                .iter()
                .filter(|&&i| {
                    let x = positions[i as usize];
                    lat.patches[v].nearest(x).0 == lat.patches[w].nearest(x).0
                })
                .count();
            lowest = lowest.min(same as f64 / atoms.len() as f64);
        }
    }
    lowest
}

/// The lowest agreement across the periodic box faces the lattice tiles
/// (`wrap`), which [`face_agreement`] and placement never compare: the
/// first and last patches along a periodic axis meet only through the box.
/// A grain whose patches drift by a lattice vector across the box (a tilt
/// or shear that varies along another axis, snapped to one translation)
/// carries the drift to that face, where the census would read the two
/// labellings of one plane of sites as a plane of vacancies and one of
/// interstitials. Per axis, each atom within a quarter patch of the box's
/// upper face is labelled by its own patch and, at its image across the
/// face, by the patch there; the two labels are compared as the census
/// keys them (wrapped into the box), per pair of patches.
fn periodic_face_agreement(
    lat: &PatchLattice,
    members: &[Vec<u32>],
    positions: &[Coord3],
    cell: &UnitCell,
    wrap: [bool; 3],
) -> f64 {
    let Ok(inv_t) = cell.inverse_transposed() else {
        return 1.0;
    };
    let heights = cell.heights();
    let mut lowest = 1.0f64;
    for d in (0..3).filter(|&d| wrap[d] && cell.pbc[d]) {
        let a = *cell.h_matrix().row(d);
        let near = 0.25 * lat.size / heights[d];
        // (patch of the atom, patch of its image) -> (atoms, agreeing).
        let mut pairs: std::collections::BTreeMap<(usize, usize), (usize, usize)> =
            Default::default();
        for (v, m) in members.iter().enumerate() {
            if !lat.fitted[v] {
                continue;
            }
            for &i in m {
                let x = positions[i as usize];
                let l = cell.x2lamda_pre(x, &inv_t)[d];
                if l - l.floor() < 1.0 - near {
                    continue;
                }
                let y = [x[0] - a[0], x[1] - a[1], x[2] - a[2]];
                let w = lat.voxel(y);
                if w == v || !lat.fitted[w] {
                    continue;
                }
                let key =
                    |s: Site| super::census::canonical_site(&lat.global, cell, &inv_t, wrap, s);
                let same = key(lat.patches[v].nearest(x).0) == key(lat.patches[w].nearest(y).0);
                let e = pairs.entry((v, w)).or_default();
                e.0 += 1;
                e.1 += usize::from(same);
            }
        }
        for &(n, same) in pairs.values() {
            if n >= MIN_PATCH_ATOMS / 2 {
                lowest = lowest.min(same as f64 / n as f64);
            }
        }
    }
    lowest
}

/// Set patch `q`'s lattice to `t` with the origin that makes its labels
/// agree with `parent`'s (or with the global lattice for the seed) on `q`'s
/// atoms nearest the parent. Returns the fraction of those atoms whose label
/// the two then agree on.
fn place(
    lat: &mut PatchLattice,
    q: usize,
    parent: Option<usize>,
    t: crate::cell::Mat3,
    members: &[u32],
    positions: &[Coord3],
) -> f64 {
    let reference = match parent {
        Some(p) => lat.patches[p].clone(),
        None => lat.global.clone(),
    };
    // Atoms nearest the parent: the parent's lattice is most trustworthy there.
    let mut atoms: Vec<usize> = members.iter().map(|&i| i as usize).collect();
    if let Some(p) = parent {
        let c = voxel_centre(lat, p);
        atoms.sort_by(|&a, &b| {
            dist2(positions[a], c)
                .total_cmp(&dist2(positions[b], c))
                .then(a.cmp(&b))
        });
        atoms.truncate((atoms.len() / 2).max(1));
    }
    let mut new = AffineLattice::new(reference.motif.clone(), t, reference.origin)
        .unwrap_or_else(|| reference.clone());
    if atoms.is_empty() {
        lat.patches[q] = new;
        return 1.0;
    }
    let bond = reference.nn();
    let labels: Vec<Site> = atoms
        .iter()
        .map(|&i| reference.nearest(positions[i]).0)
        .collect();
    // o = mean(x − T·(n + b)), then once more over the atoms that fit.
    let resid = |o: Coord3, keep: Option<f64>| -> Coord3 {
        let mut s = [0.0; 3];
        let mut c = 0usize;
        for (&i, l) in atoms.iter().zip(&labels) {
            let b = new.motif.basis[l.k as usize];
            let f = [
                l.n[0] as f64 + b[0],
                l.n[1] as f64 + b[1],
                l.n[2] as f64 + b[2],
            ];
            let tf = t.mul_v(f);
            let r = [
                positions[i][0] - tf[0] - o[0],
                positions[i][1] - tf[1] - o[1],
                positions[i][2] - tf[2] - o[2],
            ];
            if keep.map_or(true, |k| r.iter().map(|v| v * v).sum::<f64>() < k * k) {
                for a in 0..3 {
                    s[a] += r[a];
                }
                c += 1;
            }
        }
        let c = c.max(1) as f64;
        [s[0] / c, s[1] / c, s[2] / c]
    };
    let r0 = resid([0.0; 3], None);
    let r1 = resid(r0, Some(0.35 * bond));
    new.origin = [r0[0] + r1[0], r0[1] + r1[1], r0[2] + r1[2]];
    let agree = atoms
        .iter()
        .zip(&labels)
        .filter(|(&i, l)| new.nearest(positions[i]).0 == **l)
        .count() as f64
        / atoms.len() as f64;
    lat.patches[q] = new;
    agree
}

fn voxel_centre(lat: &PatchLattice, v: usize) -> Coord3 {
    let z = v % lat.dims[2];
    let y = (v / lat.dims[2]) % lat.dims[1];
    let x = v / (lat.dims[2] * lat.dims[1]);
    [
        lat.lo[0] + (x as f64 + 0.5) * lat.size,
        lat.lo[1] + (y as f64 + 0.5) * lat.size,
        lat.lo[2] + (z as f64 + 0.5) * lat.size,
    ]
}

fn dist2(a: Coord3, b: Coord3) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Face neighbours of voxel `v`, in a fixed order.
fn neighbours(v: usize, dims: [usize; 3]) -> impl Iterator<Item = usize> {
    let z = v % dims[2];
    let y = (v / dims[2]) % dims[1];
    let x = v / (dims[2] * dims[1]);
    let at = move |x: usize, y: usize, z: usize| (x * dims[1] + y) * dims[2] + z;
    let mut out = Vec::with_capacity(6);
    if x > 0 {
        out.push(at(x - 1, y, z));
    }
    if x + 1 < dims[0] {
        out.push(at(x + 1, y, z));
    }
    if y > 0 {
        out.push(at(x, y - 1, z));
    }
    if y + 1 < dims[1] {
        out.push(at(x, y + 1, z));
    }
    if z > 0 {
        out.push(at(x, y, z - 1));
    }
    if z + 1 < dims[2] {
        out.push(at(x, y, z + 1));
    }
    out.into_iter()
}
