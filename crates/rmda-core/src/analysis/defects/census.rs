//! Wigner–Seitz occupancy of one grain against its fitted lattice
//! (design §16.1, §16.2, §16.4).
//!
//! **Assignment.** Every atom of the grain goes to its nearest site (with
//! `neighbours`, an ambiguous one to the site its neighbours point to). There
//! is no tolerance and no atom is held out, so a split dumbbell is two atoms
//! on one site: one interstitial.
//!
//! **Domain.** The sites counted are the occupied ones, plus empty sites
//! enclosed by the domain: an empty site joins when at least half its first
//! shell (plus one) is already in the domain, repeated to a fixed point (or
//! an iteration cap, reported). That fills vacancies and the interior of
//! vacancy clusters, and stops at a free surface: a site above a flat surface
//! sees at most half its shell occupied. An empty site whose position another
//! grain's atom already occupies is never this grain's vacancy.
//!
//! **Conservation.** Because every atom sits on a counted site,
//! `vacancies − interstitials = sites − atoms` holds exactly, per class and in
//! total. It is asserted, not hoped for.
//!
//! **Classes.** An occupied site is `boundary` when an atom on it lies
//! within the boundary width of another grain; an empty site when its own
//! position does (the same rule, asked through [`Surroundings::probe`]; its
//! nearest atom's flag would misclass empty sites on mismatched boundaries).
//! Otherwise `interior`. The two are counted separately and both conserve.

use super::antisite::{
    antisites_defined, chemical_order, majority_share, sublattice_species, GrainAntisites,
};
use super::lattice::{Reference, Site};
use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::{Error, Result};
use crate::exec::Par;
use std::collections::HashMap;

/// A grid larger than this many sites per atom means the grain's lattice box
/// is mostly empty (a wildly misfit lattice); refuse rather than allocate.
const MAX_SITES_PER_ATOM: usize = 16;

/// A connected set of empty sites larger than this is open space (vacuum, or
/// another grain's territory), not a cavity.
pub(super) const CAVITY_MAX: usize = 20_000;

/// Fractional tolerance for a site lying on a periodic face.
const FACE_TOL: f64 = 1e-6;

/// Counts of one class of sites.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassCount {
    pub sites: u64,
    pub atoms: u64,
    pub vacancies: u64,
    pub interstitials: u64,
}

impl ClassCount {
    /// `V − I`, equal to `sites − atoms` in a conserving count: the sites
    /// the class has beyond its atoms. In a closed crystal damaged by
    /// Frenkel pairs it is what the counts do not hold: interstitials in
    /// loops and networks the census cannot see, less vacancies in collapsed
    /// vacancy loops it cannot see, plus atoms moved to a surface or
    /// boundary.
    pub fn net_vacancies(&self) -> i64 {
        self.vacancies as i64 - self.interstitials as i64
    }

    /// `(V − I) − (sites − atoms)`; zero for a conserving count.
    pub fn residual(&self) -> i64 {
        (self.vacancies as i64 - self.interstitials as i64)
            - (self.sites as i64 - self.atoms as i64)
    }
}

/// What the caller knows about a point that the grain alone does not.
pub trait Surroundings: Sync {
    /// `(claimed_by_another_grain, in_boundary_zone)` for an empty site at
    /// Cartesian position `p`.
    fn probe(&self, p: Coord3) -> (bool, bool);
}

/// Inputs of one grain's census.
pub struct CensusInput<'a> {
    pub lattice: &'a dyn Reference,
    pub cell: &'a UnitCell,
    /// Axes on which the lattice is commensurate with the box: site keys are
    /// wrapped there, so the seam is not a plane of duplicate sites.
    pub wrap: [bool; 3],
    /// Grain atoms' positions (unwrapped on non-wrapping axes).
    pub positions: &'a [Coord3],
    /// Per grain atom: within the boundary width of another grain.
    pub boundary: &'a [bool],
    pub max_domain_iterations: usize,
    /// Per grain atom: its type, to count antisites (`None`: not counted).
    pub types: Option<&'a [i32]>,
    /// With [`DefectsConfig::resolve_ambiguous`](super::DefectsConfig):
    /// a grain atom's neighbours (grain atom, bond vector to it), for atoms
    /// whose nearest site is ambiguous. `None`: pure nearest-site
    /// assignment (Wigner–Seitz, OVITO's), the default.
    pub neighbours: Option<&'a Neighbours<'a>>,
}

/// A grain atom's neighbours: `f(atom, out)` fills `out` with (grain atom,
/// bond vector from `atom` to it).
pub type Neighbours<'a> = dyn Fn(usize, &mut Vec<(usize, Coord3)>) + Sync + 'a;

/// An atom farther than this from its nearest site, in nearest-neighbour
/// distances, takes the site its neighbours point to (with `neighbours`).
/// Inside a loop the crystal is displaced by up to b/2 from the lattice
/// fitted around it (0.5-0.58 bonds), where two sites are nearly equally
/// near; the local census's regions use the same rule.
const AMBIGUOUS: f64 = 0.3;

/// Sites of the atoms whose nearest site is ambiguous, from their
/// neighbours' (resolved outward from the unambiguous atoms): each neighbour
/// on a site votes for the site its bond points to from there; most votes
/// win, then the shorter bonds, then the lower key.
fn resolve_ambiguous(
    lat: &dyn Reference,
    positions: &[Coord3],
    neighbours: &Neighbours,
    canonical: &(dyn Fn(Site) -> Site + Sync),
    par: &Par,
) -> HashMap<usize, Site> {
    let g = lat.global();
    let nn = {
        let shell = &g.motif.shell[0];
        shell
            .iter()
            .map(|&(dn, l)| {
                let a = g.position(Site { n: [0; 3], k: 0 });
                let b = g.position(Site { n: dn, k: l });
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let lim2 = (AMBIGUOUS * nn).powi(2);
    let amb: Vec<usize> = par
        .map_chunks(positions.len(), |a, b| {
            (a..b)
                .filter(|&i| lat.nearest(positions[i]).2 > lim2)
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .collect();
    let is_amb: std::collections::HashSet<usize> = amb.iter().copied().collect();
    let bonds: Vec<Vec<(usize, Coord3)>> = par.install(|| {
        use rayon::prelude::*;
        amb.par_iter()
            .map(|&i| {
                let mut v = Vec::new();
                neighbours(i, &mut v);
                v
            })
            .collect()
    });
    let mut resolved: HashMap<usize, Site> = HashMap::new();
    let mut pending: Vec<usize> = (0..amb.len()).collect();
    loop {
        let votes: Vec<Option<Site>> = par.install(|| {
            use rayon::prelude::*;
            pending
                .par_iter()
                .map(|&a| {
                    let mut tally: Vec<(Site, u32, f64)> = Vec::new();
                    for &(j, bond) in &bonds[a] {
                        let from = if !is_amb.contains(&j) {
                            canonical(lat.nearest(positions[j]).0)
                        } else if let Some(&s) = resolved.get(&j) {
                            s
                        } else {
                            continue;
                        };
                        // `i`'s site seen from `j`'s: the bond runs from
                        // `i` to `j` (x_j - x_i), so step back along it.
                        let p = lat.position(from);
                        let (s, _, d2) =
                            lat.nearest([p[0] - bond[0], p[1] - bond[1], p[2] - bond[2]]);
                        let s = canonical(s);
                        match tally.iter_mut().find(|t| t.0 == s) {
                            Some(t) => {
                                t.1 += 1;
                                t.2 += d2.sqrt();
                            }
                            None => tally.push((s, 1, d2.sqrt())),
                        }
                    }
                    tally
                        .into_iter()
                        .min_by(|a, b| {
                            b.1.cmp(&a.1)
                                .then(a.2.total_cmp(&b.2))
                                .then((a.0.n, a.0.k).cmp(&(b.0.n, b.0.k)))
                        })
                        .map(|t| t.0)
                })
                .collect()
        });
        let mut left = Vec::new();
        let mut decided = Vec::new();
        for (&a, v) in pending.iter().zip(votes) {
            match v {
                Some(s) => decided.push((amb[a], s)),
                None => left.push(a),
            }
        }
        if decided.is_empty() {
            break;
        }
        resolved.extend(decided);
        if left.is_empty() {
            break;
        }
        pending = left;
    }
    resolved
}

/// One grain's census.
#[derive(Debug, Clone, Default)]
pub struct Census {
    pub interior: ClassCount,
    pub boundary: ClassCount,
    /// Vacant sites: position and whether in the boundary class.
    pub vacancies: Vec<(Coord3, bool)>,
    /// Interstitial atoms as indices into the grain's atom list, with class.
    pub interstitials: Vec<(u32, bool)>,
    /// Per grain atom: distance to its site, Å.
    pub displacement: Vec<f32>,
    /// RMS of `displacement`.
    pub rms_displacement: f64,
    /// Fraction of atoms more than 0.35 bonds from their site.
    pub offlattice_fraction: f64,
    /// Domain-filling iterations used, and whether it reached a fixed point.
    pub domain_iterations: usize,
    pub domain_converged: bool,
    /// With `types`: the grain's antisites, and each antisite as (grain atom,
    /// site type, site position, boundary).
    pub antisites: Option<(GrainAntisites, Vec<AntisiteAtom>)>,
    /// With `types`: the chemical order (Cramér's V) and the smallest
    /// majority share of a basis site; antisites are `None` when neither
    /// defines them ([`antisites_defined`](super::antisite::antisites_defined)).
    pub antisite_order: Option<(f64, f64)>,
}

/// An antisite of a census: (grain atom, site type, site position, boundary).
pub type AntisiteAtom = (u32, i32, Coord3, bool);

#[derive(Clone, Copy)]
struct Grid<'a> {
    lat: &'a dyn Reference,
    lo: [i32; 3],
    dim: [usize; 3],
    nb: usize,
    wrap: [bool; 3],
    any_wrap: bool,
    cell: &'a UnitCell,
    cell_inv_t: Mat3,
}

impl Grid<'_> {
    fn len(&self) -> usize {
        self.dim[0] * self.dim[1] * self.dim[2] * self.nb
    }

    #[inline]
    fn index(&self, s: Site) -> Option<usize> {
        let mut r = [0usize; 3];
        for (a, slot) in r.iter_mut().enumerate() {
            let v = s.n[a] - self.lo[a];
            if v < 0 || v as usize >= self.dim[a] {
                return None;
            }
            *slot = v as usize;
        }
        Some(((r[0] * self.dim[1] + r[1]) * self.dim[2] + r[2]) * self.nb + s.k as usize)
    }

    #[inline]
    fn site(&self, idx: usize) -> Site {
        let k = idx % self.nb;
        let c = idx / self.nb;
        let z = c % self.dim[2];
        let y = (c / self.dim[2]) % self.dim[1];
        let x = c / (self.dim[2] * self.dim[1]);
        Site {
            n: [
                x as i32 + self.lo[0],
                y as i32 + self.lo[1],
                z as i32 + self.lo[2],
            ],
            k: k as u8,
        }
    }

    /// The one key a physical site has: wrapped into the box on the
    /// commensurate axes.
    #[inline]
    fn canonical(&self, s: Site) -> Site {
        if !self.any_wrap {
            return s;
        }
        canonical_site(self.lat.global(), self.cell, &self.cell_inv_t, self.wrap, s)
    }

    #[inline]
    fn neighbour(&self, idx: usize, dn: [i32; 3], l: u8) -> Option<usize> {
        let s = self.site(idx);
        let t = Site {
            n: [s.n[0] + dn[0], s.n[1] + dn[1], s.n[2] + dn[2]],
            k: l,
        };
        self.index(self.canonical(t))
    }
}

/// The one key a physical site has: `s` wrapped into the box on the axes in
/// `wrap` (where the lattice is commensurate with the box). Index arithmetic
/// on the global lattice, which is the one snapped to the box: a site and
/// its periodic image differ by a translation.
pub(super) fn canonical_site(
    g: &super::lattice::AffineLattice,
    cell: &UnitCell,
    cell_inv_t: &Mat3,
    wrap: [bool; 3],
    s: Site,
) -> Site {
    let p = g.position(s);
    let mut l = cell.x2lamda_pre(p, cell_inv_t);
    // A site on a periodic face must wrap the same way from either side,
    // or it gets two keys (one occupied, one "vacant"). Fractions within
    // `FACE_TOL` below 1 wrap to 0.
    for (v, &w) in l.iter_mut().zip(&wrap) {
        if w {
            *v -= (*v + FACE_TOL).floor();
        }
    }
    g.nearest(cell.lamda2x(l)).0
}

/// Run the census of one grain.
pub fn census(input: &CensusInput, env: &dyn Surroundings, par: &Par) -> Result<Census> {
    let lat = input.lattice;
    let n = input.positions.len();
    let motif = &lat.global().motif;
    let nb = motif.basis.len();
    let cell_inv_t = input.cell.inverse_transposed()?;
    let mut grid = Grid {
        lat,
        lo: [0; 3],
        dim: [1; 3],
        nb,
        wrap: input.wrap,
        any_wrap: input.wrap.iter().any(|&w| w),
        cell: input.cell,
        cell_inv_t,
    };

    if n == 0 {
        return Ok(Census::default());
    }
    // 1. The occupied keys' bounding box, without storing the keys: the
    // assignment is cheap enough to compute twice, and a stored key is 16 B
    // per atom.
    // Canonical keys do not depend on the grid's bounds, so the closure keeps
    // its own copy while `grid` gets them.
    let keys = grid;
    let resolved: HashMap<usize, Site> = match input.neighbours {
        Some(nb) => resolve_ambiguous(lat, input.positions, nb, &|s| keys.canonical(s), par),
        None => HashMap::new(),
    };
    let key = |i: usize| -> (Site, f32) {
        if let Some(&s) = resolved.get(&i) {
            let p = lat.position(s);
            let x = input.positions[i];
            let (d, _) = input
                .cell
                .min_image_with_shift_pre([x[0] - p[0], x[1] - p[1], x[2] - p[2]], &cell_inv_t);
            return (s, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() as f32);
        }
        let (s, _, d2) = lat.nearest(input.positions[i]);
        (keys.canonical(s), d2.sqrt() as f32)
    };
    let boxes = par.map_chunks(n, |a, b| {
        let mut lo = [i32::MAX; 3];
        let mut hi = [i32::MIN; 3];
        for i in a..b {
            let s = key(i).0;
            for d in 0..3 {
                lo[d] = lo[d].min(s.n[d]);
                hi[d] = hi[d].max(s.n[d]);
            }
        }
        (lo, hi)
    });
    let mut lo = [i32::MAX; 3];
    let mut hi = [i32::MIN; 3];
    for (l, h) in boxes {
        for d in 0..3 {
            lo[d] = lo[d].min(l[d]);
            hi[d] = hi[d].max(h[d]);
        }
    }
    // 2. Grid over the occupied keys, with a margin for neighbours, and the
    // assignment as (grid index, displacement): 8 B per atom.
    for a in 0..3 {
        grid.lo[a] = lo[a] - 2;
        grid.dim[a] = (hi[a] - lo[a] + 5) as usize;
    }
    let total = grid.len();
    check_site_count(total, n)?;
    let mut assign = vec![(0u32, 0.0f32); n];
    par.fill(&mut assign, |i, out| {
        let (s, d) = key(i);
        let idx = grid.index(s).expect("occupied site inside its own grid");
        *out = (idx as u32, d);
    });
    let mut occ = vec![0u16; total];
    for &(idx, _) in &assign {
        occ[idx as usize] = occ[idx as usize].saturating_add(1);
    }
    let atom_site = |i: usize| assign[i].0;

    // 3. Domain: occupied sites, then enclosed empty ones.
    let shell = &motif.shell;
    let need = motif.coordination() / 2 + 1;
    let mut in_dom: Vec<bool> = occ.iter().map(|&o| o > 0).collect();
    let mut probed: HashMap<u32, (bool, bool)> = HashMap::new();
    let mut candidates: Vec<u32> = Vec::new();
    let mut seen = vec![false; total];
    let push_empty_neighbours =
        |from: &[u32], in_dom: &[bool], seen: &mut [bool], out: &mut Vec<u32>| {
            for &idx in from {
                let k = idx as usize % nb;
                for &(dn, l) in &shell[k] {
                    if let Some(j) = grid.neighbour(idx as usize, dn, l) {
                        if !in_dom[j] && !seen[j] {
                            seen[j] = true;
                            out.push(j as u32);
                        }
                    }
                }
            }
        };
    let occupied: Vec<u32> = (0..total as u32).filter(|&i| occ[i as usize] > 0).collect();
    push_empty_neighbours(&occupied, &in_dom, &mut seen, &mut candidates);
    let mut iterations = 0;
    let mut converged = false;
    while iterations < input.max_domain_iterations {
        iterations += 1;
        let mut added = Vec::new();
        for &c in &candidates {
            let ci = c as usize;
            if in_dom[ci] {
                continue;
            }
            let k = ci % nb;
            let count = shell[k]
                .iter()
                .filter(|&&(dn, l)| grid.neighbour(ci, dn, l).map_or(false, |j| in_dom[j]))
                .count();
            if count < need {
                continue;
            }
            let probe = *probed
                .entry(c)
                .or_insert_with(|| env.probe(lat.position(grid.site(ci))));
            if !probe.0 {
                added.push(c);
            }
        }
        if added.is_empty() {
            converged = true;
            break;
        }
        for &a in &added {
            in_dom[a as usize] = true;
        }
        let mut next = Vec::new();
        push_empty_neighbours(&added, &in_dom, &mut seen, &mut next);
        // Rejected candidates stay eligible: a cavity fills from its rim inward.
        candidates.retain(|&c| !in_dom[c as usize]);
        candidates.extend(next);
    }

    // 3b. Cavities. No site of a compact void has the held neighbours the
    // rule above needs, so it never starts filling one. A connected set of
    // empty sites that neither reaches past the grid, nor wraps round a
    // periodic axis (a gap through the whole box is two surfaces, not a
    // void), nor grows past CAVITY_MAX, and has no site in another grain's
    // boundary zone, is enclosed by the grain: a void, counted whole.
    // Vacuum and other grains' territory reach the grid's edge, wrap, or
    // hit the cap.
    {
        // 0 unseen, 1 open, 2 being walked.
        let mut mark = vec![0u8; total];
        candidates.sort_unstable();
        candidates.dedup();
        for &c0 in &candidates {
            let c0 = c0 as usize;
            if in_dom[c0] || mark[c0] != 0 {
                continue;
            }
            let mut comp = vec![c0];
            mark[c0] = 2;
            // Each walked site's cell, unwrapped along the walk.
            let mut cell_of: HashMap<usize, [i32; 3]> = HashMap::new();
            cell_of.insert(c0, grid.site(c0).n);
            let mut open = false;
            let mut q = 0;
            while q < comp.len() && !open {
                let ci = comp[q];
                q += 1;
                let u = cell_of[&ci];
                for &(dn, l) in &shell[ci % nb] {
                    let uj = [u[0] + dn[0], u[1] + dn[1], u[2] + dn[2]];
                    match grid.neighbour(ci, dn, l) {
                        None => open = true,
                        Some(j) if !in_dom[j] => match mark[j] {
                            0 => {
                                mark[j] = 2;
                                cell_of.insert(j, uj);
                                comp.push(j);
                            }
                            1 => open = true,
                            _ => open |= cell_of[&j] != uj,
                        },
                        Some(_) => {}
                    }
                }
                open |= comp.len() > CAVITY_MAX;
            }
            if !open {
                open = comp.iter().any(|&ci| {
                    let p = *probed
                        .entry(ci as u32)
                        .or_insert_with(|| env.probe(lat.position(grid.site(ci))));
                    p.0 || p.1
                });
            }
            for &ci in &comp {
                if open {
                    mark[ci] = 1;
                } else {
                    in_dom[ci] = true;
                }
            }
        }
    }

    // 4. Classes and counts.
    let mut site_boundary: HashMap<u32, bool> = HashMap::new();
    for i in 0..n {
        if input.boundary[i] {
            site_boundary.insert(atom_site(i), true);
        }
    }
    let mut out = Census {
        domain_iterations: iterations,
        domain_converged: converged,
        ..Default::default()
    };
    for idx in 0..total {
        if !in_dom[idx] {
            continue;
        }
        let o = occ[idx] as u64;
        let b = if o > 0 {
            site_boundary.get(&(idx as u32)).copied().unwrap_or(false)
        } else {
            probed
                .get(&(idx as u32))
                .map(|p| p.1)
                .unwrap_or_else(|| env.probe(lat.position(grid.site(idx))).1)
        };
        let c = if b {
            &mut out.boundary
        } else {
            &mut out.interior
        };
        c.sites += 1;
        c.atoms += o;
        if o == 0 {
            c.vacancies += 1;
            out.vacancies.push((lat.position(grid.site(idx)), b));
        } else {
            c.interstitials += o - 1;
        }
    }

    // Interstitial atoms: on an over-occupied site, all but the closest.
    let mut closest: HashMap<u32, (f32, u32)> = HashMap::new();
    for (i, &(s, d)) in assign.iter().enumerate() {
        if occ[s as usize] > 1 {
            let e = closest.entry(s).or_insert((d, i as u32));
            if d < e.0 || (d == e.0 && (i as u32) < e.1) {
                *e = (d, i as u32);
            }
        }
    }
    for i in 0..n {
        let s = atom_site(i);
        if let Some(&(_, keep)) = closest.get(&s) {
            if keep != i as u32 {
                let b = site_boundary.get(&s).copied().unwrap_or(false);
                out.interstitials.push((i as u32, b));
            }
        }
    }

    // Antisites (§16.1a): per basis site, the majority occupant type of its
    // singly occupied sites is its species; a singly occupied site holding
    // another type is an antisite. Read off the assignment, which it does
    // not change.
    if let Some(types) = input.types {
        let single = |i: usize| occ[atom_site(i) as usize] == 1;
        let species = sublattice_species(
            nb,
            (0..n)
                .filter(|&i| single(i))
                .map(|i| ((atom_site(i) as usize % nb) as u8, types[i])),
        );
        let occ_of = || {
            (0..n)
                .filter(|&i| single(i))
                .map(|i| ((atom_site(i) as usize % nb) as u8, types[i]))
        };
        let order = chemical_order(occ_of());
        let share = majority_share(occ_of());
        let mut counts = GrainAntisites {
            sublattice: species.clone(),
            order,
            share,
            ..Default::default()
        };
        let mut atoms = Vec::new();
        for i in (0..n).filter(|&i| single(i)) {
            let s = atom_site(i);
            let want = species[s as usize % nb];
            if types[i] != want {
                let b = site_boundary.get(&s).copied().unwrap_or(false);
                counts.add(want, types[i], b);
                atoms.push((i as u32, want, lat.position(grid.site(s as usize)), b));
            }
        }
        // Neither chemical order nor a host species (a concentrated random
        // alloy): antisites are not defined.
        out.antisites = antisites_defined(order, share).then_some((counts, atoms));
        out.antisite_order = Some((order, share));
    }

    let bond = lat.global().nn();
    out.displacement = assign.iter().map(|a| a.1).collect();
    let sum2: f64 = out.displacement.iter().map(|&d| (d as f64).powi(2)).sum();
    out.rms_displacement = (sum2 / n as f64).sqrt();
    out.offlattice_fraction = out
        .displacement
        .iter()
        .filter(|&&d| d as f64 > 0.35 * bond)
        .count() as f64
        / n as f64;

    for c in [&out.interior, &out.boundary] {
        if c.residual() != 0 {
            return Err(Error::invalid(format!(
                "census does not conserve atoms (residual {}); this is a bug",
                c.residual()
            )));
        }
    }
    Ok(out)
}

/// The grain's site grid is usable: not absurdly larger than its atoms (a
/// fit that does not describe the grain), and addressable by the `u32` site
/// index the assignment stores.
fn check_site_count(total: usize, n: usize) -> Result<()> {
    if total > MAX_SITES_PER_ATOM * n + 1_000_000 {
        return Err(Error::invalid(format!(
            "grain lattice box holds {total} sites for {n} atoms; the fit does not describe it"
        )));
    }
    if total > u32::MAX as usize {
        return Err(Error::invalid(format!(
            "grain lattice box holds {total} sites, more than the census can index ({}); \
             split the grain with --region",
            u32::MAX
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::defects::lattice::{AffineLattice, Motif};
    use crate::analysis::ptm::structure;

    struct Nobody;
    impl Surroundings for Nobody {
        fn probe(&self, _: Coord3) -> (bool, bool) {
            (false, false)
        }
    }

    fn bcc_block(n: i32, a: f64, skip: &[(i32, i32, i32, u8)]) -> Vec<Coord3> {
        let mut v = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for (b, off) in [(0u8, 0.0), (1u8, 0.5)] {
                        if skip.contains(&(i, j, k, b)) {
                            continue;
                        }
                        v.push([
                            (i as f64 + off) * a,
                            (j as f64 + off) * a,
                            (k as f64 + off) * a,
                        ]);
                    }
                }
            }
        }
        v
    }

    fn run(pos: &[Coord3], wrap: [bool; 3], cell: &UnitCell) -> Census {
        let m = Motif::for_structure(structure::BCC).unwrap();
        let t = Mat3::from_rows([3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]);
        let lat = AffineLattice::new(m, t, [0.0; 3]).unwrap();
        let b = vec![false; pos.len()];
        census(
            &CensusInput {
                lattice: &lat,
                cell,
                wrap,
                positions: pos,
                boundary: &b,
                max_domain_iterations: 12,
                types: None,
                neighbours: None,
            },
            &Nobody,
            &Par::new(1),
        )
        .unwrap()
    }

    #[test]
    fn sites_on_a_periodic_face_have_one_key() {
        // Origin exactly on the box face, with the cell a hair off from an
        // integer multiple: without the tolerance, face sites wrap to 0 from
        // one side and to 1 - eps from the other and read as vacancies.
        let pos = bcc_block(6, 3.0, &[]);
        let cell = UnitCell::orthorhombic(18.0 * (1.0 + 1e-13), 18.0, 18.0);
        let c = run(&pos, [true; 3], &cell);
        assert_eq!(c.interior.vacancies, 0, "{:?}", c.interior);
    }

    #[test]
    fn a_bulk_vacancy_is_counted_with_and_without_wrapping() {
        let pos = bcc_block(6, 3.0, &[(3, 3, 3, 0), (2, 2, 2, 1)]);
        let cell = UnitCell::orthorhombic(18.0, 18.0, 18.0);
        let w = run(&pos, [true; 3], &cell);
        assert_eq!((w.interior.vacancies, w.interior.interstitials), (2, 0));
        let open = cell.with_pbc([false; 3]);
        let o = run(&pos, [false; 3], &open);
        assert_eq!(
            (o.interior.vacancies, o.interior.interstitials),
            (2, 0),
            "{o:?}"
        );
    }

    #[test]
    fn a_site_grid_past_the_u32_index_is_refused_not_wrapped() {
        let max = u32::MAX as usize;
        // Enough atoms that the sites-per-atom check passes either way.
        let n = max / 4;
        assert!(check_site_count(max, n).is_ok());
        let e = check_site_count(max + 1, n).unwrap_err().to_string();
        assert!(e.contains("more than the census can index"), "{e}");
        assert!(check_site_count(17 * 1000 + 1_000_001, 1000).is_err());
    }
}
