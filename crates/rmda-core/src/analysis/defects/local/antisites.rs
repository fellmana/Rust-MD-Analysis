//! Antisites in the local census (design §16.1a): a sublattice vote per
//! vote region.
//!
//! The local census has no index space shared over the grain: a lattice atom
//! is its own site and a defect atom's site comes from a lattice fitted to
//! its neighbours. A site's motif basis index `k` therefore means nothing
//! beyond the lattice that gave it. Antisites need it, so the grain's
//! singly occupied sites are labelled region by region:
//!
//! - **Vote regions.** The box is cut into cells about [`VOTE_CELL`] bonds on
//!   a side (at least two per periodic axis). A cell's *vote region* is the
//!   cell grown by half a cell on each side, but never wider than half the
//!   box on a periodic axis, so that every member's displacement from the
//!   region's centre is the minimum image whatever the grain's shape.
//! - **One lattice per region.** `T` is refined from the first shells of the
//!   region's lattice atoms (as a census chunk's), and the origin is fitted to
//!   them, nearest the centre first ([`super::fit_frame`]). Every singly
//!   occupied site of the region is labelled with the basis index of its
//!   nearest site of that lattice.
//! - **Vote.** Each basis index takes the majority occupant type over the
//!   region's singly occupied sites (ties to the lower type), as the global
//!   census does over a grain. A site in the cell (not in the margin) whose
//!   atom is of another type is an antisite; each site is decided by exactly
//!   one region, and the margin only adds votes.
//!
//! The fit's choice of origin only relabels `k`, which the vote absorbs, so
//! regions need not agree on labels. A region whose lattice atoms are too few
//! for a fit decides none of its sites. Unresolved defect atoms (no local
//! lattice) are not on a lattice site and are never antisites. Everything is
//! ordered by position, never by the atoms' order in the file.

use super::{fit_frame, nearest_site, pmap, LocalInput, INLIER};
use crate::analysis::defects::antisite::{
    chemical_order, majority_share, sublattice_species, GrainAntisites,
};
use crate::analysis::defects::census::AntisiteAtom;
use crate::cell::Coord3;
use crate::exec::Par;
use crate::neighbor::KnnScratch;

/// Vote cell edge, in bonds.
const VOTE_CELL: f64 = 10.0;
/// Lattice atoms whose first shells refine a region's `T`.
const T_SAMPLE: usize = 48;
/// Lattice atoms a region's origin is fitted to.
const FRAME_SAMPLE: usize = 256;

/// A singly occupied site of the local census.
#[derive(Clone, Copy, Debug)]
pub(in crate::analysis::defects) struct Single {
    /// Grain-local index of its atom.
    pub atom: u32,
    /// The census's site, Cartesian (a lattice atom's own position). An
    /// antisite is reported at its region lattice's site nearest to it.
    pub site: Coord3,
    /// Its class.
    pub boundary: bool,
    /// Its atom is a lattice atom (PTM structure the grain's).
    pub lattice: bool,
}

/// The antisites among `singles` (§16.1a), with `types` per grain atom.
/// The grain's `sublattice` is the sorted species list (one per basis site)
/// that the vote regions holding most sites agree on: labels differ between
/// regions, so only the list of species is comparable.
pub(super) fn count(
    inp: &LocalInput,
    nn: f64,
    types: &[i32],
    singles: &[Single],
    par: &Par,
) -> (GrainAntisites, Vec<AntisiteAtom>) {
    let cell = inp.cell;
    let m = inp.motif;
    let basis = &m.basis;
    let nb = basis.len();
    let Ok(inv_t) = cell.inverse_transposed() else {
        return (GrainAntisites::default(), Vec::new());
    };
    let lengths = cell.lengths();
    let nv: [usize; 3] = std::array::from_fn(|a| {
        let n = (lengths[a] / (VOTE_CELL * nn)).floor() as usize;
        if cell.pbc[a] {
            n.max(2)
        } else {
            n.max(1)
        }
    });
    // Half-width of a vote region, fractional.
    let half: [f64; 3] = std::array::from_fn(|a| {
        let w = 1.0 / nv[a] as f64;
        if cell.pbc[a] {
            w.min(0.25)
        } else {
            w
        }
    });
    let frac = |x: Coord3| -> Coord3 {
        let l = cell.x2lamda_pre(x, &inv_t);
        std::array::from_fn(|a| {
            if cell.pbc[a] {
                l[a] - l[a].floor()
            } else {
                l[a]
            }
        })
    };
    let cell_of = |f: Coord3| -> [usize; 3] {
        std::array::from_fn(|a| ((f[a] * nv[a] as f64).floor().max(0.0) as usize).min(nv[a] - 1))
    };
    let id = |c: [usize; 3]| (c[0] * nv[1] + c[1]) * nv[2] + c[2];
    let fr: Vec<Coord3> = singles.iter().map(|s| frac(s.site)).collect();
    let ncell = nv[0] * nv[1] * nv[2];
    let mut by_cell: Vec<Vec<u32>> = vec![Vec::new(); ncell];
    for (q, f) in fr.iter().enumerate() {
        by_cell[id(cell_of(*f))].push(q as u32);
    }
    let occupied: Vec<usize> = (0..ncell).filter(|&c| !by_cell[c].is_empty()).collect();
    let o0 = cell.lamda2x([0.0; 3]);
    let cart = |d: Coord3| -> Coord3 {
        let x = cell.lamda2x(d);
        [x[0] - o0[0], x[1] - o0[1], x[2] - o0[2]]
    };
    let in_grain = |j: usize| inp.grain_of[j] == inp.grain;
    let is_lat = |j: usize| in_grain(j) && (inp.lattice_atom)(j);
    let inlier2 = (INLIER * nn).powi(2);
    let serial = Par::new(1);
    let z = m.coordination();

    // Per occupied cell: its antisites (single, site type, site), its
    // species per basis index and its own (core) site count.
    type CellOut = (Vec<(u32, i32, Coord3)>, Vec<i32>, usize, (f64, f64));
    let per_cell: Vec<Option<CellOut>> = pmap(par, &occupied, 1, |&c| {
        let ci = [c / (nv[1] * nv[2]), (c / nv[2]) % nv[1], c % nv[2]];
        let centre: Coord3 = std::array::from_fn(|a| (ci[a] as f64 + 0.5) / nv[a] as f64);
        // Members: the singles of the neighbouring cells within the region,
        // with their displacement from the centre.
        let mut cells: Vec<usize> = Vec::new();
        for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                for dz in -1i64..=1 {
                    let d = [dx, dy, dz];
                    let mut k = [0usize; 3];
                    let mut ok = true;
                    for a in 0..3 {
                        let v = ci[a] as i64 + d[a];
                        if cell.pbc[a] {
                            k[a] = v.rem_euclid(nv[a] as i64) as usize;
                        } else if v < 0 || v >= nv[a] as i64 {
                            ok = false;
                        } else {
                            k[a] = v as usize;
                        }
                    }
                    if ok {
                        cells.push(id(k));
                    }
                }
            }
        }
        cells.sort_unstable();
        cells.dedup();
        let mut members: Vec<(u32, Coord3)> = Vec::new();
        for &k in &cells {
            for &q in &by_cell[k] {
                let f = fr[q as usize];
                let mut d = [0.0; 3];
                let mut inside = true;
                for a in 0..3 {
                    let mut v = f[a] - centre[a];
                    if cell.pbc[a] {
                        v -= v.round();
                    }
                    inside &= v.abs() <= half[a];
                    d[a] = v;
                }
                if inside {
                    members.push((q, cart(d)));
                }
            }
        }
        // Ordered by position (the singles' order follows the file's).
        members.sort_by_cached_key(|&(q, _)| {
            crate::analysis::defects::place_key(cell, singles[q as usize].site)
        });
        let lattice: Vec<(u32, Coord3)> = members
            .iter()
            .copied()
            .filter(|&(q, _)| singles[q as usize].lattice)
            .collect();
        // `T` from the first shells of a sample of the region's lattice atoms.
        let t = {
            let sample = crate::analysis::defects::place_sample(
                cell,
                &lattice,
                |(q, _)| singles[q as usize].site,
                T_SAMPLE,
            );
            let mut sc = KnnScratch::default();
            let mut nbr = Vec::new();
            let mut pairs = Vec::new();
            for &(q, _) in &sample {
                let i = inp.atoms[singles[q as usize].atom as usize] as usize;
                inp.index.knn(
                    i,
                    inp.positions[i],
                    crate::analysis::defects::fit::fit_neighbours(z),
                    &mut sc,
                    &mut nbr,
                );
                pairs.extend(nbr.iter().filter(|e| is_lat(e.0 as usize)).map(|e| e.1));
            }
            crate::analysis::defects::fit::refine(m, inp.t, &pairs, 3, &serial)
                .map_or(inp.t, |r| r.0)
        };
        // The origin, from lattice atoms nearest the centre first.
        let mut frame: Vec<Coord3> = crate::analysis::defects::place_sample(
            cell,
            &lattice,
            |(q, _)| singles[q as usize].site,
            FRAME_SAMPLE,
        )
        .into_iter()
        .map(|e| e.1)
        .collect();
        let d2 = |e: &Coord3| e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        frame.sort_by(|a, b| d2(a).total_cmp(&d2(b)));
        let f = fit_frame(t, basis, &frame, inlier2, &serial)?;
        // Basis index, and the vector to the site.
        let labels: Vec<(u8, Coord3)> = members
            .iter()
            .map(|&(_, e)| nearest_site(&f.t, &f.ti, basis, f.o, e))
            .collect();
        let species = sublattice_species(
            nb,
            members
                .iter()
                .zip(&labels)
                .map(|(&(q, _), &(k, _))| (k, types[singles[q as usize].atom as usize])),
        );
        let core = by_cell[c].len();
        // The region's chemical order, weighted by its core sites below.
        let occ_of = || {
            members
                .iter()
                .zip(&labels)
                .map(|(&(q, _), &(k, _))| (k, types[singles[q as usize].atom as usize]))
        };
        let order = (chemical_order(occ_of()), majority_share(occ_of()));
        let mut found = Vec::new();
        for (&(q, _), &(k, d)) in members.iter().zip(&labels) {
            let s = &singles[q as usize];
            if id(cell_of(fr[q as usize])) != c {
                continue;
            }
            let want = species[k as usize];
            if types[s.atom as usize] != want {
                let p = s.site;
                found.push((q, want, [p[0] + d[0], p[1] + d[1], p[2] + d[2]]));
            }
        }
        Some((found, species, core, order))
    });

    let mut out = GrainAntisites::default();
    let mut atoms: Vec<AntisiteAtom> = Vec::new();
    let mut lists: std::collections::BTreeMap<Vec<i32>, usize> = Default::default();
    let (mut vsum, mut ssum, mut wsum) = (0.0, 0.0, 0.0);
    for (found, species, core, order) in per_cell.into_iter().flatten() {
        vsum += order.0 * core as f64;
        ssum += order.1 * core as f64;
        wsum += core as f64;
        let mut s = species;
        s.sort_unstable();
        *lists.entry(s).or_insert(0) += core;
        for (q, want, site) in found {
            let s = &singles[q as usize];
            out.add(want, types[s.atom as usize], s.boundary);
            atoms.push((s.atom, want, cell.wrap(site), s.boundary));
        }
    }
    // Most sites; ties to the lower list (ascending map, strict `>`).
    let mut best = (0usize, Vec::new());
    for (s, n) in lists {
        if n > best.0 {
            best = (n, s);
        }
    }
    out.sublattice = best.1;
    out.order = if wsum > 0.0 { vsum / wsum } else { 1.0 };
    out.share = if wsum > 0.0 { ssum / wsum } else { 1.0 };
    atoms.sort_unstable_by_key(|a| a.0);
    (out, atoms)
}
