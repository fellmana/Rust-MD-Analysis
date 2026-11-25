//! Void detection, method A on the defect census (design §17).
//!
//! A void is a cluster of **vacancy sites only**: the vacancies of measured
//! grains ([`DefectsResult::vacancies`]) joined when within the cluster cutoff
//! (by default each grain's own, halfway between its first and second
//! neighbour shells, as the defect clusters of §16.5 use, so two vacancies
//! join exactly when they are first neighbours). Interstitials take no part.
//!
//! - **Volume** `V = Σ Ω_site` over the members, `Ω_site = |det T| / n_basis`
//!   of the member's grain (the grain's fitted conventional cell over its
//!   basis count; for the local reference, the grain-wide prefit). A grain
//!   with no fitted cell (explicit reference) uses the ideal cell of its
//!   structure at its measured bond.
//! - **Equivalent radius** `(3V / 4π)^(1/3)`.
//! - **Class.** `surface` when the void is open: some member has a first-shell
//!   lattice position that holds neither an atom nor a vacancy (within 0.45
//!   bonds, so a grain bent up to ~25° away from its fitted cell still reads
//!   closed voids as closed; a grain with no fitted cell counts held points
//!   against the coordination instead), i.e. empty
//!   space the census did not count — vacuum above a free surface, a pit, or
//!   (rarely) an open region the census refused as a cavity (§16.2: one that
//!   reaches another grain's boundary zone or exceeds the cavity cap).
//!   Otherwise `boundary` when a member is in the grain-boundary class, else
//!   `bulk`. A closed void, however close to the surface, is `bulk`: one
//!   atomic layer over it closes it.
//!
//! Because the census fills enclosed cavities whole (§16.2), a closed void's
//! interior is counted and its volume is exact on a lattice. An open void's
//! empty sites are mostly outside the counted domain (that is what open
//! means), so a `surface` void's volume counts only the sites the census
//! counted: a lower bound.
//!
//! The order of atoms in the file does not enter: vacancies come from the
//! census in site order, and voids are sorted by size, then position.

use super::clusters::{self, Cutoff};
use super::lattice::Motif;
use super::{place_key, DefectsResult, GrainReport, Status};
use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::Result;
use crate::exec::Par;
use crate::neighbor::{KnnIndex, KnnScratch};

/// A first-shell position is held when an atom or a vacancy lies within this
/// many bonds of it. Interstitial positions are at least 0.5 bonds from any
/// site in BCC, FCC and HCP, and the next site at least one bond away.
const HELD_TOLERANCE: f64 = 0.45;

/// Where a void lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VoidClass {
    /// Closed, every member interior.
    Bulk,
    /// Closed, some member in a grain-boundary zone.
    Boundary,
    /// Open to empty space the census did not count (a free surface).
    Surface,
}

impl VoidClass {
    pub fn label(self) -> &'static str {
        match self {
            VoidClass::Bulk => "bulk",
            VoidClass::Boundary => "boundary",
            VoidClass::Surface => "surface",
        }
    }
}

/// One void.
#[derive(Debug, Clone)]
pub struct Void {
    /// Dense id, 0 for the largest.
    pub id: u32,
    pub n_sites: u32,
    /// `Σ Ω_site`, Å³.
    pub volume: f64,
    /// `(3V/4π)^(1/3)`, Å.
    pub radius: f64,
    /// Centre of the member sites, unwrapped along the cluster.
    pub centre: Coord3,
    /// Grain of the first member; `multi_grain` when members differ.
    pub grain: i32,
    pub multi_grain: bool,
    pub class: VoidClass,
}

/// Settings.
#[derive(Debug, Clone)]
pub struct VoidConfig {
    /// Smallest cluster reported as a void; smaller clusters' vacancies are
    /// counted as isolated.
    pub min_sites: usize,
    /// Joining distance; `None` is each grain's shell cutoff (§16.5).
    pub cutoff: Option<f64>,
}

impl Default for VoidConfig {
    fn default() -> Self {
        Self {
            min_sites: 2,
            cutoff: None,
        }
    }
}

/// The voids of one frame.
#[derive(Debug, Clone, Default)]
pub struct Voids {
    /// Largest first (then by position).
    pub voids: Vec<Void>,
    /// Per vacancy of the census: its void's id, `None` when its cluster is
    /// smaller than `min_sites`.
    pub vacancy_void: Vec<Option<u32>>,
    /// Vacancies in clusters smaller than `min_sites`.
    pub n_isolated: usize,
    /// The cutoff the clustering used (the largest measured grain's when per
    /// grain), Å.
    pub cutoff: f64,
    pub min_sites: usize,
}

impl Voids {
    pub fn n_sites(&self) -> u64 {
        self.voids.iter().map(|v| v.n_sites as u64).sum()
    }
    /// Total void volume, Å³; `0.0` (not `-0.0`, an empty float sum) with no
    /// voids.
    pub fn volume(&self) -> f64 {
        self.voids.iter().map(|v| v.volume).sum::<f64>() + 0.0
    }
    pub fn count(&self, class: VoidClass) -> usize {
        self.voids.iter().filter(|v| v.class == class).count()
    }
}

/// The grain's fitted cell; for the local reference the grain-wide prefit.
/// Its shell vectors locate first-shell positions to within the 0.45-bond
/// [`HELD_TOLERANCE`], i.e. up to ~25° of local misorientation, which covers
/// any elastic bend, and its volume is the grain's mean site volume. Counting
/// held points within the first-shell cutoff instead (the explicit
/// reference's fallback) is fragile there: the local census places atom-by-
/// atom vacancies a little off the ideal sites, and with thermal noise an
/// eighth neighbour of a closed void's member sits just past the cutoff.
fn grain_wide_cell(g: &GrainReport) -> Option<Mat3> {
    if g.reference == "explicit" {
        return None;
    }
    g.t.iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(Mat3 { m: g.t })
}

/// Volume of one site of a grain, Å³: `|det T| / n_basis` of its grain-wide
/// cell, else the ideal cell of its structure at its bond; `NaN` without
/// either.
pub fn site_volume(g: &GrainReport) -> f64 {
    let Some(m) = Motif::for_structure(g.structure) else {
        return f64::NAN;
    };
    let nb = m.basis.len() as f64;
    if let Some(t) = grain_wide_cell(g) {
        let d = t.det().abs();
        if d > 0.0 {
            return d / nb;
        }
    }
    let scale = g.bond / m.nn;
    m.cell.det().abs() * scale.powi(3) / nb
}

/// First-shell vectors of a grain, Cartesian, one set per motif basis site
/// with distinct shells (one set for Bravais lattices); empty when the grain
/// has no grain-wide cell.
fn shell_sets(g: &GrainReport) -> Vec<Vec<Coord3>> {
    let Some(m) = Motif::for_structure(g.structure) else {
        return Vec::new();
    };
    let Some(t) = grain_wide_cell(g) else {
        return Vec::new();
    };
    let mut sets: Vec<Vec<Coord3>> = Vec::new();
    for (k, shell) in m.shell.iter().enumerate() {
        let mut v: Vec<Coord3> = shell
            .iter()
            .map(|&(dn, l)| {
                let b = m.basis[l as usize];
                let a = m.basis[k];
                t.mul_v([
                    dn[0] as f64 + b[0] - a[0],
                    dn[1] as f64 + b[1] - a[1],
                    dn[2] as f64 + b[2] - a[2],
                ])
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let same = |w: &Vec<Coord3>| {
            w.len() == v.len()
                && w.iter()
                    .zip(&v)
                    .all(|(p, q)| (0..3).all(|d| (p[d] - q[d]).abs() < 1e-9))
        };
        if !sets.iter().any(same) {
            sets.push(std::mem::take(&mut v));
        }
    }
    sets
}

/// Find the voids among `result`'s vacancies. `positions` are the frame's
/// atoms (all of them, any grain), which decide whether a first-shell
/// position is held.
pub fn find(
    cell: &UnitCell,
    positions: &[Coord3],
    result: &DefectsResult,
    cfg: &VoidConfig,
    par: &Par,
) -> Result<Voids> {
    let nv = result.vacancies.len();
    let cutoff = match cfg.cutoff {
        Some(c) => Cutoff::Uniform(c),
        None => clusters::default_cutoff(result),
    };
    let mut out = Voids {
        vacancy_void: vec![None; nv],
        cutoff: cutoff.reported(),
        min_sites: cfg.min_sites,
        ..Default::default()
    };
    if nv == 0 {
        return Ok(out);
    }
    let vp: Vec<Coord3> = result.vacancies.iter().map(|v| v.position).collect();
    let vg: Vec<i32> = result.vacancies.iter().map(|v| v.grain).collect();
    let cl = clusters::cluster(cell, &vp, &vg, &[], &[], &cutoff, par)?;

    let grain = |g: i32| {
        result
            .grains
            .iter()
            .find(|r| r.grain == g && r.status == Status::Measured)
    };
    let mut omega = std::collections::HashMap::new();
    let mut shells = std::collections::HashMap::new();
    for &g in &vg {
        omega
            .entry(g)
            .or_insert_with(|| grain(g).map_or(f64::NAN, site_volume));
        shells
            .entry(g)
            .or_insert_with(|| grain(g).map_or_else(Vec::new, shell_sets));
    }

    let mut members: Vec<Vec<u32>> = vec![Vec::new(); cl.clusters.len()];
    for (k, &c) in cl.vacancy_cluster.iter().enumerate() {
        members[c as usize].push(k as u32);
    }

    // Held positions: atoms and vacancies, one index.
    let mut pts: Vec<Coord3> = positions.to_vec();
    pts.extend_from_slice(&vp);
    let index = KnnIndex::build(cell, &pts, par)?;
    let open = |k: usize| -> bool {
        let v = &result.vacancies[k];
        let Some(g) = grain(v.grain) else {
            return false;
        };
        let bond = g.bond;
        let mut s = KnnScratch::default();
        let mut nb = Vec::new();
        let sets = &shells[&v.grain];
        if sets.is_empty() {
            // No fitted cell (explicit reference): count the held points in the
            // first shell against the coordination.
            let Some(m) = Motif::for_structure(g.structure) else {
                return false;
            };
            let rc = clusters::shell_cutoff(g.structure) * bond;
            index.knn_point(v.position, 32, &mut s, &mut nb);
            let held = nb
                .iter()
                .filter(|e| e.2 <= rc * rc && e.2 > 1e-12 * bond * bond)
                .count();
            return held < m.coordination();
        }
        let tol2 = (HELD_TOLERANCE * bond).powi(2);
        // The basis site whose shell is held best is the member's own; on a
        // Bravais lattice there is one set.
        sets.iter().all(|set| {
            set.iter().any(|d| {
                let p = [
                    v.position[0] + d[0],
                    v.position[1] + d[1],
                    v.position[2] + d[2],
                ];
                index.knn_point(p, 1, &mut s, &mut nb);
                nb.first().map_or(true, |e| e.2 > tol2)
            })
        })
    };

    let mut voids: Vec<(Void, Vec<u32>)> = Vec::new();
    for (c, m) in cl.clusters.iter().zip(members) {
        if m.len() < cfg.min_sites.max(1) {
            out.n_isolated += m.len();
            continue;
        }
        let volume: f64 = m.iter().map(|&k| omega[&vg[k as usize]]).sum();
        let class = if m.iter().any(|&k| open(k as usize)) {
            VoidClass::Surface
        } else if m.iter().any(|&k| result.vacancies[k as usize].boundary) {
            VoidClass::Boundary
        } else {
            VoidClass::Bulk
        };
        voids.push((
            Void {
                id: 0,
                n_sites: m.len() as u32,
                volume,
                radius: (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt(),
                centre: c.centre,
                grain: c.grain,
                multi_grain: c.multi_grain,
                class,
            },
            m,
        ));
    }
    voids.sort_by(|a, b| {
        b.0.n_sites
            .cmp(&a.0.n_sites)
            .then_with(|| place_key(cell, a.0.centre).cmp(&place_key(cell, b.0.centre)))
    });
    for (id, (v, m)) in voids.iter_mut().enumerate() {
        v.id = id as u32;
        for &k in m.iter() {
            out.vacancy_void[k as usize] = Some(id as u32);
        }
    }
    out.voids = voids.into_iter().map(|e| e.0).collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::defects::Vacancy;
    use crate::analysis::ptm::structure;

    fn bcc_grain(a: f64) -> GrainReport {
        let mut g = GrainReport::new(0, 1000);
        g.status = Status::Measured;
        g.structure = structure::BCC;
        g.t = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]];
        g.origin = [0.0; 3];
        g.bond = a * 3f64.sqrt() / 2.0;
        g
    }

    /// Perfect BCC block, with the sites within `r` of `c` removed (returned
    /// as the census would report them).
    fn block(n: usize, a: f64, c: Coord3, r: f64) -> (UnitCell, Vec<Coord3>, DefectsResult) {
        let l = n as f64 * a;
        let cell = UnitCell::orthorhombic(l, l, l);
        let (mut atoms, mut vac) = (Vec::new(), Vec::new());
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for o in [0.0, 0.5] {
                        let p = [(i as f64 + o) * a, (j as f64 + o) * a, (k as f64 + o) * a];
                        if (0..3).map(|d| (p[d] - c[d]).powi(2)).sum::<f64>() < r * r {
                            vac.push(Vacancy {
                                position: p,
                                grain: 0,
                                boundary: false,
                            });
                        } else {
                            atoms.push(p);
                        }
                    }
                }
            }
        }
        let res = DefectsResult {
            grains: vec![bcc_grain(a)],
            vacancies: vac,
            ..Default::default()
        };
        (cell, atoms, res)
    }

    #[test]
    fn site_volume_is_the_cell_over_its_basis_with_or_without_a_fit() {
        let a = 3.165;
        let g = bcc_grain(a);
        assert!((site_volume(&g) - a * a * a / 2.0).abs() < 1e-12);
        let mut h = g.clone();
        h.t = [[f64::NAN; 3]; 3];
        assert!((site_volume(&h) - a * a * a / 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_spherical_void_is_one_bulk_void_of_its_planted_sites() {
        let a = 3.165;
        let (cell, atoms, res) = block(12, a, [6.0 * a + 0.1; 3], 5.5);
        let n = res.vacancies.len();
        assert_eq!(n, 55);
        let v = find(&cell, &atoms, &res, &VoidConfig::default(), &Par::new(2)).unwrap();
        assert_eq!(v.voids.len(), 1);
        let w = &v.voids[0];
        assert_eq!(w.n_sites as usize, n);
        assert_eq!(w.class, VoidClass::Bulk);
        let vol = n as f64 * a * a * a / 2.0;
        assert!((w.volume - vol).abs() < 1e-9 * vol);
        assert!((w.radius - (3.0 * vol / (4.0 * std::f64::consts::PI)).cbrt()).abs() < 1e-9);
    }

    #[test]
    fn a_void_with_an_unheld_neighbour_position_is_open() {
        let a = 3.165;
        let c = [6.0 * a + 0.1; 3];
        let (cell, mut atoms, mut res) = block(12, a, c, 4.0);
        assert_eq!(res.vacancies.len(), 15);
        // Remove one atom next to the void without reporting it as a vacancy:
        // an empty position the census did not count.
        let far = res.vacancies[0].position;
        let (i, _) = atoms
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (0..3).map(|d| (p[d] - far[d]).powi(2)).sum::<f64>()))
            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();
        atoms.remove(i);
        let v = find(&cell, &atoms, &res, &VoidConfig::default(), &Par::new(1)).unwrap();
        assert_eq!(v.voids[0].class, VoidClass::Surface);
        // A boundary member makes a closed void a boundary void.
        let (cell, atoms, _) = block(12, a, c, 4.0);
        res.vacancies[3].boundary = true;
        let v = find(&cell, &atoms, &res, &VoidConfig::default(), &Par::new(1)).unwrap();
        assert_eq!(v.voids[0].class, VoidClass::Boundary);
    }

    #[test]
    fn small_clusters_are_isolated_vacancies() {
        let a = 3.165;
        let (cell, atoms, res) = block(10, a, [5.0 * a + 0.1; 3], 0.5);
        assert_eq!(res.vacancies.len(), 1);
        let v = find(&cell, &atoms, &res, &VoidConfig::default(), &Par::new(1)).unwrap();
        assert!(v.voids.is_empty());
        assert_eq!(v.n_isolated, 1);
        assert_eq!(v.vacancy_void, vec![None]);
        let cfg = VoidConfig {
            min_sites: 1,
            ..Default::default()
        };
        let v = find(&cell, &atoms, &res, &cfg, &Par::new(1)).unwrap();
        assert_eq!(v.voids.len(), 1);
        assert_eq!(v.vacancy_void, vec![Some(0)]);
    }
}
