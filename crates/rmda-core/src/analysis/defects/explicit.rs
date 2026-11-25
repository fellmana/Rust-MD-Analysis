//! Wigner–Seitz against an explicit reference frame (design §16.3a), on
//! the census result types: OVITO's `WignerSeitzAnalysisModifier`.
//!
//! The sites are the reference frame's atoms. Every current atom goes to its
//! nearest site (pure WS: no tolerance, no cutoff, ties to the lower site
//! index); a vacancy is a site with no atom, the interstitial count is
//! `Σ max(occ − 1, 0)`, and the interstitial atoms are, per over-occupied
//! site, all but the atom nearest to it (ties to the lower atom index), as in
//! the lattice census (§16.1).
//!
//! **Everything about a site is decided on the reference, once.** Its grain
//! (one grain, `--region` evaluated on the reference frame, a reference
//! column, or a segmentation of the reference), its interior/boundary class
//! (within the boundary width of a site of another grain or of none), and
//! each grain's structure and bond (PTM of the reference). An atom counts in
//! the grain and class of the site it is assigned to, so every class
//! conserves: `V − I = sites − atoms` with `atoms` the atoms assigned to the
//! class's sites. Atoms on sites of no grain are not counted, like atoms of
//! no grain in the lattice census. With one grain (no labels) every site is
//! interior and the totals are OVITO's.
//!
//! **Affine mapping** ([`AffineMapping`], OVITO's `affine_mapping`):
//! `Off` (OVITO's default) assigns current positions as they are, periodic
//! in the reference cell; `ToReference` maps current positions into the
//! reference cell through fractional coordinates first (OVITO's
//! `ToReference`); `ToCurrent` maps the sites into the current cell instead
//! (OVITO refuses it for WS; here it differs from `ToReference` only in
//! which cell's metric decides "nearest"). Vacancy positions are reported in
//! the current cell whenever a mapping is on, so they overlay the current
//! frame; with `Off` they are the reference positions.

use super::{Antisite, DefectsResult, GrainReport, Interstitial, Status, Vacancy};
use crate::analysis::ptm::{structure, PtmAtom};
use crate::cell::{Coord3, UnitCell};
use crate::error::{Error, Result};
use crate::exec::Par;
use crate::neighbor::{KnnIndex, KnnScratch};

/// How the reference cell is related to the current one (OVITO's
/// `ReferenceConfigurationModifier.AffineMapping`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AffineMapping {
    /// Positions as they are; periodic images in the reference cell.
    #[default]
    Off,
    /// Current positions mapped into the reference cell.
    ToReference,
    /// Reference sites mapped into the current cell.
    ToCurrent,
}

impl AffineMapping {
    pub fn label(&self) -> &'static str {
        match self {
            AffineMapping::Off => "off",
            AffineMapping::ToReference => "to-reference",
            AffineMapping::ToCurrent => "to-current",
        }
    }
}

/// Per grain of the reference.
#[derive(Debug, Clone)]
struct GrainInfo {
    grain: i32,
    structure: i32,
    n_crystalline: usize,
    bond: f64,
}

/// A prepared explicit reference: sites, their grains and classes, and a
/// nearest-site index in the reference cell. Built once per run.
pub struct ExplicitReference {
    cell: UnitCell,
    positions: Vec<Coord3>,
    grain_of: Vec<i32>,
    boundary: Vec<bool>,
    grains: Vec<GrainInfo>,
    /// Dense slot of each site's grain in `grains`, `u32::MAX` for none.
    slot: Vec<u32>,
    index: KnnIndex,
}

impl ExplicitReference {
    /// `grain_of[s]` is site `s`'s grain (negative: none); `ptm` is the
    /// reference frame's PTM, for each grain's structure and bond and the
    /// default boundary width (1.5 mean first-neighbour distances).
    pub fn new(
        cell: &UnitCell,
        positions: &[Coord3],
        grain_of: &[i32],
        ptm: &[PtmAtom],
        boundary_width: Option<f64>,
        par: &Par,
    ) -> Result<Self> {
        let n = positions.len();
        if n == 0 {
            return Err(Error::invalid("the WS reference frame has no atoms"));
        }
        if grain_of.len() != n || ptm.len() != n {
            return Err(Error::invalid(
                "WS reference: labels or PTM do not match its atoms",
            ));
        }
        let index = {
            let _t = crate::prof::StageTimer::start("defects.ws_reference_index");
            KnnIndex::build(cell, positions, par)?
        };
        let mut ids: Vec<i32> = grain_of.iter().copied().filter(|&g| g >= 0).collect();
        ids.sort_unstable();
        ids.dedup();
        let slot: Vec<u32> = grain_of
            .iter()
            .map(|&g| {
                if g < 0 {
                    u32::MAX
                } else {
                    ids.binary_search(&g).unwrap() as u32
                }
            })
            .collect();

        // Per grain: majority lattice structure and its mean PTM bond.
        let lattice_types = [
            structure::FCC,
            structure::HCP,
            structure::BCC,
            structure::SC,
            structure::DHCP,
        ];
        let nt = structure::DHCP as usize + 1;
        let mut counts = vec![0usize; ids.len() * nt];
        let mut bond_sum = vec![0.0f64; ids.len() * nt];
        for (s, p) in ptm.iter().enumerate() {
            let k = slot[s];
            let t = p.structure();
            if k == u32::MAX || !(0..nt as i32).contains(&t) {
                continue;
            }
            let at = k as usize * nt + t as usize;
            counts[at] += 1;
            if p.interatomic_distance.is_finite() {
                bond_sum[at] += f64::from(p.interatomic_distance);
            }
        }
        let mut nn_fallback: Option<f64> = None;
        let mut grains = Vec::with_capacity(ids.len());
        for (k, &g) in ids.iter().enumerate() {
            let st = *lattice_types
                .iter()
                .max_by_key(|&&t| counts[k * nt + t as usize])
                .unwrap();
            let c = counts[k * nt + st as usize];
            let (structure, bond) = if c > 0 {
                (st, bond_sum[k * nt + st as usize] / c as f64)
            } else {
                let b = match nn_fallback {
                    Some(b) => b,
                    None => {
                        let b = median_nn(&index, positions);
                        nn_fallback = Some(b);
                        b
                    }
                };
                (structure::OTHER, b)
            };
            grains.push(GrainInfo {
                grain: g,
                structure,
                n_crystalline: c,
                bond,
            });
        }

        let width = boundary_width.unwrap_or_else(|| {
            let (s, c) = ptm
                .iter()
                .filter(|p| p.structure() != structure::OTHER && p.interatomic_distance.is_finite())
                .fold((0.0, 0usize), |(s, c), p| {
                    (s + f64::from(p.interatomic_distance), c + 1)
                });
            if c > 0 {
                1.5 * s / c as f64
            } else {
                1.5 * median_nn(&index, positions)
            }
        });
        // One label everywhere: no boundary to look for.
        let boundary = if ids.len() == 1 && grain_of.iter().all(|&g| g >= 0) {
            vec![false; n]
        } else {
            let _t = crate::prof::StageTimer::start("defects.boundary");
            super::boundary_flags(&index, positions, grain_of, width, par)
        };
        Ok(Self {
            cell: *cell,
            positions: positions.to_vec(),
            grain_of: grain_of.to_vec(),
            boundary,
            grains,
            slot,
            index,
        })
    }

    /// Number of sites.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether there are no sites (never, once built).
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The reference cell.
    pub fn cell(&self) -> &UnitCell {
        &self.cell
    }

    /// Site `s`'s grain (negative: none).
    pub fn site_grain(&self, s: usize) -> i32 {
        self.grain_of[s]
    }

    /// Median first-neighbour distance over about `sample` evenly spaced
    /// sites (every `len / sample`-th).
    pub fn median_nn(&self, sample: usize) -> f64 {
        median_nn_sampled(&self.index, &self.positions, sample)
    }
}

/// Median nearest-neighbour distance over up to 20 000 sites.
fn median_nn(index: &KnnIndex, positions: &[Coord3]) -> f64 {
    median_nn_sampled(index, positions, 20_000)
}

fn median_nn_sampled(index: &KnnIndex, positions: &[Coord3], sample: usize) -> f64 {
    let step = (positions.len() / sample.max(1)).max(1);
    let mut s = KnnScratch::default();
    let mut out = Vec::new();
    let mut d: Vec<f64> = positions
        .iter()
        .enumerate()
        .step_by(step)
        .filter_map(|(i, &p)| {
            index.knn(i, p, 1, &mut s, &mut out);
            out.first().map(|e| e.2.sqrt())
        })
        .collect();
    if d.is_empty() {
        return 1.0;
    }
    d.sort_by(f64::total_cmp);
    d[d.len() / 2]
}

/// What a run produces beyond the shared result.
pub struct ExplicitOutcome {
    pub result: DefectsResult,
    /// Per current atom: the grain of the site it was assigned to.
    pub atom_grain: Vec<i32>,
    /// Per current atom: the site it was assigned to.
    pub site_of: Vec<u32>,
    /// Per site: its occupancy.
    pub occupancy: Vec<u32>,
    /// Site index of each of `result.vacancies`, and of the site each of
    /// `result.interstitials` sits on.
    pub vacancy_sites: Vec<u32>,
    pub interstitial_sites: Vec<u32>,
    /// Site index of each of `result.antisites`.
    pub antisite_sites: Vec<u32>,
}

/// Per-type occupancy for antisites: the type of each site (reference atom)
/// and of each current atom, in one numbering.
#[derive(Debug, Clone, Copy)]
pub struct Types<'a> {
    pub sites: &'a [i32],
    pub atoms: &'a [i32],
}

/// Wigner–Seitz of the current frame against `r`.
pub fn run(
    r: &ExplicitReference,
    cell: &UnitCell,
    positions: &[Coord3],
    mapping: AffineMapping,
    par: &Par,
) -> Result<ExplicitOutcome> {
    run_typed(r, cell, positions, mapping, None, par)
}

/// [`run`], also counting antisites (§16.1a) when `types` is given: a singly
/// occupied site whose atom's type is not the site's (OVITO's
/// `per_type_occupancies`: total occupancy 1, own-type occupancy 0). The
/// assignment, vacancies and interstitials are the same as without.
pub fn run_typed(
    r: &ExplicitReference,
    cell: &UnitCell,
    positions: &[Coord3],
    mapping: AffineMapping,
    types: Option<Types>,
    par: &Par,
) -> Result<ExplicitOutcome> {
    if let Some(t) = types {
        if t.sites.len() != r.positions.len() || t.atoms.len() != positions.len() {
            return Err(Error::invalid(
                "WS reference: types do not match the sites or the atoms",
            ));
        }
    }
    if cell.pbc != r.cell.pbc {
        return Err(Error::invalid(format!(
            "WS reference has periodic flags {:?}, the current frame {:?}",
            r.cell.pbc, cell.pbc
        )));
    }
    let n = positions.len();
    let ns = r.positions.len();
    if ns > u32::MAX as usize || n > u32::MAX as usize {
        return Err(Error::invalid("WS reference: more than u32::MAX atoms"));
    }

    // Sites in the frame the census runs in, and the index over them.
    let ref_inv = r.cell.inverse_transposed()?;
    let cur_inv = cell.inverse_transposed()?;
    let to_current = |x: Coord3| cell.lamda2x(r.cell.x2lamda_pre(x, &ref_inv));
    let mapped_sites: Option<Vec<Coord3>> = match mapping {
        AffineMapping::ToCurrent => {
            let mut v = vec![[0.0; 3]; ns];
            par.fill(&mut v, |s, o| *o = to_current(r.positions[s]));
            Some(v)
        }
        _ => None,
    };
    let own_index;
    let index: &KnnIndex = match &mapped_sites {
        Some(v) => {
            let _t = crate::prof::StageTimer::start("defects.ws_reference_index");
            own_index = KnnIndex::build(cell, v, par)?;
            &own_index
        }
        None => &r.index,
    };

    // Assignment: nearest site, ties to the lower site index.
    let mut assigned: Vec<(u32, f64)> = vec![(0, 0.0); n];
    {
        let _t = crate::prof::StageTimer::start("defects.ws_assign");
        par.fill(&mut assigned, |i, o| {
            thread_local! {
                static BUF: std::cell::RefCell<(KnnScratch, Vec<crate::neighbor::knn::Nbr>)> =
                    std::cell::RefCell::new((KnnScratch::default(), Vec::new()));
            }
            let q = match mapping {
                AffineMapping::ToReference => {
                    r.cell.lamda2x(cell.x2lamda_pre(positions[i], &cur_inv))
                }
                _ => positions[i],
            };
            BUF.with(|b| {
                let (s, nb) = &mut *b.borrow_mut();
                index.knn_point(q, 1, s, nb);
                let e = nb[0];
                *o = (e.0, e.2);
            });
        });
    }

    // Occupancy and, per site, the nearest of its atoms (lower index on a tie).
    let mut occupancy = vec![0u32; ns];
    let mut keeper = vec![(f64::INFINITY, u32::MAX); ns];
    for (i, &(s, d2)) in assigned.iter().enumerate() {
        let s = s as usize;
        occupancy[s] += 1;
        if d2 < keeper[s].0 {
            keeper[s] = (d2, i as u32);
        }
    }

    // Counts per grain and class.
    let ng = r.grains.len();
    let mut reps: Vec<GrainReport> = r
        .grains
        .iter()
        .map(|g| {
            let mut rep = GrainReport::new(g.grain, 0);
            rep.status = Status::Measured;
            rep.structure = g.structure;
            rep.n_crystalline = g.n_crystalline;
            rep.bond = g.bond;
            rep.reference = "explicit";
            rep.domain_converged = true;
            rep
        })
        .collect();
    let mut sum2 = vec![0.0f64; ng];
    let mut off = vec![0usize; ng];
    let mut result = DefectsResult {
        displacement: vec![f32::NAN; n],
        ..Default::default()
    };
    let mut atom_grain = vec![-1i32; n];
    let mut site_of = vec![0u32; n];
    let mut vacancy_sites = Vec::new();
    let mut interstitial_sites = Vec::new();
    for (i, &(s, d2)) in assigned.iter().enumerate() {
        site_of[i] = s;
        let k = r.slot[s as usize];
        if k == u32::MAX {
            continue;
        }
        let k = k as usize;
        atom_grain[i] = r.grains[k].grain;
        let d = d2.sqrt();
        result.displacement[i] = d as f32;
        sum2[k] += d2;
        if d > 0.35 * r.grains[k].bond {
            off[k] += 1;
        }
        reps[k].n_atoms += 1;
    }
    for s in 0..ns {
        let k = r.slot[s];
        if k == u32::MAX {
            continue;
        }
        let rep = &mut reps[k as usize];
        let b = r.boundary[s];
        let c = if b {
            &mut rep.boundary
        } else {
            &mut rep.interior
        };
        let o = u64::from(occupancy[s]);
        c.sites += 1;
        c.atoms += o;
        if o == 0 {
            c.vacancies += 1;
            let p = match (&mapped_sites, mapping) {
                (Some(v), _) => v[s],
                (None, AffineMapping::ToReference) => to_current(r.positions[s]),
                (None, _) => r.positions[s],
            };
            vacancy_sites.push(s as u32);
            result.vacancies.push(Vacancy {
                position: p,
                grain: r.grain_of[s],
                boundary: b,
            });
        } else {
            c.interstitials += o - 1;
        }
    }
    for (i, &(s, _)) in assigned.iter().enumerate() {
        let s = s as usize;
        if r.slot[s] != u32::MAX && occupancy[s] > 1 && keeper[s].1 != i as u32 {
            interstitial_sites.push(s as u32);
            result.interstitials.push(Interstitial {
                atom: i as u32,
                grain: r.grain_of[s],
                boundary: r.boundary[s],
            });
        }
    }
    // Antisites: the one atom of a singly occupied site, of another type.
    let mut antisite_sites = Vec::new();
    if let Some(t) = types {
        for rep in reps.iter_mut() {
            // The reference's atoms carry their own types: no order to infer.
            rep.antisites = Some(super::antisite::GrainAntisites {
                order: 1.0,
                share: 1.0,
                ..Default::default()
            });
        }
        for (i, &(s, _)) in assigned.iter().enumerate() {
            let s = s as usize;
            let k = r.slot[s];
            if k == u32::MAX || occupancy[s] != 1 || t.atoms[i] == t.sites[s] {
                continue;
            }
            let b = r.boundary[s];
            reps[k as usize]
                .antisites
                .as_mut()
                .unwrap()
                .add(t.sites[s], t.atoms[i], b);
            let p = match (&mapped_sites, mapping) {
                (Some(v), _) => v[s],
                (None, AffineMapping::ToReference) => to_current(r.positions[s]),
                (None, _) => r.positions[s],
            };
            antisite_sites.push(s as u32);
            result.antisites.push(Antisite {
                atom: i as u32,
                site_type: t.sites[s],
                site_position: p,
                grain: r.grain_of[s],
                boundary: b,
            });
        }
    }
    for (k, rep) in reps.iter_mut().enumerate() {
        if rep.n_atoms > 0 {
            rep.rms_displacement = (sum2[k] / rep.n_atoms as f64).sqrt();
            rep.offlattice_fraction = off[k] as f64 / rep.n_atoms as f64;
        }
        for c in [&rep.interior, &rep.boundary] {
            debug_assert_eq!(c.residual(), 0);
        }
    }
    result.grains = reps;
    Ok(ExplicitOutcome {
        result,
        atom_grain,
        site_of,
        occupancy,
        vacancy_sites,
        interstitial_sites,
        antisite_sites,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ptm::PtmAtom;

    fn bcc(n: usize, a: f64) -> (UnitCell, Vec<Coord3>) {
        let mut p = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let b = [i as f64, j as f64, k as f64];
                    p.push([b[0] * a, b[1] * a, b[2] * a]);
                    p.push([(b[0] + 0.5) * a, (b[1] + 0.5) * a, (b[2] + 0.5) * a]);
                }
            }
        }
        let l = n as f64 * a;
        (UnitCell::from_h([l, l, l, 0.0, 0.0, 0.0]), p)
    }

    fn ptm_bcc(n: usize, a: f64) -> Vec<PtmAtom> {
        vec![
            PtmAtom {
                structure: structure::BCC as u8,
                rmsd: 0.0,
                interatomic_distance: (a * 3f64.sqrt() / 2.0) as f32,
                orientation: [1.0, 0.0, 0.0, 0.0],
            };
            n
        ]
    }

    #[test]
    fn frenkel_pairs_and_mapping() {
        let a = 3.165;
        let (cell, sites) = bcc(8, a);
        let ptm = ptm_bcc(sites.len(), a);
        let par = Par::new(2);
        let r =
            ExplicitReference::new(&cell, &sites, &vec![0; sites.len()], &ptm, None, &par).unwrap();
        // Two vacancies, one <110> dumbbell on site 100.
        let mut cur: Vec<Coord3> = sites.clone();
        let c = cur[100];
        let h = 0.35 * a;
        cur[100] = [c[0] + h, c[1] + h, c[2]];
        cur.push([c[0] - h, c[1] - h, c[2]]);
        cur.remove(300);
        cur.remove(7);
        let o = run(&r, &cell, &cur, AffineMapping::Off, &par).unwrap();
        let g = &o.result.grains[0];
        assert_eq!((g.interior.vacancies, g.interior.interstitials), (2, 1));
        assert_eq!(g.interior.residual(), 0);
        assert_eq!(g.boundary, Default::default());
        assert_eq!(o.result.vacancies.len(), 2);
        assert_eq!(o.result.interstitials.len(), 1);

        // Strain the current box by 6 % (with shear): off breaks, mapping fixes.
        let f = [[1.06, 0.0, 0.04], [0.0, 1.0, 0.0], [0.0, 0.0, 0.97]];
        let apply = |x: Coord3| {
            [
                f[0][0] * x[0] + f[0][1] * x[1] + f[0][2] * x[2],
                f[1][0] * x[0] + f[1][1] * x[1] + f[1][2] * x[2],
                f[2][0] * x[0] + f[2][1] * x[1] + f[2][2] * x[2],
            ]
        };
        let l = 8.0 * a;
        let ccur = UnitCell::from_matrix(
            crate::cell::Mat3::from_rows(
                apply([l, 0.0, 0.0]),
                apply([0.0, l, 0.0]),
                apply([0.0, 0.0, l]),
            ),
            [true; 3],
            [0.0; 3],
        );
        let strained: Vec<Coord3> = cur.iter().map(|&x| apply(x)).collect();
        for m in [AffineMapping::ToReference, AffineMapping::ToCurrent] {
            let o = run(&r, &ccur, &strained, m, &par).unwrap();
            let g = &o.result.grains[0];
            assert_eq!(
                (g.interior.vacancies, g.interior.interstitials),
                (2, 1),
                "{m:?}"
            );
            // Vacancy positions are in the current cell.
            let want = apply(sites[7]);
            assert!(o
                .result
                .vacancies
                .iter()
                .any(|v| (0..3).all(|d| (v.position[d] - want[d]).abs() < 1e-9)));
        }
        let o = run(&r, &ccur, &strained, AffineMapping::Off, &par).unwrap();
        assert!(o.result.grains[0].interior.vacancies > 2);
        assert_eq!(o.result.grains[0].interior.residual(), 0);
    }

    #[test]
    fn grains_and_classes_conserve() {
        let a = 3.165;
        let (cell, sites) = bcc(8, a);
        let ptm = ptm_bcc(sites.len(), a);
        let par = Par::new(3);
        let l = 8.0 * a;
        // Two slabs along x; a site between them is boundary.
        let lab: Vec<i32> = sites.iter().map(|p| i32::from(p[0] >= l / 2.0)).collect();
        let r = ExplicitReference::new(&cell, &sites, &lab, &ptm, None, &par).unwrap();
        let mut cur = sites.clone();
        cur.remove(1000);
        cur.push([0.25 * a + 0.1, 0.5 * a, 1.0 * a + 0.05]);
        let o = run(&r, &cell, &cur, AffineMapping::Off, &par).unwrap();
        assert_eq!(o.result.grains.len(), 2);
        let (i, b) = o.result.totals();
        assert_eq!(i.vacancies + b.vacancies, 1);
        assert_eq!(i.interstitials + b.interstitials, 1);
        for g in &o.result.grains {
            assert_eq!(g.interior.residual(), 0);
            assert_eq!(g.boundary.residual(), 0);
            assert!(g.boundary.sites > 0);
        }
        assert_eq!(i.sites + b.sites, sites.len() as u64);
        assert_eq!(i.atoms + b.atoms, cur.len() as u64);
    }

    #[test]
    fn antisites_on_singly_occupied_sites() {
        let a = 2.88;
        let (cell, sites) = bcc(6, a);
        let ptm = ptm_bcc(sites.len(), a);
        let par = Par::new(2);
        let r =
            ExplicitReference::new(&cell, &sites, &vec![0; sites.len()], &ptm, None, &par).unwrap();
        // B2: corner sites type 1, centre sites type 2.
        let site_types: Vec<i32> = (0..sites.len()).map(|s| 1 + (s % 2) as i32).collect();
        let mut cur = sites.clone();
        let mut types = site_types.clone();
        // Swap a pair (two antisites), plant a lone antisite (type 2 on a
        // type-1 site), and a dumbbell of a type-2 atom on a type-1 site
        // (an interstitial, not an antisite).
        types.swap(10, 11);
        types[40] = 2;
        let c = cur[60];
        cur[60] = [c[0] + 0.8, c[1] + 0.8, c[2]];
        cur.push([c[0] - 0.8, c[1] - 0.8, c[2]]);
        types.push(2);
        // A vacancy.
        cur.remove(100);
        types.remove(100);
        let t = Types {
            sites: &site_types,
            atoms: &types,
        };
        let o = run_typed(&r, &cell, &cur, AffineMapping::Off, Some(t), &par).unwrap();
        let g = &o.result.grains[0];
        assert_eq!((g.interior.vacancies, g.interior.interstitials), (1, 1));
        let an = g.antisites.as_ref().unwrap();
        assert_eq!((an.interior, an.boundary), (3, 0));
        assert_eq!(an.pairs[&(1, 2)], [2, 0]);
        assert_eq!(an.pairs[&(2, 1)], [1, 0]);
        let mut s = o.antisite_sites.clone();
        s.sort_unstable();
        assert_eq!(s, vec![10, 11, 40]);
        // Without types, the same V and I and no antisites.
        let p = run(&r, &cell, &cur, AffineMapping::Off, &par).unwrap();
        assert_eq!(p.result.grains[0].interior, g.interior);
        assert!(p.result.grains[0].antisites.is_none());
    }
}
