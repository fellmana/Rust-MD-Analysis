//! `rmda ws --reference` / Python `ws(reference=)` on the explicit-reference
//! core ([`super::explicit`]), as a pipeline [`Analysis`].
//!
//! The assignment is [`super::explicit::run`] with one grain: OVITO's
//! `WignerSeitzAnalysisModifier`, every current atom on its nearest reference
//! site (no tolerance, no search radius, ties to the lower site index). This
//! adapter gives it the `ws` interface — scalars, per-atom properties and the
//! defect-file detail:
//!
//! | name | meaning |
//! |---|---|
//! | `ws_vacancies`, `ws_interstitials`, `ws_interstitial_sites`, `ws_sites` | the census (`Σ max(occ − 1, 0)`; sites with occ > 1; reference atoms) |
//! | `ws_config_mean_displacement`, `ws_config_max_displacement` | over every atom, site to atom |
//! | `ws_config_tolerance` | `--tolerance`, or 1.05 × the reference's median first-neighbour distance |
//! | `ws_config_unmatched` | atoms farther than the tolerance from their site |
//!
//! **The tolerance is a diagnostic.** Every atom is counted, as OVITO does, so
//! `V − I = sites − atoms` holds; `ws_config_unmatched` counts the atoms far
//! from their site, which are still in the census. `--cutoff` (a search
//! radius) has no meaning in pure WS and is ignored. The current and reference
//! frames must have the same periodic flags.

use super::explicit::{self, AffineMapping, ExplicitReference};
use crate::analysis::defect_dump::{DefectDump, SiteGrid};
use crate::analysis::ptm::PtmAtom;
use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::cell::{Coord3, UnitCell};
use crate::data::property::{DType, Property, PropertyStore};
use crate::error::{Error, Result};
use crate::exec::Par;

/// Slack on the measured first-neighbour distance for the default tolerance.
const TOLERANCE_SLACK: f64 = 1.05;

/// Sites sampled for that distance.
const NN_SAMPLE: usize = 4096;

impl From<crate::analysis::ws::AffineMapping> for AffineMapping {
    fn from(m: crate::analysis::ws::AffineMapping) -> Self {
        use crate::analysis::ws::AffineMapping as W;
        match m {
            W::None => AffineMapping::Off,
            W::ToReferenceCell => AffineMapping::ToReference,
            W::ToCurrent => AffineMapping::ToCurrent,
        }
    }
}

/// Wigner–Seitz against a reference frame, as a pipeline analysis.
pub struct ExplicitWs {
    ref_cell: UnitCell,
    ref_positions: Vec<Coord3>,
    affine: AffineMapping,
    tolerance: Option<f64>,
    /// Last run's defect detail, for `take_defects`.
    defects: std::sync::Mutex<Option<DefectDump>>,
}

/// Per-run results.
#[derive(Debug, Clone)]
pub struct ExplicitWsOutcome {
    /// Reference site of each current atom (never negative: pure WS).
    pub site_of_atom: Vec<i64>,
    /// Minimum-image vector from each atom's site to the atom, 3 per atom.
    pub displacement: Vec<f64>,
    /// Occupancy per reference site.
    pub occupancy: Vec<u32>,
    pub vacancy_count: usize,
    /// `Σ max(occ − 1, 0)`.
    pub interstitial_count: usize,
    /// Sites holding more than one atom.
    pub interstitial_site_count: usize,
    /// Atoms farther than `tolerance` from their site (still counted).
    pub unmatched_count: usize,
    pub mean_displacement: f64,
    pub max_displacement: f64,
    pub tolerance: f64,
}

impl ExplicitWsOutcome {
    pub fn defect_count(&self) -> usize {
        self.vacancy_count + self.interstitial_count
    }
}

impl ExplicitWs {
    /// Reference sites are the reference frame's atom positions.
    pub fn new(ref_cell: UnitCell, ref_positions: Vec<Coord3>) -> Self {
        ExplicitWs {
            ref_cell,
            ref_positions,
            affine: AffineMapping::Off,
            tolerance: None,
            defects: std::sync::Mutex::new(None),
        }
    }

    /// `--map` / `affine_map` (either mapping type).
    pub fn with_affine(mut self, m: impl Into<AffineMapping>) -> Self {
        self.affine = m.into();
        self
    }

    /// The distance beyond which an atom is reported in
    /// `ws_config_unmatched`. Does not change the census.
    pub fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = Some(t);
        self
    }

    /// Run against a position set.
    pub fn run(
        &self,
        cell: &UnitCell,
        positions: &[Coord3],
        par: &Par,
    ) -> Result<ExplicitWsOutcome> {
        if let Some(t) = self.tolerance {
            // NaN included: it would pass every comparison silently.
            if !t.is_finite() || t <= 0.0 {
                return Err(Error::invalid("tolerance must be positive and finite"));
            }
        }
        let ns = self.ref_positions.len();
        if ns == 0 {
            return Err(Error::invalid(
                "reference configuration has no sites: nothing to assign against",
            ));
        }
        // One grain, every site interior: the totals are OVITO's. The PTM is
        // only read for a grain's structure and bond, which this report does
        // not use.
        let r = ExplicitReference::new(
            &self.ref_cell,
            &self.ref_positions,
            &vec![0; ns],
            &vec![PtmAtom::default(); ns],
            None,
            par,
        )?;
        let tolerance = match self.tolerance {
            Some(t) => t,
            None => r.median_nn(NN_SAMPLE) * TOLERANCE_SLACK,
        };
        let o = explicit::run(&r, cell, positions, self.affine, par)?;

        let mut vacancy_count = 0usize;
        let mut interstitial_count = 0usize;
        let mut interstitial_site_count = 0usize;
        for &occ in &o.occupancy {
            if occ == 0 {
                vacancy_count += 1;
            } else if occ > 1 {
                interstitial_site_count += 1;
                interstitial_count += (occ - 1) as usize;
            }
        }

        // Site-to-atom vectors in the cell the assignment ran in.
        let ref_inv = self.ref_cell.inverse_transposed()?;
        let cur_inv = cell.inverse_transposed()?;
        let n = positions.len();
        let mut disp3 = vec![[0.0; 3]; n];
        par.fill(&mut disp3, |i, d| {
            let s = self.ref_positions[o.site_of[i] as usize];
            let p = positions[i];
            let v = match self.affine {
                AffineMapping::Off => {
                    self.ref_cell
                        .min_image_with_shift_pre([p[0] - s[0], p[1] - s[1], p[2] - s[2]], &ref_inv)
                        .0
                }
                AffineMapping::ToReference => {
                    let q = self.ref_cell.lamda2x(cell.x2lamda_pre(p, &cur_inv));
                    self.ref_cell
                        .min_image_with_shift_pre([q[0] - s[0], q[1] - s[1], q[2] - s[2]], &ref_inv)
                        .0
                }
                AffineMapping::ToCurrent => {
                    let t = cell.lamda2x(self.ref_cell.x2lamda_pre(s, &ref_inv));
                    cell.min_image_with_shift_pre([p[0] - t[0], p[1] - t[1], p[2] - t[2]], &cur_inv)
                        .0
                }
            };
            *d = v;
        });
        let displacement: Vec<f64> = disp3.into_iter().flatten().collect();
        let mut sum_d = 0.0f64;
        let mut max_d = 0.0f64;
        let mut unmatched_count = 0usize;
        for d in displacement.chunks_exact(3) {
            let m = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            sum_d += m;
            max_d = max_d.max(m);
            if m > tolerance {
                unmatched_count += 1;
            }
        }
        Ok(ExplicitWsOutcome {
            site_of_atom: o.site_of.iter().map(|&s| i64::from(s)).collect(),
            displacement,
            occupancy: o.occupancy,
            vacancy_count,
            interstitial_count,
            interstitial_site_count,
            unmatched_count,
            mean_displacement: if n == 0 { 0.0 } else { sum_d / n as f64 },
            max_displacement: max_d,
            tolerance,
        })
    }
}

impl Analysis for ExplicitWs {
    fn name(&self) -> &'static str {
        "ws-config"
    }

    fn take_defects(&self) -> Option<DefectDump> {
        self.defects.lock().expect("ws defect stash").take()
    }

    /// Indexes the reference frame's sites, not the current atoms, so the
    /// pipeline's shared list is of no use.
    fn cutoffs(&self) -> Vec<f64> {
        Vec::new()
    }

    fn run(
        &self,
        ctx: &AnalysisContext<'_>,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        let cell = &ctx.frame.cell;
        let n = ctx.frame.natoms();
        let res = self.run(cell, &ctx.frame.particles.positions[..n], ctx.par)?;

        let mut sp = Property::with_capacity("ws_site".to_owned(), DType::I64, true, 1, n);
        for v in &res.site_of_atom {
            sp.push(*v as f64);
        }
        out.register(sp);
        out.register_flat(DType::F64, "ws_displacement", 3, res.displacement.clone());
        // An atom on an over-occupied site, as the lattice engines flag it.
        let mut flag =
            Property::with_capacity("ws_interstitial".to_owned(), DType::I32, true, 1, n);
        for &s in &res.site_of_atom {
            flag.push(if res.occupancy[s as usize] > 1 {
                1.0
            } else {
                0.0
            });
        }
        out.register(flag);
        // The sites as the defect files see them: the reference positions in
        // the reference cell, or mapped into the current cell under ToCurrent.
        let (dump_cell, dump_sites) = if self.affine == AffineMapping::ToCurrent {
            let inv = self.ref_cell.inverse_transposed()?;
            let v = self
                .ref_positions
                .iter()
                .map(|&s| cell.lamda2x(self.ref_cell.x2lamda_pre(s, &inv)))
                .collect();
            (*cell, v)
        } else {
            (self.ref_cell, self.ref_positions.clone())
        };
        *self.defects.lock().expect("ws defect stash") = Some(DefectDump {
            cell: dump_cell,
            occupancy: res.occupancy.clone(),
            sites: SiteGrid::Positions(dump_sites),
            site_of_atom: res.site_of_atom.clone(),
            void_of_site: None,
        });

        let mut p = Property::with_capacity(
            "ws_occupancy".to_owned(),
            DType::I32,
            false,
            1,
            res.occupancy.len(),
        );
        for o in &res.occupancy {
            p.push(*o as f64);
        }
        out.register(p);

        summary.merge_from(Summary {
            scalars: vec![
                ("ws_vacancies".to_owned(), res.vacancy_count as f64),
                ("ws_interstitials".to_owned(), res.interstitial_count as f64),
                (
                    "ws_interstitial_sites".to_owned(),
                    res.interstitial_site_count as f64,
                ),
                ("ws_sites".to_owned(), self.ref_positions.len() as f64),
                // Named apart from the lattice engines' `ws_mean_displacement`:
                // against a configuration this is thermal displacement.
                (
                    "ws_config_mean_displacement".to_owned(),
                    res.mean_displacement,
                ),
                (
                    "ws_config_max_displacement".to_owned(),
                    res.max_displacement,
                ),
                ("ws_config_unmatched".to_owned(), res.unmatched_count as f64),
                ("ws_config_tolerance".to_owned(), res.tolerance),
            ],
            histograms: vec![],
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::fcc::Fcc;
    use crate::lattice::Lattice;

    fn block(a: f64, cells: usize) -> (UnitCell, Vec<Coord3>) {
        let len = cells as f64 * a;
        let cell = UnitCell::orthorhombic(len, len, len);
        let proto = Fcc { a }.primitive();
        let mut pos = Vec::new();
        for i in 0..cells {
            for j in 0..cells {
                for k in 0..cells {
                    for &p in &proto.basis {
                        pos.push([
                            (i as f64 + p[0]) * a,
                            (j as f64 + p[1]) * a,
                            (k as f64 + p[2]) * a,
                        ]);
                    }
                }
            }
        }
        (cell, pos)
    }

    #[test]
    fn a_frame_against_itself_is_clean() {
        let (cell, pos) = block(3.6, 5);
        let r = ExplicitWs::new(cell, pos.clone())
            .run(&cell, &pos, &Par::new(2))
            .unwrap();
        assert_eq!(r.defect_count(), 0);
        assert_eq!(r.max_displacement, 0.0);
        assert_eq!(r.unmatched_count, 0);
        assert!((r.tolerance / (3.6 / 2f64.sqrt()) - 1.05).abs() < 1e-9);
    }

    #[test]
    fn a_far_atom_is_counted_and_reported() {
        // An atom pushed 0.6 bonds off its site still sits nearest to it; with
        // a 0.5 A tolerance it is reported, and the census does not change.
        let (cell, pos) = block(3.6, 5);
        let mut cur = pos.clone();
        cur[7][0] += 1.0;
        let r = ExplicitWs::new(cell, pos)
            .with_tolerance(0.5)
            .run(&cell, &cur, &Par::new(1))
            .unwrap();
        assert_eq!(r.defect_count(), 0);
        assert_eq!(r.unmatched_count, 1);
        assert!((r.max_displacement - 1.0).abs() < 1e-12);
        assert!((r.displacement[21] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_displaced_atom_is_a_vacancy_and_an_interstitial() {
        let (cell, pos) = block(3.6, 5);
        let mut cur = pos.clone();
        let nn = 3.6 / 2f64.sqrt();
        cur[0][0] += nn / 2f64.sqrt();
        cur[0][1] += nn / 2f64.sqrt();
        let r = ExplicitWs::new(cell, pos)
            .run(&cell, &cur, &Par::new(1))
            .unwrap();
        assert_eq!(r.vacancy_count, 1);
        assert_eq!(r.interstitial_count, 1);
        assert_eq!(r.interstitial_site_count, 1);
    }

    #[test]
    fn to_reference_maps_a_strained_frame_back() {
        let (cell, pos) = block(3.6, 4);
        let s = 1.03;
        let big = UnitCell::orthorhombic(4.0 * 3.6 * s, 4.0 * 3.6 * s, 4.0 * 3.6 * s);
        let cur: Vec<Coord3> = pos.iter().map(|p| [p[0] * s, p[1] * s, p[2] * s]).collect();
        let r = ExplicitWs::new(cell, pos)
            .with_affine(crate::analysis::ws::AffineMapping::ToReferenceCell)
            .run(&big, &cur, &Par::new(1))
            .unwrap();
        assert_eq!(r.defect_count(), 0);
        assert!(r.max_displacement < 1e-9);
    }
}
