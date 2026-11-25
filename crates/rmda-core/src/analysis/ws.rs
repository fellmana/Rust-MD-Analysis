//! Wigner–Seitz analysis against a reference lattice (`rmda ws`; `--self` fits it).
//!
//! Point defects are defined *relative to a known perfect lattice*: build the
//! reference site set, assign every current atom to its nearest site, and call a
//! site with no atom a vacancy and a site with more than one an interstitial.
//!
//! # Why this needs no neighbour search
//!
//! A tree over the reference sites would work, but for a regular lattice the
//! nearest site is available by arithmetic: see
//! [`SiteLattice`](crate::lattice::sites::SiteLattice), which rounds into each
//! basis atom's own cell instead of descending a tree. Two consequences:
//!
//! - `cutoffs()` is empty, so the pipeline builds **no cell list at all** for this
//!   analysis. That is a genuine saving at scale, not an optimisation detail.
//! - Site lookup is exact, not approximate, so occupancy cannot depend on
//!   neighbour-search edge cases.
//!
//! # Determinism
//!
//! Each atom's site is a pure function of its position, so the mapping is
//! lock-free and identical for any thread count. Several atoms may land on one
//! site, so the occupancy counter is atomic — but because the counter is only
//! ever incremented (never read mid-flight to decide anything), the final counts
//! cannot depend on scheduling.

use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::cell::{Coord3, UnitCell};
use crate::data::property::{DType, PropertyStore};
use crate::error::{Error, Result};
use crate::lattice::fit::{
    assign_region, fit_region, nn_distance, Candidate, FitOptions, FitScore, PerAtom, RegionFit,
    SiteAssignment,
};
use crate::lattice::sites::{SiteAssign, SiteLattice};
use crate::lattice::Lattice;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::PoisonError;

/// Tolerance on "the box holds a whole number of conventional cells".
///
/// Tight on purpose: the failure mode is silent and large, so the check is worth
/// being annoying about. A user who hits it should be told to use the self-built
/// engine rather than nudged into adding digits until it passes.
pub const COMMENSURATE_TOL: f64 = 1e-6;

/// OVITO's `affine_mapping` option.
///
/// OVITO supports `None` and `ToReferenceCell` and *errors* on `ToCurrent`; we
/// match that rather than silently doing something different, because the choice
/// changes which cell the reference sites live in and therefore every count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AffineMapping {
    /// Use positions as they are. Correct when the reference and current frames
    /// share a cell.
    #[default]
    None,
    /// Map current positions into the reference cell before assigning.
    ToReferenceCell,
    /// Unsupported, as in OVITO. Present so the CLI can report it clearly.
    ToCurrent,
}

/// Wigner–Seitz analysis against a reference lattice.
pub struct WsRef {
    lattice: Box<dyn Lattice>,
    affine: AffineMapping,
    /// Cartesian position of the reference cell's (0,0,0) corner.
    origin: Coord3,
    /// Reference cell, required for [`AffineMapping::ToReferenceCell`].
    ref_cell: Option<UnitCell>,
    /// Last run's defect detail, for `take_defects`.
    defects: std::sync::Mutex<Option<crate::analysis::defect_dump::DefectDump>>,
}

/// Register `ws_interstitial` and build the [`DefectDump`] for one run.
///
/// The flag is per *atom* and the dump is per *site*, and both come out of the same
/// pass: an atom is an interstitial exactly when its site holds more than one atom,
/// which is the same predicate that decides a site is over-occupied. Doing it twice
/// would be two chances to disagree.
fn interstitial_flag_and_dump(
    cell: &UnitCell,
    occupancy: &[u32],
    site_of_atom: &[i64],
    sites: crate::analysis::defect_dump::SiteGrid,
    voids: Option<&crate::analysis::void::VoidCounts>,
) -> (
    crate::data::property::Property,
    crate::analysis::defect_dump::DefectDump,
) {
    let mut flag = crate::data::property::Property::with_capacity(
        "ws_interstitial".to_owned(),
        DType::I32,
        true,
        1,
        site_of_atom.len(),
    );
    for &s in site_of_atom {
        // A negative slot is an atom past tolerance or outside the region. It is
        // off-lattice, which is a displacement problem, not an interstitial: calling
        // it one would double-count it against a site that may hold exactly one atom.
        let v = if s >= 0 && occupancy[s as usize] > 1 {
            1.0
        } else {
            0.0
        };
        flag.push(v);
    }
    let mut dump = crate::analysis::defect_dump::DefectDump {
        cell: *cell,
        occupancy: occupancy.to_vec(),
        sites,
        site_of_atom: site_of_atom.to_vec(),
        void_of_site: None,
    };
    if let Some(v) = voids {
        dump = dump.with_voids(v);
    }
    (flag, dump)
}

impl WsRef {
    pub fn new(lattice: Box<dyn Lattice>) -> Self {
        WsRef {
            lattice,
            affine: AffineMapping::default(),
            origin: [0.0, 0.0, 0.0],
            ref_cell: None,
            defects: std::sync::Mutex::new(None),
        }
    }

    pub fn with_affine(mut self, m: AffineMapping) -> Self {
        self.affine = m;
        self
    }

    pub fn with_origin(mut self, o: Coord3) -> Self {
        self.origin = o;
        self
    }

    pub fn with_reference_cell(mut self, c: UnitCell) -> Self {
        self.ref_cell = Some(c);
        self
    }

    /// The cell the reference sites are built in, or an error if the requested
    /// mapping does not supply one.
    fn reference_cell<'a>(&'a self, current: &'a UnitCell) -> Result<&'a UnitCell> {
        match self.affine {
            AffineMapping::None | AffineMapping::ToReferenceCell => {
                Ok(self.ref_cell.as_ref().unwrap_or(current))
            }
            // OVITO errors here, and a silently wrong cell would misreport
            // every defect count in the run.
            AffineMapping::ToCurrent => Err(Error::invalid(
                "affine mapping ToCurrent is not supported (OVITO errors on it too)",
            )),
        }
    }
}

/// Per-run results, kept separate from the property plumbing so tests and future
/// callers (the self-built path, void detection) can use them directly.
#[derive(Debug, Clone)]
pub struct WsOutcome {
    /// Reference site index per atom.
    pub site_of_atom: Vec<i64>,
    /// Minimum-image vector from each atom's site to the atom, 3 per atom.
    pub displacement: Vec<f64>,
    /// Occupancy per reference site.
    pub occupancy: Vec<u32>,
    /// The reference site grid those occupancies index into.
    ///
    /// Lazy rather than materialised: it is O(1) to keep and answers a position
    /// query per index, so a run that only wanted the counts pays nothing, and a
    /// caller writing vacancy sites out gets positions without a second lattice
    /// being built and quietly disagreeing with the first.
    pub sites: SiteLattice,
    pub vacancy_count: usize,
    /// Sum of `occupancy - 1` over over-occupied sites.
    pub interstitial_count: usize,
    /// Sites holding more than one atom, as distinct from `interstitial_count`
    /// (which counts the *extra* atoms).
    pub interstitial_site_count: usize,
    pub mean_displacement: f64,
    pub max_displacement: f64,
}

impl WsOutcome {
    /// Vacancies and interstitial extra-atoms, the headline numbers.
    pub fn defect_count(&self) -> usize {
        self.vacancy_count + self.interstitial_count
    }
}

impl WsRef {
    /// Run against an arbitrary position set, in the reference cell.
    ///
    /// Separate from the [`Analysis`] path so other callers can reuse the counting
    /// half with a site lattice they built themselves.
    pub fn run_positions(
        &self,
        cell: &UnitCell,
        positions: &[Coord3],
        par: &crate::exec::Par,
    ) -> Result<WsOutcome> {
        let ref_cell = self.reference_cell(cell)?;
        let sites = SiteLattice::build(ref_cell, &self.lattice.primitive(), self.origin)?;
        // A non-commensurate box makes `dims` a ceil, so the extra slab of sites
        // lies outside the crystal and every one of them reports as a vacancy.
        // On a 6x6x6 FCC block that is 508 phantom vacancies from a lattice
        // constant given to 5 significant figures. Silently producing those would
        // be worse than refusing: the incommensurate case is what the self-built
        // engine ([`WsSelf`]) exists to solve, by fitting the reference to the box
        // instead of assuming it tiles.
        if !sites.is_commensurate(COMMENSURATE_TOL) {
            let spans = sites.cell_spans();
            return Err(Error::invalid(format!(
                "reference lattice does not tile the box: spans {:.6}, {:.6}, {:.6} cells \
                 (lattice '{}'). Vacancy counts would include the sites outside the box. \
                 Supply a lattice constant giving whole cells, or use --self, which \
                 fits the reference to the data instead of assuming it tiles.",
                spans[0],
                spans[1],
                spans[2],
                self.lattice.name()
            )));
        }
        let assign = SiteAssign::new(ref_cell, &sites)?;

        // The mapping is deterministic; only this counter is shared.
        let occupancy: Vec<AtomicU32> = (0..sites.len()).map(|_| AtomicU32::new(0)).collect();

        let _t = crate::prof::StageTimer::start("ws.assign");
        let mapped = if self.affine == AffineMapping::ToReferenceCell && self.ref_cell.is_some() {
            // Carry positions into the reference cell through fractional space,
            // which is the only safe route for a triclinic pair.
            let lam: Vec<Coord3> = positions
                .iter()
                .map(|&p| cell.x2lamda(p))
                .collect::<Result<Vec<_>>>()?;
            lam.iter().map(|&l| ref_cell.lamda2x(l)).collect::<Vec<_>>()
        } else {
            positions.to_vec()
        };

        let results: Vec<(usize, Coord3)> =
            par.map_collect::<_, crate::error::Error, _>(positions.len(), |i| {
                let a = assign.assign(mapped[i]);
                occupancy[a.site].fetch_add(1, Ordering::Relaxed);
                Ok((a.site, a.displacement))
            })?;

        // Deterministic readback: Relaxed is sound here because the increments all
        // happen-before the joins that precede this loop.
        let occ: Vec<u32> = occupancy
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();

        let mut vacancy_count = 0usize;
        let mut interstitial_count = 0usize;
        let mut interstitial_site_count = 0usize;
        for &o in &occ {
            if o == 0 {
                vacancy_count += 1;
            } else if o > 1 {
                interstitial_site_count += 1;
                interstitial_count += (o - 1) as usize;
            }
        }

        let n_atoms = results.len();
        let mut site_of_atom = Vec::with_capacity(n_atoms);
        let mut displacement = Vec::with_capacity(n_atoms * 3);
        let mut sum_d = 0.0f64;
        let mut max_d = 0.0f64;
        for (site, d) in results {
            site_of_atom.push(site as i64);
            displacement.extend_from_slice(&d);
            let m = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            sum_d += m;
            if m > max_d {
                max_d = m;
            }
        }
        let mean_displacement = if n_atoms == 0 {
            0.0
        } else {
            sum_d / n_atoms as f64
        };

        Ok(WsOutcome {
            site_of_atom,
            displacement,
            occupancy: occ,
            sites,
            vacancy_count,
            interstitial_count,
            interstitial_site_count,
            mean_displacement,
            max_displacement: max_d,
        })
    }
}

impl Analysis for WsRef {
    fn name(&self) -> &'static str {
        "ws-ref"
    }

    fn take_defects(&self) -> Option<crate::analysis::defect_dump::DefectDump> {
        self.defects
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Empty on purpose: the lattice-index assignment needs no neighbours, so the
    /// pipeline must not build a cell list for this analysis.
    fn cutoffs(&self) -> Vec<f64> {
        Vec::new()
    }

    fn run(
        &self,
        ctx: &AnalysisContext,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        let cell = &ctx.frame.cell;
        let n = ctx.frame.natoms();
        let sites = SiteLattice::build(
            self.reference_cell(cell)?,
            &self.lattice.primitive(),
            self.origin,
        )?;
        let res = self.run_positions(cell, &ctx.frame.particles.positions[..n], ctx.par)?;
        debug_assert_eq!(res.site_of_atom.len(), n);

        // Per-atom outputs: which site each atom claimed, and how far it
        // sits from it.
        // `register_scalar` needs `Into<f64>`, which i64 does not implement, and
        // site indices must stay exact — so build the I64 property by hand.
        let mut sp = crate::data::property::Property::with_capacity(
            "ws_site".to_owned(),
            DType::I64,
            true,
            1,
            res.site_of_atom.len(),
        );
        for v in &res.site_of_atom {
            sp.push(*v as f64);
        }
        out.register(sp);
        out.register_flat(DType::F64, "ws_displacement", 3, res.displacement.clone());

        let (flag, dump) = interstitial_flag_and_dump(
            self.reference_cell(cell)?,
            &res.occupancy,
            &res.site_of_atom,
            crate::analysis::defect_dump::SiteGrid::Lattice(res.sites.clone()),
            None,
        );
        out.register(flag);
        *self.defects.lock().unwrap_or_else(PoisonError::into_inner) = Some(dump);

        // Per-*site* output. `per_particle = false` keeps this out of the per-atom
        // property stream; writers filter on that flag, so it cannot corrupt an
        // extXYZ file.
        let mut p = crate::data::property::Property::with_capacity(
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
                ("ws_sites".to_owned(), sites.len() as f64),
                ("ws_mean_displacement".to_owned(), res.mean_displacement),
                ("ws_max_displacement".to_owned(), res.max_displacement),
            ],
            histograms: vec![],
        });
        Ok(())
    }
}

/// Wigner–Seitz analysis against a reference fitted to the data (`rmda ws --self`).
///
/// [`WsRef`] needs a reference that is already known exactly, and refuses a box
/// the reference does not tile rather than report the sites outside it as
/// vacancies. That is right for a single crystal whose reference you already
/// hold, and it is exactly the requirement a polycrystal, an interface, or a
/// tilted grain cannot meet: no single reference describes any of them, and a box
/// that fails to tile is the normal case rather than a user mistake.
///
/// This fits the reference to the atoms of the region
/// ([`fit_region`](crate::lattice::fit::fit_region)) and then counts using the
/// same assignment the fit itself converged on, so the numbers reported are the
/// ones the search arrived at rather than a second opinion from a different
/// engine.
///
/// # What the caller still has to supply
///
/// The *prototype* and an orientation seed, and neither can be derived:
///
/// - The prototype is the lattice type -- FCC, BCC, HCP. That is chemistry, not
///   geometry, and this fits a known type to a region rather than identifying it.
/// - The seed is a PTM quaternion. There is deliberately no global orientation
///   search (see the `lattice::fit` module docs for the measurements behind that),
///   so a region more than a few degrees off the identity will not be recovered.
///   `mean_displacement` is how a caller notices that rather than believing a
///   confident zero-defect count; `unseeded` below reports whether a seed was
///   used at all, because an unseeded fit succeeding on a tilted region is luck.
pub struct WsSelf {
    lattice: Box<dyn Lattice>,
    options: FitOptions,
    tolerance: Option<f64>,
    /// Smallest empty-site cluster reported as a void (design §17.1).
    ///
    /// Not a defect threshold, and unrelated to `tolerance`: that decides whether an
    /// atom sits *on* a site, this decides whether a run of empty sites is a *void*.
    void_min_sites: usize,
    /// Last run's defect detail, for `take_defects`.
    defects: std::sync::Mutex<Option<crate::analysis::defect_dump::DefectDump>>,
}

/// Result of a self-built Wigner–Seitz analysis over one region.
#[derive(Debug, Clone)]
pub struct WsSelfOutcome {
    /// The fitted reference and the score it achieved.
    pub fit: RegionFit,
    /// The defect tally this run reports.
    ///
    /// Equal to `fit.score` for a single-set run. A
    /// [split](WsSelf::run_region_split) run reports a tally restricted to the
    /// counted atoms, so this and `fit.score` are two different questions and
    /// reading the wrong one is how a seam exclusion turns into a defect count.
    pub tally: FitScore,
    /// Occupancy per in-region site.
    pub occupancy: Vec<u32>,
    /// Site index per atom, `-1` where the atom has no lattice point within
    /// tolerance and `-2` where it has one outside the region.
    pub site_of_atom: Vec<i64>,
    /// Atom-minus-site vector, three components per atom.
    pub displacement: Vec<f64>,
    /// Largest `|displacement|`.
    pub max_displacement: f64,
    /// The full assignment, for anything that needs the site grid itself.
    ///
    /// Void detection needs the empty-site *indices*, which is a different question
    /// from the vacant-site count, so it is answered from here rather than by
    /// recomputing the index arithmetic and hoping it matches.
    pub assign: SiteAssignment,
}

impl WsSelfOutcome {
    /// Count void clusters: connected runs of empty sites (design §17.1, method A).
    ///
    /// `min_sites` is the smallest cluster to call a void. A value of 1 counts every
    /// empty site, which is a vacancy tally under another name.
    ///
    /// Adjacency is the fitted lattice's nearest-neighbour distance. That is a property
    /// of the fit rather than of the caller, so it is not a parameter here;
    /// [`Self::count_voids_with_cutoff`] is for a different criterion.
    pub fn count_voids(&self, min_sites: usize) -> Result<crate::analysis::void::VoidCounts> {
        self.count_voids_with_cutoff(min_sites, None)
    }

    /// [`Self::count_voids`] with an explicit adjacency cutoff, in Å.
    ///
    /// `None` uses the fitted nearest-neighbour distance. A supplied value must be
    /// large enough to include a bond or the cluster count comes out wrong in a way
    /// that looks like a physical result.
    pub fn count_voids_with_cutoff(
        &self,
        min_sites: usize,
        cutoff: Option<f64>,
    ) -> Result<crate::analysis::void::VoidCounts> {
        let cell = &self.fit.cell;
        let cutoff = match cutoff {
            Some(c) => c,
            None => crate::analysis::void::default_cutoff(cell)?,
        };
        crate::analysis::void::count_voids(cell, self.fit.origin, &self.assign, min_sites, cutoff)
    }
}

impl WsSelfOutcome {
    pub fn vacancy_count(&self) -> usize {
        self.tally.vacancies
    }
    pub fn interstitial_count(&self) -> usize {
        self.tally.interstitials
    }
    pub fn interstitial_site_count(&self) -> usize {
        self.tally.interstitial_sites
    }
    /// Sites the region holds. Greater than the atom count means the reference is
    /// larger than the atoms need, which is what void detection wants: an empty site
    /// is a candidate crater, and the count is only a lower bound on those because a
    /// real crater is several adjacent empty cells.
    pub fn n_sites(&self) -> usize {
        self.tally.n_sites
    }
    pub fn defect_count(&self) -> usize {
        self.tally.defects()
    }
    /// Mean atom-to-site distance over the *fitted* set.
    ///
    /// Read from `fit.score`, not `tally`: this measures how well the reference
    /// describes the crystal it was fitted to, so restricting it to a counted
    /// subset would report the quality of a fit nobody made.
    pub fn mean_displacement(&self) -> f64 {
        self.fit.score.mean_displacement
    }
}

impl WsSelf {
    pub fn new(lattice: Box<dyn Lattice>) -> Self {
        WsSelf {
            lattice,
            options: FitOptions::default(),
            tolerance: None,
            void_min_sites: 1,
            defects: std::sync::Mutex::new(None),
        }
    }

    /// Smallest cluster of empty sites to report as a void.
    ///
    /// One by default, which counts every empty site and so is a vacancy tally wearing
    /// a different name. A real crater is several adjacent empty cells, so a threshold
    /// above one is how a caller asks for craters rather than vacancies -- and it is a
    /// parameter rather than a constant because what counts as a void depends on the
    /// material's thermal broadening, which this engine does not model.
    pub fn with_void_min_sites(mut self, n: usize) -> Self {
        self.void_min_sites = n.max(1);
        self
    }

    pub fn with_options(mut self, options: FitOptions) -> Self {
        self.options = options;
        self
    }

    /// Override the site-matching tolerance.
    ///
    /// Default is half the *fitted* nearest-neighbour distance, which scales with
    /// the material rather than being a number the user has to know.
    pub fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = Some(t);
        self
    }

    /// Whether no orientation seed was supplied, i.e. whether the fit was local.
    ///
    /// Named from the condition rather than the outcome so a caller cannot read
    /// `unseeded() == true` as "seeding was done".
    pub fn unseeded(&self) -> bool {
        self.options.seed_orientation.is_none()
    }

    /// Fit a reference to `positions` and count defects against it.
    pub fn run_region(
        &self,
        cell: &UnitCell,
        positions: &[Coord3],
        par: &crate::exec::Par,
    ) -> Result<WsSelfOutcome> {
        let out = self.run_region_split(cell, positions, positions, None, par)?;
        // The fit already scored this reference over every atom. Scoring it again
        // through the assignment path must not change a single count or a single bit
        // of the mean, or one of the two is lying. The check lives here rather than in
        // `run_region_split` because only here are the fit set, the assigned set and
        // the counted set the same list by construction -- in a split run they
        // deliberately differ, and asserting there would fail on the feature's whole
        // reason for existing.
        debug_assert_eq!(out.assign.score, out.fit.score);
        Ok(out)
    }

    /// Fit a reference to one set of atoms and tally a possibly different set.
    ///
    /// The two sets exist because *excluding* atoms is the opposite of what a
    /// vacancy count needs. Occupancy is a property of a site, not of an atom, so
    /// dropping an atom from `assign_positions` does not remove it from the
    /// crystal -- it leaves an empty site behind, and the site then reads as a
    /// vacancy. Measured on a defect-free 10x10x10-cell FCC block: remove one
    /// plane of 400 atoms and the crystal reports exactly 400 vacancies, one per
    /// atom removed. So a seam exclusion written the obvious way manufactures a
    /// vacancy per excluded atom and reports the boundary count back to the user
    /// wearing a defect label.
    ///
    /// The split fixes it without pretending the seam is crystalline. Fit on
    /// `fit_positions` -- the ones whose disorder must not set the reference --
    /// then assign `assign_positions`, so a seam atom still *occupies* its site and
    /// that site is not reported empty, while `counted` decides which atoms are
    /// allowed to contribute an interstitial. Vacancy and interstitial tallies
    /// then disagree about the seam on purpose: the seam cannot create a vacancy,
    /// because it is there, and it cannot create an interstitial either, because it
    /// belongs to the boundary bin rather than to the grain interior.
    ///
    /// `counted` is indexed like `assign_positions`; `None` counts every atom,
    /// which makes this exactly [`run_region`](Self::run_region).
    pub fn run_region_split(
        &self,
        cell: &UnitCell,
        fit_positions: &[Coord3],
        assign_positions: &[Coord3],
        counted: Option<&[bool]>,
        par: &crate::exec::Par,
    ) -> Result<WsSelfOutcome> {
        if let Some(c) = counted {
            if c.len() != assign_positions.len() {
                return Err(Error::invalid(format!(
                    "counted mask has {} entries for {} assigned atoms",
                    c.len(),
                    assign_positions.len()
                )));
            }
        }
        // Only a starting guess for the neighbour search that measures the real
        // spacing; the fitted scale comes from the atoms, not from here.
        let nn = nn_distance(&self.lattice.primitive())?;
        let mut opts = self.options.clone();
        if let Some(t) = self.tolerance {
            opts.tolerance = Some(t);
        }
        let fit = fit_region(cell, fit_positions, self.lattice.as_ref(), nn, &opts)?;

        let cand = Candidate {
            pc: fit.cell.clone(),
            origin: fit.origin,
        };
        let mut per_atom = PerAtom::default();
        let assign = assign_region(
            cell,
            &cand,
            assign_positions,
            fit.tolerance,
            Some(par),
            Some(&mut per_atom),
        )?;

        let tally = match counted {
            None => assign.score,
            Some(mask) => restricted_tally(&assign, &per_atom.site_of_atom, mask),
        };
        let assign_ref = assign.clone();

        let max_displacement = per_atom
            .displacement
            .chunks_exact(3)
            .map(|c| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt())
            .fold(0.0f64, f64::max);

        Ok(WsSelfOutcome {
            fit,
            tally,
            assign,
            occupancy: assign_ref.occupancy,
            site_of_atom: per_atom.site_of_atom,
            displacement: per_atom.displacement,
            max_displacement,
        })
    }
}

/// Tally over the `mask`ed atoms only, keeping the assignment's vacancy count.
///
/// Vacancies are left alone deliberately. They come from the occupancy array, and
/// the whole point of the split is that excluded atoms stay *in* that array: an
/// excluded atom covers its site, so its site is not vacant. Filtering occupancy
/// would reintroduce the one-per-atom vacancy this exists to prevent.
///
/// Interstitials are rebuilt from the per-atom slots rather than read off
/// `assign.score`, because the engine's own figure is `sum(max(0, occ-1))` over
/// all atoms plus the off-lattice count, and both terms can be seam artefacts: a
/// disordered seam atom landing on a good site pushes it over one atom per site,
/// and a seam atom off its site is off-lattice by definition. Neither is a defect
/// of the grain interior.
fn restricted_tally(assign: &SiteAssignment, slot_of_atom: &[i64], mask: &[bool]) -> FitScore {
    let mut off_lattice = 0usize;
    let mut per_slot: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for (i, &counted) in mask.iter().enumerate() {
        if !counted {
            continue;
        }
        match slot_of_atom[i] {
            // `-2`: on a lattice point outside the region. Not a defect (the
            // region's edge, not the crystal's), so it counts for nothing here
            // either -- and it consumed no site, so no vacancy is owed.
            s if s >= 0 => *per_slot.entry(s).or_insert(0) += 1,
            _ => {}
        }
        if slot_of_atom[i] == -1 {
            off_lattice += 1;
        }
    }
    let interstitial_sites = per_slot.values().filter(|&&c| c > 1).count();
    let interstitials: usize = per_slot.values().map(|&c| c.saturating_sub(1)).sum();
    FitScore {
        vacancies: assign.score.vacancies,
        interstitials: interstitials + off_lattice,
        interstitial_sites,
        n_sites: assign.score.n_sites,
        // Atoms *scored*, not atoms tallied: this is the reference-quality
        // figure, and the reference was fitted over the fit set.
        n_atoms: assign.score.n_atoms,
        mean_displacement: assign.score.mean_displacement,
    }
}

impl Analysis for WsSelf {
    fn name(&self) -> &'static str {
        "ws-self"
    }

    fn take_defects(&self) -> Option<crate::analysis::defect_dump::DefectDump> {
        self.defects
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Empty for the same reason as [`WsRef`]: the fit reads positions directly
    /// and the assignment rounds into each basis atom's own cell.
    fn cutoffs(&self) -> Vec<f64> {
        Vec::new()
    }

    fn run(
        &self,
        ctx: &AnalysisContext,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        let cell = &ctx.frame.cell;
        let n = ctx.frame.natoms();
        let res = self.run_region(cell, &ctx.frame.particles.positions[..n], ctx.par)?;
        let voids = res.count_voids(self.void_min_sites)?;

        let mut sp = crate::data::property::Property::with_capacity(
            "ws_site".to_owned(),
            DType::I64,
            true,
            1,
            res.site_of_atom.len(),
        );
        for v in &res.site_of_atom {
            sp.push(*v as f64);
        }
        out.register(sp);
        out.register_flat(DType::F64, "ws_displacement", 3, res.displacement.clone());

        // The input cell, not the fitted primitive cell: the region was fitted to
        // atoms inside this box and the site positions are absolute, so the box the
        // user handed us is the one they want the vacancy file to overlay on.
        let (flag, dump) = interstitial_flag_and_dump(
            cell,
            &res.occupancy,
            &res.site_of_atom,
            crate::analysis::defect_dump::SiteGrid::Region(res.assign.sites.clone()),
            Some(&voids),
        );
        out.register(flag);
        *self.defects.lock().unwrap_or_else(PoisonError::into_inner) = Some(dump);

        let mut p = crate::data::property::Property::with_capacity(
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

        // Scalar names match `ws-ref` exactly so a writer or script does not care
        // which engine produced them. The two `ws_self_*` scalars are added because a
        // fit with no orientation seed is only meaningful near the identity, and that
        // is the single easiest way to get a wrong answer out of this.
        summary.merge_from(Summary {
            scalars: vec![
                ("ws_vacancies".to_owned(), res.vacancy_count() as f64),
                (
                    "ws_interstitials".to_owned(),
                    res.interstitial_count() as f64,
                ),
                (
                    "ws_interstitial_sites".to_owned(),
                    res.interstitial_site_count() as f64,
                ),
                ("ws_sites".to_owned(), res.n_sites() as f64),
                ("ws_mean_displacement".to_owned(), res.mean_displacement()),
                ("ws_max_displacement".to_owned(), res.max_displacement),
                // 1 seeded, 0 unseeded. Reported rather than inferred, because a fit
                // with no seed starts from the identity and the orientation descent
                // cannot find a grain turned away from it -- there is a test pinning
                // that as an honest failure. Reading a bad displacement without
                // knowing this is how a wrong answer looks like a good one.
                (
                    "ws_self_seeded".to_owned(),
                    if self.unseeded() { 0.0 } else { 1.0 },
                ),
                // The fitted lattice constant and orientation, so the run can be
                // checked rather than trusted: the point of a self-fit is that its
                // answer came from the data, and the data it came from is the a it
                // settled on. `col_norm(0)`: `row(0)[0]` turns into `a·R₁₁` on a
                // rotated grain, and `row_norm(0)` reads 1.12·a on
                // the hexagonal cell, whose lattice vectors are columns — column 0
                // is (a,0,0), row 0 is (a, a/2, 0).
                ("ws_fit_a".to_owned(), res.fit.cell.h.col_norm(0)),
                ("ws_fit_tolerance".to_owned(), res.fit.tolerance),
                // Void clusters, which are *not* the vacancy count: one crater of 50
                // empty cells is one void and fifty vacancies, and the design's method
                // A is the one that groups them.
                ("void_count".to_owned(), voids.n_voids() as f64),
                ("void_sites".to_owned(), voids.n_void_sites() as f64),
                (
                    "void_isolated_sites".to_owned(),
                    voids.n_isolated_sites as f64,
                ),
                ("void_max_equivalent_radius".to_owned(), voids.max_radius()),
                ("void_min_sites".to_owned(), self.void_min_sites as f64),
                // The adjacency cutoff, because a void count means nothing without the
                // bond length it was measured against.
                ("void_cutoff".to_owned(), voids.cutoff),
            ],
            histograms: vec![],
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{bcc::Bcc, fcc::Fcc, hcp::Hcp, sc::Sc};

    fn par() -> crate::exec::Par {
        crate::exec::Par::new(1)
    }

    /// A defect-free FCC block with the given atoms removed from `remove`.
    fn fcc_block(ncells: usize, a: f64, remove: &[usize]) -> (UnitCell, Vec<Coord3>) {
        let cell = UnitCell::orthorhombic(ncells as f64 * a, ncells as f64 * a, ncells as f64 * a);
        let sites = SiteLattice::build(&cell, &Fcc { a }.primitive(), [0.0; 3]).unwrap();
        let all: Vec<Coord3> = sites.iter().collect();
        let kept: Vec<Coord3> = all
            .iter()
            .enumerate()
            .filter(|(i, _)| !remove.contains(i))
            .map(|(_, &p)| p)
            .collect();
        (cell, kept)
    }

    /// The whole point of `WsSelf`: a box the reference does not tile.
    ///
    /// `WsRef` refuses this outright, correctly, because counting the sites of
    /// the extra slab would report them as vacancies. `WsSelf` fits the reference
    /// to the atoms instead, so the region is the crystal and the answer is the
    /// one a user expects from a defect-free crystal.
    ///
    /// The offset is 1.1% of a lattice constant on a 14 A box: enough that
    /// `ceil(14.11/3.5) = 5` cells against the 4 the crystal actually occupies,
    /// so `WsRef` cannot be used at all here. Nothing about the crystal changed.
    #[test]
    fn ws_self_handles_the_box_ws_ref_has_to_refuse() {
        let a = 3.5;
        let n = 4usize;
        let len = n as f64 * a;
        let (_exact_cell, sites) = fcc_block(n, a, &[]);
        assert_eq!(sites.len(), n * n * n * 4);

        let slack = 0.011 * len;
        let cell = UnitCell::orthorhombic(len + slack, len + slack, len + slack);

        // The premise: the reference genuinely does not tile this box.
        let tiled = SiteLattice::build(&cell, &Fcc { a }.primitive(), [0.0; 3]).unwrap();
        assert!(
            !tiled.is_commensurate(COMMENSURATE_TOL),
            "test box must be incommensurate, or it proves nothing"
        );
        let refused = WsRef::new(Box::new(Fcc { a })).run_positions(&cell, &sites, &par());
        assert!(
            refused.is_err(),
            "WsRef accepting an incommensurate box would be the bug"
        );

        let r = WsSelf::new(Box::new(Fcc { a }))
            .run_region(&cell, &sites, &par())
            .unwrap();
        assert_eq!(
            r.vacancy_count(),
            0,
            "self-fitted reference reported vacancies on a defect-free crystal"
        );
        assert_eq!(r.interstitial_count(), 0);
        assert_eq!(r.n_sites(), sites.len());
        // Not 1e-15: the reference is found by alternating numerical searches,
        // so it converges to a small residual rather than solving exactly. The
        // defect counts are the assertion that matters, and they are exact.
        assert!(
            r.mean_displacement() < 1e-3,
            "self-fitted reference displaced by {} A",
            r.mean_displacement()
        );
        assert_eq!(r.site_of_atom.len(), sites.len());
        assert!(r.site_of_atom.iter().all(|&s| s >= 0));
        assert_eq!(r.occupancy.iter().filter(|&&o| o == 1).count(), sites.len());
    }

    /// An orthohexagonal HCP block, four atoms per box, built from the geometry
    /// rather than from [`SiteLattice`] so a tiling bug cannot hide in the fixture.
    fn hcp_block(nx: usize, ny: usize, nz: usize, a: f64, c: f64) -> (UnitCell, Vec<Coord3>) {
        let la = a * 3.0_f64.sqrt();
        let cell = UnitCell::orthorhombic(nx as f64 * a, ny as f64 * la, nz as f64 * c);
        let mut pos = Vec::with_capacity(nx * ny * nz * 4);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x0 = i as f64 * a;
                    let y0 = j as f64 * la;
                    let z0 = k as f64 * c;
                    pos.push([x0, y0, z0]);
                    pos.push([x0 + a / 2.0, y0 + la / 2.0, z0]);
                    pos.push([x0, y0 + la / 3.0, z0 + c / 2.0]);
                    pos.push([x0 + a / 2.0, y0 + la / 3.0 + la / 2.0, z0 + c / 2.0]);
                }
            }
        }
        (cell, pos)
    }

    /// The same guarantee for the hexagonal lattice, which the cubic fixtures
    /// cannot make: its cell has to be the axis-aligned orthohexagonal box
    /// *and* the data's stacking phase, or half the atoms of a perfect ABAB
    /// crystal come out off-lattice by a/√3 (phase) or the site set holds 640
    /// sites for 384 atoms (sheared bounding-box overfill → phantom vacancies).
    #[test]
    fn ws_self_seats_every_atom_of_a_perfect_hcp_crystal() {
        let a = 2.51;
        let c = a * (8.0_f64 / 3.0).sqrt();
        let (cell, atoms) = hcp_block(6, 4, 4, a, c);
        assert_eq!(atoms.len(), 6 * 4 * 4 * 4);

        let r = WsSelf::new(Box::new(Hcp { a, c }))
            .run_region(&cell, &atoms, &par())
            .unwrap();

        assert_eq!(r.n_sites(), atoms.len(), "one site per atom");
        assert_eq!(r.vacancy_count(), 0, "perfect HCP reported vacancies");
        assert_eq!(
            r.interstitial_count(),
            0,
            "perfect HCP reported interstitials"
        );
        assert!(
            r.mean_displacement() < 1e-3,
            "mean {}",
            r.mean_displacement()
        );
    }

    /// A region that is not axis aligned, the case where the eight sign flips would
    /// look like the right answer: a sign change is not a rotation.
    ///
    /// Vacancies are asserted to be zero as well, which needs the region to be
    /// bounded by the cells the atoms occupy rather than by the atoms' bounding box:
    /// the box around a tilted grain is not a whole number of cells in any direction
    /// and holds 546 sites where the grain has 256 atoms.
    #[test]
    fn ws_self_seats_every_atom_of_a_tilted_grain() {
        let a = 3.5;
        let n = 5usize;
        let (exact_cell, sites) = fcc_block(n, a, &[]);
        let q = crate::lattice::fit::to_orientation(crate::lattice::fit::mat_to_quat(
            &crate::lattice::fit::axis_angle([0.3, -0.5, 0.81], 0.6f64),
        ));
        let rot = crate::lattice::fit::quat_to_mat(q);
        let mid = 40.0;
        let cell = UnitCell::orthorhombic(mid, mid, mid);
        let (_, chi) = exact_cell.bounds();
        let centre = [chi[0] / 2.0, chi[1] / 2.0, chi[2] / 2.0];
        let atoms: Vec<Coord3> = sites
            .iter()
            .map(|&p| {
                let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
                let r = rot.mul_v(d);
                [mid / 2.0 + r[0], mid / 2.0 + r[1], mid / 2.0 + r[2]]
            })
            .collect();

        let opts = FitOptions {
            seed_orientation: Some(q),
            ..FitOptions::default()
        };
        let r = WsSelf::new(Box::new(Fcc { a }))
            .with_options(opts)
            .run_region(&cell, &atoms, &par())
            .unwrap();

        assert!(
            r.mean_displacement() < 1e-3,
            "tilted grain not seated: mean displacement {} A",
            r.mean_displacement()
        );
        assert_eq!(
            r.interstitial_count(),
            0,
            "a defect-free tilted grain reported interstitials"
        );
        assert!(r.site_of_atom.iter().all(|&s| s >= 0));
        assert_eq!(
            r.occupancy.iter().filter(|&&o| o == 1).count(),
            atoms.len(),
            "every atom should hold its own site"
        );
        assert_eq!(
            r.n_sites(),
            atoms.len(),
            "a tilted grain has one site per atom: {:?}",
            r
        );
        assert_eq!(
            r.vacancy_count(),
            0,
            "a defect-free tilted grain reported vacancies: {:?}",
            r
        );

        // The fitted scale is invariant under the tilt: the grain
        // is turned by 0.6 rad about a skewed axis, so reading a single row
        // component of the fitted cell reports ~0.84 a while the vector's length
        // is the lattice parameter the fit actually found. Pin both so a
        // regression in either direction has a name. `col_norm(0)` is what
        // `ws_fit_a` reports; on this cubic fixture it equals `row_norm(0)`.
        let a_fit = r.fit.cell.h.col_norm(0);
        assert!(
            (a_fit / a - 1.0).abs() < 1e-4,
            "fitted a = {a_fit} A against the {a} A it was tilted from"
        );
        assert!((r.fit.cell.h.row_norm(0) - a_fit).abs() < 1e-9 * a);
        let a_component = r.fit.cell.h.row(0)[0];
        assert!(
            (a_component / a - 1.0).abs() > 0.05,
            "fixture no longer rotated: row(0)[0] = {a_component} A == a, test has gone vacuous"
        );
    }

    /// Planted defects have to survive the whole pipeline: fit, then count.
    #[test]
    fn ws_self_recovers_planted_vacancies_and_interstitials() {
        let a = 3.5;
        let vacancies: Vec<usize> = vec![5, 61, 130, 400];
        let (cell, mut atoms) = fcc_block(8, a, &vacancies);

        let clean = WsSelf::new(Box::new(Fcc { a }))
            .run_region(&cell, &atoms, &par())
            .unwrap();
        assert_eq!(clean.vacancy_count(), vacancies.len());

        // Interstitials at the octahedral sites, so they sit a lattice half-width
        // from the nearest real site and are not absorbed by it.
        for k in 0..3usize {
            let p = atoms[k * 37 % atoms.len()];
            atoms.push([p[0] + a / 2.0, p[1] + a / 2.0, p[2] + a / 2.0]);
        }
        let both = WsSelf::new(Box::new(Fcc { a }))
            .run_region(&cell, &atoms, &par())
            .unwrap();
        assert_eq!(
            both.vacancy_count(),
            vacancies.len(),
            "adding interstitials changed the vacancy count"
        );
        assert_eq!(both.interstitial_count(), 3);
    }

    /// `WsRef` defaults `origin` to `[0,0,0]`, but every LAMMPS `write_data` file
    /// has nonzero bounds (the Co-Fe-Mn-Ni fixtures sit at
    /// `xlo = 2.3213234469867925`) and the reader keeps positions unwrapped *and*
    /// sets `cell.origin`, so the lattice must be anchored at the box corner
    /// (`with_origin(cell.origin)`, as `rmda ws` does). Anchoring it at absolute
    /// zero translates the whole site grid by `-xlo` relative to the atoms --
    /// 2.32 A on a 2.49 A nearest-neighbour spacing. Two identical FCC crystals
    /// differing only in `xlo` then give 0 vacancies and 64289 respectively, and a
    /// *defect-free* reference crystal reads as 57614 vacancies.
    #[test]
    fn lattice_origin_defaults_to_the_box_corner_not_absolute_zero() {
        let a = 3.525_933_827_650_641; // L/40 from the HEA fixture
        let nc = 4usize;
        let xlo = 2.321_323_446_986_792_5;
        let mut cell = UnitCell::orthorhombic(nc as f64 * a, nc as f64 * a, nc as f64 * a);
        cell.origin = [xlo; 3];

        // The crystal is anchored at the box corner, so it moves with `xlo`.
        let sites = SiteLattice::build(&cell, &Fcc { a }.primitive(), cell.origin).unwrap();
        assert!(
            sites.is_commensurate(COMMENSURATE_TOL),
            "test block must be commensurate for this to mean anything"
        );
        let pos: Vec<Coord3> = sites.iter().collect();
        assert_eq!(pos.len(), nc * nc * nc * 4);

        let r = WsRef::new(Box::new(Fcc { a }))
            .with_origin(cell.origin)
            .run_positions(&cell, &pos, &par())
            .unwrap();
        assert_eq!(
            r.vacancy_count, 0,
            "perfect FCC at xlo!=0 reported vacancies"
        );
        assert_eq!(
            r.interstitial_count, 0,
            "perfect FCC at xlo!=0 reported interstitials"
        );
        assert!(
            r.mean_displacement < 1e-9,
            "perfect FCC at xlo!=0 displaced by {}",
            r.mean_displacement
        );

        // And anchoring at absolute zero really is wrong: it misassigns a large
        // fraction of a small block, in proportion to how badly `xlo` happens to
        // sit relative to the lattice.
        let bad = WsRef::new(Box::new(Fcc { a }))
            .run_positions(&cell, &pos, &par())
            .unwrap();
        assert!(
            bad.vacancy_count * 4 > pos.len(),
            "anchoring at absolute zero is expected to misassign most sites; got \
             {} vacancies out of {} sites -- if this now passes, the origin is \
             being applied somewhere else and the CLI fix can be moved down into \
             WsRef itself",
            bad.vacancy_count,
            pos.len()
        );
    }

    #[test]
    fn perfect_crystal_has_no_defects() {
        let (cell, pos) = fcc_block(3, 2.0, &[]);
        let ws = WsRef::new(Box::new(Fcc { a: 2.0 }));
        let r = ws.run_positions(&cell, &pos, &par()).unwrap();
        assert_eq!(r.vacancy_count, 0, "perfect FCC reported vacancies");
        assert_eq!(
            r.interstitial_count, 0,
            "perfect FCC reported interstitials"
        );
        assert_eq!(r.occupancy.len(), 3 * 3 * 3 * 4);
        assert!(
            r.mean_displacement < 1e-9,
            "perfect FCC displaced by {}",
            r.mean_displacement
        );
        // Every atom on a distinct site.
        let mut seen: Vec<i64> = r.site_of_atom.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), pos.len());
    }

    #[test]
    fn n_vacancies_are_recovered_exactly() {
        for n_remove in [1usize, 2, 5, 17] {
            let victims: Vec<usize> = (0..n_remove).map(|k| k * 7 + 3).collect();
            let (cell, pos) = fcc_block(4, 2.0, &victims);
            let ws = WsRef::new(Box::new(Fcc { a: 2.0 }));
            let r = ws.run_positions(&cell, &pos, &par()).unwrap();
            assert_eq!(
                r.vacancy_count, n_remove,
                "removed {n_remove}, found {} vacancies",
                r.vacancy_count
            );
            assert_eq!(r.interstitial_count, 0, "vacancies produced interstitials");
            assert!(r.occupancy.iter().all(|&o| o <= 1));
        }
    }

    #[test]
    fn n_interstitials_are_recovered_exactly() {
        // Duplicate atoms at random sites: each extra atom must show up as one
        // interstitial and must not create a vacancy.
        let a = 2.0;
        let nc = 4;
        let (cell, pos) = fcc_block(nc, a, &[]);
        let extras = 4usize;
        let mut p = pos.clone();
        for k in 0..extras {
            // Offset each duplicate off-site by a small amount so it is not a
            // bit-exact repeat.
            let src = pos[k * 11 % pos.len()];
            p.push([src[0] + 0.05, src[1] - 0.03, src[2] + 0.04]);
        }
        let ws = WsRef::new(Box::new(Fcc { a }));
        let r = ws.run_positions(&cell, &p, &par()).unwrap();
        assert_eq!(r.vacancy_count, 0, "interstitials created vacancies");
        assert_eq!(
            r.interstitial_count, extras,
            "added {extras} interstitial atoms, found {}",
            r.interstitial_count
        );
        assert_eq!(r.interstitial_site_count, extras);
    }

    #[test]
    fn vacancies_and_interstitials_are_counted_together() {
        let a = 2.0;
        let (cell, pos) = fcc_block(4, a, &[5, 9, 40]);
        let mut p = pos.clone();
        let s0 = pos[0];
        let s1 = pos[1];
        p.push([s0[0] + 0.04, s0[1], s0[2]]);
        p.push([s1[0], s1[1] + 0.04, s1[2]]);
        let ws = WsRef::new(Box::new(Fcc { a }));
        let r = ws.run_positions(&cell, &p, &par()).unwrap();
        assert_eq!(r.vacancy_count, 3);
        assert_eq!(r.interstitial_count, 2);
        assert_eq!(r.defect_count(), 5);
    }

    #[test]
    fn bcc_perfect_block_is_defect_free() {
        let a = 2.0;
        let nc = 4;
        let cell = UnitCell::orthorhombic(nc as f64 * a, nc as f64 * a, nc as f64 * a);
        let sites = SiteLattice::build(&cell, &Bcc { a }.primitive(), [0.0; 3]).unwrap();
        let pos: Vec<Coord3> = sites.iter().collect();
        let ws = WsRef::new(Box::new(Bcc { a }));
        let r = ws.run_positions(&cell, &pos, &par()).unwrap();
        assert_eq!(r.vacancy_count, 0);
        assert_eq!(r.interstitial_count, 0);
        assert_eq!(r.occupancy.len(), nc * nc * nc * 2);
    }

    #[test]
    fn sc_perfect_block_is_defect_free() {
        let a = 3.0;
        let nc = 3;
        let cell = UnitCell::orthorhombic(nc as f64 * a, nc as f64 * a, nc as f64 * a);
        let sites = SiteLattice::build(&cell, &Sc { a }.primitive(), [0.0; 3]).unwrap();
        let pos: Vec<Coord3> = sites.iter().collect();
        let ws = WsRef::new(Box::new(Sc { a }));
        let r = ws.run_positions(&cell, &pos, &par()).unwrap();
        assert_eq!(r.vacancy_count, 0);
        assert_eq!(r.interstitial_count, 0);
        assert_eq!(r.occupancy.len(), nc * nc * nc);
    }

    #[test]
    fn thermally_displaced_atoms_produce_no_spurious_defects() {
        // The sanity check that matters for real input: atoms displaced by far
        // more than a thermal amplitude must not cross a W–S boundary and invent
        // vacancy/interstitial pairs. Displacements here are ~20% of the
        // nearest-neighbour distance.
        let a = 4.0;
        let (cell, pos) = fcc_block(3, a, &[]);
        let nn = a / 2.0_f64.sqrt();
        let jittered: Vec<Coord3> = pos
            .iter()
            .enumerate()
            .map(|(i, p)| {
                // Deterministic pseudo-random offsets in +/- 0.2*nn.
                let f = |k: f64| ((i as f64 * (k + 1.7)).sin()) * 0.2 * nn;
                [p[0] + f(1.0), p[1] + f(2.0), p[2] + f(3.0)]
            })
            .collect();
        let ws = WsRef::new(Box::new(Fcc { a }));
        let r = ws.run_positions(&cell, &jittered, &par()).unwrap();
        assert_eq!(
            r.vacancy_count,
            0,
            "{} vacancies from jitter of {:.3} A (nn {:.3})",
            r.vacancy_count,
            0.2 * nn,
            nn
        );
        assert_eq!(r.interstitial_count, 0);
        assert!(r.mean_displacement > 0.0, "jitter produced no displacement");
    }

    #[test]
    fn results_are_identical_across_thread_counts() {
        // The module docs' determinism claim, tested rather than asserted in a comment.
        let a = 2.0;
        let (cell, pos) = fcc_block(5, a, &[1, 2, 3, 88, 400]);
        let mut p = pos.clone();
        for k in 0..3 {
            let s = pos[k * 5 % pos.len()];
            p.push([s[0] + 0.05, s[1], s[2]]);
        }
        let ws = WsRef::new(Box::new(Fcc { a }));

        let serial = ws
            .run_positions(&cell, &p, &crate::exec::Par::new(1))
            .unwrap();
        let threaded = ws
            .run_positions(&cell, &p, &crate::exec::Par::new(4))
            .unwrap();
        assert_eq!(serial.vacancy_count, threaded.vacancy_count);
        assert_eq!(serial.interstitial_count, threaded.interstitial_count);
        assert_eq!(serial.occupancy, threaded.occupancy);
        assert_eq!(serial.site_of_atom, threaded.site_of_atom);
    }

    #[test]
    fn incommensurate_box_is_refused_rather_than_reporting_phantom_vacancies() {
        // The regression this guards: `dims` is a ceil, so a box spanning
        // 6.000058 cells stores 7 cells and every site in the extra slab reads as
        // a vacancy. On this block that is 508 phantom vacancies. The refusal
        // is the whole point.
        let a = 4.0;
        let nc = 6usize;
        let cell = UnitCell::orthorhombic(nc as f64 * a, nc as f64 * a, nc as f64 * a);
        let sites = SiteLattice::build(&cell, &Fcc { a }.primitive(), [0.0; 3]).unwrap();
        assert!(sites.is_commensurate(COMMENSURATE_TOL));

        // Perturb the lattice constant by 1 part in 1e5 — i.e. the 5th significant
        // figure of a perfectly reasonable input.
        let a2 = a * (1.0 - 1e-5);
        let bad = SiteLattice::build(&cell, &Fcc { a: a2 }.primitive(), [0.0; 3]).unwrap();
        assert!(!bad.is_commensurate(COMMENSURATE_TOL));

        let pos: Vec<Coord3> = sites.iter().collect();
        let ws = WsRef::new(Box::new(Fcc { a: a2 }));
        let err = ws.run_positions(&cell, &pos, &par()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("does not tile the box"), "unexpected: {msg}");
        assert!(msg.contains("--self"), "error should name the fix: {msg}");
    }

    #[test]
    fn lattice_constant_precision_decides_the_answer() {
        // Documents the trap explicitly: at 5 significant figures the run is
        // refused, and with the exact value the counts are perfect. Same input
        // file, same lattice type.
        let a = 4.0;
        let nc = 6usize;
        let cell = UnitCell::orthorhombic(nc as f64 * a, nc as f64 * a, nc as f64 * a);
        let victims: Vec<usize> = (0..5).collect();
        let sites = SiteLattice::build(&cell, &Fcc { a }.primitive(), [0.0; 3]).unwrap();
        let all: Vec<Coord3> = sites.iter().collect();
        let pos: Vec<Coord3> = all
            .iter()
            .enumerate()
            .filter(|(i, _)| !victims.contains(i))
            .map(|(_, &p)| p)
            .collect();

        // `Fcc::a` is the conventional cubic constant, so an exact `a` tiles the
        // box and the counts must be perfect.
        let r = WsRef::new(Box::new(Fcc { a }))
            .run_positions(&cell, &pos, &par())
            .unwrap();
        assert_eq!(r.vacancy_count, 5);
        assert_eq!(r.interstitial_count, 0);
        assert_eq!(r.occupancy.len(), nc * nc * nc * 4);
    }

    #[test]
    fn affine_mapping_to_current_is_rejected() {
        let (cell, pos) = fcc_block(2, 2.0, &[]);
        let ws = WsRef::new(Box::new(Fcc { a: 2.0 })).with_affine(AffineMapping::ToCurrent);
        let err = ws.run_positions(&cell, &pos, &par()).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("tocurrent"),
            "unexpected error: {err}"
        );
    }

    /// The phantom-vacancy trap, measured rather than asserted.
    ///
    /// Occupancy belongs to a site, so an atom missing from the position set does
    /// not stop being in the crystal -- it leaves an empty site, and the site
    /// reports as a vacancy. One per atom removed, exactly. This is why a seam
    /// exclusion cannot be written as "drop the seam atoms from the run": on a
    /// defect-free crystal it would report the seam length back as a defect count.
    #[test]
    fn excluding_atoms_from_the_position_set_manufactures_vacancies() {
        let a = 4.0;
        let (cell, all) = fcc_block(10, a, &[]);
        let ws = WsSelf::new(Box::new(Fcc { a }));
        let clean = ws.run_region(&cell, &all, &par()).unwrap();
        assert_eq!(
            (clean.vacancy_count(), clean.interstitial_count()),
            (0, 0),
            "a defect-free crystal must report no defects"
        );

        // Remove every site in one cell plane: the stand-in for a seam.
        let plane: Vec<usize> = (0..all.len())
            .filter(|&i| (all[i][0] / (a / 2.0)).round() as i64 == 10)
            .collect();
        assert!(!plane.is_empty(), "the plane must not be empty");
        let kept: Vec<Coord3> = all
            .iter()
            .enumerate()
            .filter(|(i, _)| !plane.contains(i))
            .map(|(_, &p)| p)
            .collect();

        let dropped = ws.run_region(&cell, &kept, &par()).unwrap();
        assert_eq!(
            dropped.vacancy_count(),
            plane.len(),
            "the trap is one phantom vacancy per excluded atom; if this ever stops \
             holding, the split is no longer load-bearing and the reason for it has \
             gone stale"
        );

        // The split: fit on the kept atoms, assign all of them.
        let split = ws
            .run_region_split(&cell, &kept, &all, None, &par())
            .unwrap();
        assert_eq!(
            split.vacancy_count(),
            0,
            "keeping the excluded atoms in the assignment is what stops the phantom \
             vacancies, because their sites stay occupied"
        );
    }

    /// The mask changes the interstitial tally and not the vacancy tally.
    ///
    /// Both halves matter and they pull in opposite directions: a masked-out atom
    /// must not be able to *create* a vacancy (it is still there, so its site is
    /// occupied), and must not be able to *report* one either (a disordered seam
    /// atom off its own site is not a defect of the grain interior).
    #[test]
    fn split_mask_restricts_interstitials_not_vacancies() {
        let a = 4.0;
        let (cell, all) = fcc_block(6, a, &[]);
        // An explicit tolerance rather than the default half-NN: the default is
        // ~1.4 A here, so a plausible-looking 0.9 A displacement would sit
        // comfortably *inside* it and the test would count nothing for a reason
        // that has nothing to do with the mask.
        let ws = WsSelf::new(Box::new(Fcc { a })).with_tolerance(0.2);

        // Displace a handful of atoms well off their sites: genuine interstitials.
        let bumped: Vec<Coord3> = all
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                if i % 97 == 3 {
                    [p[0] + 0.9, p[1], p[2]]
                } else {
                    p
                }
            })
            .collect();
        let open = ws
            .run_region_split(&cell, &bumped, &bumped, None, &par())
            .unwrap();
        let n_bumped = (0..all.len()).filter(|&i| i % 97 == 3).count();
        assert!(open.interstitial_count() >= n_bumped);

        // Mask those same atoms out of the tally: the interstitial figure must
        // fall, because a masked atom is in the boundary bin, not the interior.
        let mask: Vec<bool> = (0..all.len()).map(|i| i % 97 != 3).collect();
        let masked = ws
            .run_region_split(&cell, &bumped, &bumped, Some(&mask), &par())
            .unwrap();
        assert_eq!(
            masked.interstitial_count(),
            0,
            "masking out every displaced atom must leave no interstitial, since \
             nothing that could report one is left counted"
        );
        assert_eq!(
            masked.vacancy_count(),
            open.vacancy_count(),
            "the vacancy tally is over sites and the mask does not touch it"
        );
        assert_eq!(masked.n_sites(), open.n_sites());
    }

    /// A mask that does not line up with the atoms it is supposed to index is a
    /// bug in the caller, not a shorter count.
    #[test]
    fn split_rejects_a_mismatched_mask() {
        let (cell, all) = fcc_block(3, 4.0, &[]);
        let ws = WsSelf::new(Box::new(Fcc { a: 4.0 }));
        let err = ws
            .run_region_split(&cell, &all, &all, Some(&[true; 3]), &par())
            .unwrap_err();
        assert!(
            err.to_string().contains("counted mask"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn analysis_path_emits_properties_and_scalars() {
        use crate::analysis::{AnalysisContext, NeighborhoodContext};
        use crate::data::frame::Frame;
        use crate::data::particle::ParticleData;

        let a = 2.0;
        let (cell, pos) = fcc_block(3, a, &[7]);
        let mut pd = ParticleData::with_capacity(pos.len());
        for (i, q) in pos.iter().enumerate() {
            pd.add_atom(*q, i as i64 + 1, 1);
        }
        let frame = Frame::new(cell, pd);

        // WS never touches the cell list, but AnalysisContext requires one. The
        // cutoff here is arbitrary and irrelevant: `cutoffs()` being empty is what
        // stops the pipeline building a real one for this analysis.
        let cl = crate::neighbor::cell_list::CellList::build(&cell, &pos, 1.0).unwrap();
        let nk = NeighborhoodContext { cell: Some(&cl) };
        let par = par();
        let ctx = AnalysisContext {
            frame: &frame,
            nkeys: &nk,
            par: &par,
        };
        let ws = WsRef::new(Box::new(Fcc { a }));
        assert!(
            ws.cutoffs().is_empty(),
            "WS must not request a neighbour cutoff"
        );

        let mut out = PropertyStore::new();
        let mut summary = Summary::default();
        ws.run(&ctx, &mut out, &mut summary).unwrap();

        let site = out.get("ws_site").unwrap();
        assert_eq!(site.len(), pos.len());
        assert!(out.get("ws_displacement").unwrap().as_f64().is_some());
        let occ = out.get("ws_occupancy").unwrap();
        assert_eq!(occ.len(), 3 * 3 * 3 * 4);
        assert!(!occ.per_particle, "occupancy is per site, not per atom");

        let vac = summary
            .scalars
            .iter()
            .find(|(k, _)| k == "ws_vacancies")
            .map(|(_, v)| *v)
            .unwrap();
        assert_eq!(vac, 1.0);
    }
}
