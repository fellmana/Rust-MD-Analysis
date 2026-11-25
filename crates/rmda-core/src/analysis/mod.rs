//! Analysis framework (design §12).
//!
//! Analyses plug into a per-frame pipeline: they receive an immutable
//! `AnalysisContext` (the frame and the shared spatial index) and write
//! results back into the frame's `PropertyStore` (per-atom) or a `Summary`
//! (scalar/global quantities, e.g. RDF histograms).
//!
//! All analyses are deterministic (design §10.2) and run through `exec::Par`.

use crate::data::frame::Frame;
use crate::error::Result;
use crate::exec::Par;

pub mod cluster;
pub mod cna;
pub mod defect_dump;
pub mod defects;
pub mod grain_auto;
pub mod grain_boundaries;
pub mod grain_seg;
pub mod grain_table;
pub mod grain_track;

pub mod grains;
pub mod neighbor_census;
pub mod occupancy;
pub mod orient;
pub mod ptm;
pub mod rdf;
/// Registry of the analyses a pipeline runs.
pub mod registry;
pub mod roto_strain;
pub mod stacking;
pub mod void;
pub mod void_b;
pub mod warren_cowley;
pub mod ws;

/// Density estimate of the radius containing `k` atoms, inflated 1.5×.
///
/// The k-th nearest neighbour in a crystal sits farther out than the ideal-gas
/// sphere holding k atoms, so a bare `(3k/4πρ)^(1/3)` under-covers a lattice;
/// the margin absorbs that. `INFINITY` for an empty frame means callers must
/// handle `n == 0` before building a cell list from the result.
pub(crate) fn knear_radius(k: usize, rho: f64) -> f64 {
    if rho <= 0.0 {
        return f64::INFINITY;
    }
    (3.0 * k as f64 / (4.0 * std::f64::consts::PI * rho)).cbrt() * 1.5
}

/// Result a single-frame analysis writes.
#[derive(Debug, Default, Clone)]
pub struct Summary {
    /// Named scalar/global results (e.g. "mean_cn", "total_pairs").
    pub scalars: Vec<(String, f64)>,
    /// Named histograms (e.g. RDF g(r)).
    pub histograms: Vec<(String, Vec<f64>)>,
}

impl Summary {
    pub fn scalar(name: &str, v: f64) -> Self {
        Summary {
            scalars: vec![(name.to_string(), v)],
            histograms: Vec::new(),
        }
    }

    pub fn merge_from(&mut self, other: Summary) {
        self.scalars.extend(other.scalars);
        self.histograms.extend(other.histograms);
    }
}

/// Immutable view handed to an analysis.
pub struct AnalysisContext<'a> {
    pub frame: &'a Frame,
    pub nkeys: &'a NeighborhoodContext<'a>,
    pub par: &'a Par,
}

/// Neighbourhood data a single analysis needs: the shared cell list, queried
/// in place at the analysis's own cutoffs (design §9.2 — no materialised
/// neighbour pool).
pub struct NeighborhoodContext<'a> {
    /// The shared cell list, if one was built.
    ///
    /// Optional because not every analysis needs it, and at 128M atoms the
    /// difference is a whole neighbour index over the input frame rather than a
    /// rounding error: the Wigner-Seitz engines all index their own reference
    /// sites instead, and they say so by reporting no cutoffs.
    pub cell: Option<&'a crate::neighbor::cell_list::CellList<'a>>,
}

impl<'a> NeighborhoodContext<'a> {
    /// The shared cell list, or an error naming the analysis that wanted it.
    pub fn require_cell(&self, who: &str) -> Result<&'a crate::neighbor::cell_list::CellList<'a>> {
        self.cell.ok_or_else(|| {
            crate::error::Error::invalid(format!(
                "{who} needs the shared neighbour list, which was not built for this run"
            ))
        })
    }
}

/// The analysis trait.
pub trait Analysis {
    fn name(&self) -> &'static str;

    /// Returns the cutoff(s) this analysis needs; the pipeline builds cell
    /// lists up to the max.
    ///
    /// Owned rather than borrowed because an analysis may carry its cutoff as
    /// configuration (e.g. `--cutoff 4.2`), so a `&'static [f64]` table would
    /// silently report a default the analysis never uses.
    fn cutoffs(&self) -> Vec<f64>;

    /// Cutoffs for a concrete frame, for analyses that size their own search
    /// from the geometry rather than from configuration alone.
    ///
    /// The adaptive CNA radius is the motivating case: OVITO gates its 12/14
    /// neighbour tests on *coordination*, not on a radius, so a fixed pipeline
    /// radius silently under-coordinates atoms in a hot, expanded lattice. The
    /// radius therefore has to scale with the number density, which is only
    /// known once there is a frame. Defaults to [`Analysis::cutoffs`].
    fn cutoffs_for(&self, frame: &crate::data::frame::Frame) -> Vec<f64> {
        let _ = frame;
        self.cutoffs()
    }

    /// Run on one frame. `ctx.nkeys` provides the shared cell list. Per-atom
    /// results go into `out`, scalar/global results into `summary`.
    fn run(
        &self,
        ctx: &AnalysisContext,
        out: &mut crate::data::property::PropertyStore,
        summary: &mut Summary,
    ) -> Result<()>;

    /// This run's Wigner–Seitz detail, if the analysis has any.
    ///
    /// [`run`](Self::run) hands back per-*atom* properties and scalars, which is
    /// lossy for defects in a specific way: `ws_occupancy` has one entry per
    /// *site*, so it is registered `per_particle = false` and never reaches the
    /// atom table. On a block with three vacancies that is 864 entries against 861
    /// atoms, and a vacancy has no atom to attach to at all.
    ///
    /// So an analysis that counts defects keeps its outcome and hands it over
    /// here, rather than the caller reconstructing it from scalars that cannot
    /// distinguish *which* sites are empty. `None` by default: most analyses have
    /// no site grid, and returning a default-constructed one would invite the
    /// caller to write an empty defects directory and call it a result.
    fn take_defects(&self) -> Option<crate::analysis::defect_dump::DefectDump> {
        None
    }
}
