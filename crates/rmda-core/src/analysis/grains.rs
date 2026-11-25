//! Bond-graph grain labelling, used by `rmda grain-boundaries`.
//!
//! A polycrystal is not a crystal, and a single-lattice Wigner-Seitz analysis
//! assumes it is: it fits one reference lattice to the whole frame, which is
//! exactly the assumption a grain boundary invalidates. This module labels which
//! atoms belong to the same grain, so the seam pass has grains to compare. The
//! segmentation behind `rmda segment` and `rmda defects` is `grain_seg`.
//!
//! The criterion is OVITO's: two atoms share a grain when they are neighbours,
//! their structure types are compatible, and the **disorientation** angle between
//! their orientations is small. Structure type alone is not enough (a stacking
//! fault is a boundary in the structure-type view and not a grain boundary in the
//! material), and orientation alone is not enough either, so type is used as a
//! cheap pre-filter and orientation as the actual test.
//!
//! **This is not OVITO's full algorithm.** Implemented: the structure-aware
//! adjacency graph, the disorientation criterion at OVITO's
//! `_misorientationThreshold`, grain-graph merging, and the artificial-grain
//! cleanup. Not here: the Node-Pair-Sampling partition and the structure-only
//! fallback (`grain_auto` ports OVITO's algorithm). The partitioning here is
//! connected-components labelling — well defined, but not OVITO's algorithm, so
//! the grain count is stable but not identical to OVITO's on the same input.
//! The grain-graph merge threshold is explicit rather than OVITO's fitted
//! `ThresholdSelection`, and it merges over-split grains; it cannot undo a
//! bridge, since labelling has already fused such a pair before the grain graph
//! exists.
//!
//! Orientation is read from the shared [`PropertyStore`] as PTM's per-atom
//! `StructureOrientation`; it is not recomputed here.

use crate::analysis::ptm::quat::quat_rot;
use crate::analysis::ptm::quat_tables::{CUBIC, HCP as HCP_GROUP, ICOSAHEDRAL};
use crate::analysis::ptm::structure;
use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::data::property::{DType, PropertyStore};
use crate::error::{Error, Result};
use crate::graph::{canonical_labels, DisjointSet};
use std::collections::BTreeMap;

/// OVITO's `GrainSegmentationModifier::_misorientationThreshold`, degrees.
pub const DEFAULT_MISORIENTATION_THRESHOLD: f64 = 4.0;

/// Disorientation below this is reported as exactly zero.
///
/// Two atoms in the same grain produce a `dot` a few ULPs below 1, and `acos` of
/// that is a small positive angle rather than zero. At 12 bonds per atom the
/// accumulated bias is enough to flip a pair that sits exactly on the threshold.
const DISORIENTATION_EPS: f64 = 1e-5;

/// Label for an atom that belongs to no grain, because its grain was rejected.
///
/// Internal representation. The stored `Grain` property is `i32` and writes `-1`
/// for these atoms, since `DType` has no unsigned 32-bit variant; keeping the
/// sentinel distinct from grain 0 is the point either way.
pub const NO_GRAIN: u32 = u32::MAX;

/// The value written to the stored `Grain` property for an atom in no grain.
pub const NO_GRAIN_EXPORT: i32 = -1;

/// How far the orphan adoption may reach, in bond lengths.
///
/// One bond is too short. The shell around a vacancy is exactly the population
/// PTM cannot classify, so the nearest atom that still belongs to a grain is the
/// second or third coordination shell — 1.633 bond lengths for BCC's third shell
/// and 1.732 for FCC's. A reach that covers both is what turns the shell of a
/// hole back into members of the grain the hole is in, and that membership is
/// what lets a Wigner-Seitz census find an owner for the empty site.
///
/// Adoption is a flood fill, so a disordered region thicker than this is filled
/// from its edges inward one reach at a time; the factor bounds each step, not
/// the total.
const ADOPTION_REACH: f64 = 1.75;

/// The point group of a structure type, or `None` when it has none.
///
/// `None` is load-bearing: `OTHER` has no orientation symmetry to quotient by, so
/// two `OTHER` atoms have *undefined* mutual orientation. Treating them as
/// congruent would merge every non-crystalline atom into whichever grain happened
/// to touch it first.
pub fn point_group(kind: i32) -> Option<&'static [[f64; 4]]> {
    Some(match kind {
        // FCC, BCC and SC share the cubic group: all three are invariant under
        // the same 24 proper rotations.
        structure::FCC | structure::BCC | structure::SC => &CUBIC,
        structure::HCP => &HCP_GROUP,
        structure::ICO => &ICOSAHEDRAL,
        _ => return None,
    })
}

/// Whether two structure types may share a grain.
///
/// Same type always; with `stacking_faults`, also the pair OVITO allows across a
/// stacking fault. Getting this wrong in the strict direction shatters a stacking
/// fault into a grain boundary, which is the more damaging error because it
/// inflates the grain count.
fn types_compatible(a: i32, b: i32, stacking_faults: bool) -> bool {
    if a == b {
        return true;
    }
    if !stacking_faults {
        return false;
    }
    matches!(
        (a, b),
        (structure::FCC, structure::HCP) | (structure::HCP, structure::FCC)
    )
}

/// Per-atom PTM inputs: the structure type, and the orientation where one
/// exists.
type PtmInputs = (Vec<i32>, Vec<Option<[f64; 4]>>);

/// One row of a per-grain table: a scalar or a short tuple of `f64`.
trait Row {
    const COMPONENTS: usize;
    fn write(&self, out: &mut [f64]);
}

impl Row for f64 {
    const COMPONENTS: usize = 1;
    fn write(&self, out: &mut [f64]) {
        out[0] = *self;
    }
}

impl Row for i32 {
    const COMPONENTS: usize = 1;
    fn write(&self, out: &mut [f64]) {
        out[0] = f64::from(*self);
    }
}

impl Row for usize {
    const COMPONENTS: usize = 1;
    fn write(&self, out: &mut [f64]) {
        out[0] = *self as f64;
    }
}

impl Row for [f64; 3] {
    const COMPONENTS: usize = 3;
    fn write(&self, out: &mut [f64]) {
        out.copy_from_slice(self);
    }
}

impl Row for [f64; 4] {
    const COMPONENTS: usize = 4;
    fn write(&self, out: &mut [f64]) {
        out.copy_from_slice(self);
    }
}

/// Build a per-*grain* property: one row per grain, row index equal to the grain
/// id, flagged `per_particle = false`.
///
/// The row count is the grain count, not the atom count, so a consumer must not
/// index it per atom. That is exactly why it is not flagged per-particle: the
/// atom-table writers would otherwise emit a column of the wrong length and
/// silently misalign every value after it.
fn per_grain<T: Row>(name: &str, dtype: DType, rows: &[T]) -> crate::data::property::Property {
    let comps = T::COMPONENTS;
    let mut p = crate::data::property::Property::with_capacity(
        name.to_owned(),
        dtype,
        false,
        comps,
        rows.len(),
    );
    let mut buf = [0.0f64; 4];
    for r in rows {
        r.write(&mut buf[..comps]);
        p.push_value(&buf[..comps]);
    }
    p
}

/// Scale a quaternion to unit length in place; no-op when the norm is unusable.
fn quat_normalize(q: &mut [f64; 4]) {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l > 1e-12 && l.is_finite() {
        for v in q.iter_mut() {
            *v /= l;
        }
    }
}

/// Whether every component is a real number.
fn quat_is_finite(q: &[f64; 4]) -> bool {
    q.iter().all(|v| v.is_finite())
}

/// Disorientation angle between two orientations of the same structure, degrees.
///
/// `2·acos(max |dot(qa, qb·g)|)` maximised over the point group (design
/// §11.3). Maximising `|dot|` minimises the angle, and taking it over the whole
/// group is what makes this a *dis*orientation: `qa` and `qb` may differ by a
/// symmetry operation of the lattice and still describe the same physical
/// orientation.
///
/// The symmetry multiplies from the right because PTM's quaternion rotates the
/// template frame into the sample frame and the lattice symmetry acts in the
/// template frame — the same side `ptm::quat` folds into the fundamental zone
/// on. Left multiplication (`g·qb`) applies the symmetry in the sample frame
/// and reports tens of degrees between symmetry-equivalent orientations.
pub fn disorientation(qa: &[f64; 4], qb: &[f64; 4], group: &[[f64; 4]]) -> f64 {
    // An orientation that is not a real number has no angle. Returning NaN makes
    // the bond test fail, which keeps an atom with a bad orientation out of every
    // grain; returning 0 instead would merge grains across a broken measurement.
    if !quat_is_finite(qa) || !quat_is_finite(qb) {
        return f64::NAN;
    }
    let mut best = 0.0f64;
    let mut found = false;
    for g in group {
        let gq = quat_rot(*qb, *g);
        let d = (qa[0] * gq[0] + qa[1] * gq[1] + qa[2] * gq[2] + qa[3] * gq[3]).abs();
        if !found || d > best {
            best = d;
            found = true;
        }
    }
    if !found {
        return 0.0;
    }
    // Clamp before acos: rounding can push |dot| a hair above 1 and acos of that
    // is NaN, which compares false against every threshold and so would keep
    // every bond rather than dropping it.
    let angle = 2.0 * best.clamp(-1.0, 1.0).acos().to_degrees();
    if angle < DISORIENTATION_EPS {
        0.0
    } else {
        angle
    }
}

/// Merge grains on the grain graph, weighted by the
/// smallest disorientation between them.
///
/// This fixes grains that were split, not grains that were bridged. One atom pair
/// sitting above `misorientation_threshold` welds two otherwise separate crystals
/// into a single component, and by the time a grain graph exists that pair is
/// already inside one node -- no threshold on the grain graph can separate it
/// again, only a different `misorientation_threshold`, which unbridges them at the
/// cost of dissolving every real boundary too. Bridging is diagnosed, not
/// repaired, here.
///
/// What *is* repairable is over-splitting: one grain whose bonds all sit a little
/// above `misorientation_threshold` comes back in pieces, and raising that
/// threshold to glue it back dissolves real boundaries in the same breath.
/// Deciding on the grain graph separates the two. Two grains that touch *and*
/// whose closest approach is a low disorientation were one grain that a
/// measurement split, so they get merged. Crucially the quantity compared against
/// `merge_threshold` is the **minimum disorientation over every bond between the
/// two grains**, not the mean and not the bond that happened to bridge them: a
/// grain boundary that is clean everywhere except for one noisy pair still reads
/// as clean, which is the behaviour the mean would get exactly backwards.
///
/// Type-incompatible pairs never produce an edge. An FCC/BCC boundary is a real
/// boundary, and no disorientation threshold should be able to dissolve it.
///
/// # Why one pass suffices
///
/// A "hierarchical" merge suggests iterating. This one does not
/// need to: merging two grains replaces each of their outgoing edges with the
/// *minimum* of the two, so an edge's weight can only decrease. A set of edges
/// below the threshold is therefore closed under merging, and a single union-find
/// pass over the sub-threshold edges reaches the same fixed point an iterative
/// agglomeration would.
///
/// # Labels
///
/// Renumbering keeps the "lowest member index first" convention of
/// [`label_sets`]: ids are handed out while walking atoms in index order, so a
/// merged grain keeps the id of whichever grain started earliest.
fn merge_grains(
    labels: &[u32],
    n_sets: usize,
    bridges: &[(u32, u32, f64)],
    threshold: f64,
) -> (Vec<u32>, usize, usize) {
    // Collapse the per-bond list to one minimum per grain pair. Done before any
    // union so the decision sees exactly the quantity documented above.
    let mut best: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    for &(i, j, angle) in bridges {
        let (a, b) = (labels[i as usize], labels[j as usize]);
        // An atom in no grain cannot bound anything, and a NaN angle is an
        // orientation that failed to measure rather than a small one.
        if a == NO_GRAIN || b == NO_GRAIN || a == b || !angle.is_finite() {
            continue;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        best.entry(key)
            .and_modify(|v| *v = v.min(angle))
            .or_insert(angle);
    }

    let mut ds = DisjointSet::new(n_sets);
    let mut edges_used = 0usize;
    for (&(a, b), &angle) in &best {
        if angle <= threshold && ds.union(a, b) {
            edges_used += 1;
        }
    }

    // Hand out ids in first-atom order, as `label_sets` does.
    let mut root_to_new = vec![NO_GRAIN; n_sets];
    let mut next = 0u32;
    let mut out = vec![NO_GRAIN; labels.len()];
    for (i, &l) in labels.iter().enumerate() {
        if l == NO_GRAIN {
            continue;
        }
        let r = ds.find(l) as usize;
        if root_to_new[r] == NO_GRAIN {
            root_to_new[r] = next;
            next += 1;
        }
        out[i] = root_to_new[r];
    }
    (out, next as usize, edges_used)
}

/// Hand each orphan atom the grain id of the nearest atom that has one.
///
/// An *orphan* is an atom the caller asks to adopt (`adopt[i]`) that currently
/// has [`NO_GRAIN`]. Adoption is a flood fill: one pass would leave a disordered
/// region thicker than `reach` half-orphaned, so each pass adopts the orphans
/// within `reach` of an atom that already has a grain, and the next pass works
/// inward from there. Every pass is a snapshot — the search reads the ids from
/// before the pass — so the result does not depend on atom order or on which
/// thread got there first.
///
/// Ties are broken by distance, then by the lower grain id: two grains meeting
/// at a boundary put atoms at the same distance from the seam, and the tie has
/// to break the same way on every run.
///
/// Returns the number of atoms adopted.
fn adopt_orphans(
    cell: &crate::cell::UnitCell,
    positions: &[crate::cell::Coord3],
    labels: &mut [u32],
    adopt: &[bool],
    reach: f64,
) -> Result<usize> {
    let n = labels.len();
    if n == 0 || !reach.is_finite() || reach <= 0.0 {
        return Ok(0);
    }
    let list = crate::neighbor::cell_list::CellList::build(cell, positions, reach)?;
    let reach2 = reach * reach;
    let mut scratch: Vec<(usize, crate::cell::Coord3)> = Vec::new();
    let mut adopted = 0usize;
    loop {
        let snapshot = labels.to_vec();
        let mut pending: Vec<(usize, u32)> = Vec::new();
        for i in 0..n {
            if !adopt[i] || snapshot[i] != NO_GRAIN {
                continue;
            }
            list.neighbors_of(i, reach, &mut scratch)?;
            let mut best: Option<(f64, u32)> = None;
            for &(j, dr) in scratch.iter() {
                let l = snapshot[j];
                if l == NO_GRAIN {
                    continue;
                }
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 > reach2 {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some((bd, bl)) => d2 < bd || (d2 == bd && l < bl),
                };
                if better {
                    best = Some((d2, l));
                }
            }
            if let Some((_, l)) = best {
                pending.push((i, l));
            }
        }
        if pending.is_empty() {
            break;
        }
        adopted += pending.len();
        for (i, l) in pending {
            labels[i] = l;
        }
    }
    Ok(adopted)
}

/// Label every eligible atom with the id of its set, ids following first-atom
/// order.
/// Ids come from the **lowest eligible member index**, not from discovery order
/// or from size. Discovery order depends on the neighbour iteration order, which
/// depends on the cell list's bin layout and so on the frame's origin; size
/// order makes a grain's id change when an unrelated grain grows. Both produce
/// output that looks stable and is not, which is worse than an obviously
/// unstable labelling because it survives a casual eyeball check.
///
/// Ineligible atoms get [`NO_GRAIN`] and do not consume an id. They have to be
/// excluded here rather than filtered afterwards: union-find gives every atom a
/// set, so an atom that was never a candidate still holds a singleton, and
/// filtering its label afterwards leaves a hole in the id space that makes
/// `grain_count` disagree with the largest id seen.
fn label_sets(ds: &mut DisjointSet, eligible: &[bool]) -> (Vec<u32>, usize) {
    // Every atom is its own set, so an ineligible atom has to be hidden before
    // the shared relabeller runs: otherwise its singleton becomes a grain.
    let sets: Vec<u32> = (0..eligible.len() as u32)
        .map(|i| if eligible[i as usize] { i } else { NO_GRAIN })
        .collect();
    canonical_labels(ds, &sets)
}

/// Grain segmentation over an already-PTM-classified frame.
#[derive(Clone, Debug)]
pub struct Grains {
    /// Maximum disorientation, degrees, for two neighbours to be one grain.
    pub misorientation_threshold: f64,
    /// Neighbour cutoff for the bond test.
    ///
    /// `None` derives it from `StructureInteratomicDistance`, and fails loudly
    /// when that is absent rather than guessing a radius — a wrong guess yields
    /// a plausible-looking grain map at the wrong connectivity.
    pub cutoff: Option<f64>,
    /// Discard grains smaller than this.
    ///
    /// Their atoms keep [`NO_GRAIN`] and are reported under
    /// `grain_rejected_atoms`. This is the "these atoms are not a grain and are
    /// not to be counted" knob; [`Grains::min_grain_size`] is the "this cluster
    /// is not a grain, give it to the grain beside it" knob.
    pub min_size: usize,
    /// Dissolve crystallite clusters smaller than this into their neighbours.
    ///
    /// OVITO's `min_grain_size`, whose default is 100 atoms: a crystallite
    /// below it is a dislocation core, a patch of disordered boundary, or the
    /// shell around a vacancy, not a grain. Its atoms lose their own id and are
    /// then handed to the nearest surviving grain (see
    /// [`Grains::orphan_adoption`]). Zero (the default) dissolves nothing.
    pub min_grain_size: usize,
    /// Accept FCC/HCP neighbours across a stacking fault.
    pub stacking_faults: bool,
    /// Give non-crystalline atoms (`StructureType == OTHER`) their own grains.
    ///
    /// Off by default: they have no point group, so their mutual orientation is
    /// undefined, and every rule for an undefined quantity merges the amorphous
    /// phase into a neighbouring grain.
    pub include_other: bool,
    /// Merge grains whose *grain-graph* edge is at or below this disorientation,
    /// in degrees.
    ///
    /// `None` (the default) skips the merging pass.
    pub merge_threshold: Option<f64>,
    /// Hand non-crystalline atoms left without a grain to the nearest adjacent
    /// grain.
    ///
    /// OVITO's `orphan_adoption`, on by default there. The orphans are the
    /// non-crystalline atoms PTM could not classify (the `include_other` =
    /// false population). Atoms of clusters dissolved by
    /// [`Grains::min_grain_size`] are adopted whether or not this is set, as
    /// OVITO's `min_grain_size` merges them into their neighbours. Adopting
    /// the orphans is what makes a polycrystal whose vacancies sit in
    /// disordered shells still report one id per atom: the shell around a hole
    /// belongs to the grain the hole is in, so the census has an atom there to
    /// own the empty site.
    pub orphan_adoption: bool,
}

impl Default for Grains {
    fn default() -> Self {
        Self {
            misorientation_threshold: DEFAULT_MISORIENTATION_THRESHOLD,
            cutoff: None,
            min_size: 1,
            min_grain_size: 0,
            stacking_faults: false,
            include_other: false,
            merge_threshold: None,
            orphan_adoption: false,
        }
    }
}

impl Grains {
    pub fn new(misorientation_threshold: f64) -> Self {
        Self {
            misorientation_threshold,
            ..Default::default()
        }
    }

    /// The bond cutoff, from configuration or from PTM's own output.
    fn resolve_cutoff(&self, store: &PropertyStore) -> Result<f64> {
        if let Some(c) = self.cutoff {
            // `!(c > 0.0)` would also reject NaN, which is wanted, but says so
            // obscurely; the explicit form names both cases.
            if !c.is_finite() || c <= 0.0 {
                return Err(Error::invalid(format!(
                    "grain cutoff must be positive and finite, got {c}"
                )));
            }
            return Ok(c);
        }
        let p = store.get("StructureInteratomicDistance").ok_or_else(|| {
            Error::invalid(
                "grain segmentation needs a bond cutoff: pass one, or run PTM first so \
                 StructureInteratomicDistance is available",
            )
        })?;
        let vals = p
            .as_f64()
            .ok_or_else(|| Error::invalid("StructureInteratomicDistance is not f64"))?;
        // The largest measured nearest-neighbour distance, not the mean: a cutoff
        // below the true maximum severs the bonds of the most expanded atom in
        // the frame, which in a relaxed sample is exactly the thermal tail.
        let max = vals
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(0.0f64, f64::max);
        if max <= 0.0 {
            return Err(Error::invalid(
                "StructureInteratomicDistance holds no positive finite value",
            ));
        }
        Ok(max)
    }

    /// Read and normalise the per-atom inputs.
    ///
    /// Quaternions are normalised here rather than trusted: PTM returns unit
    /// quaternions, but `dot` feeds straight into `acos`, and a quaternion one
    /// ULP long gives an angle error that is not one ULP.
    fn inputs(&self, store: &PropertyStore, n: usize) -> Result<PtmInputs> {
        let st = store
            .get("StructureType")
            .ok_or_else(|| Error::invalid("grain segmentation needs StructureType from PTM"))?;
        let raw = st
            .as_u8()
            .ok_or_else(|| Error::invalid("StructureType is not u8"))?;
        if raw.len() < n {
            return Err(Error::invalid(format!(
                "StructureType has {} entries for {n} atoms",
                raw.len()
            )));
        }
        let types: Vec<i32> = raw[..n].iter().map(|&v| v as i32).collect();

        let ori = store.get("StructureOrientation").ok_or_else(|| {
            Error::invalid("grain segmentation needs StructureOrientation from PTM")
        })?;
        let flat = ori
            .as_f64()
            .ok_or_else(|| Error::invalid("StructureOrientation is not f64"))?;
        if flat.len() < 4 * n {
            return Err(Error::invalid(format!(
                "StructureOrientation has {} values for {n} atoms",
                flat.len()
            )));
        }
        let quats: Vec<Option<[f64; 4]>> = (0..n)
            .map(|i| {
                let mut q = [
                    flat[4 * i],
                    flat[4 * i + 1],
                    flat[4 * i + 2],
                    flat[4 * i + 3],
                ];
                if q.iter().all(|v| v.is_finite()) {
                    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                    if l > 1e-12 {
                        quat_normalize(&mut q);
                        return Some(q);
                    }
                }
                None
            })
            .collect();
        Ok((types, quats))
    }
}

impl Analysis for Grains {
    fn name(&self) -> &'static str {
        "grains"
    }

    /// Empty unless a cutoff was configured.
    ///
    /// The derived cutoff needs `StructureInteratomicDistance`, which only exists
    /// once PTM has run, and [`cutoffs`](Self::cutoffs) is asked before that.
    /// Reporting a radius here that the analysis does not use would size the
    /// shared cell list for nothing; when the cutoff is derived, the analysis
    /// builds its own list at the resolved value instead.
    fn cutoffs(&self) -> Vec<f64> {
        match self.cutoff {
            Some(c) => vec![c],
            None => Vec::new(),
        }
    }

    fn run(
        &self,
        ctx: &AnalysisContext<'_>,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        let n = ctx.frame.natoms();
        let (types, quats) = self.inputs(out, n)?;
        let cutoff = self.resolve_cutoff(out)?;

        // Eligibility first, for every atom, before any pair is considered: the
        // bond test reads its partner's flag, and a partner with a higher index
        // has not been visited yet when the loop reaches `i`.
        let mut eligible = vec![false; n];
        for i in 0..n {
            let has_group = point_group(types[i]).is_some();
            eligible[i] = if has_group {
                quats[i].is_some()
            } else {
                self.include_other
            };
        }

        // Reuse the pipeline's shared list only when it can actually answer a
        // query this wide. A `CellList` sizes its candidate stencil once, at
        // build time, so asking a list built for a narrower cutoff than ours
        // returns a *truncated* neighbour set rather than an error -- and a
        // missing neighbour here silently splits a grain in two, which is
        // exactly the kind of wrong answer that is hard to spot downstream.
        // Anything else gets its own list at the cutoff we actually use.
        let owned;
        let cell = match ctx.nkeys.cell {
            Some(cl) if cutoff <= cl.max_query_cutoff() => cl,
            _ => {
                owned = crate::neighbor::cell_list::CellList::build(
                    &ctx.frame.cell,
                    &ctx.frame.particles.positions[..n],
                    cutoff,
                )?;
                &owned
            }
        };

        let mut ds = DisjointSet::new(n);
        let mut scratch = Vec::new();
        // Grain-graph merge input: the bonds that were *not* taken. Only
        // collected when a merge threshold is set, so the default path pays
        // nothing for a pass it will not run.
        let merging = self.merge_threshold.is_some();
        let mut bridges: Vec<(u32, u32, f64)> = Vec::new();
        let cutoff2 = cutoff * cutoff;
        for i in 0..n {
            if !eligible[i] {
                continue;
            }
            let ti = types[i];
            cell.neighbors_of(i, cutoff, &mut scratch)?;
            for &(j, dr) in scratch.iter() {
                // Each pair once: the test is symmetric, and running it twice
                // costs a second `acos` per bond for no new information.
                if j <= i || !eligible[j] {
                    continue;
                }
                if dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2] > cutoff2 {
                    continue;
                }
                if !types_compatible(ti, types[j], self.stacking_faults) {
                    continue;
                }
                // An eligible atom without an orientation is only reachable
                // under `include_other`, since a type with a point group is
                // eligible only when it has a quaternion. Those atoms have no
                // defined mutual orientation, so type agreement is the whole
                // test -- and `types_compatible` above has already rejected
                // pairing them with a crystalline neighbour.
                let bonded = match (quats[i], quats[j], point_group(types[j])) {
                    (Some(qi), Some(qj), Some(g)) => {
                        let d = disorientation(&qi, &qj, g);
                        if merging {
                            bridges.push((i as u32, j as u32, d));
                        }
                        d <= self.misorientation_threshold
                    }
                    _ => true,
                };
                if bonded {
                    ds.union(i as u32, j as u32);
                }
            }
        }

        let (mut labels, mut n_sets) = label_sets(&mut ds, &eligible);

        // Merge on the grain graph. This fixes over-splitting -- one
        // grain that a low bond threshold came back in pieces -- and cannot undo
        // a bridge, which is already fused in `labels` by this point. Runs
        // before the cleanup below, so fragments rejoined here are not discarded
        // as undersized on the strength of having been split in the first place.
        let mut merge_edges = 0usize;
        if let Some(t) = self.merge_threshold {
            if !t.is_finite() || t < 0.0 {
                return Err(crate::error::Error::InvalidData(format!(
                    "merge_threshold must be non-negative and finite, got {t}"
                )));
            }
            let (l, n, edges) = merge_grains(&labels, n_sets, &bridges, t);
            labels = l;
            n_sets = n;
            merge_edges = edges;
        }

        // Artificial-grain cleanup, in the two forms OVITO distinguishes.
        //
        // `min_size` *rejects* an undersized cluster: its atoms keep `NO_GRAIN`
        // and are reported, because "not a grain" is the honest answer for a
        // fragment born of a misclassified interface, and "in no grain" and "in
        // grain 0" must stay distinguishable downstream.
        //
        // `min_grain_size` *dissolves* one: the atoms still leave the label map,
        // but only so that the adoption pass can hand them to the neighbouring
        // grains. A crystallite below a hundred atoms is the shell around a
        // vacancy or a patch of disordered boundary, not a grain, and counting
        // it as one makes a clean polycrystal read as a hundred grains.
        //
        // A cluster below both thresholds is rejected rather than dissolved:
        // `min_size` is the caller saying those atoms are not to be counted at
        // all. Surviving ids are renumbered contiguously so the count and the
        // largest id agree.
        let mut dropped = 0usize;
        let mut dissolved = 0usize;
        let mut adopt = vec![false; n];
        if self.min_size > 1 || self.min_grain_size > 0 {
            let threshold = self.min_size.max(self.min_grain_size);
            let mut sizes = vec![0usize; n_sets];
            for &l in &labels {
                // `NO_GRAIN` is already in `labels` for atoms that were never
                // candidates, and it is not a set index. Counting it would index
                // a `u32::MAX` element of `sizes`, so the unassigned-atoms case
                // and the cleanup case would panic in combination rather than
                // separately.
                if l != NO_GRAIN {
                    sizes[l as usize] += 1;
                }
            }
            let mut remap = vec![NO_GRAIN; n_sets];
            let mut next = 0u32;
            for (g, &s) in sizes.iter().enumerate() {
                if s >= threshold {
                    remap[g] = next;
                    next += 1;
                }
            }
            for (i, l) in labels.iter_mut().enumerate() {
                // An atom that was never a candidate is already `NO_GRAIN`;
                // leave it alone and do not count it as rejected, since
                // `grain_rejected_atoms` reports what `min_size` discarded, and
                // `grain_other_atoms` already accounts for these.
                if *l == NO_GRAIN {
                    continue;
                }
                let g = *l as usize;
                let m = remap[g];
                *l = m;
                if m == NO_GRAIN {
                    if sizes[g] < self.min_size {
                        dropped += 1;
                    } else {
                        adopt[i] = true;
                        dissolved += 1;
                    }
                }
            }
        }

        // The labels before adoption: the atoms PTM classified into a
        // surviving crystallite. The mean orientation and structure type have
        // to come from these, not from the adopted orphans -- an orphan is by
        // definition not crystalline, and averaging its quaternion (or its
        // `OTHER` type) into the grain would describe neither the grain nor
        // the orphan. The size and the centre, unlike these, count every
        // member.
        let core = labels.clone();

        // The rest of OVITO's orphan population: atoms PTM could not classify.
        // Adopting them is what makes "one id per atom" true for a real
        // polycrystal, whose boundaries are exactly the atoms PTM misses.
        if self.orphan_adoption {
            for (i, &eligible) in eligible.iter().enumerate() {
                if !eligible && labels[i] == NO_GRAIN {
                    adopt[i] = true;
                }
            }
        }

        let adopted = if adopt.iter().any(|&a| a) {
            adopt_orphans(
                &ctx.frame.cell,
                &ctx.frame.particles.positions[..n],
                &mut labels,
                &adopt,
                ADOPTION_REACH * cutoff,
            )?
        } else {
            0
        };

        // What is left is what the adoption could not reach: an atom farther
        // than `ADOPTION_REACH` bonds from any surviving grain, which is a
        // genuine hole in the sample rather than a disordered patch of it.

        // Per-grain table, in one pass over the atoms.
        let n_grains = labels
            .iter()
            .filter(|&&l| l != NO_GRAIN)
            .max()
            .map_or(0, |&m| m as usize + 1);
        let mut sizes = vec![0usize; n_grains];
        let mut centres = vec![[0.0f64; 3]; n_grains];
        let mut quat_sum = vec![[0.0f64; 4]; n_grains];
        let mut type_counts: Vec<std::collections::BTreeMap<i32, usize>> =
            vec![Default::default(); n_grains];
        for i in 0..n {
            let l = labels[i];
            if l == NO_GRAIN {
                continue;
            }
            let g = l as usize;
            // The size and the centre count every atom that belongs to the
            // grain, adopted orphans included: OVITO's grain size depends on
            // the adoption option the same way.
            sizes[g] += 1;
            for (acc, v) in centres[g].iter_mut().zip(ctx.frame.particles.positions[i]) {
                *acc += v;
            }
            let c = core[i];
            if c == NO_GRAIN {
                continue;
            }
            let cg = c as usize;
            if let Some(q) = quats[i] {
                for (acc, v) in quat_sum[cg].iter_mut().zip(q) {
                    *acc += v;
                }
            }
            *type_counts[cg].entry(types[i]).or_insert(0) += 1;
        }
        // Majority vote, not the mean: a structure type is a label, and averaging
        // 1 and 4 gives 2.5, which is HCP, a structure no atom in the grain is.
        // Ties go to the lower type id so the table is reproducible.
        let mut mean_type = vec![structure::OTHER; n_grains];
        for (g, counts) in type_counts.iter().enumerate() {
            if let Some((&t, _)) = counts
                .iter()
                .max_by_key(|(&t, &c)| (c, std::cmp::Reverse(t)))
            {
                mean_type[g] = t;
            }
        }
        for g in 0..n_grains {
            let inv = 1.0 / sizes[g].max(1) as f64;
            for d in 0..3 {
                centres[g][d] *= inv;
            }
            quat_normalize(&mut quat_sum[g]);
            // A grain whose atoms carried no orientation sums to the zero
            // quaternion, which is not an orientation at all. Left as zeros it
            // reads as a real one and puts every angle to it at a confident
            // 180 degrees, so mark it undefined instead.
            let norm = (quat_sum[g][0] * quat_sum[g][0]
                + quat_sum[g][1] * quat_sum[g][1]
                + quat_sum[g][2] * quat_sum[g][2]
                + quat_sum[g][3] * quat_sum[g][3])
                .sqrt();
            if norm <= 1e-12 {
                quat_sum[g] = [f64::NAN; 4];
            }
        }

        let live_total: usize = sizes.iter().sum();
        // Reference grain for the misorientation column: the largest grain that
        // actually has an orientation, ties to the lower id. Restricting the
        // choice to grains with a mean orientation matters -- an amorphous grain
        // can outsize the crystalline ones, and anchoring on it would report NaN
        // for the entire frame. Deterministic either way, and reported in the
        // scalars rather than left implicit, because the column is only
        // interpretable against a known reference.
        let reference = (0..n_grains)
            .filter(|&g| quat_is_finite(&quat_sum[g]))
            .max_by_key(|&g| (sizes[g], std::cmp::Reverse(g)));
        let misorientation: Vec<f64> = match reference {
            None => Vec::new(),
            Some(r) => (0..n_grains)
                .map(|g| match point_group(mean_type[g]) {
                    // Undefined for a grain of atoms with no point group, since
                    // the angles are between quantities that do not exist. NaN
                    // says so; zero would claim they are aligned.
                    None => f64::NAN,
                    Some(grp) => disorientation(&quat_sum[g], &quat_sum[r], grp),
                })
                .collect(),
        };

        // The grain table, one row per grain, row index == grain id. Kept out
        // of the per-atom stream (`per_particle = false`) because it has one row
        // per grain rather than per atom, and the writers filter on that flag.
        out.register(per_grain("GrainSize", DType::I32, &sizes));
        out.register(per_grain("GrainCenter", DType::F64, &centres));
        out.register(per_grain("GrainOrientation", DType::F64, &quat_sum));
        out.register(per_grain("GrainStructureType", DType::I32, &mean_type));
        out.register(per_grain(
            "GrainMisorientation",
            DType::F64,
            &misorientation,
        ));

        out.register_flat(
            DType::I32,
            "Grain",
            1,
            labels
                .iter()
                .map(|&l| {
                    if l == NO_GRAIN {
                        NO_GRAIN_EXPORT as f64
                    } else {
                        l as f64
                    }
                })
                .collect(),
        );

        let largest = sizes.iter().copied().max().unwrap_or(0);
        let mut scalars = vec![
            ("grain_count".to_owned(), n_grains as f64),
            (
                "grain_misorientation_threshold".to_owned(),
                self.misorientation_threshold,
            ),
            ("grain_cutoff".to_owned(), cutoff),
            ("grain_min_size".to_owned(), self.min_size as f64),
            (
                "grain_min_grain_size".to_owned(),
                self.min_grain_size as f64,
            ),
            ("grain_dissolved_atoms".to_owned(), dissolved as f64),
            ("grain_adopted_atoms".to_owned(), adopted as f64),
            (
                "grain_orphan_adoption".to_owned(),
                f64::from(u8::from(self.orphan_adoption)),
            ),
            ("grain_rejected_atoms".to_owned(), dropped as f64),
            ("grain_largest".to_owned(), largest as f64),
            (
                "grain_largest_fraction".to_owned(),
                if n > 0 {
                    largest as f64 / n as f64
                } else {
                    0.0
                },
            ),
            ("grain_unassigned".to_owned(), (n - live_total) as f64),
        ];
        if let Some(r) = reference {
            // Named in the output because the `GrainMisorientation` column is
            // only interpretable against a known reference.
            scalars.push(("grain_reference".to_owned(), r as f64));
        }
        if !self.include_other {
            let others = types.iter().filter(|&&t| t == structure::OTHER).count();
            scalars.push(("grain_other_atoms".to_owned(), others as f64));
        }
        if let Some(t) = self.merge_threshold {
            // Reported so a user can tell a merge pass that did nothing from one
            // that was never asked for: the count alone cannot.
            scalars.push(("grain_merge_threshold".to_owned(), t));
            scalars.push(("grain_merge_edges".to_owned(), merge_edges as f64));
        }
        summary.merge_from(Summary {
            scalars,
            ..Default::default()
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(w: f64, x: f64, y: f64, z: f64) -> [f64; 4] {
        let mut v = [w, x, y, z];
        quat_normalize(&mut v);
        v
    }

    /// 90 degrees about z, as a unit quaternion.
    fn rot90z() -> [f64; 4] {
        q(
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
        )
    }

    #[test]
    fn disorientation_of_an_orientation_with_itself_is_zero() {
        let a = q(0.923_879_532_5, 0.0, 0.0, 0.382_683_432_4);
        assert_eq!(disorientation(&a, &a, &CUBIC), 0.0);
    }

    #[test]
    fn disorientation_quotients_by_the_point_group() {
        // Two orientations differing by a 90-degree cubic rotation describe the
        // same physical orientation, so the disorientation must be zero even
        // though the quaternions are far apart. Without the group minimum this
        // reports 90 degrees and cuts a grain in half.
        let a = q(1.0, 0.0, 0.0, 0.0);
        let b = rot90z();
        let d = disorientation(&a, &b, &CUBIC);
        assert!(d < 1e-6, "symmetry-equivalent orientations read as {d} deg");

        // The same pair against the hexagonal group must *not* vanish. It reads
        // 30 degrees rather than 90 because the group holds -120 degrees about
        // z, which carries a 90-degree rotation to 330 -- i.e. 30 degrees from
        // identity. Still non-zero, which is the property under test: the
        // hexagonal group cannot absorb a cubic 4-fold rotation, so the cubic
        // and hexagonal grains stay separate.
        let h = disorientation(&a, &b, &HCP_GROUP);
        assert!(
            h > 1.0,
            "hexagonal group must not quotient a cubic rotation, read as {h} deg"
        );
        assert!(
            (h - 30.0).abs() < 1e-6,
            "expected 30 deg for the hexagonal group, got {h}"
        );
    }

    #[test]
    fn disorientation_vanishes_between_symmetry_equivalent_generic_orientations() {
        // `qb = qa·g` is the same crystal orientation for every lattice
        // symmetry `g`. Near-identity or z-axis test orientations commute with
        // enough of the group to hide which side the symmetry is applied on, so
        // this uses generic ones.
        let generic = [
            q(
                0.719_846_310_4,
                0.439_385_041_2,
                -0.326_351_822_3,
                0.433_012_701_9,
            ),
            q(
                0.270_598_050_1,
                -0.653_281_482_4,
                0.270_598_050_1,
                0.653_281_482_4,
            ),
            q(0.5, 0.5, -0.5, 0.5),
        ];
        for (name, group) in [("cubic", &CUBIC[..]), ("hcp", &HCP_GROUP[..])] {
            for qa in generic {
                let mut qa = qa;
                quat_normalize(&mut qa);
                for g in group {
                    let qb = quat_rot(qa, *g);
                    let d = disorientation(&qa, &qb, group);
                    assert!(d < 1e-6, "{name}: qa·g read as {d} deg from qa");
                }
            }
        }
    }

    #[test]
    fn disorientation_is_symmetric_and_bounded() {
        let a = q(0.965_925_826_3, 0.0, 0.258_819_045_1, 0.0);
        let b = q(0.965_925_826_3, 0.0, 0.0, 0.258_819_045_1);
        let ab = disorientation(&a, &b, &CUBIC);
        let ba = disorientation(&b, &a, &CUBIC);
        assert!((ab - ba).abs() < 1e-9, "asymmetric: {ab} vs {ba}");
        // The cubic group's maximum is 62.8 degrees; more than that would mean
        // the group search is not covering the coset.
        assert!(
            ab > 0.0 && ab <= 62.9,
            "implausible disorientation {ab} deg"
        );
    }

    #[test]
    fn disorientation_never_returns_nan() {
        // acos of a dot product that rounding pushed above 1 is the classic way
        // this returns NaN, and NaN fails every threshold comparison, so it would
        // keep bonds rather than drop them — a silent pass-everything failure.
        for k in 0..64 {
            let t = k as f64 / 64.0;
            for (a, b) in [
                (q(1.0, t, 0.0, 0.0), q(1.0, 0.0, t, 0.0)),
                (q(1.0, t, t, 0.0), q(t, 1.0, 0.0, t)),
            ] {
                let d = disorientation(&a, &b, &CUBIC);
                assert!(d.is_finite(), "NaN at t={t}");
                assert!((0.0..=180.0).contains(&d), "{d} deg out of range");
            }
        }
    }

    #[test]
    fn types_compatible_is_strict_unless_stacking_faults_is_set() {
        assert!(types_compatible(structure::FCC, structure::FCC, false));
        assert!(!types_compatible(structure::FCC, structure::BCC, false));
        assert!(!types_compatible(structure::FCC, structure::HCP, false));
        assert!(types_compatible(structure::FCC, structure::HCP, true));
        assert!(types_compatible(structure::HCP, structure::FCC, true));
        // Permission for a stacking fault is not a free pass for everything.
        assert!(!types_compatible(structure::BCC, structure::ICO, true));
    }

    #[test]
    fn other_has_no_point_group() {
        assert!(point_group(structure::OTHER).is_none());
        assert_eq!(point_group(structure::FCC).map(|g| g.len()), Some(24));
        assert_eq!(point_group(structure::BCC).map(|g| g.len()), Some(24));
        assert_eq!(point_group(structure::ICO).map(|g| g.len()), Some(60));
        assert_eq!(point_group(structure::HCP).map(|g| g.len()), Some(6));
    }

    #[test]
    fn disjoint_set_unions_and_finds() {
        let mut ds = DisjointSet::new(6);
        assert!(ds.union(0, 1));
        assert!(ds.union(1, 2));
        assert!(!ds.union(0, 2), "re-union must report no change");
        assert_eq!(ds.find(0), ds.find(2));
        assert!(ds.union(3, 4));
        assert_ne!(ds.find(0), ds.find(3));
    }

    #[test]
    fn disjoint_set_survives_a_long_chain() {
        // Union in an order that builds a deep tree without union-by-size
        // protection, then confirm every atom still lands in one set. Guards
        // the path-halving loop against an off-by-one that would spin or panic.
        let n = 1000;
        let mut ds = DisjointSet::new(n);
        for i in 1..n {
            ds.union(0, i as u32);
        }
        let root = ds.find(0);
        for i in 0..n {
            assert_eq!(ds.find(i as u32), root, "atom {i} is in another set");
        }
    }

    #[test]
    fn labels_are_deterministic_and_follow_first_index() {
        let build = || {
            let mut ds = DisjointSet::new(5);
            ds.union(3, 4);
            ds.union(0, 1);
            label_sets(&mut ds, &[true; 5]).0
        };
        let a = build();
        assert_eq!(a, build(), "labels are not deterministic");
        // Ids follow the lowest member index of each set, not the order the
        // unions arrived in: {0,1} -> 0, then the singleton {2} -> 1, then
        // {3,4} -> 2. Atom 3 has a higher index than atom 2's set but sits in
        // the set unioned first, so discovery order would have put {3,4} at id
        // 1 and this would catch it.
        assert_eq!(a, vec![0, 0, 1, 2, 2]);
    }

    #[test]
    fn ineligible_atoms_get_no_grain_and_consume_no_id() {
        // An atom that was never a candidate still holds a singleton set, so if
        // labelling ignored the mask it would be handed a real grain id and the
        // ids would not be contiguous. Atom 2 is ineligible and sits between
        // two eligible ones, which is where a leaked id would show up.
        let mut ds = DisjointSet::new(4);
        ds.union(0, 1);
        ds.union(2, 3);
        let (labels, count) = label_sets(&mut ds, &[true, true, false, true]);
        assert_eq!(labels[2], NO_GRAIN, "ineligible atom was labelled");
        assert_eq!(count, 2, "ineligible atom consumed a grain id");
        assert_eq!(labels[3], 1, "ids must stay contiguous: {labels:?}");
    }

    #[test]
    fn cutoff_resolution_rejects_a_nonpositive_value() {
        let g = Grains {
            cutoff: Some(-1.0),
            ..Default::default()
        };
        let err = g
            .resolve_cutoff(&PropertyStore::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("positive"), "{err}");
    }

    #[test]
    fn cutoff_falls_back_to_the_largest_interatomic_distance() {
        let g = Grains::default();
        let mut store = PropertyStore::new();
        store.register_flat(
            DType::F64,
            "StructureInteratomicDistance",
            1,
            vec![2.4, 2.9, 2.6, f64::NAN, 0.0],
        );
        let c = g.resolve_cutoff(&store).unwrap();
        assert!(
            (c - 2.9).abs() < 1e-12,
            "expected the max of the finite positive values, got {c}"
        );
    }

    #[test]
    fn cutoff_without_ptm_output_is_an_error_not_a_guess() {
        let g = Grains::default();
        let err = g
            .resolve_cutoff(&PropertyStore::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("cutoff"), "{err}");
    }

    /// End-to-end through [`Grains::run`]: the graph, the disorientation test
    /// and the labelling, on fixtures whose answer is known by hand.
    /// Grain-graph merging, on hand-built label and bridge lists so the
    /// decision rule is tested without a geometric fixture.
    mod merging {
        use super::*;

        /// `labels[i]` for `n` atoms, `bridges` as (i, j, angle) triples.
        fn run(
            labels: &[u32],
            n_sets: usize,
            bridges: &[(u32, u32, f64)],
            threshold: f64,
        ) -> (Vec<u32>, usize, usize) {
            merge_grains(labels, n_sets, bridges, threshold)
        }

        #[test]
        fn an_edge_above_the_threshold_keeps_the_grains_apart() {
            let labels = vec![0, 0, 1, 1];
            let (out, n, edges) = run(&labels, 2, &[(0, 2, 30.0)], 4.0);
            assert_eq!(n, 2);
            assert_eq!(edges, 0);
            assert_eq!(out, vec![0, 0, 1, 1]);
        }

        #[test]
        fn an_edge_at_or_below_the_threshold_merges_them() {
            let labels = vec![0, 0, 1, 1];
            let (out, n, edges) = run(&labels, 2, &[(0, 2, 4.0)], 4.0);
            assert_eq!(n, 1);
            assert_eq!(edges, 1);
            // Every atom follows, and the merged grain keeps the id that started
            // earliest -- the same convention `label_sets` uses.
            assert_eq!(out, vec![0, 0, 0, 0]);
        }

        /// The reason the minimum is taken rather than the mean or the
        /// bridging bond. A boundary that is clean everywhere except for one
        /// noisy pair is a real grain boundary that a measurement split, and the
        /// mean over that pair's bonds would call it a strong boundary forever.
        #[test]
        fn the_minimum_is_what_decides_not_the_mean() {
            let labels = vec![0, 0, 1, 1];
            // One near-zero bond and one at 30 degrees. Mean is 15.
            let (out, n, _) = run(&labels, 2, &[(0, 2, 0.5), (1, 3, 30.0)], 1.0);
            assert_eq!(n, 1, "the 0.5 degree bond should merge them");
            assert_eq!(out, vec![0, 0, 0, 0]);
        }

        #[test]
        fn a_transitive_chain_of_low_edges_collapses_to_one() {
            // 0-1-2-3 in a chain, each link weak. One pass must reach the same
            // fixed point an iterative agglomeration would.
            let labels = vec![0, 1, 2, 3];
            let (out, n, edges) = run(&labels, 4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)], 4.0);
            assert_eq!(n, 1);
            assert_eq!(edges, 3);
            assert_eq!(out, vec![0, 0, 0, 0]);
        }

        #[test]
        fn two_weak_links_do_not_join_through_a_strong_one() {
            let labels = vec![0, 1, 2];
            let (out, n, _) = run(&labels, 3, &[(0, 1, 1.0), (1, 2, 50.0)], 4.0);
            assert_eq!(n, 2);
            assert_eq!(out, vec![0, 0, 1]);
        }

        #[test]
        fn an_atom_in_no_grain_bounds_nothing() {
            // Atom 1 is unassigned. Its bonds must not pull grains 0 and 2
            // together, and must not be counted as an edge.
            let labels = vec![0, NO_GRAIN, 2];
            let (out, n, edges) = run(&labels, 3, &[(0, 1, 0.1), (1, 2, 0.1)], 4.0);
            assert_eq!(n, 2);
            assert_eq!(edges, 0);
            assert_eq!(out, vec![0, NO_GRAIN, 1]);
        }

        #[test]
        fn an_unmeasurable_angle_never_merges() {
            // NaN is an orientation that failed to measure, not a small angle.
            let labels = vec![0, 1];
            let (out, n, _) = run(&labels, 2, &[(0, 1, f64::NAN)], 1e9);
            assert_eq!(n, 2);
            assert_eq!(out, vec![0, 1]);
        }

        #[test]
        fn ids_are_handed_out_in_first_atom_order_not_root_order() {
            // Union-find will make 3 the root (union by size), but the grain
            // holding atom 0 must still come first in the output numbering.
            let labels = vec![0, 1, 1, 1];
            let (out, n, _) = run(&labels, 4, &[(0, 1, 1.0)], 4.0);
            assert_eq!(n, 1);
            assert_eq!(out, vec![0, 0, 0, 0]);

            // And when the merged grain starts later, it takes the later id.
            let labels = vec![5, 5, 7];
            let (out, n, _) = run(&labels, 8, &[(0, 2, 1.0)], 4.0);
            assert_eq!(n, 1);
            assert_eq!(out, vec![0, 0, 0]);
        }

        #[test]
        fn the_threshold_is_inclusive_at_the_boundary() {
            let labels = vec![0, 1];
            let (_, n_at, _) = run(&labels, 2, &[(0, 1, 4.0)], 4.0);
            assert_eq!(n_at, 1, "at the threshold must merge");
            let (_, n_above, _) = run(&labels, 2, &[(0, 1, 4.001)], 4.0);
            assert_eq!(n_above, 2, "just above must not");
        }

        #[test]
        fn one_grain_touching_itself_is_not_an_edge() {
            // Bonds within a grain are dropped before the decision, so a grain
            // with many internal bonds does not report edges merged with itself.
            let labels = vec![0, 0, 0];
            let (_, n, edges) = run(&labels, 1, &[(0, 1, 0.0), (1, 2, 0.0)], 4.0);
            assert_eq!(n, 1);
            assert_eq!(edges, 0);
        }

        #[test]
        fn no_bridges_means_no_change() {
            let labels = vec![0, 0, 1, 1, 2, 2];
            let (out, n, edges) = run(&labels, 3, &[], 4.0);
            assert_eq!((n, edges), (3, 0));
            assert_eq!(out, labels);
        }

        #[test]
        fn a_zero_threshold_merges_exactly_coincident_grains() {
            let labels = vec![0, 1];
            let (_, n, _) = run(&labels, 2, &[(0, 1, 0.0)], 0.0);
            assert_eq!(n, 1);
        }
    }

    mod segmentation {
        use super::*;
        use crate::analysis::NeighborhoodContext;
        use crate::cell::UnitCell;
        use crate::data::frame::Frame;
        use crate::data::particle::ParticleData;
        use crate::exec::Par;

        /// Frame of `positions`, with the PTM outputs `run` reads.
        ///
        /// Returns the store separately rather than hiding it: it stands in for
        /// whatever ran PTM, and the analysis must read its inputs from the
        /// shared store, not from the frame.
        fn fixture(
            cell: UnitCell,
            positions: &[[f64; 3]],
            types: &[i32],
            quats: &[[f64; 4]],
        ) -> (Frame, PropertyStore) {
            let n = positions.len();
            assert_eq!(types.len(), n);
            assert_eq!(quats.len(), n);
            let mut p = ParticleData::with_capacity(n);
            for (i, q) in positions.iter().enumerate() {
                p.add_atom(*q, i as i64 + 1, 1);
            }
            let frame = Frame::new(cell, p);
            let mut store = PropertyStore::new();
            store.register_flat(
                DType::U8,
                "StructureType",
                1,
                types.iter().map(|&t| t as f64).collect(),
            );
            store.register_flat(
                DType::F64,
                "StructureOrientation",
                4,
                quats.iter().flatten().copied().collect(),
            );
            store.register_flat(DType::F64, "StructureInteratomicDistance", 1, vec![1.0; n]);
            (frame, store)
        }

        /// A periodic chain along x, spacing 1.0, the first `split` atoms
        /// orientation `qa` and the rest `qb`. All FCC.
        ///
        /// A chain is legitimate input: this module keys off neighbours,
        /// structure type and orientation, never off coordination, so a real
        /// lattice would only add noise between the fixture and the assertion.
        /// y and z are wide enough that no off-chain bond can form, which keeps
        /// the expected answer readable from the geometry alone; x is periodic,
        /// so the chain closes on itself.
        fn chain(n: usize, split: usize, qa: [f64; 4], qb: [f64; 4]) -> (Frame, PropertyStore) {
            let pos: Vec<[f64; 3]> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
            let quats: Vec<[f64; 4]> = (0..n).map(|i| if i < split { qa } else { qb }).collect();
            fixture(
                UnitCell::orthorhombic(n as f64, 20.0, 20.0),
                &pos,
                &vec![structure::FCC; n],
                &quats,
            )
        }

        /// Run `g` over `frame` against `store`, returning `Grain` and scalars.
        fn segment(
            frame: &Frame,
            mut store: PropertyStore,
            g: &Grains,
        ) -> (Vec<i32>, Vec<(String, f64)>) {
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut summary = Summary::default();
            g.run(&ctx, &mut store, &mut summary).unwrap();
            let labels = store.get("Grain").unwrap().as_i32().unwrap().clone();
            assert_eq!(labels.len(), frame.natoms(), "Grain has the wrong length");
            (labels, summary.scalars)
        }

        fn scalar(scalars: &[(String, f64)], key: &str) -> f64 {
            scalars
                .iter()
                .find(|(k, _)| k == key)
                .unwrap_or_else(|| panic!("missing scalar {key}"))
                .1
        }

        fn distinct(labels: &[i32]) -> usize {
            let mut v = labels.to_vec();
            v.sort_unstable();
            v.dedup();
            v.len()
        }

        /// Rotation of `deg` degrees about z.
        fn about_z(deg: f64) -> [f64; 4] {
            let h = deg.to_radians() / 2.0;
            q(h.cos(), 0.0, 0.0, h.sin())
        }

        #[test]
        fn a_uniform_chain_is_one_grain() {
            let id = q(1.0, 0.0, 0.0, 0.0);
            let (f, s) = chain(10, 5, id, id);
            let (labels, sc) = segment(&f, s, &Grains::new(4.0));
            assert_eq!(distinct(&labels), 1, "uniform chain split: {labels:?}");
            assert_eq!(scalar(&sc, "grain_count"), 1.0);
            assert_eq!(scalar(&sc, "grain_largest"), 10.0);
            assert_eq!(scalar(&sc, "grain_unassigned"), 0.0);
        }

        #[test]
        fn a_misorientation_beyond_the_threshold_splits_the_chain() {
            // 30 degrees about z: past a 4-degree threshold and not a cubic
            // symmetry rotation, so the two halves must not bond.
            let (f, s) = chain(10, 5, q(1.0, 0.0, 0.0, 0.0), about_z(30.0));
            let (labels, sc) = segment(&f, s, &Grains::new(4.0));
            assert_eq!(distinct(&labels), 2, "chain did not split: {labels:?}");
            assert_eq!(scalar(&sc, "grain_count"), 2.0);
            // Ids follow the first atom, so the leading half is grain 0.
            assert!(labels[..5].iter().all(|&l| l == 0), "{labels:?}");
            assert!(labels[5..].iter().all(|&l| l == 1), "{labels:?}");
            assert_eq!(scalar(&sc, "grain_largest"), 5.0);
        }

        #[test]
        fn a_symmetry_equivalent_orientation_does_not_split_the_chain() {
            // The same fixture with the second half rotated 90 degrees about z, a
            // cubic 4-fold rotation. The physical orientation is unchanged, so
            // the chain stays one grain. This is what the point group search is
            // for: comparing raw quaternions reads 90 degrees here and shatters
            // the grain.
            let (f, s) = chain(10, 5, q(1.0, 0.0, 0.0, 0.0), about_z(90.0));
            let (labels, sc) = segment(&f, s, &Grains::new(4.0));
            assert_eq!(
                distinct(&labels),
                1,
                "symmetry-equivalent halves split: {labels:?}"
            );
            assert_eq!(scalar(&sc, "grain_count"), 1.0);
        }

        #[test]
        fn the_threshold_is_honoured_from_both_sides() {
            // 10 degrees: splits at a 4-degree threshold, joins at 15. Pins the
            // comparison to the configured value, so a threshold that is read
            // once and then ignored cannot pass.
            let mk = || chain(10, 5, q(1.0, 0.0, 0.0, 0.0), about_z(10.0));
            let (f1, s1) = mk();
            let tight = segment(&f1, s1, &Grains::new(4.0));
            let (f2, s2) = mk();
            let loose = segment(&f2, s2, &Grains::new(15.0));
            assert_eq!(distinct(&tight.0), 2, "4 deg should split");
            assert_eq!(distinct(&loose.0), 1, "15 deg should join");
        }

        #[test]
        fn type_mismatch_blocks_the_bond_unless_stacking_faults_is_allowed() {
            // Identical orientations, so only the structure types differ: FCC
            // against HCP. Strict mode must split; with `stacking_faults` the same
            // fixture joins, which is the pair OVITO allows across a stacking
            // fault. Note the HCP atoms need an orientation, since a type with no
            // quaternion would be unassigned for an unrelated reason and the
            // test would pass for the wrong reason.
            let pos: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
            let types: Vec<i32> = (0..10)
                .map(|i| {
                    if i < 5 {
                        structure::FCC
                    } else {
                        structure::HCP
                    }
                })
                .collect();
            let quats = vec![[1.0, 0.0, 0.0, 0.0]; 10];
            let run = |faults: bool| {
                let (f, s) = fixture(
                    UnitCell::orthorhombic(10.0, 20.0, 20.0),
                    &pos,
                    &types,
                    &quats,
                );
                let g = Grains {
                    stacking_faults: faults,
                    ..Grains::new(4.0)
                };
                distinct(&segment(&f, s, &g).0)
            };
            assert_eq!(run(false), 2, "FCC/HCP must not bond strictly");
            assert_eq!(run(true), 1, "stacking fault must bond");
        }

        #[test]
        fn other_atoms_are_unassigned_by_default() {
            // Trailing half is OTHER with NaN orientations. With `include_other`
            // off those atoms are in no grain: they must not be folded into
            // grain 0, which would quietly enlarge it, and must not become their
            // own grain either, since their mutual orientation is undefined.
            let pos: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
            let types: Vec<i32> = (0..10)
                .map(|i| {
                    if i < 5 {
                        structure::FCC
                    } else {
                        structure::OTHER
                    }
                })
                .collect();
            let quats: Vec<[f64; 4]> = (0..10)
                .map(|i| {
                    if i < 5 {
                        [1.0, 0.0, 0.0, 0.0]
                    } else {
                        [f64::NAN; 4]
                    }
                })
                .collect();
            let (f, s) = fixture(
                UnitCell::orthorhombic(10.0, 20.0, 20.0),
                &pos,
                &types,
                &quats,
            );
            let (labels, sc) = segment(&f, s, &Grains::new(4.0));
            assert!(labels[..5].iter().all(|&l| l == 0), "{labels:?}");
            assert!(
                labels[5..].iter().all(|&l| l == NO_GRAIN_EXPORT),
                "OTHER atoms must be unassigned, got {labels:?}"
            );
            assert_eq!(scalar(&sc, "grain_count"), 1.0);
            assert_eq!(scalar(&sc, "grain_largest"), 5.0);
            assert_eq!(scalar(&sc, "grain_other_atoms"), 5.0);
            assert_eq!(scalar(&sc, "grain_unassigned"), 5.0);
        }

        #[test]
        fn include_other_gives_them_their_own_grain() {
            // The opposite choice, on the same fixture: with `include_other` the
            // OTHER atoms form a grain of their own rather than being dropped,
            // so the total assigned count still accounts for every atom.
            let pos: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
            let types: Vec<i32> = (0..10)
                .map(|i| {
                    if i < 5 {
                        structure::FCC
                    } else {
                        structure::OTHER
                    }
                })
                .collect();
            let quats: Vec<[f64; 4]> = (0..10)
                .map(|i| {
                    if i < 5 {
                        [1.0, 0.0, 0.0, 0.0]
                    } else {
                        [f64::NAN; 4]
                    }
                })
                .collect();
            let g = Grains {
                include_other: true,
                ..Grains::new(4.0)
            };
            let (f, s) = fixture(
                UnitCell::orthorhombic(10.0, 20.0, 20.0),
                &pos,
                &types,
                &quats,
            );
            let (labels, sc) = segment(&f, s, &g);
            assert_eq!(distinct(&labels), 2, "{labels:?}");
            assert_eq!(scalar(&sc, "grain_count"), 2.0);
            assert_eq!(scalar(&sc, "grain_unassigned"), 0.0);
        }

        #[test]
        fn min_size_rejects_small_grains_and_renumbers() {
            // 10 atoms split 3/7 at a misorientation. With `min_size` = 5 the
            // 3-atom grain goes to `NO_GRAIN` and the survivor is renumbered to
            // id 0, so `grain_count` and the largest id cannot disagree.
            let (f, s) = chain(10, 3, q(1.0, 0.0, 0.0, 0.0), about_z(30.0));
            let g = Grains {
                min_size: 5,
                ..Grains::new(4.0)
            };
            let (labels, sc) = segment(&f, s, &g);
            assert!(
                labels[..3].iter().all(|&l| l == NO_GRAIN_EXPORT),
                "small grain must be rejected, got {labels:?}"
            );
            assert!(
                labels[3..].iter().all(|&l| l == 0),
                "survivor must be renumbered to 0, got {labels:?}"
            );
            assert_eq!(scalar(&sc, "grain_count"), 1.0);
            assert_eq!(scalar(&sc, "grain_largest"), 7.0);
            assert_eq!(scalar(&sc, "grain_rejected_atoms"), 3.0);
            assert_eq!(scalar(&sc, "grain_unassigned"), 3.0);
        }

        #[test]
        fn min_grain_size_dissolves_and_absorbs_rather_than_rejects() {
            // The same 3/7 split, but now the 3-atom cluster is *dissolved*:
            // it is still not a grain, yet its atoms come back as members of
            // the surviving grain, because OVITO's `min_grain_size` merges a
            // dissolved crystallite into its neighbours. That is independent of
            // `orphan_adoption`, which governs the *non-crystalline* atoms --
            // and it is the whole difference from `min_size` above, where the
            // rejected atoms stay at -1.
            //
            // The flood fill has to make several passes: the far end of the
            // cluster is more than one reach from the surviving grain.
            let (f, s) = chain(10, 3, q(1.0, 0.0, 0.0, 0.0), about_z(30.0));
            let g = Grains {
                min_grain_size: 5,
                ..Grains::new(4.0)
            };
            let (labels, sc) = segment(&f, s, &g);
            assert!(
                labels.iter().all(|&l| l == 0),
                "every atom must end up in the surviving grain, got {labels:?}"
            );
            assert_eq!(scalar(&sc, "grain_count"), 1.0);
            assert_eq!(scalar(&sc, "grain_largest"), 10.0);
            assert_eq!(scalar(&sc, "grain_dissolved_atoms"), 3.0);
            assert_eq!(scalar(&sc, "grain_adopted_atoms"), 3.0);
            assert_eq!(scalar(&sc, "grain_rejected_atoms"), 0.0);
            assert_eq!(scalar(&sc, "grain_unassigned"), 0.0);
        }

        #[test]
        fn min_size_rejection_wins_over_min_grain_size_dissolution() {
            // Both thresholds are set and the 3-atom cluster is below both. It
            // is *rejected*, not dissolved: `min_size` is the caller saying
            // those atoms are not to be counted, and adoption must not quietly
            // overrule that.
            let (f, s) = chain(10, 3, q(1.0, 0.0, 0.0, 0.0), about_z(30.0));
            let g = Grains {
                min_size: 5,
                min_grain_size: 5,
                orphan_adoption: true,
                ..Grains::new(4.0)
            };
            let (labels, sc) = segment(&f, s, &g);
            assert!(
                labels[..3].iter().all(|&l| l == NO_GRAIN_EXPORT),
                "{labels:?}"
            );
            assert!(labels[3..].iter().all(|&l| l == 0), "{labels:?}");
            assert_eq!(scalar(&sc, "grain_rejected_atoms"), 3.0);
            assert_eq!(scalar(&sc, "grain_dissolved_atoms"), 0.0);
            assert_eq!(scalar(&sc, "grain_adopted_atoms"), 0.0);
            assert_eq!(scalar(&sc, "grain_unassigned"), 3.0);
        }

        #[test]
        fn orphan_adoption_folds_unclassified_atoms_into_the_grain() {
            // The other orphan population: atoms PTM could not classify. They
            // are ineligible with `include_other` off, so without adoption they
            // stay at -1 even though the whole frame is one crystalline block.
            // These are the atoms a Wigner-Seitz census needs when a vacancy
            // removes a neighbour and PTM loses the shell.
            let n = 9;
            let pos: Vec<[f64; 3]> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
            let mut types = vec![structure::FCC; n];
            let mut quats = vec![[1.0, 0.0, 0.0, 0.0]; n];
            for i in 3..6 {
                types[i] = structure::OTHER;
                quats[i] = [f64::NAN; 4];
            }
            let (f, s) = fixture(
                UnitCell::orthorhombic(n as f64, 20.0, 20.0),
                &pos,
                &types,
                &quats,
            );
            let g = Grains {
                orphan_adoption: true,
                ..Grains::new(4.0)
            };
            let (labels, sc) = segment(&f, s, &g);
            assert!(
                labels.iter().all(|&l| l == 0),
                "the unclassified run must join the grain it interrupts, got {labels:?}"
            );
            // The *whole* frame is one grain after adoption, including the
            // three atoms that are not crystalline; the interior type and
            // orientation still come from the crystalline six.
            assert_eq!(scalar(&sc, "grain_count"), 1.0);
            assert_eq!(scalar(&sc, "grain_largest"), 9.0);
            assert_eq!(scalar(&sc, "grain_unassigned"), 0.0);
            assert!(scalar(&sc, "grain_adopted_atoms") >= 3.0);
        }

        #[test]
        fn adoption_does_not_reach_across_a_gap() {
            // An orphan farther than the reach from every grain must stay
            // unassigned: adoption is "hand it to the grain beside it", not
            // "invent a grain for it". The isolated atom sits 10 A from the
            // chain, well past 1.75 bond lengths.
            let mut pos: Vec<[f64; 3]> = (0..5).map(|i| [i as f64, 0.0, 0.0]).collect();
            pos.push([15.0, 0.0, 0.0]);
            let mut types = vec![structure::FCC; 6];
            types[5] = structure::OTHER;
            let mut quats = vec![[1.0, 0.0, 0.0, 0.0]; 6];
            quats[5] = [f64::NAN; 4];
            let (f, s) = fixture(
                UnitCell::orthorhombic(40.0, 40.0, 40.0),
                &pos,
                &types,
                &quats,
            );
            let g = Grains {
                orphan_adoption: true,
                ..Grains::new(4.0)
            };
            let (labels, sc) = segment(&f, s, &g);
            assert_eq!(labels[5], NO_GRAIN_EXPORT, "{labels:?}");
            assert_eq!(scalar(&sc, "grain_unassigned"), 1.0);
        }

        #[test]
        fn disconnected_chains_stay_separate() {
            // Two chains with a gap wider than the cutoff, same orientation
            // throughout. Type and orientation agree everywhere, so only the
            // bond test keeps them apart; a pass that skipped the distance check
            // would report one grain for the whole frame.
            let n = 6;
            let mut pos = Vec::new();
            for i in 0..n {
                pos.push([i as f64, 0.0, 0.0]);
                pos.push([20.0 + i as f64, 0.0, 0.0]);
            }
            let types = vec![structure::FCC; 2 * n];
            let quats = vec![[1.0, 0.0, 0.0, 0.0]; 2 * n];
            let (f, s) = fixture(
                UnitCell::orthorhombic(40.0, 40.0, 40.0),
                &pos,
                &types,
                &quats,
            );
            let (labels, sc) = segment(&f, s, &Grains::new(4.0));
            assert_eq!(distinct(&labels), 2, "gap not respected: {labels:?}");
            assert_eq!(scalar(&sc, "grain_count"), 2.0);
            assert_eq!(scalar(&sc, "grain_largest"), n as f64);
        }

        #[test]
        fn a_narrow_shared_list_is_not_trusted() {
            // A `CellList` sizes its candidate stencil once, at build time, so
            // asking a narrow list about wider neighbours returns a *truncated*
            // set -- silently, with no error. Grains must not inherit such a
            // list, because a missing neighbour splits a grain in two and that
            // wrong answer is hard to spot downstream.
            //
            // The chain is 50 atoms 1.9 A apart along x, so consecutive atoms
            // land two bins apart in a list built with 1.0 A bins: a +/-1
            // stencil cannot see them. Enough atoms are used to keep the
            // one-bin-per-atom cap from raising the bin width back past the
            // 2.0 A cutoff (on a two-atom fixture the cap would rescue the
            // narrow list and this test would pass either way).
            let n = 50usize;
            let step = 1.9;
            let pos: Vec<[f64; 3]> = (0..n).map(|i| [0.95 + i as f64 * step, 0.0, 0.0]).collect();
            let id = q(1.0, 0.0, 0.0, 0.0);
            let quats = vec![id; n];
            let (f, s) = fixture(
                UnitCell::orthorhombic(0.95 + n as f64 * step, 1.0, 1.0),
                &pos,
                &vec![structure::FCC; n],
                &quats,
            );

            let narrow = crate::neighbor::cell_list::CellList::build_with_binsize(
                &f.cell,
                &f.particles.positions,
                0.5,
                1.0,
            )
            .unwrap();
            assert!(
                narrow.max_query_cutoff() < 2.0,
                "test needs a genuinely too-narrow list, got {}",
                narrow.max_query_cutoff()
            );

            let nk = NeighborhoodContext {
                cell: Some(&narrow),
            };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut store = s;
            let mut sc = Summary::default();
            Grains {
                cutoff: Some(2.0),
                ..Grains::new(4.0)
            }
            .run(&ctx, &mut store, &mut sc)
            .unwrap();

            let labels = store.get("Grain").unwrap().as_i32().unwrap().clone();
            assert_eq!(distinct(&labels), 1, "truncated stencil split: {labels:?}");
            assert_eq!(scalar(&sc.scalars, "grain_count"), 1.0);
        }

        #[test]
        fn the_grain_table_has_one_row_per_grain() {
            // The grain table. Two grains of 3 and 7 atoms at x = 0..2 and x = 3..9,
            // so the centroids are checkable by hand: (1,0,0) and (6,0,0).
            let (f, s) = chain(10, 3, q(1.0, 0.0, 0.0, 0.0), about_z(30.0));
            let mut store = s;
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut sc = Summary::default();
            Grains::new(4.0).run(&ctx, &mut store, &mut sc).unwrap();

            let size = store.get("GrainSize").unwrap();
            assert!(!size.per_particle, "grain table must not be per-particle");
            assert_eq!(size.as_i32().unwrap().as_slice(), &[3, 7]);

            let centre = store.get("GrainCenter").unwrap();
            assert_eq!(centre.components, 3);
            let c = centre.as_f64().unwrap();
            assert_eq!(c.len(), 2 * 3, "two grains of three components");
            assert!(
                (c[0] - 1.0).abs() < 1e-12 && c[1].abs() < 1e-12 && c[2].abs() < 1e-12,
                "grain 0 centroid: {:?}",
                &c[..3]
            );
            assert!(
                (c[3] - 6.0).abs() < 1e-12 && c[4].abs() < 1e-12 && c[5].abs() < 1e-12,
                "grain 1 centroid: {:?}",
                &c[3..]
            );

            let ori = store.get("GrainOrientation").unwrap();
            assert_eq!(ori.components, 4);
            let o = ori.as_f64().unwrap();
            // Each grain is internally uniform, so its mean is its own
            // orientation, renormalized.
            for row in o.chunks_exact(4) {
                let l = (row.iter().map(|v| v * v).sum::<f64>()).sqrt();
                assert!((l - 1.0).abs() < 1e-12, "mean quaternion not unit: {row:?}");
            }

            assert_eq!(
                store
                    .get("GrainStructureType")
                    .unwrap()
                    .as_i32()
                    .unwrap()
                    .as_slice(),
                &[structure::FCC, structure::FCC]
            );
            // Reference is the largest grain, id 1, and grain 0 is 30 degrees
            // from it while the reference is 0 from itself.
            assert_eq!(scalar(&sc.scalars, "grain_reference"), 1.0);
            let mo = store.get("GrainMisorientation").unwrap().as_f64().unwrap();
            assert!((mo[1]).abs() < 1e-6, "reference is {mo:?} from itself");
            assert!((mo[0] - 30.0).abs() < 1e-6, "grain 0 reads {mo:?}");
        }

        #[test]
        fn mean_structure_type_is_a_majority_vote_not_an_average() {
            // FCC and HCP in one grain, which is only reachable with
            // `stacking_faults` -- FCC and BCC can never share a grain, since
            // the type test blocks the bond outright, so a "mean type" that
            // averaged mixed labels could only ever arise across a stacking
            // fault. Five FCC against three HCP must report FCC: averaging the
            // labels gives 1.375, which is not a structure, and rounding it
            // lands on HCP, a structure only three atoms actually have.
            let n = 8;
            let pos: Vec<[f64; 3]> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
            let types: Vec<i32> = (0..n)
                .map(|i| {
                    if i < 5 {
                        structure::FCC
                    } else {
                        structure::HCP
                    }
                })
                .collect();
            let quats = vec![[1.0, 0.0, 0.0, 0.0]; n];
            let (f, mut store) = fixture(
                UnitCell::orthorhombic(n as f64, 20.0, 20.0),
                &pos,
                &types,
                &quats,
            );
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut sc = Summary::default();
            Grains {
                stacking_faults: true,
                ..Grains::new(4.0)
            }
            .run(&ctx, &mut store, &mut sc)
            .unwrap();
            let labels = store.get("Grain").unwrap().as_i32().unwrap().clone();
            assert!(
                labels.iter().all(|&l| l == 0),
                "expected one stacking-fault grain: {labels:?}"
            );
            assert_eq!(
                store
                    .get("GrainStructureType")
                    .unwrap()
                    .as_i32()
                    .unwrap()
                    .as_slice(),
                &[structure::FCC],
                "5:3 FCC majority must report FCC"
            );
        }

        #[test]
        fn a_type_tie_reports_the_lower_type_id() {
            // One FCC and one HCP: no majority, so the tie-break decides. It has
            // to be the lower id for the table to be reproducible -- picking
            // whichever the map happened to yield first would make the reported
            // type depend on iteration order.
            let n = 2;
            let pos: Vec<[f64; 3]> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
            let types = vec![structure::HCP, structure::FCC];
            let quats = vec![[1.0, 0.0, 0.0, 0.0]; n];
            let (f, mut store) = fixture(
                UnitCell::orthorhombic(2.0, 20.0, 20.0),
                &pos,
                &types,
                &quats,
            );
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut sc = Summary::default();
            Grains {
                stacking_faults: true,
                ..Grains::new(4.0)
            }
            .run(&ctx, &mut store, &mut sc)
            .unwrap();
            assert_eq!(
                store
                    .get("GrainStructureType")
                    .unwrap()
                    .as_i32()
                    .unwrap()
                    .as_slice(),
                &[structure::FCC],
                "a 1:1 tie must resolve to the lower type id"
            );
        }

        #[test]
        fn misorientation_is_nan_where_no_point_group_exists() {
            // A grain of OTHER atoms has no point group, so its disorientation
            // to the reference is undefined. NaN reports that; zero would claim
            // the grain is aligned with the reference.
            let pos: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
            let types: Vec<i32> = (0..10)
                .map(|i| {
                    if i < 5 {
                        structure::OTHER
                    } else {
                        structure::FCC
                    }
                })
                .collect();
            let quats: Vec<[f64; 4]> = (0..10)
                .map(|i| {
                    if i < 5 {
                        [f64::NAN; 4]
                    } else {
                        [1.0, 0.0, 0.0, 0.0]
                    }
                })
                .collect();
            let g = Grains {
                include_other: true,
                ..Grains::new(4.0)
            };
            let (f, mut store) = fixture(
                UnitCell::orthorhombic(10.0, 20.0, 20.0),
                &pos,
                &types,
                &quats,
            );
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut sc = Summary::default();
            g.run(&ctx, &mut store, &mut sc).unwrap();
            let mo = store
                .get("GrainMisorientation")
                .unwrap()
                .as_f64()
                .unwrap()
                .clone();
            let labels = store.get("Grain").unwrap().as_i32().unwrap().clone();
            let other_grain = labels[0];
            let fcc_grain = labels[5];
            assert_eq!(other_grain, 0);
            assert_eq!(fcc_grain, 1);
            assert!(
                mo[other_grain as usize].is_nan(),
                "OTHER grain must read NaN, got {mo:?}"
            );
            assert!(
                (mo[fcc_grain as usize]).abs() < 1e-6,
                "reference grain reads {mo:?}"
            );
        }

        #[test]
        fn the_table_is_empty_when_there_are_no_grains() {
            // Every atom OTHER and `include_other` off: no grains, so the table
            // must be empty rather than one NaN row per atom.
            let n = 4;
            let pos: Vec<[f64; 3]> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
            let types = vec![structure::OTHER; n];
            let quats = vec![[f64::NAN; 4]; n];
            let (f, s) = fixture(
                UnitCell::orthorhombic(n as f64, 20.0, 20.0),
                &pos,
                &types,
                &quats,
            );
            let mut store = s;
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut sc = Summary::default();
            Grains::new(4.0).run(&ctx, &mut store, &mut sc).unwrap();
            assert!(store.get("GrainSize").unwrap().as_i32().unwrap().is_empty());
            assert!(store
                .get("GrainMisorientation")
                .unwrap()
                .as_f64()
                .unwrap()
                .is_empty());
            // With no grain there is no reference either.
            assert!(!sc.scalars.iter().any(|(k, _)| k == "grain_reference"));
        }

        #[test]
        fn min_size_combines_with_unassigned_other_atoms() {
            // `OTHER` atoms are `NO_GRAIN` before the `min_size` pass, which must
            // not index `sizes` with that sentinel (it would panic). The
            // two features are each tested alone, so only running both together
            // reaches it -- this fixture is the HEA case, where a real snapshot
            // has plenty of non-crystalline atoms at its grain boundaries.
            let n = 12;
            let pos: Vec<[f64; 3]> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
            // Two FCC blocks split by a misorientation, with OTHER atoms
            // scattered through the second half.
            let types: Vec<i32> = (0..n)
                .map(|i| match i {
                    0..=4 => structure::FCC,
                    5 | 9 => structure::OTHER,
                    _ => structure::FCC,
                })
                .collect();
            let quats: Vec<[f64; 4]> = (0..n)
                .map(|i| {
                    if types[i] == structure::OTHER {
                        [f64::NAN; 4]
                    } else if i < 5 {
                        q(1.0, 0.0, 0.0, 0.0)
                    } else {
                        about_z(30.0)
                    }
                })
                .collect();
            let (f, mut store) = fixture(
                UnitCell::orthorhombic(n as f64, 20.0, 20.0),
                &pos,
                &types,
                &quats,
            );
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut sc = Summary::default();
            // `min_size` 4 keeps the 5-atom block and discards the 3- and
            // 2-atom ones that the OTHER atoms cut off from each other.
            Grains {
                min_size: 4,
                ..Grains::new(4.0)
            }
            .run(&ctx, &mut store, &mut sc)
            .unwrap();
            let labels = store.get("Grain").unwrap().as_i32().unwrap().clone();
            // The first block survives intact and is renumbered to 0; every other
            // atom is either too small or not crystalline.
            assert!(labels[..5].iter().all(|&l| l == 0), "{labels:?}");
            assert_eq!(scalar(&sc.scalars, "grain_largest"), 5.0);
            // The rejected count covers the small grains only; the OTHER atoms
            // were never candidates, so they are unassigned rather than rejected.
            assert_eq!(scalar(&sc.scalars, "grain_other_atoms"), 2.0);
            assert_eq!(scalar(&sc.scalars, "grain_rejected_atoms"), 5.0);
            assert_eq!(scalar(&sc.scalars, "grain_count"), 1.0);
            assert_eq!(
                scalar(&sc.scalars, "grain_unassigned"),
                (n - 5) as f64,
                "everything outside grain 0 is unassigned"
            );
        }

        #[test]
        fn a_missing_orientation_property_is_an_error() {
            // Defaulting the orientation to identity would merge every grain in
            // the frame into one, so this has to fail loudly instead.
            let n = 4;
            let pos: Vec<[f64; 3]> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
            let mut p = ParticleData::with_capacity(n);
            for (i, q) in pos.iter().enumerate() {
                p.add_atom(*q, i as i64 + 1, 1);
            }
            let f = Frame::new(UnitCell::orthorhombic(n as f64, 20.0, 20.0), p);
            let mut store = PropertyStore::new();
            store.register_flat(
                DType::U8,
                "StructureType",
                1,
                vec![structure::FCC as f64; n],
            );
            store.register_flat(DType::F64, "StructureInteratomicDistance", 1, vec![1.0; n]);
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut sc = Summary::default();
            let err = Grains::new(4.0)
                .run(&ctx, &mut store, &mut sc)
                .unwrap_err()
                .to_string();
            assert!(err.contains("StructureOrientation"), "{err}");
        }

        #[test]
        fn an_empty_frame_yields_no_grains() {
            // Zero atoms with an explicit cutoff: the derived one cannot exist
            // here, and failing on that is honest. What this pins is that the
            // loops, the centroid pass and the count do not index, underflow or
            // panic on an empty frame.
            let f = Frame::new(
                UnitCell::orthorhombic(10.0, 10.0, 10.0),
                ParticleData::with_capacity(0),
            );
            let mut store = PropertyStore::new();
            store.register_flat(DType::U8, "StructureType", 1, Vec::new());
            store.register_flat(DType::F64, "StructureOrientation", 4, Vec::new());
            store.register_flat(DType::F64, "StructureInteratomicDistance", 1, Vec::new());
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut sc = Summary::default();
            let g = Grains {
                cutoff: Some(1.0),
                ..Grains::new(4.0)
            };
            g.run(&ctx, &mut store, &mut sc).unwrap();
            assert_eq!(scalar(&sc.scalars, "grain_count"), 0.0);
            assert_eq!(scalar(&sc.scalars, "grain_largest"), 0.0);
            assert_eq!(scalar(&sc.scalars, "grain_largest_fraction"), 0.0);
            assert_eq!(store.get("Grain").unwrap().as_i32().unwrap().len(), 0);
        }

        #[test]
        fn an_empty_frame_cannot_derive_a_cutoff() {
            // Same empty frame, but relying on PTM's derived cutoff. There is no
            // maximum interatomic distance to take, and inventing one would give
            // an arbitrary connectivity.
            let f = Frame::new(
                UnitCell::orthorhombic(10.0, 10.0, 10.0),
                ParticleData::with_capacity(0),
            );
            let mut store = PropertyStore::new();
            store.register_flat(DType::U8, "StructureType", 1, Vec::new());
            store.register_flat(DType::F64, "StructureOrientation", 4, Vec::new());
            store.register_flat(DType::F64, "StructureInteratomicDistance", 1, Vec::new());
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &f,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut sc = Summary::default();
            let err = Grains::new(4.0)
                .run(&ctx, &mut store, &mut sc)
                .unwrap_err()
                .to_string();
            assert!(err.contains("StructureInteratomicDistance"), "{err}");
        }

        /// Grain-graph merging end to end, on a chain whose orientation creeps.
        ///
        /// The scenario is over-splitting rather than bridging, because that is the
        /// half merging can actually fix. Connected-components labelling that has
        /// been given a *correct* low threshold still shatters a grain whose bonds
        /// all sit slightly above it, and the only in-band fix -- raising the bond
        /// threshold -- dissolves real boundaries at the same time. Deciding on the
        /// grain graph separates the two cases.
        mod merging_e2e {
            use super::segmentation::{fixture, scalar, segment};
            use super::*;
            use crate::cell::UnitCell;
            use crate::data::frame::Frame;
            use crate::data::property::PropertyStore;

            fn quat_about_z(deg: f64) -> [f64; 4] {
                let (s, c) = (deg.to_radians() / 2.0).sin_cos();
                [c, 0.0, 0.0, s]
            }

            /// A chain whose orientation creeps by `step` degrees per bond, with a
            /// 30 degree discontinuity after `n1` atoms -- one real boundary.
            fn creeping_chain(n1: usize, n2: usize, step: f64) -> (Frame, PropertyStore) {
                let n = n1 + n2;
                let pos: Vec<[f64; 3]> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
                // Second stretch continues from where the first left off, plus a
                // 30 degree step, so the bond across index n1 is the only one
                // inside the chain that is not `step` degrees.
                let second_base = (n1 - 1) as f64 * step + 30.0;
                let quats: Vec<[f64; 4]> = (0..n)
                    .map(|i| {
                        let (base, k) = if i < n1 {
                            (0.0, i)
                        } else {
                            (second_base, i - n1)
                        };
                        quat_about_z(base + k as f64 * step)
                    })
                    .collect();
                fixture(
                    UnitCell::orthorhombic(n as f64, 20.0, 20.0),
                    &pos,
                    &vec![structure::FCC; n],
                    &quats,
                )
            }

            /// 10 atoms creeping 3 degrees apart, then a 30 degree jump, then 4 more.
            /// Every bond inside each stretch is 3 degrees, and the bond across the
            /// middle is 30.
            const N1: usize = 10;
            const N2: usize = 4;
            const STEP: f64 = 3.0;

            #[test]
            fn a_low_bond_threshold_shatters_one_grain() {
                // The premise: 2 degrees cannot bond 3-degree neighbours, so a grain
                // with no boundary in it comes back in pieces.
                let (f, s) = creeping_chain(N1, N2, STEP);
                let g = Grains {
                    misorientation_threshold: 2.0,
                    ..Default::default()
                };
                let (_, scalars) = segment(&f, s, &g);
                assert!(
                    scalar(&scalars, "grain_count") > 2.0,
                    "expected over-splitting, got {}",
                    scalar(&scalars, "grain_count")
                );
            }

            #[test]
            fn raising_the_bond_threshold_would_dissolve_the_real_boundary() {
                // 30 degrees passes, so the two stretches become one grain -- the
                // failure mode that makes "just raise --threshold" the wrong fix.
                let (f, s) = creeping_chain(N1, N2, STEP);
                let g = Grains {
                    misorientation_threshold: 40.0,
                    ..Default::default()
                };
                let (_, scalars) = segment(&f, s, &g);
                assert_eq!(scalar(&scalars, "grain_count"), 1.0);
            }

            #[test]
            fn merging_reunites_the_split_grain_and_keeps_the_real_boundary() {
                // The point of the pass: bond threshold 2 (so nothing bonds),
                // merge threshold 4 (so every 3-degree grain-graph edge is below it,
                // and the 30-degree one is far above).
                let (f, s) = creeping_chain(N1, N2, STEP);
                let g = Grains {
                    misorientation_threshold: 2.0,
                    merge_threshold: Some(4.0),
                    min_size: 0,
                    ..Default::default()
                };
                let (labels, scalars) = segment(&f, s, &g);
                assert_eq!(
                    scalar(&scalars, "grain_count"),
                    2.0,
                    "one grain per stretch, not one per atom: {labels:?}"
                );
                assert_eq!(
                    scalar(&scalars, "grain_merge_edges"),
                    (N1 + N2 - 2) as f64,
                    "every bond but the one across the boundary should merge"
                );
                // The boundary is between atom N1-1 and N1.
                assert_eq!(labels[N1 - 1], 0);
                assert_eq!(labels[N1], 1);
                assert!(labels[..N1].iter().all(|&l| l == 0), "{labels:?}");
                assert!(labels[N1..].iter().all(|&l| l == 1), "{labels:?}");
            }

            #[test]
            fn a_merge_threshold_below_the_creep_merges_nothing() {
                let (f, s) = creeping_chain(N1, N2, STEP);
                let g = Grains {
                    misorientation_threshold: 2.0,
                    merge_threshold: Some(1.0),
                    ..Default::default()
                };
                let (_, scalars) = segment(&f, s, &g);
                assert_eq!(scalar(&scalars, "grain_merge_edges"), 0.0);
                assert!(scalar(&scalars, "grain_count") > 2.0);
            }

            /// Scalar names the default report must contain, in order.
            fn plain_report_keys() -> Vec<String> {
                [
                    "grain_count",
                    "grain_misorientation_threshold",
                    "grain_cutoff",
                    "grain_min_size",
                    "grain_min_grain_size",
                    "grain_dissolved_atoms",
                    "grain_adopted_atoms",
                    "grain_orphan_adoption",
                    "grain_rejected_atoms",
                    "grain_largest",
                    "grain_largest_fraction",
                    "grain_unassigned",
                    "grain_reference",
                    "grain_other_atoms",
                ]
                .iter()
                .map(|s| (*s).to_owned())
                .collect()
            }

            #[test]
            fn the_default_reports_no_merge_scalars_at_any_bond_threshold() {
                // The default runs without the merging pass. This pins that:
                // switching it on by accident, or emitting its scalars when it is
                // off, would change the default output.
                for t in [0.0, 2.0, 40.0, 90.0] {
                    let (f, s) = creeping_chain(N1, N2, STEP);
                    let (_, scalars) = segment(
                        &f,
                        s,
                        &Grains {
                            misorientation_threshold: t,
                            ..Default::default()
                        },
                    );
                    let keys: Vec<String> = scalars.iter().map(|(k, _)| k.clone()).collect();
                    assert_eq!(keys, plain_report_keys(), "at bond threshold {t}");
                }
            }

            #[test]
            fn the_default_is_deterministic_at_every_bond_threshold() {
                for t in [0.0, 2.0, 40.0, 90.0] {
                    let (f1, s1) = creeping_chain(N1, N2, STEP);
                    let (f2, s2) = creeping_chain(N1, N2, STEP);
                    let g = Grains {
                        misorientation_threshold: t,
                        ..Default::default()
                    };
                    let a = segment(&f1, s1, &g).1;
                    let b = segment(&f2, s2, &g).1;
                    assert_eq!(a, b, "not deterministic at bond threshold {t}");
                }
            }

            #[test]
            fn enabling_merging_adds_exactly_two_scalars() {
                let (f, s) = creeping_chain(N1, N2, STEP);
                let (_, scalars) = segment(
                    &f,
                    s,
                    &Grains {
                        misorientation_threshold: 2.0,
                        merge_threshold: Some(4.0),
                        min_size: 0,
                        ..Default::default()
                    },
                );
                let keys: Vec<String> = scalars.iter().map(|(k, _)| k.clone()).collect();
                let mut expected = plain_report_keys();
                // Appended at the end, so the shared prefix stays diffable.
                expected.push("grain_merge_threshold".to_owned());
                expected.push("grain_merge_edges".to_owned());
                assert_eq!(keys, expected);
            }

            /// `segment` unwraps, so the rejection path needs its own runner.
            fn try_segment(
                frame: &Frame,
                store: PropertyStore,
                g: &Grains,
            ) -> std::result::Result<(), String> {
                let ctx = AnalysisContext {
                    frame,
                    nkeys: &NeighborhoodContext { cell: None },
                    par: &Par::new(1),
                };
                let mut store = store;
                let mut sc = Summary::default();
                g.run(&ctx, &mut store, &mut sc).map_err(|e| e.to_string())
            }

            #[test]
            fn an_invalid_merge_threshold_is_refused_before_it_can_merge_or_not_merge() {
                // A threshold of NaN would compare false against every edge and
                // silently behave as if merging were off, which is exactly the
                // kind of quiet wrong answer a threshold option must not give.
                for bad in [f64::NAN, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
                    let (f, s) = creeping_chain(N1, N2, STEP);
                    let g = Grains {
                        misorientation_threshold: 2.0,
                        merge_threshold: Some(bad),
                        ..Default::default()
                    };
                    let err = try_segment(&f, s, &g).expect_err(&format!("{bad} must be refused"));
                    assert!(err.contains("merge_threshold"), "{err}");
                }
            }

            #[test]
            fn a_finite_positive_merge_threshold_is_accepted() {
                // The counterpart, so the check above is not just refusing all
                // thresholds: 0.0 is the boundary the validation must admit.
                for good in [0.0_f64, 0.5, 4.0, 180.0] {
                    let (f, s) = creeping_chain(N1, N2, STEP);
                    let g = Grains {
                        misorientation_threshold: 2.0,
                        merge_threshold: Some(good),
                        min_size: 0,
                        ..Default::default()
                    };
                    try_segment(&f, s, &g)
                        .unwrap_or_else(|e| panic!("{good} must be accepted: {e}"));
                }
            }
        }
    }
}
