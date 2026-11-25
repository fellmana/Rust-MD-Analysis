//! Warren-Cowley short-range-order parameters.
//!
//! For species `i`, `j` and shell `m`,
//!
//! ```text
//! alpha_ij^m = 1 - p_ij^m / c_j
//! ```
//!
//! where `p_ij^m` is the average probability of finding a `j` atom among the
//! neighbours in shell `m` of an `i` atom, and `c_j` is the concentration of `j`
//! in the system.
//!
//! **Sign convention.** `alpha = 0` is a random alloy. `p > c` means `j` is
//! enriched around `i`, so `alpha < 0`: negative is clustering, i.e. short-range
//! order. `p < c` means `j` is depleted, so `alpha > 0`: positive is repulsion,
//! i.e. segregation (design §18.1). Glossing the positive case as "clustering"
//! is backwards; the reference implementation's own README states it correctly.
//!
//! **Shells are k-nearest ranks, not radii.** `nneigh` holds the cumulative
//! neighbour counts that end each shell, exactly as OVITO's `nneigh=[0, 12, 18]`
//! does: shell 1 is neighbour ranks `[0, 12)`, shell 2 is `[12, 18)`. A leading
//! zero carries no information and may be omitted, so `[12, 18]` and
//! `[0, 12, 18]` mean the same thing.
//!
//! That choice has a consequence worth stating: shells are a statement about
//! neighbour *ordering*, so a shell's population is fixed by the configuration
//! rather than measured. On a relaxed alloy the 12th neighbour is not at one
//! radius, and a thermal displacement can reorder the shells near a boundary.
//! That is inherent to the definition (OVITO and LAMMPS both use it), but it
//! means `alpha^m` is not a sharp radial shell integral.
//!
//! **Where this deliberately differs from the reference implementation:**
//!
//! 1. *A short neighbour list cannot be padded, and the guard against it is a
//!    reported number.* The reference asks OVITO for exactly `max(nneigh)`
//!    neighbours and then indexes `particle_types[neigh_idx]`, where OVITO pads a
//!    short list with `-1`; in NumPy that silently reads the *last* atom in the
//!    frame. An atom at a surface with 11 neighbours where 18 were asked for
//!    would contribute its shell-1 neighbours a second time in shell 2, and
//!    nothing in the output would say so.
//!
//!    Here a short list is never measured. `validate` refuses `k + 1 > n`, and
//!    the search widens within the cell list's stencil, which is built at 1.5x
//!    the radius the frame's mean density gives for the k-th neighbour: enough
//!    for an atom whose neighbourhood is half empty (beside a void or a free
//!    surface needs ~1.26x). An atom that still finds fewer than `k` (one far
//!    from all others, e.g. a sputtered atom in vacuum) is left out of every
//!    shell and counted in `wc_atoms_skipped`, never measured on a truncated
//!    list.
//! 2. *Concentrations use the distinct types present*, not `1..=ntypes`, matching
//!    `np.unique` in the reference. A frame whose types are `{1, 7}` gets a 2x2
//!    matrix; using the maximum index would add a row of `NaN` for every type
//!    that happens not to occur.
//! 3. *Symmetry is a reported number, 0 only on a perfect lattice.* `alpha_ij`
//!    and `alpha_ji` divide the bond counts `b_ij`, `b_ji` by `N_i * N_j`, but a
//!    k-nearest bond is not symmetric (j among the k nearest of i does not put i
//!    among the k nearest of j), so on a thermal or relaxed frame the two
//!    counts, and the two alphas, differ. `--only-selected` breaks it further,
//!    because there the central populations no longer match the concentration
//!    set.
//!    `wc_asymmetry_max` is the off-diagonal disagreement, so a frame where the
//!    user expects the reference's symmetry warning to fire can be checked rather
//!    than assumed -- and `only_same_type`, which drops the off-diagonal entirely,
//!    leaves nothing to check and does not report one.
//!
//! ## Outputs
//!
//! One scalar per matrix entry, named `wc_m{m}_{i}_{j}`, with `i`, `j` the actual
//! type ids and `m` 1-based to match OVITO's property names. Scalars rather than
//! a flattened histogram because `Summary` histograms carry no axis metadata: a
//! vector indexed by `i*ntypes + j` would print correctly and be unreadable.
//!
//! A NaN entry travels as NaN rather than being dropped, so "not measured" is
//! visible in the output instead of absent from it.
//!
//! Per-atom `WarrenCowleyShell{m}` with `ntypes * ntypes` components in
//! flattened `i*ntypes + j` order, written only when `per_particle` is set. The
//! component labels are not in the property name; `rmda wc` prints them.
//!
//! With `edge_correct`, `wc_atoms_surface` and `wc_atoms_measured` say how many
//! central atoms were excluded for a free surface nearer than their k-th
//! neighbour and how many were measured.
//!
//! ## Memory and threads (design §18.1)
//!
//! The neighbour pass is parallel; each atom writes its own row of u16 shell
//! histograms (2 B per shell and type), and the averages are summed over the
//! rows serially in atom order, so every output is bit-identical at any thread
//! count and to a serial loop.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::analysis::occupancy::Occupancy;
use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::data::frame::Frame;
use crate::data::property::{DType, Property, PropertyStore};
use crate::error::{Error, Result};
use crate::neighbor::cell_list::CellList;
use crate::neighbor::nnearest::nearest_k_excluding_self;

/// Radius to probe for `k` neighbours at number density `rho`, with margin.
///
/// Where the `k`-th neighbour is expected to sit, from the mean density.
///
/// The same estimate `analysis::cluster` uses for its k-nearest mode (1.5x the
/// ideal-gas radius). It sizes the cell list, whose stencil bounds how far
/// `nearest_k_excluding_self` can widen its probe: an atom with fewer than `k`
/// neighbours within it is skipped (see the module notes), not measured short.
fn knear_radius(k: usize, rho: f64) -> f64 {
    crate::analysis::knear_radius(k, rho)
}

/// Atoms per parallel task of the neighbour pass. Small so a 40k-atom frame
/// still spreads over every thread; nothing depends on it but the schedule.
const WC_GRAIN: usize = 1024;

/// Per-atom status word of the shell histogram rows.
const STATUS_MEASURED: u16 = 1;
/// Fewer than `k` neighbours (unreachable after `validate`; kept as a tripwire).
const STATUS_SHORT: u16 = 2;
/// Excluded by `edge_correct`: a free surface nearer than its k-th neighbour.
const STATUS_SURFACE: u16 = 3;

/// Cumulative shell ends, validated and normalised to include a leading zero.
///
/// The shared shape for both the parsed input and the struct field, so the two
/// can never disagree about what a shell is.
fn normalise_ends(ends: &[usize]) -> Result<Vec<usize>> {
    let mut v: Vec<usize> = ends.to_vec();
    if v.first() == Some(&0) {
        v.remove(0);
    }
    if v.is_empty() {
        return Err(Error::invalid(
            "warren-cowley needs at least one shell end, e.g. --wc-shells 12",
        ));
    }
    if v.contains(&0) {
        return Err(Error::invalid(format!(
            "warren-cowley shell ends must be positive, got {ends:?} (only a leading 0 may be \
             implied, not one in the middle)"
        )));
    }
    if v.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::invalid(format!(
            "warren-cowley shell ends must be strictly increasing, got {ends:?}"
        )));
    }
    v.insert(0, 0);
    Ok(v)
}

/// Parse a comma-separated `--wc-shells` value.
pub fn parse_shells(text: &str) -> Result<Vec<usize>> {
    let mut ends = Vec::new();
    for part in text.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        ends.push(t.parse::<usize>().map_err(|_| {
            Error::invalid(format!(
                "warren-cowley shell end {t:?} is not a whole number of neighbours"
            ))
        })?);
    }
    normalise_ends(&ends)
}

/// Warren-Cowley parameters over k-nearest-defined shells.
#[derive(Clone, Debug)]
pub struct WarrenCowley {
    /// Cumulative neighbour counts ending each shell, strictly increasing.
    ///
    /// `[0, 12]` is the first shell only, which is OVITO's default. `[0, 12, 18]`
    /// adds the FCC second shell. A leading zero is optional.
    pub nneigh: Vec<usize>,
    /// Use only selected atoms as central atoms.
    ///
    /// Concentrations are then computed over the selected atoms *and their
    /// outermost-shell neighbours*, as in the reference: measuring a region's
    /// order against the global composition of a box that is mostly something
    /// else reports segregation of the region from the rest of the cell, not
    /// order within the region.
    pub only_selected: bool,
    /// Emit the per-atom `WarrenCowleyShell{m}` properties.
    pub per_particle: bool,
    /// Report only `alpha_ii`, dropping the off-diagonal terms.
    ///
    /// An extension past the reference implementation, which has no such flag.
    /// The diagonal is the like-atom term, so this is the "is this alloy ordered
    /// at all" summary; the full matrix is what says which pairs prefer each
    /// other.
    pub only_same_type: bool,
    /// Measure only central atoms deeper below a free surface than their own
    /// k-th neighbour (design §18.1), so no reported shell is truncated by a
    /// surface: in a film the k nearest of a surface atom reach into its
    /// second shell, which biases alpha. Depth is the lower bound of
    /// `Occupancy::depth_of`, as `rdf --edge-correct` uses; concentrations stay
    /// over every atom (or the selection).
    pub edge_correct: bool,
}

impl Default for WarrenCowley {
    /// OVITO's default: the first shell, twelve neighbours.
    fn default() -> Self {
        WarrenCowley {
            nneigh: vec![0, 12],
            only_selected: false,
            per_particle: false,
            only_same_type: false,
            edge_correct: false,
        }
    }
}

impl WarrenCowley {
    /// Shell ends including the leading zero, or the stored list if invalid.
    ///
    /// Reporting methods must not return `Result`, so they cannot refuse; `run`
    /// validates separately and an invalid list here only means "no shell
    /// information to report".
    pub fn ends(&self) -> Vec<usize> {
        normalise_ends(&self.nneigh).unwrap_or_else(|_| vec![0])
    }

    pub fn nshells(&self) -> usize {
        self.ends().len().saturating_sub(1)
    }

    /// Neighbours required by the outermost shell.
    pub fn k(&self) -> usize {
        self.ends().last().copied().unwrap_or(0)
    }

    /// Type ids present, ascending. Matrix indices are positions in this list.
    pub fn types_present(&self, frame: &Frame) -> Vec<u32> {
        let mut set: Vec<u32> = frame.particles.types.to_vec();
        set.sort_unstable();
        set.dedup();
        set
    }

    /// Whether atom `i` is a central atom for this configuration.
    ///
    /// Only `only_selected` restricts anything. A frame that *carries* a
    /// selection is still measured in full without the flag: reading a mask off
    /// the file is not the same as asking for a subset, and silently honouring it
    /// would make the same command measure different things on two files.
    ///
    /// The `None` arm cannot be reached with the flag set -- `validate` refuses
    /// that -- and is `true` rather than an error so the helper stays total.
    fn is_central(&self, frame: &Frame, i: usize) -> bool {
        if !self.only_selected {
            return true;
        }
        match &frame.particles.selection {
            Some(sel) => sel.get(i).is_some_and(|&s| s != 0),
            None => true,
        }
    }

    /// Checks the request against the frame before any search runs.
    ///
    /// Public so a CLI can refuse a bad request before allocating a neighbour list
    /// for it, the same reason `Rdf::validate` is: otherwise a malformed rank list
    /// is reported as a failure of the search rather than of the request.
    pub fn validate(&self, frame: &Frame) -> Result<()> {
        normalise_ends(&self.nneigh)?;
        let n = frame.natoms();
        let k = self.k();
        if n == 0 {
            return Ok(());
        }
        // One neighbour needs one other atom. Checking here means the error is
        // about the request rather than about a search that cannot converge.
        if k + 1 > n {
            return Err(Error::invalid(format!(
                "warren-cowley asks for {k} neighbours but the frame has only {n} atoms"
            )));
        }
        if self.only_selected && frame.particles.selection.is_none() {
            return Err(Error::invalid(
                "--only-selected needs a Selection property on the input, and the frame has none. \
                 Analysing every atom instead would silently ignore the request",
            ));
        }
        Ok(())
    }
}

impl Analysis for WarrenCowley {
    fn name(&self) -> &'static str {
        "wc"
    }

    /// No fixed cutoff: shells are k-nearest ranks. The radius the search needs
    /// is derived per frame in `cutoffs_for`.
    fn cutoffs(&self) -> Vec<f64> {
        Vec::new()
    }

    fn cutoffs_for(&self, frame: &Frame) -> Vec<f64> {
        let n = frame.natoms();
        if n == 0 {
            return Vec::new();
        }
        let rho = n as f64 / frame.cell.volume();
        let r = knear_radius(self.k(), rho);
        if r.is_finite() && r > 0.0 {
            vec![r]
        } else {
            Vec::new()
        }
    }

    fn run(
        &self,
        ctx: &AnalysisContext<'_>,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        let frame = ctx.frame;
        let n = frame.natoms();
        let ends = normalise_ends(&self.nneigh)?;
        let nshells = ends.len() - 1;
        let k = *ends.last().unwrap();
        self.validate(frame)?;

        // An empty frame has no shells to measure, and that is an answer rather
        // than a failure: an `analyze` run over a trajectory should not stop.
        if n == 0 {
            summary.merge_from(Summary::scalar("wc_shells", nshells as f64));
            summary.merge_from(Summary::scalar("wc_ntypes", 0.0));
            summary.merge_from(Summary::scalar("wc_atoms_central", 0.0));
            summary.merge_from(Summary::scalar("wc_atoms_skipped", 0.0));
            return Ok(());
        }

        let types = self.types_present(frame);
        let ntypes = types.len();
        let index_of = |t: u32| types.binary_search(&t).ok();
        let n_central = (0..n).filter(|&i| self.is_central(frame, i)).count();
        let width: Vec<usize> = ends.windows(2).map(|w| w[1] - w[0]).collect();
        if width.iter().any(|&w| w > u16::MAX as usize) {
            return Err(Error::invalid(format!(
                "warren-cowley shells wider than {} neighbours are not supported",
                u16::MAX
            )));
        }

        // Own cell list at the density estimate, the same radius `cutoffs_for`
        // reports. A shared narrower list would truncate the stencil and shorten
        // shells near the box edge, which is exactly where the count is checked.
        let rho = n as f64 / frame.cell.volume();
        let radius = knear_radius(k, rho);
        let positions = &frame.particles.positions[..n];
        let cell = CellList::build(&frame.cell, positions, radius)?;
        // Depth below the free surface, for `edge_correct`: an atom is a centre
        // only when no free surface is nearer than its own k-th neighbour, so
        // every shell it reports is whole.
        let depth_grid = if self.edge_correct {
            Some((
                Occupancy::build(&frame.cell, positions)?,
                frame.cell.inverse_transposed()?,
            ))
        } else {
            None
        };

        // One parallel pass over the neighbour search. Each atom owns a row of
        // `1 + nshells * ntypes` u16: a status word, then the type histogram of
        // each shell (`row[1 + m * ntypes + j]` neighbours of type `types[j]` in
        // shell `m`). The status is tracked explicitly rather than inferred from
        // an all-zero row, because an all-zero row is also what a skipped atom
        // looks like and conflating the two would quietly drop a real atom.
        //
        // Rows are disjoint and a function of the atom alone, so they are the
        // same at every thread count, and the reductions below run over them in
        // atom order. That keeps the f64 sums bit-identical to a serial loop,
        // at 2 B per (shell, type) plus 2 B per atom rather than the 150-250
        // B/atom of per-atom nested `Vec<Vec<Vec<f64>>>`.
        let stride = 1 + nshells * ntypes;
        let mut rows = vec![0u16; n * stride];
        // Atoms each shell reached, for the `only_selected` concentration set.
        // Marking is an OR, so the order the threads reach an atom in is moot.
        let reached: Vec<AtomicBool> = if self.only_selected {
            (0..n).map(|_| AtomicBool::new(false)).collect()
        } else {
            Vec::new()
        };
        {
            let _t = crate::prof::StageTimer::start("wc.shells");
            let types_of = &frame.particles.types;
            let run = |(c, chunk): (usize, &mut [u16])| -> Result<()> {
                let mut entries: Vec<(u32, [f64; 3], f64)> = Vec::with_capacity(k);
                for (r, row) in chunk.chunks_exact_mut(stride).enumerate() {
                    let i = c * WC_GRAIN + r;
                    if !self.is_central(frame, i) {
                        continue;
                    }
                    nearest_k_excluding_self(&cell, i, k, &mut entries)?;
                    if entries.len() < k {
                        row[0] = STATUS_SHORT;
                        continue;
                    }
                    if let Some((grid, inv_t)) = &depth_grid {
                        let dk = entries[k - 1].2.sqrt();
                        let depth =
                            grid.depth_of(Occupancy::frac(&frame.cell, inv_t, positions[i]));
                        if depth <= dk {
                            row[0] = STATUS_SURFACE;
                            continue;
                        }
                    }
                    row[0] = STATUS_MEASURED;
                    for (m, w) in ends.windows(2).enumerate() {
                        let hist = &mut row[1 + m * ntypes..1 + (m + 1) * ntypes];
                        for e in &entries[w[0]..w[1]] {
                            if let Some(j) = index_of(types_of[e.0 as usize]) {
                                hist[j] += 1;
                            }
                        }
                    }
                    if !reached.is_empty() {
                        reached[i].store(true, Ordering::Relaxed);
                        for e in &entries {
                            reached[e.0 as usize].store(true, Ordering::Relaxed);
                        }
                    }
                }
                Ok(())
            };
            let rows_mut = &mut rows[..];
            let results: Vec<Result<()>> = if ctx.par.is_parallel() {
                let run = &run;
                ctx.par.install(move || {
                    rows_mut
                        .par_chunks_mut(stride * WC_GRAIN)
                        .enumerate()
                        .map(run)
                        .collect()
                })
            } else {
                rows_mut
                    .chunks_mut(stride * WC_GRAIN)
                    .enumerate()
                    .map(run)
                    .collect()
            };
            // The lowest chunk's error, whatever the scheduling.
            for r in results {
                r?;
            }
        }
        let status = |i: usize| rows[i * stride];
        let skipped = (0..n).filter(|&i| status(i) == STATUS_SHORT).count();
        let surface = (0..n).filter(|&i| status(i) == STATUS_SURFACE).count();
        let measured = (0..n).filter(|&i| status(i) == STATUS_MEASURED).count();

        // Every central atom falling short is not a frame with a few awkward
        // atoms in it, it is a frame with no measurable shell at all: an
        // all-`NaN` matrix with a zero denominator everywhere is a much worse
        // thing to print than an error. `validate` has already refused the
        // outright impossible case of `k + 1 > n`, so reaching this means the
        // atoms are there but none of them has `k` neighbours.
        if measured == 0 && surface == 0 {
            return Err(Error::invalid(format!(
                "warren-cowley found no atom with {k} neighbours among {n} atoms, so no shell \
                 could be measured; the frame is too sparse or too heterogeneous for a \
                 fixed-shell description"
            )));
        }
        if measured == 0 {
            return Err(Error::invalid(format!(
                "warren-cowley --edge-correct: none of the {surface} central atoms is deeper below \
                 a free surface than its {k}th neighbour; the structure is too thin for these \
                 shells, so drop --edge-correct or use fewer neighbours"
            )));
        }

        // Concentrations over the atoms the parameters are measured *against*:
        // everything, or the selected atoms plus their outermost shell (a
        // selected atom counted once however many shells reach it).
        let mut conc_counts = vec![0f64; ntypes];
        for a in 0..n {
            if reached.is_empty() || reached[a].load(Ordering::Relaxed) {
                if let Some(j) = index_of(frame.particles.types[a]) {
                    conc_counts[j] += 1.0;
                }
            }
        }
        drop(reached);
        let conc_total: f64 = conc_counts.iter().sum();
        if conc_total > 0.0 {
            for c in conc_counts.iter_mut() {
                *c /= conc_total;
            }
        }

        // The parameters. The reference averages *ratios* over the central atoms
        // of each type (`nanmean` of `1 - (count/Nb)/c_j`), not counts, so this
        // accumulates per-atom `alpha` and divides by the number of atoms that
        // produced one. The pooled-count form agrees exactly when every central
        // atom has the same shell size, and differs only where shells were short
        // -- and those atoms are skipped rather than weighted.
        //
        // Serial, in atom order, from the rows: the same additions in the same
        // order as the serial loop, so the sums do not move by an ulp with the
        // thread count. It is O(n * nshells * ntypes) adds, small beside the
        // search.
        let _t = crate::prof::StageTimer::start("wc.reduce");
        let mut sum = vec![vec![vec![0f64; ntypes]; ntypes]; nshells];
        let mut cnt = vec![vec![vec![0u64; ntypes]; ntypes]; nshells];
        for i in 0..n {
            if status(i) != STATUS_MEASURED {
                continue;
            }
            let row = &rows[i * stride..(i + 1) * stride];
            let i_t = index_of(frame.particles.types[i]).expect("type occurs, so it is in `types`");
            for m in 0..nshells {
                let nb = width[m] as f64;
                for j in 0..ntypes {
                    // `c_j > 0` for every type present by construction. A
                    // selected region can contain no central atom of a type its
                    // halo reached, which leaves the *row* NaN rather than this
                    // entry, so this branch is a guard, not a normal path.
                    let cj = conc_counts[j];
                    if cj <= 0.0 || cj.is_nan() {
                        continue;
                    }
                    let a = 1.0 - (row[1 + m * ntypes + j] as f64 / nb) / cj;
                    sum[m][i_t][j] += a;
                    cnt[m][i_t][j] += 1;
                }
            }
        }
        let mut alpha = vec![vec![vec![f64::NAN; ntypes]; ntypes]; nshells];
        for m in 0..nshells {
            for i in 0..ntypes {
                for j in 0..ntypes {
                    if cnt[m][i][j] > 0 {
                        alpha[m][i][j] = sum[m][i][j] / cnt[m][i][j] as f64;
                    }
                }
            }
        }
        drop(_t);
        // ---- outputs ----
        summary.merge_from(Summary::scalar("wc_shells", nshells as f64));
        summary.merge_from(Summary::scalar("wc_ntypes", ntypes as f64));
        summary.merge_from(Summary::scalar("wc_atoms_central", n_central as f64));
        summary.merge_from(Summary::scalar("wc_atoms_skipped", skipped as f64));
        if self.edge_correct {
            summary.merge_from(Summary::scalar("wc_atoms_surface", surface as f64));
            summary.merge_from(Summary::scalar("wc_atoms_measured", measured as f64));
        }
        summary.merge_from(Summary::scalar("wc_probe_radius", radius));
        for (t, c) in types.iter().zip(conc_counts.iter()) {
            summary.scalars.push((format!("wc_conc_{t}"), *c));
        }
        for (m, w) in ends.windows(2).enumerate() {
            summary
                .scalars
                .push((format!("wc_m{}_nneigh", m + 1), (w[1] - w[0]) as f64));
        }

        // `alpha_ij != alpha_ji` once a partial selection has broken the tie
        // between the central populations and the concentration set. Reported so
        // that a matrix which *should* be symmetric can be checked rather than
        // assumed -- and not reported at all under `only_same_type`, where the
        // off-diagonal is gone and the number could only ever be a zero that
        // reads like a check that passed.
        if !self.only_same_type {
            let mut asym = 0f64;
            for shell in &alpha {
                for (i, row) in shell.iter().enumerate() {
                    for (j, &b) in row.iter().take(i).enumerate() {
                        let a = shell[j][i];
                        if a.is_finite() && b.is_finite() {
                            asym = asym.max((a - b).abs());
                        }
                    }
                }
            }
            summary.merge_from(Summary::scalar("wc_asymmetry_max", asym));
        }

        for (m, shell) in alpha.iter().enumerate() {
            for (i, row) in shell.iter().enumerate() {
                if self.only_same_type {
                    summary
                        .scalars
                        .push((format!("wc_m{}_{}", m + 1, types[i]), row[i]));
                    continue;
                }
                for (j, &a) in row.iter().enumerate() {
                    summary
                        .scalars
                        .push((format!("wc_m{}_{}_{}", m + 1, types[i], types[j]), a));
                }
            }
        }

        if self.per_particle {
            // One property per shell, filled atom by atom straight from the
            // rows: the matrix row of the atom's own type, NaN elsewhere. An
            // atom that is not a central atom, was skipped for a short list or
            // excluded at a surface is NaN throughout rather than zero, since
            // zero is a real value meaning "this neighbour type does not occur
            // here".
            let mut props: Vec<Property> = (0..nshells)
                .map(|m| {
                    Property::with_capacity(
                        format!("WarrenCowleyShell{}", m + 1),
                        DType::F64,
                        true,
                        ntypes * ntypes,
                        n,
                    )
                })
                .collect();
            for i in 0..n {
                let measured = status(i) == STATUS_MEASURED;
                let row = &rows[i * stride..(i + 1) * stride];
                let i_t = index_of(frame.particles.types[i]);
                for (m, prop) in props.iter_mut().enumerate() {
                    let nb = width[m] as f64;
                    for ii in 0..ntypes {
                        for j in 0..ntypes {
                            let cj = conc_counts[j];
                            let v = if measured && Some(ii) == i_t && cj > 0.0 && !cj.is_nan() {
                                1.0 - (row[1 + m * ntypes + j] as f64 / nb) / cj
                            } else {
                                f64::NAN
                            };
                            prop.push(v);
                        }
                    }
                }
            }
            for prop in props {
                out.register(prop);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::NeighborhoodContext;
    use crate::cell::UnitCell;
    use crate::data::particle::ParticleData;
    use crate::exec::Par;

    /// Simple cubic lattice of `ncell` cells per side at spacing `a`.
    ///
    /// SC because its shells are countable by hand. The catch, which several
    /// tests below depend on: SC's first shell is 6 wide, and ranks 6..18 are
    /// **twelve vectors at one distance**, so a `k = 12` shell takes only half of
    /// them and which half is a tie-break. Exact fixtures therefore use
    /// `[0, 6]`; a `k = 12` expectation on SC would be pinning the tie-break
    /// rather than the physics.
    fn sc(ncell: usize, a: f64) -> Frame {
        sc_typed(ncell, a, |_, _, _| 1)
    }

    fn sc_typed(ncell: usize, a: f64, ty: impl Fn(usize, usize, usize) -> u32) -> Frame {
        let mut pd = ParticleData::with_capacity(ncell.pow(3));
        let mut id = 0i64;
        for i in 0..ncell {
            for j in 0..ncell {
                for k in 0..ncell {
                    id += 1;
                    pd.add_atom([i as f64 * a, j as f64 * a, k as f64 * a], id, ty(i, j, k));
                }
            }
        }
        Frame::new(
            UnitCell::orthorhombic(ncell as f64 * a, ncell as f64 * a, ncell as f64 * a),
            pd,
        )
    }

    /// Two well-separated pure blocks: `n_a` cells of type 1 and `n_b` of type 2.
    ///
    /// Every atom's six nearest neighbours are inside its own block, so each
    /// species sits at `p_jj = 1` and `p_jk = 0` with no boundary effect and no
    /// arithmetic left to guess. That makes the two-block frame the workhorse for
    /// exact expectations.
    fn two_blocks(n_a: usize, n_b: usize, a: f64) -> Frame {
        let na = n_a as f64 * a;
        let mut pd = ParticleData::with_capacity(n_a.pow(3) + n_b.pow(3));
        let mut id = 0i64;
        for (base, n, ty) in [(0.0, n_a, 1u32), (10.0 * na, n_b, 2u32)] {
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        id += 1;
                        pd.add_atom([base + i as f64 * a, j as f64 * a, k as f64 * a], id, ty);
                    }
                }
            }
        }
        let span = na + 10.0 * na;
        Frame::new(UnitCell::orthorhombic(12.0 * span, span, span), pd)
    }

    /// An ordered FCC alloy of the L1_2 (Cu3Au) kind: `ncell` conventional cells
    /// per side at lattice constant `a`, corners of type 1 and face centres of
    /// type 2. The cell is periodic and the lattice is continuous across the
    /// faces, so no atom is a surface atom.
    ///
    /// Here to be checked against crystallography rather than against itself. The
    /// bond census is a textbook one:
    ///
    ///   * a corner atom's twelve nearest neighbours are all face centres
    ///   * a face-centre atom's twelve nearest are four corners and eight face
    ///     centres
    ///
    /// giving `c_1 = 1/4`, `c_2 = 3/4` and a matrix with a real signature, not a
    /// diagonal of zeros.
    fn fcc_l12(ncell: usize, a: f64) -> Frame {
        let basis: [(f64, f64, f64); 4] = [
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ];
        let mut pd = ParticleData::with_capacity(4 * ncell.pow(3));
        let mut id = 0i64;
        for i in 0..ncell {
            for j in 0..ncell {
                for k in 0..ncell {
                    for b in basis {
                        id += 1;
                        pd.add_atom(
                            [
                                (i as f64 + b.0) * a,
                                (j as f64 + b.1) * a,
                                (k as f64 + b.2) * a,
                            ],
                            id,
                            if b == (0.0, 0.0, 0.0) { 1 } else { 2 },
                        );
                    }
                }
            }
        }
        Frame::new(
            UnitCell::orthorhombic(ncell as f64 * a, ncell as f64 * a, ncell as f64 * a),
            pd,
        )
    }

    fn wc(nneigh: &[usize]) -> WarrenCowley {
        WarrenCowley {
            nneigh: normalise_ends(nneigh).expect("test shells are valid"),
            ..Default::default()
        }
    }

    fn run(f: &Frame, w: &WarrenCowley) -> (Summary, PropertyStore) {
        let nk = NeighborhoodContext { cell: None };
        let par = Par::new(1);
        let ctx = AnalysisContext {
            frame: f,
            nkeys: &nk,
            par: &par,
        };
        let mut out = PropertyStore::new();
        let mut sc = Summary::default();
        w.run(&ctx, &mut out, &mut sc).expect("run");
        (sc, out)
    }

    fn run_err(f: &Frame, w: &WarrenCowley) -> String {
        let nk = NeighborhoodContext { cell: None };
        let par = Par::new(1);
        let ctx = AnalysisContext {
            frame: f,
            nkeys: &nk,
            par: &par,
        };
        let mut out = PropertyStore::new();
        let mut sc = Summary::default();
        w.run(&ctx, &mut out, &mut sc)
            .expect_err("should have been refused")
            .to_string()
    }

    fn scalar(sc: &Summary, key: &str) -> f64 {
        sc.scalars
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| {
                let names: Vec<&str> = sc.scalars.iter().map(|(k, _)| k.as_str()).collect();
                panic!("no {key} in summary; have {names:?}")
            })
    }

    fn assert_close(got: f64, want: f64, what: &str) {
        assert!(
            (got - want).abs() < 1e-12,
            "{what}: got {got}, expected {want} (difference {})",
            got - want
        );
    }

    #[test]
    fn a_single_species_reads_exactly_zero_because_it_cannot_be_ordered() {
        // One type: p_jj = 1 and c_j = 1, so alpha = 1 - 1/1 = 0 exactly. Any
        // stray factor of two, or a `4*pi` left in the formula, shows up here.
        let (sc, _) = run(&sc(14, 3.0), &wc(&[0, 6]));
        assert_eq!(scalar(&sc, "wc_ntypes"), 1.0);
        assert_eq!(scalar(&sc, "wc_m1_1_1"), 0.0);
        assert_eq!(scalar(&sc, "wc_asymmetry_max"), 0.0);
    }

    #[test]
    fn two_pure_blocks_read_minus_one_on_the_diagonal_which_is_maximum_order() {
        // Each species has only itself in its own first shell, so p_jj = 1 against
        // c_j = 1/2: alpha_jj = 1 - 2 = -1. Each species is absent from the
        // other's, so alpha_jk = 1 - 0 = +1.
        //
        // This is the exact end of the scale in both directions at once, and it
        // pins the sign: the "perfectly ordered" answer is the *negative* one.
        let (sc, _) = run(&two_blocks(8, 8, 3.0), &wc(&[0, 6]));
        assert_close(scalar(&sc, "wc_conc_1"), 0.5, "c_1");
        assert_close(scalar(&sc, "wc_m1_1_1"), -1.0, "alpha_11");
        assert_close(scalar(&sc, "wc_m1_2_2"), -1.0, "alpha_22");
        assert_close(scalar(&sc, "wc_m1_1_2"), 1.0, "alpha_12");
        assert_close(scalar(&sc, "wc_m1_2_1"), 1.0, "alpha_21");
        // Equal concentrations, so the matrix is symmetric here.
        assert_close(scalar(&sc, "wc_asymmetry_max"), 0.0, "asymmetry");
    }

    #[test]
    fn a_species_absent_from_its_own_shell_reads_positive_and_that_is_repulsion() {
        // Type by x-parity in a block with an **even** number of cells, so every
        // atom's two x-neighbours really are the other type with no wrap-around
        // exception. Each site then has 4 same-type (along y, z) and 2 other-type
        // (along x) neighbours out of 6, against equal concentrations:
        //
        //     alpha_jj = 1 - (4/6)/(1/2) = 1 - 4/3 = -1/3
        //     alpha_jk = 1 - (2/6)/(1/2) = 1 - 2/3 = +1/3
        //
        // The off-diagonal is positive, i.e. repulsion, which is the half of the
        // convention that is easy to state backwards (design §18.1).
        let f = sc_typed(8, 3.0, |i, _, _| 1 + (i % 2) as u32);
        let (sc, _) = run(&f, &wc(&[0, 6]));
        assert_close(scalar(&sc, "wc_conc_1"), 0.5, "c_1");
        assert_close(scalar(&sc, "wc_m1_1_1"), -1.0 / 3.0, "alpha_11");
        assert_close(scalar(&sc, "wc_m1_2_2"), -1.0 / 3.0, "alpha_22");
        assert_close(scalar(&sc, "wc_m1_1_2"), 1.0 / 3.0, "alpha_12");
        assert_close(scalar(&sc, "wc_m1_2_1"), 1.0 / 3.0, "alpha_21");
        assert_close(scalar(&sc, "wc_asymmetry_max"), 0.0, "asymmetry");
    }

    #[test]
    fn unequal_concentrations_scale_the_diagonal() {
        // Two pure blocks of *different* sizes: 8^3 = 512 sites of type 1 and
        // 6^3 = 216 of type 2, so
        //
        //     c_1 = 512/728 = 64/91        c_2 = 216/728 = 27/91
        //
        // Each species still has only itself in its own shell (`p_jj = 1`,
        // `p_jk = 0`), so the diagonal is just `1 - 1/c_j` and the off-diagonal
        // is exactly 1:
        //
        //     alpha_11 = 1 - 91/64 = -27/64      alpha_22 = 1 - 91/27 = -64/27
        //
        // Both local environments are equally perfect -- `p_jj = 1` in each block --
        // and the diagonal still differs by nearly 2, because `alpha` divides by a
        // global concentration. A factor of two left in the normalisation, or a
        // `c` taken from the wrong block, would not reproduce either fraction.
        let (sc, _) = run(&two_blocks(8, 6, 3.0), &wc(&[0, 6]));
        assert_close(scalar(&sc, "wc_conc_1"), 64.0 / 91.0, "c_1");
        assert_close(scalar(&sc, "wc_conc_2"), 27.0 / 91.0, "c_2");
        assert_close(scalar(&sc, "wc_m1_1_1"), -27.0 / 64.0, "alpha_11");
        assert_close(scalar(&sc, "wc_m1_2_2"), -64.0 / 27.0, "alpha_22");
        assert_close(scalar(&sc, "wc_m1_1_2"), 1.0, "alpha_12");
        assert_close(scalar(&sc, "wc_m1_2_1"), 1.0, "alpha_21");
        // Identical off-diagonals, so no asymmetry -- see
        // `only_selected_is_the_way_to_make_alpha_asymmetric` for where a non-zero
        // one comes from. A non-zero here would mean the matrix had been
        // symmetrised *or* that the diagonal had been folded into the comparison.
        assert_close(scalar(&sc, "wc_asymmetry_max"), 0.0, "asymmetry");
    }

    #[test]
    fn only_selected_is_the_way_to_make_alpha_asymmetric() {
        // An eight-atom alternating chain, but the selection is *not* a species:
        // atoms 2, 4 and 6 (all type 1) plus atom 3 (type 2) only. Ends 0 and 1 are
        // unselected, atom 5, 7 and 8 are only reached as neighbours.
        //
        // Central atoms are 2, 3, 4, 6. Each type-1 central site has two type-2
        // neighbours, and type-2 site 3 has two type-1 neighbours, so
        //
        //     p_11 = 0   p_12 = 1   p_21 = 1   p_22 = 0
        //
        // The concentration set is the selection *plus its halo*, which is
        // {1,2,3,4,5,6,7} -- 3 sites of type 1 and 4 of type 2 -- so
        //
        //     c_1 = 3/7   c_2 = 4/7
        //     alpha_11 = 1 - 0/(3/7) = +1
        //     alpha_12 = 1 - 1/(4/7) = -3/4
        //     alpha_21 = 1 - 1/(3/7) = -4/3
        //     alpha_22 = 1 - 0       = +1
        //
        // `alpha_12` and `alpha_21` now differ by 7/12, and that is structural:
        // with every atom central the two directions divide the *same* bond counts
        // by the product of the same two populations, so they agree identically.
        // Breaking the symmetry requires central atoms of one type to see a
        // population the other direction does not, which is exactly what a
        // partial selection does.
        let mut f = chain(8, 3.0);
        let mut sel = vec![0u8; f.natoms()];
        for a in [2usize, 3, 4, 6] {
            sel[a] = 1;
        }
        f.particles.selection = Some(sel);
        let mut w = wc(&[0, 2]);
        w.only_selected = true;
        let (sc, _) = run(&f, &w);
        assert_close(scalar(&sc, "wc_atoms_central"), 4.0, "central atoms");
        assert_close(scalar(&sc, "wc_conc_1"), 3.0 / 7.0, "c_1");
        assert_close(scalar(&sc, "wc_conc_2"), 4.0 / 7.0, "c_2");
        assert_close(scalar(&sc, "wc_m1_1_1"), 1.0, "alpha_11");
        assert_close(scalar(&sc, "wc_m1_2_2"), 1.0, "alpha_22");
        assert_close(scalar(&sc, "wc_m1_1_2"), -0.75, "alpha_12");
        assert_close(scalar(&sc, "wc_m1_2_1"), -4.0 / 3.0, "alpha_21");
        assert_close(scalar(&sc, "wc_asymmetry_max"), 7.0 / 12.0, "asymmetry");
        // The same frame without the flag measures the whole chain: equal
        // populations, and both directions agree again.
        let (all, _) = run(&f, &wc(&[0, 2]));
        assert_close(scalar(&all, "wc_conc_1"), 0.5, "c_1 unselected");
        assert_close(
            scalar(&all, "wc_asymmetry_max"),
            0.0,
            "asymmetry unselected",
        );
    }

    #[test]
    fn a_three_species_slab_pins_every_matrix_entry_against_hand_counted_bonds() {
        // One SC block of 8^3 sites typed by x-parity, except that the last column
        // (x = 7, which parity would make type 2) is type 3. Populations are
        // 256 / 192 / 64 out of 512, so
        //
        //     c_1 = 1/2      c_2 = 3/8      c_3 = 1/8
        //
        // Bond counts per central atom, counting the six nearest:
        //
        //   * type 1 (x even): x = 0 and x = 6 each see one type-3 x-neighbour and
        //     one type-2; x = 2 and x = 4 see two type-2. All four see 4 type-1
        //     along y and z. Averages: p_11 = 4/6, p_12 = 6/4/6 = 1/4,
        //     p_13 = 2/4/6 = 1/12.
        //   * type 2 (x = 1, 3, 5): both x-neighbours are type 1, the four
        //     y/z neighbours are type 2. p_21 = 2/6, p_22 = 4/6, p_23 = 0.
        //   * type 3 (x = 7): both x-neighbours wrap to even x, so both are
        //     type 1. p_31 = 2/6, p_33 = 4/6, p_32 = 0.
        //
        // One invariant checks all nine at once: since `p_ij` is a probability,
        // every row obeys `sum_j (1 - alpha_ij) c_j == 1`. A dropped bond, a
        // transposed index or a sign flip breaks it.
        let f = sc_typed(8, 3.0, |i, _, _| match i {
            7 => 3,
            i if i % 2 == 0 => 1,
            _ => 2,
        });
        let (sc, _) = run(&f, &wc(&[0, 6]));
        assert_close(scalar(&sc, "wc_conc_1"), 1.0 / 2.0, "c_1");
        assert_close(scalar(&sc, "wc_conc_2"), 3.0 / 8.0, "c_2");
        assert_close(scalar(&sc, "wc_conc_3"), 1.0 / 8.0, "c_3");
        let want: [[f64; 3]; 3] = [
            // central type 1: 1-4/3, 1-2/3, 1-2/3
            [-1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            // central type 2: 1-2/3, 1-16/9, 1-0
            [1.0 / 3.0, -7.0 / 9.0, 1.0],
            // central type 3: 1-2/3, 1-0, 1-16/3
            [1.0 / 3.0, 1.0, -13.0 / 3.0],
        ];
        for (i, row) in want.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                assert_close(
                    scalar(&sc, &format!("wc_m1_{}_{}", i + 1, j + 1)),
                    w,
                    &format!("alpha_{}{}", i + 1, j + 1),
                );
            }
            let prob: f64 = row
                .iter()
                .zip([0.5, 3.0 / 8.0, 1.0 / 8.0])
                .map(|(&a, c)| (1.0 - a) * c)
                .sum();
            assert_close(
                prob,
                1.0,
                &format!("row {} must be a probability distribution", i + 1),
            );
        }
        // This fixture happens to *be* symmetric -- `p_ij/c_j` agrees in both
        // directions for all three pairs. Pinning the 0 says the agreement was
        // measured, not imposed: the previous test shows the same code reporting a
        // real 7/12 on a frame where it is genuinely there.
        assert_close(scalar(&sc, "wc_asymmetry_max"), 0.0, "asymmetry");
    }

    #[test]
    fn an_ordered_fcc_alloy_matches_its_textbook_matrix() {
        // L1_2, with the bond census above and `c_1 = 1/4`, `c_2 = 3/4`:
        //
        //     central type 1 (corners):     p_11 = 0      p_12 = 1
        //     central type 2 (face):       p_21 = 4/12   p_22 = 8/12
        //
        //     alpha_11 = 1 - 0/(1/4)       = +1
        //     alpha_12 = 1 - 1/(3/4)       = -1/3
        //     alpha_21 = 1 - (1/3)/(1/4)   = -1/3
        //     alpha_22 = 1 - (2/3)/(3/4)   = +1/9
        //
        // A random FCC alloy reads 0 everywhere; the chemical ordering is what
        // makes unlike neighbours *preferred* (negative across), while the minority
        // species' own diagonal is positive because it sees itself 2/3 of the time
        // against a global 3/4 -- it is slightly *less* ordered with itself than a
        // random alloy would be.
        //
        // Each entry is checked against its own fraction rather than against the
        // others: swapping `c` and `p`, or flipping the sign, would still leave the
        // row sums at 1 and still produce a matrix that looks like a parameter.
        let (sc, _) = run(&fcc_l12(10, 4.0), &wc(&[0, 12]));
        assert_close(scalar(&sc, "wc_ntypes"), 2.0, "ntypes");
        assert_close(scalar(&sc, "wc_conc_1"), 0.25, "c_1");
        assert_close(scalar(&sc, "wc_conc_2"), 0.75, "c_2");
        assert_close(scalar(&sc, "wc_m1_1_1"), 1.0, "alpha_11");
        assert_close(scalar(&sc, "wc_m1_1_2"), -1.0 / 3.0, "alpha_12");
        assert_close(scalar(&sc, "wc_m1_2_1"), -1.0 / 3.0, "alpha_21");
        assert_close(scalar(&sc, "wc_m1_2_2"), 1.0 / 9.0, "alpha_22");
        assert_close(scalar(&sc, "wc_asymmetry_max"), 0.0, "asymmetry");
        // Row 1: no own-type neighbour at all, so the whole of `c_2` shows up as
        // pair probability. Row 2: a third of the shell is type 1 against a
        // global 1/4.
        let row1 = (1.0 - 1.0) * 0.25 + (1.0 - (-1.0 / 3.0)) * 0.75;
        let row2 = (1.0 - (-1.0 / 3.0)) * 0.25 + (1.0 - 1.0 / 9.0) * 0.75;
        assert_close(row1, 1.0, "row 1 is a probability distribution");
        assert_close(row2, 1.0, "row 2 is a probability distribution");
    }

    #[test]
    fn the_shell_width_is_the_cumulative_difference_not_the_cumulative_count() {
        // SC shell 1 is 6 wide, so `[0, 6]` and `[0, 6, 12]` must not agree: the
        // second adds ranks 6..12, which are six of the twelve tied vectors at
        // distance a*sqrt(2). Reading `ends` as widths instead of ends would put
        // the same six neighbours in both shells and count them twice.
        let f = sc(14, 3.0);
        let (s1, _) = run(&f, &wc(&[0, 6]));
        let (s2, _) = run(&f, &wc(&[0, 6, 12]));
        assert_eq!(scalar(&s1, "wc_shells"), 1.0);
        assert_eq!(scalar(&s2, "wc_shells"), 2.0);
        assert_close(scalar(&s1, "wc_m1_nneigh"), 6.0, "shell 1 width");
        assert_close(scalar(&s2, "wc_m1_nneigh"), 6.0, "shell 1 width");
        assert_close(scalar(&s2, "wc_m2_nneigh"), 6.0, "shell 2 width");
        // A single species is exactly 0 in every shell, so this cannot catch a
        // mis-set width on its own; the two-block fixture below can.
        assert_eq!(scalar(&s2, "wc_m1_1_1"), 0.0);
        assert_eq!(scalar(&s2, "wc_m2_1_1"), 0.0);
    }

    #[test]
    fn a_second_shell_of_another_species_shows_up_only_in_the_second_shell() {
        // Two blocks of equal size, `a = 3`. The six second-nearest vectors are at
        // 3*sqrt(2) = 4.243 and belong to the atom's own block. With `[0, 6]` only
        // the pure first shell is measured; with `[0, 12]` six of the twelve
        // second-shell vectors are included as well. Both are entirely
        // same-type, so the diagonal is -1 either way -- but the *pair counts*
        // differ, and a second shell that silently reused the first shell's
        // neighbours would show up as the same alpha for a shell that measured a
        // different set. The invariant to check is the per-atom shell
        // populations, which `wc_m2_nneigh` records.
        let (sc, _) = run(&two_blocks(8, 8, 3.0), &wc(&[0, 6, 12]));
        assert_close(scalar(&sc, "wc_m1_nneigh"), 6.0, "shell 1");
        assert_close(scalar(&sc, "wc_m2_nneigh"), 6.0, "shell 2");
        assert_close(scalar(&sc, "wc_m1_1_1"), -1.0, "alpha_11 shell 1");
        assert_close(scalar(&sc, "wc_m2_1_1"), -1.0, "alpha_11 shell 2");
    }

    #[test]
    fn the_per_particle_property_is_flattened_in_type_index_order_and_averages_back() {
        let f = sc_typed(8, 3.0, |i, _, _| match i {
            7 => 3,
            i if i % 2 == 0 => 1,
            _ => 2,
        });
        let mut w = wc(&[0, 6]);
        w.per_particle = true;
        let (sc, store) = run(&f, &w);
        let prop = store.get("WarrenCowleyShell1").expect("property");
        assert!(prop.per_particle, "a heatmap value is per atom");
        assert_eq!(prop.components, 9, "3 types squared");
        assert_eq!(prop.len(), f.natoms(), "one value per atom");
        assert_eq!(
            prop.as_f64().expect("f64").len(),
            f.natoms() * 9,
            "flat length"
        );
        let n = f.natoms();
        let vals = prop.as_f64().expect("f64").clone();
        assert_eq!(vals.len(), n * 9);
        // Only the central atom's own row of three is filled. That is what makes
        // the property a per-atom value rather than a copy of the global matrix
        // pasted nine times, and a row filled for every type would be the other
        // plausible mistake.
        for row in vals.chunks(9) {
            let finite: Vec<usize> = (0..9).filter(|&c| row[c].is_finite()).collect();
            assert_eq!(finite.len(), 3, "exactly one row of 3 per atom");
            assert_eq!(finite[0] % 3, 0, "a filled row is the atom's own type");
        }
        // The mean over the central atoms of a type must equal the global entry,
        // so the two outputs cannot drift apart. Only each atom's own row is
        // filled, so the mean has to be taken over that type's atoms: the NaNs
        // elsewhere would otherwise poison it and every comparison would pass
        // vacuously against NaN.
        for (i, &ti) in [1u32, 2, 3].iter().enumerate() {
            let of_type: Vec<usize> = (0..n).filter(|&a| f.particles.types[a] == ti).collect();
            assert!(!of_type.is_empty(), "type {ti} exists in the fixture");
            for (j, &tj) in [1u32, 2, 3].iter().enumerate() {
                let col: Vec<f64> = of_type.iter().map(|&a| vals[a * 9 + i * 3 + j]).collect();
                assert!(
                    col.iter().all(|v| v.is_finite()),
                    "every central {ti} atom has a finite row"
                );
                let m = col.iter().sum::<f64>() / col.len() as f64;
                assert_close(
                    m,
                    scalar(&sc, &format!("wc_m1_{ti}_{tj}")),
                    &format!("global vs mean for {ti}->{tj}"),
                );
            }
        }
    }

    /// A straight chain of `n` atoms spaced `d` apart, in a box with no periodic
    /// boundary conditions.
    ///
    /// Used where neighbour counts have to be *derived* rather than assumed. Every
    /// atom has a neighbour on each side within reach, and an end atom's second
    /// nearest neighbour is the second atom of the chain -- k-nearest ranks are
    /// always well defined, and never short.
    fn chain(n: usize, d: f64) -> Frame {
        let mut pd = ParticleData::with_capacity(n);
        for i in 0..n {
            pd.add_atom([i as f64 * d, 0.0, 0.0], i as i64 + 1, 1 + (i % 2) as u32);
        }
        let span = (n as f64 * d) + 40.0;
        let mut cell = UnitCell::orthorhombic(span, 60.0, 60.0);
        cell.pbc = [false, false, false];
        Frame::new(cell, pd)
    }

    #[test]
    fn a_shell_never_reuses_a_neighbour_to_fill_out_a_rank() {
        // A ten-atom alternating chain with `[0, 2]`. The end atoms have only one
        // neighbour at a bond length, so their second nearest neighbour is the
        // *second atom of the chain* rather than a repeat of the first. That is
        // the only correct reading of "ranks", and it is what separates this from
        // the reference's padded `-1` index, which would count the same neighbour
        // twice and turn an alternating chain into a self-ordered one.
        //
        // The tell is that the two off-diagonal entries must come out equal. The
        // chain alternates 1,2,1,2,..., so `alpha_12` and `alpha_21` state the same
        // physical fact and both must be strongly negative:
        //
        //     p_12 = p_21 = (0.5 + 1 + 1 + 1 + 1) / 5 = 0.9    c_1 = c_2 = 1/2
        //     alpha_12 = alpha_21 = 1 - 0.9 / 0.5 = -0.8
        //
        // A repeat would push the end atoms' type-2 count to a full 2/2, lifting
        // `p_21` to 1.0 and giving `alpha_21 = -1` against `alpha_12 = -0.8`. The
        // mismatch would also surface as `wc_asymmetry_max` = 0.2.
        let f = chain(10, 3.0);
        let (sc, _) = run(&f, &wc(&[0, 2]));
        assert_close(scalar(&sc, "wc_m1_nneigh"), 2.0, "shell width");
        assert_close(scalar(&sc, "wc_conc_1"), 0.5, "c_1");
        assert_close(scalar(&sc, "wc_conc_2"), 0.5, "c_2");
        // Atom 0 keeps one neighbour of its own type and atom 9 likewise, since
        // each has only one neighbour inside a bond length and the next atom of
        // the chain supplies the other rank. So p_11 = p_22 = 0.1 and
        // p_12 = p_21 = 0.9, and the two diagonal entries are equal as an
        // alternating chain requires.
        assert_close(scalar(&sc, "wc_m1_1_1"), 0.8, "alpha_11");
        assert_close(scalar(&sc, "wc_m1_2_2"), 0.8, "alpha_22");
        assert_close(scalar(&sc, "wc_m1_1_2"), -0.8, "alpha_12");
        assert_close(scalar(&sc, "wc_m1_2_1"), -0.8, "alpha_21");
        assert_close(scalar(&sc, "wc_asymmetry_max"), 0.0, "asymmetry");
        // Both rows are still probability distributions, so the counts sum right.
        let row1: f64 = [0.5, 0.5]
            .iter()
            .zip([0.8, -0.8])
            .map(|(&c, a)| (1.0 - a) * c)
            .sum();
        let row2: f64 = [0.5, 0.5]
            .iter()
            .zip([-0.8, 0.8])
            .map(|(&c, a)| (1.0 - a) * c)
            .sum();
        assert_close(row1, 1.0, "row 1 is a probability distribution");
        assert_close(row2, 1.0, "row 2 is a probability distribution");
    }

    #[test]
    fn asking_for_more_neighbours_than_there_are_atoms_is_refused_up_front() {
        // Three atoms cannot supply twelve neighbours each, and the error should
        // be about the request rather than about a search that cannot converge.
        let mut pd = ParticleData::with_capacity(3);
        for (k, q) in [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
            .into_iter()
            .enumerate()
        {
            pd.add_atom(q, k as i64 + 1, 1);
        }
        let f = Frame::new(UnitCell::orthorhombic(40.0, 40.0, 40.0), pd);
        let msg = run_err(&f, &wc(&[0, 12]));
        assert!(msg.contains("only 3 atoms"), "{msg}");
    }

    #[test]
    fn only_selected_refuses_a_frame_with_no_selection() {
        // Analysing every atom when the caller asked for a subset would report a
        // whole-box average under a flag that says otherwise.
        let f = sc(10, 3.0);
        let mut w = wc(&[0, 6]);
        w.only_selected = true;
        let msg = run_err(&f, &w);
        assert!(msg.contains("Selection"), "{msg}");
    }

    #[test]
    fn only_selected_measures_the_subset_against_its_own_neighbourhood() {
        // Two blocks 40 A apart, selecting only the first. Its halo contains only
        // type 1, so c_2 = 0 and the type-2 row is NaN rather than a number
        // computed against a global concentration the selection does not contain.
        let mut f = two_blocks(8, 8, 3.0);
        let n1 = 8 * 8 * 8;
        let mut sel = vec![0u8; f.natoms()];
        for s in sel.iter_mut().take(n1) {
            *s = 1;
        }
        f.particles.selection = Some(sel);
        let mut w = wc(&[0, 6]);
        w.only_selected = true;
        let (sc, _) = run(&f, &w);
        assert_close(scalar(&sc, "wc_atoms_central"), n1 as f64, "central atoms");
        assert_close(scalar(&sc, "wc_conc_1"), 1.0, "c_1");
        assert_close(scalar(&sc, "wc_conc_2"), 0.0, "c_2");
        assert_eq!(scalar(&sc, "wc_m1_1_1"), 0.0);
        assert!(
            scalar(&sc, "wc_m1_1_2").is_nan(),
            "a type absent from the selection's halo has no alpha"
        );
        // Without the flag the whole box is measured and c_2 is a real number.
        let (all, _) = run(&f, &wc(&[0, 6]));
        assert_close(scalar(&all, "wc_conc_2"), 0.5, "c_2 unselected");
    }

    #[test]
    fn a_selection_on_the_frame_restricts_nothing_without_the_flag() {
        // A frame that carries a mask is still measured in full when
        // `only_selected` is off. Reading a selection off the file is not the same
        // as asking for a subset, and honouring it silently would make the same
        // command measure different things on two files that differ only in
        // whether someone exported a selection.
        let mut f = two_blocks(6, 6, 3.0);
        let n1 = 6 * 6 * 6;
        let mut sel = vec![0u8; f.natoms()];
        for s in sel.iter_mut().take(n1) {
            *s = 1;
        }
        f.particles.selection = Some(sel);
        let plain = two_blocks(6, 6, 3.0);
        let (masked, _) = run(&f, &wc(&[0, 6]));
        let (bare, _) = run(&plain, &wc(&[0, 6]));
        assert_close(
            scalar(&masked, "wc_atoms_central"),
            scalar(&bare, "wc_atoms_central"),
            "every atom is central without the flag",
        );
        assert_close(
            scalar(&masked, "wc_conc_2"),
            scalar(&bare, "wc_conc_2"),
            "concentrations are over the whole frame",
        );
        for name in ["wc_m1_1_1", "wc_m1_1_2", "wc_m1_2_1", "wc_m1_2_2"] {
            assert_close(scalar(&masked, name), scalar(&bare, name), name);
        }
    }

    #[test]
    fn a_type_with_no_selected_central_atom_leaves_its_row_nan() {
        // Checkerboard by x-parity, selecting only the type-1 half. Type 2 appears
        // in the halo -- so its concentration is real -- but no *selected* central
        // atom is type 2, so its row has nothing to average and is NaN. The
        // reference reaches the same place through `nanmean` of an empty axis.
        let mut f = sc_typed(8, 3.0, |i, _, _| 1 + (i % 2) as u32);
        let mut sel = vec![0u8; f.natoms()];
        for (a, s) in sel.iter_mut().enumerate() {
            // Type 1 sits at even x, i.e. even atom index within each 8x8 slab.
            let x = (a / 64) % 8;
            *s = u8::from(x % 2 == 0);
        }
        f.particles.selection = Some(sel);
        let mut w = wc(&[0, 6]);
        w.only_selected = true;
        let (sc, _) = run(&f, &w);
        assert_close(scalar(&sc, "wc_atoms_central"), 256.0, "central atoms");
        assert!(
            scalar(&sc, "wc_m1_2_2").is_nan(),
            "type 2 has no selected central atom, so its row is NaN"
        );
        assert!(
            scalar(&sc, "wc_m1_1_1").is_finite(),
            "type 1 has central atoms, so its row is a number"
        );
    }

    #[test]
    fn only_same_type_drops_the_off_diagonal_entries() {
        let f = sc_typed(8, 3.0, |i, _, _| match i {
            7 => 3,
            i if i % 2 == 0 => 1,
            _ => 2,
        });
        let mut w = wc(&[0, 6]);
        w.only_same_type = true;
        let (sc, _) = run(&f, &w);
        let names: Vec<&str> = sc.scalars.iter().map(|(k, _)| k.as_str()).collect();
        for t in ["wc_m1_1", "wc_m1_2", "wc_m1_3"] {
            assert!(names.contains(&t), "{t} missing from {names:?}");
        }
        // Every alpha name is `wc_m1_<type>`: three underscores would mean an
        // off-diagonal pair survived.
        for n in names.iter().filter(|n| n.starts_with("wc_m1_")) {
            assert_eq!(
                n.matches('_').count(),
                2,
                "{n} looks like an off-diagonal entry"
            );
        }
        assert_close(scalar(&sc, "wc_m1_1"), -1.0 / 3.0, "alpha_11");
        assert_close(scalar(&sc, "wc_m1_2"), -7.0 / 9.0, "alpha_22");
        assert_close(scalar(&sc, "wc_m1_3"), -13.0 / 3.0, "alpha_33");
    }

    #[test]
    fn a_leading_zero_is_optional_and_a_bad_list_is_refused() {
        assert_eq!(parse_shells("12").unwrap(), vec![0, 12]);
        assert_eq!(parse_shells("0,12").unwrap(), vec![0, 12]);
        assert_eq!(parse_shells("12,18").unwrap(), vec![0, 12, 18]);
        assert_eq!(parse_shells(" 12 , 18 ").unwrap(), vec![0, 12, 18]);
        // Not increasing, not a number, or empty: each says which.
        for (bad, needle) in [
            ("12,12", "strictly increasing"),
            ("18,12", "strictly increasing"),
            ("12,6", "strictly increasing"),
            ("6,0,12", "positive"),
            ("", "at least one shell end"),
            ("12,x", "whole number"),
            ("-1", "whole number"),
        ] {
            let e = parse_shells(bad).unwrap_err().to_string();
            assert!(
                e.contains(needle),
                "{bad:?} should mention {needle:?}, said {e}"
            );
        }
    }

    #[test]
    fn an_empty_frame_reports_zero_shells_rather_than_failing() {
        let f = Frame::new(
            UnitCell::orthorhombic(10.0, 10.0, 10.0),
            ParticleData::new(0),
        );
        let (sc, _) = run(&f, &wc(&[0, 12]));
        assert_eq!(scalar(&sc, "wc_atoms_central"), 0.0);
        assert_eq!(scalar(&sc, "wc_ntypes"), 0.0);
        assert_eq!(scalar(&sc, "wc_shells"), 1.0);
    }

    #[test]
    fn the_result_does_not_depend_on_a_shared_cell_list_being_present() {
        // The module builds its own list, so passing a narrower one in must change
        // nothing. A shared narrow list would truncate the stencil and shorten
        // shells near the box edge, which is exactly what the six-neighbour SC
        // shell at the box face depends on.
        let f = sc_typed(8, 3.0, |i, _, _| match i {
            7 => 3,
            i if i % 2 == 0 => 1,
            _ => 2,
        });
        let w = wc(&[0, 6]);
        let (bare, _) = run(&f, &w);
        let narrow = CellList::build(&f.cell, &f.particles.positions[..f.natoms()], 3.1).unwrap();
        let nk = NeighborhoodContext {
            cell: Some(&narrow),
        };
        let par = Par::new(1);
        let ctx = AnalysisContext {
            frame: &f,
            nkeys: &nk,
            par: &par,
        };
        let mut out = PropertyStore::new();
        let mut with = Summary::default();
        w.run(&ctx, &mut out, &mut with).unwrap();
        assert_eq!(bare.scalars.len(), with.scalars.len());
        for ((k, v), (k2, v2)) in bare.scalars.iter().zip(with.scalars.iter()) {
            assert_eq!(k, k2);
            assert!(
                (v - v2).abs() < 1e-15 || (v.is_nan() && v2.is_nan()),
                "{k}: {v} vs {v2}"
            );
        }
    }

    #[test]
    fn sparse_type_ids_do_not_produce_empty_rows() {
        // Types {1, 7} must give a 2x2 matrix, not 7x7 with five NaN rows. The
        // reference gets this from `np.unique`; indexing by the maximum type id
        // would add rows no atom occupies.
        let f = sc_typed(8, 3.0, |i, _, _| if i % 2 == 0 { 1 } else { 7 });
        let (sc, _) = run(&f, &wc(&[0, 6]));
        assert_eq!(scalar(&sc, "wc_ntypes"), 2.0);
        let names: Vec<&str> = sc.scalars.iter().map(|(k, _)| k.as_str()).collect();
        assert!(names.contains(&"wc_m1_1_7"), "{names:?}");
        assert!(names.contains(&"wc_m1_7_1"), "{names:?}");
        assert!(
            !names.iter().any(|n| n.starts_with("wc_m1_2")),
            "no row for an absent type: {names:?}"
        );
    }

    #[test]
    fn the_probe_radius_grows_until_every_central_atom_can_see_k_neighbours() {
        // A frame with one dense block and one sparse block: the density
        // estimate is set by the mean, so the first probe cannot serve every
        // atom and the search has to widen within the stencil. No atom may be
        // skipped.
        let mut pd = ParticleData::with_capacity(2 * 6 * 6 * 6);
        let mut id = 0i64;
        for (base, n) in [(0.0f64, 6usize), (60.0, 3usize)] {
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        id += 1;
                        pd.add_atom(
                            [base + i as f64 * 3.0, j as f64 * 3.0, k as f64 * 3.0],
                            id,
                            1 + ((i + j + k) % 2) as u32,
                        );
                    }
                }
            }
        }
        let natoms = pd.natoms;
        let f = Frame::new(UnitCell::orthorhombic(120.0, 24.0, 24.0), pd);
        let (sc, _) = run(&f, &wc(&[0, 6]));
        assert_close(scalar(&sc, "wc_atoms_skipped"), 0.0, "skipped");
        assert_close(scalar(&sc, "wc_atoms_central"), natoms as f64, "central");
        assert!(
            scalar(&sc, "wc_probe_radius") > 0.0,
            "the probe radius is reported so the widening is visible"
        );
    }

    fn run_threads(f: &Frame, w: &WarrenCowley, threads: usize) -> (Summary, PropertyStore) {
        let nk = NeighborhoodContext { cell: None };
        let par = Par::new(threads);
        let ctx = AnalysisContext {
            frame: f,
            nkeys: &nk,
            par: &par,
        };
        let mut out = PropertyStore::new();
        let mut sc = Summary::default();
        w.run(&ctx, &mut out, &mut sc).expect("run");
        (sc, out)
    }

    /// A thermally jittered FCC block with pseudo-random types 1..=4 and a
    /// pseudo-random selection, large enough for several parallel tasks.
    fn jittered_alloy(ncell: usize) -> Frame {
        let mut f = fcc_l12(ncell, 3.6);
        let mut s = 12345u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let n = f.natoms();
        let mut sel = Vec::with_capacity(n);
        for i in 0..n {
            for d in 0..3 {
                f.particles.positions[i][d] += 0.15 * (next() - 0.5);
            }
            f.particles.types[i] = 1 + (next() * 4.0) as u32;
            sel.push(u8::from(next() < 0.6));
        }
        f.particles.selection = Some(sel);
        f
    }

    #[test]
    fn the_result_is_bit_identical_at_any_thread_count() {
        let f = jittered_alloy(10);
        assert!(f.natoms() > 3 * WC_GRAIN, "several parallel tasks");
        for only_selected in [false, true] {
            let w = WarrenCowley {
                nneigh: vec![0, 12, 18],
                only_selected,
                per_particle: true,
                ..Default::default()
            };
            let (s1, o1) = run_threads(&f, &w, 1);
            let (s4, o4) = run_threads(&f, &w, 4);
            let bits = |sc: &Summary| -> Vec<(String, u64)> {
                sc.scalars
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_bits()))
                    .collect()
            };
            assert_eq!(bits(&s1), bits(&s4), "only_selected={only_selected}");
            for m in ["WarrenCowleyShell1", "WarrenCowleyShell2"] {
                let b = |o: &PropertyStore| -> Vec<u64> {
                    o.get(m)
                        .unwrap()
                        .as_f64()
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect()
                };
                assert_eq!(b(&o1), b(&o4), "{m} only_selected={only_selected}");
            }
        }
    }

    #[test]
    fn edge_correct_reads_a_film_as_its_bulk() {
        // An ordered L1_2 film (open along z): the surface atoms' twelve
        // nearest reach into the second shell and move alpha; measuring only
        // atoms deeper than their 12th neighbour gives the periodic bulk's
        // matrix (to the rounding of a sum over fewer atoms).
        let bulk = fcc_l12(6, 3.6);
        let mut film = bulk.clone();
        film.cell = film.cell.with_pbc([true, true, false]);
        let w = wc(&[0, 12]);
        let edge = WarrenCowley {
            edge_correct: true,
            ..w.clone()
        };
        let (sb, _) = run(&bulk, &w);
        let (sf, _) = run(&film, &w);
        let (se, _) = run(&film, &edge);
        let keys = ["wc_m1_1_1", "wc_m1_1_2", "wc_m1_2_1", "wc_m1_2_2"];
        assert!(
            keys.iter()
                .any(|k| (scalar(&sf, k) - scalar(&sb, k)).abs() > 1e-3),
            "the surface must bias the uncorrected film"
        );
        for k in keys {
            assert!((scalar(&se, k) - scalar(&sb, k)).abs() < 1e-12, "{k}");
        }
        let surface = scalar(&se, "wc_atoms_surface");
        assert!(surface > 0.0);
        assert_eq!(
            surface + scalar(&se, "wc_atoms_measured"),
            film.natoms() as f64
        );
        // Periodic bulk excludes nothing.
        let (sbe, _) = run(&bulk, &edge);
        assert_eq!(scalar(&sbe, "wc_atoms_surface"), 0.0);
        // The default output carries no edge-correct scalars.
        assert!(sb.scalars.iter().all(|(k, _)| k != "wc_atoms_surface"));
    }
}
